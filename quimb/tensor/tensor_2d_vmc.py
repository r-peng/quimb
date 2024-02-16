import time,itertools,functools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from .tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    AmplitudeFactory,
    Model,
    ExchangeSampler,
)
def flatten(i,j,Ly): # flattern site to row order
    return i*Ly+j
def flat2site(ix,Ly): # ix in row order
    return ix//Ly,ix%Ly
def cache_key(config,i,direction,step,Ly):
    if direction=='row':
        if step==1: # bottom env
            config = config[:(i+1)*Ly] 
        else: # top env
            config = config[i*Ly:]
    else: 
        cols = range(i+1) if step==1 else range(i,Ly)
        nsite = len(config)
        config = tuple([config[slc] for slc in [slice(col,nsite,Ly) for col in cols]])
    return config,direction,step
####################################################################################
# amplitude fxns 
####################################################################################
class AmplitudeFactory2D(AmplitudeFactory):
    def __init__(self,psi,model,blks=None,phys_dim=2,backend='numpy',from_plq=True,**compress_opts):
        # init wfn
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.nsite = self.Lx * self.Ly
        self.sites = list(itertools.product(range(self.Lx),range(self.Ly)))
        psi.add_tag('KET')
        self.set_psi(psi) # current state stored in self.psi

        self.data_map = self.get_data_map(phys_dim)

        self.model = model
        self.backend = backend 
        self.from_plq = from_plq 
        self.wfn2backend()

        # init contraction
        self.compress_opts = compress_opts
        self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2

        self.blks = [self.sites] if blks is None else blks
        self.site_map = self.get_site_map()
        self.constructors = self.get_constructors(psi)
        self.get_block_dict()
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)

        self.is_tn = True
        self.dmc = False 
        self.pbc = False
        self.deterministic = False 
##### wfn methods #####
    def flatten(self,site):
        i,j = site
        return flatten(i,j,self.Ly) 
    def flat2site(self,ix):
        return flat2site(ix,self.Ly)
    def site_tag(self,site):
        return self.psi.site_tag(*site)
    def site_tags(self,site):
        return self.site_tag(site),self.psi.row_tag(site[0]),self.col_tag(site[1])
    def site_ind(self,site):
        return self.psi.site_ind(*site)
    def col_tag(self,col):
        return self.psi.col_tag(col)
    def row_tag(self,row):
        return self.psi.row_tag(row)
    def plq_sites(self,plq_key):
        (i0,j0),(x_bsz,y_bsz) = plq_key
        sites = list(itertools.product(range(i0,i0+x_bsz),range(j0,j0+y_bsz)))
        return sites
    def cache_key(self,config,i,direction,step):
        return cache_key(config,i,direction,step,self.Ly)
    def _new_cache(self,config,cache,direction='row'):
        cache_new = dict()
        for i in range(self.rix1+1):
            key = self.cache_key(config,i,direction,1) 
            cache_new[key] = cache[key]
        for i in range(self.rix2,self.Lx):
            key = self.cache_key(config,i,direction,-1) 
            cache_new[key] = cache[key]
        return cache_new
    def update_cache(self,config):
        # TODO: enable vertical compression
        self.cache = self._new_cache(config,self.cache)
    def free_ad_cache(self):
        self._cache = dict()
    def free_sweep_cache(self,step,direction='row'):
        keys = [key for key in self.cache if key[1:]==(direction,step)]
        for key in keys:
            self.cache.pop(key)
    def set_psi(self,psi):
        self.psi = psi
        self.cache = dict()
        self.free_ad_cache() 
##### compress row methods  #####
    def get_mid_env(self,i,config,append='',psi=None,direction='row'):
        psi = self.psi if psi is None else psi 
        if direction=='row':
            row_tag = self.row_tag
            Ly = self.Ly
            slc = slice(i*self.Ly,(i+1)*self.Ly,1) 
        else:
            row_tag = self.col_tag
            Ly = self.Lx
            slc = slice(i,self.nsite,self.Ly) 
        row = psi.select(row_tag(i),virtual=False)
        key = config[slc]
        # compute mid env for row i
        for j in range(Ly-1,-1,-1):
            site = (i,j) if direction=='row' else (j,i)
            row.add_tensor(self.get_bra_tsr(key[j],site,append=append),virtual=True)
        return row
    def contract_mid_env(self,i,row,direction='row'):
        Ly = self.Ly if direction=='row' else self.Lx
        try: 
            for j in range(Ly-1,-1,-1):
                site = (i,j) if direction=='row' else (j,i)
                row.contract_tags(self.site_tag(site),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def compress_row_pbc(self,tn,i,direction='row'):
        Ly = self.Ly if direction=='row' else self.Lx
        for j in range(Ly): # compress between j,j+1
            if direction=='row':
                site1,site2 = (i,j),(i,(j+1)%Ly) 
            else:
                site1,site2 = (j,i),((j+1)%Ly,i) 
            tn.compress_between(self.site_tag(site1),self.site_tag(site2),
                                **self.compress_opts)
        return tn
    def compress_row_obc(self,tn,i,direction='row'):
        if direction=='row':
            tn.canonize_row(i,sweep='left')
            Ly = self.Ly
        else:
            tn.canonize_column(i,sweep='up')
            Ly = self.Lx
        for j in range(Ly-1):
            if direction=='row':
                site1,site2 = (i,j),(i,j+1) 
            else:
                site1,site2 = (j,i),(j+1,i) 
            self.tensor_compress_bond(tn[site1],tn[site2],absorb='right')        
        return tn
    def contract_boundary_single(self,tn,i,iprev,direction='row'):
        Ly = self.Ly if direction=='row' else self.Lx
        for j in range(Ly):
            if direction=='row':
                sites = (iprev,j),(i,j) 
            else:
                sites = (j,iprev),(j,i) 
            tn.contract_([self.site_tag(site) for site in sites],which='any')
        if self.pbc:
            return self.compress_row_pbc(tn,i,direction=direction)
        else:
            return self.compress_row_obc(tn,i,direction=direction)
    def get_benv(self,i,row,env_prev,config,step,direction='row'):
        cache = self.cache if self._backend=='numpy' else self._cache
        key = self.cache_key(config,i,direction,step)
        if key in cache: # reusable
            return cache[key]
        row = self.contract_mid_env(i,row,direction=direction)

        # is terminal
        if step==1 and i==0:
            cache[key] = row
            return row
        Lx = self.Lx if direction=='row' else self.Ly
        if step==-1 and i==Lx-1:
            cache[key] = row
            return row

        # contraction fails
        if row is None:
            cache[key] = row
            return row
        if env_prev is None:
            cache[key] = None 
            return None

        if step==1:
            tn = env_prev.copy()
            tn.add_tensor_network(row,virtual=False)
        else:
            tn = row
            tn.add_tensor_network(env_prev,virtual=False)
        try:
            tn = self.contract_boundary_single(tn,i,i-step,direction=direction)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_env_prev(self,config,i,step,direction='row'):
        if step==1 and i==0:
            return None 
        Lx = self.Lx if direction=='row' else self.Ly
        if step==-1 and i==Lx-1:
            return None
        cache = self.cache if self._backend=='numpy' else self._cache
        key = self.cache_key(config,i-step,direction,step)
        return cache[key]
    def _get_all_benvs(self,config,step,psi=None,start=None,stop=None,append='',direction='row'):
        psi = self.psi if psi is None else psi

        Lx = self.Lx if direction=='row' else self.Ly
        if step==1:
            start = 0 if start is None else start
            stop = Lx-1 if stop is None else stop
        else:
            start = Lx-1 if start is None else start
            stop = 0 if stop is None else stop 
        env_prev = self.get_env_prev(config,start,step,direction=direction)
        for i in range(start,stop,step):
            row = self.get_mid_env(i,config,append=append,psi=psi,direction=direction)
            env_prev = self.get_benv(i,row,env_prev,config,step,direction=direction)
        return env_prev
    def get_all_benvs(self,config,psi=None,x_bsz=1,compute_bot=True,compute_top=True,imax=None,imin=None,direction='row'):
        psi = self.psi if psi is None else psi
        Lx = self.Lx if direction=='row' else self.Ly

        env_bot = None
        env_top = None
        if compute_bot: 
            stop = Lx-x_bsz if imax is None else imax+1 
            env_bot = self._get_all_benvs(config,1,psi=psi,stop=stop,direction=direction)
        if compute_top:
            stop = x_bsz-1 if imin is None else imin-1
            env_top = self._get_all_benvs(config,-1,psi=psi,stop=stop,direction=direction)
        return env_bot,env_top
    def _contract_cols(self,cols,js,direction='col'):
        col_tag = self.col_tag if direction=='col' else self.row_tag
        tags = [col_tag(j) for j in js]
        cols ^= tags
        return cols
    def get_all_envs(self,cols,step,stop=None,inplace=False,direction='col'):
        tmp = cols if inplace else cols.copy()
        if direction=='col':
            Ly = self.Ly 
            col_tag = self.col_tag
        else:
            Ly = self.Lx
            col_tag = self.row_tag
        if step==1:
            start = 0
            stop = Ly - 1 if stop is None else stop
        else:
            start = Ly - 1
            stop = 0 if stop is None else stop
        first_col = col_tag(start)
        envs = [None] * Ly
        for j in range(start,stop,step): 
            try:
                tmp = self._contract_cols(tmp,(start,j),direction=direction)
                envs[j] = tmp.select(first_col,virtual=False)
            except (ValueError,IndexError):
                return cols,envs
        return cols,envs
    def _get_plq_sweep(self,j,y_bsz,cols,envs,step,direction='col'):
        if direction=='col':
            Ly = self.Ly 
            col_tag = self.col_tag
        else:
            Ly = self.Lx
            col_tag = self.row_tag
        tags = [col_tag(j+ix) for ix in range(y_bsz)]
        plq = cols.select(tags,which='any',virtual=False)
        if step==1:
            if j>0:
                other = plq
                plq = cols.select(col_tag(0),virtual=False)
                plq.add_tensor_network(other,virtual=False)
            if j<Ly - y_bsz:
                plq.add_tensor_network(envs[j+y_bsz],virtual=False)
        else:
            if j>0:
                other = plq
                plq = envs[j-1]
                plq.add_tensor_network(other,virtual=False)
            if j<Ly - y_bsz:
                plq.add_tensor_network(cols.select(col_tag(Ly-1),virtual=False),virtual=False)
        plq.view_like_(cols)
        return plq
    def _get_plq(self,j,y_bsz,cols,lenvs,renvs,direction='col'):
        # lenvs from lenvs, renvs from renvs
        if direction=='col':
            Ly = self.Ly 
            col_tag = self.col_tag
        else:
            Ly = self.Lx
            col_tag = self.row_tag
        tags = [col_tag(j+ix) for ix in range(y_bsz)]
        plq = cols.select(tags,which='any',virtual=False)
        if j>0:
            other = plq
            plq = lenvs[j-1]
            plq.add_tensor_network(other,virtual=False)
        if j<Ly - y_bsz:
            plq.add_tensor_network(renvs[j+y_bsz],virtual=False)
        plq.view_like_(cols)
        return plq
    def update_plq_from_3row(self,plq,cols,i,x_bsz,y_bsz,psi=None,direction='col'):
        psi = self.psi if psi is None else psi
        Ly = self.Ly if direction=='col' else self.Lx
        cols,lenvs = self.get_all_envs(cols,1,stop=Ly-y_bsz,inplace=False,direction=direction)
        cols,renvs = self.get_all_envs(cols,-1,stop=y_bsz-1,inplace=False,direction=direction)
        for j in range(self.Ly - y_bsz +1): 
            try:
                plq[(i,j),(x_bsz,y_bsz)] = self._get_plq(j,y_bsz,cols,lenvs,renvs,direction=direction) 
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def build_3row_tn(self,config,i,x_bsz,psi=None,direction='row'):
        psi = self.psi if psi is None else psi
        cache = self.cache if self._backend=='numpy' else self._cache
        Lx = self.Lx if direction=='row' else self.Ly
        try:
            tn = self.get_mid_env(i,config,psi=psi,direction=direction)
            for ix in range(1,x_bsz):
                tn.add_tensor_network(self.get_mid_env(i+ix,config,psi=psi,direction=direction),virtual=False)
            if i>0:
                other = tn 
                tn = cache[self.cache_key(config,i-1,direction,1)].copy()
                tn.add_tensor_network(other,virtual=False)
            if i+x_bsz<Lx:
                tn.add_tensor_network(cache[self.cache_key(config,i+x_bsz,direction,-1)],virtual=False)
        except AttributeError:
            tn = None
        return tn 
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,imin=None,imax=None,direction='row'):
        if direction=='row':
            Lx = self.Lx
            direction_ = 'col'
        else:
            Lx = self.Ly
            direction_ = 'row'
        imin = 0 if imin is None else imin
        imax = Lx-x_bsz if imax is None else imax
        psi = self.psi if psi is None else psi
        plq = dict()
        for i in range(imin,imax+1):
            cols = self.build_3row_tn(config,i,x_bsz,psi=psi,direction=direction)
            if cols is not None:
                plq = self.update_plq_from_3row(plq,cols,i,x_bsz,y_bsz,psi=psi,direction=direction_)
        return plq
    def unsigned_amplitude(self,config,to_numpy=True,i=None,direction='row'): 
        if i is None:
            imax,imin = self.rix1,self.rix2
        elif i==0:
            imax,imin = 0,1
        else:
            imax,imin = i-1,i
        # always contract rows into the middle
        env_bot,env_top = self.get_all_benvs(config,imax=imax,imin=imin,direction=direction)

        if env_bot is None and env_top is None:
            return None

        try:
            tn = env_bot.copy()
            tn.add_tensor_network(env_top,virtual=False)
        except AttributeError:
            return None 

        cx = safe_contract(tn)
        if cx is None:
            return None
        if to_numpy:
            cx = tensor2backend(cx,'numpy')
        return cx  
##### hamiltonian methods #######
    def batch_benvs(self,batch_key,new_cache):
        b = self.model.batched_pairs[batch_key]
        if b.direction=='row':
            psi = self.psi
        else:
            psi = self.psi.copy()
            psi.reorder('col',inplace=True)
        self._get_all_benvs(self.config,1,psi=psi,stop=b.bix+1,direction=b.direction)
        self._get_all_benvs(self.config,-1,psi=psi,stop=b.tix-1,direction=b.direction)
    def batch_pair_energies(self,batch_key,new_cache):
        b = self.model.batched_pairs[batch_key]
        ex = dict() 
        if b.deterministic:
            cx = self.amplitude(self.config,to_numpy=False)
            self.cx['deterministic'] = cx

            for where in b.pairs:
                ex_ij = self.update_pair_energy_from_benvs(where)
                for tag,eij in ex_ij.items():
                    ex[where,tag] = eij,cx,eij/cx
            return ex,None

        self.batch_benvs(batch_key,new_cache) 
        if self.from_plq and (not self.dmc):
            # form plqs
            plq = dict()
            for imin,imax,x_bsz,y_bsz in b.plq_types:
                plq.update(self.get_plq_from_benvs(self.config,x_bsz,y_bsz,imin=imin,imax=imax))
            return self.pair_energies_from_plq(plq,b.pairs),plq

        for where in b.pairs:
            i = min([i for i,_ in where])
            if i not in self.cx:
                self.cx[i] = self.amplitude(self.config,direction=b.direction,i=i,to_numpy=False)
            cij = self.cx[i]
            
            ex_ij = self.update_pair_energy_from_benvs(where,direction=b.direction,i=i) 
            if self.dmc:
                for tag,eij in ex_ij.items():
                    ex[tag] = eij/cij 
            else:
                for tag,eij in ex_ij.items():
                    ex[where,tag] = eij,cij,eij/cij 
        return ex,None
    def pair_energies_from_plq(self,plq,pairs):
        ex = dict() 
        for where in pairs:
            key = self.model.pair_key(*where)

            tn = plq.get(key,None) 
            if tn is None:
                continue
            if key not in self.cx:
                self.cx[key] = safe_contract(tn.copy())
            cij = self.cx[key]

            ex_ij = self.update_pair_energy_from_plq(tn,where) 
            for tag,eij in ex_ij.items():
                ex[where,tag] = eij,cij,eij/cij
        return ex
class PerturbedAmplitudeFactory2D(AmplitudeFactory2D):
    def __init__(self,psi,nn,model,**kwargs):
        self.nn = nn
        super().__init__(psi,model,from_plq=False,**kwargs)
    def get_block_dict(self):
        super().get_block_dict()
        self.nn.get_block_dict()
        start = self.nparam
        for _start,_stop in self.nn.block_dict:
            stop = start + _stop - _start
            self.block_dict.append((start,stop)) 
            start = stop
        self.nparam = stop
    def get_x(self):
        return np.concatenate([super().get_x(),self.nn.get_x()])
    def update(self,x,fname=None,root=0):
        npsi = self.nparam - self.nn.nparam
        fname_ = None if fname is None else fname+f'_0' 
        super().update(x[:npsi],fname=fname_,root=root)

        fname_ = None if fname is None else fname+f'_1' 
        self.nn.update(x[npsi:],fname=fname,root=root)
    def wfn2backend(self,**kwargs):
        super().wfn2backend(**kwargs)
        self.nn.wfn2backend(**kwargs)
    def extract_ad_grad(self):
        return np.concatenate([super().extract_ad_grad(),self.nn.extract_ad_grad()])
    def free_ad_cache(self):
        super().free_ad_cache()
        self.nn.free_ad_cache()
    def get_benv(self,i,row,env_prev,config,step,direction='row'):
        cache = self.cache if self._backend=='numpy' else self._cache    
        key = self.cache_key(config,i,direction,step)
        if key in cache: # reusable
            return cache[key]
        row = self.contract_mid_env(i,row,direction=direction)

        # is terminal
        if step==1 and i==0:
            cache[key] = row
            return row
        Lx = self.Lx if direction=='row' else self.Ly
        if step==-1 and i==Lx-1:
            cache[key] = row
            return row

        # contraction fails
        if row is None:
            cache[key] = row
            return row
        if env_prev is None:
            cache[key] = None 
            return None

        try:
            env_prev = self.nn.modify_mps_old(env_prev,config,i-step,step)
        except NotImplementedError:
            pass
        if step==1:
            tn = env_prev.copy()
            tn.add_tensor_network(row,virtual=False)
        else:
            tn = row
            tn.add_tensor_network(env_prev,virtual=False)
        try:
            tn = self.contract_boundary_single(tn,i,i-step,direction=direction)
        except (ValueError,IndexError):
            tn = None
        try:
            tn = self.nn.modify_mps_new(tn,config,i,step)
        except NotImplementedError:
            pass

        cache[key] = tn 
        return tn 
    def unsigned_amplitude(self,config,to_numpy=True,i=None,direction='row'): 
        if i is None:
            imax,imin = self.rix1,self.rix2
        elif i==0:
            imax,imin = 0,1
        else:
            imax,imin = i-1,i
        # always contract rows into the middle
        env_bot,env_top = self.get_all_benvs(config,imax=imax,imin=imin,direction=direction)

        if env_bot is None and env_top is None:
            return None

        try:
            env_bot = self.nn.modify_bot_mps(env_bot,config,imax)
        except NotImplementedError:
            pass
        try:
            env_top = self.nn.modify_top_mps(env_top,config,imin)
        except NotImplementedError:
            pass
        try:
            tn = env_bot.copy()
            tn.add_tensor_network(env_top,virtual=False)
        except AttributeError:
            return None 

        cx = safe_contract(tn)
        if cx is None:
            return None
        if to_numpy:
            cx = tensor2backend(cx,'numpy')
        return cx  
####################################################################
# models
####################################################################
class Batch:
    def __init__(self,bix=None,tix=None,direction='row',deterministic=False):
        self.bix,self.tix = bix,tix
        self.plq_types = []
        self.pairs = []
        self.direction = direction
        self.deterministic = deterministic
class Model2D(Model):
    def __init__(self,Lx,Ly,nbatch=1,pbc=False,deterministic=False):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly
        self.nbatch = nbatch 
        self.pbc = pbc
        self.deterministic = deterministic
    def flatten(self,site):
        i,j = site
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Ly)
    def get_batch_pairs(self,dx,dy,irange,jrange=None):
        if jrange is None:
            jrange = range(self.Ly)
        ls = []
        for i,j in itertools.product(irange,jrange):
            site1 = i,j
            if self.pbc:
                site2 = (i+dx)%self.Lx,(j+dy)%self.Ly
                ix1,ix2 = self.flatten(site1),self.flatten(site2)
                if ix1>ix2:
                    ix1,ix2 = ix2,ix1
                where = self.flat2site(ix1),self.flat2site(ix2)
                ls.append(where)
                if not(dx==1 and dy==1):
                    continue
                site1,site2 = (i,(j+dy)%self.Ly),((i+dx)%self.Lx,j)
                ix1,ix2 = self.flatten(site1),self.flatten(site2)
                if ix1>ix2:
                    ix1,ix2 = ix2,ix1
                where = self.flat2site(ix1),self.flat2site(ix2)
                ls.append(where)
            elif j+dy<self.Ly:
                where = site1,(i+dx,j+dy)
                ls.append(where)
                if not(dx==1 and dy==1):
                    continue
                where = (i,j+dy),(i+dx,j)
                ls.append(where)
            else:
                pass
        return ls
    def get_batch_deterministic(self,imin,imax,dx,dy):
        if imin is None:
            key = 'pbc'
            irange = range(self.Lx-dx,self.Lx)
        else:
            key = imin,imax
            irange = range(imin,imax+1-dx)
        if key not in self.batched_pairs:
            self.batched_pairs[key] = Batch(deterministic=True) 
        self.batched_pairs[key].pairs += self.get_batch_pairs(dx,dy,irange)
    def get_batch_plq(self,imin,imax,dx,dy):
        key = imin,imax
        irange = range(imin,imax+1-dx)
        if key not in self.batched_pairs:
            self.batched_pairs[key] = Batch(bix=0,tix=self.Lx-1) 
        b = self.batched_pairs[key]
        b.bix = max(b.bix,imax-dx-1)
        b.tix = min(b.tix,imin+dx+1)
        b.plq_types.append((imin,imax-dx,dx+1,dy+1))
        b.pairs += self.get_batch_pairs(dx,dy,irange)
    def get_batch_plq_ver(self,dx,dy):
        key = 'pbc'
        imin,imax = 0,self.Ly-1
        irange,jrange = range(self.Lx-dx,self.Lx),range(self.Ly-dy)
        if key not in self.batched_pairs:
            self.bached_pairs[key] = Batch(bix=0,tix=self.Ly-1,direction='col') 
        b = self.batched_pairs[key]
        b.bix = max(b.bix,imax-dy-1)
        b.tix = min(b.tix,imin+dy+1)
        b.plq_types.append((imin,imax-dy,dy+1,dx+1))
        b.pairs += self.get_batch_pairs(dx,dy,irange,jrange=jrange)
from .tensor_vmc import get_gate2
class Heisenberg(Model2D): 
    def __init__(self,J,h,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        try:
            self.Jx,self.Jy,self.Jz = J
        except TypeError:
            self.Jx,self.Jy,self.Jz = J,J,J
        self.h = h

        self.gate = get_gate2((self.Jx,self.Jy,0.))
        self.order = 'b1,k1,b2,k2'

        if self.deterministic:
            self.batched_pairs = dict()
            self.batch_deterministic_nnh() 
            self.batch_deterministic_nnv() 
        else:
            self.batch_plq_nn()
    def pair_valid(self,i1,i2):
        return True
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
    def pair_coeff(self,site1,site2): # coeff for pair tsr
        return 1.
    def compute_local_energy_eigen(self,config):
        eh = 0.
        ez = 0.
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            s1 = (-1) ** config[self.flatten((i,j))]
            eh += s1 
            if j+1<self.Ly:
                ez += s1 * (-1)**config[self.flatten((i,j+1))]
            else:
                if self.pbc:
                    ez += s1 * (-1)**config[self.flatten((i,0))]
            if i+1<self.Lx:
                ez += s1 * (-1)**config[self.flatten((i+1,j))]
            else:
                if self.pbc:
                    ez += s1 * (-1)**config[self.flatten((0,j))]
        return eh * .5 * self.h + ez * .25 * self.Jz
    def pair_terms(self,i1,i2):
        if i1!=i2:
            return [(1-i1,1-i2,.25*(self.Jx+self.Jy))]
        else:
            return [(1-i1,1-i2,.25*(self.Jx-self.Jy))]
class J1J2(Model2D): # prototypical next nn model
    def __init__(self,J1,J2,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.J1,self.J2 = J1,J2
        self.gate = {None:(get_gate2((1.,1.,0.)),'b1,k1,b2,k2')}

        self.batched_pairs = dict()
        if self.deterministic:
            self.get_batch_deterministic(0,self.Lx-1,0,1)
            self.get_batch_deterministic(0,self.Lx-1,1,0)
            self.get_batch_deterministic(0,self.Lx-1,1,1)
        else:
            self.get_batch_plq(0,self.Lx-1,0,1)
            self.get_batch_plq(0,self.Lx-1,1,0)
            self.get_batch_plq(0,self.Lx-1,1,1)
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def pair_key(self,site1,site2):
        (i1,j1),(i2,j2) = site1,site2
        if abs(i1-i2)==1 and abs(j1-j2)==1:
            i0 = min(site1[0],site2[0],self.Lx-2)
            j0 = min(site1[1],site2[1],self.Ly-2)
            return (i0,j0),(2,2) 
        elif i1==i2:
            return (i1,j1),(1,2)
        elif j1==j2:
            return (i1,j1),(2,1)
        else:
            raise ValueError(f'pair={(site1,site2)}')
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        dx = abs(site2[0]-site1[0])
        dy = abs(site2[1]-site1[1])
        if dx == 0:
            return self.J1
        if dy == 0:
            return self.J1
        return self.J2
    def compute_local_energy_eigen(self,config):
        # NN
        e1 = 0.
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            s1 = (-1) ** config[self.flatten((i,j))]
            if j+1<self.Ly:
                e1 += s1 * (-1)**config[self.flatten((i,j+1))]
            else:
                if self.pbc:
                    e1 += s1 * (-1)**config[self.flatten((i,0))]
            if i+1<self.Lx:
                e1 += s1 * (-1)**config[self.flatten((i+1,j))]
            else:
                if self.pbc:
                    e1 += s1 * (-1)**config[self.flatten((0,j))]
        # next NN
        e2 = 0. 
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            if i+1<self.Lx and j+1<self.Ly: 
                ix1,ix2 = self.flatten((i,j)), self.flatten((i+1,j+1))
                e2 += (-1)**(config[ix1]+config[ix2])
                ix1,ix2 = self.flatten((i,j+1)), self.flatten((i+1,j))
                e2 += (-1)**(config[ix1]+config[ix2])
            else:
                if self.pbc:
                    ix1,ix2 = self.flatten((i,j)), self.flatten(((i+1)%self.Lx,(j+1)%self.Ly))
                    e2 += (-1)**(config[ix1]+config[ix2])
                    ix1,ix2 = self.flatten((i,(j+1)%self.Ly)), self.flatten(((i+1)%self.Lx,j))
                    e2 += (-1)**(config[ix1]+config[ix2])
        return .25 * (e1 * self.J1 + e2 * self.J2) 
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.5,None)]
#class SpinDensity(Hamiltonian):
#    def __init__(self,Lx,Ly):
#        self.Lx,self.Ly = Lx,Ly 
#        self.data = np.zeros((Lx,Ly))
#        self.n = 0.
#    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
#        self.n += 1.
#        for i in range(self.Lx):
#            for j in range(self.Ly):
#                self.data[i,j] += (-1) ** config[self.flatten(i,j)]
#        return 0.,0.,None,None,0. 
#    def _print(self,fname,data):
#        print(fname)
#        print(data)
#class Mz(Hamiltonian):
#    def __init__(self,Lx,Ly):
#        self.Lx,self.Ly = Lx,Ly 
#        self.nsites = Lx * Ly
#        self.data = np.zeros(1)
#        self.n = 0.
#    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
#        self.n += 1.
#
#        data = 0.
#        for ix1 in range(self.nsites):
#            s1 = (-1) ** config[ix1]
#            site1 = self.flat2site(ix1)
#            for ix2 in range(ix1+1,self.nsites):
#                s2 = (-1) ** config[ix2]
#                site2 = self.flat2site(ix2)
#                  
#                dx,dy = site1[0]-site2[0],site1[1]-site2[1]
#                data += s1 * s2 * (-1)**(dx+dy)
#        self.data += data / self.nsites**2
#        return 0.,0.,None,None,0. 
#    def _print(self,fname,data):
#        print(f'fname={fname},data={data[0]}') 
####################################################################################
# sampler 
####################################################################################
class ExchangeSampler2D(ExchangeSampler):
    def __init__(self,Lx,Ly,seed=None,burn_in=0,scheme='hv'):
        self.Lx, self.Ly = Lx,Ly
        self.nsite = Lx * Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.af = None

        assert scheme in ('hv','blks','random')
        self.scheme = scheme
        if scheme=='random':
            self.pairs = [((i,j),(i,j+1)) for i,j in itertools.product(range(Lx),range(Ly-1))]
            self.pairs += [((i,j),(i+1,j)) for i,j in itertools.product(range(Lx-1),range(Ly))]
            self.npair = len(self.pairs)
    def flatten(self,site):
        i,j = site
        return flatten(i,j,self.Ly)
    def flat2site(self,site):
        i,j = site
        return flat2site(i,j,self.Ly)
    def get_pairs(self,i,j,x_bsz,y_bsz):
        if (x_bsz,y_bsz)==(1,2):
            pairs = [((i,j),(i,(j+1)%self.Ly))]
        elif (x_bsz,y_bsz)==(2,1):
            pairs = [((i,j),((i+1)%self.Lx,j))]
        elif (x_bsz,y_bsz)==(2,2):
            bonds_map = {'l':((i,j),((i+1)%self.Lx,j)),
                         'd':((i,j),(i,(j+1)%self.Ly)),
                         'r':((i,(j+1)%self.Ly),((i+1)%self.Lx,(j+1)%self.Ly)),
                         'u':(((i+1)%self.Lx,j),((i+1)%self.Lx,(j+1)%self.Ly)),
                         'x':((i,j),((i+1)%self.Lx,(j+1)%self.Ly)),
                         'y':((i,(j+1)%self.Ly),((i+1)%self.Lx,j))}
            order = 'ldru' 
            pairs = [bonds_map[key] for key in order]
        else:
            raise NotImplementedError
        return pairs
    def sweep_col_from_plq(self,i,cols,x_bsz,y_bsz):
        step = self.rng.choice([-1,1])
        stop = y_bsz-1 if step==1 else self.Ly-y_bsz
        cols,envs = self.af.get_all_envs(cols,-step,stop=stop,inplace=False)
        sweep = range(self.Ly - y_bsz + 1) if step==1 else \
                range(self.Ly - y_bsz,-1,-1)
        for j in sweep: 
            plq = self.af._get_plq_sweep(j,y_bsz,cols,envs,step)
            pairs = self.get_pairs(i,j,x_bsz,y_bsz)
            for site1,site2 in pairs:
                config_sites,config_new = self._new_pair(site1,site2)
                if config_sites is None:
                    continue
                config_sites = self.af.parse_config(config_sites)
                config_new_ = self.af.parse_config(config_new)
                _,py = self.af._new_log_prob_from_plq(plq,(site1,site2),config_sites,config_new_)
                if py is None:
                    continue
                acceptance = np.exp(py - self.px)
                if acceptance < self.rng.uniform(): # reject
                    continue
                # accept, update px & config & env_m
                self.px = py
                self.config = config_new
                cols = self.af.replace_sites(cols,(site1,site2),config_sites)
            cix = (0,j) if step==1 else (j+y_bsz-1,self.Ly-1)
            cols = self.af._contract_cols(cols,cix)
    def sweep_col_from_benv(self,i,x_bsz,y_bsz):
        step = self.rng.choice([-1,1])
        sweep = range(self.Ly - y_bsz + 1) if step==1 else \
                range(self.Ly - y_bsz,-1,-1)
        for j in sweep:
            pairs = self.get_pairs(i,j,x_bsz,y_bsz)
            for site1,site2 in pairs:
                _,config_new = self._new_pair(site1,site2)
                if config_new is None:
                    continue
                config_new_ = self.af.parse_config(config_new)
                py = self.af.log_prob(config_new_,i=i)
                if py is None:
                    continue
                acceptance = np.exp(py - self.px)
                if acceptance < self.rng.uniform(): # reject
                    continue
                self.px = py
                self.config = config_new
    def sweep_row(self,x_bsz,y_bsz):
        step = self.rng.choice([-1,1])
        self.af.free_sweep_cache(step)
        stop = x_bsz-1 if step==1 else self.Lx-x_bsz 
        self.af._get_all_benvs(self.af.parse_config(self.config),-step,stop=stop)
        sweep = range(self.Lx-x_bsz+1) if step==1 else range(self.Lx-x_bsz-1,-1)
        for i in sweep:
            if self.af.from_plq:
                tn = self.af.build_3row_tn(self.af.parse_config(self.config),i,x_bsz)
                self.sweep_col_from_plq(i,tn,x_bsz,y_bsz)
            else:
                self.sweep_col_from_benv(i,x_bsz,y_bsz)
            start = i if step==1 else i+x_bsz-1
            stop = i+x_bsz if step==1 else i-1
            self.af._get_all_benvs(self.af.parse_config(self.config),step,start=start,stop=stop)
    def get_bsz(self):
        if self.scheme=='hv':
            bsz = [(1,2),(2,1)][::self.rng.choice([1,-1])]
        elif self.scheme=='blks':
            bsz = [(2,2)]
        else:
            raise NotImplementedError
        return bsz
    def _sample(self):
        bsz = self.get_bsz()
        for (x_bsz,y_bsz) in bsz: 
            self.sweep_row(x_bsz,y_bsz) 
    def _update_pair(self,site1,site2):
        config_sites,config_new = self._new_pair(site1,site2)
        if config_sites is None:
            return
        py = self.af.log_prob(self.af.parse_config(config_new))
        if py is None:
            return
        acceptance = np.exp(py - self.px)
        if self.rng.uniform() < acceptance: # accept, update px & config & env_m
            self.px = py
            self.config = tuple(config_new) 
    def _sample_deterministic(self):
        bsz = self.get_bsz()
        for x_bsz,y_bsz in bsz:
            imax = self.Lx-1 if self.af.pbc else self.Lx-x_bsz
            jmax = self.Ly-1 if self.af.pbc else self.Ly-y_bsz
            rdir = self.rng.choice([-1,1]) 
            cdir = self.rng.choice([-1,1]) 
            sweep_row = range(0,imax+1) if rdir==1 else range(imax,-1,-1)
            sweep_col = range(0,jmax+1) if cdir==1 else range(jmax,-1,-1)
            for i,j in itertools.product(sweep_row,sweep_col):
                pairs = self.get_pairs(i,j,x_bsz,y_bsz)
                for site1,site2 in pairs:
                    self._update_pair(site1,site2)
        self.af.update_cache(self.af.parse_config(self.config))

def get_product_state(Lx,Ly,config=None,bdim=1,eps=0.,pdim=2,normalize=True):
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [bdim] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (pdim,)

            if config is None:
                data = np.ones(shape) 
            else:
                data = np.zeros(shape) 
                ix = flatten(i,j,Ly)
                ix = config[ix]
                data[(0,)*(len(shape)-1)+(ix,)] = 1.
            data += eps * np.random.rand(*shape)
            if normalize:
                data /= np.linalg.norm(data)
            row.append(data)
        arrays.append(row)
    from .tensor_2d import PEPS
    return PEPS(arrays)
def expand_peps(peps,Dnew,eps=0.):
    tn = peps.copy()
    for i,j in itertools.product(range(tn.Lx),range(tn.Ly)):
        T = tn[i,j]
        data = T.data
        shape = data.shape
        ndim = len(shape)

        shape_new = (Dnew,)*(ndim-1) + (data.shape[-1],)
        data_new = eps * np.random.rand(*shape_new)
        slices = tuple([slice(D) for D in shape])
        data_new[slices] = data

        T.modify(data=data_new)
    return tn

from .tensor_core import Tensor,rand_uuid
def peps2pbc(peps):
    vbonds = [rand_uuid() for j in range(peps.Ly)]
    hbonds = [rand_uuid() for i in range(peps.Lx)]
    # i = 0
    for j in range(peps.Ly):
        tsr = peps[0,j]
        bdim,pdim = tsr.data.shape[0],tsr.data.shape[-1]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        d = vbonds[j]
        if j==0:
            u,r,p = tsr.inds
            l = hbonds[0]
        elif j==peps.Ly-1:
            u,l,p = tsr.inds
            r = hbonds[0]
        else:
            u,r,l,p = tsr.inds
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)

    for i in range(1,peps.Lx-1):
        tsr = peps[i,0]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        u,r,d,p = tsr.inds
        l = hbonds[i]
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)

        tsr = peps[i,peps.Ly-1]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        u,d,l,p = tsr.inds
        r = hbonds[i]
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)

    # i = Lx-1
    for j in range(peps.Ly):
        tsr = peps[peps.Lx-1,j]
        bdim,pdim = tsr.data.shape[0],tsr.data.shape[-1]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        u = vbonds[j]
        if j==0:
            r,d,p = tsr.inds
            l = hbonds[peps.Lx-1]
        elif j==peps.Ly-1:
            d,l,p = tsr.inds
            r = hbonds[peps.Lx-1]
        else:
            r,d,l,p = tsr.inds
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)
    return peps
