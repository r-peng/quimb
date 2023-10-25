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
####################################################################################
# amplitude fxns 
####################################################################################
class AmplitudeFactory2D(AmplitudeFactory):
    def __init__(self,psi,blks=None,phys_dim=2,backend='numpy',pbc=False,deterministic=False,**compress_opts):
        # init wfn
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.nsite = self.Lx * self.Ly
        self.sites = list(itertools.product(range(self.Lx),range(self.Ly)))
        psi.add_tag('KET')
        self.set_psi(psi) # current state stored in self.psi
        self.backend = backend 

        self.data_map = self.get_data_map(phys_dim)
        self.wfn2backend()

        # init contraction
        self.compress_opts = compress_opts
        self.pbc = pbc 
        self.deterministic = deterministic
        self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2

        if blks is None:
            blks = [self.sites]
        self.site_map = self.get_site_map(blks)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict(blks)
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)
        self.nparam = len(self.get_x())
        self.is_tn = True
##### wfn methods #####
    def flatten(self,site):
        i,j = site
        return flatten(i,j,self.Ly) 
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
    def get_cache(self,direction,step):
        if direction=='row':
            cache = self.cache_bot if step==1 else self.cache_top
        else:
            cache = self.cache_left if step==1 else self.cache_right
        return cache
    def cache_key(self,config,i,direction,step):
        if direction=='row':
            if step==1: # bottom env
                return config[:(i+1)*self.Ly] 
            else: # top env
                return config[i*self.Ly:]
        else: 
            cols = range(i+1) if step==1 else range(i,self.Ly)
            return tuple([config[slc] for slc in [slice(col,self.nsite,self.Ly) for col in cols]])
    def _new_cache(self,config,cache_bot,cache_top):
        cache_bot_new = dict()
        for i in range(self.rix1+1):
            key = config[:(i+1)*self.Ly]
            cache_bot_new[key] = cache_bot[key]
        cache_top_new = dict()
        for i in range(self.rix2,self.Lx):
            key = config[i*self.Ly:]
            cache_top_new[key] = cache_top[key]
        return cache_bot_new,cache_top_new
    def update_cache(self,config):
        self.cache_bot,self.cache_top = self._new_cache(config,self.cache_bot,self.cache_top)
    def set_psi(self,psi):
        self.psi = psi

        self.cache_bot = dict()
        self.cache_top = dict()
        self.cache_left = dict()
        self.cache_right = dict()
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
    def get_benv(self,i,row,env_prev,config,step,cache=None,direction='row'):
        cache = self.get_cache(direction,step) if cache is None else cache 
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
    def get_env_prev(self,config,i,step,cache=None,direction='row'):
        if step==1 and i==0:
            return None 
        Lx = self.Lx if direction=='row' else self.Ly
        if step==-1 and i==Lx-1:
            return None
        cache = self.get_cache(direction,step) if cache is None else cache 
        key = self.cache_key(config,i-step,direction,step)
        return cache[key]
    def _get_all_benvs(self,config,step,psi=None,cache=None,start=None,stop=None,append='',direction='row'):
        psi = self.psi if psi is None else psi
        cache = self.get_cache(direction,step) if cache is None else cache 

        Lx = self.Lx if direction=='row' else self.Ly
        if step==1:
            start = 0 if start is None else start
            stop = Lx-1 if stop is None else stop
        else:
            start = Lx-1 if start is None else start
            stop = 0 if stop is None else stop 
        env_prev = self.get_env_prev(config,start,step,cache=cache,direction=direction)
        for i in range(start,stop,step):
            row = self.get_mid_env(i,config,append=append,psi=psi,direction=direction)
            env_prev = self.get_benv(i,row,env_prev,config,step,cache=cache,direction=direction)
        return env_prev
    def get_all_benvs(self,config,psi=None,cache_bot=None,cache_top=None,x_bsz=1,
                      compute_bot=True,compute_top=True,imax=None,imin=None,direction='row'):
        psi = self.psi if psi is None else psi
        cache_bot = self.get_cache(direction,1) if cache_bot is None else cache_bot
        cache_top = self.get_cache(direction,-1) if cache_top is None else cache_top
        Lx = self.Lx if direction=='row' else self.Ly

        env_bot = None
        env_top = None
        if compute_bot: 
            stop = Lx-x_bsz if imax is None else imax+1 
            env_bot = self._get_all_benvs(config,1,psi=psi,cache=cache_bot,stop=stop,direction=direction)
        if compute_top:
            stop = x_bsz-1 if imin is None else imin-1
            env_top = self._get_all_benvs(config,-1,psi=psi,cache=cache_top,stop=stop,direction=direction)
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
    def _get_plq_forward(self,j,y_bsz,cols,renvs,direction='col'):
        # lenv from col, renv from renv
        if direction=='col':
            Ly = self.Ly 
            col_tag = self.col_tag
        else:
            Ly = self.Lx
            col_tag = self.row_tag
        first_col = col_tag(0)
        tags = [col_tag(j+ix) for ix in range(y_bsz)]
        plq = cols.select(tags,which='any',virtual=False)
        if j>0:
            other = plq
            plq = cols.select(first_col,virtual=False)
            plq.add_tensor_network(other,virtual=False)
        if j<Ly - y_bsz:
            plq.add_tensor_network(renvs[j+y_bsz],virtual=False)
        plq.view_like_(cols)
        return plq
    def _get_plq_backward(self,j,y_bsz,cols,lenvs,direction='col'):
        # lenv from lenv, renv from cols
        if direction=='col':
            Ly = self.Ly 
            col_tag = self.col_tag
        else:
            Ly = self.Lx
            col_tag = self.row_tag
        last_col = col_tag(Ly-1)
        tags = [col_tag(j+ix) for ix in range(y_bsz)] 
        plq = cols.select(tags,which='any',virtual=False)
        if j>0:
            other = plq
            plq = lenvs[j-1]
            plq.add_tensor_network(other,virtual=False)
        if j<Ly - y_bsz:
            plq.add_tensor_network(cols.select(last_col,virtual=False),virtual=False)
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
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None,direction='row'):
        psi = self.psi if psi is None else psi
        cache_bot = self.get_cache(direction,1) if cache_bot is None else cache_bot
        cache_top = self.get_cache(direction,-1) if cache_top is None else cache_top
        Lx = self.Lx if direction=='row' else self.Ly
        try:
            tn = self.get_mid_env(i,config,psi=psi,direction=direction)
            for ix in range(1,x_bsz):
                tn.add_tensor_network(self.get_mid_env(i+ix,config,psi=psi,direction=direction),virtual=False)
            if i>0:
                other = tn 
                tn = cache_bot[self.cache_key(config,i-1,direction,1)].copy()
                tn.add_tensor_network(other,virtual=False)
            if i+x_bsz<Lx:
                tn.add_tensor_network(cache_top[self.cache_key(config,i+x_bsz,direction,-1)],virtual=False)
        except AttributeError:
            tn = None
        return tn 
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,cache_bot=None,cache_top=None,imin=None,imax=None,direction='row'):
        if direction=='row':
            Lx = self.Lx
            direction_ = 'col'
        else:
            Lx = self.Ly
            direction_ = 'row'
        imin = 0 if imin is None else imin
        imax = Lx-x_bsz if imax is None else imax
        psi = self.psi if psi is None else psi
        cache_bot = self.get_cache(direction,1) if cache_bot is None else cache_bot
        cache_top = self.get_cache(direction,-1) if cache_top is None else cache_top
        plq = dict()
        for i in range(imin,imax+1):
            cols = self.build_3row_tn(config,i,x_bsz,psi=psi,cache_bot=cache_bot,cache_top=cache_top,direction=direction)
            #exit()
            if cols is not None:
                plq = self.update_plq_from_3row(plq,cols,i,x_bsz,y_bsz,psi=psi,direction=direction_)
        return plq
    def unsigned_amplitude(self,config,cache_bot=None,cache_top=None,to_numpy=True): 
        # always contract rows into the middle
        env_bot,env_top = self.get_all_benvs(config,cache_bot=cache_bot,cache_top=cache_top,imax=self.rix1,imin=self.rix2)

        if env_bot is None and env_top is None:
            cx = 0. if to_numpy else None
            return cx 

        try:
            tn = env_bot.copy()
            tn.add_tensor_network(env_top,virtual=False)
        except AttributeError:
            cx = 0. if to_numpy else None
            return cx 

        cx = safe_contract(tn)
        if to_numpy:
            cx = 0. if cx is None else tensor2backend(cx,'numpy')
        return cx  
##### hamiltonian methods #######
    def pair_energy_deterministic(self,site1,site2,cache_bot=None,cache_top=None):
        ix1,ix2 = [self.flatten(site) for site in (site1,site2)]
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return None 
        ex = [] 
        coeff_comm = self.intermediate_sign(self.config,ix1,ix2) * self.model.pair_coeff(site1,site2)
        for i1_new,i2_new,coeff in self.model.pair_terms(i1,i2):
            config_new = list(self.config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = self.parse_config(tuple(config_new))
            sign_new = self.config_sign(config_new)

            cx_new = self.unsigned_amplitude(config_new,cache_bot=cache_bot,cache_top=cache_top,to_numpy=False)
            if cx_new is None:
                continue
            ex.append(coeff * sign_new * cx_new)
        if len(ex)==0:
            return None
        return sum(ex) * coeff_comm
    def batch_pair_energies_from_plq(self,batch_key,new_cache=False):
        fxn = self.batch_pair_energies_from_plq_pbc if self.pbc else \
              self.batch_pair_energies_from_plq_obc
        return fxn(batch_key,new_cache=new_cache) 
    def batch_pair_energies_from_plq_obc(self,batch_key,new_cache=False):
        bix,tix,plq_types,pairs,direction = self.model.batched_pairs[batch_key]
        assert direction=='row'
        cache_bot = dict() if new_cache else None
        cache_top = dict() if new_cache else None
        self._get_all_benvs(self.config,1,cache=cache_bot,stop=bix+1)
        self._get_all_benvs(self.config,-1,cache=cache_top,stop=tix-1)

        # form plqs
        plq = dict()
        for imin,imax,x_bsz,y_bsz in plq_types:
            plq.update(self.get_plq_from_benvs(self.config,x_bsz,y_bsz,cache_bot=cache_bot,cache_top=cache_top,imin=imin,imax=imax))

        # compute energy numerator 
        ex,cx = self.pair_energies_from_plq(plq,pairs)
        return ex,cx,plq
    def batch_pair_energies_from_plq_pbc(self,batch_key,new_cache=False):
        bix,tix,plq_types,pairs,direction = self.model.batched_pairs[batch_key]
        if direction=='row':
            psi = self.psi
        else:
            psi = self.psi.copy()
            psi.reorder('col',inplace=True)
        cache_bot = dict() if new_cache else None
        cache_top = dict() if new_cache else None
        self._get_all_benvs(self.config,1,psi=psi,cache=cache_bot,stop=bix+1,direction=direction)
        self._get_all_benvs(self.config,-1,psi=psi,cache=cache_top,stop=tix-1,direction=direction)

        # form 3row_tn
        plq = dict()
        for imin,imax,x_bsz,_ in plq_types:
            for i in range(imin,imax+1):
                plq[i,x_bsz] = self.build_3row_tn(self.config,i,x_bsz,psi=psi,cache_bot=cache_bot,cache_top=cache_top,direction=direction) 

        def _contract_3row(self,tn,imin,x_bsz):
            for i in range(imin,imin+x_bsz):
                tn = self._contract_boundary_single(tn,i,i-1,direction=direction) 
            return safe_contract_(tn) 

        ex,cx = dict(),dict()
        sign = self.config_sign(self.config)
        for where in pairs:
            i = min([i for (i,_) in where])
            x_bsz = abs(where[0][1]-where[1][1])
            tn = plq.get((i,x_bsz),None)
            if tn is None:
                continue
            # cij
            cx[where] = _contract_3row(tn.copy(),i,x_bsz) * sign,(i,x_bsz)

            # ex_ij
            ix1,ix2 = [self.flatten(site) for site in where]
            i1,i2 = self.config[ix1],self.config[ix2] 
            if not self.model.pair_valid(i1,i2):
                for tag in self.model.gate:
                    ex[where,tag] = 0,0
                continue
            coeff_comm = self.model.pair_coeff(*where) * self.intermediate_sign(self.config,ix1,ix2)
            for i1_new,i2_new,coeff in self.model.pair_terms(i1,i2):
                config_new = list(self.config)
                config_new[ix1] = i1_new
                config_new[ix2] = i2_new 
                config_new = self.parse_config(tuple(config_new)) 
                sign_new = self.config_sign(config_new)
                eij = self.replace_sites(tn.copy(),where,(i1_new,i2_new))
                eij = _contract_3row(eij,i,x_bsz) * coeff * coeff_comm *  sign * sign_new 
                ex[where,tag] = eij,eij/cij 
        return ex,cx,None
    def batch_pair_energies_deterministic(self,batch_key,new_cache=False):
        cache_bot = dict() if new_cache else None
        cache_top = dict() if new_cache else None

        ex = dict() 
        for site1,site2 in self.model.batched_pairs[batch_key]:
            eij = self.pair_energy_deterministic(site1,site2,cache_bot=cache_bot,cache_top=cache_top)
            if eij is not None:
                ex[site1,site2] = eij
        return ex
####################################################################
# models
####################################################################
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
    #def pairs_nn(self,d=1):
    #    ls = [] 
    #    for i,j in itertools.product(range(self.Lx),range(self.Ly)):
    #        if j+d<self.Ly:
    #            where = (i,j),(i,j+d)
    #            ls.append(where)
    #        else:
    #            if _PBC:
    #                where = (i,(j+d)%self.Ly),(i,j)
    #                ls.append(where)
    #        if i+d<self.Lx:
    #            where = (i,j),(i+d,j)
    #            ls.append(where)
    #        else:
    #            if _PBC:
    #                where = ((i+d)%self.Lx,j),(i,j)
    #                ls.append(where)
    #    return ls
    #def pairs_diag(self):
    #    ls = [] 
    #    for i in range(self.Lx):
    #        for j in range(self.Ly):
    #            if i+1<self.Lx and j+1<self.Ly:
    #                where = (i,j),(i+1,j+1)
    #                ls.append(where)
    #                where = (i,j+1),(i+1,j)
    #                ls.append(where)
    #            else:
    #                if _PBC:
    #                    ix1,ix2 = self.flatten((i,j)),self.flatten(((i+1)%self.Lx,(j+1)%self.Ly))
    #                    where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
    #                    ls.append(where)
    #                    
    #                    ix1,ix2 = self.flatten((i,(j+1)%self.Ly)),self.flatten(((i+1)%self.Lx,j))
    #                    where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
    #                    ls.append(where)
    #    return ls
    def batch_deterministic_nnh(self,d=1):
        for i in range(self.Lx):
            ls = self.batched_pairs.get((i,i),[])
            for j in range(self.Ly):
                if j+d<self.Ly:
                    where = (i,j),(i,j+d)
                    ls.append(where)
                elif self.pbc:
                    where = (i,(j+d)%self.Ly),(i,j)
                    ls.append(where)
                else:
                    pass
            self.batched_pairs[i,i] = ls
    def batch_deterministic_nnv(self,d=1):
        for i in range(self.Lx-d):
            ls = self.batched_pairs.get((i,i+d),[])
            for j in range(self.Ly):
                where = (i,j),(i+d,j)
                ls.append(where)
            self.batched_pairs[i,i+d] = ls
        if not self.pbc:
            return
        ls = self.batched_pairs.get('pbc',[]) 
        for (i,j) in itertools.product(range(self.Lx-d,self.Lx),range(self.Ly)):
            where = ((i+d)%self.Lx,j),(i,j)
            ls.append(where)
        self.batched_pairs['pbc'] = ls
    def batch_deterministic_diag(self):
        for i in range(self.Lx-1):
            ls = self.batched_pairs.get((i,i+1),[])
            for j in range(self.Ly):
                if i+1<self.Lx and j+1<self.Ly:
                    where = (i,j),(i+1,j+1)
                    ls.append(where)
                    where = (i,j+1),(i+1,j)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (i,j),(i+1,(j+1)%self.Ly)
                        ls.append(where)
                        
                        where = (i,(j+1)%self.Ly),(i+1,j)
                        ls.append(where)
            self.batched_pairs[i,i+1] = ls
        if not self.pbc:
            return
        ls = self.batched_pairs.get('pbc',[])
        for j in range(self.Ly):
            where = (0,(j+1)%self.Ly),(self.Lx-1,j)
            ls.append(where)
            
            where = (0,j),(self.Lx-1,(j+1)%self.Ly)
            ls.append(where)
        self.batched_pairs['pbc'] = ls 
    def batch_plq_nn(self):
        # OBC plqs
        self.batched_pairs = dict() 
        batchsize = max(self.Lx // self.nbatch,2)
        pbc_terms = []
        for i in range(self.Lx):
            batch_idx = i // batchsize
            if batch_idx not in self.batched_pairs:
                self.batched_pairs[batch_idx] = [],[] 
            rows,pairs = self.batched_pairs[batch_idx]
            rows.append(i)
            if i+1 < self.Lx:
                rows.append(i+1)
            for j in range(self.Ly):
                if j+1<self.Ly:
                    where = (i,j),(i,j+1)
                    pairs.append(where)
                elif self.pbc:
                    where = (i,(j+1)%self.Ly),(i,j)
                    pairs.append(where)
                else:
                    pass 

                if i+1<self.Lx:
                    where = (i,j),(i+1,j)
                    pairs.append(where)
                elif self.pbc:
                    where = ((i+1)%self.Lx,j),(i,j)
                    pbc_terms.append(where)
                else:
                    pass
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            bix,tix = max(0,imax-1),min(imin+1,self.Lx-1) # bot_ix,top_ix 
            plq_types = [(imin,imax,1,2), (imin,imax-1,2,1)] # i0_min,i0_max,x_bsz,y_bsz
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs,'row' 

        if not self.pbc:
            if RANK==0:
                print('nbatch=',len(self.batched_pairs))
            return
        # adding PBC terms
        plq_types = (self.Ly-2,1,1,self.Lx),
        self.batch_pairs['ver'] = self.Ly-2,1,plq_types,pbc_terms,'col' 
    def batch_plq_diag(self):
        self.batched_pairs = dict()
        batchsize = max(self.Lx // self.nbatch, 2)
        pbc_ver = []
        pbc_diag = []
        for i in range(self.Lx):
            batch_idx = i // batchsize
            if batch_idx not in self.batched_pairs:
                self.batched_pairs[batch_idx] = [],[] 
            rows,pairs = self.batched_pairs[batch_idx]
            rows.append(i)
            if i+1 < self.Lx:
                rows.append(i+1)
            for j in range(self.Ly):
                if j+1<self.Ly: # NN
                    where = (i,j),(i,j+1)
                    pairs.append(where)
                elif self.pbc:
                    where = (i,(j+1)%self.Ly),(i,j)
                    pairs.append(where)
                else:
                    pass 

                if i+1<self.Lx: # NN
                    where = (i,j),(i+1,j)
                    pairs.append(where)
                elif self.pbc:
                    where = ((i+1)%self.Lx,j),(i,j)
                    pbc_ver.append(where)
                else:
                    pass

                if i+1<self.Lx and j+1<self.Ly: # diag
                    where = (i,j),(i+1,j+1)
                    pairs.append(where)
                    where = (i,j+1),(i+1,j)
                    pairs.append(where)
                elif i+1<self.Ly and self.pbc:
                    where = (i,j),(i+1,(j+1)%self.Ly) 
                    pairs.append(where)
                    where = (i,(j+1)%self.Ly),(i+1,j)
                    pairs.append(where)
                elif j+1<self.Ly and self.pbc:
                    where = ((i+1)%self.Lx,j+1),(i,j)
                    pbc_ver.append(where)
                    where = ((i+1)%self.Lx,j),(i,j+1)
                    pbc_ver.append(where)
                elif self.pbc:
                    where = ((i+1)%self.Lx,(j+1)%self.Ly),(i,j) 
                    pbc_diag.append(where)
                    where = ((i+1)%self.Lx,j),(i,(j+1)%self.Ly)
                    pbc_diag.append(where)
                else:
                    pass 
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            plq_types = (imin,imax-1,2,2),
            bix,tix = max(0,imax-2),min(imin+2,self.Lx-1)
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs,'row' # bot_ix,top_ix,pairs 
        #self.plq_sz = (2,2),
        if not self.pbc:
            if RANK==0:
                print('nbatch=',len(self.batched_pairs))
            return
        # adding PBC terms
        plq_types = (self.Ly-2,1,2,self.Lx),
        self.batch_pairs['ver'] = self.Ly-2,1,plq_types,pbc_ver,'col' 
        self.batch_pairs['deterministic'] = pbc_diag
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

        if self.deterministic:
            self.batched_pairs = dict()
            self.batch_deterministic_nnh() 
            self.batch_deterministic_nnv() 
            self.batch_deterministic_diag() 
        else:
            self.batch_plq_diag()
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def pair_key(self,site1,site2):
        i0 = min(site1[0],site2[0],self.Lx-2)
        j0 = min(site1[1],site2[1],self.Ly-2)
        return (i0,j0),(2,2) 
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
        return [(1-i1,1-i2,.5)]
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
        self.scheme = scheme
    def flatten(self,site):
        i,j = site
        return flatten(i,j,self.Ly)
    def flat2site(self,site):
        i,j = site
        return flat2site(i,j,self.Ly)
    def _new_pair(self,site1,site2):
        ix1,ix2 = self.flatten(site1),self.flatten(site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if i1==i2: # continue
            return (None,) * 2
        i1_new,i2_new = self.propose_new_pair(i1,i2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        return (i1_new,i2_new),tuple(config_new)
    def _update_pair(self,site1,site2,plq,cols):
        config_sites,config_new = self._new_pair(site1,site2)
        if config_sites is None:
            return plq,cols
        config_sites = self.af.parse_config(config_sites)
        self.af.config_new = config_new
        plq_new,py = self.af._new_log_prob_from_plq(plq,(site1,site2),config_sites)
        if py is None:
            return plq,cols
        acceptance = np.exp(py - self.px)
        if acceptance < self.rng.uniform(): # reject
            return plq,cols
        # accept, update px & config & env_m
        self.px = py
        self.config = config_new
        cols = self.af.replace_sites(cols,(site1,site2),config_sites)
        return plq_new,cols 
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
    def update_plq(self,i,j,x_bsz,y_bsz,plq,cols):
        pairs = self.get_pairs(i,j,x_bsz,y_bsz)
        for site1,site2 in pairs:
            plq,cols = self._update_pair(site1,site2,plq,cols) 
        return cols
    def sweep_col_forward(self,i,cols,x_bsz,y_bsz):
        cols,renvs = self.af.get_all_envs(cols,-1,stop=y_bsz-1,inplace=False)
        for j in range(self.Ly - y_bsz + 1): 
            plq = self.af._get_plq_forward(j,y_bsz,cols,renvs)
            cols = self.update_plq(i,j,x_bsz,y_bsz,plq,cols) 
            cols = self.af._contract_cols(cols,(0,j))
    def sweep_col_backward(self,i,cols,x_bsz,y_bsz):
        cols,lenvs = self.af.get_all_envs(cols,1,stop=self.Ly-y_bsz,inplace=False)
        for j in range(self.Ly - y_bsz,-1,-1): # Ly-1,...,1
            plq = self.af._get_plq_backward(j,y_bsz,cols,lenvs)
            cols = self.update_plq(i,j,x_bsz,y_bsz,plq,cols) 
            cols = self.af._contract_cols(cols,(j+y_bsz-1,self.Ly-1))
    def sweep_col_pbc(self,i,cols,x_bsz,y_bsz,step):
        sweep = range(self.Ly-y_bsz+1) if step==1 else range(self.Ly-y_bsz,-1,-1)
        for j in sweep:
            pairs = self.get_pairs(i,j,x_bsz,y_bsz)
            for site1,site2 in pairs:
                config_sites,config_new = self._new_pair(site1,site2)
                if config_sites is None:
                    continue
                config_sites = self.af.parse_config(config_sites)
                self.af.config_new = config_new
                cols_new,py = af._new_log_prob_pbc(cols,(site1,site2),config_sites,i,x_bsz) 
                if py is None:
                    continue
                acceptance = np.exp(py - self.px)
                if acceptance < self.rng.uniform(): # reject
                    continue
                # accept, update px & config & env_m
                self.px = py
                self.config = config_new
                cols = cols_new
    def sweep_row_forward(self,x_bsz,y_bsz):
        self.af.cache_bot = dict()
        self.af._get_all_benvs(self.af.parse_config(self.config),-1,stop=x_bsz-1)

        cdir = self.rng.choice([-1,1]) 
        if self.af.pbc:
            sweep_col = functools.partialmethod(self.sweep_col_pbc,step=cdir) 
        else:
            sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax+1):
            tn = self.af.build_3row_tn(self.af.parse_config(self.config),i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)

            self.af._get_all_benvs(self.af.parse_config(self.config),1,start=i,stop=i+x_bsz)
    def sweep_row_backward(self,x_bsz,y_bsz):
        self.af.cache_top = dict()
        self.af._get_all_benvs(self.af.parse_config(self.config),1,stop=self.Lx-x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax,-1,-1):
            tn = self.af.build_3row_tn(self.af.parse_config(self.config),i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)

            self.af._get_all_benvs(self.af.parse_config(self.config),-1,stop=i-1,start=i+x_bsz-1)
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
            sweep_fn = {0:self.sweep_row_forward,
                        1:self.sweep_row_backward}[self.rng.choice([0,1])] 
            sweep_fn(x_bsz,y_bsz) 
    def update_plq_deterministic(self,i,j,x_bsz,y_bsz):
        pairs = self.get_pairs(i,j,x_bsz,y_bsz)
        for (site1,site2) in pairs:
            config_sites,config_new = self._new_pair(site1,site2)
            if config_sites is None:
                continue
            cy = self.af.unsigned_amplitude(self.af.parse_config(config_new))
            if cy is None:
                continue
            py = np.log(tensor2backend(cy,'numpy')**2)
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
                self.update_plq_deterministic(i,j,x_bsz,y_bsz)
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
