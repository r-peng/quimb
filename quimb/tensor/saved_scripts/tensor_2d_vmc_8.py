import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import sys
this = sys.modules[__name__]
from .tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
)
from .tensor_vmc import AmplitudeFactory as AmplitudeFactory_
def set_options(pbc=False,deterministic=False):
    this._PBC = pbc
    this._DETERMINISTIC = deterministic 
def flatten(i,j,Ly): # flattern site to row order
    return i*Ly+j
def flat2site(ix,Ly): # ix in row order
    return ix//Ly,ix%Ly
####################################################################################
# amplitude fxns 
####################################################################################
class AmplitudeFactory(AmplitudeFactory_):
    def __init__(self,psi,blks=None,phys_dim=2,backend='numpy',**compress_opts):
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
        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        if self.deterministic:
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
    def site_tag(self,site):
        return self.psi.site_tag(*site)
    def site_tags(self,site):
        return self.site_tag(site),self.psi.row_tag(site[0]),self.col_tag(site[1])
    def site_ind(self,site):
        return self.psi.site_ind(*site)
    def col_tag(self,col):
        return self.psi.col_tag(col)
    def plq_sites(self,plq_key):
        (i0,j0),(x_bsz,y_bsz) = plq_key
        sites = list(itertools.product(range(i0,i0+x_bsz),range(j0,j0+y_bsz)))
        return sites
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
##################################################################################################
# contraction methods                                          
##################################################################################################
    def get_mid_env(self,i,config,append='',psi=None):
        psi = self.psi if psi is None else psi 
        row = psi.select(psi.row_tag(i),virtual=False)
        key = config[i*self.Ly:(i+1)*self.Ly]
        # compute mid env for row i
        for j in range(self.Ly-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(key[j],(i,j),append=append),virtual=True)
        return row
    def contract_mid_env(self,i,row):
        try: 
            for j in range(self.Ly-1,-1,-1):
                row.contract_tags(row.site_tag(i,j),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def compress_row_pbc(self,tn,i):
        for j in range(self.Ly): # compress between j,j+1
            tn.compress_between(tn.site_tag(i,j),tn.site_tag(i,(j+1)%self.Ly),
                                **self.compress_opts)
        return tn
    def compress_row_obc(self,tn,i):
        tn.canonize_row(i,sweep='left')
        for j in range(self.Ly-1):
            self.tensor_compress_bond(tn[i,j],tn[i,j+1],absorb='right')        
        return tn
    def contract_boundary_single(self,tn,i,iprev):
        for j in range(self.Ly):
            tag1,tag2 = tn.site_tag(iprev,j),tn.site_tag(i,j)
            tn.contract_((tag1,tag2),which='any')
        if _PBC:
            return self.compress_row_pbc(tn,i)
        else:
            return self.compress_row_obc(tn,i)
    def get_bot_env(self,i,row,env_prev,config,cache=None):
        # contract mid env for row i with prev bot env 
        key = config[:(i+1)*row.Ly]
        cache = self.cache_bot if cache is None else cache
        if key in cache: # reusable
            return cache[key]
        row = self.contract_mid_env(i,row)
        if i==0:
            cache[key] = row
            return row
        if row is None:
            cache[key] = row
            return row
        if env_prev is None:
            cache[key] = None 
            return None
        tn = env_prev.copy()
        tn.add_tensor_network(row,virtual=False)
        try:
            tn = self.contract_boundary_single(tn,i,i-1)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_all_bot_envs(self,config,psi=None,cache=None,imin=None,imax=None,append=''):
        imin = 0 if imin is None else imin
        imax = self.Lx-2 if imax is None else imax
        psi = self.psi if psi is None else psi
        cache = self.cache_bot if cache is None else cache
        env_prev = None if imin==0 else cache[config[:imin*self.Ly]] 
        for i in range(imin,imax+1):
            row = self.get_mid_env(i,config,append=append,psi=psi)
            env_prev = self.get_bot_env(i,row,env_prev,config,cache=cache)
        return env_prev
    def get_top_env(self,i,row,env_prev,config,cache=None):
        # contract mid env for row i with prev top env 
        key = config[i*row.Ly:]
        cache = self.cache_top if cache is None else cache
        if key in cache: # reusable
            return cache[key]
        row = self.contract_mid_env(i,row)
        if i==row.Lx-1:
            cache[key] = row
            return row
        if row is None:
            cache[key] = row
            return row
        if env_prev is None:
            cache[key] = None 
            return None
        tn = row
        tn.add_tensor_network(env_prev,virtual=False)
        try:
            tn = self.contract_boundary_single(tn,i,i+1)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_all_top_envs(self,config,psi=None,cache=None,imin=None,imax=None,append=''):
        imin = 1 if imin is None else imin
        imax = self.Lx - 1 if imax is None else imax
        psi = self.psi if psi is None else psi 
        cache = self.cache_top if cache is None else cache
        env_prev = None if imax==self.Lx-1 else cache[config[(imax+1)*self.Ly:]]
        for i in range(imax,imin-1,-1):
             row = self.get_mid_env(i,config,append=append,psi=psi)
             env_prev = self.get_top_env(i,row,env_prev,config,cache=cache)
        return env_prev
    def get_all_benvs(self,config,psi=None,cache_bot=None,cache_top=None,x_bsz=1,
                      compute_bot=True,compute_top=True,rix1=None,rix2=None):
        psi = self.psi if psi is None else psi
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top

        env_bot = None
        env_top = None
        if compute_bot: 
            imax = self.Lx-1-x_bsz if rix1 is None else rix1 
            env_bot = self.get_all_bot_envs(config,psi=psi,cache=cache_bot,imax=imax)
        if compute_top:
            imin = x_bsz if rix2 is None else rix2 
            env_top = self.get_all_top_envs(config,psi=psi,cache=cache_top,imin=imin)
        #print(imin,imax)
        return env_bot,env_top
    def _contract_cols(self,cols,js):
        tags = [self.col_tag(j) for j in js]
        cols ^= tags
        return cols
    def get_all_lenvs(self,cols,jmax=None,inplace=False):
        tmp = cols if inplace else cols.copy()
        jmax = self.Ly-2 if jmax is None else jmax
        first_col = self.col_tag(0)
        lenvs = [None] * self.Ly
        for j in range(jmax+1): 
            try:
                tmp = self._contract_cols(tmp,(0,j))
                lenvs[j] = tmp.select(first_col,virtual=False)
            except (ValueError,IndexError):
                return cols,lenvs
        return cols,lenvs
    def get_all_renvs(self,cols,jmin=None,inplace=False):
        tmp = cols if inplace else cols.copy()
        jmin = 1 if jmin is None else jmin
        last_col = self.col_tag(self.Ly-1)
        renvs = [None] * self.Ly
        for j in range(self.Ly-1,jmin-1,-1): 
            try:
                tmp = self._contract_cols(tmp,(j,self.Ly-1))
                renvs[j] = tmp.select(last_col,virtual=False)
            except (ValueError,IndexError):
                return cols,renvs
        return cols,renvs
    def _get_plq_forward(self,j,y_bsz,cols,renvs):
        # lenv from col, renv from renv
        first_col = self.col_tag(0)
        tags = [self.col_tag(j+ix) for ix in range(y_bsz)]
        plq = cols.select(tags,which='any',virtual=False)
        if j>0:
            other = plq
            plq = cols.select(first_col,virtual=False)
            plq.add_tensor_network(other,virtual=False)
        if j<self.Ly - y_bsz:
            plq.add_tensor_network(renvs[j+y_bsz],virtual=False)
        plq.view_like_(cols)
        return plq
    def _get_plq_backward(self,j,y_bsz,cols,lenvs):
        # lenv from lenv, renv from cols
        last_col = self.col_tag(self.Ly-1)
        tags = [self.col_tag(j+ix) for ix in range(y_bsz)] 
        plq = cols.select(tags,which='any',virtual=False)
        if j>0:
            other = plq
            plq = lenvs[j-1]
            plq.add_tensor_network(other,virtual=False)
        if j<self.Ly - y_bsz:
            plq.add_tensor_network(cols.select(last_col,virtual=False),virtual=False)
        plq.view_like_(cols)
        return plq
    def _get_plq(self,j,y_bsz,cols,lenvs,renvs):
        # lenvs from lenvs, renvs from renvs
        tags = [self.col_tag(j+ix) for ix in range(y_bsz)]
        plq = cols.select(tags,which='any',virtual=False)
        if j>0:
            other = plq
            plq = lenvs[j-1]
            plq.add_tensor_network(other,virtual=False)
        if j<self.Ly - y_bsz:
            plq.add_tensor_network(renvs[j+y_bsz],virtual=False)
        plq.view_like_(cols)
        return plq
    def update_plq_from_3row(self,plq,cols,i,x_bsz,y_bsz,psi=None):
        psi = self.psi if psi is None else psi
        cols,lenvs = self.get_all_lenvs(cols,jmax=self.Ly-y_bsz-1,inplace=False)
        cols,renvs = self.get_all_renvs(cols,jmin=y_bsz,inplace=False)
        for j in range(self.Ly - y_bsz +1): 
            try:
                plq[(i,j),(x_bsz,y_bsz)] = self._get_plq(j,y_bsz,cols,lenvs,renvs) 
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None):
        psi = self.psi if psi is None else psi
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        try:
            tn = self.get_mid_env(i,config,psi=psi)
            for ix in range(1,x_bsz):
                tn.add_tensor_network(self.get_mid_env(i+ix,config,psi=psi),virtual=False)
            if i>0:
                other = tn 
                tn = cache_bot[config[:i*self.Ly]].copy()
                tn.add_tensor_network(other,virtual=False)
            if i+x_bsz<self.Lx:
                tn.add_tensor_network(cache_top[config[(i+x_bsz)*self.Ly:]],virtual=False)
        except AttributeError:
            tn = None
        return tn 
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,cache_bot=None,cache_top=None,imin=None,imax=None):
        #if self.compute_bot and self.compute_top:
        #    raise ValueError
        imin = 0 if imin is None else imin
        imax = self.Lx-x_bsz if imax is None else imax
        psi = self.psi if psi is None else psi
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        plq = dict()
        for i in range(imin,imax+1):
            tn = self.build_3row_tn(config,i,x_bsz,psi=psi,cache_bot=cache_bot,cache_top=cache_top)
            #exit()
            if tn is not None:
                plq = self.update_plq_from_3row(plq,tn,i,x_bsz,y_bsz,psi=psi)
        return plq
    def _unsigned_amplitude_from_benvs(self,bot,top):
        try:
            tn = bot.copy()
            tn.add_tensor_network(top,virtual=False)
            return safe_contract(tn)
        except AttributeError:
            return None
    def unsigned_amplitude(self,config,cache_bot=None,cache_top=None,to_numpy=True): 
        # always contract into the middle
        rix1,rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
        env_bot,env_top = self.get_all_benvs(config,cache_bot=cache_bot,cache_top=cache_top,rix1=rix1,rix2=rix2)
        if env_bot is None and env_top is None:
            return 0.
        cx = self._unsigned_amplitude_from_benvs(env_bot,env_top)
        if to_numpy:
            cx = 0. if cx is None else tensor2backend(cx,'numpy')
        return cx  
################################################################################
# for hamiltonian
################################################################################
    def _get_bot(self,imin,config,cache_bot=None):
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        bot = None if imin==0 else cache_bot[config[:imin*self.Ly]]
        return bot
    def _get_top(self,imax,config,cache_top=None):
        cache_top = self.cache_top if cache_top is None else cache_top
        top = None if imax==self.Lx-1 else cache_top[config[(imax+1)*self.Ly:]]
        return top
    def _get_boundary_mps_deterministic(self,config,imin,imax,cache_bot=None,cache_top=None):
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        imin = min(self.rix1+1,imin) 
        imax = max(self.rix2-1,imax) 
        self.get_all_bot_envs(config,cache=cache_bot,imax=imin-1)
        self.get_all_top_envs(config,cache=cache_top,imin=imax+1)
        bot = self._get_bot(imin,config,cache_bot=cache_bot)
        top = self._get_top(imax,config,cache_top=cache_top)
        return bot,top
    def _unsigned_amplitude_deterministic(self,config,imin,imax,bot,top,cache_bot=None,cache_top=None):
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        imin = min(self.rix1+1,imin) 
        imax = max(self.rix2-1,imax) 
        bot_term = None if bot is None else bot.copy()
        for i in range(imin,self.rix1+1):
            row = self.get_mid_env(i,config)
            bot_term = self.get_bot_env(i,row,bot_term,config,cache=cache_bot)
        if bot_term is None:
            return None

        top_term = None if top is None else top.copy()
        for i in range(imax,self.rix2-1,-1):
            row = self.get_mid_env(i,config)
            top_term = self.get_top_env(i,row,top_term,config,cache=cache_top)
        if top_term is None:
            return None
        return self._unsigned_amplitude_from_benvs(bot_term,top_term)
    def pair_energy_deterministic(self,config,site1,site2,top,bot,model,cache_bot=None,cache_top=None):
        ix1,ix2 = [model.flatten(site) for site in (site1,site2)]
        i1,i2 = config[ix1],config[ix2]
        if not model.pair_valid(i1,i2): # term vanishes 
            return None 
        imin = min(self.rix1+1,site1[0],site2[0]) 
        imax = max(self.rix2-1,site1[0],site2[0]) 
        ex = [] 
        coeff_comm = self.intermediate_sign(config,ix1,ix2) * model.pair_coeff(site1,site2)
        for i1_new,i2_new,coeff in model.pair_terms(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            sign_new = self.config_sign(config_new)

            cx_new = self._unsigned_amplitude_deterministic(config_new,imin,imax,bot,top,cache_bot=cache_bot,cache_top=cache_top)
            if cx_new is None:
                continue
            ex.append(coeff * sign_new * cx_new)
        if len(ex)==0:
            return None
        return sum(ex) * coeff_comm
####################################################################################
###################################################################
# ham
###################################################################
from .tensor_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(Hamiltonian_):
    def batch_pair_energies_from_plq(self,batch_idx,config):
        af = self.amplitude_factory 
        bix,tix,plq_types,pairs = self.model.batched_pairs[batch_idx]
        cache_bot,cache_top = dict(),dict()
        af.get_all_bot_envs(config,cache=cache_bot,imax=bix)
        af.get_all_top_envs(config,cache=cache_top,imin=tix)

        # form plqs
        plq = dict()
        for imin,imax,x_bsz,y_bsz in plq_types:
            plq.update(af.get_plq_from_benvs(config,x_bsz,y_bsz,cache_bot=cache_bot,cache_top=cache_top,imin=imin,imax=imax))

        # compute energy numerator 
        ex,cx = self._pair_energies_from_plq(plq,pairs,config,af=af)
        return ex,cx,plq
    def pair_energies_from_plq(self,config): 
        af = self.amplitude_factory  
        x_bsz_min = min([x_bsz for x_bsz,_ in self.model.plq_sz])
        af.get_all_benvs(config,x_bsz=x_bsz_min)

        plq = dict()
        for x_bsz,y_bsz in self.model.plq_sz:
            plq.update(af.get_plq_from_benvs(config,x_bsz,y_bsz))

        ex,cx = self._pair_energies_from_plq(plq,self.model.pairs,config,af=af)
        return ex,cx,plq
    def pair_energy_deterministic(self,config,site1,site2):
        af = self.amplitude_factory 
        ix1,ix2 = [self.model.flatten(*site) for site in (site1,site2)]
        i1,i2 = config[ix1],config[ix2]
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return None 

        cache_top = dict()
        cache_bot = dict()
        imin = min(site1[0],site2[0])
        imax = max(site1[0],site2[0])
        bot,top = af._get_boundary_mps_deterministic(config,imin,imax,cache_bot=cache_bot,cache_top=cache_top)

        ex = af.pair_energy_deterministic(config,site1,site2,top,bot,self.model,cache_bot=cache_bot,cache_top=cache_top)
        return {(site1,site2):ex} 
    def batch_pair_energies_deterministic(self,config,batch_imin,batch_imax):
        af = self.amplitude_factory
        cache_top = dict()
        cache_bot = dict()
        bot,top = af._get_boundary_mps_deterministic(config,batch_imin,batch_imax,cache_bot=cache_bot,cache_top=cache_top)

        ex = dict() 
        for site1,site2 in self.model.batched_pairs[batch_imin,batch_imax]:
            eij = af.pair_energy_deterministic(config,site1,site2,top,bot,self.model,cache_bot=cache_bot,cache_top=cache_top)
            if eij is not None:
                ex[site1,site2] = eij
        return ex
    def pair_energies_deterministic(self,config):
        af = self.amplitude_factory
        #print('called')
        #exit()
        af.get_all_benvs(config)
        ex = dict() 
        for (site1,site2) in self.model.pairs:
            imin = min(af.rix1+1,site1[0],site2[0]) 
            imax = max(af.rix2-1,site1[0],site2[0]) 
            bot = af._get_bot(imin,config)  
            top = af._get_top(imax,config)  

            eij = af.pair_energy_deterministic(config,site1,site2,top,bot,self.model)
            if eij is not None:
                ex[site1,site2] = eij
        return ex
####################################################################
# models
####################################################################
from .tensor_vmc import Model as Model_
class Model(Model_):
    def __init__(self,Lx,Ly,nbatch=1):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly
        self.nbatch = nbatch 
    def flatten(self,site):
        i,j = site
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Ly)
    def pairs_nn(self,d=1):
        ls = [] 
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            if j+d<self.Ly:
                where = (i,j),(i,j+d)
                ls.append(where)
            else:
                if _PBC:
                    where = (i,(j+d)%self.Ly),(i,j)
                    ls.append(where)
            if i+d<self.Lx:
                where = (i,j),(i+d,j)
                ls.append(where)
            else:
                if _PBC:
                    where = ((i+d)%self.Lx,j),(i,j)
                    ls.append(where)
        return ls
    def pairs_diag(self):
        ls = [] 
        for i in range(self.Lx):
            for j in range(self.Ly):
                if i+1<self.Lx and j+1<self.Ly:
                    where = (i,j),(i+1,j+1)
                    ls.append(where)
                    where = (i,j+1),(i+1,j)
                    ls.append(where)
                else:
                    if _PBC:
                        ix1,ix2 = self.flatten(i,j),self.flatten((i+1)%self.Lx,(j+1)%self.Ly)
                        where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
                        ls.append(where)
                        
                        ix1,ix2 = self.flatten(i,(j+1)%self.Ly),self.flatten((i+1)%self.Lx,j)
                        where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
                        ls.append(where)
        return ls
    def batch_deterministic_nnh(self,d=1):
        for i in range(self.Lx):
            ls = self.batched_pairs.get((i,i),[])
            for j in range(self.Ly):
                if j+d<self.Ly:
                    where = (i,j),(i,j+d)
                    ls.append(where)
                else:
                    if _PBC:
                        where = (i,(j+d)%self.Ly),(i,j)
                        ls.append(where)
            self.batched_pairs[i,i] = ls
    def batch_deterministic_nnv(self,d=1):
        for i in range(self.Lx-d):
            ls = self.batched_pairs.get((i,i+d),[])
            for j in range(self.Ly):
                where = (i,j),(i+d,j)
                ls.append(where)
            self.batched_pairs[i,i+d] = ls
        if not _PBC:
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
                    if _PBC:
                        where = (i,j),(i+1,(j+1)%self.Ly)
                        ls.append(where)
                        
                        where = (i,(j+1)%self.Ly),(i+1,j)
                        ls.append(where)
            self.batched_pairs[i,i+1] = ls
        if not _PBC:
            return
        ls = self.batched_pairs.get('pbc',[])
        for j in range(self.Ly):
            where = (0,(j+1)%self.Ly),(self.Lx-1,j)
            ls.append(where)
            
            where = (0,j),(self.Lx-1,(j+1)%self.Ly)
            ls.append(where)
        self.batched_pairs['pbc'] = ls 
    def batch_plq_nn(self,d=1):
        self.batched_pairs = dict() 
        batchsize = max(self.Lx // self.nbatch,d+1)
        for i in range(self.Lx):
            batch_idx = i // batchsize
            if batch_idx not in self.batched_pairs:
                self.batched_pairs[batch_idx] = [],[] 
            rows,pairs = self.batched_pairs[batch_idx]
            for ix in range(d+1):
                if i+ix < self.Lx:
                    rows.append(i+ix)
            for j in range(self.Ly):
                for ix in range(1,d+1):
                    if j+ix<self.Ly:
                        where = (i,j),(i,j+ix)
                        pairs.append(where)
                for ix in range(1,d+1):
                    if i+ix<self.Lx:
                        where = (i,j),(i+ix,j)
                        pairs.append(where)
        self.pairs = []
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            bix,tix = max(0,imax-1),min(imin+1,self.Lx-1) # bot_ix,top_ix 
            plq_types = (imin,imax,1,d+1), (imin,imax-d,d+1,1),# i0_min,i0_max,x_bsz,y_bsz
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs 
            self.pairs += pairs
        self.plq_sz = (1,d+1),(d+1,1)
        if RANK==0:
            print('nbatch=',len(self.batched_pairs))
    def batch_plq_diag(self):
        self.batched_pairs = dict()
        batchsize = max(self.Lx // self.nbatch, 2)
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
                if i+1<self.Lx: # NN
                    where = (i,j),(i+1,j)
                    pairs.append(where)
                if i+1<self.Lx and j+1<self.Ly: # diag
                    where = (i,j),(i+1,j+1)
                    pairs.append(where)
                    where = (i,j+1),(i+1,j)
                    pairs.append(where)
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            plq_types = (imin,imax-1,2,2),
            bix,tix = max(0,imax-2),min(imin+2,self.Lx-1)
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs # bot_ix,top_ix,pairs 
        self.plq_sz = (2,2),
from .tensor_vmc import get_gate2
class Heisenberg(Model): 
    def __init__(self,J,h,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        try:
            self.Jx,self.Jy,self.Jz = J
        except TypeError:
            self.Jx,self.Jy,self.Jz = J,J,J
        self.h = h

        self.gate = get_gate2((self.Jx,self.Jy,0.))
        self.order = 'b1,k1,b2,k2'

        self.pairs = self.pairs_nn()
        if _DETERMINISTIC:
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
class J1J2(Model): # prototypical next nn model
    def __init__(self,J1,J2,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.J1,self.J2 = J1,J2

        self.gate = get_gate2((1.,1.,0.))
        self.order = 'b1,k1,b2,k2'

        self.pairs = self.pairs_nn() + self.pairs_diag() # list of all pairs, for SR
        if _DETERMINISTIC:
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
                if _PBC:
                    e1 += s1 * (-1)**config[self.flatten((i,0))]
            if i+1<self.Lx:
                e1 += s1 * (-1)**config[self.flatten((i+1,j))]
            else:
                if _PBC:
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
                if _PBC:
                    ix1,ix2 = self.flatten((i,j)), self.flatten(((i+1)%self.Lx,(j+1)%self.Ly))
                    e2 += (-1)**(config[ix1]+config[ix2])
                    ix1,ix2 = self.flatten((i,(j+1)%self.Ly)), self.flatten(((i+1)%self.Lx,j))
                    e2 += (-1)**(config[ix1]+config[ix2])
        return .25 * (e1 * self.J1 + e2 * self.J2) 
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.5)]
class SpinDensity(Hamiltonian):
    def __init__(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly 
        self.data = np.zeros((Lx,Ly))
        self.n = 0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
        self.n += 1.
        for i in range(self.Lx):
            for j in range(self.Ly):
                self.data[i,j] += (-1) ** config[self.flatten(i,j)]
        return 0.,0.,None,None,0. 
    def _print(self,fname,data):
        print(fname)
        print(data)
class Mz(Hamiltonian):
    def __init__(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly 
        self.nsites = Lx * Ly
        self.data = np.zeros(1)
        self.n = 0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
        self.n += 1.

        data = 0.
        for ix1 in range(self.nsites):
            s1 = (-1) ** config[ix1]
            site1 = self.flat2site(ix1)
            for ix2 in range(ix1+1,self.nsites):
                s2 = (-1) ** config[ix2]
                site2 = self.flat2site(ix2)
                  
                dx,dy = site1[0]-site2[0],site1[1]-site2[1]
                data += s1 * s2 * (-1)**(dx+dy)
        self.data += data / self.nsites**2
        return 0.,0.,None,None,0. 
    def _print(self,fname,data):
        print(f'fname={fname},data={data[0]}') 
####################################################################################
# sampler 
####################################################################################
from .tensor_vmc import ExchangeSampler as ExchangeSampler_
class ExchangeSampler(ExchangeSampler_):
    def __init__(self,Lx,Ly,seed=None,burn_in=0,scheme='hv'):
        self.Lx, self.Ly = Lx,Ly
        self.nsite = Lx * Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
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
        config_sites = self.amplitude_factory.parse_config(config_sites)
        plq_new,py = self.amplitude_factory._new_prob_from_plq(plq,(site1,site2),config_sites)
        if py is None:
            return plq,cols
        try:
            acceptance = py / self.px
        except ZeroDivisionError:
            acceptance = 1. if py > self.px else 0.
        if acceptance < self.rng.uniform(): # reject
            return plq,cols
        # accept, update px & config & env_m
        self.px = py
        self.config = config_new
        cols = self.amplitude_factory.replace_sites(cols,(site1,site2),config_sites)
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
        af = self.amplitude_factory
        try:
            cols,renvs = af.get_all_renvs(cols,jmin=y_bsz,inplace=False)
        except AttributeError:
            return
        for j in range(self.Ly - y_bsz + 1): 
            try:
                plq = af._get_plq_forward(j,y_bsz,cols,renvs)
            except AttributeError:
                return
            cols = self.update_plq(i,j,x_bsz,y_bsz,plq,cols) 
            try:
                cols = af._contract_cols(cols,(0,j))
            except (ValueError,IndexError):
                return
    def sweep_col_backward(self,i,cols,x_bsz,y_bsz):
        af = self.amplitude_factory
        try:
            cols,lenvs = af.get_all_lenvs(cols,jmax=self.Ly-1-y_bsz,inplace=False)
        except AttributeError:
            return
        for j in range(self.Ly - y_bsz,-1,-1): # Ly-1,...,1
            try:
                plq = af._get_plq_backward(j,y_bsz,cols,lenvs)
            except AttributeError:
                return
            cols = self.update_plq(i,j,x_bsz,y_bsz,plq,cols) 
            try:
                cols = af._contract_cols(cols,(j+y_bsz-1,self.Ly-1))
            except (ValueError,IndexError):
                return
    def sweep_row_forward(self,x_bsz,y_bsz):
        af = self.amplitude_factory
        af.cache_bot = dict()
        af.get_all_top_envs(af.parse_config(self.config),imin=x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax+1):
            tn = af.build_3row_tn(af.parse_config(self.config),i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)

            af.get_all_bot_envs(af.parse_config(self.config),imin=i,imax=i+x_bsz-1)
    def sweep_row_backward(self,x_bsz,y_bsz):
        af = self.amplitude_factory
        af.cache_top = dict()
        af.get_all_bot_envs(af.parse_config(self.config),imax=self.Lx-1-x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax,-1,-1):
            tn = af.build_3row_tn(af.parse_config(self.config),i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)

            af.get_all_top_envs(af.parse_config(self.config),imin=i,imax=i+x_bsz-1)
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
        af = self.amplitude_factory
        pairs = self.get_pairs(i,j,x_bsz,y_bsz)
        for (site1,site2) in pairs:
            config_sites,config_new = self._new_pair(site1,site2)
            if config_sites is None:
                continue
            imin = min(site1[0],site2[0]) 
            imax = max(site1[0],site2[0]) 
            config = af.parse_config(config_new)
            bot,top = af._get_boundary_mps_deterministic(config,imin,imax)
            cy = af._unsigned_amplitude_deterministic(config,imin,imax,bot,top)
            if cy is None:
                continue
            py = tensor2backend(cy ** 2,'numpy')
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept, update px & config & env_m
                self.px = py
                self.config = tuple(config_new) 
    def _sample_deterministic(self):
        bsz = self.get_bsz()
        for x_bsz,y_bsz in bsz:
            imax = self.Lx-1 if _PBC else self.Lx-x_bsz
            jmax = self.Ly-1 if _PBC else self.Ly-y_bsz
            rdir = self.rng.choice([-1,1]) 
            cdir = self.rng.choice([-1,1]) 
            sweep_row = range(0,imax+1) if rdir==1 else range(imax,-1,-1)
            sweep_col = range(0,jmax+1) if cdir==1 else range(jmax,-1,-1)
            for i,j in itertools.product(sweep_row,sweep_col):
                self.update_plq_deterministic(i,j,x_bsz,y_bsz)
        af = self.amplitude_factory
        af.update_cache(af.parse_config(self.config))

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