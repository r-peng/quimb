from .fermion_2d_vmc import AmplitudeFactory2D as AmplitudeFactory
from .fermion_2d_vmc import Hubbard2D as Hubbard
from .fermion_2d_vmc import (
    flat2site,
    site_grad,
    get_bra_tsr,
    get_mid_env,
    get_all_bot_envs,
    get_all_top_envs,
)
from .fermion_core import FermionTensorNetwork 
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
#####################################################################################
# double layer cache
#####################################################################################
def get_3col_ftn(norm,config,cache_bot,cache_top,i,**compress_opts):
    norm.reorder('row',layer_tags=('KET','BRA'),inplace=True)
    ls = []
    if i>0:
        bot = get_all_bot_envs(norm,config,cache_bot,imax=i-1,layer_tags=('KET','BRA'),append='*',**compress_opts)
        ls.append(bot)
    ls.append(get_mid_env(i,norm,config,append='*'))
    if i<norm.Lx-1:
        top = get_all_top_envs(norm,config,cache_top,imin=i+1,layer_tags=('KET','BRA'),append='*',**compress_opts)
        ls.append(top)
    ftn = FermionTensorNetwork(ls,virtual=False).view_like_(norm)
    return ftn 
######################################################################################
# DMRG cahce update
######################################################################################
def cache_update(cache_bot,cache_top,ix,Lx,Ly):
    i,_ = flat2site(ix,Lx,Ly) 
    keys = list(cache_bot.keys())
    l = i * Ly
    for key in keys:
        if len(key) > l:
            cache_bot.pop(key)

    keys = list(cache_top.keys())
    l = (Lx - i - 1) * Ly
    for key in keys:
        if len(key) > l:
            cache_top.pop(key)
    return cache_bot,cache_top
class AmplitudeFactory2D(AmplitudeFactory):
    def _set_psi(self,psi):
        self.psi = psi
        self.store = dict()
        self.store_grad = dict()

        self.compute_bot = True
        self.compute_top = True
        if self.ix is None:
            self.cache_bot = dict()
            self.cache_top = dict()
            return
        self.cache_bot,self.cache_top = cache_update(self.cache_bot,self.cache_top,self.ix,self.Lx,self.Ly)
    def get_grad_from_plq(self,plq,config=None,compute_cx=True):
        i,j = self.flat2site(self.ix)
        _,(x_bsz,y_bsz) = list(plq.keys())[0]
        i0 = min(self.Lx-x_bsz,i)
        j0 = min(self.Ly-y_bsz,j)
        ftn_plq = plq[(i0,j0),(x_bsz,y_bsz)]
        cx = ftn_plq.copy().contract() if compute_cx else 1.
        vx = site_grad(ftn_plq.copy(),i,j) / cx 
        cons = self.constructors[self.ix][0]
        vx = cons.tensor_to_vector(vx) 
        if config is not None:
            self.store[config] = cx
            self.store_grad[config] = vx
        return cx,vx 
class Hubbard2D(Hubbard):
    def update_cache(self,ix):
        self.cache_bot,self.cache_top = cache_update(self.cache_bot,self.cache_top,ix,self.Lx,self.Ly)
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        amplitude_factory.get_all_benvs(config,x_bsz=1) 
        # form all (1,2),(2,1) plqs
        plq12 = amplitude_factory.get_plq_from_benvs(config,x_bsz=1,y_bsz=2)
        plq21 = amplitude_factory.get_plq_from_benvs(config,x_bsz=2,y_bsz=1)
        # get gradient form plq12
        cx12 = {key:ftn_plq.copy().contract() for (key,_),ftn_plq in plq12.items()}
        unsigned_cx = sum(cx12.values()) / len(cx12)
        vx = None
        if compute_v:
            _,vx = amplitude_factory.get_grad_from_plq(plq12,config=config) 
        # get h/v bonds
        eh = self.nn(config,plq12,x_bsz=1,y_bsz=2,inplace=True,cx=cx12) 
        ev = self.nn(config,plq21,x_bsz=2,y_bsz=1,inplace=True,cx=None) 
        # onsite terms
        config = np.array(config,dtype=int)
        eu = self.u*len(config[config==3])

        ex = eh+ev+eu
        if not compute_Hv: 
            return unsigned_cx,ex,vx,None 
        if self.Lx > self.Ly:
            Hvx = self.compute_Hv_hop_col_first(tuple(config),amplitude_factory)
            sign = amplitude_factory.compute_config_sign(tuple(config)) 
        else:
            Hvx = self.compute_Hv_hop_row_first(tuple(config),amplitude_factory)
            sign = 1.
        Hvx /= sign * unsigned_cx
        Hvx += eu * vx
        return unsigned_cx,ex,vx,Hvx
    def compute_Hv_hop_col_first(self,config,amplitude_factory):
        # make bra
        bra = self.pepo.copy()
        fpeps = amplitude_factory.psi.copy()
        for ix,ci in reversed(list(enumerate(config))):
            i,j = self.flat2site(ix)
            tsr = get_bra_tsr(fpeps,ci,i,j,append='*')
            bra.add_tensor(tsr,virtual=True) 
        for i in range(self.Lx):
            for j in range(self.Ly):
                bra.contract_tags(bra.site_tag(i,j),inplace=True)
        norm = fpeps 
        norm.add_tensor_network(bra,virtual=True)

        ix = amplitude_factory.ix
        i,j = self.flat2site(ix)
        norm.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        for j_ in range(1,j):
            norm.contract_boundary_from_left_(yrange=(j_-1,j_),layer_tags=('KET','BRA'),**self.contract_opts)
        for j_ in range(self.Ly-2,j,-1):
            norm.contract_boundary_from_right_(yrange=(j_,j_+1),layer_tags=('KET','BRA'),**self.contract_opts)
        Hvx = site_grad(norm,i,j)
        cons = amplitude_factory.constructors[ix][0]
        Hvx = cons.tensor_to_vector(Hvx) 
        return Hvx
    def compute_Hv_hop_row_first(self,config,amplitude_factory):
        # make bra
        norm = amplitude_factory.psi.copy()
        norm.add_tensor_network(self.pepo.copy(),virtual=True)
        ix = amplitude_factory.ix
        i,j = self.flat2site(ix)

        ftn = get_3col_ftn(norm,config,self.cache_bot,self.cache_top,i,**self.contract_opts) 
        Hvx = site_grad(ftn,i,j)
        cons = amplitude_factory.constructors[ix][0]
        Hvx = cons.tensor_to_vector(Hvx) 
        return Hvx
