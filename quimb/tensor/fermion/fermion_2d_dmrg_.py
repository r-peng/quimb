from .fermion_2d_vmc import AmplitudeFactory2D as AmplitudeFactory
from .fermion_2d_vmc import Hubbard2D as Hubbard
from .fermion_2d_vmc import flat2site,site_grad,get_bra_tsr
from .fermion_core import FermionTensorNetwork 
import numpy as np
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
        return vx 
class Hubbard2D(Hubbard):
    def initialize_pepo(self,fpeps=None):
        pass
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
            vx = amplitude_factory.get_grad_from_plq(plq12,config=config) 
        # get h/v bonds
        eh = self.nn(config,plq12,x_bsz=1,y_bsz=2,inplace=True,cx=cx12) 
        ev = self.nn(config,plq21,x_bsz=2,y_bsz=1,inplace=True,cx=None) 
        # onsite terms
        config = np.array(config,dtype=int)
        eu = self.u*len(config[config==3])

        ex = eh+ev+eu
        if not compute_Hv: 
            return unsigned_cx,ex,vx,None 
        sign = amplitude_factory.compute_config_sign(tuple(config)) 
        Hvx = self.compute_Hv_hop(tuple(config),amplitude_factory)
        Hvx /= sign * unsigned_cx
        Hvx += eu * vx
        return unsigned_cx,ex,vx,Hvx
    def compute_Hv_hop(self,config,amplitude_factory):
        cache_top = amplitude_factory.cache_top
        cache_bot = amplitude_factory.cache_bot
        ix = amplitude_factory.ix
        i0,j0 = self.flat2site(ix)
        fpeps = amplitude_factory.psi
        compress_opts = amplitude_facory.contract_opts

        # horizontal,lower
        top = FermionTensorNetwork([]) if i0==self.Lx-1 else cache_top[config[(i0+1)*self.Ly:]]
        mid = get_mid_env_(i0,fpeps,config)
        for i in range(i0-1,-1,-1):
            env_prev = None if i==0 else cache_bot[config[:i*self.Ly]].copy()
            for j in range(self.Ly-1):
                config_new,sign = self.terms(config,i,j)
                for i_ in range(i,i0):
                    row = get_mid_env(i_,fpeps,config_new) 
                    env_prev = get_bot_env(i_,row,env_prev,config_new,cache_bot,**compress_opts)
                if env_
                ftn = FermionTensorNetwork([]) 
        Hvx = site_grad(ftn,i,j)
        cons = amplitude_factory.constructors[ix][0]
        Hvx = cons.tensor_to_vector(Hvx) 
        return Hvx
