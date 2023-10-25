import time,itertools
import numpy as np

#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()

import sys
this = sys.modules[__name__]
def set_options(pbc=False,deterministic=False):
    this._PBC = pbc
    this._DETERMINISTIC = deterministic
    from .tensor_2d_vmc import set_options as set_options
    return set_options(pbc=pbc,deterministic=deterministic)

from .product_vmc import (
    RBM,FNN,SIGN,
    ProductAmplitudeFactory,
)
from .tensor_2d_vmc import AmplitudeFactory2D 
class ProductAmplitudeFactory2D(ProductAmplitudeFactory):
    def __init__(self,af):
        self.af = af 
        self.get_sections()

        self.Lx,self.Ly = self.af[0].Lx,self.af[0].Ly
        self.sites = self.af[0].sites
        self.model = self.af[0].model
        self.nsite = self.af[0].nsite
        self.backend = self.af[0].backend

        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
        self.spins = None,
##### wfn methods #####
    def update_cache(self,config):
        for af,config_ in zip(self.af,config):
            af.update_cache(config_)
##### compress row methods  #####
    def _get_all_benvs(self,config,step,psi=None,cache=None,start=None,stop=None,append='',direction='row'):
        env_prev = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            cache_ = af.get_cache(direction,step) if cache is None else cache[ix]
            env_prev[ix] = af._get_all_benvs(config[ix],step,psi=psi_,cache=cache_,
                               start=start,stop=stop,append=append,direction=direction)
        return env_prev
    def _contract_cols(self,cols,js,direction='col'):
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            cols[ix] = af._contract_cols(cols[ix],js,direction=direction)
        return cols 
    def get_all_envs(self,cols,step,stop=None,inplace=False,direction='col'):
        cols_new = [None] * self.naf 
        envs = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            cols_new[ix],envs[ix] = af.get_all_envs(cols[ix],step,stop=stop,inplace=inplace)
        return cols_new,envs
    def _get_plq_forward(self,j,y_bsz,cols,renvs,direction='col'):
        plq = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            plq[ix] = af._get_plq_forward(j,y_bsz,cols[ix],renvs[ix],direction=direction)
        return plq
    def _get_plq_backward(self,j,y_bsz,cols,lenvs,direction='col'):
        plq = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            plq[ix] = af._get_plq_backward(j,y_bsz,cols[ix],lenvs[ix],direction=direction)
        return plq 
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None,direction='row'):
        tn = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            cache_bot_ = af.get_cache(direction,1) if cache_bot is None else cache_bot[ix]
            cache_top_ = af.get_cache(direction,-1) if cache_top is None else cache_top[ix]
            tn[ix] = af.build_3row_tn(config[ix],i,x_bsz,psi=psi_,
                         cache_bot=cache_bot_,cache_top=cache_top_,direction=direction) 
        return tn
    def unsigned_amplitude(self,config,cache_bot=None,cache_top=None,to_numpy=True):
        cx = np.zeros(self.naf) 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cache_top_ = af.cache_top if cache_top is None else cache_top[ix]
                cache_bot_ = af.cache_bot if cache_bot is None else cache_bot[ix]
                cx[ix] = af.unsigned_amplitude(config[ix],
                             cache_bot=cache_bot_,cache_top=cache_top_,to_numpy=to_numpy)
            else:
                cx[ix] = af.unsigned_amplitude(config[ix],to_numpy=to_numpy)
        #print(config[2],cx)
        #print(self.config_sign(config))
        #exit()
        return np.prod(cx)
class RBM2D(RBM,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nh,**kwargs)
class FNN2D(FNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nl,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nl,**kwargs)
class SIGN2D(SIGN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nl,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nl,**kwargs)
#class CNN2D(FNN2D):
#    def __init__(self,Lx,Ly,nl,kx,ky,**kwargs):
#        self.kx,self.ky = kx,ky 
#        #self.blks = []
#        #for i,j in itertools.product(range(Lx-kx),range(Ly-ky)):
#        #    self.blks.append(list(itertools.product(range(i,i+kx),range(j,j+ky))))
#        #self.get_site_map(self.blks)
#        super().__init__(Lx,Ly,nl,**kwargs) 
#    def log_amplitude(self,config,to_numpy=True):
#        c = np.array(config,dtype=float).reshape((self.Lx,self.Ly)) 
#        jnp,c = self.get_backend(c=c)
#        for i in range(self.nl-1):
#            c = self.convolve(c,self.w[i],jnp) + self.b[i]
#            c = jnp.log(jnp.cosh(c)) 
#        c = jnp.dot(c,self.w[-1])
#        #exit()
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#        return c,0
#    def convolve(self,x,w,jnp):
#        v = jnp.zeros((self.Lx,self.Ly),requires_grad=True) 
#        for i,j in itertools.product(range(self.Lx-kx),range(self.Ly-ky)):
#            v[i:i+kx,j:j+ky] += jnp.matmul(w[i,j,:,:],x[i:i+kx,j:j+ky].flatten()).reshape((kx,ky))
#        return v 
