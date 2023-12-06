import time,itertools
import numpy as np

#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()

from .product_vmc import (
    RBM,FNN,LRelu,
    CompoundAmplitudeFactory,
    ProductAmplitudeFactory,
    SumAmplitudeFactory,
)
class CompoundAmplitudeFactory2D(CompoundAmplitudeFactory):
    def update_cache(self,config):
        for af,config_ in zip(self.af,config):
            if af.is_tn:
                af.update_cache(config_)
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
class ProductAmplitudeFactory2D(ProductAmplitudeFactory,
                                CompoundAmplitudeFactory2D):
    def get_mid_env(self,config,append='',psi=None):
        envs = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            envs[ix] = af.get_mid_env(config[ix],append=append,psi=psi_)
        return envs
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
class SumAmplitudeFactory2D(SumAmplitudeFactory,
                            CompoundAmplitudeFactory2D):
    def build_3row_tn(self,*args):
        pass
    def get_all_envs(self,*args,**kwargs):
        return None,None 
    def _get_plq_forward(self,*args):
        pass 
    def _get_plq_backward(self,*args):
        pass 
    def _contract_cols(self,*args):
        pass

from .tensor_2d_vmc import AmplitudeFactory2D 
from .tensor_1d_vmc import AmplitudeFactory1D 
class RBM2D(RBM,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nh,**kwargs)
class FNN2D(FNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,afn,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nh,afn,**kwargs)
class LRelu2D(LRelu,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nh,**kwargs)
#class SIGN2D(SIGN,AmplitudeFactory2D):
#    def __init__(self,Lx,Ly,nv,nl,**kwargs):
#        self.Lx = Lx
#        self.Ly = Ly 
#        super().__init__(nv,nl,**kwargs)
class RBM1D(RBM,AmplitudeFactory1D):
    def __init__(self,nsite,nv,nh,**kwargs):
        self.nsite = nsite 
        super().__init__(nv,nh,**kwargs)
class FNN1D(FNN,AmplitudeFactory1D):
    def __init__(self,nsite,nv,nh,afn,**kwargs):
        self.nsite = nsite 
        super().__init__(nv,nh,afn,**kwargs)
#class SIGN1D(SIGN,AmplitudeFactory1D):
#    def __init__(self,nsite,nv,nl,**kwargs):
#        self.nsite = nsite 
#        super().__init__(nv,nl,**kwargs)
#class CNN2D(FNN,AmplitudeFactory2D):
#    def __init__(self,Lx,Ly,nv,kx=2,ky=2,**kwargs):
#        super().__init__(nv,None,**kwargs)
#        self.Lx,self.Ly = Lx,Ly
#        self.kx,self.ky = kx,ky 
#        self.ksize = kx * ky
#    def init(self,eps,a=-1,b=1,fname=None):
#        Lx,Ly = self.Lx,self.Ly
#        self.w = [] 
#        c = b-a
#        while Lx>self.kx and Ly>self.ky:
#            Lx -= self.kx
#            Ly -= self.ky
#            nn = Lx * Ly
# 
#            wi = (np.random.rand(nn,self.ksize) * c + a) * eps
#            COMM.Bcast(wi,root=0)
#            self.w.append(wi)
#
#            bi = (np.random.rand(nn) * c + a) * eps
#            COMM.Bcast(bi,root=0)
#            self.b.append(bi)
#        self.w.append(np.ones(ksize))    
#    def load_from_disc(self,fname):
#        return super.__init__(fname,min(self.Lx,self.Ly))
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
