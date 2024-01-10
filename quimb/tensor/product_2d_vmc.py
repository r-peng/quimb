import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from .product_vmc import (
    RBM,Dense,NN,
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
        env_prev = [None] * len(self.af) 
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
        envs = [None] * len(self.af) 
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
        cols_new = [None] * len(self.af )
        envs = [None] * len(self.af)
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            cols_new[ix],envs[ix] = af.get_all_envs(cols[ix],step,stop=stop,inplace=inplace)
        return cols_new,envs
    def _get_plq_forward(self,j,y_bsz,cols,renvs,direction='col'):
        plq = [None] * len(self.af) 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            plq[ix] = af._get_plq_forward(j,y_bsz,cols[ix],renvs[ix],direction=direction)
        return plq
    def _get_plq_backward(self,j,y_bsz,cols,lenvs,direction='col'):
        plq = [None] * len(self.af) 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            plq[ix] = af._get_plq_backward(j,y_bsz,cols[ix],lenvs[ix],direction=direction)
        return plq 
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None,direction='row'):
        tn = [None] * len(self.af) 
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
class NN2D(NN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,lr,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(lr,**kwargs)
class CNN2D(Dense):
    def __init__(self,lx,ly,nx,ny,afn,npool=1,**kwargs):
        super().__init__(nx,ny,afn,**kwargs)
        self.lx,self.ly = lx,ly
        self.npool = npool
        lx,ly = max(1,self.lx-1),max(1,self.ly-1)
        self.sh = (lx,ly,4*nx,ny*npool),(lx*ly,ny*npool)
    def apply_w(self,y):
        y = y.reshape(self.lx,self.ly,y.shape[-1]) 
        lx,ly = max(1,self.lx-1),max(1,self.ly-1)
        W = self.params[0]
        ynew = []
        for i,j in itertools.product(range(lx),range(ly)):
            yij = [(i,j)]
            if self.ly>1:
                yij.append((i,j+1))
            if self.lx>1:
                yij.append((i+1,j))
            if self.lx>1 and self.ly>1:
                yij.append((i+1,j+1))
            yij = [y[i_,j_] for i_,j_ in yij]
            try:
                yij = self.jnp.concatenate(yij)
            except:
                yij = self.jnp.cat(yij)
            ynew.append(self.jnp.matmul(yij,W[i,j]))
        return self.jnp.stack(ynew,axis=0)
    def set_backend(self,backend):
        super().set_backend(backend)
        if self.npool==1:
            return
        def _afn(x):
            x = x.reshape(x.shape[0],self.ny,self.npool)
            return self.jnp.max(x,-1) 
        self._afn = _afn
