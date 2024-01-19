import time,itertools
import numpy as np
import torch

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from .product_vmc import (
    CompoundAmplitudeFactory,
    ProductAmplitudeFactory,
    SumAmplitudeFactory,
)
class CompoundAmplitudeFactory2D(CompoundAmplitudeFactory):
    def update_cache(self,config):
        for af,config_ in zip(self.af,config):
            if af.is_tn:
                af.update_cache(config_)
    def _get_all_benvs(self,config,step,psi=None,start=None,stop=None,append='',direction='row'):
        env_prev = [None] * len(self.af) 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            env_prev[ix] = af._get_all_benvs(config[ix],step,psi=psi_,
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
    def build_3row_tn(self,config,i,x_bsz,psi=None,direction='row'):
        tn = [None] * len(self.af) 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            tn[ix] = af.build_3row_tn(config[ix],i,x_bsz,psi=psi_,direction=direction) 
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

#from .tensor_1d_vmc import AmplitudeFactory1D 
#class CNN2D(Dense):
#    def __init__(self,lx,ly,nx,ny,afn,npool=1,**kwargs):
#        super().__init__(nx,ny,afn,**kwargs)
#        self.lx,self.ly = lx,ly
#        self.npool = npool
#        lx,ly = max(1,self.lx-1),max(1,self.ly-1)
#        self.sh = [(lx,ly,4*nx,ny*npool),(lx*ly,ny*npool)]
#        if not self.bias:
#            self.sh.pop()
#        self.flatten = False
#    def apply_w(self,x):
#        x = x.reshape(self.lx,self.ly,x.shape[-1]) 
#        lx,ly = max(1,self.lx-1),max(1,self.ly-1)
#        W = self.params[0]
#        y = []
#        for i,j in itertools.product(range(lx),range(ly)):
#            yij = [(i,j)]
#            if self.ly>1:
#                yij.append((i,j+1))
#            if self.lx>1:
#                yij.append((i+1,j))
#            if self.lx>1 and self.ly>1:
#                yij.append((i+1,j+1))
#            yij = [x[i_,j_] for i_,j_ in yij]
#            try:
#                yij = self.jnp.concatenate(yij)
#            except:
#                yij = self.jnp.cat(yij)
#            y.append(self.jnp.matmul(yij,W[i,j]))
#        return self.jnp.stack(y,axis=0)
#    def _combine(self,x,y):
#        if not self.combine:
#            return y
#        x = x.reshape(self.lx,self.ly,x.shape[-1]) 
#        lx,ly = max(1,self.lx-1),max(1,self.ly-1)
#        y = y.reshape(lx,ly,y.shape[-1]) 
#        ynew = []
#        for i,j in itertools.product(range(lx),range(ly)):
#            yij = [(i,j)]
#            if self.ly>1:
#                yij.append((i,j+1))
#            if self.lx>1:
#                yij.append((i+1,j))
#            if self.lx>1 and self.ly>1:
#                yij.append((i+1,j+1))
#            yij = [x[i_,j_] for i_,j_ in yij] + [y[i,j]]
#            try:
#                yij = self.jnp.concatenate(yij)
#            except:
#                yij = self.jnp.cat(yij)
#            ynew.append(yij)
#        y = self.jnp.stack(ynew,axis=0)
#        if self.flatten:
#            y = y.flatten()
#        return y
#    def set_backend(self,backend):
#        if self.npool==1:
#            super().set_backend(backend)
#            return
#        if backend=='numpy': 
#            self.jnp = np
#            def _afn(x):
#                x = x.reshape(x.shape[0],self.ny,self.npool)
#                y = self.jnp.max(x,axis=-1) 
#                return y
#        else:
#            self.jnp = torch
#            def _afn(x):
#                x = x.reshape(x.shape[0],self.ny,self.npool)
#                y,_ = self.jnp.max(x,-1) 
#                return y
#        self._afn = _afn
