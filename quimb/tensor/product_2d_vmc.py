import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from .product_vmc import (
    RBM,FNN,LRelu,RNN,CNN,
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
def get_bond_map_2d(self):
    bmap = dict()
    ix = 0
    for i,j in itertools.product(range(self.Lx),range(self.Ly)):
        ix1 = self.flatten((i,j))
        if j<self.Ly-1:
            ix2 = self.flatten((i,j+1)) 
            bmap[ix1,ix2] = ix
            ix += 1
        if i<self.Lx-1:
            ix2 = self.flatten((i+1,j)) 
            bmap[ix1,ix2] = ix
            ix += 1
    return bmap
class RBM2D(RBM,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        self.bmap = get_bond_map_2d(self)
        super().__init__(nv,nh,**kwargs)
class FNN2D(FNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,afn,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        self.bmap = get_bond_map_2d(self)
        super().__init__(nv,nh,afn,**kwargs)
class LRelu2D(LRelu,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        self.bmap = get_bond_map_2d(self)
        super().__init__(nv,nh,**kwargs)
class RNN2D(RNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,D,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(Lx*Ly,D,input_format=(-1,1),**kwargs)
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
class CNN2D(CNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,D,nf=1,**kwargs):
        self.Lx,self.Ly = Lx,Ly
        super().__init__(Lx*Ly,D,**kwargs)
    def init_dim(self):
        self.lx,self.ly = self.Lx,self.Ly
class CNN2D1(CNN2D):
    def set_layer(self,l,Dix,D):
        D1 = D[Dix] if Dix<len(D) else D[-1]
        D2 = D[Dix+1] if Dix+1<len(D) else D[-1]
        if self.ly > 1:
            key = l,'h'
            self.param_keys.append(key)
            self.sh[key] = self.lx,self.ly-1,D1,D1,D2
        if self.lx > 1:
            key = l,'v'
            self.param_keys.append(key)
            self.sh[key] = self.lx-1,self.ly,D1,D1,D2

        if self.lx > 1:
            self.lx -= 1
        if self.ly > 1:
            self.ly -= 1
        _break = False
        if self.lx==1 and self.ly==1:
            _break = True
            for key in [(l,'h'),(l,'v')]:
                if key in self.sh:
                    self.sh[key] = self.sh[key][:-1] + (self.nf,)  

        return l+1,Dix+1,_break
    def input_layer(self,config):
        l = 0
        vnew = dict()
        if self.lx == 1:
            i = 0
            T = self.params[l,'h']
            for j in range(self.ly-1):
                vnew[i,j] = T[i,j,config[j],config[j+1],:]
            self.ly -= 1
            return vnew
        if self.ly == 1:
            j = 0
            T = self.params[l,'v']
            for i in range(self.lx-1):
                vnew[i,j] = T[i,j,config[i],config[i+1],:]
            self.lx -= 1
            return vnew

        b = dict()
        Th = self.params[l,'h']
        Tv = self.params[l,'v']
        for i,j in itertools.product(range(self.lx),range(self.ly)):
            ix1 = config[self.flatten((i,j))]
            if j+1<self.ly:
                ix2 = config[self.flatten((i,j+1))]
                b[(i,j),(i,j+1)] = Th[i,j,ix1,ix2,:]
            if i+1<self.lx:
                ix2 = config[self.flatten((i+1,j))]
                b[(i,j),(i+1,j)] = Tv[i,j,ix1,ix2,:]

        for i,j in itertools.product(range(self.lx-1),range(self.ly-1)):
            vnew[i,j] = b[(i,j),(i,j+1)]*b[(i+1,j),(i+1,j+1)] + b[(i,j),(i+1,j)]*b[(i,j+1),(i+1,j+1)]
        self.lx -= 1
        self.ly -= 1
        return vnew 
    def layer_forward(self,l,v):
        vnew = dict()
        if self.lx == 1:
            i = 0
            T = self.params[l,'h']
            for j in range(self.ly-1):
                vnew[i,j] = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i,j+1],T[i,j,...])
            self.ly -= 1
            if self.ly == 1:
                return None,vnew[0,0],True
            return l+1,vnew,False
        if self.ly == 1:
            j = 0
            T = self.params[l,'v']
            for i in range(self.lx-1):
                vnew[i,j] = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i+1,j],T[i,j,...])
            self.lx -= 1
            if self.lx == 1:
                return None,vnew[0,0],True
            return l+1,vnew,False

        b = dict()
        Th = self.params[l,'h']
        Tv = self.params[l,'v']
        for i,j in itertools.product(range(self.lx),range(self.ly)):
            if j+1<self.ly:
                b[(i,j),(i,j+1)] = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i,j+1],Th[i,j,...])
            if i+1<self.lx:
                b[(i,j),(i+1,j)] = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i+1,j],Tv[i,j,...])

        for i,j in itertools.product(range(self.lx-1),range(self.ly-1)):
            vnew[i,j] = b[(i,j),(i,j+1)]*b[(i+1,j),(i+1,j+1)] + b[(i,j),(i+1,j)]*b[(i,j+1),(i+1,j+1)]
        self.lx -= 1
        self.ly -= 1
        if self.lx == 1 and self.ly == 1:
            return None,vnew[0,0],True
        return l+1,vnew,False 
class CNN2D2(CNN2D,AmplitudeFactory2D):
    def set_layer(self,l,Dix,D):
        D1 = D[Dix] if Dix<len(D) else D[-1]
        D2 = D[Dix+1] if Dix+1<len(D) else D[-1]
        key = l,1
        self.param_keys.append(key) 
        self.sh[key] = max(self.lx-1,1),max(self.ly-1,1),D1,D1,D2
        if self.lx > 1 and self.ly > 1:
            key = l,2
            self.param_keys.append(key) 
            self.sh[key] = self.lx-1,self.ly-1,D1,D2,D2

            key = l,3
            self.param_keys.append(key) 
            self.sh[key] = self.lx-1,self.ly-1,D1,D2,D2

        if self.lx > 1:
            self.lx -= 1
        if self.ly > 1:
            self.ly -= 1

        _break = False
        if self.lx==1 and self.ly==1:
            _break = True
            self.sh[key] = self.sh[key][:-1] + (self.nf,)  
        return l+1,Dix+1,_break
    def input_layer(self,config):
        l = 0
        vnew = dict()
        T1 = self.params[l,1]
        if self.lx == 1:
            i = 0
            for j in range(self.ly-1):
                vnew[i,j] = T1[i,j,config[j],config[j+1],:]
            self.ly -= 1
            return vnew
        if self.ly == 1:
            j = 0
            for i in range(self.lx-1):
                vnew[i,j] = T1[i,j,config[i],config[i+1],:]
            self.lx -= 1
            return vnew
        T2 = self.params[l,2]
        T3 = self.params[l,3]
        for i,j in itertools.product(range(self.lx-1),range(self.ly-1)):
            ix1 = config[self.flatten((i,j))]
            ix2 = config[self.flatten((i,j+1))]
            ix3 = config[self.flatten((i+1,j+1))]
            ix4 = config[self.flatten((i+1,j))]

            vij = T1[i,j,ix1,ix2,:]
            vij = self.jnp.matmul(vij,T2[i,j,ix3,...]) 
            vnew[i,j] = self.jnp.matmul(vij,T3[i,j,ix4,...])
        self.lx -= 1
        self.ly -= 1
        return vnew 
    def layer_forward(self,l,v):
        vnew = dict()
        T1 = self.params[l,1]
        if self.lx == 1:
            i = 0
            for j in range(self.ly-1):
                vnew[i,j] = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i,j+1],T1[i,j,...])
            self.ly -= 1
            if self.ly == 1:
                return None,vnew[0,0],True
            return l+1,vnew,False
        if self.ly == 1:
            j = 0
            for i in range(self.lx-1):
                vnew[i,j] = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i+1,j],T1[i,j,...])
            self.lx -= 1
            if self.lx == 1:
                return None,vnew[0,0],True
            return l+1,vnew,False
        T2 = self.params[l,2]
        T3 = self.params[l,3]
        for i,j in itertools.product(range(self.lx-1),range(self.ly-1)):
            vij = self.jnp.einsum('i,j,ijk->k',v[i,j],v[i,j+1],T1[i,j,...])
            vij = self.jnp.einsum('i,j,ijk->k',v[i+1,j+1],vij,T2[i,j,...]) 
            vnew[i,j] = self.jnp.einsum('i,j,ijk->k',v[i+1,j],vij,T3[i,j,...])
        self.lx -= 1
        self.ly -= 1
        if self.lx == 1 and self.ly == 1:
            return None,vnew[0,0],True
        return l+1,vnew,False 
