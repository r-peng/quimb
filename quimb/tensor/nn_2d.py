import time,itertools,functools,h5py
import numpy as np
from .nn_core import (
#    Dense,FNN,AmplitudeFNN,AmplitudeSNN,TensorFNN,
    AmplitudeFNN,Fourier,CP,HP,
    #tensor2backend,
    #relu_init_normal,
)
#from .tensor_core import Tensor
from .tensor_2d_vmc import AmplitudeFactory2D,cache_key
#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()
class Fourier2D(Fourier,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nx,ny,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nx,ny,**kwargs)
class CP2D(CP,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nx,ny,pdim,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nx,ny,pdim,**kwargs)
class HP2D(HP,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nx,ny,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nx,ny,**kwargs)
class AmplitudeFNN2D(AmplitudeFNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,af,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(af,**kwargs)

##def get_inds_info(psi):
##    ind_map = dict()
##    for i,j in itertools.product(range(psi.Lx-1),range(psi.Ly)):
##        ind_map[(i,j),(i+1,j)] = tuple(psi[i,j].bonds(psi[i+1,j]))[0] 
##    return ind_map
#def init_gauge_tanh(psi,D,scale=.5,const=1,log=False,eps=1e-3):
#    nx = psi.Lx * psi.Ly
#    ny = D
#    afn = 'tanh'
#    af = dict()
#    for i,j in itertools.product(range(psi.Lx-1),range(psi.Ly)):
#        nn = Dense(nx,ny,afn)
#        nn.scale = scale 
#        nn.init(0,eps)
#        nn.init(1,eps)
#
#        nn = TensorNN([nn])
#        nn.input_format = (-1,1),None
#        nn.shape = (D,)
#        nn.ind = tuple(psi[i,j].bonds(psi[i+1,j]))[0] 
#        nn.const = const 
#        nn.log = log
#        af[(i,j),(i+1,j)] = nn 
#    nn = GaugeNN2D(psi.Lx,psi.Ly,af)
#    return nn 
#def init_gauge_relu(Lx,Ly,D,const=1,log=False,eps=1e-3):
#    nx = Lx * Ly
#    ny = D
#    afn = 'relu'
#    af = dict()
#    for i,j in itertools.product(range(Lx-1),range(Ly)):
#        nn = Dense(nx,ny,afn)
#        nn = relu_init_normal(nn,-1,1,1/nx,eps)
#
#        nn = TensorNN([nn])
#        nn.input_format = (-1,1),None
#        nn.shape = (D,)
#        nn.const = const 
#        af[(i,j),(i+1,j)] = nn 
#    nn = GaugeNN2D(Lx,Ly,af)
#    return af
#class GaugeNN2D(NN):
#    def __init__(self,Lx,Ly,af):
#        self.Lx,self.Ly = Lx,Ly
#        self.af = af
#        self.keys = list(af.keys())
#        self.backend = af[self.keys[0]].backend
#    def modify_mps_old(self,tn,config,i,step):
#        # for forming boundaries
#        if tn is None:
#            return tn
#        tn = tn.copy()
#        for j in range(self.Ly):
#            where = ((i,j),(i+1,j)) if step==1 else ((i-1,j),(i,j))
#            y = self.af[where].forward(config)
#            ind = self.af[where].ind
#            tn[i,j].multiply_index_diagonal_(ind,y) 
#        return tn 
#    def modify_mps_new(self,*args):
#        raise NotImplementedError
#    def modify_top_mps(self,*args):
#        raise NotImplementedError
#    def modify_bot_mps(self,tn,config,i):
#        # for 2-row contraction into amplitude
#        return self.modify_mps_old(tn,config,i,1)
#    def save_to_disc(self,fname,root=0):
#        if RANK!=root:
#            return
#        f = h5py.File(fname+'.hdf5','w')
#        for i,key in enumerate(self.keys):
#            for j,lr in enumerate(self.af[key].af):
#                for k,tsr in enumerate(lr.params):
#                    f.create_dataset(f'p{key},{j},{k}',data=tensor2backend(tsr,'numpy'))
#        f.close()
#    def load_from_disc(self,fname):  
#        f = h5py.File(fname+'.hdf5','r')
#        for i,key in enumerate(keys):
#            af = self.af[key]
#            for j,lr in enumerate(af.af):
#                for k in range(lr.sh):
#                    lr.params[k] = f[f'p{key},{j},{k}'][:]
#        f.close() 
#
#def out_info(shapes):
#    sizes = [np.array(sh).prod() for sh in shapes]
#    out_sections = {'numpy':np.cumsum(np.array(sizes))[:-1],
#                    'torch':sizes}
#    return sum(sizes),out_sections
#def init_env_tanh(Lx,Ly,D,scale=.5,const=0,log=False,eps=1e-3):
#    afn = 'tanh'
#    sh = [None] * Ly
#    for j in range(Ly):
#        sh[j] = (D,) * 2 if j in (0,Ly-1) else (D,)*3
#    ny,out_sections = out_info(sh)
#    nx = Lx * Ly
#    nn = Dense(nx,ny,afn)
#    nn.scale = scale 
#    nn.init(0,eps)
#    nn.init(1,eps)
#
#    nn = EnvNN2D(Lx,Ly,[nn])
#    nn.input_format = (-1,1),None
#    nn.const = const 
#    nn.log = log
#    nn.shapes = sh
#    nn.out_sections = out_sections 
#    return nn 
#def add(T1,T2,jnp):
#    sh1,sh2 = T1.shape,T2.shape
#    T = jnp.zeros([max(d1,d2) for d1,d2 in zip(sh1,sh2)]) 
#    for shi,Ti in zip((sh1,sh2),(T1,T2)): 
#        T[tuple([slice(d) for d in shi])] += Ti
#    return T
#class EnvNN2D(NN):
#    def __init__(self,Lx,Ly,lr,**kwargs):
#        super().__init__(lr,**kwargs)
#        self.Lx,self.Ly = Lx,Ly
#        self.cache = dict()
#        self._cache = dict()
#        self.shapes = None
#        self.out_sections = None
#    def modify_mps_new(self,*args):
#        raise NotImplementedError
#    def modify_mps_old(self,*args):
#        raise NotImplementedError
#    def forward(self,config,i,step):
#        cache = self.cache if self._backend=='numpy' else self._cache
#        key = cache_key(config,i,'row',step,self.Ly)
#        if key in cache:
#            return cache[key] 
#
#        y = self._input(key[0])
#        for l,lr in enumerate(self.af):
#            y = lr.forward(y,step=step)
#        if self.log:
#            y = self.jnp.exp(y)
#        y += self.const 
#         
#        y = self.jnp.split(y,self.out_sections[self._backend])
#        cache[key] = [y[j].reshape(sh) for j,sh in enumerate(self.shapes)]
#        return cache[key]  
#    def modify_bot_mps(self,tn,config,i,step=1):
#        if tn is None:
#            return tn
#        y = self.forward(config,i,step)
#        for j in range(self.Ly):
#            T = tn[i,j]
#            lind = [] if j==0 else list(T.bonds(tn[i,j-1]))
#            rind = [] if j==self.Ly-1 else list(T.bonds(tn[i,j+1]))
#            where = ((i,j),(i+1,j)) if step==1 else ((i-1,j),(i,j))
#            inds = lind+rind+[self.ind_map[where]]
#            T.transpose_(*inds)
#            T.modify(data=add(y[j],T.data,self.jnp))
#        return tn
#    def modify_top_mps(self,tn,config,i):
#        return self.modify_bot_mps(tn,config,i,step=-1)
#    def free_ad_cache(self):
#        self._cache = dict()
