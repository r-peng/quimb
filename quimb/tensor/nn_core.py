import time,itertools,h5py,scipy
import numpy as np
from .tensor_vmc import (
    tensor2backend,
    tensor_grad,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

import autoray as ar
import torch
ar.register_function('torch','relu',torch.nn.functional.relu)
def config_to_ab(config):
    config = np.array(config)
    return tuple(config % 2), tuple(config // 2)
def config_from_ab(config_a,config_b):
    return tuple(np.array(config_a) + np.array(config_b) * 2)
def pair_terms(i1,i2,spin):
    if spin=='a':
        map_ = {(0,1):(1,0),(1,0):(0,1),
                (2,3):(3,2),(3,2):(2,3),
                (0,3):(1,2),(3,0):(2,1),
                (1,2):(0,3),(2,1):(3,0)}
    elif spin=='b':
        map_ = {(0,2):(2,0),(2,0):(0,2),
                (1,3):(3,1),(3,1):(1,3),
                (0,3):(2,1),(3,0):(1,2),
                (1,2):(3,0),(2,1):(0,3)}
    else:
        raise ValueError
    return map_.get((i1,i2),(None,)*2)
class AmplitudeFactory:
    def pair_terms(self,i1,i2,spin=None):
        if self.fermion:
            return pair_terms(i1,i2,spin) 
        else:
            return i2,i1
    def log_prob(self,config,**kwargs):
        if self.phase:
            return 1
        cx = self.forward(config)
        cx = tensor2backend(cx,'numpy') 
        if self.log:
            return 2 * cx 
        else:
            return np.log(cx**2) 
    def amplitude(self,config,to_numpy=True):
        cx = self.forward(config)
        if self.log:
            logcx = cx
            if self.phase:
                logcx = 1j*logcx
            cx = self.jnp.exp(logcx)
        if to_numpy:
            cx = tensor2backend(cx,'numpy')
        return cx
    def batch_pair_energies(self,batch_key,new_cache):
        spins = ('a','b') if self.fermion else (None,)

        cx = self.forward(self.config)
        if self.log:
            logcx = cx 
            if self.phase: 
                logcx = 1j * logcx
            cx = self.jnp.exp(logcx)

        self.cx['deterministic'] = cx
        ex = dict()
        for where in self.model.batched_pairs[batch_key].pairs:
            ix1,ix2 = [self.flatten(site) for site in where]
            i1,i2 = self.config[ix1],self.config[ix2]
            if not self.model.pair_valid(i1,i2): # term vanishes 
                for spin in spins:
                    ex[where,spin] = 0,cx,0
                continue
            for spin in spins:
                i1_new,i2_new = self.pair_terms(i1,i2,spin)
                if i1_new is None:
                    ex[where,spin] = 0,cx,0 
                    continue
                config_new = list(self.config)
                config_new[ix1] = i1_new
                config_new[ix2] = i2_new 
                cx_new = self.forward(config_new) 
                if self.log:
                    logcx_new = cx_new
                    if self.phase:
                        logcx_new = 1j * logcx_new
                    cx_new = self.jnp.exp(logcx_new) 
                    ratio = cx_new / cx if logcx is None else \
                            self.jnp.exp(logcx_new-logcx)   
                else:
                    ratio = cx_new / cx
                ex[where,spin] = cx_new, cx, ratio 
        return ex,cx
    def get_grad_deterministic(self,config,save=False):
        if self.vx is not None:
            cx = self.cx.get('deterministic',None)
            return cx,self.vx
        self.wfn2backend(backend='torch',requires_grad=True)
        cx = self.forward(config) 
        cx,self.vx = self.propagate(cx) 
        if self.log:
            if self.phase:
                self.vx = 1j * self.vx
                cx = 1j * cx
            cx = np.exp(cx) 
        else:
            self.vx /= cx
        self.wfn2backend()
        if save:
            self.cx['deterministic'] = cx
        return cx,self.vx 
    def get_grad_from_plq(self,plq):
        return self.get_grad_deterministic(self.config,save=True)[1]

class Layer:
    def get_block_dict(self):
        self.block_dict = []
        start = 0
        for sh in self.sh:
            stop = start + np.prod(np.array(sh))
            self.block_dict.append((start,stop))
            start = stop
        self.nparam = stop
    def set_backend(self,backend):
        self._backend = backend
        self.jnp = np if backend=='numpy' else torch 
    def wfn2backend(self,backend=None,requires_grad=False):
        self.params = [tensor2backend(p,backend,requires_grad=requires_grad) for p in self.params]
    def get_x(self,grad=False):
        ls = []
        for p in self.params:
            tsr = tensor_grad(p) if grad else p
            ls.append(tensor2backend(tsr,'numpy').flatten())
        return np.concatenate(ls)
    def extract_ad_grad(self):
        return self.get_x(grad=True)
    def update(self,x):
        for i,sh in enumerate(self.sh):
            start,stop = self.block_dict[i]
            size = stop - start
            xi,x = x[:size],x[size:]
            self.params[i] = xi.reshape(sh) 
    def init(self,key,eps,loc=0,iprint=False,normal=True):
        if normal:
            tsr = np.random.normal(loc=loc,scale=eps,size=self.sh[key])
        else:
            tsr = (np.random.rand(*self.sh[key])*2-1) * eps + loc 
        COMM.Bcast(tsr,root=0)
        self.params[key] = tsr
        if RANK==0 and iprint:
            print(key)
            print(tsr)
class RBM(Layer):
    def __init__(self,nv,nh):
        self.sh = (nv,),(nh,),(nv,nh)
        self.params = [None] * 3
    def forward(self,x):
        a,b,w = self.params
        return self.jnp.dot(a,x) + self.jnp.sum(self.jnp.log(self.jnp.cosh(self.jnp.matmul(x,w) + b)))
class Dense(Layer):
    def __init__(self,nx,ny,afn,bias=True):
        self.nx,self.ny = nx,ny
        self.afn = afn
        self.sh = [(nx,ny),(ny,)]
        self.params = [None] * 2
        self.bias = bias
        if not bias:
            self.sh.pop()
            self.params.pop()

        # to be set 
        self.combine = False 
        self.pre_act = False 
        self.post_act = True 
        self.scale = 1
    def apply_w(self,x,step=1):
        dim = len(x)
        w = self.params[0]
        if dim==w.shape[0]:
            return self.jnp.matmul(x,w)    
        if dim>w.shape[0]:
            raise ValueError
        if step==1: 
            return self.jnp.matmul(x,w[:dim,:])    
        else:
            return self.jnp.matmul(x,w[-dim:,:])    
    def _combine(self,x,y):
        if not self.combine:
            return y
        try:
            return self.jnp.concatenate([x,y])
        except:
            return self.jnp.cat([x,y])
    def forward(self,x,step=1):
        if self.pre_act:
            x = self._afn(x) 
        y = self.apply_w(x,step=step)
        if self.bias:
            y = y + self.params[1]
        if self.post_act:
            y = self._afn(y)
        return self._combine(x,y)
    def set_backend(self,backend):
        super().set_backend(backend)
        _afn = None
        if self.afn=='tanh':
            def _afn(x):
                return self.scale * self.jnp.tanh(x)
        if self.afn=='relu':
            def _afn(x):
                try:
                    return self.jnp.relu(x)
                except AttributeError:
                    return x*(x>0)
        if self.afn=='softplus':
            def _afn(x):
                return self.jnp.log(1+self.jnp.exp(x))
        if self.afn=='sinh':
            def _afn(x):
                return self.jnp.sinh(x)
        if self.afn=='cosh':
            def _afn(x):
                return self.jnp.cosh(x)
        if self.afn=='exp':
            def _afn(x):
                return self.jnp.exp(x)
        self._afn = _afn
def input_determinant(config):
    ls = [None] * 2
    for ix,c in enumerate(config_to_ab(config)):
        ls[ix] = np.where(c)[0]
    c = np.concatenate(ls) 
    return c/len(config)
def input_bond(config,bmap):
    ls = []
    nb = len(bmap)
    for conf in config_to_ab(config):
        v = np.zeros(nb) 
        for (ix1,ix2),ix in bmap.items():
            c1,c2 = conf[ix1],conf[ix2]
            if c1+c2==1:
                v[ix] = 1
        ls.append(v)
    conf = np.array(config)
    ls.append(np.array([len(conf[conf==3])]))
    return np.concatenate(ls)
def input_pnsz(config):
    v = np.zeros((len(config),2))
    pn_map = [0,1,1,2]
    sz_map = [0,1,-1,0]
    for i,ci in enumerate(config):
        v[i,0] = pn_map[ci]
        v[i,1] = sz_map[ci]
    return v
def _input(config,input_format,backend,bmap=None):
    _format,_order = input_format
    if _format=='det':
        x = input_determinant(config)
    elif _format=='fermion':
        x = np.array(config,dtype=float) 
    elif _format=='bond':
        x = input_bond(config,bmap)
    elif _format=='pnsz':
        x = input_pnsz(config)
    elif _format=='conv1':
        x = np.array(config,dtype=float).reshape(len(config),1) 
    elif _format=='conv2':
        x = np.stack([np.array(cf,dtype=float) for cf in config_to_ab(config)],axis=1) * 2 - 1
    else:
        if _order is not None:
            config = np.stack([np.array(cf,dtype=float) for cf in config_to_ab(config)],axis=0).flatten(order=_order)
        else:
            config = np.array(config,dtype=float)
        xmin,xmax = _format
        x = config * (xmax-xmin) + xmin 
    if backend=='numpy':
        return x 
    else:
        return tensor2backend(x,backend) 
class Fourier(Layer,AmplitudeFactory):
    def __init__(self,nx,ny,backend='numpy'):
        self.nx,self.ny = nx,ny
        self.sh = [(2,nx,ny),(2,ny)]
        self.params = [None] * len(self.sh) 

        self.backend = backend
        self.set_backend(backend)

        self.is_tn = False
        self.from_plq = False
        self.spin = None
        self.vx = None

        # to be set
        self.fermion = None 
        self.log = None 
        self.phase = None 
        self.const = None
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self.set_backend(backend)
        super().wfn2backend(backend=backend,requires_grad=requires_grad)
    def set_backend(self,backend):
        super().set_backend(backend)
        ar.set_backend(self.jnp.zeros(1))
    def forward(self,config):
        y = _input(config,self.input_format,self._backend)
        w,b = self.params
        ls = [self.jnp.matmul(y,w[ix])+b[ix] for ix in (0,1)]
        try:
            ls[0] = self.jnp.relu(ls[0]) 
        except AttributeError:
            ls[0] = ls[0] * (ls[0]>0)
        ls[1] = self.jnp.sin(np.pi*ls[1])
        return self.jnp.dot(ls[0],ls[1])/self.nx + self.const
    def load_from_disc(self,fname):
        f = h5py.File(fname+'.hdf5','r')
        for j in range(len(self.sh)):
            self.params[j] = f[f'p{j}'][:]
        f.close() 
    def save_to_disc(self,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for j,tsr in enumerate(self.params):
            f.create_dataset(f'p{j}',data=tensor2backend(tsr,'numpy'))
        f.close()
    def update(self,x,fname=None,root=0):
        super().update(x)
        if fname is not None:
            self.save_to_disc(fname,root=root) 
        self.wfn2backend()
def get_block_dict(afs,keys=None):
    keys = range(len(afs)) if keys is None else keys
    [afs[key].get_block_dict() for key in keys]

    nparam = np.array([afs[key].nparam for key in keys])
    sections = np.cumsum(nparam)[:-1]

    block_dict = []
    for i,key in enumerate(keys):
        shift = 0 if i==0 else sections[i-1]
        block_dict += [(start+shift,stop+shift) for start,stop in afs[key].block_dict]
    nparam = sum(nparam)
    return nparam,sections,block_dict 
def wfn2backend(afs,backend,requires_grad,keys=None):
    keys = range(len(afs)) if keys is None else keys
    [afs[key].wfn2backend(backend=backend,requires_grad=requires_grad)for key in keys]
def get_x(afs,keys=None):
    keys = range(len(afs)) if keys is None else keys
    return np.concatenate([afs[key].get_x() for key in keys])
def extract_ad_grad(afs,keys=None):
    keys = range(len(afs)) if keys is None else keys
    return np.concatenate([afs[key].extract_ad_grad() for key in keys]) 
def update(afs,x,sections,keys=None):
    keys = range(len(afs)) if keys is None else keys
    x = np.split(x,sections)
    for i,key in enumerate(keys):
        afs[key].update(x[i])
def free_ad_cache(afs,keys=None):
    keys = range(len(afs)) if keys is None else keys
    for key in keys:
        try:
            afs[key].free_ad_cache()
        except:
            continue
def free_sweep_cache(afs,step,keys=None):
    keys = range(len(afs)) if keys is None else keys
    for key in keys:
        try:
            afs[key].free_sweep_cache(step)
        except:
            continue
class FNN:
    def __init__(self,lr,backend='numpy'):
        self.af = lr
        self.nl = len(lr)
        self.keys = range(self.nl) 
        self.backend = backend
        self.set_backend(backend)

        # to be set
        self.input_format = None 
        self.const = None
    def free_ad_cache(self):
        free_ad_cache(self.af,keys=self.keys) 
    def get_block_dict(self):
        self.nparam,self.sections,self.block_dict = get_block_dict(self.af,keys=self.keys)
    def set_backend(self,backend):
        self._backend = backend
        for key in self.keys:
            self.af[key].set_backend(backend)
        self.jnp = self.af[key].jnp
        ar.set_backend(self.jnp.zeros(1))
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self.set_backend(backend)
        wfn2backend(self.af,backend,requires_grad,keys=self.keys)
    def get_x(self):
        return get_x(self.af,keys=self.keys) 
    def extract_ad_grad(self):
        return extract_ad_grad(self.af,keys=self.keys) 
    def load_from_disc(self,fname):
        f = h5py.File(fname+'.hdf5','r')
        for i,key in enumerate(self.keys):
            lr = self.af[key]
            for j in range(len(lr.sh)):
                lr.params[j] = f[f'p{key},{j}'][:]
        f.close() 
    def save_to_disc(self,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for i,key in enumerate(self.keys):
            for j,tsr in enumerate(self.af[key].params):
                f.create_dataset(f'p{key},{j}',data=tensor2backend(tsr,'numpy'))
        f.close()
    def update(self,x,fname=None,root=0):
        update(self.af,x,self.sections,keys=self.keys)
        if fname is not None:
            self.save_to_disc(fname,root=root) 
        self.wfn2backend()
class AmplitudeFNN(FNN,AmplitudeFactory):
    def __init__(self,lr,**kwargs):
        super().__init__(lr,**kwargs)

        self.is_tn = False
        self.from_plq = False
        self.spin = None
        self.vx = None

        # to be set
        self.fermion = None 
        self.sum_all = None 
        self.log = None 
        self.phase = None 
    def forward(self,config):
        y = _input(config,self.input_format,self._backend)
        _sum = 0
        for l,lr in enumerate(self.af):
            y = lr.forward(y)
            if l==self.nl-1 or self.sum_all:
                _sum = _sum + y.sum() 
        return _sum + self.const 
class TensorFNN(FNN):
    def __init__(self,lr,**kwargs):
        super().__init__(lr,**kwargs)
        # to be set
        self.log = None
        self.ind = None
    def forward(self,config):
        y = _input(config,self.input_format,self._backend)
        for l,lr in enumerate(self.af):
            y = lr.forward(y)
        if self.log:
            y = self.jnp.exp(y)
        y = y.reshape(self.shape) + self.const 
        return y
    def free_ad_cache(self):
        self._tensor = None
 
def rotate(x,scale):
    nx = len(x)
    K = np.random.normal(loc=0,scale=scale,size=(nx,nx))
    U = scipy.linalg.expm(K-K.T)
    return np.dot(U,x) 
def compute_colinear(nx,ny,scale,cos_max=.9,thresh=1e-6):
    nc = 0
    while True:
        w = np.random.normal(loc=0,scale=scale,size=nx)
        norm = np.linalg.norm(w)
        if norm < thresh:
            continue
        w /= norm 
        break
    ls = [w] 
    for i in range(ny-1):
        w = rotate(ls[-1],scale)
        for wj in ls:
            if np.fabs(np.dot(w,wj)) > cos_max:
                nc += 1
        ls.append(w)
    if RANK==0:
        print('collinear ratio=',nc/(ny*(ny-1)/2))
    return np.array(ls) 
def relu_init_rand(nx,ny,xmin,xmax):
    w = np.random.rand(ny,nx) * 2 - 1
    w = compute_colinear(w)
    x = np.random.rand(ny,nx) * (xmax - xmin) + xmin 
    b = np.sum(w*x,axis=1) 
    return w,b
def relu_init_normal(nx,ny,xmin,xmax,s1=1,s2=None):
    if RANK==0:
        w = compute_colinear(nx,ny,s1)
        loc = (xmin+xmax) / 2 
        s2 = (xmax-xmin) / 4 if s2 is None else s2
        x = np.array([np.random.normal(loc=loc,scale=s2,size=nx) for _ in range(ny)])
        b = np.sum(w*x,axis=1) 
    else:
        w = np.zeros((ny,nx))
        b = np.zeros(ny)
    COMM.Bcast(w,root=0)
    COMM.Bcast(b,root=0)
    return w.T,b 
def relu_init_sobol(nx,ny,xmin,xmax,eps):
    import qmcpy
    sampler = qmcpy.discrete_distribution.digital_net_b2.Sobol(dimension=nx,randomize=False)
    m = int(np.log2(ny*2)) + 1
    p = sampler.gen_samples(2**m) 
    w = p * 2 - 1
    w = compute_colinear(w[ny:,],ny=ny,eps=eps)
    x = p[:ny] * (xmax - xmin) + xmin
    x += np.random.normal(loc=0,scale=eps,size=(ny,nx))  
    b = np.sum(w*x,axis=1) 
    return w,b
def relu_init_grid(nx,ndiv,xmin,xmax,eps):
    # generate ndiv * nx plaines
    w = []
    b = []
    dx = (xmax - xmin) / ndiv
    for i in range(nx):
        wi = np.zeros(nx)
        wi[i] = 1
        for n in range(ndiv):
            w.append(rotate(wi,eps))

            x = np.random.normal(loc=0,scale=eps,size=nx)
            x[i] += n * dx + xmin + dx / 2
 
            b.append(np.dot(w[-1],x))
    return np.array(w),np.array(b)
def relu_init_quad(nx,ndiv,xmin,xmax,eps):
    # generate ndiv * nx * (nx-1) / 2 plaines
    w = []
    b = [] 
    dtheta = np.pi / ndiv
    center = (xmin + xmax) / 2 
    for i in range(nx):
        for j in range(i):
            theta0 = np.random.normal(loc=0,scale=np.pi)
            for n in range(ndiv):
                theta = theta0 + dtheta * n 
                wi = np.zeros(nx) 
                wi[i] = np.cos(theta)
                wi[j] = np.sin(theta)
                w.append(rotate(wi,eps))
           
                x = np.random.normal(loc=0,scale=eps*xmax,size=nx)
                x[i] += center 
                x[j] += center 

                b.append(np.dot(w[-1],x))
    return np.array(w),np.array(b)
def relu_init_spin(nx,eps,eps_init=None):
    if eps_init is None:
        K = np.random.rand(nx,nx)
    else:
        K = np.random.normal(loc=0,scale=eps_init,size=(nx,nx))
    U = scipy.linalg.expm(K-K.T)
    w = []
    b = []
    for i in range(nx):
        w.append(rotate(U[i,:],eps))
        x = np.random.normal(loc=0,scale=eps,size=nx)
        b.append(np.dot(w[-1],x))
    return np.array(w),np.array(b)
