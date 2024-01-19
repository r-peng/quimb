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
class Layer:
    def __init__(self):
        self.grad = True
    def get_block_dict(self):
        self.block_dict = []
        start = 0
        for sh in self.sh:
            stop = start + np.prod(np.array(sh))
            self.block_dict.append((start,stop))
            start = stop
        self.nparam = stop
    def set_backend(self,backend):
        self.jnp = np if backend=='numpy' else torch 
    def wfn2backend(self,backend=None,requires_grad=False):
        grad = requires_grad if self.grad else False
        self.params = [tensor2backend(p,backend,requires_grad=grad) for p in self.params]
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
    def __init__(self,nv,nh,**kwargs):
        super().__init__(**kwargs)
        self.sh = (nv,),(nh,),(nv,nh)
        self.params = [None] * 3
    def forward(self,x):
        a,b,w = self.params
        return self.jnp.dot(a,x) + self.jnp.sum(self.jnp.log(self.jnp.cosh(self.jnp.matmul(x,w) + b)))
class Dense(Layer):
    def __init__(self,nx,ny,afn,bias=True,**kwargs):
        super().__init__(**kwargs)
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
    def forward(self,x):
        if self.pre_act:
            x = self._afn(x) 
        y = self.apply_w(x)
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
                return self.jnp.log(1+self.exp(x))
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
class NN:
    def __init__(self,af,backend='numpy'):
        self.af = af
        self.nl = len(af)
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
        if backend=='numpy':
            self._input = self.input
            tsr = np.zeros(1)
            self.jnp = np 
        else:
            tsr = torch.zeros(1)
            self.jnp = torch
            def _input(config):
                return tensor2backend(self.input(config),backend) 
            self._input = _input
        ar.set_backend(tsr)
        for key in self.keys:
            self.af[key].set_backend(backend)
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
            af = self.af[key]
            for j in range(len(af.sh)):
                af.params[j] = f[f'p{key},{j}'][:]
        f.close() 
    def save_to_disc(self,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for i,key in enumerate(keys):
            for j,tsr in enumerate(self.af[key].params):
                f.create_dataset(f'p{key},{j}',data=tensor2backend(tsr,'numpy'))
        f.close()
    def update(self,x,fname=None,root=0):
        update(self.af,x,self.sections,keys=self.keys)
        if fname is not None:
            self.save_to_disc(fname,root=root) 
        self.wfn2backend()
    def input_determinant(self,config):
        ls = [None] * 2
        for ix,c in enumerate(config_to_ab(config)):
            ls[ix] = np.where(c)[0]
        c = np.concatenate(ls) 
        return c/len(config)
    def input_bond(self,config):
        ls = []
        nb = len(self.bmap)
        for conf in config_to_ab(config):
            v = np.zeros(nb) 
            for (ix1,ix2),ix in self.bmap.items():
                c1,c2 = conf[ix1],conf[ix2]
                if c1+c2==1:
                    v[ix] = 1
            ls.append(v)
        conf = np.array(config)
        ls.append(np.array([len(conf[conf==3])]))
        return np.concatenate(ls)
    def input_pnsz(self,config):
        v = np.zeros((len(config),2))
        pn_map = [0,1,1,2]
        sz_map = [0,1,-1,0]
        for i,ci in enumerate(config):
            v[i,0] = pn_map[ci]
            v[i,1] = sz_map[ci]
        return v
    def input(self,config):
        _format,_order = self.input_format
        if _format=='det':
            return self.input_determinant(config)
        if _format=='fermion':
            return np.array(config,dtype=float) 
        if _format=='bond':
            return self.input_bond(config)
        if _format=='pnsz':
            return self.input_pnsz(config)
        if _format=='conv1':
            return np.array(config,dtype=float).reshape(len(config),1) 
        if _format=='conv2':
            return np.stack([np.array(cf,dtype=float) for cf in config_to_ab(config)],axis=1) * 2 - 1
        if _order is not None:
            config = np.stack([np.array(cf,dtype=float) for cf in config_to_ab(config)],axis=0).flatten(order=_order)
        else:
            config = np.array(config,dtype=float)
        xmin,xmax = _format
        return config * (xmax-xmin) + xmin 
class AmplitudeNN(NN):
    def __init__(self,af,**kwargs):
        super().__init__(af,**kwargs)

        self.is_tn = False
        self.from_plq = False
        self.spin = None
        self.vx = None

        # to be set
        self.fermion = None 
        self.sum_all = None 
        self.log = None 
        self.phase = None 
    def pair_terms(self,i1,i2,spin=None):
        if self.fermion:
            return pair_terms(i1,i2,spin) 
        else:
            return i2,i1
    def forward(self,config):
        y = self._input(config)
        _sum = 0
        for l,af in enumerate(self.af):
            y = af.forward(y)
            if l==self.nl-1 or self.sum_all:
                _sum = _sum + y.sum() 
        return _sum + self.const 
    def log_prob(self,config):
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
class TensorNN(NN):
    def __init__(self,af,**kwargs):
        super().__init__(af,**kwargs)
        #self.tensor = None
        #self._tensor = None

        # to be set
        self.log = None
    def forward(self,config):
        #tensor = self.tensor if self._backend=='numpy' else self._tensor
        #if tensor is not None:
        #    _config,y = tensor 
        #    if _config==config:
        #        return y
        
        y = self._input(config)
        for l,af in enumerate(self.af):
            y = af.forward(y)
        if self.log:
            y = self.jnp.exp(y)
        y += self.const 
        #if self._backend=='numpy':
        #    self.tensor = config,y
        #else:
        #    self._tensor = config,y
        return y
    def free_ad_cache(self):
        self._tensor = None
 
def rotate(x,eps):
    nx = len(x)
    K = np.random.normal(loc=0,scale=eps,size=(nx,nx))
    U = scipy.linalg.expm(K-K.T)
    return np.dot(U,x) 
def compute_colinear(w,ny=None,eps=0,cos_max=.9,thresh=1e-6):
    ny = w.shape[0] if ny is None else ny
    nx = w.shape[1]
    nc = 0
    ls = []
    for i in range(w.shape[0]):
        if len(ls)==ny:
            break
        norm = np.linalg.norm(w[i,:]) 
        if norm < thresh:
            continue
        wi = rotate(w[i,:] / norm,eps)
        for wj in ls:
            if np.fabs(np.dot(wi,wj)) > cos_max:
                nc += 1
        ls.append(wi)
    if RANK==0:
        print('collinear ratio=',nc/(ny*(ny-1)/2))
    return np.array(ls) 
def relu_init_rand(nx,ny,xmin,xmax):
    w = np.random.rand(ny,nx) * 2 - 1
    w = compute_colinear(w)
    x = np.random.rand(ny,nx) * (xmax - xmin) + xmin 
    b = np.sum(w*x,axis=1) 
    return w,b
def relu_init_normal(af,xmin,xmax,scale,eps):
    if RANK==0:
        w = np.array([np.random.normal(loc=0,scale=scale,size=af.nx) for _ in range(af.ny)])
        w = compute_colinear(w)
        loc = (xmin+xmax) / 2 
        x = np.array([np.random.normal(loc=loc,scale=eps,size=af.nx) for _ in range(af.ny)])
        b = np.sum(w*x,axis=1) 
    else:
        w = np.zeros((af.ny,af.nx))
        b = np.zeros(af.ny)
    COMM.Bcast(w,root=0)
    COMM.Bcast(b,root=0)
    af.params[0] = w.T * eps
    af.params[1] = b * eps
    return af
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
