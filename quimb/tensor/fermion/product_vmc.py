import numpy as np
import itertools
import scipy
#####################################################
# for separate ansatz
#####################################################
from ..tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    AmplitudeFactory,
)
from ..product_vmc import (
    ProductAmplitudeFactory,
    NN,
    RBM,
    FNN,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
def config_to_ab(config):
    config = np.array(config)
    return tuple(config % 2), tuple(config // 2)
def config_from_ab(config_a,config_b):
    return tuple(np.array(config_a) + np.array(config_b) * 2)
class FermionProductAmplitudeFactory(ProductAmplitudeFactory):
    def parse_config(self,config):
        ca,cb = config_to_ab(config)
        return [{'a':ca,'b':cb,None:config}[af.spin] for af in self.af]
    def parse_energy(self,ex,batch_key,cx=None):
        pairs = self.model.batched_pairs[batch_key][3]
        e = 0.
        p = 1 if cx is None else 0
        for where,spin in itertools.product(pairs,('a','b')):
            term = 1.
            for ix,ex_ in enumerate(ex):
                if (where,spin) in ex_:
                    term *= ex_[where,spin][p] 
                else:
                    if p==1:
                        continue 
                    if isinstance(cx[ix],dict):
                        term *= cx[ix][where][0]
                    else:
                        term *= cx[ix]
            e += term
        if p==1:
            e = tensor2backend(e,'numpy')
        return e
#######################################################################
# some jastrow forms
#######################################################################
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
class TNJastrow(AmplitudeFactory):
    def update_pair_energy_from_plq(self,tn,where):
        ix1,ix2 = [self.flatten(site) for site in where]
        i1,i2 = self.config[ix1],self.config[ix2] 
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return {spin:0 for spin in ('a','b')}
        ex = dict()
        for spin in ('a','b'):
            i1_new,i2_new = pair_terms(i1,i2,spin)
            if i1_new is None:
                ex[spin] = 0
                continue
            tn_new = self.replace_sites(tn.copy(),where,(i1_new,i2_new))
            ex_ij = safe_contract(tn_new)
            if ex_ij is None:
                ex_ij = 0
            ex[spin] = ex_ij 
        return ex 
import autoray as ar
import torch
import h5py
def to_spin(config,order='F'):
    ca,cb = config_to_ab(config) 
    return np.stack([np.array(tsr,dtype=float) for tsr in (ca,cb)],axis=0).flatten(order=order)
class FermionNN(NN):
    def __init__(self,to_spin=True,order='F',**kwargs):
        self.to_spin = to_spin
        self.order = order
        super().__init__(**kwargs)
    def log_amplitude(self,config,to_numpy=True):
        c = to_spin(config,self.order) if self.to_spin else np.array(config,dtype=float)
        return super().log_amplitude(c,to_numpy=to_numpy)
    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
        c = to_spin(config,self.order) if self.to_spin else np.array(config,dtype=float)
        return super().log_amplitude(c,to_numpy=to_numpy)
    def batch_pair_energies_from_plq(self,batch_key,new_cache=None):
        bix,tix,plq_types,pairs,direction = self.model.batched_pairs[batch_key]
        jnp = torch if new_cache else np
        if self.log_amp:
            logcx,sx = self.log_amplitude(self.config,to_numpy=False)
            cx = jnp.exp(logcx) * sx
        else:
            cx = self.unsigned_amplitude(self.config,to_numpy=False)
        ex = dict()
        for where in pairs:
            ix1,ix2 = [self.flatten(site) for site in where]
            i1,i2 = self.config[ix1],self.config[ix2]
            if self.model.pair_valid(i1,i2): # term vanishes 
                for spin in ('a','b'):
                    i1_new,i2_new = pair_terms(i1,i2,spin)
                    if i1_new is None:
                        ex[where,spin] = 0,0 
                    else:
                        config_new = list(self.config)
                        config_new[ix1] = i1_new
                        config_new[ix2] = i2_new 
                        if self.log_amp:
                            logcx_new,sx_new = self.log_amplitude(config_new,to_numpy=False) 
                            cx_new = jnp.exp(logcx_new) * sx_new
                            ex[where,spin] = cx_new,jnp.exp(logcx_new-logcx) * sx_new / sx 
                        else:
                            cx_new = self.unsigned_amplitude(config_new,to_numpy=False)
                            ex[where,spin] = cx_new, cx_new/cx 
            else:
                for spin in ('a','b'):
                    ex[where,spin] = 0,0
        return ex,cx,None
class FermionRBM(FermionNN,RBM):
    def __init__(self,nv,nh,**kwargs):
        self.nv,self.nh = nv,nh
        self.nparam = nv + nh + nv * nh 
        self.block_dict = [(0,nv),(nv,nv+nh),(nv+nh,self.nparam)]
        super().__init__(**kwargs)
class FermionFNN(FermionNN,FNN):
    def __init__(self,nv,nl,afn='logcosh',**kwargs):
        self.nv = nv
        self.nl = nl # number of hidden layer
        assert afn in ('logcosh','logistic','tanh','softplus','silu','cos')
        self.afn = afn 
        super().__init__(**kwargs)
class FermionSIGN(FermionFNN):
    def __init__(self,nv,nl,afn='tanh',**kwargs):
        super().__init__(nv,nl,afn=afn,log_amp=False,**kwargs)
    def forward(self,c,jnp):
        c = super().forward(c,jnp) 
        return self._afn(c)
class ORB(NN): # 1-particle orbital rotation
    def __init__(self,nsite,nelec,spin,orth=True,**kwargs):
        super().__init__(log_amp=False)
        self.nsite = nsite
        self.nelec = nelec
        self.spin = spin
        self.orth = orth 
        self.nparam = nsite * nelec
        self.block_dict = [(0,self.nparam)] 
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
        ar.set_backend(tsr)
        self.U = tensor2backend(self.U,backend=backend,requires_grad=requires_grad) 
    def get_x(self):
        return tensor2backend(self.U,'numpy').flatten()
    def load_from_disc(self,fname):
        self.U = np.load(fname) 
        return self.U
    def save_to_disc(self,U,fname,root=0): 
        if RANK!=root:
            return
        np.save(fname+'.npy',U)
    def update(self,x,fname=None,root=0):
        self.U = x.reshape((self.nsite,self.nelec))
        if self.orth:
            self.U,_ = np.linalg.qr(self.U)
        if fname is not None:
            self.save_to_disc(self.U,fname,root=root) 
        self.wfn2backend()
    def extract_grad(self):
        return tensor2backend(self.tensor_grad(self.U),'numpy').flatten()
    def get_backend(self):
        if isinstance(self.U,torch.Tensor):
            jnp = torch
            def _select(det,U):
                return torch.index_select(U,0,torch.tensor(det[0]))
        else:
            jnp = np
            def _select(det,U):
                return U[det,:]
        self._select = _select
        return jnp
    def forward(self,config,jnp):
        det = np.where(np.array(config))
        return jnp.linalg.det(self._select(det,self.U)) 
    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
        jnp = self.get_backend() 
        c = self.forward(config,jnp) 
        if to_numpy:
            c = tensor2backend(c,'numpy') 
        return c
    def log_amplitude(self,config,to_numpy=True):
        jnp = self.get_backend() 
        c = self.forward(config,jnp) 
        c,s = jnp.log(jnp.abs(c)),jnp.sign(c)
        if to_numpy:
            c = tensor2backend(c,'numpy') 
            s = tensor2backend(s,'numpy') 
        return c,s
def BackFlow(FNN):
    def __init__(self,nsite,nelec,nv,nl,**kwargs):
        self.nsite = nsite
        self.nelec = nelec 
        super().__init__(nv,nl,log_amp=False,**kwargs)
    
#class ORB(NN):
#    def __init__(self,nsite,nelec,**kwargs):
#        self.nsite = nsite
#        self.nelec = nelec
#        self.nparam = 2 * nsite ** 2
#        self.block_dict = [(0,nsite**2),(nsite**2,self.nparam)]
#        super().__init__(to_spin=True,order='C',log_amp=False,**kwargs)
#    def init(self,eps,a=-1,b=1,fname=None):
#        c = b-a
#        self.K = (np.random.rand(2,self.nsite,self.nsite) * c + a) * eps
#        COMM.Bcast(self.K,root=0)
#        if fname is not None:
#            self.save_to_disc(self.K,fname)
#        self.U = [scipy.linalg.expm(self.K[ix]-self.K[ix].T)[:,:self.nelec[ix]] for ix in (0,1)]
#    def wfn2backend(self,backend=None,requires_grad=False):
#        backend = self.backend if backend is None else backend
#        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
#        ar.set_backend(tsr)
#        self.K = tensor2backend(self.K,backend=backend,requires_grad=requires_grad) 
#        expm = scipy.linalg.expm if backend=='numpy' else torch.linalg.matrix_exp
#        self.U = [expm(self.K[ix]-self.K[ix].T)[:,:self.nelec[ix]] for ix in (0,1)]
#    def get_x(self):
#        return tensor2backend(self.K,'numpy').flatten()
#    def load_from_disc(self,fname):
#        self.K = np.load(fname) 
#        self.U = [scipy.linalg.expm(self.K[ix]-self.K[ix].T)[:,:self.nelec[ix]] for ix in (0,1)]
#        return self.K
#    def save_to_disc(self,K,fname,root=0): 
#        if RANK!=root:
#            return
#        np.save(fname+'.npy',K)
#    def update(self,x,fname=None,root=0):
#        self.K = x.reshape((2,)+(self.nsite,)*2)
#        if fname is not None:
#            self.save_to_disc(self.K,fname,root=root) 
#        self.wfn2backend()
#    def extract_grad(self):
#        return tensor2backend(self.tensor_grad(self.K),'numpy').flatten()
#    def get_backend(self):
#        if isinstance(self.K,torch.Tensor):
#            jnp = torch
#            def _select(det,U):
#                return torch.index_select(U,0,torch.tensor(det[0]))
#        else:
#            jnp = np
#            def _select(det,U):
#                return U[det,:]
#        self._select = _select
#        return jnp,None
#    def forward(self,config,jnp):
#        config = config_to_ab(config) 
#        c = 1. 
#        for config_,U_ in zip(config,self.U):
#            det = np.where(np.array(config_))
#            c *= jnp.linalg.det(self._select(det,U_)) 
#        return c
#    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
#        jnp,_ = self.get_backend() 
#        c = self.forward(config,jnp) 
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#        return c
#    def log_amplitude(self,config,to_numpy=True):
#        jnp,_ = self.get_backend() 
#        c = self.forward(config,jnp) 
#        c,s = jnp.log(jnp.abs(c)),jnp.sign(c)
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#            s = tensor2backend(s,'numpy') 
#        return c,s
#class SIGN(NN):
#    def __init__(self,n,afn='tanh',**kwargs):
#        assert afn in ('tanh','cos','sin') 
#        self.afn = afn
#        self.nparam = n 
#        self.block_dict = [(0,n)] 
#        super().__init__(log_amp=False,**kwargs)
#    def init(self,eps,fname=None):
#        self.w = (np.random.rand(self.nparam) * 2 - 1) * eps 
#        COMM.Bcast(self.w,root=0)
#        if fname is not None: 
#            self.save_to_disc(self.w,fname) 
#        return self.w
#    def init_from(self,w,eps,fname):
#        self.init(eps)
#        self.w[:len(w)] = w
#        if fname is not None: 
#            self.save_to_disc(self.w,fname) 
#        return self.w
#    def wfn2backend(self,backend=None,requires_grad=False):
#        backend = self.backend if backend is None else backend
#        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
#        ar.set_backend(tsr)
#        self.w = tensor2backend(self.w,backend=backend,requires_grad=requires_grad)
#    def get_x(self):
#        return tensor2backend(self.w,'numpy')
#    def load_from_disc(self,fname):
#        self.w = np.load(fname)
#        return self.w
#    def save_to_disc(self,w,fname,root=0):
#        if RANK!=root:
#            return
#        np.save(fname+'.npy',w)
#    def update(self,x,fname=None,root=0):
#        self.w = x
#        if fname is not None:
#            self.save_to_disc(self.w,fname,root=root) 
#        self.wfn2backend()
#    def extract_grad(self):
#        return tensor2backend(self.tensor_grad(self.w),'numpy')
#    def get_backend(self,c=None):
#        if isinstance(self.w,torch.Tensor):
#            if c is not None:
#                c = tensor2backend(c,backend='torch')
#            jnp = torch
#        else:
#            jnp = np
#        if self.afn=='tahn':
#            _afn = jnp.tahn
#        elif self.afn=='cos':
#            def _afn(x):
#                return jnp.cos(np.pi*x)    
#        elif self.afn=='sin':
#            def _afn(x):
#                return jnp.sin(np.pi*x)
#        else:
#            raise NotImplementedError
#        self._afn = _afn 
#        return jnp,c
#    def forward(self,c,jnp):
#        return self._afn(jnp.dot(c,self.w))
