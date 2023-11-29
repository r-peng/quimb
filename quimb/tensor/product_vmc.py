import numpy as np
import itertools
import scipy
#####################################################
# for separate ansatz
#####################################################
from .tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    AmplitudeFactory,
    scale_wfn,
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
class CompoundAmplitudeFactory(AmplitudeFactory):
    def __init__(self,af,fermion=False):
        self.af = af 
        self.Lx,self.Ly = af[0].Lx,af[0].Ly
        self.get_sections()

        self.sites = af[0].sites
        self.model = af[0].model
        self.nsite = af[0].nsite
        self.backend = af[0].backend

        self.pbc = af[0].pbc 
        self.from_plq = af[0].from_plq
        self.deterministic = af[0].deterministic 

        self.fermion = fermion 
        if self.fermion:
            self.spinless = af[0].spinless

        self.flatten = af[0].flatten
        self.flat2site = af[0].flat2site
        self.intermediate_sign = af[0].intermediate_sign
    def extract_grad(self):
        Hvx = [af.extract_grad() for af in self.af] 
        return np.concatenate(Hvx) 
    def parse_config(self,config):
        if self.fermion:
            ca,cb = config_to_ab(config)
            return [{'a':ca,'b':cb,None:config}[af.spin] for af in self.af]
        else:
            return [config] * self.naf
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        for af in self.af:
            af.wfn2backend(backend=backend,requires_grad=requires_grad)
    def get_x(self):
        return np.concatenate([af.get_x() for af in self.af])
    def get_sections(self):
        self.naf = len(self.af)
        self.nparam = np.array([af.nparam for af in self.af])
        self.sections = np.cumsum(self.nparam)[:-1]

        self.block_dict = self.af[0].block_dict.copy()
        for af,shift in zip(self.af[1:],self.sections):
            self.block_dict += [(start+shift,stop+shift) for start,stop in af.block_dict]
    def update(self,x,fname=None,root=0):
        x = np.split(x,self.sections)
        for ix,af in enumerate(self.af):
            fname_ = None if fname is None else fname+f'_{ix}' 
            af.update(x[ix],fname=fname_,root=root)
    def set_config(self,config,compute_v):
        self.config = config 
        self.cx = dict()
        config = self.parse_config(self.config)
        for af,config_ in zip(self.af,config):
            af.set_config(config_,compute_v)
        return config
    def amplitude2scalar(self):
        for af in self.af:
            af.amplitude2scalar()
class ProductAmplitudeFactory(CompoundAmplitudeFactory):
    def amplitude(self,config,sign=True,cache_bot=None,cache_top=None,to_numpy=True):
        if cache_bot is None:
            cache_bot = (None,) * self.naf
        if cache_top is None:
            cache_top = (None,) * self.naf
        cx = 1 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cx_ix = af.amplitude(config[ix],sign=sign,cache_bot=cache_bot[ix],cache_top=cache_top[ix],to_numpy=to_numpy)
                if cx_ix is None:
                    return None 
            else:
                cx_ix = af.amplitude(config[ix],to_numpy=to_numpy)
            cx = cx * cx_ix
        return cx 
    def get_grad_deterministic(self,config):
        cx = [None] * self.naf
        vx = [None] * self.naf 
        for ix,af in enumerate(self.af):
            cx[ix],vx[ix] = af.get_grad_deterministic(config[ix])
        return np.prod(np.array(cx)),np.concatenate(vx)
    def parse_gradient(self):
        vx = [None] * self.naf
        for ix,af in enumerate(self.af):
            if af.is_tn:
                vx[ix] = af.dict2vec(af.vx)
            else:
                vx[ix] = af.vx
            af.vx = None
            af.cx = None
        return np.concatenate(vx)
    def parse_energy(self,exs,batch_key,ratio):
        pairs = self.model.batched_pairs[batch_key].pairs
        e = 0.
        spins = ('a','b') if self.fermion else (None,)
        for where,spin in itertools.product(pairs,spins):
            term = 1.
            for af,ex in zip(self.af,exs):
                if (where,spin) in ex:
                    ix = -1 if ratio else 0
                    fac = ex[where,spin][ix] 
                else:
                    fac = 1 if ratio else ex[where,af.spin][1]
                term *= fac 
            e = e + term
        if ratio:
            e = tensor2backend(e,'numpy')
        return e
    def batch_pair_energies(self,batch_key,compute_Hv):
        ex = [None] * self.naf
        plq = [None] * self.naf
        for ix,af in enumerate(self.af):
            ex[ix],plq[ix] = af.batch_pair_energies(batch_key,compute_Hv)
        return ex,plq
    def get_grad_from_plq(self,plq):
        for ix,af in enumerate(self.af):
            af.get_grad_from_plq(plq[ix])
    def contraction_error(self):
        cx,err = np.zeros(self.naf),np.zeros(self.naf)
        for ix,af in enumerate(self.af): 
            cx[ix],err[ix] = contraction_error(af.cx)
        return np.prod(cx),np.amax(err)
    def _new_log_prob_from_plq(self,plq,sites,config_sites,config_new):
        py = [None] * self.naf 
        plq_new = [None] * self.naf
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq_new[ix],py[ix] = af._new_log_prob_from_plq(plq[ix],sites,config_sites[ix],None)
            else:
                py[ix] = af.log_prob(config_new[ix])
            if py[ix] is None:
                return plq_new,None 
        return plq_new,sum(py)
    def log_prob(self,config):
        p = [None] * self.naf 
        for ix,af in enumerate(self.af):
            p[ix] = af.log_prob(config[ix])
            if p[ix] is None:
                return None 
        return sum(p) 
    def replace_sites(self,tn,sites,cis):
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            tn[ix] = af.replace_sites(tn[ix],sites,cis[ix])
        return tn 
##### ham methods #####
    def config_sign(self,config):
        sign = [] 
        for af,config_ in zip(self.af,config):
            sign.append(af.config_sign(config_))
        return np.array(sign) 
class SumAmplitudeFactory(CompoundAmplitudeFactory):
    def check(self,configs,n=10):
        if RANK>0:
            return
        if configs.shape[0] == 1:
            rng = np.random.default_rng()
            config = configs[0,:]
            configs = np.array([config] + [rng.permuted(config) for _ in range(n)])
        for i in range(configs.shape[0]):
            config_i = tuple(configs[i,:])
            cx = self.amplitude(self.parse_config(config_i),_sum=False)
            print(config_i,cx)
    def amplitude(self,config,_sum=True,cache_bot=None,cache_top=None,to_numpy=True,**kwargs):
        if cache_bot is None:
            cache_bot = (None,) * self.naf
        if cache_top is None:
            cache_top = (None,) * self.naf

        cx = [0] * self.naf 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cx[ix] = af.amplitude(config[ix],cache_bot=cache_bot[ix],cache_top=cache_top[ix],to_numpy=to_numpy,**kwargs)
                if cx[ix] is None:
                    cx[ix] = 0
            else:
                cx[ix] = af.amplitude(config[ix],to_numpy=to_numpy)
        if _sum:
            cx = sum(cx)
        return cx 
    def unsigned_amplitude(self,config,**kwargs):
        return self.amplitude(config,**kwargs)
    def get_grad_deterministic(self,config):
        self.wfn2backend(backend='torch',requires_grad=True)
        cache_bot = [None] * self.naf 
        cache_top = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cache_bot[ix] = dict() 
                cache_top[ix] = dict() 
        cx = self.amplitude(config,cache_bot=cache_bot,cache_top=cache_top,to_numpy=False)
        cx,vx = self.propagate(cx)
        self.wfn2backend()
        return cx,vx/cx 
    def get_grad_from_plq(self,plq):
        pass
    def parse_gradient(self):
        for ix,af in enumerate(self.af):
            af.cx = None
            af.vx = None
    def batch_pair_energies(self,batch_key,compute_Hv):
        cache_bot = [None] * self.naf
        cache_top = [None] * self.naf
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            cache_bot[ix],cache_top[ix] = af.batch_benvs(batch_key,compute_Hv)
        ex = dict() 
        b = self.model.batched_pairs[batch_key]
        config = [af.config for af in self.af]
        cx = self.amplitude(config,cache_bot=cache_bot,cache_top=cache_top,direction=b.direction,to_numpy=False)
        self.cx['deterministic'] = cx
        for where in b.pairs:
            i = min([i for i,_ in where])
            ex_ij = self.update_pair_energy_from_benvs(where,cache_bot,cache_top,direction=b.direction,i=i) 
            for tag,eij in ex_ij.items():
                ex[where,tag] = eij,cx,eij/cx
        return ex,None
import autoray as ar
import torch
import h5py
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
class NN(AmplitudeFactory):
    def __init__(self,backend='numpy',log=True,phase=False,fermion=False,to_spin=True,order='F'):
        self.backend = backend
        self.set_backend(backend)
        self.log = log # if output is amplitude or log amplitude 
        self.phase = phase
        if phase:
            assert log

        self.is_tn = False
        self.from_plq = False
        self.spin = None
        self.vx = None

        self.fermion = fermion
        if fermion:
            self.to_spin = to_spin
            self.order = order
        else:
            self.to_spin = False
    def pair_terms(self,i1,i2,spin=None):
        if self.fermion:
            return pair_terms(i1,i2,spin) 
        else:
            return i2,i1
    def input(self,config):
        if self.fermion:
            if self.to_spin:
                ca,cb = config_to_ab(config) 
                config = np.stack([np.array(tsr,dtype=float) for tsr in (ca,cb)],axis=0).flatten(order=self.order)
            else:
                return np.array(config,dtype=float)
        else:
            config = np.array(config,dtype=float)
        return config * 2 - 1
    def set_backend(self,backend):
        if backend=='numpy':
            tsr = np.zeros(1)
            self.jnp = np 
            self._input = self.input
        else:
            tsr = torch.zeros(1)
            self.jnp = torch
            def _input(config):
                return tensor2backend(self.input(config),backend) 
            self._input = _input
        ar.set_backend(tsr)
    def log_prob(self,config):
        if self.phase:
            return 1
        c = self._input(config)
        c = self.forward(c)
        c = tensor2backend(c,'numpy') 
        if self.log:
            return 2 * c 
        else:
            return np.log(c**2) 
    def amplitude(self,config,to_numpy=True):
        c = self._input(config)
        c = self.forward(c)
        if self.log:
            logc = c
            if self.phase:
                loc = 1j*logc
            c = self.jnp.exp(logc)
        if to_numpy:
            c = tensor2backend(c,'numpy')
        return c 
    def batch_pair_energies(self,batch_key,new_cache):
        spins = ('a','b') if self.fermion else (None,)

        config = self._input(self.config)
        cx = self.forward(config)
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
                config_new = self._input(config_new)
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
        config = self._input(config)
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
class RBM(NN):
    def __init__(self,nv,nh,**kwargs):
        self.nv,self.nh = nv,nh
        self.nparam = nv + nh + nv * nh 
        self.block_dict = [(0,nv),(nv,nv+nh),(nv+nh,self.nparam)]
        super().__init__(**kwargs)
    def init(self,eps,a=-1,b=1,fname=None,shift=[0,0,0]):
        # va(config) = config
        # vb(config) = tanh(w*config+b)
        # vw(config) = tanh(w*config+b) * config

        # rule of thumb
        # small numbers initialized in range (a,b)
        c = b-a
        self.a = (np.random.rand(self.nv) * c + a) * eps + shift[0] 
        self.b = (np.random.rand(self.nh) * c + a) * eps + shift[1]
        self.w = (np.random.rand(self.nv,self.nh) * c + a) * eps + shift[2]
        COMM.Bcast(self.a,root=0)
        COMM.Bcast(self.b,root=0)
        COMM.Bcast(self.w,root=0)
        if fname is not None: 
            self.save_to_disc(self.a,self.b,self.w,fname) 
        return self.a,self.b,self.w
    def init_from(self,a,b,w,eps,fname=None):
        self.init(eps)
        self.a[:len(a)] = a
        self.b[:len(b)] = b 
        sh1,sh2 = w.shape
        self.w[:sh1,:sh2] = w
        if fname is not None: 
            self.save_to_disc(self.a,self.b,self.w,fname) 
        return self.a,self.b,self.w
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self.set_backend(backend)
        self.a,self.b,self.w = [tensor2backend(tsr,backend,requires_grad=requires_grad) \
                                for tsr in [self.a,self.b,self.w]]
    def get_x(self):
        ls = [tensor2backend(tsr,'numpy') for tsr in [self.a,self.b,self.w]]
        ls[-1] = ls[-1].flatten()
        return np.concatenate(ls)
    def load_from_disc(self,fname):
        f = h5py.File(fname,'r')
        self.a = f['a'][:]
        self.b = f['b'][:]
        self.w = f['w'][:]
        f.close()
        return self.a,self.b,self.w
    def save_to_disc(self,a,b,w,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        f.create_dataset('a',data=a) 
        f.create_dataset('b',data=b) 
        f.create_dataset('w',data=w) 
        f.close()
    def update(self,x,fname=None,root=0):
        x = np.split(x,[self.nv,self.nv+self.nh])
        self.a = x[0]
        self.b = x[1]
        self.w = x[2].reshape(self.nv,self.nh)
        if fname is not None:
            self.save_to_disc(self.a,self.b,self.w,fname,root=root)
        self.wfn2backend()
    def extract_grad(self):
        ls = [tensor2backend(self.tensor_grad(tsr),'numpy') for tsr in [self.a,self.b,self.w]] 
        ls[-1] = ls[-1].flatten()
        return np.concatenate(ls)
    def forward(self,c): # NN output
        c = self.jnp.dot(self.a,c) + self.jnp.sum(self.jnp.log(self.jnp.cosh(self.jnp.matmul(c,self.w) + self.b)))
        return c
class FNN(NN):
    def __init__(self,nv,nf=1,afn='logcosh',scale=1,coeff=1,change_basis=False,**kwargs):
        self.nv = nv
        self.nf = nf
        assert afn in ('logcosh','logistic','tanh','softplus','silu','cos')
        self.afn = afn 
        self.coeff = coeff
        self.scale = scale
        self.change_basis = change_basis
        super().__init__(**kwargs)
    def input(self,config):
        if not self.change_basis: 
            return super().input(config)
        if not self.fermion:
            return np.array(config,dtype=float)
        config = [tuple(np.where(np.array(c))[0]) for c in config_to_ab(config)]
        config = np.array([[self.flat2site(ix) for ix in c] for c in config],dtype=float)
        return config.flatten()
    def init(self,nn,eps,a=-1,b=1,fname=None,iprint=0): # nn is number of nodes in each hidden layer
        self.w = []
        self.b = [] 
        c = b-a
        for i,ni in enumerate(nn):
            sh1 = self.nv if i==0 else nn[i-1]
            wi = (np.random.rand(sh1,nn[i]) * c + a) * eps 
            COMM.Bcast(wi,root=0)
            self.w.append(wi)

            bi = (np.random.rand(nn[i]) * c + a) * eps 
            COMM.Bcast(bi,root=0)
            self.b.append(bi)
            if iprint>0 and RANK==0:
                print(i)
                print(wi)
                print(bi)
        self.w.append(np.ones((nn[-1],self.nf)))
        if fname is not None: 
            self.save_to_disc(self.w,self.b,fname)
        return self.w,self.b
    def init_from(self,nn,w,b,eps,fname=None):
        self.init(nn,eps)
        for i,wi in enumerate(w):
            sh1,sh2 = wi.shape  
            self.w[i][:sh1,:sh2] = wi
        for i,bi in enumerate(b):
            self.b[i][:len(bi)] = bi
        if fname is not None: 
            self.save_to_disc(self.w,self.b,fname) 
        return self.w,self.b
    def get_block_dict(self,w,b):
        self.w,self.b = w,b
        self.block_dict = []
        start = 0
        for i,wi in enumerate(self.w):
            tsrs = [wi] if i==len(self.b) else [wi,b[i]]
            for tsr in tsrs:
                stop = start + tsr.size
                self.block_dict.append((start,stop))
                start = stop
        self.nparam = stop
    def set_backend(self,backend):
        super().set_backend(backend)
        if self.afn=='logcosh':
            def _afn(x):
                return self.jnp.log(self.jnp.cosh(x))
        elif self.afn=='logistic':
            def _afn(x):
                return 1./(1.+self.jnp.exp(-x))    
        elif self.afn=='tanh':
            def _afn(x):
                return self.scale * self.jnp.tanh(self.coeff * x)
        elif self.afn=='softplus':
            def _afn(x):
                return self.jnp.log(1.+self.jnp.exp(x))
        elif self.afn=='silu':
            def _afn(x):
                return x/(1.+self.jnp.exp(-x))
        elif self.afn=='cos':
            def _afn(x):
                return self.jnp.cos(self.coeff * x)
        else:
            raise NotImplementedError
        self._afn = _afn 
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self.set_backend(backend)
        self.w = [tensor2backend(tsr,backend,requires_grad=requires_grad) for tsr in self.w]
        self.b = [tensor2backend(tsr,backend,requires_grad=requires_grad) for tsr in self.b]
    def get_x(self):
        ls = []
        for i,wi in enumerate(self.w):
            ls.append(tensor2backend(wi,'numpy').flatten())
            if i<len(self.b): 
                ls.append(tensor2backend(self.b[i],'numpy'))
        return np.concatenate(ls)
    def load_from_disc(self,fname,nl):
        f = h5py.File(fname,'r')
        self.w = []
        self.b = []
        for i in range(nl):
            try:
                self.w.append(f[f'w{i}'][:])
            except:
                pass
            try:
                self.b.append(f[f'b{i}'][:])
            except:
                pass
        f.close()
        return self.w,self.b
    def save_to_disc(self,w,b,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for i,wi in enumerate(w):
            f.create_dataset(f'w{i}',data=wi) 
            try:
                f.create_dataset(f'b{i}',data=b[i]) 
            except:
                pass
        f.close()
    def update(self,x,fname=None,root=0):
        for i in range(len(self.w)):
            start,stop = self.block_dict[2*i]
            size = stop-start
            xi,x = x[:size],x[size:]
            self.w[i] = xi.reshape(self.w[i].shape)

            if i<len(self.b):
                start,stop = self.block_dict[2*i+1]
                size = stop-start
                xi,x = x[:size],x[size:]
                self.b[i] = xi
        if fname is not None:
            self.save_to_disc(self.w,self.b,fname,root=root) 
        self.wfn2backend()
    def extract_grad(self):
        ls = []
        for i,wi in enumerate(self.w):
            ls.append(tensor2backend(self.tensor_grad(wi),'numpy').flatten())
            if i<len(self.b): 
                ls.append(tensor2backend(self.tensor_grad(self.b[i]),'numpy'))
        return np.concatenate(ls)
    def forward(self,c):
        for i in range(len(self.b)):
            c = self.jnp.matmul(c,self.w[i]) + self.b[i]    
            c = self._afn(c)
        return self.jnp.matmul(c,self.w[-1])
class SIGN(FNN):
    def __init__(self,nv,nl,afn='tanh',**kwargs):
        super().__init__(nv,nl,afn=afn,log_amp=False,**kwargs)
    def forward(self,c):
        c = super().forward(c) 
        return self._afn(c)
