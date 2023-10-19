import numpy as np
import itertools
#####################################################
# for separate ansatz
#####################################################
from ..tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    AmplitudeFactory,
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
class ProductAmplitudeFactory:
    def parse_config(self,config):
        ca,cb = config_to_ab(config)
        return [{'a':ca,'b':cb,None:config}[af.spin] for af in self.af]
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
    def get_grad_deterministic(self,config,unsigned=False):
        cx = [None] * self.naf
        vx = [None] * self.naf 
        for ix,af in enumerate(self.af):
            cx[ix],vx[ix] = af.get_grad_deterministic(config[ix],unsigned=unsigned)
        return np.array(cx),np.concatenate(vx)
    def _new_log_prob_from_plq(self,plq,sites,cis):
        py = [None] * self.naf 
        plq_new = [None] * self.naf
        config_new = self.parse_config(self.config_new)
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq_new[ix],py[ix] = af._new_log_prob_from_plq(plq[ix],sites,cis[ix])
            else:
                py[ix] = 2 * af.log_amplitude(config_new[ix])[0]
            if py[ix] is None:
                return plq_new,None 
        return plq_new,sum(py)
    def prob(self,config):
        p = 1
        for ix,af in enumerate(self.af):
            pix = af.prob(config[ix])
            if pix is None:
                return 0
            p *= pix
        return p 
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
    def propagate(self,ex_num):
        ex_num.backward()
        Hvx = [af.extract_grad() for af in self.af] 
        return np.concatenate(Hvx) 
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
    def batch_quantities_from_plq(self,batch_key,compute_v,compute_Hv): # only used for Hessian
        ex = [None] * self.naf
        cx = [None] * self.naf
        plq = [None] * self.naf
        for ix,af in enumerate(self.af):
            if compute_Hv:
                af.wfn2backend(backend='torch',requires_grad=True)
                af.model.gate2backend('torch')
            ex[ix],cx[ix],plq[ix] = af.batch_pair_energies_from_plq(batch_key,new_cache=compute_Hv)

        if compute_Hv:
            Hvx = self.propagate(self.parse_energy(ex,batch_key,cx=cx))
        else:
            Hvx = 0.

        ex = self.parse_energy(ex,batch_key)
        for ix,af in enumerate(self.af):
            if af.is_tn:
                af.cx.update({plq_key:tensor2backend(cij,'numpy') for where,(cij,plq_key) in cx[ix].items()})
            else:
                af.cx = {None:tensor2backend(cx[ix],'numpy')}
             
        if compute_v: 
            for ix,af in enumerate(self.af):
                af.get_grad_from_plq(plq[ix])
        if compute_Hv: 
            self.wfn2backend()
            for af in self.af:
                af.model.gate2backend(self.backend)
        return ex,Hvx
    def contraction_error(self,multiply=True):
        cx,err = np.zeros(self.naf),np.zeros(self.naf)
        for ix,af in enumerate(self.af): 
            cx[ix],err[ix] = contraction_error(af.cx)
        if multiply:
            cx = np.prod(cx)
        return cx,np.amax(err)
    def compute_local_quantities_from_plq(self,compute_v,compute_Hv): 
        for af in self.af:
            af.cx = dict()
            if af.is_tn:
                af.vx = dict()
        ex,Hvx = 0.,0.
        for batch_key in self.model.batched_pairs:
            ex_,Hvx_ = self.batch_quantities_from_plq(batch_key,compute_v,compute_Hv)  
            ex += ex_
            Hvx += Hvx_
        cx,err = self.contraction_error() 
        eu = self.model.compute_local_energy_eigen(self.config)
        ex += eu

        if compute_v:
            vx = [None] * self.naf
            for ix,af in enumerate(self.af):
                if af.is_tn:
                    vx[ix] = af.dict2vec(af.vx)
                else:
                    vx[ix] = af.vx
                af.vx = None
            vx = np.concatenate(vx)
        else:
            vx = None

        if compute_Hv:
            Hvx = Hvx / cx + eu * vx
        else:
            Hvx = None
        return cx,ex,vx,Hvx,err 
    def compute_local_energy(self,config,compute_v=True,compute_Hv=False):
        self.config = config 
        for af,config_ in zip(self.af,self.parse_config(config)):
            af.config = config_ 
        return self.compute_local_quantities_from_plq(compute_v,compute_Hv)
    #def batch_hessian_deterministic(self,config,batch_key):
    #    af = self.amplitude_factory
    #    af.wfn2backend(backend='torch',requires_grad=True)
    #    ex = self.batch_pair_energies_deterministic(config,batch_key,new_cache=True)
    #    _,ex,Hvx = af.compute_hessian(ex)
    #    af.wfn2backend()
    #    return ex,Hvx
    #def compute_local_energy_hessian_deterministic(self,config):
    #    af = self.amplitude_factory
    #    cx,vx = af.get_grad_deterministic(config)

    #    ex = 0. 
    #    Hvx = 0.
    #    for key in self.model.batched_pairs:
    #        ex_,Hvx_ = self.batch_hessian_deterministic(config,key) 
    #        ex += ex_
    #        Hvx += Hvx_
    #     
    #    eu = self.model.compute_local_energy_eigen(config)
    #    return af.parse_hessian_deterministic(Hvx,vx,ex,eu,cx)
    #def compute_local_energy_gradient_deterministic(self,config,compute_v=True):
    #    af = self.amplitude_factory
    #    ex = dict() 
    #    for key in self.model.batched_pairs:
    #        ex_ = self.batch_pair_energies_deterministic(config,key)
    #        ex.update(ex_)

    #    if compute_v:
    #        cx,vx = af.get_grad_deterministic(config)
    #    else:
    #        cx = af.unsigned_amplitude(config)
    #        sign = af.config_sign(config)
    #        cx *= sign
    #        vx = None
    #    if cx is None:
    #        return 0.,0.,vx,None,0.
    #    ex,cx = af.parse_energy_deterministic(ex,cx) 
    #    eu = self.model.compute_local_energy_eigen(config)
    #    ex += eu
    #    return cx,ex,vx,None,0.
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
class NN(AmplitudeFactory):
    def __init__(self,to_spin=True,order='F',backend='numpy',log_amp=True):
        self.backend = backend
        self.order = order
        self.log_amp = log_amp # if output is amplitude or log amplitude 
        self.to_spin = to_spin

        self.is_tn = False
        self.spin = None
        self.vx = None
    def log_amplitude(self,config,to_numpy=True):
        c = to_spin(config,self.order) if self.to_spin else np.array(config,dtype=float)
        jnp,c = self.get_backend(c=c) 
        c,s = self.forward(c,jnp),1
        if not self.log_amp: # NN outputs amplitude
            c,s = jnp.log(jnp.abs(c)),jnp.sign(c)
        if to_numpy:
            c = tensor2backend(c,'numpy') 
            s = tensor2backend(s,'numpy') 
        return c,s
    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
        c = to_spin(config,self.order) if self.to_spin else np.array(config,dtype=float)
        jnp,c = self.get_backend(c=c) 
        c = self.forward(c,jnp)
        if self.log_amp: # NN outputs log amplitude
            c = jnp.exp(c)
        if to_numpy:
            c = tensor2backend(c,'numpy') 
        return c
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
    def get_grad_from_plq(self,plq=None):
        if self.vx is not None:
            return self.vx
        self.wfn2backend(backend='torch',requires_grad=True)
        c,_ = self.log_amplitude(self.config,to_numpy=False) 
        _,self.vx = self.propagate(c) 
        #print(self.config,self.vx)
        #exit()
        self.wfn2backend()
        return self.vx 
class RBM(NN):
    def __init__(self,nv,nh,**kwargs):
        self.nv,self.nh = nv,nh
        self.nparam = nv + nh + nv * nh 
        self.block_dict = [(0,nv),(nv,nv+nh),(nv+nh,self.nparam)]
        super().__init__(**kwargs)
    def init(self,eps,a=-1,b=1,fname=None):
        # small numbers initialized in range (a,b)
        c = b-a
        self.a = (np.random.rand(self.nv) * c + a) * eps 
        self.b = (np.random.rand(self.nh) * c + a) * eps 
        self.w = (np.random.rand(self.nv,self.nh) * c + a) * eps
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
        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
        ar.set_backend(tsr)
        self.a,self.b,self.w = [tensor2backend(tsr,backend=backend,requires_grad=requires_grad) \
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
    def get_backend(self,c=None):
        if isinstance(self.a,torch.Tensor):
            if c is not None:
                c = tensor2backend(c,backend='torch')
            return torch,c
        else:
            return np,c
    def forward(self,c,jnp): # NN output
        c = jnp.dot(self.a,c) + jnp.sum(jnp.log(jnp.cosh(jnp.matmul(c,self.w) + self.b)))
        return c
class FNN(NN):
    def __init__(self,nv,nl,afn='logcosh',**kwargs):
        self.nv = nv
        self.nl = nl # number of hidden layer
        assert afn in ('logcosh','logistic','tanh','softplus','silu','cos')
        self.afn = afn 
        super().__init__(**kwargs)
    def init(self,nn,eps,a=-1,b=1,fname=None): # nn is number of nodes in each hidden layer
        if isinstance(nn,int):
            nn = (nn,) * self.nl 
        else:
            assert len(nn)==self.nl

        self.w = []
        self.b = [] 
        c = b-a
        for i in range(self.nl):
            sh1 = self.nv if i==0 else nn[i-1]
            wi = (np.random.rand(sh1,nn[i]) * c + a) * eps 
            COMM.Bcast(wi,root=0)
            self.w.append(wi)

            bi = (np.random.rand(nn[i]) * c + a) * eps 
            COMM.Bcast(bi,root=0)
            self.b.append(bi)
        self.w.append(np.ones(nn[-1]))
        if fname is not None: 
            self.save_to_disc(self.w,self.b,fname)
        return self.w,self.b
    def init_from(self,nn,w,b,eps,fname=None):
        self.init(nn,eps)
        for i,wi in enumerate(w):
            if i==len(w)-1:
                self.w[i][:len(wi),0] = wi
            else:
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
        for i in range(self.nl+1):
            tsrs = [w[i]] if i==self.nl else [w[i],b[i]]
            for tsr in tsrs:
                stop = start + tsr.size
                self.block_dict.append((start,stop))
                start = stop
        self.nparam = stop
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
        ar.set_backend(tsr)
        self.w = [tensor2backend(tsr,backend=backend,requires_grad=requires_grad) for tsr in self.w]
        self.b = [tensor2backend(tsr,backend=backend,requires_grad=requires_grad) for tsr in self.b]
    def get_x(self):
        ls = []
        for i in range(self.nl+1):
            ls.append(tensor2backend(self.w[i],'numpy').flatten())
            if i<self.nl: 
                ls.append(tensor2backend(self.b[i],'numpy'))
        return np.concatenate(ls)
    def load_from_disc(self,fname):
        f = h5py.File(fname,'r')
        self.w = []
        self.b = []
        for i in range(self.nl+1):
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
        for i in range(self.nl+1):
            start,stop = self.block_dict[2*i]
            size = stop-start
            xi,x = x[:size],x[size:]
            self.w[i] = xi.reshape(self.w[i].shape)

            if i<self.nl:
                start,stop = self.block_dict[2*i+1]
                size = stop-start
                xi,x = x[:size],x[size:]
                self.b[i] = xi
        if fname is not None:
            self.save_to_disc(self.w,self.b,fname,root=root) 
        self.wfn2backend()
    def extract_grad(self):
        ls = []
        for i in range(self.nl+1):
            ls.append(tensor2backend(self.tensor_grad(self.w[i]),'numpy').flatten())
            if i<self.nl: 
                ls.append(tensor2backend(self.tensor_grad(self.b[i]),'numpy'))
        return np.concatenate(ls)
    def get_backend(self,c=None):
        if isinstance(self.w[0],torch.Tensor):
            if c is not None:
                c = tensor2backend(c,backend='torch')
            jnp = torch
        else:
            jnp = np
        if self.afn=='logcosh':
            def _afn(x):
                return jnp.log(jnp.cosh(x))
        elif self.afn=='logistic':
            def _afn(x):
                return 1./(1.+jnp.exp(-x))    
        elif self.afn=='tanh':
            def _afn(x):
                return jnp.tanh(self.coeff * x)
        elif self.afn=='softplus':
            def _afn(x):
                return jnp.log(1.+jnp.exp(x))
        elif self.afn=='silu':
            def _afn(x):
                return x/(1.+jnp.exp(-x))
        elif self.afn=='cos':
            def _afn(x):
                return jnp.cos(self.coeff * x)
        else:
            raise NotImplementedError
        self._afn = _afn 
        return jnp,c
    def forward(self,c,jnp):
        for i in range(self.nl):
            c = jnp.matmul(c,self.w[i]) + self.b[i]    
            c = self._afn(c)
        return jnp.dot(c,self.w[-1])
class SIGN(FNN):
    def __init__(self,nv,nl,afn='tanh',**kwargs):
        super().__init__(nv,nl,afn=afn,log_amp=False,**kwargs)
    def forward(self,c,jnp):
        c = super().forward(c,jnp) 
        return self._afn(c)
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