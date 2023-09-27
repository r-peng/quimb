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
class NN(AmplitudeFactory):
    def unsigned_amplitude(self,config,cx=None,cache_top=None,cache_bot=None,to_numpy=True):
        if cx is None:
            cx,_ = self.log_amplitude(config,to_numpy=to_numpy) 
        jnp,_ = self.get_backend()
        return jnp.exp(cx)
    def batch_pair_energies_from_plq(self,batch_key,new_cache=None):
        bix,tix,plq_types,pairs,direction = self.model.batched_pairs[batch_key]
        jnp = torch if new_cache else np
        logcx,_ = self.log_amplitude(self.config,to_numpy=False)
        cx = jnp.exp(logcx) 
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
                        logcx_new,_ = self.log_amplitude(config_new,to_numpy=False) 
                        cx_new = jnp.exp(logcx_new)
                        ex[where,spin] = cx_new,jnp.exp(logcx_new-logcx) 
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
    def __init__(self,a=None,b=None,w=None,nv=None,nh=None,backend='numpy'):
        self.is_tn = False
        self.backend = backend
        self.spin = None
        self.vx = None
        if a is None:
            self.nv,self.nh = nv,nh
        else:
            self.a = a 
            self.b = b
            self.w = w
            self.nv = len(a)
            self.nh = len(b)
        self.nparam = self.nv + self.nh + self.nh * self.nv 
        self.block_dict = [(0,self.nv),(self.nv,self.nv+self.nh),(self.nv+self.nh,self.nparam)]
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
    def log_amplitude(self,config,to_numpy=True):
        ca,cb = config_to_ab(config) 
        #c = np.array(ca+cb,dtype=float)
        c = np.stack([np.array(tsr,dtype=float) for tsr in (ca,cb)],axis=1).flatten()
        jnp,c = self.get_backend(c=c) 
        c = jnp.dot(self.a,c) + jnp.sum(jnp.log(jnp.cosh(jnp.matmul(c,self.w) + self.b)))
        if to_numpy:
            c = tensor2backend(c,'numpy') 
        return c,0 
class FNN(NN):
    def __init__(self,w=None,b=None,nl=None,backend='numpy'):
        self.is_tn = False
        self.backend = backend
        self.spin = None
        self.vx = None
        if w is None:
            self.nl = nl
            return
        self.w = w
        self.b = b
        self.nl = len(w)
        self.init_block_dict() 
    def init_block_dict(self,w=None,b=None):
        if w is None:
            w,b = self.w,self.b
        else:
            self.w,self.b = w,b
        self.block_dict = []
        start = 0
        for i in range(self.nl):
            tsrs = [w[i]] if i==self.nl-1 else [w[i],b[i]]
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
        for i in range(self.nl):
            ls.append(tensor2backend(self.w[i],'numpy').flatten())
            if i<self.nl-1: 
                ls.append(tensor2backend(self.b[i],'numpy'))
        return np.concatenate(ls)
    def load_from_disc(self,fname):
        f = h5py.File(fname,'r')
        self.w = []
        self.b = []
        for i in range(self.nl):
            self.w.append(f[f'w{i}'][:])
            if i<self.nl-1:
                self.b.append(f[f'b{i}'][:])
        f.close()
        return self.w,self.b
    def save_to_disc(self,w,b,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for i in range(self.nl):
            f.create_dataset(f'w{i}',data=w[i]) 
            if i<self.nl-1:
                f.create_dataset(f'b{i}',data=b[i]) 
        f.close()
    def update(self,x,fname=None,root=0):
        for i in range(self.nl):
            start,stop = self.block_dict[2*i]
            size = stop-start
            xi,x = x[:size],x[size:]
            self.w[i] = xi.reshape(self.w[i].shape)

            if i<self.nl-1:
                start,stop = self.block_dict[2*i+1]
                size = stop-start
                xi,x = x[:size],x[size:]
                self.b[i] = xi
        if fname is not None:
            self.save_to_disc(self.w,self.b,fname,root=root) 
        self.wfn2backend()
    def extract_grad(self):
        ls = []
        for i in range(self.nl):
            ls.append(tensor2backend(self.tensor_grad(self.w[i]),'numpy').flatten())
            if i<self.nl-1: 
                ls.append(tensor2backend(self.tensor_grad(self.b[i]),'numpy'))
        return np.concatenate(ls)
    def get_backend(self,c=None):
        if isinstance(self.w[0],torch.Tensor):
            if c is not None:
                c = tensor2backend(c,backend='torch')
            return torch,c
        else:
            return np,c
    def log_amplitude(self,config,to_numpy=True):
        c = np.array(config,dtype=float)
        jnp,c = self.get_backend(c=c) 
        for i in range(self.nl-1):
            c = jnp.matmul(c,self.w[i])     
            c = c + self.b[i]
            #c = jnp.tanh(c)
            c = jnp.log(jnp.cosh(c))
        c = jnp.dot(c,self.w[-1])
        #exit()
        if to_numpy:
            c = tensor2backend(c,'numpy') 
        return c,0
