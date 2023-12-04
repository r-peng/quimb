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
class ProductAmplitudeFactory:
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
class SumAmplitudeFactory:
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
    def get_grad_from_plq(self,*args):
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
    def _new_log_prob_from_plq(self,plq,sites,config_sites,config_new):
        cx = self.amplitude(config_new)
        return None,np.log(cx**2)
    def replace_sites(self,*args):
        pass
import autoray as ar
import torch
ar.register_function('torch','relu',torch.nn.functional.relu)
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
    def __init__(self,backend='numpy',log=True,phase=False,cinput=False,fermion=False,to_spin=True,order='F'):
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
        self.cinput = cinput
        self.params = dict()
        self.param_keys = []
        self.sh = dict()
        if cinput:
            key = 'in'
            self.param_keys.append(key)
            sh = 4 if fermion else 2
            self.sh[key] = self.nv,sh 
        else:
            if fermion:
                self.to_spin = to_spin
                self.order = order
            #else:
            #    self.to_spin = False
    def init(self,key,eps,a=-1,b=1,c=0,iprint=0):
        tsr = (np.random.rand(*self.sh[key])*(b-a)+a) * eps + c
        COMM.Bcast(tsr,root=0)
        self.params[key] = tsr
        if RANK==0 and iprint>0:
            print(key)
            print(tsr)
    def get_block_dict(self):
        self.block_dict = []
        start = 0
        for key in self.param_keys:
            stop = start + self.params[key].size
            self.block_dict.append((start,stop))
            start = stop
        self.nparam = stop
    def set_backend(self,backend):
        if backend=='numpy':
            tsr = np.zeros(1)
            self.jnp = np 
            self._input = self.input
        else:
            tsr = torch.zeros(1)
            self.jnp = torch
            if self.cinput:
                self._input = self.input
            else:
                def _input(config):
                    return tensor2backend(self.input(config),backend) 
                self._input = _input
        ar.set_backend(tsr)
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self.set_backend(backend)
        for key in self.param_keys:
            self.params[key] = tensor2backend(self.params[key],backend,requires_grad=requires_grad)
    def get_x(self,grad=False):
        ls = []
        for key in self.param_keys:
            tsr = self.params[key]
            if grad:
                tsr = self.tensor_grad(tsr)
            ls.append(tensor2backend(tsr,'numpy').flatten())
        return np.concatenate(ls)
    def extract_grad(self):
        return self.get_x(grad=True) 
    def load_from_disc(self,fname):
        self.sh = dict()
        f = h5py.File(fname,'r')
        for key in self.param_keys:
            tsr = f[str(key)][:]
            self.params[key] = tsr 
            self.sh[key] = tsr.shape
        f.close()
        return self.params
    def save_to_disc(self,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for key in self.param_keys:
            f.create_dataset(str(key),data=self.params[key]) 
        f.close()
    def update(self,x,fname=None,root=0):
        for i,key in enumerate(self.param_keys):
            start,stop = self.block_dict[i]
            size = stop - start
            xi,x = x[:size],x[size:]
            self.params[key] = xi.reshape(self.sh[key])
        if fname is not None:
            self.save_to_disc(fname,root=root) 
        self.wfn2backend()
    def pair_terms(self,i1,i2,spin=None):
        if self.fermion:
            return pair_terms(i1,i2,spin) 
        else:
            return i2,i1
    def input_continous(self,config):
        x = self.params['in']
        c = self.jnp.zeros_like(x)
        for i,ci in enumerate(config):
            c[i,ci] = 1
        return self.jnp.sum(x*c,axis=1)
    def input(self,config):
        if self.cinput:
            return self.input_continous(config)
        if self.fermion:
            if self.to_spin:
                ca,cb = config_to_ab(config) 
                config = np.stack([np.array(tsr,dtype=float) for tsr in (ca,cb)],axis=0).flatten(order=self.order)
            else:
                return np.array(config,dtype=float)
        else:
            config = np.array(config,dtype=float)
        return config * 2 - 1
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
        super().__init__(**kwargs)
        self.param_keys += ['a','b','w']
        self.sh.update({'a':(self.nv,),
                        'b':(self.nh,),
                        'w':(self.nv,self.nh)})
    def forward(self,c): # NN output
        a = self.params['a']
        b = self.params['b']
        w = self.params['w']
        c = self.jnp.dot(a,c) + self.jnp.sum(self.jnp.log(self.jnp.cosh(self.jnp.matmul(c,w) + b)))
        return c
class FNN(NN):
    def __init__(self,nv,nh,afn,nf=1,nbasis=None,bias=False,wf=True,scale=None,**kwargs):
        self.nv,self.nh,self.nf = nv,nh,nf
        self.afn = afn 
        self.scale = scale
        super().__init__(**kwargs)

        self.nbasis = nbasis
        for i in range(len(nh)):
            key = i,'w'
            self.param_keys.append(key)
            sh1 = self.nv if i==0 else nh[i-1]
            self.sh[key] = sh1,nh[i]
            if bias:
                key = i,'b'
                self.param_keys.append(key)
                self.sh[key] = nh[i],
            if nbasis is not None:
                key = i,'a'
                self.param_keys.append(key)
                self.sh[key] = nbasis[i],
        if wf:
            key = 'wf'
            self.param_keys.append(key)
            self.sh[key] = nh[-1],nf
        if len(afn)==len(nh):
            pass
        elif len(afn)==len(nh)+1:
            assert (not self.log)
        else:
            raise ValueError(f'number of hidden nodes={len(nh)},number of activation fxn={len(afn)}')
    def get_layer_afn(self,i):
        afn = self.afn[i]
        # unsaturating 
        if afn=='id':
            def _afn(x):
                return x
        elif afn=='logcosh':
            def _afn(x):
                return self.jnp.log(self.jnp.cosh(x))
        elif afn=='softplus':
            def _afn(x):
                return self.jnp.log(1.+self.jnp.exp(x))
        elif afn=='silu':
            def _afn(x):
                return x/(1.+self.jnp.exp(-x))
        elif afn=='relu':
            def _afn(x):
                try:
                    return self.jnp.relu(x)
                except AttributeError:
                    return x*(x>0)
        elif afn=='exp':
            def _afn(x):
                return self.jnp.exp(x)
        elif afn=='sinh':
            def _afn(x):
                return self.jnp.sinh(x)
        # saturating
        elif afn=='logistic':
            def _afn(x):
                return self.scale[i] / (1.+self.jnp.exp(-x))    
        elif afn=='tanh':
            def _afn(x):
                return self.scale[i] * self.jnp.tanh(x)
        elif afn=='cos':
            def _afn(x):
                return self.scale[i] * self.jnp.cos(x)
        elif afn=='sin':
            def _afn(x):
                return self.scale[i] * self.jnp.sin(x)
        # linear combination
        elif afn=='pow':
            def _afn(x):
                return [x**p for p in range(1,self.nbasis[i]+1)]
        elif afn=='fsin':
            def _afn(x):
                return [self.jnp.sin(x*p) for p in range(1,self.nbasis[i]+1)]
        else:
            raise NotImplementedError
        return _afn
    def set_backend(self,backend):
        super().set_backend(backend)
        self._afn = [self.get_layer_afn(i) for i in range(len(self.afn))]
    def forward(self,c):
        for i in range(len(self.nh)):
            c = self.jnp.matmul(c,self.params[i,'w'])    
            if (i,'b') in self.params:
                c = c + self.params[i,'b']
            c = self._afn[i](c)
            if (i,'a') in self.params:
                c = sum([ci*ai for ci,ai in zip(c,self.params[i,'a'])]) 
        if len(self.afn)==len(self.nh)+1:
            c = self._afn[-1](c)
        if 'wf' in self.params:
            return self.jnp.matmul(c,self.params['wf'])
        return self.jnp.sum(c)
    
