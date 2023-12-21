import numpy as np
import itertools,pickle,scipy
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
    def __init__(self,af,fermion=False,update=None):
        self.af = af 
        self._update = tuple(range(len(af))) if update is None else update
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
        vx = [self.af[ix].extract_grad() for ix in self._update] 
        return np.concatenate(vx) 
    def parse_config(self,config):
        if self.fermion:
            ca,cb = config_to_ab(config)
            return [{'a':ca,'b':cb,None:config}[af.spin] for af in self.af]
        else:
            return [config] * len(self.af)
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        for ix in self._update:
            self.af[ix].wfn2backend(backend=backend,requires_grad=requires_grad)
    def get_x(self):
        return np.concatenate([self.af[ix].get_x() for ix in self._update])
    def get_sections(self):
        self.nparam = np.array([self.af[ix].nparam for ix in self._update])
        self.sections = np.cumsum(self.nparam)[:-1]

        self.block_dict = []
        for i,ix in enumerate(self._update):
            shift = 0 if i==0 else self.sections[i-1]
            self.block_dict += [(start+shift,stop+shift) for start,stop in self.af[ix].block_dict]
        self.nparam = sum(self.nparam)
    def update(self,x,fname=None,root=0):
        x = np.split(x,self.sections)
        for i,ix in enumerate(self._update):
            fname_ = None if fname is None else fname+f'_{ix}' 
            self.af[ix].update(x[i],fname=fname_,root=root)
        self.get_sections()
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
            cache_bot = (None,) * len(self.af)
        if cache_top is None:
            cache_top = (None,) * len(self.af)
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
        cx = [None] * len(self.af)
        vx = [None] * len(self._update)
        for i,ix in enumerate(self._update):
            cx[ix],vx[i] = af.get_grad_deterministic(config[ix])
        for ix,af in enumerate(self.af):
            if cx[ix] is not None:
                continue
            cx[ix] = af.amplitude(config[ix])
        return np.prod(np.array(cx)),np.concatenate(vx)
    def parse_gradient(self):
        vx = [None] * len(self._update) 
        for i,ix in enumerate(self._update):
            af = self.af[ix]
            vx[i] = af.dict2vec(af.vx) if af.is_tn else af.vx
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
        ex = [None] * len(self.af)
        plq = [None] * len(self.af)
        for ix,af in enumerate(self.af):
            ex[ix],plq[ix] = af.batch_pair_energies(batch_key,compute_Hv)
        return ex,plq
    def get_grad_from_plq(self,plq):
        for ix,af in enumerate(self.af):
            af.get_grad_from_plq(plq[ix])
    def contraction_error(self):
        cx,err = np.zeros(len(self.af)),np.zeros(len(self.af))
        for ix,af in enumerate(self.af): 
            cx[ix],err[ix] = contraction_error(af.cx)
        return np.prod(cx),np.amax(err)
    def _new_log_prob_from_plq(self,plq,sites,config_sites,config_new):
        py = [None] * len(self.af) 
        plq_new = [None] * len(self.af)
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq_new[ix],py[ix] = af._new_log_prob_from_plq(plq[ix],sites,config_sites[ix],None)
            else:
                py[ix] = af.log_prob(config_new[ix])
            if py[ix] is None:
                return plq_new,None 
        return plq_new,sum(py)
    def log_prob(self,config):
        p = [None] * len(self.af) 
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
            cache_bot = (None,) * len(self.af)
        if cache_top is None:
            cache_top = (None,) * len(self.af)

        cx = [0] * len(self.af) 
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
        cache_bot = [None] * len(self.af) 
        cache_top = [None] * len(self.af) 
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
        cache_bot = [None] * len(self.af)
        cache_top = [None] * len(self.af)
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
#ar.register_function('torch','lrelu',torch.nn.functional.leaky_relu)
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
    def __init__(self,backend='numpy',log=True,phase=False,input_format=(0,1),order='F',fermion=False):
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
        assert input_format in ('det','fermion',(0,1),(-1,1))
        self.input_format = input_format
        self.order = order
        self.params = dict()
        self.param_keys = []
        self.sh = dict()
        self.change_layer_every = None 
        self.act_pattern = None
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
    def get_block_dict(self):
        self.block_dict = []
        start = 0
        keys = self.param_keys if self.change_layer_every is None else self.layer_keys[self.lcurr]
        for key in keys:
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
            def _input(config):
                return tensor2backend(self.input(config),backend) 
            self._input = _input
        ar.set_backend(tsr)
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self.set_backend(backend)
        for key in self.param_keys:
            if self.change_layer_every is None:
                grad = requires_grad
            else:
                grad = requires_grad if key in self.layer_keys[self.lcurr] else False
            self.params[key] = tensor2backend(self.params[key],backend,requires_grad=grad)
    def get_x(self,grad=False):
        ls = []
        keys = self.param_keys if self.change_layer_every is None else self.layer_keys[self.lcurr]
        for key in keys:
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
            try:
                tsr = f[str(key)][:]
                self.params[key] = tsr 
                self.sh[key] = tsr.shape
                if self.change_layer_every is not None:
                    self.layer_keys[i].append(key)
            except:
                continue
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
        if self.change_layer_every is None: 
            keys = self.param_keys 
        else:
            keys = self.layer_keys[self.lcurr]
            if RANK==0:
                print('update layer=',self.lcurr) 
        for i,key in enumerate(keys):
            start,stop = self.block_dict[i]
            size = stop - start
            xi,x = x[:size],x[size:]
            self.params[key] = xi.reshape(self.sh[key])
        if fname is not None:
            self.save_to_disc(fname,root=root) 
        if self.change_layer_every is not None:
            self.nstep = (self.nstep + 1) % self.change_layer_every
            if self.nstep==0:
                self.lcurr = (self.lcurr + 1) % len(self.nh)
            self.get_block_dict()
        self.wfn2backend()
        if self.act_pattern is None:
            return 
        for i,amap in enumerate(self.act_pattern):
            with open(fname+f'l{i}RANK{RANK}.pkl','wb') as f:
                pickle.dump(amap,f) 
        self.act_pattern = [dict() for _ in self.afn] 
    def pair_terms(self,i1,i2,spin=None):
        if self.fermion:
            return pair_terms(i1,i2,spin) 
        else:
            return i2,i1
    def input_determinant(self,config):
        ls = [None] * 2
        for ix,c in enumerate(config_to_ab(config)):
            ls[ix] = np.where(c)[0]
        c = np.concatenate(ls) 
        return c/len(config)
    def input(self,config):
        if self.input_format=='det':
            return self.input_determinant(config)
        if self.input_format=='fermion':
            return np.array(config,dtype=float) 
        if self.fermion:
            ca,cb = config_to_ab(config) 
            config = np.stack([np.array(tsr,dtype=float) for tsr in (ca,cb)],axis=0).flatten(order=self.order)
        xmin,xmax = self.input_format
        return config * (xmax-xmin) + xmin 
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
class RBM(NN):
    def __init__(self,nv,nh,**kwargs):
        super().__init__(**kwargs)
        self.nv,self.nh = nv,nh
        self.param_keys += ['a','b','w']
        self.sh.update({'a':(self.nv,),
                        'b':(self.nh,),
                        'w':(self.nv,self.nh)})
        self.change_layer_every = None 
    def forward(self,config): # NN output
        y = self._input(config)
        a = self.params['a']
        b = self.params['b']
        w = self.params['w']
        y = self.jnp.dot(a,y) + self.jnp.sum(self.jnp.log(self.jnp.cosh(self.jnp.matmul(y,w) + b)))
        return y
class FNN(NN):
    def __init__(self,nv,nh,afn,nf=1,bias=False,wf=True,scale=None,change_layer_every=None,act_pattern=False,**kwargs):
        super().__init__(**kwargs)
        self.nv,self.nh,self.nf = nv,nh,nf
        self.afn = afn 
        self.act_pattern = [dict() for _ in afn] if act_pattern else None 
        self.scale = scale
        self.bias = bias

        self.change_layer_every = change_layer_every
        if change_layer_every is not None:
            self.lcurr = 0
            self.nstep = 0
            self.layer_keys = [[] for i in range(len(nh))]
        for i in range(len(nh)):
            self.set_layer(i)
        if wf:
            key = 'wf'
            self.param_keys.append(key)
            self.sh[key] = nh[-1],nf
            if self.change_layer_every is not None: # group with final layer
                self.layer_keys[len(nh)-1].append(key)
        if len(afn)==len(nh):
            pass
        elif len(afn)==len(nh)+1:
            assert (not self.log)
        else:
            raise ValueError(f'number of hidden nodes={len(nh)},number of activation fxn={len(afn)}')
    def set_layer(self,i):
        key = i,'w'
        self.param_keys.append(key)
        sh1 = self.nv if i==0 else self.nh[i-1]
        self.sh[key] = sh1,self.nh[i]
        if self.change_layer_every is not None:
            self.layer_keys[i].append(key)
        if not self.bias:
            return
        key = i,'b'
        self.param_keys.append(key)
        self.sh[key] = self.nh[i],
        if self.change_layer_every is not None:
            self.layer_keys[i].append(key)
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
        return _afn
    def set_backend(self,backend):
        super().set_backend(backend)
        self._afn = [self.get_layer_afn(i) for i in range(len(self.afn))]
    def layer_forward(self,c,i):
        c = self.jnp.matmul(c,self.params[i,'w'])    
        if (i,'b') in self.params:
            c = c + self.params[i,'b']
        return self._afn[i](c)
    def add_act_pattern(self,y,i,config):
        if self.act_pattern is None:
            return   
        y = tensor2backend(y,'numpy')
        key = tuple(np.nonzero(y > 1e-10)[0])
        if key not in self.act_pattern[i]:
            self.act_pattern[i][key] = set() 
        self.act_pattern[i][key].add(tuple(config))
        return y
    def forward(self,config):
        y = self._input(config)
        for i in range(len(self.nh)):
            y = self.layer_forward(y,i)
            self.add_act_pattern(y,i,config)
        if len(self.afn)==len(self.nh)+1:
            y = self._afn[-1](y)
        if 'wf' in self.params:
            return self.jnp.matmul(y,self.params['wf'])
        return self.jnp.sum(y)
class LCFNN(FNN):
    def __init__(self,nv,nh,afn,nbasis,**kwargs):
        self.nbasis = nbasis
        super().__init__(nv,nh,afn,**kwargs)
    def set_layer(self,i):
        super().set_layer(i)
        key = i,'a'
        self.param_keys.append(key)
        self.sh[key] = self.nbasis[i],
    def get_layer_afn(self,i):
        afn = self.afn[i]
        if afn=='pow':
            def _afn(x):
                return [x**p for p in range(1,self.nbasis[i]+1)]
        elif afn=='fsin':
            def _afn(x):
                return [self.jnp.sin(x*p) for p in range(1,self.nbasis[i]+1)]
        else:
            raise NotImplementedError
        return _afn
    def forward(self,c):
        for i in range(len(self.nh)):
            c = self.jnp.matmul(c,self.params[i,'w'])    
            if (i,'b') in self.params:
                c = c + self.params[i,'b']
            c = self._afn[i](c)
            c = sum([ci*ai for ci,ai in zip(c,self.params[i,'a'])]) 
        if len(self.afn)==len(self.nh)+1:
            c = self._afn[-1](c)
        if 'wf' in self.params:
            return self.jnp.matmul(c,self.params['wf'])
        return self.jnp.sum(c)
class LRelu(FNN):
    def __init__(self,nv,nh,afn=None,**kwargs):
        _afn = ('id',)*len(nh)
        if afn is not None:
            _afn = _afn + (afn,) 
        super().__init__(nv,nh,_afn,**kwargs)
    def set_layer(self,i):
        super().set_layer(i)
        key = i,'a'
        self.param_keys.append(key)
        self.sh[key] = self.nh[i],
    def load_from_relu(self,fname,eps=1e-2,a=0,b=1):
        self.sh = dict()
        f = h5py.File(fname,'r')
        for key in self.param_keys:
            try:
                tsr = f[str(key)][:]
                self.params[key] = tsr 
                self.sh[key] = tsr.shape
            except:
                continue
        f.close()
        for i in range(len(self.nh)):
            key = (i,'a')
            self.sh[key] = self.nh[i],
            self.init(key,eps,a=a,b=b)
        return self.params
    def forward(self,c):
        for i in range(len(self.nh)):
            c = self.jnp.matmul(c,self.params[i,'w'])    
            if (i,'b') in self.params:
                c = c + self.params[i,'b']
            try:
                cp = self.jnp.relu(c)
                cm = self.jnp.relu(-c)
                c = cp - self.params[i,'a'] * cm
            except AttributeError:
                cp = c*(c>0)
                cm = c*(c<0)
                c = cp + self.params[i,'a'] * cm
        if len(self.afn)==len(self.nh)+1:
            c = self._afn[-1](c)
        if 'wf' in self.params:
            return self.jnp.matmul(c,self.params['wf'])
        return self.jnp.sum(c)
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
def relu_init_normal(nx,ny,xmin,xmax,eps):
    w = np.array([np.random.normal(loc=0,scale=eps,size=nx) for _ in range(ny)])
    w = compute_colinear(w)
    loc = (xmin+xmax) / 2 
    x = np.array([np.random.normal(loc=loc,scale=eps,size=nx) for _ in range(ny)])
    b = np.sum(w*x,axis=1) 
    return w,b
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
class RNN(NN):
    def __init__(self,nsite,D,nf=1,**kwargs):
        super().__init__(**kwargs)
        self.nsite = nsite
        self.D = D
        self.nf = nf
        self.pdim = 4 if self.fermion else 2
        for i in range(1,nsite):
            key = f'v{i}'
            self.param_keys.append(key)
            self.sh[key] = self.pdim,self.pdim,D
            if i==1:
                continue
            key = f'T{i}' 
            self.param_keys.append(key)
            self.sh[key] = i-1,self.pdim,D,D,D
    #def input(self,config):
    #    array = np.zeros((self.nsite,self.pdim)) 
    #    for i,ci in enumerate(config):
    #        array[i,ci] = 1
    #    return array
    def forward(self,config): 
        #y = self._input(config)
        ls = [None] * len(config)
        n = 0
        for i,ci in enumerate(config):
            if i==0:
                continue
            v = self.params[f'v{i}'][config[0],ci,:]
            for j in range(1,i):
                v = self.jnp.einsum('i,j,ijk->k',v,ls[j],self.params[f'T{i}'][j-1,ci,...]) 
            ls[i] = v
        return sum(v)
class CNN(NN):
    def __init__(self,nsite,D,nf=1,**kwargs):
        super().__init__(**kwargs)

        self.pdim = 4 if self.fermion else 2
        key = 'in'
        self.param_keys.append(key)
        self.sh[key] = nsite,self.pdim,self.pdim 

        if isinstance(D,int):
            D = (D,)
        self.D = (self.pdim,) + tuple(D) 
        Dix = 0
        i = 0
        self.init_dim()
        while True:
            if self._break():
                self.sh[key] = self.sh[key][:-1] + (nf,)
                break 
            i,Dix,key = self.set_layer(i,Dix,D)
    def forward(self,config):
        v = {self.flat2site(i):self.params['in'][i,ci,:] for i,ci in enumerate(config)}
        self.init_dim()
        l = 0
        while True:
            if self._break():
                break 
            l,v = self.layer_forward(l,v)
        if isinstance(v,dict):
            v = list(v.values())
        return v[0]
            
