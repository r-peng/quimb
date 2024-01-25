import numpy as np
import itertools,pickle,scipy
#####################################################
# for separate ansatz
#####################################################
from .tensor_vmc import (
    tensor2backend,
    contraction_error,
    AmplitudeFactory,
)
from .nn_core import (
    config_to_ab,
    get_block_dict,
    wfn2backend,
    get_x,
    extract_ad_grad,
    free_ad_cache,
    free_sweep_cache,
)    
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
class CompoundAmplitudeFactory(AmplitudeFactory):
    def __init__(self,af,fermion=False,update=None):
        self.af = af 
        self.Lx,self.Ly = af[0].Lx,af[0].Ly
        self.get_block_dict()

        self.sites = af[0].sites
        self.model = af[0].model
        self.nsite = af[0].nsite
        self.backend = af[0].backend

        self.pbc = af[0].pbc 
        self.from_plq = af[0].from_plq
        self.dmc = af[0].dmc
        self.deterministic = af[0].deterministic 

        self.fermion = fermion 
        if self.fermion:
            self.spinless = af[0].spinless

        self.flatten = af[0].flatten
        self.flat2site = af[0].flat2site
        self.intermediate_sign = af[0].intermediate_sign
    def extract_ad_grad(self):
        return extract_ad_grad(self.af)
    def parse_config(self,config):
        if self.fermion:
            ca,cb = config_to_ab(config)
            return [{'a':ca,'b':cb,None:config}[af.spin] for af in self.af]
        else:
            return [config] * len(self.af)
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        wfn2backend(self.af,backend,requires_grad)
    def get_x(self):
        return get_x(self.af) 
    def get_block_dict(self):
        self.nparam,self.sections,self.block_dict = get_block_dict(self.af)
    def update(self,x,fname=None,root=0):
        x = np.split(x,self.sections)
        for i,af in enumerate(self.af):
            fname_ = None if fname is None else fname+f'_{i}' 
            af.update(x[i],fname=fname_,root=root)
        self.wfn2backend()
    def set_config(self,config,compute_v):
        self.config = config 
        self.cx = dict()
        config = self.parse_config(self.config)
        for af,config_ in zip(self.af,config):
            af.set_config(config_,compute_v)
        return config
    def free_ad_cache(self):
        free_ad_cache(self.af)
    def free_sweep_cache(self,step):
        free_sweep_cache(self.af,step)
    def amplitude2scalar(self):
        for af in self.af:
            af.amplitude2scalar()
class ProductAmplitudeFactory:
    def check(self,config):
        cx = [None] * len(self.af) 
        for ix,af in enumerate(self.af):
            cx[ix] = af.amplitude(config[ix])
        print(config[0],cx)
    def amplitude(self,config,sign=True,to_numpy=True):
        cx = 1 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cx_ix = af.amplitude(config[ix],sign=sign,to_numpy=to_numpy)
                if cx_ix is None:
                    return None 
            else:
                cx_ix = af.amplitude(config[ix],to_numpy=to_numpy)
            cx = cx * cx_ix
        return cx 
    def get_grad_deterministic(self,config):
        cx = [None] * len(self.af)
        vx = [None] * len(self.af)
        for i,af in enumerate(self.af):
            cx[i],vx[i] = af.get_grad_deterministic(config[i])
        return np.prod(np.array(cx)),np.concatenate(vx)
    def parse_gradient(self):
        vx = [None] * len(self.af) 
        for i,af in enumerate(self.af):
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
    def log_prob(self,config,i=None):
        p = [None] * len(self.af) 
        for ix,af in enumerate(self.af):
            p[ix] = af.log_prob(config[ix],i=i)
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
    def amplitude(self,config,_sum=True,to_numpy=True,**kwargs):
        cx = [0] * len(self.af) 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cx[ix] = af.amplitude(config[ix],to_numpy=to_numpy,**kwargs)
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
        cx = self.amplitude(config,to_numpy=False)
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
        ex = dict() 
        b = self.model.batched_pairs[batch_key]
        config = [af.config for af in self.af]
        cx = self.amplitude(config,direction=b.direction,to_numpy=False)
        self.cx['deterministic'] = cx
        for where in b.pairs:
            i = min([i for i,_ in where])
            ex_ij = self.update_pair_energy_from_benvs(where,direction=b.direction,i=i) 
            for tag,eij in ex_ij.items():
                ex[where,tag] = eij,cx,eij/cx
        return ex,None
    def _new_log_prob_from_plq(self,plq,sites,config_sites,config_new):
        cx = self.amplitude(config_new)
        return None,np.log(cx**2)
    def replace_sites(self,*args):
        pass
