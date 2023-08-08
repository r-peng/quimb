import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import sys
this = sys.modules[__name__]
from .tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    Hamiltonian,
    Model,
    get_gate2,
    _add_gate,
)
from .tensor_2d_vmc import AmplitudeFactory2D,ExchangeSampler2D 
def set_options(pbc=False,deterministic=False,backend='numpy'):
    this._PBC = pbc
    from .tensor_2d_vmc import set_options
    set_options(pbc=pbc)
class AmplitudeFactory1D(AmplitudeFactory2D):
    def __init__(self,psi,blks=None,phys_dim=2,backend='numpy',**compress_opts):
        self.L = psi.L
        self.Ly = psi.L
        self.nsite = psi.L
        self.sites = list(range(self.L))
        psi.add_tag('KET')
        self.set_psi(psi)
        self.backend = backend 

        self.data_map = self.get_data_map(phys_dim)
        self.wfn2backend()
        self.pbc = _PBC

        if blks is None:
            blks = [self.sites]
        self.site_map = self.get_site_map(blks)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict(blks)
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)
        self.nparam = len(self.get_x())
    def site_tag(self,site):
        return self.psi.site_tag(site)
    def site_tags(self,site):
        return (self.site_tag(site),)
    def site_ind(self,site):
        return self.psi.site_ind(site)
    def col_tag(self,col):
        return self.psi.site_tag(col)    
    def plq_sites(self,plq_key):
        i0,bsz = plq_key
        sites = list(range(i0,i0+bsz))
        return sites
    def update_cache(self,config=None):
        pass
    def set_psi(self,psi):
        self.psi = psi
    def get_mid_env(self,config,append='',psi=None):
        psi = self.psi if psi is None else psi 
        row = psi.copy()
        # compute mid env for row i
        for j in range(self.L-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(config[j],j,append=append),virtual=True)
        return row
    def contract_mid_env(self,row):
        try: 
            for j in range(self.L-1,-1,-1):
                row.contract_tags(row.site_tag(j),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def get_plq(self,config,bsz,psi=None):
        psi = self.psi if psi is None else psi
        cols = self.get_mid_env(config,psi=psi) 
        plq = dict()
        if self.pbc:
            plq[self.L-1,self.L] = tn.copy()
        cols,lenvs = self.get_all_envs(cols,1,stop=self.Ly-bsz,inplace=False)
        cols,renvs = self.get_all_envs(cols,-1,stop=bsz-1,inplace=False)
        for j in range(self.L-bsz+1):
            plq[j,bsz] = self._get_plq(j,bsz,cols,lenvs,renvs)
        return plq 
    def unsigned_amplitude(self,config,to_numpy=True):
        tn = self.get_mid_env(config)
        cx = safe_contract(tn)
        if to_numpy:
            cx = 0. if cx is None else cx
        return cx  
    def amplitude(self,config):
        raise NotImplementedError
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
class Hamiltonian1D(Hamiltonian):
    def batch_pair_energies_from_plq(self,config,batch_key,new_cache=None,compute_v=True,to_vec=False):
        af = self.amplitude_factory  
        # form plqs
        plq = dict()
        for bsz in self.model.plq_sz:
            plq.update(af.get_plq(config,bsz))

        # compute energy numerator 
        ex,cx = self._pair_energies_from_plq(plq,self.model.pairs,config,af=af)
        if compute_v:
            vx = af.get_grad_from_plq(plq,to_vec=to_vec) 
        else:
            vx = None if to_vec else dict()
        return ex,cx,vx
    def compute_local_energy(self,config,compute_v=True,compute_Hv=False):
        if compute_Hv:
            return self.compute_local_energy_hessian_from_plq(config)
        else:
            return self.compute_local_energy_gradient_from_plq(config,compute_v=compute_v)
class Model1D(Model):
    def __init__(self,L):
        self.L = L
        self.nsite = L
    def flatten(self,site):
        return site
    def flat2site(self,ix):
        return ix
    def pairs_nn(self,d=1):
        ls = [] 
        for j in range(self.L):
            if j+d<self.L:
                where = j,j+d
                ls.append(where)
            else:
                if _PBC:
                    where = j,(j+d)%self.L
                    ls.append(where)
        return ls
class J1J2(Model1D): 
    def __init__(self,J1,J2,L):
        super().__init__(L)
        self.J1,self.J2 = J1,J2

        self.gate = get_gate2((1.,1.,0.))
        self.order = 'b1,k1,b2,k2'

        self.pairs = self.pairs_nn() + self.pairs_nn(d=2)
        self.batched_pairs = [None]
        self.plq_sz = 3,
    def pair_key(self,i,j):
        return min(i,j,self.L-3),3
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def pair_coeff(self,i,j): # coeff for pair tsr
        d = abs(i-j)
        if d==1:
            return self.J1
        if d==2:
            return self.J2
    def compute_local_energy_eigen(self,config):
        e = np.zeros(2) 
        for d in (1,2):
            for i in range(self.L):
                s1 = (-1) ** config[i]
                if i+d<self.L:
                    e[d-1] += s1 * (-1)**config[i+d]
                else:
                    if _PBC:
                        e[d-1] += s1 * (-1)**config[(i+d)%self.L]
        return .25 * (e[0]*self.J1 + e[1]*self.J2) 
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.5)]
class ExchangeSampler1D(ExchangeSampler2D):
    def __init__(self,L,seed=None,burn_in=0):
        self.L = L
        self.nsite = L

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
    def flatten(self,i):
        return i
    def flat2site(self,i):
        return i
    def sweep_col_forward(self,cols,bsz):
        af = self.amplitude_factory
        cols,renvs = af.get_all_envs(cols,-1,stop=bsz-1,inplace=False)
        for j in range(self.L - bsz + 1): 
            plq = af._get_plq_forward(j,bsz,cols,renvs)
            _,cols = self._update_pair(j,j+1,plq,cols) 
            try:
                cols = af._contract_cols(cols,(0,j))
            except (ValueError,IndexError):
                return
    def sweep_col_backward(self,cols,bsz):
        af = self.amplitude_factory
        cols,lenvs = af.get_all_envs(cols,1,stop=self.L-bsz,inplace=False)
        for j in range(self.L - bsz,-1,-1): # Ly-1,...,1
            plq = af._get_plq_backward(j,bsz,cols,lenvs)
            _,cols = self._update_pair(j,j+1,plq,cols) 
            try:
                cols = af._contract_cols(cols,(j+bsz-1,self.L-1))
            except (ValueError,IndexError):
                return
    def sample(self):
        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward
        tn = self.amplitude_factory.get_mid_env(self.config)
        sweep_col(tn,2)
        return self.config,self.px
from .tensor_1d import MatrixProductState,MatrixProductOperator
from .tensor_core import Tensor,TensorNetwork,rand_uuid
def get_product_state(L,config=None,bdim=1,eps=0.,pdim=4,normalize=True,pbc=False):
    arrays = []
    for i in range(L):
        shape = [bdim] * 2
        if not pbc and (i==0 or i==L-1):
            shape.pop()
        shape = tuple(shape) + (pdim,)

        if config is None:
            data = np.ones(shape)
        else:
            data = np.zeros(shape)
            ix = config[i]
            data[(0,)*(len(shape)-1)+(ix,)] = 1.
        data += eps * np.random.rand(*shape)
        if normalize:
            data /= np.linalg.norm(data)
        arrays.append(data)
    return MatrixProductState(arrays) 
def compute_energy(mps,terms,order,pbc=False):
    # terms: tebd ham terms
    norm,_,bra = mps.make_norm(return_all=True)
    bsz = max([abs(i-j)+1 for i,j in terms.keys()])
    set_options(pbc=pbc)
    af = AmplitudeFactory(mps) 
    _,lenvs = af.get_all_envs(norm,1,stop=mps.L-bsz,inplace=False) 
    _,renvs = af.get_all_envs(norm,-1,stop=bsz-1,inplace=False)
    n = lenvs[bsz-1].copy()
    n.add_tensor_network(renvs[bsz],virtual=False)
    n = n.contract()
    print('norm=',n)

    plq = dict()
    for j in range(mps.L-bsz+1):
        plq[j,bsz] = af._get_plq(j,bsz,norm,lenvs,renvs)
    e = 0.
    for (i,j),gate in terms.items():
        if gate.shape==(4,4):
            gate = gate.reshape((2,)*4)
        e += _add_gate(plq[min(i,j,mps.L-bsz),bsz].copy(),gate,order,(i,j),mps.site_ind,mps.site_tag,contract=True)
    return e/n  
def build_mpo(L,terms,order):
    self = None
    for (i,j),gate in terms.items():
        if gate.shape==(4,4):
            gate = gate.reshape((2,)*4)
        if order=='b1,b2,k1,k2':
            gate = gate.transpose(0,2,1,3)
        gate = gate.reshape((4,)*2)
        u,s,v = np.linalg.svd(gate)
        bdim = len(s)
        s **= .5
        u *= s.reshape((1,-1))
        v *= s.reshape((-1,1))
        u = u.reshape((2,2,bdim))
        v = v.reshape((bdim,2,2))
        

        arrs = []
        for site in range(i):
            arr = np.eye(2)
            if site>0:
                arr = arr.reshape((1,)+arr.shape)
            arr = arr.reshape(arr.shape+(1,))
            arrs.append(arr)

        if i>0:
            u = u.reshape((1,)+u.shape)
        arrs.append(u)
        for site in range(i+1,j):
            arr = np.einsum('lr,ud->ludr',np.eye(bdim),np.eye(2))
            arrs.append(arr)
        if j<L-1:
            v = v.reshape(v.shape+(1,))
        arrs.append(v)

        for site in range(j+1,L):
            arr = np.eye(2)
            arr = arr.reshape((1,)+arr.shape)
            if site<L-1:
                arr = arr.reshape(arr.shape+(1,))
            arrs.append(arr)
        other = MatrixProductOperator(arrs,shape='ludr')
        if self is None:
            self = other 
        else:
            self.add_MPO(other,inplace=True,compress=True)
    return self
