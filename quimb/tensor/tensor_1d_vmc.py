import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

from .tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    Model,
    get_gate2,
    _add_gate,
)
from .tensor_2d_vmc import (
    AmplitudeFactory2D,
    ExchangeSampler2D,
)
class AmplitudeFactory1D(AmplitudeFactory2D):
    def __init__(self,psi,model,blks=None,phys_dim=2,backend='numpy',normalize=1.,pbc=False,**compress_opts):
        self.Lx,self.Ly = 1,psi.L
        self.nsite = psi.L
        self.sites = list(range(self.nsite))
        psi.add_tag('KET')
        self.normalize = normalize
        self.set_psi(psi)

        self.data_map = self.get_data_map(phys_dim)

        self.model = model
        self.backend = backend 
        self.from_plq = True
        self.wfn2backend()

        self.blks = [self.sites] if blks is None else blks
        self.site_map = self.get_site_map()
        self.constructors = self.get_constructors(psi)
        self.get_block_dict()
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)
        self.is_tn = True
        self.dmc = False 
        self.pbc = pbc 
        self.deterministic = False
    def flatten(self,site):
        return site
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
    def get_mid_env(self,config,append='',psi=None):
        psi = self.psi if psi is None else psi 
        row = psi.copy()
        # compute mid env for row i
        for j in range(self.nsite-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(config[j],j,append=append),virtual=True)
        return row
    def contract_mid_env(self,row):
        try: 
            for j in range(self.nsite-1,-1,-1):
                row.contract_tags(row.site_tag(j),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def get_plq(self,config,bsz,psi=None):
        psi = self.psi if psi is None else psi
        cols = self.get_mid_env(config,psi=psi) 
        plq = dict()
        if self.pbc:
            plq[self.nsite-1,self.nsite] = cols.copy()
        cols,lenvs = self.get_all_envs(cols,1,stop=self.nsite-bsz,inplace=False)
        cols,renvs = self.get_all_envs(cols,-1,stop=bsz-1,inplace=False)
        for j in range(self.nsite-bsz+1):
            plq[j,bsz] = self._get_plq(j,bsz,cols,lenvs,renvs)
        return plq 
    def unsigned_amplitude(self,config,i=None,to_numpy=True):
        tn = self.get_mid_env(config)
        tn = self.contract_mid_env(tn)
        cx = safe_contract(tn)
        if cx is None:
            return None
        if to_numpy:
            cx = tensor2backend(cx,'numpy')
        return cx  
    def batch_pair_energies(self,batch_key,new_cache):
        b = self.model.batched_pairs[batch_key]
        # form plqs
        plq = self.get_plq(self.config,b.plq_sz)
        return self.pair_energies_from_plq(plq,b.pairs),plq
class Batch:
    def __init__(self):
        self.pairs = []
        self.plq_sz = 1
class Model1D(Model):
    def __init__(self,L,pbc=False):
        self.nsite = L
        self.pbc = pbc
    def flatten(self,site):
        return site
    def flat2site(self,ix):
        return ix
    def pairs_nn(self,d=1):
        key = 'all'
        if key not in self.batched_pairs:
            self.batched_pairs[key] = Batch()
        b = self.batched_pairs[key]
        for j in range(self.nsite):
            if j+d<self.nsite:
                where = j,j+d
                b.pairs.append(where)
            else:
                if self.pbc:
                    where = j,(j+d)%self.nsite
                    b.pairs.append(where)
        b.plq_sz = max(d+1,b.plq_sz)
class J1J2(Model1D): 
    def __init__(self,J1,J2,L,**kwargs):
        super().__init__(L,**kwargs)
        self.J1,self.J2 = J1,J2

        self.gate = {None:(get_gate2((1.,1.,0.)),'b1,k1,b2,k2')}

        self.batched_pairs = dict()
        self.pairs_nn()
        self.pairs_nn(d=2)
    def pair_key(self,i,j):
        return min(i,j,self.nsite-3),3
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
            for i in range(self.nsite):
                s1 = (-1) ** config[i]
                if i+d<self.nsite:
                    e[d-1] += s1 * (-1)**config[i+d]
                else:
                    if self.pbc:
                        e[d-1] += s1 * (-1)**config[(i+d)%self.nsite]
        return .25 * (e[0]*self.J1 + e[1]*self.J2) 
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.5)]
class ExchangeSampler1D(ExchangeSampler2D):
    def __init__(self,L,beta=1.,burn_in=None,seed=None,nsweep=1):
        self.Lx,self.Ly = 1,L
        self.nsite = L
        self.af = None
        self.px = None
        self.beta = beta
        self.burn_in = burn_in

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.nsweep = nsweep
    def flatten(self,i):
        return i
    def flat2site(self,i):
        return i
    def get_pairs(self,i,j,x_bsz,y_bsz):
        assert y_bsz == 2
        return [(j,(j+1)%self.Ly)]
    #def sweep_col_forward(self,cols,bsz):
    #    cols,renvs = self.af.get_all_envs(cols,-1,stop=bsz-1,inplace=False)
    #    for j in range(self.nsite - bsz + 1): 
    #        plq = self.af._get_plq_sweep(j,bsz,cols,renvs,1)
    #        plq,cols = self._update_pair_from_plq(j,j+1,plq,cols) 
    #        try:
    #            cols = self.af._contract_cols(cols,(0,j))
    #        except (ValueError,IndexError):
    #            return
    #def sweep_col_backward(self,cols,bsz):
    #    cols,lenvs = self.af.get_all_envs(cols,1,stop=self.nsite-bsz,inplace=False)
    #    for j in range(self.nsite - bsz,-1,-1): # Ly-1,...,1
    #        plq = self.af._get_plq_backward(j,bsz,cols,lenvs)
    #        plq,cols = self._update_pair_from_plq(j,j+1,plq,cols) 
    #        try:
    #            cols = self.af._contract_cols(cols,(j+bsz-1,self.nsite-1))
    #        except (ValueError,IndexError):
    #            return
    def sample(self):
        cdir = self.rng.choice([-1,1]) 
        for _ in range(self.nsweep):
            tn = self.af.get_mid_env(self.af.parse_config(self.config))
            self.sweep_col_from_plq(None,tn,1,2)
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
        dataqueue += eps * np.random.rand(*shape)
        if normalize:
            data /= np.linalg.norm(data)
        arrays.append(data)
    return MatrixProductState(arrays) 
def compute_energy(mps,terms,order,pbc=False):
    # terms: tebd ham terms
    norm,_,bra = mps.make_norm(return_all=True)
    bsz = max([abs(i-j)+1 for i,j in terms.keys()])
    af = AmplitudeFactory1D(mps) 
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
