import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

# set tensor symmetry
import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)

from .torch_utils import SVD,QR
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

import sys
this = sys.modules[__name__]
from .tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
)
from .tensor_vmc import AmplitudeFactory as AmplitudeFactory_
def set_options(pbc=False,deterministic=False,backend='numpy'):
    this._PBC = pbc
def get_all_lenvs(L,tn,jmax=None):
    jmax = L-2 if jmax is None else jmax
    first_col = tn.site_tag(0)
    lenvs = [None] * L
    for j in range(jmax+1): 
        tags = first_col if j==0 else (first_col,tn.site_tag(j))
        try:
            tn ^= tags
            lenvs[j] = tn.select(first_col,virtual=False)
        except (ValueError,IndexError):
            return lenvs
    return lenvs
def get_all_renvs(L,tn,jmin=None):
    jmin = 1 if jmin is None else jmin
    last_col = tn.site_tag(L-1)
    renvs = [None] * L
    for j in range(L-1,jmin-1,-1): 
        tags = last_col if j==L-1 else (tn.site_tag(j),last_col)
        try:
            tn ^= tags
            renvs[j] = tn.select(last_col,virtual=False)
        except (ValueError,IndexError):
            return renvs
    return renvs
def get_plq_from_envs(L,bsz,tn,lenvs,renvs):
    plq = dict()
    for j in range(L-bsz+1): 
        tags = [tn.site_tag(j+ix) for ix in range(bsz)]
        cols = tn.select(tags,which='any',virtual=False)
        try:
            if j>0:
                other = cols
                cols = lenvs[j-1]
                cols.add_tensor_network(other,virtual=False)
            if j<L-bsz:
                cols.add_tensor_network(renvs[j+bsz],virtual=False)
            plq[j,bsz] = cols.view_like_(tn)
        except (AttributeError,TypeError): # lenv/renv is None
            return plq
    return plq
class AmplitudeFactory(AmplitudeFactory_):
    def __init__(self,psi,blks=None,phys_dim=2,**compress_opts):
        self.L = L
        self.nsite = L
        self.sites = list(range(self.L))
        psi.add_tag('KET')
        self.set_psi(psi)
        self.backend = _BACKEND
        self.wfn2backend()

        if blks is None:
            blks = [self.sites]
        self.site_map = self.get_site_map(blks)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict(blks)
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)
    def site_tag(self,site):
        return self.psi.site_tag(site)
    def site_tags(self,site):
        return (self.site_tag(site),)
    def site_ind(self,site):
        return self.psi.site_ind(site)
    def plq_site(self,plq_key):
        raise NotImplementedError
    def update_cache(self,config=None):
        pass
    def get_mid_env(self,config,append='',psi=None):
        psi = self.psi if psi is None else psi 
        row = psi.copy()
        # compute mid env for row i
        for j in range(self.L-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(config[j],j,append=append,tn=row),virtual=True)
        return row
    def contract_mid_env(self,row):
        try: 
            for j in range(self.L-1,-1,-1):
                row.contract_tags(row.site_tag(j),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def get_all_lenvs(self,tn,jmax=None):
        return get_all_lenvs(self.L,tn,jmax=jmax)
    def get_all_renvs(self,tn,jmin=None):
        return get_all_renvs(self.L,tn,jmin=jmin)
    def get_plq(self,config,bsz,psi=None):
        psi = self.psi if psi is None else psi
        tn = self.get_mid_env(config,psi=psi) 
        plq = dict()
        if self.pbc:
            plq[self.L-1,self.L] = tn.copy()
        lenvs = self.get_all_lenvs(tn.copy(),jmax=self.Ly-bsz-1)
        renvs = self.get_all_renvs(tn.copy(),jmin=bsz)
        return get_plq_from_envs(self.L,bsz,tn,lenvs,renvs)
    def get_grad_dict_from_plq(self,plq,cx,backend='numpy'):
        # gradient
        vx = dict()
        for (i0,bsz),tn in plq.items():
            if bsz == self.L:
                continue
            where = i0,i0+bsz-1
            for i in range(i0,i0+bsz):
                if i in vx:
                    continue
                vx[i] = self._2numpy(self.site_grad(tn.copy(),i)/cx[where],backend=backend)
        return vx
    def config_sign(self,config=None):
        return 1.
    def get_constructors(self,psi):
        constructors = [None] * (self.L)
        for i in range(self.L):
            data = psi[i].data
            constructors[i] = data.shape,len(data.flatten())
        return constructors
    def get_block_dict(self):
        start = 0
        blk_dict = [(0,sum([size for _,size in self.constructors]))]
        return blk_dict 
    def tensor2vec(self,tsr,ix=None):
        return tsr.flatten()
    def dict2vecs(self,dict_):
        ls = [None] * len(self.constructors)
        for i,(_,size) in enumerate(self.constructors):
            vec = np.zeros(size)
            g = dict_.get(i,None)
            if g is not None:
                vec = self.tensor2vec(g,ix=i) 
            ls[i] = vec
        return ls
    def dict2vec(self,dict_):
        return np.concatenate(self.dict2vecs(dict_))
    def psi2vecs(self,psi=None):
        psi = self.psi if psi is None else psi
        ls = [None] * len(self.constructors)
        for i,(_,size) in enumerate(self.constructors):
            ls[i] = self.tensor2vec(psi[i].data,ix=i)
        return ls
    def psi2vec(self,psi=None):
        return np.concatenate(self.psi2vecs(psi)) 
    def get_x(self):
        return self.psi2vec()
    def split_vec(self,x):
        ls = [None] * len(self.constructors)
        start = 0
        for i,(_,size) in enumerate(self.constructors):
            stop = start + size
            ls[i] = x[start:stop]
            start = stop
        return ls 
    def vec2tensor(self,x,i):
        shape = self.constructors[i][0]
        return x.reshape(shape)
    def vec2dict(self,x): 
        ls = self.split_vec(x)
        return {i:x for i,x in zip(range(self.L),ls)} 
    def vec2psi(self,x,inplace=True): 
        psi = self.psi if inplace else self.psi.copy()
        ls = self.split_vec(x)
        for i in range(self.L):
            psi[i].modify(data=self.vec2tensor(ls[i],i))
        return psi
    def update(self,x,fname=None,root=0):
        psi = self.vec2psi(x,inplace=True)
        self.set_psi(psi) 
        if RANK==root:
            if fname is not None: # save psi to disc
                write_tn_to_disc(psi,fname,provided_filename=True)
        return psi
    def set_psi(self,psi):
        self.psi = psi
    def unsigned_amplitude(self,config):
        tn = self.get_mid_env(config)
        cx = self.safe_contract(tn)
        cx = 0. if cx is None else cx
        return cx  
    def amplitude(self,config):
        raise NotImplementedError
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def get_grad_from_plq(self,plq,cx,backend=None):
        backend = self.backend if backend is None else backend
        vx = self.get_grad_dict_from_plq(plq,cx,backend=backend)
        return self.dict2vec(vx) 
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
from .tensor_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(Hamiltonian_):
    def batch_pair_energies_from_plq(self,config,batch_idx):
        return self.pair_energies_from_plq(config)
    def pair_energy_from_plq(self,config):
        # form plqs
        plq = dict()
        for bsz in self.model.plq_sz:
            plq.update(self.amplitude_factory.get_plq(config,))

        # compute energy numerator 
        ex,cx = self._pair_energies_from_plq(plq,self.model.pairs,config)
        return ex,cx,plq
    def compute_local_energy(self,config,compute_v=True,compute_Hv=False):
        if compute_Hv:
            return self.compute_local_energy_hessian_from_plq(config)
        else:
            return self.compute_local_energy_gradient_from_plq(config,compute_v=compute_v)
class Model:
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
    def pair_key(self,i,j):
        return i,abs(j-i)+1
from .tensor_vmc import get_gate2,_add_gate
class J1J2(Model): 
    def __init__(self,J1,J2,L):
        super().__init__(L)
        self.J1,self.J2 = J1,J2

        self.gate = tensor2backend(get_gate2((1.,1.,0.),to_bk=False),_BACKEND)

        self.pairs = self.pairs_nn() + self.pairs_nn(d=2)
        self.plq_sz = 3,
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
                s1 = (-1) ** config[self.flatten(i)]
                if i+d<self.L:
                    e1 += s1 * (-1)**config[i+d]
                else:
                    if _PBC:
                        e1 += s1 * (-1)**config[(i+d)%self.L]
        return .25 * (e[0]*self.J1 + e[1]*self.J2) 
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.5)]
#class ExchangeSampler2:
#    def __init__(self,L,seed=None,burn_in=0):
#        self.L = L
#        self.nsite = L
#
#        self.rng = np.random.default_rng(seed)
#        self.exact = False
#        self.dense = False
#        self.burn_in = burn_in 
#        self.amplitude_factory = None
#        self.backend = 'numpy'
#    def preprocess(self):
#        self._burn_in()
#    def _burn_in(self,config=None,burn_in=None):
#        if config is not None:
#            self.config = config 
#        self.px = self.amplitude_factory.prob(self.config)
#
#        if RANK==0:
#            print('\tprob=',self.px)
#            return 
#        t0 = time.time()
#        burn_in = self.burn_in if burn_in is None else burn_in
#        for n in range(burn_in):
#            self.config,self.omega = self.sample()
#        if RANK==SIZE-1:
#            print('\tburn in time=',time.time()-t0)
#    def new_pair(self,i1,i2):
#        return i2,i1
#    def _new_pair(self,i,bsz):
#        ix1,ix2 = i,(i+1)%self.L
#        i1,i2 = self.config[ix1],self.config[ix2]
#        if not self.amplitude_factory.pair_valid(i1,i2): # continue
#            return (None,) * 3
#        i1_new,i2_new = self.new_pair(i1,i2)
#        config_new = list(self.config)
#        config_new[ix1] = i1_new
#        config_new[ix2] = i2_new
#        return (ix1,ix2),(i1_new,i2_new),tuple(config_new)
#    def update_pair(self,i,bsz,cols,tn):
#        sites,config_sites,config_new = self._new_pair(i,bsz)
#        if config_sites is None:
#            return tn
#
#        cols = self.amplitude_factory.replace_sites(cols,sites,config_sites) 
#        py = self.amplitude_factory.safe_contract(cols)
#        if py is None:
#            return tn 
#        py = py ** 2
#
#        try:
#            acceptance = py / self.px
#        except ZeroDivisionError:
#            acceptance = 1. if py > self.px else 0.
#        if self.rng.uniform() < acceptance: # accept, update px & config & env_m
#            self.px = py
#            self.config = config_new
#            tn = self.amplitude_factory.replace_sites(tn,sites,config_sites)
#        return tn
#    def _get_cols_forward(self,first_col,j,bsz,tn,renvs):
#        tags = [tn.site_tag(j+ix) for ix in range(bsz)]
#        cols = tn.select(tags,which='any',virtual=False)
#        if j>0:
#            other = cols
#            cols = tn.select(first_col,virtual=False)
#            cols.add_tensor_network(other,virtual=False)
#        if j<self.L - bsz:
#            cols.add_tensor_network(renvs[j+bsz],virtual=False)
#        cols.view_like_(tn)
#        return cols
#    def sweep_col_forward(self,tn,bsz):
#        try:
#            tn.reorder(inplace=True)
#        except (NotImplementedError,AttributeError):
#            pass
#        renvs = self.amplitude_factory.get_all_renvs(tn.copy(),jmin=bsz)
#        first_col = tn.site_tag(0)
#        for j in range(self.L - bsz + 1): 
#            cols = self._get_cols_forward(first_col,j,bsz,tn,renvs)
#            tn = self.update_pair(j,bsz,cols,tn) 
#            tn ^= first_col,tn.site_tag(j) 
#    def _get_cols_backward(self,last_col,j,bsz,tn,lenvs):
#        tags = [tn.site_tag(j+ix) for ix in range(bsz)] 
#        cols = tn.select(tags,which='any',virtual=False)
#        if j>0:
#            other = cols
#            cols = lenvs[j-1]
#            cols.add_tensor_network(other,virtual=False)
#        if j<self.L - bsz:
#            cols.add_tensor_network(tn.select(last_col,virtual=False),virtual=False)
#        cols.view_like_(tn)
#        return cols
#    def sweep_col_backward(self,tn,bsz):
#        try:
#            tn.reorder(inplace=True)
#        except (NotImplementedError,AttributeError):
#            pass
#        lenvs = self.amplitude_factory.get_all_lenvs(tn.copy(),jmax=self.L-1-bsz)
#        last_col = tn.site_tag(self.L-1)
#        for j in range(self.L - bsz,-1,-1): # Ly-1,...,1
#            cols = self._get_cols_backward(last_col,j,bsz,tn,lenvs)
#            tn = self.update_pair(j,bsz,cols,tn) 
#            tn ^= tn.site_tag(j+bsz-1),last_col
#    def sample(self):
#        cdir = self.rng.choice([-1,1]) 
#        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward
#        tn = self.amplitude_factory.get_mid_env(self.config)
#        sweep_col(tn,2)
#        return self.config,self.px
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
def compute_energy(mps,terms):
    # terms: tebd ham terms
    norm,_,bra = mps.make_norm(return_all=True)
    bsz = max([abs(i-j)+1 for i,j in terms.keys()])
    lenvs = get_all_lenvs(mps.L,norm.copy(),jmax=mps.L-bsz-1) 
    renvs = get_all_renvs(mps.L,norm.copy(),jmin=bsz)
    n = lenvs[bsz-1].copy()
    n.add_tensor_network(renvs[bsz],virtual=False)
    n = n.contract()
    print('norm=',n)

    plq = get_plq_from_envs(mps.L,bsz,norm,lenvs,renvs)
    e = 0.
    for (i,j),gate in terms.items():
        if gate.shape==(4,4):
            gate = gate.reshape((2,)*4)
        e += _add_gate(plq[min(i,j,mps.L-bsz),bsz].copy(),gate,(i,j),mps.site_ind,mps.site_tag,contract=True)
    return e/n  
def build_mpo(L,terms):
    self = None
    for (i,j),gate in terms.items():
        if gate.shape==(4,4):
            gate = gate.reshape((2,)*4) # gate=(b1,b2,k1,k2)
        gate = gate.transpose(0,2,1,3)
        gate = gate.reshape((4,4))
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
            arrs.append(arr.reshape((1,)+arr.shape+(1,)))
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
        print(i,j)
        print(other)
        if self is None:
            self = other 
        else:
            self.add_MPO(other,inplace=True,compress=True)
        print(self) 
