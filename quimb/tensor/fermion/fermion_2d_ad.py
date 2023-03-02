import time,itertools,sys,warnings
import numpy as np

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = True
pyblock3.algebra.ad.ENABLE_AUTORAY = True

from pyblock3.algebra import fermion_setting as setting
setting.set_ad(True)

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False

#########################################################################################
# convert pyblock3 flat fermion tensor to ad-compatible
#########################################################################################
from pyblock3.algebra.ad.core import SubTensor
from pyblock3.algebra.ad.fermion import SparseFermionTensor
from jax.tree_util import tree_flatten, tree_unflatten
def get_params(self):
    params,tree = tree_flatten(self)
    self._tree = tree
    return params
def set_params(self,params):
    x = tree_unflatten(self._tree,params)
    self.blocks = x.blocks
    self._pattern = x.pattern
    self._shape = x.shape
SparseFermionTensor.get_params = get_params
SparseFermionTensor.set_params = set_params
# set tensor symmetry
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True):
    from .fermion_2d_vmc import set_options 
    # flat tensors for non-grad contraction
    this.data_map = set_options(symmetry=symmetry,flat=flat) 

    this.flat = flat
    tsr = this.data_map[0]
    try:
        spt= tsr.to_sparse()
    except NotImplementedError:
        spt = tsr 
    this.spt_cls = spt.__class__ # non-ad SparseFermionTensor
    this.blk_cls = spt.blocks[0].__class__ # non-ad SubTensor
    this.tsr_cls = tsr.__class__
    this.state_map = [convert_tsr(this.data_map[i]) for i in range(4)] # converts only states to ad
def convert_blk(x):
    if isinstance(x,SubTensor):
        return blk_cls(reduced=np.asarray(x.data),q_labels=x.q_labels)
    elif isinstance(x,blk_cls):
        return SubTensor(data=np.asarray(x.data),q_labels=x.q_labels)
    else:
        raise ValueError(f'blk type = {type(x)}')
def convert_tsr(x):
    if isinstance(x,SparseFermionTensor):
        new_tsr = spt_cls(blocks=[convert_blk(b) for b in x.blocks],pattern=x.pattern,shape=x.shape)
        if flat:
            new_tsr = new_tsr.to_flat()
        return new_tsr
    elif isinstance(x,tsr_cls):
        try:  
            x = x.to_sparse() 
        except NotImplementedError:
            pass
        return SparseFermionTensor(blocks=[convert_blk(b) for b in x.blocks],pattern=x.pattern,shape=x.shape)
    else:
        raise ValueError(f'tsr type = {type(x)}')
import autoray as ar
ar.register_function('pyblock3','array',lambda x:x)
ar.register_function('pyblock3','to_numpy',lambda x:x)
####################################################################################
# amplitude fxns 
####################################################################################
from .fermion_2d_vmc import (
    RANK,pn_map,
    get_mid_env,
    contract_mid_env,
    #get_top_env,
    get_all_top_envs, 
    compute_fpeps_parity,
    AmplitudeFactory2D,
    ExchangeSampler2D,DenseSampler2D,
    Hubbard2D,
)
from .fermion_core import FermionTensor, FermionTensorNetwork
from ..optimize import contract_backend,tree_map,to_numpy,_VARIABLE_TAG,Vectorizer
#######################################################################################
# torch amplitude factory 
#######################################################################################
import torch
torch.set_num_threads(28)
ar.register_function('torch','conjugate',torch.conj)
warnings.filterwarnings(action='ignore',category=torch.jit.TracerWarning)
class TorchAmplitudeFactory2D(AmplitudeFactory2D):
    def __init__(self,psi,device=None,**contract_opts):
        self.contract_opts = contract_opts
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.state_map = [None] * 4
        for ix,data in enumerate(state_map):
            data_torch = data.copy()
            params = data_torch.get_params()
            params = tree_map(self.to_constant,params)
            data_torch.set_params(params)
            self.state_map[ix] = data_torch

        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.nsite = self.Lx * self.Ly
        psi.reorder(direction='row',inplace=True)
        self.psi = psi # flat cls with numpy arr
        self.psi_ad = self.convert_psi(psi) # ad cls with numpy array
        self.parity_cum = self.get_parity_cum()

        # initialize parameters
        self.variables = self._get_variables(self.psi_ad) # numpy arrays
        self.vectorizer = Vectorizer(self.variables) 
        self.nparam = len(self.vectorizer.pack(self.variables))
        self.sign = dict()
        self.store = dict()
        self.store_grad = dict()
        self.cache_top = dict()
        self.compute_top = True
        self.compute_bot = False
    def get_x(self):
        return self.vectorizer.pack(self.variables).copy() 
    def update(self,x):
        self.variables = self.vectorizer.unpack(vector=x) # numpy arrays
        self.psi_ad = self._inject(self.variables,inplace=True) # numpy arrays
        self.psi = self.convert_psi(self.psi_ad)
        self.store = dict()
        self.store_grad = dict()
        self.cache_top = dict()
        return self.psi
    def _psi2vec(self):
        raise NotImplementedError
    def _vec2psi(self):
        raise NotImplementedError
    def _set_psi(self):
        raise NotImplementedError
    def update_scheme(self,benv_dir=None):
        pass
    def _get_variables(self,psi):
        variables = [None] * self.nsite
        for ix in range(self.nsite):
            tsr = psi[_VARIABLE_TAG.format(ix)]
            variables[ix] = tsr.data.get_params()
        return variables
    def _inject(self,variables,inplace=True):
        psi = self.psi_ad if inplace else self.psi_ad.copy()
        for ix in range(self.nsite):
            tsr = psi[_VARIABLE_TAG.format(ix)]
            tsr.data.set_params(variables[ix])
        return psi
    def convert_psi(self,psi):
        psi_new = FermionTensorNetwork([])
        for ix in range(self.nsite):
            tsr = psi[self.flat2site(ix)]
            data = convert_tsr(tsr.data)
            tsr = FermionTensor(data=data,inds=tsr.inds,tags=tsr.tags,left_inds=tsr.left_inds)
            tsr.add_tag(_VARIABLE_TAG.format(ix))
            psi_new.add_tensor(tsr)
        psi_new.view_like_(psi)
        return psi_new
    def to_variable(self, x):
        return torch.tensor(x).to(self.device).requires_grad_()
    def to_constant(self, x):
        return torch.tensor(x).to(self.device)
    def get_bra_tsr(self,ci,i,j):
        inds = self.psi.site_ind(i,j),
        tags = self.psi.site_tag(i,j),self.psi.row_tag(i),self.psi.col_tag(j),'BRA'
        data = self.state_map[ci].dagger 
        return FermionTensor(data=data,inds=inds,tags=tags)
    def get_mid_env(self,i,fpeps,config):
        row = fpeps.select(fpeps.row_tag(i)).copy()
        key = config[i*fpeps.Ly:(i+1)*fpeps.Ly]
        # compute mid env for row i
        for j in range(row.Ly-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(key[j],i,j),virtual=True)
        return row
    def get_top_env(self,i,row,env_prev):
        row = contract_mid_env(i,row)
        if i==row.Lx-1:
            return row
        if row is None:
            return row
        if env_prev is None:
            return None
        ftn = FermionTensorNetwork([row,env_prev],virtual=True).view_like_(row)
        try:
            ftn.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,row.Ly-1),**self.contract_opts)
        except (ValueError,IndexError):
            ftn = None
        return ftn 
    def grad(self,config):
        sign = self.compute_config_sign(config)
        if config in self.store_grad:
            unsigned_cx = self.store[config]
            unsigned_gx = self.store_grad[config]
            return sign * unsigned_cx, sign * unsigned_gx
        variables_ad = tree_map(self.to_variable,self.variables) # torch arrays
        self.psi_ad = self._inject(variables_ad,inplace=True) # ad cls with torch arrays
        imin = 1
        env_top = None
        with contract_backend('torch'): 
            for i in range(self.Lx-1,imin-1,-1):
                row = self.get_mid_env(i,self.psi_ad,config)
                env_top = self.get_top_env(i,row,env_top)
            if env_top is None:
                unsigned_cx = 0.
                unsigned_gx = np.zeros(self.nparam) 
                self.store[config] = unsigned_cx
                self.store_grad[config] = unsigned_gx
                return unsigned_cx,unsigned_gx
            row = self.get_mid_env(0,self.psi_ad,config)
            ftn = FermionTensorNetwork([row,env_top],virtual=True)
            try:
                unsigned_cx = ftn.contract()
            except (ValueError,IndexError):
                unsigned_cx = 0.
                unsigned_gx = np.zeros(self.nparam) 
                self.store[config] = unsigned_cx
                self.store_grad[config] = unsigned_gx
                return unsigned_cx,unsigned_gx
        unsigned_cx.backward()
        gx = [None] * self.nsite
        for ix1,blks in enumerate(variables_ad):
            gix1 = [None] * len(blks) 
            for ix2,t in enumerate(blks):
                if t.grad is None:
                    gix1[ix2] = np.zeros(t.shape)
                else:
                    gt = t.grad
                    mask = torch.isnan(gt)
                    gt.masked_fill_(mask,0.)
                    #gix1[ix2] = to_numpy(gt).conj()
                    gix1[ix2] = to_numpy(gt)
            gx[ix1] = gix1
        unsigned_cx = to_numpy(unsigned_cx)
        unsigned_gx = self.vectorizer.pack(gx).copy()
        self.store[config] = unsigned_cx
        self.store_grad[config] = unsigned_gx
        return unsigned_cx*sign,unsigned_gx*sign
####################################################################################
# ham class 
####################################################################################
def hop(i1,i2):
    n1,n2 = pn_map[i1],pn_map[i2]
    nsum,ndiff = n1+n2,abs(n1-n2)
    if ndiff==1:
        sign = 1 if nsum==1 else -1
        return [(i2,i1,sign)]
    if ndiff==2:
        return [(1,2,-1),(2,1,1)] 
    if ndiff==0:
        sign = i1-i2
        return [(0,3,sign),(3,0,sign)]
class Hubbard2D(Hubbard2D):
    def hop(self,config,site1,site2,unsigned_amp_fn,sign_fn=None):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if i1==i2: # no hopping possible
            return 0.
        ex = 0.
        for i1_new,i2_new,hop_sign in hop(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            unsigned_cy = unsigned_amp_fn(config_new) 
            sign_cy = 1. if sign_fn is None else sign_fn(config_new) 
            ex += sign_cy * unsigned_cy * hop_sign
        parity = sum([pn_map[ci] for ci in config[ix1+1:ix2]]) % 2
        return ex * self.hop_coeff(site1,site2) * (-1)**parity 
    def nn(self,config,amplitude_factory):
        unsigned_cx = amplitude_factory.store[config]
        sign_cx = amplitude_factory.sign[config]

        unsigned_amp_fn = amplitude_factory.unsigned_amplitude
        sign_fn = amplitude_factory.compute_config_sign

        # all horizontal bonds
        # adjacent, no sign in between
        eh = 0.
        for i in range(self.Lx):
            for j in range(self.Ly-1):
                site1,site2 = (i,j),(i,j+1)
                eh += self.hop(config,site1,site2,unsigned_amp_fn)
        eh /= unsigned_cx
       
        # all vertical bonds
        ev = 0.
        for i in range(1,self.Lx):
            for j in range(self.Ly):
                site1,site2 = (i-1,j),(i,j)
                ev += self.hop(config,site1,site2,unsigned_amp_fn,sign_fn=sign_fn)
        ev /= sign_cx * unsigned_cx
        return eh+ev
class ExchangeSampler2D(ExchangeSampler2D):
    def __init__(self,Lx,Ly,nelec,seed=None,burn_in=0,sweep=True):
        super().__init__(Lx,Ly,nelec,seed=seed,burn_in=burn_in)
        self.sweep = sweep
        self.hbonds = [((i,j),(i,j+1)) for i in range(self.Lx) for j in range(self.Ly-1)]
        self.vbonds = [((i,j),(i+1,j)) for j in range(self.Ly) for i in range(self.Lx-1)]
    def initialize(self,config,thresh=1e-10):
        self.config = config
        self.px = self.amplitude_factory.prob(config)
        if self.px < thresh:
            raise ValueError 
    def sample(self):
        # randomly choose to sweep h or v bonds
        if self.sweep:
            path = self.rng.choice([0,1])
            bonds = self.hbonds if path==0 else self.vbonds
            # randomly choose to sweep forward or backward
            direction = self.rng.choice([1,-1])
            if direction == -1:
                bonds = bonds[::-1]
        else:
            bonds = self.rng.permutation(self.hbonds+self.vbonds)
        for site1,site2 in bonds: 
            ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
            i1,i2 = self.config[ix1],self.config[ix2]
            if i1==i2: # continue
                #print(i,j,site1,site2,ix1,ix2,'pass')
                continue
            i1_new,i2_new = self.new_pair(i1,i2)
            config = list(self.config)
            config[ix1] = i1_new
            config[ix2] = i2_new 
            config = tuple(config)
            py = self.amplitude_factory.prob(config)
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept
                self.px = py 
                self.config = config 
        return self.config,self.px 
