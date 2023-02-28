import time,itertools,sys,warnings
import numpy as np

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_JAX = True
pyblock3.algebra.ad.ENABLE_AUTORAY = True

from pyblock3.algebra import fermion_setting as setting
setting.set_ad(True)

from .block_interface import set_options
flat = False
set_options(use_cpp=flat)

from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False

#########################################################################################
# convert flat fermion tensor to ad-compatible
#########################################################################################
from pyblock3.algebra.ad.core import SubTensor,SparseTensor,FermionTensor
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
def subtensor2ad(x):
    return SubTensor(data=np.asarray(x.data),q_labels=x.q_labels)
def fermiontensor2ad(x):
    try: # data is flat 
        x = x.to_sparse() 
    except NotImplementedError:
        pass
    return SparseFermionTensor(blocks=[subtensor2ad(b) for b in x.blocks],pattern=x.pattern,shape=x.shape)
def fpeps2ad(fpeps):
    fpeps.reorder(direction='row',inplace=True)
    nsite = fpeps.Lx * fpeps.Ly
    psi = FermionTensorNetwork([])
    for ix in range(nsite):
        tsr = fpeps[flat2site(ix,fpeps.Lx,fpeps.Ly)]
        data = fermiontensor2ad(tsr.data)
        tsr = FermionTensor(data=data,inds=tsr.inds,tags=tsr.tags,left_inds=tsr.left_inds)
        tsr.add_tag(_VARIABLE_TAG.format(ix))
        psi.add_tensor(tsr)
    psi.view_like_(fpeps)
    return psi
# set tensor symmetry
from pyblock3.algebra.ad.fermion_ops import vaccum,creation
this = sys.modules[__name__]
def set_options(symmetry,precision=1e-10):
    this.PRECISION = precision

    this.cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    this.cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    this.vac = vaccum(n=1,symmetry=symmetry,flat=flat)
    this.occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
    this.occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
    this.occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
    this.state_map = [vac,occ_a,occ_b,occ_db]
import autoray as ar
SparseTensor.shape = property(lambda x:tuple(ix.n_bonds for ix in x.infos))
FermionTensor.shape = property(lambda x:tuple(ix.n_bonds for ix in x.infos))
ar.register_function('pyblock3','array',lambda x:x)
ar.register_function('pyblock3','to_numpy',lambda x:x)
#######################################################################################
# torch amplitude factory 
#######################################################################################
import torch
from torch import nn
torch.set_num_threads(28)
ar.register_function('torch','conjugate',torch.conj)
warnings.filterwarnings(action='ignore',category=torch.jit.TracerWarning)
from ..optimize import contract_backend,tree_map,to_numpy,_VARIABLE_TAG,Vectorizer

####################################################################################
# amplitude fxns 
####################################################################################
from .fermion_2d_vmc import (
    pn_map,
    flatten,flat2site,
    contract_mid_env,
    AmplitudeFactory2D,
    ExchangeSampler2D,DenseSampler2D,
    Hubbard2D,
)
from .fermion_core import FermionTensor, FermionTensorNetwork
def contract_top_down(fpeps,**compress_opts):
    # form top env
    fpeps = contract_mid_env(fpeps.Lx-1,fpeps)
    if fpeps is None:
        return fpeps
    #for i in range(fpeps.Lx-2,-1,-1):
    #    fpeps = contract_mid_env(i,fpeps)
    #    if i>0:
    #        fpeps.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,fpeps.Ly-1),**compress_opts)
    #return fpeps.contract()
    try:
        for i in range(fpeps.Lx-2,-1,-1):
            fpeps = contract_mid_env(i,fpeps)
            if i>0:
                fpeps.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,fpeps.Ly-1),**compress_opts)
        return fpeps.contract()
    except (ValueError,IndexError):
        return None
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
        psi.add_tag('KET')
        self.psi = psi 

        # initialize parameters
        self.variables = self._get_variables(psi)
        self.vectorizer = Vectorizer(self.variables) 
        self.nparam = len(self.vectorizer.pack(self.variables))
        self.store = dict()
        self.store_grad = dict()
    def get_x(self):
        return self.vectorizer.pack(self.variables).copy() 
    def update(self,x):
        self.variables = self.vectorizer.unpack(vector=x)
        self.psi = self._inject(self.variables,inplace=True)
        self.store = dict()
        self.store_grad = dict()
        return self.psi
    def _psi2vec(self):
        raise NotImplementedError
    def _vec2psi(self):
        raise NotImplementedError
    def _set_psi(self):
        raise NotImplementedError
    def _get_variables(self,psi):
        variables = [None] * self.nsite
        for ix in range(self.nsite):
            tsr = psi[_VARIABLE_TAG.format(ix)]
            variables[ix] = tsr.data.get_params()
        return variables
    def _inject(self,variables=None,inplace=True):
        variables = self.variables if variables is None else variables
        psi = self.psi if inplace else self.psi.copy()
        for ix in range(self.nsite):
            tsr = psi[_VARIABLE_TAG.format(ix)]
            tsr.data.set_params(variables[ix])
        return psi
    def to_variable(self, x):
        return torch.tensor(x).to(self.device).requires_grad_()
    def to_constant(self, x):
        return torch.tensor(x).to(self.device)
    def get_bra_tsr(self,ci,ix,use_torch=True):
        i,j = self.flat2site(ix)
        inds = self.psi.site_ind(i,j),
        tags = self.psi.site_tag(i,j),self.psi.row_tag(i),self.psi.col_tag(j),'BRA'
        data = self.state_map[ci].dagger if use_torch else state_map[ci].dagger
        return FermionTensor(data=data,inds=inds,tags=tags)
    def amplitude(self,config):
        if config in self.store:
            return self.store[config]
        psi = self.psi.copy() # numpy arrays
        for ix,ci in reversed(list(enumerate(config))):
            psi.add_tensor(self.get_bra_tsr(ci,ix,use_torch=False))
        cx = contract_top_down(psi,**self.contract_opts)
        cx = 0. if cx is None else cx
        self.store[config] = cx
        return cx
    def grad(self,config):
        if config in self.store_grad:
            return self.store[config],self.store_grad[config]
        variables = tree_map(self.to_variable,self.variables) 
        psi = self._inject(variables=variables,inplace=False)
        for ix,ci in reversed(list(enumerate(config))):
            psi.add_tensor(self.get_bra_tsr(ci,ix,use_torch=True))
        with contract_backend('torch'): 
            cx = contract_top_down(psi,**self.contract_opts)
        if cx is None:
            cx = 0. 
            gx = np.zeros(self.nparam) 
        else: 
            cx.backward()
            gx = [None] * self.nsite
            for ix1,blks in enumerate(variables):
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
            cx = to_numpy(cx)
            gx = self.vectorizer.pack(gx).copy()
        self.store[config] = cx
        self.store_grad[config] = gx
        return cx,gx
    def prob(self,config):
        return self.amplitude(config)**2
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
    def hop(self,config,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if i1==i2: # no hopping possible
            return 0.
        ex = 0.
        parity = sum([pn_map[ci] for ci in config[ix1+1:ix2]]) % 2
        for i1_new,i2_new,sign in hop(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            ex += sign * self.hop_coeff(site1,site2) * self.cx(tuple(config_new)) 
        return ex * (-1)**parity 
    def nn(self,config,amplitude_factory):
        cx = amplitude_factory.store[config]
        self.cx = amplitude_factory.amplitude
        e = 0.
        # all horizontal bonds
        for i in range(self.Lx):
            for j in range(self.Ly-1):
                site1,site2 = (i,j),(i,j+1)
                e += self.hop(config,site1,site2)
                #print(site1,site2,self.hop(config,site1,site2))
        # all vertical bonds
        for i in range(self.Lx-1):
            for j in range(self.Ly):
                site1,site2 = (i,j),(i+1,j)
                e += self.hop(config,site1,site2)
                #print(site1,site2,self.hop(config,site1,site2))
        return e/cx
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
