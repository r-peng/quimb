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
    pn_map,
    get_mid_env,
    contract_mid_env,
    get_top_env,
    get_all_top_env, 
    compute_fpeps_parity,
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
    try:
        for i in range(fpeps.Lx-2,-1,-1):
            fpeps = contract_mid_env(i,fpeps)
            if i>0:
                fpeps.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,fpeps.Ly-1),**compress_opts)
        return fpeps.contract()
    except (ValueError,IndexError):
        return None
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
        self.variables_ad = tree_map(self.to_variable,self.variables) # torch arrays
        self.psi_ad = self._inject(self.variables_ad,inplace=True) # ad cls with torch arrays

        self.vectorizer = Vectorizer(self.variables) 
        self.nparam = len(self.vectorizer.pack(self.variables))
        self.store = dict()
        self.store_grad = dict()
        self.cache_top = dict()
    def get_x(self):
        return self.vectorizer.pack(self.variables).copy() 
    def update(self,x):
        self.variables = self.vectorizer.unpack(vector=x) # numpy arrays
        self.psi_ad = self._inject(self.variables,inplace=True) # numpy arrays
        self.psi = self.convert_psi(self.psi_ad)

        self.variables_ad = tree_map(self.to_variable,self.variables) # torch arrays
        self.psi_ad = self._inject(self.variables_ad,inplace=True) # numpy arrays
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
    def get_bra_tsr(self,ci,ix):
        i,j = self.flat2site(ix)
        inds = self.psi.site_ind(i,j),
        tags = self.psi.site_tag(i,j),self.psi.row_tag(i),self.psi.col_tag(j),'BRA'
        data = self.state_map[ci].dagger 
        return FermionTensor(data=data,inds=inds,tags=tags)
    def unsigned_amplitude(self,config):
        if config in self.store:
            return self.store[config]
        env_prev = get_all_top_envs(self.psi,config,self.cache_top,imin=0,**self.contract_opts)
        if env_prev is None:
            self.unsigned_cx = 0.
        else:
            try:
                self.unsigned_cx = env_prev.contract()
            except (ValueError,IndexError):
                self.unsigned_cx = 0.
        return self.unsigned_cx
    def grad(self,config):
        if config in self.store_grad:
            return self.store[config],self.store_grad[config]
        psi = self.psi_ad.copy()
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
            for ix1,blks in enumerate(self.variables_ad):
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
    def __init__(self,Lx,Ly,t,u):
        super().__init__(Lx,Ly,t,u)
        cre_a = data_map['cre_a']
        cre_b = data_map['cre_b']
        ann_a = data_map['ann_a']
        ann_b = data_map['ann_b']
        self.op_map = {(0,1):[(ann_a,cre_a,-1.)],(1,0):[(cre_a,ann_a,1.)],
                       (0,2):[(ann_b,cre_b,-1.)],(2,0):[(cre_b,ann_b,1.)],
                       (0,3):[(ann_a,cre_a,-1.),(ann_b,cre_b,-1.)],
                       (3,0):[(cre_a,ann_a,1.), (cre_b,ann_b,1.)],
                       (1,2):[(cre_a,ann_a,1.), (ann_b,cre_b,-1.)],
                       (2,1):[(cre_b,ann_b,-1.),(ann_a,cre_a,1.)],
                       (1,3):[(ann_b,cre_b,-1.)],(3,1):[(cre_b,ann_b,1.)],
                       (2,3):[(ann_a,cre_a,-1.)],(3,2):[(cre_a,ann_a,1.)]}
    def nnv(self,env_prev,fpeps,config,site1,site2,**compress_opts):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if i1==i2: # no hopping possible
            return 0.
        ex = 0.
        imax = max([site[0] for site in (site1,site2)])
        #print(site1,site2)
        for op1,op2,sign in self.op_map[i1,i2]:
            ftn = env_prev.copy()
            for site,op in zip((site1,site2),(op1,op2)):
                pix = fpeps.site_ind(*site)
                tags = fpeps.row_tag(site[0]),fpeps.col_tag(site[1]),fpeps.site_tag(*site),'OP'
                ftn.add_tensor(FermionTensor(data=op.copy(),inds=(pix+'*',pix),tags=tags)) 
            #print(ftn)
            try:
                for i in range(imax,-1,-1):
                    row = get_mid_env(i,fpeps,config) 
                    if i==site1[0]:
                        tsr = row[row.site_tag(*site1),'BRA']
                        pix = row.site_ind(*site1)
                        tsr.reindex_({pix:pix+'*'})
                    if i==site2[0]:
                        tsr = row[row.site_tag(*site2),'BRA']
                        pix = row.site_ind(*site2)
                        tsr.reindex_({pix:pix+'*'})
                    ftn = FermionTensorNetwork([row,ftn],virtual=True).view_like_(fpeps)
                    #print(ftn)
                    ftn = contract_mid_env(i,ftn) 
                    if ftn is None:
                        break
                    if i<self.Lx-1:
                        ftn.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,fpeps.Ly-1),
                                                        **compress_opts) 
                if ftn is not None:
                    ex += sign * ftn.contract()
            except (ValueError,IndexError):
                continue 
        #exit()
        return ex * self.hop_coeff(site1,site2) 
    def nnh(self,env_prev,fpeps,config,site1,site2,**compress_opts):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if i1==i2: # no hopping possible
            return 0.
        imax = site1[0]
        ex = 0.
        for i1_new,i2_new,sign in hop(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            benvs = get_top_envs(fpeps,config_new,env_prev=env_prev,imax=imax,imin=0,**compress_opts)
            ftn = benvs[0]
            if ftn is None:
                continue
            try:
                ex += sign * self.hop_coeff(site1,site2) * ftn.contract()
            except (ValueError,IndexError):
                continue 
        return ex  
    def nn(self,config,amplitude_factory):
        fpeps = amplitude_factory.psi
        fs = fpeps.fermion_space
        compress_opts = amplitude_factory.contract_opts
        benvs = get_top_envs(fpeps,config,imax=self.Lx-1,imin=0,**compress_opts)
        ftn = benvs[0]
        cx = ftn.contract() # unsigned 
        e = 0.
        # all horizontal bonds
        # adjacent, no sign in between
        #for i in range(self.Lx):
        #    env_prev = None if i==self.Lx-1 else benvs[i+1]
        #    for j in range(self.Ly-1):
        #        site1,site2 = (i,j),(i,j+1)
        #        e += self.nnh(env_prev,fpeps,config,site1,site2,**compress_opts)
        #        #print(site1,site2,self.hop(config,site1,site2))
        # all vertical bonds
        for i in range(1,self.Lx):
            env_prev = FermionTensorNetwork([]) if i==self.Lx-1 else benvs[i+1]
            for j in range(self.Ly):
                site1,site2 = (i-1,j),(i,j)
                e += self.nnv(env_prev,fpeps,config,site1,site2,**compress_opts)
        #        #print(site1,site2,self.hop(config,site1,site2))
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
