import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

SYMMETRY = 'u11' # sampler symmetry
# set tensor symmetry
import sys
this = sys.modules[__name__]
# set backend
import autoray as ar
import torch
#torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(False)

from ..torch_utils import SVD,QR,set_max_bond
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_AUTORAY = True
from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False
####################################################################################
# amplitude fxns 
####################################################################################
from ..tensor_2d_vmc import ContractionEngine as ContractionEngine_
class ContractionEngine(ContractionEngine_): 
    def init_contraction(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly
        self.pbc = pbc
        self.deterministic = deterministic
        if self.deterministic:
            self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
        self.compress_opts = compress_opts

        self.data_map = data_map
    def _2backend(self,data,requires_grad):
        if self.backend=='numpy':
            return data.copy()
        else:
            return SparseFermionTensor.from_flat(data,requires_grad=requires_grad)
    def intermediate_sign(self,config,ix1,ix2):
        return (-1)**(sum(config[ix1+1:ix2]) % 2)
    def _2numpy(self,data,backend=None):
        backend = self.backend if backend is None else backend 
        if backend=='torch':
            try:
                data = data.to_flat()
            except AttributeError:
                data = self._torch2numpy(data,backend=backend) 
        return data
    def tsr_grad(self,tsr,set_zero=True):
        return tsr.get_grad(set_zero=set_zero) 
    def get_bra_tsr(self,fpeps,cij,i,j,append=''):
        inds = fpeps.site_ind(i,j)+append,
        tags = fpeps.site_tag(i,j),fpeps.row_tag(i),fpeps.col_tag(j),'BRA'
        key = {0:'vac_',1:'occ_'}[ci]
        data = self._2backend(data_map[key+self.spin].dagger,False)
        return FermionTensor(data=data,inds=inds,tags=tags)
    def site_grad(self,ftn_plq,i,j):
        ket = ftn_plq[ftn_plq.site_tag(i,j),'KET']
        tid = ket.get_fermion_info()[0]
        ket = ftn_plq._pop_tensor(tid,remove_from_fermion_space='end')
        g = ftn_plq.contract(output_inds=ket.inds[::-1])
        return g.data.dagger 
def compute_fpeps_parity(fs,start,stop):
    if start==stop:
        return 0
    tids = [fs.get_tid_from_site(site) for site in range(start,stop)]
    tsrs = [fs.tensor_order[tid][0] for tid in tids] 
    return sum([tsr.parity for tsr in tsrs]) % 2
def get_parity_cum(fpeps):
    parity = []
    fs = fpeps.fermion_space
    for i in range(1,fpeps.Lx): # only need parity of row 1,...,Lx-1
        start,stop = i*fpeps.Ly,(i+1)*fpeps.Ly
        parity.append(compute_fpeps_parity(fs,start,stop))
    return np.cumsum(np.array(parity[::-1]))
from ..tensor_2d_vmc import AmplitudeFactory as BosonAmplitudeFactory
class SubspaceAmplitudeFactory(ContractionEngine,BosonAmplitudeFactory):
    def __init__(self,psi,spin):
        super().init_contraction(psi.Lx,psi.Ly)
        self.spin = spin 
        psi.reorder(direction='row',inplace=True)
        psi.add_tag('KET')
        self.parity_cum = get_parity_cum(psi)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict()

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
        self.small_mem = True
    def config_sign(self,config): # config is spin separated
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum(config[i*self.Ly:(i+1)*self.Ly]) % 2
        parity = np.array(parity[::-1])
        parity_cum = np.cumsum(parity[:-1])
        parity_cum += self.parity_cum 
        return (-1)**(np.dot(parity[1:],parity_cum) % 2)
    def get_constructors(self,fpeps):
        from .block_interface import Constructor
        constructors = [None] * (fpeps.Lx * fpeps.Ly)
        for i,j in itertools.product(range(fpeps.Lx),range(fpeps.Ly)):
            data = fpeps[fpeps.site_tag(i,j)].data
            bond_infos = [data.get_bond_info(ax,flip=False) \
                          for ax in range(data.ndim)]
            cons = Constructor.from_bond_infos(bond_infos,data.pattern,flat=this.flat)
            dq = data.dq
            size = cons.vector_size(dq)
            ix = flatten(i,j,fpeps.Ly)
            constructors[ix] = (cons,dq),size,(i,j)
        return constructors
    def tensor2vec(self,tsr,ix):
        cons,dq = self.constructors[ix][0]
        return cons.tensor_to_vector(tsr) 
    def vec2tensor(self,x,ix):
        cons,dq = self.constructors[ix][0]
        return cons.vector_to_tensor(x,dq)
    def update(self,x,fname=None,root=0):
        psi = self.vec2psi(x,inplace=True)
        self.set_psi(psi) 
        if RANK==root:
            if fname is not None: # save psi to disc
                write_ftn_to_disc(psi,fname,provided_filename=True)
        return psi
def config_to_ab(config):
    config_a = [None] * len(config)
    config_b = [None] * len(config)
    map_a = {0:0,1:1,2:0,3:1}
    map_b = {0:0,1:0,2:1,3:1}
    for ix,ci in enumerate(config):
        config_a[ix] = map_a[ci] 
        config_b[ix] = map_b[ci] 
    return tuple(config_a),tuple(config_b)
def config_from_ab(config_a,config_b):
    map_ = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}
    return tuple([map_[config_a[ix],config_b[ix]] for ix in len(config_a)])
def parse_config(config):
    if len(config)==2:
        config_a,config_b = config
        config_full = self.config_from_ab(config_a,config_b)
    else:
        config_full = config
        config_a,config_b = self.config_to_ab(config)
    return config_a,config_b,config_full
class AmplitudeFactory:
    def __init__(self,psi_a,psi_b,psi_boson):
        self.psi = [None] * 3
        self.psi[0] = SubspaceAmplitudeFactory(psi_a,'a')
        self.psi[1] = SubspaceAmplitudeFactory(psi_b,'b')
        self.psi[2] = BosonAmplitudeFactory(psi_boson)

        self.nparam = [len(amp_fac.get_x()) for amp_fac in self.psi] 
        self.block_dict = self.psi[0].block_dict.copy()
        self.block_dict += [(start+self.nparam[0],stop+self.nparam[0]) \
                           for start,stop in self.psi[1].block_dict]
        self.block_dict += [(start+self.nparam[1],stop+self.nparam[1]) \
                           for start,stop in self.psi[2].block_dict]

    def config_sign(self,config=None):
        raise NotImplementedError
    def get_constructors(self,peps=None):
        raise NotImplementedError
    def get_block_dict(self):
        raise NotImplementedError
    def tensor2vec(self,tsr,ix=None):
        raise NotImplementedError
    def dict2vecs(self,dict_=None):
        raise NotImplementedError
    def dict2vec(self,dict_=None):
        raise NotImplementedError
    def psi2vecs(self,psi=None):
        raise NotImplementedError
    def psi2vec(self,psi=None):
        raise NotImplementedError
    def split_vec(self,x=None):
        raise NotImplementedError
    def vec2tensor(self,x=None,ix=None):
        raise NotImplementedError
    def vec2dict(self,x=None): 
        raise NotImplementedError
    def vec2psi(self,x=None,inplace=None): 
        raise NotImplementedError
    def get_x(self):
        return np.concatenate([amp_fac.get_x() for amp_fac in self.psi])
    def update(self,x,fname=None,root=0):
        fname_ = fname + '_a' if fname is not None else fname
        self.psi[0].update(x[:self.nparam[0]],fname=fname_,root=root)

        fname_ = fname + '_b' if fname is not None else fname
        self.psi[1].update(x[self.nparam[0]:self.nparam[0]+self.nparam[1]],fname=fname_,root=root)

        fname_ = fname + '_boson' if fname is not None else fname
        self.psi[2].update(x[self.nparam[0]+self.nparam[1]:],fname=fname_,root=root)
    def set_psi(self,psi=None):
        raise NotImplementedError
    def unsigned_amplitude(self,config):
        config_a,config_b,config_full = parse_config(config)
        c0 = self.psi[0].unsigned_amplitude(config_a) 
        c1 = self.psi[1].unsigned_amplitude(config_b) 
        c2 = self.psi[2].unsigned_amplitude(config_full) 
        return c0 * c1 * c2
    def amplitude(self,config=None):
        raise NotImplementedError
    def get_grad_from_plq(self,plq=None,cx=None,backend=None):
        raise NotImplementedError
####################################################################################
# ham class 
####################################################################################
from ..tensor_2d_vmc import Hamiltonian as BosonHamiltonian
class SubspaceHamiltonian(ContractionEngine,BosonHamiltonian):
    def __init__(self,Lx,Ly,spin,nbatch=1):
        super().init_contraction(Lx,Ly)
        self.spin = spin
        self.nbatch = nbatch
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self._2backend(self.data_map[self.key+self.spin],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return FermionTensor(data=data,inds=inds,tags=tags) 
class Hamiltonian:
    def __init__(self,Lx,Ly,nbatch=1):
        self.ham = [None] * 3
        self.ham[0] = SubspaceHamiltonian(Lx,Ly,'a',nbatch=nbatch)
        self.ham[1] = SubspaceHamiltonian(Lx,Ly,'b',nbatch=nbatch)
        self.ham[2] = BosonHamiltonian(Lx,Ly,nbatch=nbatch)
class Hubbard(Hamiltonian):
    def __init__(self,t,u,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.t,self.u = t,u
        self.key = 'h1_'

        if self.pbc:
            self.pairs = self.nn_pairs_pbc()
        else:
            self.batched_pairs = dict() 
            batchsize = self.Lx // self.nbatch 
            for i in range(self.Lx):
                batch_idx = i // batchsize
                if batch_idx not in self.batched_pairs:
                    self.batched_pairs[batch_idx] = [],[] 
                rows,pairs = self.batched_pairs[batch_idx]
                rows.append(i)
                if i+1 < self.Lx:
                    rows.append(i+1)
                for j in range(self.Ly):
                    if j+1<self.Ly:
                        where = (i,j),(i,j+1)
                        pairs.append(where)
                    if i+1<self.Lx:
                        where = (i,j),(i+1,j)
                        pairs.append(where)
            self.pairs = []
            for batch_idx in self.batched_pairs:
                rows,pairs = self.batched_pairs[batch_idx]
                imin,imax = min(rows),max(rows)
                bix,tix = max(0,imax-1),min(imin+1,self.Lx-1) # bot_ix,top_ix,pairs 
                plq_types = (imin,imax,1,2), (imin,imax-1,2,1),# i0_min,i0_max,x_bsz,y_bsz
                self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs 
                self.pairs += pairs
            self.plq_sz = (1,2),(2,1)
            #for batch_idx in self.batched_pairs:
            #    bix,tix,plq_types,pairs = self.batched_pairs[batch_idx]
            #    print(batch_idx,bix,tix,plq_types)
            #    print(pairs)
            if RANK==0:
                print('nbatch=',len(self.batched_pairs))
            #exit()
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
    def pair_coeff(self,site1,site2):
        return -self.t
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def compute_local_energy_eigen(self,config):
        config = np.array(config,dtype=int)
        return self.u*len(config[config==3])
    def pair_terms(self,i1,i2):
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
class DensityMatrix(Hamiltonian):
    def __init__(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly 
        self.pairs = [] 
        self.data = np.zeros((Lx,Ly))
        self.n = 0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
        self.n += 1.
        for i in range(self.Lx):
            for j in range(self.Ly):
                self.data[i,j] += pn_map[config[self.flatten(i,j)]]
        return 0.,0.,None,None,0. 
####################################################################################
# sampler 
####################################################################################
from ..tensor_2d_vmc import ExchangeSampler as ExchangeSampler_
class ExchangeSampler(ContractionEngine,ExchangeSampler_):
    def __init__(self,Lx,Ly,seed=None,burn_in=0,thresh=1e-14):
        super().init_contraction(Lx,Ly)
        self.nsite = self.Lx * self.Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.alternate = False # True if autodiff else False
        self.backend = 'numpy'
        self.thresh = thresh
    def new_pair(self,i1,i2):
        if SYMMETRY=='u11':
            return self.new_pair_u11(i1,i2)
        else:
            raise NotImplementedError
    def new_pair_u11(self,i1,i2):
        n = abs(pn_map[i1]-pn_map[i2])
        if n==1:
            i1_new,i2_new = i2,i1
        else:
            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
            i1_new,i2_new = self.rng.choice(choices)
        return i1_new,i2_new 
    def new_pair_u1(self,i1,i2):
        return
from ..tensor_2d_vmc import DenseSampler as DenseSampler_
class DenseSampler(DenseSampler_):
    def __init__(self,Lx,Ly,nelec,**kwargs):
        self.nelec = nelec
        super().__init__(Lx,Ly,None,**kwargs)
    def get_all_configs(self):
        if SYMMETRY=='u1':
            return self.get_all_configs_u1()
        elif SYMMETRY=='u11':
            return self.get_all_configs_u11()
        else:
            raise NotImplementedError
    def get_all_configs_u11(self):
        assert isinstance(self.nelec,tuple)
        sites = list(range(self.nsite))
        ls = [None] * 2
        for spin in (0,1):
            occs = list(itertools.combinations(sites,self.nelec[spin]))
            configs = [None] * len(occs) 
            for i,occ in enumerate(occs):
                config = [0] * self.nsite 
                for ix in occ:
                    config[ix] = 1
                configs[i] = tuple(config)
            ls[spin] = configs

        na,nb = len(ls[0]),len(ls[1])
        configs = [None] * (na*nb)
        for ixa,configa in enumerate(ls[0]):
            for ixb,configb in enumerate(ls[1]):
                config = [config_map[configa[i],configb[i]] \
                          for i in range(self.nsite)]
                ix = ixa * nb + ixb
                configs[ix] = tuple(config)
        return configs
    def get_all_configs_u1(self):
        if isinstance(self.nelec,tuple):
            self.nelec = sum(self.nelec)
        sites = list(range(self.nsite*2))
        occs = list(itertools.combinations(sites,self.nelec))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = [0] * (self.nsite*2) 
            for ix in occ:
                config[ix] = 1
            configs[i] = tuple(config)

        for ix in range(len(configs)):
            config = configs[ix]
            configa,configb = config[:self.nsite],config[self.nsite:]
            config = [config_map[configa[i],configb[i]] for i in range(self.nsite)]
            configs[ix] = tuple(config)
        return configs

