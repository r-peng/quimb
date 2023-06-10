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
from ..tensor_2d_vmc import AmplitudeFactory as BosonAmplitudeFactory
from .fermion_2d_vmc_ import AmplitudeFactory as FermionAmplitudeFactory
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
        self.psi[0] = FermionAmplitudeFactory(psi_a,subspace='a')
        self.psi[1] = FermionAmplitudeFactory(psi_b,subspace='b')
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
from ..tensor_2d_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(Hamiltonian_):
    def parse_energy_numerator(self,exs):
        ex = []
        for ix in range(2):
            ex1,ex2 = exs[ix],exs[2][ix]
            for site1,site2 in ex1:
                ex.append(ex1[site1,site2] * ex2[site1,site2])
        return ex
    def parse_energy(self,exs,cxs):
        ex = 0.
        cx2 = cxs[2]
        np2 = self.ham[2]._2numpy
        for ix in range(2):
            ex1,ex2,cx1 = exs[ix],exs[2][ix],cxs[ix]
            np1 = self.ham[ix]._2numpy
            for site1,site2 in ex1:
                ex += np1(ex1[site1,site2]) * np2(ex2[site1,site2]) / (cx1[site1] * cx2[site1])
        return ex
    def parse_hessian(self,ex,wfns,amplitude_factory):
        if len(ex)==0:
            return 0.,0.
        ex_num = sum(ex)
        ex_num.backward()
        Hvxs = [None] * 3
        for ix in range(3):
            Hvx = dict()
            peps = wfns[ix]
            _2numpy = self.ham[ix]._2numpy
            tsr_grad = self.ham[ix].tsr_grad
            for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
                Hvx[i,j] = _2numpy(tsr_grad(peps[i,j].data))
            Hvxs[ix] = amplitude_factory.psi[ix].dict2vec(Hvx)  
        return ex_num,np.concatenate(Hvxs)
    def contraction_error(self,cxs):
        cx = 1.
        err = 0.
        for ix in range(3): 
            cx_,err_ = self.ham[ix].contraction_error(cxs[ix])
            cx *= cx_
            err = max(err,err_)
        return cx,err
    def batch_hessian_from_plq(self,batch_idx,config,amplitude_factory): # only used for Hessian
        exs,cxs,plqs,wfns = [None] * 3,[None] * 3,[None] * 3,[None] * 3
        configs = parse_config(config)
        for ix in range(3):
            peps = amplitude_factory.psi[ix].psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self.ham[ix]._2backend(peps[i,j].data,True))
            wfns[ix] = peps
            exs[ix],cxs[ix],plqs[ix] = self.ham[ix].batch_pair_energies_from_plq(configs[ix],peps)
        ex = self.parse_energy_numerator(exs)
        _,Hvx = self.parse_hessian(ex,wfns,amplitude_factory)
        ex = self.parse_energy(exs,cxs)

        vxs = [None] * 3
        for ix in range(3):
            vxs[ix] = self.ham[ix].get_grad_dict_from_plq(plqs[ix],cxs[ix],backend=self.backend)
        return ex,Hvx,cxs,vxs 
    def compute_local_energy_hessian_from_plq(config,amplitude_factory):
        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))

        ex,Hvx = 0.,0.
        cxs,vxs = [dict()] * 3,[dict()] * 3
        for batch_idx in self.batched_pairs:
            ex_,Hvx_,cxs_,vxs_ = self.batch_hessian_from_plq(batch_idx,config,amplitude_factory)  
            ex += ex_
            Hvx += Hvx_
            for ix in range(3):
                cxs[ix].update(cxs_[ix])
                vxs[ix].update(vxs_[ix])

        eu = self.compute_local_energy_eigen(config)
        ex += eu

        vx = np.concatenate([amplitude_factory.psi[ix].dict2vec(vx[ix]) for ix in range(3)])
        cx,err = self.contraction_error(cxs)

        Hvx = Hvx/cx + eu*vx
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,err 
    def compute_local_energy_gradient_from_plq(config,amplitude_factory,compute_v=True):
        exs,cxs,plqs = [None] * 3,[None] * 3,[None] * 3
        configs = parse_config(config)
        for ix in range(3):
            exs[ix],cxs[ix],plqs[ix] = self.ham[ix].pair_energies_from_plq(
                                           configs[ix],amplitude_factory.psi[ix])
        ex = self.parse_energy(exs,cxs)
        eu = self.compute_local_energy_eigen(config_full)
        ex += eu

        if not compute_v:
            cx,err = self.contraction_error(cxs)
            return cx,ex,None,None,err 

        vx = np.concatenate([amplitude_factory.psi[ix].get_grad_from_plq(
                 plqs[ix],cxs[ix],backend=self.backend) for ix in range(3)])
        cx,err = self.contraction_error(cxs)
        return cx,ex,vx,None,err
    def compute_local_amplitude_gradient_deterministic(self,config,amplitude_factory):
        cx,vx = np.zeros(3),[None] * 3 
        configs = parse_config(config)
        for ix in range(3):
            cx[ix],vx[ix] = self.ham[ix].amplitude_gradient_deterministic(
                                  configs[ix],amplitude_factory.psi[ix])
        return np.prod(cx),np.concatenate(vx)
    def batch_hessian_deterministic(self,config,amplitude_factory,imin,imax):
        exs,wfns = [None] * 3,[None] * 3
        configs = parse_config(config)
        for ix in range(3):
            psi = amplitude_factory.psi[ix]
            peps = psi.psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self.ham[ix]._2backend(peps[i,j].data,True))
            wfns[ix] = peps
            exs[ix] = self.ham[ix].batch_pair_energies_deterministic(configs[ix],peps,psi.config_sign,
                                                                     imin,imax)
        ex = self.parse_energy_numrator(exs)
        ex,Hvx = self.parse_hessian(ex,wfns,amplitude_factory)
        return self._2numpy(ex_num),Hvx
    def pair_hessian_deterministic(self,config,amplitude_factory,site1,site2):
        exs,wfns = [None] * 3,[None] * 3
        configs = parse_config(config)
        for ix in range(3):
            psi = amplitude_factory.psi[ix]
            peps = psi.psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self.ham[ix]._2backend(peps[i,j].data,True))
            wfns[ix] = peps
            ex = self.ham[ix].pair_energy_deterministic(configs[ix],peps,psi.config_sign,
                                                               site1,site2)
            if ex is None:
                return 0.,0.
            exs[ix] = {(site1,site2):ex} 
        ex = self.parse_energy_numrator(exs)
        ex,Hvx = self.parse_hessian(ex,wfns,amplitude_factory)
        return self._2numpy(ex_num),Hvx
    def compute_local_energy_gradient_deterministic(config,amplitude_factory,compute_v=True):
        configs = parse_config(config)
        ex,cx = [None] * 3,np.zeros(3)
        for ix in range(3):
            ex[ix],cx[ix] = self.ham[ix].pair_energies_deterministic(
                                  configs[ix],amplitude_factory.psi[ix]) 
        cx = np.prod(cx) 
        ex = sum(self.parse_energy_numerator(ex)) / cx
        eu = self.compute_local_energy_eigen(config)
        ex += eu
        if not compute_v:
            return cx,ex,None,None,0.

        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))
        _,vx = self.amplitude_gradient_deterministic(config,amplitude_factory)
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,None,0.
class BosonHamiltonian(Hamiltonian_):
    def pair_tensor(self,bixs,kixs,spin,tags=None):
        data = self._2backend(self.data_map[self.key+spin],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return Tensor(data=data,inds=inds,tags=tags) 
    def pair_energy_from_plq(self,tn,config,site1,site2,spin):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        kixs = [tn.site_ind(*site) for site in [site1,site2]]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip([site1,site2],kixs,bixs):
            tn[tn.site_tag(*site),'BRA'].reindex_({kix:bix})
        tn.add_tensor(self.pair_tensor(bixs,kixs,spin),virtual=True)
        try:
            t0 = time.time()
            ex = tn.contract()
            #print(time.time()-t0)
            return self.pair_coeff(site1,site2) * ex 
        except (ValueError,IndexError):
            return None 
    def _pair_energies_from_plq(self,plq,pairs):
        exa = dict()
        exb = dict()
        cx = dict()
        for (site1,site2) in pairs:
            key = self.pair_key(site1,site2)

            tn = plq.get(key,None) 
            if tn is not None:
                eija = self.pair_energy_from_plq(tn.copy(),config,site1,site2,'a') 
                if eija is not None:
                    exa[site1,site2] = eija
                eijb = self.pair_energy_from_plq(tn.copy(),config,site1,site2,'b') 
                if eijb is not None:
                    exb[site1,site2] = eijb

                if site1 in cx:
                    cij = cx[site1]
                elif site2 in cx:
                    cij = cx[site2]
                else:
                    cij = self._2numpy(tn.copy().contract())
                cx[site1] = cij 
                cx[site2] = cij 
        #print(f'e,time={time.time()-t0}')
        return (exa,exb),cx
class HubbardBoson(BosonHamiltonian):
    def __init__(self,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,phys_dim=4)

        # alpha
        h1a = np.zeros((4,)*4)
        for ki in [1,3]:
            for kj in [0,2]:
                bi = {1:0,3:2}[ki]
                bj = {0:1,2:3}[kj]
                h1a[bi,ki,bj,kj] = 1.
                h1a[bj,kj,bi,ki] = 1.

        # beta
        h1b = np.zeros((4,)*4)
        for ki in [2,3]:
            for kj in [0,1]:
                bi = {2:0,3:1}[ki]
                bj = {0:2,1:3}[kj]
                h1b[bi,ki,bj,kj] = 1.
                h1b[bj,kj,bi,ki] = 1.
        self.key = 'h1'
        self.data_map[self.key+'a'] = h1a 
        self.data_map[self.key+'b'] = h1b 

        self.pairs = self.nn_pairs()
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_nn_plq()
    def batch_deterministic(self):
        self.batched_pairs = dict()
        self.batch_nnh() 
        self.batch_nnv() 
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
    def pair_coeff(self,site1,site2):
        return 1. 
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def pair_terms(self,i1,i2):
        pn_map = {0:0,1:1,2:1,3:2}
        n1,n2 = pn_map[i1],pn_map[i2]
        nsum,ndiff = n1+n2,abs(n1-n2)
        if ndiff==1:
            return [(i2,i1,1)]
        if ndiff==2:
            return [(1,2,1),(2,1,1)] 
        if ndiff==0:
            return [(0,3,1),(3,0,1)]
from .fermion_2d_vmc_ import Hubbard as HubbardFermion
class Hubbard(Hamiltonian):
    def __init__(self,t,u,Lx,Ly,**kwargs):
        self.ham = [None] * 4
        self.ham[0] = HubbardFermion(t,u,Lx,Ly,subspace='a',**kwargs)
        self.ham[1] = HubbardFermion(t,u,Lx,Ly,subspace='b',**kwargs)
        self.ham[2] = HubbardBoson(Lx,Ly,subspace='a'**kwargs)
        self.ham[3] = HubbardBoson(Lx,Ly,subspace='b'**kwargs)
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

