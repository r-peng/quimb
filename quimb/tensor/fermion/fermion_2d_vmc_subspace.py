import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

# set backend
import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)

from ..torch_utils import SVD,QR
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_AUTORAY = True
from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False

def set_options(symmetry='u1',flat=True,pbc=False,deterministic=False,**compress_opts):
    from ..tensor_2d_vmc_ import set_options as set_options1
    set_options1(pbc=pbc,deterministic=deterministic,**compress_opts)
    from .fermion_2d_vmc_ import set_options as set_options2
    set_options2(symmetry=symmetry,flat=flat,pbc=pbc,deterministic=deterministic,**compress_opts)

from ..tensor_2d_vmc_ import AmplitudeFactory as BosonAmplitudeFactory
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
        config_full = config_from_ab(config_a,config_b)
    else:
        config_full = config
        config_a,config_b = config_to_ab(config)
    return config_a,config_b,config_full
class AmplitudeFactory:
    def __init__(self,psi_a,psi_b,psi_boson,blks=None,flat=True):
        self.psi = [None] * 3
        self.psi[0] = FermionAmplitudeFactory(psi_a,blks=blks,subspace='a',flat=flat)
        self.psi[1] = FermionAmplitudeFactory(psi_b,blks=blks,subspace='b',flat=flat)
        self.psi[2] = BosonAmplitudeFactory(psi_boson,blks=blks,phys_dim=4)

        self.nparam = [len(amp_fac.get_x()) for amp_fac in self.psi] 
        self.block_dict = self.psi[0].block_dict.copy()
        self.block_dict += [(start+self.nparam[0],stop+self.nparam[0]) \
                           for start,stop in self.psi[1].block_dict]
        self.block_dict += [(start+self.nparam[1],stop+self.nparam[1]) \
                           for start,stop in self.psi[2].block_dict]
    def get_x(self):
        return np.concatenate([amp_fac.get_x() for amp_fac in self.psi])
    def update(self,x,fname=None,root=0):
        fname_ = fname + '_a' if fname is not None else fname
        self.psi[0].update(x[:self.nparam[0]],fname=fname_,root=root)

        fname_ = fname + '_b' if fname is not None else fname
        self.psi[1].update(x[self.nparam[0]:self.nparam[0]+self.nparam[1]],fname=fname_,root=root)

        fname_ = fname + '_boson' if fname is not None else fname
        self.psi[2].update(x[self.nparam[0]+self.nparam[1]:],fname=fname_,root=root)
    def unsigned_amplitude(self,config):
        configs = parse_config(config)
        cx = 1. 
        for ix in range(3):
            cx *= self.psi[ix].unsigned_amplitude(configs[ix])
        return cx
    def prob(self,config):
        return self.unsigned_amplitude(config) ** 2
####################################################################################
# ham class 
####################################################################################
from ..tensor_2d_vmc_ import Hamiltonian as Hamiltonian_
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
        return amplitude_factory._2numpy(ex_num),np.concatenate(Hvxs)
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
    def compute_local_energy_hessian_from_plq(self,config,amplitude_factory):
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
    def compute_local_energy_gradient_from_plq(self,config,amplitude_factory,compute_v=True):
        exs,cxs,plqs = [None] * 3,[None] * 3,[None] * 3
        configs = parse_config(config)
        for ix in range(3):
            exs[ix],cxs[ix],plqs[ix] = self.ham[ix].pair_energies_from_plq(
                                           configs[ix],amplitude_factory.psi[ix])
        ex = self.parse_energy(exs,cxs)
        eu = self.compute_local_energy_eigen(configs[ix])
        ex += eu

        if not compute_v:
            cx,err = self.contraction_error(cxs)
            return cx,ex,None,None,err 

        vx = np.concatenate([amplitude_factory.psi[ix].get_grad_from_plq(plqs[ix],cxs[ix]) for ix in range(3)])
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
        ex = self.parse_energy_numerator(exs)
        return self.parse_hessian(ex,wfns,amplitude_factory)
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
        ex = self.parse_energy_numerator(exs)
        return self.parse_hessian(ex,wfns,amplitude_factory)
    def compute_local_energy_gradient_deterministic(self,config,amplitude_factory,compute_v=True):
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
from ..tensor_core import Tensor
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
            ex = tn.contract()
            return self.pair_coeff(site1,site2) * ex 
        except (ValueError,IndexError):
            return None 
    def _pair_energies_from_plq(self,plq,pairs,config):
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

        self.pairs = self.pairs_nn()
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_plq_nn()
    def pair_coeff(self,site1,site2):
        return 1. 
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
        self.Lx,self.Ly = Lx,Ly
        self.t,self.u = t,u
        self.ham = [None] * 3 
        self.ham[0] = HubbardFermion(t,u,Lx,Ly,subspace='a',**kwargs)
        self.ham[1] = HubbardFermion(t,u,Lx,Ly,subspace='b',**kwargs)
        self.ham[2] = HubbardBoson(Lx,Ly,**kwargs)

        self.pbc = self.ham[0].pbc
        self.deterministic = self.ham[0].deterministic
    def pair_coeff(self,site1,site2):
        return -self.t
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
from .fermion_2d_vmc_ import ExchangeSampler as ExchangeSampler_
class ExchangeSampler(ExchangeSampler_):
    def update_pair(self,i,j,x_bsz,y_bsz,cols,tn):
        config_sites = [None] * 3
        sites,config_sites[2],config_new = self._new_pair(i,j,x_bsz,y_bsz)
        if config_sites[2] is None:
            return tn
        config_sites[0],config_sites[1] = config_to_ab(config_sites[2])

        py = 1.
        for ix in range(3):
            py_ = self._prob_from_plq(cols[ix],tn[ix],sites,config_sites[ix])
            if py_ is None:
                return tn
            py *= py_
                
        try:
            acceptance = py / self.px
        except ZeroDivisionError:
            acceptance = 1. if py > self.px else 0.
        if self.rng.uniform() < acceptance: # accept, update px & config & env_m
            #print('acc')
            self.px = py
            self.config = config_new
            for ix in range(3):
                tn[ix] = self.replace_sites(tn[ix],sites,configs[ix])
        return tn
    def sweep_col_forward(self,i,tn,x_bsz,y_bsz):
        self.config = list(self.config)
        renvs = [self.get_all_renvs(tn[ix].copy(),jmin=y_bsz) for ix in range(3)]

        first_col = [tn[ix].col_tag(0) for ix in range(3)]
        for j in range(self.Ly - y_bsz + 1): 
            cols = [self._get_cols_forward(first_col[ix],j,y_bsz,tn[ix],renvs[ix]) for ix in range(3)]
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            for ix in range(3):
                tn[ix] ^= first_col[ix],tn[ix].col_tag(j) 
        self.config = tuple(self.config)
    def sweep_col_backward(self,i,tn,x_bsz,y_bsz):
        self.config = list(self.config)
        lenvs = [self.get_all_lenvs(tn[ix].copy(),jmax=self.Ly-1-y_bsz) for ix in range(3)]

        last_col = [tn[ix].col_tag(self.Ly-1) for ix in range(3)]
        for j in range(self.Ly - y_bsz,-1,-1): # Ly-1,...,1
            cols = [self._get_cols_backward(last_col[ix],j,y_bsz,tn[ix],lenvs[ix]) for ix in range(3)] 
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            for ix in range(3):
                tn[ix] ^= tn[ix].col_tag(j+y_bsz-1),last_col[ix]
        self.config = tuple(self.config)
    def sweep_row_forward(self,x_bsz,y_bsz):
        config = parse_config(self.config)
        for ix in range(3): 
            psi = self.amplitude_factory.psi[ix]
            psi.cache_bot = dict()
            psi.get_all_top_envs(config[ix],imin=x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax+1):
            tn = [None] * 3
            for ix in range(3):
                psi = self.amplitude_factory.psi[ix]
                tn[ix] = psi.build_3row_tn(config[ix],i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)
            config = parse_config(self.config)

            for ix in range(3):
                psi = self.amplitude_factory.psi[ix]
                for inew in range(i,i+x_bsz):
                    row = psi.get_mid_env(inew,config[ix])
                    env_prev = None if inew==0 else psi.cache_bot[config[ix][:inew*self.Ly]] 
                    psi.get_bot_env(inew,row,env_prev,config[ix])
    def sweep_row_backward(self,x_bsz,y_bsz):
        config = parse_config(self.config)
        for ix in range(3):
            psi = self.amplitude_factory.psi[ix]
            psi.cache_top = dict()
            psi.get_all_bot_envs(config[ix],imax=self.Lx-1-x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax,-1,-1):
            tn = [None] * 3
            for ix in range(3):
                psi = self.amplitude_factory.psi[ix]
                tn[ix] = psi.build_3row_tn(config[ix],i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)
            config = parse_config(self.config)

            for ix in range(3):
                psi = self.amplitude_factory.psi[ix]
                for inew in range(i+x_bsz-1,i-1,-1):
                    row = psi.get_mid_env(inew,config[ix])
                    env_prev = None if inew==self.Lx-1 else psi.cache_top[config[ix][(inew+1)*self.Ly:]] 
                    psi.get_top_env(inew,row,env_prev,config[ix])
    def update_cache(self,amplitude_factory):
        for ix in range(3):
            super().update_cache(self,amplitude_factory.psi[ix])
    def _prob_deterministic(self,config_old,config_new,amplitude_factory,site1,site2):
        config_old_ = parse_config(config_old)
        config_new_ = parse_config(config_new)
        py = 1.
        for ix in range(3):
            py_ = self._prob_deterministic(config_old_[ix],config_new_[ix],amplitude_factory.psi[ix],
                                           site1,site2)
            if py_ is None:
                return None
            py *= py_
        return py
