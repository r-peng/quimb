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
    return set_options2(symmetry=symmetry,flat=flat,pbc=pbc,deterministic=deterministic,**compress_opts)
from ..tensor_2d import PEPS
def get_gutzwiller(Lx,Ly,bdim=1,eps=None,g=1.):
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [bdim] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (4,)

            data = np.ones(shape)
            if eps is not None:
                data += eps * np.random.rand(*shape)
            data[...,3] = g * np.random.rand()
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)

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

        self.pbc = self.psi[0].pbc
        self.deterministic = self.psi[0].deterministic
        self.pair_valid = self.psi[0].pair_valid
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
        cx = np.ones(3) 
        for ix in range(3):
            cx[ix] = self.psi[ix].unsigned_amplitude(configs[ix])
        return np.prod(cx)
    def prob(self,config):
        return self.unsigned_amplitude(config) ** 2
####################################################################################
# ham class 
####################################################################################
from ..tensor_2d_vmc_ import Hamiltonian as Hamiltonian_
class Hamiltonian(Hamiltonian_):
    def parse_energy_from_plq(self,ex_num,cx):
        ex = np.zeros(2)
        cx2 = cx[2]
        np2 = self.ham[2]._2numpy
        for ix in range(2):
            ex2 = ex_num[2][ix]
            ex1,cx1 = ex_num[ix],cx[ix]
            np1 = self.ham[ix]._2numpy
            for where in ex1:
                if where not in ex2:
                    continue
                ex[ix] += np1(ex1[where]) * np2(ex2[where]) / (cx1[where] * cx2[where])
        return ex
    def parse_energy_numerator(self,ex_num):
        ex = [None] * 2 
        for ix in range(2):
            ex_ix = [] 
            ex1,ex2 = ex_num[ix],ex_num[2][ix]
            for site1,site2 in ex1:
                ex_ix.append(ex1[site1,site2] * ex2[site1,site2])
            if len(ex_ix)>0:
                ex[ix] = sum(ex_ix)
        return ex
    def propagate_hessian(self,ex_num,wfn,amplitude_factory):
        Hvx_fermion = [None] * 2 
        Hvx_boson = np.zeros((2,amplitude_factory.nparam[2])) 
        ex = np.zeros(2)
        for spin in range(2):
            if ex_num[spin] is None:
                continue
            ex_num[spin].backward(retain_graph=True)

            for ix in (spin,2):
                Hvx = dict()
                peps = wfn[ix]
                _2numpy = self.ham[ix]._2numpy
                tsr_grad = self.ham[ix].tsr_grad
                for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                    Hvx[i,j] = _2numpy(tsr_grad(peps[i,j].data))
                Hvx = amplitude_factory.psi[ix].dict2vec(Hvx)  
                if ix == 2:
                    Hvx_boson[spin,:] = Hvx
                else:
                    Hvx_fermion[spin] = Hvx

            ex[spin] = self.ham[spin]._2numpy(ex_num[spin])
        return ex,Hvx_fermion + [Hvx_boson]
    def parse_hessian(self,Hvx,vx,cx,ex,eu):
        ls = [0.] * 3
        for ix in range(2):
            denom = cx[ix] * cx[2]
            ls[ix] += Hvx[ix] / denom + ex[1-ix] * vx[ix]
            ls[2] += Hvx[2][ix,:] / denom
        Hvx = np.concatenate(ls)
        cx = np.prod(cx)
        ex = ex.sum() + eu

        vx = np.concatenate(vx)
        Hvx += eu*vx
        return Hvx,vx,cx,ex
    def contraction_error(self,cxs,multiply=True):
        cx,err = np.zeros(3),np.zeros(3)
        for ix in range(3): 
            cx[ix],err[ix] = self.ham[ix].contraction_error(cxs[ix])
        if multiply:
            cx = np.prod(cx)
        return cx,np.amax(err)
    def batch_hessian_from_plq(self,batch_idx,configs,amplitude_factory): # only used for Hessian
        ex,cx,plq,wfn = [None] * 3,[None] * 3,[None] * 3,[None] * 3
        for ix in range(3):
            self.ham[ix].backend = 'torch'
            peps = amplitude_factory.psi[ix].psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self.ham[ix]._2backend(peps[i,j].data,True))
            wfn[ix] = peps
            ex[ix],cx[ix],plq[ix] = self.ham[ix].batch_pair_energies_from_plq(batch_idx,configs[ix],peps)
        vx = [self.ham[ix].get_grad_dict_from_plq(plq[ix],cx[ix],backend=self.ham[ix].backend) for ix in range(3)]

        ex_num = self.parse_energy_numerator(ex)
        _,Hvx = self.propagate_hessian(ex_num,wfn,amplitude_factory)
        ex = self.parse_energy_from_plq(ex,cx)
        return ex,Hvx,cx,vx 
    def compute_local_energy_hessian_from_plq(self,config,amplitude_factory):
        ar.set_backend(torch.zeros(1))

        ex = 0.
        Hvx = [0.,0.,0.]
        cx,vx = [dict(),dict(),dict()],[dict(),dict(),dict()]
        configs = parse_config(config)
        for batch_idx in self.ham[0].batched_pairs:
            ex_,Hvx_,cx_,vx_ = self.batch_hessian_from_plq(batch_idx,configs,amplitude_factory)  
            ex += ex_
            for ix in range(3):
                Hvx[ix] += Hvx_[ix] 
                cx[ix].update(cx_[ix])
                vx[ix].update(vx_[ix])

        vx = [amplitude_factory.psi[ix].dict2vec(vx[ix]) for ix in range(3)]
        cx,err = self.contraction_error(cx,multiply=False)

        eu = self.compute_local_energy_eigen(config)
        Hvx,vx,cx,ex = self.parse_hessian(Hvx,vx,cx,ex,eu)
        ar.set_backend(np.zeros(1))
        #print(cx,ex)
        return cx,ex,vx,Hvx,err 
    def compute_local_energy_gradient_from_plq(self,config,amplitude_factory,compute_v=True):
        ex,cx,plq = [None] * 3,[None] * 3,[None] * 3
        configs = parse_config(config)
        for ix in range(3):
            ex[ix],cx[ix],plq[ix] = self.ham[ix].pair_energies_from_plq(
                                           configs[ix],amplitude_factory.psi[ix])
        ex = np.sum(self.parse_energy_from_plq(ex,cx))
        eu = self.compute_local_energy_eigen(config)
        ex += eu

        if not compute_v:
            cx,err = self.contraction_error(cx)
            return cx,ex,None,None,err 

        vx = np.concatenate([amplitude_factory.psi[ix].get_grad_from_plq(plq[ix],cx[ix]) for ix in range(3)])
        cx,err = self.contraction_error(cx)
        return cx,ex,vx,None,err
    def amplitude_gradient_deterministic(self,configs,amplitude_factory,multiply=True):
        cx,vx = np.zeros(3),[None] * 3 
        for ix in range(3):
            cx[ix],vx[ix] = self.ham[ix].amplitude_gradient_deterministic(
                                  configs[ix],amplitude_factory.psi[ix])
        if multiply:
            cx = np.prod(cx)
            vx = np.concatenate(vx)
        return cx,vx
    def batch_hessian_deterministic(self,configs,amplitude_factory,imin,imax):
        ex,wfn = [None] * 3,[None] * 3
        for ix in range(3):
            self.ham[ix].backend = 'torch'
            psi = amplitude_factory.psi[ix]
            peps = psi.psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self.ham[ix]._2backend(peps[i,j].data,True))
            wfn[ix] = peps
            ex[ix] = self.ham[ix].batch_pair_energies_deterministic(configs[ix],peps,imin,imax,
                                                                    psi.config_sign)
        ex = self.parse_energy_numerator(ex)
        return self.propagate_hessian(ex,wfn,amplitude_factory)
    def pair_hessian_deterministic(self,configs,amplitude_factory,site1,site2):
        ex,wfn = [None] * 3,[None] * 3
        for ix in range(3):
            self.ham[ix].backend = 'torch'
            psi = amplitude_factory.psi[ix]
            peps = psi.psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self.ham[ix]._2backend(peps[i,j].data,True))
            wfn[ix] = peps
            ex = self.ham[ix].pair_energy_deterministic(configs[ix],peps,site1,site2,psi.config_sign)
            if ex is None:
                return 0.,0.
            ex[ix] = ex 
        ex = [ex[ix]*ex[2][ix] for ix in range(2)]
        return self.propagate_hessian(ex,wfn,amplitude_factory)
    def compute_local_energy_hessian_deterministic(self,config,amplitude_factory):
        ar.set_backend(torch.zeros(1))

        configs = parse_config(config)
        cx,vx = self.amplitude_gradient_deterministic(configs,amplitude_factory,multiply=False)
        for ix in range(2):
            cx[ix] *= amplitude_factory.psi[ix].config_sign(configs[ix])

        ex = 0. 
        Hvx = [0.,0.,0.]
        for key in self.ham[0].batched_pairs:
            if key=='pbc':
                continue
            imin,imax = key
            ex_,Hvx_ = self.batch_hessian_deterministic(configs,amplitude_factory,imin,imax) 
            ex += ex_
            for ix in range(3):
                Hvx[ix] += Hvx_[ix]
        if self.pbc:
            for site1,site2 in self.ham[0].batched_pairs['pbc']:
                ex_,Hvx_ = self.pair_hessian_deterministic(configs,amplitude_factory,site1,site2)
                ex += ex_
                for ix in range(3):
                    Hvx[ix] += Hvx_[ix]
         
        ex = [ex[ix] / (cx[ix]*cx[2]) for ix in range(2)]
        eu = self.compute_local_energy_eigen(config)
        Hvx,vx,cx,ex = self.parse_hessian(Hvx,vx,cx,ex,eu)
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,0. 
    def compute_local_energy_gradient_deterministic(self,config,amplitude_factory,compute_v=True):
        configs = parse_config(config)
        ex,cx = [None] * 3,np.zeros(3)
        for ix in range(3):
            config,psi = configs[ix],amplitude_factory.psi[ix]
            ex[ix],cx[ix] = self.ham[ix].pair_energies_deterministic(config,psi)
            cx[ix] *= psi.config_sign(config) 
        ex = self.parse_energy_numerator(ex)
        ex = sum([ex[ix] / (cx[ix]*cx[2]) for ix in range(2)])
        eu = self.compute_local_energy_eigen(config)
        ex += eu
        cx = np.prod(cx)
        if not compute_v:
            return cx,ex,None,None,0.

        ar.set_backend(torch.zeros(1))
        _,vx = self.amplitude_gradient_deterministic(configs,amplitude_factory)
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,None,0.
class BosonHamiltonian(Hamiltonian_):
    def pair_terms(self,i1,i2,spin):
        if spin=='a':
            map_ = {(0,1):(1,0),(1,0):(0,1),
                    (2,3):(3,2),(3,2):(2,3),
                    (0,3):(1,2),(3,0):(2,1),
                    (1,2):(0,3),(2,1):(3,0)}
        elif spin=='b':
            map_ = {(0,2):(2,0),(2,0):(0,2),
                    (1,3):(3,1),(3,1):(1,3),
                    (0,3):(2,1),(3,0):(1,2),
                    (1,2):(3,0),(2,1):(0,3)}
        else:
            raise ValueError
        return map_.get((i1,i2),(None,)*2)
    def _pair_energy_from_plq(self,tn,config,where):
        ix1,ix2 = self.flatten(*where[0]),self.flatten(*where[1])
        i1,i2 = config[ix1],config[ix2] 
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        cx = [None] * 2
        for ix,spin in zip((0,1),('a','b')):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue
            tn_new = self.replace_sites(tn.copy(),where,(i1_new,i2_new))
            cx_new = self.safe_contract(tn_new)
            if cx_new is not None:
                cx[ix] = cx_new 
        return cx 
    def _pair_energies_from_plq(self,plq,pairs,config):
        exa = dict()
        exb = dict()
        cx = dict()
        for where in pairs:
            key = self.pair_key(*where)

            tn = plq.get(key,None) 
            if tn is not None:
                cij = self._2numpy(tn.copy().contract())
                cx[where] = cij 

                eij = self._pair_energy_from_plq(tn,config,where) 
                if eij is None:
                    continue
                if eij[0] is not None:
                    exa[where] = eij[0]
                if eij[1] is not None:
                    exb[where] = eij[1]
        return (exa,exb),cx
    def _pair_energy_deterministic(self,config,site1,site2,psi,top,bot):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        imin = min(self.rix1+1,site1[0],site2[0]) 
        imax = max(self.rix2-1,site1[0],site2[0]) 
        cx = [None] * 2 
        cache_top = dict()
        cache_bot = dict()
        for ix,spin in zip((0,1),('a','b')):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue 
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)

            bot_term = None if bot is None else bot.copy()
            for i in range(imin,self.rix1+1):
                row = self.get_mid_env(i,config_new,psi=psi)
                bot_term = self.get_bot_env(i,row,bot_term,config_new,cache=cache_bot)
            if bot_term is None:
                continue

            top_term = None if top is None else top.copy()
            for i in range(imax,self.rix2-1,-1):
                row = self.get_mid_env(i,config_new,psi=psi)
                top_term = self.get_top_env(i,row,top_term,config_new,cache=cache_top)
            if top_term is None:
                continue

            tn = bot_term.copy()
            tn.add_tensor_network(top_term,virtual=False)
            cx_new = self.safe_contract(tn)
            if cx_new is not None:
                cx[ix] = cx_new
        return cx 
    def batch_pair_energies_deterministic(self,config,psi,batch_imin,batch_imax,sign_fn=None):
        cache_top = dict()
        cache_bot = dict()
        
        imin = min(self.rix1+1,batch_imin) 
        imax = max(self.rix2-1,batch_imax) 
        self.get_all_bot_envs(config,psi=psi,cache=cache_bot,imax=imin-1)
        self.get_all_top_envs(config,psi=psi,cache=cache_top,imin=imax+1)
        top = None if imax==self.Lx-1 else cache_top[config[(imax+1)*self.Ly:]]
        bot = None if imin==0 else cache_bot[config[:imin*self.Ly]]

        exa,exb = dict(),dict()
        for site1,site2 in self.batched_pairs[batch_imin,batch_imax]:
            eij = self._pair_energy_deterministic(config,site1,site2,psi,top,bot)
            if eij is None:
                continue 
            if eij[0] is not None:
                exa[site1,site2] = eij[0]
            if eij[1] is not None:
                exb[site1,site2] = eij[1]
        return exa,exb
    def pair_energies_deterministic(self,config,amplitude_factory):
        self.backend = 'numpy'
        psi = amplitude_factory.psi
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top
        env_bot,env_top = amplitude_factory.get_all_benvs(config)

        tn = env_bot.copy()
        tn.add_tensor_network(env_top,virtual=False)
        cx = tn.contract() 

        exa = dict() 
        exb = dict() 
        for (site1,site2) in self.pairs:
            imin = min(self.rix1+1,site1[0],site2[0]) 
            imax = max(self.rix2-1,site1[0],site2[0]) 
            top = None if imax==self.Lx-1 else cache_top[config[(imax+1)*self.Ly:]]
            bot = None if imin==0 else cache_bot[config[:imin*self.Ly]]

            eij = self._pair_energy_deterministic(config,site1,site2,psi,top,bot)
            if eij is None:
                continue 
            if eij[0] is not None:
                exa[site1,site2] = eij[0]
            if eij[1] is not None:
                exb[site1,site2] = eij[1]
        return (exa,exb),cx
class HubbardBoson(BosonHamiltonian):
    def __init__(self,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,phys_dim=4)
        self.pairs = self.pairs_nn()
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_plq_nn()
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
    def new_pair(self,i1,i2):
        return super().new_pair_full(i1,i2)
    def update_pair(self,i,j,x_bsz,y_bsz,cols,tn):
        sites,config_sites,config_new = self._new_pair(i,j,x_bsz,y_bsz)
        if config_sites is None:
            return tn

        configs = [None,None,config_sites]
        configs[0],configs[1] = config_to_ab(config_sites)
        py = 1.
        for ix in range(3):
            psi = self.amplitude_factory.psi[ix]
            cols_ix = psi.replace_sites(cols[ix],sites,configs[ix]) 
            py_ix = psi.safe_contract(cols_ix)
            if py_ix is None:
                return tn 
            py *= py_ix ** 2

        try:
            acceptance = py / self.px
        except ZeroDivisionError:
            acceptance = 1. if py > self.px else 0.
        if self.rng.uniform() < acceptance: # accept, update px & config & env_m
            self.px = py
            self.config = config_new
            for ix in range(3):
                psi = self.amplitude_factory.psi[ix]
                tn[ix] = psi.replace_sites(tn[ix],sites,configs[ix])
        return tn
    def sweep_col_forward(self,i,tn,x_bsz,y_bsz):
        renvs = [None] * 3
        for ix in range(3):
            try:
                tn[ix].reorder('col',inplace=True)
            except (NotImplementedError,AttributeError):
                pass
            renvs[ix] = self.amplitude_factory.psi[ix].get_all_renvs(tn[ix].copy(),jmin=y_bsz)

        first_col = [tn[ix].col_tag(0) for ix in range(3)]
        for j in range(self.Ly - y_bsz + 1): 
            cols = [self._get_cols_forward(first_col[ix],j,y_bsz,tn[ix],renvs[ix]) for ix in range(3)]
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            for ix in range(3):
                tn[ix] ^= first_col[ix],tn[ix].col_tag(j) 
    def sweep_col_backward(self,i,tn,x_bsz,y_bsz):
        lenvs = [None] * 3
        for ix in range(3):
            try:
                tn[ix].reorder('col',inplace=True)
            except (NotImplementedError,AttributeError):
                pass
            lenvs[ix] = self.amplitude_factory.psi[ix].get_all_lenvs(tn[ix].copy(),jmax=self.Ly-1-y_bsz)

        last_col = [tn[ix].col_tag(self.Ly-1) for ix in range(3)]
        for j in range(self.Ly - y_bsz,-1,-1): # Ly-1,...,1
            cols = [self._get_cols_backward(last_col[ix],j,y_bsz,tn[ix],lenvs[ix]) for ix in range(3)] 
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            for ix in range(3):
                tn[ix] ^= tn[ix].col_tag(j+y_bsz-1),last_col[ix]
    def sweep_row_forward(self,x_bsz,y_bsz):
        config = parse_config(self.config)
        for ix in range(3): 
            psi = self.amplitude_factory.psi[ix]
            psi.cache_bot = dict()
            psi.get_all_top_envs(config[ix],imin=x_bsz)

        #cdir = self.rng.choice([-1,1]) 
        cdir = 1
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

        #cdir = self.rng.choice([-1,1]) 
        cdir = 1
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
    def update_cache(self,amplitude_factory,config):
        configs = parse_config(config)
        for ix in range(3):
            super().update_cache(amplitude_factory.psi[ix],configs[ix])
    def _prob_deterministic(self,config_old,config_new,amplitude_factory,site1,site2):
        config_old_ = parse_config(config_old)
        config_new_ = parse_config(config_new)
        py = 1.
        for ix in range(3):
            py_ = super()._prob_deterministic(config_old_[ix],config_new_[ix],amplitude_factory.psi[ix],
                                           site1,site2)
            if py_ is None:
                return None
            py *= py_
        return py
