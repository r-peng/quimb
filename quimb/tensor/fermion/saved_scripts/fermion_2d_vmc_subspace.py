import time,itertools
import numpy as np

#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import sys
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True,pbc=False,deterministic=False):
    this._PBC = pbc
    this._DETERMINISTIC = deterministic
    this._SYMMETRY = symmetry
    this._FLAT = flat 
    from .fermion_2d_vmc import set_options as set_options
    return set_options(symmetry=symmetry,flat=flat,pbc=pbc,deterministic=deterministic)

from ..tensor_2d_vmc import AmplitudeFactory2D,Hamiltonian2D 
from .fermion_2d_vmc import FermionAmplitudeFactory2D
from .fermion_product_vmc import (
    JastrowAmplitudeFactory,
    ProductAmplitudeFactory,
    ProductHamiltonian,
)
class JastrowAmplitudeFactory2D(JastrowAmplitudeFactory,AmplitudeFactory2D):
    def pair_energy_deterministic(self,config,site1,site2,model,cache_top=None,cache_bot=None):
        ix1,ix2 = model.flatten(site1),model.flatten(site2)
        i1,i2 = config[ix1],config[ix2]
        if not model.pair_valid(i1,i2): # term vanishes 
            return None 
        cx = [None] * 2 
        for ix,spin in zip((0,1),('a','b')):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue 
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)

            cx_new = self.unsigned_amplitude(config_new,cache_bot=cache_bot,cache_top=cache_top,to_numpy=False)
            if cx_new is not None:
                cx[ix] = cx_new
        return cx 
class ProductAmplitudeFactory2D(ProductAmplitudeFactory,AmplitudeFactory2D):
    def __init__(self,psi,blks=None,backend='numpy',**compress_opts):
        self.psi = [None] * 3
        self.psi[0] = FermionAmplitudeFactory2D(psi[0],blks=blks,spinless=True,backend=backend,**compress_opts)
        self.psi[1] = FermionAmplitudeFactory2D(psi[1],blks=blks,spinless=True,backend=backend,**compress_opts)
        self.psi[2] = JastrowAmplitudeFactory2D(psi[2],blks=blks,phys_dim=4,backend=backend,**compress_opts)

        self.nparam = [len(amp_fac.get_x()) for amp_fac in self.psi] 
        self.block_dict = self.psi[0].block_dict.copy()
        shift = 0
        for ix in (1,2):
            shift += self.nparam[ix-1]
            self.block_dict += [(start+shift,stop+shift) for start,stop in self.psi[ix].block_dict]

        self.Lx,self.Ly = self.psi[0].Lx,self.psi[0].Ly
        self.sites = self.psi[0].sites
        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        self.backend = backend
        self.spinless = False
        self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
##### wfn methods #####
    def update_cache(self,config):
        for ix in range(3):
            self.psi[ix].update_cache(config[ix])
##### compress row methods  #####
    def _get_all_benvs(self,config,step,psi=None,cache=None,start=None,stop=None,append='',direction='row'):
        env_prev = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_ = self.psi[ix].get_cache(direction,step) if cache is None else cache[ix]
            env_prev[ix] = self.psi[ix]._get_all_benvs(config[ix],step,psi=psi_,cache=cache_,start=start,stop=stop,append=append,direction=direction)
        return env_prev
    def get_all_benvs(self,config,psi=None,cache_bot=None,cache_top=None,x_bsz=1,
                      compute_bot=True,compute_top=True,imin=None,imax=None,direction='row'):
        env_top = [None] * 3
        env_bot = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_top_ = self.psi[ix].get_cache(direction,-1) if cache_top is None else cache_top[ix]
            cache_bot_ = self.psi[ix].get_cache(direction,1) if cache_bot is None else cache_bot[ix]
            env_bot[ix],env_top[ix] = self.psi[ix].get_all_benvs(config[ix],psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_,x_bsz=x_bsz,compute_bot=compute_bot,compute_top=compute_top,imax=imax,imin=imin,direction=direction)
        return env_bot,env_top
    def _contract_cols(self,cols,js,direction='col'):
        return [af._contract_cols(cols_,js,direction=direction) for af,cols_ in zip(self.psi,cols)]
    def get_all_envs(self,cols,step,stop=None,inplace=False,direction='col'):
        cols_new = [None] * 3
        envs = [None] * 3
        for ix in range(3):
            cols_new[ix],envs[ix] = self.psi[ix].get_all_envs(cols[ix],step,stop=stop,inplace=inplace)
        return cols_new,envs
    def _get_plq_forward(self,j,y_bsz,cols,renvs,direction='col'):
        return [af._get_plq_forward(j,y_bsz,cols_,renvs_,direction=direction) for af,cols_,renvs_ in zip(self.psi,cols,renvs)]
    def _get_plq_backward(self,j,y_bsz,cols,lenvs,direction='col'):
        return [af._get_plq_backward(j,y_bsz,cols_,lenvs_,direction=direction) for af,cols_,lenvs_ in zip(self.psi,cols,lenvs)]
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None,direction='row'):
        tn = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_bot_ = self.psi[ix].get_cache(direction,1) if cache_bot is None else cache_bot[ix]
            cache_top_ = self.psi[ix].get_cache(direction,-1) if cache_top is None else cache_top[ix]
            tn[ix] = self.psi[ix].build_3row_tn(config[ix],i,x_bsz,psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_,direction=direction) 
        return tn
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,cache_bot=None,cache_top=None,imin=0,imax=None,direction='row'):
        plq = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_bot_ = self.psi[ix].get_cache(direction,1) if cache_bot is None else cache_bot[ix]
            cache_top_ = self.psi[ix].get_cache(direction,-1) if cache_top is None else cache_top[ix]
            plq[ix] = self.psi[ix].get_plq_from_benvs(config[ix],x_bsz,y_bsz,psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_,imin=imin,imax=imax,direction=direction)
        return plq
    def unsigned_amplitude(self,config,cache_bot=None,cache_top=None,to_numpy=True):
        cx = [None] * 3 
        for ix in range(3):
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            cx[ix] = self.psi[ix].unsigned_amplitude(config[ix],cache_bot=cache_bot_,cache_top=cache_top_,to_numpy=to_numpy)
        #print(config[2],cx)
        #print(self.config_sign(config))
        #exit()
        return cx[0] * cx[1] * cx[2]
class ProductHamiltonian2D(ProductHamiltonian,Hamiltonian2D):
    def batch_pair_energies_from_plq(self,batch_key,config,new_cache=False,compute_v=True,to_vec=False): # only used for Hessian
        af = self.amplitude_factory 
        bix,tix,plq_types,pairs,direction = self.model.batched_pairs[batch_key]
        cache_bot = [dict(),dict(),dict()] if new_cache else None
        cache_top = [dict(),dict(),dict()] if new_cache else None
        af._get_all_benvs(config,1,cache=cache_bot,stop=bix+1,direction=direction)
        af._get_all_benvs(config,-1,cache=cache_top,stop=tix-1,direction=direction)

        # form plqs
        plq = [dict(),dict(),dict()]  
        for imin,imax,x_bsz,y_bsz in plq_types:
            plq_new = af.get_plq_from_benvs(config,x_bsz,y_bsz,cache_bot=cache_bot,cache_top=cache_top,imin=imin,imax=imax,direction=direction)
            for ix in range(3):
                plq[ix].update(plq_new[ix])

        # compute energy numerator 
        ex = [None] * 3
        cx = [None] * 3
        for ix in range(3):
            ex[ix],cx[ix] = self._pair_energies_from_plq(plq[ix],pairs,config[ix],af=af.psi[ix])
        if compute_v:
            vx = af.get_grad_from_plq(plq,to_vec=to_vec) 
        else:
            vx = None if to_vec else [dict(),dict(),dict()]
        return ex,cx,vx
    #def pair_energies_from_plq(self,config,direction='row'): 
    #    af = self.amplitude_factory 
    #    x_bsz_min = min([x_bsz for x_bsz,_ in self.model.plq_sz])
    #    af.get_all_benvs(config,x_bsz=x_bsz_min)

    #    plq = [dict(),dict(),dict()]  
    #    for x_bsz,y_bsz in self.model.plq_sz:
    #        plq_new = af.get_plq_from_benvs(config,x_bsz,y_bsz)
    #        for ix in range(3):
    #            plq[ix].update(plq_new[ix])

    #    # compute energy numerator 
    #    ex = [None] * 3
    #    cx = [None] * 3
    #    for ix in range(3):
    #        ex[ix],cx[ix] = self._pair_energies_from_plq(plq[ix],self.model.pairs,config[ix],af=af.psi[ix])
    #    return ex,cx,plq
    #def pair_energy_deterministic(self,config,site1,site2):
    #    af = self.amplitude_factory 
    #    ix1,ix2 = [self.model.flatten(*site) for site in (site1,site2)]
    #    i1,i2 = config[2][ix1],config[2][ix2]
    #    if not self.model.pair_valid(i1,i2): # term vanishes 
    #        return None 

    #    cache_bot = [dict(),dict(),dict()]
    #    cache_top = [dict(),dict(),dict()]
    #    imin = min(site1[0],site2[0])
    #    imax = max(site1[0],site2[0])
    #    bot,top = af._get_boundary_mps_deterministic(config,imin,imax,cache_bot=cache_bot,cache_top=cache_top)

    #    ex = [None] * 3
    #    for ix in range(3):
    #        ex_ix = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,top[ix],bot[ix],self.model,cache_bot=cache_bot[ix],cache_top=cache_top[ix])
    #        ex[ix] = {(site1,site2):ex_ix}
    #    return ex
    def batch_pair_energies_deterministic(self,config,batch_key,new_cache=False):
        af = self.amplitude_factory
        cache_bot = [dict(),dict(),dict()] if new_cache else None
        cache_top = [dict(),dict(),dict()] if new_cache else None

        ex = [None] * 3
        for ix in range(3):
            ex_ix = dict() 
            for site1,site2 in self.model.batched_pairs[batch_key]:
                eij = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,self.model,cache_bot=cache_bot[ix],cache_top=cache_top[ix])
                if eij is not None:
                    ex_ix[site1,site2] = eij
            ex[ix] = ex_ix
        return ex
    #def pair_energies_deterministic(self,config):
    #    af = self.amplitude_factory
    #    af.get_all_benvs(config)

    #    ex = [None] * 3
    #    for ix in range(3):
    #        ex_ix = dict() 
    #        for (site1,site2) in self.model.pairs:
    #            imin = min(af.rix1+1,site1[0],site2[0]) 
    #            imax = max(af.rix2-1,site1[0],site2[0]) 
    #            bot = af.psi[ix]._get_bot(imin,config[ix])  
    #            top = af.psi[ix]._get_top(imax,config[ix])  

    #            eij = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,top,bot,self.model)
    #            if eij is not None:
    #                ex_ix[site1,site2] = eij
    #        ex[ix] = ex_ix
    #    return ex
from ..tensor_2d import PEPS
def get_gutzwiller(Lx,Ly,coeffs,bdim=1,eps=0.,normalize=False):
    if isinstance(coeffs,np.ndarray):
        assert len(coeffs)==4
        coeffs = {(i,j):coeffs for i,j in itertools.product(range(Lx),range(Ly))}
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

            data = eps * np.random.rand(*shape)
            for ix in range(4):
                data[(0,)*(len(shape)-1)+(ix,)] = coeffs[i,j][ix] 
            if normalize:
                data /= np.linalg.norm(data)
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
