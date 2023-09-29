import time,itertools
import numpy as np

#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()

import sys
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True,pbc=False,deterministic=False):
    this._PBC = pbc
    this._DETERMINISTIC = deterministic
    this._SYMMETRY = symmetry
    this._FLAT = flat 
    from .fermion_2d_vmc import set_options as set_options
    return set_options(symmetry=symmetry,flat=flat,pbc=pbc,deterministic=deterministic)

from .product_vmc import (
    TNJastrow,
    RBM,FNN,
    ProductAmplitudeFactory,
)
#class ProductAmplitudeFactory2D(ProductAmplitudeFactory,AmplitudeFactory2D):
class ProductAmplitudeFactory2D(ProductAmplitudeFactory):
    def __init__(self,af):
        self.af = af 
        self.get_sections()

        self.Lx,self.Ly = self.af[0].Lx,self.af[0].Ly
        self.sites = self.af[0].sites
        self.model = self.af[0].model
        self.nsite = self.af[0].nsite
        self.backend = self.af[0].backend

        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        self.spinless = False
        self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
##### wfn methods #####
    def update_cache(self,config):
        for af,config_ in zip(self.af,config):
            af.update_cache(config_)
##### compress row methods  #####
    def _get_all_benvs(self,config,step,psi=None,cache=None,start=None,stop=None,append='',direction='row'):
        env_prev = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            cache_ = af.get_cache(direction,step) if cache is None else cache[ix]
            env_prev[ix] = af._get_all_benvs(config[ix],step,psi=psi_,cache=cache_,
                               start=start,stop=stop,append=append,direction=direction)
        return env_prev
    #def get_all_benvs(self,config,psi=None,cache_bot=None,cache_top=None,x_bsz=1,
    #                  compute_bot=True,compute_top=True,imin=None,imax=None,direction='row'):
    #    env_top = [None] * self.naf 
    #    env_bot = [None] * self.naf 
    #    for ix,af in enumerate(self.af):
    #        psi_ = af.psi if psi is None else psi[ix]
    #        cache_top_ = af.get_cache(direction,-1) if cache_top is None else cache_top[ix]
    #        cache_bot_ = af.get_cache(direction,1) if cache_bot is None else cache_bot[ix]
    #        env_bot[ix],env_top[ix] = af.get_all_benvs(
    #            config[ix],psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_,
    #            x_bsz=x_bsz,compute_bot=compute_bot,compute_top=compute_top,
    #            imax=imax,imin=imin,direction=direction)
    #    return env_bot,env_top
    def _contract_cols(self,cols,js,direction='col'):
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            cols[ix] = af._contract_cols(cols[ix],js,direction=direction)
        return cols 
    def get_all_envs(self,cols,step,stop=None,inplace=False,direction='col'):
        cols_new = [None] * self.naf 
        envs = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            cols_new[ix],envs[ix] = af.get_all_envs(cols[ix],step,stop=stop,inplace=inplace)
        return cols_new,envs
    def _get_plq_forward(self,j,y_bsz,cols,renvs,direction='col'):
        plq = [None] * 3
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            plq[ix] = af._get_plq_forward(j,y_bsz,cols[ix],renvs[ix],direction=direction)
        return plq
    def _get_plq_backward(self,j,y_bsz,cols,lenvs,direction='col'):
        plq = [None] * 3
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            plq[ix] = af._get_plq_backward(j,y_bsz,cols[ix],lenvs[ix],direction=direction)
        return plq 
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None,direction='row'):
        tn = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            psi_ = af.psi if psi is None else psi[ix]
            cache_bot_ = af.get_cache(direction,1) if cache_bot is None else cache_bot[ix]
            cache_top_ = af.get_cache(direction,-1) if cache_top is None else cache_top[ix]
            tn[ix] = af.build_3row_tn(config[ix],i,x_bsz,psi=psi_,
                         cache_bot=cache_bot_,cache_top=cache_top_,direction=direction) 
        return tn
    #def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,cache_bot=None,cache_top=None,imin=0,imax=None,direction='row'):
    #    plq = [None] * self.naf 
    #    for ix,af in enumerate(self.af):
    #        psi_ = af.psi if psi is None else psi[ix]
    #        cache_bot_ = af.get_cache(direction,1) if cache_bot is None else cache_bot[ix]
    #        cache_top_ = af.get_cache(direction,-1) if cache_top is None else cache_top[ix]
    #        plq[ix] = af.get_plq_from_benvs(config[ix],x_bsz,y_bsz,psi=psi_,
    #            cache_bot=cache_bot_,cache_top=cache_top_,imin=imin,imax=imax,direction=direction)
    #    return plq
    def unsigned_amplitude(self,config,cache_bot=None,cache_top=None,to_numpy=True):
        cx = np.zeros(self.naf) 
        for ix,af in enumerate(self.af):
            cache_top_ = af.cache_top if cache_top is None else cache_top[ix]
            cache_bot_ = af.cache_bot if cache_bot is None else cache_bot[ix]
            cx[ix] = af.unsigned_amplitude(config[ix],
                         cache_bot=cache_bot_,cache_top=cache_top_,to_numpy=to_numpy)
        #print(config[2],cx)
        #print(self.config_sign(config))
        #exit()
        return np.prod(cx)
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
    #def batch_pair_energies_deterministic(self,config,batch_key,new_cache=False):
    #    af = self.amplitude_factory
    #    cache_bot = [dict(),dict(),dict()] if new_cache else None
    #    cache_top = [dict(),dict(),dict()] if new_cache else None

    #    ex = [None] * 3
    #    for ix in range(3):
    #        ex_ix = dict() 
    #        for site1,site2 in self.model.batched_pairs[batch_key]:
    #            eij = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,self.model,cache_bot=cache_bot[ix],cache_top=cache_top[ix])
    #            if eij is not None:
    #                ex_ix[site1,site2] = eij
    #        ex[ix] = ex_ix
    #    return ex
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
from ..tensor_2d_vmc import AmplitudeFactory2D 
class PEPSJastrow(TNJastrow,AmplitudeFactory2D):
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
class RBM2D(RBM,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nv,nh,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nv,nh,**kwargs)
class FNN2D(FNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,nl,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(nl,**kwargs)
class CNN2D(FNN2D):
    def __init__(self,Lx,Ly,nl,kx,ky,**kwargs):
        self.kx,self.ky = kx,ky 
        #self.blks = []
        #for i,j in itertools.product(range(Lx-kx),range(Ly-ky)):
        #    self.blks.append(list(itertools.product(range(i,i+kx),range(j,j+ky))))
        #self.get_site_map(self.blks)
        super().__init__(Lx,Ly,nl,**kwargs) 
    def log_amplitude(self,config,to_numpy=True):
        c = np.array(config,dtype=float).reshape((self.Lx,self.Ly)) 
        jnp,c = self.get_backend(c=c)
        for i in range(self.nl-1):
            c = self.convolve(c,self.w[i],jnp) + self.b[i]
            c = jnp.log(jnp.cosh(c)) 
        c = jnp.dot(c,self.w[-1])
        #exit()
        if to_numpy:
            c = tensor2backend(c,'numpy') 
        return c,0
    def convolve(self,x,w,jnp):
        v = jnp.zeros((self.Lx,self.Ly),requires_grad=True) 
        for i,j in itertools.product(range(self.Lx-kx),range(self.Ly-ky)):
            v[i:i+kx,j:j+ky] += jnp.matmul(w[i,j,:,:],x[i:i+kx,j:j+ky].flatten()).reshape((kx,ky))
        return v 
