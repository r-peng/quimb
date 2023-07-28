import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import sys
this = sys.modules[__name__]
def set_options(pbc=False,deterministic=False,symmetry='u1',flat=True):
    this._PBC = pbc
    this._DETERMINISTIC = deterministic
    this._SYMMETRY = symmetry
    this._FLAT = flat 
    from ..tensor_2d_vmc import set_options
    set_options(pbc=pbc,deterministic=deterministic)
from ..tensor_2d_vmc import flatten,flat2site 
from .fermion_core import FermionTensor,FermionTensorNetwork
####################################################################################
# amplitude fxns 
####################################################################################
from ..tensor_2d_vmc import AmplitudeFactory as AmplitudeFactory2D
from .fermion_vmc import AmplitudeFactory as FermionAmplitudeFactory
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
class AmplitudeFactory(FermionAmplitudeFactory,AmplitudeFactory2D): 
    def __init__(self,psi,blks=None,spinless=False,backend='numpy',**compress_opts):
        # init wfn
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.nsite = self.Lx * self.Ly
        self.sites = list(itertools.product(range(self.Lx),range(self.Ly)))
        psi.add_tag('KET')
        psi.reorder(direction='row',inplace=True)
        self.set_psi(psi) # current state stored in self.psi
        self.backend = backend

        self.symmetry = _SYMMETRY
        self.flat = _FLAT
        self.spinless = spinless
        self.data_map = self.get_data_map()
        self.wfn2backend()

        # init contraction
        self.compress_opts = compress_opts
        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        if self.deterministic:
            self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2

        self.parity_cum = get_parity_cum(psi)

        if blks is None:
            blks = [list(itertools.product(range(self.Lx),range(self.Ly)))]
        self.site_map = self.get_site_map(blks)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict(blks)
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)
        self.nparam = len(self.get_x())

    def config_sign(self,config):
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum(self.config2pn(config,i*self.Ly,(i+1)*self.Ly)) % 2
        parity = np.array(parity[::-1])
        parity_cum = np.cumsum(parity[:-1])
        parity_cum += self.parity_cum 
        return (-1)**(np.dot(parity[1:],parity_cum) % 2)
    def get_all_lenvs(self,cols,jmax=None,inplace=False):
        cols.reorder('col',inplace=True)
        return super().get_all_lenvs(cols,jmax=jmax,inplace=inplace)
    def get_all_renvs(self,cols,jmin=None,inplace=False):
        cols.reorder('col',inplace=True)
        return super().get_all_renvs(cols,jmin=jmin,inplace=inplace)

from pyblock3.algebra.fermion_ops import H1
from ..tensor_2d_vmc import Model as Model2D
from .fermion_vmc import Model as FermionModel
from .fermion_vmc import pn_map
class Model(FermionModel,Model2D):
    pass 
class Hubbard(Model):
    def __init__(self,t,u,Lx,Ly,spinless=False,nbatch=1):
        super().__init__(Lx,Ly,nbatch=nbatch)
        self.t,self.u = t,u
        self.gate = H1(symmetry=_SYMMETRY,flat=_FLAT,spinless=spinless)
        self.order = 'b1,b2,k1,k2'
        self.spinless = spinless

        self.pairs = self.pairs_nn()
        if _DETERMINISTIC:
            self.batched_pairs = dict()
            self.batch_deterministic_nnh() 
            self.batch_deterministic_nnv() 
        else:
            self.batch_plq_nn()
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def pair_coeff(self,site1,site2):
        return -self.t
    def compute_local_energy_eigen(self,config):
        config = np.array(config,dtype=int)
        return self.u*len(config[config==3])
    def pair_terms(self,i1,i2):
        if self.spinless:
            return [(i2,i1,1)]
        else:
            return self.pair_terms_full(i1,i2)
    def pair_terms_full(self,i1,i2):
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
class FDKineticO2(Hubbard):
    def __init__(self,N,L,**kwargs):
        self.eps = L/(N+1e-15) if pbc else L/(N+1.)

        t = .5 / self.eps**2
        self.l = 2. / self.eps**2
        super().__init__(t,0.,N,N,**kwargs)
    def compute_local_energy_eigen(self,config):
        return self.l * sum(self.config2pn(config,0,len(config)))
    def get_h(self):
        nsite = self.Lx * self.Ly
        h = np.zeros((nsite,)*2)
        for i in range(self.Lx):
            for j in range(self.Ly):
                ix1 = self.flatten(i,j)
                h[ix1,ix1] = self.l
                if i+1<self.Lx or self.pbc:
                    ix2 = self.flatten((i+1)%self.Lx,j) 
                    h[ix1,ix2] = -self.t
                    h[ix2,ix1] = -self.t
                if j+1<self.Ly or self.pbc:
                    ix2 = self.flatten(i,(j+1)%self.Ly) 
                    h[ix1,ix2] = -self.t
                    h[ix2,ix1] = -self.t
        return h
class FDKineticO4(Model):
    def __init__(self,N,L,**kwargs):
        super().__init__(N,N,**kwargs)
        self.eps = L/(N+1e-15) if pbc else L/(N+1.)
        self.t1 = 16./(24.*self.eps**2)
        self.t2 = 1./(24.*self.eps**2)
        self.li = 60./(24.*self.eps**2) 
        self.lb = 59./(24.*self.eps**2) 
        self.lc = 58./(24.*self.eps**2) 
    
        self.pairs = self.pairs_nn() + self.pairs_nn(d=2)
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_plq_nn(d=2)
    def batch_deterministic(self):
        self.batched_pairs = dict()
        self.batch_nnh() 
        self.batch_nnh(d=2) 
        self.batch_nnv() 
        self.batch_nnv(d=2) 
    def pair_key(self,site1,site2):
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        if dx==0:
            j0 = min(site1[1],site2[1],self.Ly-3)
            return (site1[0],j0),(1,3)
        elif dy==0:
            i0 = min(site1[0],site2[0],self.Lx-3)
            return (i0,site1[1]),(3,1)
        else:
            raise ValueError(f'site1={site1},site2={site2}')
    def pair_coeff(self,site1,site2):
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        if dx+dy == 1:
            return -self.t1
        elif dx+dy == 2:
            return self.t2
        else:
            raise ValueError(f'site1={site1},site2={site2}')
    def compute_local_energy_eigen(self,config):
        pn = self.config2pn(config,0,len(config))
        if self.pbc:
            return self.li * sum(pn)
        # interior
        ei = sum([pn[self.flatten(i,j)] for i in range(1,self.Lx-1) for j in range(1,self.Ly-1)])
        # border
        eb = sum([pn[self.flatten(i,j)] for i in (0,self.Lx-1) for j in range(1,self.Ly-1)]) \
           + sum([pn[self.flatten(i,j)] for i in range(1,self.Lx-1) for j in (0,self.Ly-1)])
        # corner
        ec = pn[self.flatten(0,0)] + pn[self.flatten(0,self.Ly-1)] \
           + pn[self.flatten(self.Ly-1,0)] + pn[self.flatten(self.Lx-1,self.Ly-1)]
        return self.li * ei + self.lb * eb + self.lc * ec
    def get_h(self):
        nsite = self.Lx * self.Ly
        h = np.zeros((nsite,)*2)
        for i in range(self.Lx):
             for j in range(self.Ly):
                 ix1 = self.flatten(i,j)
                 if i+1<self.Lx or self.pbc:
                     ix2 = self.flatten((i+1)%self.Lx,j) 
                     h[ix1,ix2] = -self.t1
                     h[ix2,ix1] = -self.t1
                 if j+1<self.Ly or self.pbc:
                     ix2 = self.flatten(i,(j+1)%self.Ly) 
                     h[ix1,ix2] = -self.t1
                     h[ix2,ix1] = -self.t1
                 if i+2<self.Lx or self.pbc:
                     ix2 = self.flatten((i+2)%self.Lx,j) 
                     h[ix1,ix2] = self.t2
                     h[ix2,ix1] = self.t2
                 if j+2<self.Ly or self.pbc:
                     ix2 = self.flatten(i,(j+2)%self.Ly) 
                     h[ix1,ix2] = self.t2
                     h[ix2,ix1] = self.t2
                 if self.pbc:
                     h[ix1,ix1] = self.li
                 else:
                     if i in (0,self.Lx-1) and j in (0,self.Ly-1):
                         h[ix1,ix1] = self.lc
                     elif i in (0,self.Lx-1) or j in (0,self.Ly-1):
                         h[ix1,ix1] = self.lb
                     else:
                         h[ix1,ix1] = self.li
        return h
def coulomb(config,Lx,Ly,eps,spinless):
    pn = config2pn(config,0,len(config),spinless)
    e = 0.
    for ix1 in range(len(config)):
        n1 = pn[ix1]
        if n1==0:
            continue
        i1,j1 = flat2site(ix1,Lx,Ly)
        for ix2 in range(ix1+1,len(config)):
            n2 = pn[ix2]
            if n2==0:
                continue
            i2,j2 = flat2site(ix2,Lx,Ly)
            dist = np.sqrt((i1-i2)**2+(j1-j2)**2)
            e += n1 * n2 / dist    
    return e / eps
class UEGO2(FDKineticO2):
    def compute_local_energy_eigen(self,config):
        ke = super().compute_local_energy_eigen(config)
        v = coulomb(config,self.Lx,self.Ly,self.eps,self.spinless)
class UEGO4(FDKineticO4):
    def compute_local_energy_eigen(self,config):
        ke = super().compute_local_energy_eigen(config)
        v = coulomb(config,self.Lx,self.Ly,self.eps,self.spinless)
class DensityMatrix:
    def __init__(self,Lx,Ly,spinless=False):
        self.Lx,self.Ly = Lx,Ly 
        self.pairs = [] 
        self.data = np.zeros((Lx,Ly))
        self.n = 0.
        self.spinless = spinless
    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
        self.n += 1.
        pn = self.config2pn(config,0,len(config)) 
        for i in range(self.Lx):
            for j in range(self.Ly):
                self.data[i,j] += pn[self.flatten(i,j)]
        return 0.,0.,None,None,0. 
    def _print(self,fname,data):
        print(data)
####################################################################################
# sampler 
####################################################################################
from ..tensor_2d_vmc import ExchangeSampler as ExchangeSampler2D
from .fermion_vmc import ExchangeSampler as FermionExchangeSampler
class ExchangeSampler(FermionExchangeSampler,ExchangeSampler2D):
    pass
pattern_map = {'u':'+','r':'+','d':'-','l':'-','p':'+'}
def get_patterns(Lx,Ly,shape='urdlp'):
    patterns = dict()
    for i,j in itertools.product(range(Lx),range(Ly)):
        arr_order = shape
        if i==Lx-1:
            arr_order = arr_order.replace('u','')
        if j==Ly-1:
            arr_order = arr_order.replace('r','')
        if i==0:
            arr_order = arr_order.replace('d','')
        if j==0:
            arr_order = arr_order.replace('l','')
        patterns[i,j] = ''.join([pattern_map[c] for c in arr_order])
    return patterns
def get_vaccum(Lx,Ly,shape='urdlp',symmetry='u1',flat=True,spinless=False):
    from pyblock3.algebra.fermion_ops import bonded_vaccum
    from ..tensor_2d import PEPS
    from .fermion_2d import FPEPS
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=1,shape=shape)
    patterns = get_patterns(Lx,Ly,shape=shape)
    ftn = FermionTensorNetwork([])
    for ix, iy in itertools.product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        data = bonded_vaccum((1,)*(T.ndim-1), pattern=patterns[ix,iy],
                             symmetry=symmetry,flat=flat,spinless=spinless)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn
def create_particle(fpeps,site,spin,spinless=False):
    if spinless:
        cre = data_map['cre'].copy()
    else:
        cre = data_map[f'cre_{spin}'].copy()
    T = fpeps[fpeps.site_tag(*site)]
    trans_order = list(range(1,T.ndim))+[0] 
    data = np.tensordot(cre, T.data, axes=((1,), (-1,))).transpose(trans_order)
    T.modify(data=data)
    return fpeps
def get_product_state(Lx,Ly,config,symmetry='u1',flat=True,spinless=False):
    fpeps = get_vaccum(Lx,Ly,symmetry=symmetry,flat=flat,spinless=spinless)
    if spinless: 
        for ix,ci in enumerate(config):
            site = flat2site(ix,Ly)
            if ci==1:
                fpeps = create_particle(fpeps,site,None,spinless=spinless)
    else:
        for ix,ci in enumerate(config):
            site = flat2site(ix,Ly)
            if ci==1:
                fpeps = create_particle(fpeps,site,'a',spinless=spinless)
            if ci==2:
                fpeps = create_particle(fpeps,site,'b',spinless=spinless)
            if ci==3:
                fpeps = create_particle(fpeps,site,'b',spinless=spinless)
                fpeps = create_particle(fpeps,site,'a',spinless=spinless)
    return fpeps
