import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from ..tensor_2d_vmc import (
    flatten,flat2site,
    AmplitudeFactory2D,
    Model2D,
    ExchangeSampler2D,
)
from .fermion_vmc import (
    FermionAmplitudeFactory,
    FermionModel,
    pn_map,
    get_data_map,
    config2pn,
    FermionExchangeSampler,
)
from .fermion_core import FermionTensor,FermionTensorNetwork
class FermionModel2D(FermionModel,Model2D):
    pass 
class FermionExchangeSampler2D(FermionExchangeSampler,ExchangeSampler2D):
    pass
####################################################################################
# amplitude fxns 
####################################################################################
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
class FermionAmplitudeFactory2D(FermionAmplitudeFactory,AmplitudeFactory2D): 
    def __init__(self,psi,blks=None,spinless=False,backend='numpy',pbc=False,deterministic=False,symmetry='u1',flat=True,**compress_opts):
        # init wfn
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.nsite = self.Lx * self.Ly
        self.sites = list(itertools.product(range(self.Lx),range(self.Ly)))
        psi.add_tag('KET')
        psi.reorder(direction='row',inplace=True)
        self.set_psi(psi) # current state stored in self.psi
        self.backend = backend

        self.symmetry = symmetry 
        self.flat = flat
        self.spinless = spinless
        self.data_map = self.get_data_map()
        self.wfn2backend()

        # init contraction
        self.compress_opts = compress_opts
        self.pbc = pbc
        self.deterministic = deterministic
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
        self.is_tn = True
    def config_sign(self,config):
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum(self.config2pn(config,i*self.Ly,(i+1)*self.Ly)) % 2
        parity = np.array(parity[::-1])
        parity_cum = np.cumsum(parity[:-1])
        parity_cum += self.parity_cum 
        return (-1)**(np.dot(parity[1:],parity_cum) % 2)
    def get_all_envs(self,cols,step,stop=None,inplace=False,direction='col'):
        cols.reorder('col',inplace=True)
        return super().get_all_envs(cols,step,stop=stop,inplace=inplace,direction=direction)

from pyblock3.algebra.fermion_ops import H1
class Hubbard(FermionModel2D):
    def __init__(self,t,u,Lx,Ly,spinless=False,spin=None,sep=False,symmetry='u1',flat=True,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.t,self.u = t,u
        #self.gate = {spin:(H1(symmetry=_SYMMETRY,flat=_FLAT,spinless=spinless),'b1,b2,k1,k2')}
        data_map = get_data_map(symmetry=symmetry,flat=flat,spinless=spinless) 
        order = 'b1,k1,b2,k2'
        if spinless:
            cre = data_map['cre'].copy() 
            ann = data_map['ann'].copy() 
            h1 = np.tensordot(cre,ann,axes=([],[]))
            h1 = h1 + h1.transpose((2,3,0,1))
            h1.shape = (2,)*4
            self.gate = {spin:(h1,order)}
        else:
            h1 = []
            for spin in ('a','b'):
                cre = data_map[f'cre_{spin}'].copy() 
                ann = data_map[f'ann_{spin}'].copy() 
                h1.append(np.tensordot(cre,ann,axes=([],[])))
                h1[-1] = h1[-1] + h1[-1].transpose((2,3,0,1))
                h1[-1].shape = (4,)*4
            if sep:
                self.gate = {'a':(h1[0],order),'b':(h1[1],order)}
            else:
                self.gate = {None:((h1[0]+h1[1]),order)}
        self.spinless = spinless

        #self.pairs = self.pairs_nn()
        if self.deterministic:
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
    def get_h(self):
        h = np.zeros((self.nsite,)*2)
        for ix1 in range(self.nsite):
            i,j = self.flat2site(ix1)
            if i+1<self.Lx or self.pbc:
                ix2 = self.flatten(((i+1)%self.Lx,j))
                h[ix1,ix2] = -self.t
                h[ix2,ix1] = -self.t
            if j+1<self.Ly or self.pbc:
                ix2 = self.flatten((i,(j+1)%self.Ly))
                h[ix1,ix2] = -self.t
                h[ix2,ix1] = -self.t
        return h
#class FDKineticO2(Hubbard):
#    def __init__(self,N,L,**kwargs):
#        super().__init__(t,0.,N,N,**kwargs)
#        self.eps = L/N if self.pbc else L/(N+1.)
#
#        t = .5 / self.eps**2
#        self.l = 2. / self.eps**2
#    def compute_local_energy_eigen(self,config):
#        return self.l * sum(self.config2pn(config,0,len(config)))
#    def get_h(self):
#        h = np.zeros((self.nsite,)*2)
#        for ix1 in range(self.nsite):
#            i,j = self.flat2site(ix1)
#            h[ix1,ix1] = self.l
#            if i+1<self.Lx or self.pbc:
#                ix2 = self.flatten(((i+1)%self.Lx,j))
#                h[ix1,ix2] = -self.t
#                h[ix2,ix1] = -self.t
#            if j+1<self.Ly or self.pbc:
#                ix2 = self.flatten((i,(j+1)%self.Ly))
#                h[ix1,ix2] = -self.t
#                h[ix2,ix1] = -self.t
#        return h
#class FDKineticO4(FermionModel2D):
#    def __init__(self,N,L,spinless=False,**kwargs):
#        super().__init__(N,N,**kwargs)
#        self.spinless = spinless
#
#        self.eps = L/N if self.pbc else L/(N+1.)
#        self.t1 = 16./(24.*self.eps**2)
#        self.t2 = 1./(24.*self.eps**2)
#        self.li = 60./(24.*self.eps**2) 
#        self.lb = 59./(24.*self.eps**2) 
#        self.lc = 58./(24.*self.eps**2) 
#    
#        self.pairs = self.pairs_nn() + self.pairs_nn(d=2)
#        if self.deterministic:
#            self.batch_deterministic()
#        else:
#            self.batch_plq_nn(d=2)
#    def batch_deterministic(self):
#        self.batched_pairs = dict()
#        self.batch_nnh() 
#        self.batch_nnh(d=2) 
#        self.batch_nnv() 
#        self.batch_nnv(d=2) 
#    def pair_key(self,site1,site2):
#        dx = site2[0]-site1[0]
#        dy = site2[1]-site1[1]
#        if dx==0:
#            j0 = min(site1[1],site2[1],self.Ly-3)
#            return (site1[0],j0),(1,3)
#        elif dy==0:
#            i0 = min(site1[0],site2[0],self.Lx-3)
#            return (i0,site1[1]),(3,1)
#        else:
#            raise ValueError(f'site1={site1},site2={site2}')
#    def pair_coeff(self,site1,site2):
#        dx = site2[0]-site1[0]
#        dy = site2[1]-site1[1]
#        if dx+dy == 1:
#            return -self.t1
#        elif dx+dy == 2:
#            return self.t2
#        else:
#            raise ValueError(f'site1={site1},site2={site2}')
#    def compute_local_energy_eigen(self,config):
#        pn = self.config2pn(config,0,len(config))
#        if self.pbc:
#            return self.li * sum(pn)
#        # interior
#        ei = sum([pn[self.flatten((i,j))] for i in range(1,self.Lx-1) for j in range(1,self.Ly-1)])
#        # border
#        eb = sum([pn[self.flatten((i,j))] for i in (0,self.Lx-1) for j in range(1,self.Ly-1)]) \
#           + sum([pn[self.flatten((i,j))] for i in range(1,self.Lx-1) for j in (0,self.Ly-1)])
#        # corner
#        ec = pn[self.flatten((0,0))] + pn[self.flatten((0,self.Ly-1))] \
#           + pn[self.flatten((self.Ly-1,0))] + pn[self.flatten((self.Lx-1,self.Ly-1))]
#        return self.li * ei + self.lb * eb + self.lc * ec
#    def get_h(self):
#        h = np.zeros((self.nsite,)*2)
#        for ix1 in range(self.nsite):
#            i,j = self.flat2site(ix1)
#            if i+1<self.Lx or self.pbc:
#                ix2 = self.flatten(((i+1)%self.Lx,j))
#                h[ix1,ix2] = -self.t1
#                h[ix2,ix1] = -self.t1
#            if j+1<self.Ly or self.pbc:
#                ix2 = self.flatten((i,(j+1)%self.Ly))
#                h[ix1,ix2] = -self.t1
#                h[ix2,ix1] = -self.t1
#            if i+2<self.Lx or self.pbc:
#                ix2 = self.flatten(((i+2)%self.Lx,j))
#                h[ix1,ix2] = self.t2
#                h[ix2,ix1] = self.t2
#            if j+2<self.Ly or self.pbc:
#                ix2 = self.flatten((i,(j+2)%self.Ly))
#                h[ix1,ix2] = self.t2
#                h[ix2,ix1] = self.t2
#            if self.pbc:
#                h[ix1,ix1] = self.li
#            else:
#                if i in (0,self.Lx-1) and j in (0,self.Ly-1):
#                    h[ix1,ix1] = self.lc
#                elif i in (0,self.Lx-1) or j in (0,self.Ly-1):
#                    h[ix1,ix1] = self.lb
#                else:
#                    h[ix1,ix1] = self.li
#        return h
#def _idx_map(M):
#    idxs = np.triu_indices(M)
#    idxs = list(zip(tuple(idxs[0]),tuple(idxs[1]))) 
#    return {tuple(idx):ix for ix,idx in enumerate(idxs)}
#class CoulombOBC(Model2D):
#    def __init__(self,N,eps,spinless=False): # FD parameters
#        super().__init__(N,N)
#        self.eps = eps
#        self.spinless = spinless
#    def coulomb(self,config):
#        pn = config2pn(config,0,self.nsite,self.spinless)
#        e = 0.
#        for ix1,n1 in enumerate(pn):
#            if n1==0:
#                continue
#            i1,j1 = self.flat2site(ix1)
#            for ix2 in range(ix1+1,len(pn)):
#                n2 = pn[ix2]
#                if n2==0:
#                    continue
#                i2,j2 = self.flat2site(ix2)
#                dist = np.sqrt((i1-i2)**2+(j1-j2)**2+1e-15)
#                e += n1 * n2 / dist    
#        return e / self.eps
#    def get_eri_s8(self):
#        idx_map1 = _idx_map(self.nsite)        
#        M4 = self.nsite * (self.nsite + 1) // 2
#        idx_map2 = _idx_map(M4) 
#        M8 = M4 * (M4 + 1) // 2
#        eri = np.zeros(M8)
#        for ix1 in range(self.nsite):
#            ix1_ = idx_map1[ix1,ix1]
#            i1,j1 = self.flat2site(ix1) 
#            for ix2 in range(ix1+1,self.nsite):
#                ix2_ = idx_map1[ix2,ix2]
#                i1,j1 = self.flat2site(ix1) 
#                dist = np.sqrt((i1-i2)**2+(j1-j2)**2+1e-15) * self.eps
#                eri[idx_map[ix1_,ix2_]] = 1./dist
#        return eri
#    def get_eri_s1(self):
#        eri = np.zeros((self.nsite,)*4)
#        for ix1 in range(self.nsite):
#            i1,j1 = self.flat2site(ix1) 
#            for ix2 in range(ix1+1,self.nsite):
#                i2,j2 = self.flat2site(ix2) 
#                dist = np.sqrt((i1-i2)**2+(j1-j2)**2+1e-15) * self.eps
#                eri[ix1,ix1,ix2,ix2] = eri[ix2,ix2,ix2,ix1] = 1./dist
#        return eri
#class CoulombPBC(Model2D):
#    def __init__(self,N,L,M,spinless=False): # N is FD parameter 
#        super().__init__(N,N)
#        self.L = L
#        self.eps = L/N 
#        M = N if M is None else M
#        self.k = [[nx,ny] for nx in range(-M,M+1) for ny in range(0,M+1)] 
#        self.k.remove([0,0])
#
#        self.k = np.array(self.k) * 2. * np.pi / L
#        self.knorm = np.sqrt(np.sum(self.k**2,axis=1))
#    def coulomb(self,config):
#        pn = config2pn(config,0,self.nsite,self.spinless)
#        e = 0.
#        for ix1,n1 in enumerate(pn):
#            if n1==0:
#                continue
#            i1,j1 = self.flat2site(ix1)
#            for ix2 in range(ix1+1,len(pn)):
#                n2 = pn[ix2]
#                if n2==0:
#                    continue
#                i2,j2 = self.flat2site(ix2)
#                r = np.array([i1-i2,j1-j2]) * self.eps
#                 
#                e += n1 * n2 * np.sum(np.cos(np.dot(self.k,r)) / self.knorm)
#        return e * 4 * np.pi / self.L ** 2 
#    def get_eri_s1(self):
#        eri = np.zeros((self.nsite,)*4)
#        for ix1 in range(self.nsite):
#            i1,j1 = self.flat2site(ix1) 
#            for ix2 in range(ix1+1,self.nsite):
#                i2,j2 = self.flat2site(ix2) 
#                r = np.array([i1-i2,j1-j2]) * self.eps
#                eri[ix1,ix1,ix2,ix2] = eri[ix2,ix2,ix1,ix1] = np.sum(np.cos(np.dot(self.k,r)) / self.knorm)
#        return eri * 4 * np.pi / self.L**2
#class UEGO2(FDKineticO2):
#    def __init__(self,N,L,M=None,**kwargs): 
#        super().__init__(N,L,**kwargs)
#        self.coulomb = CoulombPBC(N,L,M,spinless=spinless) if self.pbc else \
#                       CoulombOBC(N,self.eps,spinless=spinless)
#    def compute_local_energy_eigen(self,config):
#        ke = super().compute_local_energy_eigen(config)
#        v = self.coulomb.coulomb(config)
#        return ke + v
#class UEGO4(FDKineticO4):
#    def __init__(self,N,L,M=None,spinless=False,nbatch=1): 
#        super().__init__(N,L,spinless=spinless,nbatch=nbatch)
#        self.coulomb = CoulombPBC(N,L,M,spinless=spinless) if _PBC else \
#                       CoulombOBC(N,self.eps,spinless=spinless)
#    def compute_local_energy_eigen(self,config):
#        ke = super().compute_local_energy_eigen(config)
#        v = self.coulomb.coulomb(config)
#        return ke + v
class DensityMatrix:
    def __init__(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly 
        self.pairs = [] 
        self.data = np.zeros((Lx,Ly))
        self.n = 0.
        self.nsite = Lx * Ly
    def flat2site(self,i,j):
        return i*self.Ly+j
    def compute_local_energy(self,config,compute_v=False,compute_Hv=False):
        self.n += 1.
        pn = self.af.config2pn(config,0,len(config)) 
        for i in range(self.Lx):
            for j in range(self.Ly):
                self.data[i,j] += pn[self.flat2site(i,j)]
        return 0.,0.,None,None,0. 
    def _print(self,fname,data):
        print(data)
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
def create_particle(fpeps,site,spin,spinless=False,data_map=None):
    if data_map is None:
        data_map = get_data_map(symmetry=_SYMMETRY,flat=_FLAT,spinless=spinless)
    if spinless:
        cre = data_map['cre'].copy()
    else:
        if spin=='sum':
            cre = data_map[f'cre_a'] + data_map['cre_b']
            cre.shape = (4,4)
        else:
            cre = data_map[f'cre_{spin}'].copy()
    T = fpeps[fpeps.site_tag(*site)]
    trans_order = list(range(1,T.ndim))+[0] 
    data = np.tensordot(cre, T.data, axes=((1,), (-1,))).transpose(trans_order)
    T.modify(data=data)
    return fpeps
def get_product_state(Lx,Ly,config,symmetry='u1',flat=True,spinless=False):
    fpeps = get_vaccum(Lx,Ly,symmetry=symmetry,flat=flat,spinless=spinless)
    data_map = get_data_map(symmetry=_SYMMETRY,flat=_FLAT,spinless=spinless)
    if spinless: 
        for ix,ci in enumerate(config):
            site = flat2site(ix,Ly)
            if ci==1:
                fpeps = create_particle(fpeps,site,None,spinless=spinless,data_map=data_map)
    else:
        for ix,ci in enumerate(config):
            site = flat2site(ix,Ly)
            if ci==1:
                fpeps = create_particle(fpeps,site,'a',spinless=spinless,data_map=data_map)
            if ci==2:
                fpeps = create_particle(fpeps,site,'b',spinless=spinless,data_map=data_map)
            if ci==3:
                fpeps = create_particle(fpeps,site,'b',spinless=spinless,data_map=data_map)
                fpeps = create_particle(fpeps,site,'a',spinless=spinless,data_map=data_map)
    return fpeps
def get_hole_doped_product_state(Lx,Ly,holes,symmetry='u1',flat=True):
    fpeps = get_vaccum(Lx,Ly,symmetry=symmetry,flat=flat,spinless=False)
    data_map = get_data_map(symmetry=_SYMMETRY,flat=_FLAT,spinless=False)
    for site in itertools.product(range(Lx),range(Ly)):
        if site in holes:
            continue
        fpeps = create_particle(fpeps,site,'sum',spinless=False,data_map=data_map)
    return fpeps 
def get_fpeps_z2(Lx,Ly,D,typ='one',eps=0,flat=True,normalize=False):
    from pyblock3.algebra.fermion_ops import peps_tensor_z2 
    from ..tensor_2d import PEPS
    from .fermion_2d import FPEPS
    shape = 'urdlp'
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=1,shape=shape)
    patterns = get_patterns(Lx,Ly,shape=shape)
    ftn = FermionTensorNetwork([])
    for ix, iy in itertools.product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = patterns[ix,iy]
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        shape = (D,) * (len(pattern)-1) + (4,)
        data = peps_tensor_z2(shape, pattern,parity=0,typ=typ,eps=eps,flat=flat,normalize=normalize)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn 
