import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

from ..tensor_1d_vmc import (
    AmplitudeFactory1D,
    Model1D,
    ExchangeSampler1D,
)
from .fermion_vmc import (
    FermionAmplitudeFactory,
    Hubbard,
    pn_map,
    get_data_map,
    config2pn,
    FermionExchangeSampler,
)
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid
class FermionExchangeSampler1D(FermionExchangeSampler,ExchangeSampler1D):
    pass
class FermionAmplitudeFactory1D(FermionAmplitudeFactory,AmplitudeFactory1D): 
    def __init__(self,psi,blks=None,spinless=False,backend='numpy',pbc=False,symmetry='u1',flat=True,**compress_opts):
        # init wfn
        self.L,self.Ly = psi.Ly,psi.Ly
        self.nsite = self.L
        self.sites = list(range(self.Ly))
        psi.add_tag('KET')
        psi.reorder(direction='col',inplace=True)
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
        self.deterministic = False 

        if blks is None:
            blks = [self.sites]
        self.site_map = self.get_site_map(blks)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict(blks)
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)
        self.nparam = len(self.get_x())
        self.is_tn = True
    def site_tag(self,site):
        return self.psi.col_tag(site)
    def site_tags(self,site):
        return (self.psi.row_tag(0),self.psi.col_tag(site),self.psi.site_tag(0,site))
    def site_ind(self,site):
        return self.psi.site_ind(0,site)
    def col_tag(self,col):
        return self.psi.col_tag(col)    
class Hubbard1D(Hubbard,Model1D):
    def __init__(self,t,u,L,spinless=False,spin=None,sep=False,symmetry='u1',flat=True,**kwargs):
        super().__init__(L,**kwargs)
        self.t,self.u = t,u
        self.get_gate(spinless=spinless,spin=spin,sep=sep,symmetry=symmetry,flat=flat)

        self.batched_pairs = dict()
        self.pairs_nn()
    def pair_key(self,i,j):
        return min(i,j,self.L-2),2
    def get_h(self):
        h = np.zeros((self.L,)*2)
        for ix1 in range(self.L):
            if ix1+1<self.L or self.pbc:
                ix2 = (ix1+1)%self.L
                h[ix1,ix2] = -self.t
                h[ix2,ix1] = -self.t
        return h
