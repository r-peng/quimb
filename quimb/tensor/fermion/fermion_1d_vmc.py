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
from pyblock3.algebra.ad.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_ops import vaccum,creation,H1

import sys
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True,pbc=False,deterministic=False,**compress_opts):
    this.pbc = pbc

    this.data_map = dict()
    # spinless
    cre = creation(spin='a',symmetry=symmetry,flat=flat,spinless=True)
    vac = vaccum(n=1,symmetry=symmetry,flat=flat,spinless=True)
    occ = np.tensordot(cre,vac,axes=([1],[0])) 
    this.data_map['cre'] = cre
    this.data_map['ann'] = cre.dagger 
    this.data_map['occ'] = occ 
    this.data_map['vac',True] = vac 
    this.data_map['h1',True] = H1(symmetry=symmetry,flat=flat,spinless=True).transpose((0,2,1,3))

    # spin-1/2
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat,spinless=False)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat,spinless=False)
    this.data_map['cre_a'] = cre_a
    this.data_map['cre_b'] = cre_b
    this.data_map['ann_a'] = cre_a.dagger
    this.data_map['ann_b'] = cre_b.dagger

    vac = vaccum(n=1,symmetry=symmetry,flat=flat,spinless=False)
    occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
    occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
    occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
    this.data_map['vac',False] = vac
    this.data_map['occ_a'] = occ_a
    this.data_map['occ_b'] = occ_b
    this.data_map['occ_db'] = occ_db
    this.data_map['h1',False] = H1(symmetry=symmetry,flat=flat,spinless=False).transpose((0,2,1,3))
    return this.data_map
pn_map = [0,1,1,2]
config_map = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}

from ..tensor_1d import MatrixProductState 
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid,tensor_split
class FMPS(FermionTensorNetwork,MatrixProductState):

    _EXTRA_PROPS = (
        '_site_tag_id',
        '_L',
        '_site_ind_id',
    )
    def __init__(self):
        self._site_tag_id = None
        self._site_ind_id = None
        self._L = None
    def reorder(self,layer_tags=None,inplace=True):
        tid_map = dict()
        current_position = 0
        for i in range(self.L):
            site_tag = self.site_tag(i)
            tids = self._get_tids_from_tags(site_tag)
            if len(tids) == 1:
                tid, = tids
                if tid not in tid_map:
                    tid_map[tid] = current_position
                    current_position +=1
            else:
                if layer_tags is None:
                    _tags = [self.tensor_map[ix].tags for ix in tids]
                    _tmp_tags = _tags[0].copy()
                    for itag in _tags[1:]:
                        _tmp_tags &= itag
                    _layer_tags = sorted([list(i-_tmp_tags)[0] for i in _tags])
                else:
                    _layer_tags = layer_tags
                for tag in _layer_tags:
                    tid, = self._get_tids_from_tags((site_tag, tag))
                    if tid not in tid_map:
                        tid_map[tid] = current_position
                        current_position += 1
        return self._reorder_from_tid(tid_map, inplace)

#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
import pickle,uuid
def load_ftn_from_disc(fname, delete_file=False):

    # Get the data
    if type(fname) != str:
        data = fname
    else:
        # Open up the file
        with open(fname, 'rb') as f:
            data = pickle.load(f)

    # Set up a dummy fermionic tensor network
    tn = FermionTensorNetwork([])

    # Put the tensors into the ftn
    tensors = [None,] * data['ntensors']
    for i in range(data['ntensors']):

        # Get the tensor
        ten_info = data['tensors'][i]
        ten = ten_info['tensor']
        ten = FermionTensor(ten.data, inds=ten.inds, tags=ten.tags)

        # Get/set tensor info
        tid, site = ten_info['fermion_info']
        ten.fermion_owner = None
        ten._avoid_phase = False

        # Add the required phase
        ten.phase = ten_info['phase']

        # Add to tensor list
        tensors[site] = (tid, ten)

    # Add tensors to the tn
    for (tid, ten) in tensors:
        tn.add_tensor(ten, tid=tid, virtual=True)

    # Get addition attributes needed
    tn_info = data['tn_info']

    # Set all attributes in the ftn
    extra_props = dict()
    for props in tn_info:
        extra_props[props[1:]] = tn_info[props]

    # Convert it to the correct type of fermionic tensor network
    tn = tn.view_as_(data['class'], **extra_props)

    # Remove file (if desired)
    if delete_file:
        delete_ftn_from_disc(fname)

    # Return resulting tn
    return tn

def rand_fname():
    return str(uuid.uuid4())
def write_ftn_to_disc(tn, tmpdir, provided_filename=False):

    # Create a generic dictionary to hold all information
    data = dict()

    # Save which type of tn this is
    data['class'] = type(tn)

    # Add information relevant to the tensors
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)

    # Add the tensors themselves
    data['tensors'] = []
    ntensors = 0
    for ten in tn.tensors:
        ten_info = dict()
        ten_info['fermion_info'] = ten.get_fermion_info()
        ten_info['phase'] = ten.phase
        ten_info['tensor'] = ten
        data['tensors'].append(ten_info)
        ntensors += 1
    data['ntensors'] = ntensors

    # If tmpdir is None, then return the dictionary
    if tmpdir is None:
        return data

    # Write fermionic tensor network to disc
    else:
        # Create a temporary file
        if provided_filename:
            fname = tmpdir
            print('saving to ', fname)
        else:
            if tmpdir[-1] != '/': 
                tmpdir = tmpdir + '/'
            fname = tmpdir + rand_fname()

        # Write to a file
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        # Return the filename
        return fname
####################################################################################
# amplitude fxns 
####################################################################################
from ..tensor_1d_vmc import ContractionEngine as ContractionEngine_
def config2pn(config,start,stop,spinless):
    if spinless:
        return config[start:stop]
    else:
        return [pn_map[ci] for ci in config[start:stop]]
class ContractionEngine(ContractionEngine_): 
    def init_contraction(self,L):
        self.L = L
        self.pbc = pbc
        self.data_map = data_map
    def _2backend(self,data,requires_grad):
        if self.backend=='numpy':
            return data.copy()
        else:
            return SparseFermionTensor.from_flat(data,requires_grad=requires_grad)
    def config2pn(self,config,start,stop):
        return config2pn(config,start,stop,self.spinless)
    def intermediate_sign(self,config,ix1,ix2):
        return (-1)**(sum(self.config2pn(config,ix1+1,ix2)) % 2)
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
    def get_bra_tsr(self,ci,i,append='',tn=None):
        tn = self.psi if tn is None else tn 
        inds = tn.site_ind(i)+append,
        tags = tn.site_tag(i),'BRA' 
        if self.spinless:
            key = {0:('vac',True),1:'occ'}[ci]
        else:
            key = {0:('vac',False),1:'occ_a',2:'occ_b',3:'occ_db'}[ci] 
        data = self._2backend(data_map[key].dagger,False)
        return FermionTensor(data=data,inds=inds,tags=tags)
    def site_grad(self,ftn_plq,i):
        ket = ftn_plq[ftn_plq.site_tag(i),'KET']
        tid = ket.get_fermion_info()[0]
        ket = ftn_plq._pop_tensor(tid,remove_from_fermion_space='end')
        g = ftn_plq.contract(output_inds=ket.inds[::-1])
        return g.data.dagger 
from ..tensor_1d_vmc import AmplitudeFactory as AmplitudeFactory_
class AmplitudeFactory(ContractionEngine,AmplitudeFactory_):
    def __init__(self,psi,spinless=False,flat=True):
        super().init_contraction(psi.L)
        psi.reorder(inplace=True)
        psi.add_tag('KET')
        self.spinless = spinless
        self.flat = flat

        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict()
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
    def config_sign(self,config):
        return 1. 
    def get_constructors(self,psi):
        from .block_interface import Constructor
        constructors = [None] * (self.L)
        for i in range(psi.L):
            data = psi[i].data
            bond_infos = [data.get_bond_info(ax,flip=False) \
                          for ax in range(data.ndim)]
            cons = Constructor.from_bond_infos(bond_infos,data.pattern,flat=self.flat)
            dq = data.dq
            size = cons.vector_size(dq)
            constructors[i] = (cons,dq),size
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
####################################################################################
# ham class 
####################################################################################
from ..tensor_1d_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(ContractionEngine,Hamiltonian_):
    def __init__(self,L,nbatch=1,spinless=False):
        super().init_contraction(L)
        self.nbatch = nbatch
        self.spinless = spinless
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self._2backend(self.data_map[self.key,self.spinless],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return FermionTensor(data=data,inds=inds,tags=tags) 
class Hubbard(Hamiltonian):
    def __init__(self,t,u,L,**kwargs):
        super().__init__(L,**kwargs)
        self.t,self.u = t,u
        self.key = 'h1'

        self.pairs = self.pairs_nn()
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
#class FDKineticO2(Hubbard):
#    def __init__(self,N,L,**kwargs):
#        self.eps = L/(N+1e-15) if pbc else L/(N+1.)
#
#        t = .5 / self.eps**2
#        self.l = 2. / self.eps**2
#        super().__init__(t,0.,N,N,**kwargs)
#    def compute_local_energy_eigen(self,config):
#        return self.l * sum(self.config2pn(config,0,len(config)))
#    def get_h(self):
#        nsite = self.Lx * self.Ly
#        h = np.zeros((nsite,)*2)
#        for i in range(self.Lx):
#            for j in range(self.Ly):
#                ix1 = self.flatten(i,j)
#                h[ix1,ix1] = self.l
#                if i+1<self.Lx or self.pbc:
#                    ix2 = self.flatten((i+1)%self.Lx,j) 
#                    h[ix1,ix2] = -self.t
#                    h[ix2,ix1] = -self.t
#                if j+1<self.Ly or self.pbc:
#                    ix2 = self.flatten(i,(j+1)%self.Ly) 
#                    h[ix1,ix2] = -self.t
#                    h[ix2,ix1] = -self.t
#        return h
#class FDKineticO4(Hamiltonian):
#    def __init__(self,N,L,**kwargs):
#        super().__init__(N,N,**kwargs)
#        self.eps = L/(N+1e-15) if pbc else L/(N+1.)
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
#        ei = sum([pn[self.flatten(i,j)] for i in range(1,self.Lx-1) for j in range(1,self.Ly-1)])
#        # border
#        eb = sum([pn[self.flatten(i,j)] for i in (0,self.Lx-1) for j in range(1,self.Ly-1)]) \
#           + sum([pn[self.flatten(i,j)] for i in range(1,self.Lx-1) for j in (0,self.Ly-1)])
#        # corner
#        ec = pn[self.flatten(0,0)] + pn[self.flatten(0,self.Ly-1)] \
#           + pn[self.flatten(self.Ly-1,0)] + pn[self.flatten(self.Lx-1,self.Ly-1)]
#        return self.li * ei + self.lb * eb + self.lc * ec
#    def get_h(self):
#        nsite = self.Lx * self.Ly
#        h = np.zeros((nsite,)*2)
#        for i in range(self.Lx):
#             for j in range(self.Ly):
#                 ix1 = self.flatten(i,j)
#                 if i+1<self.Lx or self.pbc:
#                     ix2 = self.flatten((i+1)%self.Lx,j) 
#                     h[ix1,ix2] = -self.t1
#                     h[ix2,ix1] = -self.t1
#                 if j+1<self.Ly or self.pbc:
#                     ix2 = self.flatten(i,(j+1)%self.Ly) 
#                     h[ix1,ix2] = -self.t1
#                     h[ix2,ix1] = -self.t1
#                 if i+2<self.Lx or self.pbc:
#                     ix2 = self.flatten((i+2)%self.Lx,j) 
#                     h[ix1,ix2] = self.t2
#                     h[ix2,ix1] = self.t2
#                 if j+2<self.Ly or self.pbc:
#                     ix2 = self.flatten(i,(j+2)%self.Ly) 
#                     h[ix1,ix2] = self.t2
#                     h[ix2,ix1] = self.t2
#                 if self.pbc:
#                     h[ix1,ix1] = self.li
#                 else:
#                     if i in (0,self.Lx-1) and j in (0,self.Ly-1):
#                         h[ix1,ix1] = self.lc
#                     elif i in (0,self.Lx-1) or j in (0,self.Ly-1):
#                         h[ix1,ix1] = self.lb
#                     else:
#                         h[ix1,ix1] = self.li
#        return h
#def coulomb(config,Lx,Ly,eps,spinless):
#    pn = config2pn(config,0,len(config),spinless)
#    e = 0.
#    for ix1 in range(len(config)):
#        n1 = pn[ix1]
#        if n1==0:
#            continue
#        i1,j1 = flat2site(ix1,Lx,Ly)
#        for ix2 in range(ix1+1,len(config)):
#            n2 = pn[ix2]
#            if n2==0:
#                continue
#            i2,j2 = flat2site(ix2,Lx,Ly)
#            dist = np.sqrt((i1-i2)**2+(j1-j2)**2)
#            e += n1 * n2 / dist    
#    return e / eps
#class UEGO2(FDKineticO2):
#    def compute_local_energy_eigen(self,config):
#        ke = super().compute_local_energy_eigen(config)
#        v = coulomb(config,self.Lx,self.Ly,self.eps,self.spinless)
#class UEGO4(FDKineticO4):
#    def compute_local_energy_eigen(self,config):
#        ke = super().compute_local_energy_eigen(config)
#        v = coulomb(config,self.Lx,self.Ly,self.eps,self.spinless)
#class DensityMatrix(Hamiltonian):
#    def __init__(self,Lx,Ly,spinless=False):
#        self.Lx,self.Ly = Lx,Ly 
#        self.pairs = [] 
#        self.data = np.zeros((Lx,Ly))
#        self.n = 0.
#        self.spinless = spinless
#    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
#        self.n += 1.
#        pn = self.config2pn(config,0,len(config)) 
#        for i in range(self.Lx):
#            for j in range(self.Ly):
#                self.data[i,j] += pn[self.flatten(i,j)]
#        return 0.,0.,None,None,0. 
#    def _print(self,fname,data):
#        print(data)
#####################################################################################
## sampler 
#####################################################################################
#from ..tensor_2d_vmc_ import ExchangeSampler2 as ExchangeSampler_
#class ExchangeSampler(ExchangeSampler_):
#    def new_pair(self,i1,i2):
#        if self.amplitude_factory.spinless:
#            return i2,i1
#        return self.new_pair_full(i1,i2)
#    def new_pair_full(self,i1,i2):
#        n = abs(pn_map[i1]-pn_map[i2])
#        if n==1:
#            i1_new,i2_new = i2,i1
#        else:
#            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
#            i1_new,i2_new = self.rng.choice(choices)
#        return i1_new,i2_new 
from ..tensor_1d_vmc import DenseSampler as DenseSampler_
class DenseSampler(DenseSampler_):
    def __init__(self,L,nelec,**kwargs):
        self.nelec = nelec
        super().__init__(L,None,**kwargs)
    def get_all_configs(self):
        return self.get_all_configs_u11()
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

