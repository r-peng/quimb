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
from pyblock3.algebra.ad.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_setting import set_subspace
def set_options(symmetry='u1',flat=True,pbc=False,deterministic=False,**compress_opts):
    this.pbc = pbc
    this.deterministic = True if pbc else deterministic
    this.compress_opts = compress_opts
    set_max_bond(compress_opts.get('max_bond',None))

    this.data_map = dict()
    for subspace in ['a','b']:
        set_subspace(subspace)
        from pyblock3.algebra.fermion_ops import vaccum,creation,H1
        cre = creation(spin=subspace,symmetry=symmetry,flat=flat)
        vac = vaccum(n=1,symmetry=symmetry,flat=flat)
        occ = np.tensordot(cre,vac,axes=([1],[0])) 
        this.data_map[f'cre_{subspace}_{subspace}'] = cre
        this.data_map[f'ann_{subspace}_{subspace}'] = cre.dagger 
        this.data_map[f'occ_{subspace}_{subspace}'] = occ 
        this.data_map['vac_'+subspace] = vac 
        this.data_map['h1_'+subspace] = H1(symmetry=symmetry,flat=flat).transpose((0,2,1,3))

    set_subspace('full')
    from pyblock3.algebra.fermion_ops import vaccum,creation,H1
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    this.data_map['cre_a_full'] = cre_a
    this.data_map['cre_b_full'] = cre_b
    this.data_map['ann_a_full'] = cre_a.dagger
    this.data_map['ann_b_full'] = cre_b.dagger

    vac = vaccum(n=1,symmetry=symmetry,flat=flat)
    occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
    occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
    occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
    this.data_map['vac_full'] = vac
    this.data_map['occ_a_full'] = occ_a
    this.data_map['occ_b_full'] = occ_b
    this.data_map['occ_db_full'] = occ_db
    this.data_map['h1_full'] = H1(symmetry=symmetry,flat=flat).transpose((0,2,1,3))
    return this.data_map
pn_map = [0,1,1,2]
config_map = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}
from ..tensor_2d_vmc import flatten,flat2site 
#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
import pickle,uuid
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid,tensor_split
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
# initialization fxns
####################################################################################
def get_vaccum(Lx,Ly,symmetry='u1',flat=True,subspace='full'):
    set_subspace(subspace)
    from pyblock3.algebra.fermion_ops import bonded_vaccum
    from ..tensor_2d import PEPS
    from .fermion_2d import FPEPS
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=1)
    ftn = FermionTensorNetwork([])
    ind_to_pattern_map = dict()
    inv_pattern = {"+":"-", "-":"+"}
    def get_pattern(inds):
        """
        make sure patterns match in input tensors, eg,
        --->A--->B--->
         i    j    k
        pattern for A_ij = +-
        pattern for B_jk = +-
        the pattern of j index must be reversed in two operands
        """
        pattern = ""
        for ix in inds[:-1]:
            if ix in ind_to_pattern_map:
                ipattern = inv_pattern[ind_to_pattern_map[ix]]
            else:
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<len(pattern) else "+"
                ind_to_pattern_map[ix] = ipattern
            pattern += ipattern
        pattern += "+" # assuming last index is the physical index
        return pattern
    for ix, iy in itertools.product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = get_pattern(T.inds)
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        data = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern,
                             symmetry=symmetry,flat=flat)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn
def create_particle(fpeps,site,spin,subspace='full'):
    cre = data_map[f'cre_{spin}_{subspace}'].copy()
    T = fpeps[fpeps.site_tag(*site)]
    trans_order = list(range(1,T.ndim))+[0] 
    data = np.tensordot(cre, T.data, axes=((1,), (-1,))).transpose(trans_order)
    T.modify(data=data)
    return fpeps
def get_product_state(Lx,Ly,spin_map,symmetry='u1',flat=True,subspace='full'):
    fpeps = get_vaccum(Lx,Ly,symmetry=symmetry,flat=flat,subspace=subspace)
    for spin,sites in spin_map.items():
        for site in sites:
            fpeps = create_particle(fpeps,site,spin,subspace=subspace)
    return fpeps
####################################################################################
# amplitude fxns 
####################################################################################
from ..tensor_2d_vmc import ContractionEngine as ContractionEngine_
class ContractionEngine(ContractionEngine_): 
    def init_contraction(self,Lx,Ly,subspace='full'):
        self.Lx,self.Ly = Lx,Ly
        self.subspace = subspace
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
    def pn(self,config,start,stop):
        if self.subspace=='full':
            return config[start:stop] 
        else:
            return [pn_map[ci] for ci in config[start:stop]]
    def intermediate_sign(self,config,ix1,ix2):
        return (-1)**(sum(self.pn(config,ix1+1,ix2)) % 2)
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
    def get_bra_tsr(self,fpeps,ci,i,j,append=''):
        inds = fpeps.site_ind(i,j)+append,
        tags = fpeps.site_tag(i,j),fpeps.row_tag(i),fpeps.col_tag(j),'BRA' 
        if self.subspace=='full':
            key = {0:'vac_full',1:'occ_a_full',2:'occ_b_full',3:'occ_db_full'}[ci] 
        else:
            key = {0:f'vac_{self.subspace}',1:'occ_{self.subspace}_{self.subspace}'}[ci]
        data = self._2backend(data_map[key].dagger,False)
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
from ..tensor_2d_vmc import AmplitudeFactory as AmplitudeFactory_
class AmplitudeFactory(ContractionEngine,AmplitudeFactory_):
    def __init__(self,psi,subspace='full'):
        super().init_contraction(psi.Lx,psi.Ly,subspace=subspace)
        psi.reorder(direction='row',inplace=True)
        psi.add_tag('KET')
        self.parity_cum = get_parity_cum(psi)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict()

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
        self.small_mem = True
    def config_sign(self,config):
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum(self.pn(config,i*self.Ly,(i+1)*self.Ly)) % 2
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
####################################################################################
# ham class 
####################################################################################
from ..tensor_2d_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(ContractionEngine,Hamiltonian_):
    def __init__(self,Lx,Ly,nbatch=1,subspace='full'):
        super().init_contraction(Lx,Ly,subspace=subspace)
        self.nbatch = nbatch
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self._2backend(self.data_map[f'{self.key}_{self.subspace}'],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return FermionTensor(data=data,inds=inds,tags=tags) 
class Hubbard(Hamiltonian):
    def __init__(self,t,u,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.t,self.u = t,u
        self.key = 'h1'

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

