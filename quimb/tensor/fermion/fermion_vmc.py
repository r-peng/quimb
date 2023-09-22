import numpy as np
import itertools
class FermionModel:
    def gate2backend(self,backend):
        self.gate = tensor2backend(self.gate,backend)
from ..tensor_vmc import (
    safe_contract,
    DenseSampler,
    ExchangeSampler,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
class FermionDenseSampler(DenseSampler):
    def __init__(self,nsite,nelec,spinless=False,**kwargs):
        self.nelec = nelec
        self.spinless = spinless
        nspin = (nelec,) if spinless else None
        super().__init__(nsite,nspin,**kwargs)
    def get_all_configs(self,fix_sector=True):
        if self.spinless:
            return super().get_all_configs(fix_sector=fix_sector)
        if not fix_sector:
            return list(itertools.product((0,1,2,3),repeat=self.nsite)) 
        return self.get_all_configs_u11()
    def get_all_configs_u11(self):
        assert isinstance(self.nelec,tuple)
        sites = list(range(self.nsite))
        ls = [None] * 2
        for spin in (0,1):
            occs = list(itertools.combinations(sites,self.nelec[spin]))
            configs = [None] * len(occs) 
            for i,occ in enumerate(occs):
                config = np.zeros(self.nsite,dtype=int) 
                config[occ,] = 1
                configs[i] = config
            ls[spin] = configs

        na,nb = len(ls[0]),len(ls[1])
        configs = [None] * (na*nb)
        for ixa,config_a in enumerate(ls[0]):
            for ixb,config_b in enumerate(ls[1]):
                ix = ixa * nb + ixb
                configs[ix] = tuple(config_a + config_b * 2)
        return configs
    def get_all_configs_u1(self):
        if isinstance(self.nelec,tuple):
            self.nelec = sum(self.nelec)
        sites = list(range(self.nsite*2))
        occs = list(itertools.combinations(sites,self.nelec))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = np.zeros(self.nsite,dtype=int) 
            config[occ,] = 1
            configs[i] = config

        for ix in range(len(configs)):
            config = configs[ix]
            config_a,config_b = config[:self.nsite],config[self.nsite:]
            configs[ix] = tuple(config_a + config_b * 2)
        return configs
class FermionExchangeSampler(ExchangeSampler):
    def propose_new_pair(self,i1,i2):
        if self.af.spinless:
            return i2,i1
        n = abs(pn_map[i1]-pn_map[i2])
        if n==1:
            i1_new,i2_new = i2,i1
        else:
            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
            i1_new,i2_new = self.rng.choice(choices)
        return i1_new,i2_new 
from .fermion_core import FermionTensor,FermionTensorNetwork
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
pn_map = [0,1,1,2]
def config2pn(config,start,stop,spinless):
    if spinless:
        return config[start:stop]
    else:
        return [pn_map[ci] for ci in config[start:stop]]
import pyblock3.algebra.ad
pyblock3.algebra.ad.ENABLE_AUTORAY = True
from pyblock3.algebra.ad import core
core.ENABLE_FUSED_IMPLS = False
from pyblock3.algebra.ad.fermion import SparseFermionTensor
from pyblock3.algebra.fermion import FlatFermionTensor
from pyblock3.algebra.fermion_ops import vaccum,creation,H1
def get_data_map(symmetry='u1',flat=True,spinless=False):
    data_map = dict()
    if spinless: # spinless
        cre = creation(spin='a',symmetry=symmetry,flat=flat,spinless=True)
        vac = vaccum(n=1,symmetry=symmetry,flat=flat,spinless=True)
        occ = np.tensordot(cre,vac,axes=([1],[0])) 
        data_map['cre'] = cre
        data_map['ann'] = cre.dagger 
        data_map[0] = vac 
        data_map[1] = occ 
    else: # spin-1/2
        cre_a = creation(spin='a',symmetry=symmetry,flat=flat,spinless=False)
        cre_b = creation(spin='b',symmetry=symmetry,flat=flat,spinless=False)
        data_map['cre_a'] = cre_a
        data_map['cre_b'] = cre_b
        data_map['ann_a'] = cre_a.dagger
        data_map['ann_b'] = cre_b.dagger

        vac = vaccum(n=1,symmetry=symmetry,flat=flat,spinless=False)
        occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
        occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
        occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
        data_map[0] = vac
        data_map[1] = occ_a
        data_map[2] = occ_b
        data_map[3] = occ_db
    return data_map
def tensor2backend(data,backend,requires_grad=False):
    if isinstance(data,FlatFermionTensor): 
        if backend=='torch':
            data = SparseFermionTensor.from_flat(data,requires_grad=requires_grad)
    elif isinstance(data,SparseFermionTensor):
        if backend=='numpy':
            data = data.to_flat()
        else:
            data.requires_grad_(requires_grad=requires_grad)
    elif isinstance(data,np.ndarray):
        if backend=='numpy':
            pass
        else:
            print(data)
            raise TypeError
    return data
def _add_gate(tn,gate,order,where,site_ind,site_tag,contract=True):
    # reindex
    kixs = [site_ind(site) for site in where]
    bixs = [kix+'*' for kix in kixs]
    for site,kix,bix in zip(where,kixs,bixs):
        tn[site_tag(site),'BRA'].reindex_({kix:bix})

    # add gate
    if order=='b1,k1,b2,k2':
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
    elif order=='b1,b2,k1,k2':
        inds = bixs + kixs
    else:
        raise NotImplementedError
    T = FermionTensor(data=gate,inds=inds)
    tn.add_tensor(T,virtual=True)
    if not contract:
        return tn  
    return safe_contract(tn)
class FermionAmplitudeFactory:
    def write_tn_to_disc(self,tn,fname):
        return write_ftn_to_disc(tn,fname,provided_filename=True)
    def get_constructors(self,psi):
        from .block_interface import Constructor
        constructors = [None] * self.nsite 
        for site in self.sites:
            data = psi[self.site_tag(site)].data
            bond_infos = [data.get_bond_info(ax,flip=False) \
                          for ax in range(data.ndim)]
            cons = Constructor.from_bond_infos(bond_infos,data.pattern,flat=self.flat)
            dq = data.dq
            size = cons.vector_size(dq)
            ix = self.site_map[site]
            constructors[ix] = (cons,dq,data.shape),size,site
        return constructors
    def get_data_map(self):
        return get_data_map(symmetry=self.symmetry,flat=self.flat,spinless=self.spinless)
    def tensor2backend(self,data,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        return tensor2backend(data,backend,requires_grad=requires_grad)
    def tensor2vec(self,tsr,ix):
        cons,dq,_ = self.constructors[ix][0]
        return cons.tensor_to_vector(self.tensor2backend(tsr,'numpy'))
    def vec2tensor(self,x,ix):
        cons,dq,shape = self.constructors[ix][0]
        data = cons.vector_to_tensor(x,dq)
        data.shape = shape
        return self.tensor2backend(data,self.backend)
    def tensor_grad(self,tsr,set_zero=True):
        return tsr.get_grad(set_zero=set_zero) 
    def config2pn(self,config,start,stop):
        return config2pn(config,start,stop,self.spinless)
    def intermediate_sign(self,config,ix1,ix2):
        return (-1)**(sum(self.config2pn(config,ix1+1,ix2)) % 2)
    def get_bra_tsr(self,ci,site,append=''):
        inds = self.site_ind(site)+append,
        tags = self.site_tags(site) + ('BRA',)
        data = self.data_map[ci].dagger
        return FermionTensor(data=data,inds=inds,tags=tags)
    def site_grad(self,tn,site):
        ket = tn[self.site_tag(site),'KET']
        tid = ket.get_fermion_info()[0]
        ket = tn._pop_tensor(tid,remove_from_fermion_space='end')
        #ket = tn._pop_tensor(tid,remove_from_fermion_space=True)
        g = tn.contract(output_inds=ket.inds[::-1])
        return g.data.dagger 
    def tensor_compress_bond(self,T1,T2,absorb='right'):
        site1 = T1.get_fermion_info()[1]
        site2 = T2.get_fermion_info()[1]
        if site1<site2:
            self._tensor_compress_bond(T1,T2,absorb=absorb)
        else:
            absorb = {'left':'right','right':'left'}[absorb]
            self._tensor_compress_bond(T2,T1,absorb=absorb)
    def _add_gate(self,tn,where,contract=True):
        return _add_gate(tn,self.model.gate.copy(),self.model.order,
                         where,self.site_ind,self.site_tag,contract=contract)
