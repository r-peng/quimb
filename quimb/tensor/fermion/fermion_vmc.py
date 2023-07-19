from ..tensor_vmc import DenseSampler as DenseSampler_ 
class DenseSampler(DenseSampler_):
    def __init__(self,nsite,nelec,spinless=False,**kwargs):
        self.nelec = nelec
        self.spinless = spinless
        nspin = (nelec,) if spinless else None
        super().__init__(nsite,nspin,**kwargs)
    def get_all_configs(self):
        if self.spinless:
            return super().get_all_configs()
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
from .fermion_core import FermionTensor,FermionTensorNetwork
def load_tn_from_disc(fname, delete_file=False):

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
def write_tn_to_disc(tn, tmpdir, provided_filename=False):

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
def config2pn(config,start,stop,spinless):
    pn_map = [0,1,1,2]
    if spinless:
        return config[start:stop]
    else:
        return [pn_map[ci] for ci in config[start:stop]]
from pyblock3.algebra.ad.fermion import SparseFermionTensor
from pyblock3.algebra.fermion import FlatFermionTensor
from pyblock3.algebra.fermion_ops import vaccum,creation,H1
def get_data_map(self,backend,symmetry='u1',flat=True,spinless=False):
    data_map = dict()
    if spinless: # spinless
        cre = creation(spin='a',symmetry=symmetry,flat=flat,spinless=True)
        vac = vaccum(n=1,symmetry=symmetry,flat=flat,spinless=True)
        occ = np.tensordot(cre,vac,axes=([1],[0])) 
        data_map['cre'] = cre
        data_map['ann'] = cre.dagger 
        data_map[0] = occ 
        data_map[1] = vac 
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

    for key in data_map:
        data_map[key] = tensor2backend(data_map[key],backend)   
    return data_map
def tensor2backend(self,data,backend,requires_grad=False):
    if isinstance(data,FlatFermionTensor): 
        if backend=='torch':
            data = SparsefermionTensor.from_flat(data,requires_grad=requires_grad)
    elif instance(data,SparseFermionTensor):
        if backend=='numpy':
            data = data.to_flat()
        else:
            data/requires_grad_(requires_grad=requires_grad)
    else:
        print(data)
        raise TypeError
    return data
class AmplitudeFactory:
    def write_tn_to_disc(self,tn,fname):
        return write_tn_to_disc(tn,fname)
    def get_data_map(self,backend,symmetry='u1',flat=True):
        return get_data_map(backend,symmetry=symmetry,flat=flat)
    def tensor_grad(self,tsr,set_zero=True):
        return tsr.get_grad(set_zero=set_zero) 
    def config2pn(self,config,start,stop):
        return config2pn(config,start,stop,self.spinless)
    def intermediate_sign(self,config,ix1,ix2):
        return (-1)**(sum(self.config2pn(config,ix1+1,ix2)) % 2)
    def tensor_compress_bond(self,T1,T2,absorb='right'):
        site1 = T1.get_fermion_info()[1]
        site2 = T2.get_fermion_info()[1]
        if site1<site2:
            self._tensor_compress_bond(T1,T2,absorb=absorb)
        else:
            absorb = {'left':'right','right':'left'}[absorb]
            self._tensor_compress_bond(T2,T1,absorb=absorb)
    def _tensor_compress_bond(self,T1,T2,absorb='right'):
        shared_ix, left_env_ix = T1.filter_bonds(T2)
        if not shared_ix:
            raise ValueError("The tensors specified don't share an bond.")
        T1_inds,T2_inds = T1.inds,T2.inds
    
        tmp_ix = rand_uuid()
        T1.reindex_({shared_ix[0]:tmp_ix})
        T2.reindex_({shared_ix[0]:tmp_ix})
        if absorb=='right':
            T1_L, T1_R = T1.split(left_inds=left_env_ix, right_inds=(tmp_ix,), absorb="right",
                                  get='tensors', method='qr')
            M,T2_R = T1_R,T2
        elif absorb=='left':
            T2_L, T2_R = T2.split(left_inds=(tmp_ix,), absorb="left", get='tensors', method='qr')
            T1_L,M = T1,T2_L
        else:
            raise NotImplementedError(f'absorb={absorb}')
        M.drop_tags()
        M_L, *s, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                               absorb=absorb, **self.compress_opts)
    
        # make sure old bond being used
        ns_ix, = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})
    
        T1C = T1_L.contract(M_L)
        T2C = M_R.contract(T2_R)
    
        # update with the new compressed data
        T1.modify(data=T1C.data, inds=T1C.inds)
        T2.modify(data=T2C.data, inds=T2C.inds)
