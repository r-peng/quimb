import uuid,pickle,itertools
import numpy as np
from .fermion_core import FermionTensor, FermionTensorNetwork
#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
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
################################################################################
# some helper fxns  
################################################################################    
def psi2vecs(constructors,psi):
    is_dict = isinstance(psi,dict)
    ls = [None] * len(constructors)
    for ix,(cons,_,size,site) in enumerate(constructors):
        if is_dict: 
            vec = np.zeros(size)
            if site in psi:
                g = psi[site] 
                if g is not None:
                    vec = cons.tensor_to_vector(g) 
        else:
            vec = cons.tensor_to_vector(psi[psi.site_tag(*site)].data)
        ls[ix] = vec
    return ls
def split_vec(constructors,x):
    ls = [None] * len(constructors)
    start = 0
    for ix,(_,_,size,_) in enumerate(constructors):
        stop = start + size
        ls[ix] = x[start:stop]
        start = stop
    return ls 
def vec2psi(constructors,x,psi=None): 
    psi_new = dict() if psi is None else psi
    ls = split_vec(constructors,x)
    for ix,(cons,dq,_,site) in enumerate(constructors):
        data = cons.vector_to_tensor(ls[ix],dq)
        if psi is None:
            psi_new[site] = data
        else:
            psi_new[psi.site_tag(*site)].modify(data=data)
    return psi_new
