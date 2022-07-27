import numpy as np
import time,uuid,pickle,resource
from .fermion_core import FermionTensor,FermionTensorNetwork
import sys
#####################################################################################
# MPI STUFF
#####################################################################################
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

def parallelized_looped_function_balanced(func, large, small, args, kwargs):
    results = []
    sizes = []
    iterate_over = large + small 
    for worker in reversed(range(SIZE)):
        #i,j = worker,SIZE-1-worker
        #tasks = large[i::SIZE] + small[j::SIZE]
        tasks = iterate_over[worker::SIZE]
        sizes.append(len(tasks))
        if worker != 0:
            worker_info = [func, tasks, args, kwargs]
            COMM.send(worker_info, dest=worker)
        else:
            for task in tasks:
                results.append(func(task, *args, **kwargs))
    for worker in range(1, SIZE):
        results += COMM.recv(source=worker)
    print('large,small=',len(large),len(small))
    print('sizes=',sizes)
    return results

def parallelized_looped_function(func, iterate_over, args, kwargs):
    """
    When a function must be called many times for a set of parameters
    then this implements a parallelized loop controlled by the rank 0 process.
    
    Args:
        func: Function
            The function that will be called. 
        iterate_over: iterable
            The argument of the function that will be iterated over.
        args: list
            A list of arguments to be supplied to the function
        kwargs: dict
            A dictrionary of arguments to be supplied to the function
            at each call

    Returns:
        results: list
            This is a list of the results of each function call stored in 
            a list with the same ordering as was supplied in 'iterate_over'
    """
    ## Figure out which items are done by which worker
    #min_per_worker = len(iterate_over) // SIZE

    #per_worker = [min_per_worker for _ in range(SIZE)]
    #for i in range(len(iterate_over) - min_per_worker * SIZE):
    #    per_worker[SIZE-1-i] += 1

    #randomly_permuted_tasks = np.random.permutation(len(iterate_over))
    #worker_ranges = []
    #for worker in range(SIZE):
    #    start = sum(per_worker[:worker])
    #    end = sum(per_worker[:worker+1])
    #    tasks = [randomly_permuted_tasks[ind] for ind in range(start, end)]
    #    worker_ranges.append(tasks)

    results = []
    #sizes = []
    for worker in reversed(range(SIZE)):
        #worker_iterate_over = [iterate_over[i] for i in worker_ranges[worker]]
        worker_iterate_over = iterate_over[worker::SIZE]
        #sizes.append(len(worker_iterate_over))
        if worker != 0:
            worker_info = [func, worker_iterate_over, args, kwargs]
            COMM.send(worker_info, dest=worker)
        else:
            for func_call in range(len(worker_iterate_over)):
                results.append(func(worker_iterate_over[func_call],*args,**kwargs))
    for worker in range(1, SIZE):
        results += COMM.recv(source=worker)
    #print('total=',len(iterate_over))
    #print('sizes=',sizes)
    return results

def worker_execution():
    """
    All but the rank 0 process should initially be called
    with this function. It is an infinite loop that continuously 
    checks if an assignment has been given to this process. 
    Once an assignment is recieved, it is executed and sends
    the results back to the rank 0 process. 
    """
    # Create an infinite loop
    while True:

        # Loop to see if this process has a message
        # (helps keep processor usage low so other workers
        #  can use this process until it is needed)
        while not COMM.Iprobe(source=0):
            time.sleep(0.01)

        # Recieve the assignments from RANK 0
        assignment = COMM.recv()

        # End execution if received message 'finished'
        if assignment == 'finished': 
            break

        # Otherwise, call function
        function,iterate_over,args,kwargs = assignment
        results = [None] * len(iterate_over)
        for func_call in range(len(iterate_over)):
            results[func_call] = function(iterate_over[func_call], 
                                          *args, **kwargs)

        # Send the results back to the rank 0 process
        COMM.send(results, dest=0)

#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################

def _profile(title):
    res = resource.getrusage(resource.RUSAGE_SELF)
    with open(f'./dmrg_log/cpu_{RANK}.log','a') as f:
        f.write(title+'\n')
        f.write('    ru_utime    = {}\n'.format(res.ru_utime   )) 
        f.write('    ru_stime    = {}\n'.format(res.ru_stime   )) 
        f.write('    ru_maxrss   = {}\n'.format(res.ru_maxrss  )) 
        f.write('    ru_ixrss    = {}\n'.format(res.ru_ixrss   )) 
        f.write('    ru_idrss    = {}\n'.format(res.ru_idrss   )) 
        f.write('    ru_isrss    = {}\n'.format(res.ru_isrss   )) 
        f.write('    ru_minflt   = {}\n'.format(res.ru_minflt  )) 
        f.write('    ru_majflt   = {}\n'.format(res.ru_majflt  )) 
        f.write('    ru_nswap    = {}\n'.format(res.ru_nswap   )) 
        f.write('    ru_inblock  = {}\n'.format(res.ru_inblock )) 
        f.write('    ru_oublock  = {}\n'.format(res.ru_oublock )) 
        f.write('    ru_msgsnd   = {}\n'.format(res.ru_msgsnd  )) 
        f.write('    ru_msgrcv   = {}\n'.format(res.ru_msgrcv  )) 
        f.write('    ru_nsignals = {}\n'.format(res.ru_nsignals)) 
        f.write('    ru_nvcsw    = {}\n'.format(res.ru_nvcsw   )) 
        f.write('    ru_nivcsw   = {}\n'.format(res.ru_nivcsw  )) 

def rand_fname():
    return str(uuid.uuid4())

def delete_ftn_from_disc(fname):
    """
    Simple wrapper that removes a file from disc. 
    Args:
        fname: str
            A string indicating the file to be removed
    """
    try:
        os.remove(fname)
    except:
        pass

def remove_env_from_disc(benv):
    """
    Simple wrapper that removes all files associated
    with a boundary environment from disc. 
    Args:
        benv: dict
            This is the dictionary holding the boundary environment
            tensor networks. Each entry in the dictionary should be a 
            dictionary. We check if the key 'tn' is in that dictionary 
            and if so, remove that file from disc.
    """
    for key in benv:
        if 'tn' in benv[key]:
            delete_ftn_from_disc(benv[key]['tn'])

def load_ftn_from_disc(fname, delete_file=False):
    """
    If a fermionic tensor network has been written to disc
    this function loads it back as a fermionic tensor network
    of the same class. if 'delete_file' is True, then the
    supplied file will also be removed from disc.
    """

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

def write_ftn_to_disc(tn, tmpdir, provided_filename=False):
    """
    This function takes a fermionic tensor network that is supplied 'tn'
    and saves it as a random filename inside of the supplied directory
    'tmpdir'. If 'provided_filename' is True, then it will assume that 
    'tmpdir' includes a previously assigned filename and will overwrite
    that file
    """

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
            # print('saving to ', fname)
        else:
            if tmpdir[-1] != '/': 
                tmpdir = tmpdir + '/'
            fname = tmpdir + rand_fname()

        # Write to a file
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        # Return the filename
        return fname

def _write_ftn_to_disc(tn,fname):
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
        ten_info['global_flip'] = ten.phase.get('global_flip',False)
        ten_info['local_inds'] = tuple(ten.phase.get('local_inds',[]))
        if fname is None:
            ten_info['tensor_data'] = ten.data.copy()
        else:
            ten_info['tensor_data'] = ten.data
        ten_info['tensor_inds'] = ten.inds
        ten_info['tensor_tags'] = ten.tags
        data['tensors'].append(ten_info)
        ntensors += 1
    data['ntensors'] = ntensors
    # Write to a file
    if fname is None:
        return data
    else:
        with open(fname, 'wb') as f:
            pickle.dump(data, f)
        return fname 
def _load_ftn_from_disc(fname, delete_file=False):
    # Open up the file
    if isinstance(fname,str):
        with open(fname, 'rb') as f:
            data = pickle.load(f)
        copy_data = False
    else:
        data = fname
        copy_data = True
    # Set up a dummy fermionic tensor network
    tn = FermionTensorNetwork([])
    # Put the tensors into the ftn
    tensors = [None,] * data['ntensors']
    for i in range(data['ntensors']):
        # Get the tensor
        ten_info = data['tensors'][i]
        ten_data = ten_info['tensor_data']
        inds = ten_info['tensor_inds']
        tags = ten_info['tensor_tags']
        global_flip = ten_info['global_flip']
        local_inds  = ten_info['local_inds']
        if copy_data:
            ten = FermionTensor(ten_data.copy(), inds=inds, tags=tags)
        else:
            ten = FermionTensor(ten_data, inds=inds, tags=tags)
        # Get/set tensor info
        tid, site = ten_info['fermion_info']
        ten.fermion_owner = None
        ten._avoid_phase = False
        # Add the required phase
        ten.phase = {'global_flip':global_flip,'local_inds':list(local_inds)}
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


##################################################################################
# some sum of op class for 2d-dmrg 
##################################################################################
class OPTERM:
    """
    Simple class that holds a single term in a operator.
    """
    def __init__(self, sites, ops, prefactor):
        """
        Creates an OPTERM object to hold a single operator

        Args:
            sites: list of tuples
                A list of sites that the operators are applied to, i.e.
                [(x1, y1), (x2, y2), ...]
            ops: a list
                The operators corresponding to each site in 'sites'
            prefactor: float
                A value to multiply against each operator. This is helpful
                for reusing intermediates because prefactors on hamiltonian
                terms do not need to be absorbed into the operators themselves.
        """
        self.sites = sites
        self.nops = len(self.sites)
        self.ops = dict(zip(sites, ops))
        self.prefactor = prefactor
        
        # Figure out the operator tags
        self.optags = dict()
        for site in self.sites:
            for tag in list(self.ops[site][2]):
                if tag[:7] == 'OPLABEL':
                    self.optags[site] = tag
                    break

    def get_op(self, site, op_dict):
        op_tag,inds,tags = self.ops[site]
        return FermionTensor(op_dict[op_tag].copy(),inds=inds,tags=tags)

    def copy(self):
        # Return a new OPTERM object with copies of the operator
        return OPTERM(self.sites,
                      [self.ops[site] for site in self.sites],
                      self.prefactor)
class SumOpDMRG:
    def __init__(self,op_dict,ham_terms):
        self.op_dict = op_dict
        self.ham_terms = ham_terms
        print('number of H terms=',len(self.ham_terms))

##################################################################################
# some helper ftn fxn
##################################################################################
def remove_phase(peps):
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            peps[i,j].phase = dict()
    return peps
def match_phase(T_tar,T_ref):
    # match phase of T_tar to T_ref
    global_flip_tar = T_tar.phase.get('global_flip',False)
    local_inds_tar = set(T_tar.phase.get('local_inds',[]))
    global_flip_ref = T_ref.phase.get('global_flip',False)
    local_inds_ref = set(T_ref.phase.get('local_inds',[]))

    data = T_tar.data.copy()
    global_flip = (global_flip_tar!=global_flip_ref)
    if global_flip:
        data._global_flip()
    local_inds = list(local_inds_tar.symmetric_difference(local_inds_ref))
    if len(local_inds)>0:
        axes = [T.inds.index(ind) for ind in local_inds]
        data._local_flip(axes)
    return data
def tensor_copy(tsr):
    new_tsr = FermionTensor(data=tsr.data.copy(),inds=tsr.inds,
                            tags=tsr.tags.copy(),left_inds=tsr.left_inds)
    new_tsr.avoid_phase = tsr.avoid_phase
    new_tsr.phase = tsr.phase.copy()
    return new_tsr
def copy(ftn,full=True):
    new_ftn = FermionTensorNetwork([])
    new_order = dict()
    for tid,tsr in ftn.tensor_map.items():
        site = tsr.get_fermion_info()[1]
        new_tsr = tensor_copy(tsr)
        # add to fs
        new_order[tid] = (new_tsr,site)
        # add to tn
        new_ftn.tensor_map[tid] = new_tsr
        new_tsr.add_owner(new_ftn,tid)
        new_ftn._link_tags(new_tsr.tags,tid)
        new_ftn._link_inds(new_tsr.inds,tid)
    new_fs = FermionSpace(tensor_order=new_order,virtual=True)
    if full:
        new_ftn.view_like_(ftn)
    return new_ftn
def insert(ftn,isite,T):
    fs = ftn.fermion_space
    fs.insert_tensor(isite,T,virtual=True)
    tid = T.get_fermion_info()[0]
    ftn.tensor_map[tid] = T 
    T.add_owner(ftn,tid)
    ftn._link_tags(T.tags,tid)
    ftn._link_inds(T.inds,tid)
    return ftn
def replace(ftn,isite,T):
    fs = ftn.fermion_space
    tid = fs.get_tid_from_site(isite)
    t = ftn.tensor_map.pop(tid)
    ftn._unlink_tags(t.tags,tid)
    ftn._unlink_inds(t.inds,tid)
    t.remove_owner(ftn)

    fs.replace_tensor(isite,T,tid=tid,virtual=True)    
    ftn.tensor_map[tid] = T 
    T.add_owner(ftn,tid)
    ftn._link_tags(T.tags,tid)
    ftn._link_inds(T.inds,tid)
    return ftn
