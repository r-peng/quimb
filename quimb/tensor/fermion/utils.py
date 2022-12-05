import uuid,time,pickle
import numpy as np
import scipy.sparse.linalg as spla
from .fermion_core import FermionTensor, FermionTensorNetwork, tensor_contract
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
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
# MPI stuff
################################################################################    
def distribute(ntotal):
    batchsize,remain = ntotal // SIZE, ntotal % SIZE
    batchsizes = [batchsize] * SIZE
    for worker in range(SIZE-remain,SIZE):
        batchsizes[worker] += 1
    ls = [None] * SIZE
    start = 0
    for worker in range(SIZE):
        stop = start + batchsizes[worker]
        ls[worker] = start,stop
        start = stop
    return ls
def parallelized_looped_fxn(fxn,ls,args):
    stop = min(SIZE,len(ls))
    results = [None] * stop 
    for worker in range(stop-1,-1,-1):
        worker_info = fxn,ls[worker],args 
        if worker > 0:
            COMM.send(worker_info,dest=worker) 
        else:
            results[0] = fxn(ls[0],*args)
    for worker in range(1,stop):
        results[worker] = COMM.recv(source=worker)
    return results
def worker_execution():
    """
    Simple function for workers that waits to be given
    a function to call. Once called, it executes the function
    and sends the results back
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
        fxn,local_ls,args = assignment
        result = fxn(local_ls,*args)
        COMM.send(result, dest=0)
################################################################################
# gradient optimizer  
################################################################################    
class GradientAccumulator:

    def __init__(self,optimizer='sgd',learning_rate=1e-2,ovlp=False,hess=False):
        self._eloc = None
        self._glog = None
        self._glog_eloc = None
        self._num_samples = 0

        self.ovlp = ovlp
        self._glog_glog = None

        self.hess = hess
        self._geloc = None
        self._glog_glog_eloc = None
        self._glog_geloc = None

        self.optimizer = optimizer
        self.learning_rate = learning_rate
    def _init_storage(self, gx):
        size = len(gx)
        self._eloc = 0.0
        self._glog = np.zeros(size)
        self._glog_eloc = np.zeros(size)
        if self.ovlp:
            self._glog_glog = np.zeros((size,)*2)
        if self.hess:
            self._geloc = np.zeros(size)
            self._glog_geloc = np.zeros((size,)*2)

    def update(self, glx, ex, gex):
        if self._eloc is None:
            self._init_storage(glx)

        self._eloc += ex
        self._glog += glx 
        self._glog_eloc += glx * ex 
        if self.ovlp:
            self._glog_glog += np.outer(glx,glx)
        if self.hess:
            self._geloc += gex
            self._glog_geloc += np.outer(glx,gex)
        self._num_samples += 1
    def update_exact(self, glx, ex, gex,cx):
        if self._eloc is None:
            self._init_storage(glx)
            self._num_samples = 0.

        self._eloc += ex * cx**2
        self._glog += glx * cx**2
        self._glog_eloc += glx * ex * cx**2
        if self.ovlp:
            self._glog_glog += np.outer(glx,glx) * cx**2 
        if self.hess:
            self._geloc += gex * cx**2
            self._glog_geloc += np.outer(glx,gex) * cx**2
        self._num_samples += cx**2
    def update_from_worker(self,other):
        if self._eloc is None:
            self._init_storage(other._glog)

        self._eloc += other._eloc
        self._glog += other._glog
        self._glog_eloc += other._glog_eloc
        if self.ovlp:
            self._glog_glog += other._glog_glog
        if self.hess:
            self._geloc += other._geloc
            self._glog_geloc += other._glog_geloc
        self._num_samples += other._num_samples
    def extract_grads_energy(self):
        self._eloc /= self._num_samples
        self._glog /= self._num_samples
        self._glog_eloc /= self._num_samples
        g = self._glog_eloc - self._glog * self._eloc 
        sij,hij = None,None
        if self.ovlp:
            self._glog_glog /= self._num_samples
            sij = self._glog_glog - np.outer(self._glog,self._glog)
        if self.hess:
            self._geloc /= self._num_samples
            self._glog_geloc /= self._num_samples
            hij = self._glog_geloc - np.outer(self._glog,self._geloc)
            hij -= np.outer(g,self._glog)
        return g,sij,hij 
    def reset(self):
        self._eloc = 0.
        self._glog.fill(0.)
        self._glog_eloc.fill(0.)
        if self.ovlp:
            self._glog_glog.fill(0.)
        if self.hess:
            self._geloc.fill(0.)
            self._glog_geloc.fill(0.)
        self._num_samples = 0
    def transform_gradients(self):
        g,_,_ = self.extract_grads_energy()
        if self.optimizer=='sgd':
            return self.learning_rate * g
        elif self.optimizer=='sign':
            return self.learning_rate * np.sign(g)
        elif self.optimizer=='signu':
            return self.learning_rate * np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError
class Adam(GradientAccumulator):
    def __init__(self,learning_rate=1e-2,beta1=.9,beta2=.999,eps=1e-8):
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self._num_its = 0
        self._ms = None
        self._vs = None
        super().__init__(optimizer=None,learning_rate=learning_rate,
                         ovlp=False,hess=False)
    def transform_gradients(self):
        g,_,_ = self.extract_grads_energy()

        self._num_its += 1
        if self._num_its == 1:
            self._ms = np.zeros_like(g)
            self._vs = np.zeros_like(g)
    
        self._ms = (1.-self.beta1) * g + self.beta1 * self._ms
        self._vs = (1.-self.beta2) * g**2 + self.beta2 * self._vs 
        mhat = self._ms / (1. - self.beta1**(self._num_its))
        vhat = self._vs / (1. - self.beta2**(self._num_its))
        return self.learning_rate * mhat / (np.sqrt(vhat)+self.eps)
class SR(GradientAccumulator):
    def __init__(self,learning_rate=1e-2,delta=1e-5):
        self.delta = delta
        super().__init__(optimizer=None,learning_rate=learning_rate,
                         ovlp=True,hess=False)
    def transform_gradients(self):
        g,sij,_ = self.extract_grads_energy()
        idxs = tuple(range(len(g)))
        sij[idxs,idxs] += self.delta
        return self.learning_rate * np.linalg.solve(sij,g)
class RGN(GradientAccumulator):
    def __init__(self,learning_rate=1e-2,delta=1e-5):
        self.delta = delta
        super().__init__(optimizer=None,learning_rate=learning_rate,
                         ovlp=True,hess=True)
    def transform_gradients(self):
        g,sij,hij = self.extract_grads_energy()
        hij -= self._eloc * sij

        idxs = tuple(range(len(g)))
        sij[idxs,idxs] += self.delta
        return np.linalg.solve(hij+sij/self.learning_rate,g)
class LinOpt(GradientAccumulator):
    def __init__(self,learning_rate=1e-2,delta=1e-5,sr_damping=False):
        self.delta = delta
        self.sr_damping = sr_damping
        super().__init__(optimizer=None,learning_rate=learning_rate,
                         ovlp=True,hess=True)
    def transform_gradients(self):
        g,sij,hij = self.extract_grads_energy()
        hij -= self._eloc * sij

        size = len(g)
        idxs = tuple(range(size))
        #sij[idxs,idxs] += self.delta
        s = np.block([[np.ones((1,1)),np.zeros((1,size))],
                      [np.zeros((size,1)),sij]])     

        if self.sr_damping:
            sij[idxs,idxs] += self.delta
            hij += sij / self.learning_rate
        else:
            hij[idxs,idxs] += 1./self.learning_rate

        h = np.block([[np.zeros((1,1)),g.reshape(1,size)],
                      [g.reshape(size,1),hij]])
        dE,deltas = spla.eigs(h,M=s,k=1,sigma=h[0,0]) 
        print('dE=',dE)
        deltas = deltas[:,0].real
        scale = deltas[0] * (1. - np.dot(self._glog,deltas[1:]))
        return - deltas[1:] / scale 
def get_optimizer(optimizer_opts):
    _optimizer = optimizer_opts['optimizer']
    learning_rate = optimizer_opts.get('learning_rate',1e-2)
    delta = optimizer_opts.get('delta',1e-5)
    hess = False
    if _optimizer=='adam':
        beta1 = optimizer_opts.get('beta1',.9)
        beta2 = optimizer_opts.get('beta2',.999)
        eps = optimizer_opts.get('eps',1e-8)
        optimizer = Adam(learning_rate=learning_rate,
                         beta1=beta1,beta2=beta2,eps=eps)
    elif _optimizer=='sr':
        optimizer = SR(learning_rate=learning_rate,delta=delta)
    elif _optimizer=='lin':
        hess = True 
        sr_damping = optimizer_opts.get('sr_damping',False)
        optimizer = LinOpt(learning_rate=learning_rate,
                           delta=delta,sr_damping=sr_damping)
    elif _optimizer=='rgn':
        hess = True
        optimizer = RGN(learning_rate=learning_rate,delta=delta)
    else:
        optimizer = GradientAccumulator(optimizer=_optimizer,
                        learning_rate=learning_rate,ovlp=False,hess=False)
    return optimizer,hess
