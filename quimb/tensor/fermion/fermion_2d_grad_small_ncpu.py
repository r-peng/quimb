import numpy as np
import time,scipy,uuid,os,atexit,shutil,pickle,resource

from .minimize import (
    _minimize_bfgs
)
from .fermion_2d_tebd import (
    insert,
)
from .fermion_core import FermionTensor,FermionTensorNetwork
from .fermion_2d import FermionTensorNetwork2D
from .block_interface import (
    Constructor,
    creation,
    onsite_U,
    ParticleNumber,
)
#####################################################################################
# MPI STUFF
#####################################################################################
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

if RANK == 0:
    TMPDIR = os.environ.get('TMPDIR','.')
    if TMPDIR[-1] == '/':
        TMPDIR = TMPDIR[:-1]
    RANDTMPDIR = TMPDIR + '/' + str(uuid.uuid4()) + '/'
    print(f'Saving temporary files in: {RANDTMPDIR}')
    os.mkdir(RANDTMPDIR)

def create_rand_tmpdir():
    # This funciton is poorly named... it just returns the already created
    # temporary directory for this calculation
    return RANDTMPDIR

def rand_fname():
    return str(uuid.uuid4())

def clear_tmpdir(tmpdir):
    try:
        shutil.rmtree(tmpdir)
    except OSError as e:
        pass

if RANK == 0:
    atexit.register(clear_tmpdir, RANDTMPDIR)

def parallelized_looped_function(func, iterate_over, args, kwargs):
    """
    A parallelized loop (controlled by the rank 0 process)
    """
    # Figure out which items are done by which worker
    min_per_worker = len(iterate_over) // SIZE

    per_worker = [min_per_worker for _ in range(SIZE)]
    for i in range(len(iterate_over) - min_per_worker * SIZE):
        per_worker[SIZE-1-i] += 1

    randomly_permuted_tasks = np.random.permutation(len(iterate_over))
    worker_ranges = []
    for worker in range(SIZE):
        start = sum(per_worker[:worker])
        end = sum(per_worker[:worker+1])
        tasks = [randomly_permuted_tasks[ind] for ind in range(start, end)]
        worker_ranges.append(tasks)

    # Container for all the results
    worker_results = [None for _ in range(SIZE)]

    # Loop over all the processes (backwards so zero starts last
    for worker in reversed(range(SIZE)):

        # Collect all info needed for workers
        worker_iterate_over = [iterate_over[i] for i in worker_ranges[worker]]
        worker_info = [func, worker_iterate_over, args, kwargs]

        # Send to worker
        if worker != 0:
            COMM.send(worker_info, dest=worker)

        # Do task with this worker
        else:
            worker_results[0] = [None for _ in worker_ranges[worker]]
            for func_call in range(len(worker_iterate_over)):
                result = func(worker_iterate_over[func_call], 
                              *args, **kwargs)
                worker_results[0][func_call] = result

    # Collect all the results
    for worker in range(1, SIZE):
        worker_results[worker] = COMM.recv(source=worker)

    results = [None for _ in range(len(iterate_over))]
    for worker in range(SIZE):
        worker_ind = 0
        for i in worker_ranges[worker]:
            results[i] = worker_results[worker][worker_ind]
            worker_ind += 1

    # Return the results
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
        function = assignment[0]
        iterate_over = assignment[1]
        args = assignment[2]
        kwargs = assignment[3]
        results = [None for _ in range(len(iterate_over))]
        for func_call in range(len(iterate_over)):
            results[func_call] = function(iterate_over[func_call], 
                                          *args, **kwargs)
        COMM.send(results, dest=0)

def _profile(title):
    res = resource.getrusage(resource.RUSAGE_SELF)
    with open(f'./grad_log/cpu_{RANK}.log','a') as f:
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

#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################

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

##################################################################
# Hamiltonians
##################################################################
class OPi:
    def __init__(self,data_map,site):
        self.data_map = data_map
        self.site = site
        self.tag = '_'.join([key+'{},{}'.format(*site) for key in data_map])
    def get_data(self,data_map):
        data = 0.0
        for key,fac in self.data_map.items():
            data = data + fac*data_map[key]
        return data
class UEG_PWDB:
    def __init__(self,imax,jmax,Lx,Ly,maxdist=1000,dist_type='graph',
                 symmetry='u1',flat=True):
        self.Lx = Lx
        self.Ly = Ly
        self.Omega = Lx*Ly
        self.imax = imax # nx=0,...,+/-Nx
        self.jmax = jmax
        self.Nx = 2*imax+1
        self.Ny = 2*jmax+1
        self.N = self.Nx*self.Ny
        self.dist_type = dist_type

        self.g  = []
        self.g1 = [] # norm
        self.g2 = [] # normsq
        for gx in range(-imax,imax+1):
            for gy in range(-jmax,jmax+1):
                self.g.append(2.0*np.pi*np.array([gx/self.Lx,gy/self.Ly]))
                self.g2.append(np.dot(self.g[-1],self.g[-1]))
                g1_inv = 0.0 if (gx==0 and gy==0) else 1.0/np.sqrt(self.g2[-1])
                self.g1.append(g1_inv)
        self.ke1 = sum(self.g2)/(2.0*self.N)
        self.ee1 = sum(self.g1)*2.0*np.pi/self.Omega 
        self.g1 = np.array(self.g1)
        self.g2 = np.array(self.g2)

        cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
        cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
        ann_a = cre_a.dagger
        ann_b = cre_b.dagger
        pn    = ParticleNumber(symmetry=symmetry,flat=flat)
        nanb  = onsite_U(u=1.0,symmetry=symmetry)
        sign_a = (-1)**(cre_a.parity*ann_a.parity)
        sign_b = (-1)**(cre_b.parity*ann_b.parity)
        self.data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
                        'pn':pn,'nanb':nanb}

        self._1col_terms = []
        self._2col_terms = []
        sites = [(x,y) for y in range(self.Ny) for x in range(self.Nx)]
        for i,site1 in enumerate(sites):
            opis = OPi({'pn':self.ke1,'nanb':self.ee1},site1),
            self._1col_terms.append((opis,1.))
            for site2 in sites[i+1:]:
                if self.dist(site1,site2)<=maxdist:
                    assert site1[1]<=site2[1]
                    if site1[1]==site2[1]:
                        assert site1[0]<site2[0]
                    ke2,ee2 = self.compute_fac2(site1,site2)
                    tmp = []
                    if abs(ke2)>1e-12:
                        opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
                        tmp.append((opis,ke2*sign_a)) 
                        opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
                        tmp.append((opis,ke2)) 
                        opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
                        tmp.append((opis,ke2*sign_b)) 
                        opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
                        tmp.append((opis,ke2))
                    if abs(ee2)>1e-12:
                        opis = OPi({'pn':1.},site1),OPi({'pn':1.},site2)
                        tmp.append((opis,ee2))
                    if site1[1]==site2[1]:
                        self._1col_terms += tmp
                    else:
                        self._2col_terms += tmp
        keys = ['cre_a','ann_a','cre_b','ann_b','pn']
        self.reuse = [(OPi({key:1.},site),maxdist) \
                       for key in keys for site in sites] 
        print('number of 1 col terms=',len(self._1col_terms))
        print('number of 2 col terms=',len(self._2col_terms))
    def compute_fac2(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        r = np.array([dx*self.Lx/self.Nx,dy*self.Ly/self.Ny])
        cos = np.array([np.cos(np.dot(g,r)) for g in self.g])
        ke2 = np.dot(cos,self.g2)/(2.0*self.N)
        ee2 = np.dot(cos,self.g1)*2.0*np.pi/self.Omega
        return ke2,ee2
    def dist(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        if self.dist_type=='graph':
            return abs(dx)+abs(dy)
        elif self.dist_type=='site':
            return np.sqrt(dx**2+dy**2)
        else:
            raise NotImplementedError(f'distance type {self.dist_type} not implemented')

class UEG_FD:
    def __init__(self,Nx,Ny,Lx,Ly,Ne,scheme,symmetry='u1',flat=True,
                 maxdist=1000,dist_type='graph'):
        assert Nx==Ny
        self.Nx = Nx
        self.Ny = Ny
        self.eps = Lx/(Nx+2.) # spacing for ke terms
        self.h = Lx/(Nx+1.) # spacing for distance
        self.dist_type = dist_type
        sites = [(x,y) for y in range(self.Ny) for x in range(self.Nx)]
        den = (Ne+1e-15) / (Nx*Ny)
        if scheme==1:
            const = 1.
            def compute_lambda(site):
                return 1./self.h
            def compute_u(site1):
                u = sum([1./self.phys_dist(site1,site2,const=const) \
                        for site2 in sites])
                return - u * self.h**2 * den 
            background = 0.0
            for site1 in sites:
                for site2 in sites:
                    background += 1./self.phys_dist(site1,site2,const=const)
            background *= self.h**4*den**2/2.
        elif scheme==2:
            const = 0. 
            def compute_lambda(site):
                return 1.4866/self.h
            def compute_u(site1):
                sites_ = sites.copy()
                sites_.remove(site1)
                u = sum([1./self.phys_dist(site1,site2,const=const) \
                         for site2 in sites_])
                return - u * self.h**2 * den
            background = 0.0
            for i,site1 in enumerate(sites):
                for site2 in sites[i+1:]:
                    background += 1./self.phys_dist(site1,site2,const=const)
            background *= self.h**4*den**2
        else:
            raise NotImplementedError(f'scheme {scheme} not implemented')

        print('background=',background)
        print('analytical=',1.4866*Ne**2/Lx)
        cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
        cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
        ann_a = cre_a.dagger
        ann_b = cre_b.dagger
        pn = ParticleNumber(symmetry=symmetry,flat=flat)
        nanb = onsite_U(u=1.0,symmetry=symmetry)
        sign_a = (-1)**(cre_a.parity*ann_a.parity)
        sign_b = (-1)**(cre_b.parity*ann_b.parity)
        self.data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
                        'pn':pn,'nanb':nanb}

        self._1col_terms = []
        self._2col_terms = []
        for i,site1 in enumerate(sites):
            pn1 = compute_u(site1)+self.compute_ke1(site1)
            ee1 = compute_lambda(site1)
            opis = OPi({'pn':pn1,'nanb':ee1},site1),
            self._1col_terms.append((opis,1.))
#            print(site1,pn1,ee1)
            for site2 in sites[i+1:]:
                assert site1[1]<=site2[1]
                if site1[1]==site2[1]:
                    assert site1[0]<site2[0]
                tmp = []
                # long-range ee
                if self.dist(site1,site2)<=maxdist:
                    opis = OPi({'pn':1.},site1),OPi({'pn':1.},site2)
                    tmp.append((opis,1./self.phys_dist(site1,site2,const=const)))
#                    print(site1,site2,'ee',1./self.phys_dist(site1,site2,const=const))
                # NN/3rd-NN ke
                ke2 = self.compute_ke2(site1,site2)
                if ke2 is not None:
                    opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
                    tmp.append((opis,ke2*sign_a)) 
                    opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
                    tmp.append((opis,ke2)) 
                    opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
                    tmp.append((opis,ke2*sign_b)) 
                    opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
                    tmp.append((opis,ke2))
#                    print(site1,site2,'ke',ke2)
                if site1[1]==site2[1]:
                    self._1col_terms += tmp
                else:
                    self._2col_terms += tmp
        self.reuse = [(OPi({'pn':1.},site),maxdist) for site in sites] 
        keys = ['cre_a','ann_a','cre_b','ann_b']
        self.reuse += [(OPi({key:1.},site),3) for key in keys for site in sites] 
        print('number of 1 col terms=',len(self._1col_terms))
        print('number of 2 col terms=',len(self._2col_terms))
    def phys_dist(self,site1,site2,const=1.):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        return self.h*np.sqrt(dx**2+dy**2+const)
    def compute_ke1(self,site):
        x,y = site
        if (x==0 and y==0) or (x==0 and y==self.Ny-1) or\
           (x==self.Nx-1 and y==0) or (x==self.Nx-1 and y==self.Ny-1):
            return 58./(24.*self.eps**2)
        elif x==0 or x==self.Nx-1 or y==0 or y==self.Ny-1:
            return 59./(24.*self.eps**2)
        else:
            return 60./(24.*self.eps**2)
    def dist(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        if self.dist_type=='graph':
            return abs(dx)+abs(dy)
        elif self.dist_type=='site':
            return np.sqrt(dx**2+dy**2)
        else:
            raise NotImplementedError(f'distance type {self.dist_type} not implemented')
    def compute_ke2(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        if dx!=0 and dy!=0:
            return None
        else:
            dist = abs(dx)+abs(dy)
            if dist==1:
                return -16./(24.*self.eps**2)
            elif dist==3:
                return 1./(24.*self.eps**2)
            else:
                return None
class Hubbard:
    def __init__(self,t,u,Lx,Ly,symmetry='u1',flat=True):
        cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
        cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
        ann_a = cre_a.dagger
        ann_b = cre_b.dagger
        nanb  = onsite_U(u=1.0,symmetry=symmetry)
        sign_a = (-1)**(cre_a.parity*ann_a.parity)
        sign_b = (-1)**(cre_b.parity*ann_b.parity)
        self.data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
                         'nanb':nanb}
        self._1col_terms = []
        self._2col_terms = []
        for i in range(Lx):
            for j in range(Ly):
                opis = OPi({'nanb':1.},(i,j)),
                self._1col_terms.append((opis,u)) 
                if i+1 != Lx:
                    opis = OPi({'cre_a':1.},(i,j)),OPi({'ann_a':1.},(i+1,j))
                    self._1col_terms.append((opis,-t*sign_a)) 
                    opis = OPi({'ann_a':1.},(i,j)),OPi({'cre_a':1.},(i+1,j))
                    self._1col_terms.append((opis,-t)) 
                    opis = OPi({'cre_b':1.},(i,j)),OPi({'ann_b':1.},(i+1,j))
                    self._1col_terms.append((opis,-t*sign_b)) 
                    opis = OPi({'ann_b':1.},(i,j)),OPi({'cre_b':1.},(i+1,j))
                    self._1col_terms.append((opis,-t)) 
                if j+1 != Ly:
                    opis = OPi({'cre_a':1.},(i,j)),OPi({'ann_a':1.},(i,j+1))
                    self._2col_terms.append((opis,-t*sign_a))             
                    opis = OPi({'ann_a':1.},(i,j)),OPi({'cre_a':1.},(i,j+1))
                    self._2col_terms.append((opis,-t))                    
                    opis = OPi({'cre_b':1.},(i,j)),OPi({'ann_b':1.},(i,j+1))
                    self._2col_terms.append((opis,-t*sign_b))             
                    opis = OPi({'ann_b':1.},(i,j)),OPi({'cre_b':1.},(i,j+1))
                    self._2col_terms.append((opis,-t)) 
        keys = ['cre_a','ann_a','cre_b','ann_b']
        self.reuse = [(OPi({key:1.},(i,j)),1) for key in keys \
                       for i in range(Lx) for j in range(Ly)] 
        print('number of 1 col terms=',len(self._1col_terms))
        print('number of 2 col terms=',len(self._2col_terms))

#############################################################
# gradient functions
#############################################################
def _norm_benvs(side,norm,tmpdir,profile,**compress_opts):
    norm = load_ftn_from_disc(norm)
    if side == 'left':
        benvs = norm.compute_left_environments(**compress_opts)
    else:
        benvs = norm.compute_right_environments(**compress_opts)
    benvs_ = dict()
    if side=='left':
        for j in range(2,norm.Ly):
            benvs_['norm','left',j] = write_ftn_to_disc(benvs['left',j],tmpdir)
        for j in range(norm.Ly-1):
            benvs_['norm','mid',j] = write_ftn_to_disc(benvs['mid',j],tmpdir)
    else:
        for j in range(norm.Ly-2):
            benvs_['norm','right',j] = write_ftn_to_disc(benvs['right',j],tmpdir)
        j = norm.Ly-1
        benvs_['norm','mid',j] = write_ftn_to_disc(benvs['mid',j],tmpdir)
    if profile:
        _profile('_norm_benvs')
    return benvs_ 
def _1col_mid(opis,data_map,norm,tmpdir,profile):
    ftn = load_ftn_from_disc(norm)

    N = ftn.num_tensors//2
    site_range = (N,max(ftn.fermion_space.sites)+1)
    tsrs = []
    for opi in opis:
        ket = ftn[ftn.site_tag(*opi.site),'KET']
        pix = ket.inds[-1] 
        TG = FermionTensor(data=opi.get_data(data_map),tags=ket.tags,
                           inds=(pix,pix+'_'),left_inds=(pix,))
        tsrs.append(ftn.fermion_space.move_past(TG,site_range))

    ftn.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    for TG,opi in zip(tsrs,opis):
        bra = ftn[ftn.site_tag(*opi.site),'BRA']
        bra_tid,bra_site = bra.get_fermion_info()
        site_range = (bra_site,max(ftn.fermion_space.sites)+1)
        TG = ftn.fermion_space.move_past(TG,site_range)

        ket = ftn[ftn.site_tag(*opi.site),'KET']
        pix = ket.inds[-1] 
        ket.reindex_({pix:pix+'_'})
        ket_tid,ket_site = ket.get_fermion_info()
        ftn = insert(ftn,ket_site+1,TG)
        ftn.contract_tags(ket.tags,which='all',inplace=True)

    y = opis[0].site[1]
    term = tuple([opi.tag for opi in opis])
    term = term[0] if len(term)==1 else term
    ftn = write_ftn_to_disc(ftn.select(ftn.col_tag(y)).copy(),tmpdir)
    if profile:
        _profile(f'_1col_mid')
    return {(term,'mid',y):ftn}
def _1col_left(info,benvs,tmpdir,Ly,profile,**compress_opts):
    opis,ix = info
    y = opis[0].site[1]
    if y<Ly-1:
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        benvs_ = dict()

        if y<2:
            l = [] if y==0 else [benvs['norm','mid',0]]
        else:
            l = [benvs['norm','left',y]]
        ls = l + [benvs[term,'mid',y]]
        ls += [benvs['norm','mid',j] for j in range(y+1,ix+1)]
        ls = [load_ftn_from_disc(fname) for fname in ls]
        like = ls[0] if ls[0].num_tensors>0 else ls[1]
        ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

        first_col = ftn.col_tag(0)
        j0 = y+2 if y==0 else y+1
        for j in range(j0,ix+1):
            ftn.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,ftn.Lx-1),
                                             **compress_opts) 
            benvs_[term,'left',j] = write_ftn_to_disc(ftn.select(first_col).copy(),
                                                      tmpdir)
        if profile:
            _profile(f'_1col_left')
        return benvs_
def _1col_right(info,benvs,tmpdir,Ly,profile,**compress_opts):
    opis,ix = info
    y = opis[0].site[1]
    if y>0:
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        benvs_ = dict()

        ls = [benvs['norm','mid',j] for j in range(ix,y)]
        if y>Ly-3:
            r = [] if y==Ly-1 else [benvs['norm','mid',Ly-1]]
        else: 
            r = [benvs['norm','right',y]]  
        ls += [benvs[term,'mid',y]] + r
        ls = [load_ftn_from_disc(fname) for fname in ls]
        like = ls[0] if ls[0].num_tensors>0 else ls[1]
        ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

        last_col = ftn.col_tag(Ly-1)
        j0 = y-2 if y==Ly-1 else y-1
        for j in range(j0,ix-1,-1):
            ftn.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,ftn.Lx-1),
                                              **compress_opts) 
            benvs_[term,'right',j] = write_ftn_to_disc(ftn.select(last_col).copy(),
                                                       tmpdir)
        if profile:
            _profile(f'_1col_right')
        return benvs_ 
def _1col_benvs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = _1col_left if info[0]=='left' else _1col_right
    return fxn(info[1:],benvs,tmpdir,Ly,profile,**compress_opts)
def _2col_left(opis,benvs,tmpdir,Ly,profile,**compress_opts):
    (x1,y1),(x2,y2) = [opi.site for opi in opis]
    tag1,tag2 = [opi.tag for opi in opis]
    if y2==1:
        ls = [benvs[tag1,'mid',0],benvs[tag2,'mid',y2]]
    else:
        ls = [benvs[tag1,'left',y2],benvs[tag2,'mid',y2]] 
    ls += [benvs['norm','mid',j] for j in range(y2+1,Ly)]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    benvs = ftn.compute_left_environments(yrange=(y2-1,Ly-1),**compress_opts) 

    benvs_ = dict()
    term = tag1,tag2
    for j in range(y2+1,Ly):
        benvs_[term,'left',j] = write_ftn_to_disc(benvs['left',j],tmpdir)
    if profile:
        _profile(f'_2col_left')
    return benvs_
def _2col_right(opis,benvs,tmpdir,Ly,profile,**compress_opts):
    tag1,tag2 = [opi.tag for opi in opis]
    (x1,y1),(x2,y2) = [opi.site for opi in opis]
    ls = [benvs['norm','mid',j] for j in range(y1)]
    if y1==Ly-2:
        ls += [benvs[tag1,'mid',y1],benvs[tag2,'mid',Ly-1]]
    else:
        ls += [benvs[tag1,'mid',y1],benvs[tag2,'right',y1]]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    benvs = ftn.compute_right_environments(yrange=(0,y1+1),**compress_opts) 

    benvs_ = dict()
    term = tag1,tag2
    for j in range(0,y1):
        benvs_[term,'right',j] = write_ftn_to_disc(benvs['right',j],tmpdir)
    if profile:
        _profile(f'_2col_right')
    return benvs_
def _2col_benvs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = _2col_left if info[0]=='left' else _2col_right
    return fxn(info[1],benvs,tmpdir,Ly,profile,**compress_opts)
def _row_envs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    opis,j,fac = info
    if opis is None: # norm term
        term = 'norm'
        if j<2:
            l = [] if j==0 else [benvs['norm','mid',0]]
        else:
            l = [benvs['norm','left',j]]
        if j>Ly-3:
            r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
        else: 
            r = [benvs['norm','right',j]]
        ls = l + [benvs['norm','mid',j]] + r
    else: # H terms
        sites = [opi.site for opi in opis]
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        if (len(sites)==2) and (sites[0][1]<sites[1][1]): # 2col terms 
            tag1,tag2 = term
            (x1,y1),(x2,y2) = sites
            if j<y1:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                ls = l + [benvs['norm','mid',j],benvs[term,'right',j]]
            elif j==y1:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                r = [benvs[tag2,'mid',Ly-1]] if j==Ly-2 else \
                    [benvs[tag2,'right',j]]
                ls = l + [benvs[tag1,'mid',j]] + r
            elif j<y2: # y1<j<y2
                l = benvs[tag1,'mid',0] if j==1 else benvs[tag1,'left',j]
                r = benvs[tag2,'mid',Ly-1] if j==Ly-2 else \
                    benvs[tag2,'right',j]
                ls = l,benvs['norm','mid',j],r
            elif j==y2:
                l = [benvs[tag1,'mid',0]] if j==1 else [benvs[tag1,'left',j]]
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = l + [benvs[tag2,'mid',j]] + r
            else: # j>y2
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = [benvs[term,'left',j],benvs['norm','mid',j]] + r
        else: # 1col terms
            y = sites[0][1]
            if j<y:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                r = [benvs[term,'mid',Ly-1]] if j==Ly-2 else \
                    [benvs[term,'right',j]]
                ls = l + [benvs['norm','mid',j]] + r
            elif j==y:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = l + [benvs[term,'mid',j]] + r
            else: # j>y
                l = [benvs[term,'mid',0]] if j==1 else [benvs[term,'left',j]]
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = l + [benvs['norm','mid',j]] + r
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    envs = ftn.compute_row_environments(
               yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)

    data_map = dict()
    for i in range(ftn.Lx):
        ftn_ij = FermionTensorNetwork(
              [envs[side,i] for side in ['bottom','mid','top']]
              ).view_as_(FermionTensorNetwork2D,like=ftn)
        site_tag = ftn_ij.site_tag(i,j)
        ftn_ij.select((site_tag,'BRA'),which='!all').add_tag('grad')
        bra = ftn_ij[site_tag,'BRA']
        ftn_ij.contract_tags('grad',which='any',inplace=True,
                          output_inds=bra.inds[::-1])
        assert ftn_ij.num_tensors==2
        scal = ftn_ij.contract()
        bra_tid = bra.get_fermion_info()[0]
        bra = ftn_ij._pop_tensor(bra_tid,remove_from_fermion_space='end')
        data = ftn_ij['grad'].data
        data_map[(i,j),term] = scal*fac,data*fac
    if profile:
        _profile(f'_row_envs')
    envs = None
    return data_map
def compute_grad(H,psi,tmpdir,profile=False,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm = write_ftn_to_disc(norm,tmpdir)

    fxn = _norm_benvs
    iterate_over = ['left','right']
    args = [norm,tmpdir,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for benvs_ in ls:
        benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    # treate 1col terms
    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    args = [H.data_map,norm,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _1col_benvs
    iterate_over  = [('left', opis,psi.Ly-1) for (opis,_) in H._1col_terms]
    iterate_over += [('right',opis,0) for (opis,_) in H._1col_terms]
    args = [benvs,tmpdir,psi.Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        if benvs_ is not None:
            benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _row_envs
    iterate_over  = [(None,j,1.0) for j in range(psi.Ly)]
    iterate_over += [(opis,j,fac) for j in range(psi.Ly)
                                  for (opis,fac) in H._1col_terms]
    args = [benvs,tmpdir,psi.Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    H0 = {(i,j):0.0 for i in range(psi.Lx) for j in range(psi.Ly)}
    H1 = {(i,j):0.0 for i in range(psi.Lx) for j in range(psi.Ly)}
    N0 = dict() 
    N1 = dict()
    for data_map in ls:
        for ((i,j),term),(scal,data) in data_map.items():
            if term == 'norm':
                N0[i,j] = scal
                N1[i,j] = data
            else:
                H0[i,j] = H0[i,j] + scal
                H1[i,j] = H1[i,j] + data
    compress_opts.pop('dense')
    keys = [key for key in benvs if key[0]!='norm']
    for key in keys:
        fname = benvs.pop(key)
        delete_ftn_from_disc(fname)
#    print('nfile=',len(os.listdir(tmpdir)))

    # treat 2col terms
    fxn = _1col_mid
    iterate_over = [(opi,) for (opi,_) in H.reuse]
    args = [H.data_map,norm,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    delete_ftn_from_disc(norm) 
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _1col_benvs
    iterate_over  = [('left',(opi,),min(opi.site[1]+dy,psi.Ly-1)) \
                     for (opi,dy) in H.reuse]
    iterate_over += [('right',(opi,),max(opi.site[1]-dy,0)) for (opi,dy) in H.reuse]
    args = [benvs,tmpdir,psi.Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        if benvs_ is not None:
            benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _2col_benvs
    iterate_over = [(side,opis) for (opis,_) in H._2col_terms \
                                for side in ['left','right']]
    args = [benvs,tmpdir,psi.Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _row_envs
    iterate_over = [(opis,j,fac) for j in range(psi.Ly)
                     for (opis,fac) in H._2col_terms]
    args = [benvs,tmpdir,psi.Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for data_map in ls:
        for ((i,j),_),(scal,data) in data_map.items():
            H0[i,j] = H0[i,j] + scal
            H1[i,j] = H1[i,j] + data
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname)

    g = dict()
    E = []
    for i in range(psi.Lx):
        for j in range(psi.Ly):
            E.append(H0[i,j]/N0[i,j])
            g[psi.site_tag(i,j)] = (H1[i,j]-N1[i,j]*E[-1])/N0[i,j]
    dE = max(E)-min(E)
    E = sum(E)/len(E)
    N = N0[0,0]
    H0 = N0 = H1 = N1 = None
    return g,E,N,dE 
def _energy_term(info,benvs,**compress_opts):
    opis,fac = info
    j = 0 
    if opis is None: # norm term
        term = 'norm'
        ls = [benvs['norm',side,j] for side in ['mid','right']]
    else: # H terms
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        sites = [opi.site for opi in opis]
        if (len(sites)==2) and (sites[0][1]<sites[1][1]): # 2col terms 
            tag1,tag2 = term 
            (x1,y1),(x2,y2) = sites
            if j<y1:
                ls = benvs['norm','mid',j],benvs[term,'right',j]
            elif j==y1:
                ls = benvs[tag1,'mid',j],benvs[tag2,'right',j]
            elif j<y2:
                ls = benvs['norm','mid',j],benvs[tag2,'right',j]
            elif j==y2:
                ls = benvs[tag2,'mid',j],benvs['norm','right',j]
            else: # j>y2
                ls = benvs['norm','mid',j],benvs['norm','right',j]
        else: # 1col terms
            y = sites[0][1]
            if j<y:
                ls = benvs['norm','mid',j],benvs[term,'right',j]
            elif j==y:
                ls = benvs[term,'mid',j],benvs['norm','right',j]
            else: # j>y
                ls = benvs['norm','mid',j],benvs['norm','right',j]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    return term,ftn.contract()*fac
def compute_energy(H,psi,tmpdir,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    norm = psi.make_norm(layer_tags=('KET','BRA'))
    benvs_ = norm.compute_right_environments(**compress_opts)
    benvs = dict()
    for (side,j),ftn in benvs_.items():
        benvs['norm',side,j] = write_ftn_to_disc(ftn,tmpdir)
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    benvs['norm','mid',0] = write_ftn_to_disc(norm.select(norm.col_tag(0)).copy(),
                                              tmpdir)

    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm = write_ftn_to_disc(norm,tmpdir)

    # treate 1col terms
    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    args = [H.data_map,norm,tmpdir,False]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _1col_right
    iterate_over  = [(opis,0)   for (opis,_) in H._1col_terms]
    args = [benvs,tmpdir,psi.Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        if benvs_ is not None:
            benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _energy_term
    iterate_over = H._1col_terms + [(None,1.)]
    args = [benvs]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    H0 = 0.
    for (term,scal) in ls:
        if term == 'norm':
            N = scal
        else:
            H0 += scal
    compress_opts.pop('dense')
    keys = [key for key in benvs if key[0]!='norm']
    for key in keys:
        fname = benvs.pop(key)
        delete_ftn_from_disc(fname)
#    print('nfile=',len(os.listdir(tmpdir)))

    # treat 2col terms
    fxn = _1col_mid
    iterate_over = [(opi,) for (opi,_) in H.reuse]
    args = [H.data_map,norm,tmpdir,False]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    delete_ftn_from_disc(norm) 
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _1col_right
    iterate_over = [((opi,),max(opi.site[1]-dy,0)) for (opi,dy) in H.reuse]
    args = [benvs,tmpdir,psi.Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        if benvs_ is not None:
            benvs.update(benvs_)

    fxn = _2col_right
    iterate_over = [opis for (opis,_) in H._2col_terms]
    args = [benvs,tmpdir,psi.Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
#    print('nfile=',len(os.listdir(tmpdir)))

    fxn = _energy_term
    iterate_over =  H._2col_terms
    args = [benvs]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for _,e in ls:
        H0 += e
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname) 
    return H0/N,N
def _pn_site(info,envs,Lx):
    (i,j),data = info
    b = [] if i==0 else [envs['norm',j,'bottom',i]]
    t = [] if i==Lx-1 else [envs['norm',j,'top',i]]
    ls = b + [envs['norm',j,'mid',i]] + t
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

    N = ftn.contract()
    ket = ftn[ftn.site_tag(i,j),'KET']
    pix = ket.inds[-1]
    TG = FermionTensor(data=data.copy(),inds=(pix,pix+'_'),left_inds=(pix,))
    ket.reindex_({pix:pix+'_'})
    ftn.add_tensor(TG,virtual=True)
    return ftn.contract()/N,N
class PN:
    def __init__(self,Nx,Ny,symmetry='u1',flat=True):
        self.data_map = {'pn': ParticleNumber(symmetry=symmetry,flat=flat)}
        self._2col_terms = []
        self.reuse = []
        self._1col_terms = []
        for i in range(Nx):
            for j in range(Ny):
                opis = OPi({'pn':1.},(i,j)),
                self._1col_terms.append((opis,1.))
def compute_particle_number(psi,tmpdir,symmetry='u1',flat=True,
    dense_row=True,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    H = PN(psi.Lx,psi.Ly,symmetry=symmetry,flat=flat)
    return compute_energy(H,psi,tmpdir,dense_row=dense_row,**compress_opts)
def compute_norm(psi,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    ftn_dict = norm.compute_right_environments(**compress_opts)
    ftn = FermionTensorNetwork(
          (norm.select(norm.col_tag(0)).copy(),ftn_dict['right',0])
          ).view_as_(FermionTensorNetwork2D,like=norm)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    return ftn.contract() 
class GlobalGrad():
    def __init__(self,H,peps,chi,psi_fname,profile=False,dense_row=True):
        self.start_time = time.time()
        self.H = H
        self.D = peps[peps.site_tag(0,0)].shape[0]
        self.chi = chi
        self.profile = profile
        self.dense_row = dense_row
        self.tmpdir = RANDTMPDIR 
        print(f'D={self.D},chi={self.chi}')

        PN,N = compute_particle_number(peps,self.tmpdir,max_bond=128)  
        print('PN=',PN)
        peps = peps.multiply_each(N**(-1.0/(2.0*peps.num_tensors)),inplace=True)
        peps.balance_bonds_()
        peps.equalize_norms_()
        self.fac = 1.0

        self.psi_fname = psi_fname
        write_ftn_to_disc(peps,self.psi_fname, provided_filename=True) # save state
        self.constructors = dict()
        for i in range(peps.Lx):
            for j in range(peps.Ly):
                site_tag = peps.site_tag(i,j)
                data = peps[site_tag].data
                bond_infos = [data.get_bond_info(ax,flip=False) \
                              for ax in range(data.ndim)]
                cons = Constructor.from_bond_infos(bond_infos,data.pattern)
                self.constructors[site_tag] = cons,data.dq
        self.ng = 0
        self.ne = 0
    def fpeps2vec(self,psi):
        ls = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = self.constructors[site_tag][0]
                ls.append(cons.tensor_to_vector(psi[site_tag].data))
        return np.concatenate(ls)
    def vec2fpeps(self,x):
        psi = load_ftn_from_disc(self.psi_fname)
        start = 0
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons,dq = self.constructors[site_tag]
                stop = int(start+cons.get_info(dq)[-1][-1]+1e-6)
                psi[site_tag].modify(data=cons.vector_to_tensor(x[start:stop],dq))
                start = stop
        return psi
    def compute_energy(self,x):
        psi = self.vec2fpeps(x)
        E,N = compute_energy(self.H,psi,self.tmpdir,dense_row=self.dense_row,
                               max_bond=self.chi,cutoff=1e-15)
        print('    ne={},time={}'.format(self.ne,time.time()-self.start_time))
        print('        E={},N={}'.format(E,N))
        self.ne += 1
        self.fac = N**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,N
    def compute_grad(self,x):
        psi = self.vec2fpeps(x)
        grad,E,N,dE = compute_grad(self.H,psi,self.tmpdir,profile=self.profile,
            dense_row=self.dense_row,max_bond=self.chi,cutoff=1e-15)
        g = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = self.constructors[site_tag][0]
                g.append(cons.tensor_to_vector(grad[site_tag]))
        g = np.concatenate(g)
        gmax = np.amax(abs(g))
        print('    ng={},time={}'.format(self.ng,time.time()-self.start_time))
        print('        E={},N={},gmax={},dE={}'.format(E,N,gmax,dE))
        self.ng += 1
        self.fac = N**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,g
    def callback(self,x,g):
        x *= self.fac
        g /= self.fac
        psi = self.vec2fpeps(x)
        write_ftn_to_disc(psi,self.psi_fname,provided_filename=True)
        return x,g
    def kernel(self,method=_minimize_bfgs,options={'maxiter':200,'gtol':1e-5}):
        self.ng = 0
        self.ne = 0
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi_fname))
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
