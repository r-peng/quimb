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
        self.ee1 = sum(self.g1)*4.0*np.pi/self.Omega 
        self.ee1 /= 2.0
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
            for j,site2 in enumerate(sites[i+1:]):
                if self.dist(site1,site2)<=maxdist:
                    assert site1[1]<=site2[1]
                    if site1[1]==site2[1]:
                        assert site1[0]<site2[0]
                    ke2,ee2 = self.compute_fac2(site1,site2)
                    ee2 /= 2.0
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
        ee2 = np.dot(cos,self.g1)*4.0*np.pi/self.Omega
        return ke2,ee2
    def dist(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        if self.dist_type=='graph':
            return abs(dx)+abs(dy)
        elif self.dist_type=='site':
            return np.sqrt(dx**2+dy**2)
        else:
            raise NotImplementedError('distant type {} not implemented'.format(
                                       self.dist_type))
class UEG_FD:
    def __init__(self,Nx,Ny,Lx,Ly,Ne,symmetry='u1',flat=True,
                 maxdist=1000,dist_type='graph'):
        assert Nx==Ny
        self.Nx = Nx
        self.Ny = Ny
        self.Ne = Ne
        self.eps = Lx/(Nx+2.) # spacing for ke terms
        self.h = Lx/(Nx+1.) # spacing for distance

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
        self.sites = [(x,y) for y in range(self.Ny) for x in range(self.Nx)]
        for i,site1 in enumerate(sites):
            pn1 = self.compute_u(site1)+self.compute_ke1(site1)
            ee1 = self.compute_lambda(site1)
            opis = OPi({'pn':pn1,'nanb':ee1},site1),
            self._1col_terms.append((opis,1.))
            for j,site2 in enumerate(sites[i+1:]):
                assert site1[1]<=site2[1]
                if site1[1]==site2[1]:
                    assert site1[0]<site2[0]
                dist = self.dist(site1,site2)
                tmp = []
                # long-range ee
                if dist<=maxdist:
                    opis = OPi({'pn':1.},(x1,y1)),OPi({'pn':1.},(x2,y2))
                    tmp.append((opis,1./self.phys_dist(site1,site2)))
                # NN/3rd-NN ke
                if dist==1 or dist==3:
                    ke2 = -16./(24.*eps**2) if dist==1 else 1./(24.*eps**2)
                    opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
                    tmp.append((opis,ke2*sign_a)) 
                    opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
                    tmp.append((opis,ke2)) 
                    opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
                    tmp.append((opis,ke2*sign_b)) 
                    opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
                    tmp.append((opis,ke2)) 
                if y1==y2:
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
    def compute_u(self,site1):
        ke1 = sum([1./self.phys_dist(site1,site2) for site2 in self.sites])
        return ke1 * self.Ne / (self.Nx*self.Ny)
    def compute_ke1(self,site):
        x,y = site
        if (x==0 and y==0) or (x==0 and y==self.Ny-1) or\
           (x==self.Nx-1 and y==0) or (x==self.Nx-1 and y==self.Ny-1):
            return 58./(24.*self.eps**2)
        elif x==0 or x==self.Nx-1 or y==0 or y==self.Ny-1:
            return 59./(24.*self.eps**2)
        else:
            return 60./(24.*self.eps**2)
    def compute_lambda(self,site1):
        return 1./self.phys_dist(site1,site1,const=1.)
    def dist(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        if self.dist_type=='graph':
            return abs(dx)+abs(dy)
        elif self.dist_type=='site':
            return np.sqrt(dx**2+dy**2)
        else:
            raise NotImplementedError('distant type {} not implemented'.format(
                                       self.dist_type))
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
    ftn_dict = ftn.compute_left_environments(yrange=(y2-1,Ly-1),**compress_opts) 

    benvs_ = dict()
    term = tag1,tag2
    for j in range(y2+1,Ly):
        benvs_[term,'left',j] = write_ftn_to_disc(ftn_dict['left',j],tmpdir)
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
    ftn_dict = ftn.compute_right_environments(yrange=(0,y1+1),**compress_opts) 

    benvs_ = dict()
    term = tag1,tag2
    for j in range(0,y1):
        benvs_[term,'right',j] = write_ftn_to_disc(ftn_dict['right',j],tmpdir)
    if profile:
        _profile(f'_2col_right')
    return benvs_
def _2col_benvs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = _2col_left if info[0]=='left' else _2col_right
    return fxn(info[1],benvs,tmpdir,Ly,profile,**compress_opts)
def _row_envs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    side,opis,j = info
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
    if side=='top':
        envs = ftn.compute_top_environments(
                   yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)
    else:
        envs = ftn.compute_bottom_environments(
                   yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)

    envs_ = dict()
    if side=='bottom':
        for i in range(2,ftn.Lx):
            envs_[term,j,'bottom',i] = write_ftn_to_disc(envs['bottom',i],tmpdir)
        for i in range(ftn.Lx-1):
            envs_[term,j,'mid',i] = write_ftn_to_disc(envs['mid',i],tmpdir)
    else: 
        for i in range(ftn.Lx-2):
            envs_[term,j,'top',i] = write_ftn_to_disc(envs['top',i],tmpdir)
        i = ftn.Lx-1
        envs_[term,j,'mid',i] = write_ftn_to_disc(envs['mid',i],tmpdir)
    if profile:
        _profile(f'_row_envs')
    return envs_
def _contract_plq(info,envs,Lx):
    opis,i,j,fac = info
    term = 'norm' if opis is None else tuple([opi.tag for opi in opis])
    term = term[0] if len(term)==1 else term
    if i<2:
        b = [] if i==0 else [envs[term,j,'mid',0]]
    else:
        b = [envs[term,j,'bottom',i]]
    if i>Lx-3:
        t = [] if i==Lx-1 else [envs[term,j,'mid',Lx-1]]
    else: 
        t = [envs[term,j,'top',i]]
    ls = b + [envs[term,j,'mid',i]] + t
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    site_tag = ftn.site_tag(i,j)
    ftn.select((site_tag,'BRA'),which='!all').add_tag('grad')
    bra = ftn[site_tag,'BRA']
    ftn.contract_tags('grad',which='any',inplace=True,
                      output_inds=bra.inds[::-1])
    assert ftn.num_tensors==2
    scal = ftn.contract()
    bra_tid = bra.get_fermion_info()[0]
    bra = ftn._pop_tensor(bra_tid,remove_from_fermion_space='end')
    data = ftn['grad'].data
    return (i,j),term,scal*fac,data*fac
def _site_grad(info):
    site,site_data_map = info
    H0,H1 = 0.0,0.0 # scalar/tsr H
    for term,(scal,data) in site_data_map.items():
        if term=='norm':
            N0,N1 = scal,data
        else:
            H0,H1 = H0+scal,H1+data
    E = H0/N0
    g = (H1-N1*E)/N0
    return site,g,E,N0
def compute_grad(H,psi,tmpdir,layer_tags=('KET','BRA'),profile=False,
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
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

    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    iterate_over += [(opi,) for (opi,_) in H.reuse]
    args = [H.data_map,norm,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _1col_benvs
    iterate_over  = [('left',opis,psi.Ly-1) for (opis,_) in H._1col_terms]
    iterate_over += [('right',opis,0)   for (opis,_) in H._1col_terms]
    iterate_over += [('left',(opi,),min(opi.site[1]+dy,psi.Ly-1)) \
                     for (opi,dy) in H.reuse]
    iterate_over += [('right',(opi,),max(opi.site[1]-dy,0)) for (opi,dy) in H.reuse]
    args = [benvs,tmpdir,psi.Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        if benvs_ is not None:
            benvs.update(benvs_)

    fxn = _2col_benvs
    iterate_over = [(side,opis) for (opis,_) in H._2col_terms \
                                for side in ['left','right']]
    args = [benvs,tmpdir,psi.Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _row_envs
    iterate_over  = [(side,None,j) for side in ['top','bottom'] \
                     for j in range(psi.Ly)]
    iterate_over += [(side,opis,j) for side in ['top','bottom'] \
                     for (opis,_) in H._1col_terms+H._2col_terms \
                     for j in range(psi.Ly)]
    args = [benvs,tmpdir,psi.Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    envs = dict()
    for envs_ in ls:
        envs.update(envs_)

    fxn = _contract_plq
    iterate_over = [(opis,i,j,fac) for (opis,fac) in H._1col_terms + H._2col_terms \
                     for i in range(psi.Lx) for j in range(psi.Ly)]
    iterate_over += [(None,i,j,1.) for i in range(psi.Lx) for j in range(psi.Ly)] 
    args = [envs,psi.Lx]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    data_map = dict()
    for (site,term,scal,data) in ls:
        if site not in data_map:
            data_map[site] = dict()
        data_map[site][term] = scal,data

    fxn = _site_grad
    iterate_over = [(key,val) for key,val in data_map.items()]
    args = []
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    g = dict()
    for (site,site_g,site_E,site_N) in ls:
        g[psi.site_tag(*site)] = site_g
        if site==(0,0):
            E = site_E
            N = site_N

    delete_ftn_from_disc(norm) 
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname)
    for _,fname in envs.items():
        delete_ftn_from_disc(fname)
    return g,E,N 
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
def compute_energy(H,psi,tmpdir,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    norm = psi.make_norm(layer_tags=('KET','BRA'))
    ftn_dict = norm.compute_right_environments(**compress_opts)
    benvs = dict()
    for (side,j),ftn in ftn_dict.items():
        benvs['norm',side,j] = write_ftn_to_disc(ftn,tmpdir)
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    benvs['norm','mid',0] = write_ftn_to_disc(norm.select(norm.col_tag(0)).copy(),
                                              tmpdir)

    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm = write_ftn_to_disc(norm,tmpdir)

    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    iterate_over += [(opi,) for (opi,_) in H.reuse]
    args = [H.data_map,norm,tmpdir,False]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _1col_right
    iterate_over  = [(opis,0)   for (opis,_) in H._1col_terms]
    iterate_over += [((opi,),max(opi.site[1]-dy,0)) for (opi,dy) in H.reuse]
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

    fxn = _energy_term
    iterate_over =  H._1col_terms + H._2col_terms + [(None,1.)]
    args = [benvs]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    E = 0.
    for (term,scal) in ls:
        if term=='norm':
            N = scal
        else:
            E += scal

    delete_ftn_from_disc(norm) 
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname) 
    return E/N,N
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
    def __init__(self,H,peps,chi,psi_fname,profile=False):
        self.start_time = time.time()
        self.H = H
        self.D = peps[peps.site_tag(0,0)].shape[0]
        self.chi = chi
        self.tmpdir = create_rand_tmpdir()
        self.profile = profile
        print(f'D={self.D},chi={self.chi}')

        N = compute_norm(peps,max_bond=chi)  
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
        E,N = compute_energy(self.H,psi,self.tmpdir,max_bond=self.chi,cutoff=1e-15)
        print('    ne={},time={}'.format(self.ne,time.time()-self.start_time))
        print('        E={},N={}'.format(E,N))
        self.ne += 1
        self.fac = N**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,N
    def compute_grad(self,x):
        psi = self.vec2fpeps(x)
        grad,E,N = compute_grad(self.H,psi,self.tmpdir,max_bond=self.chi,
                                cutoff=1e-15,profile=self.profile)
        E_,N_ = (E,N) if self.chi is None else \
             compute_energy(self.H,psi,self.tmpdir,max_bond=self.chi+5,cutoff=1e-15)

        g = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = self.constructors[site_tag][0]
                g.append(cons.tensor_to_vector(grad[site_tag]))
        g = np.concatenate(g)
        gmax = np.amax(abs(g))
        print('    ng={},time={}'.format(self.ng,time.time()-self.start_time))
        print('        E={},N={},gmax={}'.format(E,N,gmax))
        print('        E_err={},n_err={}'.format(abs((E-E_)/E_),abs((N-N_)/N_)))
        self.ng += 1
        self.fac = N**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,g
    def callback(self,x,g):
        x *= self.fac
        g /= self.fac
        psi = self.vec2fpeps(x)
        write_ftn_to_disc(psi,self.psi_fname,provided_filename=True)
        print('nfile=',len(os.listdir(self.tmpdir)))
        return x,g
    def kernel(self,method=_minimize_bfgs,options={'maxiter':200,'gtol':1e-5}):
        self.ng = 0
        self.ne = 0
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi_fname))
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
