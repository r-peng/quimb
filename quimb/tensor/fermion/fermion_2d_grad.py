import numpy as np
import time,scipy

from .minimize import (
    _minimize_bfgs
)
from .fermion_2d_tebd import (
    insert,
    match_phase,
    write_ftn_to_disc,
    load_ftn_from_disc,
    delete_ftn_from_disc,
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
##################################################################
# Hamiltonians
##################################################################
class UEG:
    def __init__(self,Nx,Ny,Lx,Ly,Ne,maxdist=1000,dist_type='graph',
                 symmetry='u1',flat=True):
        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny
        self.max_dy = Ny if maxdist>Ny else int(maxdist)
        self.dist_type = dist_type
        self.n_dict,self.ke1,self.ee1 = self.compute_fac1()

        cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
        cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
        ann_a = cre_a.dagger
        ann_b = cre_b.dagger
        pn = ParticleNumber(spin='sum',symmetry=symmetry,flat=flat)
        nanb = onsite_U(u=1.0,symmetry=symmetry)
        onsite = pn*self.ke1 + nanb*self.ee1
        self.op_dict = dict()
        self.op_dict['cre_a'] = cre_a
        self.op_dict['ann_a'] = ann_a
        self.op_dict['cre_b'] = cre_b
        self.op_dict['ann_b'] = ann_b
        self.op_dict['pn'] = pn
        self.op_dict['onsite'] = onsite

        # useful 2col info
        self._op_tags = ['cre_a','ann_a','cre_b','ann_b','pn']
        self._sites = set()

        self._1col_terms = []
        self._2col_terms = []
        sites = [(x,y) for y in range(Ny) for x in range(Nx)]
        sign_a = (-1)**(cre_a.parity*ann_a.parity)
        sign_b = (-1)**(cre_b.parity*ann_b.parity)
        for i in range(len(sites)):
            self._1col_terms.append(([sites[i]],['onsite'],1.0))
            x1,y1 = sites[i]
            for j in range(i+1,len(sites)):
                x2,y2 = sites[j]
                if self.dist(sites[i],sites[j])<=maxdist:
                    assert y1<=y2
                    if y1==y2:
                        assert x1<x2
                    ke2,ee2 = self.compute_fac2(sites[i],sites[j])
                    where = sites[i],sites[j]

                    tmp = [(where,['cre_a','ann_a'],ke2*sign_a),\
                           (where,['ann_a','cre_a'],ke2),\
                           (where,['cre_b','ann_b'],ke2*sign_b),\
                           (where,['ann_b','cre_b'],ke2),\
                           (where,['pn','pn'],ee2)]
                    if y1==y2:
                        self._1col_terms += tmp
                    else:
                        self._2col_terms += tmp
                        self._sites.update({sites[i],sites[j]})
        self._sites = list(self._sites)
        print('number of 1 col terms=',len(self._1col_terms))
        print('number of 2 col terms=',len(self._2col_terms))
    def compute_fac1(self):
        imax = self.Nx//2
        jmax = self.Ny//2
        n_dict = dict()
        for nx in range(-imax,-imax+self.Nx):
            for ny in range(-jmax,-jmax+self.Ny):
                kn = 2.0*np.pi*np.array([nx/self.Lx,ny/self.Ly])
                normsq = np.dot(kn,kn)
                norm = np.sqrt(normsq)
                n_dict[nx,ny] = {'k':kn,'norm':norm,'normsq':normsq}
        print('imax={},jmax={}'.format(imax,jmax))
        print('number of plane waves=',len(n_dict))
        # kinetic onsite prefactor
        ke1 = sum([kn_dict['normsq'] for kn_dict in n_dict.values()])
#        ke1 /= 2.0*len(n_dict)
        ke1 /= 2.0*(imax*jmax)
        # e-e onsite prefactor
        ee1 = sum([1.0/kn_dict['norm'] \
                   for (nx,ny),kn_dict in n_dict.items() if not (nx==0 and ny==0)])
        ee1 *= 2.0*2.0*np.pi/(self.Lx*self.Ly)
        return n_dict,ke1,ee1
    def compute_fac2(self,site1,site2):
        (x1,y1),(x2,y2) = site1,site2
        dx,dy = x1-x2,y1-y2
        imax = self.Nx//2
        jmax = self.Ny//2
#        r = np.array([dx*self.Lx/self.Nx,dy*self.Ly/self.Ny])
        r = np.array([dx*self.Lx/imax,dy*self.Ly/jmax])
        cos_dict = dict()
        for (nx,ny),kn_dict in self.n_dict.items():
            cos_dict[nx,ny] = np.cos(np.dot(kn_dict['k'],r))
        # kinetic
        ke2 = sum([kn_dict['normsq']*cos_dict[nx,ny] \
                   for (nx,ny),kn_dict in self.n_dict.items()])
#        ke2 /= 2.0*len(self.n_dict) 
        ke2 /= 2.0*(imax*jmax)
        # e-e
        ee2 = sum([cos_dict[nx,ny]/kn_dict['norm']\
                   for (nx,ny),kn_dict in self.n_dict.items() \
                   if not (nx==0 and ny==0)])
        ee2 *= 2.0*2.0*np.pi/(self.Lx*self.Ly)
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
#############################################################
# gradient functions
#############################################################
def _norm_benvs(side_,norm_fname,directory,**compress_opts):
    norm = load_ftn_from_disc(norm_fname)
    if side_ == 'left':
        ftn_dict = norm.compute_left_environments(**compress_opts)
    else:
        ftn_dict = norm.compute_right_environments(**compress_opts)
    fname_dict = dict()
    str_ = None if directory is None else directory+'_norm_'
    if side_=='left':
        for j in range(2,norm.Ly):
            fname = None if directory is None else str_+'left'+str(j)
            fname = write_ftn_to_disc(ftn_dict['left',j],fname)
            fname_dict['norm','left',j] = fname
        for j in range(norm.Ly-1):
            fname = None if directory is None else str_+'mid'+str(j)
            fname = write_ftn_to_disc(ftn_dict['mid',j],fname)
            fname_dict['norm','mid',j] = fname
    else:
        for j in range(norm.Ly-2):
            fname = None if directory is None else str_+'right'+str(j)
            fname = write_ftn_to_disc(ftn_dict['right',j],fname)
            fname_dict['norm','right',j] = fname
        j = norm.Ly-1
        fname = None if directory is None else str_+'mid'+str(j)
        fname = write_ftn_to_disc(ftn_dict['mid',j],fname)
        fname_dict['norm','mid',j] = fname
    return fname_dict
def _1col_mid(info,op_dict,norm_fname,directory):
    op_tags,xs,y = info
    ftn = load_ftn_from_disc(norm_fname)

    N = ftn.num_tensors//2
    site_range = (N,max(ftn.fermion_space.sites)+1)
    tsrs = []
    for op_tag,x in zip(op_tags,xs):
        ket = ftn[ftn.site_tag(x,y),'KET']
        pix = ket.inds[-1]
        TG = FermionTensor(op_dict[op_tag].copy(),inds=(pix,pix+'_'),
                           left_inds=(pix,),tags=ket.tags)
        tsrs.append(ftn.fermion_space.move_past(TG,site_range))

    ftn.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    for TG,x in zip(tsrs,xs):
        bra = ftn[ftn.site_tag(x,y),'BRA']
        bra_tid,bra_site = bra.get_fermion_info()
        site_range = (bra_site,max(ftn.fermion_space.sites)+1)
        TG = ftn.fermion_space.move_past(TG,site_range)

        ket = ftn[ftn.site_tag(x,y),'KET']
        pix = ket.inds[-1]
        ket.reindex_({pix:pix+'_'})
        ket_tid,ket_site = ket.get_fermion_info()
        ftn = insert(ftn,ket_site+1,TG)
        ftn.contract_tags(ket.tags,which='all',inplace=True)

    op_tags = '_'.join([op_tag+'{},{}'.format(x,y) for op_tag,x in zip(op_tags,xs)])
    fname = None if directory is None else \
            '_'.join([directory,op_tags,'mid{}'.format(y)])
    fname = write_ftn_to_disc(ftn.select(ftn.col_tag(y)).copy(),fname)
    return {(op_tags,'mid',y):fname}
def _1col_left(info,benvs,directory,Ly,**compress_opts):
    op_tags,xs,y,ix = info
    fname_dict = dict()
    if y<Ly-1:
        op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                            for op_tag,x in zip(op_tags,xs)])
        if y<2:
            l = [] if y==0 else [benvs['norm','mid',0]]
        else:
            l = [benvs['norm','left',y]]
        ls = l + [benvs[op_tags,'mid',y]]
        ls += [benvs['norm','mid',j] for j in range(y+1,ix+1)]
        ls = [load_ftn_from_disc(fname) for fname in ls]
        like = ls[0] if ls[0].num_tensors>0 else ls[1]
        ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

        str_ = None if directory is None else '_'.join([directory,op_tags,'left'])
        first_col = ftn.col_tag(0)
        j0 = y+2 if y==0 else y+1
        for j in range(j0,ix+1):
            ftn.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,ftn.Lx-1),
                                             **compress_opts) 
            fname = None if directory is None else str_+str(j)
            fname = write_ftn_to_disc(ftn.select(first_col).copy(),fname)
            fname_dict[op_tags,'left',j] = fname
    return fname_dict
def _1col_right(info,benvs,directory,Ly,**compress_opts):
    op_tags,xs,y,ix = info
    fname_dict = dict()
    if y>0:
        op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                            for op_tag,x in zip(op_tags,xs)])
        ls = [benvs['norm','mid',j] for j in range(ix,y)]
        if y>Ly-3:
            r = [] if y==Ly-1 else [benvs['norm','mid',Ly-1]]
        else: 
            r = [benvs['norm','right',y]]  
        ls += [benvs[op_tags,'mid',y]] + r
        ls = [load_ftn_from_disc(fname) for fname in ls]
        like = ls[0] if ls[0].num_tensors>0 else ls[1]
        ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

        last_col = ftn.col_tag(Ly-1)
        str_ = None if directory is None else '_'.join([directory,op_tags,'right']) 
        j0 = y-2 if y==Ly-1 else y-1
        for j in range(j0,ix-1,-1):
            ftn.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,ftn.Lx-1),
                                             **compress_opts) 
            fname = None if directory is None else str_+str(j)
            fname = write_ftn_to_disc(ftn.select(last_col).copy(),fname)
            fname_dict[op_tags,'right',j] = fname
    return fname_dict
def _1col_benvs(info,benvs,directory,Ly,**compress_opts):
    fxn = _1col_left if info[0]=='left' else _1col_right
    return fxn(info[1:],benvs,directory,Ly,**compress_opts)
def _2col_left(info,benvs,directory,Ly,**compress_opts):
    [op_tag1,op_tag2],[(x1,y1),(x2,y2)] = info
    op_tag1 += '{},{}'.format(x1,y1)
    op_tag2 += '{},{}'.format(x2,y2)
    if y2==1:
        ls = [benvs[op_tag1,'mid',0],benvs[op_tag2,'mid',y2]]
    else:
        ls = [benvs[op_tag1,'left',y2],benvs[op_tag2,'mid',y2]] 
    ls += [benvs['norm','mid',j] for j in range(y2+1,Ly)]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    ftn_dict = ftn.compute_left_environments(yrange=(y2-1,Ly-1),**compress_opts) 

    fname_dict = dict()
    op_tags = '_'.join([op_tag1,op_tag2])
    str_ = None if directory is None else  '_'.join([directory,op_tags,'left']) 
    for j in range(y2+1,Ly):
        fname = None if directory is None else str_+str(j)
        fname = write_ftn_to_disc(ftn_dict['left',j],fname)
        fname_dict[op_tags,'left',j] = fname
    return fname_dict
def _2col_right(info,benvs,directory,Ly,**compress_opts):
    [op_tag1,op_tag2],[(x1,y1),(x2,y2)] = info
    op_tag1 += '{},{}'.format(x1,y1)
    op_tag2 += '{},{}'.format(x2,y2)
    ls = [benvs['norm','mid',j] for j in range(y1)]
    if y1==Ly-2:
        ls += [benvs[op_tag1,'mid',y1],benvs[op_tag2,'mid',Ly-1]]
    else:
        ls += [benvs[op_tag1,'mid',y1],benvs[op_tag2,'right',y1]]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    ftn_dict = ftn.compute_right_environments(yrange=(0,y1+1),**compress_opts) 

    fname_dict = dict()
    op_tags = '_'.join([op_tag1,op_tag2])
    str_ = None if directory is None else '_'.join([directory,op_tags,'right']) 
    for j in range(0,y1):
        fname = None if directory is None else str_+str(j)
        fname = write_ftn_to_disc(ftn_dict['right',j],fname)
        fname_dict[op_tags,'right',j] = fname
    return fname_dict
def _2col_benvs(info,benvs,directory,Ly,**compress_opts):
    fxn = _2col_left if info[0]=='left' else _2col_right
    return fxn(info[1:],benvs,directory,Ly,**compress_opts)
def _row_envs(info,benvs,directory,Ly,**compress_opts):
    side_,op_tags,sites,j = info
    if sites is None: # norm term
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
        if (len(sites)==2) and (sites[0][1]<sites[1][1]): # 2col terms 
            op_tag1,op_tag2 = op_tags
            (x1,y1),(x2,y2) = sites
            op_tag1 += '{},{}'.format(x1,y1)
            op_tag2 += '{},{}'.format(x2,y2)
            op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                                for op_tag,(x,y) in zip(op_tags,sites)])
            if j<y1:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                ls = l + [benvs['norm','mid',j],benvs[op_tags,'right',j]]
            elif j==y1:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                r = [benvs[op_tag2,'mid',Ly-1]] if j==Ly-2 else \
                    [benvs[op_tag2,'right',j]]
                ls = l + [benvs[op_tag1,'mid',j]] + r
            elif j<y2: # y1<j<y2
                l = benvs[op_tag1,'mid',0] if j==1 else benvs[op_tag1,'left',j]
                r = benvs[op_tag2,'mid',Ly-1] if j==Ly-2 else \
                    benvs[op_tag2,'right',j]
                ls = l,benvs['norm','mid',j],r
            elif j==y2:
                l = [benvs[op_tag1,'mid',0]] if j==1 else [benvs[op_tag1,'left',j]]
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = l + [benvs[op_tag2,'mid',j]] + r
            else: # j>y2
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = [benvs[op_tags,'left',j],benvs['norm','mid',j]] + r
        else: # 1col terms
            y = sites[0][1]
            op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                                for op_tag,(x,y) in zip(op_tags,sites)])
            if j<y:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                r = [benvs[op_tags,'mid',Ly-1]] if j==Ly-2 else \
                    [benvs[op_tags,'right',j]]
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
                ls = l + [benvs[op_tags,'mid',j]] + r
            else: # j>y
                l = [benvs[op_tags,'mid',0]] if j==1 else [benvs[op_tags,'left',j]]
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = l + [benvs['norm','mid',j]] + r
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    if side_=='top':
        ftn_dict = ftn.compute_top_environments(
                   yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)
    else:
        ftn_dict = ftn.compute_bottom_environments(
                   yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)

    fname_dict = dict()
    str_ = None if directory is None else '_'.join([directory,op_tags,ftn.col_tag(j)])
    if side_=='bottom':
        for i in range(2,ftn.Lx):
            fname = None if directory is None else str_+'bottom'+str(i)
            fname = write_ftn_to_disc(ftn_dict['bottom',i],fname)
            fname_dict[op_tags,j,'bottom',i] = fname
        for i in range(ftn.Lx-1):
            fname = None if directory is None else str_+'mid'+str(i)
            fname = write_ftn_to_disc(ftn_dict['mid',i],fname)
            fname_dict[op_tags,j,'mid',i] = fname
    else: 
        for i in range(ftn.Lx-2):
            fname = None if directory is None else str_+'top'+str(i)
            fname = write_ftn_to_disc(ftn_dict['top',i],fname)
            fname_dict[op_tags,j,'top',i] = fname
        i = ftn.Lx-1
        fname = None if directory is None else str_+'mid'+str(i)
        fname = write_ftn_to_disc(ftn_dict['mid',i],fname)
        fname_dict[op_tags,j,'mid',i] = fname
    return fname_dict
def _contract_plq(info,envs,Lx):
    op_tags,sites,i,j,fac = info
    if sites is not None:
        op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                            for op_tag,(x,y) in zip(op_tags,sites)])
    if i<2:
        b = [] if i==0 else [envs[op_tags,j,'mid',0]]
    else:
        b = [envs[op_tags,j,'bottom',i]]
    if i>Lx-3:
        t = [] if i==Lx-1 else [envs[op_tags,j,'mid',Lx-1]]
    else: 
        t = [envs[op_tags,j,'top',i]]
    ls = b + [envs[op_tags,j,'mid',i]] + t
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
    return (i,j),op_tags,scal*fac,data*fac
def _site_grad(info):
    site,site_data_map = info
    H0,H1 = 0.0,0.0 # scalar/tsr H
    for term_key,(scal,data) in site_data_map.items():
        if term_key=='norm':
            N0,N1 = scal,data
        else:
            H0,H1 = H0+scal,H1+data
    E = H0/N0
    g = (H1-N1*E)/N0
    return site,g,E,N0
def compute_grad(H,psi,directory,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm_fname = None if directory is None else directory+'_norm'
    norm_fname = write_ftn_to_disc(norm,norm_fname)

    fxn = _norm_benvs
    iterate_over = ['left','right']
    args = [norm_fname,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _1col_mid
    ls1 = [(op_tags,[x for (x,_) in sites],sites[0][1]) \
           for (sites,op_tags,_) in H._1col_terms]
    ls2 = [([op_tag],[x],y) for op_tag in H._op_tags for (x,y) in H._sites]
    iterate_over = ls1 + ls2 
    args = [H.op_dict,norm_fname,directory]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _1col_benvs
    ix_dict = {'left':Ly-1,'right':0}
    ls1 = [(side,op_tags,xs,y,ix_dict[side]) for side in ['left','right']\
           for (op_tags,xs,y) in ls1]
    ls2_ = [('left',op_tags,xs,y,min(Ly-1,y+H.max_dy)) for (op_tags,xs,y) in ls2]
    ls2_ += [('right',op_tags,xs,y,max(0,y-H.max_dy)) for (op_tags,xs,y) in ls2]
    iterate_over = ls1 + ls2_ 
    args = [benvs,directory,Ly]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _2col_benvs
    iterate_over = [(side,op_tags,sites) for (sites,op_tags,_) in H._2col_terms\
                    for side in ['left','right']]
    args = [benvs,directory,Ly]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _row_envs
    iterate_over = [(side,op_tags,sites,j) \
                     for side in ['top','bottom'] for j in range(Ly)\
                     for (sites,op_tags,_) in H._1col_terms + H._2col_terms ]
    iterate_over += [(side,'norm',None,j) for side in ['top','bottom']\
                     for j in range(Ly)]
    args = [benvs,directory,Ly]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    envs = dict()
    for fname_dict in ls:
        envs.update(fname_dict)

    fxn = _contract_plq
    iterate_over = []
    iterate_over = [(op_tags,sites,i,j,fac) \
                     for (sites,op_tags,fac) in H._1col_terms + H._2col_terms \
                     for i in range(Lx) for j in range(Ly)]
    iterate_over += [('norm',None,i,j,1.0) for i in range(Lx) for j in range(Ly)] 
    args = [envs,Lx]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    data_map = dict()
    for (site,term_key,scal,data) in ls:
        if site not in data_map:
            data_map[site] = dict()
        data_map[site][term_key] = scal,data

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

    delete_ftn_from_disc(norm_fname) 
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname)
    for _,fname in envs.items():
        delete_ftn_from_disc(fname)
    return g,E,N 
def _energy_term(info,benvs,**compress_opts):
    sites,op_tags,fac = info
    j = 0 
    if sites is None: # norm term
        ls = [benvs['norm',side,j] for side in ['mid','right']]
    else: # H terms
        if (len(sites)==2) and (sites[0][1]<sites[1][1]): # 2col terms 
            op_tag1,op_tag2 = op_tags
            (x1,y1),(x2,y2) = sites
            op_tag1 += '{},{}'.format(x1,y1)
            op_tag2 += '{},{}'.format(x2,y2)
            op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                                for op_tag,(x,y) in zip(op_tags,sites)])
            if j<y1:
                ls = benvs['norm','mid',j],benvs[op_tags,'right',j]
            elif j==y1:
                ls = benvs[op_tag1,'mid',j],benvs[op_tag2,'right',j]
            elif j<y2:
                ls = benvs['norm','mid',j],benvs[op_tag2,'right',j]
            elif j==y2:
                ls = benvs[op_tag2,'mid',j],benvs['norm','right',j]
            else: # j>y2
                ls = benvs['norm','mid',j],benvs['norm','right',j]
        else: # 1col terms
            y = sites[0][1]
            op_tags = '_'.join([op_tag+'{},{}'.format(x,y) \
                                for op_tag,(x,y) in zip(op_tags,sites)])
            if j<y:
                ls = benvs['norm','mid',j],benvs[op_tags,'right',j]
            elif j==y:
                ls = benvs[op_tags,'mid',j],benvs['norm','right',j]
            else: # j>y
                ls = benvs['norm','mid',j],benvs['norm','right',j]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    return op_tags,ftn.contract()*fac
def compute_energy(H,psi,directory,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm_fname = None if directory is None else directory+'_norm'
    norm_fname = write_ftn_to_disc(norm,norm_fname)

    ftn_dict = norm.compute_right_environments(**compress_opts)
    benvs = dict()
    for (side,j),ftn in ftn_dict.items():
        fname = None if directory is None else directory+'_norm_'+side+str(j)
        fname = write_ftn_to_disc(ftn,fname)
        benvs['norm',side,j] = fname
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    ftn = norm.select(norm.col_tag(0)).copy()
    fname = None if directory is None else directory+'_norm_mid0'
    fname = write_ftn_to_disc(ftn,fname)
    benvs['norm','mid',0] = fname

    fxn = _1col_mid
    ls1 = [(op_tags,[x for (x,_) in sites],sites[0][1]) \
           for (sites,op_tags,_) in H._1col_terms]
    ls2 = [([op_tag],[x],y) for op_tag in H._op_tags for (x,y) in H._sites]
    iterate_over = ls1 + ls2 
    args = [H.op_dict,norm_fname,directory]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _1col_right
    ls1 = [(op_tags,xs,y,0) for (op_tags,xs,y) in ls1]
    ls2 = [(op_tags,xs,y,max(0,y-H.max_dy)) for (op_tags,xs,y) in ls2]
    iterate_over = ls1 + ls2 
    args = [benvs,directory,Ly]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _2col_right
    iterate_over = [(op_tags,sites) for (sites,op_tags,_) in H._2col_terms]
    args = [benvs,directory,Ly]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)

    fxn = _energy_term
    iterate_over =  H._1col_terms + H._2col_terms + [(None,'norm',1.0)]
    args = [benvs]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    E = 0.0
    for (term_key,scal) in ls:
        if term_key=='norm':
            N = scal
        else:
            E += scal

    delete_ftn_from_disc(norm_fname) 
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
    def __init__(self,H,peps,D,chi,directory,psi_fname,
                 write_int_to_disc=True):
        self.start_time = time.time()
        self.H = H
        self.D = D
        self.chi = chi
        self.directory = directory
        self.write_int_to_disc = write_int_to_disc

        N = compute_norm(peps,max_bond=chi)  
        peps = peps.multiply_each(N**(-1.0/(2.0*peps.num_tensors)),inplace=True)
        peps.balance_bonds_()
        peps.equalize_norms_()
        self.fac = 1.0

        self.psi_fname = directory+psi_fname
        write_ftn_to_disc(peps,self.psi_fname) # save state
        self.constructors = dict()
        for i in range(peps.Lx):
            for j in range(peps.Ly):
                site_tag = peps.site_tag(i,j)
                data = peps[site_tag].data
                bond_infos = [data.get_bond_info(ax,flip=False) \
                              for ax in range(data.ndim)]
                cons = Constructor.from_bond_infos(bond_infos,data.pattern)
                self.constructors[site_tag] = cons,data.dq
        self.skeleton = peps
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
        directory = self.directory if self.write_int_to_disc else None 
        E,N = compute_energy(self.H,psi,directory,max_bond=self.chi,cutoff=1e-15)
        print('    ne={},time={}'.format(self.ne,time.time()-self.start_time))
        print('        E={},N={}'.format(E,N))
        self.ne += 1
        self.fac = N**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,N
    def compute_grad(self,x):
        psi = self.vec2fpeps(x)
        directory = self.directory if self.write_int_to_disc else None 
        grad,E,N = compute_grad(self.H,psi,directory,max_bond=self.chi,cutoff=1e-15)
#        E_,N_ = compute_energy(self.H,psi,directory,
#                            max_bond=self.chi,cutoff=1e-15)
#        print(abs((E-E_)/E_),abs((N-N_)/N_))
        E_,N_ = (E,N) if self.chi is None else \
             compute_energy(self.H,psi,directory,max_bond=self.chi+5,cutoff=1e-15)

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
        write_ftn_to_disc(psi,self.psi_fname)
        return x,g
    def kernel(self,method=_minimize_bfgs,options={'maxiter':200,'gtol':1e-5}):
        self.ng = 0
        self.ne = 0
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi_fname))
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
