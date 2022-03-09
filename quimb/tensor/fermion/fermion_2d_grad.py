import numpy as np
import time

from itertools import product
from .fermion_2d_tebd import (
    insert,
    write_ftn_to_disc,
    load_ftn_from_disc,
    delete_ftn_from_disc,
)
from .fermion_core import FermionTensor,FermionTensorNetwork
from .fermion_2d import FermionTensorNetwork2D
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
def apply_term(term_key,H,psi,directory):
    sites,typ = term_key
    ops = H[term_key][0] 
    psi = load_ftn_from_disc(psi)
    ftn = psi.make_norm(layer_tags=('KET','BRA'))

    N = ftn.num_tensors//2
    site_range = (N,max(ftn.fermion_space.sites)+1)
    tsrs = []
    for op,site in zip(ops,sites):
        ket = ftn[ftn.site_tag(*site),'KET']
        pix = ket.inds[-1]
        TG = FermionTensor(op.copy(),inds=(pix,pix+'_'),left_inds=(pix,),
                           tags=ket.tags)
        tsrs.append(ftn.fermion_space.move_past(TG,site_range))

    ftn.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    for TG,site in zip(tsrs,sites):
        bra = ftn[ftn.site_tag(*site),'BRA']
        bra_tid,bra_site = bra.get_fermion_info()
        site_range = (bra_site,max(ftn.fermion_space.sites)+1)
        TG = ftn.fermion_space.move_past(TG,site_range)

        ket = ftn[ftn.site_tag(*site),'KET']
        pix = ket.inds[-1]
        ket.reindex_({pix:pix+'_'})
        ket_tid,ket_site = ket.get_fermion_info()
        ftn = insert(ftn,ket_site+1,TG)
        ftn.contract_tags(ket.tags,which='all',inplace=True)
    term_str = '_'.join([ftn.site_tag(*site) for site in sites]+[str(typ)])+'_'
    term_dict = dict()
    for idx,site in enumerate(sites):
        if site[1] not in term_dict:
            fname = directory+term_str+ftn.col_tag(site[1])
            tmp = ftn.select(ftn.col_tag(site[1])).copy()
            write_ftn_to_disc(tmp,fname)
            term_dict[site[1]] = fname
    return term_key,term_dict
def compute_energy_term(info,norm_benvs,directory,**compress_opts):
    term_key,term_dict = info
    cols = list(term_dict.keys())
    cols.sort()
    like = load_ftn_from_disc(norm_benvs['mid',0])
    Lx,Ly = like.Lx,like.Ly

    nl,nr = cols[-1]+1,Ly-1-cols[0]+1
    jmin = max(cols[0]-1,0)
    jmax = min(cols[-1]+1,Ly-1)
    if nl<nr:
        _,benvs = compute_left_envs_wrapper(info,norm_benvs,directory,
                  True,**compress_opts) 
        ls = benvs['left',jmax],norm_benvs['right',jmax-1] 
        yrange = jmax-1,jmax
    else:
        _,benvs = compute_right_envs_wrapper(info,norm_benvs,directory,
                  True,**compress_opts) 
        ls = norm_benvs['left',jmin+1],benvs['right',jmin]
        yrange = jmin,jmin+1
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=like)
    i = Lx//2
    row_envs = dict()
    ftn.compute_bottom_environments(envs=row_envs,xrange=(0,i),yrange=yrange,
                                    **compress_opts)
    ftn.compute_top_environments(envs=row_envs,xrange=(i-1,Lx-1),yrange=yrange,
                                 **compress_opts)
    ftn = FermionTensorNetwork((row_envs['bottom',i],row_envs['top',i-1]))
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname)
    return term_key,ftn.contract()
def compute_left_envs_wrapper(info,norm_benvs,directory,partial,**compress_opts):
    term_key,term_dict = info
    cols = list(term_dict.keys())
    cols.sort()
    like = load_ftn_from_disc(norm_benvs['mid',0])
    Lx,Ly = like.Lx,like.Ly

    ls = [norm_benvs['left',cols[0]]]
    for idx,j1 in enumerate(cols):
        j2 = Ly if idx==len(cols)-1 else cols[idx+1]
        ls += [term_dict[j1]]+[norm_benvs['mid',j] for j in range(j1+1,j2)]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=like)
    jmin = max(cols[0]-1,0)
    jmax = min(cols[-1]+1,Ly-1) if partial else Ly-1
    term_benvs = ftn.compute_left_environments(yrange=(jmin,jmax),**compress_opts) 

    sites,typ = term_key
    term_str = '_'.join([ftn.site_tag(*site) for site in sites]+[str(typ)])+'_'
    for key in term_benvs.keys():
        fname = directory+term_str+key[0]+'_'+ftn.col_tag(key[1])
        write_ftn_to_disc(term_benvs[key],fname)
        term_benvs[key] = fname
    return term_key,term_benvs
def compute_right_envs_wrapper(info,norm_benvs,directory,partial,**compress_opts): 
    term_key,term_dict = info
    cols = list(term_dict.keys())
    cols.sort()
    like = load_ftn_from_disc(norm_benvs['mid',0])
    Lx,Ly = like.Lx,like.Ly

    ls = []
    for idx,j2 in enumerate(cols):
        j1 = -1 if idx==0 else cols[idx-1]
        ls += [norm_benvs['mid',j] for j in range(j1+1,j2)]+[term_dict[j2]]
    ls.append(norm_benvs['right',cols[-1]])
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=like)
    jmin = max(cols[0]-1,0) if partial else 0
    jmax = min(cols[-1]+1,Ly-1)
    term_benvs = ftn.compute_right_environments(yrange=(jmin,jmax),**compress_opts)

    sites,typ = term_key
    term_str = '_'.join([ftn.site_tag(*site) for site in sites]+[str(typ)])+'_'
    for key in term_benvs.keys():
        fname = directory+term_str+key[0]+'_'+ftn.col_tag(key[1])
        write_ftn_to_disc(term_benvs[key],fname)
        term_benvs[key] = fname
    return term_key,term_benvs
def compute_plq_envs_wrapper(info,norm_benvs,directory,**compress_opts):
    term_key,term_benvs,j = info
    like = load_ftn_from_disc(norm_benvs['mid',0])
    Lx,Ly = like.Lx,like.Ly
    if term_key=='norm':
        ls = [norm_benvs[side,j] for side in ['left','mid','right']]
    else:
        cols = list(set([site[1] for site in term_key[0]]))
        cols.sort()
        if j<cols[0]:
            ls = norm_benvs['left',j],norm_benvs['mid',j],term_benvs['right',j]
        elif j>cols[-1]:
            ls = term_benvs['left',j],norm_benvs['mid',j],norm_benvs['right',j]
        else:
            ls = [term_benvs[side,j] for side in ['left','mid','right']]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=like)
    row_envs = ftn.compute_row_environments(yrange=(max(j-1,0),min(j+1,Ly-1)),
                                            **compress_opts)
    if term_key=='norm':
        term_str = term_key+'_'
    else:
        sites,typ = term_key
        term_str = '_'.join([ftn.site_tag(*site) for site in sites]+[str(typ)])+'_'
    plq_envs = dict()
    for i in range(Lx):
        ftn = FermionTensorNetwork(
              [row_envs[side,i] for side in ['bottom','mid','top']],
              check_collisions=False)
        site_tag = like.site_tag(i,j)
        fname = directory+term_str+site_tag
        write_ftn_to_disc(ftn,fname)
        plq_envs[term_key,site_tag] = fname
    return plq_envs
def compute_site_grad(site_tag,H,plq_envs):
    H0,H1 = 0.0,0.0 # scalar/tsr H
    for key in ['norm']+list(H.keys()):
        ftn = load_ftn_from_disc(plq_envs[key,site_tag])
        ftn.select((site_tag,'BRA'),which='!all').add_tag('grad')
        bra = ftn[site_tag,'BRA']
        ftn.contract_tags('grad',which='any',inplace=True,output_inds=bra.inds[::-1])
        assert ftn.num_tensors==2
        scal = ftn.contract()
        bra_tid = bra.get_fermion_info()[0]
        ftn._pop_tensor(bra_tid,remove_from_fermion_space='end')
        data = ftn['grad'].data
        if key=='norm':
            N0,N1 = scal,data
        else:
            H0,H1 = H0+scal*H[key][-1],H1+data*H[key][-1]
    return site_tag,(H1-N1*H0/N0)/N0
def compute_energy(H,psi_fname,directory,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    # get norm col envs
    psi = load_ftn_from_disc(psi_fname)
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm_benvs = norm.compute_col_environments(**compress_opts)
    for key in norm_benvs.keys():
        fname = directory+'norm_'+key[0]+'_'+norm.col_tag(key[1])
        write_ftn_to_disc(norm_benvs[key],fname)
        norm_benvs[key] = fname
    # get term cols
    fxn = apply_term
    iterate_over = list(H.keys())
    args = [H,psi_fname,directory]
    kwargs = dict()
    terms = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    # compute energy numerator  
    fxn = compute_energy_term
    iterate_over = terms
    args = [norm_benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    E = 0.0
    for (term_key,Ei) in ls:
        Ei *= H[term_key][-1]
        E += Ei
    # compute norm
    j = Ly//2
    ls = norm_benvs['left',j],norm_benvs['right',j-1]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=norm)
    i = (Lx-1)//2
    row_envs = dict()
    ftn.compute_bottom_environments(envs=row_envs,xrange=(0,i),yrange=(j-1,j),
                                    **compress_opts)
    ftn.compute_top_environments(envs=row_envs,xrange=(i-1,Lx-1),yrange=(j-1,j),
                                 **compress_opts)
    ftn = FermionTensorNetwork((row_envs['bottom',i],row_envs['top',i-1]))
    # delete files
    for _,fname in norm_benvs.items():
        delete_ftn_from_disc(fname) 
    for (_,term_dict) in terms:
        for _,fname in term_dict.items():
            delete_ftn_from_disc(fname)
    return E/ftn.contract() 
def compute_grad(H,psi_fname,directory,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    # get norm col envs
    psi = load_ftn_from_disc(psi_fname)
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm_benvs = norm.compute_col_environments(**compress_opts)
    for key in norm_benvs.keys():
        fname = directory+'norm_'+key[0]+'_'+norm.col_tag(key[1])
        write_ftn_to_disc(norm_benvs[key],fname)
        norm_benvs[key] = fname
    # get term cols
    fxn = apply_term
    iterate_over = list(H.keys())
    args = [H,psi_fname,directory]
    kwargs = dict()
    terms = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    # get terms col envs
    benvs = {'norm':norm_benvs}

    fxn = compute_left_envs_wrapper
    iterate_over = terms
    args = [norm_benvs,directory,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for (term_key,term_benvs) in ls:
        benvs[term_key] = term_benvs

    fxn = compute_right_envs_wrapper
    iterate_over = terms
    args = [norm_benvs,directory,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for (term_key,term_benvs) in ls:
        benvs[term_key].update(term_benvs)
    # get terms & norm plq_envs
    plq_envs = dict()
    fxn = compute_plq_envs_wrapper
    iterate_over = [(term_key,term_benvs,j) \
                    for term_key,term_benvs in benvs.items() for j in range(Ly)]
    args = [norm_benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for plq_envs_term in ls:
        plq_envs.update(plq_envs_term)
    # compute grad
    fxn = compute_site_grad
    iterate_over = [norm.site_tag(i,j) for i in range(Lx) for j in range(Ly)]
    args = [H,plq_envs]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    grad = dict()
    for (site_tag,gsite) in ls:
        grad[site_tag] = gsite
    # delete files 
    for _,fname in norm_benvs.items():
        delete_ftn_from_disc(fname) 
    for (_,term_dict) in terms:
        for _,fname in term_dict.items():
            delete_ftn_from_disc(fname)
    for _,term_benvs in benvs.items():
        for _,fname in term_benvs.items():
            delete_ftn_from_disc(fname)
    for _,fname in plq_envs.items():
        delete_ftn_from_disc(fname)
    return grad 
def conjugate_grad(grad,d_old,gnormsq_old):
    gnormsq = sum([np.dot(g.data,g.data) for g in grad.values()])
    if d_old is None:
        d = {key:-g for key,g in grad.items()}
    else:
        beta = gnormsq/gnormsq_old
        d = {key:-g+beta*d_old[key] for key,g in grad.items()}
    return d,gnormsq
def line_search(d,EL,t0,H,psi_name,directory,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    def energy(t):
        psi = load_ftn_from_disc(psi_name)
        for site_tag,data in d.items():
            psi[site_tag].modify(data=psi[site_tag].data+t*data)
        write_ftn_to_disc(psi,psi_name+'_tmp')
        return compute_energy(H,psi_name+'_tmp',directory,**compress_opts)
    tR,ER = 2.0*t0,0.0
    while ER>EL:
        tR = 0.5*tR
        ER = energy(tR)
    tM = 0.5*tR
    EM = energy(tM)
    delete_ftn_from_disc(psi_name+'_tmp')
    if ER<EM:
        return tR,ER
    else:
        return tM,EM
class GlobalGrad():
    def __init__(self,H,peps,D,chi,directory='./saved_states/'):
        self.H = H
        self.D = D
        self.chi = chi
        self.directory = directory

        self._psi = directory+'_psi'
        write_ftn_to_disc(peps,self._psi)
    def kernel(self,maxiter=100,tol=1e-5,t0=0.5):
        E_old = compute_energy(self.H,self._psi,self.directory,max_bond=self.chi)
        print('initial energy =',E_old)
        d_old,gnormsq_old = None,None
        for i in range(maxiter):
            grad = compute_grad(self.H,self._psi,self.directory,max_bond=self.chi)
            d,gnormsq = conjugate_grad(grad,d_old,gnormsq_old)
            t,E = line_search(d,E_old,t0,self.H,self._psi,self.directory,
                              max_bond=self.chi) 
            psi = load_ftn_from_disc(self._psi)
            for site_tag,data in d.items():
                psi[site_tag].modify(data=psi[site_tag].data+t*data)
            print('iter={},E={},gnorm={},t={}'.format(i,E,np.sqrt(gnormsq),t))
            write_ftn_to_disc(psi,self._psi)
            if E-E_old>tol:
                break
            d_old,gnormsq_old,E_old = d.copy(),gnormsq,E
        return 
def SpinlessFermion(t,v,Lx,Ly,symmetry='u1'):
    from .spinless import creation
    ham = dict()
    cre = creation(symmetry=symmetry)
    ann = cre.dagger
    pn = np.tensordot(cre,ann,axes=((1,),(0,)))
    def get_terms(sites):
        # cre0,ann1 = (-1)**S ann1,cre0 
        ops = cre.copy(),ann.copy()
        phase = (-1)**(cre.parity*ann.parity)
        ham[sites,0] = ops,-t*phase
        # cre1,ann0
        ops = ann.copy(),cre.copy()
        phase = 1.0
        ham[sites,1] = ops,-t*phase
        # pn1,pn2
        ops = pn.copy(),pn.copy()
        phase = 1.0
        ham[sites,2] = ops,v*phase
        return 
    for i, j in product(range(Lx), range(Ly)):
        if i+1 != Lx:
            get_terms(((i,j),(i+1,j)))
        if j+1 != Ly:
            get_terms(((i,j),(i,j+1)))
    return ham
