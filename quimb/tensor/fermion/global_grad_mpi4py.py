import numpy as np
import time,functools

from itertools import product
from ..tensor_2d import Rotator2D
from .fermion_2d_tebd import insert,copy
from .fermion_2d import FermionTensorNetwork2D 
from .fermion_core import FermionTensor,FermionTensorNetwork
########### MPI ##################
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

########### serial fxns #######################
def apply_term(term,norm):
#    print('############ after copy ##########')
#    for tid,(tsr,_) in norm.fermion_space.tensor_order.items():
#        assert norm.tensor_map[tid] is tsr
#        print(tsr.inds)
    ftn = norm.copy(full=True)
#    ftn = copy(norm)
#    print('############ after copy ##########')
#    for tid,(tsr,_) in ftn.fermion_space.tensor_order.items():
#        assert ftn.tensor_map[tid] is tsr
#        print(tsr.inds)
    n = ftn.num_tensors//2
    apply_order = list(term.ops.keys())
    apply_order.sort()
    for key in apply_order:
        where,op = term.ops[key]
        ket = ftn[ftn.site_tag(*where),'KET']
        site_ix = ket.inds[-1]
        bnd = site_ix+'_'
        ket.reindex_({site_ix:bnd})
        ket_tid,ket_site = ket.get_fermion_info()

        TG = FermionTensor(op.copy(),inds=(site_ix,bnd),left_inds=(site_ix,),
                           tags=ket.tags)
        ftn = insert(ftn,n,TG)
        TG_tid,TG_site = TG.get_fermion_info()
        ftn.fermion_space.move(TG_tid,ket_site+1)
        
        ftn.contract_tags(ket.tags,which='all',inplace=True)
    return ftn,term.fac
def compute_environments(
    tn,
    from_which,
    xrange=None,
    yrange=None,
    max_bond=None,
    *,
    cutoff=1e-10,
    canonize=True,
    mode='mps',
    layer_tags=None,
    dense=False,
    compress_opts=None,
    envs=None,
    **contract_boundary_opts
):
    """Compute the ``self.Lx`` 1D boundary tensor networks describing
    the environments of rows and columns. The returned tensor network
    also contains the original plaquettes
    """
    direction = {"left": "col",
                 "right": "col",
                 "top": "row",
                 "bottom": "row"}[from_which]
    tn.reorder(direction, layer_tags=layer_tags, inplace=True)

    r2d = Rotator2D(tn, xrange, yrange, from_which)
    sweep, row_tag = r2d.vertical_sweep, r2d.row_tag
    contract_boundary_fn = r2d.get_contract_boundary_fn()

    if envs is None:
        envs = {}

    if mode == 'full-bond':
        # set shared storage for opposite env contractions
        contract_boundary_opts.setdefault('opposite_envs', {})

    envs[from_which, sweep[0]] = FermionTensorNetwork([])
    first_row = row_tag(sweep[0])
    envs['mid', sweep[0]] = tn.select(first_row).copy()
    if len(sweep)==1:
        return envs
    if dense:
        tn ^= first_row
    envs[from_which, sweep[1]] = tn.select(first_row).copy()

    for i in sweep[2:]:
        iprevprev = i - 2 * sweep.step
        iprev = i - sweep.step
        envs['mid', iprev] = tn.select(row_tag(iprev)).copy()
        if dense:
            tn ^= (row_tag(iprevprev), row_tag(iprev))
        else:
            contract_boundary_fn(
                iprevprev, iprev,
                max_bond=max_bond,
                cutoff=cutoff,
                mode=mode,
                canonize=canonize,
                layer_tags=layer_tags,
                compress_opts=compress_opts,
                **contract_boundary_opts,
            )

        envs[from_which, i] = tn.select(first_row).copy()

    return envs

#compute_bottom_environments = functools.partialmethod(
compute_bottom_environments = functools.partial(
    compute_environments, from_which='bottom')

#compute_top_environments = functools.partialmethod(
compute_top_environments = functools.partial(
    compute_environments, from_which='top')

#compute_left_environments = functools.partialmethod(
compute_left_environments = functools.partial(
    compute_environments, from_which='left')

#compute_right_environments = functools.partialmethod(
compute_right_environments = functools.partial(
    compute_environments, from_which='right')
def compute_top_envs(info,**plq_env_opts):
    key,ftn = info
    envs = compute_top_environments(ftn,**plq_env_opts)
    return key,envs
def compute_bottom_envs(info,**plq_env_opts):
    key,ftn = info
    envs = compute_bottom_environments(ftn,envs,**plq_env_opts)
    return key,envs
def compute_col_envs(ftn,xrange,**plq_env_opts):
    envs = {}
    compute_left_environments(ftn,envs=envs,xrange=xrange,**plq_env_opts)
    compute_right_environments(ftn,envs=envs,xrange=xrange,**plq_env_opts)
    return envs
def compute_plq_envs(info,x_bsz,y_bsz,**plq_env_opts):
    key,ftn,fac = info
#    plq_envs = ftn.compute_plaquette_environments(x_bsz=x_bsz,y_bsz=y_bsz,
#               **plq_env_opts)
    print('############')
    print(ftn)
    row_envs = compute_row_envs(ftn,**plq_env_opts)
    col_envs = dict()
    for i in range(ftn.Lx - x_bsz + 1):
        row_i = FermionTensorNetwork((
            row_envs['bottom', i],
            *[row_envs['mid', i+x] for x in range(x_bsz)],
            row_envs['top', i + x_bsz - 1],
        )).view_as_(FermionTensorNetwork2D, like=ftn)
        col_envs[i] = compute_col_envs(row_i,
            xrange=(max(i - 1, 0), min(i + x_bsz, ftn.Lx - 1)),
            **plq_env_opts)
    plq_envs = dict()
    for i0, j0 in product(range(ftn.Lx - x_bsz + 1),
                          range(ftn.Ly - y_bsz + 1)):
        env_ij = FermionTensorNetwork((
            col_envs[i0]['left', j0],
            *[col_envs[i0]['mid', ix] for ix in range(j0, j0+y_bsz)],
            col_envs[i0]['right', j0 + y_bsz - 1]
        ), check_collisions=False)

        plq_envs[(i0, j0), (x_bsz, y_bsz)] = env_ij
    return key,plq_envs,fac
def term_to_flat(term_map):
    flat_map = dict()
    for term_key,(plq_envs,fac) in term_map.items():
        for (where,_),env in plq_envs.items():
            flat_map[where,term_key] = env,fac
    return flat_map
def get_components_site(site,env,fac=1.0):
    scalar = env.contract()
    bra = env[site,'BRA']
    tid = bra.get_fermion_info()[0]
    env._pop_tensor(tid,remove_from_fermion_space='end')
    grad = env.contract(output_inds=bra.inds[::-1])
    return scalar*fac,grad*fac
def flat_to_site(flat_map):
    site_map = dict()
    for (where,term_key),val in flat_map.items():
        if where not in site_map:
            site_map[where] = dict()
        local_map = site_map[where]
        local_map[term_key] = val
    return site_map
def get_local_grad(local_map):
    scalar_H = 0.0
    grad_H = 0.0
    for key,(scalar,grad) in local_map.items():
        if key=='norm':
            scalar_N,grad_N = scalar,grad
        else:
            scalar_H += scalar
            grad_H += grad
    print('energy=',scalar_H/scalar_N)
    return grad_H/scalar_N-grad_N*scalar_H/scalar_N**2
############### parallelizable fxns ##############################
def apply_terms(norm,H,parallel=False):
    ftn_map = dict()
    ftn_map['norm'] = norm,1.0
    if parallel:
        fxn = apply_term
        iterate_over = H
        args = [norm]
        kwargs = {}
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        count = 0
        for i in range(len(results)):
            if results[i] is not None:
                ftn = results[i][0]
                fac = results[i][1]
                ftn_map[count] = ftn,fac
                count += 1
        assert count==len(H)
    else:
        for i,term in enumerate(H):
            ftn_map[i] = apply_term(term,norm)
    return ftn_map
def get_plq_envs(ftn_map,parallel=True,
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps',layer_tags=('KET','BRA'),
    **plq_env_opts):
    plq_env_opts['max_bond'] = max_bond
    plq_env_opts['cutoff'] = cutoff
    plq_env_opts['canonize'] = canonize
    plq_env_opts['mode'] = mode
    plq_env_opts['layer_tags'] = layer_tags
    x_bsz = y_bsz = 1
    env_map = dict()
    if parallel:
        fxn = compute_top_envs
        iterate_over = [(key,ftn.copy()) for key,(ftn,_) in ftn_map.items()]
        args = []
        kwargs = plq_env_opts
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        top_env_map = dict()
        for i in range(len(results)):
            if results[i] is not None:
                key = results[i][0]
                envs = results[i][1]
                top_env_map[key] = envs
        assert len(top_env_map)==len(ftn_map)

        fxn = compute_bottom_envs
        iterate_over = [(key,ftn.copy()) for key,(ftn,_) in ftn_map.items()]
        args = []
        kwargs = plq_env_opts
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        bottom_env_map = dict()
        for i in range(len(results)):
            if results[i] is not None:
                key = results[i][0]
                envs = results[i][1]
                bottom_env_map[key] = envs
        assert len(bottom_env_map)==len(ftn_map)

        row_env_map = dict()
        for key in top_envs_map.keys():
            row_envs = dict()
            row_envs.update(top_env_map[key])
            row_envs.update(bottom_env_map[key])
            row_env_map[key] = row_envs


    else:
        for key,(ftn,fac) in ftn_map.items():
            plq_envs = ftn.compute_plaquette_environments(x_bsz=x_bsz,y_bsz=y_bsz,
                       **plq_env_opts)
            env_map[key] = plq_envs,fac
    return env_map
def get_components(flat_env_map,ftn):
    component_map = dict()
    for key,(env,fac) in flat_env_map.items():
        site = ftn.site_tag(*key[0])
        component_map[key] = get_components_site(site,env,fac)
    return component_map
def get_grad(component_map):
    grad_map = dict()
    for where,local_map in component_map.items():
        grad_map[where] = get_local_grad(local_map)
    return grad_map
class GlobalGrad():
    def __init__(self,H,peps,D,chi,maxiter=1000,tol=1e-5):
        self.H = H
        self._psi = peps
        self.D = D
        self.chi = chi
        self.maxiter = maxiter
        self.tol = tol
    def kernel(self,maxiter=None,tol=None):
        maxiter = self.maxiter if maxiter is None else maxiter
        tol = self.tol if tol is None else tol
        for i in range(maxiter):
            norm,_,self._bra = self._psi.make_norm(return_all=True,
                                        layer_tags=('KET','BRA'))
            ftn_map = apply_terms(norm,self.H)
            env_map = get_plq_envs(ftn_map,max_bond=self.chi)
            flat_env_map = term_to_flat(env_map)
            flat_component_map = get_components(flat_env_map,norm)
            component_map = flat_to_site(flat_component_map)
            grad_map = get_grad(component_map)
            exit() 
class Term():
    def __init__(self,ops,fac):
        self.ops = ops
        self.fac = fac
def SpinlessFermion(t,v,Lx,Ly,symmetry='u1'):
    from .spinless import creation
    ham = []
    cre = creation(symmetry=symmetry)
    ann = cre.dagger
    pn = np.tensordot(cre,ann,axes=((1,),(0,)))
    for i, j in product(range(Lx), range(Ly)):
        if i+1 != Lx:
            where = ((i,j),(i+1,j)) 
            ops = {1:(where[0],cre.copy()),0:(where[1],ann.copy())}
            ham.append(Term(ops,-t))
            ops = {1:(where[1],cre.copy()),0:(where[0],ann.copy())}
            ham.append(Term(ops,-t))
            ops = {1:(where[0],pn.copy()),0:(where[1],pn.copy())}
            ham.append(Term(ops,v))
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            ops = {1:(where[0],cre.copy()),0:(where[1],ann.copy())}
            ham.append(Term(ops,-t))
            ops = {1:(where[1],cre.copy()),0:(where[0],ann.copy())}
            ham.append(Term(ops,-t))
            ops = {1:(where[0],pn.copy()),0:(where[1],pn.copy())}
            ham.append(Term(ops,v))
    return ham 
