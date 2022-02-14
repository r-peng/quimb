import numpy as np
import time#,functools

from itertools import product
from .fermion_2d_tebd import (
    insert,
    match_phase,
#    copy,
    write_ftn_to_disc,
    load_ftn_from_disc,
    delete_ftn_from_disc,
)
from .fermion_core import FermionTensor,FermionTensorNetwork
from .fermion_2d import FermionTensorNetwork2D
#from ..tensor_2d import Rotator2D

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
def _apply_term(ftn,sites,ops):
    n = ftn.num_tensors//2
    for where,op in zip(sites,ops):
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
    return ftn
def apply_term(args,norm,directory):
    sites,term_type,ops,fac = args
    ftn = load_ftn_from_disc(norm)
    ftn = _apply_term(ftn,sites,ops) 
    new_fname = ftn.site_tag(*sites[0]) + '_' + ftn.site_tag(*sites[1]) \
              + '_term{}'.format(term_type)
    new_fname = directory+new_fname
    write_ftn_to_disc(ftn,new_fname)
    return sites,term_type,new_fname,fac
def get_terms(norm,H,directory='./saved_states/'):
    fname = directory+'norm'
    write_ftn_to_disc(norm,fname)
    term_map = dict()
    term_map['norm'] = 1.0,fname

    fxn = apply_term
    iterate_over = [tuple(key)+tuple(val) for key,val in H.items()] 
    args = [fname,directory]
    kwargs = dict()
    results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for i in range(len(results)):
        if results[i] is not None:
            sites,term_type,new_fname,fac = results[i]
            term_map[sites,term_type] = fac,new_fname
    return term_map
def contract_boundary(args,bix,from_which,**kwargs):
    key,fname,orth_range = args
    tn = load_ftn_from_disc(fname)
    step = 1 if from_which in {'bottom','left'} else -1
    if from_which in {'bottom','top'}:
        tn.contract_boundary_from_(
            xrange=(bix-2*step,bix-step),yrange=orth_range,
            from_which=from_which,**kwargs)
    else:
        tn.contract_boundary_from_(
            yrange=(bix-2*step,bix-step),xrange=orth_range,
            from_which=from_which,**kwargs)
    write_ftn_to_disc(tn,fname)
    return key,fname
def get_benvs(term_map,direction,Lbix,Lix,layer_tags=('KET','BRA'),
              max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    kwargs = dict()
    kwargs['max_bond'] = max_bond
    kwargs['cutoff'] = cutoff
    kwargs['canonize'] = canonize
    kwargs['mode'] = mode
    kwargs['layer_tags'] = layer_tags

    sides = ('left','right') if direction=='col' else ('bottom','top')
    benvs = dict()
    # middle:
    for key,(_,fname) in term_map.items():
        ftn = load_ftn_from_disc(fname)
        ftn.reorder(direction,layer_tags=layer_tags,inplace=True)
        write_ftn_to_disc(ftn,fname)
        tag = ftn.col_tag if direction=='col' else ftn.row_tag
        for bix in range(Lbix):
            benv_name = fname+'_mid{}'.format(bix)
            write_ftn_to_disc(ftn.select(tag(bix)).copy(),benv_name)
            benvs[key,'mid',bix] = benv_name 
        benv_name = fname+'_{}{}'.format(sides[0],0)
        write_ftn_to_disc(FermionTensorNetwork([]),benv_name)
        benvs[key,sides[0],0] = benv_name
 
        benv_name = fname+'_{}{}'.format(sides[0],1)
        write_ftn_to_disc(ftn.select(tag(0)).copy(),benv_name)
        benvs[key,sides[0],1] = benv_name 

        benv_name = fname+'_{}{}'.format(sides[1],Lbix-1)
        write_ftn_to_disc(FermionTensorNetwork([]),benv_name)
        benvs[key,sides[1],Lbix-1] = benv_name
 
        benv_name = fname+'_{}{}'.format(sides[1],Lbix-2)
        write_ftn_to_disc(ftn.select(tag(Lbix-1)).copy(),benv_name)
        benvs[key,sides[1],Lbix-2] = benv_name 

    orth_range = 0,Lix-1
    # left
    iterate_over = []
    for key,(_,fname) in term_map.items():
        ftn = load_ftn_from_disc(fname)
        tmp_fname = fname+'_tmp'
        write_ftn_to_disc(ftn,tmp_fname)
        iterate_over.append((key,tmp_fname,orth_range)) 
    for bix in range(2,Lbix):
        fxn = contract_boundary
        args = [bix,sides[0]] 
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        for i in range(len(results)):
            if results[i] is not None:
                key,tmp_fname = results[i]
                ftn = load_ftn_from_disc(tmp_fname)
                tag = ftn.col_tag if direction=='col' else ftn.row_tag
                benv_name = term_map[key][1]+'_{}{}'.format(sides[0],bix)
                write_ftn_to_disc(ftn.select(tag(0)).copy(),benv_name)
                benvs[key,sides[0],bix] = benv_name
    # right
    iterate_over = []
    for key,(_,fname) in term_map.items():
        ftn = load_ftn_from_disc(fname)
        tmp_fname = fname+'_tmp'
        write_ftn_to_disc(ftn,tmp_fname)
        iterate_over.append((key,tmp_fname,orth_range)) 
    for bix in range(Lbix-3,-1,-1):
        fxn = contract_boundary
        args = [bix,sides[1]] 
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        for i in range(len(results)):
            if results[i] is not None:
                key,tmp_fname = results[i]
                ftn = load_ftn_from_disc(tmp_fname)
                tag = ftn.col_tag if direction=='col' else ftn.row_tag
                benv_name = term_map[key][1]+'_{}{}'.format(sides[1],bix)
                write_ftn_to_disc(ftn.select(tag(Lbix-1)).copy(),benv_name)
                benvs[key,sides[1],bix] = benv_name

    fname = term_map['norm'][1]
    like = load_ftn_from_disc(fname)
    for key,(_,fname) in term_map.items():
        delete_ftn_from_disc(fname)
    for (_,tmp_fname,_) in iterate_over:
        delete_ftn_from_disc(tmp_fname)

    benv_map = dict() 
    for key,(_,fname) in term_map.items():
        for bix in range(Lbix):
            b0 = load_ftn_from_disc(benvs[key,sides[0],bix])
            m  = load_ftn_from_disc(benvs[key,'mid',bix])
            b1 = load_ftn_from_disc(benvs[key,sides[1],bix])
            ftn = FermionTensorNetwork((b0,m,b1)).view_as_(
                  FermionTensorNetwork2D,like=like)
            bftn_name = fname+'_bix{}'.format(bix) 
            write_ftn_to_disc(ftn,bftn_name)
            benv_map[key,bix] = bftn_name 
           
    for key,fname in benvs.items():
        delete_ftn_from_disc(fname)
    return benv_map
def get_plqs(benv_map,direction,Lbix,Lix,layer_tags=('KET','BRA'),
             max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    kwargs = dict()
    kwargs['max_bond'] = max_bond
    kwargs['cutoff'] = cutoff
    kwargs['canonize'] = canonize
    kwargs['mode'] = mode
    kwargs['layer_tags'] = layer_tags

    sides = ('left','right') if direction=='col' else ('bottom','top')
    plq_envs = dict()
    # middle:
    for key,fname in benv_map.items():
        ftn = load_ftn_from_disc(fname)
        ftn.reorder(direction,layer_tags=layer_tags,inplace=True)
        write_ftn_to_disc(ftn,fname)
        tag = ftn.col_tag if direction=='col' else ftn.row_tag
        for ix in range(Lix):
            plq_name = fname+'_mid{}'.format(ix)
            write_ftn_to_disc(ftn.select(tag(ix)).copy(),plq_name)
            plq_envs[key,'mid',ix] = plq_name 
        plq_name = fname+'_{}{}'.format(sides[0],0)
        write_ftn_to_disc(FermionTensorNetwork([]),plq_name)
        plq_envs[key,sides[0],0] = plq_name 

        plq_name = fname+'_{}{}'.format(sides[0],1)
        write_ftn_to_disc(ftn.select(tag(0)).copy(),plq_name)
        plq_envs[key,sides[0],1] = plq_name 

        plq_name = fname+'_{}{}'.format(sides[1],Lix-1)
        write_ftn_to_disc(FermionTensorNetwork([]),plq_name)
        plq_envs[key,sides[1],Lix-1] = plq_name

        plq_name = fname+'_{}{}'.format(sides[1],Lix-2)
        write_ftn_to_disc(ftn.select(tag(Lix-1)).copy(),plq_name)
        plq_envs[key,sides[1],Lix-2] = plq_name 

    # bottom
    iterate_over = []
    for key,fname in benv_map.items():
        ftn = load_ftn_from_disc(fname)
        tmp_fname = fname+'_tmp'
        write_ftn_to_disc(ftn,tmp_fname)
        
        bix = key[-1]
        orth_range = max(bix-1,0),min(bix+1,Lbix-1)
        iterate_over.append((key,tmp_fname,orth_range)) 
    for ix in range(2,Lix):
        fxn = contract_boundary
        args = [ix,sides[0]]
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        for i in range(len(results)):
            if results[i] is not None:
                key,tmp_fname = results[i]
                ftn = load_ftn_from_disc(tmp_fname)
                tag = ftn.col_tag if direction=='col' else ftn.row_tag
                plq_name = benv_map[key]+'_{}{}'.format(sides[0],ix)
                write_ftn_to_disc(ftn.select(tag(0)).copy(),plq_name)
                plq_envs[key,sides[0],ix] = plq_name 
    # top
    iterate_over = []
    for key,fname in benv_map.items():
        ftn = load_ftn_from_disc(fname)
        tmp_fname = fname+'_tmp'
        write_ftn_to_disc(ftn,tmp_fname)
        
        bix = key[-1]
        orth_range = max(bix-1,0),min(bix+1,Lbix-1)
        iterate_over.append((key,tmp_fname,orth_range)) 
    for ix in range(Lix-3,-1,-1):
        fxn = contract_boundary
        args = [ix,sides[1]]
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        for i in range(len(results)):
            if results[i] is not None:
                key,tmp_fname = results[i]
                ftn = load_ftn_from_disc(tmp_fname)
                tag = ftn.col_tag if direction=='col' else ftn.row_tag
                plq_name = benv_map[key]+'_{}{}'.format(sides[1],ix)
                write_ftn_to_disc(ftn.select(tag(Lix-1)).copy(),plq_name)
                plq_envs[key,sides[1],ix] = plq_name 

    fname = benv_map['norm',0]
    like = load_ftn_from_disc(fname)
    for key,fname in benv_map.items():
        delete_ftn_from_disc(fname)
    for (_,tmp_fname,_) in iterate_over:
        delete_ftn_from_disc(tmp_fname)

    plq_map = dict() 
    iterate_over = []
    for key,fname in benv_map.items():
        for ix in range(Lix):
            b0 = load_ftn_from_disc(plq_envs[key,sides[0],ix])
            m  = load_ftn_from_disc(plq_envs[key,'mid',ix])
            b1 = load_ftn_from_disc(plq_envs[key,sides[1],ix])
            ftn = FermionTensorNetwork((b0,m,b1),check_collisions=False
                  ).view_as_(FermionTensorNetwork2D,like=like)
            plq_name = fname+'_ix{}'.format(ix) 
            write_ftn_to_disc(ftn,plq_name)

            term_key,bix = key
            site = (ix,bix) if direction=='row' else (bix,ix)
            plq_map[site,term_key] = plq_name
            iterate_over.append((site,plq_name))
    # partially contract all plq_ftns
    fxn = partial_contract
    args = []
    kwargs = dict()
    results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for key,fname in plq_envs.items():
        delete_ftn_from_disc(fname)
    return plq_map
def partial_contract(args):
    site,fname = args
    ftn = load_ftn_from_disc(fname)
    site_tag = ftn.site_tag(*site)
    ftn.select(site_tag,which='!any').add_tag('env')
    bra = ftn[site_tag,'BRA']
    ket = ftn[site_tag,'KET']
    output_inds = bra.inds[1:]+ket.inds[:-1]
    ftn.contract_tags('env',inplace=True,output_inds=output_inds)
    assert ftn.num_tensors==3
    bra_tid = bra.get_fermion_info()[0]
    ket_tid = ket.get_fermion_info()[0]
    ftn.fermion_space.move(bra_tid,2)
    ftn.fermion_space.move(ket_tid,0)
    ftn._refactor_phase_from_tids((bra_tid,ket_tid))
    write_ftn_to_disc(ftn,fname)
    return 
def get_component(args,norm):
    site,term_key,fname,(ops,fac),t,d = args
    ftn = load_ftn_from_disc(fname)
    site_tag = ftn.site_tag(*site)
    bra = ftn[site_tag,'BRA']
    ket = ftn[site_tag,'KET']
    bra_tid = bra.get_fermion_info()[0]
    ket_tid = ket.get_fermion_info()[0]
#    ftn.fermion_space.move(bra_tid,ftn.num_tensors-1)
#    ftn.fermion_space.move(ket_tid,0)
#    ftn._refactor_phase_from_tids((bra_tid,ket_tid))

    if term_key=='norm':
        _apply = False
    else:
        term_sites,term_type = term_key
        _apply = True if site in term_sites else False
#    if not _apply:
#        assert (ket.data-bra.data.dagger).norm() < 1e-15
    if d is not None:
        ftn_ = load_ftn_from_disc(norm)
        ftn_[site_tag,'KET'].modify(data=ftn_[site_tag,'KET'].data+t*d)
        ftn_[site_tag,'BRA'].modify(data=ftn_[site_tag,'BRA'].data+t*d.dagger)
        if _apply:
            ftn_ = _apply_term(ftn_,term_sites,ops)
        ket_ = ftn_[site_tag,'KET']
        bra_ = ftn_[site_tag,'BRA']
        ket_tid_ = ket_.get_fermion_info()[0]
        bra_tid_ = bra_.get_fermion_info()[0]
        ftn_.fermion_space.move(bra_tid_,ftn_.num_tensors-1)
        ftn_.fermion_space.move(ket_tid_,0)
        ftn_._refactor_phase_from_tids((bra_tid_,ket_tid_))
        
        ket.modify(data=ket_.data.copy())
        bra.modify(data=bra_.data.copy())
    ftn.select((site_tag,'BRA'),which='!all').add_tag('grad')
    ftn.contract_tags('grad',inplace=True,output_inds=bra.inds[::-1])
    assert ftn.num_tensors==2
    scal = ftn.contract()
    ftn._pop_tensor(bra_tid,remove_from_fermion_space='end')
    data = ftn['grad'].data
    return site,term_key,scal*fac,data*fac
def test_t(args,m1,m2):
    # Wolfe: suppose f parabollic
    # m1: controlls the sup of allowed t, smaller m1 -> higher sup
    # m2: controlls the inf of allowed t, larger m2 -> lower inf
    site,(t,tL,tR),q0,dq0,qt,dqt = args
    if qt>q0+m1*t*dq0: # b
        return site,(t,tL,t),False
    else:
        if dqt<m2*dq0: # c
            return site,(t,t,tR),False
        else:
            return site,(t,tL,tR),True 
def line_search(plq_map,H,norm,m1=0.2,m2=0.8,maxiter=10,t0=0.5):
    def get_q(d=None,t=dict()):
        fxn = get_component
        if d is None:
            iterate_over = [(site,term,fname,H.get(term,(None,1.0)),None,None) \
                for (site,term),fname in plq_map.items()]
        else:
            iterate_over = [(site,term,fname,H.get(term,(None,1.0)),
                             t[site][0],d[site]) \
                for (site,term),fname in plq_map.items() if site in t]
        args = [norm]
        kwargs = dict()
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        Narr,Nscal,Harr,Hscal = dict(),dict(),dict(),dict()
        for i in range(len(results)):
            if results[i] is not None:
                site,term_key,scal,arr = results[i]
                if term_key=='norm':
                    Nscal[site] = scal
                    Narr[site] = arr
                else:
                    if site not in Hscal:
                        Hscal[site] = 0.0
                    if site not in Harr:
                        Harr[site] = 0.0
                    Hscal[site] = Hscal[site]+scal
                    Harr[site]  = Harr[site] +arr
        q,dq = dict(),dict()
        d = dict() if d is None else d
        grad_norm = 0.0
        for site in Narr:
            q[site] = Hscal[site]/Nscal[site]
            grad = (Harr[site]-Narr[site]*q[site])/Nscal[site]
            if site not in d:
                d[site] = -grad
                grad_norm += grad.norm()
            dq[site] = np.dot(grad.data,d[site].data)
        return d,q,dq,grad_norm
    d,q0,dq0,grad_norm = get_q()
    t_conv = dict() # converged t
    t_cont = {site:(t0,0.0,0.0) for site in d}
    print('t0={},maxiter={},m1={},m2={}'.format(t0,maxiter,m1,m2))
    for i in range(maxiter):
        print('line search=',i)
        _,qt,dqt,_ = get_q(d,t_cont)

        fxn = test_t
        iterate_over = [(site,ts,q0[site],dq0[site],qt[site],dqt[site])
                        for site,ts in t_cont.items()]
        args = [m1,m2]
        kwargs = dict()
        results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
        t_cont = dict()
        x = dict()
        for i in range(len(results)):
            if results[i] is not None:
                site,(t,tL,tR),stop = results[i]
                if stop:
                    t_conv[site] = t
                else:
                    t = 0.5*(tL+tR) if tR>tL else 2.0*t
                    t_cont[site] = t,tL,tR
        if len(t_cont)==0:
            break
    for site,(t,_,_) in t_cont.items():
        t_conv[site] = t
    return t_conv,d,grad_norm,list(q0.values())[0] 
class GlobalGrad():
    def __init__(self,H,peps,D,chi,maxiter=1000,tol=1e-5,
                 directory='./saved_states/'):
        self.H = H
        self.D = D
        self.chi = chi
        self.maxiter = maxiter
        self.tol = tol
        self.directory = directory
        self._psi = directory+'_psi'
        write_ftn_to_disc(peps,self._psi)
    def kernel(self,maxiter=None,tol=None,col_first=True):
        maxiter = self.maxiter if maxiter is None else maxiter
        tol = self.tol if tol is None else tol
        psi = load_ftn_from_disc(self._psi)
        if col_first:
            direction = 'col','row'
            Lbix,Lix = psi.Ly,psi.Lx
        else:
            direction = 'row','col'
            Lbix,Lix = psi.Lx,psi.Ly
        alpha = 0.1
        for i in range(maxiter):
            # compute grad
            norm = psi.make_norm(layer_tags=('KET','BRA'))
            fname = self.directory+'norm_as_psi'
            write_ftn_to_disc(norm,fname)
            term_map = get_terms(norm,self.H,self.directory)
            benv_map = get_benvs(term_map,direction[0],Lbix,Lix,max_bond=self.chi)
            plq_map = get_plqs(benv_map,direction[1],Lbix,Lix,max_bond=self.chi)
            tmap,dir_map,grad_norm,energy = line_search(plq_map,self.H,fname)
            print('iter={},energy={},grad_norm={}'.format(i,energy,grad_norm))
            print(tmap)
            # line search
            # cg
            # update
            for site,data in dir_map.items():
                ket = psi[psi.site_tag(*site)]
                ket.modify(data=ket.data+tmap[site]*data)
def SpinlessFermion(t,v,Lx,Ly,symmetry='u1'):
    from .spinless import creation
    ham = dict()
    cre = creation(symmetry=symmetry)
    ann = cre.dagger
    pn = np.tensordot(cre,ann,axes=((1,),(0,)))
    def get_terms(sites):
        ls = []
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
