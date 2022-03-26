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
from .block_interface import Constructor
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
def compute_norm_col_envs_wrapper(side,norm_fname,directory,**compress_opts):
    norm = load_ftn_from_disc(norm_fname)
    fname_dict = dict()
    term_str = directory+'_norm_'+side
    if side == 'mid':
        norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
        for j in range(norm.Ly):
            fname = term_str+str(j)
            write_ftn_to_disc(norm.select(norm.col_tag(j)).copy(),fname)
            fname_dict['norm','mid',j] = fname
    else: 
        if side == 'left':
            ftn_dict = norm.compute_left_environments(**compress_opts)
        else:
            ftn_dict = norm.compute_right_environments(**compress_opts)
        for j in range(norm.Ly):
            fname = term_str+str(j)
            write_ftn_to_disc(ftn_dict[side,j],fname)
            fname_dict['norm',side,j] = fname
    return fname_dict
def apply_term(term_key,H,norm_fname,directory):
    sites,typ = term_key
    ops = H[term_key][0] 
    ftn = load_ftn_from_disc(norm_fname)

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
    term_str = directory+'_'+str(term_key)+'_mid'
    fname_dict = dict()
    cols = list(set([site[1] for site in sites]))
    for j in cols:
        fname = term_str+str(j)
        write_ftn_to_disc(ftn.select(ftn.col_tag(j)).copy(),fname)
        fname_dict[term_key,'mid',j] = fname
    return fname_dict
def compute_left_envs_wrapper(term_key,benvs,directory,**compress_opts):
    cols = list(set([site[1] for site in term_key[0]]))
    cols.sort()
    like = load_ftn_from_disc(benvs['norm','mid',0])
    Lx,Ly = like.Lx,like.Ly
    fname_dict = dict()
    if cols[0]<Ly-1:
        ls = [benvs['norm','left',cols[0]]]
        for idx,j1 in enumerate(cols):
            j2 = Ly if idx==len(cols)-1 else cols[idx+1]
            ls += [benvs[term_key,'mid',j1]]
            ls += [benvs['norm','mid',j] for j in range(j1+1,j2)]
        ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
              ).view_as_(FermionTensorNetwork2D,like=like)
        jmin = max(cols[0]-1,0)
        ftn_dict = ftn.compute_left_environments(yrange=(jmin,Ly-1),**compress_opts) 

        term_str = directory+'_'+str(term_key)+'_left' 
        for j in range(cols[0]+1,Ly):
            fname = term_str+str(j)
            write_ftn_to_disc(ftn_dict['left',j],fname)
            fname_dict[term_key,'left',j] = fname
    return fname_dict
def compute_right_envs_wrapper(term_key,benvs,directory,**compress_opts): 
    cols = list(set([site[1] for site in term_key[0]]))
    cols.sort()
    fname_dict = dict()
    if cols[-1]>0:
        like = load_ftn_from_disc(benvs['norm','mid',0])
        Lx,Ly = like.Lx,like.Ly
        ls = []
        for idx,j2 in enumerate(cols):
            j1 = -1 if idx==0 else cols[idx-1]
            ls += [benvs['norm','mid',j] for j in range(j1+1,j2)]
            ls += [benvs[term_key,'mid',j2]]
        ls += [benvs['norm','right',cols[-1]]]
        ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
              ).view_as_(FermionTensorNetwork2D,like=like)
        jmax = min(cols[-1]+1,Ly-1)
        ftn_dict = ftn.compute_right_environments(yrange=(0,jmax),**compress_opts)

        term_str = directory+'_'+str(term_key)+'_right' 
        for j in range(cols[-1]):
            fname = term_str+str(j)
            write_ftn_to_disc(ftn_dict['right',j],fname)
            fname_dict[term_key,'right',j] = fname
    return fname_dict
def compute_col_envs_wrapper(info,benvs,directory,**compress_opts):
    side,term_key = info
    fxn = compute_left_envs_wrapper if side=='left' else \
          compute_right_envs_wrapper
    return fxn(term_key,benvs,directory,**compress_opts)
def compute_plq_envs_wrapper(info,benvs,directory,**compress_opts):
    term_key,j = info
    like = load_ftn_from_disc(benvs['norm','mid',0])
    Lx,Ly = like.Lx,like.Ly
    if term_key=='norm':
        cols = [j]
    else:
        cols = list(set([site[1] for site in term_key[0]]))
        cols.sort()
    l = benvs[term_key,'left',j] if j>cols[0] else benvs['norm','left',j]
    m = benvs[term_key,'mid',j] if j in cols else benvs['norm','mid',j]
    r = benvs[term_key,'right',j] if j<cols[-1] else benvs['norm','right',j]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in [l,m,r]]
          ).view_as_(FermionTensorNetwork2D,like=like)
    row_envs = ftn.compute_row_environments(yrange=(max(j-1,0),min(j+1,Ly-1)),
                                            **compress_opts)
    term_str = directory+'_'+str(term_key)+'_'
    plq_envs = dict()
    for i in range(Lx):
        ftn = FermionTensorNetwork(
              [row_envs[side,i] for side in ['bottom','mid','top']],
              check_collisions=False)
        site_tag = like.site_tag(i,j)
        fname = term_str+site_tag
        write_ftn_to_disc(ftn,fname)
        plq_envs[term_key,site_tag] = fname
    return plq_envs
def contract_plq(key,plq_envs,H):
    term_key,site_tag = key
    fname = plq_envs[key] 
    ftn = load_ftn_from_disc(fname)
    ftn.select((site_tag,'BRA'),which='!all').add_tag('grad')
    bra = ftn[site_tag,'BRA']
    ftn.contract_tags('grad',which='any',inplace=True,
                      output_inds=bra.inds[::-1])
    assert ftn.num_tensors==2
    scal = ftn.contract()
    bra_tid = bra.get_fermion_info()[0]
    bra = ftn._pop_tensor(bra_tid,remove_from_fermion_space='end')
    data = ftn['grad'].data
    if term_key!='norm':
        scal,data = scal*H[term_key][-1],data*H[term_key][-1]
    return key,scal,data
def compute_site_grad(site_tag,data_map,l,Nsites):
    H0,H1 = 0.0,0.0 # scalar/tsr H
    for (term_key,site_tag_),(scal,data) in data_map.items():
        if site_tag_==site_tag:
            if term_key=='norm':
                N0,N1 = scal,data
            else:
                H0,H1 = H0+scal,H1+data
    H0,H1 = H0/Nsites,H1/Nsites
    e = H0/N0 
    # f = (<psi|H|psi>/Nsite)/<psi|psi>+(l**2+1)*(<psi|psi>-1)**2
    g = (H1-N1*e)/N0 + (l**2+1.0)*2.0*(N0-1.0)*N1
    return site_tag,2.0*g,e,N0
def compute_grad(H,psi,l,directory,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm_fname = directory+'_norm'
    write_ftn_to_disc(norm,norm_fname)
    # get norm col envs
    fxn = compute_norm_col_envs_wrapper
    iterate_over = ['left','mid','right']
    args = [norm_fname,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for fname_dict in ls:
        benvs.update(fname_dict)
    # get term mid col envs
    fxn = apply_term
    iterate_over = list(H.keys())
    args = [H,norm_fname,directory]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)
    # get terms l&r col envs
    fxn = compute_col_envs_wrapper
    iterate_over = [(side,term_key) for side in ['left','right'] \
                    for term_key in H.keys()]
    args = [benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)
    # get terms & norm plq_envs
    fxn = compute_plq_envs_wrapper
    iterate_over = [(term_key,j) for term_key in H.keys() for j in range(Ly)]
    iterate_over += [('norm',j) for j in range(Ly)]
    args = [benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    plq_envs = dict()
    for plq_envs_term in ls:
        plq_envs.update(plq_envs_term)
    # compute components
    fxn = contract_plq
    iterate_over = list(plq_envs.keys())
    args = [plq_envs,H]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    data_map = dict()
    for (key,scal,data) in ls:
        data_map[key] = scal,data
    # sum site grad
    fxn = compute_site_grad
    iterate_over = [psi.site_tag(i,j) for i in range(Lx) for j in range(Ly)]
    args = [data_map,l,float(Lx*Ly)]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    site00 = psi.site_tag(0,0)
    g = dict()
    for (site_tag,site_g,site_e,site_N) in ls:
        g[site_tag] = site_g
        if site_tag==site00:
            e = site_e
            N = site_N
    # delete files 
    delete_ftn_from_disc(norm_fname) 
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname)
    for _,fname in plq_envs.items():
        delete_ftn_from_disc(fname)
    return g,e,N 
def compute_energy_term(term_key,benvs,directory,**compress_opts):
    like = load_ftn_from_disc(benvs['norm','mid',0])
    Lx,Ly = like.Lx,like.Ly
    if term_key=='norm':
        m,r = [benvs['norm',side,0] for side in ['mid','right']]
    else:
        right_envs = compute_right_envs_wrapper(term_key,benvs,directory,
                                                **compress_opts) 
        cols = list(set([site[1] for site in term_key[0]]))
        term = term_key if min(cols)==0 else 'norm' 
        m = benvs[term_key,'mid',0] if 0 in cols else benvs['norm','mid',0]
        r = right_envs[term_key,'right',0] if cols[-1]>0 else \
            benvs['norm','right',0]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in [m,r]]
          ).view_as_(FermionTensorNetwork2D,like=like)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    if term_key!='norm':
        for _,fname in right_envs.items():
            delete_ftn_from_disc(fname)
    return term_key,ftn.contract()
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
    norm_fname = directory+'_norm'
    write_ftn_to_disc(norm,norm_fname)
    # get norm col envs
    ftn_dict = norm.compute_right_environments(**compress_opts)
    benvs = dict()
    for (side,j),ftn in ftn_dict.items():
        fname = directory+'_norm_'+side+str(j)
        write_ftn_to_disc(ftn,fname)
        benvs['norm',side,j] = fname
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    ftn = norm.select(norm.col_tag(0)).copy()
    fname = directory+'_norm_mid0'
    write_ftn_to_disc(ftn,fname)
    benvs['norm','mid',0] = fname
    # get term cols
    fxn = apply_term
    iterate_over = list(H.keys())
    args = [H,norm_fname,directory]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        benvs.update(fname_dict)
    # compute energy numerator  
    fxn = compute_energy_term
    iterate_over = list(H.keys())+['norm'] 
    args = [benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    E = 0.0
    for (term_key,scal) in ls:
        if term_key=='norm':
            N = scal
        else:
            E += scal*H[term_key][-1]
    E /= N
    # delete files
    delete_ftn_from_disc(norm_fname) 
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname) 
    return E/float(Lx*Ly),N
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
    def __init__(self,H,peps,D,chi,directory,psi_fname):
        self.start_time = time.time()
        self.H = H
        self.D = D
        self.chi = chi
        self.directory = directory

        N = compute_norm(peps,max_bond=chi)  
        peps = peps.multiply_each(N**(-1.0/(2.0*peps.num_tensors)),inplace=True)
        peps.balance_bonds_()
        peps.equalize_norms_()

        self.psi = directory+psi_fname
        write_ftn_to_disc(peps,self.psi)
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
        psi = load_ftn_from_disc(self.psi)
        start = 0
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j) 
                ket = psi[site_tag]
                stop = start+len(ket.data.data)
                cons,dq = self.constructors[site_tag]
                ket.modify(data=cons.vector_to_tensor(x[start:stop],dq))
                start = stop
        return psi
    def compute_energy(self,x):
        psi = self.vec2fpeps(x)
        e,N = compute_energy(self.H,psi,self.directory,
                           max_bond=self.chi,cutoff=1e-15)
        print('    ne={},time={}'.format(self.ne,time.time()-self.start_time))
        print('        e={},N={}'.format(e,N))
        self.ne += 1
        return e,N
    def compute_grad(self,x):
        psi = self.vec2fpeps(x) 
        grad,e,N = compute_grad(self.H,psi,x[-1],self.directory,
                                max_bond=self.chi,cutoff=1e-15)
#        e_,N_ = compute_energy(self.H,psi,self.directory,
#                            max_bond=self.chi,cutoff=1e-15)
#        print(abs((e-e_)/e_),abs((N-N_)/N_))
        e_,N_ = (e,N) if self.chi is None else \
             compute_energy(self.H,psi,self.directory,
                            max_bond=self.chi+5,cutoff=1e-15)
        f = e+(x[-1]**2+1.0)*(N-1.0)**2  

        g = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = self.constructors[site_tag][0]
                g.append(cons.tensor_to_vector(grad[site_tag]))
        g = np.concatenate(g)
        gmax = np.amax(abs(g))
        gl = 2.0*x[-1]*(N-1.0)**2
        g = np.concatenate([g,np.ones(1)*gl])
        print('    ng={},time={}'.format(self.ng,time.time()-self.start_time))
        print('        e={},N={},gmax={},l={},gl={}'.format(e,N,gmax,x[-1],gl))
        print('        e_err={},n_err={}'.format(abs((e-e_)/e_),abs((N-N_)/N_)))
        self.ng += 1
        return f,g
    def callback(self,x):
        psi = self.vec2fpeps(x)
        write_ftn_to_disc(psi,self.psi)
    def kernel(self,method=_minimize_bfgs,options={'maxiter':200,'gtol':1e-5}):
        self.ng = 0
        self.ne = 0
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi))
        x0 = np.concatenate([x0,np.ones(1)])
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
############## Hamiltonians ##############################
def Hubbard(t,u,Lx,Ly,symmetry='u1',flat=True):
    from .block_interface import creation,onsite_U
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    ann_a = cre_a.dagger
    ann_b = cre_b.dagger
    uop = onsite_U(u=1.0,symmetry=symmetry)
    ham = dict()
    def get_on_site(site):
        ham[(site,),'u'] = (uop.copy(),),u
        return
    def get_hopping(sites):
        # cre0,ann1 = (-1)**S ann1,cre0 
        ops = cre_a.copy(),ann_a.copy()
        phase = (-1)**(cre_a.parity*ann_a.parity)
        ham[sites,'t1a'] = ops,-t*phase
        # cre1,ann0
        ops = ann_a.copy(),cre_a.copy()
        phase = 1.0
        ham[sites,'t2a'] = ops,-t*phase
        # cre0,ann1 = (-1)**S ann1,cre0 
        ops = cre_b.copy(),ann_b.copy()
        phase = (-1)**(cre_b.parity*ann_b.parity)
        ham[sites,'t1b'] = ops,-t*phase
        # cre1,ann0
        ops = ann_b.copy(),cre_b.copy()
        phase = 1.0
        ham[sites,'t2b'] = ops,-t*phase
        return
    ham = dict()
    for i in range(Lx):
        for j in range(Ly):
            get_on_site((i,j))
            if i+1 != Lx:
                get_hopping(((i,j),(i+1,j)))
            if j+1 != Ly:
                get_hopping(((i,j),(i,j+1)))
    return ham
def SpinlessFermion(t,v,Lx,Ly,symmetry='u1'):
    from .spinless import creation
    cre = creation(symmetry=symmetry)
    ann = cre.dagger
    pn = np.tensordot(cre,ann,axes=((1,),(0,)))
    def get_terms(sites):
        # cre0,ann1 = (-1)**S ann1,cre0 
        ops = cre.copy(),ann.copy()
        phase = (-1)**(cre.parity*ann.parity)
        ham[sites,'t1'] = ops,-t*phase
        # cre1,ann0
        ops = ann.copy(),cre.copy()
        phase = 1.0
        ham[sites,'t2'] = ops,-t*phase
        # pn1,pn2
        ops = pn.copy(),pn.copy()
        phase = 1.0
        ham[sites,'v'] = ops,v*phase
        return 
    ham = dict()
    for i in range(Lx):
        for j in range(Ly):
            if i+1 != Lx:
                get_terms(((i,j),(i+1,j)))
            if j+1 != Ly:
                get_terms(((i,j),(i,j+1)))
    return ham
