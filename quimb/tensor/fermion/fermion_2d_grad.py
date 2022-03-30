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
    term_str = directory+'_norm_'
    if side == 'left':
        ftn_dict = norm.compute_left_environments(**compress_opts)
    else:
        ftn_dict = norm.compute_right_environments(**compress_opts)
    for (side_,j),ftn in ftn_dict.items():
        skip = (side_=='mid') and (side=='left') and (j>0)
        if not skip:
            fname = term_str+side_+str(j)
            write_ftn_to_disc(ftn,fname)
            fname_dict['norm',side_,j] = fname
    return fname_dict
def apply_term(info,norm_fname,directory):
    sites,typ,ops = info
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
    term_key = sites,typ
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
def compute_row_envs_wrapper(info,benvs,directory,**compress_opts):
    side,term_key,j = info
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
    fname_dict = dict()
    term_str = directory+'_'+str(term_key)+'_'+ftn.col_tag(j)+'_'
    if side=='top':
        ftn_dict = ftn.compute_top_environments(
                   yrange=(max(j-1,0),min(j+1,Ly-1)),**compress_opts)
    else:
        ftn_dict = ftn.compute_bottom_environments(
                   yrange=(max(j-1,0),min(j+1,Ly-1)),**compress_opts)
    for (side_,i),ftn in ftn_dict.items():
        skip = (side_=='mid') and (side=='bottom') and (i>0)
        if not skip:
            fname = term_str+side_+str(i)
            write_ftn_to_disc(ftn,fname)
            fname_dict[term_key,j,side_,i] = fname
    return fname_dict
def contract_plq(info,envs):
    term_key,i,j,fac = info
    like = load_ftn_from_disc(envs['norm',0,'mid',0])
    ls = [envs[term_key,j,side,i] for side in ['bottom','mid','top']]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls],
          check_collisions=False)
    site_tag = like.site_tag(i,j)
    ftn.select((site_tag,'BRA'),which='!all').add_tag('grad')
    bra = ftn[site_tag,'BRA']
    ftn.contract_tags('grad',which='any',inplace=True,
                      output_inds=bra.inds[::-1])
    assert ftn.num_tensors==2
    scal = ftn.contract()
    bra_tid = bra.get_fermion_info()[0]
    bra = ftn._pop_tensor(bra_tid,remove_from_fermion_space='end')
    data = ftn['grad'].data
    return (i,j),term_key,scal*fac,data*fac
def compute_site_grad(info):
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
    norm_fname = directory+'_norm'
    write_ftn_to_disc(norm,norm_fname)
    # get norm col envs
    fxn = compute_norm_col_envs_wrapper
    iterate_over = ['left','right']
    args = [norm_fname,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for fname_dict in ls:
        benvs.update(fname_dict)
    # get term mid col envs
    fxn = apply_term
    iterate_over = [(sites,typ,ops) for (sites,typ),(ops,_) in H.items()] 
    args = [norm_fname,directory]
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
    # get terms & norm row envs
    fxn = compute_row_envs_wrapper
    iterate_over = [(side,term_key,j) for side in ['top','bottom']\
                     for term_key in list(H.keys())+['norm'] for j in range(Ly)]
    args = [benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    envs = dict()
    for fname_dict in ls:
        envs.update(fname_dict)
    # compute components
    fxn = contract_plq
    iterate_over = [(term_key,i,j,fac) for term_key,(_,fac) in H.items()\
                     for i in range(Lx) for j in range(Ly)]
    iterate_over += [('norm',i,j,1.0) for i in range(Lx) for j in range(Ly)] 
    args = [envs]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    data_map = dict()
    for (site,term_key,scal,data) in ls:
        if site not in data_map:
            data_map[site] = dict()
        data_map[site][term_key] = scal,data
    # sum site grad
    fxn = compute_site_grad
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
    # delete files 
    delete_ftn_from_disc(norm_fname) 
    for _,fname in benvs.items():
        delete_ftn_from_disc(fname)
    for _,fname in envs.items():
        delete_ftn_from_disc(fname)
    return g,E,N 
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
    iterate_over = [(sites,typ,ops) for (sites,typ),(ops,_) in H.items()]
    args = [norm_fname,directory]
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
    return E,N
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
        self.fac = 1.0

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
        E,N = compute_energy(self.H,psi,self.directory,
                           max_bond=self.chi,cutoff=1e-15)
        print('    ne={},time={}'.format(self.ne,time.time()-self.start_time))
        print('        E={},N={}'.format(E,N))
        self.ne += 1
        self.fac = N**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,N
    def compute_grad(self,x):
        psi = self.vec2fpeps(x) 
        grad,E,N = compute_grad(self.H,psi,self.directory,
                                max_bond=self.chi,cutoff=1e-15)
#        E_,N_ = compute_energy(self.H,psi,self.directory,
#                            max_bond=self.chi,cutoff=1e-15)
#        print(abs((E-E_)/E_),abs((N-N_)/N_))
        E_,N_ = (E,N) if self.chi is None else \
             compute_energy(self.H,psi,self.directory,
                            max_bond=self.chi+5,cutoff=1e-15)

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
        write_ftn_to_disc(psi,self.psi)
        return x,g
    def kernel(self,method=_minimize_bfgs,options={'maxiter':200,'gtol':1e-5}):
        self.ng = 0
        self.ne = 0
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi))
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
    nanb = onsite_U(u=1.0,symmetry=symmetry)
    ham = dict()
    def get_on_site(site):
        ham[(site,),'u'] = (nanb.copy(),),u
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
def UEG(g,N,L,Ne,symmetry='u1',flat=True,maxdist=np.inf):
    from .block_interface import creation,onsite_U,ParticleNumber
    # parameters
    eps = L/(N+2.0) # 4th order discretization parameter
    spacing = L/(N+1.0) # spacing
    n = N2/N**2 # background charge density
    # operators
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    ann_a = cre_a.dagger
    ann_b = cre_b.dagger
    nanb = onsite_U(u=1.0,symmetry=symmetry)
    pn = ParticleNumber(spin='sum',symmetry=symmetry,flat=flat)
    def get_onsite(site):
        # electron-charge
        fac = 0.0
        for site_ in sites:
            r = np.array(site)-np.array(site_)
            dsq = np.dot(r,r) # lattice distance square
            fac += 1.0/(spacing*np.sqrt(dsq+1.0))
        fac *= -n 
        # KE
        if site in [(0,0),(0,N-1),(N-1,0),(N-1,N-1)]: # corner
            fac += 58.0/(24.0*eps**2)
        elif (site[0] in [0,N-1]) or (site[1] in [0,N-1]): # border
            fac += 59.0/(24.0*eps**2)
        else:
            fac += 60.0/(24.0*eps**2) # bulk
        op = fac*pn
        # onsite e-e
        op = op + 1.0/spacing*nanb
        ham[(site,),'onsite'] = (op,),1.0
        return
    def get_pair(sites):
        site0,site1 = sites
        assert site0[0]<=site1[0]
        assert site0[1]<=site1[1]
        # electron-electron
        r = np.array(site0)-np.array(site1)
        dsq = np.dot(r,r) # lattice distance square
        dist = spacing*np.sqrt(dsq+1.0)
        if dist <= maxdist:
            fac = 1.0/dist
            ham[sites,'lr'] = (pn.copy(),pn.copy()),fac
        # NN or 3rd NN:
        if abs(dsq-1.0)<1e-3: # dsq==1
            fac,typ,include = -16.0/(24.0*eps**2),'nn',True
        elif abs(dsq-9.0)<1e-3: # dsq==9 
            fac,typ,include = -1.0/(24.0*eps**2),'3nn',True
        else:
            include = False
        if include:
            # cre0,ann1 = (-1)**S ann1,cre0
            ops = cre_a.copy(),ann_a.copy()
            phase (-1)**(cre_a.parity*ann_a.parity)
            ham[sites,typ+'1a'] = ops,fac*phase
            # cre1,ann0
            ops = ann_a.copy(),cre_a.copy()
            phase = 1.0
            ham[sites,typ+'2a'] = ops,fac*phase
            # cre0,ann1 = (-1)**S ann1,cre0 
            ops = cre_b.copy(),ann_b.copy()
            phase = (-1)**(cre_b.parity*ann_b.parity)
            ham[sites,typ+'1b'] = ops,fac*phase
            # cre1,ann0
            ops = ann_b.copy(),cre_b.copy()
            phase = 1.0
            ham[sites,typ+'2b'] = ops,fac*phase
        return
    sites = [(i,j) for i in range(N) for j in range(N)]
    ham = dict()
    for i in range(len(sites)):
        get_insite(sites[i])
        for j in range(i+1,len(sites)):
            get_pair((sites[i],sites[j]))
    # const
    const = 0.0
    for site in sites:
        for site_ in sites:
            r = np.array(site)-np.array(site_)
            dsq = np.dot(r,r) # lattice distance square
            const += 1.0/(spacing*np.sqrt(dsq+1.0))
        const *= 0.5*n**2 
    return ham,const
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
