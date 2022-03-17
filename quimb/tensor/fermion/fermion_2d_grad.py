import numpy as np
import time,scipy

from .fermion_2d_tebd import (
    insert,
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
def apply_term(term_key,H,psi_name,directory):
    sites,typ = term_key
    ops = H[term_key][0] 
    psi = load_ftn_from_disc(psi_name)
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
def compute_left_envs_wrapper(info,norm_benvs,directory,**compress_opts):
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
    term_benvs = ftn.compute_left_environments(yrange=(jmin,Ly-1),**compress_opts) 

    sites,typ = term_key
    term_str = '_'.join([ftn.site_tag(*site) for site in sites]+[str(typ)])+'_'
    for key in term_benvs.keys():
        fname = directory+term_str+key[0]+'_'+ftn.col_tag(key[1])
        write_ftn_to_disc(term_benvs[key],fname)
        term_benvs[key] = fname
    return term_key,term_benvs
def compute_right_envs_wrapper(info,norm_benvs,directory,**compress_opts): 
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
    jmax = min(cols[-1]+1,Ly-1)
    term_benvs = ftn.compute_right_environments(yrange=(0,jmax),**compress_opts)

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
def compute_site_components(site_tag,H,plq_envs):
    H0,H1 = 0.0,0.0 # scalar/tsr H
    for key in ['norm']+list(H.keys()):
        ftn = load_ftn_from_disc(plq_envs[key,site_tag])
        ftn.select((site_tag,'BRA'),which='!all').add_tag('grad')
        bra = ftn[site_tag,'BRA']
        ftn.contract_tags('grad',which='any',inplace=True,output_inds=bra.inds[::-1])
        assert ftn.num_tensors==2
        scal = ftn.contract()
        bra_tid = bra.get_fermion_info()[0]
        bra = ftn._pop_tensor(bra_tid,remove_from_fermion_space='end')
        data = ftn['grad'].data
        if key=='norm':
            N0,N1 = scal,data
        else:
            H0,H1 = H0+scal*H[key][-1],H1+data*H[key][-1]
    return site_tag,H1,N1,H0,N0
def compute_grad(H,psi_fname,directory,l=None,layer_tags=('KET','BRA'),
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
    args = [norm_benvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for (term_key,term_benvs) in ls:
        benvs[term_key] = term_benvs

    fxn = compute_right_envs_wrapper
    iterate_over = terms
    args = [norm_benvs,directory]
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
    fxn = compute_site_components
    iterate_over = [norm.site_tag(i,j) for i in range(Lx) for j in range(Ly)]
    args = [H,plq_envs]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    site00 = norm.site_tag(0,0)
    for (site_tag,H1,N1,H0,N0) in ls:
        if site_tag == site00:
            H00,N00 = H0,N0
    E = H00/N00
    grad = dict()
    for (site_tag,H1,N1,H0,N0) in ls:
        # f = <psi|H|psi>/<psi|psi>
        grad[site_tag] = (H1-N1*E)/N0
        # f = <psi|H|psi>/<psi|psi>+lambda*(<psi|psi>-1)**2
#        grad[site_tag] = (H1-N1*E)/N0+2.0*l*(N00-1.0)*N1
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
    return grad,H00,N00,E
def compute_norm(psi_name,directory,layer_tags=('KET','BRA'),
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    psi = load_ftn_from_disc(psi_fname)
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    right_envs = norm.compute_right_environments(**compress_opts)

    ftn = FermionTensorNetwork(
          (norm.select(norm.col_tag(0)).copy(),right_envs['right',0])
          ).view_as_(FermionTensorNetwork2D,like=norm)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    return ftn.contract() 
def compute_energy_term(info,norm_benvs,directory,**compress_opts):
    term_key,term_dict = info
    like = load_ftn_from_disc(norm_benvs['mid',0])
    Lx,Ly = like.Lx,like.Ly

    _,right_envs = compute_right_envs_wrapper(info,norm_benvs,directory,**compress_opts) 
    col0 = term_dict[0] if 0 in term_dict else norm_benvs['mid',0]
    ls = col0,right_envs['right',0]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=like)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    for _,fname in right_envs.items():
        delete_ftn_from_disc(fname)
    return term_key,ftn.contract()
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
    ls = norm_benvs['mid',0],norm_benvs['right',0]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=norm)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    N = ftn.contract() 
    # delete files
    for _,fname in norm_benvs.items():
        delete_ftn_from_disc(fname) 
    for (_,term_dict) in terms:
        for _,fname in term_dict.items():
            delete_ftn_from_disc(fname)
    return E/N
class GlobalGrad():
    def __init__(self,H,peps,D,chi,directory):
        self.H = H
        self.D = D
        self.chi = chi
        self.directory = directory

        self.psi = directory+'psi'
        write_ftn_to_disc(peps,self.psi)
        self.skeleton = directory+'skeleton'
        write_ftn_to_disc(peps,self.skeleton)
        self.tmp = directory+'tmp'
        self.constructors = dict()
        for i in range(peps.Lx):
            for j in range(peps.Ly):
                site_tag = peps.site_tag(i,j)
                data = peps[site_tag].data
                bond_infos = [data.get_bond_info(ax,flip=False) \
                              for ax in range(data.ndim)]
                cons = Constructor.from_bond_infos(bond_infos,data.pattern)
                self.constructors[site_tag] = cons,data.dq
    def fpeps2vec(self,psi):
        ls = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = self.constructors[site_tag][0]
                ls.append(cons.tensor_to_vector(psi[site_tag].data))
        return np.concatenate(ls)
    def vec2fpeps(self,x):
        psi = load_ftn_from_disc(self.skeleton)
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
    def compute_energy(self,x_):
        if self.has_lambda:
            x,l = x_[:len(x)-1],x_[-1]
        else:
            x,l = x_,None 
        psi = self.vec2fpeps(x) 
        write_ftn_to_disc(psi,self.tmp) 
        return compute_energy(self.H,self.tmp,self.directory,max_bond=self.chi)
    def compute_grad(self,x_):
        x = x_[:len(x_)-1] if self.has_lambda else x_
        l = x_[-1] if self.has_lambda else None
        psi = self.vec2fpeps(x) 
        write_ftn_to_disc(psi,self.tmp) 
        grad,H00,N00,E = compute_grad(self.H,self.tmp,self.directory,l=l,
                                  max_bond=self.chi)

        g = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = self.constructors[site_tag][0]
                g.append(cons.tensor_to_vector(grad[site_tag]))
        g = np.concatenate(g)
        gmax = np.amax(abs(g))
        gl = (N00-1.0)**2 if self.has_lambda else None
        if self.has_lambda:
            g = np.concatenate([g,np.ones(1)*gl])
        f = E+l*(N00-1.0)**2 if self.has_lambda else E

#        E_ = compute_energy(self.H,self.tmp,self.directory,max_bond=self.chi)
#        print(abs((E-E_)/E))
        if self.chi is None:
            print('eval={},E={},N={},gmax={},l={},gl={}'.format(
                  self.k,E,N00,gmax,l,gl))
        else:
            E_ = compute_energy(self.H,self.tmp,self.directory,max_bond=self.chi//2)
            print('eval={},E={},N={},Eerr={},gmax={},l={},gl={}'.format(
                  self.k,E,N00,abs((E-E_)/E),gmax,l,gl))
        self.k += 1
        if self.sep_energy_grad:
            return g
        else:
            return f,g
    def callback(self,x):
        psi = self.vec2fpeps(x)
        write_ftn_to_disc(psi,self.psi)
        delete_ftn_from_disc(self.tmp)
        print('iter=',self.it)
        self.it += 1
    def kernel(self,method='BFGS',options={'maxiter':100,'gtol':1e-5}):
        self.sep_energy_grad = False
        self.has_lambda = False
        fun = self.compute_energy if self.sep_energy_grad else self.compute_grad
        jac = self.compute_grad if self.sep_energy_grad else True
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi))
        if self.has_lambda:
            x0 = np.concatenate([x0,1.0*np.ones(1)])
        self.it = 0
        self.k = 0
        result = scipy.optimize.minimize(fun=fun,jac=jac,method=method,x0=x0,
                 callback=self.callback,options=options)
        x = result.x[:len(result.x)-1] if self.has_lambda else result.x
        psi = self.vec2fpeps(x) 
        write_ftn_to_disc(psi,self.psi)
        return result.fun
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
################### some naive numerical methods #####################
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
def kernel(H,psi_name,directory,chi,maxiter=50,gtol=1e-5,t0=0.5):
    print('using fpeps update')
    E_old = 0.0
    d_old,gnormsq_old = None,None
    for i in range(maxiter):
        grad,E,N = compute_grad(H,psi_name,directory,max_bond=chi)
        d,gnormsq = conjugate_grad(grad,d_old,gnormsq_old)
#        t,E = line_search(d,E_old,t0,H,psi_name,directory,max_bond=chi) 
        t = t0
        g = [gi.data for _,gi in grad.items()]
        g = np.concatenate(g)
#        E_ = compute_energy(H,psi_name,directory,max_bond=chi)
#        print(abs((E-E_)/E))
        E_ = compute_energy(H,psi_name,directory,max_bond=chi//2)
        print('iter={},E={},N={},Eerr={},gmax={},t={}'.format(
               i,E,N,abs((E-E_)/E),np.amax(abs(g)),t))
        if E-E_old>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
#        d_old,gnormsq_old,E_old = d.copy(),gnormsq,E
        E_old = E
        psi = load_ftn_from_disc(psi_name)
        for site_tag,data in d.items():
            psi[site_tag].modify(data=psi[site_tag].data+t*data)
        write_ftn_to_disc(psi,psi_name)
    print('end iteration')
    return E
def kernel_vec(H,psi_name,directory,chi,maxiter=50,gtol=1e-5,t0=0.5):
    print('using vector update')
    E_old = 0.0

    peps = load_ftn_from_disc(psi_name) 
    constructors = dict()
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            site_tag = peps.site_tag(i,j)
            data = peps[site_tag].data
            bond_infos = [data.get_bond_info(ax,flip=False) \
                          for ax in range(data.ndim)]
            cons = Constructor.from_bond_infos(bond_infos,data.pattern)
            constructors[site_tag] = cons,data.dq
    write_ftn_to_disc(peps,directory+'skeleton')

    psi = peps
    ls = []
    for i in range(psi.Lx):
        for j in range(psi.Ly):
            site_tag = psi.site_tag(i,j)
            cons = constructors[site_tag][0]
            ls.append(cons.tensor_to_vector(psi[site_tag].data))
    x = np.concatenate(ls)
    for it in range(maxiter):
        psi = load_ftn_from_disc(directory+'skeleton')
        start = 0
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j) 
                ket = psi[site_tag]
                stop = start+len(ket.data.data)
                cons,dq = constructors[site_tag]
                ket.modify(data=cons.vector_to_tensor(x[start:stop],dq))
                start = stop
        write_ftn_to_disc(psi,directory+'tmp')
        grad,E,N = compute_grad(H,directory+'tmp',directory,max_bond=chi)
        t = t0
        g = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = constructors[site_tag][0]
                g.append(cons.tensor_to_vector(grad[site_tag]))
        g = np.concatenate(g)
#        E_ = compute_energy(H,directory+'tmp',directory,max_bond=chi)
#        print(abs((E-E_)/E))
        E_ = compute_energy(H,directory+'tmp',directory,max_bond=chi//2)
        print('iter={},E={},N={},Eerr={},gmax={},t={}'.format(
              it,E,N,abs((E-E_)/E),np.amax(abs(g)),t))
        if E-E_old>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
        x = x - t*g  
    print('end iteration')
    return
def custom(fun,x0,jac=None,callback=None,gtol=1e-5,maxiter=50,t0=0.5,**options):
    print('using scipy.optimize custom')
    fun,grad = fun,jac
    t = t0
    print('t=',t)
    x = x0
    fold = 0.0
    nit = 0
    while nit < maxiter:
        f,g = fun(x),grad(x) 
        if f-fold>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
        x = x - t*g
        if callback is not None:
            callback(x)
        fold = f
        nit += 1
    return scipy.optimize.OptimizeResult(fun=f,jac=g,x=x,nit=nit,message='')
def _minimize_bfgs(fun, x0, args=(), jac=None, callback=None, maxiter=None,
                   gtol=1e-5, norm=np.Inf, eps=np.sqrt(np.finfo(float).eps), 
                   disp=False, return_all=False, finite_diff_rel_step=None,
                   hess=None,hessp=None,bounds=None,constraints=(),
                   **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    BFGS algorithm.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    maxiter : int
        Maximum number of iterations to perform.
    gtol : float
        Gradient norm must be less than `gtol` before successful
        termination.
    norm : float
        Order of norm (Inf is max, -Inf is min).
    eps : float or ndarray
        If `jac is None` the absolute step size used for numerical
        approximation of the jacobian via forward differences.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    finite_diff_rel_step : None or array_like, optional
        If `jac in ['2-point', '3-point', 'cs']` the relative step size to
        use for numerical approximation of the jacobian. The absolute step
        size is computed as ``h = rel_step * sign(x) * max(1, abs(x))``,
        possibly adjusted to fit into the bounds. For ``method='3-point'``
        the sign of `h` is ignored. If None (default) then step is selected
        automatically.
    """
    from scipy.optimize import optimize
    optimize._check_unknown_options(unknown_options)
    retall = return_all

    x0 = np.asarray(x0).flatten()
    if x0.ndim == 0:
        x0.shape = (1,)
    if maxiter is None:
        maxiter = len(x0) * 200

    sf = optimize._prepare_scalar_function(fun, x0, jac, args=args, epsilon=eps,
                                  finite_diff_rel_step=finite_diff_rel_step)

    f = sf.fun
    myfprime = sf.grad

    old_fval = f(x0)
    gfk = myfprime(x0)

    k = 0
    N = len(x0)
    I = np.eye(N, dtype=int)
    Hk = I

    # Sets the initial step guess to dx ~ 1
    old_old_fval = old_fval + np.linalg.norm(gfk) / 2

    xk = x0
    if retall:
        allvecs = [x0]
    warnflag = 0
    gnorm = optimize.vecnorm(gfk, ord=norm)
    while (gnorm > gtol) and (k < maxiter):
        pk = -np.dot(Hk, gfk)
        try:
            alpha_k, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     optimize._line_search_wolfe12(f, myfprime, xk, pk, gfk,
                         old_fval, old_old_fval, amin=1e-100, amax=1e100)
        except optimize._LineSearchError:
            # Line search failed to find a better solution.
            warnflag = 2
            break

        xkp1 = xk + alpha_k * pk
        if retall:
            allvecs.append(xkp1)
        sk = xkp1 - xk
        nsk = np.linalg.norm(sk)
        sk /= nsk
        xk = xkp1
        if gfkp1 is None:
            gfkp1 = myfprime(xkp1)

        yk = gfkp1 - gfk
        yk /= nsk
        gfk = gfkp1
        if callback is not None:
            callback(xk)
        k += 1
        gnorm = optimize.vecnorm(gfk, ord=norm)
        if (gnorm <= gtol):
            break

        if not np.isfinite(old_fval):
            # We correctly found +-Inf as optimal value, or something went
            # wrong.
            warnflag = 2
            break

        rhok_inv = np.dot(yk, sk)
        print('rhok_inv={},nsk={},nyk={},alpha={}'.format(rhok_inv,nsk,np.linalg.norm(yk),alpha_k))
        # this was handled in numeric, let it remaines for more safety
        if rhok_inv == 0.:
            rhok = 1000.0
            if disp:
                print("Divide-by-zero encountered: rhok assumed large")
        else:
            rhok = 1. / rhok_inv

        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok
        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok
        Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])

    fval = old_fval

    if warnflag == 2:
        msg = optimize._status_message['pr_loss']
    elif k >= maxiter:
        warnflag = 1
        msg = optimize._status_message['maxiter']
    elif np.isnan(gnorm) or np.isnan(fval) or np.isnan(xk).any():
        warnflag = 3
        msg = optimize._status_message['nan']
    else:
        msg = optimize._status_message['success']

    if disp:
        print("%s%s" % ("Warning: " if warnflag != 0 else "", msg))
        print("         Current function value: %f" % fval)
        print("         Iterations: %d" % k)
        print("         Function evaluations: %d" % sf.nfev)
        print("         Gradient evaluations: %d" % sf.ngev)

    result = optimize.OptimizeResult(fun=fval, jac=gfk, hess_inv=Hk, nfev=sf.nfev,
                            njev=sf.ngev, status=warnflag,
                            success=(warnflag == 0), message=msg, x=xk,
                            nit=k)
    if retall:
        result['allvecs'] = allvecs
    return result
def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=np.sqrt(np.finfo(float).eps), 
                       maxiter=None, disp=False, return_all=False,
                       bounds=None,constraints=(),
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.
    Note that the `jac` parameter (Jacobian) is required.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    from scipy.optimize import optimize
    optimize._check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0 = np.asarray(x0).flatten()
    # TODO: add hessp (callable or FD) to ScalarFunction?
    sf = optimize._prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )
    f = sf.fun
    fprime = sf.grad
    _h = sf.hess(x0)

    # Logic for hess/hessp
    # - If a callable(hess) is provided, then use that
    # - If hess is a FD_METHOD, or the output fom hess(x) is a LinearOperator
    #   then create a hessp function using those.
    # - If hess is None but you have callable(hessp) then use the hessp.
    # - If hess and hessp are None then approximate hessp using the grad/jac.

    if (hess in optimize.FD_METHODS or isinstance(_h, scipy.sparse.linalg.LinearOperator)):
        fhess = None

        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)

        fhess_p = _hessp

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = optimize.OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
#    while np.add.reduce(np.abs(update)) > xtol:
    b = np.ones(len(x0))
    while np.amax(np.abs(b)) > avextol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.add.reduce(np.abs(b))
        eta = np.min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = np.zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = sf.hess(xk)
            hcalls = hcalls + 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            if np.amax(abs(ri)) < avextol:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = optimize.approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                if isinstance(A, HessianUpdateStrategy):
                    # if hess was supplied as a HessianUpdateStrategy
                    Ap = A.dot(psupi)
                else:
                    Ap = np.dot(A, psupi)
            # check curvature
            Ap = np.asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            print('curv={},npi={},rmax={}'.format(curv/np.dot(psupi,psupi),np.linalg.norm(psupi),np.amax(abs(ri))))
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1          # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     optimize._line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except optimize._LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk = xk + update        # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1
    else:
        if np.isnan(old_fval) or np.isnan(update).any():
            return terminate(3, _status_message['nan'])

        msg = optimize._status_message['success']
        return terminate(0, msg)

