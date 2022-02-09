import numpy as np
import time,multiprocessing,functools

from itertools import product
from .fermion_2d_tebd import insert,copy
from .fermion_core import FermionTensor,FermionTensorNetwork
from ..tensor_2d import Rotator2D

def apply_term(args):
    sites,term_type,ops,fac,ftn = args
    n = ftn.num_tensors//2
    print(ftn.fermion_space)
    for where,op in zip(sites,ops):
        ket = ftn[ftn.site_tag(*where),'KET']
        site_ix = ket.inds[-1]
        bnd = site_ix+'_'
        print(ket.phase)
        ket.reindex_({site_ix:bnd})
        ket_tid,ket_site = ket.get_fermion_info()

        TG = FermionTensor(op.copy(),inds=(site_ix,bnd),left_inds=(site_ix,),
                           tags=ket.tags)
        ftn = insert(ftn,n,TG)
        TG_tid,TG_site = TG.get_fermion_info()
        ftn.fermion_space.move(TG_tid,ket_site+1)
        
        ftn.contract_tags(ket.tags,which='all',inplace=True)
    return (sites,term_type,fac,ftn)
def apply_terms(norm,H,nworkers=5):
    ftn_map = dict()
    ftn_map['norm'] = 1.0,norm

    pool = multiprocessing.Pool(nworkers)
    args_ls = [tuple(key)+tuple(val)+(norm.copy(full=True),) for key,val in H.items()]
    ls = pool.map(apply_term,args_ls)
    for (sites,term_type,fac,ftn) in ls:
        ftn_map[sites,term_type] = fac,ftn
    return ftn_map
def contract_boundary(args):
    key,bix,orth_range,from_which,tn,kwargs = args
    step = 1 if from_which in {'bottom','left'} else -1
    if from_which in {'bottom','top'}:
        tn.contract_boundary_from_(
            xrange=(bix-2*step,bix-step),yrange=orth_range,
            from_which=from_which,**kwargs)
    else:
        tn.contract_boundary_from_(
            yrange=(bix-2*step,bix-step),xrange=orth_range,
            from_which=from_which,**kwargs)
    return (key,bix,from_which,tn)
def get_benvs(ftn_map,direction,Lbix,Lix,nworkers=5,
              max_bond=None,cutoff=1e-10,canonize=True,mode='mps',
              layer_tags=('KET','BRA')):
    kwargs = dict()
    kwargs['max_bond'] = max_bond
    kwards['cutoff'] = cutoff
    kwargs['canonize'] = canonize
    kwargs['mode'] = mode
    kwargs['layer_tags'] = layer_tags
    # middle:
    benvs = dict()
    if direction=='col':
        sides = 'left','right'
    else:
        sides = 'bottom','top'
    for key in ftn_map.keys():
        ftn = ftn_map[key]
        ftn.reorder_(direction,layer_tags=layer_tags)
        tag = ftn.col_tag if direction=='col' else ftn.row_tag
        for bix in range(Lbix):
            benvs[key,'mid',bix] = ftn.select(tag(bix)).copy()
        benvs[key,sides[0],0] = FermionTensorNetwork([])
        benvs[key,sides[0],1] = ftn.select(tag(0)).copy()
        benvs[key,sides[1],Lbix-1] = FermionTensorNetwork([])
        benvs[key,sides[1],Lbix-2] = ftn.select(tag(Lbix-1)).copy()

    orth_range = 0,Lix-1
    def fxn(from_which,bix_range,bix_0):
        tmp = dict()
        for key,(_,ftn) in ftn_map.items():
            tmp[key] = ftn.copy()
        for bix in bix_range:
            pool = multiprocessing.Pool(nworkers)
            args_ls = []
            for key,ftn in tmp.items():
                args = key,bix,orth_range,from_which,ftn,kwargs
                args_ls.append(args)
            ls = pool.map(contract_boundary,args_ls)
            for (key,bix,from_which,ftn) in ls:
                tag = ftn.col_tag if direction=='col' else ftn.row_tag
                benvs[key,from_which,bix] = ftn.select(tag(bix_0)).copy()
                tmp[key] = ftn
        return
    fxn(sides[0],range(2,Lbix),0)
    fxn(sides[1],range(Lbix-3,-1,-1),Lbix-1)
    return benvs
def get_plq_envs(ftn_map,benvs,direction,Lbix,Lix,nowrkers=5, 
                 max_bond=None,cutoff=1e-10,canonize=True,mode='mps',
                 layer_tags=('KET','BRA')):
    kwargs = dict()
    kwargs['max_bond'] = max_bond
    kwards['cutoff'] = cutoff
    kwargs['canonize'] = canonize
    kwargs['mode'] = mode
    kwargs['layer_tags'] = layer_tags

    bftn_map = dict()
    if direction=='row':
       from_which,to_which = 'left','right'
       sides = 'bottom','top'  
    else:
        from_which,to_which = 'bottom','top'  
        sides = 'left','right'
    for key in ftn_map.keys():
        for bix in range(Lbix):
            bftn_map[key,bix] = FermionTensorNetwork((
                benvs[key,from_which,bix],
                benvs[key,mid,bix],
                benvs[key,to_which,bix],
            )).view_as_(FermionTensorNetwork2D,like=ftn_map[key][-1])

    plq_envs = dict()
    for key in ftn_map.keys():
        for bix in range(Lbix):
            ftn = bftn_map[key,bix]
            ftn.reorder_(direction,layer_tags=layer_tags)
            tag = ftn.col_tag if direction=='col' else ftn.row_tag
            for ix in range(Lix):
                plq_envs[key,bix,'mid',ix] = ftn.select(tag(ix)).copy()
            plq_envs[key,bix,sides[0],0] = FermionTensorNetwork([])
            plq_envs[key,bix,sides[0],1] = ftn.select(tag(0)).copy()
            plq_envs[key,bix,sides[1],Lix-1] = FermionTensorNetwork([])
            plq_envs[key,bix,sides[1],Lix-2] = ftn.select(tag(Lix-1)).copy()
    def fxn(from_which,ix_range,ix_0):
        tmp = dict()
        for key in ftn_map.keys():
            for bix in range(Lbix):
                tmp[key,bix] = bftn[key,bix].copy()
        for ix in ix_range:
            pool = multiprocessing.Pool(nworkers)
            args_ls = []
            for key_,ftn in tmp.items():
                bix = key_[-1]
                orth_range = max(bix-1,0),min(bix+1,Lbix-1)
                args = key_,ix,orth_range,from_which,ftn,kwargs
                args_ls.append(args)
            ls = pool.map(contract_boundary,args_ls)
            for (key_,ix,from_which,ftn) in ls:
                key,bix = key_
                tag = ftn.col_tag if direction=='col' else ftn.row_tag
                plq_envs[key,bix,from_which,ix] = ftn.select(tag(ix_0)).copy()
                tmp[key_] = ftn
        return
    fxn(sides[0],range(2,Lix),0)
    fxn(sides[1],range(Lix-3,-1,-1),Lix-1)
    exit()

#    plq_ftn_map = dict()
#    for key in ftn_map.keys():
#        for bix in range(Lbix):

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
def get_plq_envs(ftn_map,
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps',layer_tags=('KET','BRA'),
    **plq_env_opts):
    plq_env_opts['max_bond'] = max_bond
    plq_env_opts['cutoff'] = cutoff
    plq_env_opts['canonize'] = canonize
    plq_env_opts['mode'] = mode
    plq_env_opts['layer_tags'] = layer_tags
    x_bsz = y_bsz = 1
    env_map = dict()
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
            benvs = get_benvs(ftn_map,'col',self._psi.Ly,self._psi.Lx,max_bond=self.chi)
#            env_map = get_plq_envs(ftn_map,max_bond=self.chi)
#            flat_env_map = term_to_flat(env_map)
#            flat_component_map = get_components(flat_env_map,norm)
#            component_map = flat_to_site(flat_component_map)
#            grad_map = get_grad(component_map)
#            exit() 
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
