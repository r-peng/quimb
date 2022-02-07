import numpy as np
import time

from itertools import product
from .fermion_2d_tebd import insert,copy
from .fermion_core import FermionTensor

########### serial fxns #######################
def apply_term(term,norm):
    ftn = norm.copy()
#    ftn = copy(norm)
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
def apply_terms(norm,H):
    ftn_map = dict()
    ftn_map['norm'] = norm,1.0
    for i,term in enumerate(H):
        ftn_map[i] = apply_term(term,norm)
    return ftn_map
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
