import numpy as np
import time,scipy,os

from .minimize import (
    _minimize_bfgs
)
from .fermion_core import (
    FermionTensor,
    FermionTensorNetwork,
)
from .fermion_2d import FermionTensorNetwork2D
from .block_interface import (
    Constructor,
    creation,
    annihilation,
    onsite_U,
    ParticleNumber,
)
from .utils import (
    parallelized_looped_function,
    parallelized_looped_function_balanced,
    worker_execution,
    insert,
    load_ftn_from_disc,
    write_ftn_to_disc,
    _profile,
)
from .fermion_2d_grad_1 import (
    data_map,sign_a,sign_b,symmetry,flat,
    compute_mid_benvs,
    compute_left_benvs_1col,
    compute_right_benvs_1col,
    compute_benvs_1col,
    compute_left_benvs_2col,
    compute_right_benvs_2col,
    compute_benvs_2col,
    compute_site_term,
    compute_energy_term,
    _pn_site,
    check_pn,
    compute_norm,
)
np.set_printoptions(suppress=True,linewidth=1000,precision=4)
class SumOpGrad:
    __slots__ = ['ls1','ls2','large','small','idx_map1','idx_map2']
    def __init__(self,ls1,ls2,large,small,idx_map1,idx_map2):
        self.ls1 = ls1
        self.ls2 = ls2
        assert len(large) >= 4
        self.large = large
        self.small = small
        self.idx_map1 = idx_map1
        self.idx_map2 = idx_map2
##################################################################
# Hamiltonians
##################################################################
def hubbard(t,u,Lx,Ly):
    ls1 = []
    ls2 = []
    idx_map1 = [[]] * Ly 
    idx_map2 = [[]] * Ly
    for j in range(Ly):
        for i in range(Lx):
            ls1.append((f'nanb{i}',j,u)) 
            if i+1 != Lx:
                ls1.append((f'cre_a{i}ann_a{i+1}',j,-t*sign_a))
                ls1.append((f'ann_a{i}cre_a{i+1}',j,-t))
                ls1.append((f'cre_b{i}ann_b{i+1}',j,-t*sign_b))
                ls1.append((f'ann_b{i}cre_b{i+1}',j,-t))
            if j+1 != Ly:
                ls2.append(('cre_a',i,j,'ann_a',i,j+1,-t*sign_a))
                ls2.append(('ann_a',i,j,'cre_a',i,j+1,-t))
                ls2.append(('cre_b',i,j,'ann_b',i,j+1,-t*sign_b))
                ls2.append(('ann_b',i,j,'cre_b',i,j+1,-t))
                idxs = list(range(len(ls2)-4,len(ls2)))
                idx_map1[j] += idxs 
                idx_map2[j+1] += idxs 
    large = []
    small = []
    for j in range(Ly):
        tmpj = []
        for i in range(Lx):
            tmpj.append(('nanb',i,j,Ly-1,0))
            if i+1 != Lx:
                tmpj.append(('cre_a',i,'ann_a',i+1,j,Ly-1,0))
                tmpj.append(('ann_a',i,'cre_a',i+1,j,Ly-1,0))
                tmpj.append(('cre_b',i,'ann_b',i+1,j,Ly-1,0))
                tmpj.append(('ann_b',i,'cre_b',i+1,j,Ly-1,0))
        large.append(tmpj) 
        tmpj = [(key,i,j,min(Ly-1,j+1),max(0,j-1)) \
                for key in ['cre_a','ann_a','cre_b','ann_b'] for i in range(Lx)]
        small.append(tmpj) 
    return SumOpGrad(ls1,ls2,large,small,idx_map2,idx_map2)
#############################################################
# gradient functions
#############################################################
def compute_grad(H,psi,tmpdir,bra_parity,profile=False,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm = write_ftn_to_disc(norm,tmpdir)
    benvs = dict()

    fxn = compute_mid_benvs
    ls = []
    for j in range(Ly):
        ls += H.large[j] + H.small[j]
    iterate_over  = [None] + [info[:-2] for info in ls]
    args = [norm,bra_parity,tmpdir]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm)

    start_time = time.time()
    fxn = compute_benvs_1col
    large = [('left',None),('right',None)]
    large += [('left',)  + info for info in H.large[0]  + H.large[1]]
    large += [('right',) + info for info in H.large[-1] + H.large[-2]]
    small  = [('left',)  + info for info in H.small[0]  + H.small[1]]
    small += [('right',) + info for info in H.small[-1] + H.small[-2]]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function_balanced(fxn,large,small,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    print(f'\t\tcompute_benvs_1col_boundary={time.time()-start_time}')

    fxn = compute_benvs_1col
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    for y1 in range(2,Ly-1):
        start_time = time.time()
        y2 = Ly-1 - y1
        large  = [('left',)  + info for info in H.large[y1]]
        large += [('right',) + info for info in H.large[y2]]
        small  = [('left',)  + info for info in H.small[y1]]
        small += [('right',) + info for info in H.small[y2]]
        ls = parallelized_looped_function_balanced(fxn,large,small,args,kwargs)
        for benvs_ in ls:
            benvs.update(benvs_)
        print(f'\t\tcompute_benvs_1col_{y1}={time.time()-start_time}')

    start_time = time.time()
    fxn = compute_benvs_2col
    iterate_over = [(side,) + info for info in H.ls2 for side in ['left','right']]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    print(f'\t\tcompute_benvs_2col={time.time()-start_time}')

    fxn = compute_site_term
    iterate_over  = list(range(Ly))
    iterate_over += [(j,) + info for j in range(Ly) for info in H.ls1 + H.ls2]
    args = [benvs,tmpdir,Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)

    # parse site_map
    H0 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    H1 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    N0 = dict() 
    N1 = dict() 
    for norm,site_map in ls:
        for (i,j),(scal,data) in site_map.items():
            if norm:
                N0[i,j] = scal
                N1[i,j] = data
            else:
                H0[i,j] = H0[i,j] + scal
                H1[i,j] = H1[i,j] + data
    for fname in benvs.values():
        os.remove(fname)
    return H0,H1,N0,N1
def compute_energy(H,psi,tmpdir,bra_parity,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm = write_ftn_to_disc(norm,tmpdir)
    benvs = dict()

    fxn = compute_mid_benvs
    ls = []
    for j in range(Ly):
        ls += H.large[j] + H.small[j]
    iterate_over  = [None] + [info[:-2] for info in ls]
    args = [norm,bra_parity,tmpdir]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm) 

    fxn = compute_right_benvs_1col 
    large = [(None,)] + H.large[-1] + H.large[-2]
    small = H.small[-1] + H.small[-2]
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function_balanced(fxn,large,small,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = compute_right_benvs_1col 
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    for y1 in range(2,Ly-1):
        y2 = Ly-1 - y1
        ls = parallelized_looped_function_balanced(fxn,H.large[y2],H.small[y2],args,kwargs)
        for benvs_ in ls:
            benvs.update(benvs_)

    fxn = compute_right_benvs_2col
    iterate_over = H.ls2
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = compute_energy_term
    iterate_over = H.ls1 + H.ls2 + [None]
    args = [benvs]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    H0 = 0.
    for norm,scal in ls:
        if norm:
            N0 = scal
        else:
            H0 += scal
    for fname in benvs.values():
        os.remove(fname) 
    return H0/N0,N0
class GlobalGrad():
    def __init__(self,H,peps,chi,psi_fname,tmpdir,small_mem=False,
                 profile=False,dense_row=True):
        self.start_time = time.time()
        self.H = H
        self.D = peps[0,0].shape[0]
        self.chi = chi
        self.profile = profile
        self.dense_row = dense_row
        self.tmpdir = tmpdir
        self.small_mem = small_mem
        self.parity = sum([T.data.parity for T in peps.tensors])
        print(f'D={self.D},chi={self.chi}')

        n = compute_norm(peps,max_bond=chi,dense_row=dense_row)
        peps.multiply_each(n**(-1.0/(2.0*peps.num_tensors)),inplace=True)
        peps.balance_bonds_()
        peps.equalize_norms_()
        self.fac = 1.0

        self.psi = psi_fname
        write_ftn_to_disc(peps,self.psi,provided_filename=True) # save state
        self.constructors = dict()
        for i in range(peps.Lx):
            for j in range(peps.Ly):
                site_tag = peps.site_tag(i,j)
                data = peps[site_tag].data
                bond_infos = [data.get_bond_info(ax,flip=False) \
                              for ax in range(data.ndim)]
                cons = Constructor.from_bond_infos(bond_infos,data.pattern)
                self.constructors[i,j] = cons,data.dq
        self.ng = 0
        self.ne = 0
        self.niter = 0
    def fpeps2vec(self,psi):
        ls = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                cons = self.constructors[i,j][0]
                ls.append(cons.tensor_to_vector(psi[i,j].data))
        return np.concatenate(ls)
    def vec2fpeps(self,x):
        psi = load_ftn_from_disc(self.psi)
        start = 0
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                cons,dq = self.constructors[i,j]
                stop = int(start+cons.get_info(dq)[-1][-1]+1e-6)
                psi[i,j].modify(data=cons.vector_to_tensor(x[start:stop],dq))
                start = stop
        return psi
    def compute_energy(self,x):
        psi = self.vec2fpeps(x)
        E,n = compute_energy(self.H,psi,self.tmpdir,self.parity,
                             dense_row=self.dense_row,max_bond=self.chi)
        print(f'\tne={self.ne},time={time.time()-self.start_time}')
        print(f'\t\tE={E},norm={n}')
        self.ne += 1
        self.fac = n**(-1.0/(2.0*psi.Lx*psi.Ly))
        print('nfile_energy=',len(os.listdir(self.tmpdir)))
        return E,n
    def compute_grad(self,x):
        psi = self.vec2fpeps(x)
        if self.small_mem:
            H0,H1,n0,n1 = compute_grad_balance(self.H,psi,self.tmpdir,self.parity,
                max_bond=self.chi,dense_row=self.dense_row,profile=self.profile)
        else:
            H0,H1,n0,n1 = compute_grad(self.H,psi,self.tmpdir,self.parity,
                max_bond=self.chi,dense_row=self.dense_row,profile=self.profile)
        g = []
        E = [] # energy
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                E.append(H0[i,j]/n0[i,j])
                gij = (H1[i,j]-n1[i,j]*E[-1])/n0[i,j]
                cons = self.constructors[i,j][0]
                g.append(cons.tensor_to_vector(gij))
        g = np.concatenate(g)
        gmax = np.amax(abs(g))
        print(f'\tng={self.ng},time={time.time()-self.start_time}')
        
        dE = max(E)-min(E)
        E = sum(E)/len(E)
        print(f'\t\tE={E},norm={n0[0,0]},gmax={gmax},dE={dE}')

        self.ng += 1
        self.fac = n0[0,0]**(-1.0/(2.0*psi.Lx*psi.Ly))
        print('nfile_gradient=',len(os.listdir(self.tmpdir)))
        return E,g
    def callback(self,x,g):
        x *= self.fac
        g /= self.fac
        psi = self.vec2fpeps(x)

        write_ftn_to_disc(psi,self.psi+f'_{self.niter}',provided_filename=True)
        write_ftn_to_disc(psi,self.psi,provided_filename=True)
        print('nfile=',len(os.listdir(self.tmpdir)))
        self.niter += 1
        return x,g
    def kernel(self,method=_minimize_bfgs,options={'maxiter':200,'gtol':1e-5}):
        self.ng = 0
        self.ne = 0
        self.niter = 0
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi))
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return

