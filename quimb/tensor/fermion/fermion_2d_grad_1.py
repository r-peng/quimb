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
from .block_interface import Constructor
from .utils import (
    parallelized_looped_function,
    worker_execution,
    insert,
    load_ftn_from_disc,
    write_ftn_to_disc,
    _profile,
)
from .spin_utils import data_map,sign_a,sign_b 
np.set_printoptions(suppress=True,linewidth=1000,precision=4)

def distribute(ls):
    ls1 = []
    ls2 = []
    while len(ls)>0:
        info = ls.pop()
        if len(info)==7:
            ls1.append(info)
        else:
            if info[0]=='nanb':
                ls1.append(info)
            else:
                ls2.append(info)
    return ls1,ls2
def distribute1(ls):
    ls1 = []
    ls2 = []
    while len(ls)>0:
        info = ls.pop()
        if info[0][:2]=='pn':
            ls_ = info[0][2:].split('pn')
            if len(ls_)==1:
                ls2.append(info)
            else:
                ls1.append(info)
        else:
            ls1.append(info)
    return ls1,ls2
class SumOpGrad:
    def __init__(self,ls1,ls2,tmp):
        self.ls1 = ls1
        self.ls2 = ls2
        print('number of 1col terms =',len(ls1))
        print('number of 2col terms =',len(ls2))

        Ly = len(tmp)
        assert Ly >= 2

        self.r0 = tmp.pop()
        self.r1 = tmp.pop()
        self.m = []
        for j in range(Ly-3,1,-1):
            self.m += tmp.pop()
        try:
            self.l0 = tmp.pop(0)
        except IndexError:
            self.l0 = []
        try:
            self.l1 = tmp.pop(0)
        except IndexError:
            self.l1 = []
        print('number of reuse terms=',len(self.l0+self.l1+self.r0+self.r1+self.m))
    def distribute(self):
        self.m_1,self.m_2 = distribute(self.m)
        self.l0_1,self.l0_2 = distribute(self.l0)
        self.l1_1,self.l1_2 = distribute(self.l1)
        self.r0_1,self.r0_2 = distribute(self.r0)
        self.r1_1,self.r1_2 = distribute(self.r1)
        self.ls1_1,self.ls1_2 = distribute1(self.ls1)
##################################################################
# Hamiltonians
##################################################################
def hubbard(t,u,Lx,Ly):
    ls1 = []
    ls2 = []
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
    tmp = []
    for j in range(Ly):
        tmpj = [('nanb',i,j,Ly-1,0) for i in range(Lx)]
        for i in range(Lx-1):
            tmpj.append(('cre_a',i,'ann_a',i+1,j,Ly-1,0))
            tmpj.append(('ann_a',i,'cre_a',i+1,j,Ly-1,0))
            tmpj.append(('cre_b',i,'ann_b',i+1,j,Ly-1,0))
            tmpj.append(('ann_b',i,'cre_b',i+1,j,Ly-1,0))
        tmpj += [(key,i,j,min(Ly-1,j+1),max(0,j-1)) \
                for key in ['cre_a','ann_a','cre_b','ann_b'] for i in range(Lx)]
        tmp.append(tmpj) 
    return SumOpGrad(ls1,ls2,tmp)
#############################################################
# gradient functions
#############################################################
def compute_mid_benvs(info,norm,bra_parity,tmpdir):
    ftn = load_ftn_from_disc(norm)
    col = ftn.col_tag
    if info is None:
        return {('norm','mid',j):write_ftn_to_disc(ftn.select(col(j)).copy(),tmpdir) for j in range(ftn.Ly)}
    max_site = max(ftn.fermion_space.sites)
    nop = (len(info)-1) // 2
    if nop == 1:
        key1,x1,y = info
        tag = f'{key1}{x1},{y}'
    else:
        key1,x1,key2,x2,y = info
        tag = f'{key1}{x1}{key2}{x2},{y}'
    for i in range(nop):
        key,x = info[2*i],info[2*i+1]

        data = data_map[key].copy()
        global_parity = bra_parity * data.parity
        if global_parity != 0:
            data._global_flip()
        data._local_flip([0])

        ket = ftn[ftn.site_tag(x,y),'KET']
        pix,inds,site = ket.inds[-1],ket.inds,ket.get_fermion_info()[1]
        TG = FermionTensor(data=data,tags=ket.tags,
                           inds=(pix,pix+'_'),left_inds=(pix,))
        TG = ftn.fermion_space.move_past(TG,(site+1,max_site+1))
        ket.reindex_({pix:pix+'_'})
        ftn = insert(ftn,site+1,TG)
        ftn.contract_tags(TG.tags,which='all',output_inds=inds,inplace=True)
        #try:
        #    ftn.contract_tags(TG.tags,which='all',output_inds=inds,inplace=True)
        #except ValueError:
        #    print(key,x,y,ket.data.shape,TG.data.shape,type(ket.data))
        #    print(ket.data)
        #    print(TG.data)
    return {(tag,'mid',y):write_ftn_to_disc(ftn.select(col(y)).copy(),tmpdir)}
def compute_left_benvs_1col(info,benvs,tmpdir,Ly,profile,**compress_opts):
    # computes missing left envs: y+1,...,ix for y
    if info[0] is None:
        tag,jmin,jmax = 'norm',2,Ly-1
        ls = [benvs['norm','mid',j] for j in range(Ly)]
    else:
        if len(info) == 5:
            key1,x1,y,jmax,_ = info
            tag = f'{key1}{x1},{y}'
        else:
            key1,x1,key2,x2,y,jmax,_ = info
            tag = f'{key1}{x1}{key2}{x2},{y}'
        if y == Ly-1:
            return dict()
        jmin = y+2 if y==0 else y+1
        if y==0:
            ls = [benvs[tag,'mid',y]]
        elif y==1:
            ls = [benvs['norm','mid',0],benvs[tag,'mid',y]]
        else:
            ls = [benvs['norm','left',y],benvs[tag,'mid',y]]
        ls += [benvs['norm','mid',j] for j in range(y+1,jmax)]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])

    benvs_ = dict()
    col = ftn.col_tag(0)
    for j in range(jmin,jmax+1):
        ftn.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,ftn.Lx-1),**compress_opts) 
        benvs_[tag,'left',j] = write_ftn_to_disc(ftn.select(col).copy(),tmpdir)
    if profile:
        _profile('compute_left_benvs_1col')
    return benvs_
def compute_right_benvs_1col(info,benvs,tmpdir,Ly,profile,**compress_opts):
    # computes missing right envs: ix,...,y-1
    if info[0] is None: # norm
        tag,jmax,jmin = 'norm',Ly-3,0
        ls = [benvs['norm','mid',j] for j in range(Ly)]
    else:
        if len(info) == 5:
            key1,x1,y,_,jmin = info
            tag = f'{key1}{x1},{y}'
        else:
            key1,x1,key2,x2,y,_,jmin = info
            tag = f'{key1}{x1}{key2}{x2},{y}'
        if y == 0:
            return dict()
        jmax = y-2 if y==Ly-1 else y-1
        ls = [benvs['norm','mid',j] for j in range(jmin+1,y)]
        if y==Ly-1:
            ls += [benvs[tag,'mid',y]]
        elif y==Ly-2:
            ls += [benvs[tag,'mid',y],benvs['norm','mid',Ly-1]] 
        else: 
            ls += [benvs[tag,'mid',y],benvs['norm','right',y]]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])

    benvs_ = dict()
    col = ftn.col_tag(Ly-1)
    for j in range(jmax,jmin-1,-1):
        ftn.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,ftn.Lx-1),**compress_opts) 
        benvs_[tag,'right',j] = write_ftn_to_disc(ftn.select(col).copy(),tmpdir)
    if profile:
        _profile('compute_right_benvs_1col')
    return benvs_ 
def compute_benvs_1col(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = compute_left_benvs_1col if info[0]=='left' else \
          compute_right_benvs_1col
    return fxn(info[1:],benvs,tmpdir,Ly,profile,**compress_opts)
def compute_left_benvs_2col(info,benvs,tmpdir,Ly,profile,**compress_opts):
    key1,x1,y1,key2,x2,y2,_ = info
    tag1,tag2 = f'{key1}{x1},{y1}',f'{key2}{x2},{y2}'
    if y2==1:
        ls = [benvs[tag1,'mid',0],benvs[tag2,'mid',y2]]
    else:
        ls = [benvs[tag1,'left',y2],benvs[tag2,'mid',y2]] 
    ls += [benvs['norm','mid',j] for j in range(y2+1,Ly)]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])

    benvs_ = dict()
    col = ftn.col_tag(0)
    tag = tag1 + tag2
    for j in range(y2+1,Ly):
        ftn.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,ftn.Lx-1),**compress_opts)
        benvs_[tag,'left',j] = write_ftn_to_disc(ftn.select(col).copy(),tmpdir)
    if profile:
        _profile('compute_left_benvs_2col')
    return benvs_
def compute_right_benvs_2col(info,benvs,tmpdir,Ly,profile,**compress_opts):
    key1,x1,y1,key2,x2,y2,_ = info
    tag1,tag2 = f'{key1}{x1},{y1}',f'{key2}{x2},{y2}'
    ls = [benvs['norm','mid',j] for j in range(y1)]
    if y1==Ly-2:
        ls += [benvs[tag1,'mid',y1],benvs[tag2,'mid',Ly-1]]
    else:
        ls += [benvs[tag1,'mid',y1],benvs[tag2,'right',y1]]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])

    benvs_ = dict()
    col = ftn.col_tag(Ly-1)
    tag = tag1 + tag2
    for j in range(y1-1,-1,-1):
        ftn.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,ftn.Lx-1),
                                          **compress_opts)
        benvs_[tag,'right',j] = write_ftn_to_disc(ftn.select(col).copy(),tmpdir)
    if profile:
        _profile('compute_right_benvs_2col')
    return benvs_
def compute_benvs_2col(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = compute_left_benvs_2col if info[0]=='left' else \
          compute_right_benvs_2col
    return fxn(info[1:],benvs,tmpdir,Ly,profile,**compress_opts)
def compute_site_term(info,benvs,tmpdir,Ly,profile,**compress_opts):
    norm = False
    if isinstance(info,int):
        j,norm,fac = info,True,1.
        if j<2:
            l = [] if j==0 else [benvs['norm','mid',0]]
        else:
            l = [benvs['norm','left',j]]
        if j>Ly-3:
            r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
        else: 
            r = [benvs['norm','right',j]]
        ls = l + [benvs['norm','mid',j]] + r
    elif len(info)==4: 
        j,tag,y,fac = info
        tag = f'{tag},{y}'
        if j<y:
            if j<2:
                l = [] if j==0 else [benvs['norm','mid',0]]
            else:
                l = [benvs['norm','left',j]]
            r = [benvs[tag,'mid',Ly-1]] if j==Ly-2 else \
                [benvs[tag,'right',j]]
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
            ls = l + [benvs[tag,'mid',j]] + r
        else: # j>y
            l = [benvs[tag,'mid',0]] if j==1 else [benvs[tag,'left',j]]
            if j>Ly-3:
                r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
            else: 
                r = [benvs['norm','right',j]]
            ls = l + [benvs['norm','mid',j]] + r
    else:
        j,key1,x1,y1,key2,x2,y2,fac = info
        tag1,tag2 = f'{key1}{x1},{y1}',f'{key2}{x2},{y2}'
        if j<y1:
            if j<2:
                l = [] if j==0 else [benvs['norm','mid',0]]
            else:
                l = [benvs['norm','left',j]]
            ls = l + [benvs['norm','mid',j],benvs[tag1+tag2,'right',j]]
        elif j==y1:
            if j<2:
                l = [] if j==0 else [benvs['norm','mid',0]]
            else:
                l = [benvs['norm','left',j]]
            r = [benvs[tag2,'mid',Ly-1]] if j==Ly-2 else \
                [benvs[tag2,'right',j]]
            ls = l + [benvs[tag1,'mid',j]] + r
        elif j<y2: # y1<j<y2
            l = benvs[tag1,'mid',0] if j==1 else benvs[tag1,'left',j]
            r = benvs[tag2,'mid',Ly-1] if j==Ly-2 else \
                benvs[tag2,'right',j]
            ls = l,benvs['norm','mid',j],r
        elif j==y2:
            l = [benvs[tag1,'mid',0]] if j==1 else [benvs[tag1,'left',j]]
            if j>Ly-3:
                r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
            else: 
                r = [benvs['norm','right',j]]
            ls = l + [benvs[tag2,'mid',j]] + r
        else: # j>y2
            if j>Ly-3:
                r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
            else: 
                r = [benvs['norm','right',j]]
            ls = [benvs[tag1+tag2,'left',j],benvs['norm','mid',j]] + r
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])
    envs = ftn.compute_row_environments(
               yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)

    site_map = dict()
    for i in range(ftn.Lx):
        ftn_ij = FermionTensorNetwork(
              [envs[side,i] for side in ['bottom','mid','top']]
              ).view_as_(FermionTensorNetwork2D,like=ftn)
        site_tag = ftn_ij.site_tag(i,j)
        ftn_ij.select((site_tag,'BRA'),which='!all').add_tag('grad')
        bra = ftn_ij[site_tag,'BRA']
        ftn_ij.contract_tags('grad',which='any',inplace=True,
                          output_inds=bra.inds[::-1])
        assert ftn_ij.num_tensors==2
        scal = ftn_ij.contract()
        bra_tid = bra.get_fermion_info()[0]
        bra = ftn_ij._pop_tensor(bra_tid,remove_from_fermion_space='end')
        data = ftn_ij['grad'].data
        site_map[i,j] = scal*fac,data*fac
    if profile:
        _profile(f'compute_site_terms')
    return norm,site_map
def compute_grad(H,psi,tmpdir,bra_parity,profile=False,dense_row=True,iprint=0,
                 smallmem=False,**compress_opts):
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm = write_ftn_to_disc(norm,tmpdir)
    H0 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    H1 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    if smallmem:
        H0,H1,N0,N1,benvs = compute_grad_batch(Lx,Ly,norm,bra_parity,tmpdir,
            H.m_1,H.l0_1,H.l1_1,H.r0_1,H.r1_1,H.ls1_1,[],None,H0,H1,True,
            profile=profile,dense_row=dense_row,iprint=iprint,**compress_opts)
        keys = list(benvs.keys())
        for key in keys:
            if key[0] != 'norm':
                fname = benvs.pop(key)
                os.remove(fname)
        H0,H1,_,_,benvs = compute_grad_batch(Lx,Ly,norm,bra_parity,tmpdir,
            H.m_2,H.l0_2,H.l1_2,H.r0_2,H.r1_2,H.ls1_2,H.ls2,benvs,H0,H1,False,
            profile=profile,dense_row=dense_row,iprint=iprint,**compress_opts)
    else:
        H0,H1,N0,N1,benvs = compute_grad_batch(Lx,Ly,norm,bra_parity,tmpdir,
            H.m,H.l0,H.l1,H.r0,H.r1,H.ls1,H.ls2,None,H0,H1,True,
            profile=profile,dense_row=dense_row,iprint=iprint,**compress_opts)
    os.remove(norm)
    for fname in benvs.values():
        os.remove(fname)
    return H0,H1,N0,N1
def compute_grad_batch(Lx,Ly,norm,bra_parity,tmpdir,
                       m,l0,l1,r0,r1,ls1,ls2,norm_benvs,H0,H1,compute_N,
                       profile=False,dense_row=True,iprint=0,**compress_opts):
    benvs = dict() if norm_benvs is None else norm_benvs
    compress_opts['layer_tags'] = 'KET','BRA'

    start_time = time.time()
    fxn = compute_mid_benvs
    iterate_over = [info[:-2] for info in m + l0 + l1 + r0 + r1]
    if norm_benvs is None:
        iterate_over.append(None)
    args = [norm,bra_parity,tmpdir]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    if iprint>0:
        print(f'\t\tcompute_mid_benvs={time.time()-start_time}')

    start_time = time.time()
    fxn = compute_benvs_1col
    iterate_over = [('left',) + info for info in l0 + l1]
    iterate_over += [('right',) + info for info in r0 + r1]
    if norm_benvs is None:
        iterate_over += [('left',None),('right',None)]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    if iprint>0:
        print(f'\t\tcompute_benvs_1col_boundary={time.time()-start_time}')

    start_time = time.time()
    fxn = compute_benvs_1col
    iterate_over  = [('left',) + info for info in m + r1]
    iterate_over += [('right',) + info for info in m + l1]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    if iprint>0:
        print(f'\t\tcompute_benvs_1col={time.time()-start_time}')

    start_time = time.time()
    fxn = compute_benvs_2col
    iterate_over = [(side,) + info for info in ls2 for side in ['left','right']]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    if iprint>0:
        print(f'\t\tcompute_benvs_2col={time.time()-start_time}')

    start_time = time.time()
    compress_opts['dense'] = dense_row
    fxn = compute_site_term
    iterate_over = [(j,) + info for j in range(Ly) for info in ls1 + ls2]
    if compute_N:
        iterate_over += list(range(Ly))
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    if iprint>0:
        print(f'\t\tcompute_site_term={time.time()-start_time}')

    # parse site_map
    start_time = time.time()
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
    if iprint>0:
        print(f'\t\tsum_grad_term={time.time()-start_time}')
    return H0,H1,N0,N1,benvs
def compute_energy_term(info,benvs,**compress_opts):
    norm = False
    if info is None:
        norm,fac = True,1.
        ls = benvs['norm','mid',0],benvs['norm','right',0]
    elif len(info)==3: 
        tag,y,fac = info
        tag = f'{tag},{y}'
        if y==0:
            ls = benvs[tag,'mid',0],benvs['norm','right',0]
        else:
            ls = benvs['norm','mid',0],benvs[tag,'right',0]
    else:
        key1,x1,y1,key2,x2,y2,fac = info
        tag1,tag2 = f'{key1}{x1},{y1}',f'{key2}{x2},{y2}'
        if y1==0:
            ls = benvs[tag1,'mid',0],benvs[tag2,'right',0]
        else:
            ls = benvs['norm','mid',0],benvs[tag1+tag2,'right',0]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    dense = compress_opts.pop('dense')
    if dense:
        for i in range(ftn.Lx-3,-1,-1):
            ftn ^= (ftn.row_tag(i+1),ftn.row_tag(i+2))
    else:
        for i in range(ftn.Lx-3,-1,-1):
            ftn.contract_boundary_from_top_(yrange=(0,1),xrange=(i+1,i+2),
                                            **compress_opts)
    return norm,ftn.contract()*fac
def compute_energy(H,psi,tmpdir,bra_parity=None,dense_row=True,
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
    bra_parity = sum([T.data.parity for T in psi.tensors]) if bra_parity is None \
                 else bra_parity
    benvs = dict()

    fxn = compute_mid_benvs
    iterate_over  = [None] + [info[:-2] for info in H.m + H.l0 + H.l1 + H.r0 + H.r1]
    args = [norm,bra_parity,tmpdir]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm) 

    fxn = compute_right_benvs_1col 
    iterate_over = [(None,)] + H.r0 + H.r1
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = compute_right_benvs_1col 
    iterate_over = H.m + H.l1
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
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
def compute_local_term(info,benvs,bra_parity,Ly,**compress_opts):
    j,keys = info[0],info[1:]
    if j<2:
        l = [] if j==0 else [benvs['norm','mid',0]]
    else:
        l = [benvs['norm','left',j]]
    if j>Ly-3:
        r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
    else: 
        r = [benvs['norm','right',j]]
    ls = l + [benvs['norm','mid',j]] + r
    ls = [load_ftn_from_disc(fname) for fname in ls]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=ls[0])
    envs = ftn.compute_row_environments(yrange=(max(j-1,0),min(j+1,Ly-1)),**compress_opts)

    site_map = dict()
    for key in keys:
        data = data_map[key].copy()
        global_parity = bra_parity * data.parity
        if global_parity != 0:
            data._global_flip()
        data._local_flip([0])
        for i in range(ftn.Lx):
            ftn_ij = FermionTensorNetwork(
                  [envs[side,i] for side in ['bottom','mid','top']]
                  ).view_as_(FermionTensorNetwork2D,like=ftn)
            norm_ij = ftn_ij.contract()

            ket = ftn_ij[ftn_ij.site_tag(i,j),'KET']
            pix = ket.inds[-1]
            ket.reindex_({pix:pix+'_'})
            #ftn_ij.add_tensor(TG,virtual=True)
            #expec_ij = ftn_ij.contract()
            bk_ij = ftn_ij.contract(output_inds=(pix,pix+'_')) 
            try:
                expec_ij = np.tensordot(bk_ij.data,data,axes=((0,1),(0,1)))
            except IndexError:
                expec_ij = 0.
            site_map[key,i,j] = expec_ij,norm_ij
    return site_map
def compute_local_expectations(psi,tmpdir,keys,bra_parity=None,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    benvs = {('norm','mid',j):write_ftn_to_disc(norm.select(norm.col_tag(j)).copy(),tmpdir) for j in range(norm.Ly)}
    bra_parity = sum([T.data.parity for T in psi.tensors]) if bra_parity is None else bra_parity

    fxn = compute_benvs_1col 
    iterate_over = [('left',None),('right',None)]
    args = [benvs,tmpdir,norm.Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = compute_local_term
    iterate_over = [(j,)+keys for j in range(psi.Ly)]
    args = [benvs,bra_parity,psi.Ly]
    compress_opts['dense'] = dense_row
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    site_maps = dict()
    for site_map in ls:
        site_maps.update(site_map)
    for fname in benvs.values():
        os.remove(fname) 
    return site_maps
def check_particle_number(psi,tmpdir,spinless=False,bra_parity=None,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    keys = ('pna',)
    if not spinless:
        keys = keys + ('pnb',)
    site_map = compute_local_expectations(psi,tmpdir,keys,bra_parity=bra_parity,
        dense_row=dense_row,layer_tags=layer_tags,max_bond=max_bond,cutoff=cutoff,
        canonize=canonize,mode=mode)
    arr_map = {key:np.zeros((psi.Lx,psi.Ly)) for key in keys}
    for (key,i,j),(e,n) in site_map.items():
        arr_map[key][i,j] = e/n
    print('total alpha particle number=',sum(arr_map['pna'].reshape(-1)))
    print(arr_map['pna'])
    if not spinless: 
        print('total beta particle number=',sum(arr_map['pnb'].reshape(-1)))
        print(arr_map['pnb'])
        pn_ = arr_map['pna']+arr_map['pnb']
        print('total particle number=',sum(pn_.reshape(-1)))
        print(pn_) 
    return arr_map 
def compute_norm(psi,layer_tags=('KET','BRA'),dense_row=True,
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    for j in range(norm.Ly-3,-1,-1):
        norm.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,norm.Lx-1),
                                           **compress_opts)
    norm.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    if dense_row:
        for i in range(norm.Lx-3,-1,-1):
            norm ^= (norm.row_tag(i+1),norm.row_tag(i+2))
    else:
        for i in range(norm.Lx-3,-1,-1):
            norm.contract_boundary_from_top_(yrange=(0,1),xrange=(i+1,i+2),
                                            **compress_opts)
    return norm.contract() 
class GlobalGrad():
    def __init__(self,H,peps,chi,psi_fname,tmpdir,save_every=True,
                 profile=False,dense_row=True,iprint=0,smallmem=False):
        self.H = H
        self.D = peps[0,0].shape[0]
        self.chi = chi
        self.profile = profile
        self.dense_row = dense_row
        self.tmpdir = tmpdir
        self.parity = sum([T.data.parity for T in peps.tensors])
        self.iprint = iprint
        self.save_every = save_every
        self.smallmem = smallmem
        print(f'D={self.D},chi={self.chi}')
        if smallmem:
            self.H.distribute()    

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
        self.start_time = time.time()
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
                data = cons.vector_to_tensor(x[start:stop],dq)
                #print(i,j,start,stop,x[start:stop])
                #data.shape = psi[i,j].data.shape
                psi[i,j].modify(data=data)
                start = stop
        return psi
    def compute_energy(self,x):
        psi = self.vec2fpeps(x)
        E,n = compute_energy(self.H,psi,self.tmpdir,bra_parity=self.parity,
                             dense_row=self.dense_row,max_bond=self.chi)
        print(f'\tne={self.ne},time={time.time()-self.start_time}')
        print(f'\t\tE={E},norm={n}')
        self.ne += 1
        self.fac = n**(-1.0/(2.0*psi.Lx*psi.Ly))
        print('nfile_energy=',len(os.listdir(self.tmpdir)))
        return E,n
    def compute_grad(self,x):
        psi = self.vec2fpeps(x)
        H0,H1,n0,n1 = compute_grad(self.H,psi,self.tmpdir,self.parity,
                max_bond=self.chi,dense_row=self.dense_row,
                profile=self.profile,iprint=self.iprint,smallmem=self.smallmem)
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

        if self.save_every:
            write_ftn_to_disc(psi,self.psi+f'_{self.niter}',provided_filename=True)
        write_ftn_to_disc(psi,self.psi,provided_filename=True)
        self.niter += 1
        return x,g
    def kernel(self,method=_minimize_bfgs,maxiter=200,gtol=1e-5):
        self.ng = 0
        self.ne = 0
        self.niter = 0
        options = {'maxiter':maxiter,'gtol':gtol}
        x0 = self.fpeps2vec(load_ftn_from_disc(self.psi))
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
