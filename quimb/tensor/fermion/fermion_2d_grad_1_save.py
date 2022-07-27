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
    worker_execution,
    insert,
    load_ftn_from_disc,
    write_ftn_to_disc,
    _profile,
)
np.set_printoptions(suppress=True,linewidth=1000,precision=4)
symmetry = 'u1'
flat = True
data_map = {'cre_a':creation(spin='a',symmetry=symmetry,flat=flat),
            'cre_b':creation(spin='b',symmetry=symmetry,flat=flat),
            'ann_a':annihilation(spin='a',symmetry=symmetry,flat=flat),
            'ann_b':annihilation(spin='b',symmetry=symmetry,flat=flat),
            'pn':ParticleNumber(symmetry=symmetry,flat=flat),
            'nanb':onsite_U(u=1.,symmetry=symmetry,flat=flat)}
sign_a = (-1)**(data_map['cre_a'].parity*data_map['ann_a'].parity)
sign_b = (-1)**(data_map['cre_b'].parity*data_map['ann_b'].parity)
##################################################################################
# some sum of op class for gradient 
##################################################################################
class OPi:
    def __init__(self,coeff_map,site):
        self.coeff_map = coeff_map
        self.site = site
        self.tag = '_'.join([key+'{},{}'.format(*site) for key in coeff_map])
    def get_data(self):
        data = 0.0
        for key,fac in self.coeff_map.items():
            data = data + fac*data_map[key]
        return data
class SumOpGrad:
    def __init__(self,_1col_terms,_2col_terms,reuse):
        self._1col_terms = _1col_terms
        self._2col_terms = _2col_terms
        self.reuse = reuse
        print('number of 1 col terms=',len(self._1col_terms))
        print('number of 2 col terms=',len(self._2col_terms))
##################################################################
# Hamiltonians
##################################################################
def hubbard(t,u,Lx,Ly):
    #cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    #cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    #ann_a = cre_a.dagger
    #ann_b = cre_b.dagger
    #pn    = ParticleNumber(symmetry=symmetry,flat=flat)
    #nanb  = onsite_U(u=1.0,symmetry=symmetry)
    _1col_terms = []
    _2col_terms = []
    for i in range(Lx):
        for j in range(Ly):
            opis = OPi({'nanb':1.},(i,j)),
            _1col_terms.append((opis,u)) 
            if i+1 != Lx:
                opis = OPi({'cre_a':1.},(i,j)),OPi({'ann_a':1.},(i+1,j))
                _1col_terms.append((opis,-t*sign_a)) 
                opis = OPi({'ann_a':1.},(i,j)),OPi({'cre_a':1.},(i+1,j))
                _1col_terms.append((opis,-t)) 
                opis = OPi({'cre_b':1.},(i,j)),OPi({'ann_b':1.},(i+1,j))
                _1col_terms.append((opis,-t*sign_b)) 
                opis = OPi({'ann_b':1.},(i,j)),OPi({'cre_b':1.},(i+1,j))
                _1col_terms.append((opis,-t)) 
            if j+1 != Ly:
                opis = OPi({'cre_a':1.},(i,j)),OPi({'ann_a':1.},(i,j+1))
                _2col_terms.append((opis,-t*sign_a))             
                opis = OPi({'ann_a':1.},(i,j)),OPi({'cre_a':1.},(i,j+1))
                _2col_terms.append((opis,-t))                    
                opis = OPi({'cre_b':1.},(i,j)),OPi({'ann_b':1.},(i,j+1))
                _2col_terms.append((opis,-t*sign_b))             
                opis = OPi({'ann_b':1.},(i,j)),OPi({'cre_b':1.},(i,j+1))
                _2col_terms.append((opis,-t)) 
    reuse = [(OPi({key:1.},(i,j)),min(Ly-1,j+1),max(0,j-1)) \
             for key in ['cre_a','ann_a','cre_b','ann_b'] \
             for i in range(Lx) for j in range(Ly)] 
    return SumOpGrad(_1col_terms,_2col_terms,reuse)
def fd_grad(N,L,Ne,order=2,has_coulomb=False,soft=True,maxdist=1000):
    from .ueg_utils import _back,_ke2,_ke1,_u,_dist 
    eps = L/(N+1.)
    print('eps=',eps)
    sites = [(x,y) for x in range(N) for y in range(N)]
    norb = len(sites)
    if has_coulomb:
        rhob = Ne/(N**2*eps**2)
        Eb = _back(sites,eps,rhob,soft=soft) 
        print(f'rhob={rhob},Eb={Eb}')

    # operators
    #cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    #cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    #ann_a = cre_a.dagger
    #ann_b = cre_b.dagger
    #pn    = ParticleNumber(symmetry=symmetry,flat=flat)
    #nanb  = onsite_U(u=1.0,symmetry=symmetry)
    #sign_a = (-1)**(cre_a.parity*ann_a.parity)
    #sign_b = (-1)**(cre_b.parity*ann_b.parity)
    #data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
    #            'pn':pn,'nanb':nanb}

    _1col_terms = []
    _2col_terms = []
    sites = [(x,y) for y in range(N) for x in range(N)]
    for (x,y) in sites:
        ke2 = _ke2(eps,order=order) 
        if x+1<N:
           site1,site2 = (x,y),(x+1,y)
           opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
           _1col_terms.append((opis,ke2*sign_a)) 
           opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
           _1col_terms.append((opis,ke2)) 
           opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
           _1col_terms.append((opis,ke2*sign_b)) 
           opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
           _1col_terms.append((opis,ke2))
        if y+1<N:
           site1,site2 = (x,y),(x,y+1)
           opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
           _2col_terms.append((opis,ke2*sign_a)) 
           opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
           _2col_terms.append((opis,ke2)) 
           opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
           _2col_terms.append((opis,ke2*sign_b)) 
           opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
           _2col_terms.append((opis,ke2))
        if order==4 and maxdist>=2:
            ke2 = 1./(24.*eps**2)
            if x+2<N:
               site1,site2 = (x,y),(x+2,y)
               opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
               _1col_terms.append((opis,ke2*sign_a)) 
               opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
               _1col_terms.append((opis,ke2)) 
               opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
               _1col_terms.append((opis,ke2*sign_b)) 
               opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
               _1col_terms.append((opis,ke2))
            if y+2<N:
               site1,site2 = (x,y),(x,y+2)
               opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
               _2col_terms.append((opis,ke2*sign_a)) 
               opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
               _2col_terms.append((opis,ke2)) 
               opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
               _2col_terms.append((opis,ke2*sign_b)) 
               opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
               _2col_terms.append((opis,ke2))
    for site in sites:
        ke1 = _ke1(site,eps,N,order=order)
        if has_coulomb: 
            ke1 += _u(site,sites,eps,rhob,soft=soft)
            ee1 = 1./eps 
            opis = OPi({'pn':ke1,'nanb':ee1},site),
        else: 
            opis = OPi({'pn':ke1},site),
        _1col_terms.append((opis,1.))
    if has_coulomb:
        const = 1. if soft else 0.
        for i,site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                if abs(site1[0]-site2[0])+abs(site1[1]-site2[1])<=maxdist:
                    ee2 = 1./_dist(site1,site2,eps,const=const)
                    opis = OPi({'pn':1.},site1),OPi({'pn':1.},site2)
                    if site1[1]==site2[1]:
                        _1col_terms.append((opis,ee2))
                    else:
                        _2col_terms.append((opis,ee2))
    dy = 1 if order==2 else 2
    reuse = [(OPi({key:1.},(x,y)),min(N-1,y+dy),max(0,y-dy)) \
             for key in ['cre_a','ann_a','cre_b','ann_b'] for (x,y) in sites] 
    if has_coulomb:
        reuse += [(OPi({'pn':1.},(x,y)),min(N-1,y+maxdist),max(0,y-maxdist)) \
                  for (x,y) in sites] 
    return SumOpGrad(_1col_terms,_2col_terms,reuse)
#############################################################
# gradient functions
#############################################################
def _norm_left(norm,tmpdir,profile,**compress_opts):
    norm = load_ftn_from_disc(norm)
    first_col = norm.col_tag(0)
    benvs_ = dict()
    for j in range(2,norm.Ly):
        norm.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,norm.Lx-1),
                                          **compress_opts)
        benvs_['norm','left',j] = write_ftn_to_disc(norm.select(first_col).copy(),
                                                    tmpdir)
    if profile:
        _profile(f'_norm_left')
    return benvs_ 
def _norm_right(norm,tmpdir,profile,**compress_opts):
    norm = load_ftn_from_disc(norm)
    last_col = norm.col_tag(norm.Ly-1)
    benvs_ = dict()
    for j in range(norm.Ly-3,-1,-1):
        norm.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,norm.Lx-1),
                                           **compress_opts)
        benvs_['norm','right',j] = write_ftn_to_disc(norm.select(last_col).copy(),
                                                    tmpdir)
    if profile:
        _profile(f'_norm_right')
    return benvs_
def _norm_mid(norm,tmpdir,profile):
    norm = load_ftn_from_disc(norm)
    benvs_ = dict()
    for j in range(norm.Ly):
        benvs_['norm','mid',j] = write_ftn_to_disc(
                                     norm.select(norm.col_tag(j)).copy(),tmpdir)
    if profile:
        _profile(f'_norm_mid')
    return benvs_
def _norm_benvs(side,norm,tmpdir,profile,**compress_opts):
    if side=='left':
        return _norm_left(norm,tmpdir,profile,**compress_opts)
    elif side=='right':
        return _norm_right(norm,tmpdir,profile,**compress_opts)
    else:
        return _norm_mid(norm,tmpdir,profile)
def _1col_mid(opis,psi,tmpdir,profile):
    psi = load_ftn_from_disc(psi)
    ftn,_,bra = psi.make_norm(return_all=True,layer_tags=('KET','BRA')) 

    tsrs = []
    for opi in opis:
        ket = ftn[ftn.site_tag(*opi.site),'KET']
        pix = ket.inds[-1] 
        TG = FermionTensor(data=opi.get_data(),tags=ket.tags,
                           inds=(pix,pix+'_'),left_inds=(pix,))
        tsrs.append(bra.fermion_space.move_past(TG))

    ftn.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    for TG,opi in zip(tsrs,opis):
        site_tag = ftn.site_tag(*opi.site)
        _,bra_site = ftn[site_tag,'BRA'].get_fermion_info()
        site_range = bra_site,max(ftn.fermion_space.sites)+1
        TG = ftn.fermion_space.move_past(TG,site_range)

        inds = ftn[site_tag,'KET'].inds
        pix = inds[-1] 
        ftn[site_tag,'KET'].reindex_({pix:pix+'_'})
        _,ket_site = ftn[site_tag,'KET'].get_fermion_info()
        ftn = insert(ftn,ket_site+1,TG)
        ftn.contract_tags(TG.tags,which='all',output_inds=inds,inplace=True)

    y = opis[0].site[1]
    term = tuple([opi.tag for opi in opis])
    term = term[0] if len(term)==1 else term
    ftn = write_ftn_to_disc(ftn.select(ftn.col_tag(y)).copy(),tmpdir)
    if profile:
        _profile(f'_1col_mid')
    return {(term,'mid',y):ftn}
def _1col_left(info,benvs,tmpdir,Ly,profile,**compress_opts):
    # computes missing left envs: y,...,ix
    opis,ix = info
    y = opis[0].site[1]
    benvs_ = dict()
    if y<Ly-1:
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        ls = []
        if y>0:
            ls += [benvs['norm','mid',0]] if y==1 else [benvs['norm','left',y]]
        ls += [benvs[term,'mid',y]]
        ls += [benvs['norm','mid',j] for j in range(y+1,ix)]
        j0 = y+2 if y==0 else y+1

        ls = [load_ftn_from_disc(fname) for fname in ls]
        like = ls[0] if ls[0].num_tensors>0 else ls[1]
        ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

        first_col = ftn.col_tag(0)
        for j in range(j0,ix+1):
            ftn.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,ftn.Lx-1),
                                             **compress_opts) 
            benvs_[term,'left',j] = write_ftn_to_disc(ftn.select(first_col).copy(),
                                                      tmpdir)
        if profile:
            _profile(f'_1col_left')
    return benvs_
def _1col_right(info,benvs,tmpdir,Ly,profile,**compress_opts):
    # computes missing right envs: ix,...,y-1
    opis,ix = info
    y = opis[0].site[1]
    benvs_ = dict()
    if y>0:
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        ls = [benvs['norm','mid',j] for j in range(ix+1,y)]
        ls += [benvs[term,'mid',y]]
        if y<Ly-1:
            ls += [benvs['norm','mid',Ly-1]] if y==Ly-2 else [benvs['norm','right',y]]
        j0 = y-2 if y==Ly-1 else y-1

        ls = [load_ftn_from_disc(fname) for fname in ls]
        like = ls[0] if ls[0].num_tensors>0 else ls[1]
        ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

        last_col = ftn.col_tag(Ly-1)
        for j in range(j0,ix-1,-1):
            ftn.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,ftn.Lx-1),
                                              **compress_opts) 
            benvs_[term,'right',j] = write_ftn_to_disc(ftn.select(last_col).copy(),
                                                       tmpdir)
        if profile:
            _profile(f'_1col_right')
    return benvs_ 
def _1col_benvs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = _1col_left if info[0]=='left' else _1col_right
    return fxn(info[1:],benvs,tmpdir,Ly,profile,**compress_opts)
def _2col_left(opis,benvs,tmpdir,Ly,profile,**compress_opts):
    (x1,y1),(x2,y2) = [opi.site for opi in opis]
    tag1,tag2 = [opi.tag for opi in opis]
    if y2==1:
        ls = [benvs[tag1,'mid',0],benvs[tag2,'mid',y2]]
    else:
        ls = [benvs[tag1,'left',y2],benvs[tag2,'mid',y2]] 
    ls += [benvs['norm','mid',j] for j in range(y2+1,Ly)]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

    benvs_ = dict()
    first_col = ftn.col_tag(0)
    term = tag1,tag2
    for j in range(y2+1,Ly):
        ftn.contract_boundary_from_left_(yrange=(j-2,j-1),xrange=(0,ftn.Lx-1),
                                         **compress_opts)
        benvs_[term,'left',j] = write_ftn_to_disc(ftn.select(first_col).copy(),tmpdir)
    if profile:
        _profile(f'_2col_left')
    return benvs_
def _2col_right(opis,benvs,tmpdir,Ly,profile,**compress_opts):
    tag1,tag2 = [opi.tag for opi in opis]
    (x1,y1),(x2,y2) = [opi.site for opi in opis]
    ls = [benvs['norm','mid',j] for j in range(y1)]
    if y1==Ly-2:
        ls += [benvs[tag1,'mid',y1],benvs[tag2,'mid',Ly-1]]
    else:
        ls += [benvs[tag1,'mid',y1],benvs[tag2,'right',y1]]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)

    benvs_ = dict()
    last_col = ftn.col_tag(Ly-1)
    term = tag1,tag2
    for j in range(y1-1,-1,-1):
        ftn.contract_boundary_from_right_(yrange=(j+1,j+2),xrange=(0,ftn.Lx-1),
                                          **compress_opts)
        benvs_[term,'right',j] = write_ftn_to_disc(ftn.select(last_col).copy(),tmpdir)
    if profile:
        _profile(f'_2col_right')
    return benvs_
def _2col_benvs(info,benvs,tmpdir,Ly,profile,**compress_opts):
    fxn = _2col_left if info[0]=='left' else _2col_right
    return fxn(info[1],benvs,tmpdir,Ly,profile,**compress_opts)
def _site_term(info,benvs,tmpdir,Ly,profile,**compress_opts):
    typ,opis,j,fac = info
    if opis is None: # norm term
        term = 'norm'
        if j<2:
            l = [] if j==0 else [benvs['norm','mid',0]]
        else:
            l = [benvs['norm','left',j]]
        if j>Ly-3:
            r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
        else: 
            r = [benvs['norm','right',j]]
        ls = l + [benvs['norm','mid',j]] + r
    else: # H terms
        sites = [opi.site for opi in opis]
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        if (len(sites)==2) and (sites[0][1]<sites[1][1]): # 2col terms 
            tag1,tag2 = term
            (x1,y1),(x2,y2) = sites
            if j<y1:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                ls = l + [benvs['norm','mid',j],benvs[term,'right',j]]
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
                ls = [benvs[term,'left',j],benvs['norm','mid',j]] + r
        else: # 1col terms
            y = sites[0][1]
            if j<y:
                if j<2:
                    l = [] if j==0 else [benvs['norm','mid',0]]
                else:
                    l = [benvs['norm','left',j]]
                r = [benvs[term,'mid',Ly-1]] if j==Ly-2 else \
                    [benvs[term,'right',j]]
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
                ls = l + [benvs[term,'mid',j]] + r
            else: # j>y
                l = [benvs[term,'mid',0]] if j==1 else [benvs[term,'left',j]]
                if j>Ly-3:
                    r = [] if j==Ly-1 else [benvs['norm','mid',Ly-1]]
                else: 
                    r = [benvs['norm','right',j]]
                ls = l + [benvs['norm','mid',j]] + r
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
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
        _profile(f'_row_envs')
    return typ,site_map
def compute_grad(H,psi,tmpdir,profile=False,dense_row=True,
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
    psi = write_ftn_to_disc(psi,tmpdir) 

    start_time = time.time()
    fxn = _norm_benvs
    iterate_over = ['left','right','mid']
    args = [norm,tmpdir,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm)
    print(f'\t\tnorm_benvs_time={time.time()-start_time}')


    start_time = time.time()
    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    iterate_over += [(opi,) for (opi,_,_) in H.reuse]
    args = [psi,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(psi) 

    fxn = _1col_benvs
    iterate_over  = [('left', opis,Ly-1) for (opis,_) in H._1col_terms]
    iterate_over += [('right',opis,0)    for (opis,_) in H._1col_terms]
    iterate_over += [('left',(opi,),lix)  for (opi,lix,_) in H.reuse]
    iterate_over += [('right',(opi,),rix) for (opi,_,rix) in H.reuse]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    print(f'\t\t1col_benvs_time={time.time()-start_time}')

    start_time = time.time()
    fxn = _2col_benvs
    iterate_over = [(side,opis) for (opis,_) in H._2col_terms \
                                for side in ['left','right']]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    print(f'\t\t2col_benvs_time={time.time()-start_time}')

    fxn = _site_term
    iterate_over  = [('norm',None,j,1.) for j in range(Ly)]
    iterate_over += [('H',opis,j,fac) for j in range(Ly) \
                                      for (opis,fac) in H._1col_terms]
    iterate_over += [('H',opis,j,fac) for j in range(Ly) \
                                      for (opis,fac) in H._2col_terms]
    args = [benvs,tmpdir,Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)

    # parse site_map
    H0 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    H1 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    n0 = dict() 
    n1 = dict() 
    for typ,site_map in ls:
        for (i,j),(scal,data) in site_map.items():
            if typ == 'norm':
                n0[i,j] = scal
                n1[i,j] = data
            else:
                H0[i,j] = H0[i,j] + scal
                H1[i,j] = H1[i,j] + data
    for fname in benvs.values():
        os.remove(fname)
    return H0,H1,n0,n1
def compute_grad_small_mem(H,psi,tmpdir,profile=False,dense_row=True,
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
    psi = write_ftn_to_disc(psi,tmpdir) 

    fxn = _norm_benvs
    iterate_over = ['left','right','mid']
    args = [norm,tmpdir,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm)

    H0 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    H1 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    n0 = dict() 
    n1 = dict() 

    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    args = [psi,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _1col_benvs
    iterate_over  = [('left', opis,Ly-1) for (opis,_) in H._1col_terms]
    iterate_over += [('right',opis,0)    for (opis,_) in H._1col_terms]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    print('1col benvs=',len(os.listdir(tmpdir)))

    fxn = _site_term
    iterate_over  = [('norm',None,j,1.) for j in range(Ly)]
    iterate_over += [('H',opis,j,fac) for j in range(Ly) \
                                      for (opis,fac) in H._1col_terms]
    args = [benvs,tmpdir,Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for typ,site_map in ls:
        for (i,j),(scal,data) in site_map.items():
            if typ == 'norm':
                n0[i,j] = scal
                n1[i,j] = data
            else:
                H0[i,j] = H0[i,j] + scal
                H1[i,j] = H1[i,j] + data
    for key in list(benvs.keys()):
        if key[0]!='norm':  
            fname = benvs.pop(key)
            os.remove(fname)
    compress_opts.pop('dense')
    print('1col benvs=',len(os.listdir(tmpdir)))

    fxn = _1col_mid
    iterate_over = [(opi,) for (opi,_,_) in H.reuse]
    args = [psi,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(psi) 

    fxn = _1col_benvs
    iterate_over  = [('left',(opi,),lix)  for (opi,lix,_) in H.reuse]
    iterate_over += [('right',(opi,),rix) for (opi,_,rix) in H.reuse]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _2col_benvs
    iterate_over = [(side,opis) for (opis,_) in H._2col_terms \
                                for side in ['left','right']]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    print('2col benvs=',len(os.listdir(tmpdir)))

    fxn = _site_term
    iterate_over = [('H',opis,j,fac) for j in range(Ly) \
                                      for (opis,fac) in H._2col_terms]
    args = [benvs,tmpdir,Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for typ,site_map in ls:
        for (i,j),(scal,data) in site_map.items():
            H0[i,j] = H0[i,j] + scal
            H1[i,j] = H1[i,j] + data
    for fname in benvs.values():
        os.remove(fname)
    return H0,H1,n0,n1
def compute_grad_N(H,psi,tmpdir,profile=False,dense_row=True,
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
    psi = write_ftn_to_disc(psi,tmpdir) 

    fxn = _norm_benvs
    iterate_over = ['left','right','mid']
    args = [norm,tmpdir,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm)

    N_terms = [OPi({'pn':1.},(x,y)) for x in range(Lx) for y in range(Ly)]

    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    iterate_over += [(opi,) for opi in N_terms]
    for (opi,_,_) in H.reuse:
        if opi.tag[:2]!='pn':
            iterate_over.append((opi,))
    args = [psi,tmpdir,profile]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _1col_benvs
    iterate_over  = [('left', opis,Ly-1) for (opis,_) in H._1col_terms]
    iterate_over += [('right',opis,0)    for (opis,_) in H._1col_terms]
    iterate_over += [('left',(opi,),Ly-1) for opi in N_terms]
    iterate_over += [('right',(opi,),0)   for opi in N_terms]
    for (opi,lix,rix) in H.reuse:
        if opi.tag[:2]!='pn':
            iterate_over += [('left',(opi,),lix),('right',(opi,),rix)]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _2col_benvs
    iterate_over = [(side,opis) for (opis,_) in H._2col_terms \
                                for side in ['left','right']]
    args = [benvs,tmpdir,Ly,profile]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _site_term
    iterate_over  = [('norm',None,j,1.) for j in range(Ly)]
    iterate_over += [('H',opis,j,fac) for j in range(Ly) \
                                      for (opis,fac) in H._1col_terms]
    iterate_over += [('H',opis,j,fac) for j in range(Ly) \
                                      for (opis,fac) in H._2col_terms]
    iterate_over += [('N',(opi,),j,1.) for j in range(Ly) for opi in mu_terms]
    args = [benvs,tmpdir,Ly,profile]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)

    # parse site_map
    H0 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    H1 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    n0 = dict() 
    n1 = dict() 
    N0 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    N1 = {(i,j):0.0 for i in range(Lx) for j in range(Ly)}
    for typ,site_map in ls:
        for (i,j),(scal,data) in site_map.items():
            if typ == 'norm':
                n0[i,j] = scal
                n1[i,j] = data
            elif typ == 'H':
                H0[i,j] = H0[i,j] + scal
                H1[i,j] = H1[i,j] + data
            else:
                N0[i,j] = N0[i,j] + scal
                N1[i,j] = N1[i,j] + data
    os.remove(psi) 
    for fname in benvs.values():
        os.remove(fname)
    return H0,H1,N0,N1,n0,n1

def _energy_term(info,benvs,**compress_opts):
    opis,fac = info
    j = 0 
    if opis is None: # norm term
        term = 'norm'
        ls = [benvs['norm',side,j] for side in ['mid','right']]
    else: # H terms
        term = tuple([opi.tag for opi in opis])
        term = term[0] if len(term)==1 else term
        sites = [opi.site for opi in opis]
        if (len(sites)==2) and (sites[0][1]<sites[1][1]): # 2col terms 
            tag1,tag2 = term 
            (x1,y1),(x2,y2) = sites
            if j<y1:
                ls = benvs['norm','mid',j],benvs[term,'right',j]
            elif j==y1:
                ls = benvs[tag1,'mid',j],benvs[tag2,'right',j]
            elif j<y2:
                ls = benvs['norm','mid',j],benvs[tag2,'right',j]
            elif j==y2:
                ls = benvs[tag2,'mid',j],benvs['norm','right',j]
            else: # j>y2
                ls = benvs['norm','mid',j],benvs['norm','right',j]
        else: # 1col terms
            y = sites[0][1]
            if j<y:
                ls = benvs['norm','mid',j],benvs[term,'right',j]
            elif j==y:
                ls = benvs[term,'mid',j],benvs['norm','right',j]
            else: # j>y
                ls = benvs['norm','mid',j],benvs['norm','right',j]
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    dense = compress_opts.pop('dense')
    if dense:
        for i in range(ftn.Lx-3,-1,-1):
            ftn ^= (ftn.row_tag(i+1),ftn.row_tag(i+2))
    else:
        for i in range(ftn.Lx-3,-1,-1):
            ftn.contract_boundary_from_top_(yrange=(0,1),xrange=(i+1,i+2),
                                            **compress_opts)
    return term,ftn.contract()*fac
def compute_energy(H,psi,tmpdir,dense_row=True,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    Lx,Ly = psi.Lx,psi.Ly

    fxn = _norm_benvs
    iterate_over = ['right','mid']
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm = write_ftn_to_disc(norm,tmpdir)
    args = [norm,tmpdir,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm)

    fxn = _1col_mid
    iterate_over  = [opis for (opis,_) in H._1col_terms]
    iterate_over += [(opi,) for (opi,_,_) in H.reuse]
    psi = write_ftn_to_disc(psi,tmpdir) 
    args = [psi,tmpdir,False]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(psi) 

    fxn = _1col_right
    iterate_over  = [(opis,0)   for (opis,_) in H._1col_terms]
    iterate_over += [((opi,),rix) for (opi,_,rix) in H.reuse]
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _2col_right
    iterate_over = [opis for (opis,_) in H._2col_terms]
    args = [benvs,tmpdir,Ly,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for benvs_ in ls:
        benvs.update(benvs_)

    fxn = _energy_term
    iterate_over = H._1col_terms + H._2col_terms + [(None,1.)]
    args = [benvs]
    compress_opts['dense'] = dense_row
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    H0 = 0.
    for (term,scal) in ls:
        if term == 'norm':
            n = scal
        else:
            H0 += scal

    for fname in benvs.values():
        os.remove(fname) 
    print('nfile_energy=',len(os.listdir(tmpdir)))
    return H0/n,n
def _pn_site(j,benvs,bra,ops,**compress_opts):
    bra = load_ftn_from_disc(bra)
    term = 'norm'
    if j<2:
        l = [] if j==0 else [benvs['norm','mid',0]]
    else:
        l = [benvs['norm','left',j]]
    if j>bra.Ly-3:
        r = [] if j==bra.Ly-1 else [benvs['norm','mid',bra.Ly-1]]
    else: 
        r = [benvs['norm','right',j]]
    ls = l + [benvs['norm','mid',j]] + r
    ls = [load_ftn_from_disc(fname) for fname in ls]
    like = ls[0] if ls[0].num_tensors>0 else ls[1]
    ftn = FermionTensorNetwork(ls).view_as_(FermionTensorNetwork2D,like=like)
    envs = ftn.compute_row_environments(
               yrange=(max(j-1,0),min(j+1,ftn.Ly-1)),**compress_opts)

    arr_map = dict()
    for opix,op in enumerate(ops):
        for i in range(ftn.Lx):
            pix = bra.site_ind(i,j)
            TG = FermionTensor(op.copy(),inds=(pix,pix+'_'),left_inds=(pix,))
            TG = bra.fermion_space.move_past(TG)

            ftn_ij = FermionTensorNetwork(
                  [envs[side,i] for side in ['bottom','mid','top']]
                  ).view_as_(FermionTensorNetwork2D,like=ftn)
            norm_ij = ftn_ij.contract()

            ket = ftn_ij[ftn_ij.site_tag(i,j),'KET']
            ket.reindex_({pix:pix+'_'})
            #ftn_ij.add_tensor(TG,virtual=True)
            #expec_ij = ftn_ij.contract()
            bk_ij = ftn_ij.contract(output_inds=(pix,pix+'_')) 
            try:
                expec_ij = np.tensordot(bk_ij.data,TG.data,axes=((0,1),(0,1)))
            except IndexError:
                expec_ij = 0.
            arr_map[opix,i,j] = expec_ij,norm_ij
    return arr_map
def check_pn(psi,tmpdir,symmetry='u1',layer_tags=('KET','BRA'),dense_row=True,
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags

    fxn = _norm_benvs
    iterate_over = ['left','right','mid']
    norm,_,bra = psi.make_norm(layer_tags=('KET','BRA'),return_all=True)
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    norm = write_ftn_to_disc(norm,tmpdir)
    args = [norm,tmpdir,False]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    benvs = dict()
    for benvs_ in ls:
        benvs.update(benvs_)
    os.remove(norm)

    fxn = _pn_site
    iterate_over = list(range(psi.Ly))
    bra = write_ftn_to_disc(bra,tmpdir)
    pn = ParticleNumber(symmetry=symmetry,flat=True)
    cre_a = creation(spin='a',symmetry=symmetry,flat=True)
    cre_b = creation(spin='b',symmetry=symmetry,flat=True)
    ann_a = cre_a.dagger
    ann_b = cre_b.dagger
    pna = np.tensordot(cre_a,ann_a,axes=((-1,),(0,)))
    pnb = np.tensordot(cre_b,ann_b,axes=((-1,),(0,)))
    ops = pn,pna,pnb 
    args = [benvs,bra,ops]
    compress_opts['dense'] = dense_row
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    arrs = [np.zeros((psi.Lx,psi.Ly)) for op in ops]
    sums = [0.0 for op in ops]
    for arr_map in ls:
        for (opix,i,j),(e,n) in arr_map.items():
            arrs[opix][i,j] = e/n
            sums[opix] += e/n
    print('total particle number=',sums[0])
    print(arrs[0]) 
    print('total alpha particle number=',sums[1])
    print(arrs[1]) 
    print('total beta particle number=',sums[2])
    print(arrs[2])
    os.remove(bra)
    for fname in benvs.values():
        os.remove(fname) 
    return 
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
    def __init__(self,H,peps,chi,psi_fname,tmpdir,small_mem=False,
                 opt_norm=False,mu=None,opt_mu=False,
                 profile=False,dense_row=True):
        self.start_time = time.time()
        self.H = H
        self.D = peps[0,0].shape[0]
        self.chi = chi
        self.opt_norm = opt_norm
        self.mu = mu
        self.opt_mu = opt_mu
        self.profile = profile
        self.dense_row = dense_row
        self.tmpdir = tmpdir
        self.small_mem = small_mem
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
        E,n = compute_energy(self.H,psi,self.tmpdir,dense_row=self.dense_row,
                               max_bond=self.chi)
        print(f'\tne={self.ne},time={time.time()-self.start_time}')
        print(f'\t\tE={E},norm={n}')
        self.ne += 1
        self.fac = n**(-1.0/(2.0*psi.Lx*psi.Ly))
        return E,n
    def compute_grad(self,x):
        psi = self.vec2fpeps(x)
        if self.mu is None:
            if self.small_mem:
                H0,H1,n0,n1 = compute_grad_small_mem(self.H,psi,self.tmpdir,
                    max_bond=self.chi,dense_row=self.dense_row,profile=self.profile)
            else:
                H0,H1,n0,n1 = compute_grad(self.H,psi,self.tmpdir,
                    max_bond=self.chi,dense_row=self.dense_row,profile=self.profile)
        else:
           H0,H1,N0,N1,n0,n1 = compute_grad_N(self.H,psi,self.tmpdir,
               max_bond=self.chi,dense_row=self.dense_row,profile=self.profile)
        g = []
        E = [] # energy
        N = [] # pn
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                E.append(H0[i,j]/n0[i,j])
                gij = (H1[i,j]-n1[i,j]*E[-1])/n0[i,j]
                if self.mu is not None:
                    N.append(N0[i,j]/n0[i,j])
                    gij = gij-self.mu*(N1[i,j]-n1[i,j]*N[-1])/n0[i,j]
                if self.opt_norm:
                    fac = 2.*(n0[i,j]-1.)*psi.Lx*psi.Ly
                    gij = gij+fac*n1[i,j]
                cons = self.constructors[i,j][0]
                g.append(cons.tensor_to_vector(gij))
        g = np.concatenate(g)
        gmax = np.amax(abs(g))
        print(f'\tng={self.ng},time={time.time()-self.start_time}')
        
        dE = max(E)-min(E)
        E = sum(E)/len(E)
        print(f'\t\tE={E},norm={n0[0,0]},gmax={gmax},dE={dE}')

        if self.opt_mu:
            N = sum(N)/len(N)
            g = np.concatenate([g,-N*np.ones(1)])
            print(f'\t\tN={N},mu={self.mu},gmu={-N}')

        self.ng += 1
        self.fac = n0[0,0]**(-1.0/(2.0*psi.Lx*psi.Ly))

        f = E
        if self.mu is not None:
            f -= self.mu*N
        if self.opt_norm:
            f += psi.Lx*psi.Ly*(n0[0,0]-1.)**2
        return f,g
    def callback(self,x,g):
        if self.opt_mu:
            x = list(x)
            g = list(g)
            self.mu = x.pop()
            gmu = g.pop()
        if not self.opt_norm:
            x *= self.fac
            g /= self.fac
        psi = self.vec2fpeps(x)
        if self.opt_mu:
            x = np.array(x+[self.mu])
            g = np.array(g+[gmu])

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
        if self.opt_mu:
            x0 = np.concatenate([x0,np.ones(1)*self.mu])
        scipy.optimize.minimize(fun=self.compute_grad,jac=True,
                 method=method,x0=x0,
                 callback=self.callback,options=options)
        return
