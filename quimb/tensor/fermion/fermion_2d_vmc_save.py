import time,itertools,h5py
import numpy as np

from pyblock3.algebra.fermion_ops import vaccum,creation,bonded_vaccum
from quimb.utils import progbar as Progbar
from ..tensor_2d import PEPS
from .fermion_core import FermionTensor, FermionTensorNetwork, tensor_contract
from .fermion_2d import FPEPS,FermionTensorNetwork2D
from .block_interface import Constructor
from .tnvmc import (
    TNVMC,AmplitudeFactory,
    MetropolisHastingsSampler,ExchangeSampler,
    format_number_with_error,
)
from .utils import (
    load_ftn_from_disc,write_ftn_to_disc,rand_fname,
    parallelized_looped_fxn,worker_execution,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

symmetry = 'u11'
flat = True
thresh = 1e-6
precision = 1e-10
order = 'row'

cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
vac = vaccum(n=1,symmetry=symmetry,flat=flat)
occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
state_map = [vac,occ_a,occ_b,occ_db]
pn_map = [0,1,1,2]
config_map = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}

####################################################################################
# configuration ordering 
####################################################################################
def site2row(i,j,Lx,Ly): # flattern site to row order
    return i*Ly+j
def row2site(ix,Lx,Ly): # ix in row order
    return ix//Ly,ix%Ly
def site2col(i,j,Lx,Ly):
    return j*Lx+i
def col2site(ix,Lx,Ly):
    return ix%Lx,ix//Lx
if order=='row':
    flatten = site2row # natural order
    flat2site = row2site
    flatten_ = site2col # other order
    flat2site_ = col2site
else:
    flatten_ = site2row
    flat2site_ = row2site
    flatten = site2col
    flat2site = col2site
def reorder(config,Lx,Ly,sign=True):
    print('called')
    config_ = [None] * len(config)
    for ix,ci in enumerate(config):
        i,j = flat2site(ix,Lx,Ly) 
        config_[flatten_(i,j,Lx,Ly)] = ci
    config_ = tuple(config_)
    if not sign:
        return config_,None
    ftn = FermionTensorNetwork([])
    for ix,ci in enumerate(config):
        i,j = flat2site(ix,Lx,Ly) 
        ftn.add_tensor(FermionTensor(data=state_map[ci].copy(),inds=(f'k{i},{j}',),tags=f'k{i},{j}'),virtual=True)
    for ix,ci in reversed(list(enumerate(config_))):
        i,j = flat2site_(ix,Lx,Ly)
        ftn.add_tensor(FermionTensor(data=state_map[ci].dagger,inds=(f'k{i},{j}',),tags=f'b{i},{j}'),virtual=True)
    sign = 1.
    for i,j in itertools.product(range(Lx),range(Ly)):
        sign *= tensor_contract(ftn[f'b{i},{j}'],ftn[f'k{i},{j}'],inplace=True)
    return config_,sign

####################################################################################
# initialization fxns
####################################################################################
def get_vaccum(Lx,Ly):
    """
    helper function to generate initial guess from regular PEPS
    |psi> = \prod (|alpha> + |beta>) at each site
    this function only works for half filled case with U1 symmetry
    Note energy of this wfn is 0
    """
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=1)
    ftn = FermionTensorNetwork([])
    ind_to_pattern_map = dict()
    inv_pattern = {"+":"-", "-":"+"}
    def get_pattern(inds):
        """
        make sure patterns match in input tensors, eg,
        --->A--->B--->
         i    j    k
        pattern for A_ij = +-
        pattern for B_jk = +-
        the pattern of j index must be reversed in two operands
        """
        pattern = ""
        for ix in inds[:-1]:
            if ix in ind_to_pattern_map:
                ipattern = inv_pattern[ind_to_pattern_map[ix]]
            else:
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<len(pattern) else "+"
                ind_to_pattern_map[ix] = ipattern
            pattern += ipattern
        pattern += "+" # assuming last index is the physical index
        return pattern
    for ix, iy in itertools.product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = get_pattern(T.inds)
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        data = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern,
                             symmetry=symmetry,flat=flat)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn
def create_particle(fpeps,site,spin):
    cre = creation(spin=spin,symmetry=symmetry,flat=flat)
    T = fpeps[fpeps.site_tag(*site)]
    trans_order = list(range(1,T.ndim))+[0] 
    data = np.tensordot(cre, T.data, axes=((1,), (-1,))).transpose(trans_order)
    T.modify(data=data)
    return fpeps
def get_product_state(Lx,Ly,spin_map):
    fpeps = get_vaccum(Lx,Ly)
    for spin,sites in spin_map.items():
        for site in sites:
            fpeps = create_particle(fpeps,site,spin)
    return fpeps
def get_constructors(fpeps):
    constructors = dict()
    for i,j in itertools.product(range(fpeps.Lx),range(fpeps.Ly)):
        data = fpeps[fpeps.site_tag(i,j)].data
        bond_infos = [data.get_bond_info(ax,flip=False) \
                      for ax in range(data.ndim)]
        cons = Constructor.from_bond_infos(bond_infos,data.pattern)
        constructors[i,j] = cons,data.dq
    return constructors
def fpeps2vecs(constructors,fpeps_or_grad,Lx=None,Ly=None,concat=False):
    is_fpeps = isinstance(fpeps_or_grad,FPEPS)
    if is_fpeps:
        fpeps = fpeps_or_grad
        Lx,Ly = fpeps.Lx,fpeps.Ly 
    else:
        grad = fpeps_or_grad
    ls = [None] * (Lx*Ly)
    for i,j in itertools.product(range(Lx),range(Ly)):
        cons,dq = constructors[i,j]
        if is_fpeps:
            vec = cons.tensor_to_vector(fpeps[fpeps.site_tag(i,j)].data)
        else:
            if (i,j) in grad:
                vec = cons.tensor_to_vector(grad[i,j])
            else:
                vec = np.zeros(cons.vector_size(dq))
        ls[flatten(i,j,Lx,Ly)] = vec
    if concat:
        return np.concatenate(ls)
    else:
        return ls
def vec2fpeps(constructors,x,fpeps=None,Lx=None,Ly=None,inplace=False): 
    if fpeps is None:
        fpeps_new = dict()
    else:
        fpeps_new = fpeps if inplace else fpeps.copy()
        Lx,Ly = fpeps.Lx,fpeps.Ly 
    start = 0
    for i,j in itertools.product(range(Lx),range(Ly)):
        cons,dq = constructors[i,j]
        stop = start+cons.vector_size(dq)
        data = cons.vector_to_tensor(x[start:stop],dq)
        if fpeps is None:
            fpeps_new[i,j] = data
        else:
            fpeps_new[fpeps.site_tag(i,j)].modify(data=data)
        start = stop
    return fpeps_new

####################################################################################
# amplitude fxns 
####################################################################################
def get_bra_tsr(fpeps,ci,site):
    inds = fpeps.site_ind(*site),
    tags = fpeps.site_tag(*site)
    data = state_map[ci].dagger
    data.shape = fpeps[tags].shape[-1],
    return FermionTensor(data=data,inds=inds,tags=tags)
def update_cache_mid(fpeps,config,i,cache_mid,direction):
    if direction=='row':
        L = fpeps.Ly
        tag = fpeps.row_tag
        _flat2site = row2site
    else:
        L = fpeps.Lx
        tag = fpeps.col_tag
        _flat2site = col2site
    start,stop = i*L,(i+1)*L
    config_row = config[start:stop] 
    if (i,config_row) in cache_mid:
        row = cache_mid[i,config_row] 
        return row,cache_mid

    row = fpeps.select(tag(i)).copy()
    for ix in range(stop-1,start-1,-1):
        site = _flat2site(ix,fpeps.Lx,fpeps.Ly)
        row.add_tensor(get_bra_tsr(fpeps,config[ix],site))
    try:
        for ix in range(stop-1,start-1,-1):
            site = _flat2site(ix,fpeps.Lx,fpeps.Ly)
            row.contract_tags(fpeps.site_tag(*site),inplace=True)
        cache_mid[i,config_row] = row
    except (ValueError,IndexError):
        row = None 
        cache_mid[i,config_row] = None
    return row,cache_mid
def update_cache_head(fpeps,config,cache_head,cache_mid,imin,imax,
                      direction,**compress_opts):
    # compute all bottom envs
    ftn = FermionTensorNetwork([]) 
    L = fpeps.Ly if direction=='row' else fpeps.Lx
    imax_ = fpeps.Lx-1 if direction=='row' else fpeps.Ly-1
    for i in range(imin,imax+1):
        key = config[:(i+1)*L]
        if key in cache_head:
            ftn = cache_head[key]
            continue
        if ftn is None:
            cache_head[key] = None 
            continue
        row,cache_mid = update_cache_mid(fpeps,config,i,cache_mid,direction)
        if row is None:
            cache_head[key] = None 
            continue
        ftn = FermionTensorNetwork([ftn,row]).view_as_(
                  FermionTensorNetwork2D,like=row)
        if i>0:
            try:
                if direction=='row':
                    ftn.contract_boundary_from_bottom_(
                        xrange=(i-1,i),yrange=(0,fpeps.Ly-1),**compress_opts)
                else:
                    ftn.contract_boundary_from_left_(
                        yrange=(i-1,i),xrange=(0,fpeps.Lx-1),**compress_opts)
            except (ValueError,KeyError):
                ftn = None
                cache_head[key] = None 
                continue
        if i<imax_:
            cache_head[key] = ftn.copy()
    return cache_head,cache_mid
def update_cache_tail(fpeps,config,cache_tail,cache_mid,imin,imax,
                     direction='row',**compress_opts):
    # compute all top envs
    ftn = FermionTensorNetwork([]) 
    L = fpeps.Ly if direction=='row' else fpeps.Lx
    imax_ = fpeps.Lx-1 if direction=='row' else fpeps.Ly-1
    for i in range(imax,imin-1,-1):
        key = config[i*L:]
        if key in cache_tail:
            ftn = cache_tail[key]
            continue
        if ftn is None:
            cache_tail[key] = None 
            continue
        row,cache_mid = update_cache_mid(fpeps,config,i,cache_mid,direction)
        if row is None:
            cache_tail[key] = None 
            continue
        ftn = FermionTensorNetwork([row,ftn]).view_as_(
                  FermionTensorNetwork2D,like=row)
        if i<imax_:
            try:
                if direction=='row':
                    ftn.contract_boundary_from_top_(
                        xrange=(i,i+1),yrange=(0,fpeps.Ly-1),**compress_opts)
                else:
                    ftn.contract_boundary_from_right_(
                        yrange=(i,i+1),xrange=(0,fpeps.Lx-1),**compress_opts)
            except (ValueError,KeyError):
                ftn = None
                cache_tail[key] = None 
                continue
        if i>0:
            cache_tail[key] = ftn.copy()
    return cache_tail,cache_mid
def compute_config_parity(config,parity_fpeps,L,imax):
    parity_config = [None] * (imax+1)
    for i in range(imax+1):
        parity_config[i] = sum([pn_map[ci] for ci in config[i*L:(i+1)*L]]) % 2
    parity = 0
    for i in range(imax-1,-1,-1):
        parity_ = (sum(parity_fpeps[i+1:]) + sum(parity_config[i+1:])) % 2
        parity += parity_config[i] * parity_
    return parity % 2,parity_config
def compute_amplitude(fpeps,parity,config,direction,split,
                      cache_head,cache_mid,cache_tail,**compress_opts):
    # fpeps,config in same order
    if direction=='row':
        L,imax = fpeps.Ly,fpeps.Lx-1
    else:
        L,imax = fpeps.Lx,fpeps.Ly-1
    split = split - 1 if split == imax else split
    cache_head,cache_mid = update_cache_head(
            fpeps,config,cache_head,cache_mid,0,split,direction,**compress_opts)
    cache_tail,cache_mid = update_cache_tail(
            fpeps,config,cache_tail,cache_mid,split+1,imax,direction,**compress_opts)
    ftn_head = cache_head[config[:(split+1)*L]]
    ftn_tail = cache_tail[config[(split+1)*L:]]
    if ftn_head is None or ftn_tail is None:
        amp = 0.
    else:
        try:
            amp = FermionTensorNetwork([ftn_head,ftn_tail]).contract()
            parity,_ = compute_config_parity(config,parity,L,imax)
            if parity==1:
                amp *= -1
        except (ValueError,KeyError):
            amp = 0.
    return amp,cache_head,cache_mid,cache_tail
def compute_3col_envs(tn,direction,step,imin,imax,envs):
    sweep = range(imin,imax+1) if step==1 else range(imax,imin-1,-1)
    from_which = 'head' if step==1 else 'tail'
    row_tag = tn.row_tag if direction=='row' else tn.col_tag

    envs[from_which, sweep[0]] = FermionTensorNetwork([])
    first_row = row_tag(sweep[0])
    try:
        tn ^= first_row
        envs[from_which, sweep[1]] = tn.select(first_row).copy()
    except (ValueError,IndexError):
        tn = None
        envs[from_which, sweep[1]] = None 

    for i in sweep[2:]:
        if tn is None:
            envs[from_which,i] = None
            continue
        iprevprev = i - 2 * step
        iprev = i - step
        try:
            tn ^= (row_tag(iprevprev), row_tag(iprev))
            envs[from_which, i] = tn.select(first_row).copy()
        except (ValueError,IndexError):
            tn = None
            envs[from_which,i] = None
    return envs
def compute_grad(fpeps,parity,config,direction,cache_head,cache_mid,cache_tail,
                 **compress_opts):
    # fpeps,config in same order
    if direction=='row':
        L,imax = fpeps.Ly,fpeps.Lx-1
        _flat2site = row2site
        direction_ = 'col'
        row_tag = fpeps.col_tag
    else:
        L,imax = fpeps.Lx,fpeps.Ly-1
        _flat2site = col2site
        direction_ = 'row'
        row_tag = fpeps.row_tag
    for i in range(imax+1):
        _,cache_mid = update_cache_mid(fpeps,config,i,cache_mid,direction)
    cache_head,_ = update_cache_head(
        fpeps,config,cache_head,cache_mid,0,imax,direction,**compress_opts)
    cache_tail,_ = update_cache_tail(
        fpeps,config,cache_tail,cache_mid,0,imax,direction,**compress_opts)
    # compute grad
    amps = []
    grad = dict()
    parity_amp,parity_config = compute_config_parity(config,parity,L,imax)
    for i in range(imax+1):
        ftn_head = FermionTensorNetwork([]) if i==0 else cache_head[config[:i*L]]
        if ftn_head is None:
            continue
        ftn_mid = cache_mid[i,config[i*L:(i+1)*L]]
        if ftn_mid is None:
            continue
        ftn_tail = FermionTensorNetwork([]) if i==imax else \
                   cache_tail[config[(i+1)*L:]]
        if ftn_tail is None:
            continue
        ftn = FermionTensorNetwork([ftn_head,ftn_mid,ftn_tail]).view_as_(
                  FermionTensorNetwork2D,like=ftn_mid)
        ftn.reorder(direction_,inplace=True)
        envs = {('mid',j):ftn.select(row_tag(j)).copy() for j in range(L)}
        envs = compute_3col_envs(ftn.copy(),direction_,1,0,L-1,envs)
        envs = compute_3col_envs(ftn.copy(),direction_,-1,0,L-1,envs)
        for j in range(L):
            if envs['head',j] is None:
                continue
            if envs['tail',j] is None:
                continue
            ftn_ij = FermionTensorNetwork([envs[side,j] \
                                           for side in ['head','mid','tail']])

            ix = i*L+j
            site = _flat2site(ix,fpeps.Lx,fpeps.Ly) 
            tag = fpeps.site_tag(*site)
            tid = ftn_ij[tag].get_fermion_info()[0]
            Tv = ftn_ij._pop_tensor(tid,remove_from_fermion_space='end')

            try:
                target = fpeps[tag]
                #print(target.inds[:-1])
                g = ftn_ij.contract()
                #g = ftn_ij.contract()
                #print(g.inds)
                amps.append(FermionTensorNetwork([g,Tv]).contract())
                
                v = get_bra_tsr(fpeps,config[ix],site)
                if pn_map[config[ix]] % 2 == 1:
                    tmp = FermionTensorNetwork([target,v]).contract(
                                 output_inds=target.inds[:-1])
                    if (Tv.data-tmp.data).norm()<thresh:
                        pass
                    elif (Tv.data+tmp.data).norm()<thresh:
                        v.data._global_flip()
                    else:
                        print(T.data)
                        print(tmp.data)
                        raise ValueError('T and tmp not related by a global phase!')
                    if target.data.parity == 1:
                        v.data._global_flip()
                    v.data._local_flip([0])

                g = FermionTensorNetwork([g,v]).contract(
                        output_inds=target.inds[::-1])
                grad[site] = g.data.dagger 
            except (ValueError,IndexError):
                continue
    amp = 0.
    if len(amps)>0:
        amp = sum(amps)/len(amps)
        if np.fabs(amp)<precision:
            grad = dict()
        else:
            for key in grad:
                grad[key] = grad[key] / amp
            if parity_amp==1:
                amp *= -1 
    return amp,grad,cache_head,cache_mid,cache_tail
def compute_fpeps_parity(fs,start,stop):
    if start==stop:
        return 0
    tids = [fs.get_tid_from_site(site) for site in range(start,stop)]
    tsrs = [fs.tensor_order[tid][0] for tid in tids] 
    return sum([tsr.parity for tsr in tsrs]) % 2
def get_row_parity(fpeps):
    parity = [None] * fpeps.Lx
    fs = fpeps.fermion_space
    for i in range(fpeps.Lx):
        start,stop = i*fpeps.Ly,(i+1)*fpeps.Ly
        parity[i] = compute_fpeps_parity(fs,start,stop)
    return parity
def get_col_parity(fpeps):
    parity = [None] * fpeps.Ly
    fs = fpeps.fermion_space
    for i in range(fpeps.Ly):
        start,stop = i*fpeps.Lx,(i+1)*fpeps.Lx
        parity[i] = compute_fpeps_parity(fs,start,stop)
    return parity
class AmplitudeFactory2D(AmplitudeFactory):
    def _set_psi(self,psi):
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.psi_row = psi.reorder(direction='row',inplace=False)
        self.psi_col = psi.reorder(direction='col',inplace=False)
        self.row_parity = get_row_parity(self.psi_row)
        self.col_parity = get_col_parity(self.psi_col)
        self.cache_bottom = dict()
        self.cache_top = dict()
        self.cache_left = dict()
        self.cache_right = dict()
        self.cache_mid_row = dict()
        self.cache_mid_col = dict()
        return
    def get_cache(self,direction):
        if direction=='row':
            return self.cache_bottom,self.cache_mid_row,self.cache_top
        else:
            return self.cache_left,self.cache_mid_col,self.cache_right
    def set_cache(self,direction,cache_head,cache_mid,cache_tail):
        if direction=='row':
            self.cache_bottom = cache_head
            self.cache_mid_row = cache_mid
            self.cache_top = cache_tail
        else:
            self.cache_left = cache_head
            self.cache_mid_col = cache_mid
            self.cache_right = cache_tail
    def reorder(self,config,sign=True):
        return reorder(config,self.Lx,self.Ly,sign=sign)
    def amplitude(self, config):
        """Get the amplitude of ``config``, either from the cache or by
        computing it.
        """
        config,direction,split = config 
        self.queries += 1
        if config in self.store:
            self.hits += 1
            return self.store[config]

        if direction=='row':
            ordered_config,sign = config,1.
            fpeps,parity = self.psi_row,self.row_parity
        else:
            ordered_config,sign = self.reorder(config)
            fpeps,parity = self.psi_col,self.col_parity
        cache_head,cache_mid,cache_tail = self.get_cache(direction)
        amp,cache_head,cache_mid,cache_tail = compute_amplitude(
            fpeps,parity,ordered_config,direction,split,\
            cache_head,cache_mid,cache_tail,**self.contract_opts) 
        self.set_cache(direction,cache_head,cache_mid,cache_tail)
        amp *= sign
        self.store[config] = amp 
        return amp 
    def _amplitude(self,config):
        config,direction,split = config 
        self.queries += 1
        if config in self.store:
            self.hits += 1
            return self.store[config]

        if direction=='row':
            ordered_config,sign = config,1.
        else:
            ordered_config,sign = self.reorder(config)
        fpeps = self.psi_row.copy()
        for ix,i in reversed(list(enumerate(ordered_config))):
            site = flat2site(ix,self.Lx,self.Ly)
            fpeps.add_tensor(get_bra_tsr(fpeps,i,site))
        try:
            amp = fpeps.contract()
        except (ValueError,IndexError):
            amp = 0.
        amp *= sign
        self.store[config] = amp 
        return amp
    def grad(self,config,direction):
        self.queries += 1
        if config in self.store:
            self.hits += 1

        if direction=='row':
            ordered_config,sign = config,1.
            fpeps,parity = self.psi_row,self.row_parity
        else:
            ordered_config,sign = self.reorder(config)
            fpeps,parity = self.psi_col,self.col_parity
        cache_head,cache_mid,cache_tail = self.get_cache(direction)
        amp,grad,cache_head,cache_mid,cache_tail = compute_grad(
            fpeps,parity,ordered_config,direction,\
            cache_head,cache_mid,cache_tail,**self.contract_opts) 
        self.set_cache(direction,cache_head,cache_mid,cache_tail)
        amp *= sign
        for key in grad:
            grad[key] *= sign
        self.store[config] = amp 
        return amp,grad 
    def compute_single(self):
        raise NotImplementedError
    def compute_multi(self):
        raise NotImplementedError
    def amplitudes(self):
        raise NotImplementedError

####################################################################################
# ham class 
####################################################################################
def hop(ix1,ix2,config):
    i1,i2 = config[ix1],config[ix2]
    if i1==i2:
        return []
    n1,n2 = pn_map[i1],pn_map[i2]
    nsum,ndiff = n1+n2,abs(n1-n2)
    if ndiff==1:
        sign = 1 if nsum==1 else -1
        config_new = list(config)
        config_new[ix1] = i2 
        config_new[ix2] = i1
        return [(tuple(config_new),sign)]
    configs = []
    if ndiff==2:
        for i1_new,i2_new in ((1,2),(2,1)):
            sign = i1_new-i2_new
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new
            configs.append((tuple(config_new),sign))
    if ndiff==0:
        sign = i1-i2
        for i1_new,i2_new in ((0,3),(3,0)):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new
            configs.append((tuple(config_new),sign))
    return configs
class Hubbard2D:
    def __init__(self,Lx,Ly,t,u):
        self.Lx,self.Ly = Lx,Ly
        self.t,self.u = t,u
    def flatten(self,i,j):
        return flatten(i,j,self.Lx,self.Ly)        
    def config_coupling(self,config):
        configs = []
        coeffs = []

        direction = 'row'
        for i, j in itertools.product(range(self.Lx), range(self.Ly-1)):
            ix1,ix2 = self.flatten(i,j), self.flatten(i,j+1)
            configs_ = hop(ix1,ix2,config)
            for config_,sign_ in configs_:
                configs.append((config_,direction,i))
                coeffs.append(-self.t*sign_) 

        #direction = 'col'
        for i, j in itertools.product(range(self.Lx-1), range(self.Ly)):
            ix1,ix2 = self.flatten(i,j), self.flatten(i+1,j)
            sign = (-1)**sum([pn_map[ci] for ci in config[ix1+1:ix2]])
            configs_ = hop(ix1,ix2,config)
            for config_,sign_ in configs_:
                configs.append((config_,direction,i))
                coeffs.append(-self.t*sign*sign_) 
   
        configs.append(None)
        config = np.array(config,dtype=int)
        coeffs.append(self.u*len(config[config==3]))
        return configs,coeffs

####################################################################################
# sampler 
####################################################################################
def _dense_amplitude_wrapper(ixs,psi,all_configs,direction,compress_opts):
    psi = load_ftn_from_disc(psi)
    psi.reorder(direction,inplace=True)
    parity = get_row_parity(psi) if direction=='row' else get_col_parity(psi)

    p = np.zeros(len(all_configs)) 
    cache_head = dict()
    cache_mid = dict()
    cache_tail = dict()
    for ix in ixs:
        config = all_configs[ix]
        amp,cache_head,cache_mid,cache_tail = \
            compute_amplitude(psi,parity,config,direction,0,
            cache_head,cache_mid,cache_tail,**compress_opts)
        p[ix] = amp**2
    return p
class DenseSampler2D:
    def __init__(self,Lx,Ly,nelec,seed=None,**contract_opts):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly
        self.rng = np.random.default_rng(seed)
        self.nelec = nelec
        self.contract_opts = contract_opts
        self.configs = self.get_all_configs()
    def get_all_configs(self):
        if symmetry=='u1':
            return self.get_all_configs_u1()
        elif symmetry=='u11':
            return self.get_all_configs_u11()
        else:
            raise NotImplementedError
    def get_all_configs_u11(self):
        assert isinstance(self.nelec,tuple)
        sites = list(range(self.nsite))
        ls = [None] * 2
        for spin in (0,1):
            occs = list(itertools.combinations(sites,self.nelec[spin]))
            configs = [None] * len(occs) 
            for i,occ in enumerate(occs):
                config = [0] * self.nsite 
                for ix in occ:
                    config[ix] = 1
                configs[i] = tuple(config)
            ls[spin] = configs

        na,nb = len(ls[0]),len(ls[1])
        configs = [None] * (na*nb)
        for ixa,configa in enumerate(ls[0]):
            for ixb,configb in enumerate(ls[1]):
                config = [config_map[configa[i],configb[i]] \
                          for i in range(self.nsite)]
                ix = ixa * nb + ixb
                configs[ix] = tuple(config)
        return configs
    def get_all_configs_u1(self):
        if isinstance(self.nelec,tuple):
            self.nelec = sum(self.nelec)
        sites = list(range(self.nsite*2))
        occs = list(itertools.combinations(sites,self.nelec))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = [0] * (self.nsite*2) 
            for ix in occ:
                config[ix] = 1
            configs[i] = tuple(config)

        for ix in range(len(configs)):
            config = configs[ix]
            configa,configb = config[:self.nsite],config[self.nsite:]
            config = [config_map[configa[i],configb[i]] for i in range(self.nsite)]
            configs[ix] = tuple(config)
        return configs
    def _set_psi(self,psi=None,load_fname=None,write_fname=None):
        nconfig = len(self.configs)
        if load_fname is None:
            self.p = np.zeros(nconfig)
            direction = 'row'
            t0 = time.time()
            batchsize = nconfig // SIZE + 1
            ls = [list(range(worker*batchsize,min((worker+1)*batchsize,nconfig))) \
                  for worker in range(SIZE)]
            args = psi,self.configs,direction,self.contract_opts
            ls = parallelized_looped_fxn(_dense_amplitude_wrapper,ls,args)
            for p in ls:
                self.p += p
            self.p /= np.sum(self.p) 
            print(f'dense sampler updated ({time.time()-t0}s).')
        else:
            f = h5py.File(load_fname,'r')
            self.p = f['p'][:]
            f.close()
        self.flat_indexes = list(range(nconfig))
        if write_fname is not None:
            f = h5py.File(write_fname,'w')
            f.create_dataset('p',data=self.p)
            f.close()
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        omega = self.p[flat_idx]
        config = self.configs[flat_idx]
        return (config,'row',None),omega
    def flatten(self,i,j):
        return flatten(i,j,slef.Lx,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
class ExchangeSampler2D(ExchangeSampler):
    def __init__(self,Lx,Ly,nelec,seed=None,sweep=True):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly
        self.rng = np.random.default_rng(seed)
        self.nelec = nelec

        self.config = self.get_rand_config(),'row',0
        self.sweep = None 
        if sweep:
            self.sweep = self.rng.integers(low=0,high=1,endpoint=True),0,0
        else:
            self.blocks = [(i,j) for i in range(Lx-1) for j in range(Ly-1)] 
    def flatten(i,j):
        return flatten(i,j,self.Lx,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def get_rand_config(self):
        if symmetry=='u1':
            return self.get_rand_config_u1()
        elif symmetry=='u11':
            return self.get_rand_config_u11()
        else:
            raise NotImplementedError
    def get_rand_config_u11(self):
        assert isinstance(self.nelec,tuple)
        config = np.zeros((self.nsite,2),dtype=int)
        sites = np.array(range(self.nsite),dtype=int)
        for spin in (0,1):
            occ = self.rng.choice(sites,size=self.nelec[spin],
                                  replace=False,shuffle=False)
            for ix in occ:
                config[ix,spin] = 1
        config = [config_map[tuple(config[i,:])] for i in range(self.nsite)]
        return tuple(config)
    def get_rand_config_u1(self):
        if isinstance(self.nelec,tuple):
            self.nelec = sum(self.nelec)
        sites = np.array(range(self.nsite),dtype=int)
        occ = self.rng.choice(sites,size=self.nelec,replace=False,shuffle=False)
        for ix in occ:
            config[ix] = 1
        configa,configb = config[:self.nsite],config[self.nsite:]
        config = [config_map[configa[i],configb[i]] for i in range(self.nsite)]
        return tuple(config)
    def new_config(self,nconfig,i,j):
        randint = self.rng.integers(low=0,high=3,endpoint=True)
        if randint==0:
            site1,site2 = (i,j),(i,j+1)
            direction,split = 'row',i
        elif randint==1:
            site1,site2 = (i,j),(i+1,j)
            direction,split = 'col',j
        if randint==2:
            site1,site2 = (i+1,j),(i+1,j+1)
            direction,split = 'row',i+1
        elif randint==3:
            site1,site2 = (i,j+1),(i+1,j+1)
            direction,split = 'col',j+1
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2) 
        i1,i2 = nconfig[ix1],nconfig[ix2]
        if i1==i2:
            return False,nconfig,direction,split
        if symmetry=='u11':
            i1_new,i2_new = self.new_config_u11(i1,i2)
        else:
            raise NotImplementedError
        nconfig[ix1] = i1_new
        nconfig[ix2] = i2_new
        return True,nconfig,direction,split
    def new_config_u11(self,i1,i2):
        n = abs(pn_map[i1]-pn_map[i2])
        if n==1:
            i1_new,i2_new = i2,i1
        else:
            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
            i1_new,i2_new = self.rng.choice(choices)
        return i1_new,i2_new 
    def new_config_u1(self,i1,i2):
        return
    def candidate_rand(self):
        nconfig = list(self.config[0])
        blocks = self.rng.permutation(self.blocks)
        for i,j in blocks:
            is_new,nconfig,direction,split = self.new_config(nconfig,i,j)
            if is_new:
                return (tuple(nconfig),direction,split),1.
    def candidate_sweep(self):
        nconfig = list(self.config[0])
        direction_,i,j = self.sweep
        while True:
            is_new,nconfig,direction,split = self.new_config(nconfig,i,j)
            if i==self.Lx-2 and j==self.Ly-2:
                direction_ = self.rng.integers(low=0,high=1,endpoint=True)
                i,j = 0,0
            else:
                if direction_==0:
                    i,j = (i+1,0) if j==self.Ly-2 else (i,j+1)
                else:
                    i,j = (0,j+1) if i==self.Lx-2 else (i+1,j)
            self.sweep = direction_,i,j
            if is_new:
                return (tuple(nconfig),direction,split),1.
    def candidate(self):
        if self.sweep is None:
            return candidate_rand()
        else:
            return candidate_sweep()
####################################################################################
# vmc engine 
####################################################################################
def _parse_sampler(Lx,Ly,sampler,sampler_opts,contract_opts):
    seed = sampler_opts.get('seed',None)
    nelec = sampler_opts['nelec']
    if sampler=='dense':
        sampler = DenseSampler2D(Lx,Ly,nelec,seed=seed,**contract_opts)
        psi = sampler_opts.get('psi',None)
        load_fname = sampler_opts.get('load_fname',None)
        write_fname = sampler_opts.get('write_fname',None)
        sampler._set_psi(psi=psi,load_fname=load_fname,write_fname=write_fname)
        return sampler
    if sampler=='exchange':
        sweep = sampler_opts.get('sweep',True)
        sub_sampler = ExchangeSampler2D(Lx,Ly,nelec,seed=seed,sweep=sweep)
    else:
        raise NotImplementedError
    amplitude_factory = sampler['amplitude_factory']
    initial = sampler_opts.get('initial',None)
    burn_in = sampler_opts.get('burn_in',0)
    track = sampler_opts.get('track',False)
    sampler = MetropolisHastingsSampler(sub_sampler,
        amplitude_factory=amplitude_factory,
        initial=initial,burn_in=burn_in,seed=seed,track=track)
    return sampler
def _compute_local_energy(ham,amplitude_factory,config,cx):
    en = 0.0
    c_configs, c_coeffs = ham.config_coupling(config)
    for hxy, config_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < thresh:
            continue
        cy = cx if config_y is None else amplitude_factory.amplitude(config_y)
        en += hxy * cy 
    return en / cx 
def _local_sampling(batchsize,psi,ham,sampler,sampler_opts,progbar,contract_opts):
    psi = load_ftn_from_disc(psi)
    constructors = get_constructors(psi)

    amplitude_factory = AmplitudeFactory2D(psi=psi,**contract_opts) 
    if sampler != 'dense':
        sampler_opts['amplitude_factory'] = amplitude_factory
    sampler = _parse_sampler(psi.Lx,psi.Ly,sampler,sampler_opts,contract_opts)

    store = dict()
    local_energies = []
    grads_logpsi = []
    progbar = progbar and (RANK==0)
    if progbar:
        _progbar = Progbar(total=batchsize)
    direction = 'row'
    for _ in range(batchsize):
        (config,direction,split), omega = sampler.sample()

        # compute and track local energy
        if config in store:
            cx,gx,local_energy = store[config]
        else:
            cx,gx = amplitude_factory.grad(config,direction)
            gx = fpeps2vecs(constructors,gx,psi.Lx,psi.Ly,concat=False)
            if np.fabs(cx)<precision:
                local_energy = 0.
            else:
                local_energy = \
                    _compute_local_energy(ham,amplitude_factory,config,cx)
            store[config] = cx,gx,local_energy

        local_energies.append(local_energy)
        grads_logpsi.append(gx)
        if progbar:
            _progbar.update()
    return local_energies,grads_logpsi
class TNVMC2D(TNVMC):
    def _update_stats(self,local_energies,grads_logpsi):
        for local_energy,grads_logpsi_sample in zip(local_energies,grads_logpsi):
            self.local_energies.append(local_energy)
            self.moving_stats.update(local_energy)
            self.energies.append(self.moving_stats.mean)
            self.energy_errors.append(self.moving_stats.err)

            self.optimizer.update(grads_logpsi_sample, local_energy)

            if self._progbar is not None:
                self._progbar.update()
                self._progbar.set_description(
                    format_number_with_error(
                        self.moving_stats.mean,
                        self.moving_stats.err))
        return 
    def _parse_sampler(self,psi):
        if self.sampler=='dense':
            self.sampler_opts['psi'] = psi
            self.sampler_opts['load_fname'] = None
            self.sampler_opts['write_fname'] = self.dense_p
            _parse_sampler(self.Lx,self.Ly,self.sampler,self.sampler_opts,
                           self.contract_opts)
            self.sampler_opts['psi'] = None
            self.sampler_opts['load_fname'] = self.dense_p 
            self.sampler_opts['write_fname'] = None 
    def _run(self, steps, batchsize):
        self.Lx,self.Ly = self.psi.Lx,self.psi.Ly
        psi = write_ftn_to_disc(self.psi,self.tmpdir+'psi_init',
                                provided_filename=True) 
        self.constructors = get_constructors(self.psi)
        self._parse_sampler(psi) # only called with dense sampler
        for step in range(steps):
            ls = [batchsize] * SIZE
            args = psi,self.ham,self.sampler,self.sampler_opts,self.local_progbar,\
                   self.contract_opts
            ls = parallelized_looped_fxn(_local_sampling,ls,args)
            loc_ens = []
            for local_energies,grads_logpsi in ls:
                loc_ens += local_energies
                self._update_stats(local_energies,grads_logpsi)
            loc_ens = np.array(loc_ens)
            print(f'energy={np.mean(loc_ens)},err={np.std(loc_ens)/len(loc_ens)**.5}')
            #exit()

            # apply learning rate and other transforms to gradients
            deltas = self.optimizer.transform_gradients()

            # update the actual tensors
            self.update_psi(deltas)

            # reset having just performed a gradient step
            if self.conditioner is not None:
                self.conditioner(self.psi)
            psi = write_ftn_to_disc(self.psi,self.tmpdir+f'psi{step}',
                                    provided_filename=True) 
            self._parse_sampler(psi) # only called with dense sampler
    def update_psi(self,deltas):
        for ix,delta in enumerate(deltas):
            i,j = flat2site(ix,self.Lx,self.Ly)
            tsr = self.psi[self.psi.site_tag(i,j)]

            cons,dq = self.constructors[i,j]
            delta = cons.vector_to_tensor(delta,dq)
            data = tsr.data - delta 
            tsr.modify(data=data)
    def run(
        self,
        total=10000, 
        batchsize=100,
        progbar=True,
        local_progbar=True,
        tmpdir='./',
        sampler_opts=dict(),
    ):
        steps = total // batchsize // SIZE
        total = steps * batchsize * SIZE
        self.tmpdir = tmpdir
        self.sampler_opts = sampler_opts 
        self.dense_p = tmpdir+rand_fname()+'.hdf5'
        print('save dense amplitude to '+self.dense_p)

        if progbar:
            self._progbar = Progbar(total=total)
        self.local_progbar = local_progbar 

        try:
            self._run(steps, batchsize)
        except KeyboardInterrupt:
            pass
        finally:
            if self._progbar is not None:
                self._progbar.close()        
    
