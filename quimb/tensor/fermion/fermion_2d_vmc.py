import time,itertools,sys
import numpy as np

from ..tensor_2d import PEPS
from .utils import psi2vecs,vec2psi
from .fermion_core import (
    FermionTensor, FermionTensorNetwork, 
    rand_uuid, tensor_split,
)
from .fermion_2d import FPEPS,FermionTensorNetwork2D
from .block_interface import Constructor
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

SYMMETRY = 'u11' # sampler symmetry

# set tensor symmetry
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True):
    from pyblock3.algebra.fermion_ops import vaccum,creation,H1
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    vac = vaccum(n=1,symmetry=symmetry,flat=flat)
    occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
    occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
    occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
    this.data_map = {0:vac,1:occ_a,2:occ_b,3:occ_db,
                     'cre_a':cre_a,
                     'cre_b':cre_b,
                     'ann_a':cre_a.dagger,
                     'ann_b':cre_b.dagger,
                     'h1':H1(symmetry=symmetry,flat=flat)}
    this.symmetry = symmetry
    this.flat = flat
    return this.data_map
pn_map = [0,1,1,2]
config_map = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}

####################################################################################
# configuration ordering 
####################################################################################
def flatten(i,j,Ly): # flattern site to row order
    return i*Ly+j
def flat2site(ix,Lx,Ly): # ix in row order
    return ix//Ly,ix%Ly
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
    from pyblock3.algebra.fermion_ops import bonded_vaccum
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
def get_constructors_2d(fpeps):
    constructors = [None] * (fpeps.Lx * fpeps.Ly)
    
    for i,j in itertools.product(range(fpeps.Lx),range(fpeps.Ly)):
        data = fpeps[fpeps.site_tag(i,j)].data
        bond_infos = [data.get_bond_info(ax,flip=False) \
                      for ax in range(data.ndim)]
        cons = Constructor.from_bond_infos(bond_infos,data.pattern,flat=this.flat)
        dq = data.dq
        size = cons.vector_size(dq)
        ix = flatten(i,j,fpeps.Ly)
        constructors[ix] = cons,dq,size,(i,j)
    return constructors

####################################################################################
# amplitude fxns 
####################################################################################
def get_bra_tsr(fpeps,ci,i,j):
    inds = fpeps.site_ind(i,j),
    tags = fpeps.site_tag(i,j),fpeps.row_tag(i),fpeps.col_tag(j),'BRA'
    data = data_map[ci].dagger
    return FermionTensor(data=data,inds=inds,tags=tags)
def get_mid_env(i,fpeps,config):
    row = fpeps.select(fpeps.row_tag(i)).copy()
    key = config[i*fpeps.Ly:(i+1)*fpeps.Ly]
    # compute mid env for row i
    for j in range(row.Ly-1,-1,-1):
        row.add_tensor(get_bra_tsr(row,key[j],i,j),virtual=True)
    return row
def contract_mid_env(i,row):
    try: 
        for j in range(row.Ly-1,-1,-1):
            row.contract_tags(row.site_tag(i,j),inplace=True)
    except (ValueError,IndexError):
        row = None 
    return row
def get_bot_env(i,row,env_prev,config,cache,**compress_opts):
    # contract mid env for row i with prev bot env 
    key = config[:(i+1)*row.Ly]
    if key in cache: # reusable
        return cache[key]
    row = contract_mid_env(i,row)
    if i==0:
        cache[key] = row
        return row
    if row is None:
        cache[key] = row
        return row
    if env_prev is None:
        cache[key] = None 
        return None
    ftn = FermionTensorNetwork([env_prev,row],virtual=False).view_like_(row)
    try:
        ftn.contract_boundary_from_bottom_(xrange=(i-1,i),yrange=(0,row.Ly-1),**compress_opts)
    except (ValueError,IndexError):
        ftn = None
    cache[key] = ftn
    return ftn 
def get_all_bot_envs(fpeps,config,cache_bot,imax=None,**compress_opts):
    # imax for bot env
    imax = fpeps.Lx-2 if imax is None else imax
    env_prev = None
    for i in range(imax+1):
         row = get_mid_env(i,fpeps,config)
         env_prev = get_bot_env(i,row,env_prev,config,cache_bot,**compress_opts)
    return env_prev
def get_top_env(i,row,env_prev,config,cache,**compress_opts):
    # contract mid env for row i with prev top env 
    key = config[i*row.Ly:]
    if key in cache: # reusable
        return cache[key]
    row = contract_mid_env(i,row)
    if i==row.Lx-1:
        cache[key] = row
        return row
    if row is None:
        cache[key] = row
        return row
    if env_prev is None:
        cache[key] = None 
        return None
    ftn = FermionTensorNetwork([row,env_prev],virtual=False).view_like_(row)
    try:
        ftn.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,row.Ly-1),**compress_opts)
    except (ValueError,IndexError):
        ftn = None
    cache[key] = ftn
    return ftn 
def get_all_top_envs(fpeps,config,cache_top,imin=None,**compress_opts):
    imin = 1 if imin is None else imin
    env_prev = None
    key_bot = config
    key_top = tuple()
    for i in range(fpeps.Lx-1,imin-1,-1):
         row = get_mid_env(i,fpeps,config)
         env_prev = get_top_env(i,row,env_prev,config,cache_top,**compress_opts)
    return env_prev
def get_all_lenvs(ftn,jmax=None):
    jmax = ftn.Ly-2 if jmax is None else jmax
    first_col = ftn.col_tag(0)
    lenvs = [None] * ftn.Ly
    for j in range(jmax+1): 
        tags = first_col if j==0 else (first_col,ftn.col_tag(j))
        try:
            ftn ^= tags
            lenvs[j] = ftn.select(first_col).copy()
        except (ValueError,IndexError):
            return lenvs
    return lenvs
def get_all_renvs(ftn,jmin=None):
    jmin = 1 if jmin is None else jmin
    last_col = ftn.col_tag(ftn.Ly-1)
    renvs = [None] * ftn.Ly
    for j in range(ftn.Ly-1,jmin-1,-1): 
        tags = last_col if j==ftn.Ly-1 else (ftn.col_tag(j),last_col)
        try:
            ftn ^= tags
            renvs[j] = ftn.select(last_col).copy()
        except (ValueError,IndexError):
            return renvs
    return renvs
def replace_sites(ftn,sites,cis):
    for (i,j),ci in zip(sites,cis): 
        bra = ftn[ftn.site_tag(i,j),'BRA']
        bra_target = get_bra_tsr(ftn,ci,i,j)
        bra.modify(data=bra_target.data.copy(),inds=bra_target.inds)
    return ftn
def site_grad(ftn_plq,i,j):
    ket = ftn_plq[ftn_plq.site_tag(i,j),'KET']
    tid = ket.get_fermion_info()[0]
    ket = ftn_plq._pop_tensor(tid,remove_from_fermion_space='end')
    g = ftn_plq.contract(output_inds=ket.inds[::-1])
    return g.data.dagger 
def compute_fpeps_parity(fs,start,stop):
    if start==stop:
        return 0
    tids = [fs.get_tid_from_site(site) for site in range(start,stop)]
    tsrs = [fs.tensor_order[tid][0] for tid in tids] 
    return sum([tsr.parity for tsr in tsrs]) % 2
class AmplitudeFactory2D:
    def __init__(self,psi,**contract_opts):
        self.contract_opts=contract_opts
        self.Lx,self.Ly = psi.Lx,psi.Ly
        psi.reorder(direction='row',inplace=True)
        psi.add_tag('KET')
        self.constructors = get_constructors_2d(psi)

        self._set_psi(psi) # current state stored in self.psi
        self.parity_cum = self.get_parity_cum()
        self.sign = dict()
    def get_parity_cum(self):
        parity = []
        fs = self.psi.fermion_space
        for i in range(1,self.Lx): # only need parity of row 1,...,Lx-1
            start,stop = i*self.Ly,(i+1)*self.Ly
            parity.append(compute_fpeps_parity(fs,start,stop))
        return np.cumsum(np.array(parity[::-1]))
    def get_x(self):
        return self._psi2vec()
    def update(self,x):
        psi = self._vec2psi(x,inplace=True)
        self._set_psi(psi) 
        return psi
    def _psi2vec(self,psi=None):
        psi = self.psi if psi is None else psi
        return np.concatenate(psi2vecs(self.constructors,psi)) 
    def _vec2psi(self,x,inplace=True):
        psi = self.psi if inplace else self.psi.copy()
        psi = vec2psi(self.constructors,x,psi=psi)
        return psi
    def _set_psi(self,psi):
        self.psi = psi
        self.store = dict()
        self.store_grad = dict()

        self.cache_bot = dict()
        self.cache_top = dict()
        self.compute_bot = True
        self.compute_top = True
        return
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def update_scheme(self,benv_dir):
        if benv_dir == 1:
            self.compute_bot = True
            self.compute_top = False
        elif benv_dir == -1:
            self.compute_top = True
            self.compute_bot = False
        elif benv_dir == 0:
            self.compute_top = True
            self.compute_bot = True 
        else:
            raise NotImplementedError
    def get_all_benvs(self,config,x_bsz=1):
        env_bot = None
        env_top = None
        if self.compute_bot: 
            env_bot = get_all_bot_envs(self.psi,config,self.cache_bot,imax=self.Lx-1-x_bsz,
                                       **self.contract_opts)
        if self.compute_top:
            env_top = get_all_top_envs(self.psi,config,self.cache_top,imin=x_bsz,
                                       **self.contract_opts)
        return env_bot,env_top
    def compute_config_parity(self,config):
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum([pn_map[ci] for ci in config[i*self.Ly:(i+1)*self.Ly]]) % 2
        parity = np.array(parity[::-1])
        parity_cum = np.cumsum(parity[:-1])
        parity_cum += self.parity_cum 
        return np.dot(parity[1:],parity_cum) % 2
    def compute_config_sign(self,config):
        if config in self.sign:
            return self.sign[config]
        sign = (-1) ** self.compute_config_parity(config)
        self.sign[config] = sign
        return sign
    def unsigned_amplitude(self,config):
        # should only be used to:
        # 1. compute dense probs
        # 2. initialize MH sampler
        if config in self.store:
            return self.store[config]
        if self.compute_bot and self.compute_top:
            raise ValueError
        env_bot,env_top = self.get_all_benvs(config,x_bsz=1)
        if env_bot is None and env_top is None:
            unsigned_cx = 0.
            self.store[config] = unsigned_cx
            return unsigned_cx  
        if self.compute_bot: 
            row = get_mid_env(self.Lx-1,self.psi,config)
            ftn = FermionTensorNetwork([env_bot,row],virtual=False)
        if self.compute_top:
            row = get_mid_env(0,self.psi,config)
            ftn = FermionTensorNetwork([row,env_top],virtual=False)
        try:
            unsigned_cx = ftn.contract()
        except (ValueError,IndexError):
            unsigned_cx = 0.
        self.store[config] = unsigned_cx
        return unsigned_cx
    def amplitude(self,config):
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def _amplitude(self,fpeps,config):
        for ix,ci in reversed(list(enumerate(config))):
            i,j = self.flat2site(ix)
            fpeps.add_tensor(get_bra_tsr(fpeps,ci,i,j))
        try:
            cx = fpeps.contract()
        except (ValueError,IndexError):
            cx = 0.
        return cx 
    def get_plq_from_benvs(self,config,x_bsz,y_bsz):
        #if self.compute_bot and self.compute_top:
        #    raise ValueError
        first_col = self.psi.col_tag(0)
        imax = self.Lx-x_bsz
        jmax = self.Ly-y_bsz
        plq = dict()
        for i in range(imax+1):
            ls = []
            if i>0:
                ls.append(self.cache_bot[config[:i*self.Ly]])
            ls += [get_mid_env(i+ix,self.psi,config) for ix in range(x_bsz)]
            if i<imax:
                ls.append(self.cache_top[config[(i+x_bsz)*self.Ly:]])
            ftn = FermionTensorNetwork(ls,virtual=False).view_like_(self.psi)
            ftn.reorder('col',inplace=True)
            lenvs = get_all_lenvs(ftn.copy(),jmax=jmax-1)
            renvs = get_all_renvs(ftn.copy(),jmin=y_bsz)
            for j in range(jmax+1): 
                tags = [ftn.col_tag(j+ix) for ix in range(y_bsz)]
                cols = [ftn.select(tags,which='any').copy()]
                try:
                    if j>0:
                        cols.insert(0,lenvs[j-1].copy())
                    if j<jmax:
                        cols.append(renvs[j+y_bsz].copy())
                    plq[(i,j),(x_bsz,y_bsz)] = \
                        FermionTensorNetwork(cols,virtual=True).view_like_(self.psi)
                except AttributeError: # lenv/renv is None
                    continue
        return plq
    def grad(self,config,plq=None):
        # currently not called
        raise NotImplementedError
        sign = self.compute_config_sign(config)
        if config in self.store_grad:
            unsigned_cx = self.store[config]
            vx = self.store_grad[config]
            return sign * unsigned_cx, vx
        if plq is None:
            self.get_all_benvs(config,x_bsz=1)
            plq = self.get_plq_from_benvs(config,x_bsz=1,y_bsz=1)
        unsigned_cx,vx,_ = self.get_grad_from_plqs(plq,config=config)
        return unsigned_cx * sign, vx
    def get_grad_from_plq(self,plq,config=None,compute_cx=True):
        # gradient
        vx = dict()
        cx = dict()
        for ((i0,j0),(x_bsz,y_bsz)),ftn_plq in plq.items():
            cx[i0,j0] = ftn_plq.copy().contract() if compute_cx else 1.
            for i in range(i0,i0+x_bsz):
                for j in range(j0,j0+y_bsz):
                    if (i,j) in vx:
                        continue
                    vx[i,j] = site_grad(ftn_plq.copy(),i,j) / cx[i0,j0] 
        vx = np.concatenate(psi2vecs(self.constructors,vx)) 
        unsigned_cx = sum(cx.values()) / len(cx)
        if config is not None:
            self.store[config] = unsigned_cx
            self.store_grad[config] = vx
        return unsigned_cx, vx, cx
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
    def extract_diagonal(self,matrix):
        start = 0
        for ix,(_,_,size,_) in enumerate(self.constructors):
            stop = start+size 
            matrix[start:stop,:start] = 0.
            matrix[start:stop,stop:] = 0.
            start = stop
        return matrix
    def maskdot(self,v1,v2,x,f=None):
        y = np.zeros_like(x)
        start = 0
        for ix,(_,_,size,_) in enumerate(self.constructors):
            stop = start+size 
            v2x = np.dot(v2[:,start:stop],x[start:stop])
            if f is not None:
                v2x *= f
            y[start:stop] = np.dot(v2x,v1[:,start:stop])
            start = stop
        return y
    def maskouter(self,v1,v2,x):
        y = np.zeros_like(x)
        start = 0
        for ix,(_,_,size,_) in enumerate(self.constructors):
            stop = start+size 
            y[start:stop] = v1[start:stop] * np.dot(v2[start:stop],x[start:stop])
            start = stop
        return y
####################################################################################
# ham class 
####################################################################################
def set_pepo_tsrs(): 
    vac = data_map[0]
    occ = data_map[3]
    h1 = data_map['h1']

    from pyblock3.algebra.fermion_encoding import get_state_map
    from pyblock3.algebra.fermion import eye 
    state_map = get_state_map(symmetry)
    bond_info = dict()
    for qlab,_,sh in state_map.values():
        if qlab not in bond_info:
            bond_info[qlab] = sh
    I1 = eye(bond_info,flat=flat)
    I2 = np.tensordot(I1,I1,axes=([],[]))
    P0 = np.tensordot(vac,vac.dagger,axes=([],[]))
    P1 = np.tensordot(occ,occ.dagger,axes=([],[]))

    ADD = np.tensordot(vac,np.tensordot(vac.dagger,vac.dagger,axes=([],[])),axes=([],[])) \
        + np.tensordot(occ,np.tensordot(occ.dagger,vac.dagger,axes=([],[])),axes=([],[])) \
        + np.tensordot(occ,np.tensordot(vac.dagger,occ.dagger,axes=([],[])),axes=([],[]))
    ADD.shape = (4,)*3
    this.pepo_map = {'I1':I1,'I2':I2,'P0':P0,'P1':P1,
                     'h1':np.transpose(h1,axes=(0,2,1,3)),
                     'ADD':ADD,'occ':occ,'vac':vac}
    return
def get_data(coeff):
    data = np.tensordot(pepo_map['P0'],pepo_map['I2'],axes=([],[])) \
         + np.tensordot(pepo_map['P1'],pepo_map['h1'],axes=([],[])) * coeff
    data.shape = (4,)*6
    return data
def get_mpo(coeffs):
    # add h1
    ls = [None] * len(coeffs)
    for ix,coeff in enumerate(coeffs):
        data = get_data(coeff)
        inds = f'v{ix}*',f'v{ix}',f'k{ix}*',f'k{ix}',f'k{ix+1}*',f'k{ix+1}'
        tags = f'L{ix}',f'L{ix+1}','h'
        ls[ix] = FermionTensor(data=data,inds=inds,tags=tags)
    mpo = FermionTensorNetwork(ls[::-1],virtual=True)
    # add ADD
    npair = len(coeffs)
    bix = [rand_uuid() for ix in range(npair)]
    for ix in range(1,npair):
        tags = f'L{ix}',f'L{ix+1}','a'
        inds = bix[ix]+'*',f'v{ix}*',bix[ix-1]+'*'
        mpo.add_tensor(FermionTensor(data=pepo_map['ADD'].copy(),inds=inds,tags=tags))
        inds = bix[ix-1],f'v{ix}',bix[ix]
        mpo.add_tensor(FermionTensor(data=pepo_map['ADD'].dagger,inds=inds,tags=tags))
    # reindex vir
    mpo['L0','L1','h'].reindex_({'v0':bix[0],'v0*':bix[0]+'*'}) 
    # reindex phys
    for ix in range(1,npair):
        mpo[f'L{ix-1}',f'L{ix}','h'].reindex_({f'k{ix}':f'k{ix}_'})
        mpo[f'L{ix}',f'L{ix+1}','h'].reindex_({f'k{ix}*':f'k{ix}_'})
    for ix in range(1,npair):
        mpo.contract_tags((f'L{ix}',f'L{ix+1}'),which='all',inplace=True)
    mpo[f'L{npair-1}',f'L{npair}'].reindex_({bix[-1]:'v',bix[-1]+'*':'v*'})

    # compress
    mpo.drop_tags(('h','a'))
    for ix in range(npair):
        tid = mpo[f'L{ix}',f'L{ix+1}'].get_fermion_info()[0]
        tsr = mpo._pop_tensor(tid,remove_from_fermion_space='end')
        rix = f'k{ix}*',f'k{ix}'
        if ix>0:
            rix = rix+(bix,)
        bix = rand_uuid()
        tl,tr = tensor_split(tsr,left_inds=None,right_inds=rix,bond_ind=bix,method='svd',get='tensors')
        tr.modify(tags=f'L{ix}')
        tl.modify(tags=(f'L{ix+1}',f'L{ix+2}'))
        mpo.add_tensor(tr,virtual=True)
        mpo.add_tensor(tl,virtual=True)
        mpo.contract_tags((f'L{ix+1}',f'L{ix+2}'),which='all',inplace=True)
    mpo[f'L{npair}',f'L{npair+1}'].drop_tags(f'L{npair+1}')
    return mpo
def get_comb(mpos):
    # add mpo
    pepo = FermionTensorNetwork([])
    L = mpos[0].num_tensors
    tag = f'L{L-1}'
    for ix1,mpo in enumerate(mpos):
        for ix2 in range(L):
            mpo[f'L{ix2}'].reindex_({f'k{ix2}':f'mpo{ix1}_k{ix2}',f'k{ix2}*':f'mpo{ix1}_k{ix2}*'})
        mpo[tag].reindex_({'v':f'v{ix1}','v*':f'v{ix1}*'})
        mpo.add_tag(f'mpo{ix1}')
        pepo.add_tensor_network(mpo,virtual=True)
    # add ADD
    nmpo = len(mpos)
    bix = [rand_uuid() for ix in range(nmpo)]
    pepo['mpo0',tag].reindex_({'v0':bix[0],'v0*':bix[0]+'*'})
    for ix in range(1,nmpo):
        tags = f'mpo{ix}',tag 
        inds = bix[ix]+'*',f'v{ix}*',bix[ix-1]+'*'
        pepo.add_tensor(FermionTensor(data=pepo_map['ADD'].copy(),inds=inds,tags=tags))
        inds = bix[ix-1],f'v{ix}',bix[ix]
        pepo.add_tensor(FermionTensor(data=pepo_map['ADD'].dagger,inds=inds,tags=tags))
    for ix in range(1,nmpo):
        pepo.contract_tags((f'mpo{ix}',tag),which='all',inplace=True)
    pepo[f'mpo{nmpo-1}',tag].reindex_({bix[-1]:'v',bix[-1]+'*':'v*'})

    # compress
    for ix in range(nmpo-1):
        tid = pepo[f'mpo{ix}',tag].get_fermion_info()[0]
        tsr = pepo._pop_tensor(tid,remove_from_fermion_space='end')
        rix = set(tsr.inds).difference({bix[ix],bix[ix]+'*'})
        tl,tr = tensor_split(tsr,left_inds=None,right_inds=rix,method='svd',get='tensors')
        tl.modify(tags=(f'mpo{ix+1}',tag))
        pepo.add_tensor(tr,virtual=True)
        pepo.add_tensor(tl,virtual=True)
        pepo.contract_tags((f'mpo{ix+1}',tag),which='all',inplace=True)
    return pepo
def comb2PEPO(comb,fpeps,comb_type):
    if comb_type=='row':
        def get_comb_tag(i,j):
            return f'mpo{i}',f'L{j}'
        def get_comb_ind(i,j):
            return f'mpo{i}_k{j}'
    else:
        def get_comb_tag(i,j):
            return f'mpo{j}',f'L{i}'
        def get_comb_ind(i,j):
            return f'mpo{j}_k{i}'
    for i in range(fpeps.Lx):
        for j in range(fpeps.Ly):
            tsr = comb[get_comb_tag(i,j)]
            ind_old,ind_new = get_comb_ind(i,j),fpeps.site_ind(i,j)
            tsr.reindex_({ind_old:ind_new,ind_old+'*':ind_new+'*'}) 
            tsr.modify(tags=(fpeps.row_tag(i),fpeps.col_tag(j),fpeps.site_tag(i,j)))
    comb.view_like_(fpeps)
    return comb
def combine(top,bot):
    Lx,Ly = bot.Lx,bot.Ly
    for i in range(Lx):
        for j in range(Ly): 
            pix = top.site_ind(i,j)
            top[i,j].reindex_({pix:pix+'_'})
            bot[i,j].reindex_({pix+'*':pix+'_'})
    top[Lx-1,Ly-1].reindex_({'v':'vt','v*':'vt*'})
    bot[Lx-1,Ly-1].reindex_({'v':'vb','v*':'vb*'})

    pepo = bot
    pepo.add_tensor_network(top,virtual=True)
    tags = pepo.site_tag(Lx-1,Ly-1)
    pepo.add_tensor(
        FermionTensor(data=pepo_map['ADD'].copy(),inds=('v*','vt*','vb*'),tags=tags),virtual=True) 
    pepo.add_tensor(
        FermionTensor(data=pepo_map['ADD'].dagger,inds=('vb','vt','v'),tags=tags),virtual=True) 
    for i in range(Lx):
        for j in range(Ly): 
            pepo.contract_tags(pepo.site_tag(i,j),inplace=True) 
    return pepo 
class Hubbard2D:
    def __init__(self,Lx,Ly,t,u):
        self.Lx,self.Ly = Lx,Ly
        self.t,self.u = t,u
    def initialize_pepo(self,fpeps):
        set_pepo_tsrs()

        # row mpos
        mpos = [get_mpo(-self.t*np.ones(self.Ly-1)) for i in range(self.Lx)]
        pepo_row = get_comb(mpos)
        pepo_row = comb2PEPO(pepo_row,fpeps,'row')
        # col mpos
        mpos = [get_mpo(-self.t*np.ones(self.Lx-1)) for j in range(self.Ly)]
        pepo_col = get_comb(mpos)
        pepo_col = comb2PEPO(pepo_col,fpeps,'col')
        
        pepo = combine(pepo_row,pepo_col)

        tags = pepo.site_tag(self.Lx-1,self.Ly-1)
        pepo.add_tensor(FermionTensor(data=data_map[3].dagger,inds=('v*',),tags=tags),virtual=True)
        pepo.add_tensor(FermionTensor(data=data_map[3].copy(),inds=('v',),tags=tags),virtual=True)
        pepo.contract_tags(tags,inplace=True)
        self.pepo = pepo
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)        
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def hop_coeff(self,site1,site2):
        return -self.t
    def hop(self,ftn_plq,config,site1,site2,cx=None):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if i1==i2: # no hopping possible
            return 0.
        h1 = data_map['h1']
        cx = ftn_plq.copy().contract() if cx is None else cx
        kixs = [ftn_plq.site_ind(*site) for site in [site1,site2]]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip([site1,site2],kixs,bixs):
            ftn_plq[ftn_plq.site_tag(*site),'BRA'].reindex_({kix:bix})
        ftn_plq.add_tensor(FermionTensor(h1.copy(),inds=bixs+kixs,left_inds=bixs),virtual=True)
        try:
            return self.hop_coeff(site1,site2) * ftn_plq.contract() / cx 
        except (ValueError,IndexError):
            return 0.
    #def nnh(self,config,plq,inplace=True,cx=None):
    #    e = 0.
    #    x_bsz,y_bsz = 1,2
    #    # all horizontal bonds
    #    for i in range(self.Lx):
    #        for j in range(self.Ly-1):
    #            site1,site2 = (i,j),(i,j+1)
    #            if 
    #            ftn_plq = plq[(i,j),(x_bsz,y_bsz)] 
    #            if not inplace:
    #                ftn_plq = ftn_plq.copy()
    #            cx_ij = None if cx is None else cx[i,j]
    #            e += self.hop(ftn_plq,config,site1,site2,cx_ij)
    #    return e
    #def nnv(self,config,plq,inplace=True,cx=None):
    #    # all vertical bonds
    #    e = 0.
    #    x_bsz,y_bsz = 2,1
    #    for i in range(self.Lx-1):
    #        for j in range(self.Ly):
    #            site1,site2 = (i,j),(i+1,j)
    #            ftn_plq = plq[(i,j),(x_bsz,y_bsz)] 
    #            if not inplace:
    #                ftn_plq = ftn_plq.copy()
    #            cx_ij = None if cx is None else cx[i,j]
    #            e += self.hop(ftn_plq,config,site1,site2,cx_ij)
    #    return e
    def nn(self,config,plq,x_bsz,y_bsz,inplace=True,cx=None):
        # all vertical bonds
        e = 0.
        for i in range(self.Lx+1-x_bsz):
            for j in range(self.Ly+1-y_bsz):
                site1,site2 = (i,j),(i+x_bsz-1,j+y_bsz-1)
                if ((i,j),(x_bsz,y_bsz)) in plq:
                    ftn_plq = plq[(i,j),(x_bsz,y_bsz)] 
                    if not inplace:
                        ftn_plq = ftn_plq.copy()
                    cx_ij = None if cx is None else cx[i,j]
                    e += self.hop(ftn_plq,config,site1,site2,cx_ij)
        return e
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        amplitude_factory.get_all_benvs(config,x_bsz=1) 
        # form all (1,2),(2,1) plqs
        plq12 = amplitude_factory.get_plq_from_benvs(config,x_bsz=1,y_bsz=2)
        plq21 = amplitude_factory.get_plq_from_benvs(config,x_bsz=2,y_bsz=1)
        # get gradient form plq12
        vx = None
        if compute_v:
            unsigned_cx,vx,cx12 = amplitude_factory.get_grad_from_plq(plq12,config=config) 
        else:
            cx12 = {key:ftn_plq.copy().contract() for (key,_),ftn_plq in plq12.items()}
            unsigned_cx = sum(cx12.values()) / len(cx12)
        # get h/v bonds
        eh = self.nn(config,plq12,x_bsz=1,y_bsz=2,inplace=True,cx=cx12) 
        ev = self.nn(config,plq21,x_bsz=2,y_bsz=1,inplace=True,cx=None) 
        # onsite terms
        config = np.array(config,dtype=int)
        eu = self.u*len(config[config==3])

        ex = eh+ev+eu
        if not compute_Hv: 
            return unsigned_cx,ex,vx,None 
        sign = amplitude_factory.compute_config_sign(tuple(config)) 
        # make bra
        bra = self.pepo.copy()
        fpeps = amplitude_factory.psi.copy()
        for ix,ci in reversed(list(enumerate(config))):
            i,j = self.flat2site(ix)
            tsr = get_bra_tsr(fpeps,ci,i,j)
            tsr.reindex_({f'k{i},{j}':f'k{i},{j}*'})
            bra.add_tensor(tsr,virtual=True) 
        for i in range(self.Lx):
            for j in range(self.Ly):
                bra.contract_tags(bra.site_tag(i,j),inplace=True)
        norm = fpeps 
        norm.add_tensor_network(bra,virtual=True)
        plq = norm._compute_plaquette_environments_row_first(1,1,**amplitude_factory.contract_opts)
        for key in plq:
            plq[key].view_like_(fpeps)
        _,Hvx,_ = amplitude_factory.get_grad_from_plq(plq,compute_cx=False) # all hopping terms
        Hvx /= sign * unsigned_cx
        Hvx += eu * vx
        return unsigned_cx,ex,vx,Hvx
####################################################################################
# sampler 
####################################################################################
class ExchangeSampler2D:
    def __init__(self,Lx,Ly,nelec,seed=None,burn_in=0):
        self.Lx = Lx
        self.Ly = Ly
        self.nelec = nelec 
        self.nsite = self.Lx * self.Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.alternate = False # True if autodiff else False
    def initialize(self,config,thresh=1e-10):
        # randomly choses the initial sweep direction
        self.sweep_row_dir = self.rng.choice([-1,1]) 
        # setup to compute all opposite envs for initial sweep
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 
        self.px = self.amplitude_factory.prob(config)
        self.config = config
        # force to initialize with a better config
        #print(self.px)
        #exit()
        if self.px < thresh:
            raise ValueError 
    def preprocess(self,config):
        self._burn_in(config)
    def _burn_in(self,config,batchsize=None):
        batchsize = self.burn_in if batchsize is None else batchsize
        self.initialize(config)
        if batchsize==0:
            return
        t0 = time.time()
        _alternate = self.alternate 
        self.alternate = True # always do alternate sweep in burn in 
        for n in range(batchsize):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
        self.alternate = _alternate
        #print(f'\tRANK={RANK},burn in time={time.time()-t0},namps={len(self.amplitude_factory.store)}')
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def new_pair(self,i1,i2):
        if SYMMETRY=='u11':
            return self.new_pair_u11(i1,i2)
        else:
            raise NotImplementedError
    def new_pair_u11(self,i1,i2):
        n = abs(pn_map[i1]-pn_map[i2])
        if n==1:
            i1_new,i2_new = i2,i1
        else:
            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
            i1_new,i2_new = self.rng.choice(choices)
        return i1_new,i2_new 
    def new_pair_u1(self,i1,i2):
        return
    def get_pairs(self,i,j):
        bonds_map = {'l':((i,j),(i+1,j)),
                     'd':((i,j),(i,j+1)),
                     'r':((i,j+1),(i+1,j+1)),
                     'u':((i+1,j),(i+1,j+1)),
                     'x':((i,j),(i+1,j+1)),
                     'y':((i,j+1),(i+1,j))}
        bonds = []
        order = 'ldru' 
        for key in order:
            bonds.append(bonds_map[key])
        return bonds
    def update_plq(self,i,j,cols,ftn,saved_rows):
        ftn_plq = FermionTensorNetwork(cols,virtual=True).view_like_(ftn)
        pairs = self.get_pairs(i,j) 
        for site1,site2 in pairs:
            ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
            i1,i2 = self.config[ix1],self.config[ix2]
            if i1==i2: # continue
                #print(i,j,site1,site2,ix1,ix2,'pass')
                continue
            i1_new,i2_new = self.new_pair(i1,i2)
            ftn_pair = replace_sites(ftn_plq.copy(),(site1,site2),(i1_new,i2_new)) 
            try:
                py = ftn_pair.contract()**2
            except (ValueError,IndexError):
                py = 0.
            # test
            #config_ = self.config.copy()
            #config_[ix1] = i1_new
            #config_[ix2] = i2_new
            #fpeps_ = self.amplitude_factory.psi.copy()
            #for i_ in range(self.Lx):
            #    for j_ in range(self.Ly):
            #        fpeps_.add_tensor(get_bra_tsr(fpeps_,config_[self.flatten(i_,j_)],i_,j_))
            #try:
            #    py_ = fpeps_.contract()**2
            #except (ValueError,IndexError):
            #    py_ = 0.
            #print(i,j,site1,site2,ix1,ix2,i1_new,i2_new,self.config)
            #print(py,py_)
            #if np.fabs(py-py_)>PRECISION:
            #    raise ValueError
            # test
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept, update px & config & env_m
                #print('acc')
                self.px = py
                self.config[ix1] = i1_new
                self.config[ix2] = i2_new
                ftn_plq = replace_sites(ftn_plq,(site1,site2),(i1_new,i2_new))
                ftn = replace_sites(ftn,(site1,site2),(i1_new,i2_new))
                saved_rows = replace_sites(saved_rows,(site1,site2),(i1_new,i2_new))
        return ftn,saved_rows
    def sweep_col_forward(self,i,rows):
        self.config = list(self.config)
        ftn = FermionTensorNetwork(rows,virtual=False).view_like_(rows[0])
        saved_rows = ftn.copy()
        ftn.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        renvs = get_all_renvs(ftn.copy(),jmin=2)
        first_col = ftn.col_tag(0)
        for j in range(self.Ly-1): # 0,...,Ly-2
            tags = first_col,ftn.col_tag(j),ftn.col_tag(j+1)
            cols = [ftn.select(tags,which='any').copy()]
            if j<self.Ly-2:
                cols.append(renvs[j+2])
            ftn,saved_rows = self.update_plq(i,j,cols,ftn,saved_rows) 
            # update new lenv
            if j<self.Ly-2:
                tags = first_col if j==0 else (first_col,ftn.col_tag(j))
                ftn ^= first_col,ftn.col_tag(j) 
        self.config = tuple(self.config)
        return saved_rows
    def sweep_col_backward(self,i,rows):
        self.config = list(self.config)
        ftn = FermionTensorNetwork(rows,virtual=False).view_like_(rows[0])
        saved_rows = ftn.copy()
        ftn.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        lenvs = get_all_lenvs(ftn.copy(),jmax=self.Ly-3)
        last_col = ftn.col_tag(self.Ly-1)
        for j in range(self.Ly-1,0,-1): # Ly-1,...,1
            cols = []
            if j>1: 
                cols.append(lenvs[j-2])
            tags = ftn.col_tag(j-1),ftn.col_tag(j),last_col
            cols.append(ftn.select(tags,which='any').copy())
            ftn,saved_rows = self.update_plq(i,j-1,cols,ftn,saved_rows) 
            # update new renv
            if j>1:
                tags = last_col if j==ftn.Ly-1 else (ftn.col_tag(j),last_col)
                ftn ^= tags 
        self.config = tuple(self.config)
        return saved_rows
    def sweep_row_forward(self):
        fpeps = self.amplitude_factory.psi
        compress_opts = self.amplitude_factory.contract_opts
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        # can assume to have all opposite envs
        #get_all_top_envs(fpeps,self.config,cache_top,imin=2,**compress_opts)
        sweep_col = self.sweep_col_forward if self.sweep_col_dir == 1 else\
                    self.sweep_col_backward

        env_bot = None 
        row1 = get_mid_env(0,fpeps,self.config)
        for i in range(self.Lx-1):
            rows = []
            if i>0:
                rows.append(env_bot)
            row2 = get_mid_env(i+1,fpeps,self.config)
            rows += [row1,row2]
            if i<self.Lx-2:
                rows.append(cache_top[self.config[(i+2)*self.Ly:]]) 
            saved_rows = sweep_col(i,rows)
            row1_new = saved_rows.select(fpeps.row_tag(i),virtual=True)
            row2_new = saved_rows.select(fpeps.row_tag(i+1),virtual=True)
            # update new env_h
            env_bot = get_bot_env(i,row1_new,env_bot,self.config,cache_bot,**compress_opts)
            row1 = row2_new
    def sweep_row_backward(self):
        fpeps = self.amplitude_factory.psi
        compress_opts = self.amplitude_factory.contract_opts
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        # can assume to have all opposite envs
        #get_all_bot_envs(fpeps,self.config,cache_bot,imax=self.Lx-3,**compress_opts)
        sweep_col = self.sweep_col_forward if self.sweep_col_dir == 1 else\
                    self.sweep_col_backward

        env_top = None 
        row1 = get_mid_env(self.Lx-1,fpeps,self.config)
        for i in range(self.Lx-1,0,-1):
            rows = []
            if i>1:
                rows.append(cache_bot[self.config[:(i-1)*self.Ly]])
            row2 = get_mid_env(i-1,fpeps,self.config)
            rows += [row2,row1]
            if i<self.Lx-1:
                rows.append(env_top) 
            saved_rows = sweep_col(i-1,rows)
            row1_new = saved_rows.select(fpeps.row_tag(i),virtual=True)
            row2_new = saved_rows.select(fpeps.row_tag(i-1),virtual=True)
            # update new env_h
            env_top = get_top_env(i,row1_new,env_top,tuple(self.config),cache_top,**compress_opts)
            row1 = row2_new
    def sample(self):
        self.sweep_col_dir = -1 # randomly choses the col sweep direction
        if self.sweep_row_dir == 1:
            self.sweep_row_forward()
        else:
            self.sweep_row_backward()
        # setup to compute all opposite env for gradient
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 

        if self.alternate: # for burn in 
            self.sweep_row_dir *= -1
        else: # actual sampling 
            self.sweep_row_dir = self.rng.choice([-1,1]) 
        return self.config,self.px
class DenseSampler2D:
    def __init__(self,Lx,Ly,nelec,exact=False,seed=None,thresh=1e-14):
        self.Lx = Lx
        self.Ly = Ly
        self.nelec = nelec 
        self.nsite = self.Lx * self.Ly

        self.all_configs = self.get_all_configs()
        self.ntotal = len(self.all_configs)
        self.flat_indexes = list(range(self.ntotal))
        self.p = None

        batchsize,remain = self.ntotal//SIZE,self.ntotal%SIZE
        self.count = np.array([batchsize]*SIZE)
        if remain > 0:
            self.count[-remain:] += 1
        self.disp = np.concatenate([np.array([0]),np.cumsum(self.count[:-1])])
        self.start = self.disp[RANK]
        self.stop = self.start + self.count[RANK]

        self.rng = np.random.default_rng(seed)
        self.burn_in = 0
        self.dense = True
        self.exact = exact 
        self.amplitude_factory = None
        self.thresh = thresh
    def initialize(self,config=None):
        pass
    def preprocess(self):
        self.compute_dense_prob()
    def compute_dense_prob(self):
        t0 = time.time()
        ptotal = np.zeros(self.ntotal)
        start,stop = self.start,self.stop
        configs = self.all_configs[start:stop]

        plocal = [] 
        self.amplitude_factory.update_scheme(1)
        for config in configs:
            plocal.append(self.amplitude_factory.prob(config))
        plocal = np.array(plocal)
         
        COMM.Allgatherv(plocal,[ptotal,self.count,self.disp,MPI.DOUBLE])
        nonzeros = []
        for ix,px in enumerate(ptotal):
            if px > self.thresh:
                nonzeros.append(ix) 
        n = np.sum(ptotal)
        ptotal /= n 
        self.p = ptotal
        if RANK==SIZE-1:
            print('\tdense amplitude time=',time.time()-t0)

        ntotal = len(nonzeros)
        batchsize,remain = ntotal//SIZE,ntotal%SIZE
        L = SIZE-remain
        if RANK<L:
            start = RANK*batchsize
            stop = start+batchsize
        else:
            start = (batchsize+1)*RANK-L
            stop = start+batchsize+1
        self.nonzeros = nonzeros[start:stop]
        self.amplitude_factory.update_scheme(0)
    def get_all_configs(self):
        if SYMMETRY=='u1':
            return self.get_all_configs_u1()
        elif SYMMETRY=='u11':
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
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        config = self.all_configs[flat_idx]
        omega = self.p[flat_idx]
        return config,omega

