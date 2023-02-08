import time,itertools
import numpy as np

from pyblock3.algebra.fermion_ops import vaccum,creation,bonded_vaccum
from ..tensor_2d import PEPS
from .utils import psi2vecs
from .fermion_core import FermionTensor, FermionTensorNetwork, tensor_contract
from .fermion_2d import FPEPS,FermionTensorNetwork2D
from .block_interface import Constructor

symmetry = 'u1' # tsr symmetry
SYMMETRY = 'u11' # sampler symmetry
flat = True
PRECISION = 1e-10

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
def flatten(i,j,Lx,Ly): # flattern site to row order
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
        cons = Constructor.from_bond_infos(bond_infos,data.pattern)
        dq = data.dq
        size = cons.vector_size(dq)
        ix = flatten(i,j,fpeps.Lx,fpeps.Ly)
        constructors[ix] = cons,dq,size,(i,j)
    return constructors

####################################################################################
# amplitude fxns 
####################################################################################
def get_bra_tsr(fpeps,ci,site):
    inds = fpeps.site_ind(*site),
    tags = fpeps.site_tag(*site)
    data = state_map[ci].dagger
    data.shape = fpeps[tags].shape[-1],
    return FermionTensor(data=data,inds=inds,tags=tags)
def update_cache_mid(fpeps,config,i,cache_mid):
    L = fpeps.Ly
    tag = fpeps.row_tag
    start,stop = i*L,(i+1)*L
    config_row = config[start:stop] 
    if (i,config_row) in cache_mid:
        row = cache_mid[i,config_row] 
        return row,cache_mid

    row = fpeps.select(tag(i)).copy()
    for ix in range(stop-1,start-1,-1):
        site = flat2site(ix,fpeps.Lx,fpeps.Ly)
        row.add_tensor(get_bra_tsr(fpeps,config[ix],site))
    try:
        for ix in range(stop-1,start-1,-1):
            site = flat2site(ix,fpeps.Lx,fpeps.Ly)
            row.contract_tags(fpeps.site_tag(*site),inplace=True)
        cache_mid[i,config_row] = row
    except (ValueError,IndexError):
        row = None 
        cache_mid[i,config_row] = None
    return row,cache_mid
def update_cache_head(fpeps,config,cache_head,cache_mid,imin,imax,**compress_opts):
    # compute all bottom envs
    ftn = FermionTensorNetwork([]) 
    L = fpeps.Ly
    imax_ = fpeps.Lx-1
    for i in range(imin,imax+1):
        key = config[:(i+1)*L]
        if key in cache_head:
            ftn = cache_head[key]
            continue
        if ftn is None:
            cache_head[key] = None 
            continue
        row,cache_mid = update_cache_mid(fpeps,config,i,cache_mid)
        if row is None:
            cache_head[key] = None 
            continue
        ftn = FermionTensorNetwork([ftn,row]).view_as_(FermionTensorNetwork2D,like=row)
        if i>0:
            try:
                ftn.contract_boundary_from_bottom_(
                    xrange=(i-1,i),yrange=(0,fpeps.Ly-1),**compress_opts)
            except (ValueError,IndexError):
                ftn = None
                cache_head[key] = None 
                continue
        if i<imax_:
            cache_head[key] = ftn.copy()
    return cache_head,cache_mid
def update_cache_tail(fpeps,config,cache_tail,cache_mid,imin,imax,**compress_opts):
    # compute all top envs
    ftn = FermionTensorNetwork([]) 
    L = fpeps.Ly 
    imax_ = fpeps.Lx-1 
    for i in range(imax,imin-1,-1):
        key = config[i*L:]
        if key in cache_tail:
            ftn = cache_tail[key]
            continue
        if ftn is None:
            cache_tail[key] = None 
            continue
        row,cache_mid = update_cache_mid(fpeps,config,i,cache_mid)
        if row is None:
            cache_tail[key] = None 
            continue
        ftn = FermionTensorNetwork([row,ftn]).view_as_(FermionTensorNetwork2D,like=row)
        if i<imax_:
            try:
                ftn.contract_boundary_from_top_(
                    xrange=(i,i+1),yrange=(0,fpeps.Ly-1),**compress_opts)
            except (ValueError,IndexError):
                ftn = None
                cache_tail[key] = None 
                continue
        if i>0:
            cache_tail[key] = ftn.copy()
    return cache_tail,cache_mid
def compute_amplitude_2d(fpeps,config,split,cache_head,cache_mid,cache_tail,**compress_opts):
    L,imax = fpeps.Ly,fpeps.Lx-1
    split = split - 1 if split == imax else split
    cache_head,cache_mid = update_cache_head(
            fpeps,config,cache_head,cache_mid,0,split,**compress_opts)
    cache_tail,cache_mid = update_cache_tail(
            fpeps,config,cache_tail,cache_mid,split+1,imax,**compress_opts)
    ftn_head = cache_head[config[:(split+1)*L]]
    ftn_tail = cache_tail[config[(split+1)*L:]]
    if ftn_head is None or ftn_tail is None:
        amp = 0.
    else:
        try:
            amp = FermionTensorNetwork([ftn_head,ftn_tail]).contract()
        except (ValueError,IndexError):
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
def compute_grad(fpeps,config,cache_head,cache_mid,cache_tail,
                 **compress_opts):
    # fpeps,config in same order
    L,imax = fpeps.Ly,fpeps.Lx-1
    row_tag = fpeps.col_tag
    for i in range(imax+1):
        _,cache_mid = update_cache_mid(fpeps,config,i,cache_mid)
    cache_head,_ = update_cache_head(fpeps,config,cache_head,cache_mid,0,imax,**compress_opts)
    cache_tail,_ = update_cache_tail(fpeps,config,cache_tail,cache_mid,0,imax,**compress_opts)
    # compute grad
    amps = []
    grad = dict()
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
        ftn.reorder('col',inplace=True)
        envs = {('mid',j):ftn.select(row_tag(j)).copy() for j in range(L)}
        envs = compute_3col_envs(ftn.copy(),'col',1,0,L-1,envs)
        envs = compute_3col_envs(ftn.copy(),'col',-1,0,L-1,envs)
        for j in range(L):
            if envs['head',j] is None:
                continue
            if envs['tail',j] is None:
                continue
            ftn_ij = FermionTensorNetwork([envs[side,j] \
                                           for side in ['head','mid','tail']])

            ix = i*L+j
            site = flat2site(ix,fpeps.Lx,fpeps.Ly) 
            tag = fpeps.site_tag(*site)
            tid = ftn_ij[tag].get_fermion_info()[0]
            Tv = ftn_ij._pop_tensor(tid,remove_from_fermion_space='end')

            try:
                target = fpeps[tag]
                #print(target.inds[:-1])
                g = ftn_ij.contract()
                #print(g.inds)
                amps.append(FermionTensorNetwork([g,Tv]).contract())
                
                v = get_bra_tsr(fpeps,config[ix],site)
                if pn_map[config[ix]] % 2 == 1:
                    tmp = FermionTensorNetwork([target,v]).contract(
                                 output_inds=target.inds[:-1])
                    size = np.product(np.array(tmp.shape))
                    if (Tv.data-tmp.data).norm()/size<PRECISION:
                        pass
                    elif (Tv.data+tmp.data).norm()/size<PRECISION:
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
    amp = sum(amps)/len(amps) if len(amps)>0 else 0.
    return amp,grad,cache_head,cache_mid,cache_tail
def compute_fpeps_parity(fs,start,stop):
    if start==stop:
        return 0
    tids = [fs.get_tid_from_site(site) for site in range(start,stop)]
    tsrs = [fs.tensor_order[tid][0] for tid in tids] 
    return sum([tsr.parity for tsr in tsrs]) % 2
def compute_fpeps_parities(fpeps,L,imax):
    parity = [None] * (imax+1) 
    fs = fpeps.fermion_space
    for i in range(imax+1):
        start,stop = i*L,(i+1)*L
        parity[i] = compute_fpeps_parity(fs,start,stop)
    return parity
def compute_config_parity(config,parity_fpeps,L,imax):
    parity_config = [None] * (imax+1)
    for i in range(imax+1):
        parity_config[i] = sum([pn_map[ci] for ci in config[i*L:(i+1)*L]]) % 2
    parity = 0
    for i in range(imax-1,-1,-1):
        parity_ = (sum(parity_fpeps[i+1:]) + sum(parity_config[i+1:])) % 2
        parity += parity_config[i] * parity_
    return parity % 2
class AmplitudeFactory2D:
    def __init__(self,psi,**contract_opts):
        self.contract_opts=contract_opts
        self._set_psi(psi)
    def _set_psi(self,psi):
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.constructors = get_constructors_2d(psi)
        self.psi = psi.reorder(direction='row',inplace=True)
        self.parity = compute_fpeps_parities(self.psi,self.Ly,self.Lx-1)

        self.store = dict()
        self.store_grad = dict()

        self.cache_bottom = dict()
        self.cache_top = dict()
        self.cache_mid = dict()
        return
    def amplitude(self, info):
        """Get the amplitude of ``config``, either from the cache or by
        computing it.
        """
        config,split = info 
        if config in self.store:
            return self.store[config]

        amp,self.cache_bottom,self.cache_mid,self.cache_top = \
            compute_amplitude_2d(self.psi,config,split,
            self.cache_bottom,self.cache_mid,self.cache_top,**self.contract_opts) 

        p = compute_config_parity(config,self.parity,self.Ly,self.Lx-1)
        amp *= (-1)**p
        self.store[config] = amp 
        return amp 
    def _amplitude(self,info):
        config,split = info
        if config in self.store:
            return self.store[config]

        fpeps = self.psi.copy()
        for ix,i in reversed(list(enumerate(config))):
            site = flat2site(ix,self.Lx,self.Ly)
            fpeps.add_tensor(get_bra_tsr(fpeps,i,site))
        try:
            amp = fpeps.contract()
        except (ValueError,IndexError):
            amp = 0.
        self.store[config] = amp 
        return amp
    def grad(self,config):
        if config in self.store_grad:
            return self.store[config],self.store_grad[config]

        amp,grad,self.cache_bottom,self.cache_mid,self.cache_top = compute_grad(self.psi,config,
            self.cache_bottom,self.cache_mid,self.cache_top,**self.contract_opts) 
        grad = np.concatenate(psi2vecs(self.constructors,grad)) 

        p = compute_config_parity(config,self.parity,self.Ly,self.Lx-1)
        amp *= (-1)**p
        grad *= (-1)**p
        self.store[config] = amp 
        self.store_grad[config] = grad
        return amp,grad 
    def prob(self, info):
        """Calculate the probability of a configuration.
        """
        coeff = self.amplitude(info)
        return coeff**2
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

        for i, j in itertools.product(range(self.Lx), range(self.Ly-1)):
            ix1,ix2 = self.flatten(i,j), self.flatten(i,j+1)
            configs_ = hop(ix1,ix2,config)
            for config_,sign_ in configs_:
                configs.append((config_,i))
                coeffs.append(-self.t*sign_) 

        for i, j in itertools.product(range(self.Lx-1), range(self.Ly)):
            ix1,ix2 = self.flatten(i,j), self.flatten(i+1,j)
            sign = (-1)**sum([pn_map[ci] for ci in config[ix1+1:ix2]])
            configs_ = hop(ix1,ix2,config)
            for config_,sign_ in configs_:
                configs.append((config_,i))
                coeffs.append(-self.t*sign*sign_) 
   
        configs.append(None)
        config = np.array(config,dtype=int)
        coeffs.append(self.u*len(config[config==3]))
        return configs,coeffs
class PN2D(Hubbard2D):
    def __init__(self,Lx,Ly,spin='a'):
        super().__init__(Lx,Ly,0.,0.)
        self.spin = spin
    def config_coupling(self,config):
        config = np.array(config,dtype=int)

        na = len(config[config==1])
        nb = len(config[config==2])
        ndb = len(config[config==3])
        if self.spin=='a':
            coeff = na + ndb 
        elif self.spin=='b':
            coeff = nb + ndb
        else: # total pn
            coeff = na + nb + ndb
        return [None],[coeff]
####################################################################################
# sampler 
####################################################################################
class ExchangeSampler2D:
    def __init__(self,sampler_opts):
        self.Lx = sampler_opts['Lx']
        self.Ly = sampler_opts['Ly']
        self.nelec = sampler_opts['nelec'] 
        self.nsite = self.Lx * self.Ly

        self.sweep = sampler_opts.get('sweep',True) 
        self.blocks = [(i,j) for i in range(self.Lx-1) for j in range(self.Ly-1)] 

        seed = sampler_opts.get('seed',None)
        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = sampler_opts.get('burn_in',0)
    def _set_amplitude_factory(self,amplitude_factory,config):
        self.amplitude_factory = amplitude_factory
        self.prob_fn = self.amplitude_factory.prob
        if config is None:
            self.config = self.get_rand_config()
        else:
            assert len(config)==self.nsite
            self.config = config
        self.px = self.prob_fn((self.config,0))
        while self.px < PRECISION:
            self.config,self.px = self.sample()
    def flatten(self,i,j):
        return flatten(i,j,self.Lx,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def get_rand_config(self):
        if SYMMETRY=='u1':
            return self.get_rand_config_u1()
        elif SYMMETRY=='u11':
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
    def sample_pair(self,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if i1==i2:
            return 
        i1_new,i2_new = self.new_pair(i1,i2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        config_new = tuple(config_new)
        info = config_new,site1[0]
        py = self.prob_fn(info)
        acceptance = 1. if self.px < PRECISION else py / self.px 
        if self.rng.uniform() < acceptance:
            self.config = config_new
            self.px = py
    def sweep_row(self):
        for i in range(self.Lx):
            for j in range(self.Ly-1):
                site1,site2 = (i,j),(i,j+1)
                self.sample_pair(site1,site2)
    def sweep_col(self):
        for j in range(self.Ly):
            for i in range(self.Lx-1):
                site1,site2 = (i,j),(i+1,j)
                self.sample_pair(site1,site2)
    def sample_sweep(self):
        first = self.rng.integers(low=0,high=1,endpoint=True)
        if first==0:
            self.sweep_row()
        #    self.sweep_col()
        else:
            self.sweep_col()
        #    self.sweep_row()
    def sample_rand(self):
        blocks = self.rng.permutation(self.blocks)
        for i,j in blocks:
            site1 = i,j
            site2 = [(i,j+1),(i+1,j)][self.rgn.integers(low=0,high=1,endpoint=True)]
            self.sample_pair(site1,site2)
    def sample(self):
        if self.sweep:
            self.sample_sweep()
        else:
            self.sample_rand()
        return self.config,self.px
