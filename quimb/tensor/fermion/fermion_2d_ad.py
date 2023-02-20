import time,itertools,sys
import numpy as np

from pyblock3.algebra.fermion_ops import vaccum,creation,bonded_vaccum,H1
from ..tensor_2d import PEPS
from .utils import psi2vecs
from .fermion_core import FermionTensor, FermionTensorNetwork, tensor_contract
from .fermion_2d import FPEPS,FermionTensorNetwork2D
from .block_interface import Constructor
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

SYMMETRY = 'u11' # sampler symmetry
flat = True
PRECISION = 1e-10

# set tensor symmetry
this = sys.modules[__name__]
def set_symmetry(symmetry):
    this.h1 = H1(symmetry=symmetry,flat=flat)
    this.cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    this.cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    this.vac = vaccum(n=1,symmetry=symmetry,flat=flat)
    this.occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
    this.occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
    this.occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
    this.state_map = [vac,occ_a,occ_b,occ_db]
pn_map = [0,1,1,2]
config_map = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}

from fermion_2d_vmc import (
    flatten,flat2site,
    get_constructors_2d,
    get_bra_tsr,
)
####################################################################################
# amplitude fxns 
####################################################################################
def get_bra_tsr(fpeps,ci,i,j):
    inds = fpeps.site_ind(i,j),
    tags = fpeps.site_tag(i,j),fpeps.row_tag(i),fpeps.col_tag(j),'BRA'
    data = state_map[ci].dagger
    return FermionTensor(data=data,inds=inds,tags=tags)
class AmplitudeFactory2D:
    def __init__(self,psi=None,x_bsz=2,y_bsz=2,**contract_opts):
        self.contract_opts=contract_opts
        self.x_bsz = x_bsz
        self.y_bsz = y_bsz
        if psi is not None:
            self._set_psi(psi)
    def _set_psi(self,psi):
        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.constructors = get_constructors_2d(psi)
        self.psi = psi.reorder(direction='row',inplace=True)
        self.psi.add_tag('KET')

        parity = []
        fs = self.psi.fermion_space
        for i in range(1,self.Lx): # only need parity of row 1,...,Lx-1
            start,stop = i*self.Ly,(i+1)*self.Ly
            parity.append(compute_fpeps_parity(fs,start,stop))
        self.parity_cum = np.cumsum(np.array(parity[::-1]))

        self.store = dict()
        self.store_grad = dict()

        self.cache_bot = dict()
        self.cache_top = dict()
        self.compute_bot = True
        self.compute_top = True
        return
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
    def compute_config_parity(self,config):
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum([pn_map[ci] for ci in config[i*self.Ly:(i+1)*self.Ly]]) % 2
        parity = np.array(parity[::-1])
        parity_cum = np.cumsum(parity[:-1])
        parity_cum += self.parity_cum 
        return np.dot(parity[1:],parity_cum) % 2
    def unsigned_amplitude(self,config):
        # should only be used to:
        # 1. compute dense probs
        # 2. initialize MH sampler
        if config in self.store:
            return self.store[config]
        if self.compute_bot: 
            row = get_mid_env(self.Lx-1,self.psi,config)
            env_bot = get_all_bot_envs(self.psi,config,self.cache_bot,imax=self.Lx-2,
                                       **self.contract_opts)
            if env_bot is None:
                self.unsigned_cx = 0.
                return self.unsigned_cx
            ftn = FermionTensorNetwork([env_bot,row],virtual=False)
        if self.compute_top:
            row = get_mid_env(0,self.psi,config)
            env_top = get_all_top_envs(self.psi,config,self.cache_top,imin=1,**self.contract_opts)
            ftn = FermionTensorNetwork([row,env_top],virtual=False)
            if env_top is None:
                self.unsigned_cx = 0.
                return self.unsigned_cx
        try:
            self.unsigned_cx = ftn.contract()
        except (ValueError,IndexError):
            self.unsigned_cx = 0.
        return self.unsigned_cx
    def amplitude(self,config):
        self.unsigned_ampplitude(config)
        sign = (-1) ** self.compute_config_parity(config)
        cx = self.unsigned_cx * sign 
        self.store[config] = cx 
        return cx
    def _amplitude(self,fpeps,config):
        for ix,ci in reversed(list(enumerate(config))):
            i,j = flat2site(ix,self.Lx,self.Ly)
            fpeps.add_tensor(get_bra_tsr(fpeps,ci,i,j))
        try:
            cx = fpeps.contract()
        except (ValueError,IndexError):
            cx = 0.
        return cx 
    def get_all_plqs(self,config):
        #if self.compute_bot and self.compute_top:
        #    raise ValueError
        if self.compute_bot: 
            get_all_bot_envs(self.psi,config,self.cache_bot,imax=self.Lx-1-self.x_bsz,
                             **self.contract_opts)
        if self.compute_top:
            get_all_top_envs(self.psi,config,self.cache_top,imin=self.x_bsz,
                             **self.contract_opts)
        first_col = self.psi.col_tag(0)

        imax = self.Lx-self.x_bsz
        jmax = self.Ly-self.y_bsz
        plq = dict()
        for i in range(imax+1):
            ls = []
            if i>0:
                ls.append(self.cache_bot[config[:i*self.Ly]])
            ls += [get_mid_env(i+ix,self.psi,config) for ix in range(self.x_bsz)]
            if i<imax:
                ls.append(self.cache_top[config[(i+self.x_bsz)*self.Ly:]])
            ftn = FermionTensorNetwork(ls,virtual=False).view_like_(self.psi)
            ftn.reorder('col',inplace=True)
            renvs = get_all_renvs(ftn.copy(),jmin=self.y_bsz)
            for j in range(jmax+1): 
                tags = [first_col]+[ftn.col_tag(j+ix) for ix in range(self.y_bsz)]
                cols = [ftn.select(tags,which='any').copy()]
                if j<jmax:
                    cols.append(renvs[j+self.y_bsz].copy())
                plq[i,j] = FermionTensorNetwork(cols,virtual=True).view_like_(self.psi) 
                if j<jmax: 
                    tags = first_col if j==0 else (first_col,ftn.col_tag(j))
                    ftn ^= tags 
        return plq
    def grad(self,config):
        if config in self.store_grad:
            return self.store[config],self.store_grad[config]

        plq = self.get_all_plqs(config)
        # amplitude
        ftn_plq = plq[0,0].copy()
        try:
            self.unsigned_cx = ftn_plq.contract()
        except (IndexError,ValueError):
            self.unsigned_cx = 0.
        # gradient
        gx = dict()
        for i0,j0 in plq:
            ftn_plq = plq[i0,j0]
            for i in range(i0,i0+self.x_bsz):
                for j in range(j0,j0+self.y_bsz):
                    if (i,j) in gx:
                        continue
                    gx[i,j] = site_grad(ftn_plq.copy(),i,j) 
        self.plq = plq
        self.unsigned_gx = np.concatenate(psi2vecs(self.constructors,gx)) 

        sign = (-1) ** self.compute_config_parity(config)
        cx = self.unsigned_cx * sign 
        gx = self.unsigned_gx * sign
        self.store[config] = cx
        self.store_grad[config] = gx
        return cx,gx 
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
####################################################################################
# ham class 
####################################################################################
def hop(i1,i2):
    if i1==i2:
        return []
    n1,n2 = pn_map[i1],pn_map[i2]
    nsum,ndiff = n1+n2,abs(n1-n2)
    if ndiff==1:
        sign = 1 if nsum==1 else -1
        return [(i2,i1,sign)]
    if ndiff==2:
        return [(1,2,-1),(2,1,1)] 
    if ndiff==0:
        sign = i1-i2 
        return [(0,3,sign),(3,0,sign)]
class Hubbard2D:
    def __init__(self,Lx,Ly,t,u):
        self.Lx,self.Ly = Lx,Ly
        self.t,self.u = t,u
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)        
    def hop_coeff(self,site1,site2):
        return -self.t
    def hop(self,ftn_plq,config,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if i1==i2: # no hopping possible
            return 0.
        kixs = [ftn_plq.site_ind(*site) for site in [site1,site2]]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip([site1,site2],kixs,bixs):
            ftn_plq[ftn_plq.site_tag(*site),'BRA'].reindex_({kix:bix})
        ftn_plq.add_tensor(FermionTensor(h1.copy(),inds=bixs+kixs,left_inds=bixs),virtual=True)
        try:
            return self.hop_coeff(site1,site2) * ftn_plq.contract()
        except (ValueError,IndexError):
            return 0.
    def nn(self,config,amplitude_factory):
        plq = amplitude_factory.plq 
        cx = amplitude_factory.unsigned_cx
        e = 0.
        # all horizontal bonds
        for i in range(self.Lx):
            for j in range(self.Ly-1):
                site1,site2 = (i,j),(i,j+1)
                i0,j0 = i,j
                if i0==self.Lx-1:
                    i0 -= 1
                ftn_plq = plq[i0,j0]
                e += self.hop(ftn_plq.copy(),config,site1,site2)
        # all vertical bonds
        for i in range(self.Lx-1):
            for j in range(self.Ly):
                site1,site2 = (i,j),(i+1,j)
                i0,j0 = i,j
                if j0==self.Ly-1:
                    j0 -= 1
                ftn_plq = plq[i0,j0]
                e += self.hop(ftn_plq.copy(),config,site1,site2)
        return e/cx
    def compute_local_energy(self,config,amplitude_factory):
        e = self.nn(config,amplitude_factory) 
        # onsite terms
        config = np.array(config,dtype=int)
        e += self.u*len(config[config==3])
        return e 
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
#class ExchangeSampler2D:
#    def __init__(self,Lx,Ly,nelec,sweep=True,seed=None,burn_in=0):
#        self.Lx = Lx
#        self.Ly = Ly
#        self.nelec = nelec 
#        self.nsite = self.Lx * self.Ly
#
#        self.sweep = sweep 
#        self.blocks = [(i,j) for i in range(self.Lx-1) for j in range(self.Ly-1)] 
#
#        self.rng = np.random.default_rng(seed)
#        self.exact = False
#        self.dense = False
#        self.burn_in = burn_in 
#    def _set_amplitude_factory(self,amplitude_factory,config):
#        self.amplitude_factory = amplitude_factory
#        self.amplitude_factory.update_scheme(0)
#        self.prob_fn = self.amplitude_factory.prob
#        if config is None:
#            self.config = self.get_rand_config()
#        else:
#            assert len(config)==self.nsite
#            self.config = config
#        self.px = self.prob_fn(self.config)
#        #print(f'initialize nonvanishing config. Start at {self.config},{self.px}.')
#        cnt = 0
#        maxcnt = 5
#        while self.px < PRECISION:
#            self.config,self.px = self.sample()
#            cnt += 1
#            if cnt > maxcnt:
#                raise ValueError(f'failes to generate nonvanishing config. Final at {self.config},{self.px}.') 
#    def preprocess(self):
#        self._burn_in()
#    def _burn_in(self,batchsize=None):
#        batchsize = self.burn_in if batchsize is None else batchsize
#        if batchsize==0:
#            return
#        t0 = time.time()
#        for n in range(batchsize):
#            self.config,self.omega = self.sample()
#        if RANK==SIZE-1:
#            print('\tburn in time=',time.time()-t0)
#            print('\tsaved amplitudes=',len(self.amplitude_factory.store))
#        #print(f'\tRANK={RANK},burn in time={time.time()-t0},namps={len(self.amplitude_factory.store)}')
#    def flatten(self,i,j):
#        return flatten(i,j,self.Ly)
#    def flat2site(self,ix):
#        return flat2site(ix,self.Lx,self.Ly)
#    def get_rand_config(self):
#        if SYMMETRY=='u1':
#            return self.get_rand_config_u1()
#        elif SYMMETRY=='u11':
#            return self.get_rand_config_u11()
#        else:
#            raise NotImplementedError
#    def get_rand_config_u11(self):
#        assert isinstance(self.nelec,tuple)
#        config = np.zeros((self.nsite,2),dtype=int)
#        sites = np.array(range(self.nsite),dtype=int)
#        for spin in (0,1):
#            occ = self.rng.choice(sites,size=self.nelec[spin],
#                                  replace=False,shuffle=False)
#            for ix in occ:
#                config[ix,spin] = 1
#        config = [config_map[tuple(config[i,:])] for i in range(self.nsite)]
#        return tuple(config)
#    def get_rand_config_u1(self):
#        if isinstance(self.nelec,tuple):
#            self.nelec = sum(self.nelec)
#        sites = np.array(range(self.nsite),dtype=int)
#        occ = self.rng.choice(sites,size=self.nelec,replace=False,shuffle=False)
#        for ix in occ:
#            config[ix] = 1
#        configa,configb = config[:self.nsite],config[self.nsite:]
#        config = [config_map[configa[i],configb[i]] for i in range(self.nsite)]
#        return tuple(config)
#    def new_pair(self,i1,i2):
#        if SYMMETRY=='u11':
#            return self.new_pair_u11(i1,i2)
#        else:
#            raise NotImplementedError
#    def new_pair_u11(self,i1,i2):
#        n = abs(pn_map[i1]-pn_map[i2])
#        if n==1:
#            i1_new,i2_new = i2,i1
#        else:
#            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
#            i1_new,i2_new = self.rng.choice(choices)
#        return i1_new,i2_new 
#    def new_pair_u1(self,i1,i2):
#        return
#    def sample_pair(self,site1,site2):
#        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
#        i1,i2 = self.config[ix1],self.config[ix2]
#        if i1==i2:
#            return 
#        i1_new,i2_new = self.new_pair(i1,i2)
#        config_new = list(self.config)
#        config_new[ix1] = i1_new
#        config_new[ix2] = i2_new
#        config_new = tuple(config_new)
#        py = self.prob_fn(config_new)
#        try:
#            acceptance = py / self.px
#        except ZeroDivisionError:
#            acceptance = 1. if py > self.px else 0.
#        #print(py,self.px,acceptance)
#        if self.rng.uniform() < acceptance:
#            self.config = config_new
#            self.px = py
#    def sweep_row(self):
#        for i in range(self.Lx):
#            for j in range(self.Ly-1):
#                site1,site2 = (i,j),(i,j+1)
#                self.sample_pair(site1,site2)
#    def sweep_col(self):
#        for j in range(self.Ly):
#            for i in range(self.Lx-1):
#                site1,site2 = (i,j),(i+1,j)
#                self.sample_pair(site1,site2)
#    def sample_sweep(self):
#        first = self.rng.integers(low=0,high=1,endpoint=True)
#        if first==0:
#            self.sweep_row()
#            self.sweep_col()
#        else:
#            self.sweep_col()
#            self.sweep_row()
#    def sample_rand(self):
#        blocks = self.rng.permutation(self.blocks)
#        for i,j in blocks:
#            site1 = i,j
#            site2 = [(i,j+1),(i+1,j)][self.rgn.integers(low=0,high=1,endpoint=True)]
#            self.sample_pair(site1,site2)
#    def sample(self):
#        if self.sweep:
#            self.sample_sweep()
#        else:
#            self.sample_rand()
#        return self.config,self.px
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
        self.alternate = None
    def initialize(self,config):
        # randomly choses the initial sweep direction
        self.sweep_row_dir = self.rng.choice([-1,1]) 
        # setup to compute all opposite envs for initial sweep
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 
        self.px = self.amplitude_factory.prob(config)
        self.config = config
        # force to initialize with a better config
        if self.px < PRECISION:
            raise ValueError 
    def preprocess(self):
        self._burn_in()
    def _burn_in(self,batchsize=None):
        self.alternate = True
        batchsize = self.burn_in if batchsize is None else batchsize
        if batchsize==0:
            return
        t0 = time.time()
        for n in range(batchsize):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
        self.alternate = False 
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
        #print(self.sweep_row_dir,self.sweep_col_dir)
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
    def __init__(self,Lx,Ly,nelec,exact=False,seed=None):
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
        self.disp = [0]
        for batchsize in self.count[:-1]:
            self.disp.append(self.disp[-1]+batchsize)
        self.start = self.disp[RANK]
        self.stop = self.start + self.count[RANK]

        self.rng = np.random.default_rng(seed)
        self.burn_in = 0
        self.dense = True
        self.exact = exact 
        self.amplitude_factory = None
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
        nonzeros = np.nonzero(ptotal)[0]
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

