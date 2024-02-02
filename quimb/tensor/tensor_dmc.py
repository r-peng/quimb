import time,h5py,gc,itertools
#from pympler import muppy,summary
import numpy as np
from quimb.utils import progbar as Progbar
from .tensor_vmc import (
    blocking_analysis,
    scale_wfn,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
DISCARD = 1e3
class Walker:
    def __init__(self,af,tau,kp=None,discrete=False,seed=None):
        self.af = af
        self.tau = tau
        self.kp = kp 
        self.discrete = discrete
        self.shift = 0 
        self.rng = np.random.default_rng(seed=seed)
        self.thresh = 1e-10
    def propagate(self,config):
        self.config = config
        self.weight = 1 
        self.nhop = 0
        self.remain = self.kp if self.discrete else self.tau
        while True:
            self.sample()
            if self.remain <= self.thresh:
                break
    def sample(self):
        self.af.config = self.config 
        self.af.cx = dict()

        h = dict() 
        for batch_key in self.af.model.batched_pairs:
           h.update(self.af.batch_pair_energies(batch_key,False)[0])
        u = self.af.model.compute_local_energy_eigen(self.config) - self.shift 
        self.e = sum(h.values()) + u

        u += sum([val for val in h.values() if val > 0])
        if self.discrete:
            self._sample_discrete(h,u)
        else:
            self._sample_continuous(h,u)
    def _sample_discrete(self,h,u):
        keys = [self.config]
        p = [1-self.tau * u]
        if p[0]<0:
            raise ValueError(f'diagonal={p[0]},u={u}')
        for key,val in h.items():
            if val >= 0:
                continue
            p.append(-self.tau * val)
            keys.append(key)
        
        p = np.array(p)
        self.weight *= (1-self.tau*self.e)
        idx = self.rng.choice(len(p),p=p/(1-self.tau * self.e))
        self.config = keys[idx]
        if idx>0:
            self.nhop += 1
        self.remain -= 1
    def _sample_continuous(self,h,u):
        logxi = np.log(1-self.rng.random())
        td = min(self.remain,logxi/(self.e-u))
        self.weight *= np.exp(-td*self.e)
        self.remain -= td
        if self.remain <= self.thresh:
            return

        keys = []
        p = []
        for key,val in h.items():
            if val >= 0:
                continue
            keys.append(key)
            p.append(val)
        p = - np.array(p)

        idx = self.rng.choice(len(p),p=p/p.sum())
        self.config = keys[idx]
        self.nhop += 1
def load(fname):
    f = h5py.File(fname,'r')
    config = f['config_new'][:] 
    ws = f['ws'][:]
    we = f['we'][:]
    f.close()
    return config,ws,we
def compute_expectation(we,ws,L,start,size):
    ws = np.log(ws)
    G = np.array([ws[i:i+L].sum() for i in range(start,start+size)])
    G = np.exp(G - G.sum()/size)
    return blocking_analysis(we[start+L+1:start+L+1+size],weights=G,printQ=False)
class Sampler:
    def __init__(self,wk,seed=None):
        self.wk = wk
        self.nsite = wk.af.nsite 
        self.ws = []
        self.we = [] 

        self.rng = np.random.default_rng(seed=seed)
        self.progbar = False 
        self.clear_every = 1
    def run(self,start,stop,tmpdir=None):
        for step in range(start,stop):
            self.step = step
            self.sample()
            self.SR(tmpdir=tmpdir)
    def sample(self):
        if RANK==0 and self.progbar:
            pg = Progbar(total=self.M)
        t0 = time.time()
        for i in range(self.M):
            config = tuple(np.array(self.config[i,:self.nsite] + .1,dtype=int))
            self.wk.propagate(config)
            self.config[i,:self.nsite] = np.array(self.wk.config)
            self.config[i,-1] = self.wk.e
            self.config[i,-2] = self.wk.weight
            self.config[i,-3] = self.wk.nhop
            if RANK==0 and self.progbar:
                pg.update()
        if self.step % self.clear_every == 0:
            self.wk.af.cache = dict() 
            gc.collect()
        if RANK==0:
            print('\tsample time=',time.time()-t0)
    def SR(self,tmpdir=None):
        if RANK>0:
            COMM.Send(self.config,dest=0,tag=0)
            COMM.Barrier() 
            COMM.Recv(self.config,source=0,tag=1)
            return
        config = [self.config.copy()]
        for worker in range(1,SIZE):
            COMM.Recv(self.config,source=worker,tag=0)
            config.append(self.config.copy())
        config = np.concatenate(config,axis=0)
        e = config[:,-1]
        w = config[:,-2]
        hop = config[:,-3]

        wsum = w.sum()
        self.ws.append(wsum/len(w)) 
        e,err = blocking_analysis(e,weights=w)
        self.we.append(e)
        print('\thop max,mean=',max(hop),hop.sum()/len(hop))
        print(f'step={self.step},e={(e+self.wk.shift)/self.nsite},err={err/self.nsite},wmean={self.ws[-1]}')

        #idx = self.rng.choice(config.shape[0],size=config.shape[0],replace=True,p=w/wsum)
        #print('num config surviving=',len(set(idx)))
        #config_new = self.rng.choice(config,size=config.shape[0],replace=True,p=w/wsum,axis=0)

        xi = self.rng.random()
        pc = np.cumsum(w/wsum)
        config_new = np.zeros_like(config) 
        idx = [None] * config.shape[0]
        for i in range(config.shape[0]):
            z = (i+xi)/config.shape[0]
            idx[i] = len(pc[pc<z])
            config_new[i] = config[idx[i]]
        print('num config surviving=',len(set(idx)))

        if tmpdir is not None:
            f = h5py.File(tmpdir+f'step{self.step}.hdf5','w')
            f.create_dataset('config_new',data=config_new[:,:self.nsite])
            f.create_dataset('ws',data=np.array(self.ws)) 
            f.create_dataset('we',data=np.array(self.we)) 
            f.close()
        COMM.Barrier() 
        for worker in range(1,SIZE):
            self.config = config_new[worker*self.M:(worker+1)*self.M]
            COMM.Send(self.config,dest=worker,tag=1)
        self.config = config_new[:self.M]
    def init(self,config):
        self.M = config.shape[0]
        self.config = np.concatenate([config,np.zeros((self.M,3))],axis=1)
    def load(self,fname):
        config,ws,we = load(fname) 
        batchsize = config.shape[0] // SIZE
        self.init(config[RANK*batchsize:(RANK+1)*batchsize])
        if RANK!=0:
            return
        self.ws = list(ws)
        self.we = list(we)
class Hubbard1D:
    def __init__(self,u,L):
        self.u = u
        self.L = L
        self.batched_pairs = [None]       
    def compute_local_energy_eigen(self,config):
        config = np.array(config)
        return self.u * len(config[config==3])
    def pair_coeff(self,*args):
        return -1
class J1J2_2D:
    def __init__(self,Lx,Ly,j1,j2):
        self.J1,self.J2 = j1,j2
        self.Lx,self.Ly = Lx,Ly
        self.batched_pairs = [None]       
        self.pbc = False
    def flatten(self,site):
        i,j = site
        return i * self.Ly + j
    def flat2site(self,ix):
        return ix // self.Ly, ix % self.Ly
    def compute_local_energy_eigen(self,config):
        # NN
        e1 = 0.
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            s1 = (-1) ** config[self.flatten((i,j))]
            if j+1<self.Ly:
                e1 += s1 * (-1)**config[self.flatten((i,j+1))]
            else:
                if self.pbc:
                    e1 += s1 * (-1)**config[self.flatten((i,0))]
            if i+1<self.Lx:
                e1 += s1 * (-1)**config[self.flatten((i+1,j))]
            else:
                if self.pbc:
                    e1 += s1 * (-1)**config[self.flatten((0,j))]
        # next NN
        e2 = 0. 
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            if i+1<self.Lx and j+1<self.Ly: 
                ix1,ix2 = self.flatten((i,j)), self.flatten((i+1,j+1))
                e2 += (-1)**(config[ix1]+config[ix2])
                ix1,ix2 = self.flatten((i,j+1)), self.flatten((i+1,j))
                e2 += (-1)**(config[ix1]+config[ix2])
            else:
                if self.pbc:
                    ix1,ix2 = self.flatten((i,j)), self.flatten(((i+1)%self.Lx,(j+1)%self.Ly))
                    e2 += (-1)**(config[ix1]+config[ix2])
                    ix1,ix2 = self.flatten((i,(j+1)%self.Ly)), self.flatten(((i+1)%self.Lx,j))
                    e2 += (-1)**(config[ix1]+config[ix2])
        return .25 * (e1 * self.J1 + e2 * self.J2) 
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        dx = abs(site2[0]-site1[0])
        dy = abs(site2[1]-site1[1])
        if dx == 0:
            return self.J1
        if dy == 0:
            return self.J1
        return self.J2
    def pair_terms(self,i1,i2):
        return 1-i1,1-i2,0.5
def get_config(i1,i2):
    pn_map = 0,1,1,2
    n1,n2 = pn_map[i1],pn_map[i2]
    ndiff = abs(n1-n2)
    if ndiff == 1:
        return [(i2,i1)]
    if ndiff == 0:
        return [(0,3),(3,0)] 
    if ndiff == 2:
        return [(1,2),(2,1)]
class ConstWFN1D:
    def __init__(self,model):
        self.model = model
        self.nsite = self.model.L
        self.psi = None
    def batch_pair_energies(self,*args):
        e = dict()
        for i in range(self.model.L-1):
            i1,i2 = self.config[i],self.config[i+1]
            if i1==i2:
                continue
            for i1_new,i2_new in get_config(i1,i2):
                config_new = list(self.config)
                config_new[i] = i1_new
                config_new[i+1] = i2_new
                e[tuple(config_new)] = self.model.pair_coeff(i,i+1) 
        return e,None
class ConstWFN2D:
    def __init__(self,model,fermion=False):
        self.model = model
        self.Lx,self.Ly = model.Lx,model.Ly
        self.nsite = self.model.Lx * self.model.Ly
        self.psi = None
    def flatten(self,i,j):
        return i * self.Ly + j
    def flat2site(self,ix):
        return ix // self.Ly, ix % self.Ly
    def update_e(self,e,site1,site2,ix1,sign1):
        (i1,j1),(i2,j2) = site1,site2
        ix2 = self.flatten(i2,j2)
        i1_old,i2_old = self.config[ix1],self.config[ix2]
        if i1_old==i2_old:
            return e
        i1_new,i2_new,coeff = self.model.pair_terms(i1_old,i2_old) 
        coeff *= self.model.pair_coeff(site1,site2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        sign2 = self.config_sign(config_new)
        e[tuple(config_new)] = coeff * sign1 * sign2 
        return e
    def config_sign(self,config):
        n = 0
        for i in range(self.Lx):
            conf = config[i*self.Ly:(i+1)*self.Ly]
            n += sum([conf[j] for j in range(i%2,self.Ly,2)])
        return (-1)**n
    def batch_pair_energies(self,*args):
        e = dict()
        sign1 = self.config_sign(self.config)
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            ix1 = self.flatten(i,j) 
            if i+1<self.Lx:
                e = self.update_e(e,(i,j),(i+1,j),ix1,sign1)
            if j+1<self.Ly:
                e = self.update_e(e,(i,j),(i,j+1),ix1,sign1)
            if i+1<self.Lx and j+1<self.Ly:
                e = self.update_e(e,(i,j),(i+1,j+1),ix1,sign1)
                ix1 = self.flatten(i,j+1)
                e = self.update_e(e,(i,j+1),(i+1,j),ix1,sign1)
        return e,None


