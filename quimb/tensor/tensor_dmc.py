import time,h5py,gc
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
    def __init__(self,af,tau,seed=None):
        self.af = af
        self.tau = tau
        self.gamma = 0
        self.shift = 0 
        self.kp = 10
        self.rng = np.random.default_rng(seed=seed)
    def propagate(self,config):
        self.config = config
        self.weight = 1 
        self.nhop = 0
        #self.remain = self.tau
        self.remain = self.kp
        while True:
            self.sample()
            if self.remain <= 0:
                break
    def sample(self):
        self.af.config = self.config 
        self.af.cx = dict()

        h = dict() 
        for batch_key in self.af.model.batched_pairs:
           h.update(self.af.batch_pair_energies(batch_key,False)[0])
        u = self.af.model.compute_local_energy_eigen(self.config) - self.shift 
        self.e = sum(h.values()) + u

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
        b = p.sum()
        self.weight *= b
        idx = self.rng.choice(len(p),p=p/b)
        self.config = keys[idx]
        if idx>0:
            self.nhop += 1
        self.remain -= 1
    def _sample(self):
        self.af.config = self.config 
        self.af.cx = dict()

        h = dict()
        for batch_key in self.af.model.batched_pairs:
           h.update(self.af.batch_pair_energies(batch_key,False)[0])
        u = self.af.model.compute_local_energy_eigen(self.config) - self.shift
        self.e = sum(h.values()) + u

        u += (1+self.gamma)*sum([val for val in h.values() if val > 0])
        
        logxi = np.log(self.rng.random())
        td = min(self.remain,logxi/(self.e-u))
        #print(self.config,td,self.e,h)
        #exit()
        self.weight *= np.exp(-td*self.e)
        self.remain -= td
        if self.remain <= 0:
            return

        keys = []
        p = []
        for key,val in h.items():
            keys.append(key)
            sign = -1 if val<0 else self.gamma
            p.append(val*sign)
        p = np.array(p)

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
def compute_expectation(we,ws,L,eq=0):
    we = we[eq:]
    ws = ws[eq:]
    N = len(ws)-L-1
    G = np.array([ws[i:i+L].prod() for i in range(N)])
    return blocking_analysis(we[L+1:],weights=G,printQ=False)
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
            self.SR()
            self.save(tmpdir)
    def sample(self):
        #if RANK==self.print_rank:
        #    all_ = muppy.get_objects()
        #    sum_ = summary.summarize(all_)
        #    summary.print_(sum_)
        self.terminate = np.zeros(self.nsite+1,dtype=int)
        self.buf = np.zeros(5)
        self.buf[0] = self.step 
        self.buf[1] = RANK  

        if RANK==0:
            self._ctr()
        else:
            self._sample()
    def _ctr(self):
        if self.progbar:
            pg = Progbar(total=self.M)
        ncurr = 0
        self.e = np.zeros(self.M) 
        self.w = np.zeros(self.M)
        nhop = np.zeros(self.M) 
        t0 = time.time()
        for worker in range(1,SIZE): 
            self.terminate[:-1] = self.config[worker-1]
            COMM.Send(self.terminate,dest=worker,tag=1)
        while True:
            COMM.Recv(self.buf,tag=0)
            step = int(self.buf[0].real+.1)
            if step>self.step:
                raise ValueError
            elif step<self.step:
                continue

            rank = int(self.buf[1].real+.1)
            self.e[ncurr] = self.buf[2] 
            self.w[ncurr] = self.buf[3] 
            nhop[ncurr] = self.buf[4]
            ncurr += 1
            if ncurr>=self.M:
                break
            if self.progbar:
                pg.update()
            self.terminate[:-1] = self.config[ncurr]
            COMM.Send(self.terminate,dest=rank,tag=1)
        self.terminate[-1] = 1
        for worker in range(1,SIZE): 
            COMM.Send(self.terminate,dest=worker,tag=1)
        print('\tsample time=',time.time()-t0)
        print('\thop max,mean=',max(nhop),nhop.sum()/self.M)
    def _sample(self):
        while True:
            COMM.Recv(self.terminate,source=0,tag=1)
            if self.terminate[-1]==1:
                break
            
            config = tuple(self.terminate[:-1])
            self.wk.propagate(config)
            self.buf[2] = self.wk.e
            self.buf[3] = self.wk.weight
            self.buf[4] = self.wk.nhop
            COMM.Send(self.buf,dest=0,tag=0)
        if self.step % self.clear_every == 0:
            self.wk.af.cache = dict() 
            gc.collect()
    def SR(self):
        if RANK!=0:
            return
        wsum = self.w.sum()
        self.ws.append(wsum/self.M) 
        e,err = blocking_analysis(self.e,weights=self.w)
        self.we.append(e)
        print(f'step={self.step},e={(e+self.wk.shift)/self.nsite},err={err/self.nsite},wmean={self.ws[-1]}')

        xi = self.rng.random()
        pc = np.cumsum(self.w/wsum)
        config = np.zeros_like(self.config) 
        for i in range(self.M):
            z = (i+xi)/self.M 
            idx = len(pc[pc<z])
            config[i] = self.config[idx]
        self.config = config
    def save(self,tmpdir):
        if RANK!=0:
            return
        f = h5py.File(tmpdir+f'step{self.step}.hdf5','w')
        f.create_dataset('config_new',data=self.config)
        f.create_dataset('ws',data=np.array(self.ws)) 
        f.create_dataset('we',data=np.array(self.we)) 
        f.close()
    def init(self,config):
        if RANK!=0:
            return
        self.config = np.array(config+.1,dtype=int)
        self.M = config.shape[0]
        print(f'number of walkers=',self.M)
    def load(self,fname):
        if RANK!=0:
            return
        self.config,self.ws,self.we = load(fname) 
        self.ws = list(self.ws)
        self.we = list(self.we)
        self.M = self.config.shape[0] # num walkers
        print(f'number of walkers=',self.M)

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
def get_config(i1,i2):
    pn_map = 0,1,1,2
    n1,n2 = pn_map[i1],pn_map[i2]
    if abs(n1-n2) == 1:
        return [(i2,i1)]
    if n1+n2==2 and n1==n2:
        return [(0,3),(3,0)] 
    if n1+n2==2 and n1!=n2:
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
