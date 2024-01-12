import time,scipy,functools,h5py,gc
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
        self.rng = np.random.default_rng(seed=seed)
    def propagate(self,config):
        self.config = config
        self.weight = 1 
        self.nhop = 0
        self.remain = self.tau
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

        u += (1+self.gamma)*sum([val for val in self.h.values() if val > 0])
        
        logxi = np.log(self.rng.random())
        td = min(self.remain,self.logxi/(self.e-u))
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
    def run(self,start,stop,tmpdir=None):
        for step in range(start,stop):
            self.step = step
            self.sample()
            self.SR()
            self.save(tmpdir)
    def sample(self):
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
            if int(self.terminate[-1]+.1)==1:
                break
            
            config = np.array(self.terminate[:-1]
            self.wk.propagate(config)
            self.buf[2] = self.wk.e
            self.buf[3] = self.wk.weight
            self.buf[4] = self.wk.nhop
            COMM.Send(self.buf,dest=0,tag=0)
    def SR(self):
        if RANK!=0:
            return
        w = self.w.sum()/self.M
        self.ws.append(w) 
        e,err = blocking_analysis(self.e,weights=self.w)
        self.we.append(e)
        print(f'step={self.step},e={(e+shift)/self.nsite},err={err/self.nsite}')

        xi = self.rng.random()
        p = w/w.sum()
        pcum = np.cumsum(p)
        config = np.zeros_like(self.config) 
        for i in range(self.M):
            z = i/self.M + xi 
            idx = len(pc[pc<xi])
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
        self.config = config
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
