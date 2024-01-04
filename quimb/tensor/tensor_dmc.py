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
    def __init__(self,af,tau,lamb=None,gamma=0.,seed=None):
        self.af = af
        self.tau = tau
        self.lamb = lamb 
        self.gamma = gamma
        self.rng = np.random.default_rng(seed=seed)
    def propagate(self,config,sign):
        self.config = config
        self.weight = np.array([sign,1]) 
        self.tl = self.tau
        self.terminate = False
        self.nhop = 0
        while True:
            if self.terminate:
                break
            e = self.sample()
        return e
    def sample(self):
        self.af.config = self.config 
        self.af.cx = dict()

        # (a)
        self.h = dict()
        self.u = self.af.model.compute_local_energy_eigen(self.config)
        for batch_key in af.model.batched_pairs:
           self.h.update(self.af.batch_pair_energies(batch_key,False)[0])
        self.e = sum(h.values()) + self.u
        self.vsf = sum([val for val in self.h.values() if val > 0])
        self.ueff = self.u + (1+self.gamma) * self.vsf
        
        if self.lamb is None:
            self.propagate1()
        else:
            self.propagate2()
    def propagate1(self):
        logxi = np.log(self.rng.random())
        pid = self.e - self.ueff
        td = min(self.tl,logxi/pid)
        if RANK==1:
            print('td=',logxi/pid,pid)

        # (b)
        self.tl -= td
        self.weight[0] *= np.exp((-self.e+(1+self.gamma)*self.vsf)*td)
        self.weight[1] *= np.exp(-self.e*td)

        if self.tl <= 0:
            self.terminate = True
            return
        keys = list(self.h.keys())
        p =  - np.array(list(self.h.values()))
        p = np.where(p>0,p,-self.gamma*p)
        fac = 1 if self.gamma < 1e-6 else -1./self.gamma
        s = np.where(p>0,1,fac)

        p /= p.sum() 
        idx = self.rng.choice(len(p),p=p)
        self.weight[0] *= s[idx] 
        self.config = keys[idx]
        self.nhop += 1
    def propagate2(self):
        if self.tl<=0:
            self.terminate = True
            return  
        keys = list(self.h.keys())
        p =  - np.array(list(self.h.values()))
        p = np.where(p>0,p,-self.gamma*p)
        fac = 1 if self.gamma < 1e-6 else -1./self.gamma
        s = np.where(p>0,1,fac)

        keys.append(self.config)
        p = np.concatenate([p,np.array([self.lamb - self.ueff])]) 
        s = np.concatenate([s,np.array([(self.lamb-self.u)/p[-1]])]) 
        
        b = p.sum()
        self.weight *= b

        p /= b 
        idx = self.rng.choice(len(p),p=p)
        self.weight[0] *= s[idx] 
        self.config = keys[idx]

        self.tl -= 1
        if idx<len(p)-1:
            self.nhop += 1
class Sampler:
    def __init__(self,wk,L,N,seed=None):
        self.wk = wk
        self.nsite = wk.af.nsite 
        self.L = L # product length
        self.N = N # sum length
        self.f = []
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
        self.terminate = np.zeros(self.nsite+2,dtype=int)
        self.buf = np.zeros(6)
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
        self.w = np.zeros((2,self.M))
        nhop = np.zeros(self.M) 
        t0 = time.time()
        for worker in range(1,SIZE): 
            self.terminate[:self.nsite+1] = self.config[worker-1]
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
            self.w[0,ncurr] = self.buf[3] 
            self.w[1,ncurr] = self.buf[4] 
            nhop[ncurr] = self.buf[5]
            ncurr += 1
            if ncurr>=self.M:
                break
            if self.progbar:
                pg.update()
            self.terminate[:self.nsite+1] = self.config[ncurr]
            COMM.Send(self.terminate,dest=rank,tag=1)
        self.terminate[-1] = 1
        for worker in range(1,SIZE): 
            COMM.Send(self.terminate,dest=worker,tag=1)
        print('sample time=',time.time()-t0)

        # update quantities
        self.ws.append(np.sum(self.w,axis=1)) 
        self.we.append(np.dot(self.w,self.e))
        # compute energy
        self.E = self.compute_expectation(self.we) 
        print(f'step={self.step},e={self.E[0]/self.nsite},e_eff={self.E[1]/self.nsite}')
        # 
        print('hop max,mean=',max(nhop),nhop.sum()/self.M)
    def _sample(self):
        while True:
            COMM.Recv(self.terminate,source=0,tag=1)
            if self.terminate[-1]==1:
                break
            
            config = tuple(self.terminate[:self.nsite])
            sign = self.terminate[self.nsite]
            self.wk.propagate(config,sign)
            self.buf[2] = self.wk.e
            self.buf[3] = self.wk.weight[0]
            self.buf[4] = self.wk.weight[1]
            self.buf[5] = self.wk.nhop
            COMM.Send(self.buf,dest=0,tag=0)
    def compute_expectation(self,we):
        if len(self.f)<=self.L:
            return we[-1]/self.ws[-1]
        N = min(len(self.f)-self.L,self.N)
        f = np.array(self.f)
        G = np.array([f[n:n+self.L].prod() for n in range(N)]) 
        we = np.array(we[-N:])
        ws = np.array(self.ws[-N:])
        #print(N)
        #print(G)
        #print(we)
        #print(ws)
        #exit()
        return np.dot(G,we)/np.dot(G,ws)
    def SR(self):
        if RANK!=0:
            return
        ws = self.ws[-1]
        ws_ = np.fabs(self.w[0]).sum()
        print('ave sign=',ws[0],ws_,ws[0]/ws_)

        ops = [self.e]
        ops_eff = [self.E[1]]
        v = np.zeros((1,self.M))
        for ix in range(1):
            v[ix] = ops[ix] - ops_eff[ix]
        S = np.einsum('kj,lj,j->kl',v,v,self.w[1]) / ws[1]
        b = np.dot(v,self.w[0]) / ws[0]
        alpha = np.linalg.solve(S,b)
        print('alpha',alpha)
        p = self.w[1] * (1 + np.dot(alpha,v))
        psum = p.sum()

        prob = np.fabs(p)
        prob_sum = prob.sum()
        print('beta=',psum,prob_sum,psum/prob_sum)
        prob /= prob_sum 
        idx = self.rng.choice(self.M,size=self.M,p=prob) 
        self.config[:,:-1] = self.config[idx,:-1]
        self.config[:,-1] = np.array(np.sign(p[idx])+.1,dtype=int)

        f = ws[0]/self.M*prob_sum/psum
        self.f.append(f)
        if len(self.f)>self.N+self.L:
            self.f.pop(0)
            self.we.pop(0)
            self.ws.pop(0)
    def save(self,tmpdir):
        if RANK!=0:
            return
        f = h5py.File(tmpdir+f'step{self.step}.hdf5','w')
        f.create_dataset('config_new',data=self.config)
        f.create_dataset('f',data=np.array(self.f)) 
        f.create_dataset('ws',data=np.array(self.ws)) 
        f.create_dataset('we',data=np.array(self.we)) 
        f.close()
    def init(self,fname=None,config=None):
        if RANK!=0:
            return
        if config is not None:
            self.config = config
            M = config.shape[0]
            self.config = np.concatenate([config,np.ones((M,1))],axis=1) 
        if fname is not None:
            f = h5py.File(fname,'r')
            self.config = f['config_new'][:] 
            self.f += list(f['f'][:])
            self.ws += list(f['ws'][:])
            self.we += list(f['we'][:])
            f.close()
            self.M = self.config.shape[0] # num walkers
        self.config = np.array(self.config+.1,dtype=int)
        
