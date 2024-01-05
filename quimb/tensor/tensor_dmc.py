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
    def propagate(self,config,w,weff):
        self.config = config
        self.weight = np.array([w,weff]) 
        self.tl = self.tau
        self.terminate = False
        self.nhop = 0
        while True:
            if self.terminate:
                break
            self.sample()
    def sample(self):
        self.af.config = self.config 
        self.af.cx = dict()

        self.h = dict()
        self.u = self.af.model.compute_local_energy_eigen(self.config)
        for batch_key in self.af.model.batched_pairs:
           self.h.update(self.af.batch_pair_energies(batch_key,False)[0])
        self.vsf = sum([val for val in self.h.values() if val > 0])
        self.ueff = self.u + (1+self.gamma) * self.vsf
        self.e = sum(self.h.values()) + self.u
        
        self.logxi = np.log(self.rng.random())
        if self.lamb is None:
            self.propagate1()
        else:
            self.propagate2()
    def sample_offdiag(self):
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
    def propagate1(self):
        pid = self.e - self.ueff
        td = min(self.tl,self.logxi/pid)

        self.weight[0] *= np.exp(-(self.e-(1+self.gamma)*self.vsf)*td)
        self.weight[1] *= np.exp(-self.e*td)

        self.tl -= td
        if self.tl <= 0:
            self.terminate = True
            return
        self.sample_offdiag() 
    def propagate2(self):
        pd = self.lamb - self.ueff
        if pd<0:
            raise ValueError(f'u={self.u},vsf={self.vsf},ueff={self.ueff},{self.h}')
        pd /= self.lamb - self.e
        k = min(self.tl,int(self.logxi/np.log(pd)))

        self.weight *= (1-self.e/self.lamb)**k 
        self.weight[0] *= ((self.lamb-self.u)/(self.lamb-self.ueff))**k
        self.tl -= k+1
        if self.tl <= 0:
            self.terminate = True
            return
        self.sample_offdiag() 
class SamplerSR:
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
        self.terminate = np.zeros(self.nsite+3)
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
            self.config[ncurr,-2] = self.buf[3] 
            self.config[ncurr,-1] = self.buf[4] 
            nhop[ncurr] = self.buf[5]
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
        self.compute_energy(self)
    def _sample(self):
        while True:
            COMM.Recv(self.terminate,source=0,tag=1)
            if int(self.terminate[-1]+.1)==1:
                break
            
            config = tuple(np.array(self.terminate[:self.nsite]+.1,dtype=int))
            w = self.terminate[self.nsite]
            weff = self.terminate[self.nsite+1]
            self.wk.propagate(config,w,weff)
            self.buf[2] = self.wk.e
            self.buf[3] = self.wk.weight[0]
            self.buf[4] = self.wk.weight[1]
            self.buf[5] = self.wk.nhop
            COMM.Send(self.buf,dest=0,tag=0)
    def compute_energy(self):
        # update quantities
        self.ws.append(np.sum(self.config[:,-2:],axis=0)) 
        self.we.append(np.dot(self.e,self.config[:,-2:]))
        # compute energy
        self.E = self.compute_expectation(self.we) 
        print(f'step={self.step},e={self.E[0]/self.nsite},e_eff={self.E[1]/self.nsite}')
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
        ws_ = np.fabs(self.w[:,-2]).sum()
        print('\tave sign=',ws[0],ws_,ws[0]/ws_)

        ops = [self.e]
        ops_eff = [self.E[1]]
        v = np.zeros((1,self.M))
        for ix in range(1):
            v[ix] = ops[ix] - ops_eff[ix]
        S = np.einsum('kj,lj,j->kl',v,v,self.w[:,-1]) / ws[1]
        b = np.dot(v,self.w[:,-2]) / ws[0]
        alpha = np.linalg.solve(S,b)
        print('\talpha',alpha)
        print('\tw=',ws[0]/self.M,ws[1]/self.M)
        #print(self.w[0])
        #print(self.w[1])
        p = self.w[:,-1] * (1 + np.dot(alpha,v))
        psum = p.sum()

        prob = np.fabs(p)
        prob_sum = prob.sum()
        print('\tbeta=',psum,prob_sum,psum/prob_sum)
        prob /= prob_sum 
        idx = self.rng.choice(self.M,size=self.M,p=prob) 
        self.config = self.config[idx]
        self.config[:,-2] = np.sign(p[idx])
        self.config[:,-1] = 1

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
    def init(self,config):
        if RANK!=0:
            return
        self.config = config
        self.M = config.shape[0]
        self.config = np.concatenate([config,np.ones((self.M,2))],axis=1) 
        print(f'number of walkers=',self.M)
    def load(self,fname):
        if RANK!=0:
            return
        f = h5py.File(fname,'r')
        self.config = f['config_new'][:] 
        self.f += list(f['f'][:])
        self.ws += list(f['ws'][:])
        self.we += list(f['we'][:])
        f.close()
        self.M = self.config.shape[0] # num walkers
        print(f'number of walkers=',self.M)
class SampleBranch(SampleSR):
    def __init__(self,wk,nmin,nmax,thresh=1e-6,seed=None):
        self.wk = wk
        self.nsite = wk.af.nsite 
        self.nmin = nmin
        self.nmax = nmax
    
        self.rng = np.random.default_rng(seed=seed)
        self.progbar = False 
    def compute_energy(self):
        self.E = self.compute_expectation(self.e) 
        print(f'step={self.step},e={self.E[0]/self.nsite},e_eff={self.E[1]/self.nsite}')
    def compute_expectation(self,e):
        E = np.zeros(2)
        E[0] = blocking_analysis(e,weights=self.config[:,-2])
        E[1] = blocking_analysis(e,weights=self.config[:,-1])
        return E
    def SR(self):
        if RANK!=0:
            return
        wmean = self.config[:,-1].sum()/self.M
        print('\tave weff=',wmean)
        self.config[:,-2:] /= wmean
        config_new = []
        for ix in range(self.M):
            config = self.config[ix] 
            m = int(config[-1]+self.rng.rand())
            if m==1:
                config_new.append(config)
            elif m==0:
                if self.M < self.nmin:
                    config_new.append(config)
            else:
                if self.M > self.nmax:
                    config_new.append(config)
                else:
                    config[-2:] /= m
                    config_new += [config] * m
        self.config = np.array(config_new)
        self.M = self.config.shape[0]
    def save(self,tmpdir):
        np.save(tmpdir+f'step{self.step}.npy',self.config)
