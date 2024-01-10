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
    def __init__(self,af,tau,kp,seed=None):
        self.af = af
        self.tau = tau
        self.kp = kp
        self.gamma = 0
        self.shift = 0 
        self.method = None 
        self.thresh = 1e-6
        self.rng = np.random.default_rng(seed=seed)
    def propagate(self,config,w,weff):
        self.config = config
        self.weight = np.array([w,weff]) 
        self.nhop = 0
        self.remain = self.tau * self.kp if self.method==1 else self.kp
        while True:
            self.sample()
            if self.remain <= 0:
                break
    def sample(self):
        self.af.config = self.config 
        self.af.cx = dict()

        self.h = dict()
        self.u = self.af.model.compute_local_energy_eigen(self.config) - self.shift
        for batch_key in self.af.model.batched_pairs:
           self.h.update(self.af.batch_pair_energies(batch_key,False)[0])
        self.vsf = sum([val for val in self.h.values() if val > 0])
        self.ueff = self.u + (1+self.gamma) * self.vsf
        self.e = sum(self.h.values()) + self.u
        
        self.logxi = np.log(self.rng.random())
        if self.method == 1:
            return self.propagate1()
        else:
            return self.propagate2()
    def sample_offdiag(self):
        keys = list(self.h.keys())
        p =  - np.array(list(self.h.values()))
        p = np.where(p>0,p,-self.gamma*p)
        fac = 1 if self.gamma < self.thresh else -1./self.gamma
        s = np.where(p>0,1,fac)

        idx = self.rng.choice(len(p),p=p/p.sum())
        self.weight[0] *= s[idx] 
        self.config = keys[idx]
        self.nhop += 1
    def propagate1(self):
        pid = self.e - self.ueff
        td = min(self.remain,self.logxi/pid)

        self.weight[0] *= np.exp(-(self.e-(1+self.gamma)*self.vsf)*td)
        self.weight[1] *= np.exp(-self.e*td)
        self.remain -= td
        if self.remain > 0:
            self.sample_offdiag() 
    def propagate2(self):
        pd = 1 - self.ueff * self.tau
        if pd<0:
            print(f'u={self.u},vsf={self.vsf},ueff={self.ueff}')
            exit()
        b = 1 - self.e * self.tau
        k = min(self.remain,int(self.logxi/np.log(pd/b)))

        self.weight *= b**k 
        self.weight[0] *= ((1-self.u * self.tau)/pd)**k
        self.remain -= k+1
        if self.remain > 0:
            self.sample_offdiag() 
def load(fname):
    f = h5py.File(fname,'r')
    config = f['config_new'][:] 
    wf = f['f'][:]
    ws = f['ws'][:]
    we = f['we'][:]
    f.close()
    return config,wf,ws,we
def compute_expectation(we,ws,f,L,eq=0):
    we = we[eq:]
    ws = ws[eq:]
    f = f[eq:]
    N = len(f)-L-1
    G = np.array([f[i:i+L].prod() for i in range(N)])
    return np.dot(G,we[L+1:])/np.dot(G,ws[L+1:])
class SamplerSR:
    def __init__(self,wk,seed=None):
        self.wk = wk
        self.nsite = wk.af.nsite 
        self.f = []
        self.ws = []
        self.we = [] 

        self.rng = np.random.default_rng(seed=seed)
        self.progbar = False 
    def run(self,start,stop,tmpdir=None):
        for step in range(start,stop):
            self.step = step
            self.sample()
            terminate = self.SR()
            self.save(tmpdir)
            if terminate:
                break
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
        print('\thop max,min,mean=',max(nhop),min(nhop),nhop.sum()/self.M)
        self.compute_energy()
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
        #self.E = self.compute_expectation(self.we) 
        self.E = self.we[-1] / self.ws[-1]
        e = self.E + self.wk.shift 
        e /= self.nsite
        print(f'step={self.step},e={e[0]},e_eff={e[1]}')
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
        w = self.config[:,-2:]
        ws_ = np.fabs(w[:,1]).sum()
        print('\tave sign=',ws[0],ws_,ws[0]/ws_)

        v = self.e - self.E[1] 
        var  = np.dot(self.e**2,w[:,1])-self.E[1]**2
        alpha = (self.E[0]-self.E[1])/var
        print('\terr_eff=',np.sqrt(var/self.M))
        print('\talpha',alpha)
        print('\tw=',ws[0]/self.M,ws[1]/self.M)
        p = w[:,1] * (1 + alpha*v)
        psum = p.sum()

        prob = np.fabs(p)
        prob_sum = prob.sum()
        beta = psum/prob_sum
        print('\tbeta=',psum,prob_sum,beta)
        prob /= prob_sum 
        idx = self.rng.choice(self.M,size=self.M,p=prob) 
        self.config = self.config[idx]
        self.config[:,-2] = np.sign(p[idx])
        self.config[:,-1] = 1

        f = ws[0]/self.M/beta
        self.f.append(f)
        return False
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
        self.config = load(fname)[0] 
        self.M = self.config.shape[0] # num walkers
        print(f'number of walkers=',self.M)
class SamplerBranch(SamplerSR):
    def __init__(self,wk,nmin,nmax,branch=False,seed=None):
        self.wk = wk
        self.nsite = wk.af.nsite 
        self.nmin = nmin
        self.nmax = nmax
    
        self.rng = np.random.default_rng(seed=seed)
        self.thresh = 1e-10
        self.progbar = False 
        self.branch = branch
    def compute_energy(self):
        e = self.compute_expectation(self.e)
        e[:,0] += self.wk.shift
        e /= self.nsite
        w = self.config[:,-2:]
        rms = np.sqrt(np.sum(w**2,axis=0)/self.M)
        print(f'step={self.step},e={e[0,0]},err={e[0,1]},e_eff={e[1,0]},err_eff={e[1,1]},rms={rms}')
    def compute_expectation(self,e):
        E = np.zeros((2,2))
        E[0] = blocking_analysis(e,weights=self.config[:,-2])
        E[1] = blocking_analysis(e,weights=self.config[:,-1])
        return E
    def SR(self):
        if RANK!=0:
            return
        w = self.config[:,-1]
        wmax = max(w)
        wmean = w.sum()/self.M
        print('\tweff max,min,ave=',wmax,min(w),wmean)
        if not self.branch:
            return False

        #idx = np.nonzero(w > wmax * self.thresh)[0]
        #self.config = self.config[idx] 
        #self.config[:,-2:] /= wmax 
        #self.M = len(idx)
        #if self.M >= self.nmin:
        #    return

        #w = self.config[:,-1]
        #idx = np.argsort(w)
        #self.config = self.config[idx]
        #N = self.nmax - self.M 
        #self.config[-N:,-2:] /= 2
        #self.config = np.concatenate([self.config,self.config[-N:]],axis=0)

        config_new = []
        for ix in range(self.M):
            config = self.config[ix] 
            m = int(config[-1]+self.rng.random())
            if m==0:
                #if self.M < self.nmin:
                #    config_new.append(config)
                continue
            if m==1:
                config_new.append(config)
                continue
            #if self.M > self.nmax:
            #    config_new.append(config)
            #    continue
            config[-2:] /= m
            config_new += [config.copy() for _ in range(m)]
        self.config = np.array(config_new)
        self.M = self.config.shape[0]
        print(f'\tnumber of walkers=',self.M)
        if self.M > self.nmax:
            return True
        if self.M < self.nmin:
            return True
        return False
    def save(self,tmpdir):
        if RANK!=0:
            return
        np.save(tmpdir+f'step{self.step}.npy',self.config)
