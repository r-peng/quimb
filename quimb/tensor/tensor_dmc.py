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
    def __init__(self,config,weight,tau,gamma,seed=None):
        self.config = config
        self.weight = weight
        self.tl = tau
        self.gamma

        self.rng = np.random.default_rng(seed=seed)
        self.terminate = False
    def sample(self,af):
        af.config = self.config 
        af.cx = dict()

        # (a)
        h = dict()
        u = af.model.compute_local_energy_eigen(self.config)
        for batch_key in af.model.batched_pairs:
           h.update(af.batch_pair_energies(batch_key,False)[0])
        e = sum(h.values()) + u
        vsf = sum([val for val in h.values() if val > 0])
        ueff = u + (1+self.gamma) * vsf

        xi = self.rng.random()
        pid = e - ueff
        td = min(self.tl,np.log(xi)/pid)

        # (b)
        self.tl -= td
        self.weight[0] *= np.exp((-e+(1+self.gamma)*vsf)*td)
        self.weight[1] *= np.exp(-e*td)

        if self.tl <= 0:
            self.terminate = True
            return e 
        keys = list(h.keys())
        prob =  - np.array(list(h.values()))
        prob = np.where(prob>0,prob,-self.gamma*prob)
        sign = np.where(prob>0,1,-1./self.gamma)
        prob /= sum(prob) 
        idx = self.rng.choice(len(prob),p=prob)
        self.weight[0] *= sign[idx] 
        self.config = keys[idx]
        return e 
class Sampler:
    def __init__(self,af,configs,weights,seed=None):
        self.af = af
        self.nsite = af.nsite 

        self.configs = configs
        self.weights = weights

        self.rng = np.random.default_rng(seed=seed)
        self.scheme = None 
        self.fix_accum = True 
        self.progbar = False 
        self.print_energy_every = 1
        self.step = 0  
        self.E = None 
        self.buf = np.zeros(3)
    def sample(self,fname=None):
        self.redistribute()
        if RANK==0:
            self._ctr()
        else:
            self._sample()
            if fname is not None:
                self.save(fname)
    def SR(self,v,weight,weight_eff):
        sum_weight = weight.sum()
        sum_weight_eff = weight_eff.sum()
        S = (v * v * weight_eff).sum() / sum_weight_eff
        b = np.dot(weight,v) / sum_weight
        alpha = b/S
        p = weight_eff * (1 + alpha*v)
        return np.fabs(p)
    def _ctr(self):
        if self.progbar:
            pg = Progbar(total=self.ntot)
        n = 0
        energy = []
        weight = []
        weight_eff = []
        t0 = time.time()
        while True:
            if n>= self.ntot:
                break
            COMM.Recv(self.buf,tag=2)
            energy.append(self.buf[0]) 
            weight.append(self.buf[1]) 
            weight_eff.append(self.buf[2]) 
            n += 1
            if self.progbar:
                pg.update()
        print('sample time=',time.time()-t0)

        # compute energy
        self.step += 1
        if self.step%self.print_energy_every!=0:
            return
        energy = np.array(energy)
        weight = np.array(weight)
        weight_eff = np.array(weight_eff)

        num,num_err = blocking_analysis(energy,weights=weight) 
        denom,denom_err = blocking_analysis(weight) 
        E = num / denom 
        dE = 0 if self.E is None else E - self.E 
        self.E = E 

        # SR
        num,num_err = blocking_analysis(energy,weights=weight_eff) 
        denom,denom_err = blocking_analysis(weight_eff) 
        Eeff = num / denom 
        v = energy - Eeff
        p = self.SR(v,weight,weight_eff)
        

        print(f'step={self.step},E={E/self.nsite},dE={dE/self.nsite},num_err={num_err/self.nsite},denom_err={denom_err/self.nsite}')
        print(f'rms(w)={rms},min={min(w)},max={max(w)}')
    def _sample(self):
        for i in range(self.configs.shape[0]):
            wk = Walker(tuple(self.configs[i,:]),self.weights[i,:],self.tau,self.gamma)
            while True:
                if wk.terminate:
                    break
                e = wk.sample(af)
            self.buf[0] = e
            self.buf[1] = wk.weight[0]
            self.buf[2] = wk.weight[1]
            COMM.Send(self.buf,dest=0,tag=2)
            self.configs[i,:] = np.array(wk.config)
            self.weights[i] = wk.weight

        # branching
        configs = []
        weights = []
        for i in range(self.configs.shape[0]): 
            m = int(np.fabs(self.weights[i]) + self.rng.random())
            if m == 0:
                continue
            else:
                config = self.configs[i,:]
                weight = self.weights[i] / m
                for _ in range(m):
                    configs.append(config)
                    weights.append(weight)
        self.configs = np.array(configs)
        self.weights = np.array(weights)
    def redistribute(self):
        pop = 0 if RANK==0 else self.configs.shape[0] 
        pop = np.array([pop])
        tot = np.zeros(SIZE,dtype=int)
        COMM.Gather(pop,tot,root=0)
        if RANK==0:
            disp = np.cumsum(tot) 
            self.ntot = disp[-1]
            print('population=',self.ntot)
            configs = np.zeros((self.ntot,self.nsite),dtype=int)
            weights = np.zeros(self.ntot) 
            for rank in range(1,SIZE):
                start,stop = disp[rank-1],disp[rank]
                COMM.Recv(configs[start:stop,:],source=rank,tag=0) 
                COMM.Recv(weights[start:stop],source=rank,tag=1) 
        else:
            COMM.Send(self.configs,dest=0,tag=0) 
            COMM.Send(self.weights,dest=0,tag=1) 

        if RANK==0:
            pop,remain = self.ntot//(SIZE-1),self.ntot%(SIZE-1)
            tot = np.array([0] + [pop]*(SIZE-1))
            if remain > 0:
                tot[-remain:] += 1
        COMM.Bcast(tot,root=0)

        if RANK==0:
            disp = np.cumsum(tot) 
            for rank in range(1,SIZE):
                start,stop = disp[rank-1],disp[rank]
                COMM.Send(configs[start:stop,:],dest=rank,tag=0) 
                COMM.Send(weights[start:stop],dest=rank,tag=1) 
        else:
            pop = tot[RANK]
            self.configs = np.zeros((pop,self.nsite),dtype=int)
            self.weights = np.zeros(pop)
            COMM.Recv(self.configs,source=0,tag=0) 
            COMM.Recv(self.weights,source=0,tag=1) 
    def save(self,fname):
        np.save(fname+f'_RANK{RANK}_configs.npy',self.configs)
        np.save(fname+f'_RANK{RANK}_weights.npy',self.weights)
         
