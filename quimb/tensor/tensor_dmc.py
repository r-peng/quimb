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
    def __init__(self,config,weight,tau,shift,scheme,seed=None):
        self.config = config
        self.tau = tau
        self.shift = shift 

        self.rng = np.random.default_rng(seed=seed)
        self.seed = seed
        self.weight = weight

        self.scheme = scheme 
    def sample(self,af):
        af.config = self.config 
        af.cx = dict()
        if self.scheme==1:
            return self.sample1(af)  
        elif self.scheme==2:
            return self.sample2(af)
        elif self.scheme==3:
            return self.sample3(af)
        else:
            raise NotImplementedError
    def sample1(self,af):
        # no fix-node, use abs for p
        e = {self.config:af.model.compute_local_energy_eigen(self.config)}
        for batch_key in af.model.batched_pairs:
           e.update(af.batch_pair_energies(batch_key,False)[0])

        gf = {key:-self.tau * val for key,val in e.items()}
        gf[self.config] += 1 + self.tau * self.shift

        keys = list(gf.keys())
        prob = np.array(list(gf.values()))
        norm = sum(p) 
        prob = np.fabs(prob)
        prob /= norm
        self.config = tuple(self.rng.choice(keys,p=prob))

        w = self.weight
        e = sum(e.values())
        self.weight *= np.sign(gf[self.config]) * norm
        return e,w
    def sample2(self,af):
        # An & Leeuwen, Phys.Rev.B,44(1991) 
        # energy still computed with H itself
        e = dict()
        u = af.model.compute_local_energy_eigen(self.config)
        for batch_key in af.model.batched_pairs:
           e.update(af.batch_pair_energies(batch_key,False)[0])

        # Heff
        gf = {key:-self.tau * val for key,val in e.items() if val < 0}
        gf[self.config] = 1 - self.tau * (u - self.shift)

        keys = list(gf.keys())
        prob = np.array(list(gf.values()))
        norm = sum(prob) 
        prob /= norm
        self.config = tuple(self.rng.choice(keys,p=prob))

        w = self.weight
        e = sum(e.values()) + u
        self.weight *= norm
        return e,w
    def sample3(self,af):
        # Haaf & Bemmel & Leeuwen & Saarloos, Phys.Rev.B,51(1995) 
        e = dict()
        u = af.model.compute_local_energy_eigen(self.config)
        for batch_key in af.model.batched_pairs:
           e.update(af.batch_pair_energies(batch_key,False)[0])

        # Heff
        ep = sum([val for val in e.values() if val > 0])
        gf = {key:-self.tau * val for key,val in e.items() if val < 0}
        gf[self.config] = 1 - self.tau * (u + ep - self.shift)
        if gf[self.config] < 0:
            raise ValueError(f'diagonal term = {gf[self.config]}')

        keys = list(gf.keys())
        prob = np.array(list(gf.values()))
        norm = sum(prob) 
        prob /= norm
        self.config = tuple(self.rng.choice(keys,p=prob))

        w = self.weight
        e = sum(e.values()) + u
        self.weight *= norm
        return e,w
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
        self.buf = np.zeros(2)
    def sample(self,fname=None):
        self.redistribute()
        if RANK==0:
            self._ctr()
        else:
            self._sample()
            if fname is not None:
                self.save(fname)
    def _ctr(self):
        if self.progbar:
            pg = Progbar(total=self.ntot)
        n = 0
        e = []
        w = []
        while True:
            if n>= self.ntot:
                break
            COMM.Recv(self.buf,tag=2)
            e.append(self.buf[0]) 
            w.append(self.buf[1]) 
            n += 1
            pg.update()

        # compute energy
        self.step += 1
        if self.step%self.print_energy_every!=0:
            return
        e = np.array(e)
        w = np.array(w)
        num,num_err = blocking_analysis(e,weights=w) 
        denom,denom_err = blocking_analysis(w) 
        E = num / denom 
        dE = 0 if self.E is None else E - self.E 
        print(f'step={self.step},E={E/self.nsite},dE={dE/self.nsite},num_err={num_err/self.nsite},denom_err={denom_err/self.nsite}')
        self.E = E 
        rms = np.sqrt(np.dot(w,w)/len(w))
        print(f'rms(w)={rms},min={min(w)},max={max(w)}')
    def _sample(self):
        for i in range(self.configs.shape[0]):
            wk = Walker(tuple(self.configs[i,:]),self.weights[i],self.tau,self.shift,self.scheme)
            for _ in range(self.accum):
                e,w = wk.sample(self.af)
            self.buf[0] = e
            self.buf[1] = w
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
         
