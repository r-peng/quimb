import time,h5py,itertools,pickle
import numpy as np
import scipy.sparse.linalg as spla
from scipy.optimize import line_search

from quimb.utils import progbar as Progbar
from .utils import load_ftn_from_disc,write_ftn_to_disc
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
DEFAULT_RATE_MIN = 1e-2
DEFAULT_RATE_MAX = 1e-1
DEFAULT_COND_MIN = 1e-3
DEFAULT_COND_MAX = 1e-3
DEFAULT_NUM_STEP = 1e-6
PRECISION = 1e-10
CG_TOL = 1e-4
class TNVMC: # stochastic sampling
    def __init__(
        self,
        ham,
        sampler,
        amplitude_factory,
        #conditioner='auto',
        conditioner=None,
        optimizer='sr',
        extrapolator=None,
        search_rate=None,
        search_cond=False,
        **kwargs,
    ):
        # parse ham
        self.ham = ham

        # parse sampler
        self.config = None
        self.batchsize = None
        self.sampler = sampler
        self.dense_sampling = sampler.dense
        self.exact_sampling = sampler.exact

        # parse wfn 
        self.amplitude_factory = amplitude_factory         
        self.x = self.amplitude_factory.get_x()

        # TODO: if need to condition, try making the element of psi-vec O(1)
#        if conditioner == 'auto':
#            def conditioner(psi):
#                psi.equalize_norms_(1.0)
#            self.conditioner = conditioner
#        else:
#            self.conditioner = None
        self.conditioner = None
        if self.conditioner is not None:
            self.conditioner(self.x)

        # parse gradient optimizer
        self.optimizer = optimizer
        self.compute_Hv = False
        if self.optimizer=='rgn':
            self.compute_Hv = kwargs.get('compute_Hv',True) 
            if self.compute_Hv:
                self.ham.initialize_pepo(self.amplitude_factory.psi)
            else:
                self.num_step = kwargs.get('num_step',DEFAULT_NUM_STEP)

        # parse extrapolator
        self.extrapolator = extrapolator 
        self.extrapolate_direction = kwargs.get('extrapolate_direction',True)
        if self.extrapolator=='adam':
            self.beta1 = kwargs.get('beta1',.9)
            self.beta2 = kwargs.get('beta2',.999)
            self.eps = kwargs.get('eps',1e-8)
            self._ms = None
            self._vs = None
        if self.extrapolator=='diis':
            from .diis import DIIS
            self.diis = DIIS()
            self.diis_start = kwargs.get('diis_start',0) 
            self.diis_every = kwargs.get('diis_every',1)
            self.diis_size  = kwargs.get('diis_size',10)
            self.diis.space = self.diis_size

        # TODO: not sure how to do line search
        self.search_rate = search_rate 
        self.search_cond = search_cond
    def run(self,start,stop,tmpdir=None,
            rate_min=DEFAULT_RATE_MIN,
            rate_max=DEFAULT_RATE_MAX,
            cond_min=DEFAULT_COND_MIN,
            cond_max=DEFAULT_COND_MAX,
            rate_itv=None, # prapagate rate over rate_itv
            cond_itv=None, # propagate cond over cond_itv
        ):
        # change rate & conditioner as in Webber & Lindsey
        self.start = start
        self.stop = stop
        steps = stop - start
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.rate_itv = steps if rate_itv is None else rate_itv
        self.rate_base = (self.rate_max/self.rate_min)**(1./self.rate_itv)
        self.cond_min = cond_min
        self.cond_max = cond_max
        self.cond_itv = steps if cond_itv is None else cond_itv
        self.cond_base = (self.rate_max/self.rate_min)**(1./self.rate_itv)
        self.rate = self.rate_min
        self.cond = self.cond_min 
        if RANK==0:
            print('rate_min=', self.rate_min)
            print('rate_max=', self.rate_max)
            print('rate_itv=', self.rate_itv)
            print('rate_base=',self.rate_base)
            print('cond_min=', self.cond_min)
            print('cond_max=', self.cond_max)
            print('cond_itv=', self.cond_itv)
            print('cond_base=',self.cond_base)
        self.delta_norm = np.zeros(1)
        for step in range(start,stop):
            self.step = step
            self.sample()
            if RANK==self.cix:
                self.propagate_rate_cond()
                self.transform_gradients()
                self.regularize()
                self.extrapolate()
                if self.conditioner is not None:
                    self.conditioner(self.x)
                print('\tx norm=',np.linalg.norm(self.x))
            COMM.Bcast(self.x,root=self.cix) 
            COMM.Bcast(self.delta_norm,root=self.cix) 
            psi = self.amplitude_factory.update(self.x)
            if RANK==0:
                if tmpdir is not None: # save psi to disc
                    write_ftn_to_disc(psi,tmpdir+f'psi{step+1}',provided_filename=True)
    def regularize(self):
        delta_norm = np.array([np.linalg.norm(self.deltas)])
        print(f'\tdelta norm={delta_norm[0]}')
        if self.step == self.start:
            self.delta_norm = delta_norm
            return 
        ratio = delta_norm / self.delta_norm
        cnt = 0
        while ratio[0] > 2.:
            self.deltas /= 2.
            delta_norm /= 2.
            ratio /= 2. 
            cnt += 1
            if cnt>10:
                raise ValueError
        self.delta_norm = delta_norm
        if cnt>0:
            print(f'\tregularized delta norm={delta_norm[0]}')
            self.rate = self.rate_min
            self.cond = self.cond_min
    def propagate_rate_cond(self):
        if self.step < self.start + self.rate_itv:
            self.rate *= self.rate_base
        if self.step < self.start + self.cond_itv:
            self.cond *= self.cond_base
        print('\trate=',self.rate)
        print('\tcond=',self.cond)
    def extrapolate(self):
        if self.extrapolator is None:
            self.x -= self.rate * self.deltas
            return
        g =  self.deltas if self.extrapolate_direction else self.g
        if self.extrapolator=='adam':
            self._extrapolate_adam(g)
        elif self.extrapolator=='diis':
            self._extrapolate_diis(g)
        else:
            raise NotImplementedError
    def _extrapolate_adam(self,g):
        if self.step == 0:
            self._ms = np.zeros_like(g)
            self._vs = np.zeros_like(g)
    
        self._ms = (1.-self.beta1) * g + self.beta1 * self._ms
        self._vs = (1.-self.beta2) * g**2 + self.beta2 * self._vs 
        mhat = self._ms / (1. - self.beta1**(self.step+1))
        vhat = self._vs / (1. - self.beta2**(self.step+1))
        deltas = mhat / (np.sqrt(vhat)+self.eps)
        self.x -= self.rate * deltas 
        print('\tAdam delta norm=',np.linalg.norm(deltas))
        print('\tAdam beta ratio=',(1.-self.beta1)/np.sqrt(1.-self.beta2))
    def _extrapolate_diis(self,g):
        self.x -= self.rate * self.deltas
        if self.step < self.diis_start: # skip the first couple of updates
            return
        if (self.step - self.diis_start) % self.diis_every != 0: # space out 
            return
        xerr = g
        # add perturbation
        #gmax = np.amax(np.fabs(xerr))
        #print('gmax=',gmax)
        #pb = np.random.normal(size=len(xerr))
        #eps = .1
        #xerr += eps*gmax*pb
        #
        self.x = self.diis.update(self.x,xerr=xerr)
        #print('\tDIIS error vector norm=',np.linalg.norm(e))  
        print('\tDIIS extrapolated x norm=',np.linalg.norm(self.x))  
    def update_local(self,config):
        ex,vx,Hvx = self.ham.compute_local_energy(config,self.amplitude_factory)
        self.elocal.append(ex)
        self.vlocal.append(vx)
        if self.compute_Hv:
            self.Hv_local.append(Hvx)
    def sample(self):
        self.sampler.amplitude_factory = self.amplitude_factory
        if self.exact_sampling:
            self.sample_exact()
        else:
            self.sample_stochastic()
    def sample_stochastic(self): 
        # randomly select a process to be the control process
        self.cix = np.random.randint(low=0,high=SIZE,size=1)
        COMM.Bcast(self.cix,root=0)
        self.cix = self.cix[0]

        self.terminate = np.array([0])
        self.rix = np.array([RANK])
        if RANK==self.cix:
            self._ctr()
        else:
            self._sample()
    def _ctr(self):
        print('\tcontrol rank=',RANK)
        t0 = time.time()
        ncurr = 0
        ntotal = self.batchsize * SIZE
        tdest = list(set(range(SIZE)).difference({RANK}))
        while self.terminate[0]==0:
            COMM.Recv(self.rix,tag=0)
            ncurr += 1
            if ncurr > ntotal: # send termination message to all workers
                self.terminate[0] = 1
                for worker in tdest:
                    COMM.Bsend(self.terminate,dest=worker,tag=1)
            else:
                COMM.Bsend(self.terminate,dest=self.rix[0],tag=1)
        print('\tstochastic sample time=',time.time()-t0)

        # gather sample sizes
        sendbuf = np.array([0])
        recvbuf = np.array([0]*SIZE)
        COMM.Gather(sendbuf,recvbuf,root=self.cix)

        t0 = time.time()
        self.recv(sources=tdest,sizes=recvbuf)
        self.f = np.concatenate(self.f,axis=0) 
        self.e = np.concatenate(self.e,axis=0) 
        self.v = np.concatenate(self.v,axis=0) 
        self.extract_energy_gradient()
        print('\tcollect data time=',time.time()-t0)
    def _sample(self):
        self.sampler.preprocess(self.config) 

        self.samples = []
        self.flocal = dict()
        self.elocal = []
        self.vlocal = []
        if self.compute_Hv:
            self.Hv_local = []

        while self.terminate[0]==0:
            self.config,omega = self.sampler.sample()
            #if omega < 1e-10:
            #    raise ValueError(f'omega={omega}')
            if self.config in self.flocal:
                self.flocal[self.config] += 1
            else:
                self.flocal[self.config] = 1
                self.samples.append(self.config)
                self.update_local(self.config)

            COMM.Bsend(self.rix,dest=self.cix,tag=0) 
            COMM.Recv(self.terminate,source=self.cix,tag=1)

        # gather sample sizes
        sendbuf = np.array([len(self.samples)])
        recvbuf = np.array([0]*SIZE)
        COMM.Gather(sendbuf,recvbuf,root=self.cix)

        self.flocal = np.array([self.flocal[config] for config in self.samples])
        self.elocal = np.array(self.elocal)
        self.vlocal = np.array(self.vlocal)
        if self.compute_Hv:
            self.Hv_local = np.array(self.Hv_local)
        self.send(dest=self.cix)
    def sample_exact(self): 
        self.sampler.compute_dense_prob() # runs only for dense sampler 

        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros

        self.flocal = []
        self.samples = []
        self.elocal = []
        self.vlocal = []
        if self.compute_Hv:
            self.Hv_local = [] 

        t0 = time.time()
        for ix in ixs:
            self.flocal.append(p[ix])
            self.config = all_configs[ix]
            self.samples.append(self.config) 
            self.update_local(self.config)
        if RANK==SIZE-1:
            print('\texact sample time=',time.time()-t0)

        self.cix = SIZE-1 
        sendbuf = np.array([len(self.samples)])
        recvbuf = np.array([0]*SIZE)
        COMM.Gather(sendbuf,recvbuf,root=self.cix)

        self.flocal = np.array(self.flocal)
        self.elocal = np.array(self.elocal)
        self.vlocal = np.array(self.vlocal)
        if self.compute_Hv:
            self.Hv_local = np.array(self.Hv_local)
        t0 = time.time()
        if RANK<self.cix:
            self.send(dest=self.cix)
            return
        self.recv(sources=range(self.cix),sizes=recvbuf)  
        self.f.append(self.flocal)
        self.e.append(self.elocal)
        self.v.append(self.vlocal)
        if self.compute_Hv:
            self.Hv.append(self.Hv_local)
        self.f = np.concatenate(self.f,axis=0) 
        self.e = np.concatenate(self.e,axis=0) 
        self.v = np.concatenate(self.v,axis=0) 
        self.extract_energy_gradient()
        print('\tcollect data time=',time.time()-t0)
    def extract_energy_gradient(self):
        if self.exact_sampling:
            fe = self.f * self.e
            self.E,self.Eerr,self.n = np.sum(fe),0.,1.
        else:
            self.E,self.Eerr,fe,self.n = blocking_analysis(self.f,self.e,0,True)
        self.vmean = np.dot(self.f,self.v) / self.n  
        self.g = np.dot(fe,self.v) / self.n - self.vmean * self.E
        gmax = np.amax(np.fabs(self.g))
        print(f'step={self.step},energy={self.E},err={self.Eerr},gmax={gmax}')
    def send(self,dest):
        COMM.Ssend(self.flocal,dest=dest,tag=2)
        COMM.Ssend(self.elocal,dest=dest,tag=3)
        COMM.Ssend(self.vlocal,dest=dest,tag=4)
        if self.compute_Hv:
            COMM.Ssend(self.Hv_local,dest=dest,tag=5)
    def recv(self,sources,sizes):
        self.f = []
        self.e = []
        self.v = []
        if self.compute_Hv:
            self.Hv = [] 
        fdtype = float if self.exact_sampling else int
        for worker in sources:
            nlocal = sizes[worker]
            buf = np.zeros(nlocal,dtype=fdtype) 
            COMM.Recv(buf,source=worker,tag=2)
            self.f.append(buf)    

            buf = np.zeros(nlocal) 
            COMM.Recv(buf,source=worker,tag=3)
            self.e.append(buf)    

            buf = np.zeros((nlocal,len(self.x)))
            COMM.Recv(buf,source=worker,tag=4)
            self.v.append(buf)    
            if self.compute_Hv:
                buf = np.zeros((nlocal,len(self.x)))
                COMM.Recv(buf,source=worker,tag=5)
                self.Hv.append(buf)    
    def getS(self):
        def S(x):
            Sx1 = np.dot(self.f*np.dot(self.v,x),self.v) / self.n
            Sx2 = self.vmean * np.dot(self.vmean,x)
            return Sx1-Sx2
        return S
    def getH(self):
        Hv = np.concatenate(self.Hv,axis=0) 
        Hv_mean = np.dot(self.f,Hv) / self.n 
        def H(x):
            Hx1 = np.dot(self.f*np.dot(Hv,x),self.v) / self.n
            Hx2 = self.vmean * np.dot(Hv_mean,x)
            Hx3 = self.g * np.dot(self.vmean,x)
            return Hx1-Hx2-Hx3
        return H
    def transform_gradients(self):
        if self.optimizer=='sr':
            self._transform_gradients_sr()
        elif self.optimizer=='rgn':
            self._transform_gradients_rgn()
        elif self.optimizer=='lin':
            raise NotImplementedError
        else:
            self._transform_gradients_sgd()
    def _transform_gradients_sgd(self):
        g = self.g
        if self.optimizer=='sgd':
            self.deltas = g
        elif self.optimizer=='sign':
            self.deltas = np.sign(g)
        elif self.optimizer=='signu':
            self.deltas = np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError
    def _transform_gradients_sr(self):
        t0 = time.time()
        sh = len(self.g)
        S = self.getS()
        def A(vec):
            return S(vec) + self.cond * vec
        LinOp = spla.LinearOperator((sh,sh),matvec=A)
        self.deltas,info = spla.minres(LinOp,self.g,tol=CG_TOL)
        print('\tSR solver time=',time.time()-t0)
        print('\tSR solver exit status=',info)
    def _transform_gradients_rgn(self):
        if self.compute_Hv:
            self._transform_gradients_rgn_WL()
        else:
            raise NotImplementedError
    def _transform_gradients_rgn_WL(self):
        t0 = time.time()
        sh = len(self.g)
        S = self.getS()
        H = self.getH()
        def A(vec):
            return H(vec) - self.E * S(vec) + self.cond * vec 
        LinOp = spla.LinearOperator((sh,sh),matvec=A)
        self.deltas,info = spla.lgmres(LinOp,self.g,tol=CG_TOL)
        print('\tRGN solver time=',time.time()-t0)
        print('\tRGN solver exit status=',info)

def blocking_analysis(weights, energies, neql, printQ=False):
    nSamples = weights.shape[0] - neql
    weights = weights[neql:]
    energies = energies[neql:]
    weightedEnergies = np.multiply(weights, energies)
    sumWeights = weights.sum()
    meanEnergy = weightedEnergies.sum() / sumWeights 
    if printQ:
        print(f'\nMean energy: {meanEnergy:.8e}')
        print('Block size    # of blocks        Mean                Error')
    blockSizes = np.array([ 1, 2, 5, 10, 20, 50, 70, 100, 200, 500 ])
    prevError = 0.
    plateauError = None
    for i in blockSizes[blockSizes < nSamples/2.]:
        nBlocks = nSamples//i
        blockedWeights = np.zeros(nBlocks)
        blockedEnergies = np.zeros(nBlocks)
        for j in range(nBlocks):
            blockedWeights[j] = weights[j*i:(j+1)*i].sum()
            blockedEnergies[j] = weightedEnergies[j*i:(j+1)*i].sum() / blockedWeights[j]
        v1 = blockedWeights.sum()
        v2 = (blockedWeights**2).sum()
        mean = np.multiply(blockedWeights, blockedEnergies).sum() / v1
        error = (np.multiply(blockedWeights, (blockedEnergies - mean)**2).sum() / (v1 - v2 / v1) / (nBlocks - 1))**0.5
        if printQ:
            print(f'  {i:4d}           {nBlocks:4d}       {mean:.8e}       {error:.6e}')
        if error < 1.05 * prevError and plateauError is None:
            plateauError = max(error, prevError)
        prevError = error

    if printQ:
        if plateauError is not None:
            print(f'Stocahstic error estimate: {plateauError:.6e}\n')

    return meanEnergy, plateauError, weightedEnergies, sumWeights
