import time,scipy,functools,h5py
import numpy as np
import scipy.sparse.linalg as spla
from .tfqmr import tfqmr

from quimb.utils import progbar as Progbar
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
DISCARD = 1e3
CG_TOL = 1e-4
MAXITER = 100
#MAXITER = 2
##################################################################################################
# VMC utils
##################################################################################################
def _rgn_block_solve(H,E,S,g,cond):
    # hessian 
    hess = H - E * S
    # smallest eigenvalue
    w = np.linalg.eigvals(hess)
    idx = np.argmin(w.real)
    wmin = w[idx]
    # solve
    deltas = np.linalg.solve(hess+max(0.,cond-wmin.real)*np.eye(len(g)),g)
    # compute model energy
    dE = - np.dot(deltas,g) + .5 * np.dot(deltas,np.dot(hess,deltas))
    return wmin,deltas,dE
def _lin_block_solve(H,E,S,g,Hvmean,vmean,cond):
    Hi0 = g
    H0j = Hvmean - E * vmean
    sh = len(g)

    A = np.block([[np.array([[E]]),H0j.reshape(1,sh)],
                  [Hi0.reshape(sh,1),H]])
    B = np.block([[np.ones((1,1)),np.zeros((1,sh))],
                  [np.zeros((sh,1)),S+cond*np.eye(sh)]])
    w,v = scipy.linalg.eig(A,b=B) 
    w,deltas,idx = _select_eigenvector(w.real,v.real)
    return w,deltas,v[0,idx],np.linalg.norm(v[:,idx].imag)
def _select_eigenvector(w,v):
    #if min(w) < self.E - self.revert:
    #    dist = (w-self.E)**2
    #    idx = np.argmin(dist)
    #else:
    #    idx = np.argmin(w)
    z0_sq = v[0,:] ** 2
    idx = np.argmax(z0_sq)
    #v = v[1:,idx]/v[0,idx]
    v = v[1:,idx]/np.sign(v[0,idx])
    return w[idx],v,idx
def blocking_analysis(weights, energies, neql, printQ=False):
    nSamples = weights.shape[0] - neql
    weights = weights[neql:]
    energies = energies[neql:]
    weightedEnergies = np.multiply(weights, energies)
    meanEnergy = weightedEnergies.sum() / weights.sum()
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

    print(RANK,plateauError,error)
    if plateauError is None:
        plateauError = error
    else:
        if printQ:
            print(f'Stocahstic error estimate: {plateauError:.6e}\n')

    return meanEnergy, plateauError
##################################################################################################
# VMC engine 
##################################################################################################
class TNVMC: # stochastic sampling
    def __init__(
        self,
        ham,
        sampler,
        amplitude_factory,
        normalize=False,
        optimizer='sr',
        solve_full=True,
        solve_dense=False,
        **kwargs,
    ):
        # parse ham
        self.ham = ham
        self.nsite = ham.Lx * ham.Ly

        # parse sampler
        self.sampler = sampler
        self.exact_sampling = sampler.exact

        # parse wfn 
        self.amplitude_factory = amplitude_factory         
        self.x = self.amplitude_factory.get_x()
        self.nparam = len(self.x)
        self.init_norm = None
        if normalize:
            self.init_norm = np.linalg.norm(self.x)    

        # parse gradient optimizer
        self.optimizer = optimizer
        self.solve_full = solve_full
        if not self.solve_full:
            self.block_dict = self.amplitude_factory.get_block_dict()
        self.solve_dense = solve_dense
        self.compute_Hv = False
        if self.optimizer in ['rgn','lin']:
            self.compute_Hv = True
        if self.optimizer=='rgn':
            solver = kwargs.get('solver','lgmres')
            self.solver = {'lgmres':spla.lgmres,
                           'tfqmr':tfqmr}[solver] 
        if self.optimizer=='lin':
            #self.xi = None
            self.xi = kwargs.get('xi',0.5)
            # only used for iterative full Hessian
            #self.solver = kwargs.get('solver','davidson')
            #if self.solver=='davidson':
            #    maxsize = kwargs.get('maxsize',25)
            #    maxiter = kwargs.get('maxiter',100)
            #    restart_size = kwargs.get('restart_size',5)
            #    from .davidson import davidson
            #    self.davidson = functools.partial(davidson,
            #        maxsize=maxsize,restart_size=restart_size,maxiter=maxiter,tol=CG_TOL) 

        # to be set before run
        self.tmpdir = None
        self.config = None
        self.omega = None
        self.batchsize = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR
        self.rate2 = None # rate for LIN,RGN
        self.cond1 = None
        self.cond2 = None
        self.check = None 
        self.accept_ratio = None
    def normalize(self,x):
        if self.init_norm is not None:
            norm = np.linalg.norm(x)
            x *= self.init_norm / norm    
        return x
    def run(self,start,stop,tmpdir=None):
        self.Eold = 0.
        for step in range(start,stop):
            self.step = step
            self.sample()
            self.extract_energy_gradient()
            self.transform_gradients()
            COMM.Bcast(self.x,root=0) 
            fname = None if tmpdir is None else tmpdir+f'psi{step+1}' 
            psi = self.amplitude_factory.update(self.x,fname=fname,root=0)
    def sample(self,samplesize=None,compute_v=True,compute_Hv=None):
        self.sampler.amplitude_factory = self.amplitude_factory
        #self.config,self.omega = self.sampler.preprocess(self.config) 
        self.sampler.preprocess(self.config) 
        compute_Hv = self.compute_Hv if compute_Hv is None else compute_Hv

        self.buf = np.zeros(5)
        self.terminate = np.array([0])

        self.buf[0] = RANK + .1
        self.samples = []
        if compute_v:
            self.evsum = np.zeros_like(self.x)
            self.vsum = np.zeros_like(self.x)
            self.v = []
        if compute_Hv:
            self.Hvsum = np.zeros_like(self.x)
            self.Hv = [] 

        if RANK==0:
            self._ctr(samplesize=samplesize)
        else:
            if self.exact_sampling:
                self._sample_exact(compute_v=compute_v,compute_Hv=compute_Hv)
            else:
                self._sample_stochastic(compute_v=compute_v,compute_Hv=compute_Hv)
    def _ctr(self,samplesize=None):
        if self.exact_sampling:
            samplesize = len(self.sampler.nonzeros)
        else:
            samplesize = self.batchsize if samplesize is None else samplesize
        self.f = []
        self.e = []
        err_mean = 0.
        err_max = 0.
        ncurr = 0
        t0 = time.time()
        while self.terminate[0]==0:
            COMM.Recv(self.buf,tag=0)
            rank = int(self.buf[0])
            self.f.append(self.buf[1]) 
            self.e.append(self.buf[2])
            err_mean += self.buf[3]
            err_max = max(err_max,self.buf[3])
            ncurr += 1
            #print(ncurr)
            if ncurr >= samplesize: # send termination message to all workers
                self.terminate[0] = 1
                for worker in range(1,SIZE):
                    COMM.Send(self.terminate,dest=worker,tag=1)
            else:
                COMM.Send(self.terminate,dest=rank,tag=1)
        print('\tsample time=',time.time()-t0)
        print('\tcontraction err=',err_mean / len(self.e),err_max)
    def _sample_stochastic(self,compute_v=True,compute_Hv=False):
        self.buf[1] = 1.

        self.store = dict()
        self.p0 = dict()
        while self.terminate[0]==0:
            config,omega = self.sampler.sample()
            #if omega > self.omega:
            #    self.config,self.omega = config,omega
            if config in self.store:
                ex,vx,Hvx = self.store[config]
            else:
                cx,ex,vx,Hvx,err = self.ham.compute_local_energy(
                    config,self.amplitude_factory,compute_v=compute_v,compute_Hv=compute_Hv)[:5]
                #if cx is None:
                #    raise ValueError
                if cx is None or np.fabs(ex) > DISCARD:
                    #for config,(exi,_,_) in self.store.items():
                    #    print(exi)
                    #raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
                    print(f'RANK={RANK},config={config},cx={cx},ex={ex}')
                    ex = 0.
                    if compute_v:
                        vx = np.zeros_like(vx)
                    if compute_Hv:
                        Hvx = np.zeros_like(Hvx)
                self.store[config] = ex,vx,Hvx
                self.p0[config] = cx**2
            self.samples.append(config)
            self.buf[2] = ex
            self.buf[3] = err
            if compute_v:
                self.vsum += vx
                self.evsum += vx * ex
                self.v.append(vx)
            if compute_Hv:
                self.Hvsum += Hvx
                self.Hv.append(Hvx)

            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
    def _sample_exact(self,compute_v=True,compute_Hv=None): 
        # assumes exact contraction
        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros
        ntotal = len(ixs)
        if RANK==SIZE-1:
            print('\tnsamples per process=',ntotal)

        self.f = []
        for ix in ixs:
            config = all_configs[ix]
            cx,ex,vx,Hvx,err = self.ham.compute_local_energy(config,self.amplitude_factory,
                                                        compute_v=compute_v,compute_Hv=compute_Hv)[:5]
            if cx is None:
                raise ValueError
            if np.fabs(ex)*p[ix] > DISCARD:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self.samples.append(config) 
            self.buf[1] = p[ix]
            self.f.append(p[ix])
            self.buf[2] = ex
            self.buf[3] = err
            if compute_v:
                self.vsum += vx * p[ix]
                self.evsum += vx * ex * p[ix]
                self.v.append(vx)
            if compute_Hv:
                self.Hvsum += Hvx * p[ix]
                self.Hv.append(Hvx)
            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        if self.optimizer in ['sr','rgn','lin']:
            self.extract_S()
        if self.optimizer in ['rgn','lin']:
            self.extract_H()
        if RANK==0:
            try:
                print(f'step={self.step},E={self.E/self.nsite},dE={(self.E-self.Eold)/self.nsite},err={self.Eerr/self.nsite},gmax={np.amax(np.fabs(self.g))}')
            except TypeError:
                print('E=',self.E)
                print('Eerr=',self.Eerr)
            print('\tcollect data time=',time.time()-t0)
            self.Eold = self.E
    def gather_sizes(self):
        self.count = np.array([0]*SIZE)
        COMM.Allgather(np.array([self.nlocal]),self.count)
        self.disp = np.concatenate([np.array([0]),np.cumsum(self.count[:-1])])
    def extract_energy(self):
        self.E = 0
        if RANK>0:
            return
        self.e = np.array(self.e)
        self.f = np.array(self.f)
        if self.exact_sampling:
            self.n = 1.
            self.E = np.dot(self.f,self.e)
            self.Eerr = 0.
        else:
            self.n = len(self.e)
            self.E,self.Eerr = blocking_analysis(self.f,self.e,0,True)
    def extract_gradient(self):
        self.vmean = np.zeros_like(self.x)
        COMM.Reduce(self.vsum,self.vmean,op=MPI.SUM,root=0)
        self.evmean = np.zeros_like(self.x)
        COMM.Reduce(self.evsum,self.evmean,op=MPI.SUM,root=0)
        self.g = None
        if RANK==0:
            self.vmean /= self.n
            self.evmean /= self.n
            self.g = self.evmean - self.E * self.vmean
    def _extract_Hvmean(self):
        self.Hvmean = np.zeros_like(self.x)
        COMM.Reduce(self.Hvsum,self.Hvmean,op=MPI.SUM,root=0)
        if RANK==0:
            self.Hvmean /= self.n
    def extract_S(self,solve_full=None,solve_dense=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        if RANK>0:
            self.v = np.array(self.v)
        fxn = self._get_Smatrix if solve_dense else self._get_S_iterative
        self.Sx1 = np.zeros_like(self.x)
        if solve_full:
            self.S = fxn() 
        else:
            self.S = [None] * self.nsite
            for ix,(start,stop) in enumerate(self.block_dict):
                self.S[ix] = fxn(start=start,stop=stop)
    def extract_H(self,solve_full=None,solve_dense=None):
        self._extract_Hvmean()
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        if RANK>0:
            self.Hv = np.array(self.Hv)
        fxn = self._get_Hmatrix if solve_dense else self._get_H_iterative
        self.Hx1 = np.zeros_like(self.x)
        if solve_full:
            self.H = fxn() 
        else:
            self.H = [None] * self.nsite
            for ix,(start,stop) in enumerate(self.block_dict):
                self.H[ix] = fxn(start=start,stop=stop)
    def _get_Smatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        if RANK==0:
            sh = stop-start
            vvsum_ = np.zeros((sh,)*2,dtype=self.x.dtype)
        else:
            v = self.v[:,start:stop] 
            if self.exact_sampling:
                self.f = np.array(self.f)
                vvsum_ = np.einsum('s,si,sj->ij',self.f,v,v)
            else:
                vvsum_ = np.dot(v.T,v)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean = self.vmean[start:stop]
            S = vvsum / self.n - np.outer(vmean,vmean)
        return S
    def _get_Hmatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        if RANK==0:
            sh = stop-start
            vHvsum_ = np.zeros((sh,)*2,dtype=self.x.dtype)
        else:
            v = self.v[:,start:stop] 
            self.Hv = np.array(self.Hv)
            Hv = self.Hv[:,start:stop] 
            if self.exact_sampling:
                vHvsum_ = np.einsum('s,si,sj->ij',self.f,v,Hv)
            else:
                vHvsum_ = np.dot(v.T,Hv)
        vHvsum = np.zeros_like(vHvsum_)
        COMM.Reduce(vHvsum_,vHvsum,op=MPI.SUM,root=0)
        H = None
        if RANK==0:
            Hvmean = self.Hvmean[start:stop]
            vmean = self.vmean[start:stop]
            g = self.g[start:stop]
            H = vHvsum / self.n - np.outer(vmean,Hvmean) - np.outer(g,vmean)
        return H
    def _get_S_iterative(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop 
        if RANK==0:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                COMM.Reduce(np.zeros_like(self.Sx1[start:stop]),self.Sx1[start:stop],op=MPI.SUM,root=0)     
                return self.Sx1[start:stop] / self.n \
                     - self.vmean[start:stop] * np.dot(self.vmean[start:stop],x)
        else: 
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                if self.exact_sampling:
                    Sx1 = np.dot(self.f * np.dot(self.v[:,start:stop],x),self.v[:,start:stop])
                else:
                    Sx1 = np.dot(np.dot(self.v[:,start:stop],x),self.v[:,start:stop])
                COMM.Reduce(Sx1,self.Sx1[start:stop],op=MPI.SUM,root=0)     
                return 0 
        return matvec
    def _get_H_iterative(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop 
        if RANK==0:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                COMM.Reduce(np.zeros_like(self.Hx1[start:stop]),self.Hx1[start:stop],op=MPI.SUM,root=0)     
                return self.Hx1[start:stop] / self.n \
                     - self.vmean[start:stop] * np.dot(self.Hvmean[start:stop],x) \
                     - self.g[start:stop] * np.dot(self.vmean[start:stop],x)
        else:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                if self.exact_sampling:
                    Hx1 = np.dot(self.f * np.dot(self.Hv[:,start:stop],x),self.v[:,start:stop])
                else:
                    Hx1 = np.dot(np.dot(self.Hv[:,start:stop],x),self.v[:,start:stop])
                COMM.Reduce(Hx1,self.Hx1[start:stop],op=MPI.SUM,root=0)     
                return 0 
        return matvec
    def transform_gradients(self):
        if self.optimizer=='sr':
            self._transform_gradients_sr()
        elif self.optimizer in ['rgn','lin']:
            self._transform_gradients_o2()
        else:
            self._transform_gradients_sgd()
        if RANK==0:
            print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(self.deltas)},dot={np.dot(self.deltas,self.g)},x={np.linalg.norm(self.x)}')
    def update(self,rate):
        return self.normalize(self.x - rate * self.deltas)
    def _transform_gradients_sgd(self):
        if RANK>0:
            return 
        g = self.g
        if self.optimizer=='sgd':
            self.deltas = g
        elif self.optimizer=='sign':
            self.deltas = np.sign(g)
        elif self.optimizer=='signu':
            self.deltas = np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError
        self.x = self.update(self.rate1)
    def _transform_gradients_sr(self,solve_dense=None,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        if solve_dense:
            self._transform_gradients_sr_dense(solve_full=solve_full)
        else:
            self._transform_gradients_sr_iterative(solve_full=solve_full)
    def _transform_gradients_rgn(self,solve_dense=None,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        if solve_dense:
            dE = self._transform_gradients_rgn_dense(solve_full=solve_full)
        else:
            dE = self._transform_gradients_rgn_iterative(solve_full=solve_full)
        return dE
    def _transform_gradients_sr_dense(self,solve_full=None):
        if RANK>0:
            return 
        t0 = time.time()
        self.deltas = np.zeros_like(self.x)
        solve_full = self.solve_full if solve_full is None else solve_full
        if solve_full:
            self.deltas = np.linalg.solve(self.S,self.g)
        else:
            for ix,(start,stop) in enumerate(self.block_dict):
                S = self.S[ix] + self.cond1 * np.eye(stop-start)
                self.deltas[start:stop] = np.linalg.solve(S,self.g[start:stop])
        print('\tSR solver time=',time.time()-t0)
        self.x = self.update(self.rate1)
        if self.tmpdir is not None:
            if self.solve_full:
                S = self.S
            else:
                S = np.zeros((self.nparam,self.nparam))
                for ix,(start,stop) in enumerate(self.block_dict):
                    S[start:stop,start:stop] = self.S[ix]
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('S',data=S) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
    def _transform_gradients_rgn_dense(self,solve_full=None):
        if RANK>0:
            return 
        t0 = time.time()
        self.deltas = np.zeros_like(self.x)
        solve_full = self.solve_full if solve_full is None else solve_full
        if solve_full:
            w,self.deltas,dE = _rgn_block_solve(self.H,self.E,self.S,self.g,self.cond2) 
        else:
            w = [None] * self.nsite
            dE = np.zeros(self.nsite)  
            for ix,(start,stop) in enumerate(self.block_dict):
                w[ix],self.deltas[start:stop],dE[ix] = \
                    _rgn_block_solve(self.H[ix],self.E,self.S[ix],self.g[start:stop],self.cond2)
            w = min(np.array(w).real)
        print(f'\tRGN solver time={time.time()-t0},least eigenvalue={w}')
        if self.tmpdir is not None:
            if self.solve_full:
                H = self.H
                S = self.S
            else:
                H = np.zeros((self.nparam,self.nparam))
                S = np.zeros((self.nparam,self.nparam))
                for ix,(start,stop) in enumerate(self.block_dict):
                    H[start:stop,start:stop] = self.H[ix] 
                    S[start:stop,start:stop] = self.S[ix]
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('H',data=H) 
            f.create_dataset('S',data=S) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
        return np.sum(dE)
    def solve_iterative(self,A,b,cond,symm):
        self.terminate = np.array([0])
        deltas = np.zeros_like(b)
        sh = len(b)
        if RANK==0:
            t0 = time.time()
            def _A(x):
                return A(x) + cond * x
            LinOp = spla.LinearOperator((sh,sh),matvec=_A,dtype=b.dtype)
            if symm:
                deltas,info = spla.minres(LinOp,b,tol=CG_TOL,maxiter=MAXITER)
            else: 
                deltas,info = self.solver(LinOp,b,tol=CG_TOL,maxiter=MAXITER)
            self.terminate[0] = 1
            COMM.Bcast(self.terminate,root=0)
            print(f'\tsolver time={time.time()-t0},exit status={info}')
        else:
            nit = 0
            while self.terminate[0]==0:
                nit += 1
                #self.terminate = A(buf)
                A(deltas)
            if RANK==1:
                print('niter=',nit)
        return deltas
    def _transform_gradients_sr_iterative(self,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        g = self.g if RANK==0 else np.zeros_like(self.x)
        if solve_full: 
            self.deltas = self.solve_iterative(self.S,g,self.cond1,True)
            #self.deltas = self.solve_iterative(self.S,g,self.cond1,False)
        else:
            self.deltas = np.zeros_like(self.x)
            for ix,(start,stop) in enumerate(self.block_dict):
                self.deltas[strt:stop] = self.solve_iterative(self.S[ix],g[start:stop],self.cond1,True)
        if RANK==0:
            self.x = self.update(self.rate1)
    def _transform_gradients_rgn_iterative(self,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        g = self.g if RANK==0 else np.zeros_like(self.x)
        if solve_full: 
            def hess(x):
                if self.terminate[0]==1:
                    return 0
                Hx = self.H(x)
                if self.terminate[0]==1:
                    return 0
                Sx = self.S(x)
                if self.terminate[0]==1:
                    return 0
                return Hx - self.E * Sx
            self.deltas = self.solve_iterative(hess,g,self.cond2,False)
            #self.deltas = self.solve_iterative(hess,g,self.cond2,True)
            self.terminate[0] = 0
            hessp = hess(self.deltas)
            if RANK==0:
                dE = np.dot(self.deltas,hessp)
        else:
            dE = 0.
            self.deltas = np.zeros_like(self.x)
            for ix,(start,stop) in enumerate(self.block_dict):
                def hess(x):
                    if self.terminate[0]==1:
                        return 0
                    Hx = self.H[ix](x)
                    if self.terminate[0]==1:
                        return 0
                    Sx = self.S[ix](x)
                    if self.terminate[0]==1:
                        return 0
                    return Hx - self.E * Sx
                deltas = self.solve_iterative(hess,g[start:stop],self.cond2,False)
                self.deltas[start:stop] = deltas 
                self.terminate[0] = 0
                hessp = hess(deltas)
                if RANK==0:
                    dE += np.dot(hessp,deltas)
        if RANK==0:
            return - np.dot(self.g,self.deltas) + .5 * dE
        else:
            return 0 
    def _transform_gradients_o2(self,full_sr=True,dense_sr=False):
        xnew_rgn = np.zeros_like(self.x)
        dEm = 0.
        if self.optimizer=='rgn':
            dEm = self._transform_gradients_rgn()
        elif self.optimizer=='lin':
            dEm = self._transform_gradients_lin()
        else:
            raise NotImplementedError
        if RANK==0:
            xnew_rgn = self.update(self.rate2)    
        if self.check is None:
            self.x = xnew_rgn
            return
        # SR
        if not ((full_sr==self.solve_full) and (dense_sr==self.solve_dense)):
            self.extract_S(solve_full=full_sr,solve_dense=dense_sr)
        self._transform_gradients_sr(solve_full=full_sr,solve_dense=dense_sr)
        xnew_sr = self.x
        
        COMM.Bcast(xnew_rgn,root=0) 
        self.amplitude_factory.update(xnew_rgn)
        if self.check=='trust_region':
            update_rgn = self._check_by_trust_region(dEm)
        elif self.check=='energy':
            update_rgn = self._check_by_energy(dEm)
        else:
            raise NotImplementedError
        if RANK>0:
            return
        if update_rgn: 
            self.x = xnew_rgn
        else:
            self.x = xnew_sr
    def _transform_gradients_lin(self,solve_dense=None,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        if solve_dense:
            dE = self._transform_gradients_lin_dense(solve_full=solve_full)
        else:
            dE = self._transform_gradients_lin_iterative(solve_full=solve_full)
        return dE
    def _scale_eigenvector(self):
        if self.xi is None:
            Ns = self.vmean
        else:
            if self.solve_full:
                Sp = np.dot(self.S,self.deltas) if self.solve_dense else self.S(self.deltas)
            else:
                Sp = np.zeros_like(self.x)
                for ix,(start,stop) in enumerate(self.block_dict):
                    Sp[start:stop] = np.dot(self.S[ix],self.deltas[start:stop]) if self.solve_dense else \
                                     self.S[ix](self.deltas[start:stop])
            Ns  = - (1.-self.xi) * Sp 
            Ns /= 1.-self.xi + self.xi * (1.+np.dot(self.deltas,Sp))**.5
        denom = 1. - np.dot(Ns,self.deltas)
        self.deltas /= -denom
        print('\tscale2=',denom)
    def _transform_gradients_lin_dense(self,solve_full=None):
        if RANK>0:
            return 
        self.deltas = np.zeros_like(self.x)
        solve_full = self.solve_full if solve_full is None else solve_full
        
        t0 = time.time()
        if solve_full:
            w,self.deltas,v0,inorm = \
                _lin_block_solve(self.H,self.E,self.S,self.g,self.Hvmean,self.vmean,self.cond2) 
        else:
            w = np.zeros(self.nsite)
            v0 = np.zeros(self.nsite)
            inorm = np.zeros(self.nsite)
            self.deltas = np.zeros_like(self.x)
            for ix,(start,stop) in enumerate(self.block_dict):
                w[ix],self.deltas[start:stop],v0[ix],inorm[ix] = \
                    _lin_block_solve(self.H[ix],self.E,self.S[ix],self.g[start:stop],
                                     self.Hvmean[start:stop],self.vmean[start:stop],self.cond2) 
            inorm = inorm.sum()
            w = w.sum()
        print(f'\tLIN solver time={time.time()-t0},inorm={inorm},eigenvalue={w},scale1={v0}')
        self._scale_eigenvector()
        if self.tmpdir is not None:
            Hi0 = self.g
            H0j = self.Hvmean - self.E * self.vmean
            if self.solve_full:
                H = self.H
                S = self.S
            else:
                H = np.zeros((self.nparam,self.nparam))
                S = np.zeros((self.nparam,self.nparam))
                for ix,(start,stop) in enumerate(self.block_dict):
                    H[start:stop,start:stop] = self.H[ix] 
                    S[start:stop,start:stop] = self.S[ix]
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('H',data=H) 
            f.create_dataset('S',data=S) 
            f.create_dataset('Hi0',data=Hi0) 
            f.create_dataset('H0j',data=H0j) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
        return w-self.E 
    def _transform_gradients_lin_iterative(self,cond):
        Hi0 = self.g
        H0j = self.Hvmean - self.E * self.vmean
        def A(x):
            x0,x1 = x[0],x[1:]
            y = np.zeros_like(x)
            y[0] = self.E * x0 + np.dot(H0j,x1)
            y[1:] = Hi0 * x0 + self.H(x1) 
            return y
        def B(x):
            x0,x1 = x[0],x[1:]
            y = np.zeros_like(x)
            y[0] = x0
            y[1:] = self.S(x1) + cond * x1
            return y
        x0 = np.zeros(1+self.nparam)
        x0[0] = 1.
        if self.solver == 'davidson':
            w,v = self.davidson(A,B,x0,self.E)
            self.deltas = v[1:]/v[0]
            print('\teigenvalue =',w)
            print('\tscale1=',v[0])
        else:
            A = spla.LinearOperator((self.nparam+1,self.nparam+1),matvec=A,dtype=self.x.dtype)
            B = spla.LinearOperator((self.nparam+1,self.nparam+1),matvec=B,dtype=self.x.dtype)
            w,v = spla.eigs(A,k=1,M=B,sigma=self.E,v0=x0,tol=CG_TOL)
            w,self.deltas = w[0].real,v[1:,0].real/v[0,0].real
            print('\timaginary norm=',np.linalg.norm(v[:,0].imag))
            print('\teigenvalue =',w)
            print('\tscale1=',v[0,0].real)
        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
        return w - self.E
    def current_energy(self):
        if self.exact_sampling:
            if RANK==0:
                self.E0 = self.E
        else:
            if RANK>0:
                nlarge = len(self.samples)
                nsmall = self.batchsize_small // (SIZE-1)
                every = nlarge // nsmall
                self.idxs = list(range(0,nlarge,every))
                self.elocal = self.elocal[self.idxs]
            self._extract_energy_stochastic() 
    def correlated_energy(self):
        # samples new energy
        self.sampler.amplitude_factory = self.amplitude_factory
        if self.exact_sampling:
            self.sample_exact(compute_v=False,compute_Hv=False) 
            self._extract_energy_exact()
            if RANK==0:
                self.Enew = self.E
        else:
            if RANK>0:
                self._correlated_sampling(self.idxs)
            self._extract_energy_stochastic(weighted=True)
    def _correlated_sampling(self,idxs):
        self.elocal = []
        self.flocal = []
        self.amplitude_factory.update_scheme(0)
        for idx in idxs:
            config = self.samples[idx]
            cx,ex,_,_ = self.ham.compute_local_energy(config,self.amplitude_factory,
                                                         compute_v=False,compute_Hv=False) 
            self.elocal.append(ex) 
            self.flocal.append(cx**2/self.p0[config])
    def _check_by_trust_region(self,dEm):
        self.current_energy()
        if RANK==0:
            self.E0,self.Eerr0 = self.E,self.Eerr
        self.correlated_energy()
        if RANK>0:
            return 
        if self.Eerr is None:
            return False
        err = (self.Eerr0**2 + self.Eerr**2)**.5
        dE = self.E - self.E0
        rho_plus = (dE+err) / dEm 
        rho_minus = (dE-err) / dEm
        print(f'\tpredict={dEm},actual={(dE,err)},ratio={(rho_plus,rho_minus)}')
        if rho_plus < self.accept_ratio and rho_minus < accept_ratio:
            return False
        else:
            return True
    def _check_by_energy(self,dEm):
        if RANK==0:
            self.E0,self.Eerr0 = self.E,self.Eerr
        self.sample(samplesize=self.batchsize_small,compute_v=False,compute_Hv=False)
        self.extract_energy()
        if RANK>0:
            return 
        if self.Eerr is None:
            return False
        dE = self.E - self.E0
        err = (self.Eerr0**2 + self.Eerr**2)**.5
        print(f'\tpredict={dEm},actual={(dE,err)}')
        #if dE + err > 0. and dE - err > 0.:
        if dE > 0.:
            return False
        else:
            return True
class DMRG(TNVMC):
    def __init__(
        self,
        ham,
        sampler,
        amplitude_factory,
        optimizer='sr',
        **kwargs,
    ):
        # parse ham
        self.ham = ham
        self.nsite = ham.Lx * ham.Ly

        # parse sampler
        self.sampler = sampler
        self.exact_sampling = sampler.exact

        # parse wfn 
        self.amplitude_factory = amplitude_factory         
        self.x = self.amplitude_factory.get_x()
        self.block_dict = self.amplitude_factory.get_block_dict()

        # parse gradient optimizer
        self.optimizer = optimizer
        self.compute_Hv = False
        self.solve = 'matrix'
        if self.optimizer in ['rgn','lin']:
            self.ham.initialize_pepo(self.amplitude_factory.psi)
            self.compute_Hv = True
        if self.optimizer=='lin':
            self.xi = kwargs.get('xi',0.5)

        # to be set before run
        self.ix = 0 
        self.direction = 1
        self.config = None
        self.batchsize = None
        self.ratio = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR
        self.rate2 = None # rate for LIN,RGN
        self.cond1 = None
        self.cond2 = None
        self.check = None 
        self.accept_ratio = None
    def next_ix(self):
        if self.direction == 1 and self.ix == self.nsite-1:
            self.direction = -1
        elif self.direction == -1 and self.ix == 0:
            self.direction = 1
        self.ix += self.direction 
    def set_nparam(self):
        start,stop = self.block_dict[self.ix]
        self.nparam = stop - start
        if self.ratio is not None:
            self.batchsize = max(self.batchsize_small,self.nparam * self.ratio) 
    def run(self,start,stop,tmpdir=None):
        for step in range(start,stop):
            self.amplitude_factory.ix = self.ix
            self.set_nparam()
            if RANK==0:
                print(f'ix={self.ix},nparam={self.nparam}')
            self.step = step
            self.sample()
            self.extract_energy_gradient()
            self.transform_gradients()
            COMM.Bcast(self.x,root=0) 
            psi = self.amplitude_factory.update(self.x)
            self.ham.update_cache(self.ix)
            if RANK==0:
                if tmpdir is not None: # save psi to disc
                    write_ftn_to_disc(psi,tmpdir+f'psi{step+1}',provided_filename=True)
            #self.next_ix()
            self.ix = (self.ix + 1) % self.nsite
    def update(self,rate):
        start,stop = self.block_dict[self.ix]
        x = self.x.copy()
        x[start:stop] -= rate * self.deltas
        return x 
    def _transform_gradients_o2(self):
        super()._transform_gradients_o2(solve_sr='matrix')
