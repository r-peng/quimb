import time,scipy,functools,h5py,gc
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
def _rgn_block_solve(H,E,S,g,eta,eps0,enforce_pos=True):
    sh = len(g)
    # hessian 
    hess = H - E * S
    R = S + eta * np.eye(sh)

    wmin = -1. + 0.j
    eps = eps0 * 2.
    while wmin.real < 0.:
        # smallest eigenvalue
        eps /= 2.
        w = np.linalg.eigvals(hess + R/eps)
        idx = np.argmin(w.real)
        wmin = w[idx]
    # solve
    deltas = np.linalg.solve(hess + R/eps,g)
    # compute model energy
    dE = - np.dot(deltas,g) + .5 * np.dot(deltas,np.dot(hess + R/eps,deltas)) 
    return deltas,dE,wmin,eps
def _newton_block_solve(H,E,S,g,cond):
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
    return deltas,dE,wmin
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
        normalize=False,
        optimizer='sr',
        solve_full=True,
        solve_dense=False,
        **kwargs,
    ):
        # parse ham
        self.ham = ham
        self.nsite = ham.model.nsite

        # parse sampler
        self.sampler = sampler
        self.exact_sampling = sampler.exact

        # parse wfn 
        x = self.sampler.amplitude_factory.get_x()
        self.nparam = len(x)
        self.dtype = x.dtype
        self.init_norm = None
        if normalize:
            self.init_norm = np.linalg.norm(x)    

        # parse gradient optimizer
        self.optimizer = optimizer
        self.solve_full = solve_full
        self.solve_dense = solve_dense
        self.compute_Hv = False
        if self.optimizer in ['rgn','lin']:
            self.compute_Hv = True
        if self.optimizer=='rgn':
            solver = kwargs.get('solver','lgmres')
            self.solver = {'lgmres':spla.lgmres,
                           'tfqmr':tfqmr}[solver] 
            self.pure_newton = kwargs.get('pure_newton',True)
 
        if self.optimizer=='lin':
            self.xi = kwargs.get('xi',0.5)

        # to be set before run
        self.progbar = False
        self.tmpdir = None
        self.config = None
        self.batchsize = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR
        self.rate2 = None # rate for LIN,RGN
        self.cond1 = None
        self.cond2 = None
        self.check = None 
        self.debug = False

        self.free_quantities()
    def free_quantities(self):
        self.f = None
        self.e = None
        self.g = None
        self.v = None
        self.vmean = None
        self.Hv = None
        self.Hvmean = None
        self.S = None
        self.H = None
        self.Sx1 = None
        self.Hx1 = None
        self.deltas = None
        gc.collect()
    def normalize(self,x):
        if self.init_norm is not None:
            norm = np.linalg.norm(x)
            x *= self.init_norm / norm    
        return x
    def run(self,start,stop,tmpdir=None):
        self.Eold = 0.
        for step in range(start,stop):
            self.ham.amplitude_factory = self.sampler.amplitude_factory
            self.step = step
            self.sample()
            self.extract_energy_gradient()
            x = self.transform_gradients()
            self.free_quantities()
            COMM.Bcast(x,root=0) 
            fname = None if tmpdir is None else tmpdir+f'psi{step+1}' 
            psi = self.sampler.amplitude_factory.update(x,fname=fname,root=0)
    def sample(self,samplesize=None,compute_v=True,compute_Hv=None):
        self.sampler.preprocess()
        compute_Hv = self.compute_Hv if compute_Hv is None else compute_Hv

        self.buf = np.zeros(4)
        self.terminate = np.array([0])

        self.buf[0] = RANK + .1
        if compute_v:
            self.evsum = np.zeros(self.nparam,dtype=self.dtype)
            self.vsum = np.zeros(self.nparam,dtype=self.dtype)
            self.v = []
        if compute_Hv:
            self.Hvsum = np.zeros(self.nparam,dtype=self.dtype)
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

        if self.progbar:
            pg = Progbar(total=samplesize)

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
            if self.progbar:
                pg.update()
            #print(ncurr)
            if ncurr >= samplesize: # send termination message to all workers
                self.terminate[0] = 1
                for worker in range(1,SIZE):
                    COMM.Send(self.terminate,dest=worker,tag=1)
            else:
                COMM.Send(self.terminate,dest=rank,tag=1)
        print('\tsample time=',time.time()-t0)
        print('\tcontraction err=',err_mean / len(self.e),err_max)
        self.e = np.array(self.e)
        self.f = np.array(self.f)
    def _sample_stochastic(self,compute_v=True,compute_Hv=False):
        self.buf[1] = 1.
        c = []
        e = []
        configs = []
        while self.terminate[0]==0:
            config,omega = self.sampler.sample()
            #if omega > self.omega:
            #    self.config,self.omega = config,omega
            cx,ex,vx,Hvx,err = self.ham.compute_local_energy(config,compute_v=compute_v,compute_Hv=compute_Hv)
            if cx is None or np.fabs(ex) > DISCARD:
                print(f'RANK={RANK},cx={cx},ex={ex}')
                ex = 0.
                err = 0.
                if compute_v:
                    vx = np.zeros(self.nparam,dtype=self.dtype)
                if compute_Hv:
                    Hvx = np.zeros(self.nparam,dtype=self.dtype)
            self.buf[2] = ex
            self.buf[3] = err
            if compute_v:
                self.vsum += vx
                self.evsum += vx * ex
                self.v.append(vx)
            if compute_Hv:
                self.Hvsum += Hvx
                self.Hv.append(Hvx)
            if self.debug:
                c.append(cx)
                e.append(ex)
                configs.append(list(config))

            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
        self.sampler.config = self.config
        if compute_v:
            self.v = np.array(self.v)
        if compute_Hv:
            self.Hv = np.array(self.Hv)
        if self.debug:
            f = h5py.File(f'./step{self.step}RANK{RANK}.hdf5','w')
            if compute_Hv:
                f.create_dataset('Hv',data=self.Hv)
            if compute_v:
                f.create_dataset('v',data=self.v)
            f.create_dataset('e',data=np.array(e))
            f.create_dataset('c',data=np.array(c))
            f.create_dataset('config',data=np.array(configs))
            f.close()
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
            cx,ex,vx,Hvx,err = self.ham.compute_local_energy(config,compute_v=compute_v,compute_Hv=compute_Hv)
            if cx is None:
                raise ValueError
            if np.fabs(ex)*p[ix] > DISCARD:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self.f.append(p[ix])
            self.buf[1] = p[ix]
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
        self.f = np.array(self.f)
        if compute_v:
            self.v = np.array(self.v)
        if compute_Hv:
            self.Hv = np.array(self.Hv)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        if self.optimizer in ['rgn','lin']:
            self._extract_Hvmean()
        if RANK==0:
            try:
                print(f'step={self.step},E={self.E/self.nsite},dE={(self.E-self.Eold)/self.nsite},err={self.Eerr/self.nsite},gmax={np.amax(np.fabs(self.g))}')
            except TypeError:
                print('E=',self.E)
                print('Eerr=',self.Eerr)
            print('\tcollect g,Hv time=',time.time()-t0)
            self.Eold = self.E
    def extract_energy(self):
        if RANK>0:
            return
        if self.exact_sampling:
            self.n = 1.
            self.E = np.dot(self.f,self.e)
            self.Eerr = 0.
        else:
            self.n = len(self.e)
            self.E,self.Eerr = blocking_analysis(self.f,self.e,0,True)
    def extract_gradient(self):
        vmean = np.zeros(self.nparam,dtype=self.dtype)
        COMM.Reduce(self.vsum,vmean,op=MPI.SUM,root=0)
        evmean = np.zeros(self.nparam,dtype=self.dtype)
        COMM.Reduce(self.evsum,evmean,op=MPI.SUM,root=0)
        self.g = None
        self.vsum = None
        self.evsum = None
        if RANK==0:
            vmean /= self.n
            evmean /= self.n
            #print(evmean)
            self.g = evmean - self.E * vmean
            self.vmean = vmean
    def _extract_Hvmean(self):
        Hvmean = np.zeros(self.nparam,dtype=self.dtype)
        COMM.Reduce(self.Hvsum,Hvmean,op=MPI.SUM,root=0)
        self.Hvsum = None
        if RANK==0:
            Hvmean /= self.n
            #print(Hvmean)
            self.Hvmean = Hvmean
    def extract_S(self,solve_full=None,solve_dense=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        fxn = self._get_Smatrix if solve_dense else self._get_S_iterative
        self.Sx1 = np.zeros(self.nparam,dtype=self.dtype)
        if solve_full:
            self.S = fxn() 
        else:
            self.S = [None] * self.nsite
            for ix,(start,stop) in enumerate(self.sampler.amplitude_factory.block_dict):
                self.S[ix] = fxn(start=start,stop=stop)
    def extract_H(self,solve_full=None,solve_dense=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        fxn = self._get_Hmatrix if solve_dense else self._get_H_iterative
        self.Hx1 = np.zeros(self.nparam,dtype=self.dtype)
        if solve_full:
            self.H = fxn() 
        else:
            self.H = [None] * self.nsite
            for ix,(start,stop) in enumerate(self.sampler.amplitude_factory.block_dict):
                self.H[ix] = fxn(start=start,stop=stop)
    def _get_Smatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        t0 = time.time()
        if RANK==0:
            sh = stop-start
            vvsum_ = np.zeros((sh,)*2,dtype=self.dtype)
        else:
            v = self.v[:,start:stop] 
            if self.exact_sampling:
                vvsum_ = np.einsum('s,si,sj->ij',self.f,v,v)
            else:
                vvsum_ = np.dot(v.T,v)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean = self.vmean[start:stop]
            S = vvsum / self.n - np.outer(vmean,vmean)
            print('\tcollect S matrix time=',time.time()-t0)
        return S
    def _get_Hmatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        t0 = time.time()
        if RANK==0:
            sh = stop-start
            vHvsum_ = np.zeros((sh,)*2,dtype=self.dtype)
        else:
            v = self.v[:,start:stop] 
            Hv = self.Hv[:,start:stop] 
            if self.exact_sampling:
                vHvsum_ = np.einsum('s,si,sj->ij',self.f,v,Hv)
            else:
                vHvsum_ = np.dot(v.T,Hv)
        vHvsum = np.zeros_like(vHvsum_)
        COMM.Reduce(vHvsum_,vHvsum,op=MPI.SUM,root=0)
        H = None
        if RANK==0:
            #print(start,stop,np.linalg.norm(vHvsum-vHvsum.T))
            Hvmean = self.Hvmean[start:stop]
            vmean = self.vmean[start:stop]
            g = self.g[start:stop]
            H = vHvsum / self.n - np.outer(vmean,Hvmean) - np.outer(g,vmean)
            print('\tcollect H matrix time=',time.time()-t0)
        return H
    def _get_S_iterative(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop 
        if RANK==0:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                Sx1 = np.zeros_like(self.Sx1[start:stop])
                COMM.Reduce(Sx1,self.Sx1[start:stop],op=MPI.SUM,root=0)     
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
                Hx1 = np.zeros_like(self.Hx1[start:stop])
                COMM.Reduce(Hx1,self.Hx1[start:stop],op=MPI.SUM,root=0)     
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
            x = self._transform_gradients_sr()
        elif self.optimizer in ['rgn','lin']:
            x = self._transform_gradients_o2()
        else:
            x = self._transform_gradients_sgd()
        if RANK==0:
            print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(self.deltas)},dot={np.dot(self.deltas,self.g)},x={np.linalg.norm(x)}')
        return x
    def update(self,rate):
        x = self.sampler.amplitude_factory.get_x()
        return self.normalize(x - rate * self.deltas)
    def _transform_gradients_sgd(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype) 
        if self.optimizer=='sgd':
            self.deltas = self.g
        elif self.optimizer=='sign':
            self.deltas = np.sign(self.g)
        elif self.optimizer=='signu':
            self.deltas = np.sign(self.g) * np.random.uniform(size=self.nparam)
        else:
            raise NotImplementedError
        return self.update(self.rate1)
    def _transform_gradients_sr(self,solve_dense=None,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        self.extract_S(solve_full=solve_full,solve_dense=solve_dense)
        if solve_dense:
            return self._transform_gradients_sr_dense(solve_full=solve_full)
        else:
            return self._transform_gradients_sr_iterative(solve_full=solve_full)
    def _transform_gradients_rgn(self,solve_dense=None,solve_full=None,x0=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        self.extract_S(solve_full=solve_full,solve_dense=solve_dense)
        self.extract_H(solve_full=solve_full,solve_dense=solve_dense)
        if solve_dense:
            dE = self._transform_gradients_rgn_dense(solve_full=solve_full)
        else:
            dE = self._transform_gradients_rgn_iterative(solve_full=solve_full,x0=x0)
        return dE
    def _transform_gradients_sr_dense(self,solve_full=None):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype) 
        t0 = time.time()
        solve_full = self.solve_full if solve_full is None else solve_full
        if solve_full:
            self.deltas = np.linalg.solve(self.S,self.g)
        else:
            self.deltas = np.empty(self.nparam,dtype=self.dtype)
            for ix,(start,stop) in enumerate(self.sampler.amplitude_factory.block_dict):
                S = self.S[ix] + self.cond1 * np.eye(stop-start)
                self.deltas[start:stop] = np.linalg.solve(S,self.g[start:stop])
        print('\tSR solver time=',time.time()-t0)
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
        return self.update(self.rate1)
    def _transform_gradients_rgn_dense(self,solve_full=None):
        if RANK>0:
            return 0. 
        t0 = time.time()
        solve_full = self.solve_full if solve_full is None else solve_full
        if solve_full:
            if self.pure_newton:
                self.deltas,dE,w = _newton_block_solve(self.H,self.E,self.S,self.g,self.cond2) 
                print(f'\tRGN solver time={time.time()-t0},least eigenvalue={w}')
            else:
                self.deltas,dE,w,eps = _rgn_block_solve(self.H,self.E,self.S,self.g,self.cond1,self.rate2) 
                print(f'\tRGN solver time={time.time()-t0},least eigenvalue={w},eps={eps}')
        else:
            blk_dict = self.sampler.amplitude_factory.block_dict
            w = [None] * len(blk_dict)
            dE = np.zeros(len(blk_dict))  
            self.deltas = np.empty(self.nparam,dtype=self.dtype)
            for ix,(start,stop) in enumerate(blk_dict):
                if self.pure_newton:
                    self.deltas[start:stop],dE[ix],w[ix] = _newton_block_solve(
                        self.H[ix],self.E,self.S[ix],self.g[start:stop],self.cond2)
                    print(f'ix={ix},eigval={w[ix]}')
                else:
                    self.deltas[start:stop],dE[ix],w[ix],eps = _rgn_block_solve(
                        self.H[ix],self.E,self.S[ix],self.g[start:stop],self.cond1,self.rate2)
                    print(f'ix={ix},eigval={w[ix]},eps={eps}')
            w = min(np.array(w).real)
            dE = np.sum(dE)
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
        return dE
    def solve_iterative(self,A,b,symm,x0=None):
        self.terminate = np.array([0])
        deltas = np.zeros_like(b)
        sh = len(b)
        if RANK==0:
            t0 = time.time()
            LinOp = spla.LinearOperator((sh,sh),matvec=A,dtype=b.dtype)
            if symm:
                deltas,info = spla.minres(LinOp,b,x0=x0,tol=CG_TOL,maxiter=MAXITER)
            else: 
                deltas,info = self.solver(LinOp,b,x0=x0,tol=CG_TOL,maxiter=MAXITER)
            self.terminate[0] = 1
            COMM.Bcast(self.terminate,root=0)
            print(f'\tsolver time={time.time()-t0},exit status={info}')
        else:
            nit = 0
            while self.terminate[0]==0:
                nit += 1
                A(deltas)
            if RANK==1:
                print('niter=',nit)
        return deltas
    def _transform_gradients_sr_iterative(self,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype)
        if solve_full: 
            def R(x):
                return self.S(x) + self.cond1 * x
            self.deltas = self.solve_iterative(R,g,True,x0=g)
        else:
            self.deltas = np.empty(self.nparam,dtype=self.dtype)
            for ix,(start,stop) in enumerate(self.sampler.amplitude_factory.block_dict):
                def R(x):
                    return self.S[ix](x) + self.cond1 * x
                self.deltas[strt:stop] = self.solve_iterative(R,g[start:stop],True,x0=g[start:stop])
        if RANK==0:
            return self.update(self.rate1)
        else:
            return np.zeros(self.nparam,dtype=self.dtype)
    def _transform_gradients_rgn_iterative(self,solve_full=None,x0=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype)
        E = self.E if RANK==0 else 0
        if RANK==0:
            print('pure_newton=',self.pure_newton)
        if solve_full: 
            def A(x):
                if self.terminate[0]==1:
                    return 0
                Hx = self.H(x)
                if self.terminate[0]==1:
                    return 0
                Sx = self.S(x)
                if self.terminate[0]==1:
                    return 0
                if self.pure_newton:
                    return Hx - E * Sx + self.cond2 * x
                else:
                    return Hx + (1./self.rate2 - E) * Sx + self.cond1 / self.rate2 * x
            self.deltas = self.solve_iterative(A,g,False,x0=x0)
            self.terminate[0] = 0
            hessp = A(self.deltas)
            if RANK==0:
                dE = np.dot(self.deltas,hessp)
        else:
            dE = 0.
            self.deltas = np.empty(self.nparam,dtype=self.dtype)
            for ix,(start,stop) in enumerate(self.sampler.amplitude_factory.block_dict):
                if RANK==0:
                    print(f'ix={ix},sh={stop-start}')
                def A(x):
                    if self.terminate[0]==1:
                        return 0
                    Hx = self.H[ix](x)
                    if self.terminate[0]==1:
                        return 0
                    Sx = self.S[ix](x)
                    if self.terminate[0]==1:
                        return 0
                    if self.pure_newton:
                        return Hx - E * Sx + self.cond2 * x
                    else:
                        return Hx + (1./self.rate2 - E) * Sx + self.cond1 / self.rate2 * x
                x0_ = None if x0 is None else x0[start:stop]
                deltas = self.solve_iterative(A,g[start:stop],False,x0=x0_)
                self.deltas[start:stop] = deltas 
                self.terminate[0] = 0
                hessp = A(deltas)
                if RANK==0:
                    dE += np.dot(hessp,deltas)
        if RANK==0:
            return - np.dot(self.g,self.deltas) + .5 * dE
        else:
            return 0. 
    def _transform_gradients_o2(self,full_sr=True,dense_sr=False):
        # SR
        xnew_sr = self._transform_gradients_sr(solve_full=full_sr,solve_dense=dense_sr)
        deltas_sr = self.deltas

        if self.optimizer=='rgn':
            dE = self._transform_gradients_rgn(x0=deltas_sr)
        elif self.optimizer=='lin':
            dE = self._transform_gradients_lin()
        else:
            raise NotImplementedError
        if self.pure_newton:
            xnew_rgn = self.update(self.rate2) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype)
        else:
            xnew_rgn = self.update(1.) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype)
        deltas_rgn = self.deltas
        if self.check is None:
            return xnew_rgn
        
        if RANK==0:
            g = self.g
        COMM.Bcast(xnew_rgn,root=0) 
        self.sampler.amplitude_factory.update(xnew_rgn)
        if self.check=='energy':
            update_rgn = self._check_by_energy(dE)
        else:
            raise NotImplementedError
        if RANK==0:
            self.g = g
        if update_rgn: 
            self.deltas = deltas_rgn
            return xnew_rgn
        else:
            self.deltas = deltas_sr
            return xnew_sr
    def _transform_gradients_lin(self,solve_dense=None,solve_full=None):
        raise NotImplementedError
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
    def _check_by_energy(self,dEm):
        if RANK==0:
            E,Eerr = self.E,self.Eerr
        self.free_quantities()
        debug = self.debug
        self.debug = False
        self.sample(samplesize=self.batchsize_small,compute_v=False,compute_Hv=False)
        self.debug = debug
        self.extract_energy()
        if RANK>0:
            return True 
        if self.Eerr is None:
            return False
        dE = self.E - E
        err = (Eerr**2 + self.Eerr**2)**.5
        print(f'\tpredict={dEm},actual={(dE,err)}')
        return (dE < 0.) 
    def measure(self,fname=None):
        self.sample(compute_v=False,compute_Hv=False)

        sendbuf = np.array([self.ham.n])
        recvbuf = np.zeros_like(sendbuf) 
        COMM.Reduce(sendbuf,recvbuf,op=MPI.SUM,root=0)
        n = recvbuf[0]

        sendbuf = self.ham.data
        recvbuf = np.zeros_like(sendbuf) 
        COMM.Reduce(sendbuf,recvbuf,op=MPI.SUM,root=0)
        if RANK>0:
            return
        data = recvbuf / n
        if fname is not None:
            f = h5py.File(fname,'w')
            f.create_dataset('data',data=data) 
            f.close()
        self.ham._print(fname,data)
    def debug_torch(self,tmpdir,step,rank=None):
        if RANK==0:
            return
        if rank is not None:
            if RANK!=rank:
                return
        f = h5py.File(tmpdir+f'step{step}RANK{RANK}.hdf5','r')
        e = f['e'][:]
        v = f['v'][:]
        configs = f['config'][:]
        f.close()
        e_new = []
        c_new = []
        v_new = []
        Hv_new = []
        n = len(e)
        print(f'RANK={RANK},n={n}')
        for i in range(n):
            config = tuple(configs[i,:])
            cx,ex,vx,Hvx,err = self.ham.compute_local_energy(
                config,self.sampler.amplitude_factory,compute_v=True,compute_Hv=True)
            if cx is None or np.fabs(ex) > DISCARD:
                print(f'RANK={RANK},cx={cx},ex={ex}')
                ex = 0.
                err = 0.
                if compute_v:
                    vx = np.zeros(self.nparam,dtype=self.dtype)
                if compute_Hv:
                    Hvx = np.zeros(self.nparam,dtype=self.dtype)
            e_new.append(ex) 
            c_new.append(cx) 
            v_new.append(vx)
            Hv_new.append(Hvx)
            err_e = np.fabs(ex-e[i])
            err_v = np.linalg.norm(vx-v[i,:])
            if err_e > 1e-6 or err_v > 1e-6: 
                print(f'RANK={RANK},config={config},ex={ex},ex_sr={e[i]},err_e={err_e},err_v={err_v}')
            #else:
            #    print(f'RANK={RANK},i={i}')
        f = h5py.File(f'./step{step}RANK{RANK}.hdf5','w')
        f.create_dataset('Hv',data=np.array(Hv_new))
        f.create_dataset('v',data=np.array(v_new))
        f.create_dataset('e',data=np.array(e_new))
        f.create_dataset('c',data=np.array(c_new))
        f.close()
    def load(self,tmpdir):
        if RANK==0:
            vmean = np.zeros(self.nparam,dtype=self.dtype)
            evmean = np.zeros(self.nparam,dtype=self.dtype)
            Hvmean = np.zeros(self.nparam,dtype=self.dtype)

            self.e = [] 
            for rank in range(1,SIZE):
                print('rank=',rank)
                e,vsum,evsum,Hvsum = COMM.recv(source=rank)
                self.e.append(e)
                vmean += vsum
                evmean += evsum
                Hvmean += Hvsum

            self.e = np.concatenate(self.e)
            self.f = np.ones_like(self.e)
            self.E,self.Eerr = blocking_analysis(self.f,self.e,0,True)
            self.n = len(self.e)

            vmean /= self.n
            evmean /= self.n
            self.g = evmean - self.E * vmean
            self.vmean = vmean 

            Hvmean /= self.n
            self.Hvmean = Hvmean
        else:
            f = h5py.File(tmpdir+f'step{self.step}RANK{RANK}.hdf5','r')
            self.Hv = f['Hv'][:]
            self.v = f['v'][:]
            e = f['e'][:]
            f.close()
            vsum = self.v.sum(axis=0)
            evsum = np.dot(e,self.v)
            Hvsum = self.Hv.sum(axis=0)
            COMM.send([e,vsum,evsum,Hvsum],dest=0)
        COMM.Barrier()
##############################################################################################
# sampler
#############################################################################################
import itertools
class DenseSampler:
    def __init__(self,nsite,nspin,exact=False,seed=None,thresh=1e-14,fix_sector=True):
        self.nsite = nsite 
        self.nspin = nspin

        self.all_configs = self.get_all_configs(fix_sector=fix_sector)
        self.ntotal = len(self.all_configs)
        if RANK==0:
            print('ntotal=',self.ntotal)
        self.flat_indexes = list(range(self.ntotal))
        self.p = None

        batchsize,remain = self.ntotal//SIZE,self.ntotal%SIZE
        self.count = np.array([batchsize]*SIZE)
        if remain > 0:
            self.count[-remain:] += 1
        self.disp = np.concatenate([np.array([0]),np.cumsum(self.count[:-1])])
        self.start = self.disp[RANK]
        self.stop = self.start + self.count[RANK]

        self.rng = np.random.default_rng(seed)
        self.burn_in = 0
        self.dense = True
        self.exact = exact 
        self.amplitude_factory = None
        self.thresh = thresh
    def preprocess(self):
        self.compute_dense_prob()
    def compute_dense_prob(self):
        t0 = time.time()
        ptotal = np.zeros(self.ntotal)
        start,stop = self.start,self.stop
        configs = self.all_configs[start:stop]

        plocal = [] 
        for config in configs:
            #print('sampler',config)
            config = self.amplitude_factory.parse_config(config)
            plocal.append(self.amplitude_factory.prob(config))
        plocal = np.array(plocal)
         
        COMM.Allgatherv(plocal,[ptotal,self.count,self.disp,MPI.DOUBLE])
        nonzeros = []
        for ix,px in enumerate(ptotal):
            if px > self.thresh:
                nonzeros.append(ix) 
        n = np.sum(ptotal)
        ptotal /= n 
        self.p = ptotal
        if RANK==SIZE-1:
            print('\tdense amplitude time=',time.time()-t0)

        ntotal = len(nonzeros)
        batchsize,remain = ntotal//(SIZE-1),ntotal%(SIZE-1)
        L = SIZE-1-remain
        if RANK-1<L:
            start = (RANK-1)*batchsize
            stop = start+batchsize
        else:
            start = (batchsize+1)*(RANK-1)-L
            stop = start+batchsize+1
        self.nonzeros = nonzeros if RANK==0 else nonzeros[start:stop]
    def get_all_configs(self,fix_sector=True):
        if not fix_sector:
            return list(itertools.product((0,1),repeat=self.nsite))
        assert isinstance(self.nspin,tuple)
        sites = list(range(self.nsite))
        occs = list(itertools.combinations(sites,self.nspin[0]))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = np.zeros(self.nsite,dtype=int) 
            config[occ,] = 1
            configs[i] = tuple(config)
        return configs
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        config = self.all_configs[flat_idx]
        omega = self.p[flat_idx]
        return config,omega
class ExchangeSampler:
    def preprocess(self):
        self._burn_in()
    def _burn_in(self,config=None,burn_in=None):
        if config is not None:
            self.config = config 
        af = self.amplitude_factory
        self.px = af.prob(af.parse_config(self.config))

        if RANK==0:
            print('\tprob=',self.px)
            return 
        t0 = time.time()
        burn_in = self.burn_in if burn_in is None else burn_in
        for n in range(burn_in):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
    def propose_new_pair(self,i1,i2):
        return i2,i1
    def sample(self):
        if self.amplitude_factory.deterministic:
            self._sample_deterministic()
        else:
            self._sample()
        return self.config,self.px
##############################################################################################
# HELPER FUNCS 
##############################################################################################
def load_tn_from_disc(fname, delete_file=False):
    if type(fname) != str:
        data = fname
    else:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    return data
def write_tn_to_disc(tn, fname, provided_filename=False):
    with open(fname, 'wb') as f:
        pickle.dump(tn, f)
    return fname
import pickle,uuid
def scale_wfn(psi,scale):
    for tid in psi.tensor_map:
        tsr = psi.tensor_map[tid]
        tsr.modify(data=tsr.data*scale)
    return psi
def safe_contract(tn):
    try:
        data = tn.contract(tags=all)
    except (ValueError,IndexError):
        return None
    if isinstance(data,int):
        return None
    return data
def contraction_error(cx,multiply=True):
    def _contraction_error(cx):
        if isinstance(cx,dict):
            if len(cx)==0:
                return 0.,0.
            cx = np.array(list(cx.values()))
            max_,min_,mean_ = np.amax(cx),np.amin(cx),np.mean(cx)
            return mean_,np.fabs((max_-min_)/mean_)
        else:
            return cx,0.

    if isinstance(cx,dict):
        return _contraction_error(cx)
    if isinstance(cx,list):
        cx_,err = np.zeros(3),np.zeros(3)
        for ix in range(3): 
            cx_[ix],err[ix] = _contraction_error(cx[ix])
        if multiply:
            cx_ = np.prod(cx_)
        return cx_,np.amax(err)
    return cx,0.
def list2dict(ls):
    if isinstance(ls,dict):
        return ls
    dict_ = dict()
    for ix in range(3):
        for key,val in ls[ix].items():
            dict_[key,ix] = val
    return dict_
def dict2list(dict_):
    if isinstance(dict_,list):
        return dict_
    ls = [dict(),dict(),dict()] 
    for (key,ix),val in dict_.items():
        ls[ix][key] = val
    return ls
import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)
from .torch_utils import SVD,QR
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)
def tensor2backend(data,backend,requires_grad=False):
    if isinstance(data,np.ndarray):
        if backend=='torch': # numpy to torch 
            data = torch.tensor(data,requires_grad=requires_grad)
    elif isinstance(data,torch.Tensor):
        if backend=='numpy': # torch to numpy
            data = data.detach().numpy()
            if data.size==1:
                data = data.reshape(-1)[0]
        else: # torch to torch
            data.requires_grad_(requires_grad=requires_grad)
    else:
        pass
    return data
from .tensor_core import Tensor,rand_uuid
def _add_gate(tn,gate,order,where,site_ind,site_tag,contract=True):
    # reindex
    kixs = [site_ind(site) for site in where]
    bixs = [kix+'*' for kix in kixs]
    for site,kix,bix in zip(where,kixs,bixs):
        tn[site_tag(site),'BRA'].reindex_({kix:bix})

    # add gate
    if order=='b1,k1,b2,k2':
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
    elif order=='b1,b2,k1,k2':
        inds = bixs + kixs
    else:
        raise NotImplementedError
    T = Tensor(data=gate,inds=inds)
    tn.add_tensor(T,virtual=True)
    if not contract:
        return tn
    return safe_contract(tn)
class AmplitudeFactory:
##### wfn methods #####
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
        ar.set_backend(tsr)
        for site in self.sites:
            tag = self.site_tag(site)
            self.psi[tag].modify(data=self.tensor2backend(self.psi[tag].data,backend=backend,requires_grad=requires_grad))
        for key in self.data_map:
            self.data_map[key] = self.tensor2backend(self.data_map[key],backend=backend,requires_grad=False)
    def get_site_map(self,blks):
        site_order = []
        for blk in blks:
            site_order += blk
        site_map = dict()
        for ix,site in enumerate(site_order):
            site_map[site] = ix
        return site_map
    def get_constructors(self,psi):
        constructors = [None] * self.nsite 
        for site in self.sites:
            data = psi[self.site_tag(site)].data
            ix = self.site_map[site]
            constructors[ix] = data.shape,len(data.flatten()),site
        return constructors
    def get_block_dict(self,blks):
        start = 0
        blk_dict = [None] * len(blks)
        for bix,blk in enumerate(blks):
            site_min,site_max = blk[0],blk[-1]
            ix_min,ix_max = self.site_map[site_min],self.site_map[site_max]
            stop = start
            for ix in range(ix_min,ix_max+1):
                _,size,_ = self.constructors[ix]
                stop += size
            blk_dict[bix] = start,stop
            start = stop
        return blk_dict 
    def dict2vecs(self,dict_):
        ls = [None] * len(self.constructors)
        for ix,(_,size,site) in enumerate(self.constructors):
            vec = np.zeros(size)
            g = dict_.get(site,None)
            if g is not None:
                vec = self.tensor2vec(g,ix=ix) 
            ls[ix] = vec
        return ls
    def dict2vec(self,dict_):
        return np.concatenate(self.dict2vecs(dict_))
    def psi2vecs(self,psi=None):
        psi = self.psi if psi is None else psi
        ls = [None] * len(self.constructors)
        for ix,(_,size,site) in enumerate(self.constructors):
            ls[ix] = self.tensor2vec(psi[self.site_tag(site)].data,ix=ix)
        return ls
    def psi2vec(self,psi=None):
        return np.concatenate(self.psi2vecs(psi)) 
    def get_x(self):
        return self.psi2vec()
    def split_vec(self,x):
        ls = [None] * len(self.constructors)
        start = 0
        for ix,(_,size,_) in enumerate(self.constructors):
            stop = start + size
            ls[ix] = x[start:stop]
            start = stop
        return ls 
    def vec2dict(self,x): 
        dict_ = dict() 
        ls = self.split_vec(x)
        for ix,(_,_,site) in enumerate(self.constructors):
            dict_[site] = self.vec2tensor(ls[ix],ix) 
        return dict_ 
    def vec2psi(self,x,inplace=True): 
        psi = self.psi if inplace else self.psi.copy()
        ls = self.split_vec(x)
        for ix,(_,_,site) in enumerate(self.constructors):
            psi[self.site_tag(site)].modify(data=self.vec2tensor(ls[ix],ix))
        return psi
    def write_tn_to_disc(self,tn,fname):
        return write_tn_to_disc(tn,fname)
    def update(self,x,fname=None,root=0):
        psi = self.vec2psi(x,inplace=True)
        self.set_psi(psi) 
        if RANK==root:
            if fname is not None: # save psi to disc
                self.write_tn_to_disc(psi,fname)
        return psi
##### tsr methods #####
    def get_data_map(self,phys_dim=2):
        data_map = dict()
        for i in range(phys_dim):
            data = np.zeros(phys_dim)
            data[i] = 1.
            data_map[i] = data
        return data_map
    def tensor2backend(self,data,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        return tensor2backend(data,backend,requires_grad=requires_grad)
    def tensor2vec(self,tsr,ix=None):
        return self.tensor2backend(tsr,backend='numpy').flatten()
    def vec2tensor(self,x,ix):
        shape = self.constructors[ix][0]
        return self.tensor2backend(x.reshape(shape))
    def tensor_grad(self,tsr,set_zero=True):
        grad = tsr.grad
        if set_zero:
            tsr.grad = None
        return grad 
    def get_bra_tsr(self,ci,site,append=''):
        inds = self.site_ind(site)+append,
        tags = self.site_tags(site) + ('BRA',)
        data = self.data_map[ci]
        return Tensor(data=data,inds=inds,tags=tags)
    def site_grad(self,tn,site):
        tid = tuple(tn._get_tids_from_tags((self.site_tag(site),'KET'),which='all'))[0]
        ket = tn._pop_tensor(tid)
        g = tn.contract(output_inds=ket.inds,tags=all)
        return g.data 
    def replace_sites(self,tn,sites,cis):
        for site,ci in zip(sites,cis): 
            bra = tn[self.site_tag(site),'BRA']
            bra_target = self.get_bra_tsr(ci,site)
            bra.modify(data=bra_target.data,inds=bra_target.inds)
        return tn
    def tensor_compress_bond(self,T1,T2,absorb='right'):
        self._tensor_compress_bond(T1,T2,absorb=absorb)
    def _tensor_compress_bond(self,T1,T2,absorb='right'):
        # TODO:check for absorb='left'
        shared_ix, left_env_ix = T1.filter_bonds(T2)
        if not shared_ix:
            raise ValueError("The tensors specified don't share an bond.")
        assert len(shared_ix)==1
        T1_inds,T2_inds = T1.inds,T2.inds
    
        tmp_ix = rand_uuid()
        T1.reindex_({shared_ix[0]:tmp_ix})
        T2.reindex_({shared_ix[0]:tmp_ix})
        if absorb=='right': # assume T2 is isometric
            T1_L, T1_R = T1.split(left_inds=left_env_ix, right_inds=(tmp_ix,), absorb='right',
                                  get='tensors', method='qr')
            M,T2_R = T1_R,T2
        elif absorb=='left': # assume T1 is isometric
            T2_L, T2_R = T2.split(left_inds=(tmp_ix,), absorb='left',get='tensors', method='lq')
            T1_L,M = T1,T2_L
        else:
            raise NotImplementedError(f'absorb={absorb}')
        M_L, *s, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                               absorb=absorb, **self.compress_opts)
    
        # make sure old bond being used
        ns_ix, = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})
    
        T1C = T1_L.contract(M_L)
        T2C = M_R.contract(T2_R)
    
        # update with the new compressed data
        T1.modify(data=T1C.data, inds=T1C.inds)
        T2.modify(data=T2C.data, inds=T2C.inds)
##### ham methods #####
    def parse_config(self,config):
        return config
    def intermediate_sign(self,config=None,ix1=None,ix2=None):
        return 1.
    def config_sign(self,config=None):
        return 1.
    def get_grad_from_plq(self,plq,to_vec=True):
        vx = dict()
        for plq_key,tn in plq.items():
            cij = tensor2backend(safe_contract(tn.copy()),'numpy')
            if cij is None:
                continue
            sites = self.plq_sites(plq_key)
            for site in sites:
                if site in vx:
                    continue
                vx[site] = self.site_grad(tn.copy(),site)/cij
        if to_vec:
            vx = self.dict2vec(vx)
        return vx
    def _add_gate(self,tn,gate,order,where,contract=True):
        return _add_gate(tn.copy(),gate,order,where,self.site_ind,self.site_tag,contract=contract)
    def pair_energy_from_plq(self,tn,config,where,model):
        ix1,ix2 = [model.flatten(where[ix]) for ix in (0,1)]
        i1,i2 = config[ix1],config[ix2] 
        if not model.pair_valid(i1,i2): # term vanishes 
            return None 
        ex = self._add_gate(tn.copy(),model.gate,model.order,where,contract=True)
        if ex is None:
            return None
        return model.pair_coeff(*where) * ex 
    def parse_hessian_from_plq(self,Hvx,vx,ex,eu,cx):
        vx = self.dict2vec(vx)
        cx,err = contraction_error(cx) 
        return cx,ex+eu,vx,Hvx/cx+eu*vx,err
    def parse_hessian_deterministic(self,Hvx,vx,ex,eu,cx):
        return cx,ex/cx+eu,vx,Hvx/cx+eu*vx,0.
    def parse_energy_deterministic(self,ex,cx,to_numpy=True):
        ex = sum(ex.values())/cx
        if to_numpy:
            ex = tensor2backend(ex,'numpy')
        return ex,cx
    def parse_energy_from_plq(self,ex,cx):
        if len(cx)==0:
            return 0.
        ex = sum([eij/cx[where] for where,eij in ex.items()])
        return tensor2backend(ex,'numpy')
    def parse_derivative(self,ex,cx=None):
        if len(ex)==0:
            return 0.,0.,np.zeros(self.nparam)
        ex_num = sum(ex.values())
        ex_num.backward()
        Hvx = {site:self.tensor_grad(self.psi[self.site_tag(site)].data) for site in self.sites}
        if cx is None:
            ex = None
        elif isinstance(cx,dict):
            ex = self.parse_energy_from_plq(ex,cx)
        else:
            ex = self.parse_energy_deterministic(ex,cx)
        return ex,tensor2backend(ex_num,'numpy'),self.dict2vec(Hvx)  
    def get_grad_deterministic(self,config,unsigned=False):
        self.wfn2backend(backend='torch',requires_grad=True)
        cache_top = dict()
        cache_bot = dict()
        cx = self.unsigned_amplitude(config,cache_top=cache_top,cache_bot=cache_bot,to_numpy=False)
        if cx is None:
            vx = np.zeros(self.nparam)
        else:
            _,cx,vx = self.parse_derivative({0:cx})
            vx /= cx
            sign = 1. if unsigned else self.config_sign(config)
            cx *= sign
        self.wfn2backend()
        return cx,vx
##### sampler methods ######
    def amplitude(self,config):
        raise NotImplementedError
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
    def _new_prob_from_plq(self,plq,sites,cis):
        plq_new = self.replace_sites(plq.copy(),sites,cis) 
        cy = safe_contract(plq_new)
        if cy is None:
            return plq_new,None
        return plq_new,tensor2backend(cy**2,'numpy') 
class Model:
    def gate2backend(self,backend):
        self.gate = tensor2backend(self.gate,backend)
class Hamiltonian:
    def __init__(self,model):
        self.model = model
    def _pair_energies_from_plq(self,plq,pairs,config,af=None):
        ex = dict()
        cx = dict()
        af = self.amplitude_factory if af is None else af
        for where in pairs:
            key = self.model.pair_key(*where)

            tn = plq.get(key,None) 
            if tn is not None:
                cij = tensor2backend(safe_contract(tn.copy()),'numpy')
                if cij is None:
                    continue
                cx[where] = cij

                eij = af.pair_energy_from_plq(tn,config,where,self.model) 
                if eij is None:
                    continue
                ex[where] = eij
        return ex,cx
    def batch_hessian_from_plq(self,batch_idx,config): # only used for Hessian
        af = self.amplitude_factory
        af.wfn2backend(backend='torch',requires_grad=True)
        self.model.gate2backend('torch')
        ex,cx,plq = self.batch_pair_energies_from_plq(batch_idx,config)

        ex,_,Hvx = af.parse_derivative(ex,cx=cx)
        vx = af.get_grad_from_plq(plq,to_vec=False) 
        vx = list2dict(vx)
        cx = list2dict(cx)
        af.wfn2backend()
        self.model.gate2backend(af.backend)
        return ex,Hvx,cx,vx
    def compute_local_energy_hessian_from_plq(self,config): 
        ex,Hvx = 0.,0.
        cx,vx = dict(),dict()
        for batch_idx in self.model.batched_pairs:
            ex_,Hvx_,cx_,vx_ = self.batch_hessian_from_plq(batch_idx,config)  
            ex += ex_
            Hvx += Hvx_
            cx.update(cx_)
            vx.update(vx_)
        eu = self.model.compute_local_energy_eigen(config)
        return self.amplitude_factory.parse_hessian_from_plq(Hvx,vx,ex,eu,cx)
    def compute_local_energy_gradient_from_plq(self,config,compute_v=True):
        ex,cx,plq = self.pair_energies_from_plq(config)

        af = self.amplitude_factory
        ex = af.parse_energy_from_plq(ex,cx)
        eu = self.model.compute_local_energy_eigen(config)
        ex += eu

        if not compute_v:
            cx,err = contraction_error(cx)
            return cx,ex,None,None,err 
        vx = af.get_grad_from_plq(plq)  
        cx,err = contraction_error(cx)
        return cx,ex,vx,None,err
    def batch_hessian_deterministic(self,config,batch_imin,batch_imax):
        af = self.amplitude_factory
        af.wfn2backend(backend='torch',requires_grad=True)
        ex = self.batch_pair_energies_deterministic(config,batch_imin,batch_imax)
        _,ex,Hvx = af.parse_derivative(ex)
        af.wfn2backend()
        return ex,Hvx
    def pair_hessian_deterministic(self,config,site1,site2):
        af = self.amplitude_factory
        af.wfn2backend(backend='torch',requires_grad=True)
        ex = self.pair_energy_deterministic(config,site1,site2)
        _,ex,Hvx = af.parse_derivative(ex)
        af.wfn2backend()
        return ex,Hvx 
    def compute_local_energy_hessian_deterministic(self,config):
        af = self.amplitude_factory
        cx,vx = af.get_grad_deterministic(config)

        ex = 0. 
        Hvx = 0.
        for key in self.model.batched_pairs:
            if key=='pbc':
                continue
            imin,imax = key
            ex_,Hvx_ = self.batch_hessian_deterministic(config,imin,imax) 
            ex += ex_
            Hvx += Hvx_
        if af.pbc:
            for site1,site2 in self.batched_pairs['pbc']:
                ex_,Hvx_ = self.pair_hessian_deterministic(self,config,site1,site2)
                ex += ex_
                Hvx += Hvx_
         
        eu = self.model.compute_local_energy_eigen(config)
        return af.parse_hessian_deterministic(Hvx,vx,ex,eu,cx)
    def compute_local_energy_gradient_deterministic(self,config,compute_v=True):
        af = self.amplitude_factory
        ex = self.pair_energies_deterministic(config)
        if compute_v:
            cx,vx = af.get_grad_deterministic(config)
        else:
            cx = af.unsigned_amplitude(config)
            sign = af.config_sign(config)
            cx *= sign
            vx = None
        if cx is None:
            return 0.,0.,vx,None,0.
        ex,cx = af.parse_energy_deterministic(ex,cx) 
        eu = self.model.compute_local_energy_eigen(config)
        ex += eu
        return cx,ex,vx,None,0.
    def compute_local_energy(self,config,compute_v=True,compute_Hv=False):
        config = self.amplitude_factory.parse_config(config)
        if self.amplitude_factory.deterministic:
            if compute_Hv:
                return self.compute_local_energy_hessian_deterministic(config)
            else:
                return self.compute_local_energy_gradient_deterministic(config,compute_v=compute_v)
        else:
            if compute_Hv:
                return self.compute_local_energy_hessian_from_plq(config)
            else:
                return self.compute_local_energy_gradient_from_plq(config,compute_v=compute_v)
def get_gate1():
    return np.array([[1,0],
                   [0,-1]]) * .5
def get_gate2(j,to_bk=False,to_matrix=False):
    sx = np.array([[0,1],
                   [1,0]]) * .5
    sy = np.array([[0,-1],
                   [1,0]]) * 1j * .5
    sz = np.array([[1,0],
                   [0,-1]]) * .5
    try:
        jx,jy,jz = j
    except TypeError:
        j = j,j,j
    data = 0.
    for coeff,op in zip(j,[sx,sy,sz]):
        data += coeff * np.tensordot(op,op,axes=0).real
    if to_bk:
        data = data.transpose(0,2,1,3)
    if to_matrix:
        data = data.reshape((4,4))
    return data
