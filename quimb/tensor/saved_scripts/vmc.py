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
        self.nsite = ham.nsite

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
            cx,ex,vx,Hvx,err = self.ham.compute_local_energy(
                config,self.sampler.amplitude_factory,compute_v=compute_v,compute_Hv=compute_Hv)
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
            cx,ex,vx,Hvx,err = self.ham.compute_local_energy(config,self.sampler.amplitude_factory,
                                                        compute_v=compute_v,compute_Hv=compute_Hv)
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
class SpinDenseSampler:
    def __init__(self,nsite,nspin,exact=False,seed=None,thresh=1e-14):
        self.nsite = nsite 
        self.nspin = nspin

        self.all_configs = self.get_all_configs()
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
    def get_all_configs(self):
        assert isinstance(self.nspin,tuple)
        sites = list(range(self.nsite))
        occs = list(itertools.combinations(sites,self.nspin[0]))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = [0] * (self.nsite) 
            for ix in occ:
                config[ix] = 1
            configs[i] = tuple(config)
        return configs
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        config = self.all_configs[flat_idx]
        omega = self.p[flat_idx]
        return config,omega
class FermionDenseSampler(SpinDenseSampler):
    def __init__(self,nsite,nelec,spinless=False,**kwargs):
        self.nelec = nelec
        self.spinless = spinless
        nspin = (nelec,) if spinless else None
        super().__init__(nsite,nspin,**kwargs)
    def get_all_configs(self):
        if self.spinless:
            return super().get_all_configs()
        return self.get_all_configs_u11()
    def get_all_configs_u11(self):
        assert isinstance(self.nelec,tuple)
        sites = list(range(self.nsite))
        ls = [None] * 2
        for spin in (0,1):
            occs = list(itertools.combinations(sites,self.nelec[spin]))
            configs = [None] * len(occs) 
            for i,occ in enumerate(occs):
                config = [0] * self.nsite 
                for ix in occ:
                    config[ix] = 1
                configs[i] = tuple(config)
            ls[spin] = configs

        na,nb = len(ls[0]),len(ls[1])
        configs = [None] * (na*nb)
        for ixa,configa in enumerate(ls[0]):
            for ixb,configb in enumerate(ls[1]):
                config = [config_map[configa[i],configb[i]] \
                          for i in range(self.nsite)]
                ix = ixa * nb + ixb
                configs[ix] = tuple(config)
        return configs
    def get_all_configs_u1(self):
        if isinstance(self.nelec,tuple):
            self.nelec = sum(self.nelec)
        sites = list(range(self.nsite*2))
        occs = list(itertools.combinations(sites,self.nelec))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = [0] * (self.nsite*2) 
            for ix in occ:
                config[ix] = 1
            configs[i] = tuple(config)

        for ix in range(len(configs)):
            config = configs[ix]
            configa,configb = config[:self.nsite],config[self.nsite:]
            config = [config_map[configa[i],configb[i]] for i in range(self.nsite)]
            configs[ix] = tuple(config)
        return configs
#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
import pickle,uuid
from .fermion.fermion_core import FermionTensor,FermionTensorNetwork
def scale_wfn(psi,scale):
    for tid in psi.tensor_map:
        tsr = psi.tensor_map[tid]
        tsr.modify(data=tsr.data*scale)
    return psi
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
def load_ftn_from_disc(fname, delete_file=False):

    # Get the data
    if type(fname) != str:
        data = fname
    else:
        # Open up the file
        with open(fname, 'rb') as f:
            data = pickle.load(f)

    # Set up a dummy fermionic tensor network
    tn = FermionTensorNetwork([])

    # Put the tensors into the ftn
    tensors = [None,] * data['ntensors']
    for i in range(data['ntensors']):

        # Get the tensor
        ten_info = data['tensors'][i]
        ten = ten_info['tensor']
        ten = FermionTensor(ten.data, inds=ten.inds, tags=ten.tags)

        # Get/set tensor info
        tid, site = ten_info['fermion_info']
        ten.fermion_owner = None
        ten._avoid_phase = False

        # Add the required phase
        ten.phase = ten_info['phase']

        # Add to tensor list
        tensors[site] = (tid, ten)

    # Add tensors to the tn
    for (tid, ten) in tensors:
        tn.add_tensor(ten, tid=tid, virtual=True)

    # Get addition attributes needed
    tn_info = data['tn_info']

    # Set all attributes in the ftn
    extra_props = dict()
    for props in tn_info:
        extra_props[props[1:]] = tn_info[props]

    # Convert it to the correct type of fermionic tensor network
    tn = tn.view_as_(data['class'], **extra_props)

    # Remove file (if desired)
    if delete_file:
        delete_ftn_from_disc(fname)

    # Return resulting tn
    return tn
def rand_fname():
    return str(uuid.uuid4())
def write_ftn_to_disc(tn, tmpdir, provided_filename=False):

    # Create a generic dictionary to hold all information
    data = dict()

    # Save which type of tn this is
    data['class'] = type(tn)

    # Add information relevant to the tensors
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)

    # Add the tensors themselves
    data['tensors'] = []
    ntensors = 0
    for ten in tn.tensors:
        ten_info = dict()
        ten_info['fermion_info'] = ten.get_fermion_info()
        ten_info['phase'] = ten.phase
        ten_info['tensor'] = ten
        data['tensors'].append(ten_info)
        ntensors += 1
    data['ntensors'] = ntensors

    # If tmpdir is None, then return the dictionary
    if tmpdir is None:
        return data

    # Write fermionic tensor network to disc
    else:
        # Create a temporary file
        if provided_filename:
            fname = tmpdir
            print('saving to ', fname)
        else:
            if tmpdir[-1] != '/': 
                tmpdir = tmpdir + '/'
            fname = tmpdir + rand_fname()

        # Write to a file
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        # Return the filename
        return fname

