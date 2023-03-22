import time,scipy,functools
import numpy as np
import scipy.sparse.linalg as spla

from quimb.utils import progbar as Progbar
from .utils import load_ftn_from_disc,write_ftn_to_disc
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
DISCARD = 1e3
CG_TOL = 1e-4
class TNVMC: # stochastic sampling
    def __init__(
        self,
        ham,
        sampler,
        amplitude_factory,
        optimizer='sr',
        full_matrix=False,
        **kwargs,
    ):
        # parse ham
        self.ham = ham

        # parse sampler
        self.config = None
        self.batchsize = None
        self.sampler = sampler
        self.exact_sampling = sampler.exact

        # parse wfn 
        self.amplitude_factory = amplitude_factory         
        self.x = self.amplitude_factory.get_x()
        self.nparam = len(self.x)

        # parse gradient optimizer
        self.optimizer = optimizer
        if self.optimizer in ['rgn','lin']:
            self.ham.initialize_pepo(self.amplitude_factory.psi)
        if self.optimizer=='lin':
            self.xi = kwargs.get('xi',0.5)
            self.solver = kwargs.get('solver','davidson')
            # only used for iterative full Hessian
            if self.solver == 'davidson':
                maxsize = kwargs.get('maxsize',25)
                maxiter = kwargs.get('maxiter',100)
                restart_size = kwargs.get('restart_size',5)
                from .davidson import davidson
                self.davidson = functools.partial(davidson,
                    maxsize=maxsize,restart_size=restart_size,maxiter=maxiter,tol=CG_TOL) 
        self.full_matrix = full_matrix
        if self.optimizer=='sr':
            self.mask = False
        elif self.exact_sampling:
            self.mask = False
        elif self.full_matrix:
            self.mask = False
        else:
            self.mask = kwargs.get('mask',False)
        if self.mask:
            self.block_dict = self.amplitude_factory.get_block_dict()
    def run(self,start,stop,tmpdir=None,
            rate_start=1e-1,
            rate_stop=1e-1,
            cond_start=1e-5,
            cond_stop=1e-5,
            rate_itv=None, # prapagate rate over rate_itv
            cond_itv=None, # propagate cond over cond_itv
        ):
        # change rate & conditioner as in Webber & Lindsey
        self.start = start
        self.stop = stop
        steps = stop - start
        self.rate_start = rate_start
        self.rate_stop = rate_stop
        self.rate_itv = steps if rate_itv is None else rate_itv
        self.rate_base = (self.rate_stop/self.rate_start)**(1./self.rate_itv)
        self.cond_start = cond_start
        self.cond_stop = cond_stop
        self.cond_itv = steps if cond_itv is None else cond_itv
        self.cond_base = (self.rate_stop/self.rate_start)**(1./self.rate_itv)
        self.rate = self.rate_start
        self.cond = self.cond_start

        for step in range(start,stop):
            self.step = step
            self.propagate_rate_cond()
            self.sample()
            self.extract_energy_gradient()
            self.transform_gradients(self.cond)
            if RANK==0:
                self.x -= self.rate * self.deltas
                print('\tx norm=',np.linalg.norm(self.x))
            
            COMM.Bcast(self.x,root=0) 
            psi = self.amplitude_factory.update(self.x)
            if RANK==0:
                if tmpdir is not None: # save psi to disc
                    write_ftn_to_disc(psi,tmpdir+f'psi{step+1}',provided_filename=True)
    def propagate_rate_cond(self):
        if RANK>0:
            return
        if self.step < self.start + self.rate_itv:
            self.rate *= self.rate_base
        if self.step < self.start + self.cond_itv:
            self.cond *= self.cond_base
    def sample(self):
        self.sampler.amplitude_factory = self.amplitude_factory
        if self.exact_sampling:
            self.sample_exact()
        else:
            self.sample_stochastic()
    def sample_stochastic(self): 
        self.terminate = np.array([0])
        self.rank = np.array([RANK])
        if RANK==0:
            self._ctr()
        else:
            self._sample()
    def _ctr(self):
        ncurr = 0
        ntotal = self.batchsize * SIZE
        while self.terminate[0]==0:
            COMM.Recv(self.rank,tag=0)
            ncurr += 1
            if ncurr > ntotal: # send termination message to all workers
                self.terminate[0] = 1
                for worker in range(1,SIZE):
                    COMM.Bsend(self.terminate,dest=worker,tag=1)
            else:
                COMM.Bsend(self.terminate,dest=self.rank[0],tag=1)
    def _sample(self):
        self.sampler.preprocess(self.config) 

        self.samples = []
        self.elocal = []
        self.vlocal = []
        if self.optimizer in ['rgn','lin']:
            compute_Hv = True
            self.Hv_local = [] 
        else: 
            compute_Hv = False

        self.store = dict()
        self.p0 = dict()
        t0 = time.time()
        while self.terminate[0]==0:
            config,omega = self.sampler.sample()
            if config in self.store:
                info = self.store[config]
                if info is None:
                    continue
                ex,vx,Hvx = info 
            else:
                cx,ex,vx,Hvx = self.ham.compute_local_energy(config,self.amplitude_factory,
                                                             compute_Hv=compute_Hv)
                if np.fabs(ex) > DISCARD:
                    self.store[config] = None
                    continue
                self.store[config] = ex,vx,Hvx
                self.p0[config] = cx**2
            self.samples.append(config)
            self.elocal.append(ex)
            self.vlocal.append(vx)
            if compute_Hv:
                self.Hv_local.append(Hvx)

            COMM.Bsend(self.rank,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
        if RANK==SIZE-1:
            print('\tstochastic sample time=',time.time()-t0)
    def sample_exact(self): 
        self.sampler.compute_dense_prob() # runs only for dense sampler 

        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros

        self.samples = []
        self.flocal = []
        self.elocal = []
        self.vlocal = []
        if self.optimizer in ['rgn','lin']:
            compute_Hv = True 
            self.Hv_local = [] 
        else: 
            compute_Hv = False

        t0 = time.time()
        self.store = dict()
        for ix in ixs:
            self.flocal.append(p[ix])
            config = all_configs[ix]
            self.samples.append(config) 
            _,ex,vx,Hvx = self.ham.compute_local_energy(config,self.amplitude_factory,
                                                      compute_Hv=compute_Hv)
            self.elocal.append(ex)
            self.vlocal.append(vx)
            if compute_Hv:
                self.Hv_local.append(Hvx)
        if RANK==SIZE-1:
            print('\texact sample time=',time.time()-t0)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        if self.optimizer in ['sr','rgn','lin']:
            self.extract_S()
        if self.optimizer in ['rgn','lin']:
            self.extract_H()
        if RANK==0:
            print('\tcollect data time=',time.time()-t0)
            print('\tnormalization=',self.n)
            print('\tgradient norm=',np.linalg.norm(self.g))
            print(f'step={self.step},energy={self.E},err={self.Eerr}')
    def gather_sizes(self):
        self.count = np.array([0]*SIZE)
        COMM.Allgather(np.array([self.nlocal]),self.count)
        self.disp = np.concatenate([np.array([0]),np.cumsum(self.count[:-1])])
    def extract_energy(self):
        if self.exact_sampling:
            self._extract_energy_exact()
        else:
            self._extract_energy_stochastic()
    def _extract_energy_stochastic(self):
        # collect all energies for blocking analysis 
        if RANK==0:
            self.nlocal = 1
            self.elocal = np.zeros(1)
        else:
            self.nlocal = len(self.samples)
            self.elocal = np.array(self.elocal)
        self.gather_sizes()
        self.n = self.count.sum()
        e = np.zeros(self.n)
        COMM.Gatherv(self.elocal,[e,self.count,self.disp,MPI.DOUBLE],root=0)
        if RANK>0:
            return
        e = e[1:]
        self.n -= 1
        self.E,self.Eerr = blocking_analysis(np.ones_like(e),e,0,True)
    def _extract_energy_exact(self):
        # reduce scalar energy
        self.nlocal = len(self.samples)
        self.gather_sizes()

        self.elocal = np.array(self.elocal)
        self.flocal = np.array(self.flocal)
        e = np.array([np.dot(self.elocal,self.flocal)])
        self.E = np.zeros_like(e) 
        COMM.Reduce(e,self.E,op=MPI.SUM,root=0)
        if RANK>0:
            return
        self.E = self.E[0]
        self.Eerr = 0.
        self.n = 1.
    def extract_gradient(self):
        # reduce vectors
        if self.exact_sampling:
            self.vlocal = np.array(self.vlocal)
            vsum_ = np.dot(self.flocal,self.vlocal) 
            vesum_ = np.dot(self.elocal * self.flocal,self.vlocal)
        else:
            if RANK==0:
                vsum_ = np.zeros(self.nparam)
                vesum_ = np.zeros(self.nparam)
            else:
                self.vlocal = np.array(self.vlocal)
                vsum_ = self.vlocal.sum(axis=0)
                vesum_ = np.dot(self.elocal,self.vlocal)
        self.vmean = np.zeros_like(vsum_)
        COMM.Reduce(vsum_,self.vmean,op=MPI.SUM,root=0)
        vesum = np.zeros_like(vesum_)
        COMM.Reduce(vesum_,vesum,op=MPI.SUM,root=0)
        if RANK>0:
            return
        self.vmean /= self.n
        self.g = vesum / self.n - self.E * self.vmean 
    def extract_S(self):
        if self.mask:
            assert not self.exact_sampling
            self._extract_S_mask()
        else:
            if self.full_matrix:
                self.S = self._get_Smatrix()
            else:
                self._extract_S_full()
    def _get_Smatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        if RANK==0:
            sh = stop-start
            vvsum_ = np.zeros((sh,)*2)
        else:
            v = self.vlocal[:,start:stop] 
            vvsum_ = np.dot(v.T,v)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean = self.vmean[start:stop]
            S = vvsum / self.n - np.outer(vmean,vmean)
        return S
    def _extract_S_mask(self):
        # stochastic only, collect matrix blocks
        ls = [None] * len(self.block_dict)
        for ix,(start,stop) in enumerate(self.block_dict):
            ls[ix] = self._get_Smatrix(start=start,stop=stop)
        self.S = ls
    def _extract_S_full(self):
        if self.exact_sampling:
            self.f = np.zeros(self.count.sum())
            COMM.Gatherv(self.flocal,[self.f,self.count,self.disp,MPI.DOUBLE],root=0)
        # construct matvec
        if RANK>0:
            COMM.Ssend(self.vlocal,dest=0,tag=4)
            return
        v = [self.vlocal] if self.exact_sampling else []
        for worker in range(1,SIZE):
            nlocal = self.count[worker]
            buf = np.zeros((nlocal,self.nparam))
            COMM.Recv(buf,source=worker,tag=4)
            v.append(buf)    
        self.v = np.concatenate(v,axis=0) 
        if self.exact_sampling:
            def matvec(x):
                return np.dot(self.f * np.dot(self.v,x),self.v) - self.vmean * np.dot(self.vmean,x)
        else:
            def matvec(x):
                return np.dot(np.dot(self.v,x),self.v) / self.n - self.vmean * np.dot(self.vmean,x)
        self.S = matvec
    def extract_H(self):
        self._extract_Hvmean()
        if self.mask:
            assert not self.exact_sampling
            self._extract_H_mask()
        else:
            if self.full_matrix:
                self.H = self._get_Hmatrix()
            else:
                self._extract_H_full()
    def _extract_Hvmean(self):
        if self.exact_sampling:
            self.Hv_local = np.array(self.Hv_local)
            Hvsum_ = np.dot(self.flocal,self.Hv_local)
        else:
            if RANK==0:
                Hvsum_ = np.zeros(self.nparam)
            else:
                self.Hv_local = np.array(self.Hv_local)
                Hvsum_ = self.Hv_local.sum(axis=0)
        self.Hvmean = np.zeros_like(Hvsum_)
        COMM.Reduce(Hvsum_,self.Hvmean,op=MPI.SUM,root=0)
        if RANK==0:
            self.Hvmean /= self.n
    def _get_Hmatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        if RANK==0:
            sh = stop-start
            vHvsum_ = np.zeros((sh,)*2)
        else:
            v = self.vlocal[:,start:stop] 
            Hv = self.Hv_local[:,start:stop] 
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
    def _extract_H_mask(self):
        ls = [None] * len(self.block_dict)
        for ix,(start,stop) in enumerate(self.block_dict):
            ls[ix] = self._get_Hmatrix(start=start,stop=stop)
        self.H = ls
    def _extract_H_full(self):
        if RANK>0:
            COMM.Ssend(self.Hv_local,dest=0,tag=5)
            return
        Hv = [self.Hv_local] if self.exact_sampling else []
        for worker in range(1,SIZE):
            nlocal = self.count[worker]
            buf = np.zeros((nlocal,self.nparam))
            COMM.Recv(buf,source=worker,tag=5)
            Hv.append(buf)    
        Hv = np.concatenate(Hv,axis=0) 
        if self.exact_sampling:
            def matvec(x):
                return np.dot(self.f * np.dot(Hv,x),self.v) - self.vmean * np.dot(self.Hvmean,x) \
                                                            - self.g * np.dot(self.vmean,x)
        else:
            def matvec(x):
                return np.dot(np.dot(Hv,x),self.v) / self.n - self.vmean * np.dot(self.Hvmean,x) \
                                                            - self.g * np.dot(self.vmean,x)
        self.Hv = Hv
        self.H = matvec
    def transform_gradients(self,cond):
        if RANK>0:
            return 
        if self.optimizer=='sr':
            self._transform_gradients_sr(cond)
        elif self.optimizer=='rgn':
            self._transform_gradients_rgn(cond)
        elif self.optimizer=='lin':
            self._transform_gradients_lin(cond)
        else:
            self._transform_gradients_sgd()
        print('\tdelta norm=',np.linalg.norm(self.deltas))
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
    def _transform_gradients_sr(self,cond):
        t0 = time.time()
        def A(x):
            return self.S(x) + cond * x
        LinOp = spla.LinearOperator((self.nparam,self.nparam),matvec=A,dtype=self.g.dtype)
        self.deltas,info = spla.minres(LinOp,self.g,tol=CG_TOL)
        print('\tSR solver exit status=',info)
        print('\tSR solver time=',time.time()-t0)
    def _transform_gradients_rgn(self,cond):
        t0 = time.time()
        if self.mask:
            self._transform_gradients_rgn_mask(cond)
        else:
            self._transform_gradients_rgn_full(cond)
        print('\tRGN solver time=',time.time()-t0)
    def _transform_gradients_rgn_mask(self,cond):
        self.deltas = np.zeros_like(self.x)
        for ix,(start,stop) in enumerate(self.block_dict):
            H = self.H[ix] - self.E * self.S[ix] + cond * np.eye(stop-start)
            self.deltas[start:stop] = np.linalg.solve(H,self.g[start:stop])
        #H = np.zeros((self.nparam,self.nparam))
        #for ix,(start,stop) in enumerate(self.block_dict):
        #    H[start:stop,start:stop] = self.H[ix] - self.E * self.S[ix] + cond * np.eye(stop-start)
        #self.deltas = np.linalg.solve(H,self.g)
    def _transform_gradients_rgn_full(self,cond):
        if self.full_matrix:
            H = self.H - self.E * self.S + cond * np.eye(self.nparam)
            self.deltas = np.linalg.solve(H,self.g)
        else:
            def A(x):
                return self.H(x) - self.E * self.S(x) + cond * x 
            LinOp = spla.LinearOperator((self.nparam,self.nparam),matvec=A,dtype=self.g.dtype)
            self.deltas,info = spla.lgmres(LinOp,self.g,tol=CG_TOL)
            print('\tRGN solver exit status=',info)
    def _transform_gradients_lin(self,cond):
        t0 = time.time()
        if self.mask:
            self._transform_gradients_lin_mask(cond)
        else:
            self._transform_gradients_lin_full(cond)
        self._scale_eigenvector()
        print('\tEIG solver time=',time.time()-t0)
    def _scale_eigenvector(self):
        if self.xi is None:
            Ns = self.vmean
        else:
            if self.full_matrix:
                Sp = np.dot(self.S,self.deltas)
            else:
                if self.mask:
                    Sp = np.zeros_like(self.x)
                    for ix,(start,stop) in enumerate(self.block_dict):
                        Sp[start:stop] = np.dot(self.S[ix],self.deltas[start:stop])
                else:
                    Sp = self.S(self.deltas)
            Ns  = - (1.-self.xi) * Sp 
            Ns /= 1.-self.xi + self.xi * (1.+np.dot(self.deltas,Sp))**.5
        denom = 1. - np.dot(Ns,self.deltas)
        self.deltas /= -denom
        print('\tscale2=',denom)
    def _transform_gradients_lin_mask(self,cond):
        Hi0 = self.g
        H0j = self.Hvmean - self.E * self.vmean
        ws = np.zeros(len(self.block_dict))
        v0 = np.zeros(len(self.block_dict))
        self.deltas = np.zeros_like(self.x)
        imag_norm = 0. 
        for ix,(start,stop) in enumerate(self.block_dict):
            sh = stop - start
            A = np.block([[np.ones((1,1))*self.E,H0j[start:stop].reshape(1,sh)],
                          [Hi0[start:stop].reshape(sh,1),self.H[ix]]])
            B = np.block([[np.ones((1,1)),np.zeros((1,sh))],
                          [np.zeros((sh,1)),self.S[ix]+cond*np.eye(sh)]])
            w,v = scipy.linalg.eig(A,b=B) 
            ws[ix],self.deltas[start:stop],idx = _select_eigenvector(w.real,v.real)
            imag_norm += np.linalg.norm(v[:,idx].imag)
            v0[ix] = v[0,idx].real
        print('\timaginary norm=',imag_norm)
        print('\teigenvalue =',ws)
        print('\tscale1=',v0)
    def _transform_gradients_lin_full(self,cond):
        Hi0 = self.g
        H0j = self.Hvmean - self.E * self.vmean
        if self.full_matrix:
            A = np.block([[np.ones((1,1))*self.E,H0j.reshape(1,self.nparam)],
                          [Hi0.reshape(self.nparam,1),self.H]])
            B = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
                          [np.zeros((self.nparam,1)),self.S+cond*np.eye(self.nparam)]])
            w,v = scipy.linalg.eig(A,b=B) 
            w,self.deltas,idx = _select_eigenvector(w.real,v.real)
            print('\timaginary norm=',np.linalg.norm(v[:,idx].imag))
            print('\teigenvalue =',w)
            print('\tscale1=',v[0,idx].real)
        else:
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
def _select_eigenvector(w,v):
    #if min(w) < self.E - self.revert:
    #    dist = (w-self.E)**2
    #    idx = np.argmin(dist)
    #else:
    #    idx = np.argmin(w)
    z0_sq = v[0,:] ** 2
    idx = np.argmax(z0_sq)
    v = v[1:,idx]/v[0,idx]
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

    if printQ:
        if plateauError is not None:
            print(f'Stocahstic error estimate: {plateauError:.6e}\n')

    return meanEnergy, plateauError
