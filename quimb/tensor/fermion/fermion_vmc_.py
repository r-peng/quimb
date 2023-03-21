import time,h5py,itertools,pickle,sys,scipy
import numpy as np
import scipy.sparse.linalg as spla
#from scipy.optimize import line_search

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
        #conditioner='auto',
        conditioner=None,
        optimizer='sr',
        extrapolator=None,
        search_rate=False,
        search_cond=False,
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

        # if need to condition, try making the element of psi-vec O(1)
        if conditioner == 'auto':
            def conditioner(x):
                self.x /= np.amax(np.fabs(self.x))
            self.conditioner = conditioner
        else:
            self.conditioner = None
        if self.conditioner is not None:
            self.conditioner(self.x)

        # parse gradient optimizer
        self.optimizer = optimizer
        if self.optimizer in ['rgn','lin']:
            self.ham.initialize_pepo(self.amplitude_factory.psi)
        if self.optimizer=='lin':
            self.xi = kwargs.get('xi',None)
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
            rate_start=1e-1,
            rate_stop=1e-1,
            cond_start=1e-5,
            cond_stop=1e-5,
            rate_itv=None, # prapagate rate over rate_itv
            cond_itv=None, # propagate cond over cond_itv
            revert=1., 
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
        self.revert = revert

        self.E_old = 0.
        for step in range(start,stop):
            self.step = step
            self.propagate_rate_cond()
            self.sample()
            cond,search_cond = self.check_update()
            self.get_direction(cond,search_cond)
            self.regularize_size()
            self.get_rate()
            self.extrapolate()
            if RANK==0:
                if self.conditioner is not None:
                    self.conditioner(self.x)
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
    def check_update(self):
        if RANK>0:
            return None,None
        if self.E - self.E_old > self.revert:
            print('Bad step detected, revert to previous wfn.')
            self.n = self.n_old
            self.g = self.g_old
            if self.optimizer in ['sr','rgn','lin']:
                self.S = self.S_old
            if self.optimizer in ['rgn','lin']:
                self.H = self.H_old
            cond = 1.
            search_cond = False
        else: # valid step, save quantities for later revert
            self.n_old = self.n
            self.g_old = self.g
            if self.optimizer in ['sr','rgn','lin']:
                self.S_old = self.S
            if self.optimizer in ['rgn','lin']:
                self.H_old = self.H
            cond = self.cond
            search_cond = self.search_cond
        print('\tcond=',cond)
        return cond,search_cond
    def regularize_size(self):
        if RANK>0:
            return
        delta_norm = np.linalg.norm(self.deltas)
        print(f'\tdelta norm={delta_norm}')
        #self.deltas *= self.gnorm / delta_norm
        #if self.step == self.start:
        #    denom = np.linalg.norm(self.x)
        #    ratio1 = 1.
        #    ratio2 = 1.
        #else:
        #    denom = self.delta_norm
        #    ratio1 = 10.
        #    ratio2 = 2.
        #ratio = delta_norm / denom 
        #if ratio > ratio1:
        #    print('Warning! Delta ratio=',ratio)
        #if ratio > ratio2:
        #    self.deltas /= ratio
        #    delta_norm /= ratio
        #self.delta_norm = delta_norm
    def extrapolate(self):
        if RANK>0:
            return
        print('\trate=',self.rate)
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
    def sample(self):
        self.sampler.amplitude_factory = self.amplitude_factory
        if self.exact_sampling:
            self.sample_exact()
        else:
            self.sample_stochastic()
        self.extract_energy_gradient()
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
            print(f'step={self.step},energy={self.E},err={self.Eerr}')
            self.gnorm = np.linalg.norm(self.g)
            print('\tgradient norm=',self.gnorm)
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
        #self.deltas = np.zeros_like(self.x)
        #for ix,(start,stop) in enumerate(self.block_dict):
        #    H = self.H[ix] - self.E * self.S[ix] + cond * np.eye(stop-start)
        #    self.deltas[start:stop] = np.linalg.solve(H,self.g[start:stop])
        H = np.zeros((self.nparam,self.nparam))
        for ix,(start,stop) in enumerate(self.block_dict):
            H[start:stop,start:stop] = self.H[ix] - self.E * self.S[ix] + cond * np.eye(stop-start)
        self.deltas = np.linalg.solve(H,self.g)
    def _transform_gradients_rgn_full(self,cond):
        if self.full_matrix:
            H = self.H - self.E * self.S + cond * np.eye(self.nparam)
            self.deltas = np.linalg.solve(H,self.g)
        else:
            def A(x):
                return self.H(x) - self.E * self.S(x) + cond * x 
            LinOp = spla.LinearOperator((self.nparam,self.nparam),matvec=A,dtype=self.g.dtype)
            self.deltas,info = spla.lgmres(LinOp,self.g,tol=CG_TOL)
            #self.deltas,info = spla.minres(LinOp,self.g,tol=CG_TOL)
            print('\tRGN solver exit status=',info)
    def _transform_gradients_lin(self,cond):
        t0 = time.time()
        if self.mask:
            self._transform_gradients_lin_mask(cond)
        else:
            self._transform_gradients_lin_full(cond)
        print('\tEIG solver time=',time.time()-t0)
    def _scale_eigenvector(self,v):
        self.deltas = v[1:].real/v[0].real
        self.deltas = self.deltas.real
        if self.xi is None:
            Ns = self.vmean
        else:
            if self.full_matrix:
                Sp = np.dot(self.S,self.deltas)
            else:
                Sp = self.S(self.deltas)
            Ns  = - (1.-self.xi) * Sp 
            Ns /= 1.-self.xi + self.xi * (1.+np.dot(self.deltas,Sp))**.5
        denom = 1. - np.dot(Ns,self.deltas)
        self.deltas /= -denom
        print('\tscale1=',v[0].real)
        print('\tscale2=',denom)
        print('\timaginary norm=',np.linalg.norm(v.imag))
    def _select_eigenvalue(self,w,v):
        #if min(w.real) < self.E - self.revert:
        #    dist = (w.real-self.E)**2
        #    idx = np.argmin(dist)
        #else:
        #    idx = np.argmin(w.real)
        z0_sq = v[0,:].real ** 2
        idx = np.argmax(z0_sq)
        return w[idx],v[:,idx]
    def _transform_gradients_lin_mask(self,cond):
        H = np.zeros((self.nparam,self.nparam))
        S = np.zeros((self.nparam,self.nparam))
        for ix,(start,stop) in enumerate(self.block_dict):
            H[start:stop,start:stop] = self.H[ix] 
            S[start:stop,start:stop] = self.S[ix] 
        A = np.block([[np.ones((1,1))*self.E,np.zeros((1,self.nparam))],
                      [np.zeros((self.nparam,1)),H]])
        B = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
                      [np.zeros((self.nparam,1)),S+cond*np.eye(self.nparam)]])
        w,v = scipy.linalg.eig(A,b=B) 
        w,v = self._select_eigenvalue(w,v)
        print('\teigenvalue =',w)
        self._scale_eigenvector(v)
    def _transform_gradients_lin_full(self,cond):
        Hi0 = self.g
        H0j = self.Hvmean - self.E * self.vmean
        if self.full_matrix:
            A = np.block([[np.ones((1,1))*self.E,H0j.reshape(1,self.nparam)],
                          [Hi0.reshape(self.nparam,1),self.H]])
            B = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
                          [np.zeros((self.nparam,1)),self.S+cond*np.eye(self.nparam)]])
            w,v = scipy.linalg.eig(A,b=B) 
            w,v = self._select_eigenvalue(w,v)
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
            A = spla.LinearOperator((self.nparam+1,self.nparam+1),matvec=A,dtype=self.g.dtype)
            B = spla.LinearOperator((self.nparam+1,self.nparam+1),matvec=B,dtype=self.g.dtype)
            w,v = spla.eigs(A,k=1,M=B,sigma=self.E,tol=CG_TOL)
            w,v = w[0],v[:,0]
        print('\teigenvalue =',w)
        self._scale_eigenvector(v)
    def sample_correlated(self):
        if self.exact_sampling:
            raise NotImplementedError
        if RANK>0:
            self.flocal = []
            self.elocal = []
             
            self.store = dict()
            for config in self.samples:
                if config in self.store:
                    fx,ex = self.store[config]
                else:
                    cx,ex,_,_ = self.ham.compute_local_energy(config,self.amplitude_factory,
                                                              compute_v=False,compute_Hv=False)
                    fx = cx**2 / self.p0[config]
                    self.store[config] = fx,ex
                self.flocal.append(fx)
                self.elocal.append(ex)
        return self.extract_correlated_energy()
    def extract_correlated_energy(self):
        if RANK==0:
            flocal = np.zeros(1)
            elocal = np.zeros(1)
        else:
            flocal = np.array(self.flocal)
            elocal = np.array(self.elocal)
        # gather local quantities
        n = self.count.sum()
        e = np.zeros(n)
        COMM.Gatherv(elocal,[e,self.count,self.disp,MPI.DOUBLE],root=0)
        f = np.zeros(n)
        COMM.Gatherv(flocal,[f,self.count,self.disp,MPI.DOUBLE],root=0)
        if RANK==0:
           return blocking_analysis(f[1:],e[1:],0,False)
        else:
            return None,None
    def search(self,xs,params):
        self.amplitude_factory.update_scheme(0)
        Es = np.zeros(len(xs))
        for ix,x in enumerate(xs):
            COMM.Bcast(x,root=0)
            self.amplitude_factory.update(x) 
            t0 = time.time()
            E,err = self.sample_correlated()
            if RANK==0:
                Es[ix] = E
                print('\tcorrelated sample time=',time.time()-t0)
                print(f'\tix={ix},param={params[ix]},E={E},err={err}')
        if RANK==0:
            #return solve_quad(params,Es)
            return solve_min(params,Es)
        return None,None
    def get_rate(self):
        if self.extrapolator is not None:
            print(f'Extrapolator={self.extrapolate}, disable rate search.')
            return
        if not self.search_rate:
            return 
        if RANK==0:
            params = np.array([self.rate/2.,self.rate,self.rate*2.])
            xs = [self.x-rate * self.deltas for rate in params]
        else:
            params = None
            xs = [np.zeros_like(self.x)] * 3
        self.rate,_ = self.search(xs,params)
    def get_direction(self,cond,search_cond):
        if not search_cond:
            self.transform_gradients(cond)
            return 
        if RANK==0:
            params = np.array([cond/10.,cond,cond*10.]) 
            deltas = [self.transform_gradients(cond) for cond in params]
            xs = [self.x - self.rate * delta for delta in deltas]
        else:
            params = None
            xs = [np.zeros_like(self.x)] * 3
        cond,_ = self.search(xs,params)
        self.transform_gradients(cond)
def solve_min(x,y):
    idx = np.argmin(y)
    return x[idx],y[idx]
def solve_quad(x,y):
    if len(x)!=3:
        return solve_min(x,y)
    m = np.stack([np.square(x),x,np.ones(3)],axis=1)
    a,b,c = list(np.dot(np.linalg.inv(m),y))
    if a < 0. or a*b > 0.:
        x0,y0 = solve_min(x,y)
    else:
        x0,y0 = -b/(2.*a),-b**2/(4.*a)+c
    print(f'\ta={a},b={b},x0={x0},y0={y0}')
    return x0,y0
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
