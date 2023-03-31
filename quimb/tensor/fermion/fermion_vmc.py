import time,scipy,functools,h5py
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
        **kwargs,
    ):
        # parse ham
        self.ham = ham

        # parse sampler
        self.sampler = sampler
        self.exact_sampling = sampler.exact

        # parse wfn 
        self.amplitude_factory = amplitude_factory         
        self.x = self.amplitude_factory.get_x()
        self.nparam = len(self.x)

        # parse gradient optimizer
        self.optimizer = optimizer
        self.solve = kwargs.get('solve','iterative')
        self.compute_Hv = False
        if self.optimizer in ['rgn','lin']:
            self.ham.initialize_pepo(self.amplitude_factory.psi)
            self.compute_Hv = True
        if self.optimizer=='lin':
            self.xi = kwargs.get('xi',0.5)
            # only used for iterative full Hessian
            self.solver = kwargs.get('solver','davidson')
            if self.solver=='davidson':
                maxsize = kwargs.get('maxsize',25)
                maxiter = kwargs.get('maxiter',100)
                restart_size = kwargs.get('restart_size',5)
                from .davidson import davidson
                self.davidson = functools.partial(davidson,
                    maxsize=maxsize,restart_size=restart_size,maxiter=maxiter,tol=CG_TOL) 
        if self.solve=='mask':
            self.block_dict = self.amplitude_factory.get_block_dict()

        # to be set before run
        self.config = None
        self.batchsize = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR
        self.rate2 = None # rate for LIN,RGN
        self.cond1 = None
        self.cond2 = None
        self.check_update = None 
        self.accept_ratio = None
    def run(self,start,stop,tmpdir=None):
        for step in range(start,stop):
            self.step = step
            self.sample()
            self.extract_energy_gradient()
            self.transform_gradients()
            COMM.Bcast(self.x,root=0) 
            psi = self.amplitude_factory.update(self.x)
            if RANK==0:
                if tmpdir is not None: # save psi to disc
                    write_ftn_to_disc(psi,tmpdir+f'psi{step+1}',provided_filename=True)
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
        while self.terminate[0]==0:
            COMM.Recv(self.rank,tag=0)
            ncurr += 1
            if ncurr > self.batchsize: # send termination message to all workers
                self.terminate[0] = 1
                for worker in range(1,SIZE):
                    COMM.Send(self.terminate,dest=worker,tag=1)
            else:
                COMM.Send(self.terminate,dest=self.rank[0],tag=1)
    def _sample(self):
        self.sampler.preprocess(self.config) 

        self.samples = []
        self.elocal = []
        self.vlocal = []
        if self.compute_Hv:
            self.Hv_local = [] 

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
                                                             compute_Hv=self.compute_Hv)
                if np.fabs(ex) > DISCARD:
                    self.store[config] = None
                    continue
                self.store[config] = ex,vx,Hvx
                self.p0[config] = cx**2
            self.samples.append(config)
            self.elocal.append(ex)
            self.vlocal.append(vx)
            if self.compute_Hv:
                self.Hv_local.append(Hvx)

            COMM.Send(self.rank,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
        if RANK==SIZE-1:
            print('\tstochastic sample time=',time.time()-t0)
    def sample_exact(self,compute_v=True,compute_Hv=None): 
        # can be reused for correlated sampling
        self.sampler.compute_dense_prob() # runs only for dense sampler 

        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros

        self.samples = []
        self.flocal = []
        self.elocal = []
        self.vlocal = []
        compute_Hv = self.compute_Hv if compute_Hv is None else compute_Hv
        if compute_Hv:
            self.Hv_local = [] 

        t0 = time.time()
        self.store = dict()
        if RANK==SIZE-1:
            print('\tnsamples=',len(ixs))
        for ix in ixs:
            self.flocal.append(p[ix])
            config = all_configs[ix]
            self.samples.append(config) 
            _,ex,vx,Hvx = self.ham.compute_local_energy(config,self.amplitude_factory,
                                                        compute_v=compute_v,compute_Hv=compute_Hv)
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
    def gather_sizes(self):
        self.count = np.array([0]*SIZE)
        COMM.Allgather(np.array([self.nlocal]),self.count)
        self.disp = np.concatenate([np.array([0]),np.cumsum(self.count[:-1])])
    def extract_energy(self):
        if self.exact_sampling:
            self._extract_energy_exact()
        else:
            self._extract_energy_stochastic()
    def _extract_energy_stochastic(self,weighted=False):
        # collect all energies for blocking analysis 
        if RANK==0:
            self.nlocal = 1
            self.elocal = np.zeros(1)
            if weighted:
                self.flocal = np.zeros(1)
        else:
            self.nlocal = len(self.elocal)
            self.elocal = np.array(self.elocal)
            if weighted:
                self.flocal = np.array(self.flocal)
        self.gather_sizes()
        self.n = self.count.sum()
        e = np.zeros(self.n)
        COMM.Gatherv(self.elocal,[e,self.count,self.disp,MPI.DOUBLE],root=0)
        e = e[1:]
        if weighted:
            f = np.zeros(self.n)
            COMM.Gatherv(self.flocal,[f,self.count,self.disp,MPI.DOUBLE],root=0)
            f = f[1:] 
        else:
            f = np.ones_like(e) 
        if RANK>0:
            return
        self.n -= 1
        self.E,self.Eerr = blocking_analysis(f,e,0,True)
    def _extract_energy_exact(self):
        # reduce scalar energy
        self.nlocal = len(self.elocal)
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
                vsum_ = np.zeros_like(self.x)
                vesum_ = np.zeros_like(self.x)
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
    def extract_S(self,solve=None):
        solve = self.solve if solve is None else solve
        if solve=='mask':
            self._extract_S_mask()
        elif solve=='matrix':
            self.S = self._get_Smatrix()
        elif solve=='iterative':
            self._extract_S_iterative()
        else:
            raise NotImplementedError
    def _get_Smatrix_stochastic(self,start=0,stop=None):
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
    def _get_Smatrix_exact(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        v = self.vlocal[:,start:stop] 
        vvsum_ = np.einsum('s,si,sj->ij',self.flocal,v,v)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean = self.vmean[start:stop]
            S = vvsum - np.outer(vmean,vmean)
        return S
    def _get_Smatrix(self,start=0,stop=None):
        if self.exact_sampling:
            return self._get_Smatrix_exact(start=start,stop=stop)
        else:
            return self._get_Smatrix_stochastic(start=start,stop=stop)
    def _extract_S_mask(self):
        # stochastic only, collect matrix blocks
        ls = [None] * len(self.block_dict)
        for ix,(start,stop) in enumerate(self.block_dict):
            ls[ix] = self._get_Smatrix(start=start,stop=stop)
        self.S = ls
    def _extract_S_iterative(self):
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
        if self.solve=='mask':
            self._extract_H_mask()
        elif self.solve=='matrix':
            self.H = self._get_Hmatrix()
        elif self.solve=='iterative':
            self._extract_H_iterative()
        else:
            raise NotImplementedError
    def _extract_Hvmean(self):
        if self.exact_sampling:
            self.Hv_local = np.array(self.Hv_local)
            Hvsum_ = np.dot(self.flocal,self.Hv_local)
        else:
            if RANK==0:
                Hvsum_ = np.zeros_like(self.x)
            else:
                self.Hv_local = np.array(self.Hv_local)
                Hvsum_ = self.Hv_local.sum(axis=0)
        self.Hvmean = np.zeros_like(Hvsum_)
        COMM.Reduce(Hvsum_,self.Hvmean,op=MPI.SUM,root=0)
        if RANK==0:
            self.Hvmean /= self.n
    def _get_Hmatrix_stochastic(self,start=0,stop=None):
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
    def _get_Hmatrix_exact(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        v = self.vlocal[:,start:stop] 
        Hv = self.Hv_local[:,start:stop] 
        vHvsum_ = np.einsum('s,si,sj->ij',self.flocal,v,Hv)
        vHvsum = np.zeros_like(vHvsum_)
        COMM.Reduce(vHvsum_,vHvsum,op=MPI.SUM,root=0)
        H = None
        if RANK==0:
            Hvmean = self.Hvmean[start:stop]
            vmean = self.vmean[start:stop]
            g = self.g[start:stop]
            H = vHvsum / self.n - np.outer(vmean,Hvmean) - np.outer(g,vmean)
        return H
    def _get_Hmatrix(self,start=0,stop=None):
        if self.exact_sampling:
            return self._get_Hmatrix_exact(start=start,stop=stop)
        else:
            return self._get_Hmatrix_stochastic(start=start,stop=stop)
    def _extract_H_mask(self):
        ls = [None] * len(self.block_dict)
        for ix,(start,stop) in enumerate(self.block_dict):
            ls[ix] = self._get_Hmatrix(start=start,stop=stop)
        self.H = ls
    def _extract_H_iterative(self):
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
    def transform_gradients(self):
        if self.optimizer=='sr':
            self._transform_gradients_sr()
        elif self.optimizer in ['rgn','lin']:
            self._transform_gradients_o2()
        else:
            self._transform_gradients_sgd()
        if RANK==0:
            print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(self.deltas)},dot={np.dot(self.deltas,self.g)},x={np.linalg.norm(self.x)}')
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
        self.x -= self.rate1 * self.deltas
    def _transform_gradients_sr(self,solve=None):
        if RANK>0:
            return 
        solve = self.solve if solve is None else solve
        t0 = time.time()
        if solve=='mask':
            self._transform_gradients_sr_mask()
        elif solve=='matrix':
            self._transform_gradients_sr_matrix()
        elif solve=='iterative':
            self._transform_gradients_sr_iterative()
        else:
            raise NotImplementedError
        print('\tSR solver time=',time.time()-t0)
        self.x -= self.rate1 * self.deltas
    def _transform_gradients_sr_mask(self):
        self.deltas = np.zeros_like(self.x)
        for ix,(start,stop) in enumerate(self.block_dict):
            S = self.S[ix] + self.cond1 * np.eye(stop-start)
            self.deltas[start:stop] = np.linalg.solve(S,self.g[start:stop])
        if self.tmpdir is not None:
            S = np.zeros((self.nparam,self.nparam))
            for ix,(start,stop) in enumerate(self.block_dict):
                S[start:stop,start:stop] = self.S[ix]
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('S',data=S) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
    def _transform_gradients_sr_matrix(self):
        S = self.S + self.cond1 * np.eye(self.nparam)
        self.deltas = np.linalg.solve(S,self.g)
        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('S',data=self.S) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
    def _transform_gradients_sr_iterative(self):
        def A(x):
            return self.S(x) + self.cond1 * x
        LinOp = spla.LinearOperator((self.nparam,self.nparam),matvec=A,dtype=self.g.dtype)
        self.deltas,info = spla.minres(LinOp,self.g,tol=CG_TOL)
        print('\tSR solver exit status=',info)
        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
    def _transform_gradients_o2(self):
        xnew_rgn = np.zeros_like(self.x)
        if RANK==0:
            if self.optimizer=='rgn':
                dEm = self._transform_gradients_rgn()
            elif self.optimizer=='lin':
                dEm = self._transform_gradients_lin()
            else:
                raise NotImplementedError
            xnew_rgn = self.x - self.rate2 * self.deltas    
        if not self.check_update:
            self.x = self.xnew_rgn
            return
        # SR
        self.extract_S(solve='iterative')
        self._transform_gradients_sr(solve='iterative')
        xnew_sr = self.x
        
        COMM.Bcast(xnew_rgn,root=0) 
        self.current_energy()
        self.correlated_energy(xnew_rgn) 
        if RANK>0:
            return
        dE = self.Enew - self.E0
        rho = dE / dEm
        print(f'\tpredict={dEm},actual={dE},ratio={rho}')
        if rho > self.accept_ratio:
            self.x = xnew_rgn
        else:
            self.x = xnew_sr
    def _transform_gradients_rgn(self):
        t0 = time.time()
        if self.solve=='mask':
            dE = self._transform_gradients_rgn_mask()
        elif self.solve=='matrix':
            dE = self._transform_gradients_rgn_matrix()
        elif self.solve=='iterative':
            dE = self._transform_gradients_rgn_iterative()
        else:
            raise NotImplementedError
        print('\tRGN solver time=',time.time()-t0)
        return dE
    def _transform_gradients_rgn_mask(self):
        self.deltas = np.zeros_like(self.x)
        w = [None] * len(self.block_dict) 
        dE = np.zeros(len(self.block_dict))  
        for ix,(start,stop) in enumerate(self.block_dict):
            w[ix],self.deltas[start:stop],dE[ix] = \
                _rgn_block_solve(self.H[ix],self.E,self.S[ix],self.g[start:stop],self.cond2)
        print('\tleast eigenvalue=',w)
        if self.tmpdir is not None:
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
    def _transform_gradients_rgn_matrix(self):
        w,self.deltas,dE = _rgn_block_solve(self.H,self.E,self.S,self.g,self.cond2) 
        print('\tleast eigenvalue=',wmin)
        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('H',data=self.H) 
            f.create_dataset('S',data=self.S) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
        return dE
    def _transform_gradients_rgn_iterative(self):
        def hess(x):
            return self.H(x) - self.E * self.S(x)
        def A(x):
            return hess(x) + self.cond2 * x 
        LinOp = spla.LinearOperator((self.nparam,self.nparam),matvec=A,dtype=self.g.dtype)
        self.deltas,info = spla.lgmres(LinOp,self.g,tol=CG_TOL)
        print('\tRGN solver exit status=',info)
        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'step{self.step}','w')
            f.create_dataset('g',data=self.g) 
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
        return - np.dot(self.g,self.deltas) + np.dot(self.deltas,hess(self.deltas))
    def _transform_gradients_lin(self,cond):
        t0 = time.time()
        if self.solve=='mask':
            self._transform_gradients_lin_mask(cond)
        elif self.solve=='matrix':
            self._transform_gradients_lin_matrix(cond)
        elif self.solve=='iterative':
            self._transform_gradients_lin_iterative(cond) 
        else:
            raise NotImplementedError
        self._scale_eigenvector()
        print('\tEIG solver time=',time.time()-t0)
    def _scale_eigenvector(self):
        if self.xi is None:
            Ns = self.vmean
        else:
            if self.solve=='matrix':
                Sp = np.dot(self.S,self.deltas)
            elif self.solve=='mask':
                Sp = np.zeros_like(self.x)
                for ix,(start,stop) in enumerate(self.block_dict):
                    Sp[start:stop] = np.dot(self.S[ix],self.deltas[start:stop])
            elif self.solve=='iterative':
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
        scale = len(self.block_dict)
        for ix,(start,stop) in enumerate(self.block_dict):
            sh = stop - start
            A = np.block([[np.ones((1,1))*self.E,H0j[start:stop].reshape(1,sh)*scale],
            #A = np.block([[np.ones((1,1))*self.E,H0j[start:stop].reshape(1,sh)],
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

        H = np.zeros((self.nparam,self.nparam))
        S = np.zeros((self.nparam,self.nparam))
        for ix,(start,stop) in enumerate(self.block_dict):
            H[start:stop,start:stop] = self.H[ix] 
            S[start:stop,start:stop] = self.S[ix]
        #A = np.block([[np.ones((1,1))*self.E,H0j.reshape(1,self.nparam)],
        #              [Hi0.reshape(self.nparam,1),H]])
        #B = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
        #              [np.zeros((self.nparam,1)),S+cond*np.eye(self.nparam)]])
        #w,v = scipy.linalg.eig(A,b=B) 
        #w,self.deltas,idx = _select_eigenvector(w.real,v.real)
        #print('\timaginary norm=',np.linalg.norm(v[:,idx].imag))
        #print('\teigenvalue =',w)
        #print('\tscale1=',v[0,idx].real)

        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'msk_step{self.step}','w')
            f.create_dataset('H',data=H) 
            f.create_dataset('S',data=S) 
            f.create_dataset('Hi0',data=Hi0) 
            f.create_dataset('H0j',data=H0j) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.close()
 
    def _transform_gradients_lin_matrix(self,cond):
        Hi0 = self.g
        H0j = self.Hvmean - self.E * self.vmean
        A = np.block([[np.ones((1,1))*self.E,H0j.reshape(1,self.nparam)],
                      [Hi0.reshape(self.nparam,1),self.H]])
        B = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
                      [np.zeros((self.nparam,1)),self.S+cond*np.eye(self.nparam)]])
        w,v = scipy.linalg.eig(A,b=B) 
        w,self.deltas,idx = _select_eigenvector(w.real,v.real)
        print('\timaginary norm=',np.linalg.norm(v[:,idx].imag))
        print('\teigenvalue =',w)
        print('\tscale1=',v[0,idx].real)

        if self.tmpdir is not None:
            f = h5py.File(self.tmpdir+f'full_step{self.step}','w')
            f.create_dataset('H',data=self.H) 
            f.create_dataset('S',data=self.S) 
            f.create_dataset('Hi0',data=Hi0) 
            f.create_dataset('H0j',data=H0j) 
            f.create_dataset('E',data=np.array([self.E])) 
            f.close()
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
            if RANK==0:
                self.E0,self.Eerr0 = self.E,self.Eerr
    def correlated_energy(self,xnew):
        # samples new energy
        self.amplitude_factory.update(xnew)
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
            if RANK==0:
                self.Enew,self.Eerr_new = self.E,self.Eerr 
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
    dE = - np.dot(deltas,g) + np.dot(deltas,np.dot(hess,deltas))
    return wmin,deltas,dE
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

    if printQ:
        if plateauError is not None:
            print(f'Stocahstic error estimate: {plateauError:.6e}\n')

    return meanEnergy, plateauError
