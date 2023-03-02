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
        self.omega = None
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
            psi = self.amplitude_factory.vec2psi(self.x,inplace=True)

        # parse gradient optimizer
        self.optimizer = optimizer
        self.compute_hg = False
        if self.optimizer=='rgn':
            self.compute_hg = kwargs.get('compute_hg',True) 
            if not self.compute_hg:
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
        for step in range(start,stop):
            self.step = step
            self.sample()

            self.propagate_rate_cond()
            self.transform_gradients()

            if RANK==0:
                self.regularize()
                self.extrapolate()
                if self.conditioner is not None:
                    self.conditioner(self.x)
                print('\tx norm=',np.linalg.norm(self.x))
            COMM.Bcast(self.x,root=0) 
            psi = self.amplitude_factory.update(self.x)
            if RANK==0:
                if tmpdir is not None: # save psi to disc
                    write_ftn_to_disc(psi,tmpdir+f'psi{step+1}',provided_filename=True)
    def regularize(self):
        delta_norm = np.linalg.norm(self.deltas)
        print(f'\tdelta norm={delta_norm}')
        if self.step == self.start:
            self.delta_norm = delta_norm
            return 
        ratio = delta_norm / self.delta_norm
        cnt = 0
        while ratio > 2.:
            self.deltas /= 2.
            delta_norm /= 2.
            ratio /= 2. 
            cnt += 1
        self.delta_norm = delta_norm
        if cnt>0:
            print(f'\tregularized delta norm={delta_norm}')
            self.rate = self.rate_min
            self.cond = self.cond_min
    def propagate_rate_cond(self):
        if self.step < self.start + self.rate_itv:
            self.rate *= self.rate_base
        if self.step < self.start + self.cond_itv:
            self.cond *= self.cond_base
        if RANK==0:
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
        ex,vx = self.ham.compute_local_energy(config,self.amplitude_factory)
        self.elocal.append(ex)
        self.vlocal.append(vx)
    def sample(self,energy_only=False):
        self.sampler.amplitude_factory = self.amplitude_factory
        if self.exact_sampling:
            self.sample_exact(energy_only=energy_only)
        else:
            self.sample_stochastic(energy_only=energy_only)
        self.extract_energy_gradient()
    def sample_exact(self,energy_only=False): 
        self.sampler.compute_dense_prob() # runs only for dense sampler 

        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros

        self.flocal = []
        self.samples = []
        self.elocal = []
        self.vlocal = None 
        self.hg_local = None 
        if not energy_only:
            self.vlocal = []
            if self.compute_hg:
                self.hg_local = [] 
        t0 = time.time()
        for ix in ixs:
            self.omega = p[ix]
            self.flocal.append(self.omega)
            self.config = all_configs[ix]
            self.samples.append(self.config) 
            self.update_local(self.config)
        if RANK==SIZE-1:
            print('\texact sample time=',time.time()-t0)
        self.flocal = np.array(self.flocal)
    def sample_stochastic(self,energy_only=False): 
        # randomly select a process to be the control process
        self.cix = np.random.randint(low=0,high=SIZE,size=1)
        COMM.Bcast(self.cix,root=0)
        self.cix = self.cix[0]

        self.sampler.preprocess(self.config) 

        self.samples = []
        self.flocal = dict()
        self.elocal = []
        self.vlocal = None 
        self.hg_local = None
        if not energy_only:
            self.vlocal = []
            if self.compute_hg:
                self.hg_local = []

        self.terminate = np.array([0])
        self.rid = np.array([RANK])
        if RANK==self.cix:
            self._ctr()
        else:
            self._sample()
        self.flocal = np.array([self.flocal[config] for config in self.samples])
    def _ctr(self):
        print('\tcontrol rank=',RANK)
        t0 = time.time()
        self.config,self.omega = self.sampler.sample()
        self.flocal[self.config] = 1
        self.samples.append(self.config)
        self.update_local(self.config)

        ncurr = 1
        ntotal = self.batchsize * SIZE
        tdest = set(range(SIZE)).difference({RANK})
        while self.terminate[0]==0:
            COMM.Recv(self.rid)
            ncurr += 1
            if ncurr > ntotal: # send termination message to all workers
                self.terminate[0] = 1
                for worker in tdest:
                    COMM.Bsend(self.terminate,dest=worker)
            else:
                COMM.Bsend(self.terminate,dest=self.rid[0])
        print('\tstochastic sample time=',time.time()-t0)
    def _sample(self):
        while self.terminate[0]==0:
            self.config,self.omega = self.sampler.sample()
            if self.config in self.flocal:
                self.flocal[self.config] += 1
            else:
                self.flocal[self.config] = 1
                self.samples.append(self.config)
                self.update_local(self.config)

            COMM.Bsend(self.rid,dest=self.cix) 
            COMM.Recv(self.terminate,source=self.cix)
    def extract_energy_gradient(self,new_sample=True):
        t0 = time.time()
        if new_sample:
            nlocal = np.array([len(self.flocal)])
            n = np.zeros_like(nlocal)
            COMM.Allreduce(nlocal,n,op=MPI.SUM)
            if RANK==0:
                print('\tunique samples',n[0])
            
            nlocal = np.array([np.sum(self.flocal)])
            self.n = np.zeros_like(nlocal)
            COMM.Allreduce(nlocal,self.n,op=MPI.SUM)
            self.n = self.n[0]
            if RANK==0:
                print('\tnormalization=',self.n)

            # not sure the effect of the following
            omega_local = np.array([self.omega])
            omega_all = np.zeros(SIZE)
            COMM.Allgather(omega_local,omega_all)
            ix = np.argmax(omega_all)
            config = np.array(self.config)
            COMM.Bcast(config,root=ix)
            self.config = tuple(config)

        # mean
        self.elocal = np.array(self.elocal)
        elocal = self.elocal.reshape(len(self.elocal),1)
        self.E = compute_mean(elocal,self.flocal,self.n)[0]
        # var
        esq_local = elocal**2
        esq_mean = compute_mean(esq_local,self.flocal,self.n)[0]
        Evar = esq_mean-self.E**2
        self.Eerr = np.sqrt(Evar/self.n) 
        if RANK==0:
            print(f'step={self.step},E={self.E},Eerr={self.Eerr}')

        if self.vlocal is not None:
            # mean
            self.vlocal = np.array(self.vlocal)
            #print(RANK,self.vlocal)
            self.vmean = compute_mean(self.vlocal,self.flocal,self.n)
            # var
            vsq_local = self.vlocal**2
            vsq_mean = compute_mean(vsq_local,self.flocal,self.n)
            vvar = vsq_mean - self.vmean**2 

            # mean
            ef_local = self.elocal*self.flocal
            ev_mean = compute_mean(self.vlocal,ef_local,self.n)
            self.g = ev_mean - self.E*self.vmean
            # var
            evsq_mean = compute_mean(vsq_local,esq_local[:,0]*self.flocal,self.n)
            ev_var = evsq_mean - ev_mean**2

            # g = mean(ev) - E * mean(v) # treat E as constant
            # var(g) = var(ev) + var(v)*E**2 - 2*E*cov(ev,v)
            # cov(ev,v) = mean(ev*v) - mean(ev)*mean(v)
            evv_mean = compute_mean(vsq_local,ef_local,self.n)
            gvar = ev_var + self.E**2 * vvar - 2*self.E*(evv_mean-ev_mean*self.vmean)
            self.gerr = np.sqrt(gvar/self.n)
        if self.hg_local is not None:
            self.hg_local = np.array(self.hg_local)
        if RANK==0:
            print('\textract local time=',time.time()-t0)
            print('\tgradient max=',np.amax(np.fabs(self.g)))
            print('\tgradient err max=',np.amax(self.gerr))
    def getS(self):
        COMM.Bcast(self.vmean,root=0)
        def S(x):
            Sx1 = compute_mean(self.vlocal,self.flocal*np.dot(self.vlocal,x),self.n) 
            Sx2 = self.vmean * np.dot(self.vmean,x) 
            return Sx1-Sx2
        return S 
    def getH(self):
        self.ncalls = 0
        hg_mean = compute_mean(self.hg_local,self.flocal,self.n)
        COMM.Bcast(self.vmean,root=0)
        COMM.Bcast(self.hg_mean,root=0)
        def H(x):
            Hx1 = compute_mean(self.vlocal,self.flocal*np.dot(self.hg_local,x),self.n)
            Hx2 = self.vmean * np.dot(hg_mean,x)
            Hx3 = self.g * np.dot(self.vmean,x)
            return Hx1-Hx2-Hx3
        return H
    def sample_correlated(energy_only=False):
        self.vlocal = None if energy_only else []
        self.elocal = []
        t0 = time.time()
        for config in self.samples:
            self.update_local(config)
        tix = SIZE-1
        if tix==self.cix:
            tix -= 1
        if RANK==tix:
            print('\tcorrelated sample time=',time.time()-t0) 
        self.extract_energy_gradient(new_sample=False) 
    def getH_num(self):
        self.ncalls = 0
        g0 = self.g.copy()
        def H(x):
            self.amplitude_factory.vec2psi(self.x - x * self.num_step,inplace=True)
            if self.exact_sampling:
                self.sampe_exact()
            else:
                self.sample_correlated()
            return (self.g-g0)/self.num_step 
        return H
    def transform_gradients(self):
        self.deltas = np.zeros_like(self.g)
        if self.optimizer=='sr':
            self._transform_gradients_sr()
        elif self.optimizer=='rgn':
            self._transform_gradients_rgn()
        elif self.optimizer=='lin':
            raise NotImplementedError
            self._transform_gradients_lin()
        else:
            self._transform_gradients_sgd()
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
    def parallel_inv(self,A,b,x0=None,maxiter=None,herm=True):
        terminate = np.array([0])
        buf = np.zeros_like(b)
        sh = len(b)
        self.ncalls = 0
        def _A(x): # wrapper for synced parallel process
            COMM.Bcast(terminate,root=0)
            if terminate[0]==1:
                return x

            self.ncalls += 1
            COMM.Bcast(x,root=0)
            return A(x)
        LinOp = spla.LinearOperator((sh,sh),matvec=_A,dtype=float)
        if RANK==0:
            if herm:
                show = False if maxiter is None else True
                x,info = spla.minres(LinOp,b,x0=x0,maxiter=maxiter,tol=CG_TOL,show=show)
            else:
                x,info = spla.lgmres(LinOp,b,x0=x0,tol=CG_TOL)
            terminate[0] = 1
            COMM.Bcast(terminate,root=0)
            return x,info
        else:
            while terminate[0]==0:
                _A(buf) 
            return buf,1
    def _transform_gradients_sr(self):
        t0 = time.time()
        g = self.g
        S = self.getS()
        def R(vec):
            return S(vec) + self.cond * vec
        self.deltas,info = self.parallel_inv(R,g)
        if RANK==0:
            print('\tSR solver time=',time.time()-t0)
            print('\tSR solver exit status=',info)
    def _transform_gradients_rgn(self):
        if self.compute_hg:
            self._transform_gradients_rgn_WL()
        else:
            self._transform_gradients_rgn_num()
    def _transform_gradients_rgn_WL(self):
        t0 = time.time()
        E = self.E
        g = self.g
        S = self.getS()
        H = self.getH()
        def A(vec):
            return H(vec) - E * S(vec) + self.cond * vec 
        self.deltas,info = self.parallel_inv(A,g,herm=False)
        if RANK==0:
            print('\tRGN solver time=',time.time()-t0)
            print('\tRGN solver exit status=',info)
    def _transform_gradients_rgn_num(self,rate,cond):
        t0 = time.time()
        g = self.g
        S = self.getS()
        H = self.getH_num()
        sh = len(g)
        def R(vec):
            return S(vec) + self.cond * vec
        x0,info = self.parallel_inv(R,g)
        if RANK==0:
            print('\tSR solver time=',time.time()-t0)
            print('\tSR solver exit status=',info)

        def A(vec):
            return H(vec) + self.cond * vec 
        self.deltas,info = self.parallel_inv(A,g,x0=x0,maxiter=10)
        if RANK==0:
            print('\tRGN solver time=',time.time()-t0)
            print('\tRGN solver exit status=',info)
#    def _transform_gradients_lin(self,rate,cond):
#        E = self.E
#        g = self.g
#        S = self.S
#        H = self.H
#        sh = len(g)
#        cond,rate = self._get_cond_rate(x)
#        def _S(vec):
#            x0,x1 = vec[0],vec[1:]
#            return np.concatenate([np.ones(1)*x0,S(x1)]) 
#        if self.sr_cond:
#            def R(x1):
#                return S(x1) + cond * x1
#        else:
#            def R(x1):
#                return cond * x1
#        def _H(vec):
#            x0,x1 = vec[0],vec[1:]
#            Hx0 = np.dot(g,x1)
#            Hx1 = x0 * g + H(x1) - E*S(x1) + R(x1)/rate
#            return np.concatenate([np.ones(1)*Hx0,Hx1])
#        A = spla.LinearOperator((sh+1,sh+1),matvec=_H)
#        M = spla.LinearOperator((sh+1,sh+1),matvec=_S)
#        dE,deltas = spla.eigs(A,k=1,M=M,sigma=0.)
#        deltas = deltas[:,0].real
#        deltas = deltas[1:] / deltas[0]
#        return - deltas / (1. - np.dot(self._glog,deltas)) 
#    def transform_gradients_search_rate(self):
#        raise NotImplementedError
#        # save the current quantities
#        E = self.E
#        elocal = self.elocal.copy()
#        vlocal = self.vlocal.copy()
#        fv_local = self.fv_local.copy()
#        vmean = self.vmean.copy()
#        g = self.g.copy()
#
#        self.transform_gradients(rate=1.)
#        COMM.Bcast(self.deltas,root=0) 
#        pk = self.deltas
#        if self.search_rate=='wolfe':
#            xk = np.concatenate(psi2vecs(self.constructors,self.psi))
#            gfk = g     
#            old_fval = E
#            self.saved_f[self.step] = E
#            old_old_fval = self.saved_f.get(self.step-1,None) 
#            self.maxiter = 3
#            alpha_star,phi_star = self._search_rate_wolfe(xk,pk,gfk,old_fval,old_old_fval) 
#        else:    
#            alpha_star,phi_star = self._search_rate_quad(pk)
#        self.deltas *= alpha_star
#        # reset current quantities
#        self.E = E
#        self.elocal = elocal
#        self.vlocal = vlocal
#        self.fv_local = fv_local
#        self.vmean = vmean
#        self.g = g
#    def _search_rate_quad(self,pk):
#        E0,E1 = np.zeros(3),np.zeros(3)
#        alpha = np.array([self.rate/2.,self.rate,self.rate*2.]) 
#        for ix,alpha_i in enumerate(alpha):
#            psi = self.vec2psi(self.psi-alpha_i*pk)
#            self.correlated_sampling(psi,energy_only=True)
#            E0[ix] = self.E
#            E1[ix] = self.err 
#        return _solve_quad(alpha,E0)
#    def _search_rate_wolfe(self,xk,pk,gfk,old_fval,old_old_fval):
#        terminate = np.array([0])
#        buf = np.zeros_like(xk)
#        def f(x):
#            COMM.Bcast(terminate,root=0)
#            if terminate[0]==1:
#                return 0.
#
#            COMM.Bcast(x,root=0)
#            if np.linalg.norm(x-self.x)/np.linalg.norm(self.x)<PRECISION:
#                return self.E
#            psi = vec2psi(self.constructors,x,psi=self.psi.copy())
#            self.correlated_sampling(psi) 
#            return self.E
#        def fprime(x):
#            COMM.Bcast(terminate,root=0)
#            if terminate[0]==1:
#                return buf
#
#            COMM.Bcast(x,root=0)
#            if np.linalg.norm(x-self.x)/np.linalg.norm(self.x)<PRECISION:
#                return self.g
#            psi = vec2psi(self.constructors,x,psi=self.psi.copy())
#            self.correlated_sampling(psi) 
#            return self.g
#        if RANK==0:
#            alpha_star,fc,gc,phi_star,old_fval,derphi_star = line_search(f,fprime,xk,pk,
#                gfk=gfk,old_fval=old_fval,old_old_fval=old_old_fval,maxiter=self.maxiter)
#            terminate[0] = 1
#            COMM.Bcast(terminate,root=0)
#            return alpha_star,phi_star
#        else:
#            while True:
#                if terminate[0]==1:
#                    return 0.,0.
#                f(buf)
#                if terminate[0]==1:
#                    return 0.,0.
#                fprime(buf)
#class LBFGS(TNVMC):
#    def __init__(self,psi,ham,sampler_opts,contract_opts,optimizer_opts,conditioner=None):
#        optimizer_opts['method'] = 'bfgs'
#        super().__int__(psi,ham,sampler_opts,contract_opts,optimizer_opts,conditioner=conditioner)
#        from scipy.optimize import lbfgsb
#        self.m = optimizer_opts.get('m',10)
#        self.hessinvp = lbfgsb.LbfgsInvHessProduct
#        self.s = None
#        self.y = None
#        self.xprev = None
#        self.gprev = None
#    def getS(self):
#        raise NotImplementedError
#    def getH(self):
#        raise NotImplementedError
#    def transform_gradients(self,x=None):
#        if self.step == 0:
#            return self.transform_gradients_sgd(x=x)
#        x = self.rate if x is None else x
#        hessinvp = self.hessinvp(self.s,self.y)
#        z = hessinvp._matvec(self.g)
#        return x * z
#    #def _sample_lbfgs(self):
#    #    if self.step>0:
#    #        prev_samples = self.samples.copy()
#    #        prev_f = self.f.copy()
#    #        prev_x = self.x.copy()
#    #        prev_g = self.g.copy()
#    #    self.x = np.concatenate(psi2vecs(self.constructors,self.psi))
#    #    self._sample()
#    #    if self.step==0:
#    #        return
#    #    if self.sampler.exact:
#    #        g_corr = self.g
#    #    else:
#    #        opt = get_optimizer(self.optimizer_opts)
#    #        opt.parse_samples(prev_samples,prev_f)
#
#    #        t0 = time.time()
#    #        self.amplitude_factory,opt = update_grads(self.amplitude_factory,opt)
#    #        if RANK==SIZE-1:
#    #            print(f'\tcorrelated gradient time={time.time()-t0}')
#
#    #        t0 = time.time()
#    #        opt = compute_elocs(self.ham,self.amplitude_factory,opt)
#    #        if RANK==SIZE-1:
#    #            print('\tcorrelated eloc time=',time.time()-t0)
#
#    #        g_corr = self.extract_energy_gradient(opt)[-1]
#    #    if RANK>0:
#    #        return
#    #    # RANK==0 only
#    #    size = len(self.x)
#    #    yprev = (self.x - prev_x).reshape(1,size)
#    #    sprev = (g_corr - prev_g).reshape(1,size)
#    #    if self.y is None:
#    #        self.y = yprev 
#    #        self.s = sprev
#    #    else:
#    #        self.y = np.concatenate([self.y,yprev],axis=0) 
#    #        self.s = np.concatenate([self.s,sprev],axis=0)

#def _update_psi(psi,deltas,constructors):
#    deltas = vec2psi(constructors,deltas)
#    for ix,(_,_,_,site) in enumerate(constructors):
#        tsr = psi[psi.site_tag(*site)]
#        data = tsr.data - deltas[site] 
#        tsr.modify(data=data)
#    return psi
#def _solve_quad(x,y):
#    m = np.stack([np.square(x),x,np.ones(3)],axis=1)
#    a,b,c = list(np.dot(np.linalg.inv(m),y))
#    if a < 0. or a*b > 0.:
#        ix = np.argmin(y)
#        x0,y0 = x[ix],y[ix]
#    else:
#        x0,y0 = -b/(2.*a),-b**2/(4.*a)+c
#    if RANK==0:
#        print(f'x={x},y={y},x0={x0},y0={y0}')
#    return x0,y0
#def _solve_quad(x,y):
#    if RANK==0:
#        print(f'x={x},y={y}')
#    ix = np.argmin(y)
#    return x[ix],y[ix]
def compute_mean(x,f,n):
    xsum = np.dot(f,x)
    xmean = np.zeros_like(xsum)
    #print(RANK,xsum.shape,xmean.shape)
    COMM.Reduce(xsum,xmean,op=MPI.SUM,root=0)
    return xmean/n 
