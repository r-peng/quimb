import time,h5py,itertools,pickle
import numpy as np
import scipy.sparse.linalg as spla

from quimb.utils import progbar as Progbar
from .utils import (
    load_ftn_from_disc,write_ftn_to_disc,
    vec2psi,psi2vecs,
)
from .fermion_2d_vmc import (
    SYMMETRY,config_map,
    AmplitudeFactory2D,ExchangeSampler2D,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
################################################################################
# Sampler  
################################################################################    
GEOMETRY = '2D' 
class DenseSampler:
    def __init__(self,sampler_opts):
        if GEOMETRY=='2D':
            self.nsite = sampler_opts['Lx'] * sampler_opts['Ly']
        else:
            raise NotImplementedError
        self.nelec = sampler_opts['nelec']
        self.all_configs = self.get_all_configs()
        self.ntotal = len(self.all_configs)
        self.flat_indexes = list(range(self.ntotal))

        batchsize,remain = self.ntotal//SIZE,self.ntotal%SIZE
        self.count = np.array([batchsize]*SIZE)
        if remain > 0:
            self.count[-remain:] += 1
        self.disp = [0]
        for batchsize in self.count[:-1]:
            self.disp.append(self.disp[-1]+batchsize)
        self.start = self.disp[RANK]
        self.stop = self.start + self.count[RANK]

        seed = sampler_opts.get('seed',None)
        self.rng = np.random.default_rng(seed)
        self.burn_in = 0
    def _set_prob(self,p):
        self.p = p 
    def get_all_configs(self):
        if SYMMETRY=='u1':
            return self.get_all_configs_u1()
        elif SYMMETRY=='u11':
            return self.get_all_configs_u11()
        else:
            raise NotImplementedError
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
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        config = self.all_configs[flat_idx]
        omega = self.p[flat_idx]
        return config,omega
    def sample_exact(self,ix):
        config = self.all_configs[ix]
        omega = self.p[ix]
        return config,omega
################################################################################
# VMC engine  
################################################################################    
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
        psi,
        ham,
        sampler_opts,
        contract_opts,
        optimizer_opts,
        #conditioner='auto',
        conditioner=None,
    ):
        # parse wfn 
        self.psi = psi.reorder('row',inplace=False)

        if conditioner == 'auto':
            def conditioner(psi):
                psi.equalize_norms_(1.0)
            self.conditioner = conditioner
        else:
            self.conditioner = None

        if self.conditioner is not None:
            # want initial arrays to be in conditioned form so that gradients
            # are approximately consistent across runs (e.g. for momentum)
            self.conditioner(self.psi)

        if GEOMETRY=='2D':
            self.amplitude_factory = AmplitudeFactory2D(self.psi,**contract_opts) 
        else:
            raise NotImplementedError
        self.constructors = self.amplitude_factory.constructors 

        # parse ham
        self.ham = ham

        # parse sampler
        self.config = None
        self.batchsize = None
        self.dense = sampler_opts.get('dense',False)
        if self.dense:
            self.sampler = DenseSampler(sampler_opts)
        else:
            if GEOMETRY=='2D':
                self.sampler = ExchangeSampler2D(sampler_opts)
            else:
                raise NotImplementedError

        # parse gradient optimizer
        self.method = optimizer_opts['method']
        self.num_step = optimizer_opts.get('num_step',DEFAULT_NUM_STEP)
        self.sr_cond = optimizer_opts.get('sr_cond',True)

        self.compute_hg = False
        if self.method in ['rgn','lin']:
            self.compute_hg = optimizer_opts.get('compute_hg',False) 
            self.ncalls = 0

        self.search_rate = False
        if self.method not in ['adam','rgn','lin']:
            self.search_rate = optimizer_opts.get('search_rate',False)
        if self.search_rate:
            self.search_rate_method = optimizer.get('search_rate_method','quad')
        self.search_cond = False

        if self.method=='adam':
            self.beta1 = optimizer_opts.get('beta1',.9)
            self.beta2 = optimizer_opts.get('beta2',.999)
            self.eps = optimizer_opts.get('eps',1e-8)
            self._ms = None
            self._vs = None
    def get_rate_cond(self):
        if self.step < self.rate_itv:
            self.rate *= self.rate_base
        if self.step < self.cond_itv:
            self.cond *= self.cond_base
        return self.rate,self.cond
    def run(self,steps,tmpdir=None,
            rate_min=DEFAULT_RATE_MIN,
            rate_max=DEFAULT_RATE_MAX,
            cond_min=DEFAULT_COND_MIN,
            cond_max=DEFAULT_COND_MAX,
            rate_itv=None,
            cond_itv=None):
        # change rate & conditioner as in Webber & Lindsey
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
        for step in range(steps):
            self.step = step
            self.sample()

            rate,cond = self.get_rate_cond()
            self.transform_gradients(rate=rate,cond=cond)
            #COMM.Barrier()
            COMM.Bcast(self.deltas,root=0) 

            self.psi = _update_psi(self.psi,self.deltas,self.constructors)
            if self.conditioner is not None:
                self.conditioner(self.psi)
            if RANK==0:
                if tmpdir is not None: # save psi to disc
                    write_ftn_to_disc(self.psi,tmpdir+f'psi{step+1}',provided_filename=True)
            self.amplitude_factory._set_psi(self.psi)
            self.constructors = self.amplitude_factory.constructors
    def compute_dense_amplitude(self):
        t0 = time.time()
        p_total = np.zeros(self.sampler.ntotal)
        start,stop = self.sampler.start,self.sampler.stop
        configs = self.sampler.all_configs[start:stop]

        p_local = [] 
        for config in configs:
            amp = self.amplitude_factory.amplitude((config,0))
            p_local.append(amp**2)
        p_local = np.array(p_local)
        nz_local = np.array([len(np.nonzero(p_local)[0])])
        nz = np.zeros(1,dtype=int)

        # sync before gathering
        #COMM.Barrier()
         
        COMM.Allgatherv(p_local,[p_total,self.sampler.count,self.sampler.disp,MPI.DOUBLE])
        p_total /= np.sum(p_total)
        self.sampler._set_prob(p_total)

        COMM.Allreduce(nz_local,nz,op=MPI.MAX)
        self.progbar = (nz_local[0]==nz[0])
        if RANK==0:
            print(f'\tdense amplitude time=',time.time()-t0)
    def burn_in(self,batchsize=None):
        batchsize = self.sampler.burn_in if batchsize is None else batchsize
        self.sampler._set_amplitude_factory(self.amplitude_factory,self.config)
        if batchsize==0:
            return

        t0 = time.time()
        progbar = (RANK==SIZE-1)
        if progbar:
            _progbar = Progbar(total=batchsize)
        for n in range(batchsize):
            config,omega = self.sampler.sample()
            if progbar:
                _progbar.update()
        if progbar:
            _progbar.close()
        self.config = config
        if RANK==SIZE-1:
            print(f'\tburn in time={time.time()-t0},namps={len(self.amplitude_factory.store)}')
        #print(f'\tRANK={RANK},burn in time={time.time()-t0},namps={len(self.amplitude_factory.store)}')
    def _update_local_energy(self,config,cx):
        ex = 0.0
        c_configs, c_coeffs = self.ham.config_coupling(config)
        for hxy, info_y in zip(c_coeffs, c_configs):
            if np.fabs(hxy) < PRECISION:
                continue
            cy = cx if info_y is None else self.amplitude_factory.amplitude(info_y)
            ex += hxy * cy 
        self.elocal.append(ex/cx)
    def _update_local_hg(self,config,cx,gx):
        ex = 0.0
        hgx = 0.0 
        c_configs, c_coeffs = self.ham.config_coupling(config)
        for hxy, info_y in zip(c_coeffs, c_configs):
            if np.fabs(hxy) < PRECISION:
                continue
            cy,gy = (cx,gx) if info_y is None else self.amplitude_factory.grad(info_y[0])
            ex += hxy * cy 
            hgx += hxy * gy
        self.elocal.append(ex/cx)
        self.hg_local.append(hgx/cx)
    def update_local(self,config):
        if self.vlocal is None:
            cx = self.amplitude_factory.store[config]
        else:
            cx,gx = self.amplitude_factory.grad(config)
        if np.fabs(cx) < PRECISION:
            self.elocal.append(0.)
            if self.vlocal is not None:
                self.vlocal.append(np.zeros_like(gx))
            if self.hg_local is not None:
                self.hg_local.append(np.zeros_like(gx))
        else:
            if self.vlocal is not None:
                self.vlocal.append(gx/cx)
            if self.hg_local is not None:
                self._update_local_hg(config,cx,gx)
            else:
                self._update_local_energy(config,cx)
    def sample(self,energy_only=False): 
        if self.dense:
            self.compute_dense_amplitude() # runs only for dense sampler 
        else: # burn in
            self.burn_in()

        t0 = time.time()
        progbar = (RANK==SIZE-1)
        if progbar:
            _progbar = Progbar(total=self.batchsize)
        self.samples = []
        self.flocal = dict()
        self.elocal = []
        self.vlocal = None 
        self.hg_local = None
        if not energy_only:
            self.vlocal = []
            if self.compute_hg:
                self.hg_local = []
        for n in range(self.batchsize):
            config,omega = self.sampler.sample()
            if config in self.flocal:
                self.flocal[config] += 1
            else:
                self.flocal[config] = 1
                self.samples.append(config)
                self.update_local(config)
            if progbar:
                _progbar.update()
        self.config = config
        self.flocal = np.array([self.flocal[config] for config in self.samples])
        #print(f'RANK={RANK},sample time={time.time()-t0}')
        # sync before gathering
        #COMM.Barrier()
        if RANK==0:
            print('\tsample time=',time.time()-t0)
        self.extract_energy_gradient()
    def extract_energy_gradient(self,new_sample=True):
        if new_sample:
            nlocal = np.array([np.sum(self.flocal)])
            self.n = np.zeros_like(nlocal)
            COMM.Allreduce(nlocal,self.n,op=MPI.SUM)
            self.n = self.n[0]

        self.elocal = np.array(self.elocal)
        fe_local = self.flocal * self.elocal
        esum_local = np.array([np.sum(fe_local)])
        esqsum_local = np.array([np.dot(fe_local,self.elocal)])
        self.E = np.zeros_like(esum_local)
        esq_mean = np.zeros_like(esqsum_local)
        COMM.Allreduce(esum_local,self.E,op=MPI.SUM)
        COMM.Allreduce(esqsum_local,esq_mean,op=MPI.SUM)
        self.E = self.E[0] / self.n 
        esq_mean = esq_mean[0] / self.n
        self.err = np.sqrt((esq_mean-self.E**2)/self.n) 
        if RANK==0:
            print('\tnormalization=',self.n)
            print(f'step={self.step},E={self.E},err={self.err}')

        if self.vlocal is not None:
            self.vlocal = np.array(self.vlocal)
            self.fv_local = np.einsum('i,ij->ij',self.flocal,self.vlocal) 
            vsum_local = np.sum(self.fv_local,axis=0) 
            self.vmean = np.zeros_like(vsum_local)
            COMM.Allreduce(vsum_local,self.vmean,op=MPI.SUM)
            self.vmean /= self.n

            glocal = np.dot(self.elocal,self.fv_local)
            self.g = np.zeros_like(glocal)
            COMM.Allreduce(glocal,self.g,op=MPI.SUM)
            self.g /= self.n
            self.g -= self.E * self.vmean

        if self.hg_local is not None:
            self.hg_local = np.array(self.hg_local)
            self.fhg_local = np.einsum('i,ij->ij',self.flocal,self.hg_local) 
    def getS(self):
        def S(x):
            Sx1_local = np.dot(self.vlocal.T,np.dot(self.fv_local,x)) 
            Sx1 = np.zeros_like(Sx1_local)
            COMM.Reduce(Sx1_local,Sx1,op=MPI.SUM,root=0)
            Sx1 /= self.n
            Sx2 = self.vmean * np.dot(self.vmean,x) 
            return Sx1-Sx2
        return S 
    def getH(self):
        self.ncalls = 0
        hg_mean_local = np.sum(self.fhg_local,axis=0)
        hg_mean = np.zeros_like(hg_mean_local)
        COMM.Reduce(hg_mean_local,hg_mean,op=MPI.SUM,root=0)
        hg_mean /= self.n
        def H(x):
            Hx1_local = np.dot(self.vlocal.T,np.dot(self.fhg_local,x))
            Hx1 = np.zeros_like(Hx1_local) 
            COMM.Reduce(Hx1_local,Hx1,op=MPI.SUM,root=0)
            Hx1 /= self.n
            Hx2 = self.vmean * np.dot(hg_mean,x)
            Hx3 = self.g * np.dot(self.vmean,x)
            return Hx1-Hx2-Hx3
        return H
    def correlated_sampling(self,psi):
        self.amplitude_factory._set_psi(psi)
        self.sampler.amplitude_factory = self.amplitude_factory
        progbar = (RANK==SIZE-1)
        if progbar:
            _progbar = Progbar(total=len(self.samples))
        self.vlocal = []
        self.elocal = []
        for config in self.samples:
            self.update_local(config)
            if progbar:
                _progbar.update()
        # sync before gathering
        #COMM.Barrier()
        self.extract_energy_gradient(new_sample=False) 
    def getH_num(self):
        self.ncalls = 0
        g0 = self.g.copy()
        def H(x):
            psi = _update_psi(self.psi.copy(),-x*self.num_step,self.constructors) 
            self.correlated_sampling(psi)
            return (self.g-g0)/self.num_step 
        return H
    def transform_gradients(self,rate=None,cond=None):
        self.deltas = np.zeros_like(self.g)
        if self.method=='adam':
            self._transform_gradients_adam(rate)
        elif self.method=='sr':
            self._transform_gradients_sr(rate,cond)
        elif self.method=='rgn':
            self._transform_gradients_rgn(rate,cond)
        elif self.method=='lin':
            self._transform_gradients_lin(rate,cond)
        else:
            self._transform_gradients_sgd(rate)

        if RANK>0:
            return
        delta_norm = np.linalg.norm(self.deltas)
        print(f'\tdelta norm={delta_norm}')
        if self.step == 0:
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
    def _transform_gradients_sgd(self,rate):
        if RANK>0:
            return
        g = self.g
        if self.method=='sgd':
            self.deltas = g
        elif self.method=='sign':
            self.deltas = np.sign(g)
        elif self.method=='signu':
            self.deltas = np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError
        self.deltas *= rate
    def _transform_gradients_adam(self,rate):
        if RANK>0:
            return
        g = self.g
        x = self.rate if x is None else x
        if self.step == 0:
            self._ms = np.zeros_like(g)
            self._vs = np.zeros_like(g)
    
        self._ms = (1.-self.beta1) * g + self.beta1 * self._ms
        self._vs = (1.-self.beta2) * g**2 + self.beta2 * self._vs 
        mhat = self._ms / (1. - self.beta1**(self.step+1))
        vhat = self._vs / (1. - self.beta2**(self.step+1))
        self.deltas = rate * mhat / (np.sqrt(vhat)+self.eps)
    def parallel_inv(self,A,b,x0=None,maxiter=None,herm=True):
        terminate = np.array([0])
        buf = np.zeros_like(b)
        sh = len(b)
        def _A(x): # wrapper for synced parallel process
            COMM.Bcast(terminate,root=0)
            if terminate[0]==1:
                return x
            # sync calls to ensure all processes have the same x 
            self.ncalls += 1
            #COMM.Barrier()
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
            #while True:
            #    _A(buf) 
            #    if terminate[0]==1:
            #        return buf,1
            while terminate[0]==0:
                _A(buf)
            return buf,1
    def _transform_gradients_sr(self,rate,cond):
        t0 = time.time()
        g = self.g
        S = self.getS()
        self.ncalls = 0
        def R(vec):
            return S(vec) + cond * vec
        self.deltas,info = self.parallel_inv(R,g)
        if RANK==0:
            print(f'\tSR solver time={time.time()-t0},exit status={info}')
            self.deltas *= rate
    def _transform_gradients_rgn(self,rate,cond):
        if self.compute_hg:
            self._transform_gradients_rgn_WL(rate,cond)
        else:
            self._transform_gradients_rgn_num(rate,cond)
    def _transform_gradients_rgn_WL(self,rate,cond):
        t0 = time.time()
        E = self.E
        g = self.g
        S = self.getS()
        H = self.getH()
        def R(vec):
            return S(vec) + cond * vec
        def A(vec):
            return H(vec) - E * S(vec) + R(vec)/rate 
        self.deltas,info = self.parallel_inv(A,g,herm=False)
        if RANK==0:
            print(f'\tRGN solver time={time.time()-t0},exit status={info}')
    def _transform_gradients_rgn_num(self,rate,cond):
        t0 = time.time()
        g = self.g
        S = self.getS()
        H = self.getH_num()
        sh = len(g)
        def R(vec):
            return S(vec) + cond * vec
        x0,info = self.parallel_inv(R,g)
        if RANK==0:
            print(f'\tSR solver time={time.time()-t0},exit status={info}')

        def A(vec):
            return H(vec) + R(vec)/rate 
        self.deltas,info = self.parallel_inv(A,g,x0=x0,maxiter=10)
        if RANK==0:
            print(f'\tRGN solver time={time.time()-t0},ncalls={self.ncalls}')
    def _transform_gradients_lin(self,rate,cond):
        E = self.E
        g = self.g
        S = self.S
        H = self.H
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        def _S(vec):
            x0,x1 = vec[0],vec[1:]
            return np.concatenate([np.ones(1)*x0,S(x1)]) 
        if self.sr_cond:
            def R(x1):
                return S(x1) + cond * x1
        else:
            def R(x1):
                return cond * x1
        def _H(vec):
            x0,x1 = vec[0],vec[1:]
            Hx0 = np.dot(g,x1)
            Hx1 = x0 * g + H(x1) - E*S(x1) + R(x1)/rate
            return np.concatenate([np.ones(1)*Hx0,Hx1])
        A = spla.LinearOperator((sh+1,sh+1),matvec=_H)
        M = spla.LinearOperator((sh+1,sh+1),matvec=_S)
        dE,deltas = spla.eigs(A,k=1,M=M,sigma=0.)
        deltas = deltas[:,0].real
        deltas = deltas[1:] / deltas[0]
        return - deltas / (1. - np.dot(self._glog,deltas)) 
    def search_rate(self):
        # save the current quantities
        E = self.E
        elocal = self.elocal.copy()
        vlocal = self.vlocal.copy()
        fv_local = self.fv_local.copy()
        vmean = self.vmean.copy()
        g = self.g.copy()

        def f(x):
            if np.linalg.norm(x-self.x)/np.linalg.norm(self.x)<PRECISION:
                return self.E
            psi = vec2psi(self.constructors,x,psi=self.psi.copy())
            self.correlated_sampling(psi) 
            return self.E
        def fprime(x):
            if np.linalg.norm(x-self.x)/np.linalg.norm(self.x)<PRECISION:
                return self.g
            psi = vec2psi(self.constructors,x,psi=self.psi.copy())
            self.correlated_sampling(psi) 
            return self.g

        if self.search_rate_method=='wolfe':
            xk = np.concatenate(psi2vecs(self.concatenate,self.psi))
            gfk = g     
            pk = self.transform_gradients(x=1.)
            old_fval = E
            self.saved_f[self.step] = E
            old_old_fval = self.saved_f.get(self.step-1,None) 
            amax = self.rate_max
            amin = self.rate_min
            maxiter = 3
            from scipy.optimize._linesearch import line_search_wolfe2
            alpha_star,fc,gc,phi_star,old_fval,derphi_star = line_search_wolfe2(f,fprime,xk,pk,
                gfk=gfk,old_fval=old_fval,old_old_fval=old_old_fval,amax=amax,maxiter=maxiter)
            
        if self.search == 'rate':
            deltas = self.transform_gradients(x=1.) 
            trial_deltas = [deltas * xi for xi in x]
        elif self.search == 'cond':
            trial_deltas = [self.transform_gradients(x=xi) for xi in x]
    
        E0,E1 = np.zeros(len(x)),np.zeros(len(x))
        for ix,deltas in enumerate(trial_deltas):
            psi = _update_psi(self.psi.copy(),deltas,self.constructors)
            psi = write_ftn_to_disc(psi,self.tmpdir+'tmp',provided_filename=True)
            self.parse_sampler(psi)
            infos = [(psi,samples,f) for samples,f in zip(self.samples,self.f)]
            ls = parallelized_looped_fxn(line_search_local_sampling,infos,self.sample_args) 
            _esum = sum([opt._esum for opt in ls])
            _esqsum = sum([opt._esqsum for opt in ls])
            n = sum([opt.n for opt in ls])
            E0[ix],E1[ix] = _mean_err(_esum,_esqsum,n)
            print(f'\tx={x[ix]},energy={E0[ix]},err={E1[ix]}')
        x0,y0 = _solve_quad(x,E0)
        return x0,y0
class TNVMC_EX(TNVMC): # exact sampling
    def __init__(self,psi,ham,sampler_opts,contract_opts,optimizer_opts,conditioner=None):
        super().__init__(psi,ham,sampler_opts,contract_opts,optimizer_opts,conditioner=conditioner)
        self.sampler = DenseSampler(sampler_opts)
        self.dense = True
    def burn_in(self):
        raise NotImplementedError
    def sample(self,energy_only=False): 
        self.compute_dense_amplitude() # runs only for dense sampler 
        t0 = time.time()
        start,stop = self.sampler.start,self.sampler.stop
        if self.progbar:
            _progbar = Progbar(total=stop-start)
        self.elocal = []
        self.vlocal = None 
        self.hg_local = None 
        if not energy_only:
            self.vlocal = []
            if self.compute_hg:
                self.hg_local = [] 
        for n in range(start,stop):
            config,omega = self.sampler.sample_exact(n)
            self.update_local(config)
            if self.progbar:
                _progbar.update()
        self.flocal = self.sampler.p[start:stop]
        # sync before gathering
        #COMM.Barrier()
        if self.progbar:
            print('\tsample time=',time.time()-t0)
        self.extract_energy_gradient()
    def getH_num(self):
        g0 = self.g.copy()
        def H(x):
            # sync calls to ensure all processes have the same x 
            #self.ncalls += 1
            #COMM.Barrier()
            #COMM.Bcast(x,root=0)

            psi = _update_psi(self.psi.copy(),-x*self.num_step,self.constructors) 
            self.amplitude_factory._set_psi(psi)
            self.sampler.amplitude_factory = self.amplitude_factory
            self.sample()
            # sync before gathering
            #COMM.Barrier()
            #self.extract_energy_gradient()
            return (self.g-g0)/self.num_step 
        return H 
class LBFGS(TNVMC):
    def __init__(self,psi,ham,sampler_opts,contract_opts,optimizer_opts,conditioner=None):
        optimizer_opts['method'] = 'bfgs'
        super().__int__(psi,ham,sampler_opts,contract_opts,optimizer_opts,conditioner=conditioner)
        from scipy.optimize import lbfgsb
        self.m = optimizer_opts.get('m',10)
        self.hessinvp = lbfgsb.LbfgsInvHessProduct
        self.s = None
        self.y = None
        self.xprev = None
        self.gprev = None
    def getS(self):
        raise NotImplementedError
    def getH(self):
        raise NotImplementedError
    def transform_gradients(self,x=None):
        if self.step == 0:
            return self.transform_gradients_sgd(x=x)
        x = self.rate if x is None else x
        hessinvp = self.hessinvp(self.s,self.y)
        z = hessinvp._matvec(self.g)
        return x * z
    #def _sample_lbfgs(self):
    #    if self.step>0:
    #        prev_samples = self.samples.copy()
    #        prev_f = self.f.copy()
    #        prev_x = self.x.copy()
    #        prev_g = self.g.copy()
    #    self.x = np.concatenate(psi2vecs(self.constructors,self.psi))
    #    self._sample()
    #    if self.step==0:
    #        return
    #    if self.sampler.exact:
    #        g_corr = self.g
    #    else:
    #        opt = get_optimizer(self.optimizer_opts)
    #        opt.parse_samples(prev_samples,prev_f)

    #        t0 = time.time()
    #        self.amplitude_factory,opt = update_grads(self.amplitude_factory,opt)
    #        if RANK==SIZE-1:
    #            print(f'\tcorrelated gradient time={time.time()-t0}')

    #        t0 = time.time()
    #        opt = compute_elocs(self.ham,self.amplitude_factory,opt)
    #        if RANK==SIZE-1:
    #            print('\tcorrelated eloc time=',time.time()-t0)

    #        g_corr = self.extract_energy_gradient(opt)[-1]
    #    if RANK>0:
    #        return
    #    # RANK==0 only
    #    size = len(self.x)
    #    yprev = (self.x - prev_x).reshape(1,size)
    #    sprev = (g_corr - prev_g).reshape(1,size)
    #    if self.y is None:
    #        self.y = yprev 
    #        self.s = sprev
    #    else:
    #        self.y = np.concatenate([self.y,yprev],axis=0) 
    #        self.s = np.concatenate([self.s,sprev],axis=0)

def _update_psi(psi,deltas,constructors):
    deltas = vec2psi(constructors,deltas)
    for ix,(_,_,_,site) in enumerate(constructors):
        tsr = psi[psi.site_tag(*site)]
        data = tsr.data - deltas[site] 
        tsr.modify(data=data)
    return psi
def _solve_quad(x,y):
    m = np.stack([np.square(x),x,np.ones(3)],axis=1)
    a,b,c = list(np.dot(np.linalg.inv(m),y))
    if a < 0. or a*b > 0.:
        idx = np.argmin(y)
        x0,y0 = x[idx],y[idx]
    else:
        x0,y0 = -b/(2.*a),-b**2/(4.*a)+c
    print(f'x={x},y={y},x0={x0},y0={y0}')
    return x0,y0
#def _solve_quad(x,y):
#    ix = np.argmin(y)
#    return x[ix],y[ix]
 
