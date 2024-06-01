import time,scipy,functools,h5py,gc
import numpy as np
import scipy.sparse.linalg as spla
import scipy.optimize as opt
from .tfqmr import tfqmr

from quimb.utils import progbar as Progbar
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
##################################################################################################
# VMC utils
##################################################################################################
#def _cubic_interpolation(fun,alpha1,alpha2,alpha3,xk,pk):
#    M = np.zeros((3,3))
#    b = np.zeros(3)
#    for i,alpha in enumerate([alpha1,alpha2,alpha3]):
#        b[i] = fun(xk+alpha*pk)
#        for j,p in enumerate([3,2,1]):
#            M[i,j] = alpha**p
#    a,b,c = np.linalg.solve(M,b)
#    sq = b**2-3*a*c
#    if sq<0:
#        return -1 
#    roots = [(-b+sign*np.sqrt(sq))/(3*a) for sign in [1,-1]] 
#    if a>0:
#        return roots[1]
#    else:
#        return roots[0]
#def _3point_interpolation(fun,alpha_k,xk,pk):
#    s = .5,1.,1.1
#    if xk[-1]+s[-1]*alpha_k*pk[-1]<0:
#        smax = -xk[-1]/pk[-1]/alpha_k
#        s = np.array([.2,.5,.9])*smax
#    x = [xk+si*alpha_k*pk for si in s]
#    b = np.array([fun(xi) for xi in x])
#    ix = np.argmin(b) 
#    return alpha_k*s[ix],b[ix],x[ix]
#def _minimize(update,x0,ctr_mu=True,alpha_0=.1,maxiter=None,iprint=0):
#    maxiter = MAXITER if maxiter is None else maxiter
#    xk = x0
#    alpha_k = alpha_0
#    for i in range(maxiter):
#        fk,gk,pk,status = update(xk)
#        gnorm = np.linalg.norm(gk)
#        if iprint>0:
#            print(f'iter={i},gnorm={gnorm},fval={fk},constr={gk[-1]},mu={xk[-1]}')
#        dEm = fk - xk[-1] * gk[-1]
#        if gnorm < CG_TOL:
#            return xk,dEm,0
#        if status!=0:
#            return xk,dEm,3
#
#        xkp1 = xk + alpha_k * pk
#        #if ctr_mu and xkp1[-1]<0:
#        #    return xk,dEm,2
#        xk = xkp1
#    return xk,dEm,1 
#def _minimize(fun,jac,hess,x0,ctr_mu=True,alpha0=1e-2):
#    xk = x0
#    gk = jac(x0)
#    old_fval = fun(x0)
#    old_old_fval = old_fval + np.linalg.norm(gk)/2
#    for i in range(MAXITER):
#        Hk = hess(xk) 
#        pk = -np.linalg.solve(Hk,gk) 
#        ls = _line_search_wolfe12
#        if ctr_mu and pk[-1] < 0:
#            ls = functools.partial(ls,amax=-xk[-1]/pk[-1]) 
#        try:
#            alpha_k,_,_,old_fval,old_old_fval,gk =  ls(fun,jac,xk,pk,gk,old_fval,old_old_fval)
#        except _LineSearchError:
#            alpha_k = alpha0
#            if ctr_mu and pk[-1] < 0:
#                alpha_k = min(alpha_k,-xk[-1]/pk[-1])
#            return xk,2
#        
#        xk += alpha_k * pk
#        if ctr_mu and xk[-1]<0:
#            return xk,3
#        if gk is None:
#            gk = jac(xk)
#        gnorm = np.linalg.norm(gk)
#        print(f'iter={i},gnorm={gnorm},fval={old_fval},alpha_k={alpha_k}')
#        if gnorm < CG_TOL:
#            return xk,0
#    return xk,1 
def _lin_block_solve(H,E,S,g,hmean,vmean,cond):
    Hi0 = g
    H0j = hmean - E * vmean
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
    v = v[1:,idx]/v[0,idx]
    #v = v[1:,idx]/np.sign(v[0,idx])
    return w[idx],v,idx
def blocking_analysis(energies, weights=None, neql=0, printQ=True):
    weights = np.ones_like(energies) if weights is None else weights
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

    #print(RANK,plateauError,error)
    if plateauError is None:
        plateauError = error
    else:
        if printQ:
            print(f'Stocahstic error estimate: {plateauError:.6e}\n')

    return meanEnergy, plateauError
def to_square(S,sh,idx=None):
    idx = np.triu_indices(sh) if idx is None else idx
    # assume first dim is triangular
    Snew = np.zeros((sh,)*2+S.shape[1:],dtype=S.dtype) 
    Snew[idx] = S
    Snew = np.swapaxes(Snew,0,1)
    Snew[idx] = S
    return Snew
##################################################################################################
# VMC engine 
##################################################################################################
class SGD: # stochastic sampling
    def __init__(
        self,
        sampler,
        dtype_o=float,
        optimizer='sgd',
        solve_full=True,
        solve_dense=False,
        maxiter=None,
    ):
        # parse sampler
        self.sampler = sampler

        # parse wfn 
        self.nsite = self.sampler.af.nsite
        x = self.sampler.af.get_x()
        self.dtype_i = x.dtype
        self.dtype_o = dtype_o

        # parse gradient optimizer
        self.optimizer = optimizer
        self.solve_full = solve_full
        self.solve_dense = solve_dense
        self.compute_h = False
        self.maxiter = 500 
        self.discard = 1e3
        self.tol = 1e-4

        # to be set before run
        self.tmpdir = None
        self.batchsize = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR
        self.ctr_update = None

        self.progbar = False
        self.save_local = False
        self.save_grad_hess = False
        self.free_g = True
    def free_quantities(self):
        if RANK==0: 
            self.Eold = self.E
        self.f = None
        self.e = None
        if self.free_g:
            self.g = None
        self.v = None
        self.vmean = None
        self.evmean = None
        self.vsum = None
        self.evsum = None
        self.h = None
        self.hmean = None
        self.hsum = None

        self.S = None
        self.vvmean = None
        self.H = None
        self.vhmean = None
        self.Sx1 = None
        self.Hx1 = None
        self.Hx1h = None
        gc.collect()
    def run(self,start,stop,save_wfn=True):
        self.Eold = None 
        for step in range(start,stop):
            self.step = step
            self.nparam = self.sampler.af.nparam
            if RANK==0:
                print('\nnparam=',self.nparam)
            self.sample()
            self.extract_energy_gradient()
            x = self.transform_gradients()
            self.free_quantities()
            COMM.Bcast(x,root=0) 
            fname = self.tmpdir+f'psi{step+1}' if save_wfn else None
            psi = self.sampler.af.update(x,fname=fname,root=0)
    def sample(self,samplesize=None,save_local=None,compute_v=True,compute_h=None,save_config=True):
        self.sampler.preprocess()

        if self.sampler.exact:
            samplesize = len(self.sampler.nonzeros)
            save_config = False
        else:
            samplesize = self.batchsize if samplesize is None else samplesize
        compute_h = self.compute_h if compute_h is None else compute_h
        save_local = self.save_local if save_local is None else save_local 

        self.buf = np.zeros(3,dtype=int)
        self.buf[1] = RANK
        self.buf[2] = self.step
        self.c = []
        self.e = []
        self.err_max = 0.
        self.err_sum = 0.
        if compute_v:
            self.evsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.vsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.v = []
        if compute_h:
            self.hsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.h = [] 

        if RANK==0:
            self._ctr(samplesize,save_config)
        else:
            if self.sampler.exact:
                self._sample_exact(compute_v,compute_h)
            else:
                self._sample_stochastic(compute_v,compute_h,save_config,save_local)
    def _ctr(self,samplesize,save_config):
        if self.progbar:
            pg = Progbar(total=samplesize)

        t0 = time.time()
        ncurr = 0
        while True:
            COMM.Recv(self.buf,tag=0)
            if self.buf[-1]>self.step:
                raise ValueError
            if self.buf[-1]<self.step:
                continue

            ncurr += 1
            if ncurr >= samplesize: 
                break
            if self.progbar:
                pg.update()
            if not self.sampler.exact:
                COMM.Send(self.buf,dest=self.buf[1],tag=1)
        # send termination message to all workers
        if not self.sampler.exact:
            self.buf[0] = 1
            for worker in range(1,SIZE): 
                COMM.Send(self.buf,dest=worker,tag=1)

        if save_config:
            ls = []
            for worker in range(1,SIZE):
                config = np.zeros(self.nsite,dtype=int)
                COMM.Recv(config,source=worker,tag=2)
                ls.append(config.copy())
            np.save(self.tmpdir+f'config{self.step}.npy',np.array(ls))
        print('\tsample time=',time.time()-t0)
    def _sample_stochastic(self,compute_v,compute_h,save_config,save_local):
        configs = []
        while True:
            config,omega = self.sampler.sample()
            #if omega > self.omega:
            #    self.config,self.omega = config,omega
            cx,ex,vx,hx,err = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_h=compute_h)
            if cx is None or np.fabs(ex.real) > self.discard:
                print(f'RANK={RANK},cx={cx},ex={ex}')
                ex = np.zeros(1)[0]
                err = 0.
                if compute_v:
                    vx = np.zeros(self.nparam,dtype=self.dtype_o)
                if compute_h:
                    hx = np.zeros(self.nparam,dtype=self.dtype_o)
            self.c.append(cx)
            self.e.append(ex)
            self.err_sum += err
            self.err_max = max(err,self.err_max)
            if compute_v:
                self.vsum += vx
                self.evsum += vx * ex.conj()
                self.v.append(vx)
            if compute_h:
                self.hsum += hx
                self.h.append(hx)
            if save_local:
                configs.append(list(config))
            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.buf,source=0,tag=1)
            if self.buf[0]==1:
                break

        #self.sampler.config = self.config
        self.e = np.array(self.e)
        self.c = np.array(self.c)
        if compute_v:
            self.v = np.array(self.v)
        if compute_h:
            self.h = np.array(self.h)
        if save_config:
            config = np.array(config,dtype=int)
            COMM.Send(config,dest=0,tag=2)
        if save_local:
            f = h5py.File(self.tmpdir+f'step{self.step}RANK{RANK}.hdf5','w')
            if compute_h:
                f.create_dataset('h',data=self.h)
            if compute_v:
                f.create_dataset('v',data=self.v)
            f.create_dataset('e',data=self.e)
            f.create_dataset('c',data=self.c)
            f.create_dataset('config',data=np.array(configs))
            f.close()
    def _sample_exact(self,compute_v,compute_h): 
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
            cx,ex,vx,hx,err = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_h=compute_h)
            if cx is None:
                raise ValueError
            if np.fabs(ex.real)*p[ix] > self.discard:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self.f.append(p[ix])
            self.e.append(ex)
            self.err_sum += err
            self.err_max = max(err,self.err_max)
            if compute_v:
                self.vsum += vx * p[ix]
                self.evsum += vx * ex.conj() * p[ix]
                self.v.append(vx)
            if compute_h:
                self.hsum += hx * p[ix]
                self.h.append(hx)
            COMM.Send(self.buf,dest=0,tag=0) 
            #COMM.Recv(self.buf,source=0,tag=1)
        self.f = np.array(self.f)
        self.e = np.array(self.e)
        self.c = np.array(self.c)
        if compute_v:
            self.v = np.array(self.v)
        if compute_h:
            self.h = np.array(self.h)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        if self.optimizer in ['rgn','lin','trust']:
            self._extract_hmean()
        if RANK==0:
            try:
                dE = 0 if self.Eold is None else self.E-self.Eold
                print(f'step={self.step},E={self.E/self.nsite},dE={dE/self.nsite},err={self.Eerr/self.nsite}')
                print(f'\tgnorm=',np.linalg.norm(self.g))
            except TypeError:
                print('E=',self.E)
                print('Eerr=',self.Eerr)
            print('\tcollect g,h time=',time.time()-t0)
    def extract_energy(self):
        count = 0 if RANK==0 else len(self.e)
        count = np.array([count])
        counts = np.zeros(SIZE,dtype=int)
        COMM.Gather(count,counts,root=0)

        self.err_sum = np.zeros(1) if RANK==0 else np.array([self.err_sum])
        err_mean = np.zeros_like(self.err_sum)
        COMM.Reduce(self.err_sum,err_mean,op=MPI.SUM,root=0)

        self.err_max = np.zeros(1) if RANK==0 else np.array([self.err_max])
        err_max = np.zeros(SIZE)
        COMM.Gather(self.err_max,err_max,root=0)
        self.n = 1 if self.sampler.exact else counts.sum() 
        
        self.E = np.zeros(1)[0]
        if RANK>0:
            COMM.Send(self.e,dest=0,tag=3)
            if self.sampler.exact:
                COMM.Send(self.f,dest=0,tag=4)
            return
        err_mean = err_mean[0] / self.n
        err_max = np.amax(err_max)
        print('\tcontraction err=',err_mean,err_max)
        e = []
        f = []
        for worker in range(1,SIZE):
            self.e = np.zeros(counts[worker],dtype=self.dtype_o)
            COMM.Recv(self.e,source=worker,tag=3)
            e.append(self.e.copy())
            if self.sampler.exact:
                self.f = np.zeros(counts[worker],dtype=self.dtype_o)
                COMM.Recv(self.f,source=worker,tag=4)
                f.append(self.f.copy())
        e = np.concatenate(e)
        if self.sampler.exact:
            f = np.concatenate(f)
            self.E = np.dot(f,e)
            self.Eerr = 0.
        else:
            print('nsamples=',self.n)
            self.E,self.Eerr = blocking_analysis(e)
    def extract_gradient(self):
        vmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.vsum,vmean,op=MPI.SUM,root=0)
        evmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.evsum,evmean,op=MPI.SUM,root=0)
        if RANK>0:
            return 
        self.vmean = vmean/self.n
        self.evmean = evmean/self.n
        self.g = (self.evmean - self.E.conj() * self.vmean).real
    def update(self,deltas):
        x = self.sampler.af.get_x()
        xnorm = np.linalg.norm(x)
        dnorm = np.linalg.norm(deltas) 
        print(f'\txnorm={xnorm},dnorm={dnorm}')
        if self.ctr_update is not None:
            tg = self.ctr_update * xnorm
            if dnorm > tg:
                deltas *= tg/dnorm
        return x - deltas
    def transform_gradients(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i) 
        if self.optimizer=='sgd':
            deltas = self.g
        elif self.optimizer=='sign':
            deltas = np.sign(self.g)
        elif self.optimizer=='signu':
            deltas = np.sign(self.g) * np.random.uniform(size=self.nparam)
        else:
            raise NotImplementedError
        return self.update(self.rate1*deltas)
#    def measure(self,fname=None):
#        self.sample(compute_v=False,compute_h=False)
#
#        sendbuf = np.array([self.ham.n])
#        recvbuf = np.zeros_like(sendbuf) 
#        COMM.Reduce(sendbuf,recvbuf,op=MPI.SUM,root=0)
#        n = recvbuf[0]
#
#        sendbuf = self.ham.data
#        recvbuf = np.zeros_like(sendbuf) 
#        COMM.Reduce(sendbuf,recvbuf,op=MPI.SUM,root=0)
#        if RANK>0:
#            return
#        data = recvbuf / n
#        if fname is not None:
#            f = h5py.File(fname,'w')
#            f.create_dataset('data',data=data) 
#            f.close()
#        self.ham._print(fname,data)
#    def debug_torch(self,tmpdir,step,rank=None):
#        if RANK==0:
#            return
#        if rank is not None:
#            if RANK!=rank:
#                return
#        f = h5py.File(tmpdir+f'step{step}RANK{RANK}.hdf5','r')
#        e = f['e'][:]
#        v = f['v'][:]
#        configs = f['config'][:]
#        f.close()
#        e_new = []
#        c_new = []
#        v_new = []
#        h_new = []
#        n = len(e)
#        print(f'RANK={RANK},n={n}')
#        for i in range(n):
#            config = tuple(configs[i,:])
#            cx,ex,vx,hx,err = self.sampler.af.compute_local_energy(
#                config,compute_v=True,compute_h=True)
#            if cx is None or np.fabs(ex) > DISCARD:
#                print(f'RANK={RANK},cx={cx},ex={ex}')
#                ex = 0.
#                err = 0.
#                if compute_v:
#                    vx = np.zeros(self.nparam,dtype=self.dtype)
#                if compute_h:
#                    hx = np.zeros(self.nparam,dtype=self.dtype)
#            e_new.append(ex) 
#            c_new.append(cx) 
#            v_new.append(vx)
#            h_new.append(hx)
#            err_e = np.fabs(ex-e[i])
#            err_v = np.linalg.norm(vx-v[i,:])
#            if err_e > 1e-6 or err_v > 1e-6: 
#                print(f'RANK={RANK},config={config},ex={ex},ex_sr={e[i]},err_e={err_e},err_v={err_v}')
#            #else:
#            #    print(f'RANK={RANK},i={i}')
#        f = h5py.File(f'./step{step}RANK{RANK}.hdf5','w')
#        f.create_dataset('h',data=np.array(h_new))
#        f.create_dataset('v',data=np.array(v_new))
#        f.create_dataset('e',data=np.array(e_new))
#        f.create_dataset('c',data=np.array(c_new))
#        f.close()
    def load(self,size,tmpdir='./'):
        if RANK==0:
            self.vsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.hsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.evsum = np.zeros(self.nparam,dtype=self.dtype_o)
        else:
            start = 1 + (RANK-1) * size
            self.e = []
            self.v = []
            self.h = []
            for rank in range(start,start+size):
                try:
                    f = h5py.File(tmpdir+f'step{self.step}RANK{rank}.hdf5','r')
                    e = f['e'][:]
                    v = f['v'][:]
                    h = f['h'][:]
                    f.close()
                    self.e.append(e.newbyteorder('='))
                    self.v.append(v.newbyteorder('='))
                    self.h.append(h.newbyteorder('='))
                except FileNotFoundError:
                    break
            self.e = np.concatenate(self.e)
            self.v = np.concatenate(self.v,axis=0)
            self.h = np.concatenate(self.h,axis=0)
            self.vsum = np.sum(self.v,axis=0)
            self.hsum = np.sum(self.h,axis=0)
            self.evsum = np.dot(self.e,self.v)
        self.err_sum = 0.
        self.err_max = 0.
    def autocorrelation(self,typ):
        n = len(self.e)
        trange = range(1,n-1)
        if typ in ('e','g'):
            emean = sum(self.e) / n
            e = self.self.e - emean
        if typ in ('v','g'):
            vmean = self.vsum / n
            v = self.v - vmean.reshape(1,len(vmean)) 
        if typ=='e':
            data = e.reshape((len(e),1))
        if typ=='v':
            data = v
        if typ=='g':
            data = e * v
        if typ=='h':
            hmean = self.hsum / n
            data = self.h - hmean.reshape(1,len(hmean))
        out = 0
        for ix1,t in enumerate(trange): 
            v0,v1 = data[:-t],data[t:]
            out = out + np.dot(v0.T,v1)
        return out 
#    def _hess(self,solve_dense=None,mode='eig',gen=True,lin=True):
#        solve_dense = self.solve_dense if solve_dense is None else solve_dense
#        self.extract_S(solve_full=True,solve_dense=solve_dense)
#        self.extract_H(solve_full=True,solve_dense=solve_dense)
#        if solve_dense:
#            return self._dense_hess(mode=mode,gen=gen,lin=lin)
#        else:
#            return self._iterative_hessSVD()
#    def _dense_hess(self,mode='eig',gen=True,lin=True):
#        if RANK>0:
#            return 
#        A = self.H - self.E * self.S
#        #A = self.S
#        gen = True if lin else gen
#        if mode=='svd':
#            #A = A + (self.S + self.cond1 * np.eye(self.nparam))/self.rate2
#            #_,w,_ = spla.svds(A,k=1,tol=CG_TOL,maxiter=MAXITER)
#            _,w,_ = np.linalg.svd(A,hermitian=True)
#        elif mode=='eig':
#            M = None if not gen else self.S + self.cond1 * np.eye(self.nparam)
#            if lin:
#                A = np.block([[np.zeros((1,1)),self.g.reshape(1,self.nparam)],
#                              [self.g.reshape(self.nparam,1),A]]) 
#                M = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
#                              [np.zeros((self.nparam,1)),M]]) 
#            #w,_ = spla.eigsh(A,k=1,M=M,tol=CG_TOL,maxiter=MAXITER)
#            w,_ = scipy.linalg.eigh(A,b=M)
#        else:
#            raise NotImplementedError
#        return w
#    def _eig_iterative(self,A,M=None,symm=False):
#        self.terminate = np.array([0])
#        sh = self.nparam
#        deltas = np.zeros(sh,dtype=self.dtype)
#        w = None
#        if RANK==0:
#            t0 = time.time()
#            A = spla.LinearOperator((sh,sh),matvec=A,dtype=self.dtype)
#            if M is not None:
#                M = spla.LinearOperator((sh,sh),matvec=M,dtype=self.dtype)
#            eig = spla.eigsh if symm else spla.eigs 
#            w,_ = eig(A,k=1,tol=CG_TOL,maxiter=MAXITER)
#            self.terminate[0] = 1
#            COMM.Bcast(self.terminate,root=0)
#        else:
#            nit = 0
#            while self.terminate[0]==0:
#                nit += 1
#                A(deltas)
#            if RANK==1:
#                print('niter=',nit)
#        return w 
#    def _iterative_hessSVD(self,bare=False):
#        E = self.E if RANK==0 else 0
#        # rmatvec
#        self.xH1 = np.zeros(self.nparam,dtype=self.dtype)
#        if RANK==0:
#            def _rmatvec(x):
#                COMM.Bcast(self.terminate,root=0)
#                COMM.Bcast(x,root=0)
#                xH1 = np.zeros_like(self.xH1)
#                COMM.Reduce(xH1,self.xH1,op=MPI.SUM,root=0)     
#                return self.xH1 / self.n \
#                     - np.dot(x,self.vmean) * self.hmean \
#                     - np.dot(x,self.g) * self.vmean
#        else:
#            def _rmatvec(x):
#                COMM.Bcast(self.terminate,root=0)
#                if self.terminate[0]==1:
#                    return 0 
#                COMM.Bcast(x,root=0)
#                if self.sampler.exact:
#                    xH1 = np.dot(self.f * np.dot(self.v,x),self.h)
#                else:
#                    xH1 = np.dot(np.dot(self.v,x),self.h)
#                COMM.Reduce(xH1,self.xH1,op=MPI.SUM,root=0)     
#                return x 
#
#        def matvec(x):
#            if self.terminate[0]==1:
#                return 0
#            Hx = self.H(x)
#            if self.terminate[0]==1:
#                return 0
#            Sx = self.S(x)
#            if self.terminate[0]==1:
#                return 0
#            return Hx + (1./self.rate2 - E) * Sx + self.cond1 / self.rate2 * x
#        def rmatvec(x):
#            if self.terminate[0]==1:
#                return 0
#            xH = _rmatvec(x)
#            if self.terminate[0]==1:
#                return 0
#            xS = self.S(x)
#            if self.terminate[0]==1:
#                return 0
#            return xH + (1./self.rate2 - E) * xS + self.cond1 / self.rate2 * x
#        def MM(x):
#            return matvec(rmatvec(x))
#        w = self._eig_iterative(MM,symm=True)
#        if RANK==0:
#            print(w,np.sqrt(w))
class SR(SGD):
    def __init__(self,sampler,solve_reduce=False,eigen_thresh=None,**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'sr' 
        self.solve_reduce = solve_reduce 
        self.eigen_thresh = eigen_thresh 
    def from_square(self,S,idx=None):
        # assume first 2 dim are square
        idx = self.triu_idx if idx is None else idx
        return S[idx]
    def to_square(self,S,sh=None):
        if sh is None:
            sh = self.nparam
            idx = self.triu_idx
        else:
            idx = np.triu_indices(sh)
        return to_square(S,sh,idx=idx)
    def _save_grad_hess(self,deltas=None):
        if RANK>0:
            return
        if self.solve_full:
            S = self.S
        else:
            S = np.zeros((self.nparam,self.nparam))
            for ix,(start,stop) in enumerate(self.block_dict):
                S[start:stop,start:stop] = self.S[ix]
        f = h5py.File(self.tmpdir+f'step{self.step}.hdf5','w')
        f.create_dataset('S',data=S) 
        f.create_dataset('E',data=np.array([self.E])) 
        f.create_dataset('g',data=self.g) 
        if deltas is not None:
            f.create_dataset('deltas',data=deltas) 
        f.close()
    def extract_S(self,solve_full,solve_dense):
        if self.solve_reduce:
            return self._get_S_reduce() 
        fxn = self._get_Smatrix if solve_dense else self._get_S_iterative
        self.Sx1 = np.zeros(self.nparam,dtype=self.dtype_i)
        if solve_full:
            self.S = fxn() 
        else:
            self.S = [None] * self.nsite
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                self.S[ix] = fxn(start=start,stop=stop)
    def _get_Smatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        t0 = time.time()
        sh = stop-start
        idx = np.triu_indices(sh) 
        np2 = (1 + sh) * sh // 2
        if RANK==0:
            vvsum = np.zeros(np2,dtype=self.dtype_i)
        else:
            v = self.v[:,start:stop] 
            if self.sampler.exact:
                vvsum = np.einsum('s,si,sj->ij',self.f,v.conj(),v)
            else:
                vvsum = np.dot(v.T.conj(),v)
            vvsum = self.from_square(vvsum,idx=idx)
        vvsum = np.ascontiguousarray(vvsum.real)
        vvmean = np.zeros_like(vvsum)
        COMM.Reduce(vvsum,vvmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        vmean = self.vmean[start:stop]
        self.vvmean = vvmean / self.n
 
        print('\tcollect S matrix time=',time.time()-t0)
        return self.to_square(self.vvmean,sh=sh) \
             - np.outer(vmean.conj(),vmean).real
    def _get_S_iterative(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop 
        if RANK==0:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                Sx1 = np.zeros_like(self.Sx1[start:stop])
                COMM.Reduce(Sx1,self.Sx1[start:stop],op=MPI.SUM,root=0)     
                vmean = self.vmean[start:stop]
                return self.Sx1[start:stop] / self.n \
                     - (vmean.conj() * np.dot(vmean,x)).real
        else: 
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                v = self.v[:,start:stop]
                if self.sampler.exact:
                    Sx1 = np.dot(self.f * np.dot(v,x),v.conj())
                else:
                    Sx1 = np.dot(np.dot(v,x),v.conj())
                Sx1 = np.ascontiguousarray(Sx1.real)
                COMM.Reduce(Sx1,self.Sx1[start:stop],op=MPI.SUM,root=0)     
                return 0 
        return matvec
    def _get_S_reduce(self):
        if RANK==0:
            nsi = np.array([1])
        else:
            u,s,vh = np.linalg.svd(self.v,full_matrices=False)
            s = s[s>np.sqrt(self.eigen_thresh)]**2
            nsi = len(s)
            u = u[:,:nsi]
            vh = vh[:nsi,:]
            #self.v = {'u':u,'s':s,'v':v} 
            nsi = np.array([nsi])
        ns = np.zeros(SIZE,dtype=int)
        COMM.Gather(nsi,ns,root=0)
        if RANK>0:
            COMM.Send(vh,dest=0,tag=5)
            COMM.Send(s,dest=0,tag=6)
            return 
        vh_ls = [self.vmean.reshape(1,self.nparam)]
        s_ls = [-np.ones(1)]
        for worker in range(1,SIZE):
            vh = np.zeros((ns[worker],self.nparam))
            COMM.Recv(vh,source=worker,tag=5)
            vh_ls.append(vh) 

            s = np.zeros(ns[worker])
            COMM.Recv(s,source=worker,tag=6)
            s_ls.append(s/self.n) 
        vh = np.concatenate(vh_ls,axis=0) 
        s_mid = np.concatenate(s_ls,axis=0) 

        # decompose vh
        u,s,vh = np.linalg.svd(vh,full_matrices=False)
        s = s[s>self.eigen_thresh]
        ns = len(s)
        u = u[:,:ns]
        vh = vh[:ns,:]
        s_mid = np.einsum('si,sj,s->ij',u,u,s_mid)
        s = np.einsum('i,j,ij->ij',s,s,s_mid)

        # decompose s 
        u,s,vh_ = np.linalg.svd(s,full_matrices=False,hermitian=True)
        s = s[s>self.eigen_thresh]
        ns = len(s)
        u = u[:,:ns]
        vh_ = vh_[:ns,:]
        
        rhs = s**(-1) * np.dot(u.T,np.dot(vh,self.g))
        lhs = np.dot(vh_,vh)
        self.S = lhs,rhs
        print('number of singular values in S=',ns)
    def transform_gradients(self):
        deltas = self._transform_gradients_sr(self.solve_full,self.solve_dense)
        if RANK>0:
            return deltas
        return self.update(self.rate1*deltas)
    def _transform_gradients_sr(self,solve_full,solve_dense):
        self.extract_S(solve_full,solve_dense)
        if self.solve_reduce:
            return self._transform_gradients_sr_reduce()
        if solve_dense:
            return self._transform_gradients_sr_dense(solve_full)
        else:
            return self._transform_gradients_sr_iterative(solve_full)
    def _transform_gradients_sr_dense(self,solve_full):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i) 
        t0 = time.time()
        if solve_full:
            if self.eigen_thresh is None:
                deltas = np.linalg.solve(self.S + self.cond1 * np.eye(self.nparam),self.g)
            else:
                w,v = np.linalg.eigh(self.S)
                w = w[w>self.eigen_thresh*w[-1]]
                print(f'\tnonzero={len(w)},wmax={w[-1]}')
                v = v[:,-len(w):]
                deltas = np.dot(v/w.reshape(1,len(w)),np.dot(v.T,self.g)) 
        else:
            deltas = np.empty(self.nparam,dtype=self.dtype_i)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                S = self.S[ix] + self.cond1 * np.eye(stop-start)
                deltas[start:stop] = np.linalg.solve(S,self.g[start:stop])
        print('\tSR solver time=',time.time()-t0)
        if self.save_grad_hess:
            self._save_grad_hess(deltas=deltas)
        return deltas
    def _transform_gradients_sr_iterative(self,solve_full):
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
        if solve_full: 
            def R(x):
                return self.S(x) + self.cond1 * x
            deltas = self.solve_iterative(R,g,True,x0=g)
        else:
            deltas = np.empty(self.nparam,dtype=self.dtype_i)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                def R(x):
                    return self.S[ix](x) + self.cond1 * x
                deltas[strt:stop] = self.solve_iterative(R,g[start:stop],True,x0=g[start:stop])
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)  
        return deltas
    def _transform_gradients_sr_reduce(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i) 
        lhs,rhs = self.S
        deltas,res,rank,s = np.linalg.lstsq(lhs,rhs,rcond=self.eigen_thresh)
        print('lstsq res=',res)
        return deltas
    def solve_iterative(self,A,b,symm,x0=None):
        self.terminate = np.array([0])
        deltas = np.zeros_like(b)
        sh = len(b)
        if RANK==0:
            print('symmetric=',symm)
            t0 = time.time()
            LinOp = spla.LinearOperator((sh,sh),matvec=A,dtype=b.dtype)
            if symm:
                deltas,info = spla.minres(LinOp,b,x0=x0,tol=self.tol,maxiter=self.maxiter)
            else: 
                deltas,info = self.solver(LinOp,b,x0=x0,tol=self.tol,maxiter=self.maxiter)
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
class RGN(SR):
    def __init__(self,sampler,pure_newton=False,solver='lgmres',guess=1,**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'rgn' 
        self.compute_h = True
        self.pure_newton = pure_newton
        self.solver = {'lgmres':spla.lgmres,
                       'tfqmr':tfqmr}[solver] 
        self.guess = guess
        self.solve_symmetric = 0 

        self.rate2 = None # rate for LIN,RGN
        self.cond2 = None
        self.check = [1] 
    def _save_grad_hess(self,deltas=None):
        if RANK>0:
            return
        if self.solve_full:
            H = self.H
            S = self.S
        else:
            H = np.zeros((self.nparam,self.nparam))
            S = np.zeros((self.nparam,self.nparam))
            for ix,(start,stop) in enumerate(self.block_dict):
                H[start:stop,start:stop] = self.H[ix] 
                S[start:stop,start:stop] = self.S[ix]
        #print(H)
        f = h5py.File(self.tmpdir+f'step{self.step}.hdf5','w')
        f.create_dataset('H',data=H) 
        f.create_dataset('S',data=S) 
        f.create_dataset('E',data=np.array([self.E])) 
        f.create_dataset('g',data=self.g) 
        if deltas is not None:
            f.create_dataset('deltas',data=deltas) 
        f.close()
    def _extract_hmean(self):
        hmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.hsum,hmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        self.hmean = hmean/self.n
    def _extract_Hcov(self):
        if RANK==0:
            hhsum = np.zeros((self.nparam,)*2) 
        else:
            if self.sampler.exact:
                hhsum = np.einsum('s,si,sj->ij',self.f,self.h,self.h)
            else:
                hhsum = np.dot(self.h.T,self.h)
        hhmean = np.zeros_like(hhsum)
        COMM.Reduce(hhsum,hhmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        return hhmean/self.n-np.outer(self.hmean,self.hmean)
    def extract_H(self,solve_full,solve_dense):
        fxn = self._get_Hmatrix if solve_dense else self._get_H_iterative
        self.Hx1 = np.zeros(self.nparam,dtype=self.dtype_i)
        self.Hx1h = np.zeros(self.nparam,dtype=self.dtype_i)
        if solve_full:
            self.H = fxn() 
        else:
            self.H = [None] * len(self.sampler.af.block_dict)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                self.H[ix] = fxn(start=start,stop=stop)
    def _get_Hmatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        t0 = time.time()
        if RANK==0:
            sh = stop-start
            vhsum = np.zeros((sh,)*2,dtype=self.dtype_i)
        else:
            v = self.v[:,start:stop] 
            h = self.h[:,start:stop] 
            if self.sampler.exact:
                vhsum = np.einsum('s,si,sj->ij',self.f,v,h)
            else:
                vhsum = np.dot(v.T,h)
        vhmean = np.zeros_like(vhsum)
        COMM.Reduce(vhsum,vhmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        hmean = self.hmean[start:stop]
        vmean = self.vmean[start:stop]
        g = self.g[start:stop]
        self.vhmean = vhmean / self.n
        print('\tcollect H matrix time=',time.time()-t0)
        H = self.vhmean - np.outer(vmean,hmean) - np.outer(g,vmean)
        return H
    def _get_H_iterative(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop 
        if RANK==0:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                Hx1 = np.zeros_like(self.Hx1[start:stop])
                COMM.Reduce(Hx1,self.Hx1[start:stop],op=MPI.SUM,root=0)     
                Hx = self.Hx1[start:stop] / self.n

                Hx -= self.vmean[start:stop] * np.dot(self.hmean[start:stop],x)
                Hx -= self.g[start:stop] * np.dot(self.vmean[start:stop],x)
                return Hx
            def matvecH(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                Hx1h = np.zeros_like(self.Hx1h[start:stop])
                COMM.Reduce(Hx1h,self.Hx1h[start:stop],op=MPI.SUM,root=0)     
                Hxh = self.Hx1h[start:stop] / self.n

                Hxh -= self.hmean[start:stop] * np.dot(self.vmean[start:stop],x)
                Hxh -= self.vmean[start:stop] * np.dot(self.g[start:stop],x)
                return Hxh
        else:
            def matvec(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                if self.sampler.exact:
                    Hx1 = np.dot(self.f * np.dot(self.h[:,start:stop],x),self.v[:,start:stop])
                else:
                    Hx1 = np.dot(np.dot(self.h[:,start:stop],x),self.v[:,start:stop])
                COMM.Reduce(Hx1,self.Hx1[start:stop],op=MPI.SUM,root=0)     
                return 0
            def matvecH(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                if self.sampler.exact:
                    Hx1h = np.dot(self.f * np.dot(self.v[:,start:stop],x),self.h[:,start:stop])
                else:
                    Hx1h = np.dot(np.dot(self.v[:,start:stop],x),self.h[:,start:stop])
                COMM.Reduce(Hx1h,self.Hx1h[start:stop],op=MPI.SUM,root=0)     
                return 0 
        return matvec,matvecH
    def transform_gradients(self):
        return self._transform_gradients_rgn(self.solve_full,self.solve_dense)
    def _transform_gradients_rgn(self,solve_full,solve_dense,sr=None,enforce_pos=True):
        if sr is None:
            deltas_sr = self._transform_gradients_sr(True,False)
            xnew_sr = self.update(self.rate1*deltas_sr) if RANK==0 else np.zeros_like(deltas_sr)
        else:
            xnew_sr,deltas_sr = sr

        self.extract_S(solve_full,solve_dense)
        self.extract_H(solve_full,solve_dense)
        if solve_dense:
            deltas_rgn,dEm = self._transform_gradients_rgn_dense(solve_full,enforce_pos)
        else:
            x0 = deltas_sr * [0,self.rate1,self.rate2,1][self.guess] 
            deltas_rgn,dEm = self._transform_gradients_rgn_iterative(solve_full,x0)
        if RANK==0:
            print('SR delta norm=',np.linalg.norm(deltas_sr)*self.rate1)
            print('RGN delta norm=',np.linalg.norm(deltas_rgn))

        #solve_dense = True
        #self.extract_S(solve_full,solve_dense)
        #self.extract_H(solve_full,solve_dense)
        #deltas_rgn_dense,dEm = self._transform_gradients_rgn_dense(solve_full,enforce_pos)
        #solve_dense = False 
        #self.extract_S(solve_full,solve_dense)
        #self.extract_H(solve_full,solve_dense)
        #x0 = deltas_sr * [0,self.rate1,self.rate2,1][self.guess] 
        #deltas_rgn,dEm = self._transform_gradients_rgn_iterative(solve_full,x0)
        #if RANK==0:
        #    #print(deltas_rgn)
        #    #print(deltas_rgn_dense)
        #    print('rgn delta error=',np.linalg.norm(deltas_rgn-deltas_rgn_dense))
        #exit()

        rate = self.rate2 if self.pure_newton else 1.
        if self.check is None:
            xnew_rgn = self.update(rate*deltas_rgn) if RANK==0 else np.zeros_like(deltas_rgn)
            return xnew_rgn

        if RANK==0:
            g,E,Eerr = self.g,self.E,self.Eerr
            Enew = [None] * len(self.check)
            Eerr_new = [None] * len(self.check)
        config = self.sampler.config
        xnew_rgn = [None] * len(self.check)
        for ix,scale in enumerate(self.check): 
            deltas = deltas_rgn * rate * scale
            xnew_rgn[ix] = self.update(deltas) if RANK==0 else np.zeros_like(deltas)
            COMM.Bcast(xnew_rgn[ix],root=0) 
            self.sampler.af.update(xnew_rgn[ix])
            self.sampler.config = config

            self.free_quantities()
            self.sample(samplesize=self.batchsize_small,save_local=False,compute_v=False,compute_h=False,save_config=False)
            self.extract_energy()
            if RANK==0:
                Enew[ix] = self.E
                Eerr_new[ix] = self.Eerr
        self.sampler.config = config
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)
        self.g,self.E,self.Eerr = g,E,Eerr
        dE = np.array(Enew) - E
        Eerr_new = np.array(Eerr_new)
        print(f'\tpredict={dEm/self.nsite},actual={(dE/self.nsite,Eerr_new/self.nsite)}')
        idx = np.argmin(dE) 
        if dE[idx]<0:
            return xnew_rgn[idx]
        else:
            return xnew_sr
    def _solve_dense(self,H,S,g):
        print('solve symmetric=',self.solve_symmetric)
        hess = H - self.E * S
        print('hess norm=',np.linalg.norm(hess))
        print('ovlp norm=',np.linalg.norm(S))
        A = hess.copy()
        b = g.copy()
        if self.solve_symmetric==1:
            if not self.pure_newton:
                A += S/self.rate2 
            A += self.cond2 * np.eye(len(g))
            w = np.linalg.eigvals(A)
            wmin = min(w.real)
            wmax = max(w.real)
            print('min,max eigval=',wmin,wmax) 
            M = A
            b = np.dot(A.T,g)
            A = np.dot(A.T,A) 
        else: 
            if self.solve_symmetric==2:
                A += A.T
                A /= 2
            else:
                pass
            if not self.pure_newton:
                A = hess + S/self.rate2 
            w = np.linalg.eigvals(A)
            wmin = min(w.real)
            wmax = max(w.real)
            print('min,max eigval=',wmin,wmax) 
            A += (self.cond2 - wmin) * np.eye(len(g))
            #w = np.linalg.eigvals(A)
            #wmin = min(w.real)
            #wmax = max(w.real)
            #print('min,max eigval=',wmin,wmax) 
            M = A 
        deltas = np.linalg.solve(A,b)
        dE = - np.dot(deltas,g) + .5 * np.dot(deltas,np.dot(M,deltas)) 
        return deltas,dE
    def _transform_gradients_rgn_dense(self,solve_full,enforce_pos):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i),0. 
        t0 = time.time()
        if solve_full:
            H,S = [self.H],[self.S]
            blk_dict = [(0,self.nparam)]
        else:
            H,S = self.H,self.S
            blk_dict = self.sampler.af.block_dict
        dE = np.zeros(len(blk_dict))  
        deltas = np.empty(self.nparam,dtype=self.dtype_i)
        for ix,(start,stop) in enumerate(blk_dict):
            deltas[start:stop],dE[ix] = self._solve_dense(H[ix],S[ix],self.g[start:stop])
        dE = np.sum(dE)
        if self.save_grad_hess:
            self._save_grad_hess(deltas=self.deltas)
        return deltas,dE
    def _transform_gradients_rgn_iterative(self,solve_full,x0):
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
        E = self.E if RANK==0 else 0
        self.terminate = np.array([0])
        if RANK==0:
            print('pure_newton=',self.pure_newton)
        if solve_full:
            H,S = [self.H],[self.S]
            blk_dict = [(0,self.nparam)]
        else:
            H,S = self.H,self.S
            blk_dict = self.sampler.af.block_dict
        def get_A(ix1,ix2): 
            def A(x):
                if self.terminate[0]==1:
                    return 0
                Hx = H[ix1][ix2](x)
                if self.terminate[0]==1:
                    return 0
                Sx = S[ix1](x)
                if self.terminate[0]==1:
                    return 0
                if self.pure_newton:
                    return Hx - E * Sx + self.cond2 * x 
                else:
                    return Hx + (1./self.rate2 - E) * Sx + self.cond2 * x
            return A
        def get_Ab(ix):
            A0 = get_A(ix,0)
            A1 = get_A(ix,1)
            if self.solve_symmetric==0:
               return A0,g
            if self.solve_symmetric==1:
               b = A1(g)
               if RANK>0:
                   b = g
               def A(x):
                    if self.terminate[0]==1:
                        return 0
                    Ax = A0(x)
                    if self.terminate[0]==1:
                        return 0
                    AAx = A1(Ax)
                    return AAx
               return A,b
            else:
               def A(x):
                    if self.terminate[0]==1:
                        return 0
                    A0x = A0(x)
                    if self.terminate[0]==1:
                        return 0
                    A1x = A1(x)
                    return (A0x+A1x)/2
               return A,g

        dE = 0.
        hess_err = 0.
        deltas = np.empty(self.nparam,dtype=self.dtype_i)
        symm = (self.solve_symmetric>0)
        for ix,(start,stop) in enumerate(blk_dict):
            if RANK==0:
                print(f'ix={ix},sh={stop-start}')
            x0_ = None if x0 is None else x0[start:stop]
            A,b = get_Ab(ix)
            deltas_ = self.solve_iterative(A,b,symm,x0=x0_)
            deltas[start:stop] = deltas_
            self.terminate[0] = 0
            A = get_A(ix,0)
            hessp = A(deltas_)
            if RANK==0:
                hess_err += np.linalg.norm(b-hessp)
                dE += np.dot(hessp,deltas_)
        if RANK==0:
            print('hessian inversion error=',hess_err)
            return deltas,- np.dot(self.g,deltas) + .5 * dE
        else:
            return np.zeros(self.nparam,dtype=self.dtype_i),0. 
class lBFGS(SR):
    def __init__(self,sampler,npairs=(5,50),gamma_method=1,**kwargs):
        super().__init__(sampler,**kwargs) 
        if RANK>0:
            return
        self.npairs = npairs
        self.gamma_method = gamma_method
        self.xdiff = []
        self.gdiff = []
        self.g = None
        self.free_g = False 
    def extract_gradient(self):
        vmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.vsum,vmean,op=MPI.SUM,root=0)
        evmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.evsum,evmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        vmean /= self.n
        evmean /= self.n
        g = (evmean - self.E.conj() * vmean).real
        self.vmean = vmean
        if self.g is not None:
            self.gdiff.append(g - self.g)
            if len(self.gdiff)>self.npairs[1]:
                self.gdiff.pop(0) 
        self.g = g 
    def transform_gradients(self):
        method = np.array([0])
        if RANK==0:
            assert len(self.xdiff)==len(self.gdiff)
            if len(self.xdiff)>=self.npairs[0]:
                method[0] = 1
        COMM.Bcast(method,root=0)

        if method[0] == 0:
            self._transform_gradients_sr(self.solve_full,self.solve_dense)
            self.deltas *= self.rate1
        else:
            self._transform_gradients_lbfgs(self.solve_full,self.solve_dense)
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)
        self.xdiff.append(-self.deltas)
        if len(self.xdiff)>self.npairs[1]:
            self.xdiff.pop(0)
        x = self.sampler.af.get_x()
        return x - self.deltas 
    def _transform_gradients_lbfgs(self,solve_full,solve_dense):
        self.extract_S(solve_full,solve_dense)
        if solve_dense:
            raise NotImplementedError
            return self._transform_gradients_sr_dense(solve_full)
        else:
            return self._transform_gradients_lbfgs_iterative(solve_full)
    def _transform_gradients_lbfgs_iterative(self,solve_full):
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
        if RANK==0:
            gamma = self.compute_gamma() 
            print('gamma=',gamma)
            S = np.array(self.xdiff).T
            Y = np.array(self.gdiff).T
            SY = np.dot(S.T,Y) 
            L = np.tril(SY,k=-1)
            D = np.diag(np.diag(SY))
            M = np.linalg.inv(np.block([[-np.dot(S.T,S)*gamma,-L],[-L.T,D]]))
            Phi = np.concatenate([gamma * S,Y],axis=1)
        if solve_full: 
            def R(x):
                Rx = self.S(x) + self.cond1 * x
                if RANK>0:
                    return Rx
                Bs = gamma * x + np.dot(Phi,np.dot(M,np.dot(Phi.T,x)))
                return Bs + Rx / self.rate1 
            self.deltas = self.solve_iterative(R,g,True,x0=g)
        else:
            raise NotImplementedError
            self.deltas = np.empty(self.nparam,dtype=self.dtype_i)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                def R(x):
                    return self.S[ix](x) + self.cond1 * x
                self.deltas[strt:stop] = self.solve_iterative(R,g[start:stop],True,x0=g[start:stop])
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)  
        return self.update(self.rate1)
    def compute_gamma(self):
        if self.gamma_method==1:
            sk = self.xdiff[-1]
            yk = self.gdiff[-1]
        return max(np.dot(yk,yk)/np.dot(sk,yk),1)
class TrustRegion:
    def minimize_iterative(self,update,_apply,x0):
        self.terminate = np.array([0])
        x,dEm = x0,0
        if RANK==0:
            t0 = time.time()
            x,dEm,info = _minimize(update,x0,maxiter=self.maxiter)
            self.terminate[0] = 1
            COMM.Bcast(self.terminate,root=0)
            print(f'\tsolver time={time.time()-t0},exit status={info}')
        else:
            nit = 0
            while self.terminate[0]==0:
                nit += 1
                _apply(x)
            if RANK==1:
                print('niter=',nit)
        return x,dEm
class TrustRegionSR(SR,TrustRegion):
    def transform_gradients(self):
        self.extract_S(self.solve_full,self.solve_dense)
        if RANK==0:
            if self.Eold is not None and self.dEm is not None:
                dE = self.E - self.Eold
                ratio = dE/self.dEm
                print(f'\tdE={dE},dEm={self.dEm}')
                if ratio > 1:
                    self.trust_sz *= self.trust_scale[0] 
                elif ratio < .5:
                    self.trust_sz *= self.trust_scale[1]
            
        if self.solve_dense:
            self.dEm = self._transform_gradients_trust_dense()
        else:
            self.dEm = self._transform_gradients_trust_iterative()
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)
        print('\tnorm after=', np.sum(self.deltas**2))
        return self.update(1)
    def _transform_gradients_trust_dense(self):
        if RANK>0:
            return 0 
        print('\ttrust size=',self.trust_sz)
        x0 = - self.g * self.rate1 
        print('\tnorm before=', np.sum(x0**2))
        R = self.S + self.cond1 * np.eye(self.nparam)
        angle = np.dot(x0,np.dot(R,x0))
        print('\tangle before=', angle)
        if angle < self.trust_sz:
            x0 *= np.sqrt(self.trust_sz/angle) * 1.1
        g = self.g 
        def update(x):
            x,mu = x[:-1],x[-1]
            Rx = np.dot(R,x)
            c = np.dot(x,Rx)/2 - self.trust_sz/2

            fk = np.dot(g,x) + mu * c 
            
            Lx = g + mu*Rx
            gk = np.concatenate([Lx,c*np.ones(1)])

            Lxx = mu * R
            Lmx = Rx
            Hk = np.block([[Lxx,Lmx.reshape(self.nparam,1)],
                 [Lmx.reshape(1,self.nparam),np.zeros((1,1))]]) 
            pk = np.linalg.solve(Hk,gk)
            return fk,gk,-pk,0
        
        x0 = np.concatenate([x0,np.ones(1)*self.nsite])
        x,dEm,status = _minimize(update,x0,maxiter=self.maxiter,iprint=self.iprint)
        print('\tstatus=',status)
        self.deltas = -x[:-1] 
        return dEm 
class TrustRegionRGN(RGN,TrustRegion):
    def transform_gradients(self):
        xnew_sr = self._transform_gradients_sr(True,False)
        deltas_sr = self.deltas
        
        if self.trust_init=='rgn':
            check = self.check
            self.check = None
            self._transform_gradients_rgn(self.solve_full,self.solve_dense,(xnew_sr,deltas_sr),enforce_pos=False)
            self.check = check
        else:
            self.extract_S(self.solve_full,self.solve_dense)
            self.extract_H(self.solve_full,self.solve_dense)
            if self.trust_init=='sr': 
                self.deltas *= self.rate1
            else:
                self.deltas = self.g * self.rate1

        if RANK==0:
            print('\ttrust size=',self.trust_sz)
            print('\tnorm before=', np.sum(self.deltas**2))
        if self.solve_dense:
            dEm = self._transform_gradients_trust_dense(-self.deltas)
        else:
            dEm = self._transform_gradients_trust_iterative(-self.deltas)
        if RANK==0:
            print('\tnorm after=', np.sum(self.deltas**2))
        xnew_trust = self.update(1) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
        deltas_trust = self.deltas

        if RANK==0:
            g,E,Eerr = self.g,self.E,self.Eerr
        config = self.sampler.config
        COMM.Bcast(xnew_trust,root=0) 
        self.sampler.af.update(xnew_trust)
        self.free_quantities()
        self.sample(samplesize=self.batchsize_small,save_local=False,compute_v=False,compute_h=False,save_config=False)
        self.extract_energy()
        self.sampler.config = config
        if RANK>0:
            return xnew_trust 
        Enew,Eerr_new = self.E,self.Eerr
        dE = Enew - E 
        print(f'\tpredict={dEm},actual={(dE,Eerr_new)}')
        self.g,self.E,self.Eerr = g,E,Eerr
        if dE<0:
            ratio = dE/dEm
            if ratio > 1:
                self.trust_sz *= self.trust_scale[0] 
            elif ratio < .5:
                self.trust_sz *= self.trust_scale[1]
            self.deltas = deltas_trust
            return xnew_trust
        else:
            self.trust_sz *= self.trust_scale[1]
            self.deltas = deltas_sr
            return xnew_sr
    def _transform_gradients_trust_dense(self,x0):
        if RANK>0:
            return 0 
        R = self.S + self.cond1 * np.eye(self.nparam)
        angle = np.dot(x0,np.dot(R,x0))
        print('\tangle before=', angle)
        g = self.g 
        H = (self.H - self.E * self.S) 
        H = (H + H.T) / 2
        def update(x):
            x,mu = x[:-1],x[-1]
            Hx = np.dot(H,x)
            Rx = np.dot(R,x)
            c = np.dot(x,Rx)/2 - self.trust_sz/2

            fk = np.dot(g,x)+np.dot(x,Hx)/2 + mu * c 
            
            Lx = g + Hx + mu*Rx
            gk = np.concatenate([Lx,c*np.ones(1)])

            Lxx = H + mu * R
            Lmx = Rx
            Hk = np.block([[Lxx,Lmx.reshape(self.nparam,1)],
                 [Lmx.reshape(1,self.nparam),np.zeros((1,1))]]) 
            pk = np.linalg.solve(Hk,gk)
            return fk,gk,-pk,0
        
        x0 = np.concatenate([x0,np.ones(1)*self.nsite])
        x,dEm,status = _minimize(update,x0,maxiter=self.maxiter,iprint=self.iprint)
        print('\tstatus=',status)
        self.deltas = -x[:-1] 
        return dEm 
    def _transform_gradients_trust_iterative(self,x0):
        Rx0 = self.S(x0) + self.cond1 * x0
        data = np.array([np.dot(x0,Rx0),self.trust_sz])
        COMM.Bcast(data,root=0)
        angle,self.trust_sz = data
        if RANK==0:
            print('\tangle before=', angle)
        g = self.g / self.nsite if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
        E = self.E if RANK==0 else 0
        sh = self.nparam + 1
        def _apply(x):
            if self.terminate[0]==1:
                return 0
            x,mu = x[:-1],x[-1]
            Hx = self.H(x)
            if self.terminate[0]==1:
                return 0
            Sx = self.S(x)
            if self.terminate[0]==1:
                return 0
            return x,mu,Hx,Sx
        def update(x):
            info = _apply(x)  
            if info == 0:
                return 0 
            x,mu,Hx,Sx = info

            Hx = (Hx - E * Sx) / self.nsite
            Rx = Sx + self.cond1 * x
            c = np.dot(x,Rx)/2 - self.trust_sz/2

            fk = np.dot(g,x)+np.dot(x,HRx)/2-mu*self.trust_sz/2

            Lx = g + Hx+mu*Rx
            gk = np.concatenate([Lx,c*np.ones(1)])

            def hessp(p):
                info = _apply(p)
                if info == 0:
                    return 0
                px,pm,Hpx,Spx = info 

                Hpx = (Hpx - E * Spx) / self.nsite
                Rpx = Spx + self.cond1 * px

                Lpx = Hpx + mu * Rpx + Rx * pmu 
                Lpm = np.dot(Rx,px)
                return np.concatenate([Lpx,Lpm*np.ones(1)]) 
            LinOp = spla.LinearOperator((sh,sh),matvec=hessp,dtype=gk.dtype)
            pk,info = self.solver(LinOp,gk,x0=gk,tol=self.tol,maxiter=self.maxiter)
            return fk,gk,-pk,info 
        x0 = np.concatenate([x0,np.ones(1)])
        x,dEm = self.minimize_iterative(update,_apply,x0)
        self.deltas = -x[:-1] 
        return dEm * self.nsite
class BFGS(TrustRegionSR):
    def __init__(self,sampler,**kwargs):
        super().__init__(sampler,solve_full=True,solve_dense=True,**kwargs)
        self.Bk = None 
        self.gk = None
        self.sk = None
        self.dEm = None
    def load(self,step):
        f = h5py.File(self.tmpdir+f'step{self.step}.hdf5','r')
        self.Bk = f['Bk'][:]
        self.gk = f['gk'][:]
        self.sk = f['sk'][:]
        f.close()
    def transform_gradients(self):
        if self.Bk is None:
            x = self._transform_gradients_sr(True,True)
            if RANK==0:
                self.deltas *= self.rate1
                self.Bk = self.S + self.cond1 * np.eye(self.nparam)
        else:
            if RANK==0:
                if self.gk is not None:
                    yk = self.g - self.gk
                    ny = np.linalg.norm(yk)
                    yk /= ny
                    denom1 = np.dot(yk,self.sk) / ny
                    print('\tdenom1=',denom1)
                    if np.fabs(denom1)<self.denom_thresh[0]:
                        denom1 = self.denom_thresh[0]
                    yy = np.outer(yk,yk) / denom1 

                    ns = np.linalg.norm(self.sk)
                    self.sk /= ns
                    Bs = np.dot(self.Bk,self.sk)
                    denom2 = np.dot(self.sk,Bs)
                    print(f'\tdenom2=',denom2)
                    if np.fabs(denom2)<self.denom_thresh[1]:
                        denom2 = self.denom_thresh[1]
                    ss = np.outer(Bs,Bs) / denom2 
                    self.Bk += yy - ss
            x = super().transform_gradients()
        if RANK==0:
            self.sk = - self.deltas
            self.gk = self.g
            f = h5py.File(self.tmpdir+f'step{self.step}.hdf5','a')
            f.create_dataset('Bk',data=self.Bk) 
            f.create_dataset('sk',data=self.sk) 
            f.create_dataset('gk',data=self.gk) 
            f.close()
        return x
    def _transform_gradients_trust_dense(self):
        if RANK>0:
            return 0 
        print('\ttrust size=',self.trust_sz)
        x0 = - self.g * self.rate1
        R = self.S + self.cond1 * np.eye(self.nparam)
        angle = np.dot(x0,np.dot(R,x0))
        print('\tangle before=', angle)
        if angle < self.trust_sz:
            x0 *= np.sqrt(self.trust_sz/angle) * 1.1
        g = self.g 
        H = self.Bk
        def update(x):
            x,mu = x[:-1],x[-1]
            Hx = np.dot(H,x)
            Rx = np.dot(R,x)
            c = np.dot(x,Rx)/2 - self.trust_sz/2

            fk = np.dot(g,x)+np.dot(x,Hx)/2 + mu * c 
            
            Lx = g + Hx + mu*Rx
            gk = np.concatenate([Lx,c*np.ones(1)])

            Lxx = H + mu * R
            Lmx = Rx
            Hk = np.block([[Lxx,Lmx.reshape(self.nparam,1)],
                 [Lmx.reshape(1,self.nparam),np.zeros((1,1))]]) 
            pk = np.linalg.solve(Hk,gk)
            return fk,gk,-pk,0
        
        x0 = np.concatenate([x0,np.ones(1)*self.nsite])
        x,dEm,status = _minimize(update,x0,maxiter=self.maxiter)
        print('\tstatus=',status)
        self.deltas = -x[:-1] 
        return dEm 
class LinearMethod(RGN):
    def __init__(self,sampler,xi=.5,**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'lin' 
        self.xi = xi
    def _transform_gradients_lin(self,solve_dense=None,solve_full=None):
        solve_full = self.solve_full if solve_full is None else solve_full
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        self.extract_S(solve_full=solve_full,solve_dense=solve_dense)
        self.extract_H(solve_full=solve_full,solve_dense=solve_dense)
        if solve_dense:
            dE = self._transform_gradients_lin_dense(solve_full=solve_full)
        else:
            raise NotImplementedError
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
            return 0. 
        self.deltas = np.zeros(self.nparam)
        solve_full = self.solve_full if solve_full is None else solve_full
        
        t0 = time.time()
        if solve_full:
            w,self.deltas,v0,inorm = \
                _lin_block_solve(self.H,self.E,self.S,self.g,self.hmean,self.vmean,self.cond2) 
        else:
            w = np.zeros(self.nsite)
            v0 = np.zeros(self.nsite)
            inorm = np.zeros(self.nsite)
            self.deltas = np.zeros_like(self.x)
            for ix,(start,stop) in enumerate(self.block_dict):
                w[ix],self.deltas[start:stop],v0[ix],inorm[ix] = \
                    _lin_block_solve(self.H[ix],self.E,self.S[ix],self.g[start:stop],
                                     self.hmean[start:stop],self.vmean[start:stop],self.cond2) 
            inorm = inorm.sum()
            w = w.sum()
        print(f'\tLIN solver time={time.time()-t0},inorm={inorm},eigenvalue={w},scale1={v0}')
        self._scale_eigenvector()
        if self.save_grad_hess:
            Hi0 = self.g
            H0j = self.hmean - self.E * self.vmean
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
        H0j = self.hmean - self.E * self.vmean
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
            w,v = spla.eigs(A,k=1,M=B,sigma=self.E,v0=x0,tol=self.tol)
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
##############################################################################################
# sampler
#############################################################################################
import itertools
class DenseSampler:
    def __init__(self,nsite,nspin,exact=False,seed=None,thresh=1e-28,fix_sector=True):
        self.nsite = nsite 
        self.nspin = nspin

        self.all_configs = self.get_all_configs(fix_sector=fix_sector)
        self.ntotal = len(self.all_configs)
        if RANK==0:
            print('ntotal configs=',self.ntotal)
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
        self.af = None
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
            p = self.af.log_prob(self.af.parse_config(config))
            p = 0 if p is None else np.exp(p) 
            plocal.append(p)
        plocal = np.array(plocal)
        #print(RANK,plocal)
        #exit()
         
        COMM.Allgatherv(plocal,[ptotal,self.count,self.disp,MPI.DOUBLE])
        nonzeros = []
        for ix,px in enumerate(ptotal):
            if px > self.thresh:
                nonzeros.append(ix) 
        n = np.sum(ptotal)
        ptotal /= n 
        self.p = ptotal

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
        if RANK==SIZE-1:
            print('\tdense amplitude time=',time.time()-t0)
            print('\ttotal non-zero amplitudes=',ntotal)
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
    def _burn_in(self,config=None,burn_in=None,progbar=False,exclude_root=True):
        if config is not None:
            self.config = config 
        self.px = self.af.log_prob(self.af.parse_config(self.config))

        if exclude_root and RANK==0:
            print('\tlog prob=',self.px)
            return 
        t0 = time.time()
        burn_in = self.burn_in if burn_in is None else burn_in
        pg = None
        if progbar and RANK==SIZE-1:
            pg = Progbar(total=burn_in)
        for n in range(burn_in):
            self.config,self.omega = self.sample()
            if pg is not None:
                pg.update()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
    def propose_new_pair(self,i1,i2):
        return i2,i1
    def _new_pair(self,site1,site2):
        ix1,ix2 = self.flatten(site1),self.flatten(site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if i1==i2: # continue
            return (None,) * 2
        i1_new,i2_new = self.propose_new_pair(i1,i2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        return (i1_new,i2_new),tuple(config_new)
    def sample(self):
        if self.scheme=='random':
            self._sample_random()
        elif self.af.deterministic:
            self._sample_deterministic()
        else:
            self._sample()
        return self.config,self.px
    def _sample_random(self):
        for i in range(self.npair):
            ix = self.rng.integers(0,high=self.npair)
            self._update_pair(*self.pairs[ix])
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
def scale_wfn(psi,scale=1.):
    for tid in psi.tensor_map:
        tsr = psi.tensor_map[tid]
        s = np.amax(np.fabs(tsr.data))/scale
        tsr.modify(data=tsr.data/s)
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
    if len(cx)==0:
        return 0.,0.
    cx = np.array(list(cx.values())).real
    max_,min_,mean_ = np.amax(cx),np.amin(cx),np.mean(cx)
    return mean_,np.fabs((max_-min_)/mean_)
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
        else:
            if data.size==1:
                data = data.reshape(-1)[0] 
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
def tensor_grad(tsr,set_zero=True):
    grad = tsr.grad
    if set_zero:
        tsr.grad = None
    return grad 
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
    def parse_config(self,config):
        return config
    def intermediate_sign(self,config=None,ix1=None,ix2=None):
        return 1.
    def config_sign(self,config=None):
        return 1.
##### wfn methods #####
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        self._backend = backend
        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
        ar.set_backend(tsr)
        for site in self.sites:
            tag = self.site_tag(site)
            self.psi[tag].modify(data=self.tensor2backend(self.psi[tag].data,backend=backend,requires_grad=requires_grad))
        for key in self.data_map:
            self.data_map[key] = self.tensor2backend(self.data_map[key],backend=backend,requires_grad=False)
        if self.from_plq:
            self.model.gate2backend(backend)
    def get_site_map(self):
        site_order = []
        for blk in self.blks:
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
    def get_block_dict(self):
        start = 0
        self.block_dict = [None] * len(self.blks)
        for bix,blk in enumerate(self.blks):
            site_min,site_max = blk[0],blk[-1]
            ix_min,ix_max = self.site_map[site_min],self.site_map[site_max]
            stop = start
            for ix in range(ix_min,ix_max+1):
                _,size,_ = self.constructors[ix]
                stop += size
            self.block_dict[bix] = start,stop
            start = stop
        self.nparam = stop
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
    def set_psi(self,psi):
        if self.normalize is not None:
            psi = scale_wfn(psi,scale=self.normalize)
        self.psi = psi
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
        if cis is None:
            return tn
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
##### contraction & derivative #####
    def amplitude(self,config,sign=True,to_numpy=True,i=None,direction='row'):
        cx = self.unsigned_amplitude(config,to_numpy=to_numpy,i=i,direction=direction)
        if cx is None:
            return None 
        if sign:
            cx *= self.config_sign(config)
        return cx
    def log_prob(self, config,i=None):
        """Calculate the probability of a configuration.
        """
        cx = self.unsigned_amplitude(config,i=i)
        if cx is None:
            return None
        return np.log(cx ** 2)
    def _new_amp_from_plq(self,plq,sites,config_sites,config_new):
        if plq is None:
            return None,0
        plq_new = self.replace_sites(plq.copy(),sites,config_sites) 
        cy = safe_contract(plq_new)
        if cy is None:
            return plq_new,None
        return plq_new,tensor2backend(cy,'numpy')
    def _new_log_prob_from_plq(self,plq,sites,config_sites,config_new):
        plq_new,cx = self._new_amp_from_plq(plq,sites,config_sites,config_new)
        if cx is None:
            return plq_new,None
        return plq_new,np.log(cx**2)
    def tensor_grad(self,tsr,set_zero=True):
        return tensor_grad(tsr,set_zero=set_zero)
    def extract_ad_grad(self):
        vx = {site:self.tensor_grad(self.psi[self.site_tag(site)].data) for site in self.sites}
        return self.dict2vec(vx)
    def propagate(self,ex):
        if not isinstance(ex,torch.Tensor):
            return 0.,np.zeros(self.nparam)
        ex.backward()
        hx = self.extract_ad_grad()
        return tensor2backend(ex,'numpy'),hx 
    def get_grad_deterministic(self,config):
        self.wfn2backend(backend='torch',requires_grad=True)
        cx = self.amplitude(config,to_numpy=False)
        if cx is None:
            return cx,np.zeros(self.nparam)
        cx,vx = self.propagate(cx)
        vx /= cx
        self.wfn2backend()
        self.free_ad_cache()
        return cx,vx
    def get_grad_from_plq(self,plq):
        if plq is None:
            return 
        for plq_key,tn in plq.items():
            cij = self.cx[plq_key]
            sites = self.plq_sites(plq_key)
            for site in sites:
                if site in self.vx:
                    continue
                try:
                    self.vx[site] = self.tensor2backend(self.site_grad(tn.copy(),site)/cij,'numpy')
                except (ValueError,IndexError):
                    continue 
        return self.vx
##### ham methods #####
    def _add_gate(self,tn,gate,order,where):
        return _add_gate(tn,gate,order,where,self.site_ind,self.site_tag,contract=True)
    def update_pair_energy_from_plq(self,tn,where):
        ix1,ix2 = [self.flatten(site) for site in where]
        i1,i2 = self.config[ix1],self.config[ix2] 
        if not self.model.pair_valid(i1,i2):
            return {tag:0 for tag in self.model.gate}
        coeff = self.model.pair_coeff(*where)
        ex = dict()
        for tag,(gate,order) in self.model.gate.items():
            ex[tag] = self._add_gate(tn.copy(),gate,order,where) 
            if ex[tag] is None:
                ex[tag] = 0
            ex[tag] *= coeff 
        return ex 
    def update_pair_energy_from_benvs(self,where,**kwargs):
        ix1,ix2 = [self.flatten(site) for site in where]
        assert ix1<ix2
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.model.pair_valid(i1,i2): # term vanishes 
            if self.dmc:
                return dict()
            else:
                return {tag:0 for tag in self.model.gate}
        coeff_comm = self.intermediate_sign(self.config,ix1,ix2) * self.model.pair_coeff(*where)
        ex = dict()
        for i1_new,i2_new,coeff,tag in self.model.pair_terms(i1,i2):
            config_new = list(self.config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            key = config_new if self.dmc else tag
            config_new = self.parse_config(config_new) 
            ex[key] = self.amplitude(config_new,to_numpy=False,**kwargs) 
            if ex[key] is None:
                ex[key] = 0
            ex[key] *= coeff_comm * coeff 
        return ex 
    def amplitude2scalar(self):
        self.cx = {key:tensor2backend(cij,'numpy') for key,cij in self.cx.items()}
    def parse_energy(self,ex,batch_key,ratio):
        if ratio:
            return tensor2backend(sum([eij for _,_,eij in ex.values()]),'numpy')
        else:
            return sum([eij for eij,_,_ in ex.values()])
    def batch_quantities(self,batch_key,compute_v,compute_h): 
        if compute_h:
            self.wfn2backend(backend='torch',requires_grad=True)
        ex,plq = self.batch_pair_energies(batch_key,compute_h)

        hx = 0
        if compute_h:
            _,hx = self.propagate(self.parse_energy(ex,batch_key,False))
        ex = self.parse_energy(ex,batch_key,True)
        self.amplitude2scalar()
        if compute_v:
            self.get_grad_from_plq(plq) 
        if compute_h:
            self.wfn2backend()
        self.free_ad_cache()
        return ex,hx
    def contraction_error(self):
        return contraction_error(self.cx) 
    def set_config(self,config,compute_v):
        self.config = config
        self.cx = dict()
        if self.from_plq and compute_v:
            self.vx = dict()
        return config
    def parse_gradient(self):
        vx = self.dict2vec(self.vx) 
        return vx
    def compute_local_energy(self,config,compute_v=True,compute_h=False):
        config = self.set_config(config,compute_v)
        vx = None
        if not self.from_plq:
            if compute_v:
                cx,vx = self.get_grad_deterministic(config)
            else:
                cx,vx = self.amplitude(config),None

        ex,hx = 0.,0.
        for batch_key in self.model.batched_pairs:
            ex_,hx_ = self.batch_quantities(batch_key,compute_v,compute_h)
            ex += ex_
            hx += hx_
        cx,err = self.contraction_error() 
        eu = self.model.compute_local_energy_eigen(self.config)
        ex += eu
        if self.from_plq and compute_v:
            vx = self.parse_gradient()
        if compute_h:
            hx = hx / cx + eu * vx
        self.vx = None
        self.cx = None
        return cx,ex,vx,hx,err 
#class CIAmplitudeFactory(AmplitudeFactory): # model for testing
#    def wfn2backend(self,backend=None,requires_grad=False):
#        backend = self.backend if backend is None else backend
#        self._backend = backend
#        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
#        ar.set_backend(tsr)
#        self.psi = self.tensor2backend(self.psi,backend=backend,requires_grad=requires_grad)
#    def get_x(self):
#        return self.tensor2backend(self.psi,backend='numpy')
#    def write_tn_to_disc(self,tn,fname):
#        np.save(fname+'.npy',tn)
#    def update(self,x,fname=None,root=0):
#        self.psi = self.tensor2backend(x)
#        if RANK==root:
#            if fname is not None: # save psi to disc
#                self.write_tn_to_disc(x,fname)
#        return psi
#    def amplitude(self,config,
class Model:
    def gate2backend(self,backend):
        self.gate = {tag:(tensor2backend(tsr,backend),order) for tag,(tsr,order) in self.gate.items()}
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
