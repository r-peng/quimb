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
DISCARD = 1e3
CG_TOL = 1e-4
MAXITER = 100
#MAXITER = 2
##################################################################################################
# VMC utils
##################################################################################################
def _cubic_interpolation(fun,alpha1,alpha2,alpha3,xk,pk):
    M = np.zeros((3,3))
    b = np.zeros(3)
    for i,alpha in enumerate([alpha1,alpha2,alpha3]):
        b[i] = fun(xk+alpha*pk)
        for j,p in enumerate([3,2,1]):
            M[i,j] = alpha**p
    a,b,c = np.linalg.solve(M,b)
    sq = b**2-3*a*c
    if sq<0:
        return -1 
    roots = [(-b+sign*np.sqrt(sq))/(3*a) for sign in [1,-1]] 
    if a>0:
        return roots[1]
    else:
        return roots[0]
def _3point_interpolation(fun,alpha_k,xk,pk):
    s = .5,1.,1.1
    if xk[-1]+s[-1]*alpha_k*pk[-1]<0:
        smax = -xk[-1]/pk[-1]/alpha_k
        s = np.array([.2,.5,.9])*smax
    x = [xk+si*alpha_k*pk for si in s]
    b = np.array([fun(xi) for xi in x])
    ix = np.argmin(b) 
    return alpha_k*s[ix],b[ix],x[ix]
def _minimize(update,x0,ctr_mu=True,alpha_0=.1,maxiter=None,iprint=0):
    maxiter = MAXITER if maxiter is None else maxiter
    xk = x0
    alpha_k = alpha_0
    for i in range(maxiter):
        fk,gk,pk,status = update(xk)
        gnorm = np.linalg.norm(gk)
        if iprint>0:
            print(f'iter={i},gnorm={gnorm},fval={fk},constr={gk[-1]},mu={xk[-1]}')
        dEm = fk - xk[-1] * gk[-1]
        if gnorm < CG_TOL:
            return xk,dEm,0
        if status!=0:
            return xk,dEm,3

        xkp1 = xk + alpha_k * pk
        #if ctr_mu and xkp1[-1]<0:
        #    return xk,dEm,2
        xk = xkp1
    return xk,dEm,1 
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
def __rgn_block_solve(H,E,S,g,eta,eps0,enforce_pos=True):
    sh = len(g)
    hess = H - E * S 
    eps = eps0

    w,U = np.linalg.eigh(S)
    #print(w)
    w = w[w>eps]
    U = U[:,-len(w):]
    print(len(w),np.linalg.norm(np.eye(len(w))-np.dot(U.T,U)))

    hess = np.dot(U.T,np.dot(hess,U))
    g = np.dot(U.T,g)
    R = np.diag(w)
    deltas = np.linalg.solve(hess + R/eps,g)
    dE = - np.dot(deltas,g) + .5 * np.dot(deltas,np.dot(hess + R/eps,deltas)) 

    deltas = np.dot(U,deltas) 
    wmin = w[0]
    return deltas,dE,wmin,eps
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
def _newton_block_solve(H,E,S,g,cond,eigen=True,enforce_pos=True):
    # hessian 
    hess = H - E * S
    if eigen:
        w,v = np.linalg.eig(hess)
        wmin = np.amin(w.real)
        tau = max(0.,cond-wmin) if enforce_pos else cond
        w += tau 
        deltas = np.dot(v/w.reshape(1,len(g)),np.dot(v.T.conj(),g)).real
    else:
        # smallest eigenvalue
        w = np.linalg.eigvals(hess)
        wmin = np.amin(w.real)
        tau = max(0.,cond-wmin) if enforce_pos else cond
        # solve
        deltas = np.linalg.solve(hess+tau*np.eye(len(g)),g)
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
##################################################################################################
# VMC engine 
##################################################################################################
class SGD: # stochastic sampling
    def __init__(
        self,
        sampler,
        dtype_o=float,
        normalize=False,
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
        self.init_norm = None
        if normalize:
            self.init_norm = np.linalg.norm(x)    

        # parse gradient optimizer
        self.optimizer = optimizer
        self.solve_full = solve_full
        self.solve_dense = solve_dense
        self.compute_Hv = False
        self.maxiter = MAXITER if maxiter is None else maxiter

        # to be set before run
        self.tmpdir = None
        self.batchsize = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR

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
        self.vsum = None
        self.evsum = None
        self.Hv = None
        self.Hvmean = None
        self.Hvsum = None
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
            if RANK==0:
                print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(self.deltas)},dot={np.dot(self.deltas,self.g)},x={np.linalg.norm(x)}')
            self.free_quantities()
            COMM.Bcast(x,root=0) 
            fname = self.tmpdir+f'psi{step+1}' if save_wfn else None
            psi = self.sampler.af.update(x,fname=fname,root=0)
    def sample(self,samplesize=None,save_local=None,compute_v=True,compute_Hv=None,save_config=True):
        self.sampler.preprocess()

        if self.sampler.exact:
            samplesize = len(self.sampler.nonzeros)
            save_config = False
        else:
            samplesize = self.batchsize if samplesize is None else samplesize
        compute_Hv = self.compute_Hv if compute_Hv is None else compute_Hv
        save_local = self.save_local if save_local is None else save_local 

        self.buf = np.zeros(5,dtype=self.dtype_o)
        self.terminate = np.array([0])

        self.buf[0] = RANK 
        self.buf[4] = self.step 
        if compute_v:
            self.evsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.vsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.v = []
        if compute_Hv:
            self.Hvsum = np.zeros(self.nparam,dtype=self.dtype_o)
            self.Hv = [] 

        if RANK==0:
            self._ctr(samplesize,save_config)
        else:
            if self.sampler.exact:
                self._sample_exact(compute_v,compute_Hv)
            else:
                self._sample_stochastic(compute_v,compute_Hv,save_config,save_local)
    def _ctr(self,samplesize,save_config):
        if self.progbar:
            pg = Progbar(total=samplesize)

        self.f = []
        self.e = []
        err_mean = 0.
        err_max = 0.
        ncurr = 0
        t0 = time.time()
        while True:
            COMM.Recv(self.buf,tag=0)
            step = int(self.buf[4].real+.1)
            if step>self.step:
                raise ValueError
            elif step<self.step:
                continue

            rank = int(self.buf[0].real+.1)
            self.f.append(self.buf[1]) 
            self.e.append(self.buf[2])
            err_mean += self.buf[3]
            err_max = max(err_max,self.buf[3])
            ncurr += 1
            if ncurr >= samplesize: 
                break
            if self.progbar:
                pg.update()
            COMM.Send(self.terminate,dest=rank,tag=1)
        # send termination message to all workers
        self.terminate[0] = 1
        for worker in range(1,SIZE): 
            COMM.Send(self.terminate,dest=worker,tag=1)

        if save_config:
            ls = []
            for worker in range(1,SIZE):
                config = np.zeros(self.nsite,dtype=int)
                COMM.Recv(config,source=worker,tag=2)
                ls.append(config)
            np.save(self.tmpdir+f'config{self.step}.npy',np.array(ls))
        print('\tsample time=',time.time()-t0)
        print('\tcontraction err=',err_mean / len(self.e),err_max)
        self.e = np.array(self.e)
        self.f = np.array(self.f)
    def _sample_stochastic(self,compute_v,compute_Hv,save_config,save_local):
        self.buf[1] = 1.
        c = []
        e = []
        configs = []
        while True:
            config,omega = self.sampler.sample()
            #if omega > self.omega:
            #    self.config,self.omega = config,omega
            cx,ex,vx,Hvx,err = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_Hv=compute_Hv)
            if cx is None or np.fabs(ex.real) > DISCARD:
                print(f'RANK={RANK},cx={cx},ex={ex}')
                ex = np.zeros(1)[0]
                err = 0.
                if compute_v:
                    vx = np.zeros(self.nparam,dtype=self.dtype_o)
                if compute_Hv:
                    Hvx = np.zeros(self.nparam,dtype=self.dtype_o)
            self.buf[2] = ex
            self.buf[3] = err
            if compute_v:
                self.vsum += vx
                self.evsum += vx * ex.conj()
                self.v.append(vx)
            if compute_Hv:
                self.Hvsum += Hvx
                self.Hv.append(Hvx)
            if save_local:
                c.append(cx)
                e.append(ex)
                configs.append(list(config))
            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
            if self.terminate[0]==1:
                break

        #self.sampler.config = self.config
        if compute_v:
            self.v = np.array(self.v)
        if compute_Hv:
            self.Hv = np.array(self.Hv)
        if save_config:
            config = np.array(config,dtype=int)
            COMM.Send(config,dest=0,tag=2)
        if save_local:
            f = h5py.File(f'./step{self.step}RANK{RANK}.hdf5','w')
            if compute_Hv:
                f.create_dataset('Hv',data=self.Hv)
            if compute_v:
                f.create_dataset('v',data=self.v)
            f.create_dataset('e',data=np.array(e))
            f.create_dataset('c',data=np.array(c))
            f.create_dataset('config',data=np.array(configs))
            f.close()
    def _sample_exact(self,compute_v,compute_Hv): 
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
            cx,ex,vx,Hvx,err = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_Hv=compute_Hv)
            if cx is None:
                raise ValueError
            if np.fabs(ex.real)*p[ix] > DISCARD:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self.f.append(p[ix])
            self.buf[1] = p[ix]
            self.buf[2] = ex
            self.buf[3] = err
            if compute_v:
                self.vsum += vx * p[ix]
                self.evsum += vx * ex.conj() * p[ix]
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
        if self.optimizer in ['rgn','lin','trust']:
            self._extract_Hvmean()
        if RANK==0:
            try:
                dE = 0 if self.Eold is None else self.E-self.Eold
                print(f'step={self.step},E={self.E/self.nsite},dE={dE/self.nsite},err={self.Eerr/self.nsite},gmax={np.amax(np.fabs(self.g))}')
            except TypeError:
                print('E=',self.E)
                print('Eerr=',self.Eerr)
            print('\tcollect g,Hv time=',time.time()-t0)
    def extract_energy(self):
        if RANK>0:
            return
        if self.sampler.exact:
            self.n = 1.
            self.E = np.dot(self.f,self.e)
            self.Eerr = 0.
        else:
            self.n = len(self.e)
            self.E,self.Eerr = blocking_analysis(self.e,weights=self.f)
    def extract_gradient(self):
        vmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.vsum,vmean,op=MPI.SUM,root=0)
        evmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.evsum,evmean,op=MPI.SUM,root=0)
        if RANK>0:
            return
        vmean /= self.n
        evmean /= self.n
        self.g = (evmean - self.E.conj() * vmean).real
        self.vmean = vmean
    def update(self,rate):
        x = self.sampler.af.get_x()
        return self.normalize(x - rate * self.deltas)
    def transform_gradients(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i) 
        if self.optimizer=='sgd':
            self.deltas = self.g
        elif self.optimizer=='sign':
            self.deltas = np.sign(self.g)
        elif self.optimizer=='signu':
            self.deltas = np.sign(self.g) * np.random.uniform(size=self.nparam)
        else:
            raise NotImplementedError
        return self.update(self.rate1)
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
            cx,ex,vx,Hvx,err = self.sampler.af.compute_local_energy(
                config,compute_v=True,compute_Hv=True)
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
            self.E,self.Eerr = blocking_analysis(self.e,weights=self.f)
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
    def _hess(self,solve_dense=None,mode='eig',gen=True,lin=True):
        solve_dense = self.solve_dense if solve_dense is None else solve_dense
        self.extract_S(solve_full=True,solve_dense=solve_dense)
        self.extract_H(solve_full=True,solve_dense=solve_dense)
        if solve_dense:
            return self._dense_hess(mode=mode,gen=gen,lin=lin)
        else:
            return self._iterative_hessSVD()
    def _dense_hess(self,mode='eig',gen=True,lin=True):
        if RANK>0:
            return 
        A = self.H - self.E * self.S
        #A = self.S
        gen = True if lin else gen
        if mode=='svd':
            #A = A + (self.S + self.cond1 * np.eye(self.nparam))/self.rate2
            #_,w,_ = spla.svds(A,k=1,tol=CG_TOL,maxiter=MAXITER)
            _,w,_ = np.linalg.svd(A,hermitian=True)
        elif mode=='eig':
            M = None if not gen else self.S + self.cond1 * np.eye(self.nparam)
            if lin:
                A = np.block([[np.zeros((1,1)),self.g.reshape(1,self.nparam)],
                              [self.g.reshape(self.nparam,1),A]]) 
                M = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
                              [np.zeros((self.nparam,1)),M]]) 
            #w,_ = spla.eigsh(A,k=1,M=M,tol=CG_TOL,maxiter=MAXITER)
            w,_ = scipy.linalg.eigh(A,b=M)
        else:
            raise NotImplementedError
        return w
    def _eig_iterative(self,A,M=None,symm=False):
        self.terminate = np.array([0])
        sh = self.nparam
        deltas = np.zeros(sh,dtype=self.dtype)
        w = None
        if RANK==0:
            t0 = time.time()
            A = spla.LinearOperator((sh,sh),matvec=A,dtype=self.dtype)
            if M is not None:
                M = spla.LinearOperator((sh,sh),matvec=M,dtype=self.dtype)
            eig = spla.eigsh if symm else spla.eigs 
            w,_ = eig(A,k=1,tol=CG_TOL,maxiter=MAXITER)
            self.terminate[0] = 1
            COMM.Bcast(self.terminate,root=0)
        else:
            nit = 0
            while self.terminate[0]==0:
                nit += 1
                A(deltas)
            if RANK==1:
                print('niter=',nit)
        return w 
    def _iterative_hessSVD(self,bare=False):
        E = self.E if RANK==0 else 0
        # rmatvec
        self.xH1 = np.zeros(self.nparam,dtype=self.dtype)
        if RANK==0:
            def _rmatvec(x):
                COMM.Bcast(self.terminate,root=0)
                COMM.Bcast(x,root=0)
                xH1 = np.zeros_like(self.xH1)
                COMM.Reduce(xH1,self.xH1,op=MPI.SUM,root=0)     
                return self.xH1 / self.n \
                     - np.dot(x,self.vmean) * self.Hvmean \
                     - np.dot(x,self.g) * self.vmean
        else:
            def _rmatvec(x):
                COMM.Bcast(self.terminate,root=0)
                if self.terminate[0]==1:
                    return 0 
                COMM.Bcast(x,root=0)
                if self.sampler.exact:
                    xH1 = np.dot(self.f * np.dot(self.v,x),self.Hv)
                else:
                    xH1 = np.dot(np.dot(self.v,x),self.Hv)
                COMM.Reduce(xH1,self.xH1,op=MPI.SUM,root=0)     
                return x 

        def matvec(x):
            if self.terminate[0]==1:
                return 0
            Hx = self.H(x)
            if self.terminate[0]==1:
                return 0
            Sx = self.S(x)
            if self.terminate[0]==1:
                return 0
            return Hx + (1./self.rate2 - E) * Sx + self.cond1 / self.rate2 * x
        def rmatvec(x):
            if self.terminate[0]==1:
                return 0
            xH = _rmatvec(x)
            if self.terminate[0]==1:
                return 0
            xS = self.S(x)
            if self.terminate[0]==1:
                return 0
            return xH + (1./self.rate2 - E) * xS + self.cond1 / self.rate2 * x
        def MM(x):
            return matvec(rmatvec(x))
        w = self._eig_iterative(MM,symm=True)
        if RANK==0:
            print(w,np.sqrt(w))
class SR(SGD):
    def __init__(self,sampler,eigen_thresh=None,**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'sr' 
        self.eigen_thresh = eigen_thresh 
    def extract_S(self,solve_full,solve_dense):
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
        if RANK==0:
            sh = stop-start
            vvsum_ = np.zeros((sh,)*2,dtype=self.dtype_i)
        else:
            v = self.v[:,start:stop] 
            if self.sampler.exact:
                vvsum_ = np.einsum('s,si,sj->ij',self.f,v.conj(),v)
            else:
                vvsum_ = np.dot(v.T.conj(),v)
        vvsum_ = np.ascontiguousarray(vvsum_.real)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean = self.vmean[start:stop]
            S = vvsum / self.n - np.outer(vmean.conj(),vmean).real
            print('\tcollect S matrix time=',time.time()-t0)
        return S
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
    def transform_gradients(self):
        deltas = self._transform_gradients_sr(self.solve_full,self.solve_dense)
        if RANK>0:
            return deltas
        return self.update(self.rate1)
    def _transform_gradients_sr(self,solve_full,solve_dense):
        self.extract_S(solve_full,solve_dense)
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
                self.deltas = np.linalg.solve(self.S + self.cond1 * np.eye(self.nparam),self.g)
            else:
                w,v = np.linalg.eigh(self.S)
                w = w[w>self.eigen_thresh*w[-1]]
                print(f'\tnonzero={len(w)},wmax={w[-1]}')
                v = v[:,-len(w):]
                self.deltas = np.dot(v/w.reshape(1,len(w)),np.dot(v.T,self.g)) 
        else:
            self.deltas = np.empty(self.nparam,dtype=self.dtype_i)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                S = self.S[ix] + self.cond1 * np.eye(stop-start)
                self.deltas[start:stop] = np.linalg.solve(S,self.g[start:stop])
        print('\tSR solver time=',time.time()-t0)
        if self.save_grad_hess:
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
            f.create_dataset('deltas',data=self.deltas) 
            f.close()
        return self.deltas
    def _transform_gradients_sr_iterative(self,solve_full):
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
        if solve_full: 
            def R(x):
                return self.S(x) + self.cond1 * x
            self.deltas = self.solve_iterative(R,g,True,x0=g)
        else:
            self.deltas = np.empty(self.nparam,dtype=self.dtype_i)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                def R(x):
                    return self.S[ix](x) + self.cond1 * x
                self.deltas[strt:stop] = self.solve_iterative(R,g[start:stop],True,x0=g[start:stop])
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)  
        return self.deltas
    def solve_iterative(self,A,b,symm,x0=None):
        self.terminate = np.array([0])
        deltas = np.zeros_like(b)
        sh = len(b)
        if RANK==0:
            t0 = time.time()
            LinOp = spla.LinearOperator((sh,sh),matvec=A,dtype=b.dtype)
            if symm:
                deltas,info = spla.minres(LinOp,b,x0=x0,tol=CG_TOL,maxiter=self.maxiter)
            else: 
                deltas,info = self.solver(LinOp,b,x0=x0,tol=CG_TOL,maxiter=self.maxiter)
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
    def __init__(self,sampler,pure_newton=False,solver='lgmres',**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'rgn' 
        self.compute_Hv = True
        self.pure_newton = pure_newton
        self.solver = {'lgmres':spla.lgmres,
                       'tfqmr':tfqmr}[solver] 

        self.rate2 = None # rate for LIN,RGN
        self.cond2 = None
        self.check = [1] 
    def _extract_Hvmean(self):
        Hvmean = np.zeros(self.nparam,dtype=self.dtype_o)
        COMM.Reduce(self.Hvsum,Hvmean,op=MPI.SUM,root=0)
        self.Hvsum = None
        if RANK==0:
            Hvmean /= self.n
            #print(Hvmean)
            self.Hvmean = Hvmean
    def extract_H(self,solve_full,solve_dense):
        fxn = self._get_Hmatrix if solve_dense else self._get_H_iterative
        self.Hx1 = np.zeros(self.nparam,dtype=self.dtype_i)
        if solve_full:
            self.H = fxn() 
        else:
            self.H = [None] * self.nsite
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
                self.H[ix] = fxn(start=start,stop=stop)
    def _get_Hmatrix(self,start=0,stop=None):
        stop = self.nparam if stop is None else stop
        t0 = time.time()
        if RANK==0:
            sh = stop-start
            vHvsum_ = np.zeros((sh,)*2,dtype=self.dtype_i)
        else:
            v = self.v[:,start:stop] 
            Hv = self.Hv[:,start:stop] 
            if self.sampler.exact:
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
                if self.sampler.exact:
                    Hx1 = np.dot(self.f * np.dot(self.Hv[:,start:stop],x),self.v[:,start:stop])
                else:
                    Hx1 = np.dot(np.dot(self.Hv[:,start:stop],x),self.v[:,start:stop])
                COMM.Reduce(Hx1,self.Hx1[start:stop],op=MPI.SUM,root=0)     
                return 0 
        return matvec
    def transform_gradients(self):
        return self._transform_gradients_rgn(self.solve_full,self.solve_dense)
    def _transform_gradients_rgn(self,solve_full,solve_dense,sr=None,enforce_pos=True):
        if sr is None:
            xnew_sr = self._transform_gradients_sr(True,False)
            deltas_sr = self.deltas
        else:
            xnew_sr,deltas_sr = sr

        self.extract_S(solve_full,solve_dense)
        self.extract_H(solve_full,solve_dense)
        if solve_dense:
            dEm = self._transform_gradients_rgn_dense(solve_full,enforce_pos)
        else:
            dEm = self._transform_gradients_rgn_iterative(solve_full,delta_sr)
        deltas_rgn = self.deltas

        rate = self.rate2 if self.pure_newton else 1.
        if self.check is None:
            xnew_rgn = self.update(rate) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
            return xnew_rgn

        if RANK==0:
            g,E,Eerr = self.g,self.E,self.Eerr
            Enew = [None] * len(self.check)
            Eerr_new = [None] * len(self.check)
        config = self.sampler.config
        xnew_rgn = [None] * len(self.check)
        for ix,scale in enumerate(self.check): 
            self.deltas = deltas_rgn
            xnew_rgn[ix] = self.update(rate*scale) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
            COMM.Bcast(xnew_rgn[ix],root=0) 
            self.sampler.af.update(xnew_rgn[ix])
            self.sampler.config = config

            self.free_quantities()
            self.sample(samplesize=self.batchsize_small,save_local=False,compute_v=False,compute_Hv=False,save_config=False)
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
        print(f'\tpredict={dEm},actual={(dE,Eerr_new)}')
        idx = np.argmin(dE) 
        if dE[idx]<0:
            self.deltas = deltas_rgn
            return xnew_rgn[idx]
        else:
            self.deltas = deltas_sr
            return xnew_sr
    def _transform_gradients_rgn_dense(self,solve_full,enforce_pos):
        if RANK>0:
            return 0. 
        t0 = time.time()
        if solve_full:
            if self.pure_newton:
                self.deltas,dE,w = _newton_block_solve(self.H,self.E,self.S,self.g,self.cond2,eigen=self.sampler.exact,enforce_pos=enforce_pos) 
                print(f'\tRGN solver time={time.time()-t0},least eigenvalue={w}')
            else:
                self.deltas,dE,w,eps = _rgn_block_solve(self.H,self.E,self.S,self.g,self.cond1,self.rate2,enforce_pos=enforce_pos) 
                print(f'\tRGN solver time={time.time()-t0},least eigenvalue={w},eps={eps}')
        else:
            blk_dict = self.sampler.af.block_dict
            w = [None] * len(blk_dict)
            dE = np.zeros(len(blk_dict))  
            self.deltas = np.empty(self.nparam,dtype=self.dtype)
            for ix,(start,stop) in enumerate(blk_dict):
                if self.pure_newton:
                    self.deltas[start:stop],dE[ix],w[ix] = _newton_block_solve(
                        self.H[ix],self.E,self.S[ix],self.g[start:stop],self.cond2,enforce_pos=enforce_pos)
                    print(f'ix={ix},eigval={w[ix]}')
                else:
                    self.deltas[start:stop],dE[ix],w[ix],eps = _rgn_block_solve(
                        self.H[ix],self.E,self.S[ix],self.g[start:stop],self.cond1,self.rate2,enforce_pos=enforce_pos)
                    print(f'ix={ix},eigval={w[ix]},eps={eps}')
            w = min(np.array(w).real)
            dE = np.sum(dE)
            print(f'\tRGN solver time={time.time()-t0},least eigenvalue={w}')
        if self.save_grad_hess:
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
    def _transform_gradients_rgn_iterative(self,solve_full,x0):
        g = self.g if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
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
            self.deltas = np.empty(self.nparam,dtype=self.dtype_i)
            for ix,(start,stop) in enumerate(self.sampler.af.block_dict):
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
        self.sample(samplesize=self.batchsize_small,save_local=False,compute_v=False,compute_Hv=False,save_config=False)
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
            pk,info = self.solver(LinOp,gk,x0=gk,tol=CG_TOL,maxiter=self.maxiter)
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
    def normalize(self,x):
        return x
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
        if self.save_grad_hess:
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
    def _burn_in(self,config=None,burn_in=None,progbar=False):
        if config is not None:
            self.config = config 
        self.px = self.af.log_prob(self.af.parse_config(self.config))

        if RANK==0:
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
        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
        ar.set_backend(tsr)
        for site in self.sites:
            tag = self.site_tag(site)
            self.psi[tag].modify(data=self.tensor2backend(self.psi[tag].data,backend=backend,requires_grad=requires_grad))
        for key in self.data_map:
            self.data_map[key] = self.tensor2backend(self.data_map[key],backend=backend,requires_grad=False)
        if self.from_plq:
            self.model.gate2backend(backend)
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
    def amplitude(self,config,sign=True,cache_bot=None,cache_top=None,to_numpy=True,i=None,direction='row'):
        cx = self.unsigned_amplitude(config,cache_bot=cache_bot,cache_top=cache_top,to_numpy=to_numpy,i=i,direction=direction)
        if cx is None:
            return None 
        if sign:
            cx *= self.config_sign(config)
        return cx
    def log_prob(self, config):
        """Calculate the probability of a configuration.
        """
        cx = self.unsigned_amplitude(config)
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
    def extract_grad(self):
        vx = {site:self.tensor_grad(self.psi[self.site_tag(site)].data) for site in self.sites}
        return self.dict2vec(vx)
    def propagate(self,ex):
        if not isinstance(ex,torch.Tensor):
            return 0.,np.zeros(self.nparam)
        ex.backward()
        Hvx = self.extract_grad()
        return tensor2backend(ex,'numpy'),Hvx 
    def get_grad_deterministic(self,config):
        self.wfn2backend(backend='torch',requires_grad=True)
        cache_top = dict()
        cache_bot = dict()
        cx = self.amplitude(config,cache_top=cache_top,cache_bot=cache_bot,to_numpy=False)
        if cx is None:
            return cx,np.zeros(self.nparam)
        cx,vx = self.propagate(cx)
        vx /= cx
        self.wfn2backend()
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
    def update_pair_energy_from_benvs(self,where,cache_bot,cache_top,**kwargs):
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
            ex[key] = self.amplitude(config_new,cache_bot=cache_bot,cache_top=cache_top,to_numpy=False,**kwargs) 
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
    def batch_quantities(self,batch_key,compute_v,compute_Hv): 
        if compute_Hv:
            self.wfn2backend(backend='torch',requires_grad=True)
        ex,plq = self.batch_pair_energies(batch_key,compute_Hv)

        Hvx = 0
        if compute_Hv:
            _,Hvx = self.propagate(self.parse_energy(ex,batch_key,False))
        ex = self.parse_energy(ex,batch_key,True)
        self.amplitude2scalar()
        if compute_v:
            self.get_grad_from_plq(plq) 
        if compute_Hv:
            self.wfn2backend()
        return ex,Hvx
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
    def compute_local_energy(self,config,compute_v=True,compute_Hv=False):
        config = self.set_config(config,compute_v)
        vx = None
        if not self.from_plq:
            if compute_v:
                cx,vx = self.get_grad_deterministic(config)
            else:
                cx,vx = self.amplitude(config),None

        ex,Hvx = 0.,0.
        for batch_key in self.model.batched_pairs:
            ex_,Hvx_ = self.batch_quantities(batch_key,compute_v,compute_Hv)
            ex += ex_
            Hvx += Hvx_
        cx,err = self.contraction_error() 
        eu = self.model.compute_local_energy_eigen(self.config)
        ex += eu
        if self.from_plq and compute_v:
            vx = self.parse_gradient()
        if compute_Hv:
            Hvx = Hvx / cx + eu * vx
        self.vx = None
        self.cx = None
        return cx,ex,vx,Hvx,err 
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
