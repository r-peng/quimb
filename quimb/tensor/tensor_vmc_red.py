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
MAXITER = 500
#MAXITER = 2
##################################################################################################
# VMC utils
##################################################################################################
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
def gram_schmidt(Q,R,ak,thresh):
    nparam = len(ak)
    if Q is None:
        nk = np.linalg.norm(ak)
        ek = ak/nk
        Q = ek.reshape(nparam,1)
        R = np.array([[nk]])
        return Q,R
    nb,ns = R.shape
    #uk = ak.copy()
    #ejak = np.zeros(nb)
    #for i in range(nb):
    #    ejak[i] = np.dot(ak,Q[:,i])
    #    uk -= ejak[i] * Q[:,i] 
    ejak = np.dot(ak,Q)
    uk = ak - np.dot(Q,ejak)
    nk = np.linalg.norm(uk)
    if nk < thresh:
        R = np.concatenate([R,ejak.reshape(nb,1)],axis=1)
        return Q,R
    ek = uk / nk
    ekak = np.dot(ek,ak)
    dmax = np.amax(np.fabs(np.diag(R)))
    if abs(ekak)/dmax < thresh:
        R = np.concatenate([R,ejak.reshape(nb,1)],axis=1)
        return Q,R
    Q = np.concatenate([Q,ek.reshape(nparam,1)],axis=1) 
    Rnew = np.zeros((nb+1,ns+1))
    Rnew[:nb,:ns] = R
    Rnew[:nb,-1] = ejak
    Rnew[-1,-1] = np.dot(ek,ak)
    return Q,Rnew
def truncated_svd(M,thresh,hermitian=False):
    u,s,vh = np.linalg.svd(M,full_matrices=False,hermitian=hermitian)
    s = s[s>thresh]
    ns = len(s)
    return u[:,:ns],s,vh[:ns,:]
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
        self.compute_h = False

        # to be set before run
        self.tmpdir = None
        self.batchsize = None
        self.batchsize_small = None
        self.rate1 = None # rate for SGD,SR

        self.gs_thresh = 1e-4
        self.progbar = False
        self.save_sampled = False
    def free_quantities(self):
        if RANK==0:
            self.buf = None
            self.f = None
            self.e = None
            self.Qv = None
            self.Rv = None
            self.Qh = None
            self.Rh = None

            self.Eold = self.E
            self.g = None
            self.Rvmean = None
            self.Revmean = None
            self.Rsq = None
            self.Q = None
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
            self.free_quantities()
            COMM.Bcast(x,root=0) 
            fname = self.tmpdir+f'psi{step+1}' if save_wfn else None
            psi = self.sampler.af.update(x,fname=fname,root=0)
    def sample(self,samplesize=None,compute_v=True,compute_h=None,save_config=True):
        self.sampler.preprocess()

        compute_h = self.compute_h if compute_h is None else compute_h
        self.terminate = np.array([0])
        # 0:step,1:rank,2:f,3:c,4:err,5:e,6:config,vx,hx
        size = 6 + self.nsite 
        if compute_v:
            size += self.nparam
        if compute_h:
            size += self.nparam
        self.buf = np.zeros(size,self.dtype_o)
        self.buf[0] = self.step 
        self.buf[1] = RANK
        ix0 = 6 # float values
        ix1 = ix0 + self.nsite # config
        ix2 = ix1 + self.nparam # v
        ix3 = ix2 + self.nparam # h
        self.buf_idx = ix0,ix1,ix2,ix3

        #print(f'before, {RANK},{len(self.buf)}')
        if self.sampler.exact:
            if RANK==0:
                self._ctr_exact()
            else:
                self._sample_exact()
        else:
            if RANK==0:
                samplesize = self.batchsize if samplesize is None else samplesize
                self._ctr_stochastic(samplesize,save_config=save_config)
            else:
                self._sample_stochastic()
        #print(f'after, {RANK},{len(self.buf)}')
        #exit()
    def _init_ctr(self):
        self.e = []
        self.f = []
        self.n = 0
        ix0,ix1,ix2,ix3 = self.buf_idx
        if len(self.buf)>ix1:
            self.Qv,self.Rv = None,None
            self.v = [] 
        if len(self.buf)>ix2:
            self.Qh,self.Rh = None,None
            self.h = [] 
        self.err_sum = 0.
        self.err_max = 0.
    def _process_ctr(self):
        self.e = np.array(self.e)
        print('\tcontraction err=',self.err_sum/self.n,self.err_max)
        print('\tnsamples=',self.n)
        ix0,ix1,ix2,ix3 = self.buf_idx
        if len(self.buf) > ix1:
            print('\tnumber of v basis=',self.Rv.shape[0])
            self.v = np.array(self.v).T
            v = np.dot(self.Qv,self.Rv)
            print('check v=',np.linalg.norm(self.v-v))
            #print(self.Qv)
            #self.v = None
        if len(self.buf) > ix2:
            print('\tnumber of h basis=',self.Rh.shape[0])
            self.h = np.array(self.h).T
            h = np.dot(self.Qh,self.Rh)
            print('check h=',np.linalg.norm(self.h-h))
            #print(self.Qh)
            #print(self.h)
            #exit()
            #self.h = None
        #print('Rv diag')
        #print(np.diag(self.Rv))
        #print('Rh diag')
        #print(np.diag(self.Rh))
    def _fill_buf(self,info,config,fx=1):
        cx,ex,vx,hx,err = info
        ix0,ix1,ix2,ix3 = self.buf_idx
        self.buf[2:ix0] = np.array([fx,cx,err,ex])
        self.buf[ix0:ix1] = np.array(config)
        if len(self.buf) > ix1:
            self.buf[ix1:ix2] = vx
        if len(self.buf) > ix2:
            self.buf[ix2:] = hx
    def _parse_buffer(self):
        ix0,ix1,ix2,ix3 = self.buf_idx

        fx,cx,err,ex = self.buf[2:ix0] 
        config = self.buf[ix0:ix1]
        self.e.append(ex) 
        self.f.append(fx) 
        self.err_sum += err
        self.err_max = max(err,self.err_max)
        if len(self.buf) > ix1:
            vx = self.buf[ix1:ix2] 
            self.v.append(vx.copy())
            self.Qv,self.Rv = gram_schmidt(self.Qv,self.Rv,vx,self.gs_thresh)
            sh = self.Qv.shape
            if sh[0] < sh[1]:
                print(self.Rh.shape)
                print(np.diag(self.Rh))
                print('Qv shape=',sh)
                raise ValueError
        if len(self.buf) > ix2:
            hx = self.buf[ix2:] 
            self.h.append(hx.copy())
            self.Qh,self.Rh = gram_schmidt(self.Qh,self.Rh,hx,self.gs_thresh)
            sh = self.Qh.shape
            if sh[0] < sh[1]:
                print(self.Rh.shape)
                print(np.diag(self.Rh))
                print('Qh shape=',sh)
                raise ValueError
        return config
    def _ctr_stochastic(self,samplesize,save_config=True):
        self._init_ctr()
        configs = []
        if self.progbar:
            pg = Progbar(total=samplesize)
        t0 = time.time()
        terminate = np.array([0]*(SIZE-1))
        #print(f'{RANK},{len(self.buf)}')
        print('Qv,Rv',self.Qv,self.Rv)
        print('Qh,Rh',self.Qh,self.Rh)
        while True:
            COMM.Recv(self.buf,tag=0)
            step = int(self.buf[0]+.1)
            rank = int(self.buf[1]+.1)
            if step>self.step:
                raise ValueError
            if step<self.step:
                COMM.Send(self.terminate,dest=rank,tag=1)
                continue
            self.n += 1
            if self.n >= samplesize: 
                self.terminate[0] = 1
                terminate[rank-1] = 1
            COMM.Send(self.terminate,dest=rank,tag=1)

            config = self._parse_buffer()
            if save_config:
                if samplesize - self.n < SIZE:
                    configs.append(config.copy())
            if self.progbar:
                pg.update()
            if sum(terminate) == SIZE - 1:
                break
        print('\tsample time=',time.time()-t0)
        self._process_ctr()
        if self.save_sampled:
            self._save_sampled()
    def _save_sampled(self,compute_v=True,compute_h=False):
        ix0,ix1,ix2,ix3 = self.buf_idx
        f = h5py.File(self.tmpdir+f'step{self.step}.hdf5','w')
        f.create_dataset('e',data=self.e) 
        if self.sampler.exact:
            f.create_dataset('f',data=self.f) 
        if len(self.buf) > ix1:
            f.create_dataset('Qv',data=self.Qv) 
            f.create_dataset('Rv',data=self.Rv) 
        if len(self.buf) > ix2:
            f.create_dataset('Qv',data=self.Qv) 
            f.create_dataset('Rv',data=self.Rv) 
        f.close()
    def _sample_stochastic(self):
        ix0,ix1,ix2,ix3 = self.buf_idx
        compute_v = (len(self.buf)>ix1)
        compute_h = (len(self.buf)>ix2)
        #print(f'{RANK},{len(self.buf)}')
        while True:
            config,omega = self.sampler.sample()
            #if omega > self.omega:
            #    self.config,self.omega = config,omega
            info = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_h=compute_h)
            cx,ex,vx,hx,err = info
            if cx is None or np.fabs(ex.real) > DISCARD:
                print(f'RANK={RANK},cx={cx},ex={ex}')
                continue
            self._fill_buf(info,config)
            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.terminate,source=0,tag=1)
            if self.terminate[0]==1:
                break
    def _ctr_exact(self):
        self._init_ctr()
        samplesize = len(self.sampler.nonzeros)
        if self.progbar:
            pg = Progbar(total=samplesize)
        t0 = time.time()
        for _ in range(samplesize):
            COMM.Recv(self.buf,tag=0)
            self._parse_buffer()
            self.n += 1
            if self.progbar:
                pg.update()
        print('\tsample time=',time.time()-t0)
        self._process_ctr()
        self.f = np.array(self.f)
    def _sample_exact(self): 
        # assumes exact contraction
        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros
        ntotal = len(ixs)
        if RANK==SIZE-1:
            print('\tnsamples per process=',ntotal)

        ix0,ix1,ix2,ix3 = self.buf_idx
        compute_v = (len(self.buf)>ix1)
        compute_h = (len(self.buf)>ix2)
        for ix in ixs:
            config = all_configs[ix]
            info = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_h=compute_h)
            cx,ex,vx,hx,err = info
            if cx is None:
                raise ValueError
            if np.fabs(ex.real)*p[ix] > DISCARD:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self._fill_buf(info,config,fx=p[ix])
            COMM.Send(self.buf,dest=0,tag=0) 
    def extract_energy_gradient(self):
        if RANK>0:
            return
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()
        try:
            dE = 0 if self.Eold is None else self.E-self.Eold
            print(f'step={self.step},E={self.E/self.nsite},dE={dE/self.nsite},err={self.Eerr/self.nsite},gmax={np.amax(np.fabs(self.g))}')
        except TypeError:
            print('E=',self.E)
            print('Eerr=',self.Eerr)
        print('\tcollect g,h time=',time.time()-t0)
    def extract_energy(self):
        if RANK>0:
            return
        if self.sampler.exact:
            self.E = np.dot(self.f,self.e)
            self.Eerr = 0.
        else:
            self.E,self.Eerr = blocking_analysis(self.e)
    def extract_gradient(self):
        if self.sampler.exact:
            self.Rvmean = np.dot(self.Rv,self.f)
            self.Revmean = np.dot(self.Rv,self.f*self.e)
        else:
            self.Rvmean = np.sum(self.Rv,axis=1)/self.n
            self.Revmean = np.dot(self.Rv,self.e)/self.n
        vmean = np.dot(self.Qv,self.Rvmean)
        evmean = np.dot(self.Qv,self.Revmean)
        self.g = (evmean - self.E.conj() * vmean).real
    def update(self,rate,deltas):
        x = self.sampler.af.get_x()
        return self.normalize(x - rate * deltas)
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
        x = self.update(self.rate1,deltas)
        print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(deltas)},dot={np.dot(deltas,self.g)},x={np.linalg.norm(x)}')
        return x 
class SR(SGD):
    def __init__(self,sampler,**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'sr' 
        self.eigen_thresh = 1e-3
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
    def extract_S(self):
        if RANK>0:
            return
        rhs = np.dot(self.Qv.T,self.g)

        if self.sampler.exact:
            Rsq = np.einsum('is,js,s->ij',self.Rv,self.Rv,self.f)
        else:
            Rsq = np.dot(self.Rv,self.Rv.T)/self.n
        Rsq -= np.outer(self.Rvmean,self.Rvmean)
        u,s,vh = truncated_svd(Rsq,self.eigen_thresh,hermitian=True)
        print('SR s=',s[0],s[-1])

        rhs = s**(-1) * np.dot(u.T,rhs)
        lhs = np.dot(vh,self.Qv.T)
        print('\tnumber of singular values in S=',len(s))
        self.Rsq = Rsq
        return lhs,rhs
    def transform_gradients(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i)
        deltas = self._transform_gradients_sr()
        x = self.update(self.rate1,deltas)
        print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(deltas)},dot={np.dot(deltas,self.g)},x={np.linalg.norm(x)}')
        return x 
    def _transform_gradients_sr(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i) 
        lhs,rhs = self.extract_S()
        deltas,res,rank,s = np.linalg.lstsq(lhs,rhs,rcond=self.eigen_thresh)
        print('\tSR lstsq res=',res)

        if self.sampler.exact:
            S = np.einsum('is,js,s->ij',self.v,self.v,self.f)
            vmean = np.dot(self.v,self.f)
        else:
            S = np.dot(self.v,self.v.T)/self.n
            vmean = np.sum(self.v,axis=1)/self.n
        S -= np.outer(vmean,vmean) 
        S += np.eye(S.shape[0]) * 1e-4 
        deltas_ = np.linalg.solve(S,self.g)
        print('check SR deltas=',np.linalg.norm(deltas-deltas_))
        return deltas
class RGN(SR):
    def __init__(self,sampler,pure_newton=False,solver='lgmres',guess=3,**kwargs):
        super().__init__(sampler,**kwargs) 
        self.optimizer = 'rgn' 
        self.compute_h = True
        self.pure_newton = pure_newton

        self.rate2 = None # rate for LIN,RGN
        self.cond2 = None
        self.check = [1] 
    def extract_H(self):
        if RANK>0:
            return 
        rhs = np.dot(self.Qv.T,self.g)

        if self.sampler.exact:
            self.Rhmean = np.dot(self.Rh,self.f)
            Rsq1 = np.einsum('is,js,s->ij',self.Rv,self.Rh,self.f)
        else:
            self.Rhmean = np.sum(self.Rh,axis=1)/self.n
            Rsq1 = np.dot(self.Rv,self.Rh.T)/self.n
        Rsq1 -= np.outer(self.Rvmean,self.Rhmean)
        Q1 = np.dot(Rsq1,self.Qh.T)

        coeff = -self.E
        if not self.pure_newton:
            coeff += 1./self.rate2
        Rsq2 = coeff * self.Rsq 
        Rsq2 += self.E * np.outer(self.Rvmean,self.Rvmean)
        Rsq2 -= np.outer(self.Revmean,self.Rvmean)
        Q2 = np.dot(Rsq2,self.Qv.T)
        Q = Q1+Q2

        u,s,vh = truncated_svd(Q,self.eigen_thresh)
        print('RGN s=',s[0],s[-1])
        rhs = s**(-1) * np.dot(u.T,rhs)
        lhs = vh 
        fname = 'H' if self.pure_newton else 'H+S/rate'
        print(f'\tnumber of singular values in {fname}=',len(s))
        self.Q = Q
        return lhs,rhs 
    def _transform_gradients_rgn(self):
        if RANK>0:
            return np.zeros(self.nparam,dtype=self.dtype_i),0 
        lhs,rhs = self.extract_H()
        deltas,res,rank,s = np.linalg.lstsq(lhs,rhs,rcond=self.eigen_thresh)
        fname = 'Newton' if self.pure_newton else 'RGN'
        print(f'\t{fname} lstsq res=',res)

        dE = .5 * np.dot(np.dot(deltas,self.Qv),np.dot(self.Q,deltas))
        dE -= np.dot(self.g,deltas)
        print('dE=',dE)

        if self.sampler.exact:
            S = np.einsum('is,js,s->ij',self.v,self.v,self.f)
            H = np.einsum('is,js,s->ij',self.v,self.h,self.f)
            vmean = np.dot(self.v,self.f)
            hmean = np.dot(self.h,self.f)
        else:
            S = np.dot(self.v,self.v.T)/self.n
            H = np.dot(self.v,self.h.T)/self.n
            vmean = np.sum(self.v,axis=1)/self.n
            hmean = np.sum(self.h,axis=1)/self.n
        S -= np.outer(vmean,vmean) 
        H -= np.outer(vmean,hmean) 
        H -= np.outer(self.g,vmean)
        H -= self.E * S
        S += np.eye(S.shape[0]) * 1e-4 
        H += S/self.rate2
        deltas_ = np.linalg.solve(H,self.g)
        print('check RGN deltas=',np.linalg.norm(deltas-deltas_))
        dE_ = .5 * np.dot(np.dot(deltas_,H),deltas)
        dE_ -= np.dot(self.g,deltas_)
        print('dE=',dE_)
        return deltas,dE 
    def transform_gradients(self):
        deltas_sr = self._transform_gradients_sr()
        xnew_sr = self.update(self.rate1,deltas_sr)
        deltas_rgn,dEm = self._transform_gradients_rgn()

        rate = self.rate2 if self.pure_newton else 1.
        if self.check is None:
            xnew_rgn = self.update(rate,deltas_rgn) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
            if RANK==0:
                print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(deltas_rgn)},dot={np.dot(deltas_rgn,self.g)},x={np.linalg.norm(xnew_rgn)}')
            return xnew_rgn

        if RANK==0:
            g,E,Eerr = self.g,self.E,self.Eerr
            Enew = [None] * len(self.check)
            Eerr_new = [None] * len(self.check)
        config = self.sampler.config
        xnew_rgn = [None] * len(self.check)
        for ix,scale in enumerate(self.check): 
            xnew_rgn[ix] = self.update(rate*scale,deltas_rgn) if RANK==0 else np.zeros(self.nparam,dtype=self.dtype_i)
            COMM.Bcast(xnew_rgn[ix],root=0) 
            self.sampler.af.update(xnew_rgn[ix])
            self.sampler.config = config

            self.free_quantities()
            self.sample(samplesize=self.batchsize_small,compute_v=False,compute_h=False,save_config=False)
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
            deltas = deltas_rgn
            x = xnew_rgn[idx]
        else:
            deltas = deltas_sr
            x = xnew_sr
        print(f'\tg={np.linalg.norm(self.g)},del={np.linalg.norm(deltas)},dot={np.dot(deltas,self.g)},x={np.linalg.norm(x)}')
        return x
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
