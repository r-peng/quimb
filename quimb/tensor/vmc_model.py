import numpy as np
import itertools,h5py
import scipy.linalg
from .tensor_vmc import (
    DISCARD,
    Progbar,
    DenseSampler,
    ExchangeSampler,
    RGN,
    to_square,
)
np.set_printoptions(suppress=True,precision=6,linewidth=1000)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

H = np.array([[0., 1.], [1., 0.]])
def wf(x):
    return x/np.linalg.norm(x)
def deriv_wf2(x,eps=1e-4):
    g= np.zeros([2, 2])
    g[:,0] = (wf(x+np.array([eps,0.]))-wf(x))/eps
    g[:,1] = (wf(x+np.array([0.,eps]))-wf(x))/eps
    return g
def deriv_wf(x):
    return np.eye(2) - np.outer(x,x) 
def init(L,fname,eps=1):
    x = np.ones((L,2))
    x[:,1] = -1
    x += np.random.normal(size=(L,2)) * eps 
    x = [wf(xi) for xi in x]
    x = np.array(x)
    np.save(fname+'.npy',x)
    print(x)
def make_matrices(x):
    L = len(x)
    ovlp = np.zeros((L,2,L,2))
    g = [deriv_wf(xi) for xi in x]
    #g = [deriv_wf2(xi) for xi in x]
    xg = [np.dot(xi,gi) for gi,xi in zip(g,x)]
    for i in range(L):
        ovlp[i,:,i,:] = np.dot(g[i].T,g[i])
        for j in range(i+1,L):
            ovlp_ij = np.outer(xg[i],xg[j])
            ovlp[i,:,j,:] = ovlp_ij 
            ovlp[j,:,i,:] = ovlp_ij.T

    hess = np.zeros((L,2,L,2))
    xHg = [np.dot(np.dot(xi,H),gi) for gi,xi in zip(g,x)]
    e = [np.dot(np.dot(xi,H),xi) for xi in x] 
    E = sum(e)
    for i in range(L):
        hess[i,:,i,:] = np.dot(g[i].T,np.dot(H,g[i]))
        hess[i,:,i,:] += (E-e[i]) * ovlp[i,:,i,:] 
        for j in range(i+1,L):
            hess_ij = np.outer(xHg[i],xg[j])   
            hess_ij += np.outer(xg[i],xHg[j])   
            hess_ij += (E-e[i]-e[j]) * ovlp[i,:,j,:]
            hess[i,:,j,:] = hess_ij
            hess[j,:,i,:] = hess_ij.T

    g = np.array([xHgi + (E-ei) * xgi for xHgi,ei,xgi in zip(xHg,e,xg)]).flatten()
    ovlp = ovlp.reshape(L*2,L*2)
    hess = hess.reshape(L*2,L*2) - E * ovlp
    return E,g,ovlp,hess
def sample_matrices(sampler,sample_size=None,tmpdir=None,exact_variance=False):
    sampler.config = tuple([0] * sampler.nsite)
    eng = CovRGN if exact_variance else RGN
    vmc = eng(sampler,normalize=False,solve_full=True,solve_dense=True)
    vmc.batchsize = sample_size
    vmc.tmpdir = tmpdir
    vmc.save_grad_hess = False if tmpdir is None else True 
    vmc.progbar = True
    vmc.step = 0
    vmc.Eold = 0
    vmc.nparam = vmc.sampler.af.nparam
    vmc.np2 = (1+vmc.nparam)*vmc.nparam//2
    vmc.triu_idx = np.triu_indices(vmc.nparam)
    vmc.sample(save_config=False)
    vmc.extract_energy_gradient()
    vmc.extract_S(solve_full=True,solve_dense=True)
    vmc.extract_H(solve_full=True,solve_dense=True)
    vmc._save_grad_hess()
    if not exact_variance:
        return 
    if RANK>0:
        return
    vmc.covariance() 
def from_noisy(mean,ns,nparam,sample_size):
    x = mean + ns/np.sqrt(sample_size+1e-10)
    np2 = (1+nparam) * nparam // 2

    E,x = x[:1],x[1:]
    v,x = x[:nparam],x[nparam:]
    ev,x = x[:nparam],x[nparam:]
    h,x = x[:nparam],x[nparam:]
    vv,vh = x[:np2],x[np2:]

    g = ev - E * v
    S = to_square(vv,sh=nparam) - np.outer(v,v)
    hess = vh.reshape(nparam,nparam) - np.outer(v,h) - np.outer(g,v) - E * S
    return E,g,S,hess
class AmplitudeFactory:
    def __init__(self,x):
        self.x = x
        self.Hx = [np.dot(H,xi) for xi in x]
        self.nsite = len(self.Hx)
        self.nparam = 2 * self.nsite
    def get_x(self):
        return np.array(self.x).flatten()
    def log_prob(self,config):
        return np.log(np.array([xi[ci] for xi,ci in zip(self.x,config)])**2).sum()
    def compute_local_energy(self,config,compute_v=True,compute_h=True):
        cx = np.array([xi[ci] for xi,ci in zip(self.x,config)]).prod()
        ex = sum([Hxi[ci]/xi[ci] for Hxi,xi,ci in zip(self.Hx,self.x,config)])
        vx = None
        if compute_v:
            vx = np.zeros((self.nsite,2))
            for i,(xi,ci) in enumerate(zip(self.x,config)):
                vx[i,ci] = 1/xi[ci]
        #print(config,vx)

        hx = None
        if compute_h:
            hx = np.zeros((self.nsite,2))
            for i,(Hxi,xi,ci) in enumerate(zip(self.Hx,self.x,config)):
                hx[i] = np.dot(H,vx[i])
                hx[i] += (ex-Hxi[ci]/xi[ci]) * vx[i]

        if compute_v:
            vx = vx.flatten()
        if compute_h:
            hx = hx.flatten()
        return cx,ex,vx,hx,0.
    def update(self,x,fname=None,root=0):
        self.x = np.array([wf(xi) for xi in x.reshape(self.nsite,2)])
        #self.x = x.reshape(self.nsite,2)
    def parse_config(self,config):
        return config
    def make_matrices(self,eps=0):
        L = self.nsite
        e = np.array([np.dot(xi,Hxi) for xi,Hxi in zip(self.x,self.Hx)])
        E = sum(e)
#        E2 = L + 2 * sum([e[i]*e[j] for i in range(L) for j in range(i+1,L)]) 
#        varE = E2 - E**2
#        print(E,varE)
#        Es = E + np.random.normal(loc=0,scale=eps*np.sqrt(varE))
#        
#        ev = np.array([Hxi+xi*(E-ei) for Hxi,xi,ei in zip(self.Hx,self.x,e)]).flatten()
#        print('ev_mean=',ev)
#        sigma_ev = np.zeros((L,2,L,2))
#        log = [np.diag(1/xi) for xi in self.x]
#        for i in range(L):
#            #sigma_ev[i,:,i,:] = np.eye(2) * L
#            sigma_ev[i,:,i,:] = np.dot(log[i],log[i]) * L
#            sii2 = np.zeros((2,2))
#            for k in range(L):
#                for l in range(k+1,L):
#                    if k==i:
#                        sii2 += np.dot(log[i],np.dot(H,log[i])) * e[l] 
#                    elif l==i:
#                        sii2 += np.dot(log[i],np.dot(H,log[i])) * e[k] 
#                    else:
#                        sii2 += np.dot(log[i],log[i]) * e[k] * e[l]
#            sigma_ev[i,:,i,:] += 2 * sii2 
#            #for k in range(L):
#            #    for l in range(L):
#            #        if k==i:
#            #            if l==i:
#            #                #sigma_ev[i,:,i,:] += np.eye(2)
#
#            #                sigma_ev[i,:,i,:] += np.dot(log[i],log[i])
#            #            else:
#            #                sigma_ev[i,:,i,:] += np.dot(log[i],np.dot(H,log[i])) * e[l]
#            #        else:
#            #            if l==i:
#            #                sigma_ev[i,:,i,:] += np.dot(log[i],np.dot(H,log[i])) * e[k]
#            #            else:
#            #                sigma_ev[i,:,i,:] += np.dot(log[i],log[i]) * e[k] * e[l]
#            for j in range(i+1,L):
#                sij = np.outer(self.x[i],self.x[j]) * L
#                sij2 = np.zeros((2,2))
#                for k in range(L):
#                    for l in range(k+1,L):
#                        if k==i:
#                            if l==j:
#                                sij2 += np.outer(self.Hx[i],self.Hx[j])
#                            else:
#                                sij2 += np.outer(self.Hx[i],self.x[j]) * e[l]
#                        else:
#                            if l==j:
#                                sij2 += np.outer(self.x[i],self.Hx[j]) * e[k]
#                            else:
#                                sij2 += np.outer(self.x[i],self.x[j]) * e[k] * e[l]
#                sij += sij2
#                sigma_ev[i,:,j,:] = sij
#                sigma_ev[j,:,i,:] = sij.T
#        sigma_ev = sigma_ev.reshape((L*2,L*2)) - np.outer(ev,ev)
#        print('ev_var=',np.diag(sigma_ev))
##        g = g1 + np.random.normal(loc=0,scale=eps_g1,size=g1.shape)
        v = self.x.flatten()    
#        print('g=',ev - E * v)
##        vs = self.x.flatten()
##        vs += np.random.normal(loc=0,scale=eps_v,size=vs.shape)
##        g -= Es * vs
##
        S1 = np.zeros((L,2,L,2))
        for i in range(L):
            S1[i,:,i,:] = np.eye(2)
            for j in range(i+1,L):
                Sij = np.outer(self.x[i],self.x[j])
                S1[i,:,j,:] = Sij
                S1[j,:,i,:] = Sij.T
        S = S1.reshape(L*2,L*2)
        print('S1')
        print(S)
        w,_ = np.linalg.eigh(S)
        print(w)
        print('outer')
        ot = np.outer(v,v)
        print(ot)
        w,_ = np.linalg.eigh(ot)
        print(w)

        sigma_v = S.copy() - np.outer(v,v)
        print('sigma_v=')
        print(sigma_v)
        exit()
##        noise = np.random.normal(loc=0,scale=eps_S1,size=S.shape)
##        triu = np.triu(noise,k=2)
##        diag = np.diag(noise)
##        noise = triu + triu.T
##        np.fill_diagonal(noise,diag)
##        S += noise 
##        S -= np.outer(vs,vs)
##        
##        H1 = np.zeros((L,2,L,2))
##        for i in range(L):
##            H1[i,:,i,:] = H + np.eye(2)*(E-e[i])           
##            for j in range(i+1,L):
##                Hij = np.outer(self.Hx[i],v[j]) 
##                Hij += np.outer(v[i],self.Hx[j])
##                Hij += np.outer(v[i],v[j]) * (E-e[i]-e[j]) 
##                H1[i,:,j,:] = Hij
##                H1[j,:,i,:] = Hij.T
##        hess = H1.reshape(L*2,L*2)
##        hess += np.random.normal(loc=0,scale=eps_H1,size=hess.shape)
##        Hv = g1 + np.random.normal(loc=0,scale=eps_g1,size=g1.shape)
##        hess -= np.outer(vs,Hv)
##        hess -= np.outer(g,vs)
##        hess -= Es * S
##        return Es,g,S,hess
class Sampler(ExchangeSampler):
    def __init__(self,af,burn_in=0,seed=None,every=1):
        self.af = af
        self.nsite = self.af.nsite
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.every = every 
        self.exact = False
    def sample(self):
        for _ in range(self.every):
            step = self.rng.choice([-1,1])
            sweep = range(self.nsite) if step==1 else range(self.nsite-1,-1,-1)
            for i in sweep:
                i_old = self.config[i]
                i_new = 1-i_old
                p = self.af.x[i]**2
                acceptance = p[i_new]/p[i_old]
                if acceptance < self.rng.uniform(): # reject
                    continue

                config_new = list(self.config)
                config_new[i] = i_new 
                self.config = tuple(config_new)
        return self.config,None
class ModelDenseSampler(DenseSampler):
    def __init__(self,nsite,exact=False,seed=None,thresh=1e-28):
        super().__init__(nsite,None,exact=exact,seed=seed,thresh=thresh)
    def get_all_configs(self,fix_sector=None):
        return list(itertools.product((0,1),repeat=self.nsite))
class CovRGN(RGN):
    def _sample_exact(self,compute_v,compute_h): 
        # assumes exact contraction
        p = self.sampler.p
        all_configs = self.sampler.all_configs
        ixs = self.sampler.nonzeros
        ntotal = len(ixs)
        if RANK==SIZE-1:
            print('\tnsamples per process=',ntotal)

        self.f = []
        self.c = []
        self.e = []
        self.ev = []
       
        self.err_max = 0.
        self.err_sum = 0.
        for ix in ixs:
            config = all_configs[ix]
            cx,ex,vx,hx,err = self.sampler.af.compute_local_energy(config,compute_v=compute_v,compute_h=compute_h)
            if cx is None:
                raise ValueError
            if np.fabs(ex.real)*p[ix] > DISCARD:
                raise ValueError(f'RANK={RANK},config={config},cx={cx},ex={ex}')
            self.f.append(p[ix])
            self.e.append(ex)
            self.err_sum += err
            self.err_max = max(err,self.err_max)
            if compute_v:
                self.vsum += vx * p[ix]
                evx = vx * ex.conj() 
                self.evsum += evx * p[ix] 
                self.v.append(vx)
                self.ev.append(evx)
            if compute_h:
                self.hsum += hx * p[ix]
                self.h.append(hx)
            COMM.Send(self.buf,dest=0,tag=0) 
            COMM.Recv(self.buf,source=0,tag=1)
        self.f = np.array(self.f)
        self.e = np.array(self.e)
        self.c = np.array(self.c)
        if compute_v:
            self.v = np.array(self.v)
            self.ev = np.array(self.ev)
        if compute_h:
            self.h = np.array(self.h)
    def extract_energy(self):
        super().extract_energy()
        if RANK==0:
            eei = np.zeros(1)
        else:
            self.fe = self.f * self.e
            eei = np.array([np.dot(self.fe,self.e)])
        ee = np.zeros_like(eei)
        COMM.Reduce(eei,ee,op=MPI.SUM,root=0)
        if RANK>0:
            return
        self.ee = ee
    def extract_gradient(self):
        super().extract_gradient()
        if RANK==0:
            eevi = np.zeros(self.nparam)
            evvi = np.zeros(self.np2)
            eevvi = np.zeros(self.np2)
        else:
            self.vvi = np.einsum('si,sj->ijs',self.v,self.v)
            self.vvi = self.from_square(self.vvi)
            eevi = np.dot(self.fe,self.ev)
            evvi = np.dot(self.vvi,self.fe) 
            eevvi = np.dot(self.vvi,self.fe*self.e) 
        eev = np.zeros_like(eevi)
        evv = np.zeros_like(evvi)
        eevv = np.zeros_like(eevvi)
        COMM.Reduce(eevi,eev,op=MPI.SUM,root=0)
        COMM.Reduce(evvi,evv,op=MPI.SUM,root=0)
        COMM.Reduce(eevvi,eevv,op=MPI.SUM,root=0)
        if RANK>0:
            return
        self.eev = eev
        self.evv = evv
        self.eevv = eevv 
    def _get_Smatrix(self,start=0,stop=None):
        S = super()._get_Smatrix(start=start,stop=stop)
        np2 = (1+self.np2) * self.np2 // 2
        idx = np.triu_indices(self.np2)
        if RANK==0:
            vvvi = np.zeros((self.np2,self.nparam))
            vvevi = np.zeros((self.np2,self.nparam))
            vvvvi = np.zeros(np2)
        else:
            vvvi = np.einsum('is,s,sj->ij',self.vvi,self.f,self.v)
            vvevi = np.einsum('is,s,sj->ij',self.vvi,self.fe,self.v)
            vvvvi = self.from_square(np.einsum('s,is,js->ij',self.f,self.vvi,self.vvi),idx=idx)
        vvv = np.zeros_like(vvvi)
        vvev = np.zeros_like(vvevi)
        vvvv = np.zeros_like(vvvvi)
        COMM.Reduce(vvvi,vvv,op=MPI.SUM,root=0)
        COMM.Reduce(vvevi,vvev,op=MPI.SUM,root=0)
        COMM.Reduce(vvvvi,vvvv,op=MPI.SUM,root=0)
        if RANK>0:
            return S
        self.vvv = vvv
        self.vvev = vvev  
        self.vvvv = vvvv  
        return S
    def _extract_hmean(self):
        super()._extract_hmean()
        if RANK==0:
            ehi = np.zeros(self.nparam)
            evhi = np.zeros((self.nparam,)*2)
            hhi = np.zeros(self.np2)
            vvhi = np.zeros((self.np2,self.nparam))
        else:
            ehi = np.dot(self.fe,self.h)
            evhi = np.einsum('s,si,sj->ij',self.fe,self.v,self.h)
            vvhi = np.einsum('is,s,sj->ij',self.vvi,self.f,self.h)
            self.hhi =  np.einsum('si,sj->ijs',self.h,self.h)
            self.hhi = self.from_square(self.hhi)
            hhi = np.dot(self.hhi,self.f)
        eh = np.zeros_like(ehi)
        evh = np.zeros_like(evhi)
        vvh = np.zeros_like(vvhi)
        hh = np.zeros_like(hhi)
        COMM.Reduce(ehi,eh,op=MPI.SUM,root=0)
        COMM.Reduce(evhi,evh,op=MPI.SUM,root=0)
        COMM.Reduce(vvhi,vvh,op=MPI.SUM,root=0)
        COMM.Reduce(hhi,hh,op=MPI.SUM,root=0)
        if RANK>0:
            return H
        self.eh = eh
        self.evh = evh 
        self.vvh = vvh
        self.hh = hh 
    def _get_Hmatrix(self,start=0,stop=None):
        H = super()._get_Hmatrix(start=start,stop=stop)
        if RANK==0:
            vvehi = np.zeros((self.np2,self.nparam))
            hhvi = np.zeros((self.np2,self.nparam))
            vvvhi = np.zeros((self.np2,self.nparam,self.nparam))
            hhvvi = np.zeros((self.np2,)*2)
        else:
            vvehi = np.einsum('is,s,sj->ij',self.vvi,self.fe,self.h) 
            hhvi = np.einsum('is,s,sj->ij',self.hhi,self.f,self.v)
            vvvhi = np.einsum('ks,s,si,sj->kij',self.vvi,self.f,self.v,self.h)
            hhvvi = np.einsum('is,js,s->ij',self.hhi,self.vvi,self.f)
        vveh = np.zeros_like(vvehi)
        hhv = np.zeros_like(hhvi)
        vvvh = np.zeros_like(vvvhi)
        hhvv = np.zeros_like(hhvvi)
        COMM.Reduce(vvehi,vveh,op=MPI.SUM,root=0)
        COMM.Reduce(hhvi,hhv,op=MPI.SUM,root=0)
        COMM.Reduce(vvvhi,vvvh,op=MPI.SUM,root=0)
        COMM.Reduce(hhvvi,hhvv,op=MPI.SUM,root=0)
        if RANK>0:
            return
        self.vveh = vveh
        self.hhv = hhv
        self.vvvh = vvvh
        self.hhvv = hhvv
        return H 
    def covariance(self,iprint=0):
        size = 1 + self.nparam * 3 + self.np2 + self.nparam**2
        lab = 'e','v','ev','h','vv','vh' 
        print('size=',size)
        mean = np.zeros(size)
        sigma = np.zeros((size,)*2)

        def _fill_sigma(start,stop,ls):
            start_ = start
            ix0 = len(lab)-len(ls)
            for ix,data in enumerate(ls):
                if iprint>0:
                    lab1,lab2 = lab[ix0],lab[ix0+ix]
                    print(f'cov({lab1},{lab2})=',np.linalg.norm(data))
                stop_ = start_ + data.shape[-1] 
                data = data.reshape(stop-start,stop_-start_)
                sigma[start:stop,start_:stop_] = data
                sigma[start_:stop_,start:stop] = data.T
                start_ = stop_
            return sigma

        # cov of E
        mean[0] = self.E
        vhmean = self.vhmean.flatten()
        ls = self.ee - self.E**2, \
             self.g,\
             self.eev - self.E * self.evmean, \
             self.eh - self.E * self.hmean, \
             self.evv - self.E * self.vvmean, \
             self.evh.flatten() - self.E * vhmean,
        sigma = _fill_sigma(0,1,ls)
        # cov of v
        start = 1
        stop = start + self.nparam
        print('v start, stop=',start,stop)
        mean[start:stop] = self.vmean
        evv = self.to_square(self.evv)
        vvh = self.to_square(self.vvh)
        ls = self.S,\
             evv - np.outer(self.vmean,self.evmean), \
             self.vhmean - np.outer(self.vmean,self.hmean), \
             self.vvv.T - np.outer(self.vmean,self.vvmean), \
             vvh.reshape((self.nparam,self.nparam**2)) - np.outer(self.vmean,vhmean), 
        sigma = _fill_sigma(start,stop,ls)
        # cov of ev
        start = stop
        stop = start + self.nparam
        print('ev start, stop=',start,stop)
        mean[start:stop] = self.evmean
        eevv = self.to_square(self.eevv)
        vveh = self.to_square(self.vveh) 
        ls = eevv - np.outer(self.evmean,self.evmean), \
             self.evh - np.outer(self.evmean,self.hmean), \
             self.vvev.T - np.outer(self.evmean,self.vvmean), \
             vveh.reshape((self.nparam,self.nparam**2)) - np.outer(self.evmean,vhmean)
        sigma = _fill_sigma(start,stop,ls)
        # cov of h 
        start = stop
        stop = start + self.nparam
        print('h start, stop=',start,stop)
        mean[start:stop] = self.hmean
        hh = self.to_square(self.hh)
        hhv = self.to_square(self.hhv)
        ls = hh - np.outer(self.hmean,self.hmean), \
             self.vvh.T - np.outer(self.hmean,self.vvmean), \
             hhv.transpose(0,2,1).reshape((self.nparam,self.nparam**2)) - np.outer(self.hmean,vhmean)
        sigma = _fill_sigma(start,stop,ls)
        # cov of vv
        start = stop
        stop = start + self.np2
        print('vv start, stop=',start,stop)
        mean[start:stop] = self.vvmean
        vvvv = self.to_square(self.vvvv,sh=self.np2)
        ls = vvvv - np.outer(self.vvmean,self.vvmean),\
             self.vvvh.reshape(self.np2,self.nparam**2) - np.outer(self.vvmean,vhmean)
        sigma = _fill_sigma(start,stop,ls)
        # cov of vh
        start = stop
        mean[start:] = vhmean     
        hhvv = self.to_square(self.hhvv)  
        vvhh = self.to_square(hhvv.transpose(2,0,1))  
        ls = vvhh.transpose(0,2,1,3).reshape((self.nparam**2,)*2) - np.outer(vhmean,vhmean),
        sigma = _fill_sigma(start,size,ls)
        print('symm=',np.linalg.norm(sigma-sigma.T))
        eigval = np.linalg.eigvalsh(sigma)
        print('min eig=',eigval[0])
        print('max eig=',eigval[-1])
        if self.tmpdir is None:
            return
        #f = h5py.File(self.tmpdir+f'covariance.hdf5','w')
        #f.create_dataset('mean',data=mean) 
        #f.create_dataset('sigma',data=sigma) 
        #f.close()    
