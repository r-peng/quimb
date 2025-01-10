import numpy as np
import itertools,h5py
import scipy.linalg
from .tensor_vmc import (
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


#def wf(x):
#    return x/np.linalg.norm(x)
#def deriv_wf2(x,eps=1e-4):
#    g= np.zeros([2, 2])
#    g[:,0] = (wf(x+np.array([eps,0.]))-wf(x))/eps
#    g[:,1] = (wf(x+np.array([0.,eps]))-wf(x))/eps
#    return g
#def deriv_wf(x):
#    return np.eye(2) - np.outer(x,x) 
#def init(L,fname,eps=1):
#    x = np.ones((L,2))
#    x[:,1] = -1
#    x += np.random.normal(size=(L,2)) * eps 
#    x = [wf(xi) for xi in x]
#    x = np.array(x)
#    np.save(fname+'.npy',x)
#    print(x)
#def make_matrices(x,compute_g=True,compute_ovlp=True,compute_hess=True):
#    L = len(x)
#    e = [np.dot(np.dot(xi,H),xi) for xi in x] 
#    E = sum(e)
#    if not compute_g:
#        return E
#    g = [deriv_wf(xi) for xi in x]
#    xHg = [np.dot(np.dot(xi,H),gi) for gi,xi in zip(g,x)]
#    gvec = np.array([xHgi + (E-ei) * xgi for xHgi,ei,xgi in zip(xHg,e,xg)]).flatten()
#    if not compute_ovlp:
#        return E,gvec 
#    #g = [deriv_wf2(xi) for xi in x]
#    ovlp = np.zeros((L,2,L,2))
#    xg = [np.dot(xi,gi) for gi,xi in zip(g,x)]
#    for i in range(L):
#        ovlp[i,:,i,:] = np.dot(g[i].T,g[i])
#        for j in range(i+1,L):
#            ovlp_ij = np.outer(xg[i],xg[j])
#            ovlp[i,:,j,:] = ovlp_ij 
#            ovlp[j,:,i,:] = ovlp_ij.T
#    if not compute_hess:
#        return E,gvec,ovlp.reshape(2*L,2*L)
#
#    hess = np.zeros((L,2,L,2))
#    for i in range(L):
#        hess[i,:,i,:] = np.dot(g[i].T,np.dot(H,g[i]))
#        hess[i,:,i,:] += (E-e[i]) * ovlp[i,:,i,:] 
#        for j in range(i+1,L):
#            hess_ij = np.outer(xHg[i],xg[j])   
#            hess_ij += np.outer(xg[i],xHg[j])   
#            hess_ij += (E-e[i]-e[j]) * ovlp[i,:,j,:]
#            hess[i,:,j,:] = hess_ij
#            hess[j,:,i,:] = hess_ij.T
#
#    ovlp = ovlp.reshape(L*2,L*2)
#    hess = hess.reshape(L*2,L*2) - E * ovlp
#    return E,gvec,ovlp,hess
#def optimize(x,maxiter,tmpdir,rate1,rate2=.5,eps=.001,save_every=1):
#    L,_ = x.shape
#    for step in range(maxiter):
#        E,g,ovlp,hess = make_matrices(x)
#        err = abs(-1.-E/L)
#        print(f'step={step},e={E/L},err={err}')
#        ovlp += eps * np.eye(L*2)
#        p = np.linalg.solve(hess+ovlp/rate2,g)
#        x = x.flatten()
#        x -= rate1 * p
#        x = np.array([wf(xi) for xi in x.reshape(L,2)])
#        if step%save_every==0:
#            np.save(f'{tmpdir}step{step}.npy',x) 
#        if err<1e-6:
#            break
#def sample_matrices(sampler,samplesize=None,tmpdir=None):
#    sampler.config = tuple([0] * sampler.nsite)
#    vmc = RGN(sampler,normalize=False,solve_dense=True,solve_full=True)
#    vmc.progbar = True
#    vmc.step = 0
#    vmc.Eold = 0
#    vmc.tmpdir = tmpdir
#    vmc.batchsize = samplesize
#    vmc.nparam = vmc.sampler.af.nparam
#    vmc.rate2 = .5
#    vmc.sample(save_config=False)
#    vmc.extract_energy_gradient()
#
##    vmc.eigen_thresh = 1e-3 
##    deltas_sr = vmc._transform_gradients_sr()
#    vmc.solve_symmetric = True
#    vmc.cond1 = .001
#    vmc.rate1 = .1
#    vmc.rate2 = .5
#    vmc._transform_gradients_rgn(solve_full=True,solve_dense=False)
#    return
##    return deltas_sr,deltas_rgn
##
##    if RANK==0:
##        print(deltas0)
##        print(deltas1)
#    exit()
##
##
#    vmc.extract_S(solve_full=True,solve_dense=True)
#    vmc.extract_H(solve_full=True,solve_dense=True)
#    vmc._save_grad_hess()
#    return vmc._extract_Hcov()
##    if not exact_variance:
##        return 
##    if RANK>0:
##        return
##    vmc.covariance() 
#def from_noisy(mean,ns,nparam,sample_size):
#    x = mean + ns/np.sqrt(sample_size+1e-10)
#    np2 = (1+nparam) * nparam // 2
#
#    E,x = x[:1],x[1:]
#    v,x = x[:nparam],x[nparam:]
#    ev,x = x[:nparam],x[nparam:]
#    h,x = x[:nparam],x[nparam:]
#    vv,vh = x[:np2],x[np2:]
#
#    g = ev - E * v
#    S = to_square(vv,sh=nparam) - np.outer(v,v)
#    hess = vh.reshape(nparam,nparam) - np.outer(v,h) - np.outer(g,v) - E * S
#    return E,g,S,hess
class AmplitudeFactory:
    def __init__(self,x):
        self.x = x
        self.prob = self.x**2
        self.X = np.array([[0., 1.], [1., 0.]])
        self.Hx = [np.dot(self.X,xi) for xi in x]
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
                hx[i] = np.dot(self.X,vx[i])
                hx[i] += (ex-Hxi[ci]/xi[ci]) * vx[i]

        if compute_v:
            vx = vx.flatten()
        if compute_h:
            hx = hx.flatten()
        return cx,ex,vx,hx,0.
    def update(self,x,fname=None,root=0):
        self.x = np.array([wf(xi) for xi in x.reshape(self.nsite,2)])
        self.Hx = [np.dot(self.X,xi) for xi in self.x]
        if fname is None:
            return
        if RANK==root:
            np.save(fname+'.npy',self.x)
    def parse_config(self,config):
        return config
    def make_matrices(self,compute_g=True,compute_ovlp=True,compute_hess=True,compute_Hcov=False):
        L = self.nsite
        e = np.array([np.dot(xi,Hxi) for xi,Hxi in zip(self.x,self.Hx)])
        E = sum(e)
        E2 = L + 2 * sum([e[i]*e[j] for i in range(L) for j in range(i+1,L)]) 
        varE = E2 - E**2
        if not compute_g:
            return E,varE
        
        ev = np.array([Hxi+xi*(E-ei) for Hxi,xi,ei in zip(self.Hx,self.x,e)]).flatten()
        v = self.x.flatten()
        g = ev - E * v 
        if not compute_ovlp:
            return E,g

        S = np.zeros((L,2,L,2))
        for i in range(L):
            S[i,:,i,:] = np.eye(2)
            for j in range(i+1,L):
                Sij = np.outer(self.x[i],self.x[j])
                S[i,:,j,:] = Sij
                S[j,:,i,:] = Sij.T
        S = S.reshape(L*2,L*2)
        S -= np.outer(v,v)
        if not compute_hess:
            return E,g,S
        
        H1 = np.zeros((L,2,L,2))
        for i in range(L):
            H1[i,:,i,:] = self.X + np.eye(2)*(E-e[i])           
            for j in range(i+1,L):
                Hij = np.outer(self.Hx[i],self.x[j]) 
                Hij += np.outer(self.x[i],self.Hx[j])
                Hij += np.outer(self.x[i],self.x[j]) * (E-e[i]-e[j]) 
                H1[i,:,j,:] = Hij
                H1[j,:,i,:] = Hij.T
        hess = H1.reshape(L*2,L*2)
        Hv = ev 
        hess -= np.outer(v,Hv)
        hvcov = hess.copy()
        hess -= np.outer(g,v)
        hess -= E * S
        if not compute_Hcov:
            return E,g,S,hess,hvcov

        H2 = np.zeros((L,2,L,2)) 
        for i in range(L):
            H2[i,:,i,:] = L*np.eye(2)
            hii = np.zeros((2,2))
            for k in range(L):
                for l in range(k+1,L):
                    if k==i:
                        hii += e[l] * H
                    else:
                        if l==i:
                            hii += e[k] * H
                        else:
                            hii += e[k] * e[l] *np.eye(2)
            H2[i,:,i,:] += 2*hii
            for j in range(i+1,L):
                hij = np.outer(self.x[i],self.x[j]) * L
                H2[i,:,j,:] = hij
                H2[j,:,i,:] = hij.T
                hij = np.zeros((2,2))
                for k in range(L):
                    for l in range(k+1,L):
                        if k==i:
                            if l==j:
                                hij += np.outer(self.Hx[i],self.Hx[j])
                            else:
                                hij += np.outer(self.Hx[i],self.x[j])*e[l]
                        elif k==j:
                            hij += np.outer(self.x[i],self.Hx[j])*e[l]
                        else:
                            if l==i:
                                hij += np.outer(self.Hx[i],self.x[j])*e[k]
                            elif l==j:
                                hij += np.outer(self.x[i],self.Hx[j])*e[k]
                            else:
                                hij += np.outer(self.x[i],self.x[j])*e[k]*e[l]
                H2[i,:,j,:] += 2*hij
                H2[j,:,i,:] += 2*hij.T
        H2 = H2.reshape(2*L,2*L) - np.outer(Hv,Hv)
        return E,g,S,hess,hvcov,H2
class ModelSampler(ExchangeSampler):
    def __init__(self,af,seed=None,every=1,burn_in=10):
        self.af = af
        self.nsite = self.af.nsite
        self.rng = np.random.default_rng(seed)
        self.every = every 
        self.exact = False
        self.px = None
        self.burn_in = burn_in
    def sample(self):
        for _ in range(self.every):
            step = self.rng.choice([-1,1])
            sweep = range(self.nsite) if step==1 else range(self.nsite-1,-1,-1)
            for i in sweep:
                i_old = self.config[i]
                i_new = 1-i_old
                p = self.af.prob[i]
                acceptance = p[i_new]/p[i_old]
                if acceptance < self.rng.uniform(): # reject
                    continue

                config_new = list(self.config)
                config_new[i] = i_new 
                self.config = tuple(config_new)
        return self.config,None
class ModelDenseSampler(DenseSampler):
    def __init__(self,af,exact=False,seed=None,thresh=1e-28):
        self.af = af
        super().__init__(af.nsite,None,exact=exact,seed=seed,thresh=thresh)
    def get_all_configs(self,fix_sector=None):
        return list(itertools.product((0,1),repeat=self.nsite))
class Model:
    def __init__(self,theta,ham=None):
        self.nsite = self.nparam = len(theta)
        self.theta = theta
        self.wfn = np.stack([np.sin(theta),np.cos(theta)],axis=1)
        self.prob = self.wfn**2
        self.ham = np.ones(self.nsite) if ham is None else ham
    def parse_config(self,config):
        return config
    def get_x(self):
        return self.theta
    def log_prob(self,config):
        return np.log(np.array([xi[ci] for xi,ci in zip(self.prob,config)])).sum()
    def amplitude(self,x):
        ls = np.array([self.wfn[i,xi] for i,xi in enumerate(x)])
        return np.prod(ls)
    def eloc(self,x):
        ls = np.array([self.wfn[i,1-xi]/self.wfn[i,xi] for i,xi in enumerate(x)])
        return np.dot(ls,self.ham)
    def nu(self,x):
        ls = np.array([self.wfn[i,1-xi]/self.wfn[i,xi]*(-1)**xi for i,xi in enumerate(x)])
        return ls
    def h(self,x):
        ex = self.ham*np.array([self.wfn[i,1-xi]/self.wfn[i,xi] for i,xi in enumerate(x)])
        vx = np.array([self.wfn[i,1-xi]/self.wfn[i,xi]*(-1)**xi for i,xi in enumerate(x)])
        hx2 = self.ham*np.power(-np.ones(self.nsite),x)
        hx = vx * (ex.sum() - ex) - hx2 
        return hx
    def compute_local_energy(self,config,compute_v=True,compute_h=True):
        cx = self.amplitude(config) 
        ex = self.eloc(config)
        vx = None
        if compute_v:
            vx = self.nu(config)
        hx = None
        if compute_h:
            hx = self.h(config)
        return cx,ex,vx,hx,0.
    def update(self,theta,fname=None,root=0):
        self.theta = theta 
        self.wfn = np.stack([np.sin(theta),np.cos(theta)],axis=1)
        self.prob = self.wfn**2
        if fname is None:
            return
        if RANK==root:
            np.save(fname+'.npy',self.theta)
    def exact_expectation(self):
        self.e = np.prod(self.wfn,axis=1)*2*self.ham
        self.esum = self.e.sum()
        self.eqsum = self.quad_sum0()

        self.g = (self.wfn[:,1]**2-self.wfn[:,0]**2)*self.ham
        self.S = np.eye(self.nsite)
        self.H = np.eye(self.nsite)
        np.fill_diagonal(self.H,self.esum-2*self.e)
    def exact_variance(self):
        N = L = self.nsite
        wfn = self.wfn

        esq = self.e**2
        esqsum = esq.sum()
        self.evar = L - esqsum 

        #mu = wfn[:,1]**4/wfn[:,0]**2+wfn[:,0]**4/wfn[:,1]**2 
        mu = 4/self.e**2-3
        #xi = wfn[:,1]**3/wfn[:,0]+wfn[:,0]**3/wfn[:,1]
        xi = 2/self.e-self.e
        #self.gvar = mu + L - 1 + 2*self.e*(self.e-xi) - esqsum - self.g**2
        self.gvar = 4*(self.e-1/self.e)**2 + L - esqsum - self.g**2

        self.Svar = np.ones((N,N)) 
        np.fill_diagonal(self.Svar,4/esq-4)

        self.Hvar = np.zeros((N,N))
        for i in range(N):
            esumi = self.esum-self.e[i]
            self.Hvar[i,i] = 1-2*xi[i]*esumi+mu[i]*(L-1+self.quad_sum1(i))

            self.Hvar[i,i] += 3*self.g[i]**2-2*self.g[i]*(wfn[i,1]**3/wfn[i,0]-wfn[i,0]**3/wfn[i,1])*esumi
            self.Hvar[i,i] -= (self.esum-2*self.e[i])**2
            for j in range(N):
                if i==j:
                    continue
                self.Hvar[i,j] = mu[i]-2*xi[i]*self.e[j]+L-1+2*(xi[i]-self.e[j])*(esumi-self.e[j])+self.quad_sum2(i,j)
                self.Hvar[i,j] -= self.g[j]**2

        self.hess_var = np.zeros((N,N))
        np.fill_diagonal(self.hess_var,9*(1-esq)+(4/esq-3)*(L-1-esqsum+esq))
        for i in range(N):
            for j in range(N):
                if i==j:
                    continue
                self.hess_var[i,j] = 4*(self.e[i]+self.e[j]-1/self.e[i])**2+L-esqsum
#    def exact_variance(self):
#        N = self.nsite
#        wfn = self.wfn
#
#        ham_sq = self.ham**2
#        ham_sq_sum = ham_sq.sum()
#        self.evar_1 = ham_sq_sum + self.eqsum 
#        self.evar = ham_sq_sum - (self.e**2).sum()
#
#        mu = wfn[:,1]**4/wfn[:,0]**2+wfn[:,0]**4/wfn[:,1]**2 
#        xi = wfn[:,1]**3/wfn[:,0]+wfn[:,0]**3/wfn[:,1]
#        self.gvar_1 = (mu-1)*ham_sq + ham_sq_sum + 2*self.ham*xi*(self.esum-self.e)
#        for i in range(N):
#            self.gvar_1[i] += self.quad_sum1(i) 
#        self.gvar = self.gvar_1 - self.g**2
#
#        self.gvar2 = mu + N - 1 + 2*(self.e-xi)*self.e - (self.e**2).sum()
#
#        self.Svar_1 = np.ones((N,N)) 
#        np.fill_diagonal(self.Svar_1,mu)
#        self.Svar = self.Svar_1 - np.eye(N)
#
#        self.Hvar_1 = np.zeros((N,N))
#        tmp = np.zeros((N,N))
#        for i in range(N):
#            self.Hvar_1[i,:] = ham_sq[i]
#            for j in range(N):
#                if i==j:
#                    tmp[i,i] = self.ham[i]*xi[i]*(self.esum-self.e[i])
#                else:
#                    tmp[i,j] = self.e[i]*(self.ham[j]*xi[j]+self.esum-self.e[i]-self.e[j])
#        self.Hvar_1 -= 2*tmp
#        
#        tmp = np.zeros((N,N))
#        for i in range(N):
#            for j in range(N):
#                if i==j:
#                    tmp[i,i] = mu[i]*(self.quad_sum1(i)+ham_sq_sum-ham_sq[i])
#                else:
#                    tmp[i,j] = mu[j]*ham_sq[j] + ham_sq_sum - ham_sq[i] - ham_sq[j] + 2*xi[j]*self.ham[j]*(self.esum-self.e[i]-self.e[j]) + self.quad_sum2(i,j)
#        self.Hvar_1 += tmp
#
#        self.Hvar = np.zeros((N,N))
#        for i in range(N):
#            for j in range(N):
#                if i==j:
#                    self.Hvar[i,i] = ham_sq[i]*(1-mu[i])-2*self.ham[i]*xi[i]*(self.esum-self.e[i])+mu[i]*ham_sq_sum
#                    self.Hvar[i,i] += mu[i]*self.quad_sum1(i)-(self.esum-2*self.e[i])**2
#                else:
#                    self.Hvar[i,j] = ham_sq[j]*(mu[j]-1)-2*self.e[i]*self.ham[j]*xi[j]+ham_sq_sum
#                    self.Hvar[i,j] += 2*(self.ham[j]*xi[j]-self.e[i])*(self.esum-self.e[i]-self.e[j])+self.quad_sum2(i,j)
#        assert np.linalg.norm(self.Hvar_1-self.H**2-self.Hvar)<1e-6
    def hilbert_sum(self,thresh=1e-10,gs=False):
        e = 0
        evar = 0

        v = 0
        g = 0
        gvar = 0 

        S = 0 
        Svar = 0 

        h = 0
        H = 0
        hess = 0
        Hvar = 0
        hess_var = 0
        print('check expresion...')
        for x in itertools.product((0,1),repeat=self.nsite):
            cx = self.amplitude(x)
            px = cx**2
        
            ex = self.eloc(x)
            e += px * ex
            evar += px * ex**2
        
            vx = self.nu(x)
            v += px * vx
            evx = (ex-self.esum)*vx
            g += px * evx 
            gvar += px * evx**2
        
            Sx = np.outer(vx,vx)
            S += px * Sx 
            Svar += px * Sx**2
        
            hx = self.h(x)
            h += px*hx
            Hx = np.outer(vx,hx-self.g)
            H += px * Hx 
            Hvar += px * Hx**2 

            hess_x = np.outer(vx,hx) - self.esum*Sx
            hess += px*hess_x
            hess_var += px*hess_x**2
         
        evar -= e**2
        if gs:
            print('evar=',evar)
        assert abs(evar-self.evar)<thresh 
        assert abs(e-self.esum)<thresh 

        gvar -= g**2
        if gs:
            print('gvar=',np.linalg.norm(gvar))
        assert np.linalg.norm(gvar-self.gvar)<thresh
        assert np.linalg.norm(g-self.g)<thresh
        assert np.linalg.norm(v)<thresh

        Svar -= S**2
        if gs:
            print(Svar)
        assert np.linalg.norm(Svar-self.Svar)<thresh
        assert np.linalg.norm(S-self.S)<thresh

        Hvar -= H**2
        hess_var -= hess**2
        if gs:
            print(Hvar)
            print(hess_var)
        assert np.linalg.norm(h-g)<thresh
        assert np.linalg.norm(H-self.H)<thresh
        assert np.linalg.norm(Hvar-self.Hvar)<thresh
        hess_ = self.H-self.esum*self.S
        assert np.linalg.norm(hess-hess_)<thresh
        assert np.linalg.norm(hess_var-self.hess_var)<thresh
        print('expression checked')
    def quad_sum0(self):
        return self.esum**2-(self.e**2).sum()
    def quad_sum1(self,i):
        return self.eqsum - 2*self.e[i]*(self.esum-self.e[i])
    def quad_sum2(self,i,j):
        eij = self.e[i] + self.e[j]
        return self.eqsum - 2*eij*(self.esum-eij) - 2*self.e[j]*self.e[i]
class ModelShift(Model):
    def __init__(self,theta,shift,ham=None):
        self.shift = shift
        super().__init__(theta,ham=ham)
        self.wfn += self.shift 
        self.prob = self.wfn**2
def init(N,nrun,perr,scale=0.01):
    ei = np.random.normal(loc=perr/100,scale=scale,size=(N,nrun))-1
    assert len(ei[ei<-1])==0
    return np.arcsin(ei)/2
def get_MG_state(theta,shift=None):
    ls = []
    for i,theta_i in enumerate(theta):
        sin,cos = np.sin(theta_i),np.cos(theta_i)
        if shift is None: # derivative
            ls.append(np.array([[0,-sin],[cos,0]]))
        else:
            ls.append(np.array([[0,cos],[sin,0]])+shift*np.eye(2))
    return ls
class MG:
    def __init__(self,theta,shift=0.,pbc=False,J1=1,J2=0.5):
        self.theta = theta
        self.shift = shift
        self.wfn = get_MG_state(theta,shift=shift)
        self.dwfn = get_MG_state(theta)
        self.J1 = J1
        self.J2 = J2
        self.pbc = False
        self.nsite = len(theta)*2
        self.nparam = len(theta)
    def parse_config(self,config):
        return config
    def get_x(self):
        return self.theta
    def amplitude(self,config):
        amp = 1
        for i,M in enumerate(self.wfn):
            s1,s2 = config[2*i],config[2*i+1] 
            amp *= M[s1,s2]
        return amp
    def log_prob(self,config):
        return np.log(self.amplitude(config)**2)
    def eloc_term(self,i,j,config):
        s1,s2 = config[i],config[j] 
        if s1==s2:
            return 0
        s1_new,s2_new = 1-s1,1-s2
        if i%2==0 and i+1==j:
            # oo ** oo
            M = self.wfn[i//2]
            return M[s1_new,s2_new]/M[s1,s2]
        if i%2==1 and i+1==j:
            # o* *o
            s0,s3 = config[i-1],config[j+1]
            M1,M2 = self.wfn[i//2],self.wfn[j//2]
            return M1[s0,s1_new]*M2[s2_new,s3]/M1[s0,s1]/M2[s2,s3]
        if i%2==0 and i+2==j:
            # *o *o
            s0,s3 = config[i+1],config[j+1]
            M1,M2 = self.wfn[i//2],self.wfn[j//2]
            return M1[s1_new,s0]*M2[s2_new,s3]/M1[s1,s0]/M2[s2,s3]
        if i%2==1 and i+2==j:
            # o* o*
            s0,s3 = config[i-1],config[j-1]
            M1,M2 = self.wfn[i//2],self.wfn[j//2]
            return M1[s0,s1_new]*M2[s3,s2_new]/M1[s0,s1]/M2[s3,s2]
    def nu(self,config):
        nu = np.zeros(len(self.wfn))
        for i,M in enumerate(self.wfn):
            s1,s2 = config[2*i],config[2*i+1] 
            nu[i] = self.dwfn[i][s1,s2]/M[s1,s2]
        return nu
    def compute_local_energy_eigen(self,config):
        e = np.zeros(2) 
        for d in (1,2):
            for i in range(self.nsite):
                s1 = (-1) ** config[i]
                if i+d<self.nsite:
                    e[d-1] += s1 * (-1)**config[i+d]
                else:
                    if self.pbc:
                        e[d-1] += s1 * (-1)**config[(i+d)%self.nsite]
        return .25 * (e[0]*self.J1 + e[1]*self.J2) 
    def hterm(self,i,j,k,config):
        s1,s2 = config[i],config[j] 
        if s1==s2:
            return 0
        s1_new,s2_new = 1-s1,1-s2
        idx = i//2
        k1,k2 = config[2*k],config[2*k+1]    
        D,Mk = self.dwfn[k],self.wfn[k]
        if i%2==0 and i+1==j:
            # oo ** oo
            if k==idx:
                return D[s1_new,s2_new]/Mk[s1,s2]
            else:
                M = self.wfn[idx]
                return M[s1_new,s2_new]*D[k1,k2]/M[s1,s2]/Mk[k1,k2]
        if i%2==1 and i+1==j:
            # o* *o
            s0,s3 = config[i-1],config[j+1]
            if k==idx:
                M = self.wfn[idx+1]
                return D[s0,s1_new]*M[s2_new,s3]/Mk[s0,s1]/M[s2,s3]
            elif k==idx+1:
                M = self.wfn[idx]
                return M[s0,s1_new]*D[s2_new,s3]/M[s0,s1]/Mk[s2,s3]
            else:
                M1,M2 = self.wfn[idx],self.wfn[idx+1]
                return M1[s0,s1_new]*M2[s2_new,s3]*D[k1,k2]/M1[s0,s1]/M2[s2,s3]/Mk[k1,k2]
        if i%2==0 and i+2==j:
            # *o *o
            s0,s3 = config[i+1],config[j+1]
            if k==idx:
                M = self.wfn[idx+1]
                return D[s1_new,s0]*M[s2_new,s3]/Mk[s1,s0]/M[s2,s3]
            elif k==idx+1:
                M = self.wfn[idx]
                return M[s1_new,s0]*D[s2_new,s3]/M[s1,s0]/Mk[s2,s3]
            else:
                M1,M2 = self.wfn[idx],self.wfn[idx+1]
                return M1[s1_new,s0]*M2[s2_new,s3]*D[k1,k2]/M1[s1,s0]/M2[s2,s3]/Mk[k1,k2]
        if i%2==1 and i+2==j:
            # o* o*
            s0,s3 = config[i-1],config[j-1]
            if k==idx:
                M = self.wfn[idx+1]
                return D[s0,s1_new]*M[s3,s2_new]/Mk[s0,s1]/M[s3,s2]
            elif k==idx+1:
                M = self.wfn[idx]
                return M[s0,s1_new]*D[s3,s2_new]/M[s0,s1]/Mk[s3,s2]
            else:
                M1,M2 = self.wfn[idx],self.wfn[idx+1]
                return M1[s0,s1_new]*M2[s3,s2_new]*D[k1,k2]/M1[s0,s1]/M2[s3,s2]/D[k1,k2]
    def compute_local_energy(self,config,compute_v=True,compute_h=True):
        cx = self.amplitude(config) 
        ex = 0 
        for d,J in zip((1,2),(self.J1,self.J2)):
            for i in range(len(config)-d):
                if config[i]==config[i+d]:
                    continue
                ex += J*self.eloc_term(i,i+d,config)
        u = self.compute_local_energy_eigen(config)
        ex = ex*.5 + u 
        vx = None
        if compute_v:
            vx = self.nu(config)
        hx = None
        if not compute_h:
            return cx,ex,vx,hx,0.
        hx = np.zeros(len(self.wfn))
        for d,J in zip((1,2),(self.J1,self.J2)):
            for i in range(len(config)-d):
                if config[i]==config[i+d]:
                    continue
                for k in range(len(hx)):
                    hx[k] += J*self.hterm(i,i+d,k,config)
        hx = hx*.5 + u*vx 
        return cx,ex,vx,hx,0.
    def update(self,theta,fname=None,root=0):
        self.theta = theta 
        self.wfn = get_MG_state(theta,shift=self.shift)
        self.dwfn = get_MG_state(theta)
        if fname is None:
            return
        if RANK==root:
            np.save(fname+'.npy',self.theta)

class MGSampler(ExchangeSampler):
    def __init__(self,af,seed=None,every=1,burn_in=10):
        self.af = af
        self.nsite = self.af.nsite
        self.rng = np.random.default_rng(seed)
        self.every = every 
        self.exact = False
        self.px = None
        self.burn_in = burn_in
    def sample(self):
        for _ in range(self.every):
            step = self.rng.choice([-1,1])
            sweep = range(self.nsite-1) if step==1 else range(self.nsite-2,-1,-1)
            for i in sweep:
                i1,i2 = self.config[i],self.config[i+1]
                if i1==i2:
                    continue
                if i%2==0:
                    M = self.af.wfn[i//2]
                    acceptance = (M[i2,i1]/M[i1,i2])**2
                else:
                    M1,M2 = self.af.wfn[i//2],self.af.wfn[i//2+1]
                    i0,i3 = self.config[i-1],self.config[i+2]
                    acceptance = (M1[i0,i2]*M2[i1,i3]/M1[i0,i1]/M2[i2,i3])**2
                if acceptance < self.rng.uniform(): # reject
                    continue

                config_new = list(self.config)
                config_new[i],config_new[i+1] = i2,i1 
                self.config = tuple(config_new)
        return self.config,None
