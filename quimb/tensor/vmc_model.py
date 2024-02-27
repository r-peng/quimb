import numpy as np
import itertools
import scipy.linalg
from .tensor_vmc import Progbar,ExchangeSampler,RGN
from .tensor_vmc import DenseSampler as DenseSampler_ 
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
def sample_matrices(sampler,sample_size=None,tmpdir=None):
    sampler.config = tuple([0] * sampler.nsite)
    vmc = RGN(sampler,normalize=False,solve_full=True,solve_dense=True)
    vmc.batchsize = sample_size
    vmc.tmpdir = tmpdir
    vmc.step = 0
    vmc.Eold = 0
    vmc.nparam = vmc.sampler.af.nparam
    vmc.sample(save_config=False)
    vmc.extract_energy_gradient()
    vmc.extract_S(solve_full=True,solve_dense=True)
    vmc.extract_H(solve_full=True,solve_dense=True)
    if tmpdir is None:
        if RANK>0:
            return [None] * 4 
        return vmc.E,vmc.g,vmc.S,vmc.H-vmc.E * vmc.S
    vmc.save_grad_hess = True
    vmc.cond1 = 1e-3
    vmc.cond2 = 1e-3
    vmc.rate1 = .1
    vmc.rate2 = .5
    vmc._transform_gradients_rgn_dense(solve_full=True,enforce_pos=False)
class AmplitudeFactory:
    def __init__(self,x):
        self.x = x
        self.Hx = [np.dot(H,xi) for xi in x]
        self.nsite = len(x)
        self.nparam = 2 * self.nsite
    def get_x(self):
        return np.array(self.x).flatten()
    def log_prob(self,config):
        return np.log(np.array([xi[ci] for xi,ci in zip(self.x,config)])**2).sum()
    def compute_local_energy(self,config,compute_v=True,compute_Hv=True):
        cx = np.array([xi[ci] for xi,ci in zip(self.x,config)]).prod()
        ex = sum([Hxi[ci]/xi[ci] for Hxi,xi,ci in zip(self.Hx,self.x,config)])
        vx = None
        if compute_v:
            vx = np.zeros((self.nsite,2))
            for i,(xi,ci) in enumerate(zip(self.x,config)):
                vx[i,ci] = 1/xi[ci]
        #print(config,vx)

        Hvx = None
        if compute_Hv:
            Hvx = np.zeros((self.nsite,2))
            for i,(Hxi,xi,ci) in enumerate(zip(self.Hx,self.x,config)):
                Hvx[i] = np.dot(H,vx[i])
                Hvx[i] += (ex-Hxi[ci]/xi[ci]) * vx[i]

        if compute_v:
            vx = vx.flatten()
        if compute_Hv:
            Hvx = Hvx.flatten()
        return cx,ex,vx,Hvx,0.
    def update(self,x,fname=None,root=0):
        self.x = np.array([wf(xi) for xi in x.reshape(self.nsite,2)])
        #self.x = x.reshape(self.nsite,2)
    def parse_config(self,config):
        return config
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
class DenseSampler(DenseSampler_):
    def __init__(self,nsite,exact=False,seed=None,thresh=1e-28):
        super().__init__(nsite,None,exact=exact,seed=seed,thresh=thresh)
    def get_all_configs(self,fix_sector=None):
        return list(itertools.product((0,1),repeat=self.nsite))
