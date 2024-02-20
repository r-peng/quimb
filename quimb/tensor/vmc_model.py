import numpy as np
import itertools
import scipy.linalg
from quimb.utils import progbar as Progbar
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
    def compute_local_energy(self,config):
        #cx = np.array([xi[ci] for xi,ci in zip(self.x,config)]).prod()
        ex = sum([Hxi[ci]/xi[ci] for Hxi,xi,ci in zip(self.Hx,self.x,config)])
        vx = np.zeros((self.nsite,2))
        for i,(xi,ci) in enumerate(zip(self.x,config)):
            vx[i,ci] = 1/xi[ci]
        #print(config,vx)

        Hvx = np.zeros((self.nsite,2))
        for i,(Hxi,xi,ci) in enumerate(zip(self.Hx,self.x,config)):
            Hvx[i] = np.dot(H,vx[i])
            Hvx[i] += (ex-Hxi[ci]/xi[ci]) * vx[i]
        return ex,vx.flatten(),Hvx.flatten()
class Sampler:
    def __init__(self,af,burn_in=0,seed=None,every=1,method=1):
        self.af = af
        self.L = self.af.nsite
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.every = every 
    def sample(self):
        for _ in range(self.every):
            step = self.rng.choice([-1,1])
            sweep = range(self.L) if step==1 else range(self.L-1,-1,-1)
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
        return self.config
    def preprocess(self):
        # burn in
        if RANK==0:
            print('burn in...')
        for i in range(self.burn_in):
            self.sample()
class DenseSampler:
    def __init__(self,af,tmpdir,seed=None,**kwargs):
        self.af = af
        self.L = self.af.nsite
        self.rng = np.random.default_rng(seed)
     
        self.samples = list(itertools.product((0,1),repeat=af.nsite))
        self.ntotal = len(self.samples) 
        try:
            p = np.load(tmpdir+'p.npy')
        except FileNotFoundError:
            p = self.compute_dense_prob()
            np.save(tmpdir+'p.npy',p)
        self.p = p
    def sample(self):
        ix = self.rng.choice(self.ntotal,p=self.p) 
        return self.samples[ix] 
    def preprocess(self):
        pass
    def compute_dense_prob(self):
        if RANK==0:
            print('compute dense amplitude...')
        ptotal = np.zeros(self.ntotal)
        batchsize,remain = self.ntotal//SIZE,self.ntotal%SIZE
        count = np.array([batchsize]*SIZE)
        if remain > 0:
            count[-remain:] += 1
        disp = np.concatenate([np.array([0]),np.cumsum(count[:-1])])
        start = disp[RANK]
        stop = start + count[RANK]
        configs = self.samples[start:stop]

        plocal = [] 
        for config in configs:
            plocal.append(self.af.log_prob(config))
        plocal = np.array(plocal)
        COMM.Allgatherv(plocal,[ptotal,count,disp,MPI.DOUBLE])
        ptotal = np.exp(ptotal-np.amax(ptotal))
        return ptotal / ptotal.sum()
class VMC:
    def __init__(self):
        return
    def sample(self,sampler,samplesize,tmpdir,progbar=False):
        nparam = sampler.af.nparam
        sampler.preprocess()
        # sample
        batchsize = samplesize // SIZE
        sample_size = batchsize * SIZE
        self.e = np.zeros(batchsize)
        self.v = np.zeros((batchsize,nparam))
        self.Hv = np.zeros((batchsize,nparam))
        pg = None
        if RANK==0 and progbar:
            pg = Progbar(total=batchsize) 
        for i in range(batchsize):
            config = sampler.sample()
            self.e[i],self.v[i],self.Hv[i] = sampler.af.compute_local_energy(config)
            if pg is not None:
                pg.update() 

        # collect 
        if RANK>0:
            COMM.Send(self.e,dest=0,tag=0)
            COMM.Send(self.v,dest=0,tag=1)
            COMM.Send(self.Hv,dest=0,tag=2)
            return
        e = [self.e.copy()] 
        v = [self.v.copy()]
        Hv = [self.Hv.copy()]
        for worker in range(1,SIZE):
            COMM.Recv(self.e,source=worker,tag=0)
            COMM.Recv(self.v,source=worker,tag=1)
            COMM.Recv(self.Hv,source=worker,tag=2)
            e.append(self.e.copy())
            v.append(self.v.copy())
            Hv.append(self.Hv.copy())
        e = np.concatenate(e)
        v = np.concatenate(v,axis=0)
        Hv = np.concatenate(Hv,axis=0)
        np.save(tmpdir+f'e.npy',e)
        np.save(tmpdir+f'v.npy',v)
        np.save(tmpdir+f'Hv.npy',Hv)
    def extract(self,tmpdir): 
        e = np.load(tmpdir+'e.npy')
        v = np.load(tmpdir+'v.npy')
        Hv = np.load(tmpdir+'Hv.npy')
        samplesize = len(e)

        emean = e.sum()/samplesize
        estd = np.sqrt(((e-emean)**2).sum()/samplesize)

        vmean = np.sum(v,axis=0)/samplesize
        vstd = np.sqrt(np.sum((v-vmean)**2,axis=0)/samplesize)

        ev = e.reshape(samplesize,1) * v
        evmean = np.sum(ev,axis=0)/samplesize
        evstd = np.sqrt(np.sum((ev-evmean)**2,axis=0)/samplesize)

        Hvmean = np.sum(Hv,axis=0)/samplesize
        Hvstd = np.sqrt(np.sum((Hv-Hvmean)**2,axis=0)/samplesize)

        g = np.dot(e,v)/samplesize - emean * vmean 
        s = np.dot(v.T,v)/samplesize - np.outer(vmean,vmean)
        h = np.dot(v.T,Hv)/samplesize - np.outer(vmean,Hvmean) - np.outer(g,vmean) - emean * s 
        return samplesize,emean,g,s,h
class Exact:
    def __init__(self):
        return
    def sample(self,x,progbar=False):
        af = AmplitudeFactory(x)
        samples = list(itertools.product((0,1),repeat=af.nsite))
        nparam = af.nparam
         
        ntotal = len(samples)
        batchsize,remain = ntotal//SIZE,ntotal%SIZE
        count = np.array([batchsize]*SIZE)
        if remain > 0:
            count[-remain:] += 1
        disp = np.concatenate([np.array([0]),np.cumsum(count[:-1])])
        start = disp[RANK]
        stop = start + count[RANK]

        esum = 0.
        norm = 0.
        vsum = np.zeros(nparam) 
        Hvsum = np.zeros(nparam)
        evsum = np.zeros(nparam) 
        ssum = np.zeros((nparam,nparam))
        hsum = np.zeros((nparam,nparam)) 
        pg = None
        if RANK==SIZE-1 and progbar and count[RANK]>10:
            pg = Progbar(total=count[RANK]) 
        for ix in range(start,stop):
            config = samples[ix]
            log_px = af.log_prob(config) 
            wx = np.exp(log_px)
            norm += wx

            ex,vx,Hvx = af.compute_local_energy(config)
            esum += ex * wx
            vsum += vx * wx
            evsum += ex * vx * wx
            Hvsum += Hvx * wx 
            ssum += np.outer(vx,vx) * wx
            hsum += np.outer(vx,Hvx) * wx

            if pg is not None:
                pg.update()

        norm = np.array([norm])
        n = np.ones_like(norm)
        COMM.Reduce(norm,n,op=MPI.SUM,root=0)
        n = n[0]

        esum = np.array([esum])
        e = np.zeros_like(esum)
        COMM.Reduce(esum,e,op=MPI.SUM,root=0)
        e = e[0] / n

        v = np.zeros_like(vsum)
        COMM.Reduce(vsum,v,op=MPI.SUM,root=0)
        v /= n

        ev = np.zeros_like(evsum)
        COMM.Reduce(evsum,ev,op=MPI.SUM,root=0)
        ev /= n
        g = ev - e * v

        Hv = np.zeros_like(Hvsum)
        COMM.Reduce(Hvsum,Hv,op=MPI.SUM,root=0)
        Hv /= n

        s = np.zeros_like(ssum)
        COMM.Reduce(ssum,s,op=MPI.SUM,root=0)
        s = s/n - np.outer(v,v)

        h = np.zeros_like(hsum)
        COMM.Reduce(hsum,h,op=MPI.SUM,root=0)
        h = h/n - np.outer(v,Hv) - np.outer(g,v) - e * s 
        if RANK>0:
            return

        E,grad,ovlp,hess = make_matrices(x)
        print('L', len(x))
        print('n',n)
        print('ener',np.fabs(e-E)/E)
        print('g',np.linalg.norm(grad-g))
        print('ovlp',np.linalg.norm(s-ovlp))
        print('hess',np.linalg.norm(h-hess))
