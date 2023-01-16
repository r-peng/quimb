import time,h5py,itertools
import numpy as np
import scipy.sparse.linalg as spla

from quimb.utils import progbar as Progbar
from .utils import (
    rand_fname,load_ftn_from_disc,write_ftn_to_disc,
    vec2psi,
)
from .fermion_2d_vmc import (
    thresh,config_map,config_order,
    AmplitudeFactory2D,ExchangeSampler2D,
    compute_amplitude_2d,get_constructors_2d,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
################################################################################
# MPI stuff
################################################################################    
def distribute(ntotal):
    batchsize,remain = ntotal // SIZE, ntotal % SIZE
    batchsizes = [batchsize] * SIZE
    for worker in range(SIZE-remain,SIZE):
        batchsizes[worker] += 1
    ls = [None] * SIZE
    start = 0
    for worker in range(SIZE):
        stop = start + batchsizes[worker]
        ls[worker] = start,stop
        start = stop
    return ls
def parallelized_looped_fxn(fxn,ls,args):
    stop = min(SIZE,len(ls))
    results = [None] * stop 
    for worker in range(stop-1,-1,-1):
        worker_info = fxn,ls[worker],args 
        if worker > 0:
            COMM.send(worker_info,dest=worker) 
        else:
            results[0] = fxn(ls[0],*args)
    for worker in range(1,stop):
        results[worker] = COMM.recv(source=worker)
    return results
def worker_execution():
    """
    Simple function for workers that waits to be given
    a function to call. Once called, it executes the function
    and sends the results back
    """
    # Create an infinite loop
    while True:

        # Loop to see if this process has a message
        # (helps keep processor usage low so other workers
        #  can use this process until it is needed)
        while not COMM.Iprobe(source=0):
            time.sleep(0.01)

        # Recieve the assignments from RANK 0
        assignment = COMM.recv()

        # End execution if received message 'finished'
        if assignment == 'finished': 
            break
        # Otherwise, call function
        fxn,local_ls,args = assignment
        result = fxn(local_ls,*args)
        COMM.send(result, dest=0)
################################################################################
# Sampler  
################################################################################    
GEOMETRY = '2D' 
def dense_amplitude_wrapper_2d(info,psi,all_configs,direction,compress_opts):
    start,stop = info
    psi = load_ftn_from_disc(psi)
    psi = psi.reorder(direction,inplace=True)

    p = np.zeros(len(all_configs)) 
    cache_head = dict()
    cache_mid = dict()
    cache_tail = dict()
    for ix in range(start,stop):
        config = all_configs[ix]
        unsigned_amp,cache_head,cache_mid,cache_tail = \
            compute_amplitude_2d(psi,config,direction,0,
            cache_head,cache_mid,cache_tail,**compress_opts)
        p[ix] = unsigned_amp**2
    return p
class DenseSampler:
    def __init__(self,sampler_opts,**contract_opts):
        if GEOMETRY=='2D':
            self.nsite = sampler_opts['Lx'] * sampler_opts['Ly']
            self.direction = config_order 
        else:
            raise NotImplementedError
        self.nelec = sampler_opts['nelec']
        self.symmetry = sampler_opts['symmetry']
        self.rng = np.random.default_rng(sampler_opts.get('seed',None))
        self.contract_opts = contract_opts
        self.configs = self.get_all_configs()
    def get_all_configs(self):
        if self.symmetry=='u1':
            return self.get_all_configs_u1()
        elif self.symmetry=='u11':
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
    def _set_psi(self,psi=None,load_fname=None,write_fname=None):
        nconfig = len(self.configs)
        if load_fname is None:
            t0 = time.time()
            self.p = np.zeros(nconfig)
            if GEOMETRY=='2D':
                fxn = dense_amplitude_wrapper_2d
            else:
                raise NotImplementedError
            infos = distribute(nconfig)
            args = psi,self.configs,self.direction,self.contract_opts
            ls = parallelized_looped_fxn(fxn,infos,args)
            for p in ls:
                self.p += p
            self.p /= np.sum(self.p) 
            print(f'dense sampler updated ({time.time()-t0}s).')
        else:
            f = h5py.File(load_fname,'r')
            self.p = f['p'][:]
            f.close()
        self.flat_indexes = list(range(nconfig))
        if write_fname is not None:
            f = h5py.File(write_fname,'w')
            f.create_dataset('p',data=self.p)
            f.close()
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        omega = self.p[flat_idx]
        info = self.configs[flat_idx],self.direction,0
        return info,omega
PRECISION = 1e-15
class MetropolisHastingsSampler:
    def __init__(
        self,
        sub_sampler,
        sampler_opts,
        amplitude_factory=None,
    ):
        self.sub_sampler = sub_sampler

        if amplitude_factory is not None:
            self.prob_fn = amplitude_factory.prob
        else:
            # will initialize later
            self.prob_fn = None

        initial = sampler_opts.get('initial',None)
        if initial is not None:
            self.config, self.omega, self.prob = initial
        else:
            self.config = self.omega = self.prob = None

        self.seed = sampler_opts.get('seed',None) 
        self.rng = np.random.default_rng(self.seed)
        self.accepted = 0
        self.total = 0
        self.burn_in = sampler_opts.get('burn_in',0) 

        # should we record the history?
        self.track = sampler_opts.get('track',False) 
        if self.track:
            self.omegas = array.array('d')
            self.probs = array.array('d')
            self.acceptances = array.array('d')
        else:
            self.omegas = self.probs = self.acceptances = None

    @property
    def acceptance_ratio(self):
        if self.total == 0:
            return 0.0
        return self.accepted / self.total

    def sample(self):
        if self.config is None:
            self.prob = 0.
            while self.prob < PRECISION: 
                # check if we are starting from scratch
                self.config, self.omega = self.sub_sampler.sample()
                self.prob = self.prob_fn(self.config)

        while True:
            self.total += 1

            # generate candidate configuration
            nconfig, nomega = self.sub_sampler.candidate()
            nprob = self.prob_fn(nconfig)

            # compute acceptance probability
            acceptance = (nprob * self.omega) / (self.prob * nomega)

            if self.track:
                self.omegas.append(nomega)
                self.probs.append(nprob)
                self.acceptances.append(acceptance)

            if (self.rng.uniform() < acceptance):
                self.config = nconfig
                self.omega = nomega
                self.prob = nprob
                self.accepted += 1
                self.sub_sampler.accept(nconfig)

                if (self.total > self.burn_in):
                    return self.config, self.omega

    def update(self, **kwargs):
        self.prob_fn = kwargs['amplitude_factory'].prob
        self.sub_sampler.update(**kwargs)

################################################################################
# gradient optimizer  
################################################################################    
DEFAULT_RATE = 1e-2
DEFAULT_COND = 1e-3
DEFAULT_NUM_STEP = 1e-6
def cumulate_samples(ls):
    _,f,v,e,hg = ls.pop()
    for _,fi,vi,ei,hgi in ls:
        f += fi
        v += vi
        e += ei
        if hg is not None:
            hg += hgi
    if hg is not None: 
        hg = np.array(hg)
    return np.array(f),np.array(v),np.array(e),hg
def _extract_energy_gradient(ls):
    f,v,e,hg = cumulate_samples(ls)

    # mean energy
    esq = np.square(e)
    _xsum = np.dot(f,e) 
    _xsqsum = np.dot(f,esq)
    n = np.sum(f)
    E,err = _mean_err(_xsum,_xsqsum,n)

    # gradient
    v_mean = np.dot(f,v)/n
    g = np.dot(e*f,v)/n - E*v_mean
    return E,n,err,g,f,v,v_mean,hg
def extract_energy_gradient(ls,optimizer,
        psi,constructors,tmpdir,sampler_opts,contract_opts,ham):
    E,n,err,g,f,v,v_mean,hg = _extract_energy_gradient(ls)
    if optimizer.method not in ['sr','rgn','lin']:
        optimizer._set(E,n,g,None,None)
        return optimizer,err

    # ovlp
    if optimizer.ovlp_matrix:
        S = np.einsum('n,ni,nj,n->ij',f,v,v)/n - np.outer(v_mean,v_mean)
        def _S(x):
            return np.dot(S,x)
    else:
        def _S(x):
            Sx1 = np.dot(v.T,f*np.dot(v,x))/n
            Sx2 = v_mean * np.dot(v_mean,x) 
            return Sx1-Sx2
    if optimizer.method not in ['rgn','lin']:
        optimizer._set(E,n,g,_S,None)
        return optimizer,err
    
    # hess
    if optimizer.hg:
        hg_mean = np.dot(f,hg)/n
        if optimizer.hess_matrix:
            H = np.einsum('n,ni,nj->ij',f,v,hg)/n - np.outer(v_mean,hg_mean) 
            H -= np.outer(g,v_mean)
            def _H(x):
                return np.dot(H,x)
        else:
            def _H(x):
                Hx1 = np.dot(v.T,f*np.dot(hg,x))/n
                Hx2 = v_mean * np.dot(hg_mean,x)
                Hx3 = g * np.dot(v_mean,x)
                return Hx1-Hx2-Hx3
    else:
        eps = optimizer.num_step
        infos = [batchsize] * SIZE
        def _H(x):
            psi_ = _update_psi(psi.copy(),-x*eps,constructors)
            psi_ = write_ftn_to_disc(psi_,tmpdir+'tmp',provided_filename=True) 
            args = psi_,sampler_opts,contract_opts,ham,False
            ls_ = parallelized_looped_fxn(_local_sampling,infos,args)
            g_ = _extract_energy_gradient(ls)[3]
            return (g_-g)/eps 
    optimizer._set(E,n,g,_S,_H)
    return optimizer,err
class GradientAccumulator:
    def __init__(self,optimizer_opts):
        self.method = optimizer_opts.get('method','sgd')
        self.learning_rate = optimizer_opts.get('learning_rate',DEFAULT_RATE)
        self.search = optimizer_opts.get('search',None)
        self.hg = False
    def _set(self,E,n,g,S,H):
        self.E = E
        self.n = n
        self.g = g
        self.S = S
        self.H = H
    def transform_gradients(self,x=None):
        g = self.g
        x = self.learning_rate if x is None else x
        if self.optimizer=='sgd':
            deltas = g
        elif self.optimizer=='sign':
            deltas = np.sign(g)
        elif self.optimizer=='signu':
            deltas = np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError
        return x * deltas 
    def trial_gradients(self,x):
        deltas = self.transform_gradients(x=1.)
        return [deltas * xi for xi in x]
class Adam(GradientAccumulator):
    def __init__(self,optimizer_opts):
        super().__init__(optimizer_opts)
        self.beta1 = optimizer_opts.get('beta1',.9)
        self.beta2 = optimizer_opts.get('beta2',.999)
        self.eps = optimizer_opts.get('eps',1e-8)
        self._num_its = 0
        self._ms = None
        self._vs = None
    def transform_gradients(self):
        self._num_its += 1
        if self._num_its == 1:
            self._ms = np.zeros_like(g)
            self._vs = np.zeros_like(g)
    
        self._ms = (1.-self.beta1) * g + self.beta1 * self._ms
        self._vs = (1.-self.beta2) * g**2 + self.beta2 * self._vs 
        mhat = self._ms / (1. - self.beta1**(self._num_its))
        vhat = self._vs / (1. - self.beta2**(self._num_its))
        return E,err,self.learning_rate * mhat / (np.sqrt(vhat)+self.eps)
class SR(GradientAccumulator):
    def __init__(self,optimizer_opts):
        super().__init__(optimizer_opts)
        self.cond = optimizer_opts.get('cond',DEFAULT_COND)
        self.ovlp_matrix = optimizer_opts.get('ovlp_matrix',False)
    def _get_cond_rate(self,x):
        if x is None:
            return self.cond,self.learning_rate
        else:
            if self.search=='cond':
                return x,self.learning_rate 
            else:
                return self.cond,x
    def transform_gradients(self,x=None):
        g = self.g
        S = self.S
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        def R(x):
            return S(x) + cond * x
        LinOp = spla.LinearOperator((sh,sh),matvec=R)
        return rate * spla.cg(LinOp,g) 
    def trial_gradients(self,x):
        if self.search=='cond':
            return [self.transform_gradients(xi) for xi in x]
        else:
            deltas = self.transform_gradients(1.)
            return [deltas * xi for xi in x]
class RGN(SR):
    def __init__(self,optimizer_opts):
        super().__init__(optimizer_opts)
        self.hess_matrix = optimizer_opts.get('hess_matrix',False)  
        self.hg = optimizer_opts.get('hg',False)
        self.num_step = optimizer_opts.get('num_step',DEFAULT_NUM_STEP)
    def transform_gradients(self,x=None):
        E = self.E
        g = self.g
        S = self.S
        H = self.H
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        def R(x):
            return S(x) + cond * x
        def _H(x):
            return H(x) - E * S(x) + R(x)/rate 
        LinOp = spla.LinearOperator((sh,sh),matvec=_H)
        return rate * spla.cg(LinOp,g) 
    def trial_gradients(self,x):
        return [self.transform_gradients(xi) for xi in x]
class LinOpt(RGN):
    def __init__(self,optimizer_opts):
        super().__init__(optimizer_opts)
        self.sr_cond = optimizer_opts.get('sr_cond',True)
    def transform_gradients(self,x,psi,search_opts):
        E = self.E
        g = self.g
        S = self.S
        H = self.H
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        def _S(x):
            return np.concatenate([np.ones(1)*x[0],S(x[1:])]) 
        if self.sr_cond:
            def R(x1):
                return S(x1) + cond * x1
        else:
            def R(x1):
                return cond * x1
        def _H(x):
            x0,x1 = x[0],x[1:]
            Hx0 = np.dot(g,x1)
            Hx1 = x0 * g + H(x1) - E*S(x1) + R(x1)/rate
            return np.concatenate([np.ones(1)*Hx0,Hx1])
        A = spla.LinearOperator((sh+1,sh+1),matvec=_H)
        M = spla.LinearOperator((sh+1,sh+1),matvec=_S)
        dE,deltas = spla.eigs(A,k=1,M=M,sigma=0.)
        deltas = deltas[:,0].real
        deltas = deltas[1:] / deltas[0]
        return - deltas / (1. - np.dot(self._glog,deltas)) 
################################################################################
# sampling fxns 
################################################################################    
def parse_sampler(sampler_opts,amplitude_factory):
    _sampler = sampler_opts['sampler']
    if _sampler=='dense':
        sampler = DenseSampler(sampler_opts,**amplitude_factory.contract_opts)
        load_fname = sampler_opts['load_fname']
        sampler._set_psi(load_fname=load_fname,write_fname=None)
        return sampler
    if _sampler=='exchange':
        if GEOMETRY=='2D':
            sub_sampler = ExchangeSampler2D(sampler_opts)
    else:
        raise NotImplementedError
    return MetropolisHastingsSampler(sub_sampler,sampler_opts,amplitude_factory=amplitude_factory)
def get_amplitude_factory(psi,contract_opts):
    psi = load_ftn_from_disc(psi)
    if GEOMETRY=='2D':
        amplitude_factory = AmplitudeFactory2D(psi,**contract_opts) 
    else:
        raise NotImplementedError
    return amplitude_factory
def gen_samples(psi,batchsize,sampler_opts,contract_opts):
    t0 = time.time()
    amplitude_factory = get_amplitude_factory(psi,contract_opts)
    sampler = parse_sampler(sampler_opts,amplitude_factory)
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=batchsize)
    samples = dict()
    f = dict() # frequencies
    for n in range(batchsize):
        (config,direction,split),omega = sampler.sample()
        if config in samples:
            f[config] += 1
        else:
            samples[config] = direction,split 
            f[config] = 1  
        if progbar:
            _progbar.update()
    #exit()
    if progbar:
        _progbar.close()
        print('time=',time.time()-t0)
    return amplitude_factory,samples,f
def update_grads(amplitude_factory,samples):
    t0 = time.time()
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=len(samples))
    for info in samples:
        amplitude_factory.grad(info)
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()
        print('time=',time.time()-t0)
    return amplitude_factory 
def _compute_local_energy(ham,amplitude_factory,config):
    en = 0.0
    cx = amplitude_factory.store[config]
    c_configs, c_coeffs = ham.config_coupling(config)
    for hxy, info_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < thresh:
            continue
        cy = cx if info_y is None else amplitude_factory.amplitude(info_y)
        en += hxy * cy 
    return en / cx
def _mean_err(_xsum,_xsqsum,n):
    mean = _xsum / n
    var = _xsqsum / n - mean**2
    std = var ** .5
    err = std / n**.5 
    return mean,err
def compute_elocs(ham,amplitude_factory,samples,f):
    t0 = time.time()
    e = [] 
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=len(grads))
    _xsum = 0.
    _xsqsum = 0.
    n = 0
    for (config,_,_),fx in zip(samples,f):
        ex = _compute_local_energy(ham,amplitude_factory,config)
        e.append(ex)

        _xsum += ex * fx
        _xsqsum += ex**2 * fx
        n += fx
        mean,err = _mean_err(_xsum,_xsqsum,n)
        if progbar:
            _progbar.update()
            _progbar.set_description(f'mean={mean},err={err}')
    if progbar:
        _progbar.close()
        print('time=',time.time()-t0)
    return e
def _compute_local_hg(ham,amplitude_factory,config):
    ex = 0.0
    hgx = 0.0 
    cx = amplitude_factory.store[config]
    gx = amplitude_factory.store_grad[config]
    c_configs, c_coeffs = ham.config_coupling(config)
    for hxy, info_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < thresh:
            continue
        cy,gy = (cx,gx) if info_y is None else amplitude_factory.grad(info_y)
        ex += hxy * cy 
        hgx += hxy * gy
    return ex/cx, hgx/cx 
def compute_hg(ham,amplitude_factory,samples,f):
    t0 = time.time()
    e = []
    hg = []
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=len(grads))
    _xsum = 0.
    _xsqsum = 0.
    n = 0
    for (config,_,_),fx in zip(samples,f):
        ex,hgx = _compute_local_hg(ham,amplitude_factory,config)
        e.append(ex)
        hg.append(hgx)

        _xsum += ex * fx
        _xsqsum += ex**2 * fx
        n += fx
        mean,err = _mean_err(_xsum,_xsqsum,n)
        if progbar:
            _progbar.update()
            _progbar.set_description(f'mean={mean},err={err}')
    if progbar:
        _progbar.close()
        print('time=',time.time()-t0)
    return e,hg 
def _local_sampling(psi,batchsize,sampler_opts,contract_opts,ham,_compute_hg):
    amp_fac,samples,f = gen_samples(psi,batchsize,sampler_opts,contract_opts)
    configs = list(f.keys())
    samples = [(x,)+samples[x] for x in configs] 
    f = [f[x] for x in configs]

    amp_fac = update_grads(amp_fac,samples)
    c = amp_fac.store
    g = amp_fac.store_grad
    v = [g[x]/c[x] for x in configs]

    hg = None
    if _compute_hg:
        e,hg = compute_hg(ham,amp_fac,samples,f)
    else:
        e = compute_elocs(ham,amp_fac,samples,f)
    return samples,f,v,e,hg
def _correlated_local_sampling(info,psi,ham,contract_opts):
    samples,f = info
    configs = [x[0] for x in samples]

    amp_fac = get_amplitude_factory(psi,contract_opts)
    amp_fac = compute_grads(amp_fac,samples)
    c = amp_fac.store
    g = amp_fac.store_grad
    v = [g[x]/c[x] for x in configs]

    e = compute_elocs(ham,amp_fac,samples,f)
    return samples,f,v,e,None 
def _local_sampling_exact(info,psi,configs,contract_opts,ham,_compute_hg):
    start,stop = info
    amp_fac = get_amplitude_factory(psi,contract_opts)
    progbar = (RANK==SIZE//2)
    if progbar:
        _progbar = Progbar(total=stop-start)

    samples = []
    f = []
    v = []
    e = []
    hg = [] if _compute_hg else None
    for ix in range(start,stop):
        config = configs[ix]
        info = config,config_order,None
        cx,gx = amp_fac.grad(info)
        if np.fabs(cx)>1e-12:
            if _compute_hg:
                ex,hgx = _compute_local_hg(ham,amp_fac,config,cx,gx)
                hg.append(hgx)
            else:
                ex = _compute_local_energy(ham,amp_fac,config,cx)
            samples.append(info)
            f.append(cx**2)
            v.append(gx/cx)
            e.append(ex)
        if progbar:
            _progbar.update()
    return f,v,e,hg 
################################################################################
# linesearch   
################################################################################    
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
def line_search(self,optimizer,x,psi,constructors,tmpdir,batchsize,sampler_opts,contract_opts):
    deltas_ls = optimizer.trial_gradients(x)
    psi_new = [_update_psi(psi.copy(),deltas,constructors) for deltas in deltas_ls] 
    psi_new = [write_ftn_to_disc(psi,tmpdir+f'ls{ix}',provided_filename=True) for ix,psi in enumerate(psi_new)]

    infos = [psi_new[0]] * SIZE
    args = batchsize,sampler_opts,contract_opts
    ls = parallelized_looped_fxn(gen_samples,infos,args)

    infos = [itm[1:] for itm in ls]
    E0,E1 = np.zeros(len(psi_new)),np.zeros(len(psi_new))
    args = sampler_opts,contract_opts
    for ix,psi_ in enumerate(psi_new):
        ls = parallelized_looped_fxn(_local_line_search,infos,(psi_,sampler_opts,contract_opts)) 
        _xsum = 0.
        _xsqsum = 0.
        n = 0
        for worker,result in enumerate(ls):
            _xsum += result[0]
            _xsqsum += result[1]
            n += result[2]
        E0[ix],E1[ix] = _mean_err(_xsum,_xsqaum,n)
        print(f'ix={ix},energy={E0[ix]},err={E1[ix]}')
    x0,y0 = _solve_quad(x,E0)
    return x0,y0
class TNVMC:
    def __init__(
        self,
        psi,
        ham,
        sampler_opts,
        optimizer_opts,
        tmpdir,
        #conditioner='auto',
        conditioner=None,
        **contract_opts
    ):

        self.psi = psi.copy()
        self.tmpdir = tmpdir
        self.sampler_opts = sampler_opts

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
            self.constructors = get_constructors_2d(psi)
        else:
            raise NotImplementedError

        self.optimizer = _get_optimizer(optimizer_opts)
        self.sample_args = sampler_opts,contract_opts,ham,optimizer.hg
    def parse_sampler(self,psi):
        _sampler = self.sampler_opts['sampler']
        if _sampler=='dense':
            fname = self.tmpdir + 'tmp.hdf5'
            sampler = DenseSampler(self.sampler_opts,**self.contract_opts)
            sampler._set_psi(psi=psi,load_fname=None,write_fname=fname)
            self.sampler_opts['load_fname'] = fname
            self.sampler_opts['write_fname'] = None 
    def parse_search(self):
        if self.optimizer.search is None:
            return None
        if self.optimizer.search=='rate':
            x1 = self.optimizer.learning_rate
            return np.array([.5*x1,x1,1.5*x1])
        else:
            x1 = self.optimizer.cond
            return np.array([.1*x1,x1,10.*x1])
    def run(self, steps, batchsize):
        psi = write_ftn_to_disc(self.psi,self.tmpdir+'psi_init',
                                provided_filename=True) 
        self.parse_sampler(psi) # only called with dense sampler
        x = self.parse_search() 
        for step in range(steps):
            t0 = time.time()
            infos = [psi] * SIZE
            ls = parallelized_looped_fxn(_local_sampling,infos,(batchsize,)+self.sample_args)

            self.optimizer,err = extract_energy_gradient(ls,self.optimizer,
                batchsize,self.psi,self.constructors,self.tmpdir,
                self.sampler_opts,self.contract_opts,self.ham)
            print(f'step={step},energy={optmizer.E},err={err},time={time.time()-t0}')

            if self.optimizer.search is None:
                deltas = self.optimizer.transform_gradient()
            else:
                x0,y0 = line_search(self.optimizer,x,self.psi,self.constructors,self.tmpdir,
                                    batchsize,self.sampler_opts,self.contract_opts)  
                deltas = self.optimizer.transform_gradient(x0)
            # update the actual tensors
            self.psi = _update_psi(self.psi,deltas,self.constructors)
            if self.conditioner is not None:
                self.conditioner(self.psi)
            psi = write_ftn_to_disc(self.psi,self.tmpdir+f'psi{step}',
                                    provided_filename=True) 
            self.parse_sampler(psi) # only called with dense sampler
def _update_psi(psi,deltas,constructors):
    deltas = vec2psi(constructors,deltas)
    for ix,(_,_,_,site) in enumerate(constructors):
        tsr = psi[psi.site_tag(*site)]
        data = tsr.data - deltas[site] 
        tsr.modify(data=data)
    return psi
def _get_optimizer(optimizer_opts):
    method = optimizer_opts.get('method','sr')
    return {'adam':Adam,'sr':SR,'lin':LinOpt,'rgn':RGN}[method](optimizer_opts)
 
