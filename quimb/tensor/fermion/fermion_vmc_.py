import time,h5py,itertools
import numpy as np
import scipy.sparse.linalg as spla

from quimb.utils import progbar as Progbar
from .utils import (
    rand_fname,load_ftn_from_disc,write_ftn_to_disc,
    vec2psi,
)
from .fermion_2d_vmc import (
    thresh,config_map,order_2d,
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
            self.direction = order_2d 
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
class GradientAccumulator:
    def __init__(self,optimizer_opts,search_opts):
        self.optimizer = optimizer_opts.get('optimizer','sgd')
        self.learning_rate = optimizer_opts.get('learning_rate',DEFAULT_RATE)
        self.ovlp = optimizer_opts.get('ovlp',False)
        self.hess = optimizer_opts.get('hess',False)

        self.constructors = search_opts.get('constructors',None)
        self.ham = search_opts.get('ham',None)
        self.tmpdir = search_opts.get('tmpdir',None)
        self.sampler_opts = search_opts.get('sampler_opts',None)
        self.contract_opts = search_opts.get('contract_opts',None)
        self.search_eps = search_opts.get('search_eps',False)
        self.exact = search_opts.get('exact',False)
        self.batchsize = search_opts.get('batchsize',None)

        self._eloc = None
        self._glog = None
        self._glog_eloc = None
        self._num_samples = 0

        self._glog_glog = None

        self._geloc = None
        self._glog_geloc = None
    def _init_storage(self, gx):
        size = len(gx)
        self._eloc = 0.0
        self._glog = np.zeros(size)
        self._glog_eloc = np.zeros(size)
        if self.ovlp:
            self._glog_glog = np.zeros((size,)*2)
        if self.hess:
            self._geloc = np.zeros(size)
            self._glog_geloc = np.zeros((size,)*2)

    def update(self, glx, ex, gex):
        if self._eloc is None:
            self._init_storage(glx)

        self._eloc += ex
        self._glog += glx 
        self._glog_eloc += glx * ex 
        if self.ovlp:
            self._glog_glog += np.outer(glx,glx)
        if self.hess:
            self._geloc += gex
            self._glog_geloc += np.outer(glx,gex)
        self._num_samples += 1
    def update_exact(self, glx, ex, gex,cx):
        if self._eloc is None:
            self._init_storage(glx)
            self._num_samples = 0.

        self._eloc += ex * cx**2
        self._glog += glx * cx**2
        self._glog_eloc += glx * ex * cx**2
        if self.ovlp:
            self._glog_glog += np.outer(glx,glx) * cx**2 
        if self.hess:
            self._geloc += gex * cx**2
            self._glog_geloc += np.outer(glx,gex) * cx**2
        self._num_samples += cx**2
    def update_from_worker(self,other):
        if self._eloc is None:
            self._init_storage(other._glog)

        self._eloc += other._eloc
        self._glog += other._glog
        self._glog_eloc += other._glog_eloc
        if self.ovlp:
            self._glog_glog += other._glog_glog
        if self.hess:
            self._geloc += other._geloc
            self._glog_geloc += other._glog_geloc
        self._num_samples += other._num_samples
    def extract_grads_energy(self):
        self._eloc /= self._num_samples
        self._glog /= self._num_samples
        self._glog_eloc /= self._num_samples
        g = self._glog_eloc - self._glog * self._eloc 
        sij,hij = None,None
        if self.ovlp:
            self._glog_glog /= self._num_samples
            sij = self._glog_glog - np.outer(self._glog,self._glog)
        if self.hess:
            self._geloc /= self._num_samples
            self._glog_geloc /= self._num_samples
            hij = self._glog_geloc - np.outer(self._glog,self._geloc)
            hij -= np.outer(g,self._glog)
        return g,sij,hij
    def reset(self):
        self._eloc = 0.
        self._glog.fill(0.)
        self._glog_eloc.fill(0.)
        if self.ovlp:
            self._glog_glog.fill(0.)
        if self.hess:
            self._geloc.fill(0.)
            self._glog_geloc.fill(0.)
        self._num_samples = 0
    def transform_gradients(self,x,psi):
        g,sij,hij = self.extract_grads_energy()
        if self.optimizer=='sgd':
            deltas = g
        elif self.optimizer=='sign':
            deltas = np.sign(g)
        elif self.optimizer=='signu':
            deltas = np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError

        if x is None:
            return self.learning_rate * deltas 
        deltas_ls = [deltas * xi for xi in x]
        y,y_ = self.line_search_energies(deltas_ls,psi)
        x0,y0 =_solve_quad(x,y)
        return x0 * deltas 
    def line_search_energies(self,deltas_ls,psi):
        psi_new = [_update_psi(psi.copy(),deltas,self.constructors) for deltas in deltas_ls] 
        psi_new = [write_ftn_to_disc(psi,self.tmpdir+f'ls{ix}',provided_filename=True) for ix,psi in enumerate(psi_new)]
    
        if self.exact:
            sampler = DenseSampler(self.sampler_opts)
            nconfig = len(sampler.configs)
            infos = distribute(nconfig) 
            fxn = _local_line_search_exact
        else:
            infos = [(self.batchsize,None,None)] * SIZE
            fxn = _local_line_search
        E0,E1 = np.zeros(len(psi_new)),np.zeros(len(psi_new))
        for ix in (1,0,2):
        #for ix in range(len(psi_new)):
            args = psi_new[ix],self.ham,self.sampler_opts,self.contract_opts
            ls = parallelized_looped_fxn(fxn,infos,args) 
            r0 = 0.
            r1 = 0.
            for worker,result in enumerate(ls):
                r0 += result[0]
                r1 += result[1]
                infos[worker] = result[2]
            if self.exact:
                E0[ix],E1[ix] = r0/r1,r1
                print(f'ix={ix},energy={E0[ix]},N={E1[ix]}')
            else:
                E0[ix],E1[ix] = _mean_err(r0,r1,self.batchsize * SIZE)
                print(f'ix={ix},energy={E0[ix]},err={E1[ix]}')
        return E0,E1
class Adam(GradientAccumulator):
    def __init__(self,optimizer_opts,search_opts):
        self.beta1 = optimizer_opts.get('beta1',.9)
        self.beta2 = optimizer_opts.get('beta2',.999)
        self.eps = optimizer_opts.get('eps',1e-8)
        self._num_its = 0
        self._ms = None
        self._vs = None

        optimizer_opts['optimizer'] = None
        optimizer_opts['ovlp'] = False
        optimizer_opts['hess'] = False
        super().__init__(optimizer_opts,dict())
    def transform_gradients(self,x,psi):
        g,sij,hij = self.extract_grads_energy()
        self._num_its += 1
        if self._num_its == 1:
            self._ms = np.zeros_like(g)
            self._vs = np.zeros_like(g)
    
        self._ms = (1.-self.beta1) * g + self.beta1 * self._ms
        self._vs = (1.-self.beta2) * g**2 + self.beta2 * self._vs 
        mhat = self._ms / (1. - self.beta1**(self._num_its))
        vhat = self._vs / (1. - self.beta2**(self._num_its))
        return self.learning_rate * mhat / (np.sqrt(vhat)+self.eps)
class SR(GradientAccumulator):
    def __init__(self,optimizer_opts,search_opts):
        self.cond = optimizer_opts.get('cond',DEFAULT_COND)
        optimizer_opts['optimizer'] = None
        optimizer_opts['ovlp'] = True 
        optimizer_opts['hess'] = False
        super().__init__(optimizer_opts,search_opts)
    def transform_gradients(self,x,psi):
        g,sij,hij = self.extract_grads_energy()
        def _get_deltas(cond,rate):
            rij = sij + np.eye(len(g)) * cond
            return rate * np.linalg.solve(rij,g) 
        if x is None:
            return _get_deltas(self.cond,self.learning_rate) 
        if self.search_rate: 
            deltas = _get_deltas(self.cond,1.) 
            deltas_ls = [deltas * xi for xi in x]
            y,y_ = self.line_search_energies(deltas_ls,psi)
            x0,y0 = _solve_quad(x,y)
            return x0 * deltas 
        else:
            deltas_ls = [_get_deltas(xi,self.learning_rate) for xi in x]
            y,y_ = self.line_search_energies(deltas_ls,psi)
            x0,y0 = _solve_quad(x,y)
            return _get_deltas(x0,self.learning_rate) 
class RGN(GradientAccumulator):
    def __init__(self,optimizer_opts,search_opts):
        self.cond = optimizer_opts.get('cond',DEFAULT_COND)
        optimizer_opts['optimizer'] = None
        optimizer_opts['ovlp'] = True 
        optimizer_opts['hess'] = True 
        super().__init__(optimizer_opts,search_opts)
    def transform_gradients(self,x,psi):
        g,sij,hij = self.extract_grads_energy()
        hij -= self._eloc * sij
        def _get_deltas(cond,rate):
            rij = sij + np.eye(len(g)) * cond
            return np.linalg.solve(hij+rij/rate,g)
        if x is None:    
            return _get_deltas(self.cond,self.learning_rate) 
        if self.search_rate: 
            rij = sij + np.eye(len(g)) * self.cond
            deltas_ls = [np.linalg.solve(hij+rij/xi,g) for xi in x]
            y,y_ = self.line_search_energies(deltas_ls,psi)
            x0,y0 = _solve_quad(x,y)
            return np.linalg.solve(hij+rij/x0,g)
        else:
            deltas_ls = [_get_deltas(xi,self.learning_rate) for xi in x]
            y,y_ = self.line_search_energies(deltas_ls,psi)
            x0,y0 = _solve_quad(x,y)
            return _get_deltas(x0,self.learning_rate) 
class LinOpt(GradientAccumulator):
    def __init__(self,optimizer_opts,search_opts):
        self.cond = optimizer_opts.get('cond',DEFAULT_COND)
        self.sr_damping = optimizer_opts.get('sr_damping',True)
        optimizer_opts['optimizer'] = None
        optimizer_opts['ovlp'] = True 
        optimizer_opts['hess'] = True 
        super().__init__(optimizer_opts,search_opts)
    def transform_gradients(self,x,psi):
        g,sij,hij = self.extract_grads_energy()
        hij -= self._eloc * sij
        size = len(g)
        s = np.block([[np.ones((1,1)),np.zeros((1,size))],
                      [np.zeros((size,1)),sij]])     
        def _get_deltas(cond,rate):
            rij = sij + np.eye(size) * cond if self.sr_damping else np.eye(size)
            h = np.block([[np.zeros((1,1)),g.reshape(1,size)],
                          [g.reshape(size,1),hij+rij/rate]])
            dE,deltas = spla.eigs(h,M=s,k=1,sigma=h[0,0]) 
            deltas = deltas[:,0].real
            deltas = deltas[1:] / deltas[0]
            return - deltas / (1. - np.dot(self._glog,deltas)) 
        if x is None:
            return _get_deltas(self.cond,self.learning_rate)
        deltas_ls = [_get_deltas(self.cond,xi) for xi in x] if self.search_rate\
                     else [_get_deltas(xi,self.learning_rate) for xi in x]
        y,y_ = self.line_search_energies(deltas_ls,psi)
        x0,y0 = _solve_quad(x,y)
        deltas = _get_deltas(self.cond,x0) if self.search_eps else \
                 _get_deltas(x0,self.learning_rate)
        return deltas
################################################################################
# some vmc utils 
################################################################################    
optimizers = {'adam':Adam,'sr':SR,'lin':LinOpt,'rgn':RGN}
def get_optimizer(optimizer_opts,search_opts):
    _optimizer = optimizer_opts['optimizer']
    hess = True if _optimizer in ['lin','rgn'] else False
    cls = optimizers.get(_optimizer,GradientAccumulator)
    return cls(optimizer_opts.copy(),search_opts.copy()),hess
def get_amplitude_factory(contract_opts,psi=None):
    psi = load_ftn_from_disc(psi) if psi is not None else psi
    if GEOMETRY=='2D':
        amplitude_factory = AmplitudeFactory2D(psi=psi,**contract_opts) 
    else:
        raise NotImplementedError
    return amplitude_factory 
def parse_sampler(sampler_opts,amplitude_factory=None,psi=None,contract_opts=None):
    _sampler = sampler_opts['sampler']
    contract_opts = amplitude_factory.contract_opts if contract_opts is None else contract_opts
    if _sampler=='dense':
        sampler = DenseSampler(sampler_opts,**contract_opts)
        load_fname = sampler_opts.get('load_fname',None)
        write_fname = sampler_opts.get('write_fname',None)
        sampler._set_psi(psi=psi,load_fname=load_fname,write_fname=write_fname)
        return sampler
    if _sampler=='exchange':
        if GEOMETRY=='2D':
            sub_sampler = ExchangeSampler2D(sampler_opts)
    else:
        raise NotImplementedError
    return MetropolisHastingsSampler(sub_sampler,sampler_opts,amplitude_factory=amplitude_factory)
def _mean_err(_xsum,_xsqsum,n):
    mean = _xsum / n
    var = _xsqsum / n - mean**2
    std = var ** .5
    err = std / n**.5 
    return mean,err
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
################################################################################
# sampling fxns 
################################################################################    
def _gen_samples(batchsize,psi,sampler_opts,contract_opts):
    t0 = time.time()
    amplitude_factory = get_amplitude_factory(contract_opts,psi=psi)
    sampler = parse_sampler(sampler_opts,amplitude_factory=amplitude_factory)
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=batchsize)
    samples = dict()
    freq = dict()
    for n in range(batchsize):
        info,omega = sampler.sample()
        config = info[0]
        if config in samples:
            freq[config] += 1
        else:
            samples[config] = info,omega
            freq[config] = 1  
        if progbar:
            _progbar.update()
    #exit()
    print('time=',time.time()-t0)
    return amplitude_factory,samples,freq
def _compute_grads(amplitude_factory,samples):
    t0 = time.time()
    grads = dict()
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=len(samples))
    for config in samples:
        info,omega = samples[config]
        # compute and track local energy
        grads[config] = amplitude_factory.grad(info)
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()
    print('time=',time.time()-t0)
    return grads 
def _compute_local_energy(ham,amplitude_factory,config,cx):
    en = 0.0
    c_configs, c_coeffs = ham.config_coupling(config)
    for hxy, info_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < thresh:
            continue
        cy = cx if info_y is None else amplitude_factory.amplitude(info_y)
        en += hxy * cy 
    return en / cx
def _compute_local_energy_g(ham,amplitude_factory,config,cx,gx):
    ex = 0.0
    gex = 0.0 
    c_configs, c_coeffs = ham.config_coupling(config)
    for hxy, info_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < thresh:
            continue
        cy,gy = (cx,gx) if info_y is None else amplitude_factory.grad(info_y)
        ex += hxy * cy 
        gex += hxy * gy
    return ex/cx, gx/cx, gex/cx 
def _local_quantities(ham,amplitude_factory,config,cx,gx,hess):
    if hess:
        return _compute_local_energy_g(ham,amplitude_factory,config,cx,gx)
    else:
        ex = _compute_local_energy(ham,amplitude_factory,config,cx)
        return ex, gx/cx, None
def _update_optimizer(optimizer_opts,amplitude_factory,grads,freq,ham):
    t0 = time.time()
    optimizer,hess = get_optimizer(optimizer_opts,dict())
    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=len(grads))
    _xsum = 0.
    _xsqsum = 0.
    n = 0
    for config in grads:
        cx,gx = grads[config]
        fx = freq[config] 
        # compute and track local energy
        ex,glx,gex = _local_quantities(ham,amplitude_factory,config,cx,gx,hess)
        for fi in range(fx):
            _xsum += ex
            _xsqsum += ex**2
            n += 1
            mean,err = _mean_err(_xsum,_xsqsum,n+1)
            optimizer.update(glx,ex,gex)
        if progbar:
            _progbar.update()
            _progbar.set_description(f'mean={mean},err={err}')
    if progbar:
        _progbar.close()
    print('time=',time.time()-t0)
    return optimizer,_xsum,_xsqsum 
def _local_sampling(batchsize,psi,ham,sampler_opts,optimizer_opts,contract_opts):
    amp_fac,samples,freq = _gen_samples(batchsize,psi,sampler_opts,contract_opts)
    grads = _compute_grads(amp_fac,samples)
    return _update_optimizer(optimizer_opts,amp_fac,grads,freq,ham)
def _local_line_search(info,psi,ham,sampler_opts,contract_opts):
    batchsize,samples,freq = info
    if samples is None:
        ampilute_factory,samples,freq = _gen_samples(batchsize,psi,sampler_opts,contract_opts)
    else:
        amplitude_factory = get_amplitude_factory(contract_opts,psi=psi) 

    progbar = (RANK==0)
    if progbar:
        _progbar = Progbar(total=len(samples))
    _xsum = 0.
    _xsqsum = 0.
    n = 0
    for config in samples:
        info,omega = samples[config]
        fx = freq[config] 
        # compute and track local energy
        cx = amplitude_factory.amplitude(info)
        ex = _compute_local_energy(ham,amplitude_factory,config,cx)
        for fi in range(fx): 
            _xsum += ex
            _xsqsum += ex**2
            n += 1
            mean,err = _mean_err(_xsum,_xsqsum,n+1)
        if progbar:
            _progbar.update()
            _progbar.set_description(f'mean={mean},err={err}')
    if progbar:
        _progbar.close()
    return _xsum,_xsqsum,(batchsize,samples,freq)
################################################################################
# exact sampling  
################################################################################    
def _local_sampling_exact(info,psi,ham,sampler_opts,optimizer_opts,contract_opts):
    start,stop = info
    psi = load_ftn_from_disc(psi)

    if GEOMETRY=='2D':
        amplitude_factory = AmplitudeFactory2D(psi=psi,**contract_opts) 
        direction = order_2d
    else:
        raise NotImplementedError
    sampler = DenseSampler(sampler_opts)
    optimizer,hess = get_optimizer(optimizer_opts,dict())

    store = dict()
    progbar = (RANK==SIZE//2)
    if progbar:
        _progbar = Progbar(total=stop-start)
    for n in range(start,stop):
        config = sampler.configs[n]
        info = config,direction,None

        # compute and track local energy
        if config in store:
            cx,glx,ex,gex = store[config]
        else:
            cx,gx = amplitude_factory.grad(info)
            if np.fabs(cx)>1e-12:
                ex,glx,gex = _local_quantities(ham,amplitude_factory,config,cx,gx,hess)
            else:
                ex = 0.
                glx = np.zeros_like(gx)
                gex = np.zeros_like(gx)
            store[config] = cx,glx,ex,gex

        optimizer.update_exact(glx,ex,gex,cx)
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()
    return optimizer 
def _local_line_search_exact(info,psi,ham,sampler_opts,contract_opts):
    start,stop = info
    psi = load_ftn_from_disc(psi)

    if GEOMETRY=='2D':
        amplitude_factory = AmplitudeFactory2D(psi=psi,**contract_opts) 
        direction = order_2d
    else:
        raise NotImplementedError
    sampler = DenseSampler(sampler_opts)

    E = 0.
    N = 0.
    store = dict()
    progbar = (RANK==SIZE//2)
    if progbar:
        _progbar = Progbar(total=stop-start)
    for n in range(start,stop):
        config = sampler.configs[n]
        info = config,direction,0

        # compute and track local energy
        if config in store:
            cx,ex = store[config]
        else:
            cx = amplitude_factory.amplitude(info)
            ex = 0. if np.fabs(cx) < 1e-12 else\
                 _compute_local_energy(ham,amplitude_factory,config,cx)
            store[config] = cx,ex
        E += ex * cx**2
        N += cx**2
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()
    return E,N,(start,stop)
class TNVMC:
    def __init__(
        self,
        psi,
        ham,
        sampler_opts,
        optimizer_opts,
        search_opts,
        tmpdir,
        #conditioner='auto',
        conditioner=None,
        **contract_opts
    ):

        self.psi = psi.copy()
        self.ham = ham
        self.sampler_opts = sampler_opts
        self.contract_opts = contract_opts
        self.tmpdir = tmpdir
        if self.sampler_opts['sampler']=='dense':
            self.dense_p = self.tmpdir+rand_fname()+'.hdf5'
            print('save dense amplitude to '+self.dense_p)

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
        self.optimizer_opts = optimizer_opts
        search_opts['sampler_opts'] = sampler_opts
        search_opts['contract_opts'] = contract_opts
        search_opts['constructors'] = self.constructors
        search_opts['ham'] = ham
        search_opts['tmpdir'] = tmpdir
        self.search_opts = search_opts
        self.optimizer,_ = get_optimizer(optimizer_opts,search_opts)
    def parse_sampler(self,psi):
        _sampler = self.sampler_opts['sampler']
        if _sampler=='dense':
            self.sampler_opts['load_fname'] = None
            self.sampler_opts['write_fname'] = self.dense_p
            parse_sampler(self.sampler_opts,psi=psi,contract_opts=self.contract_opts)
            self.sampler_opts['load_fname'] = self.dense_p 
            self.sampler_opts['write_fname'] = None 
    def parse_search(self):
        _search = self.search_opts.get('search',False)
        if not _search:
            return None
        if self.search_opts.get('search_rate',True):
            x1 = self.optimizer_opts.get('learning_rate',DEFAULT_RATE)
            return np.array([.5*x1,x1,1.5*x1])
        else:
            x1 = self.optimizer_opts.get('cond',DEFAULT_COND)
            return np.array([.1*x1,x1,10.*x1])
    def _run(self, steps, batchsize):
        psi = write_ftn_to_disc(self.psi,self.tmpdir+'psi_init',
                                provided_filename=True) 
        self.parse_sampler(psi) # only called with dense sampler
        x = self.parse_search() 
        infos = [batchsize] * SIZE
        for step in range(steps):
            args = psi,self.ham,self.sampler_opts,self.optimizer_opts,\
                   self.contract_opts
            ls = parallelized_looped_fxn(_local_sampling,infos,args)
            _xsum = 0.
            _xsqsum = 0.
            for optimizer,_xsumi,_xsqsumi in ls:
                self.optimizer.update_from_worker(optimizer)

                _xsum += _xsumi 
                _xsqsum += _xsqsumi 
            mean,err = _mean_err(_xsum,_xsqsum,batchsize * SIZE)
            print(f'step={step},energy={mean},err={err}')

            # apply learning rate and other transforms to gradients
            deltas = self.optimizer.transform_gradients(x,self.psi)
            self.optimizer.reset()

            # update the actual tensors
            self.psi = _update_psi(self.psi,deltas,self.constructors)

            # reset having just performed a gradient step
            if self.conditioner is not None:
                self.conditioner(self.psi)
            psi = write_ftn_to_disc(self.psi,self.tmpdir+f'psi{step}',
                                    provided_filename=True) 
            self.parse_sampler(psi) # only called with dense sampler
    def _run_exact(self, steps):
        psi = write_ftn_to_disc(self.psi,self.tmpdir+'psi_init',
                                provided_filename=True) 

        sampler = DenseSampler(self.sampler_opts)
        nconfig = len(sampler.configs) # only called with dense sampler
        x = self.parse_search() 
        infos = distribute(nconfig) 
        for step in range(steps):
            args = psi,self.ham,self.sampler_opts,self.optimizer_opts,\
                   self.contract_opts
            ls = parallelized_looped_fxn(_local_sampling_exact,infos,args)
            for optimizer in ls:
                self.optimizer.update_from_worker(optimizer)
            print(f'step={step},energy={self.optimizer._eloc/self.optimizer._num_samples},N={self.optimizer._num_samples}')

            # apply learning rate and other transforms to gradients
            deltas = self.optimizer.transform_gradients(x,self.psi)
            self.optimizer.reset()

            # update the actual tensors
            self.psi = _update_psi(self.psi,deltas,self.constructors)

            # reset having just performed a gradient step
            if self.conditioner is not None:
                self.conditioner(self.psi)
            psi = write_ftn_to_disc(self.psi,self.tmpdir+f'psi{step}',
                                    provided_filename=True) 
    def run(
        self,
        steps=100, 
        batchsize=100,
        exact_sampling=False,
    ):
        if exact_sampling:
            self._run_exact(steps)
        else:
            self._run(steps,batchsize)

def _update_psi(psi,deltas,constructors):
    deltas = vec2psi(constructors,deltas)
    for ix,(_,_,_,site) in enumerate(constructors):
        tsr = psi[psi.site_tag(*site)]
        data = tsr.data - deltas[site] 
        tsr.modify(data=data)
    return psi
    
