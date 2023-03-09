import time,h5py,itertools
import numpy as np
import scipy.sparse.linalg as spla

from quimb.utils import progbar as Progbar
from .utils import (
    rand_fname,load_ftn_from_disc,write_ftn_to_disc,
    vec2psi,psi2vecs,
)
from .fermion_2d_vmc import (
    SYMMETRY,config_map,
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
def parallelized_looped_fxn(fxn,ls,args,kwargs=dict()):
    stop = min(SIZE,len(ls))
    results = [None] * stop 
    for worker in range(stop-1,-1,-1):
        worker_info = fxn,ls[worker],args,kwargs 
        if worker > 0:
            COMM.send(worker_info,dest=worker) 
        else:
            results[0] = fxn(ls[0],*args,**kwargs)
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
        #assignment = COMM.recv()
        assignment = COMM.recv(source=0)

        # End execution if received message 'finished'
        if assignment == 'finished': 
            break
        # Otherwise, call function
        fxn,local_ls,args,kwargs = assignment
        result = fxn(local_ls,*args,**kwargs)
        COMM.send(result, dest=0)
################################################################################
# Sampler  
################################################################################    
GEOMETRY = '2D' 
def dense_amplitude_wrapper_2d(info,psi,all_configs,compress_opts):
    start,stop = info
    psi = load_ftn_from_disc(psi)
    psi = psi.reorder('row',inplace=True)

    p = np.zeros(len(all_configs)) 
    cache_head = dict()
    cache_mid = dict()
    cache_tail = dict()
    for ix in range(start,stop):
        config = all_configs[ix]
        unsigned_amp,cache_head,cache_mid,cache_tail = \
            compute_amplitude_2d(psi,config,0,cache_head,cache_mid,cache_tail,**compress_opts)
        p[ix] = unsigned_amp**2
    return p
class DenseSampler:
    def __init__(self,sampler_opts):
        if GEOMETRY=='2D':
            self.nsite = sampler_opts['Lx'] * sampler_opts['Ly']
        else:
            raise NotImplementedError
        self.nelec = sampler_opts['nelec']
        self.configs = self.get_all_configs()
        self.nconfig = len(self.configs)
        self.infos = distribute(self.nconfig)
        self.flat_indexes = list(range(self.nconfig))

        self.exact = sampler_opts.get('exact',False)
        seed = sampler_opts.get('seed',None)
        self.rng = np.random.default_rng(seed)
        self.burn_in = 0
    def _set_amplitude_factory(self,amplitude_factory=None,config=None):
        pass
    def _set_psi(self,psi,write_fname,**contract_opts):
        t0 = time.time()
        self.p = np.zeros(self.nconfig)
        if GEOMETRY=='2D':
            fxn = dense_amplitude_wrapper_2d
        else:
            raise NotImplementedError
        args = psi,self.configs,contract_opts
        ls = parallelized_looped_fxn(fxn,self.infos,args)
        for p in ls:
            self.p += p
        self.p /= np.sum(self.p) 
        print(f'\tdense sampler updated ({time.time()-t0}s).')
        f = h5py.File(write_fname,'w')
        f.create_dataset('p',data=self.p)
        f.close()
    def _set_prob(self,load_fname):
        f = h5py.File(load_fname,'r')
        self.p = f['p'][:]
        f.close()
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
    def _sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        config = self.configs[flat_idx]
        omega = self.p[flat_idx]
        return config,omega
    def _sample_exact(self,ix):
        config = self.configs[ix]
        omega = self.p[ix]
        return config,omega
    def sample(self,ix=None):
        if self.exact:
            return self._sample_exact(ix)
        else:
            return self._sample()
################################################################################
# gradient optimizer  
################################################################################    
DEFAULT_RATE = 1e-1
DEFAULT_COND = 1e-3
DEFAULT_NUM_STEP = 1e-4
PRECISION = 1e-10
CG_TOL = 1e-4
class GradientAccumulator:
    def __init__(self):
        self.compute_hg = False
        self.n = 0
        self.esum = 0.
        self.esqsum = 0.
        self.e = []
        self.v = None
    def parse_samples(self,samples,f): 
        self.samples = samples
        self.f = np.array(f)
        #idx = np.argmax(self.f)
        idx = -1
        self.config = samples[idx]

        db = np.zeros_like(self.f)
        for ix,config in enumerate(self.samples):
            x = np.array(config)
            x = x[x==3]
            db[ix] = len(x)
            #print(config,db[ix])
            #exit()
        self.db = np.dot(self.f,db)
    def parse_grad(self,v):
        self.v = np.array(v)
        self.vsum = np.dot(self.f,self.v)
    def update_eloc(self,ex,fx): 
        self.e.append(ex)
        self.n += fx
        self.esum += ex * fx
        self.esqsum += ex**2 * fx
        return _mean_err(self.esum,self.esqsum,self.n)
    def update_hg(self,hgx=None): 
        pass
    def parse_eloc(self):
        self.e = np.array(self.e)
        if self.v is not None:
            self.vesum = np.dot(self.e*self.f,self.v) 
    def parse_hg(self):
        pass
class HessAccumulator(GradientAccumulator):
    def __init__(self):
        super().__init__()
        self.compute_hg = True 
        self.hg = []
    def update_hg(self,hgx): 
        self.hg.append(hgx)
    def parse_hg(self):
        self.hg = np.array(self.hg)
        self.hgsum = np.dot(self.f,self.hg)
################################################################################
# sampling fxns 
################################################################################    
def get_optimizer(optimizer_opts):
    method = optimizer_opts.get('method','sr')
    compute_hg = False
    if method in ['rgn','lin']:
        compute_hg = optimizer_opts.get('compute_hg')
    if compute_hg:
        return HessAccumulator()
    else:
        return GradientAccumulator()
def get_amplitude_factory(psi,contract_opts):
    if isinstance(psi,str):
        psi = load_ftn_from_disc(psi)
    if GEOMETRY=='2D':
        amplitude_factory = AmplitudeFactory2D(psi,**contract_opts) 
    else:
        raise NotImplementedError
    return amplitude_factory
def parse_sampler(sampler_opts):
    _sampler = sampler_opts['sampler']
    if _sampler=='dense':
        sampler = DenseSampler(sampler_opts)
        load_fname = sampler_opts['load_fname']
        sampler._set_prob(load_fname)
        return sampler
    if _sampler=='exchange':
        if GEOMETRY=='2D':
            return ExchangeSampler2D(sampler_opts)
    else:
        raise NotImplementedError
def _gen_samples(sampler,batchsize,optimizer):
    progbar = (RANK==SIZE-1)

    if progbar:
        _progbar = Progbar(total=sampler.burn_in)
    for n in range(sampler.burn_in):
        config,omega = sampler.sample()
        if progbar:
            _progbar.update()

    if progbar:
        _progbar = Progbar(total=batchsize)
    f = dict() 
    for n in range(batchsize):
        config,omega = sampler.sample()
        if config in f:
            f[config] += 1
        else:
            f[config] = 1  
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()

    samples = list(f.keys())
    f = [f[x] for x in samples]
    optimizer.parse_samples(samples,f)
    return optimizer
def _gen_samples_exact(sampler,optimizer):
    progbar = (RANK==SIZE-1)
    start,stop = sampler.infos[RANK]
    if progbar:
        _progbar = Progbar(total=stop-start)
    samples = [] 
    f = []
    for n in range(start,stop):
        config,omega = sampler.sample(n)
        samples.append(config)
        f.append(omega)
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()
    optimizer.parse_samples(samples,f)
    return optimizer
def gen_samples(sampler,batchsize,optimizer):
    if sampler.exact:
        return _gen_samples_exact(sampler,optimizer)
    else:
        return _gen_samples(sampler,batchsize,optimizer)
def update_grads(amplitude_factory,optimizer):
    progbar = (RANK==SIZE-1)
    if progbar:
        _progbar = Progbar(total=len(optimizer.samples))
    v = [] 
    for config in optimizer.samples:
        cx,gx = amplitude_factory.grad(config)
        if np.fabs(cx) < PRECISION:
            vx = np.zeros_like(gx) 
        else:
            vx = gx/cx
        v.append(vx)
        if progbar:
            _progbar.update()
    if progbar:
        _progbar.close()
    optimizer.parse_grad(v)
    return amplitude_factory,optimizer 
def _compute_local_energy(ham,amplitude_factory,config):
    en = 0.0
    c_configs, c_coeffs = ham.config_coupling(config)
    _,split = c_configs[0]
    cx = amplitude_factory.amplitude((config,split))
    if np.fabs(cx) < PRECISION:
        return 0.,None
    for hxy, info_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < PRECISION:
            continue
        cy = cx if info_y is None else amplitude_factory.amplitude(info_y)
        en += hxy * cy 
    return en/cx,None
def _compute_local_hg(ham,amplitude_factory,config):
    ex = 0.0
    hgx = 0.0 
    cx,gx = amplitude_factory.grad(config)
    c_configs, c_coeffs = ham.config_coupling(config)
    if np.fabs(cx) < PRECISION:
        return 0.,np.zeros_like(gx)
    for hxy, info_y in zip(c_coeffs, c_configs):
        if np.fabs(hxy) < PRECISION:
            continue
        cy,gy = (cx,gx) if info_y is None else amplitude_factory.grad(info_y[0])
        ex += hxy * cy 
        hgx += hxy * gy
    return ex/cx, hgx/cx 
def compute_elocs(ham,amplitude_factory,optimizer):
    progbar = (RANK==SIZE-1)
    if progbar:
        _progbar = Progbar(total=len(optimizer.samples))
    fxn = _compute_local_hg if optimizer.compute_hg else\
          _compute_local_energy
    for config,fx in zip(optimizer.samples,optimizer.f):
        ex,hgx = fxn(ham,amplitude_factory,config)
        mean,err = optimizer.update_eloc(ex,fx)
        optimizer.update_hg(hgx)
        if progbar:
            _progbar.update()
            _progbar.set_description(f'mean={mean},err={err}')
    if progbar:
        _progbar.close()
    optimizer.parse_eloc()
    optimizer.parse_hg()
    return optimizer
def _local_sampling(amp_fac,sampler_info,optimizer_opts,ham=None,grad=True):
    opt = get_optimizer(optimizer_opts)
    if len(sampler_info)==3:
        sampler_opts,config,batchsize = sampler_info
        sampler = parse_sampler(sampler_opts) 
        sampler._set_amplitude_factory(amp_fac,config)

        t0 = time.time()
        opt = gen_samples(sampler,batchsize,opt)
        if RANK==SIZE-1:
            print(f'\tsample time={time.time()-t0},stored amp={len(amp_fac.store)}',)
        #print(f'\tRANK={RANK},sample time={time.time()-t0},stored amp={len(amp_fac.store)},n={len(opt.f)}')
    elif len(sampler_info)==2:
        samples,f = sampler_info
        opt.parse_samples(samples,f)
    else:
        raise NotImplementedError

    if grad:
        t0 = time.time()
        amp_fac,opt = update_grads(amp_fac,opt)
        if RANK==SIZE-1:
            print('\tgradient time=',time.time()-t0)
        #print(f'\tRANK={RANK},gradient time={time.time()-t0},stored amp={len(amp_fac.store)}',)

    if ham is not None:
        t0 = time.time()
        opt = compute_elocs(ham,amp_fac,opt)
        if RANK==SIZE-1:
            print('\teloc time=',time.time()-t0)
        #print(f'\tRANK={RANK},eloc time={time.time()-t0}')
    return opt 
def local_sampling_only(info,sampler_opts,contract_opts,optimizer_opts):
    psi,config,batchsize = info
    amp_fac = get_amplitude_factory(psi,contract_opts)
    sampler_info = sampler_opts,config,batchsize
    opt = _local_sampling(amp_fac,sampler_info,optimizer_opts,grad=False) 
    return opt
def local_sampling(info,sampler_opts,contract_opts,optimizer_opts,ham):
    psi,config,batchsize = info
    amp_fac = get_amplitude_factory(psi,contract_opts)
    sampler_info = sampler_opts,config,batchsize
    opt = _local_sampling(amp_fac,sampler_info,optimizer_opts,ham=ham,grad=True) 
    return opt
def oLBFGS_local_sampling(info,sampler_opts,contract_opts,optimizer_opts,ham):
    psi,config,batchsize,samples,f = info
    amp_fac = get_amplitude_factory(psi,contract_opts)

    sampler_info = sampler_opts,config,batchsize
    opt = _local_sampling(amp_fac,sampler_info,optimizer_opts,ham=ham,grad=True) 

    exact = sampler_opts.get('exact',False)
    if exact:
        return opt,opt
    sampler_info = samples,f
    corr_opt = _local_sampling(amp_fac,sampler_info,optimizer_opts,ham=ham,grad=True)
    return opt,corr_opt
def num_hessp_local_sampling(info,sampler_opts,contract_opts,optimizer_opts,ham):
    psi,samples,f = info
    amp_fac = get_amplitude_factory(psi,contract_opts)
    exact = sampler_opts.get('exact',False)
    sampler_info = (sampler_opts,None,None) if exact else (samples,f)
    opt = _local_sampling(amp_fac,sampler_info,optimizer_opts,ham=ham,grad=True)
    return opt 
def line_search_local_sampling(info,sampler_opts,contract_opts,optimizer_opts,ham):
    psi,samples,f = info
    amp_fac = get_amplitude_factory(psi,contract_opts)
    exact = sampler_opts.get('exact',False)
    sampler_info = (sampler_opts,None,None) if exact else (samples,f)
    opt = _local_sampling(amp_fac,sampler_info,optimizer_opts,ham=ham,grad=False)
    return opt 
class TNVMC:
    def __init__(
        self,
        psi,
        ham,
        sampler_opts,
        contract_opts,
        optimizer_opts,
        tmpdir,
        #conditioner='auto',
        conditioner=None,
    ):

        self.psi = psi.copy()
        self.ham = ham
        self.tmpdir = tmpdir
        self.dense_fname = tmpdir+'tmp.hdf5'
        sampler_opts['load_fname'] = self.dense_fname
        self.sampler_opts = sampler_opts
        self.contract_opts = contract_opts
        self.optimizer_opts = optimizer_opts

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
        self.sample_args = sampler_opts,contract_opts,optimizer_opts,ham
        self.initialize()
    def burn_in(self,config,batchsize):
        self.configs = [config] * SIZE
        if self.sampler_opts['sampler']=='dense':
            return
        if batchsize==0:
            return
        psi = write_ftn_to_disc(self.psi,self.tmpdir+'psi0',provided_filename=True) 
        infos = [(psi,config,batchsize)] * SIZE
        ls = parallelized_looped_fxn(local_sampling_only,infos,self.sample_args[:-1])
        self.configs = [opt.config for opt in ls]        
        for opt in ls:
            print(opt.config,opt.p)
    def parse_search(self):
        if self.search is None:
            return None
        if self.search=='rate':
            x = self.rate
            alpha = 2.
        elif self.search=='cond':
            x = self.cond
            alpha = 10.
        else:
            raise NotImplementedError
        return np.array([x/alpha,x,alpha*x])
    def run(self, steps, batchsize):
        x = self.parse_search() # x = None if not search 
        for step in range(steps):
            self.sample(step,batchsize)

            if x is None:
                deltas = self.transform_gradients()
            else:
                x0,y0 = self.line_search(x)  
                deltas = self.transform_gradients(x0)
            # update the actual tensors
            self.psi = _update_psi(self.psi,deltas,self.constructors)
            if self.conditioner is not None:
                self.conditioner(self.psi)
    def initialize(self):
        self.method = self.optimizer_opts.get('method','sgd')
        self.rate = self.optimizer_opts.get('rate',DEFAULT_RATE)
        self.cond = self.optimizer_opts.get('cond',DEFAULT_COND)
        self.num_step = self.optimizer_opts.get('num_step',DEFAULT_NUM_STEP)
        self.sr_cond = self.optimizer_opts.get('sr_cond',True)
        self.configs = [None] * SIZE

        self.compute_hg = False
        if self.method in ['rgn','lin']:
            self.compute_hg = self.optimizer_opts.get('compute_hg',False) 

        self.search = self.optimizer_opts.get('search',None)
        if self.search is not None:
            if self.method not in ['sr','rgn','lin']:
                self.search = 'rate'

        if self.method=='adam':
            self.beta1 = self.optimizer_opts.get('beta1',.9)
            self.beta2 = self.optimizer_opts.get('beta2',.999)
            self.eps = self.optimizer_opts.get('eps',1e-8)
            self._ms = None
            self._vs = None
        if self.method=='bfgs':
            self.m = self.optimizer_opts['m']
            from scipy.optimize import lbfgsb
            self.hessinvp = lbfgsb.LbfgsInvHessProduct
            self.s = None
            self.y = None
            self.xprev = None
            self.gprev = None
    def sample(self,step,batchsize):
        self.psi_fname = write_ftn_to_disc(self.psi,self.tmpdir+f'psi{step}',provided_filename=True) 
        self.step = step
        self.batchsize = batchsize
        if self.sampler_opts['sampler']=='dense':
            sampler = DenseSampler(self.sampler_opts)
            sampler._set_psi(self.psi_fname,self.dense_fname,**self.contract_opts)
        t0 = time.time()
        if self.method=='bfgs':
            self._sample_lbfgs()
        else:
            self._sample()
        print('\ttotal sample time=',time.time()-t0)
        print('\tnsamples=',self.n)
        print(f'step={step},E={self.E},err={self.err},db={self.db}')
        #print(f'step={step},E={self.E},err={self.err}')
    def _sample(self): # non-lbfgs scheme
        infos = [(self.psi_fname,config,self.batchsize) for config in self.configs]
        ls = parallelized_looped_fxn(local_sampling,infos,self.sample_args)
        self.samples = [opt.samples for opt in ls]
        self.f       = [opt.f       for opt in ls]
        self.configs = [opt.config  for opt in ls]
        self.extract_energy_gradient(ls)
    def _sample_lbfgs(self):
        self.x = np.concatenate(psi2vecs(self.constructors,self.psi))
        if self.step == 0:
            self._sample()
            self.xprev = self.x.copy()
            self.gprev = self.g.copy()
            return  
        infos = [(self.psi_fname,config,self.batchsize,samples,f) \
                  for config,samples,f in zip(self.configs,self.samples,self.f)]
        ls = parallelized_looped_fxn(oLBFGS_local_sampling,infos,self.sample_args)
        ls_curr = [opt for opt,_ in ls]
        ls_corr = [opt for _,opt in ls]
        # current param, current samples
        self.extract_energy_gradient(ls_curr)
        self.samples = [opt.samples for opt in ls_curr]
        self.f       = [opt.f       for opt in ls_curr]
        self.configs  = [opt.config  for opt in ls_curr]
        g = self.g.copy()
        n = self.n
        E = self.E
        err = self.err
        # current param, prev samples
        self.extract_energy_gradient(ls_corr)
        size = len(self.x)
        yprev = (self.x - self.xprev).reshape(1,size)
        sprev = (self.g - self.gprev).reshape(1,size)
        if self.y is None:
            self.y = yprev 
            self.s = sprev
        else:
            self.y = np.concatenate([self.y,yprev],axis=0) 
            self.s = np.concatenate([self.s,sprev],axis=0)
        # set current
        self.g = g
        self.n = n
        self.E = E
        self.err = err
        self.xprev = self.x.copy()
        self.gprev = self.g.copy()
    def extract_energy_gradient(self,ls):
        t0 = time.time()
        self.db = sum([opt.db for opt in ls])
        esum   = sum([opt.esum   for opt in ls])
        esqsum = sum([opt.esqsum for opt in ls])
        self.n = sum([opt.n      for opt in ls])
        vsum = 0.
        vesum = 0.
        for opt in ls:
            vsum += opt.vsum
            vesum += opt.vesum

        self.E,self.err = _mean_err(esum,esqsum,self.n)
        vmean = vsum/self.n
        self.g = vesum/self.n - self.E*vmean

        if self.method in ['sr','rgn','lin']:
            f = np.concatenate([opt.f for opt in ls])
            v = np.concatenate([opt.v for opt in ls],axis=0)

            def S(x):
                if np.linalg.norm(x)/len(x)<PRECISION: # return 0 if x is 0
                    return np.zeros_like(x)
                Sx1 = np.dot(v.T,f*np.dot(v,x))/self.n
                Sx2 = vmean * np.dot(vmean,x) 
                return Sx1-Sx2
            self.S = S 
        if self.method in ['rgn','lin']:
            if self.compute_hg:
                hg = np.concatenate([opt.hg for opt in ls],axis=0)
                hgsum = 0.
                for opt in ls:
                    hgsum += opt.hgsum
                hg_mean = hgsum / self.n 

                def H(x):
                    if np.linalg.norm(x)/len(x)<PRECISION: # return 0 if x is 0
                        return np.zeros_like(x)
                    Hx1 = np.dot(v.T,f*np.dot(hg,x))/self.n
                    Hx2 = vmean * np.dot(hg_mean,x)
                    Hx3 = self.g * np.dot(vmean,x)
                    return Hx1-Hx2-Hx3
            else:
                def H(x):
                    self.ncalls += 1
                    if np.linalg.norm(x)/len(x)<PRECISION: # return 0 if x is 0
                        return np.zeros_like(x)
                    psi = _update_psi(self.psi.copy(),-x*self.num_step,self.constructors) 
                    psi_fname = write_ftn_to_disc(psi,self.tmpdir+'tmp',provided_filename=True) 
                    exact = self.sampler_opts.get('exact',False)
                    if exact:
                        sampler = DenseSampler(self.sampler_opts)
                        sampler._set_psi(psi_fname,self.dense_fname,**self.contract_opts)
                        infos = [(psi_fname,None,None)] * SIZE 
                    else:
                        infos = [(psi_fname,opt.samples,opt.f) for opt in ls]
                    ls_ = parallelized_looped_fxn(num_hessp_local_sampling,infos,self.sample_args)

                    esum = sum([opt.esum for opt in ls_])
                    n    = sum([opt.n    for opt in ls_])
                    vsum = 0.
                    vesum = 0.
                    for opt in ls_:
                        vsum += opt.vsum
                        vesum += opt.vesum
                    g = vesum/n - (esum/n)*(vsum/n)
                    return (g-self.g)/self.num_step 
            self.H = H
        print('\treduce time=',time.time()-t0)
    def transform_gradients(self,x=None):
        if self.method=='adam':
            return self._transform_gradients_adam(x=x)
        elif self.method=='lbfgs':
            return self._transform_gradients_lbfgs(x=x)
        elif self.method=='sr':
            return self._transform_gradients_sr(x=x)
        elif self.method=='rgn':
            return self._transform_gradients_rgn(x=x)
        elif self.method=='lin':
            return self._transform_gradients_lin(x=x)
        else:
            return self._transform_gradients_sgd(x=x)
    def _transform_gradients_sgd(self,x=None):
        g = self.g
        x = self.rate if x is None else x
        if self.method=='sgd':
            deltas = g
        elif self.method=='sign':
            deltas = np.sign(g)
        elif self.method=='signu':
            deltas = np.sign(g) * np.random.uniform(size=g.shape)
        else:
            raise NotImplementedError
        return x * deltas 
    def _transform_gradients_adam(self,x=None):
        g = self.g
        x = self.rate if x is None else x
        if self.step == 0:
            self._ms = np.zeros_like(g)
            self._vs = np.zeros_like(g)
    
        self._ms = (1.-self.beta1) * g + self.beta1 * self._ms
        self._vs = (1.-self.beta2) * g**2 + self.beta2 * self._vs 
        mhat = self._ms / (1. - self.beta1**(self._num_its))
        vhat = self._vs / (1. - self.beta2**(self._num_its))
        return x * mhat / (np.sqrt(vhat)+self.eps)
    def _transform_gradients_lbfgs(self,x=None):
        if self.step == 0:
            return self.transform_gradients_sgd(x=x)
        x = self.rate if x is None else x
        hessinvp = self.hessinvp(self.s,self.y)
        z = hessinvp._matvec(self.g)
        return x * z
    def _get_cond_rate(self,x):
        if x is None:
            return self.cond,self.rate
        else:
            if self.search=='cond':
                return x,self.rate 
            elif self.search=='rate':
                return self.cond,x
            else:
                raise NotImplementedError
    def _transform_gradients_sr(self,x=None):
        t0 = time.time()
        g = self.g
        S = self.S
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        def R(vec):
            return S(vec) + cond * vec
        LinOp = spla.LinearOperator(shape=(sh,sh),matvec=R,dtype=float)
        vec,info = spla.minres(LinOp,g,tol=CG_TOL)
        print(f'\tSR solver time={time.time()-t0},exit status={info}')
        return rate * vec 
    def _transform_gradients_rgn(self,x=None):
        if self.compute_hg:
            return self._transform_gradients_rgn_WL(x=x) 
        else:
            return self._transform_gradients_rgn_num(x=x)
    def _transform_gradients_rgn_WL(self,x=None):
        t0 = time.time()
        E = self.E
        g = self.g
        S = self.S
        H = self.H
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        cond_ = cond #* rate
        def R(vec):
            return S(vec) + cond_ * vec
        def Hess(vec):
            return H(vec) - E * S(vec) 
        def A(vec):
            return Hess(vec) + R(vec)/rate 
        LinOp = spla.LinearOperator(shape=(sh,sh),matvec=A,dtype=float)
        vec,info = spla.lgmres(LinOp,g,tol=CG_TOL,atol=CG_TOL) # LinOp not hermitian
        #vec,info = spla.minres(LinOp,g,tol=CG_TOL) # treat as hermitian
        print(f'\tRGN solver time={time.time()-t0},exit status={info}')
        return vec 
    def _transform_gradients_rgn_num(self,x=None):
        t0 = time.time()
        E = self.E
        g = self.g
        S = self.S
        Hess = self.H
        sh = len(g)
        cond,rate = self._get_cond_rate(x)
        cond_ = cond * rate
        def R(vec):
            return S(vec) + cond_ * vec
        def A(vec):
            return Hess(vec) + R(vec)/rate 
        LinOp = spla.LinearOperator(shape=(sh,sh),matvec=R,dtype=float)
        x0,info = spla.minres(LinOp,g,tol=CG_TOL)
        self.maxiter = 4 
        self.ncalls = 0
        #delta = _inv_rgn_cr(A,g,x0*rate,self.maxiter)
        delta = _inv_rgn_minres(A,g,x0*rate,self.maxiter)
        print(f'\tRGN solver time={time.time()-t0},ncalls={self.ncalls}')
        return delta# * rate 
    def _transform_gradients_lin(self,x=None):
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
    def line_search(self,x):
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
def _update_psi(psi,deltas,constructors):
    deltas = vec2psi(constructors,deltas)
    for ix,(_,_,_,site) in enumerate(constructors):
        tsr = psi[psi.site_tag(*site)]
        data = tsr.data - deltas[site] 
        tsr.modify(data=data)
    return psi
def _mean_err(_xsum,_xsqsum,n):
    if n<PRECISION:
        return 0.,0.
    mean = _xsum / n
    var = _xsqsum / n - mean**2
    std = np.sqrt(var)
    err = std / np.sqrt(n) 
    return mean,err
def _inv_rgn_2(H,LinOp,g,x0,rate,maxiter):
    # stationary iterative method for solving (H+R/rate)x=g
    # x^{k+1} = R^{-1}(-rate*Hx^k+rate*g)=-rate*R^{-1}Hx^k+x0
    x = x0
    incre = x0
    for k in range(1,maxiter+1):
        vec = H(incre)
        incre,info = spla.minres(LinOp,vec,tol=CG_TOL)
        x += incre * (-rate)**k 
        print(f'\titer={k},incre={np.linalg.norm(incre)}')
    return x 
def _inv_rgn_cr(A,b,x0,maxiter):
    r = b-A(x0)
    p = [r]
    s = [A(r)]
    x = x0
    for k in range(1,maxiter+1):
        s_ = s[-1]
        p_ = p[-1]
        alpha = np.dot(r,s_) / np.dot(s_,s_) 
        x += alpha * p_
        r -= alpha * s_
        print(f'\titer={k},res={np.linalg.norm(r)}')
        if k==maxiter:
            return x
        pk = s_.copy()
        sk = A(s_)

        for s_,p_ in zip(s,p):
            beta = np.dot(sk,s_) / np.dot(s_,s_)            
            pk -= beta * p_
            sk -= beta * s_
        p.append(pk)
        s.append(sk)
        if k>1:
            p.pop(0)
            s.pop(0)
def _inv_rgn_minres(A,b,x0,maxiter):
    sh = len(b)
    LinOp = spla.LinearOperator(shape=(sh,sh),matvec=A,dtype=float)
    x,info = spla.minres(LinOp,b,x0=x0,maxiter=maxiter,show=True)
    return x
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
