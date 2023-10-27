import numpy as np
from ..product_vmc import FNN,tensor2backend,pair_terms,config_to_ab
#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()
#######################################################################
# some jastrow forms
#######################################################################
class TNJastrow(FermionAmplitudeFactory):
    def update_pair_energy_from_plq(self,tn,where):
        ix1,ix2 = [self.flatten(site) for site in where]
        i1,i2 = self.config[ix1],self.config[ix2] 
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return {spin:0 for spin in ('a','b')}
        ex = dict()
        for spin in ('a','b'):
            i1_new,i2_new = pair_terms(i1,i2,spin)
            if i1_new is None:
                ex[spin] = 0
                continue
            tn_new = self.replace_sites(tn.copy(),where,(i1_new,i2_new))
            ex_ij = safe_contract(tn_new)
            if ex_ij is None:
                ex_ij = 0
            ex[spin] = ex_ij 
        return ex 
def BackFlow(FNN):
    def __init__(self,mo,nv,nl,spin,**kwargs):
        self.mo = mo
        self.nsite,self.nelec = mo.shape
        self.spin = spin
        super().__init__(nv,nl,nf=self.nsite*self.nelec,log_amp=False,fermion=True,**kwargs)
    def config_to_spin(self,config):
        return np.array(config) % 2 if self.spin=='a' else np.array(config) // 2
    def wfn2backend(self,backend=None,requires_grad=False):
        super().wfn2backend(backend=backend,requires_grad=requires_grad)
        self.mo = tensor2backend(self.mo,backend)
        if backend=='numpy':
            def _det(config,mo):
                det = np.where(config)
                return mo[det,:]
        else:
            def _det(config,mo):
                det = np.where(config)[0] 
                return torch.index_select(mo,0,torch.tensor(det))
    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
        c = self._input(config)
        mo = super().forward(c)
        mo = self.jnp.reshape(mo,self.mo.shape) + self.mo
        c = self._det(self.config_to_spin(config),mo)
        if to_numpy:
            c = tensor2backend(c,'numpy') 
        return c
    def log_amplitude(self,config,to_numpy=True):
        c = self.unsigned_amplitude(config,to_numpy=to_numpy)
        jnp = np if to_numpy else self.jnp
        return self.jnp.log(jnp.abs(c)),self.jnp.sign(c)
    def batch_pair_energies_from_plq(self,batch_key,new_cache=None):
        bix,tix,plq_types,pairs,direction = self.model.batched_pairs[batch_key]
        jnp = torch if new_cache else np
        cx = self.unsigned_amplitude(self.config,to_numpy=False)
        ex = dict()
        for where in pairs:
            ix1,ix2 = [self.flatten(site) for site in where]
            i1,i2 = self.config[ix1],self.config[ix2]
            if self.model.pair_valid(i1,i2): # term vanishes 
                i1_new,i2_new = self.pair_terms(i1,i2,self.spin)
                if i1_new is None:
                    ex[where,self.spin] = 0,0 
                else:
                    config_new = list(self.config)
                    config_new[ix1] = i1_new
                    config_new[ix2] = i2_new 
                    cx_new = self.unsigned_amplitude(config_new,to_numpy=False)

                    coeff = self.model.pair_coeff(*where)
                    coeff *= (-1)**(sum(self.config_to_spin(self.config)[ix1+1:ix2])%2)
                    cx_new *= coeff 
                    ex[where,self.spin] = cx_new, cx_new/cx 
            else:
                ex[where,self.spin] = 0,0
        return ex,cx,None
#class ORB(NN): # 1-particle orbital rotation
#    def __init__(self,nsite,nelec,spin,orth=True,**kwargs):
#        super().__init__(log_amp=False)
#        self.nsite = nsite
#        self.nelec = nelec
#        self.spin = spin
#        self.orth = orth 
#        self.nparam = nsite * nelec
#        self.block_dict = [(0,self.nparam)] 
#    def wfn2backend(self,backend=None,requires_grad=False):
#        backend = self.backend if backend is None else backend
#        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
#        ar.set_backend(tsr)
#        self.U = tensor2backend(self.U,backend=backend,requires_grad=requires_grad) 
#    def get_x(self):
#        return tensor2backend(self.U,'numpy').flatten()
#    def load_from_disc(self,fname):
#        self.U = np.load(fname) 
#        return self.U
#    def save_to_disc(self,U,fname,root=0): 
#        if RANK!=root:
#            return
#        np.save(fname+'.npy',U)
#    def update(self,x,fname=None,root=0):
#        self.U = x.reshape((self.nsite,self.nelec))
#        if self.orth:
#            self.U,_ = np.linalg.qr(self.U)
#        if fname is not None:
#            self.save_to_disc(self.U,fname,root=root) 
#        self.wfn2backend()
#    def extract_grad(self):
#        return tensor2backend(self.tensor_grad(self.U),'numpy').flatten()
#    def get_backend(self):
#        if isinstance(self.U,torch.Tensor):
#            jnp = torch
#            def _select(det,U):
#                return torch.index_select(U,0,torch.tensor(det[0]))
#        else:
#            jnp = np
#            def _select(det,U):
#                return U[det,:]
#        self._select = _select
#        return jnp
#    def forward(self,config,jnp):
#        det = np.where(np.array(config))
#        return jnp.linalg.det(self._select(det,self.U)) 
#    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
#        jnp = self.get_backend() 
#        c = self.forward(config,jnp) 
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#        return c
#    def log_amplitude(self,config,to_numpy=True):
#        jnp = self.get_backend() 
#        c = self.forward(config,jnp) 
#        c,s = jnp.log(jnp.abs(c)),jnp.sign(c)
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#            s = tensor2backend(s,'numpy') 
#        return c,s
#class ORB(NN):
#    def __init__(self,nsite,nelec,**kwargs):
#        self.nsite = nsite
#        self.nelec = nelec
#        self.nparam = 2 * nsite ** 2
#        self.block_dict = [(0,nsite**2),(nsite**2,self.nparam)]
#        super().__init__(to_spin=True,order='C',log_amp=False,**kwargs)
#    def init(self,eps,a=-1,b=1,fname=None):
#        c = b-a
#        self.K = (np.random.rand(2,self.nsite,self.nsite) * c + a) * eps
#        COMM.Bcast(self.K,root=0)
#        if fname is not None:
#            self.save_to_disc(self.K,fname)
#        self.U = [scipy.linalg.expm(self.K[ix]-self.K[ix].T)[:,:self.nelec[ix]] for ix in (0,1)]
#    def wfn2backend(self,backend=None,requires_grad=False):
#        backend = self.backend if backend is None else backend
#        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
#        ar.set_backend(tsr)
#        self.K = tensor2backend(self.K,backend=backend,requires_grad=requires_grad) 
#        expm = scipy.linalg.expm if backend=='numpy' else torch.linalg.matrix_exp
#        self.U = [expm(self.K[ix]-self.K[ix].T)[:,:self.nelec[ix]] for ix in (0,1)]
#    def get_x(self):
#        return tensor2backend(self.K,'numpy').flatten()
#    def load_from_disc(self,fname):
#        self.K = np.load(fname) 
#        self.U = [scipy.linalg.expm(self.K[ix]-self.K[ix].T)[:,:self.nelec[ix]] for ix in (0,1)]
#        return self.K
#    def save_to_disc(self,K,fname,root=0): 
#        if RANK!=root:
#            return
#        np.save(fname+'.npy',K)
#    def update(self,x,fname=None,root=0):
#        self.K = x.reshape((2,)+(self.nsite,)*2)
#        if fname is not None:
#            self.save_to_disc(self.K,fname,root=root) 
#        self.wfn2backend()
#    def extract_grad(self):
#        return tensor2backend(self.tensor_grad(self.K),'numpy').flatten()
#    def get_backend(self):
#        if isinstance(self.K,torch.Tensor):
#            jnp = torch
#            def _select(det,U):
#                return torch.index_select(U,0,torch.tensor(det[0]))
#        else:
#            jnp = np
#            def _select(det,U):
#                return U[det,:]
#        self._select = _select
#        return jnp,None
#    def forward(self,config,jnp):
#        config = config_to_ab(config) 
#        c = 1. 
#        for config_,U_ in zip(config,self.U):
#            det = np.where(np.array(config_))
#            c *= jnp.linalg.det(self._select(det,U_)) 
#        return c
#    def unsigned_amplitude(self,config,cache_top=None,cache_bot=None,to_numpy=True):
#        jnp,_ = self.get_backend() 
#        c = self.forward(config,jnp) 
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#        return c
#    def log_amplitude(self,config,to_numpy=True):
#        jnp,_ = self.get_backend() 
#        c = self.forward(config,jnp) 
#        c,s = jnp.log(jnp.abs(c)),jnp.sign(c)
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#            s = tensor2backend(s,'numpy') 
#        return c,s
#class SIGN(NN):
#    def __init__(self,n,afn='tanh',**kwargs):
#        assert afn in ('tanh','cos','sin') 
#        self.afn = afn
#        self.nparam = n 
#        self.block_dict = [(0,n)] 
#        super().__init__(log_amp=False,**kwargs)
#    def init(self,eps,fname=None):
#        self.w = (np.random.rand(self.nparam) * 2 - 1) * eps 
#        COMM.Bcast(self.w,root=0)
#        if fname is not None: 
#            self.save_to_disc(self.w,fname) 
#        return self.w
#    def init_from(self,w,eps,fname):
#        self.init(eps)
#        self.w[:len(w)] = w
#        if fname is not None: 
#            self.save_to_disc(self.w,fname) 
#        return self.w
#    def wfn2backend(self,backend=None,requires_grad=False):
#        backend = self.backend if backend is None else backend
#        tsr = np.zeros(1) if backend=='numpy' else torch.zeros(1)
#        ar.set_backend(tsr)
#        self.w = tensor2backend(self.w,backend=backend,requires_grad=requires_grad)
#    def get_x(self):
#        return tensor2backend(self.w,'numpy')
#    def load_from_disc(self,fname):
#        self.w = np.load(fname)
#        return self.w
#    def save_to_disc(self,w,fname,root=0):
#        if RANK!=root:
#            return
#        np.save(fname+'.npy',w)
#    def update(self,x,fname=None,root=0):
#        self.w = x
#        if fname is not None:
#            self.save_to_disc(self.w,fname,root=root) 
#        self.wfn2backend()
#    def extract_grad(self):
#        return tensor2backend(self.tensor_grad(self.w),'numpy')
#    def get_backend(self,c=None):
#        if isinstance(self.w,torch.Tensor):
#            if c is not None:
#                c = tensor2backend(c,backend='torch')
#            jnp = torch
#        else:
#            jnp = np
#        if self.afn=='tahn':
#            _afn = jnp.tahn
#        elif self.afn=='cos':
#            def _afn(x):
#                return jnp.cos(np.pi*x)    
#        elif self.afn=='sin':
#            def _afn(x):
#                return jnp.sin(np.pi*x)
#        else:
#            raise NotImplementedError
#        self._afn = _afn 
#        return jnp,c
#    def forward(self,c,jnp):
#        return self._afn(jnp.dot(c,self.w))
