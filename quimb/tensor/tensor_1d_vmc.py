
import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

# set tensor symmetry
import sys
import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)
from .torch_utils import SVD,QR
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)
this = sys.modules[__name__]
def set_options(pbc=False):
    this.pbc = pbc

from .tensor_1d import MatrixProductState
from .tensor_core import Tensor
def get_product_state(L,config=None,bdim=1,eps=0.,pdim=4,normalize=True,pbc=False):
    arrays = []
    for i in range(L):
        shape = [bdim] * 2
        if not pbc and (i==0 or i==L-1):
            shape.pop()
        shape = tuple(shape) + (pdim,)

        if config is None:
            data = np.ones(shape)
        else:
            data = np.zeros(shape)
            ix = config[i]
            data[(0,)*(len(shape)-1)+(ix,)] = 1.
        data += eps * np.random.rand(*shape)
        if normalize:
            data /= np.linalg.norm(data)
        arrays.append(data)
    return MatrixProductState(arrays) 
class ContractionEngine:
    def init_contraction(self,L,phys_dim=2):
        self.L = L
        self.nsite = L
        self.pbc = pbc

        self.data_map = dict()
        for i in range(phys_dim):
            data = np.zeros(phys_dim)
            data[i] = 1.
            self.data_map[i] = data
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def intermediate_sign(self,config=None,ix1=None,ix2=None):
        return 1.
    def safe_contract(self,tn):
        try:
            data = tn.contract()
        except (ValueError,IndexError):
            return None
        if self.backend=='torch':
            if isinstance(data,torch.Tensor):
                return data
            else:
                return None
        return data
    def _2backend(self,data,requires_grad):
        if self.backend=='torch':
            data = torch.tensor(data,requires_grad=requires_grad)
        return data
    def _torch2numpy(self,data,backend=None):
        backend = self.backend if backend is None else backend
        if backend=='torch':
            data = data.detach().numpy()
            if data.size==1:
                data = data.reshape(-1)[0]
        return data
    def _2numpy(self,data,backend=None):
        return self._torch2numpy(data,backend=backend)
    def tsr_grad(self,tsr,set_zero=True):
        grad = tsr.grad
        if set_zero:
            tsr.grad = None
        return grad 
    def get_bra_tsr(self,ci,i,append='',tn=None):
        tn = self.psi if tn is None else tn 
        inds = tn.site_ind(i)+append,
        tags = tn.site_tag(i),'BRA'
        data = self._2backend(self.data_map[ci],False)
        return Tensor(data=data,inds=inds,tags=tags)
    def get_mid_env(self,config,append='',psi=None):
        psi = self.psi if psi is None else psi 
        row = psi.copy()
        # compute mid env for row i
        for j in range(self.L-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(config[j],j,append=append,tn=row),virtual=True)
        return row
    def contract_mid_env(self,row):
        try: 
            for j in range(self.L-1,-1,-1):
                row.contract_tags(row.site_tag(j),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def get_all_lenvs(self,tn,jmax=None):
        jmax = self.L-2 if jmax is None else jmax
        first_col = tn.site_tag(0)
        lenvs = [None] * self.L
        for j in range(jmax+1): 
            tags = first_col if j==0 else (first_col,tn.site_tag(j))
            try:
                tn ^= tags
                lenvs[j] = tn.select(first_col,virtual=False)
            except (ValueError,IndexError):
                return lenvs
        return lenvs
    def get_all_renvs(self,tn,jmin=None):
        jmin = 1 if jmin is None else jmin
        last_col = tn.site_tag(self.L-1)
        renvs = [None] * self.L
        for j in range(self.L-1,jmin-1,-1): 
            tags = last_col if j==self.L-1 else (tn.site_tag(j),last_col)
            try:
                tn ^= tags
                renvs[j] = tn.select(last_col,virtual=False)
            except (ValueError,IndexError):
                return renvs
        return renvs
    def replace_sites(self,tn,sites,cis):
        for i,ci in zip(sites,cis): 
            bra = tn[tn.site_tag(i),'BRA']
            bra_target = self.get_bra_tsr(ci,i,tn=tn)
            bra.modify(data=bra_target.data,inds=bra_target.inds)
        return tn
    def site_grad(self,tn_plq,i):
        tid = tuple(tn_plq._get_tids_from_tags((tn_plq.site_tag(i),'KET'),which='all'))[0]
        ket = tn_plq._pop_tensor(tid)
        g = tn_plq.contract(tags=all,output_inds=ket.inds)
        return g.data 
    def update_plq_from_3row(self,plq,tn,bsz,psi=None):
        jmax = self.L - bsz
        psi = self.psi if psi is None else psi
        lenvs = self.get_all_lenvs(tn.copy(),jmax=jmax-1)
        renvs = self.get_all_renvs(tn.copy(),jmin=bsz)
        for j in range(jmax+1): 
            tags = [tn.site_tag(j+ix) for ix in range(bsz)]
            cols = tn.select(tags,which='any',virtual=False)
            try:
                if j>0:
                    other = cols
                    cols = lenvs[j-1]
                    cols.add_tensor_network(other,virtual=False)
                if j<jmax:
                    cols.add_tensor_network(renvs[j+bsz],virtual=False)
                plq[j,bsz] = cols.view_like_(psi)
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def get_plq_from_benvs(self,config,bsz,psi=None):
        psi = self.psi if psi is None else psi
        tn = self.get_mid_env(config) 
        try:
            tn.reorder(inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        plq = dict()
        if self.pbc:
            plq[self.L-1,self.L] = tn.copy()
        plq = self.update_plq_from_3row(plq,tn,bsz,psi=psi)
        return plq
    def get_grad_dict_from_plq(self,plq,cx,backend='numpy'):
        # gradient
        vx = dict()
        for (i0,bsz),tn in plq.items():
            if bsz == self.L:
                continue
            where = i0,i0+bsz-1
            for i in range(i0,i0+bsz):
                if i in vx:
                    continue
                vx[i] = self._2numpy(self.site_grad(tn.copy(),i)/cx[where],backend=backend)
        return vx
class AmplitudeFactory(ContractionEngine):
    def __init__(self,psi,phys_dim=2):
        super().init_contraction(psi.L,phys_dim=phys_dim)
        psi.add_tag('KET')

        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict()
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
    def config_sign(self,config=None):
        return 1.
    def get_constructors(self,psi):
        constructors = [None] * (self.L)
        for i in range(self.L):
            data = psi[i].data
            constructors[i] = data.shape,len(data.flatten())
        return constructors
    def get_block_dict(self):
        start = 0
        blk_dict = [(0,sum([size for _,size in self.constructors]))]
        return blk_dict 
    def tensor2vec(self,tsr,ix=None):
        return tsr.flatten()
    def dict2vecs(self,dict_):
        ls = [None] * len(self.constructors)
        for i,(_,size) in enumerate(self.constructors):
            vec = np.zeros(size)
            g = dict_.get(i,None)
            if g is not None:
                vec = self.tensor2vec(g,ix=i) 
            ls[i] = vec
        return ls
    def dict2vec(self,dict_):
        return np.concatenate(self.dict2vecs(dict_))
    def psi2vecs(self,psi=None):
        psi = self.psi if psi is None else psi
        ls = [None] * len(self.constructors)
        for i,(_,size) in enumerate(self.constructors):
            ls[i] = self.tensor2vec(psi[i].data,ix=i)
        return ls
    def psi2vec(self,psi=None):
        return np.concatenate(self.psi2vecs(psi)) 
    def get_x(self):
        return self.psi2vec()
    def split_vec(self,x):
        ls = [None] * len(self.constructors)
        start = 0
        for i,(_,size) in enumerate(self.constructors):
            stop = start + size
            ls[i] = x[start:stop]
            start = stop
        return ls 
    def vec2tensor(self,x,i):
        shape = self.constructors[i][0]
        return x.reshape(shape)
    def vec2dict(self,x): 
        ls = self.split_vec(x)
        return {i:x for i,x in zip(range(self.L),ls)} 
    def vec2psi(self,x,inplace=True): 
        psi = self.psi if inplace else self.psi.copy()
        ls = self.split_vec(x)
        for i in range(self.L):
            psi[i].modify(data=self.vec2tensor(ls[i],i))
        return psi
    def update(self,x,fname=None,root=0):
        psi = self.vec2psi(x,inplace=True)
        self.set_psi(psi) 
        if RANK==root:
            if fname is not None: # save psi to disc
                write_tn_to_disc(psi,fname,provided_filename=True)
        return psi
    def set_psi(self,psi):
        self.psi = psi
    def unsigned_amplitude(self,config):
        tn = self.get_mid_env(config)
        cx = self.safe_contract(tn)
        cx = 0. if cx is None else cx
        return cx  
    def amplitude(self,config):
        raise NotImplementedError
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def get_grad_from_plq(self,plq,cx,backend=None):
        backend = self.backend if backend is None else backend
        vx = self.get_grad_dict_from_plq(plq,cx,backend=backend)
        return self.dict2vec(vx) 
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
class Hamiltonian(ContractionEngine):
    def __init__(self,L,nbatch=1,phys_dim=2):
        super().init_contraction(L,phys_dim=phys_dim)
        self.nbatch = nbatch
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self._2backend(self.data_map[self.key],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return Tensor(data=data,inds=inds,tags=tags) 
    def _pair_energy_from_plq(self,tn,config,where):
        ix1,ix2 = where 
        i1,i2 = config[ix1],config[ix2] 
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        kixs = [tn.site_ind(site) for site in where]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip((ix1,ix2),kixs,bixs):
            tn[tn.site_tag(site),'BRA'].reindex_({kix:bix})
        tn.add_tensor(self.pair_tensor(bixs,kixs),virtual=True)
        ex = self.safe_contract(tn)
        if ex is None:
            return None
        return self.pair_coeff(*where) * ex 
    def _pair_energies_from_plq(self,plq,pairs,config):
        ex = dict()
        cx = dict()
        for where in pairs:
            key = self.pair_key(*where)

            tn = plq[key] 
            if tn is not None:
                eij = self._pair_energy_from_plq(tn.copy(),config,where) 
                if eij is not None:
                    ex[where] = eij

                cij = self._2numpy(tn.copy().contract())
                cx[where] = cij 
        return ex,cx
    def batch_pair_energies_from_plq(self,config,psi):
        # form plqs
        plq = dict()
        for bsz in self.plq_types:
            plq.update(self.get_plq_from_benvs(config,bsz,psi=psi))

        # compute energy numerator 
        ex,cx = self._pair_energies_from_plq(plq,pairs,config)
        return ex,cx,plq
    def batch_hessian_from_plq(self,config,amplitude_factory): # only used for Hessian
        self.backend = 'torch'
        psi = amplitude_factory.psi.copy()
        for i in range(self.L):
            psi[i].modify(data=self._2backend(psi[i].data,True))
        ex,cx,plq = self.batch_pair_energies_from_plq(config,psi)

        _,Hvx = self.parse_hessian(ex,peps,amplitude_factory)
        ex = sum([self._2numpy(eij)/cx[where] for where,eij in ex.items()])
        vx = self.get_grad_dict_from_plq(plq,cx,backend=self.backend) 
        return ex,Hvx,cx,vx
    def compute_local_energy_hessian_from_plq(self,config,amplitude_factory): 
        ar.set_backend(torch.zeros(1))

        ex,Hvx,cx,vx = self.batch_hessian_from_plq(config,amplitude_factory)  
        eu = self.compute_local_energy_eigen(config)
        ex += eu

        vx = amplitude_factory.dict2vec(vx)
        cx,err = self.contraction_error(cx)

        Hvx = Hvx/cx + eu*vx
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,err 
    def pair_energies_from_plq(self,config,amplitude_factory): 
        self.backend = 'numpy'
        plq = dict()
        for bsz in self.plq_sz:
            plq.update(amplitude_factory.get_plq_from_benvs(config,bsz))

        ex,cx = self._pair_energies_from_plq(plq,self.pairs,config)
        return ex,cx,plq
    def compute_local_energy_gradient_from_plq(self,config,amplitude_factory,compute_v=True):
        ex,cx,plq = self.pair_energies_from_plq(config,amplitude_factory)

        ex = sum([eij/cx[where] for where,eij in ex.items()])
        eu = self.compute_local_energy_eigen(config)
        ex += eu

        if not compute_v:
            cx,err = self.contraction_error(cx)
            return cx,ex,None,None,err 
        #vx = amplitude_factory.get_grad_from_plq(plq,cx)  
        vx = self.amplitude_gradient_deterministic(config,amplitude_factory)
        cx,err = self.contraction_error(cx)
        #print(ex,cx,err)
        return cx,ex,vx,None,err
    def amplitude_gradient_deterministic(self,config,amplitude_factory):
        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))
        cache_top = dict()
        cache_bot = dict()
        psi = amplitude_factory.psi.copy()
        for i in range(self.L):
            psi[i].modify(data=self._2backend(psi[i].data,True))

        tn = self.get_mid_env(config,psi=psi)
        cx = tn.contract() 

        cx.backward()
        vx = dict()
        for i in range(self.L):
            vx[i] = self.tsr_grad(psi[i].data)  
        vx = {site:self._2numpy(vij) for site,vij in vx.items()}
        vx = amplitude_factory.dict2vec(vx)  
        cx = self._2numpy(cx)
        return vx/cx
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        if compute_Hv:
            return self.compute_local_energy_hessian_from_plq(config,amplitude_factory)
        else:
            return self.compute_local_energy_gradient_from_plq(config,amplitude_factory,compute_v=compute_v)
    def parse_hessian(self,ex,psi,amplitude_factory):
        if len(ex)==0:
            return 0.,0.
        ex_num = sum(ex.values())
        ex_num.backward()
        Hvx = dict()
        for i in range(self.L):
            Hvx[i] = self._2numpy(self.tsr_grad(psi[i].data))
        return self._2numpy(ex_num),amplitude_factory.dict2vec(Hvx)  
    def contraction_error(self,cx):
        cx = np.array(list(cx.values()))
        return np.mean(cx),0.
    def pairs_nn(self,d=1):
        ls = [] 
        for j in range(self.L):
            if j+d<self.L:
                where = j,j+d
                ls.append(where)
            else:
                if self.pbc:
                    where = j,(j+d)%self.L
                    ls.append(where)
        return ls
    def pair_key(self,i,j):
        return i,abs(j-i)+1
class ExchangeSampler2:
    def __init__(self,L,seed=None,burn_in=0):
        self.L = L
        self.nsite = L

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.backend = 'numpy'
    def preprocess(self):
        self._burn_in()
    def _burn_in(self,config=None,burn_in=None):
        if config is not None:
            self.config = config 
        self.px = self.amplitude_factory.prob(self.config)

        if RANK==0:
            print('\tprob=',self.px)
            return 
        t0 = time.time()
        burn_in = self.burn_in if burn_in is None else burn_in
        for n in range(burn_in):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
    def new_pair(self,i1,i2):
        return i2,i1
    def _new_pair(self,i,bsz):
        ix1,ix2 = i,(i+1)%self.L
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.amplitude_factory.pair_valid(i1,i2): # continue
            return (None,) * 3
        i1_new,i2_new = self.new_pair(i1,i2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        return (ix1,ix2),(i1_new,i2_new),tuple(config_new)
    def update_pair(self,i,bsz,cols,tn):
        sites,config_sites,config_new = self._new_pair(i,bsz)
        if config_sites is None:
            return tn

        cols = self.amplitude_factory.replace_sites(cols,sites,config_sites) 
        py = self.amplitude_factory.safe_contract(cols)
        if py is None:
            return tn 
        py = py ** 2

        try:
            acceptance = py / self.px
        except ZeroDivisionError:
            acceptance = 1. if py > self.px else 0.
        if self.rng.uniform() < acceptance: # accept, update px & config & env_m
            self.px = py
            self.config = config_new
            tn = self.amplitude_factory.replace_sites(tn,sites,config_sites)
        return tn
    def _get_cols_forward(self,first_col,j,bsz,tn,renvs):
        tags = [tn.site_tag(j+ix) for ix in range(bsz)]
        cols = tn.select(tags,which='any',virtual=False)
        if j>0:
            other = cols
            cols = tn.select(first_col,virtual=False)
            cols.add_tensor_network(other,virtual=False)
        if j<self.L - bsz:
            cols.add_tensor_network(renvs[j+bsz],virtual=False)
        cols.view_like_(tn)
        return cols
    def sweep_col_forward(self,tn,bsz):
        try:
            tn.reorder(inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        renvs = self.amplitude_factory.get_all_renvs(tn.copy(),jmin=bsz)
        first_col = tn.site_tag(0)
        for j in range(self.L - bsz + 1): 
            cols = self._get_cols_forward(first_col,j,bsz,tn,renvs)
            tn = self.update_pair(j,bsz,cols,tn) 
            tn ^= first_col,tn.site_tag(j) 
    def _get_cols_backward(self,last_col,j,bsz,tn,lenvs):
        tags = [tn.site_tag(j+ix) for ix in range(bsz)] 
        cols = tn.select(tags,which='any',virtual=False)
        if j>0:
            other = cols
            cols = lenvs[j-1]
            cols.add_tensor_network(other,virtual=False)
        if j<self.L - bsz:
            cols.add_tensor_network(tn.select(last_col,virtual=False),virtual=False)
        cols.view_like_(tn)
        return cols
    def sweep_col_backward(self,tn,bsz):
        try:
            tn.reorder(inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.amplitude_factory.get_all_lenvs(tn.copy(),jmax=self.L-1-bsz)
        last_col = tn.site_tag(self.L-1)
        for j in range(self.L - bsz,-1,-1): # Ly-1,...,1
            cols = self._get_cols_backward(last_col,j,bsz,tn,lenvs)
            tn = self.update_pair(j,bsz,cols,tn) 
            tn ^= tn.site_tag(j+bsz-1),last_col
    def sample(self):
        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward
        tn = self.amplitude_factory.get_mid_env(self.config)
        sweep_col(tn,2)
        return self.config,self.px
