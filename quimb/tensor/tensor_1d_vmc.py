
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
            data[(0,)*len(shape-1)+(ix,)] = 1.
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
        g = tn_plq.contract(output_inds=ket.inds)
        return g.data 
    def update_plq_from_3row(self,plq,tn,bsz,psi=None):
        jmax = self.L - bsz
        psi = self.psi if psi is None else psi
        try:
            tn.reorder(inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.get_all_lenvs(tn.copy(),jmax=jmax-1)
        renvs = self.get_all_renvs(tn.copy(),jmin=y_bsz)
        for j in range(jmax+1): 
            tags = [tn.site_tag(j+ix) for ix in range(bsz)]
            cols = tn.select(tags,which='any',virtual=False)
            try:
                if j>0:
                    other = cols
                    cols = lenvs[j-1]
                    cols.add_tensor_network(other,virtual=False)
                if j<jmax:
                    cols.add_tensor_network(renvs[j+y_bsz],virtual=False)
                plq[i,bsz] = cols.view_like_(psi)
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def get_plq_from_benvs(self,config,bsz,psi=None):
        psi = self.psi if psi is None else psi
        tn = self.get_mid_env(config) 
        plq = dict()
        if self.pbc:
            plq[0,self.L-1] = tn.copy()
        plq = self.update_plq_from_3row(plq,tn,bsz,psi=psi)
        return plq
    def get_grad_dict_from_plq(self,plq,cx,backend='numpy'):
        # gradient
        vx = dict()
        for (i0,bsz),tn in plq.items():
            where = i0,i0+bsz-1
            for i in range(i0,i0+x_bsz):
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
            ls[ix] = vec
        return ls
    def dict2vec(self,dict_):
        return np.concatenate(self.dict2vecs(dict_))
    def psi2vecs(self,psi=None):
        psi = self.psi if psi is None else psi
        ls = [None] * len(self.constructors)
        for i,(_,size) in enumerate(self.constructors):
            ls[i] = self.tensor2vec(psi[psi.site_tag(i)].data,ix=i)
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
            psi[psi.site_tag(i)].modify(data=self.vec2tensor(ls[i],i))
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
        for site,kix,bix in zip(where,kixs,bixs):
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
        vx = amplitude_factory.get_grad_from_plq(plq,cx)  
        cx,err = self.contraction_error(cx)
        #print(ex,cx,err)
        return cx,ex,vx,None,err
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
                    where = (j+d)%self.L,j
                    ls.append(where)
        return ls
    def pair_key(self,site1,site2):
        d = site2-site1
        return site1,d
class DenseSampler:
    def __init__(self,L,nspin,exact=False,seed=None,thresh=1e-14):
        self.L = L
        self.nsite = self.L
        self.nspin = nspin

        self.all_configs = self.get_all_configs()
        self.ntotal = len(self.all_configs)
        if RANK==0:
            print('ntotal=',self.ntotal)
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
        self.amplitude_factory = None
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
            plocal.append(self.amplitude_factory.prob(config))
        plocal = np.array(plocal)
         
        COMM.Allgatherv(plocal,[ptotal,self.count,self.disp,MPI.DOUBLE])
        nonzeros = []
        for ix,px in enumerate(ptotal):
            if px > self.thresh:
                nonzeros.append(ix) 
        n = np.sum(ptotal)
        ptotal /= n 
        self.p = ptotal
        if RANK==SIZE-1:
            print('\tdense amplitude time=',time.time()-t0)

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
    def get_all_configs(self):
        assert isinstance(self.nspin,tuple)
        sites = list(range(self.nsite))
        occs = list(itertools.combinations(sites,self.nspin[0]))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = [0] * (self.nsite) 
            for ix in occ:
                config[ix] = 1
            configs[i] = tuple(config)
        return configs
    def sample(self):
        flat_idx = self.rng.choice(self.flat_indexes,p=self.p)
        config = self.all_configs[flat_idx]
        omega = self.p[flat_idx]
        return config,omega

