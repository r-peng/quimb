import time,itertools,psutil
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
def set_options(pbc=False,deterministic=False,**compress_opts):
    this.pbc = pbc
    this.deterministic = True if pbc else deterministic
    this.compress_opts = compress_opts
def flatten(i,j,Ly): # flattern site to row order
    return i*Ly+j
def flat2site(ix,Lx,Ly): # ix in row order
    return ix//Ly,ix%Ly
import pickle,uuid
#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
def load_tn_from_disc(fname, delete_file=False):
    if type(fname) != str:
        data = fname
    else:
        with open(fname,'rb') as f:
            data = pickle.load(f)
    return data
def write_tn_to_disc(tn, fname, provided_filename=False):
    with open(fname, 'wb') as f:
        pickle.dump(tn, f)
    return fname
from .tensor_2d import PEPS
def get_product_state(Lx,Ly,config,bdim=1,eps=None):
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [bdim] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (2,)

            data = np.zeros(shape) 
            ix = flatten(i,j,Ly)
            ix = config[ix]
            data[(0,)*(len(shape)-1)+(ix,)] = 1.
            if eps is not None:
                data += eps * np.random.rand(*shape)
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
from .tensor_core import Tensor,TensorNetwork,rand_uuid
def peps2pbc(peps):
    vbonds = [rand_uuid() for j in range(peps.Ly)]
    hbonds = [rand_uuid() for i in range(peps.Lx)]
    # i = 0
    for j in range(peps.Ly):
        tsr = peps[0,j]
        bdim,pdim = tsr.data.shape[0],tsr.data.shape[-1]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        d = vbonds[j]
        if j==0:
            u,r,p = tsr.inds
            l = hbonds[0]
        elif j==peps.Ly-1:
            u,l,p = tsr.inds
            r = hbonds[0]
        else:
            u,r,l,p = tsr.inds
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)

    for i in range(1,peps.Lx-1):
        tsr = peps[i,0]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        u,r,d,p = tsr.inds
        l = hbonds[i]
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)

        tsr = peps[i,peps.Ly-1]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        u,d,l,p = tsr.inds
        r = hbonds[i]
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)

    # i = Lx-1
    for j in range(peps.Ly):
        tsr = peps[peps.Lx-1,j]
        bdim,pdim = tsr.data.shape[0],tsr.data.shape[-1]
        data = np.random.rand(*((bdim,)*4+(pdim,)))
        u = vbonds[j]
        if j==0:
            r,d,p = tsr.inds
            l = hbonds[peps.Lx-1]
        elif j==peps.Ly-1:
            d,l,p = tsr.inds
            r = hbonds[peps.Lx-1]
        else:
            r,d,l,p = tsr.inds
        inds = u,r,d,l,p
        tsr.modify(data=data,inds=inds)
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            data = np.ones(2)
            inds = peps.site_ind(i,j),
            peps.add_tensor(Tensor(data=data,inds=inds))
    return peps
####################################################################################
# amplitude fxns 
####################################################################################
class ContractionEngine:
    def init_contraction(self,Lx,Ly,phys_dim=2):
        self.Lx,self.Ly = Lx,Ly
        self.pbc = pbc
        self.deterministic = deterministic
        if self.deterministic:
            self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
        self.compress_opts = compress_opts

        self.data_map = dict()
        for i in range(phys_dim):
            data = np.zeros(phys_dim)
            data[i] = 1.
            self.data_map[i] = data
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def intermediate_sign(self,config=None,ix1=None,ix2=None):
        return 1.
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
    def get_bra_tsr(self,peps,ci,i,j,append=''):
        inds = peps.site_ind(i,j)+append,
        tags = peps.site_tag(i,j),peps.row_tag(i),peps.col_tag(j),'BRA'
        data = self._2backend(self.data_map[ci],False)
        return Tensor(data=data,inds=inds,tags=tags)
    def get_mid_env(self,i,peps,config,append=''):
        row = peps.select(peps.row_tag(i)).copy()
        key = config[i*peps.Ly:(i+1)*peps.Ly]
        # compute mid env for row i
        for j in range(row.Ly-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(row,key[j],i,j,append=append),virtual=True)
        return row
    def contract_mid_env(self,i,row):
        try: 
            for j in range(row.Ly-1,-1,-1):
                row.contract_tags(row.site_tag(i,j),inplace=True)
        except (ValueError,IndexError):
            row = None 
        return row
    def canonize_row(self,tn,i,j,dist,step):
        seq = [(j+step*ix)%self.Ly for ix in range(dist+1)]
        for ix in range(dist,0,-1): # left canonize to j
            tag1,tag2 = tn.site_tag(i,seq[ix]),tn.site_tag(i,seq[ix-1])
            tn.canonize_between(tag1,tag2,absorb='right')
        return tn
    def compress_row(self,tn,i,max_bond=None,cutoff=1e-10,canonize=None):
        for j in range(self.Ly-1): # compress between j,j+1
            if canonize is not None: # canonize=canonize_dist
                if j==0:
                    tn = self.canonize_row(tn,i,j,canonize,-1)
                tn = self.canonize_row(tn,i,j+1,canonize,1)
            tn.compress_between(tn.site_tag(i,j),tn.site_tag(i,j+1),max_bond=max_bond,cutoff=cutoff,
                                absorb='right')
        return tn
    def contract_boundary_single(self,tn,i,iprev):
        for j in range(self.Ly):
            tag1,tag2 = tn.site_tag(iprev,j),tn.site_tag(i,j)
            tn.contract_((tag1,tag2),which='any')
        return self.compress_row(tn,i,**self.compress_opts)
    def get_bot_env(self,i,row,env_prev,config,cache):
        # contract mid env for row i with prev bot env 
        key = config[:(i+1)*row.Ly]
        if key in cache: # reusable
            return cache[key]
        row = self.contract_mid_env(i,row)
        if i==0:
            cache[key] = row
            return row
        if row is None:
            cache[key] = row
            return row
        if env_prev is None:
            cache[key] = None 
            return None
        tn = env_prev.copy()
        tn.add_tensor_network(row,virtual=True)
        try:
            if self.pbc:
                tn = self.contract_boundary_single(tn,i,i-1)
            else:
                tn.contract_boundary_from_bottom_(xrange=(i-1,i),**self.compress_opts)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_all_bot_envs(self,peps,config,cache_bot,imax=None,append=''):
        # imax for bot env
        imax = peps.Lx-2 if imax is None else imax
        env_prev = None
        for i in range(imax+1):
             row = self.get_mid_env(i,peps,config,append=append)
             env_prev = self.get_bot_env(i,row,env_prev,config,cache_bot)
        return env_prev
    def get_top_env(self,i,row,env_prev,config,cache):
        # contract mid env for row i with prev top env 
        key = config[i*row.Ly:]
        if key in cache: # reusable
            return cache[key]
        row = self.contract_mid_env(i,row)
        if i==row.Lx-1:
            cache[key] = row
            return row
        if row is None:
            cache[key] = row
            return row
        if env_prev is None:
            cache[key] = None 
            return None
        tn = row
        tn.add_tensor_network(env_prev.copy(),virtual=True)
        try:
            if self.pbc:
                tn = self.contract_boundary_single(tn,i,i+1)
            else:
                tn.contract_boundary_from_top_(xrange=(i,i+1),**self.compress_opts)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_all_top_envs(self,peps,config,cache_top,imin=None,append=''):
        imin = 1 if imin is None else imin
        env_prev = None
        for i in range(peps.Lx-1,imin-1,-1):
             row = self.get_mid_env(i,peps,config,append=append)
             env_prev = self.get_top_env(i,row,env_prev,config,cache_top)
        return env_prev
    def get_all_benvs(self,peps,config,cache_bot,cache_top,x_bsz=1,compute_bot=True,compute_top=True):
        env_bot = None
        env_top = None
        if compute_bot: 
            imax = self.rix1 if self.deterministic else self.Lx-1-x_bsz
            env_bot = self.get_all_bot_envs(peps,config,cache_bot,imax=imax)
        if compute_top:
            imin = self.rix2 if self.deterministic else x_bsz
            env_top = self.get_all_top_envs(peps,config,cache_top,imin=imin)
        return env_bot,env_top
    def get_all_lenvs(self,tn,jmax=None):
        jmax = tn.Ly-2 if jmax is None else jmax
        first_col = tn.col_tag(0)
        lenvs = [None] * tn.Ly
        for j in range(jmax+1): 
            tags = first_col if j==0 else (first_col,tn.col_tag(j))
            try:
                tn ^= tags
                lenvs[j] = tn.select(first_col).copy()
            except (ValueError,IndexError):
                return lenvs
        return lenvs
    def get_all_renvs(self,tn,jmin=None):
        jmin = 1 if jmin is None else jmin
        last_col = tn.col_tag(tn.Ly-1)
        renvs = [None] * tn.Ly
        for j in range(tn.Ly-1,jmin-1,-1): 
            tags = last_col if j==tn.Ly-1 else (tn.col_tag(j),last_col)
            try:
                tn ^= tags
                renvs[j] = tn.select(last_col).copy()
            except (ValueError,IndexError):
                return renvs
        return renvs
    def replace_sites(self,tn,sites,cis):
        for (i,j),ci in zip(sites,cis): 
            bra = tn[tn.site_tag(i,j),'BRA']
            bra_target = self.get_bra_tsr(tn,ci,i,j)
            bra.modify(data=bra_target.data.copy(),inds=bra_target.inds)
        return tn
    def site_grad(self,tn_plq,i,j):
        tid = tuple(tn_plq._get_tids_from_tags((tn_plq.site_tag(i,j),'KET'),which='all'))[0]
        ket = tn_plq._pop_tensor(tid)
        g = tn_plq.contract(output_inds=ket.inds)
        return g.data 
class AmplitudeFactory(ContractionEngine):
    def __init__(self,psi):
        super().init_contraction(psi.Lx,psi.Ly)
        psi.add_tag('KET')
        self.constructors = self.get_constructors(psi)
        self.get_block_dict()

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
        self.small_mem = True
    def config_sign(self,config=None):
        return 1.
    def get_constructors(self,peps):
        constructors = [None] * (peps.Lx * peps.Ly)
        for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
            data = peps[peps.site_tag(i,j)].data
            ix = flatten(i,j,peps.Ly)
            constructors[ix] = data.shape,len(data.flatten()),(i,j)
        return constructors
    def get_block_dict(self):
        start = 0
        ls = [None] * len(self.constructors)
        for ix,(_,size,_) in enumerate(self.constructors):
            stop = start + size
            ls[ix] = start,stop
            start = stop
        self.block_dict = ls
        return ls
    def tensor2vec(self,tsr,ix=None):
        return tsr.flatten()
    def dict2vecs(self,dict_):
        ls = [None] * len(self.constructors)
        for ix,(_,size,site) in enumerate(self.constructors):
            vec = np.zeros(size)
            g = dict_.get(site,None)
            if g is not None:
                vec = self.tensor2vec(g,ix=ix) 
            ls[ix] = vec
        return ls
    def dict2vec(self,dict_):
        return np.concatenate(self.dict2vecs(dict_))
    def psi2vecs(self,psi=None):
        psi = self.psi if psi is None else psi
        ls = [None] * len(self.constructors)
        for ix,(_,size,site) in enumerate(self.constructors):
            ls[ix] = self.tensor2vec(psi[psi.site_tag(*site)].data,ix=ix)
        return ls
    def psi2vec(self,psi=None):
        return np.concatenate(self.psi2vecs(psi)) 
    def get_x(self):
        return self.psi2vec()
    def split_vec(self,x):
        ls = [None] * len(self.constructors)
        start = 0
        for ix,(_,size,_) in enumerate(self.constructors):
            stop = start + size
            ls[ix] = x[start:stop]
            start = stop
        return ls 
    def vec2tensor(self,x,ix):
        shape = self.constructors[ix][0]
        return x.reshape(shape)
    def vec2dict(self,x): 
        dict_ = dict() 
        ls = self.split_vec(x)
        for ix,(_,_,site) in enumerate(self.constructors):
            dict_[site] = self.vec2tensor(ls[ix],ix) 
        return dict_ 
    def vec2psi(self,x,inplace=True): 
        psi = self.psi if inplace else self.psi.copy()
        ls = self.split_vec(x)
        for ix,(_,_,site) in enumerate(self.constructors):
            psi[psi.site_tag(*site)].modify(data=self.vec2tensor(ls[ix],ix))
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

        self.compute_bot = True
        self.compute_top = True
        self.cache_bot = dict()
        self.cache_top = dict()
    def update_scheme(self,benv_dir):
        if self.deterministic: # contract both sides to the middle
            benv_dir = 0
 
        if benv_dir == 1:
            self.compute_bot = True
            self.compute_top = False
        elif benv_dir == -1:
            self.compute_top = True
            self.compute_bot = False
        elif benv_dir == 0:
            self.compute_top = True
            self.compute_bot = True 
        else:
            raise NotImplementedError
    def unsigned_amplitude(self,config):
        # should only be used to:
        # 1. compute dense probs
        # 2. initialize MH sampler
        env_bot,env_top = self.get_all_benvs(self.psi,config,self.cache_bot,self.cache_top,
                              x_bsz=1,compute_bot=self.compute_bot,compute_top=self.compute_top)
        if env_bot is None and env_top is None:
            return 0.
        if self.deterministic:
            tn = env_bot.copy()
            tn.add_tensor_network(env_top.copy(),virtual=True)
        elif self.compute_bot: 
            tn = env_bot.copy()
            tn.add_tensor_network(self.get_mid_env(self.Lx-1,self.psi,config),virtual=True) 
        elif self.compute_top:
            tn = self.get_mid_env(0,self.psi,config)
            tn.add_tensor_network(env_top.copy(),virtual=True)
        try:
            return tn.contract()
        except (ValueError,IndexError):
            return 0.
    def amplitude(self,config):
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def get_grad_from_plq(self,plq,cx,backend='numpy'):
        self.dmrg = False
        if self.dmrg:
            fn = self.get_grad_from_plq_dmrg
        else:
            fn = self.get_grad_from_plq_full
        return fn(plq,cx,backend=backend)
    def get_grad_from_plq_full(self,plq,cx,backend='numpy'):
        # gradient
        vx = dict()
        for ((i0,j0),(x_bsz,y_bsz)),tn in plq.items():
            for i in range(i0,i0+x_bsz):
                for j in range(j0,j0+y_bsz):
                    if (i,j) in vx:
                        continue
                    cij = self._2numpy(cx[i0,j0],backend=backend)
                    vx[i,j] = self._2numpy(self.site_grad(tn.copy(),i,j)/cij,backend=backend)
        return self.dict2vec(vx) 
    def get_grad_from_plq_dmrg(self,plq,cx,backend='numpy'):
        i,j = self.flat2site(self.ix)
        _,(x_bsz,y_bsz) = list(plq.keys())[0]
        # all plqs the site could be in 
        keys = [(i0,j0) for i0 in range(i,i-x_bsz,-1) for j0 in range(j,j-y_bsz,-1)]
        for i0,j0 in keys:
            key = (i0,j0),(x_bsz,y_bsz)
            tn = plq.get(key,None)
            if tn is None:
                continue
            # returns as soon as ftn_plq exist
            cij = cx.get((i0,j0),1.)
            vx = self.site_grad(tn.copy(),i,j) / cij
            if backend=='torch':
                vij = vij.detach().numpy()
            vx = self.tensor2vec(vx,ix=self.ix)
            return cx,vx 
        # ftn_plq doesn't exist due to contraction error
        start,stop = self.block_dict[self.ix]
        vx = np.zeros(stop-start)
        return vx
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
####################################################################################
# ham class 
####################################################################################
class Hamiltonian(ContractionEngine):
    def __init__(self,Lx,Ly,discard=None,grad_by_ad=False,tmpdir=None,log_every=1):
        super().init_contraction(Lx,Ly)
        self.discard = discard 
        self.grad_by_ad = True if deterministic else grad_by_ad
        self.tmpdir = tmpdir
        if self.tmpdir is not None:
            self.log_every = log_every
            self.t0 = time.time()
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self._2backend(self.data_map[self.key],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return Tensor(data=data,inds=inds,tags=tags) 
    def pair_energy_from_plq(self,tn,config,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if not self.pair_valid(i1,i2): # term vanishes 
            return 0.
        kixs = [tn.site_ind(*site) for site in [site1,site2]]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip([site1,site2],kixs,bixs):
            tn[tn.site_tag(*site),'BRA'].reindex_({kix:bix})
        tn.add_tensor(self.pair_tensor(bixs,kixs),virtual=True)
        try:
            return self.pair_coeff(site1,site2) * tn.contract() 
        except (ValueError,IndexError):
            return 0.
    def update_plq_from_3row(self,plq,tn,i,x_bsz,y_bsz,peps):
        jmax = peps.Ly - y_bsz
        try:
            tn.reorder('col',inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.get_all_lenvs(tn.copy(),jmax=jmax-1)
        renvs = self.get_all_renvs(tn.copy(),jmin=y_bsz)
        for j in range(jmax+1): 
            tags = [tn.col_tag(j+ix) for ix in range(y_bsz)]
            cols = tn.select(tags,which='any').copy()
            try:
                if j>0:
                    other = cols
                    cols = lenvs[j-1]
                    cols.add_tensor_network(other,virtual=True)
                if j<jmax:
                    cols.add_tensor_network(renvs[j+y_bsz],virtual=True)
                plq[(i,j),(x_bsz,y_bsz)] = cols.view_like_(peps)
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,peps,cache_bot,cache_top):
        #if self.compute_bot and self.compute_top:
        #    raise ValueError
        imax = peps.Lx-x_bsz
        plq = dict()
        for i in range(imax+1):
            try:
                tn = self.get_mid_env(i,peps,config)
                for ix in range(1,x_bsz):
                    tn.add_tensor_network(self.get_mid_env(i+ix,peps,config),virtual=True)
                if i>0:
                    other = tn 
                    tn = cache_bot[config[:i*peps.Ly]].copy()
                    tn.add_tensor_network(other,virtual=True)
                if i<imax:
                    tn.add_tensor_network(cache_top[config[(i+x_bsz)*peps.Ly:]].copy(),virtual=True)
                plq = self.update_plq_from_3row(plq,tn,i,x_bsz,y_bsz,peps)
            except AttributeError:
                continue
        return plq
    def pair_energies_from_plq(self,config,peps,cache_bot,cache_top):
        x_bsz_min = min([x_bsz for x_bsz,_ in self.plq_sz])
        self.get_all_benvs(peps,config,cache_bot,cache_top)

        plq = dict()
        for x_bsz,y_bsz in self.plq_sz:
            plq.update(self.get_plq_from_benvs(config,x_bsz,y_bsz,peps,cache_bot,cache_top))

        ex = dict()
        cx = dict()
        for (site1,site2) in self.pairs:
            key = self.pair_key(site1,site2)

            tn = plq.get(key,None) 
            if tn is not None:
                ex[site1,site2] = self.pair_energy_from_plq(tn.copy(),config,site1,site2) 

                if site1 in cx:
                    cij = cx[site1]
                elif site2 in cx:
                    cij = cx[site2]
                else:
                    cij = tn.copy().contract()
                cx[site1] = cij 
                cx[site2] = cij 
        return ex,cx,plq
    def pair_energy_deterministic(self,config,peps,site1,site2,cache_bot,cache_top,sign_fn):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return 0. 
        imin = min(self.rix1+1,site1[0]) 
        imax = max(self.rix2-1,site2[0]) 
        top = None if imax==peps.Lx-1 else cache_top[config[(imax+1)*peps.Ly:]]
        bot = None if imin==0 else cache_bot[config[:imin*peps.Ly]]
        eij = 0.
        coeff_comm = self.intermediate_sign(config,ix1,ix2) * self.pair_coeff(site1,site2)
        for i1_new,i2_new,coeff in self.pair_terms(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            sign_new = sign_fn(config_new)

            bot_term = None if bot is None else bot.copy()
            for i in range(imin,self.rix1+1):
                row = self.get_mid_env(i,peps,config_new,append='')
                bot_term = self.get_bot_env(i,row,bot_term,config_new,cache_bot)
            if imin > 0 and bot_term is None:
                continue

            top_term = None if top is None else top.copy()
            for i in range(imax,self.rix2-1,-1):
                row = self.get_mid_env(i,peps,config_new,append='')
                top_term = self.get_top_env(i,row,top_term,config_new,cache_top)
            if imax < peps.Lx-1 and top_term is None:
                continue

            tn = bot_term.copy()
            tn.add_tensor_network(top_term.copy(),virtual=True)
            try:
                eij += coeff * sign_new * tn.contract() 
            except (ValueError,IndexError):
                continue
        return eij * coeff_comm
    def pair_energies_deterministic(self,config,peps,cache_bot,cache_top,sign_fn):
        env_bot,env_top = self.get_all_benvs(peps,config,cache_bot,cache_top)
        tn = env_bot.copy()
        tn.add_tensor_network(env_top.copy(),virtual=True)
        sign = sign_fn(config) 
        cx = tn.contract() * sign
        ex = dict()
        for (site1,site2) in self.pairs:
            ex[site1,site2] = self.pair_energy_deterministic(config,peps,site1,site2,cache_bot,cache_top,sign_fn) * sign
        return ex,cx,sign 
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        if self.tmpdir is not None:
             if RANK % self.log_every==0:
                 mem = psutil.virtual_memory()
                 with open(self.tmpdir+f'RANK{RANK}.log','a') as f:
                     #f.write(f'time={time.time()-self.t0},percent={mem.percent},cache_bot={len(amplitude_factory.cache_bot)},cache_top={len(amplitude_factory.cache_top)}\n')
                     f.write(f'time={time.time()-self.t0},percent={mem.percent}\n')
        if (compute_v and self.grad_by_ad) or compute_Hv:
            # torch only used here
            self.backend = 'torch'
            ar.set_backend(torch.zeros(1))
            cache_top = dict()
            cache_bot = dict()
            peps = amplitude_factory.psi.copy()
            for i,j in itertools.product(range(self.Lx),range(self.Ly)):
                peps[i,j].modify(data=self._2backend(peps[i,j].data,True))
        else:
            self.backend = 'numpy'
            peps = amplitude_factory.psi
            cache_bot = amplitude_factory.cache_bot
            cache_top = amplitude_factory.cache_top

        if self.deterministic:
            sign_fn = amplitude_factory.config_sign
            ex_dict,cx,sign = self.pair_energies_deterministic(config,peps,cache_bot,cache_top,sign_fn)
            err = 0.
        else:
            ex_dict,cx_dict,plq = self.pair_energies_from_plq(config,peps,cache_bot,cache_top) 
            sign = 1.
            cx,err = self.contraction_error(cx_dict)
            if self.discard is not None: # discard sample if contraction error too large
                if err > self.discard: 
                    self.update_cache(amplitude_factory,cache_top,cache_bot)
                    return (None,) * 5

        # energy
        ex_num = sum(ex_dict.values())
        if self.deterministic:
            ex = ex_num / cx
        else:
            ex = sum([eij/cx_dict[site1] for (site1,_),eij in ex_dict.items()])
        eu = self.compute_local_energy_eigen(config)
        ex = self._2numpy(ex) + eu
        if not compute_v:
            if self.backend=='torch':
                ar.set_backend(np.zeros(1))
            return cx,ex,None,None,err 

        # gradient
        if self.deterministic or self.grad_by_ad:
            loss = cx if self.deterministic else cx_dict[self.Lx//2,0]
            loss.backward(retain_graph=compute_Hv)
            vx = dict()
            for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
                vx[i,j] = self.tsr_grad(peps[i,j].data) / cx 
            vx = {site:self._2numpy(vij) for site,vij in vx.items()}
            vx = amplitude_factory.dict2vec(vx)  
        else:
            vx = amplitude_factory.get_grad_from_plq(plq,cx_dict,backend=self.backend) * sign 
        if not compute_Hv:
            if self.backend=='torch':
                ar.set_backend(np.zeros(1))
            return cx,ex,vx,None,err

        # back propagates energy gradient
        t0 = time.time()
        ex_num.backward()
        Hvx = dict()
        for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
            Hvx[i,j] = self.tsr_grad(peps[i,j].data) / cx
        Hvx = {site:self._2numpy(Hvij) for site,Hvij in Hvx.items()}
        Hvx = amplitude_factory.dict2vec(Hvx) 
        Hvx += eu * vx 
        if self.backend=='torch':
            ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,err
    def contraction_error(self,cx):
        nsite = self.Lx * self.Ly
        if self.backend=='torch':
            sqmean = sum(cij.pow(2) for cij in cx.values()) / nsite # mean(px) 
            mean = sum(cij for cij in cx.values()) / nsite
            err = sqmean - mean.pow(2)
            return self._2numpy(mean),self._2numpy(err.abs())
        else:
            sqmean = sum(cij**2 for cij in cx.values()) / nsite
            mean = sum(cij for cij in cx.values()) / nsite
            err = sqmean - mean**2
            return mean,np.fabs(err)
    def nn_pairs(self):
        ls = [] 
        for i in range(self.Lx):
            for j in range(self.Ly):
                if j+1<self.Ly:
                    where = (i,j),(i,j+1)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (i,0),(i,j)
                        ls.append(where)
                if i+1<self.Lx:
                    where = (i,j),(i+1,j)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (0,j),(i,j)
                        ls.append(where)
        return ls
    def diag_pairs(self):
        ls = [] 
        for i in range(self.Lx):
            for j in range(self.Ly):
                if i+1<self.Lx and j+1<self.Ly:
                    where = (i,j),(i+1,j+1)
                    ls.append(where)
                    where = (i,j+1),(i+1,j)
                    ls.append(where)
                else:
                    if self.pbc:
                        ix1 = self.flatten(i,j),self.flatten((i+1)%self.Lx,(j+1)%self.Ly)
                        where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
                        ls.append(where)
                        
                        ix1 = self.flatten(i,(j+1)%self.Ly),self.flatten((i+1)%self.Lx,j)
                        where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
                        ls.append(where)
        return ls
def get_gate1():
    return np.array([[1,0],
                   [0,-1]]) * .5
def get_gate2(j,to_bk=False):
    sx = np.array([[0,1],
                   [1,0]]) * .5
    sy = np.array([[0,-1],
                   [1,0]]) * 1j * .5
    sz = np.array([[1,0],
                   [0,-1]]) * .5
    try:
        jx,jy,jz = j
    except TypeError:
        j = j,j,j
    data = 0.
    for coeff,op in zip(j,[sx,sy,sz]):
        data += coeff * np.tensordot(op,op,axes=0).real
    if to_bk:
        data = data.transpose(0,2,1,3)
    return data
class Heisenberg(Hamiltonian):
    def __init__(self,J,h,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        try:
            self.Jx,self.Jy,self.Jz = J
        except TypeError:
            self.Jx,self.Jy,self.Jz = J,J,J
        self.h = h

        data = get_gate2((self.Jx,self.Jy,0.),to_bk=False)
        self.key = 'Jxy'
        self.data_map[self.key] = data

        self.pairs = self.nn_pairs() 
        self.plq_sz = (1,2),(2,1)
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        return 1.
    def pair_valid(self,i1,i2):
        return True
    def compute_local_energy_eigen(self,config):
        e = 0.
        for i in range(self.Lx):
            for j in range(self.Ly):
                ix1 = self.flatten(i,j)
                e += .5 * self.h * (-1) ** config[ix1]
                if j+1<self.Ly:
                    ix2 = self.flatten(i,j+1) 
                    e += .25 * self.Jz * (-1)**(config[ix1]+config[ix2])
                if i+1<self.Lx:
                    ix2 = self.flatten(i+1,j) 
                    e += .25 * self.Jz * (-1)**(config[ix1]+config[ix2])
        return e
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.25*(1-(-1)**(i1+i2)))]
class J1J2(Hamiltonian):
    def __init__(self,J1,J2,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.J1 = J1
        self.J2 = J2

        data = get_gate2((1.,1.,0.),to_bk=False)
        self.key = 'Jxy'
        self.data_map[self.key] = data

        # NN
        self.pairs = self.nn_pairs() + self.diag_pairs()
        self.plq_sz = (2,2),
    def pair_key(self,site1,site2):
        i0 = min(site1[0],site2[0],self.Lx-2)
        j0 = min(site1[1],site2[1],self.Ly-2)
        return (i0,j0),(2,2) 
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        dx = abs(site2[0]-site1[0])
        dy = abs(site2[1]-site1[1])
        if dx==1 and dy==1:
            return self.J2
        return self.J1
    def pair_valid(self,i1,i2):
        return True
    def compute_local_energy_eigen(self,config):
        e = 0.
        # NN
        for i in range(self.Lx):
            for j in range(self.Ly):
                ix1 = self.flatten(i,j)
                if j+1<self.Ly:
                    ix2 = self.flatten(i,j+1) 
                    e += .25 * self.J1 * (-1)**(config[ix1]+config[ix2])
                if i+1<self.Lx:
                    ix2 = self.flatten(i+1,j) 
                    e += .25 * self.J1 * (-1)**(config[ix1]+config[ix2])
        # next NN
        for i in range(self.Lx-1):
            for j in range(self.Ly-1):
                ix1,ix2 = self.flatten(i,j), self.flatten(i+1,j+1)
                e += .25 * self.J2 * (-1)**(config[ix1]+config[ix2])
                ix1,ix2 = self.flatten(i,j+1), self.flatten(i+1,j)
                e += .25 * self.J2 * (-1)**(config[ix1]+config[ix2])
        return e
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.25*(1-(-1)**(i1+i2)))]
####################################################################################
# sampler 
####################################################################################
class ExchangeSampler(ContractionEngine):
    def __init__(self,Lx,Ly,seed=None,burn_in=0,thresh=1e-14):
        super().init_contraction(Lx,Ly)
        self.nsite = self.Lx * self.Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.backend = 'numpy'
        self.thresh = thresh
    def initialize(self,config):
        # randomly choses the initial sweep direction
        self.sweep_row_dir = self.rng.choice([-1,1]) 
        # setup to compute all opposite envs for initial sweep
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 
        self.px = self.amplitude_factory.prob(config)
        self.config = config
        # force to initialize with a better config
        #print(self.px)
        #exit()
        if self.px < self.thresh:
            raise ValueError 
    def preprocess(self,config):
        return self._burn_in(config)
    def _burn_in(self,config,batchsize=None):
        if RANK==0:
            return None,None
        batchsize = self.burn_in if batchsize is None else batchsize
        self.initialize(config)
        if batchsize==0:
            return None,None
        t0 = time.time()
        for n in range(batchsize):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
        #print(f'RANK={RANK},burn in time={time.time()-t0}')
        return self.config,self.omega
    def new_pair(self,i1,i2):
        return i2,i1
    def get_pairs(self,i,j):
        bonds_map = {'l':((i,j),((i+1)%self.Lx,j)),
                     'd':((i,j),(i,(j+1)%self.Ly)),
                     'r':((i,(j+1)%self.Ly),((i+1)%self.Lx,(j+1)%self.Ly)),
                     'u':(((i+1)%self.Lx,j),((i+1)%self.Lx,(j+1)%self.Ly)),
                     'x':((i,j),((i+1)%self.Lx,(j+1)%self.Ly)),
                     'y':((i,(j+1)%self.Ly),((i+1)%self.Lx,j))}
        for key in bonds_map:
            site1,site2 = bonds_map[key]
            ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
            bonds_map[key] = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
        bonds = []
        order = 'ldru' 
        for key in order:
            bonds.append(bonds_map[key])
        return bonds
    def update_plq_test(self,ix1,ix2,i1_new,i2_new,py):
        config = self.config.copy()
        config[ix1] = i1_new
        config[ix2] = i2_new
        peps = self.amplitude_factory.psi.copy()
        for i in range(self.Lx):
            for j in range(self.Ly):
                peps.add_tensor(self.get_bra_tsr(peps,config[self.flatten(i,j)],i,j))
        try:
            py_ = peps.contract()**2
        except (ValueError,IndexError):
            py_ = 0.
        print(i,j,site1,site2,ix1,ix2,i1_new,i2_new,self.config,py,py_)
        if np.fabs(py-py_)>PRECISION:
            raise ValueError
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def update_plq(self,i,j,cols,tn,saved_rows):
        if cols[0] is None:
            return tn,saved_rows
        tn_plq = cols[0].copy()
        for col in cols[1:]:
            if col is None:
                return tn,saved_rows
            tn_plq.add_tensor_network(col.copy(),virtual=True)
        tn_plq.view_like_(tn)
        pairs = self.get_pairs(i,j) 
        for site1,site2 in pairs:
            ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
            i1,i2 = self.config[ix1],self.config[ix2]
            if not self.pair_valid(i1,i2): # continue
                #print(i,j,site1,site2,ix1,ix2,'pass')
                continue
            i1_new,i2_new = self.new_pair(i1,i2)
            tn_pair = self.replace_sites(tn_plq.copy(),(site1,site2),(i1_new,i2_new)) 
            try:
                py = tn_pair.contract()**2
            except (ValueError,IndexError):
                continue
            #self.update_plq_test(ix1,ix2,i1_new,i2_new,py)
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept, update px & config & env_m
                #print('acc')
                self.px = py
                self.config[ix1] = i1_new
                self.config[ix2] = i2_new
                tn_plq = self.replace_sites(tn_plq,(site1,site2),(i1_new,i2_new))
                tn = self.replace_sites(tn,(site1,site2),(i1_new,i2_new))
                saved_rows = self.replace_sites(saved_rows,(site1,site2),(i1_new,i2_new))
        return tn,saved_rows
    def sweep_col_forward(self,i,rows):
        self.config = list(self.config)
        tn = rows[0].copy()
        for row in rows[1:]:
            tn.add_tensor_network(row.copy(),virtual=True)
        saved_rows = tn.copy()
        try:
            tn.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        renvs = self.get_all_renvs(tn.copy(),jmin=2)
        first_col = tn.col_tag(0)
        for j in range(self.Ly-1): # 0,...,Ly-2
            tags = first_col,tn.col_tag(j),tn.col_tag(j+1)
            cols = [tn.select(tags,which='any').copy()]
            if j<self.Ly-2:
                cols.append(renvs[j+2])
            tn,saved_rows = self.update_plq(i,j,cols,tn,saved_rows) 
            # update new lenv
            if j<self.Ly-2:
                tn ^= first_col,tn.col_tag(j) 
        self.config = tuple(self.config)
        return saved_rows
    def sweep_col_backward(self,i,rows):
        self.config = list(self.config)
        tn = rows[0].copy()
        for row in rows[1:]:
            tn.add_tensor_network(row.copy(),virtual=True)
        saved_rows = tn.copy()
        try:
            tn.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.get_all_lenvs(tn.copy(),jmax=self.Ly-3)
        last_col = tn.col_tag(self.Ly-1)
        for j in range(self.Ly-1,0,-1): # Ly-1,...,1
            cols = []
            if j>1: 
                cols.append(lenvs[j-2])
            tags = tn.col_tag(j-1),tn.col_tag(j),last_col
            cols.append(tn.select(tags,which='any').copy())
            tn,saved_rows = self.update_plq(i,j-1,cols,tn,saved_rows) 
            # update new renv
            if j>1:
                tn ^= tn.col_tag(j),last_col
        self.config = tuple(self.config)
        return saved_rows
    def sweep_row_forward(self):
        if self.amplitude_factory.small_mem:
            # remove old bot envs
            self.amplitude_factory.cache_bot = dict()
        peps = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        # can assume to have all opposite envs
        #get_all_top_envs(fpeps,self.config,cache_top,imin=2,**compress_opts)
        sweep_col = self.sweep_col_forward if self.sweep_col_dir == 1 else\
                    self.sweep_col_backward

        env_bot = None 
        row1 = self.get_mid_env(0,peps,self.config)
        for i in range(self.Lx-1):
            rows = []
            if i>0:
                rows.append(env_bot)
            row2 = self.get_mid_env(i+1,peps,self.config)
            rows += [row1,row2]
            if i<self.Lx-2:
                rows.append(cache_top[self.config[(i+2)*self.Ly:]]) 
            saved_rows = sweep_col(i,rows)
            row1_new = saved_rows.select(peps.row_tag(i),virtual=True)
            row2_new = saved_rows.select(peps.row_tag(i+1),virtual=True)
            # update new env_h
            env_bot = self.get_bot_env(i,row1_new,env_bot,tuple(self.config),cache_bot)
            row1 = row2_new
    def sweep_row_backward(self):
        if self.amplitude_factory.small_mem:
            # remove old top envs
            self.amplitude_factory.cache_top = dict()
        peps = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        # can assume to have all opposite envs
        #get_all_bot_envs(fpeps,self.config,cache_bot,imax=self.Lx-3,**compress_opts)
        sweep_col = self.sweep_col_forward if self.sweep_col_dir == 1 else\
                    self.sweep_col_backward

        env_top = None 
        row1 = self.get_mid_env(self.Lx-1,peps,self.config)
        for i in range(self.Lx-1,0,-1):
            rows = []
            if i>1:
                rows.append(cache_bot[self.config[:(i-1)*self.Ly]])
            row2 = self.get_mid_env(i-1,peps,self.config)
            rows += [row2,row1]
            if i<self.Lx-1:
                rows.append(env_top) 
            saved_rows = sweep_col(i-1,rows)
            row1_new = saved_rows.select(peps.row_tag(i),virtual=True)
            row2_new = saved_rows.select(peps.row_tag(i-1),virtual=True)
            # update new env_h
            env_top = self.get_top_env(i,row1_new,env_top,tuple(self.config),cache_top)
            row1 = row2_new
    def sample(self):
        #self.sweep_col_dir = -1 # randomly choses the col sweep direction
        self.sweep_col_dir = self.rng.choice([-1,1]) # randomly choses the col sweep direction
        if self.deterministic:
            self._sample_deterministic()
        else:
            self._sample()
        # setup to compute all opposite env for gradient
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 
        self.sweep_row_dir *= -1
        return self.config,self.px
    def _sample(self):
        if self.sweep_row_dir == 1:
            self.sweep_row_forward()
        else:
            self.sweep_row_backward()
    def _sample_deterministic(self):
        imax = self.Lx-1 if self.pbc else self.Lx-2
        jmax = self.Ly-1 if self.pbc else self.Ly-2
        sweep_row = range(0,imax+1) if self.sweep_row_dir==1 else range(imax,-1,-1)
        sweep_col = range(0,jmax+1) if self.sweep_col_dir==1 else range(jmax,-1,-1)

        peps = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        for i,j in itertools.product(sweep_row,sweep_col):
            self.update_pair_deterministic(i,j,peps,cache_bot,cache_top)

        cache_bot_new = dict()
        for i in range(self.rix1+1):
            key = self.config[:(i+1)*self.Ly]
            cache_bot_new[key] = cache_bot[key]
        cache_top_new = dict()
        for i in range(self.rix2,self.Lx):
            key = self.config[i*self.Ly:]
            cache_top_new[key] = cache_top[key]
        self.amplitude_factory.cache_bot = cache_bot_new
        self.amplitude_factory.cache_top = cache_top_new
    def update_pair_deterministic(self,i,j,peps,cache_bot,cache_top):
        pairs = self.get_pairs(i,j)
        for site1,site2 in pairs:
            ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
            i1,i2 = self.config[ix1],self.config[ix2]
            if not self.pair_valid(i1,i2): # term vanishes 
                continue 
            imin = min(self.rix1+1,site1[0]) 
            imax = max(self.rix2-1,site2[0]) 
            top = None if imax==peps.Lx-1 else cache_top[self.config[(imax+1)*peps.Ly:]]
            bot = None if imin==0 else cache_bot[self.config[:imin*peps.Ly]]
            i1_new,i2_new = self.new_pair(i1,i2)
            config_new = list(self.config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)

            bot_term = None if bot is None else bot.copy()
            for i in range(imin,self.rix1+1):
                row = self.get_mid_env(i,peps,config_new,append='')
                bot_term = self.get_bot_env(i,row,bot_term,config_new,cache_bot)
            if imin > 0 and bot_term is None:
                continue

            top_term = None if top is None else top.copy()
            for i in range(imax,self.rix2-1,-1):
                row = self.get_mid_env(i,peps,config_new,append='')
                top_term = self.get_top_env(i,row,top_term,config_new,cache_top)
            if imax < peps.Lx-1 and top_term is None:
                continue

            tn = bot_term.copy()
            tn.add_tensor_network(top_term.copy(),virtual=True)
            try:
                py = tn.contract() ** 2 
            except (ValueError,IndexError):
                py = 0.
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept, update px & config & env_m
                #print('acc')
                self.px = py
                self.config = tuple(config_new) 
class DenseSampler:
    def __init__(self,Lx,Ly,nspin,exact=False,seed=None,thresh=1e-14):
        self.Lx = Lx
        self.Ly = Ly
        self.nsite = self.Lx * self.Ly
        self.nspin = nspin

        self.all_configs = self.get_all_configs()
        self.ntotal = len(self.all_configs)
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
    def initialize(self,config=None):
        pass
    def preprocess(self,config=None):
        return self.compute_dense_prob()
    def compute_dense_prob(self):
        t0 = time.time()
        ptotal = np.zeros(self.ntotal)
        start,stop = self.start,self.stop
        configs = self.all_configs[start:stop]

        plocal = [] 
        self.amplitude_factory.update_scheme(1)
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
        #print(RANK,start,stop,len(self.nonzeros),ntotal)
        #exit()
        self.amplitude_factory.update_scheme(0)
        return None,None
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

