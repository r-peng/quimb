import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
#from pympler import muppy,summary

# set tensor symmetry
import sys
import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)
from .torch_utils import SVD,QR,set_max_bond
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)
this = sys.modules[__name__]
def set_options(pbc=False,deterministic=False,**compress_opts):
    this.pbc = pbc
    this.deterministic = True if pbc else deterministic
    this.compress_opts = compress_opts

    set_max_bond(compress_opts.get('max_bond',None))

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
def get_product_state(Lx,Ly,config=None,bdim=1,eps=None):
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

            if config is None:
                data = np.ones(shape) 
            else:
                data = np.zeros(shape) 
                ix = flatten(i,j,Ly)
                ix = config[ix]
                data[(0,)*(len(shape)-1)+(ix,)] = 1.
            if eps is not None:
                data += eps * np.random.rand(*shape)
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
from .tensor_core import Tensor,TensorNetwork,rand_uuid,group_inds
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
        row = peps.select(peps.row_tag(i),virtual=False)
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
    def compress_row_pbc(self,tn,i):
        for j in range(self.Ly): # compress between j,j+1
            tn.compress_between(tn.site_tag(i,j),tn.site_tag(i,(j+1)%self.Ly),
                                **self.compress_opts)
        return tn
    def tensor_compress_bond(self,T1,T2,absorb='right'):
        # TODO:check for absorb='left'
        left_env_ix, shared_ix, right_env_ix = group_inds(T1, T2)
        if not shared_ix:
            raise ValueError("The tensors specified don't share an bond.")
        elif len(shared_ix) > 1:
            # fuse multibonds
            T1.fuse_({shared_ix[0]: shared_ix})
            T2.fuse_({shared_ix[0]: shared_ix})
            shared_ix = (shared_ix[0],)
        T1_inds,T2_inds = T1.inds,T2.inds

        # a) -> b)
        tmp_ix = rand_uuid()
        T1.reindex_({shared_ix[0]:tmp_ix})
        T2.reindex_({shared_ix[0]:tmp_ix})
        if absorb=='right': # assume T2 is isometric
            T1_L, T1_R = T1.split(left_inds=left_env_ix, right_inds=(tmp_ix,),
                                  get='tensors', method='qr')
            M,T2_R = T1_R,T2
        elif absorb=='left': # assume T1 is isometric
            T2_L, T2_R = T2.split(left_inds=(tmp_ix,), right_inds=right_env_ix,
                                  get='tensors', method='lq')
            T1_L,M = T1,T2_L
        else:
            raise NotImplementedError(f'absorb={absorb}')
        # c) -> d)
        M_L, *s, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                               absorb=absorb, **self.compress_opts)

        # make sure old bond being used
        ns_ix, = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})

        # d) -> e)
        T1C = T1_L.contract(M_L, output_inds=T1_inds)
        T2C = M_R.contract(T2_R, output_inds=T2_inds)

        # update with the new compressed data
        T1.modify(data=T1C.data)
        T2.modify(data=T2C.data)

        if absorb == 'right':
            T1.modify(left_inds=left_env_ix)
        else:
            T2.modify(left_inds=right_env_ix)
    def compress_row_obc(self,tn,i):
        tn.canonize_row(i,sweep='left')
        for j in range(self.Ly-1):
            self.tensor_compress_bond(tn[i,j],tn[i,j+1],absorb='right')        
        return tn
    def contract_boundary_single(self,tn,i,iprev):
        for j in range(self.Ly):
            tag1,tag2 = tn.site_tag(iprev,j),tn.site_tag(i,j)
            tn.contract_((tag1,tag2),which='any')
        if self.pbc:
            return self.compress_row_pbc(tn,i)
        else:
            return self.compress_row_obc(tn,i)
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
        tn.add_tensor_network(row,virtual=False)
        try:
            tn = self.contract_boundary_single(tn,i,i-1)
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
        tn.add_tensor_network(env_prev,virtual=False)
        try:
            tn = self.contract_boundary_single(tn,i,i+1)
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
                lenvs[j] = tn.select(first_col,virtual=False)
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
                renvs[j] = tn.select(last_col,virtual=False)
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
    def update_plq_from_3row(self,plq,tn,i,x_bsz,y_bsz,peps):
        jmax = self.Ly - y_bsz
        try:
            tn.reorder('col',inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.get_all_lenvs(tn.copy(),jmax=jmax-1)
        renvs = self.get_all_renvs(tn.copy(),jmin=y_bsz)
        for j in range(jmax+1): 
            tags = [tn.col_tag(j+ix) for ix in range(y_bsz)]
            cols = tn.select(tags,which='any',virtual=False)
            try:
                if j>0:
                    other = cols
                    cols = lenvs[j-1]
                    cols.add_tensor_network(other,virtual=False)
                if j<jmax:
                    cols.add_tensor_network(renvs[j+y_bsz],virtual=False)
                plq[(i,j),(x_bsz,y_bsz)] = cols.view_like_(peps)
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def build_3row_tn(self,config,i,x_bsz,peps,cache_bot,cache_top):
        try:
            tn = self.get_mid_env(i,peps,config)
            for ix in range(1,x_bsz):
                tn.add_tensor_network(self.get_mid_env(i+ix,peps,config),virtual=False)
            if i>0:
                other = tn 
                tn = cache_bot[config[:i*self.Ly]].copy()
                tn.add_tensor_network(other,virtual=False)
            if i+x_bsz<self.Lx:
                tn.add_tensor_network(cache_top[config[(i+x_bsz)*self.Ly:]],virtual=False)
        except AttributeError:
            tn = None
        return tn 
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,peps,cache_bot,cache_top,imin=0,imax=None):
        #if self.compute_bot and self.compute_top:
        #    raise ValueError
        imax = self.Lx-x_bsz if imax is None else imax
        plq = dict()
        for i in range(imin,imax+1):
            tn = self.build_3row_tn(config,i,x_bsz,peps,cache_bot,cache_top)
            if tn is not None:
                plq = self.update_plq_from_3row(plq,tn,i,x_bsz,y_bsz,peps)
        return plq
    def get_grad_dict_from_plq(self,plq,cx,backend='numpy'):
        # gradient
        vx = dict()
        for ((i0,j0),(x_bsz,y_bsz)),tn in plq.items():
            for i in range(i0,i0+x_bsz):
                for j in range(j0,j0+y_bsz):
                    if (i,j) in vx:
                        continue
                    vx[i,j] = self._2numpy(self.site_grad(tn.copy(),i,j)/cx[i,j],backend=backend)
        return vx
class AmplitudeFactory(ContractionEngine):
    def __init__(self,psi):
        super().init_contraction(psi.Lx,psi.Ly)
        psi.add_tag('KET')
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict()

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
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

        self.cache_bot = dict()
        self.cache_top = dict()
    def unsigned_amplitude(self,config):
        # should only be used to:
        # 1. compute dense probs
        # 2. initialize MH sampler
        if self.deterministic:
            compute_bot = True
            compute_top = True
        else:
            compute_bot = True
            compute_top = False 

        env_bot,env_top = self.get_all_benvs(self.psi,config,self.cache_bot,self.cache_top,x_bsz=1,
                                             compute_bot=compute_bot,compute_top=compute_top)
        if env_bot is None and env_top is None:
            return 0.
        if self.deterministic:
            tn = env_bot.copy()
            tn.add_tensor_network(env_top,virtual=False)
        elif compute_bot: 
            tn = env_bot.copy()
            tn.add_tensor_network(self.get_mid_env(self.Lx-1,self.psi,config),virtual=False) 
        elif compute_top:
            tn = self.get_mid_env(0,self.psi,config)
            tn.add_tensor_network(env_top,virtual=False)
        try:
            return tn.contract()
        except (ValueError,IndexError):
            return 0.
    def amplitude(self,config):
        raise NotImplementedError
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def get_grad_from_plq(self,plq,cx,backend='numpy'):
        vx = self.get_grad_dict_from_plq(plq,cx,backend=backend)
        return self.dict2vec(vx) 
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
####################################################################################
# ham class 
####################################################################################
class Hamiltonian(ContractionEngine):
    def __init__(self,Lx,Ly,nbatch=1,phys_dim=2):
        super().init_contraction(Lx,Ly,phys_dim=phys_dim)
        self.nbatch = nbatch
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self._2backend(self.data_map[self.key],False)
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return Tensor(data=data,inds=inds,tags=tags) 
    def pair_energy_from_plq(self,tn,config,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        kixs = [tn.site_ind(*site) for site in [site1,site2]]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip([site1,site2],kixs,bixs):
            tn[tn.site_tag(*site),'BRA'].reindex_({kix:bix})
        tn.add_tensor(self.pair_tensor(bixs,kixs),virtual=True)
        try:
            t0 = time.time()
            ex = tn.contract()
            #print(time.time()-t0)
            return self.pair_coeff(site1,site2) * ex 
        except (ValueError,IndexError):
            return None 
    def batch_hessian(self,batch_idx,config,amplitude_factory): # only used for Hessian
        peps = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            peps[i,j].modify(data=self._2backend(peps[i,j].data,True))

        #print(batch_idx)
        # compute benvs
        #t0 = time.time()
        cache_bot,cache_top = dict(),dict()
        bix,tix,plq_types,pairs = self.batched_pairs[batch_idx]
        self.get_all_bot_envs(peps,config,cache_bot,imax=bix)
        self.get_all_top_envs(peps,config,cache_top,imin=tix)
        #print(f'benv,time={time.time()-t0}')

        # form plqs
        #t0 = time.time()
        plq = dict()
        for imin,imax,x_bsz,y_bsz in plq_types:
            plq.update(self.get_plq_from_benvs(config,x_bsz,y_bsz,peps,cache_bot,cache_top,imin=imin,imax=imax))
        #print(f'plq,time={time.time()-t0}')

        # compute energy numerator 
        #t0 = time.time()
        ex = dict()
        cx = dict()
        for (site1,site2) in pairs:
            key = self.pair_key(site1,site2)

            tn = plq.get(key,None) 
            if tn is not None:
                eij = self.pair_energy_from_plq(tn.copy(),config,site1,site2) 
                if eij is not None:
                    ex[site1,site2] = eij

                if site1 in cx:
                    cij = cx[site1]
                elif site2 in cx:
                    cij = cx[site2]
                else:
                    cij = self._2numpy(tn.copy().contract())
                cx[site1] = cij 
                cx[site2] = cij 
        #print(f'e,time={time.time()-t0}')

        # back-prop energy numerator
        #t0 = time.time()
        if len(ex)>0:
            ex_num = sum(ex.values())
            ex_num.backward()
            Hvx = dict()
            for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
                Hvx[i,j] = self._2numpy(self.tsr_grad(peps[i,j].data))
            #print(f'H,time={time.time()-t0}')
            Hvx = amplitude_factory.dict2vec(Hvx)  
            ex = sum([self._2numpy(eij)/cx[site] for (site,_),eij in ex.items()])
        else:
            ex = 0.
            Hvx = 0.

        # compute plq grad
        vx = self.get_grad_dict_from_plq(plq,cx,backend=self.backend) 
        return ex,Hvx,cx,vx
    def compute_local_energy_hessian(self,config,amplitude_factory): 
        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))

        ex,Hvx = 0.,0.
        cx,vx = dict(),dict()
        for batch_idx in self.batched_pairs:
            ex_,Hvx_,cx_,vx_ = self.batch_hessian(batch_idx,config,amplitude_factory)  
            ex += ex_
            Hvx += Hvx_
            cx.update(cx_)
            vx.update(vx_)

        eu = self.compute_local_energy_eigen(config)
        ex += eu

        vx = amplitude_factory.dict2vec(vx)

        cx,err = self.contraction_error(cx)

        Hvx = Hvx/cx + eu*vx
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,err 
    def compute_local_energy_gradient(self,config,amplitude_factory,compute_v=True):
        self.backend = 'numpy'
        peps = amplitude_factory.psi
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top

        x_bsz_min = min([x_bsz for x_bsz,_ in self.plq_sz])
        self.get_all_benvs(peps,config,cache_bot,cache_top,x_bsz=x_bsz_min)

        plq = dict()
        for x_bsz,y_bsz in self.plq_sz:
            plq.update(self.get_plq_from_benvs(config,x_bsz,y_bsz,peps,cache_bot,cache_top))

        ex = dict()
        cx = dict()
        for (site1,site2) in self.pairs:
            key = self.pair_key(site1,site2)

            tn = plq.get(key,None) 
            if tn is not None:
                eij = self.pair_energy_from_plq(tn.copy(),config,site1,site2) 
                if eij is not None:
                    ex[site1,site2] = eij

                if site1 in cx:
                    cij = cx[site1]
                elif site2 in cx:
                    cij = cx[site2]
                else:
                    cij = tn.copy().contract()
                cx[site1] = cij 
                cx[site2] = cij 

        ex = sum([eij/cx[site1] for (site1,_),eij in ex.items()])
        eu = self.compute_local_energy_eigen(config)
        ex += eu

        if not compute_v:
            cx,err = self.contraction_error(cx)
            return cx,ex,None,None,err 
        vx = amplitude_factory.get_grad_from_plq(plq,cx,backend=self.backend)  
        cx,err = self.contraction_error(cx)
        return cx,ex,vx,None,err
    def pair_energy_deterministic(self,config,peps,site1,site2,cache_bot,cache_top,sign_fn):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        assert site1[0] <= site2[0]
        imin = min(self.rix1+1,site1[0]) 
        imax = max(self.rix2-1,site2[0]) 
        top = None if imax==peps.Lx-1 else cache_top[config[(imax+1)*peps.Ly:]]
        bot = None if imin==0 else cache_bot[config[:imin*peps.Ly]]
        ex = [] 
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
            if bot_term is None:
                continue

            top_term = None if top is None else top.copy()
            for i in range(imax,self.rix2-1,-1):
                row = self.get_mid_env(i,peps,config_new,append='')
                top_term = self.get_top_env(i,row,top_term,config_new,cache_top)
            if top_term is None:
                continue

            tn = bot_term.copy()
            tn.add_tensor_network(top_term,virtual=False)
            try:
                ex.append(coeff * sign_new * tn.contract())
            except (ValueError,IndexError):
                continue
        if len(ex)==0:
            return None
        return sum(ex) * coeff_comm
    def batch_hessian_deterministic(self,config,amplitude_factory,imin_,imax_):
        cache_top = dict()
        cache_bot = dict()
        peps = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            peps[i,j].modify(data=self._2backend(peps[i,j].data,True))
        
        #t0 = time.time()
        imin = min(self.rix1+1,imin_) 
        imax = max(self.rix2-1,imax_) 
        self.get_all_bot_envs(peps,config,cache_bot,imax=imin+1)
        self.get_all_top_envs(peps,config,cache_top,imin=imax-1)
        #print(f'benv,time={time.time()-t0}')
        sign_fn = amplitude_factory.config_sign

        #t0 = time.time()
        ex = []
        for site1,site2 in self.batched_pairs[imin_,imax_]:
            ex_ = self.pair_energy_deterministic(config,peps,site1,site2,cache_bot,cache_top,sign_fn)
            if ex_ is not None:
                ex.append(ex_)
        if len(ex)==0:
            return 0.,0.
        ex = sum(ex) 
        #print(f'ex,time={time.time()-t0}')

        #t0 = time.time()
        ex.backward()
        Hvx = dict()
        for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
            Hvx[i,j] = self.tsr_grad(peps[i,j].data)  
        Hvx = {site:self._2numpy(Hvij) for site,Hvij in Hvx.items()}
        Hvx = amplitude_factory.dict2vec(Hvx)  
        ex = self._2numpy(ex) 
        #print(f'Hvx,time={time.time()-t0}')
        return ex,Hvx
    def pair_hessian_deterministic(self,config,amplitude_factory,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return 0.,0. 

        cache_top = dict()
        cache_bot = dict()
        peps = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            peps[i,j].modify(data=self._2backend(peps[i,j].data,True))

        coeff_comm = self.intermediate_sign(config,ix1,ix2) * self.pair_coeff(site1,site2)
        ex = 0.
        Hvx = 0.
        for i1_new,i2_new,coeff in self.pair_terms(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            sign_new = amplitude_factory.config_sign(config_new)

            #t0 = time.time()
            env_bot,env_top = self.get_all_benvs(peps,config_new,cache_bot,cache_top)
            tn = env_bot.copy()
            tn.add_tensor_network(env_top,virtual=False)
            cx = tn.contract() 
            #print(f'ex,time={time.time()-t0}')

            #t0 = time.time()
            cx.backward()
            vx = dict()
            for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
                vx[i,j] = self.tsr_grad(peps[i,j].data)  
            vx = {site:self._2numpy(vij) for site,vij in vx.items()}
            vx = amplitude_factory.dict2vec(vx)  
            cx = self._2numpy(cx) 
            #print(f'Hvx,time={time.time()-t0}')

            ex += coeff * sign_new * cx 
            Hvx += coeff * sign_new * vx
        return ex * coeff_comm, Hvx * coeff_comm
    def amplitude_gradient_deterministic(self,config,amplitude_factory):
        cache_top = dict()
        cache_bot = dict()
        peps = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            peps[i,j].modify(data=self._2backend(peps[i,j].data,True))
        #t0 = time.time()
        env_bot,env_top = self.get_all_benvs(peps,config,cache_bot,cache_top)
        tn = env_bot.copy()
        tn.add_tensor_network(env_top,virtual=False)
        cx = tn.contract() 
        #print(f'cx,time={time.time()-t0}')

        #t0 = time.time()
        cx.backward()
        vx = dict()
        for i,j in itertools.product(range(peps.Lx),range(peps.Ly)):
            vx[i,j] = self.tsr_grad(peps[i,j].data)  
        vx = {site:self._2numpy(vij) for site,vij in vx.items()}
        vx = amplitude_factory.dict2vec(vx)  
        cx = self._2numpy(cx)
        #print(f'vx,time={time.time()-t0}')
        return cx,vx/cx
    def compute_local_energy_hessian_deterministic(self,config,amplitude_factory):
        #print(config)
        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))

        cx,vx = self.amplitude_gradient_deterministic(config,amplitude_factory)

        #t0 = time.time()
        ex = 0. 
        Hvx = 0.
        for key in self.batched_pairs:
            if key=='pbc':
                continue
            imin,imax = key
            ex_,Hvx_ = self.batch_hessian_deterministic(config,amplitude_factory,imin,imax) 
            ex += ex_
            Hvx += Hvx_
        if self.pbc:
            for site1,site2 in self.batched_pairs['pbc']:
                ex_,Hvx_ = self.pair_hessian_deterministic(self,config,amplitude_factory,site1,site2)
                ex += ex_
                Hvx += Hvx_
        #print(f'e,time={time.time()-t0}')
         
        eu = self.compute_local_energy_eigen(config)
        ex = ex/cx + eu
        Hvx = Hvx/cx + eu*vx
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,0. 
    def compute_local_energy_gradient_deterministic(self,config,amplitude_factory,compute_v=True):
        # compute energy
        self.backend = 'numpy'
        peps = amplitude_factory.psi
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top

        env_bot,env_top = self.get_all_benvs(peps,config,cache_bot,cache_top)

        sign_fn = amplitude_factory.config_sign
        ex = 0. 
        for (site1,site2) in self.pairs:
            ex_ = self.pair_energy_deterministic(config,peps,site1,site2,cache_bot,cache_top,sign_fn)
            if ex_ is not None:
                ex += ex_
        eu = self.compute_local_energy_eigen(config)
        if not compute_v:
            tn = env_bot.copy()
            tn.add_tensor_network(env_top,virtual=False)
            cx = tn.contract() 

            ex = ex/cx + eu
            return cx,ex,None,None,0.

        # compute_amplitude_gradient 
        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))
        cx,vx = self.amplitude_gradient_deterministic(config,amplitude_factory)
        ar.set_backend(np.zeros(1))

        ex = ex/cx + eu
        return cx,ex,vx,None,0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        if self.deterministic:
            if compute_Hv:
                return self.compute_local_energy_hessian_deterministic(config,amplitude_factory)
            else:
                return self.compute_local_energy_gradient_deterministic(config,amplitude_factory,compute_v=compute_v)
        else:
            if compute_Hv:
                return self.compute_local_energy_hessian(config,amplitude_factory)
            else:
                return self.compute_local_energy_gradient(config,amplitude_factory,compute_v=compute_v)
    def contraction_error(self,cx):
        nsite = self.Lx * self.Ly
        sqmean = sum(cij**2 for cij in cx.values()) / nsite
        mean = sum(cij for cij in cx.values()) / nsite
        err = sqmean - mean**2
        return mean,np.fabs(err/mean)
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
                        ix1,ix2 = self.flatten(i,j),self.flatten((i+1)%self.Lx,(j+1)%self.Ly)
                        where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
                        ls.append(where)
                        
                        ix1,ix2 = self.flatten(i,(j+1)%self.Ly),self.flatten((i+1)%self.Lx,j)
                        where = self.flat2site(min(ix1,ix2)),self.flat2site(max(ix1,ix2))
                        ls.append(where)
        return ls
    def batch_nnh(self):
        for i in range(self.Lx):
            ls = self.batched_pairs.get((i,i),[])
            for j in range(self.Ly):
                if j+1<self.Ly:
                    where = (i,j),(i,j+1)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (i,0),(i,j)
                        ls.append(where)
            self.batched_pairs[i,i] = ls
    def batch_nnv(self):
        for i in range(self.Lx-1):
            ls = self.batched_pairs.get((i,i+1),[])
            for j in range(self.Ly):
                where = (i,j),(i+1,j)
                ls.append(where)
            self.batched_pairs[i,i+1] = ls
        if not self.pbc:
            return
        ls = self.batched_pairs.get('pbc',[]) 
        for j in range(self.Ly):
            where = (0,j),(i,j)
            ls.append(where)
        self.batched_pairs['pbc'] = ls
    def batch_diag(self):
        for i in range(self.Lx-1):
            ls = self.batched_pairs.get((i,i+1),[])
            for j in range(self.Ly):
                if i+1<self.Lx and j+1<self.Ly:
                    where = (i,j),(i+1,j+1)
                    ls.append(where)
                    where = (i,j+1),(i+1,j)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (i,j),(i+1,(j+1)%self.Ly)
                        ls.append(where)
                        
                        where = (i,(j+1)%self.Ly),(i+1,j)
                        ls.append(where)
            self.batched_pairs[i,i+1] = ls
        if not self.pbc:
            return
        ls = self.batched_pairs.get('pbc',[])
        for j in range(self.Ly):
            where = (0,(j+1)%self.Ly),(self.Lx-1,j)
            ls.append(where)
            
            where = (0,j),(self.Lx-1,(j+1)%self.Ly)
            ls.append(where)
        self.batched_pairs['pbc'] = ls 
    def batch_nn_plq(self):
        self.batched_pairs = dict() 
        batchsize = max(self.Lx // self.nbatch,2)
        for i in range(self.Lx):
            batch_idx = i // batchsize
            if batch_idx not in self.batched_pairs:
                self.batched_pairs[batch_idx] = [],[] 
            rows,pairs = self.batched_pairs[batch_idx]
            rows.append(i)
            if i+1 < self.Lx:
                rows.append(i+1)
            for j in range(self.Ly):
                if j+1<self.Ly:
                    where = (i,j),(i,j+1)
                    pairs.append(where)
                if i+1<self.Lx:
                    where = (i,j),(i+1,j)
                    pairs.append(where)
        self.pairs = []
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            bix,tix = max(0,imax-1),min(imin+1,self.Lx-1) # bot_ix,top_ix,pairs 
            plq_types = (imin,imax,1,2), (imin,imax-1,2,1),# i0_min,i0_max,x_bsz,y_bsz
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs 
            self.pairs += pairs
        self.plq_sz = (1,2),(2,1)
        #for batch_idx in self.batched_pairs:
        #    bix,tix,plq_types,pairs = self.batched_pairs[batch_idx]
        #    print(batch_idx,bix,tix,plq_types)
        #    print(pairs)
        if RANK==0:
            print('nbatch=',len(self.batched_pairs))
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
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_nn_plq()
    def batch_deterministic(self):
        self.batched_pairs = dict()
        self.batch_nnh() 
        self.batch_nnv() 
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        return 1.
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def compute_local_energy_eigen(self,config):
        eh = 0.
        ez = 0.
        for i in range(self.Lx):
            for j in range(self.Ly):
                s1 = (-1) ** config[self.flatten(i,j)]
                eh += s1 
                if j+1<self.Ly:
                    ez += s1 * (-1)**config[self.flatten(i,j+1)]
                else:
                    if self.pbc:
                        ez += s1 * (-1)**config[self.flatten(i,0)]
                if i+1<self.Lx:
                    ez += s1 * (-1)**config[self.flatten(i+1,j)]
                else:
                    if self.pbc:
                        ez += s1 * (-1)**config[self.flatten(0,j)]
        return eh * .5 * self.h + ez * .25 * self.Jz
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.25*(self.Jx+self.Jy))]
class J1J2(Hamiltonian):
    def __init__(self,J1,J2,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.J1 = J1
        self.J2 = J2

        data = get_gate2((1.,1.,0.),to_bk=False)
        self.key = 'Jxy'
        self.data_map[self.key] = data

        self.pairs = self.nn_pairs() + self.diag_pairs() # list of all pairs, for SR
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch()
    def batch(self):
        self.batched_pairs = dict()
        batchsize = max(self.Lx // self.nbatch, 2)
        for i in range(self.Lx):
            batch_idx = i // batchsize
            if batch_idx not in self.batched_pairs:
                self.batched_pairs[batch_idx] = [],[] 
            rows,pairs = self.batched_pairs[batch_idx]
            rows.append(i)
            if i+1 < self.Lx:
                rows.append(i+1)
            for j in range(self.Ly):
                if j+1<self.Ly: # NN
                    where = (i,j),(i,j+1)
                    pairs.append(where)
                if i+1<self.Lx: # NN
                    where = (i,j),(i+1,j)
                    pairs.append(where)
                if i+1<self.Lx and j+1<self.Ly: # diag
                    where = (i,j),(i+1,j+1)
                    pairs.append(where)
                    where = (i,j+1),(i+1,j)
                    pairs.append(where)
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            plq_types = (imin,imax-1,2,2),
            bix,tix = max(0,imax-2),min(imin+2,self.Lx-1)
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs # bot_ix,top_ix,pairs 
        self.plq_sz = (2,2),
        #for batch_idx in self.batched_pairs:
        #    bix,tix,plq_types,pairs = self.batched_pairs[batch_idx]
        #    print(batch_idx,bix,tix,plq_types)
        #    print(pairs)
        if RANK==0:
            print('nbatch=',len(self.batched_pairs))
    def batch_deterministic(self):
        self.batched_pairs = dict()
        self.batch_nnh() 
        self.batch_nnv() 
        self.batch_diag() 
    def pair_key(self,site1,site2):
        i0 = min(site1[0],site2[0],self.Lx-2)
        j0 = min(site1[1],site2[1],self.Ly-2)
        return (i0,j0),(2,2) 
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        dx = abs(site2[0]-site1[0])
        dy = abs(site2[1]-site1[1])
        if dx == 0:
            return self.J1
        if dy == 0:
            return self.J1
        return self.J2
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def compute_local_energy_eigen(self,config):
        # NN
        e1 = 0.
        for i in range(self.Lx):
            for j in range(self.Ly):
                s1 = (-1) ** config[self.flatten(i,j)]
                if j+1<self.Ly:
                    e1 += s1 * (-1)**config[self.flatten(i,j+1)]
                else:
                    if self.pbc:
                        e1 += s1 * (-1)**config[self.flatten(i,0)]
                if i+1<self.Lx:
                    e1 += s1 * (-1)**config[self.flatten(i+1,j)]
                else:
                    if self.pbc:
                        e1 += s1 * (-1)**config[self.flatten(0,j)]
        # next NN
        e2 = 0. 
        for i in range(self.Lx):
            for j in range(self.Ly):
                if i+1<self.Lx and j+1<self.Ly: 
                    ix1,ix2 = self.flatten(i,j), self.flatten(i+1,j+1)
                    e2 += (-1)**(config[ix1]+config[ix2])
                    ix1,ix2 = self.flatten(i,j+1), self.flatten(i+1,j)
                    e2 += (-1)**(config[ix1]+config[ix2])
                else:
                    if self.pbc:
                        ix1,ix2 = self.flatten(i,j), self.flatten((i+1)%self.Lx,(j+1)%self.Ly)
                        e2 += (-1)**(config[ix1]+config[ix2])
                        ix1,ix2 = self.flatten(i,(j+1)%self.Ly), self.flatten((i+1)%self.Lx,j)
                        e2 += (-1)**(config[ix1]+config[ix2])
        return .25 * (e1 *self.J1 + e2 * self.J2) 
    def pair_terms(self,i1,i2):
        return [(1-i1,1-i2,.5)]

class HubbardBoson(Hamiltonian):
    def __init__(self,t,u,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,phys_dim=4)
        self.t = t
        self.u = u

class SpinDensity(Hamiltonian):
    def __init__(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly 
        self.data = np.zeros((Lx,Ly))
        self.n = 0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
        self.n += 1.
        for i in range(self.Lx):
            for j in range(self.Ly):
                self.data[i,j] += (-1) ** config[self.flatten(i,j)]
        return 0.,0.,None,None,0. 
    def _print(self,fname,data):
        print(fname)
        print(data)
class Mz(Hamiltonian):
    def __init__(self,Lx,Ly):
        self.Lx,self.Ly = Lx,Ly 
        self.nsites = Lx * Ly
        self.data = np.zeros(1)
        self.n = 0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=False,compute_Hv=False):
        self.n += 1.

        data = 0.
        for ix1 in range(self.nsites):
            s1 = (-1) ** config[ix1]
            site1 = self.flat2site(ix1)
            for ix2 in range(ix1+1,self.nsites):
                s2 = (-1) ** config[ix2]
                site2 = self.flat2site(ix2)
                  
                dx,dy = site1[0]-site2[0],site1[1]-site2[1]
                data += s1 * s2 * (-1)**(dx+dy)
        self.data += data / self.nsites**2
        return 0.,0.,None,None,0. 
    def _print(self,fname,data):
        print(f'fname={fname},data={data[0]}') 
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
    def preprocess(self,config):
        return self._burn_in(config)
    def _burn_in(self,config,batchsize=None):
        # check either: 
        # config is wrong
        # wfn update is wrong
        self.px = self.amplitude_factory.prob(config)
        self.config = config
        if self.px < self.thresh:
            raise ValueError 

        if RANK==0:
            return None,None
        batchsize = self.burn_in if batchsize is None else batchsize
        if batchsize==0:
            return None,None
        t0 = time.time()
        for n in range(batchsize):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
        #print(f'RANK={RANK},burn in time={time.time()-t0}')
        return self.config,self.omega
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def new_pair(self,i1,i2):
        return i2,i1
    def update_pair(self,i,j,x_bsz,y_bsz,cols,tn):
        if cols[0] is None:
            return tn
        tn_plq = cols[0].copy()
        for col in cols[1:]:
            if col is None:
                return tn
            tn_plq.add_tensor_network(col,virtual=False)
        tn_plq.view_like_(tn)
        if (x_bsz,y_bsz)==(1,2):
            site1,site2 = (i,j),(i,j+1)
        elif (x_bsz,y_bsz)==(2,1):
            site1,site2 = (i,j),(i+1,j)
        else:
            raise NotImplementedError
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.pair_valid(i1,i2): # continue
            #print(i,j,site1,site2,ix1,ix2,'pass')
            return tn
        i1_new,i2_new = self.new_pair(i1,i2)
        tn_plq = self.replace_sites(tn_plq,(site1,site2),(i1_new,i2_new)) 
        try:
            py = tn_plq.contract()**2
        except (ValueError,IndexError):
            return
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
            tn = self.replace_sites(tn,(site1,site2),(i1_new,i2_new))
        return tn
    def sweep_col_forward(self,i,tn,x_bsz,y_bsz):
        self.config = list(self.config)
        renvs = self.get_all_renvs(tn.copy(),jmin=y_bsz)

        first_col = tn.col_tag(0)
        jmax = self.Ly - y_bsz  
        for j in range(jmax+1): 
            tags = [first_col] + [tn.col_tag(j+ix) for ix in range(y_bsz)]
            cols = [tn.select(tags,which='any',virtual=False)]
            if j<jmax:
                cols.append(renvs[j+y_bsz])
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            tn ^= first_col,tn.col_tag(j) 
        self.config = tuple(self.config)
    def sweep_col_backward(self,i,tn,x_bsz,y_bsz):
        self.config = list(self.config)
        lenvs = self.get_all_lenvs(tn.copy(),jmax=self.Ly-1-y_bsz)

        last_col = tn.col_tag(self.Ly-1)
        jmax = self.Ly - y_bsz  
        for j in range(jmax,-1,-1): # Ly-1,...,1
            cols = []
            if j>0: 
                cols.append(lenvs[j-1])
            tags = [tn.col_tag(j+ix) for ix in range(y_bsz)] + [last_col]
            cols.append(tn.select(tags,which='any',virtual=False))
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            tn ^= tn.col_tag(j+y_bsz-1),last_col
        self.config = tuple(self.config)
    def sweep_row_forward(self,x_bsz,y_bsz):
        self.amplitude_factory.cache_bot = dict()

        peps = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        self.get_all_top_envs(peps,self.config,cache_top,imin=x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax+1):
            tn = self.build_3row_tn(self.config,i,x_bsz,peps,cache_bot,cache_top)
            sweep_col(i,tn,x_bsz,y_bsz)

            for ix in range(x_bsz):
                inew = i+ix
                row = self.get_mid_env(inew,peps,self.config)
                env_prev = None if inew==0 else cache_bot[self.config[:inew*self.Ly]] 
                self.get_bot_env(inew,row,env_prev,self.config,cache_bot)
    def sweep_row_backward(self,x_bsz,y_bsz):
        self.amplitude_factory.cache_top = dict()

        peps = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        self.get_all_bot_envs(peps,self.config,cache_bot,imax=self.Lx-1-x_bsz)

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax,-1,-1):
            tn = self.build_3row_tn(self.config,i,x_bsz,peps,cache_bot,cache_top)
            sweep_col(i,tn,x_bsz,y_bsz)

            for ix in range(x_bsz-1,-1,-1):
                inew = i+ix
                row = self.get_mid_env(inew,peps,self.config)
                env_prev = None if inew==self.Lx-1 else cache_top[self.config[(inew+1)*self.Ly:]] 
                self.get_top_env(inew,row,env_prev,self.config,cache_top)
    def sample(self):
        self.sweep_col_dir = self.rng.choice([-1,1]) 
        if self.deterministic:
            self._sample_deterministic()
        else:
            self._sample()
        return self.config,self.px
    def _sample(self):
        hdir = self.rng.choice([-1,1]) # all horizontal bonds
        if hdir == 1:
            self.sweep_row_forward(1,2) 
        else:
            self.sweep_row_backward(1,2)

        vdir = self.rng.choice([-1,1]) # all vertical bonds
        if vdir == 1:
            self.sweep_row_forward(2,1) 
        else:
            self.sweep_row_backward(2,1)
    def _sample_deterministic(self):
        peps = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        for x_bsz,y_bsz in [(1,2),(2,1)]:
            imax = self.Lx-1 if self.pbc else self.Lx-x_bsz
            jmax = self.Ly-1 if self.pbc else self.Ly-y_bsz
            rdir = self.rng.choice([-1,1]) 
            cdir = self.rng.choice([-1,1]) 
            sweep_row = range(0,imax+1) if rdir==1 else range(imax,-1,-1)
            sweep_col = range(0,jmax+1) if cdir==1 else range(jmax,-1,-1)
            for i,j in itertools.product(sweep_row,sweep_col):
                self.update_pair_deterministic(i,j,x_bsz,y_bsz,peps,cache_bot,cache_top)

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
    def update_pair_deterministic(self,i,j,x_bsz,y_bsz,peps,cache_bot,cache_top):
        if (x_bsz,y_bsz)==(1,2):
            site1,site2 = (i,j),(i,j+1)
        elif (x_bsz,y_bsz)==(2,1):
            site1,site2 = (i,j),(i+1,j)
        else:
            raise NotImplementedError

        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return 
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
            return 

        top_term = None if top is None else top.copy()
        for i in range(imax,self.rix2-1,-1):
            row = self.get_mid_env(i,peps,config_new,append='')
            top_term = self.get_top_env(i,row,top_term,config_new,cache_top)
        if imax < peps.Lx-1 and top_term is None:
            return

        tn = bot_term.copy()
        tn.add_tensor_network(top_term,virtual=False)
        try:
            py = tn.contract() ** 2 
        except (ValueError,IndexError):
            return
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
    def preprocess(self,config=None):
        return self.compute_dense_prob()
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
        #print(RANK,start,stop,len(self.nonzeros),ntotal)
        #exit()
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

