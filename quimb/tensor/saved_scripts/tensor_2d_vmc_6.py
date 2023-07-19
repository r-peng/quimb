import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)

from .torch_utils import SVD,QR
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

import sys
this = sys.modules[__name__]
def set_options(pbc=False,deterministic=False,**compress_opts):
    this.pbc = pbc
    this.deterministic = True if pbc else deterministic
    this.compress_opts = compress_opts
def flatten(i,j,Ly): # flattern site to row order
    return i*Ly+j
def flat2site(ix,Ly): # ix in row order
    return ix//Ly,ix%Ly
from .tensor_2d import PEPS
def get_product_state(Lx,Ly,config=None,bdim=1,eps=0.,pdim=2,normalize=True):
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [bdim] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (pdim,)

            if config is None:
                data = np.ones(shape) 
            else:
                data = np.zeros(shape) 
                ix = flatten(i,j,Ly)
                ix = config[ix]
                data[(0,)*(len(shape)-1)+(ix,)] = 1.
            data += eps * np.random.rand(*shape)
            if normalize:
                data /= np.linalg.norm(data)
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
def expand_peps(peps,Dnew,eps=0.):
    tn = peps.copy()
    for i,j in itertools.product(range(tn.Lx),range(tn.Ly)):
        T = tn[i,j]
        data = T.data
        shape = data.shape
        ndim = len(shape)

        shape_new = (Dnew,)*(ndim-1) + (data.shape[-1],)
        data_new = eps * np.random.rand(*shape_new)
        slices = tuple([slice(D) for D in shape])
        data_new[slices] = data

        T.modify(data=data_new)
        #print(data)
        #print(data_new)
        #exit()
    return tn

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
####################################################################################
# amplitude fxns 
####################################################################################
class ContractionEngine:
    def init_contraction(self,Lx,Ly,phys_dim=2):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly
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
        return flat2site(ix,self.Ly)
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
    def get_bra_tsr(self,ci,i,j,append='',tn=None):
        tn = self.psi if tn is None else tn 
        inds = tn.site_ind(i,j)+append,
        tags = tn.site_tag(i,j),tn.row_tag(i),tn.col_tag(j),'BRA'
        data = self._2backend(self.data_map[ci],False)
        return Tensor(data=data,inds=inds,tags=tags)
    def get_mid_env(self,i,config,append='',psi=None):
        psi = self.psi if psi is None else psi 
        row = psi.select(psi.row_tag(i),virtual=False)
        key = config[i*self.Ly:(i+1)*self.Ly]
        # compute mid env for row i
        for j in range(self.Ly-1,-1,-1):
            row.add_tensor(self.get_bra_tsr(key[j],i,j,append=append,tn=row),virtual=True)
        return row
    def contract_mid_env(self,i,row):
        try: 
            for j in range(self.Ly-1,-1,-1):
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
        M_L, *s, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                               absorb=absorb, **self.compress_opts)

        # make sure old bond being used
        ns_ix, = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})

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
    def get_bot_env(self,i,row,env_prev,config,cache=None):
        # contract mid env for row i with prev bot env 
        key = config[:(i+1)*row.Ly]
        cache = self.cache_bot if cache is None else cache
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
    def get_all_bot_envs(self,config,psi=None,cache=None,imax=None,append=''):
        imax = self.Lx-2 if imax is None else imax
        psi = self.psi if psi is None else psi
        cache = self.cache_bot if cache is None else cache
        env_prev = None
        for i in range(imax+1):
             row = self.get_mid_env(i,config,append=append,psi=psi)
             env_prev = self.get_bot_env(i,row,env_prev,config,cache=cache)
        return env_prev
    def get_top_env(self,i,row,env_prev,config,cache=None):
        # contract mid env for row i with prev top env 
        key = config[i*row.Ly:]
        cache = self.cache_top if cache is None else cache
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
    def get_all_top_envs(self,config,psi=None,cache=None,imin=None,append=''):
        imin = 1 if imin is None else imin
        psi = self.psi if psi is None else psi 
        cache = self.cache_top if cache is None else cache
        env_prev = None
        for i in range(self.Lx-1,imin-1,-1):
             row = self.get_mid_env(i,config,append=append,psi=psi)
             env_prev = self.get_top_env(i,row,env_prev,config,cache=cache)
        return env_prev
    def get_all_benvs(self,config,psi=None,cache_bot=None,cache_top=None,x_bsz=1,compute_bot=True,compute_top=True):
        psi = self.psi if psi is None else psi
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        env_bot = None
        env_top = None
        if compute_bot: 
            imax = self.rix1 if self.deterministic else self.Lx-1-x_bsz
            env_bot = self.get_all_bot_envs(config,psi=psi,cache=cache_bot,imax=imax)
        if compute_top:
            imin = self.rix2 if self.deterministic else x_bsz
            env_top = self.get_all_top_envs(config,psi=psi,cache=cache_top,imin=imin)
        return env_bot,env_top
    def get_all_lenvs(self,tn,jmax=None):
        jmax = self.Ly-2 if jmax is None else jmax
        first_col = tn.col_tag(0)
        lenvs = [None] * self.Ly
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
        last_col = tn.col_tag(self.Ly-1)
        renvs = [None] * self.Ly
        for j in range(self.Ly-1,jmin-1,-1): 
            tags = last_col if j==self.Ly-1 else (tn.col_tag(j),last_col)
            try:
                tn ^= tags
                renvs[j] = tn.select(last_col,virtual=False)
            except (ValueError,IndexError):
                return renvs
        return renvs
    def replace_sites(self,tn,sites,cis):
        for (i,j),ci in zip(sites,cis): 
            bra = tn[tn.site_tag(i,j),'BRA']
            bra_target = self.get_bra_tsr(ci,i,j,tn=tn)
            bra.modify(data=bra_target.data,inds=bra_target.inds)
        return tn
    def site_grad(self,tn_plq,i,j):
        tid = tuple(tn_plq._get_tids_from_tags((tn_plq.site_tag(i,j),'KET'),which='all'))[0]
        ket = tn_plq._pop_tensor(tid)
        g = tn_plq.contract(output_inds=ket.inds)
        return g.data 
    def update_plq_from_3row(self,plq,tn,i,x_bsz,y_bsz,psi=None):
        jmax = self.Ly - y_bsz
        psi = self.psi if psi is None else psi
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
                plq[(i,j),(x_bsz,y_bsz)] = cols.view_like_(psi)
            except (AttributeError,TypeError): # lenv/renv is None
                return plq
        return plq
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None):
        psi = self.psi if psi is None else psi
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        try:
            tn = self.get_mid_env(i,config,psi=psi)
            for ix in range(1,x_bsz):
                tn.add_tensor_network(self.get_mid_env(i+ix,config,psi=psi),virtual=False)
            if i>0:
                other = tn 
                tn = cache_bot[config[:i*self.Ly]].copy()
                tn.add_tensor_network(other,virtual=False)
            if i+x_bsz<self.Lx:
                tn.add_tensor_network(cache_top[config[(i+x_bsz)*self.Ly:]],virtual=False)
        except AttributeError:
            tn = None
        return tn 
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,cache_bot=None,cache_top=None,imin=0,imax=None):
        #if self.compute_bot and self.compute_top:
        #    raise ValueError
        imax = self.Lx-x_bsz if imax is None else imax
        psi = self.psi if psi is None else psi
        cache_bot = self.cache_bot if cache_bot is None else cache_bot
        cache_top = self.cache_top if cache_top is None else cache_top
        plq = dict()
        for i in range(imin,imax+1):
            tn = self.build_3row_tn(config,i,x_bsz,psi=psi,cache_bot=cache_bot,cache_top=cache_top)
            if tn is not None:
                plq = self.update_plq_from_3row(plq,tn,i,x_bsz,y_bsz,psi=psi)
        return plq
    def get_grad_dict_from_plq(self,plq,cx,backend='numpy'):
        # gradient
        vx = dict()
        for ((i0,j0),(x_bsz,y_bsz)),tn in plq.items():
            where = (i0,j0),(i0+x_bsz-1,j0+y_bsz-1)
            for i in range(i0,i0+x_bsz):
                for j in range(j0,j0+y_bsz):
                    if (i,j) in vx:
                        continue
                    vx[i,j] = self._2numpy(self.site_grad(tn.copy(),i,j)/cx[where],backend=backend)
        return vx
class AmplitudeFactory(ContractionEngine):
    def __init__(self,psi,blks=None,phys_dim=2):
        super().init_contraction(psi.Lx,psi.Ly,phys_dim=phys_dim)
        psi.add_tag('KET')

        if blks is None:
            blks = [list(itertools.product(range(self.Lx),range(self.Ly)))]
        self.site_map = self.get_site_map(blks)
        self.constructors = self.get_constructors(psi)
        self.block_dict = self.get_block_dict(blks)
        if RANK==0:
            sizes = [stop-start for start,stop in self.block_dict]
            print('block_dict=',self.block_dict)
            print('sizes=',sizes)

        self.set_psi(psi) # current state stored in self.psi
        self.backend = 'numpy'
    def get_site_map(self,blks):
        site_order = []
        for blk in blks:
            site_order += blk
        site_map = dict()
        for ix,site in enumerate(site_order):
            site_map[site] = ix
        return site_map
    def config_sign(self,config=None):
        return 1.
    def get_constructors(self,peps):
        constructors = [None] * (self.Lx * self.Ly)
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            data = peps[i,j].data
            #ix = flatten(i,j,peps.Ly)
            ix = self.site_map[i,j]
            constructors[ix] = data.shape,len(data.flatten()),(i,j)
        return constructors
    def get_block_dict(self,blks):
        start = 0
        blk_dict = [None] * len(blks)
        for bix,blk in enumerate(blks):
            site_min,site_max = blk[0],blk[-1]
            ix_min,ix_max = self.site_map[site_min],self.site_map[site_max]
            stop = start
            for ix in range(ix_min,ix_max+1):
                _,size,_ = self.constructors[ix]
                stop += size
            blk_dict[bix] = start,stop
            start = stop
        return blk_dict 
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
        for ix,(_,size,(i,j)) in enumerate(self.constructors):
            ls[ix] = self.tensor2vec(psi[i,j].data,ix=ix)
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
        for ix,(_,_,(i,j)) in enumerate(self.constructors):
            psi[i,j].modify(data=self.vec2tensor(ls[ix],ix))
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

        env_bot,env_top = self.get_all_benvs(config,x_bsz=1,compute_bot=compute_bot,compute_top=compute_top)
        if env_bot is None and env_top is None:
            return 0.
        try:
            if self.deterministic:
                tn = env_bot.copy()
                tn.add_tensor_network(env_top,virtual=False)
            elif compute_bot: 
                tn = env_bot.copy()
                tn.add_tensor_network(self.get_mid_env(self.Lx-1,config),virtual=False) 
            elif compute_top:
                tn = self.get_mid_env(0,config)
                tn.add_tensor_network(env_top,virtual=False)
        except AttributeError:
            return 0.
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
    def _pair_energy_from_plq(self,tn,config,where):
        ix1,ix2 = self.flatten(*where[0]),self.flatten(*where[1])
        i1,i2 = config[ix1],config[ix2] 
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        kixs = [tn.site_ind(*site) for site in where]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip(where,kixs,bixs):
            tn[tn.site_tag(*site),'BRA'].reindex_({kix:bix})
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

            tn = plq.get(key,None) 
            if tn is not None:
                eij = self._pair_energy_from_plq(tn.copy(),config,where) 
                if eij is not None:
                    ex[where] = eij

                cij = self._2numpy(tn.copy().contract())
                cx[where] = cij 
        return ex,cx
    def batch_pair_energies_from_plq(self,batch_idx,config,psi):
        cache_bot,cache_top = dict(),dict()
        bix,tix,plq_types,pairs = self.batched_pairs[batch_idx]
        self.get_all_bot_envs(config,psi=psi,cache=cache_bot,imax=bix)
        self.get_all_top_envs(config,psi=psi,cache=cache_top,imin=tix)

        # form plqs
        plq = dict()
        for imin,imax,x_bsz,y_bsz in plq_types:
            plq.update(self.get_plq_from_benvs(config,x_bsz,y_bsz,psi=psi,cache_bot=cache_bot,cache_top=cache_top,imin=imin,imax=imax))

        # compute energy numerator 
        ex,cx = self._pair_energies_from_plq(plq,pairs,config)
        return ex,cx,plq
    def batch_hessian_from_plq(self,batch_idx,config,amplitude_factory): # only used for Hessian
        self.backend = 'torch'
        psi = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            psi[i,j].modify(data=self._2backend(psi[i,j].data,True))
        ex,cx,plq = self.batch_pair_energies_from_plq(batch_idx,config,psi)

        _,Hvx = self.parse_hessian(ex,psi,amplitude_factory)
        ex = sum([self._2numpy(eij)/cx[where] for where,eij in ex.items()])
        vx = self.get_grad_dict_from_plq(plq,cx,backend=self.backend) 
        return ex,Hvx,cx,vx
    def compute_local_energy_hessian_from_plq(self,config,amplitude_factory): 
        ar.set_backend(torch.zeros(1))

        ex,Hvx = 0.,0.
        cx,vx = dict(),dict()
        for batch_idx in self.batched_pairs:
            ex_,Hvx_,cx_,vx_ = self.batch_hessian_from_plq(batch_idx,config,amplitude_factory)  
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
    def pair_energies_from_plq(self,config,amplitude_factory): 
        self.backend = 'numpy'
        x_bsz_min = min([x_bsz for x_bsz,_ in self.plq_sz])
        amplitude_factory.get_all_benvs(config,x_bsz=x_bsz_min)

        plq = dict()
        for x_bsz,y_bsz in self.plq_sz:
            plq.update(amplitude_factory.get_plq_from_benvs(config,x_bsz,y_bsz))

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
    def amplitude_gradient_deterministic(self,config,amplitude_factory):
        self.backend = 'torch'
        cache_top = dict()
        cache_bot = dict()
        psi = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            psi[i,j].modify(data=self._2backend(psi[i,j].data,True))

        #self.deterministic = True
        #self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
        env_bot,env_top = self.get_all_benvs(config,psi=psi,cache_bot=cache_bot,cache_top=cache_top)
        tn = env_bot.copy()
        tn.add_tensor_network(env_top,virtual=False)
        cx = tn.contract() 

        cx.backward()
        vx = dict()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            vx[i,j] = self.tsr_grad(psi[i,j].data)  
        vx = {site:self._2numpy(vij) for site,vij in vx.items()}
        vx = amplitude_factory.dict2vec(vx)  
        cx = self._2numpy(cx)
        return cx,vx/cx
    def _pair_energy_deterministic(self,config,site1,site2,psi,top,bot,sign_fn):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        imin = min(self.rix1+1,site1[0],site2[0]) 
        imax = max(self.rix2-1,site1[0],site2[0]) 
        ex = [] 
        coeff_comm = self.intermediate_sign(config,ix1,ix2) * self.pair_coeff(site1,site2)
        cache_top = dict()
        cache_bot = dict()
        for i1_new,i2_new,coeff in self.pair_terms(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            sign_new = sign_fn(config_new)

            bot_term = None if bot is None else bot.copy()
            for i in range(imin,self.rix1+1):
                row = self.get_mid_env(i,config_new,psi=psi)
                bot_term = self.get_bot_env(i,row,bot_term,config_new,cache=cache_bot)
            if bot_term is None:
                continue

            top_term = None if top is None else top.copy()
            for i in range(imax,self.rix2-1,-1):
                row = self.get_mid_env(i,config_new,psi=psi)
                top_term = self.get_top_env(i,row,top_term,config_new,cache=cache_top)
            if top_term is None:
                continue

            tn = bot_term.copy()
            tn.add_tensor_network(top_term,virtual=False)
            cx_new = self.safe_contract(tn)
            if cx_new is None:
                continue
            ex.append(coeff * sign_new * cx_new)
        if len(ex)==0:
            return None
        return sum(ex) * coeff_comm
    def batch_pair_energies_deterministic(self,config,psi,batch_imin,batch_imax,sign_fn):
        cache_top = dict()
        cache_bot = dict()
        
        imin = min(self.rix1+1,batch_imin) 
        imax = max(self.rix2-1,batch_imax) 
        self.get_all_bot_envs(config,psi=psi,cache=cache_bot,imax=imin-1)
        self.get_all_top_envs(config,psi=psi,cache=cache_top,imin=imax+1)
        top = None if imax==self.Lx-1 else cache_top[config[(imax+1)*self.Ly:]]
        bot = None if imin==0 else cache_bot[config[:imin*self.Ly]]

        ex = dict() 
        for site1,site2 in self.batched_pairs[batch_imin,batch_imax]:
            eij = self._pair_energy_deterministic(config,site1,site2,psi,top,bot,sign_fn)
            if eij is not None:
                ex[site1,site2] = eij
        return ex
    def batch_hessian_deterministic(self,config,amplitude_factory,batch_imin,batch_imax):
        peps = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            peps[i,j].modify(data=self._2backend(peps[i,j].data,True))
        sign_fn = amplitude_factory.config_sign
        ex = self.batch_pair_energies_deterministic(config,peps,batch_imin,batch_imax,sign_fn)
        return self.parse_hessian(ex,peps,amplitude_factory)
    def pair_energy_deterministic(self,config,psi,site1,site2,sign_fn):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 

        cache_top = dict()
        cache_bot = dict()
        imin = min(site1[0],site2[0])
        imax = max(site1[0],site2[0])
        imin = min(self.rix1+1,imin) 
        imax = max(self.rix2-1,imax) 
        self.get_all_bot_envs(config,psi=psi,cache=cache_bot,imax=imin-1)
        self.get_all_top_envs(config,psi=psi,cache=cache_top,imin=imax+1)
        top = None if imax==self.Lx-1 else cache_top[config[(imax+1)*self.Ly:]]
        bot = None if imin==0 else cache_bot[config[:imin*self.Ly]]
        return self._pair_energy_deterministic(config,site1,site2,peps,top,bot,sign_fn)
    def pair_hessian_deterministic(self,config,amplitude_factory,site1,site2):
        peps = amplitude_factory.psi.copy()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            peps[i,j].modify(data=self._2backend(peps[i,j].data,True))
        sign_fn = amplitude_factory.config_sign
        ex = self.pair_energy_deterministic(config,peps,site1,site2,sign_fn)
        if ex is None:
            return 0.,0.
        return self.parse_hessian({(site1,site2):ex},peps,amplitude_factory)
    def compute_local_energy_hessian_deterministic(self,config,amplitude_factory):
        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))

        cx,vx = self.amplitude_gradient_deterministic(config,amplitude_factory)
        sign = amplitude_factory.config_sign(config)

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
         
        eu = self.compute_local_energy_eigen(config)
        ex = ex/cx*sign + eu
        Hvx = Hvx/cx*sign + eu*vx
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,Hvx,0. 
    def pair_energies_deterministic(self,config,amplitude_factory):
        self.backend = 'numpy'
        psi = amplitude_factory.psi
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top
        env_bot,env_top = amplitude_factory.get_all_benvs(config)

        sign_fn = amplitude_factory.config_sign
        ex = dict() 
        for (site1,site2) in self.pairs:
            imin = min(self.rix1+1,site1[0],site2[0]) 
            imax = max(self.rix2-1,site1[0],site2[0]) 
            top = None if imax==self.Lx-1 else cache_top[config[(imax+1)*self.Ly:]]
            bot = None if imin==0 else cache_bot[config[:imin*self.Ly]]

            eij = self._pair_energy_deterministic(config,site1,site2,psi,top,bot,sign_fn)
            if eij is not None:
                ex[site1,site2] = eij
        tn = env_bot.copy()
        tn.add_tensor_network(env_top,virtual=False)
        cx = tn.contract() 
        return ex,cx
    def compute_local_energy_gradient_deterministic(self,config,amplitude_factory,compute_v=True):
        sign = amplitude_factory.config_sign(config)
        ex,cx = self.pair_energies_deterministic(config,amplitude_factory)
        ex = sum(ex.values()) / cx * sign
        eu = self.compute_local_energy_eigen(config)
        ex += eu
        if not compute_v:
            return cx,ex,None,None,0.

        self.backend = 'torch'
        ar.set_backend(torch.zeros(1))
        _,vx = self.amplitude_gradient_deterministic(config,amplitude_factory)
        ar.set_backend(np.zeros(1))
        return cx,ex,vx,None,0.
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        if self.deterministic:
            if compute_Hv:
                return self.compute_local_energy_hessian_deterministic(config,amplitude_factory)
            else:
                return self.compute_local_energy_gradient_deterministic(config,amplitude_factory,compute_v=compute_v)
        else:
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
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            Hvx[i,j] = self._2numpy(self.tsr_grad(psi[i,j].data))
        return self._2numpy(ex_num),amplitude_factory.dict2vec(Hvx)  
    def contraction_error(self,cx):
        cx = np.array(list(cx.values()))
        max_,min_,mean_ = np.amax(cx),np.amin(cx),np.mean(cx)
        return mean_,np.fabs((max_-min_)/mean_)
    def pairs_nn(self,d=1):
        ls = [] 
        for i in range(self.Lx):
            for j in range(self.Ly):
                if j+d<self.Ly:
                    where = (i,j),(i,j+d)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (i,(j+d)%self.Ly),(i,j)
                        ls.append(where)
                if i+d<self.Lx:
                    where = (i,j),(i+d,j)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = ((i+d)%self.Lx,j),(i,j)
                        ls.append(where)
        return ls
    def batch_nnh(self,d=1):
        for i in range(self.Lx):
            ls = self.batched_pairs.get((i,i),[])
            for j in range(self.Ly):
                if j+d<self.Ly:
                    where = (i,j),(i,j+d)
                    ls.append(where)
                else:
                    if self.pbc:
                        where = (i,(j+d)%self.Ly),(i,j)
                        ls.append(where)
            self.batched_pairs[i,i] = ls
    def batch_nnv(self,d=1):
        for i in range(self.Lx-d):
            ls = self.batched_pairs.get((i,i+d),[])
            for j in range(self.Ly):
                where = (i,j),(i+d,j)
                ls.append(where)
            self.batched_pairs[i,i+d] = ls
        if not self.pbc:
            return
        ls = self.batched_pairs.get('pbc',[]) 
        for i in range(self.Lx-d,self.Lx):
            for j in range(self.Ly):
                where = ((i+d)%self.Lx,j),(i,j)
                ls.append(where)
        self.batched_pairs['pbc'] = ls
    def batch_plq_nn(self,d=1):
        self.batched_pairs = dict() 
        batchsize = max(self.Lx // self.nbatch,d+1)
        for i in range(self.Lx):
            batch_idx = i // batchsize
            if batch_idx not in self.batched_pairs:
                self.batched_pairs[batch_idx] = [],[] 
            rows,pairs = self.batched_pairs[batch_idx]
            for ix in range(d+1):
                if i+ix < self.Lx:
                    rows.append(i+ix)
            for j in range(self.Ly):
                for ix in range(1,d+1):
                    if j+ix<self.Ly:
                        where = (i,j),(i,j+ix)
                        pairs.append(where)
                for ix in range(1,d+1):
                    if i+ix<self.Lx:
                        where = (i,j),(i+ix,j)
                        pairs.append(where)
        self.pairs = []
        for batch_idx in self.batched_pairs:
            rows,pairs = self.batched_pairs[batch_idx]
            imin,imax = min(rows),max(rows)
            bix,tix = max(0,imax-1),min(imin+1,self.Lx-1) # bot_ix,top_ix 
            plq_types = (imin,imax,1,d+1), (imin,imax-d,d+1,1),# i0_min,i0_max,x_bsz,y_bsz
            self.batched_pairs[batch_idx] = bix,tix,plq_types,pairs 
            self.pairs += pairs
        self.plq_sz = (1,d+1),(d+1,1)
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
    def pair_key(self,site1,site2):
        # site1,site2 -> (i0,j0),(x_bsz,y_bsz)
        dx = site2[0]-site1[0]
        dy = site2[1]-site1[1]
        return site1,(dx+1,dy+1)
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

        self.pairs = self.pairs_nn()
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_plq_nn()
    def pair_coeff(self,site1,site2):
        # coeff for pair tsr
        return 1.
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

        self.pairs = self.pairs_nn() + self.pairs_diag() # list of all pairs, for SR
        if self.deterministic:
            self.batch_deterministic()
        else:
            self.batch_plq()
    def pairs_diag(self):
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
    def batch_deterministic(self):
        self.batched_pairs = dict()
        self.batch_nnh() 
        self.batch_nnv() 
        self.batch_diag() 
    def batch_plq(self):
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
class ExchangeSampler1:
    def __init__(self,Lx,Ly,seed=None,burn_in=0):
        self.Lx, self.Ly = Lx,Ly
        self.nsite = Lx * Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.backend = 'numpy'
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,i,j):
        return flat2site(i,j,self.Ly)
    def preprocess(self):
        self._burn_in()
    def _burn_in(self,config=None,burn_in=None):
        if config is not None:
            self.config = config 
        self.sweep_row_dir = self.rng.choice([-1,1]) 
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
    def _new_pair(self,site1,site2):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.amplitude_factory.pair_valid(i1,i2): # continue
            return (None,) * 2
        i1_new,i2_new = self.new_pair(i1,i2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        return (i1_new,i2_new),tuple(config_new)
    def update_plq(self,i,j,cols,tn,saved_rows):
        if cols[0] is None:
            return tn,saved_rows
        tn_plq = cols[0].copy()
        for col in cols[1:]:
            if col is None:
                return tn,saved_rows
            tn_plq.add_tensor_network(col,virtual=False)
        tn_plq.view_like_(tn)
        pairs = self.get_pairs(i,j) 
        for sites in pairs:
            config_sites,config_new = self._new_pair(*sites)
            if config_sites is None:
                continue
            tn_pair = self.amplitude_factory.replace_sites(tn_plq.copy(),sites,config_sites) 
            py = self.amplitude_factory.safe_contract(tn_pair)
            if py is None:
                continue
            py = py**2
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept, update px & config & env_m
                self.px = py
                self.config = config_new 
                tn_plq = self.amplitude_factory.replace_sites(tn_plq,sites,config_sites)
                tn = self.amplitude_factory.replace_sites(tn,sites,config_sites)
                saved_rows = self.amplitude_factory.replace_sites(saved_rows,sites,config_sites)
        return tn,saved_rows
    def sweep_col_forward(self,i,rows):
        tn = rows[0].copy()
        for row in rows[1:]:
            tn.add_tensor_network(row,virtual=False)
        saved_rows = tn.copy()
        try:
            tn.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        renvs = self.amplitude_factory.get_all_renvs(tn.copy(),jmin=2)
        first_col = tn.col_tag(0)
        for j in range(self.Ly-1): # 0,...,Ly-2
            tags = first_col,tn.col_tag(j),tn.col_tag(j+1)
            cols = [tn.select(tags,which='any',virtual=False)]
            if j<self.Ly-2:
                cols.append(renvs[j+2])
            tn,saved_rows = self.update_plq(i,j,cols,tn,saved_rows) 
            # update new lenv
            if j<self.Ly-2:
                tn ^= first_col,tn.col_tag(j) 
        return saved_rows
    def sweep_col_backward(self,i,rows):
        tn = rows[0].copy()
        for row in rows[1:]:
            tn.add_tensor_network(row,virtual=False)
        saved_rows = tn.copy()
        try:
            tn.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.amplitude_factory.get_all_lenvs(tn.copy(),jmax=self.Ly-3)
        last_col = tn.col_tag(self.Ly-1)
        for j in range(self.Ly-1,0,-1): # Ly-1,...,1
            cols = []
            if j>1: 
                cols.append(lenvs[j-2])
            tags = tn.col_tag(j-1),tn.col_tag(j),last_col
            cols.append(tn.select(tags,which='any',virtual=False))
            tn,saved_rows = self.update_plq(i,j-1,cols,tn,saved_rows) 
            # update new renv
            if j>1:
                tn ^= tn.col_tag(j),last_col
        return saved_rows
    def sweep_row_forward(self):
        self.amplitude_factory.cache_bot = dict()
        psi = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        # can assume to have all opposite envs
        self.get_all_top_envs(self.config,psi=psi,cache=cache_top,imin=2)
        sweep_col = self.sweep_col_forward if self.sweep_col_dir == 1 else\
                    self.sweep_col_backward

        env_bot = None 
        row1 = amplitude_factory.get_mid_env(0,self.config)
        for i in range(self.Lx-1):
            rows = []
            if i>0:
                rows.append(env_bot)
            row2 = amplitude_factory.get_mid_env(i+1,self.config)
            rows += [row1,row2]
            if i<self.Lx-2:
                rows.append(cache_top[self.config[(i+2)*self.Ly:]]) 
            saved_rows = sweep_col(i,rows)
            row1_new = saved_rows.select(psi.row_tag(i),virtual=False)
            row2_new = saved_rows.select(psi.row_tag(i+1),virtual=False)
            # update new env_h
            env_bot = amplitude_factory.get_bot_env(i,row1_new,env_bot,tuple(self.config))
            row1 = row2_new
    def sweep_row_backward(self):
        self.amplitude_factory.cache_top = dict()
        psi = self.amplitude_factory.psi
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        # can assume to have all opposite envs
        self.get_all_bot_envs(self.config,psi=psi,cache=cache_bot,imax=self.Lx-3)
        sweep_col = self.sweep_col_forward if self.sweep_col_dir == 1 else\
                    self.sweep_col_backward

        env_top = None 
        row1 = self.amplitude_factory.get_mid_env(self.Lx-1,self.config)
        for i in range(self.Lx-1,0,-1):
            rows = []
            if i>1:
                rows.append(cache_bot[self.config[:(i-1)*self.Ly]])
            row2 = self.amplitude_factory.get_mid_env(i-1,self.config)
            rows += [row2,row1]
            if i<self.Lx-1:
                rows.append(env_top) 
            saved_rows = sweep_col(i-1,rows)
            row1_new = saved_rows.select(peps.row_tag(i),virtual=False)
            row2_new = saved_rows.select(peps.row_tag(i-1),virtual=False)
            # update new env_h
            env_top = self.amplitude_factory.get_top_env(i,row1_new,env_top,tuple(self.config))
            row1 = row2_new
    def sample(self):
        self.sweep_col_dir = self.rng.choice([-1,1]) # randomly choses the col sweep direction
        if self.amplitude_factory.deterministic:
            self._sample_deterministic()
        else:
            self._sample()
        # setup to compute all opposite env for gradient
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

        for i,j in itertools.product(sweep_row,sweep_col):
            self.update_pair_deterministic(i,j)
        self.update_cache()
    def update_cache(self):
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        rix1,rix2 = self.amplitude_factory.rix1,self.amplitude_factory.rix2

        cache_bot_new = dict()
        for i in range(rix1+1):
            key = self.config[:(i+1)*self.Ly]
            cache_bot_new[key] = cache_bot[key]
        cache_top_new = dict()
        for i in range(rix2,self.Lx):
            key = self.config[i*self.Ly:]
            cache_top_new[key] = cache_top[key]

        self.amplitude_factory.cache_bot = cache_bot_new
        self.amplitude_factory.cache_top = cache_top_new
    def update_pair_deterministic(self,i,j):
        cache_bot = self.amplitude_factory.cache_bot
        cache_top = self.amplitude_factory.cache_top
        rix1,rix2 = self.amplitude_factory.rix1,self.amplitude_factory.rix2
        pairs = self.get_pairs(i,j)
        for sites in pairs:
            config_sites,config_new = self._new_pair(*sites)
            if config_sites is None:
                continue
            imin = min(rix1+1,site1[0]) 
            imax = max(rix2-1,site2[0]) 
            top = None if imax==self.Lx-1 else cache_top[self.config[(imax+1)*self.Ly:]]
            bot = None if imin==0 else cache_bot[self.config[:imin*self.Ly]]

            bot_term = None if bot is None else bot.copy()
            for i in range(imin,rix1+1):
                row = self.amplitude_factory.get_mid_env(i,config_new)
                bot_term = self.amplitude_factory.get_bot_env(i,row,bot_term,config_new)
            if imin > 0 and bot_term is None:
                continue

            top_term = None if top is None else top.copy()
            for i in range(imax,rix2-1,-1):
                row = self.amplitude_factory.get_mid_env(i,config_new)
                top_term = self.amplitude_factory.get_top_env(i,row,top_term,config_new)
            if imax < self.Lx-1 and top_term is None:
                continue

            tn = bot_term.copy()
            tn.add_tensor_network(top_term,virtual=False)
            py = self.amplitude_factory.safe_contract(tn)
            if py is None:
                continue
            py = py ** 2 
            try:
                acceptance = py / self.px
            except ZeroDivisionError:
                acceptance = 1. if py > self.px else 0.
            if self.rng.uniform() < acceptance: # accept, update px & config & env_m
                self.px = py
                self.config = tuple(config_new) 
class ExchangeSampler2:
    def __init__(self,Lx,Ly,seed=None,burn_in=0):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.backend = 'numpy'
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,i,j):
        return flat2site(i,j,self.Ly)
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
    def _new_pair(self,i,j,x_bsz,y_bsz):
        if (x_bsz,y_bsz)==(1,2):
            site1,site2 = (i,j),(i,(j+1)%self.Ly)
        elif (x_bsz,y_bsz)==(2,1):
            site1,site2 = (i,j),((i+1)%self.Lx,j)
        else:
            raise NotImplementedError
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.amplitude_factory.pair_valid(i1,i2): # continue
            return (None,) * 3
        i1_new,i2_new = self.new_pair(i1,i2)
        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new
        return (site1,site2),(i1_new,i2_new),tuple(config_new)
    def update_pair(self,i,j,x_bsz,y_bsz,cols,tn):
        sites,config_sites,config_new = self._new_pair(i,j,x_bsz,y_bsz)
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
    def _get_cols_forward(self,first_col,j,y_bsz,tn,renvs):
        tags = [tn.col_tag(j+ix) for ix in range(y_bsz)]
        cols = tn.select(tags,which='any',virtual=False)
        if j>0:
            other = cols
            cols = tn.select(first_col,virtual=False)
            cols.add_tensor_network(other,virtual=False)
        if j<self.Ly - y_bsz:
            cols.add_tensor_network(renvs[j+y_bsz],virtual=False)
        cols.view_like_(tn)
        return cols
    def sweep_col_forward(self,i,tn,x_bsz,y_bsz):
        try:
            tn.reorder('col',inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        renvs = self.amplitude_factory.get_all_renvs(tn.copy(),jmin=y_bsz)
        first_col = tn.col_tag(0)
        for j in range(self.Ly - y_bsz + 1): 
            cols = self._get_cols_forward(first_col,j,y_bsz,tn,renvs)
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            tn ^= first_col,tn.col_tag(j) 
    def _get_cols_backward(self,last_col,j,y_bsz,tn,lenvs):
        tags = [tn.col_tag(j+ix) for ix in range(y_bsz)] 
        cols = tn.select(tags,which='any',virtual=False)
        if j>0:
            other = cols
            cols = lenvs[j-1]
            cols.add_tensor_network(other,virtual=False)
        if j<self.Ly - y_bsz:
            cols.add_tensor_network(tn.select(last_col,virtual=False),virtual=False)
        cols.view_like_(tn)
        return cols
    def sweep_col_backward(self,i,tn,x_bsz,y_bsz):
        try:
            tn.reorder('col',inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        lenvs = self.amplitude_factory.get_all_lenvs(tn.copy(),jmax=self.Ly-1-y_bsz)
        last_col = tn.col_tag(self.Ly-1)
        for j in range(self.Ly - y_bsz,-1,-1): # Ly-1,...,1
            cols = self._get_cols_backward(last_col,j,y_bsz,tn,lenvs)
            tn = self.update_pair(i,j,x_bsz,y_bsz,cols,tn) 
            tn ^= tn.col_tag(j+y_bsz-1),last_col
    def sweep_row_forward(self,x_bsz,y_bsz):
        self.amplitude_factory.cache_bot = dict()
        self.amplitude_factory.get_all_top_envs(self.config,imin=x_bsz)
        cache_bot = self.amplitude_factory.cache_bot

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax+1):
            tn = self.amplitude_factory.build_3row_tn(self.config,i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)

            for inew in range(i,i+x_bsz):
                row = self.amplitude_factory.get_mid_env(inew,self.config)
                env_prev = None if inew==0 else cache_bot[self.config[:inew*self.Ly]] 
                self.amplitude_factory.get_bot_env(inew,row,env_prev,self.config)
    def sweep_row_backward(self,x_bsz,y_bsz):
        self.amplitude_factory.cache_top = dict()
        self.amplitude_factory.get_all_bot_envs(self.config,imax=self.Lx-1-x_bsz)
        cache_top = self.amplitude_factory.cache_top

        cdir = self.rng.choice([-1,1]) 
        sweep_col = self.sweep_col_forward if cdir == 1 else self.sweep_col_backward

        imax = self.Lx-x_bsz
        for i in range(imax,-1,-1):
            tn = self.amplitude_factory.build_3row_tn(self.config,i,x_bsz)
            sweep_col(i,tn,x_bsz,y_bsz)

            for inew in range(i+x_bsz-1,i-1,-1):
                row = self.amplitude_factory.get_mid_env(inew,self.config)
                env_prev = None if inew==self.Lx-1 else cache_top[self.config[(inew+1)*self.Ly:]] 
                self.amplitude_factory.get_top_env(inew,row,env_prev,self.config)
    def sample(self):
        if self.amplitude_factory.deterministic:
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
        for x_bsz,y_bsz in [(1,2),(2,1)]:
            imax = self.Lx-1 if self.amplitude_factory.pbc else self.Lx-x_bsz
            jmax = self.Ly-1 if self.amplitude_factory.pbc else self.Ly-y_bsz
            rdir = self.rng.choice([-1,1]) 
            cdir = self.rng.choice([-1,1]) 
            sweep_row = range(0,imax+1) if rdir==1 else range(imax,-1,-1)
            sweep_col = range(0,jmax+1) if cdir==1 else range(jmax,-1,-1)
            for i,j in itertools.product(sweep_row,sweep_col):
                self.update_pair_deterministic(i,j,x_bsz,y_bsz)
        self.update_cache(self.amplitude_factory,self.config)
    def update_cache(self,amplitude_factory,config):
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top
        rix1,rix2 = amplitude_factory.rix1,amplitude_factory.rix2

        cache_bot_new = dict()
        for i in range(rix1+1):
            key = config[:(i+1)*self.Ly]
            cache_bot_new[key] = cache_bot[key]
        cache_top_new = dict()
        for i in range(rix2,self.Lx):
            key = config[i*self.Ly:]
            cache_top_new[key] = cache_top[key]

        amplitude_factory.cache_bot = cache_bot_new
        amplitude_factory.cache_top = cache_top_new
    def _prob_deterministic(self,config_old,config_new,amplitude_factory,site1,site2):
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top
        rix1,rix2 = amplitude_factory.rix1,amplitude_factory.rix2
        imin = min(rix1+1,site1[0],site2[0]) 
        imax = max(rix2-1,site1[0],site2[0]) 
        top = None if imax==self.Lx-1 else cache_top[config_old[(imax+1)*self.Ly:]]
        bot = None if imin==0 else cache_bot[config_old[:imin*self.Ly]]
        
        bot_term = None if bot is None else bot.copy()
        for i in range(imin,rix1+1):
            row = amplitude_factory.get_mid_env(i,config_new)
            bot_term = amplitude_factory.get_bot_env(i,row,bot_term,config_new)
        if imin > 0 and bot_term is None:
            return None 

        top_term = None if top is None else top.copy()
        for i in range(imax,rix2-1,-1):
            row = amplitude_factory.get_mid_env(i,config_new)
            top_term = amplitude_factory.get_top_env(i,row,top_term,config_new)
        if imax < self.Lx-1 and top_term is None:
            return None

        tn = bot_term.copy()
        tn.add_tensor_network(top_term,virtual=False)
        py = amplitude_factory.safe_contract(tn)
        if py is None:
            return None
        return py ** 2
    def update_pair_deterministic(self,i,j,x_bsz,y_bsz):
        sites,config_sites,config_new = self._new_pair(i,j,x_bsz,y_bsz)
        if config_sites is None:
            return
        site1,site2 = sites
        py = self._prob_deterministic(self.config,config_new,self.amplitude_factory,site1,site2)
        if py is None:
            return
        try:
            acceptance = py / self.px
        except ZeroDivisionError:
            acceptance = 1. if py > self.px else 0.
        if self.rng.uniform() < acceptance: # accept, update px & config & env_m
            #print('acc')
            self.px = py
            self.config = tuple(config_new) 
