import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

# set tensor symmetry
import sys
this = sys.modules[__name__]
def set_options(phys_dim=2):
    this.data_map = dict()
    for i in range(phys_dim):
        data = np.zeros(phys_dim)
        data[i] = 1.
        this.data_map[i] = data
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
def get_product_state(Lx,Ly,config):
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [1] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (2,)

            data = np.zeros(shape) 
            ix = flatten(i,j,Ly)
            ix = config[ix]
            data[...,ix] = 1.
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
####################################################################################
# amplitude fxns 
####################################################################################
from .tensor_core import Tensor,TensorNetwork,rand_uuid,tensor_split
class ContractionEngine:
    def get_bra_tsr(self,peps,ci,i,j,append=''):
        inds = peps.site_ind(i,j)+append,
        tags = peps.site_tag(i,j),peps.row_tag(i),peps.col_tag(j),'BRA'
        data = data_map[ci].copy()
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
    def get_bot_env(self,i,row,env_prev,config,cache,**compress_opts):
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
            tn.contract_boundary_from_bottom_(xrange=(i-1,i),**compress_opts)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_all_bot_envs(self,peps,config,cache_bot,imax=None,append='',**compress_opts):
        # imax for bot env
        imax = peps.Lx-2 if imax is None else imax
        env_prev = None
        for i in range(imax+1):
             row = self.get_mid_env(i,peps,config,append=append)
             env_prev = self.get_bot_env(i,row,env_prev,config,cache_bot,**compress_opts)
        return env_prev
    def get_top_env(self,i,row,env_prev,config,cache,**compress_opts):
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
            tn.contract_boundary_from_top_(xrange=(i,i+1),**compress_opts)
        except (ValueError,IndexError):
            tn = None
        cache[key] = tn
        return tn 
    def get_all_top_envs(self,peps,config,cache_top,imin=None,append='',**compress_opts):
        imin = 1 if imin is None else imin
        env_prev = None
        for i in range(peps.Lx-1,imin-1,-1):
             row = self.get_mid_env(i,peps,config,append=append)
             env_prev = self.get_top_env(i,row,env_prev,config,cache_top,**compress_opts)
        return env_prev
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
    def cache_update(self,cache_bot,cache_top,ix,Lx,Ly):
        i,_ = flat2site(ix,Lx,Ly) 
        keys = list(cache_bot.keys())
        l = i * Ly
        for key in keys:
            if len(key) > l:
                cache_bot.pop(key)
    
        keys = list(cache_top.keys())
        l = (Lx - i - 1) * Ly
        for key in keys:
            if len(key) > l:
                cache_top.pop(key)
        return cache_bot,cache_top
    def amplitude(self,peps,config):
        for ix,ci in reversed(list(enumerate(config))):
            i,j = flat2site(ix,peps.Lx,peps.Ly)
            peps.add_tensor(self.get_bra_tsr(peps,ci,i,j))
        try:
            cx = peps.contract()
        except (ValueError,IndexError):
            cx = 0.
        return cx 
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
        return plq
class AmplitudeFactory(ContractionEngine):
    def __init__(self,psi,dmrg=False,**contract_opts):
        self.contract_opts = contract_opts
        self.Lx,self.Ly = psi.Lx,psi.Ly
        psi.add_tag('KET')
        self.constructors = self.get_constructors(psi)
        self.get_block_dict()
        self.dmrg = dmrg

        self.ix = None
        self.set_psi(psi) # current state stored in self.psi
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
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
        self.store = dict()
        self.store_grad = dict()

        self.compute_bot = True
        self.compute_top = True
        if self.ix is None:
            self.cache_bot = dict()
            self.cache_top = dict()
            return
        self.cache_bot,self.cache_top = self.cache_update(
            self.cache_bot,self.cache_top,self.ix,self.Lx,self.Ly)
    def update_scheme(self,benv_dir):
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
    def get_all_benvs(self,config,x_bsz=1):
        env_bot = None
        env_top = None
        if self.compute_bot: 
            env_bot = self.get_all_bot_envs(self.psi,config,self.cache_bot,imax=self.Lx-1-x_bsz,
                                            append='',**self.contract_opts)
        if self.compute_top:
            env_top = self.get_all_top_envs(self.psi,config,self.cache_top,imin=x_bsz,
                                            append='',**self.contract_opts)
        return env_bot,env_top
    def unsigned_amplitude(self,config):
        # should only be used to:
        # 1. compute dense probs
        # 2. initialize MH sampler
        if config in self.store:
            return self.store[config]
        if self.compute_bot and self.compute_top:
            raise ValueError
        env_bot,env_top = self.get_all_benvs(config,x_bsz=1)
        if env_bot is None and env_top is None:
            unsigned_cx = 0.
            self.store[config] = unsigned_cx
            return unsigned_cx  
        if self.compute_bot: 
            row = self.get_mid_env(self.Lx-1,self.psi,config)
            tn = env_bot.copy()
            tn.add_tensor_network(row,virtual=True) 
        if self.compute_top:
            row = self.get_mid_env(0,self.psi,config)
            tn = row
            tn.add_tensor_network(env_top.copy(),virtual=True)
        try:
            unsigned_cx = tn.contract()
        except (ValueError,IndexError):
            unsigned_cx = 0.
        self.store[config] = unsigned_cx
        return unsigned_cx
    def compute_config_sign(self,config=None):
        return 1.
    def amplitude(self,config):
        unsigned_cx = self.unsigned_amplitude(config)
        sign = self.compute_config_sign(config)
        return unsigned_cx * sign 
    def grad(self,config,plq=None):
        # currently not called
        raise NotImplementedError
        sign = self.compute_config_sign(config)
        if config in self.store_grad:
            unsigned_cx = self.store[config]
            vx = self.store_grad[config]
            return sign * unsigned_cx, vx
        if plq is None:
            self.get_all_benvs(config,x_bsz=1)
            plq = self.get_plq_from_benvs(config,1,1,self.psi,self.cache_bot,self.cache_top)
        unsigned_cx,vx,_ = self.get_grad_from_plqs(plq,config=config)
        return unsigned_cx * sign, vx
    def get_grad_from_plq(self,plq,inplace=False,config=None,cx=dict()):
        if self.dmrg:
            fn = self.get_grad_from_plq_dmrg
        else:
            fn = self.get_grad_from_plq_full
        return fn(plq,inplace=inplace,config=config,cx=cx)
    def get_grad_from_plq_full(self,plq,inplace=False,config=None,cx=dict()):
        # gradient
        vx = dict()
        for ((i0,j0),(x_bsz,y_bsz)),tn in plq.items():
            cij = cx.get((i0,j0),1.)
            for i in range(i0,i0+x_bsz):
                for j in range(j0,j0+y_bsz):
                    if (i,j) in vx:
                        continue
                    tn_ = tn if inplace else tn.copy()
                    vx[i,j] = self.site_grad(tn_,i,j) / cij 
        vx = self.dict2vec(vx)
        if config is not None:
            unsigned_cx = sum(cx.values()) / len(cx)
            self.store[config] = unsigned_cx
            self.store_grad[config] = vx
        return vx
    def get_grad_from_plq_dmrg(self,plq,inplace=False,config=None,cx=dict()):
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
            tn_ = tn if inplace else tn.copy()
            vx = self.site_grad(tn_,i,j) / cij
            vx = self.tensor2vec(vx,ix=self.ix)
            if config is not None:
                self.store[config] = cij
                self.store_grad[config] = vx
            return cx,vx 
        # ftn_plq doesn't exist due to contraction error
        start,stop = self.block_dict[self.ix]
        vx = np.zeros(stop-start)
        if config is not None:
            self.store[config] = 0. 
            self.store_grad[config] = vx 
        return vx
    def prob(self, config):
        """Calculate the probability of a configuration.
        """
        return self.unsigned_amplitude(config) ** 2
####################################################################################
# ham class 
####################################################################################
class Hamiltonian(ContractionEngine):
    def set_pepo_tsrs(self):
        ADD = np.zeros((2,)*3)
        ADD[0,0,0] = ADD[1,0,1] = ADD[1,1,0] = 1.
        I1 = np.eye(2)
        I2 = np.tensordot(I1,I1,axes=0)
        P0 = np.array([1.,0.])
        P1 = np.array([0.,1.])

        self.data_map.update({'I1':I1,'I2':I2,'P0':P0,'P1':P1,'ADD':ADD})
    def get_data(self,coeff,key='h1'):
        return np.tensordot(self.data_map['P0'],self.data_map['I2'],axes=0) \
             + np.tensordot(self.data_map['P1'],self.data_map[key],axes=0) * coeff
    def get_mpo(self,coeffs,L,key='h1'):
        tsrs = []
        pixs = [None] * L
        for ix,(ix1,ix2,coeff) in enumerate(coeffs):
            data = self.get_data(coeff,key=key)
            b1 = f'k{ix1}*' if pixs[ix1] is None else pixs[ix1]
            k1 = rand_uuid()
            pixs[ix1] = k1
            b2 = f'k{ix2}*' if pixs[ix2] is None else pixs[ix2]
            k2 = rand_uuid()
            pixs[ix2] = k2
            inds = f'v{ix}',b1,k1,b2,k2
            tags = f'L{ix1}',f'L{ix2}',key
            tsrs.append(Tensor(data=data,inds=inds,tags=tags))
        mpo = TensorNetwork(tsrs[::-1],virtual=True)
        mpo.reindex_({pix:f'k{ix}' for ix,pix in enumerate(pixs)})

        bixs = ['v0']+[rand_uuid() for ix in range(1,len(coeffs))]
        for ix in range(1,len(coeffs)):
            ix1,ix2,_ = coeffs[ix]
            tags = f'L{ix1}',f'L{ix2}','a'
            inds = bixs[ix],f'v{ix}',bixs[ix-1]
            mpo.add_tensor(Tensor(data=self.data_map['ADD'].copy(),inds=inds,tags=tags))
        
        # compress
        for ix in range(L-1):
            mpo.contract_tags(f'L{ix}',which='any',inplace=True)
            tid = tuple(mpo._get_tids_from_tags(f'L{ix}'))[0]
            tsr = mpo._pop_tensor(tid)

            rix = f'k{ix}*',f'k{ix}'
            if ix>0:
                rix = rix+(bix,)
            bix = rand_uuid()
            tl,tr = tensor_split(tsr,left_inds=None,right_inds=rix,bond_ind=bix,method='svd',get='tensors')
            tr.modify(tags=f'L{ix}')
            tl.drop_tags(tags=f'L{ix}')
            mpo.add_tensor(tr,virtual=True)
            mpo.add_tensor(tl,virtual=True)
        mpo[f'L{L-1}'].reindex_({bixs[-1]:'v'})
        return mpo
    def get_comb(self,mpos):
        # add mpo
        pepo = TensorNetwork([])
        L = mpos[0].num_tensors
        tag = f'L{L-1}'
        for ix1,mpo in enumerate(mpos):
            for ix2 in range(L):
                mpo[f'L{ix2}'].reindex_({f'k{ix2}':f'mpo{ix1}_k{ix2}',f'k{ix2}*':f'mpo{ix1}_k{ix2}*'})
            mpo[tag].reindex_({'v':f'v{ix1}','v*':f'v{ix1}*'})
            mpo.add_tag(f'mpo{ix1}')
            pepo.add_tensor_network(mpo,virtual=True,check_collisions=True)
        # add ADD
        nmpo = len(mpos)
        bix = ['v0']+[rand_uuid() for ix in range(1,nmpo)]
        for ix in range(1,nmpo):
            tags = f'mpo{ix}',tag 
            inds = bix[ix],f'v{ix}',bix[ix-1]
            pepo.add_tensor(Tensor(data=self.data_map['ADD'].copy(),inds=inds,tags=tags))
    
        # compress
        for ix in range(nmpo-1):
            pepo.contract_tags((f'mpo{ix}',tag),which='all',inplace=True)
            tid = tuple(pepo._get_tids_from_tags((f'mpo{ix}',tag),which='all'))[0]
            tsr = pepo._pop_tensor(tid)

            lix = bix[ix],
            tl,tr = tensor_split(tsr,left_inds=lix,method='svd',get='tensors')
            tl.modify(tags=(f'mpo{ix+1}',tag))
            pepo.add_tensor(tr,virtual=True)
            pepo.add_tensor(tl,virtual=True)
        pepo.contract_tags((f'mpo{nmpo-1}',tag),which='all',inplace=True)
        pepo[f'mpo{nmpo-1}',tag].reindex_({bix[-1]:'v'})
        return pepo
    def comb2PEPO(self,comb,peps,comb_type):
        if comb_type=='row':
            def get_comb_tag(i,j):
                return f'mpo{i}',f'L{j}'
            def get_comb_ind(i,j):
                return f'mpo{i}_k{j}'
        else:
            def get_comb_tag(i,j):
                return f'mpo{j}',f'L{i}'
            def get_comb_ind(i,j):
                return f'mpo{j}_k{i}'
        for i in range(peps.Lx):
            for j in range(peps.Ly):
                tsr = comb[get_comb_tag(i,j)]
                ind_old,ind_new = get_comb_ind(i,j),peps.site_ind(i,j)
                tsr.reindex_({ind_old:ind_new,ind_old+'*':ind_new+'*'}) 
                tsr.modify(tags=(peps.row_tag(i),peps.col_tag(j),peps.site_tag(i,j)))
        comb.view_like_(peps)
        return comb
    def combine(self,top,bot):
        Lx,Ly = bot.Lx,bot.Ly
        for i in range(Lx):
            for j in range(Ly): 
                pix = top.site_ind(i,j)
                top[i,j].reindex_({pix:pix+'_'})
                bot[i,j].reindex_({pix+'*':pix+'_'})
        top[Lx-1,Ly-1].reindex_({'v':'vt'})
        bot[Lx-1,Ly-1].reindex_({'v':'vb'})
    
        pepo = bot
        pepo.add_tensor_network(top,virtual=True)
        tags = pepo.site_tag(Lx-1,Ly-1)
        pepo.add_tensor(
            Tensor(data=self.data_map['ADD'].copy(),inds=('v','vt','vb'),tags=tags),virtual=True) 
        for i in range(Lx):
            for j in range(Ly): 
                pepo.contract_tags(pepo.site_tag(i,j),inplace=True) 
        return pepo 
    def trace_virtual(self,pepo):
        tags = pepo.site_tag(pepo.Lx-1,pepo.Ly-1)
        pepo.add_tensor(Tensor(data=self.data_map['P1'].copy(),inds=('v',),tags=tags),virtual=True)
        pepo.contract_tags(tags,inplace=True)
        pepo.add_tag('BRA')
        return pepo

    def __init__(self,Lx,Ly,**kwargs):
        self.Lx,self.Ly = Lx,Ly
        self.hess = kwargs.get('hess','comb')
        self.dmrg = kwargs.get('dmrg',False)
        self.discard = kwargs.get('discard',None)
        self.chi_min = kwargs.get('chi_min',None)
        self.chi_max = kwargs.get('chi_max',None)
        self.contract_opts = kwargs.get('contract_opts',dict())
        self.cache_top = dict()
        self.cache_bot = dict()
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def update_cache(self,ix):
        self.cache_bot,self.cache_top = self.cache_update(
            self.cache_bot,self.cache_top,ix,self.Lx,self.Ly)
        #self.cache_bot = dict()
        #self.cache_top = dict()
    def initialize_pepo(self,peps):
        if self.hess=='single':
            return
        self.set_pepo_tsrs()

        pepos = []
        for L,nmpo,typ in [(self.Ly,self.Lx,'row'),(self.Lx,self.Ly,'col')]:
            coeffs = self.get_coeffs(L)
            mpo = self.get_mpo(coeffs,L,key=self.key)
            mpos = [mpo.copy() for i in range(nmpo)]
            pepo = self.get_comb(mpos)
            pepo = self.comb2PEPO(pepo,peps,typ)
            pepos.append(pepo)
            #print(typ,pepo)
        if self.hess=='comb':
            self.pepo_row = self.trace_virtual(pepos[0]) 
            self.pepo_col = self.trace_virtual(pepos[1]) 
            #print('row',self.pepo_row)
            #print('col',self.pepo_col)
        elif self.hess=='pepo':
            pepo = self.combine(*pepos)
            self.pepo = self.trace_virtual(pepo) 
            #print(self.pepo)
        else:
            raise NotImplementedError
        #exit()
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self.data_map[self.key].copy()
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return Tensor(data=data,inds=inds,tags=tags) 
    def pair_energy(self,tn,config,site1,site2,cx=None):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2] 
        cx = tn.copy().contract() if cx is None else cx
        if not self.pair_valid(i1,i2): # term vanishes 
            return 0.,cx
        kixs = [tn.site_ind(*site) for site in [site1,site2]]
        bixs = [kix+'*' for kix in kixs]
        for site,kix,bix in zip([site1,site2],kixs,bixs):
            tn[tn.site_tag(*site),'BRA'].reindex_({kix:bix})
        tn.add_tensor(self.pair_tensor(bixs,kixs),virtual=True)
        try:
            return self.pair_coeff(site1,site2) * tn.contract() / cx,cx
        except (ValueError,IndexError):
            return 0.,cx
    def pair_energies(self,config,plq,x_bsz,y_bsz,inplace=False,cx=dict()):
        e = 0.
        for ((i0,j0),(x_bsz_,y_bsz_)),tn in plq.items():
            for i in range(i0,i0+x_bsz_-x_bsz+1):
                for j in range(j0,j0+y_bsz_-y_bsz+1):
                    site1,site2 = (i,j),(i+x_bsz-1,j+y_bsz-1)
                    if site1 in cx:
                        cij = cx[site1]
                    elif site2 in cx:
                        cij = cx[site2]
                    else:
                        cij = None
                    tn_ = tn if inplace else tn.copy()
                    eij,cij = self.pair_energy(tn_,config,site1,site2,cij)
                    e += eij
                    cx[site1] = cij
                    cx[site2] = cij
        return e,cx
    def contraction_error(self,cx):
        ls = [] 
        for i in range(self.Lx):
            for j in range(self.Ly):
                ls.append(cx.get((i,j),0.))
        ls = np.array(ls)
        if np.linalg.norm(ls) < 1e-10:
            return 0.
        err = np.std(ls)
        mean = ls.sum() / len(cx)
        return np.fabs(err/mean)
    def _compute_Hv_comb_full(self,config,amplitude_factory,unsigned_cx):

        max_bond = self.chi_min
        t0 = time.time()
        norm = make_norm(self.pepo_row,amplitude_factory.psi,config=config)
        plq = norm._compute_plaquette_environments_row_first(1,1,
                  max_bond=max_bond,**self.contract_opts)
        plq = self.complete_plq(plq,norm)
        vals = {key:ftn_plq.copy().contract() for key,ftn_plq in plq.items()}
        err = self.contraction_error(vals,1,1)
        trr = time.time()-t0

        t0 = time.time()
        norm = make_norm(self.pepo_row,amplitude_factory.psi,config=config)
        plq = norm._compute_plaquette_environments_col_first(1,1,
                  max_bond=max_bond,**self.contract_opts)
        plq = self.complete_plq(plq,norm)
        vals = {key:ftn_plq.copy().contract() for key,ftn_plq in plq.items()}
        erc = self.contraction_error(vals,1,1)
        trc = time.time()-t0

        t0 = time.time()
        norm = make_norm(self.pepo_col,amplitude_factory.psi,config=config)
        plq = norm._compute_plaquette_environments_row_first(1,1,
                  max_bond=max_bond,**self.contract_opts)
        plq = self.complete_plq(plq,norm)
        vals = {key:ftn_plq.copy().contract() for key,ftn_plq in plq.items()}
        ecr = self.contraction_error(vals,1,1)
        tcr = time.time()-t0

        t0 = time.time()
        norm = make_norm(self.pepo_col,amplitude_factory.psi,config=config)
        plq = norm._compute_plaquette_environments_col_first(1,1,
                  max_bond=max_bond,**self.contract_opts)
        plq = self.complete_plq(plq,norm)
        vals = {key:ftn_plq.copy().contract() for key,ftn_plq in plq.items()}
        ecc = self.contraction_error(vals,1,1)
        tcc = time.time()-t0
        print(f'RANK={RANK},max_bond={max_bond},rr={(err,trr)},rc={(erc,trc)},cr={(ecr,tcr)},cc={(ecc,tcc)}')

    def compute_Hv_comb_full(self,config,amplitude_factory,unsigned_cx):
        sign = amplitude_factory.compute_config_sign(config) 
        Hvx_row,err_row = self.compute_Hv_pepo_full(config,amplitude_factory,unsigned_cx,
                                                    pepo=self.pepo_row,sign=sign,first='col') 
        if Hvx_row is None:
            return None,None
        Hvx_col,err_col = self.compute_Hv_pepo_full(config,amplitude_factory,unsigned_cx,
                                                    pepo=self.pepo_col,sign=sign,first='row') 
        if Hvx_col is None:
            return None,None
        return Hvx_row+Hvx_col, max(err_row,err_col)
    def make_norm(self,pepo,peps,config=None):
        bra = pepo.copy()
        if config is not None:
            for ix,ci in reversed(list(enumerate(config))):
                i,j = self.flat2site(ix)
                tsr = self.get_bra_tsr(peps,ci,i,j,append='*')
                bra.add_tensor(tsr,virtual=True) 
            for i in range(peps.Lx):
                for j in range(peps.Ly):
                    bra.contract_tags(bra.site_tag(i,j),inplace=True)
        norm = peps.copy() 
        norm.add_tensor_network(bra,virtual=True)
        return norm
    def complete_plq(self,plq,norm):
        for key in plq.keys():
            (i0,j0),(x_bsz,y_bsz) = key 
            tn = plq[key]
            for i in range(i0,i0+x_bsz):
                for j in range(j0,j0+y_bsz):
                    tn.add_tensor_network(norm.select(norm.site_tag(i,j)).copy())
        return plq
    def compute_Hv_pepo_full(self,config,amplitude_factory,unsigned_cx,pepo=None,sign=None,first=None):
        max_bond = self.chi_min
        pepo = self.pepo if pepo is None else pepo
        sign = amplitude_factory.compute_config_sign(config) if sign is None else sign 
        #t0 = time.time()
        while True:
            norm = self.make_norm(pepo,amplitude_factory.psi,config=config)
            if first is None:
                plq = norm.compute_plaquette_environments(x_bsz=1,y_bsz=1,max_bond=max_bond,
                                                          **self.contract_opts) 
            elif first=='row':
                plq = norm._compute_plaquette_environments_row_first(x_bsz=1,y_bsz=1,max_bond=max_bond,
                                                          **self.contract_opts) 
            elif first=='col':
                plq = norm._compute_plaquette_environments_col_first(x_bsz=1,y_bsz=1,max_bond=max_bond,
                                                          **self.contract_opts) 
            else:
                raise NotImplementedError
            plq = self.complete_plq(plq,norm)
            vals = {site:ftn_plq.copy().contract() for (site,_),ftn_plq in plq.items()}
            err = self.contraction_error(vals)
            if self.discard is None:
                break
            if err < self.discard:
                break
            max_bond *= 2
            if max_bond > self.chi_max:
                return None,None
        #print(f'RANK={RANK},max_bond={max_bond},err={err},time={time.time()-t0}')
        for key in plq:
            plq[key].view_like_(norm)
        Hvx = amplitude_factory.get_grad_from_plq(plq,inplace=True)
        return Hvx / (sign * unsigned_cx), err
    def compute_Hv_comb_dmrg(self,config,amplitude_factory,unsigned_cx):
        Hvx_row,_ = self.compute_Hv_pepo_dmrg(config,amplitude_factory,unsigned_cx,
                                                  pepo=self.pepo_row) 
        Hvx_col,_ = self.compute_Hv_pepo_dmrg(config,amplitude_factory,unsigned_cx,
                                                  pepo=self.pepo_col) 
        return Hvx_row+Hvx_col, None 
    def get_3row_ftn(self,norm,config,cache_bot,cache_top,i,**compress_opts):
        try:
            norm.reorder('row',layer_tags=('KET','BRA'),inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        tn = self.get_mid_env(i,norm,config,append='*')
        if i>0:
            other = tn
            tn = self.get_all_bot_envs(norm,config,cache_bot,imax=i-1,append='*',**compress_opts).copy()
            tn.add_tensor_network(other,virtual=True) 
        if i<norm.Lx-1:
            top = self.get_all_top_envs(norm,config,cache_top,imin=i+1,append='*',**compress_opts)
            tn.add_tensor_network(top.copy(),virtual=True) 
        return tn 
    def get_3col_ftn(self,norm,j,**compress_opts):
        try:
            norm.reorder('col',layer_tags=('KET','BRA'),inplace=True)
        except (NotImplementedError,AttributeError):
            pass
        for j_ in range(1,j):
            norm.contract_boundary_from_left_(yrange=(j_-1,j_),**compress_opts)
        for j_ in range(norm.Ly-2,j,-1):
            norm.contract_boundary_from_right_(yrange=(j_,j_+1),**compress_opts)
        return norm
    def compute_Hv_pepo_dmrg(self,config,amplitude_factory,unsigned_cx,pepo=None,sign=None):
        ix = amplitude_factory.ix
        i,j = self.flat2site(ix)
        pepo = self.pepo if pepo is None else pepo
        if self.Lx < self.Ly:
            norm = self.make_norm(self.pepo,amplitude_factory.psi,config=config)
            tn = self.get_3col_ftn(norm,j,**self.contract_opts)
            sign = amplitude_factory.compute_config_sign(config) if sign is None else sign 
        else:
            norm = self.make_norm(self.pepo,amplitude_factory.psi)
            tn = self.get_3row_ftn(norm,config,self.cache_bot,self.cache_top,i,**self.contract_opts) 
            sign = 1.
        Hvx = self.site_grad(tn,i,j)
        Hvx = amplitude_factory.tensor2vector(Hvx,ix=ix) 
        return Hvx / (sign * unsigned_cx), None
    def config_parity(self,config=None,ix1=None,ix2=None):
        return 0
    def pair_Hv(self,config,peps,grad_site,site1,site2,cache_top,cache_bot,sign_fn=None,**compress_opts):
        ix1,ix2 = self.flatten(*site1),self.flatten(*site2)
        i1,i2 = config[ix1],config[ix2]
        if not self.pair_valid(i1,i2): # term vanishes 
            return None 
        imin = min(grad_site[0],site1[0]) 
        imax = max(grad_site[0],site2[0]) 
        top = None if imax==peps.Lx-1 else cache_top[config[(imax+1)*peps.Ly:]]
        bot = None if imin==0 else cache_bot[config[:imin*peps.Ly]]
        parity = self.config_parity(config,ix1,ix2)
        sign = (-1)**parity

        Hvx = None
        for i1_new,i2_new,coeff in self.pair_terms(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            sign_term = 1 if sign_fn is None else sign_fn(config_new)
    
            bot_term = None if bot is None else bot.copy()
            for i in range(imin,grad_site[0]):
                row = self.get_mid_env(i,peps,config_new,append='')
                bot_term = self.get_bot_env(i,row,bot_term,config_new,cache_bot,**compress_opts)
            if imin > 0 and bot_term is None:
                    continue
    
            top_term = None if top is None else top.copy()
            for i in range(imax,grad_site[0],-1):
                row = self.get_mid_env(i,peps,config_new,append='')
                top_term = self.get_top_env(i,row,top_term,config_new,cache_top,**compress_opts)
            if imax < peps.Lx-1 and top_term is None:
                continue

            tn = self.get_mid_env(grad_site[0],peps,config_new,append='')
            if tn is None:
                continue
            if bot_term is not None:
                other = tn
                tn = bot_term.copy() 
                tn.add_tensor_network(other,virtual=True)
            if top_term is not None:
                tn.add_tensor_network(top_term.copy(),virtual=True)
            tn.view_like_(peps)
            try:
                Hvx_term = self.site_grad(tn,*grad_site) * (sign * sign_term * coeff)
                if Hvx is None:
                    Hvx =  Hvx_term
                else:
                    Hvx = Hvx + Hvx_term
            except (IndexError,ValueError):
                continue
        return Hvx
    def pair_Hvs(self,config,amplitude_factory,x_bsz,y_bsz,sign_fn=None,ix=None):
        cache_top = amplitude_factory.cache_top
        cache_bot = amplitude_factory.cache_bot
        ix = amplitude_factory.ix if ix is None else ix
        grad_site = self.flat2site(ix)
        fpeps = amplitude_factory.psi
        compress_opts = amplitude_factory.contract_opts
        Hvx = None 
        for i in range(self.Lx+1-x_bsz):
            for j in range(self.Ly+1-y_bsz):
                site1,site2 = (i,j),(i+x_bsz-1,j+y_bsz-1)
                Hvx_ = self.pair_Hv(config,fpeps,grad_site,site1,site2,cache_top,cache_bot,
                                    sign_fn=sign_fn,**compress_opts)
                if Hvx_ is not None:
                    Hvx_ = Hvx_ * self.pair_coeff(site1,site2)
                    if Hvx is None:
                        Hvx = Hvx_
                    else:
                        Hvx = Hvx + Hvx_
        if Hvx is None:
            start,stop = amplitude_factory.block_dict[ix]
            return np.zeros(stop-start)
        Hvx = amplitude_factory.tensor2vec(Hvx,ix=ix)
        if sign_fn is not None:
            sign = sign_fn(config) 
            Hvx /= sign
        return Hvx
    def compute_Hv_single_dmrg(self,config,amplitude_factory,unsigned_cx,ix=None):
        sign_fn = amplitude_factory.compute_config_sign
        Hvx_h = 0. 
        for bsz in range(2,self.bsz+1):
            Hvx_h += self.pair_Hvs(config,amplitude_factory,1,bsz,sign_fn=sign_fn,ix=ix)
        Hvx_v = 0. 
        for bsz in range(2,self.bsz+1):
            Hvx_v += self.pair_Hvs(config,amplitude_factory,bsz,1,sign_fn=sign_fn,ix=ix)
        Hvx = (Hvx_h + Hvx_v) / unsigned_cx
        return Hvx, None
    def compute_Hv_single_full(self,config,amplitude_factory,unsigned_cx):
        nsite = len(amplitude_factory.constructors)
        Hvx = [None] * nsite 
        for ix in range(nsite):
            Hvx[ix],_ = self.compute_Hv_single_dmrg(config,amplitude_factory,unsigned_cx,ix=ix)
        return np.concatenate(Hvx,axis=0), None
    def compute_Hv(self,config,amplitude_factory,unsigned_cx):
        if self.hess=='single':
            if self.dmrg:
                fn = self.compute_Hv_single_dmrg
            else:
                fn = self.compute_Hv_single_full
        elif self.hess=='comb':
            if self.dmrg:
                fn = self.compute_Hv_comb_dmrg
            else:
                fn = self.compute_Hv_comb_full
        elif self.hess=='pepo':
            if self.dmrg:
                fn = self.compute_Hv_pepo_dmrg
            else:
                fn = self.compute_Hv_pepo_full
        else:
            raise NotImplementedError
        return fn(config,amplitude_factory,unsigned_cx) 
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        amplitude_factory.get_all_benvs(config,x_bsz=1) 
        peps = amplitude_factory.psi
        cache_bot = amplitude_factory.cache_bot
        cache_top = amplitude_factory.cache_top

        plq_h = amplitude_factory.get_plq_from_benvs(config,1,self.bsz,peps,cache_bot,cache_top)
        eh = dict()
        ch = dict() 
        for bsz in range(2,self.bsz+1):
            eh[bsz],ch = self.pair_energies(config,plq_h,x_bsz=1,y_bsz=bsz,inplace=False,cx=ch) 
        eh = sum(eh.values())
        err = self.contraction_error(ch) # contraction error
        if self.discard is not None: # discard sample if contraction error too large
            if err > self.discard: 
                return (None,) * 6 

        plq_v = amplitude_factory.get_plq_from_benvs(config,self.bsz,1,peps,cache_bot,cache_top)
        ev = dict()
        cv = dict() 
        for bsz in range(2,self.bsz+1):
            ev[bsz],cv = self.pair_energies(config,plq_v,x_bsz=bsz,y_bsz=1,inplace=False,cx=cv) 
        ev = sum(ev.values())
        unsigned_cx = sum(cv.values()) / len(cv)

        eu = self.compute_local_energy_eigen(config)
        ex = ev + eh + eu
        if not compute_v:
            return unsigned_cx,ex,None,None,err,None

        vx = amplitude_factory.get_grad_from_plq(plq_v,inplace=False,config=config,cx=cv) 
        if not compute_Hv:
            return unsigned_cx,ex,vx,None,err,None

        Hvx,err2 = self.compute_Hv(config,amplitude_factory,unsigned_cx)
        if Hvx is None:
            return (None,) * 6
        Hvx += eu * vx
        return unsigned_cx,ex,vx,Hvx,err,err2
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
        self.bsz = 2
        self.data_map = dict()
        self.set_gate()
    def set_gate(self):
        data = get_gate2((self.Jx,self.Jy,0.),to_bk=False)
        self.key = 'Jxy'
        self.data_map[self.key] = data
    def get_coeffs(self,L):
        coeffs = []
        for i in range(L):
            if i+1 < L:
                coeffs.append((i,i+1,1.))
        return coeffs
    def pair_coeff(self,site1,site2):
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
####################################################################################
# sampler 
####################################################################################
class ExchangeSampler(ContractionEngine):
    def __init__(self,Lx,Ly,seed=None,burn_in=0):
        self.Lx = Lx
        self.Ly = Ly
        self.nsite = self.Lx * self.Ly

        self.rng = np.random.default_rng(seed)
        self.exact = False
        self.dense = False
        self.burn_in = burn_in 
        self.amplitude_factory = None
        self.alternate = False # True if autodiff else False
    def initialize(self,config,thresh=1e-10):
        # randomly choses the initial sweep direction
        self.sweep_row_dir = self.rng.choice([-1,1]) 
        # setup to compute all opposite envs for initial sweep
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 
        self.px = self.amplitude_factory.prob(config)
        self.config = config
        # force to initialize with a better config
        #print(self.px)
        #exit()
        if self.px < thresh:
            raise ValueError 
    def preprocess(self,config):
        self._burn_in(config)
    def _burn_in(self,config,batchsize=None):
        batchsize = self.burn_in if batchsize is None else batchsize
        self.initialize(config)
        if batchsize==0:
            return
        t0 = time.time()
        _alternate = self.alternate 
        self.alternate = True # always do alternate sweep in burn in 
        for n in range(batchsize):
            self.config,self.omega = self.sample()
        if RANK==SIZE-1:
            print('\tburn in time=',time.time()-t0)
        self.alternate = _alternate
        #print(f'\tRANK={RANK},burn in time={time.time()-t0},namps={len(self.amplitude_factory.store)}')
    def flatten(self,i,j):
        return flatten(i,j,self.Ly)
    def flat2site(self,ix):
        return flat2site(ix,self.Lx,self.Ly)
    def new_pair(self,i1,i2):
        return i2,i1
    def get_pairs(self,i,j):
        bonds_map = {'l':((i,j),(i+1,j)),
                     'd':((i,j),(i,j+1)),
                     'r':((i,j+1),(i+1,j+1)),
                     'u':((i+1,j),(i+1,j+1)),
                     'x':((i,j),(i+1,j+1)),
                     'y':((i,j+1),(i+1,j))}
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
                py = 0.
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
        peps = self.amplitude_factory.psi
        compress_opts = self.amplitude_factory.contract_opts
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
            env_bot = self.get_bot_env(i,row1_new,env_bot,self.config,cache_bot,**compress_opts)
            row1 = row2_new
    def sweep_row_backward(self):
        peps = self.amplitude_factory.psi
        compress_opts = self.amplitude_factory.contract_opts
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
            env_top = self.get_top_env(i,row1_new,env_top,tuple(self.config),cache_top,**compress_opts)
            row1 = row2_new
    def sample(self):
        #self.sweep_col_dir = -1 # randomly choses the col sweep direction
        self.sweep_col_dir = self.rng.choice([-1,1]) # randomly choses the col sweep direction
        if self.sweep_row_dir == 1:
            self.sweep_row_forward()
        else:
            self.sweep_row_backward()
        # setup to compute all opposite env for gradient
        self.amplitude_factory.update_scheme(-self.sweep_row_dir) 

        if self.alternate: # for burn in 
            self.sweep_row_dir *= -1
        else: # actual sampling 
            self.sweep_row_dir = self.rng.choice([-1,1]) 
        return self.config,self.px
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
    def preprocess(self):
        self.compute_dense_prob()
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
        batchsize,remain = ntotal//SIZE,ntotal%SIZE
        L = SIZE-remain
        if RANK<L:
            start = RANK*batchsize
            stop = start+batchsize
        else:
            start = (batchsize+1)*RANK-L
            stop = start+batchsize+1
        self.nonzeros = nonzeros[start:stop]
        self.amplitude_factory.update_scheme(0)
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

