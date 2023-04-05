from .fermion_2d_vmc import AmplitudeFactory2D as AmplitudeFactory
from .fermion_2d_vmc import Hubbard2D as Hubbard
from .fermion_2d_vmc import flat2site,site_grad,get_bra_tsr
from .fermion_core import FermionTensorNetwork 
import numpy as np
#####################################################################################
# double layer cache
#####################################################################################
def get_bot_env(i,norm,env_prev,config,cache,**compress_opts):
    # contract mid env for row i with prev bot env 
    key = config[:(i+1)*norm.Ly]
    if key in cache: # reusable
        return cache[key] 
    row = norm.select(norm.row_tag(i)).copy()
    if i==0:
        cache[key] = row 
        return row
    if env_prev is None:
        cache[key] = None 
        return None
    ftn = FermionTensorNetwork([env_prev,row],virtual=False).view_like_(row)
    try:
        ftn.contract_boundary_from_bottom_(xrange=(i-1,i),yrange=(0,norm.Ly-1),layer_tags=('KET','BRA'),**compress_opts)
    except (ValueError,IndexError):
        ftn = None
    cache[key] = ftn
    return ftn 
def get_top_env(i,norm,env_prev,config,cache,**compress_opts):
    # contract mid env for row i with prev top env 
    key = config[i*norm.Ly:]
    if key in cache: # reusable
        return cache[key]
    row = norm.select(norm.row_tag(i)).copy()
    if i==row.Lx-1:
        cache[key] = row 
        return row
    if env_prev is None:
        cache[key] = None 
        return None
    ftn = FermionTensorNetwork([row,env_prev],virtual=False).view_like_(row)
    try:
        ftn.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,norm.Ly-1),layer_tags=('KET','BRA'),**compress_opts)
    except (ValueError,IndexError):
        ftn = None
    cache[key] = ftn
    return ftn 
def get_all_bot_envs(norm,config,cache_bot,imax,**compress_opts):
    env_prev = None
    for i in range(imax+1):
         env_prev = get_bot_env(i,norm,env_prev,config,cache_bot,**compress_opts)
    return env_prev
def get_all_top_envs(norm,config,cache_top,imin,**compress_opts):
    env_prev = None
    for i in range(norm.Lx-1,imin-1,-1):
         env_prev = get_top_env(i,norm,env_prev,config,cache_top,**compress_opts)
    return env_prev
def get_3col_ftn(norm,config,cache_bot,cache_top,row_ix,**compress_opts):
    norm.reorder('row',inplace=True)
    ls = []
    if row_ix>0:
        bot = get_all_bot_envs(norm,config,cache_bot,row_ix-1,**compress_opts)
        ls.append(bot)
    ls.append(norm.select(norm.row_tag(row_ix)))
    if row_ix<norm.Lx-1:
        top = get_all_top_envs(norm,config,cache_top,row_ix+1,**compress_opts)
        ls.append(top)
    ftn = FermionTensorNetwork(ls,virtual=False).view_like_(norm)
    return ftn 
######################################################################################
# DMRG cahce update
######################################################################################
def cache_update(cache_bot,cache_top,ix,Lx,Ly):
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
class AmplitudeFactory2D(AmplitudeFactory):
    def _set_psi(self,psi):
        self.psi = psi
        self.store = dict()
        self.store_grad = dict()

        self.compute_bot = True
        self.compute_top = True
        if self.ix is None:
            self.cache_bot = dict()
            self.cache_top = dict()
            return
        self.cache_bot,self.cache_top = cache_update(self.cache_bot,self.cache_top,self.ix,self.Lx,self.Ly)
    def get_grad_from_plq(self,plq,config=None,compute_cx=True):
        i,j = self.flat2site(self.ix)
        _,(x_bsz,y_bsz) = list(plq.keys())[0]
        i0 = min(self.Lx-x_bsz,i)
        j0 = min(self.Ly-y_bsz,j)
        ftn_plq = plq[(i0,j0),(x_bsz,y_bsz)]
        cx = ftn_plq.copy().contract() if compute_cx else 1.
        vx = site_grad(ftn_plq.copy(),i,j) / cx 
        cons = self.constructors[self.ix][0]
        vx = cons.tensor_to_vector(vx) 
        if config is not None:
            self.store[config] = cx
            self.store_grad[config] = vx
        return vx 
class Hubbard2D(Hubbard):
    def update_cache(self,ix):
        self.cache_bot,self.cache_top = cache_update(self.cache_bot,self.cache_top,ix,self.Lx,self.Ly)
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        amplitude_factory.get_all_benvs(config,x_bsz=1) 
        # form all (1,2),(2,1) plqs
        plq12 = amplitude_factory.get_plq_from_benvs(config,x_bsz=1,y_bsz=2)
        plq21 = amplitude_factory.get_plq_from_benvs(config,x_bsz=2,y_bsz=1)
        # get gradient form plq12
        cx12 = {key:ftn_plq.copy().contract() for (key,_),ftn_plq in plq12.items()}
        unsigned_cx = sum(cx12.values()) / len(cx12)
        vx = None
        if compute_v:
            vx = amplitude_factory.get_grad_from_plq(plq12,config=config) 
        # get h/v bonds
        eh = self.nn(config,plq12,x_bsz=1,y_bsz=2,inplace=True,cx=cx12) 
        ev = self.nn(config,plq21,x_bsz=2,y_bsz=1,inplace=True,cx=None) 
        # onsite terms
        config = np.array(config,dtype=int)
        eu = self.u*len(config[config==3])

        ex = eh+ev+eu
        if not compute_Hv: 
            return unsigned_cx,ex,vx,None 
        sign = amplitude_factory.compute_config_sign(tuple(config)) 
        Hvx = self.compute_Hv_hop(tuple(config),amplitude_factory)
        Hvx /= sign * unsigned_cx
        Hvx += eu * vx
        return unsigned_cx,ex,vx,Hvx
    def compute_Hv_hop(self,config,amplitude_factory):
        # make bra
        bra = self.pepo.copy()
        fpeps = amplitude_factory.psi.copy()
        for ix,ci in reversed(list(enumerate(config))):
            i,j = self.flat2site(ix)
            tsr = get_bra_tsr(fpeps,ci,i,j)
            tsr.reindex_({f'k{i},{j}':f'k{i},{j}*'})
            bra.add_tensor(tsr,virtual=True) 
        for i in range(self.Lx):
            for j in range(self.Ly):
                bra.contract_tags(bra.site_tag(i,j),inplace=True)
        norm = fpeps 
        norm.add_tensor_network(bra,virtual=True)

        ix = amplitude_factory.ix
        i,j = self.flat2site(ix)
        if self.Lx > self.Ly:
            norm.reorder('col',inplace=True)
            left = norm.compute_environments('left',xrange=(0,self.Lx-1),yrange=(0,j-1),layer_tags=('KET','BRA'),**self.contract_opts)
            left = left['left',j]
            right = norm.compute_environments('right',xrange=(0,self.Lx-1),yrange=(j+1,self.Ly-1),layer_tags=('KET','BRA'),**self.contract_opts)
            left = left['right',j]
            mid = norm.select(norm.col_tag(j)).copy()
            ftn = FermionTensorNetwork([left,mid,right],virtual=True).view_like_(norm)
        else:
            ftn = get_3col_ftn(norm,config,self.cache_bot,self.cache_top,i,**self.contract_opts) 
        Hvx = site_grad(ftn,i,j)
        cons = amplitude_factory.constructors[ix][0]
        Hvx = cons.tensor_to_vector(Hvx) 
        return Hvx
