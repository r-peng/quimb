from .fermion_2d_vmc import Hubbard2D as Hubbard
from .fermion_2d_vmc import (
    flat2site,
    site_grad,
    get_bra_tsr,
    get_mid_env,
    get_all_bot_envs,
    get_all_top_envs,
)
from .fermion_core import FermionTensorNetwork 
import numpy as np
def hop(i1,i2):
    n1,n2 = pn_map[i1],pn_map[i2]
    nsum,ndiff = n1+n2,abs(n1-n2)
    if ndiff==1:
        sign = 1 if nsum==1 else -1
        return [(i2,i1,sign)]
    if ndiff==2:
        return [(1,2,-1),(2,1,1)] 
    if ndiff==0:
        sign = i1-i2
        return [(0,3,sign),(3,0,sign)]
class Hubbard2D(Hubbard):
    def initialize_pepo(self,fpeps=None):
        pass
    def compute_local_energy(self,config,amplitude_factory,compute_v=True,compute_Hv=False):
        amplitude_factory.get_all_benvs(config,x_bsz=1) 
        # form all (1,2),(2,1) plqs
        plq12 = amplitude_factory.get_plq_from_benvs(config,x_bsz=1,y_bsz=2)
        plq21 = amplitude_factory.get_plq_from_benvs(config,x_bsz=2,y_bsz=1)
        # get gradient form plq12
        vx = None
        if compute_v:
            unsigned_cx,vx = amplitude_factory.get_grad_from_plq(plq12,config=config) 
        # get h/v bonds
        eh = self.nn(config,plq12,x_bsz=1,y_bsz=2,inplace=True,cx=None) 
        ev = self.nn(config,plq21,x_bsz=2,y_bsz=1,inplace=True,cx=None) 
        # onsite terms
        config = np.array(config,dtype=int)
        eu = self.u*len(config[config==3])

        ex = eh+ev+eu
        if not compute_Hv: 
            return unsigned_cx,ex,vx,None 
        #sign = amplitude_factory.compute_config_sign(tuple(config)) 
        sign = 1
        Hvx = self.compute_Hv_hop(tuple(config),amplitude_factory)
        Hvx /= sign * unsigned_cx
        Hvx += eu * vx
        return unsigned_cx,ex,vx,Hvx
    def compute_Hv_hop(self,config,amplitude_factory):
        cache_top = amplitude_factory.cache_top
        cache_bot = amplitude_factory.cache_bot
        ix = amplitude_factory.ix
        grad_site = self.flat2site(ix)
        fpeps = amplitude_factory.psi
        compress_opts = amplitude_facory.contract_opts
        cons = amplitude_factory.constructors[ix][0]
        Hvx = 0.
        # hbonds
        for i in range(self.Lx):
            for j in range(self.Ly-1):
                site1,site2 = (i,j),(i,j+1)
                if i>grad_site[0]:
                    Hvx_term = self.Hv_hop_upper(config,fpeps,grad_site,site1,site2,cache_top,cache_bot,**compress_opts)
                else:
                    Hvx_term = self.Hv_hop_lower(config,fpeps,grad_site,site1,site2,cache_top,cache_bot,**compress_opts)
                if Hvx_term is not None:
                    Hvx += cons.tensor_to_vector(Hvx_term)
        # vbonds
        for i in range(self.Lx-1):
            for j in range(self.Ly):
                site1,site2 = (i,j),(i+1,j)
                if i>grad_site[0]:
                    Hvx_term = self.Hv_hop_upper(config,fpeps,grad_site,site1,site2,cache_top,cache_bot,**compress_opts)
                else:
                    Hvx_term = self.Hv_hop_lower(config,fpeps,grad_site,site1,site2,cache_top,cache_bot,**compress_opts)
                if Hvx_term is not None:
                    Hvx += cons.tensor_to_vector(Hvx_term)
        return Hvx
    def Hv_hop_lower(self,config,fpeps,grad_site,site1,site2,cache_top,cache_bot,**compress_opts):
        ix1,ix2 = self.flatten(i1,j1),self.flatten(i2,j2)
        i1,i2 = config[ix1],config[ix2]
        if i1==i2:
            return None 

        top = FermionTensorNetwork([]) if grad_site[0]==self.Lx-1 else \
              cache_top[config[(grad_site[0]+1)*self.Ly:]]
        if top is None:
            return None
        mid = get_mid_env(grad_site[0],fpeps,config)

        env_prev = None if site1[0]==0 else cache_bot[config[:site1[0]*self.Ly]]
        Hvx = None
        coeff = self.hop_coeff(site1,site2)
        for i1_new,i2_new,hop_sign in hop(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            bot = None if env_prev is None else env_prev.copy()
            for i in range(site1[0],grad_site[0]):
                row = get_mid_env(i,fpeps,config_new) 
                bot = get_bot_env(i,row,bot,config_new,cache_bot,layer_tags=None,**compress_opts)
            if bot is None:
                continue
            ftn = FermionTensorNetwork([bot,mid,top],virtual=False).view_like_(fpeps) 
            try:
                Hvx_term = site_grad(ftn,*site_grad) * (hop_sign * coeff)
                if Hvx is None:
                    Hvx =  Hvx_term
                else:
                    Hvx = Hvx + Hvx_term
            except (IndexError,ValueError):
                continue
        return Hvx
    def Hv_hop_upper(self,config,fpeps,grad_site,site1,site2,cache_top,cache_bot,**compress_opts):
        ix1,ix2 = self.flatten(i1,j1),self.flatten(i2,j2)
        i1,i2 = config[ix1],config[ix2]
        if i1==i2:
            return None 

        bot = FermionTensorNetwork([]) if grad_site[0]==0 else \
              cache_bot[config[:grad_site[0]*self.Ly]]
        if bot is None:
            return None
        mid = get_mid_env(grad_site[0],fpeps,config)

        env_prev = None if site2[0]==self.Lx-1 else cache_top[config[(site2[0]+1)*self.Ly:]]
        Hvx = None
        coeff = self.hop_coeff(site1,site2)
        for i1_new,i2_new,hop_sign in hop(i1,i2):
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)
            top = None if env_prev is None else env_prev.copy()
            for i in range(site2[0],grad_site[0],-1):
                row = get_mid_env(i,fpeps,config_new) 
                top = get_top_env(i,row,top,config_new,cache_top,layer_tags=None,**compress_opts)
            if top is None:
                continue
            ftn = FermionTensorNetwork([bot,mid,top],virtual=False).view_like_(fpeps) 
            try:
                Hvx_term = site_grad(ftn,*site_grad) * (hop_sign * coeff)
                if Hvx is None:
                    Hvx = Hvx_term
                else:
                    Hvx = Hvx + Hvx_term
            except (IndexError,ValueError):
                continue
        return Hvx
