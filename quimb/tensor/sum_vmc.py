import numpy as np
import itertools
import scipy
from .product_vmc import (
    #tensor2backend,
    #safe_contract,
    #contraction_error,
    ProductAmplitudeFactory,
)
from .tensor_2d_vmc import ExchangeSampler2D
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
class SumAmplitudeFactory(ProductAmplitudeFactory):
    def get_grad_deterministic(self,config,unsigned=False):
        raise NotImplementedError
        cx = [None] * self.naf
        vx = [None] * self.naf 
        for ix,af in enumerate(self.af):
            cx[ix],vx[ix] = af.get_grad_deterministic(config[ix],unsigned=unsigned)
        return np.array(cx),np.concatenate(vx)
    def _new_log_prob_from_plq(self,plq,sites,cis):
        py = [None] * self.naf 
        plq_new = [None] * self.naf
        config_new = self.parse_config(self.config_new)
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq_new[ix],py[ix] = af._new_log_prob_from_plq(plq[ix],sites,cis[ix])
            else:
                py[ix] = af.log_prob(config_new[ix])
            if py[ix] is None:
                return plq_new,None 
        return plq_new,sum(py)
class ExchangeSampler2D(ExchangeSampler2D):
    def _update_pair_from_plq(self,site1,site2,plq,cols):
        ix1,ix2 = [self.flatten(site) for site in (site1,site2)]
        i1,i2 = self.config[ix1],self.config[ix2]
        if i1==i2: # term vanishes 
            return plq,cols
        sign_itm = self.af[0].intermediate_sign(self.config)
        
        ls = self.af[0],model.pair_terms(i1,i2)
        ix = self.rng.choice(len(ls))
        i1_new,i2_new,sign,_ = ls[ix]

        config_new = list(self.config)
        config_new[ix1] = i1_new
        config_new[ix2] = i2_new 
        config_new = self.af.parse_config(tuple(config_new))
        sign_new = self.config_sign(config_new)
        cx = 0 
        for ix,af in enumerate(self.af.af):
            if af.is_tn:
                cx_ix = af.unsigned_amplitude(config_new[ix],imax=,imin=)
            

        config_sites = self.af.parse_config(config_sites)
        self.af.config_new = config_new
        plq_new,py = self.af._new_log_prob_from_plq(plq,(site1,site2),config_sites)
        if py is None:
            return plq,cols
        acceptance = np.exp(py - self.px)
        if acceptance < self.rng.uniform(): # reject
            return plq,cols
        # accept, update px & config & env_m
        self.px = py
        self.config = config_new
        cols = self.af.replace_sites(cols,(site1,site2),config_sites)
        return plq_new,cols 
