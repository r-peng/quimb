
import time,itertools
import numpy as np
np.set_printoptions(suppress=True,precision=4,linewidth=2000)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

import sys
this = sys.modules[__name__]
from ..tensor_vmc import Hamiltonian
from .fermion_2d_vmc import FermionAmplitudeFactory2D,FermionExchangeSampler2D
from .fermion_product_vmc import parse_config,config_from_ab
def set_options(pbc=False):
    this._PBC = pbc
    this._DETERMINISTIC = True 
    from .fermion_2d_vmc import set_options as set_options_
    set_options_(deterministic=True)
#def reorder(config_a,config_b,iprint=False):
#    config_a = np.array(config_a) 
#    config_b = np.array(config_b) 
#
#    cum_sum = np.cumsum(config_a[1:][::-1])
#    sign = (-1)**(np.dot(config_b[:-1],cum_sum[::-1])%2)
#    #if iprint:
#    #    print(RANK,config_a,config_b,cum_sum)
#    return sign

class AmplitudeFactory(FermionAmplitudeFactory2D):
    def __init__(self,moa,mob,proj,Lx,Ly):
        self.params = moa,mob,proj
        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        self.Lx,self.Ly = Lx,Ly
        self.sites = list(itertools.product(range(Lx),range(Ly)))
        self.spinless = False
    def get_x(self):
        return np.concatenate([param.flatten() for param in self.params])
    def update_cache(self,config):
        pass
    def unsigned_amplitude(self,config):
        #state_map = [0,1,2,3]
        state_map = [3,1,2,0]
        amp = np.prod(np.array([self.params[2][ix,state_map[ci]] for ix,ci in enumerate(config[2])]))
        for ix in (0,1):
            det = np.nonzero(np.array(config[ix]))[0]
            amp *= np.linalg.det(self.params[ix][det,:])
        return amp
    def prob(self,config):
        return self.unsigned_amplitude(config)**2
    def config_sign(self,config):
        return 1
    def parse_config(self,config):
        return parse_config(config)    
    def pair_energy_deterministic(self,config,site1,site2,model,cache_bot=None,cache_top=None):
        ix1,ix2 = [model.flatten(site) for site in (site1,site2)]
        ex = 0.
        for spin in (0,1):
            i1,i2 = config[spin][ix1],config[spin][ix2]
            if i1==i2: # term vanishes 
                continue 

            config_new = list(config[spin])
            config_new[ix1] = i2
            config_new[ix2] = i1 
            config_new = tuple(config_new)
            if spin==0:
                config_a,config_b = config_new,config[1-spin]
            else:
                config_a,config_b = config[1-spin],config_new
            config_full = config_from_ab(config_a,config_b)
            sign_new = (-1)**(sum(config_new[ix1+1:ix2]) % 2)
            cx_new = self.unsigned_amplitude((config_a,config_b,config_full))
            if cx_new is None:
                continue
            ex += sign_new * cx_new
        return ex * model.pair_coeff(site1,site2) 
 
