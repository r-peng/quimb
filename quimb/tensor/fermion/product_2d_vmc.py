import time,itertools
import numpy as np

#from mpi4py import MPI
#COMM = MPI.COMM_WORLD
#SIZE = COMM.Get_size()
#RANK = COMM.Get_rank()

from .product_vmc import TNJastrow,BackFlow
from ..tensor_2d import PEPS
def get_gutzwiller(Lx,Ly,coeffs,bdim=1,eps=0.,normalize=False):
    if isinstance(coeffs,np.ndarray):
        assert len(coeffs)==4
        coeffs = {(i,j):coeffs for i,j in itertools.product(range(Lx),range(Ly))}
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [bdim] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (4,)

            data = eps * np.random.rand(*shape)
            for ix in range(4):
                data[(0,)*(len(shape)-1)+(ix,)] = coeffs[i,j][ix] 
            if normalize:
                data /= np.linalg.norm(data)
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
from ..tensor_2d_vmc import AmplitudeFactory2D 
class PEPSJastrow(TNJastrow,AmplitudeFactory2D):
    def pair_energy_deterministic(self,config,site1,site2,model,cache_top=None,cache_bot=None):
        ix1,ix2 = model.flatten(site1),model.flatten(site2)
        i1,i2 = config[ix1],config[ix2]
        if not model.pair_valid(i1,i2): # term vanishes 
            return None 
        cx = [None] * 2 
        for ix,spin in zip((0,1),('a','b')):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue 
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)

            cx_new = self.unsigned_amplitude(config_new,cache_bot=cache_bot,cache_top=cache_top,to_numpy=False)
            if cx_new is not None:
                cx[ix] = cx_new
        return cx 
class BackFlow2D(BackFlow,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,mo,nv,nl,spin,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(mo,nv,nl,spin,**kwargs)
#class CNN2D(FNN2D):
#    def __init__(self,Lx,Ly,nl,kx,ky,**kwargs):
#        self.kx,self.ky = kx,ky 
#        #self.blks = []
#        #for i,j in itertools.product(range(Lx-kx),range(Ly-ky)):
#        #    self.blks.append(list(itertools.product(range(i,i+kx),range(j,j+ky))))
#        #self.get_site_map(self.blks)
#        super().__init__(Lx,Ly,nl,**kwargs) 
#    def log_amplitude(self,config,to_numpy=True):
#        c = np.array(config,dtype=float).reshape((self.Lx,self.Ly)) 
#        jnp,c = self.get_backend(c=c)
#        for i in range(self.nl-1):
#            c = self.convolve(c,self.w[i],jnp) + self.b[i]
#            c = jnp.log(jnp.cosh(c)) 
#        c = jnp.dot(c,self.w[-1])
#        #exit()
#        if to_numpy:
#            c = tensor2backend(c,'numpy') 
#        return c,0
#    def convolve(self,x,w,jnp):
#        v = jnp.zeros((self.Lx,self.Ly),requires_grad=True) 
#        for i,j in itertools.product(range(self.Lx-kx),range(self.Ly-ky)):
#            v[i:i+kx,j:j+ky] += jnp.matmul(w[i,j,:,:],x[i:i+kx,j:j+ky].flatten()).reshape((kx,ky))
#        return v 
