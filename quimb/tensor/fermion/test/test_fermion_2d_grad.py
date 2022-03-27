from quimb.tensor.fermion.fermion_2d_tebd import (
    Hubbard2D,get_half_filled_product_state,SimpleUpdate)
from quimb.tensor.fermion.fermion_2d_grad import (
    Hubbard,GlobalGrad,delete_ftn_from_disc,worker_execution,compute_energy,compute_norm)
from quimb.tensor.fermion.block_interface import set_options
from scipy.optimize import optimize
import numpy as np
set_options(symmetry="u1", use_cpp=True)

Lx = Ly = 3
t,u = 1.0,8.0
D = 3
chi = 128
directory = './'
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
if RANK==0:
    H = Hubbard2D(t,u,Lx,Ly,mu=0,symmetry='u1')
    ftn = get_half_filled_product_state(Lx,Ly,symmetry='u1')
    su = SimpleUpdate(ftn,H,D=D,chi=chi)
    su.evolve(100,0.5)
    ftn = su.get_state()

    H = Hubbard(t,u,Lx,Ly,symmetry='u1')
    gg = GlobalGrad(H,ftn,D,chi,directory,'psi')

    # check vec2fpeps    
    x = gg.fpeps2vec(ftn)
    x *= (np.random.rand()+1.0)**(1./(Lx*Ly))
    ftn_ = gg.vec2fpeps(x)
    e,N = compute_energy(H,ftn_,directory,max_bond=chi)
    print('total energy=',e*Lx*Ly ,N)

    x = np.concatenate([x,np.ones(1)*np.random.rand()])
    e,N = gg.compute_energy(x)
    print('total energy=',e*Lx*Ly ,N)

    x = np.random.rand(len(x)) 
    f,g = gg.compute_grad(x)
    ftn = gg.vec2fpeps(x)
    print('f=',f)
    def _f(x):
        e,N = gg.compute_energy(x)
        f = e + (x[-1]**2+1.0)*(N-1.0)**2
        print(f)
        return f
    for epsilon in [1e-7]:
        sf = optimize._prepare_scalar_function(
             _f,x0=x,jac=None,epsilon=epsilon,
             finite_diff_rel_step=epsilon) 
        g_ = sf.grad(x)
        print(epsilon,np.linalg.norm(g),np.linalg.norm(g_-g)/np.linalg.norm(g))
    delete_ftn_from_disc(gg.psi)
    for complete_rank in range(1, SIZE):
        COMM.send('finished', dest=complete_rank)

else:
    worker_execution()