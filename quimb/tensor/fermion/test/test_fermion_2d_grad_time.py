from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate
from quimb.tensor.fermion.utils import (
    get_half_filled_product_state,
    worker_execution,
)
from quimb.tensor.fermion.fermion_2d_grad_1 import (
    hubbard,
    GlobalGrad,
)
from quimb.tensor.fermion.fermion_2d_grad_2 import hubbard as hubbard_
from quimb.tensor.fermion.fermion_2d_grad_2 import GlobalGrad as GlobalGrad_
from quimb.tensor.fermion.block_interface import (
    set_options,
    creation,
    annihilation,
    ParticleNumber,
)
from scipy.optimize import optimize
import numpy as np
set_options(symmetry="u1", use_cpp=True)

Lx = Ly = 6
t,u = 1.0,8.0
D = 3
chi = 128
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
if RANK==0:
    H = Hubbard2D(t,u,Lx,Ly,symmetry='u1')
    ftn = get_half_filled_product_state(Lx,Ly,symmetry='u1')
    su = SimpleUpdate(ftn,H,D=D,chi=chi)
    su.evolve(100,0.5)
    ftn = su.get_state()

    print('grad_1')
    H = hubbard(t,u,Lx,Ly)
    gg = GlobalGrad(H,ftn,chi,'./psi','./tmpdir/')
    x = gg.fpeps2vec(ftn)
    E,N = gg.compute_energy(x)
    f,g = gg.compute_grad(x)

    print('grad_2')
    H_ = hubbard_(t,u,Lx,Ly)
    gg_ = GlobalGrad_(H_,ftn,chi,'./psi','./tmpdir/')
    x = gg_.fpeps2vec(ftn)
    E,N = gg_.compute_energy(x)
    f,g = gg_.compute_grad(x)
    for complete_rank in range(1, SIZE):
        COMM.send('finished', dest=complete_rank)
else:
    worker_execution()
