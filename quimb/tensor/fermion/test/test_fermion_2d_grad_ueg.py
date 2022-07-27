from quimb.tensor.fermion.fermion_2d_tebd import SimpleUpdate
from quimb.tensor.fermion.ueg_utils import (
    fd_tebd,
    fd_grad,
)
from quimb.tensor.fermion.spin_utils import (
    get_product_state,
    data_map,
    symmetry,
    flat,
    spinless,
)
from quimb.tensor.fermion.utils import (
    worker_execution,
    write_ftn_to_disc,
    load_ftn_from_disc,
)
from quimb.tensor.fermion.fermion_2d_grad_1 import (
    GlobalGrad,
    compute_energy,
    check_particle_number,
)
from quimb.tensor.fermion.block_interface import set_options
from scipy.optimize import optimize
import numpy as np
set_options(symmetry=symmetry, use_cpp=True)

N = 3
D = 3
chi = 128
L = 2.
Na = 4
Nb = 0 if spinless else 4
Ne = Na + Nb
order = 2
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
if RANK==0:
    try:
        ftn = load_ftn_from_disc(f'saved_su_states/ueg_N{N}D{D}_spinless{spinless}')
        H = fd_tebd(N,L,order=order)
        su = SimpleUpdate(ftn,H,D=D,chi=chi)
        su.evolve(1,0.1)
        ftn = su.get_state()
    except FileNotFoundError:
        ftn = get_product_state(N,N,Na=Na,Nb=Nb)
        check_particle_number(ftn,'./tmpdir/',spinless=spinless)
        H = fd_tebd(N,L,order=order)
        su = SimpleUpdate(ftn,H,D=D,chi=chi)
        su.evolve(100,0.5)
        ftn = su.get_state()
        write_ftn_to_disc(ftn,f'saved_su_states/ueg_N{N}D{D}_spinless{spinless}',
                                provided_filename=True)

    pna = data_map['pna'].copy()
    pna = {(i,j):pna.copy() for i in range(N) for j in range(N)}
    pna = ftn.compute_local_expectation(pna,return_all=True,normalized=True)
    arra = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            e,n = pna[i,j]
            arra[i,j] = e/n
    print('total pna:',sum(arra.reshape(-1)))
    print(arra)
    if not spinless:
        pnb = data_map['pnb'].copy()
        pnb = {(i,j):pnb.copy() for i in range(N) for j in range(N)}
        pnb = ftn.compute_local_expectation(pnb,return_all=True,normalized=True)
        arrb = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                e,n = pnb[i,j]
                arrb[i,j] = e/n
        print('total pnb:',sum(arrb.reshape(-1)))
        print(arrb)
        arr = arra+arrb
        print('total pn:',sum(arr.reshape(-1)))
        print(arr)
    check_particle_number(ftn,'./tmpdir/',spinless=spinless)
    #exit()

    H = fd_grad(N,L,Ne,order=order,has_coulomb=False,spinless=spinless)
    gg = GlobalGrad(H,ftn,chi,'./psi','./tmpdir/')
    x = gg.fpeps2vec(ftn)
    print(x)
    print(ftn)
    print('num_param=',len(x))
    #exit()
    f,g = gg.compute_grad(x)
    print('f=',f)
    g *= 2.0
    def _f(x):
        E,N = gg.compute_energy(x)
        return E
    for epsilon in [1e-6]:
        sf = optimize._prepare_scalar_function(
             _f,x0=x,jac=None,epsilon=epsilon,
             finite_diff_rel_step=epsilon) 
        g_ = sf.grad(x)
        print(epsilon,np.linalg.norm(g),np.linalg.norm(g_-g)/np.linalg.norm(g))
        idxs = []
        for ix in range(len(g)):
            if abs(g[ix]-g_[ix])>1e-6: 
                idxs.append(ix)
        print(idxs)
        #print(g)
        #print(g_)
    for complete_rank in range(1, SIZE):
        COMM.send('finished', dest=complete_rank)
else:
    worker_execution()
