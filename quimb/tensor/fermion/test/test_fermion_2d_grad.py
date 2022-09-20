from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,SimpleUpdate
from quimb.tensor.fermion.spin_utils import (
    get_half_filled_product_state,
    data_map,symmetry,flat,spinless,
)
from quimb.tensor.fermion.utils import (
    worker_execution,
    write_ftn_to_disc,
    load_ftn_from_disc,
)
from quimb.tensor.fermion.fermion_2d_grad_1 import (
    hubbard,
    GlobalGrad,
    compute_energy,
    check_particle_number,
)
from scipy.optimize import optimize
import numpy as np

Lx = Ly = 3
t,u = 1.0,8.0
D = 3
chi = 128
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
if RANK==0:
    try:
        ftn = load_ftn_from_disc(f'saved_su_states/hubbard_Lx{Lx}Ly{Ly}D{D}')
    except FileNotFoundError:
        H = Hubbard2D(t,u,Lx,Ly,symmetry=symmetry)
        ftn = get_half_filled_product_state(Lx,Ly)
        su = SimpleUpdate(ftn,H,D=D,chi=chi)
        su.evolve(100,0.5)
        ftn = su.get_state()
        write_ftn_to_disc(ftn,f'saved_su_states/hubbard_Lx{Lx}Ly{Ly}D{D}',
                                provided_filename=True)

    pna,pnb = data_map['pna'].copy(),data_map['pnb'].copy()
    pna = {(i,j):pna.copy() for i in range(Lx) for j in range(Ly)}
    pnb = {(i,j):pnb.copy() for i in range(Lx) for j in range(Ly)}
    pna = ftn.compute_local_expectation(pna,return_all=True,normalized=True)
    pnb = ftn.compute_local_expectation(pnb,return_all=True,normalized=True)
    arra = np.zeros((Lx,Ly))
    for i in range(Lx):
        for j in range(Ly):
            e,n = pna[i,j]
            arra[i,j] = e/n
    print('total pna:',sum(arra.reshape(-1)))
    print(arra)
    arrb = np.zeros((Lx,Ly))
    for i in range(Lx):
        for j in range(Ly):
            e,n = pnb[i,j]
            arrb[i,j] = e/n
    print('total pnb:',sum(arrb.reshape(-1)))
    print(arrb)
    arr = arra+arrb
    print('total pn:',sum(arr.reshape(-1)))
    print(arr)
    check_particle_number(ftn,'./tmpdir/')

    H = hubbard(t,u,Lx,Ly)
    gg = GlobalGrad(H,ftn,chi,'./psi','./tmpdir/',smallmem=True)
    x = gg.fpeps2vec(ftn)
    f,g = gg.compute_grad(x)
    print(g)

    # check vec2fpeps & energy
    H = hubbard(t,u,Lx,Ly)
    gg = GlobalGrad(H,ftn,chi,'./psi','./tmpdir/',smallmem=False)
    f,g = gg.compute_grad(x)
    print(g)

    def _f(x):
        E,N = gg.compute_energy(x)
        return E
    for epsilon in [1e-6]:
        sf = optimize._prepare_scalar_function(
             _f,x0=x,jac=None,epsilon=epsilon,
             finite_diff_rel_step=epsilon) 
        #sf = optimize._prepare_scalar_function(
        #     _f,x0=x,jac='2-point',epsilon=epsilon,
        #     finite_diff_rel_step=epsilon) 
        g_ = sf.grad(x)/2.
        print(g_)
        print(epsilon,np.linalg.norm(g),np.linalg.norm(g_-g)/np.linalg.norm(g))
        idxs = []
        for ix in range(len(g)):
            if abs(g[ix]-g_[ix])>1e-6: 
                idxs.append(ix)
        #print(idxs)
        #print(g[idxs])
        #print(g_[idxs])
    for complete_rank in range(1, SIZE):
        COMM.send('finished', dest=complete_rank)
else:
    worker_execution()
