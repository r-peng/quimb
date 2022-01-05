import numpy as np
import utils
import time
from quimb.tensor.tensor_2d import PEPS
from quimb.tensor.fermion.fermion_2d_tebd import Hubbard2D,FullUpdate
from quimb.tensor.fermion.block_interface import set_options
from quimb.tensor.fermion.fermion_core import FermionTensor,FermionTensorNetwork
from quimb.tensor.fermion.fermion_2d import FPEPS
from pyblock3.algebra.fermion_ops import creation,bonded_vaccum
from itertools import product
set_options(symmetry="u1", use_cpp=True)

Lx = Ly = 3 
D = 3
chi = 2*D**2
t = 1.0
u = 4.0

tn = PEPS.rand(Lx, Ly, bond_dim=2, phys_dim=2)
def from_regular_PEPS(tn,symmetry=None,flat=None):
    """
    helper function to generate initial guess from regular PEPS
    |psi> = \prod (|alpha> + |beta>) at each site
    this function only works for half filled case with U1 symmetry
    Note energy of this wfn is 0
    """
    ftn = FermionTensorNetwork([])
    ind_to_pattern_map = dict()
    inv_pattern = {"+":"-", "-":"+"}
    cre_sum = creation('sum',flat=flat) 
    def get_pattern(inds):
        """
        make sure patterns match in input tensors, eg,
        --->A--->B--->
         i    j    k
        pattern for A_ij = +-
        pattern for B_jk = +-
        the pattern of j index must be reversed in two operands
        """
        pattern = ""
        for ix in inds[:-1]:
            if ix in ind_to_pattern_map:
                ipattern = inv_pattern[ind_to_pattern_map[ix]]
            else:
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<len(pattern) else "+"
                ind_to_pattern_map[ix] = ipattern
            pattern += ipattern
        pattern += "+" # assuming last index is the physical index
        return pattern
    for ix, iy in product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = get_pattern(T.inds)
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        vac = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern)
        trans_order = list(range(1,T.ndim))+[0]
        cre = cre_sum
        data = np.tensordot(cre, vac, axes=((1,), (-1,))).transpose(trans_order)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn
ftn = from_regular_PEPS(tn)
H = Hubbard2D(t,u,Lx,Ly)
from quimb.tensor.fermion.fermion_2d_tebd import SimpleUpdate
su = SimpleUpdate(ftn,H,D=D,chi=chi)
nsteps = 100
dt = 0.5
su.evolve(nsteps,dt)
ftn = su.get_state()
ftn.normalize_(max_bond=chi)
ftn2 = ftn

fit_opts = {'tol':1e-8,'steps':20,
            'init_simple_guess':True,'condition_tensors':True,'condition_maintain_norms':True,
            'als_dense':False,'als_solver':'lgmres'}
fu = FullUpdate(ftn2,H,tau=0.05,D=D,chi=chi,compute_envs_every='term',compute_energy_every=1,fit_opts=fit_opts)
t = time.time()
steps = 5
fu.evolve(steps,0.05)
for i,tau in enumerate(fu.taus):
    print('i={},energy={}'.format(i,fu.energies[i]))
print('fu time:',time.time()-t)
