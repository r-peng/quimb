from pyblock3.algebra.fermion_ops import creation,bonded_vaccum
from .fermion_2d_tebd import LocalHam2D
from .fermion_core import FermionTensor,FermionTensorNetwork
from .fermion_2d import FPEPS
from ..tensor_2d import PEPS
def SpinlessFermion(t,v,Lx,Ly,mu=0.0,symmetry=None): 
    ham = dict()
    cre = creation(spin='a',symmetry=symmetry,flat=True)
    cre.shape = (2,)*2
    ann = cre.dagger
    h1 = np.tensordot(cre,des,axes=([],[]))
    h1 = h1+h1.transpose([2,3,0,1])
    h1.shape = (2,)*4

    pn = np.tensordot(cre,ann,axes=((1,),(0,)))
    h2 = np.tensordot(pn,pn,axes=([],[]))
    h2.shape = (2,)*4

    op = -t*h1+v*h2
    op = op.transpose([0,2,1,3])
    for i, j in product(range(Lx), range(Ly)):
        if i+1 != Lx:
            where = ((i,j), (i+1,j))
            ham[where] = op.copy()
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            ham[where] = op.copy()
    return LocalHam2D(Lx, Ly, ham)
def get_product_state(Lx,Ly,symmetry=None):
    """
    helper function to generate initial guess from regular PEPS
    |psi> = \prod (|alpha> + |>) at each site
    this function only works for half filled case with U1 symmetry
    Note energy of this wfn is 0
    """
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=2)
    ind_to_pattern_map = dict()
    inv_pattern = {"+":"-", "-":"+"}
    cre = creation(spin='a',symmetry=symmetry,flat=True) 
    cre.shape = (2,)*2
    print(cre)
    exit() 
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
