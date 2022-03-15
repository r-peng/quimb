import numpy as np
from itertools import product
from functools import reduce

from pyblock3.algebra.fermion_setting import symmetry_map
from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_ops import (
    make_phase_dict,
    _fuse_data_map,
    _compute_swap_phase
)
from .fermion_arbgeom_tebd import LocalHamGen
from .fermion_2d_tebd import get_pattern
from .fermion_core import FermionTensor,FermionTensorNetwork
from .fermion_2d import FPEPS
from ..tensor_2d import PEPS
from ..tensor_2d_tebd import LocalHam2D 
SVD_SCREENING = 1e-28
#################### pyblock-like operators ##############
def creation(symmetry='u1'):
    dat = np.array([[1.0]])
    symmetry = symmetry.upper()
    symm = symmetry_map(symmetry)
    if symmetry=='u1':
        qlab = U1(1),U1(0)
    elif symmetry=='z2':
        qlab = Z2(1),Z2(0)
    else:
       raise NotImplementedError
    blks = [SubTensor(reduced=dat,q_labels=qlab)]
    T = SparseFermionTensor(blocks=blks,pattern='+-',shape=(2,2))
    return T.to_flat()
def get_phys_identity(symmetry='u1'):
    dat = np.array([[1.0]])
    if symmetry=='u1':
        qlabs = (U1(0),U1(0)),(U1(1),U1(1))
    elif symmetry=='z2':
        qlabs = (Z2(0),Z2(0)),(Z2(1),Z2(1))
    else:
       raise NotImplementedError
    blks = [SubTensor(reduced=dat,q_labels=qlab) for qlab in qlabs]
    T = SparseFermionTensor(blocks=blks,pattern='+-',shape=(2,2))
    return T.to_flat()
def eye(bond_info):
    dq,dim = list(bond_info.items())[0]
    qlab = dq,-dq
    dat = np.eye(dim)
    blks = [SubTensor(reduced=dat,q_labels=qlab)]
    T = SparseFermionTensor(blocks=blks,pattern='+-')
    return T.to_flat() 
def bonded_vaccum(vir_shape,pattern,normalize=True,symmetry='u1'):
    if symmetry=='u1':
        q_label = U1(0)
    elif symmetry=='z2':
        q_label = Z2(0)
    else:
        raise NotImplementedError
    idx, ish = 0,1
    vir_shape = tuple(vir_shape)
    shape = vir_shape + (ish,)
    arr = np.zeros(shape)
    arr[...,idx] = 1
    q_labels = (q_label, ) * len(pattern)
    blocks = [SubTensor(reduced=arr, q_labels=q_labels)]
    T = SparseFermionTensor(blocks=blocks, pattern=pattern, shape=vir_shape+(2,))
    if normalize:
        T = T / T.norm()
    return T.to_flat()
def spinless_fermion(t,v,symmetry='u1'):
    if symmetry=='u1':
        symmetry = U1
    elif symmetry=='z2':
        symmetry = Z2
    else:
        raise NotImplementedError
    block_dict = dict()
    state_map = {0:(symmetry(0),0,1),
                 1:(symmetry(1),0,1)}
    cre_map = ann_map = {0:'',1:'+'}
    hop_map = {0:1,1:0}
    pn_dict = {0:0,1:1}
    for s1,s2 in product(cre_map.keys(),repeat=2):
        q1,ix1,d1 = state_map[s1]
        q2,ix2,d2 = state_map[s2]
        val = s1*s2*v
        if (q1,q2,q1,q2) not in block_dict:
            block_dict[(q1,q2,q1,q2)] = np.zeros((d1,d2,d1,d2))
        dat = block_dict[(q1,q2,q1,q2)]
        phase = _compute_swap_phase(s1,s2,s1,s2)
        dat[ix1,ix2,ix1,ix2] += phase*val

        s3 = hop_map[s1]
        s4 = hop_map[s2]
        q3,ix3,d3 = state_map[s3]
        q4,ix4,d4 = state_map[s4]
        input_string  = sorted(cre_map[s1]+cre_map[s2])
        output_string = sorted(cre_map[s3]+cre_map[s4])
        if input_string != output_string:
            continue
        if (q1,q2,q3,q4) not in block_dict:
            block_dict[(q1,q2,q3,q4)] = np.zeros((d1,d2,d3,d4))
        dat = block_dict[(q1,q2,q3,q4)]
        phase = _compute_swap_phase(s1,s2,s3,s4)
        dat[ix1,ix2,ix3,ix4] += phase*(-t)
    blocks = [SubTensor(reduced=dat,q_labels=q) for q,dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    return T.to_flat()
def get_flat_exponential(T,x):
    symmetry = T.dq.__class__
    state_map = {0:(symmetry(0),0,1),
                 1:(symmetry(1),0,1)}
    phase_dict = make_phase_dict(state_map, T.ndim)
    def get_phase(qlabels):
        return phase_dict[qlabels]
    split_ax = T.ndim//2
    left_pattern = T.pattern[:split_ax]
    right_pattern = T.pattern[split_ax:]
    data_map = {}
    from_parent = {}
    to_parent = {}
    for iblk in range(T.n_blocks):
        left_q = tuple(T.q_labels[iblk,:split_ax])
        right_q = tuple(T.q_labels[iblk,split_ax:])
        parent = _fuse_data_map(left_q, right_q, from_parent, to_parent, data_map)
        data_map[parent].append(iblk)
    datas = []
    shapes = []
    qlablst = []
    for slab, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        for iblk in datasets:
            lq = tuple(T.q_labels[iblk,:split_ax])
            rq = tuple(T.q_labels[iblk,split_ax:])
            if lq not in row_map:
                new_row_len = row_len + np.prod(T.shapes[iblk,:split_ax], dtype=int)
                row_map[lq] = (row_len, new_row_len, T.shapes[iblk,:split_ax])
                row_len = new_row_len
            if rq not in row_map:
                new_row_len = row_len + np.prod(T.shapes[iblk,split_ax:], dtype=int)
                row_map[rq] = (row_len, new_row_len, T.shapes[iblk,split_ax:])
                row_len = new_row_len
        data = np.zeros([row_len, row_len], dtype=T.dtype)
        for iblk in datasets:
            lq = tuple(T.q_labels[iblk,:split_ax])
            rq = tuple(T.q_labels[iblk,split_ax:])
            ist, ied = row_map[lq][:2]
            jst, jed = row_map[rq][:2]
            qlabs = tuple([symmetry.from_flat(iq) for iq in T.q_labels[iblk]])
            phase = get_phase(qlabs)
            if isinstance(phase, np.ndarray):
                phase = phase.reshape(ied-ist, jed-jst)
            data[ist:ied,jst:jed] = T.data[T.idxs[iblk]:T.idxs[iblk+1]].reshape(ied-ist, jed-jst) * phase
        if data.size ==0:
            continue
        el, ev = np.linalg.eigh(data)
        s = np.diag(np.exp(el*x))
        tmp = reduce(np.dot, (ev, s, ev.conj().T))
        for lq, (ist, ied, ish) in row_map.items():
            for rq, (jst, jed, jsh) in row_map.items():
                q_labels = lq + rq
                qlabs = tuple([symmetry.from_flat(iq) for iq in q_labels])
                phase = get_phase(qlabs)
                chunk = tmp[ist:ied, jst:jed].reshape(tuple(ish)+tuple(jsh)) * phase
                if abs(chunk).max()<SVD_SCREENING:
                    continue
                datas.append(chunk.ravel())
                shapes.append(tuple(ish)+tuple(jsh))
                qlablst.append(q_labels)
    q_labels = np.asarray(qlablst, dtype=np.uint32)
    shapes = np.asarray(shapes, dtype=np.uint32)
    datas = np.concatenate(datas)
    Texp = T.__class__(q_labels, shapes, datas, pattern=T.pattern, symmetry=T.symmetry)
    return Texp
################# quimb operator class ####################
def SpinlessFermion(t,v,Lx,Ly,symmetry='u1'): 
    ham = dict()
    op = spinless_fermion(t,v,symmetry=symmetry)
    for i, j in product(range(Lx), range(Ly)):
        if i+1 != Lx:
            where = ((i,j), (i+1,j))
            ham[where] = op.copy()
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            ham[where] = op.copy()
    return LocalHam2D(Lx, Ly, ham)
class LocalHamGen_(LocalHamGen):
    def _expm_cached(self, x, y):
        cache = self._op_cache['expm']
        key = (id(x), y)
        if key not in cache:
            out = get_flat_exponential(x, y)
            cache[key] = out
        return cache[key]
class LocalHam2D(LocalHamGen_):
    """A 2D Fermion Hamiltonian represented as local terms. Different from
    class:`~quimb.tensor.tensor_2d_tebd.LocalHam2D`, this does not combine
    two sites and one site term into a single interaction per lattice pair.

    Parameters
    ----------
    Lx : int
        The number of rows.
    Ly : int
        The number of columns.
    H2 : pyblock3 tensors or dict[tuple[tuple[int]], pyblock3 tensors]
        The two site term(s). If a single array is given, assume to be the
        default interaction for all nearest neighbours. If a dict is supplied,
        the keys should represent specific pairs of coordinates like
        ``((ia, ja), (ib, jb))`` with the values the array representing the
        interaction for that pair. A default term for all remaining nearest
        neighbours interactions can still be supplied with the key ``None``.
    H1 : array_like or dict[tuple[int], array_like], optional
        The one site term(s). If a single array is given, assume to be the
        default onsite term for all terms. If a dict is supplied,
        the keys should represent specific coordinates like
        ``(i, j)`` with the values the array representing the local term for
        that site. A default term for all remaining sites can still be supplied
        with the key ``None``.

    Attributes
    ----------
    terms : dict[tuple[tuple[int]], pyblock3 tensors]
        The total effective local term for each interaction (with single site
        terms appropriately absorbed). Each key is a pair of coordinates
        ``ija, ijb`` with ``ija < ijb``.

    """

    def __init__(self, Lx, Ly, H2, H1=None):
        self.Lx = int(Lx)
        self.Ly = int(Ly)

        # parse two site terms
        if hasattr(H2, 'shape'):
            # use as default nearest neighbour term
            H2 = {None: H2}
        else:
            H2 = dict(H2)

        # possibly fill in default gates
        default_H2 = H2.pop(None, None)
        if default_H2 is not None:
            for coo_a, coo_b in gen_2d_bonds(Lx, Ly, steppers=[
                lambda i, j: (i, j + 1),
                lambda i, j: (i + 1, j),
            ]):
                if (coo_a, coo_b) not in H2 and (coo_b, coo_a) not in H2:
                    H2[coo_a, coo_b] = default_H2

        super().__init__(H2=H2, H1=H1)
    
    @property
    def nsites(self):
        """The number of sites in the system.
        """
        return self.Lx * self.Ly
    
    draw = LocalHam2D.draw
    __repr__ = LocalHam2D.__repr__
################### wfn initialization ###################
def get_product_state(Lx,Ly,n=0.5,symmetry='u1'):
    """
    helper function to generate initial guess from regular PEPS
    |psi> = \prod |alpha> \prod | > 
    this function only works for half filled case with U1 symmetry
    Note energy of this wfn is 0
    """
    N = int(Lx*Ly*n+1e-6)
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=2)
    ftn = FermionTensorNetwork([])
    ind_to_pattern_map = dict()
    cre = creation(symmetry=symmetry)
    count = 0
    for ix, iy in product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = get_pattern(T.inds,ind_to_pattern_map)
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        vac = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern)
        if (ix+iy)%2==0 and count<N:
            trans_order = list(range(1,T.ndim))+[0]
            data = np.tensordot(cre, vac, axes=((1,), (-1,))).transpose(trans_order)
            count += 1
        else:
            data = vac
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    assert count==N
    ftn.view_as_(FPEPS, like=tn)
    return ftn,N

