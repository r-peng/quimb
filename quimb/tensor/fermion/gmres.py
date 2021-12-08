import numpy as np
from pyblock3.algebra.fermion import eye
inv_map = {'+':'-','-':'+'}
def parse(tsr,ax):
    q_labels = tsr.q_labels.copy()
    narr = q_labels[:,ax]
    narr = [tsr.symmetry.from_flat(q) for q in narr]
    narr = [-q for q in narr]
    narr = [tsr.symmetry.to_flat(q) for q in narr]
    q_labels[:,ax] = np.array(narr)

    pattern = [c for c in tsr.pattern]
    pattern[ax] = inv_map[pattern[ax]]
    pattern = ''.join(pattern)

    tsr.pattern = pattern
    tsr.q_labels = q_labels
    return tsr
def get_physical_identity(t):
    n_blocks = set(t.q_labels[:,-1])
    n_blocks = len(n_blocks)
    bond_info = t.get_bond_info(-1)
    I = eye(bond_info,flat=True)
    I.pattern = '-+'
    return I
def svd(t,left_idx,cutoff=1e-6):
    qpn_partition = (t.dq,t.symmetry(0))
    u,s,v = t.tensor_svd(left_idx=left_idx,qpn_partition=qpn_partition,absorb=1,
                         cutoff=cutoff,cutoff_mode=1)
    assert s is None
    assert u.dq==t.dq
    assert v.dq==t.symmetry(0)
    u,v = parse(u,-1),parse(v,0)
    assert u.dq==t.dq
    assert v.dq==t.symmetry(0)
#    assert (np.tensordot(u,v,axes=((-1,),(0,)))-t).norm()<cutoff

    nvir = len(t.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    lhs = np.tensordot(u.dagger,u,axes=(axs1,left_idx))
    I = get_physical_identity(u)
    assert (lhs-I).norm()<cutoff
#    if (lhs-I).norm()>cutoff:
#        print((lhs-I).norm())
#        print(lhs)
#        print(I)
    return u,v
def ovlp(D1,D2,c1,c2):
    out = 0.0
    for i in range(min(c1,c2)+2):
        out += np.tensordot(D1[i,c1].dagger,D2[i,c2],axes=((1,0),(0,1)))
    return out
