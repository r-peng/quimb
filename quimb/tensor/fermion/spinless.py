from pyblock3.algebra.core import SubTensor
from pyblock3.algebra.fermion import SparseFermionTensor
from pyblock3.algebra.fermion_symmetry import U1,Z2 
from pyblock3.algebra.fermion_ops import _compute_swap_phase,_fuse_data_map
import pyblock3.algebra.fermion_setting as setting 
import numpy as np
from itertools import product
from functools import reduce
SVD_SCREENING = setting.SVD_SCREENING

state_map_u1 = {0:(U1(0),0,1),
                1:(U1(1),0,1)}
state_map_z2 = {0:(Z2(0),0,1),
                1:(Z2(1),0,1)}
cre_map = {0:"", 1:"+"}
ann_map = {0:"", 1:"+"}
hop_map = {0:(1,), 1:(0,)}

sz_dict = {0:0, 1:.5}
pn_dict = {0:0, 1:1}

def get_state_map(symmetry):
    if isinstance(symmetry, str):
        symmetry_string = symmetry.upper()
    else:
        symmetry_string = symmetry.__name__
    return {#"U11": state_map_u11,
            "U1": state_map_u1,
            #"Z4": state_map_z4,
            #"Z22": state_map_z22,
            "Z2": state_map_z2}[symmetry_string]
def bonded_vaccum(vir_shape, pattern, normalize=True, symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(
                                 symmetry=symmetry, flat=flat)
    state_map = get_state_map(symmetry)
    q_label, idx, ish = state_map[0]
    vir_shape = tuple(vir_shape)
    shape = vir_shape + (ish,)
    arr = np.zeros(shape)
    arr[...,idx] = 1
    q_labels = (q_label, ) * len(pattern)
    blocks = [SubTensor(reduced=arr, q_labels=q_labels)]
    T = SparseFermionTensor(blocks=blocks, pattern=pattern, shape=vir_shape+(2,))
    if normalize:
        T = T / T.norm()
    if flat:
        return T.to_flat()
    else:
        return T 
def creation(symmetry=None,flat=None):
    symmetry, flat = setting.dispatch_settings(
                                 symmetry=symmetry, flat=flat)
    state_map = get_state_map(symmetry)
    block_dict = dict()
    creation_map = {0:1}
    for s1 in cre_map.keys():
        q1, ix1, d1 = state_map[s1]
        if s1 not in creation_map:
            continue
        output_s1 = creation_map[s1]
        if isinstance(output_s1, int):
            output_s1 = (output_s1, )
        for os1 in output_s1:
            q2, ix2, d2 = state_map[os1]
            if (q2, q1) not in block_dict:
                block_dict[(q2, q1)] = np.zeros([d2, d1])
            dat = block_dict[(q2, q1)]
            phase = _compute_swap_phase(os1, s1)
            dat[ix2, ix1] += phase
    blocks = [SubTensor(reduced=dat, q_labels=qlab) for qlab, dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="+-", shape=(2,2))
    if flat:
        return T.to_flat()
    else:
        return T
def Hubbard(t=1, u=1, mu=0., fac=None, symmetry=None, flat=None):
    symmetry, flat = setting.dispatch_settings(
                                 symmetry=symmetry, flat=flat)
    if fac is None:
        fac = (1, 1)
    faca, facb = fac
    state_map = get_state_map(symmetry)
    block_dict = dict()

    for s1, s2 in product(cre_map.keys(), repeat=2):
        q1, ix1, d1 = state_map[s1]
        q2, ix2, d2 = state_map[s2]
        val = (pn_dict[s1]==2) * faca * u + pn_dict[s1] * faca * mu +\
              (pn_dict[s2]==2) * facb * u + pn_dict[s2] * facb * mu
        if (q1, q2, q1, q2) not in block_dict:
            block_dict[(q1, q2, q1, q2)] = np.zeros([d1, d2, d1, d2])
        dat = block_dict[(q1, q2, q1, q2)]
        phase = _compute_swap_phase(s1, s2, s1, s2)
        dat[ix1, ix2, ix1, ix2] += phase * val
        for s3 in hop_map[s1]:
            q3, ix3, d3 = state_map[s3]
            input_string = sorted(cre_map[s1]+cre_map[s2])
            for s4 in hop_map[s2]:
                q4, ix4, d4 = state_map[s4]
                output_string = sorted(cre_map[s3]+cre_map[s4])
                if input_string != output_string:
                    continue
                if (q1, q2, q3, q4) not in block_dict:
                    block_dict[(q1, q2, q3, q4)] = np.zeros([d1, d2, d3, d4])
                dat = block_dict[(q1, q2, q3, q4)]
                phase = _compute_swap_phase(s1, s2, s3, s4)
                dat[ix1, ix2, ix3, ix4] += phase * -t

    blocks = [SubTensor(reduced=dat, q_labels=qlab) for qlab, dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    if flat:
        return T.to_flat()
    else:
        return T
def make_phase_dict(state_map, ndim):
    phase_dict = {}
    for states in product(state_map.keys(), repeat=ndim):
        all_info = [state_map[istate] for istate in states]
        qlabs = tuple([info[0] for info in all_info])
        input_string = sorted("".join([cre_map[istate] for istate in states[:ndim//2]]))
        output_string = sorted("".join([ann_map[istate] for istate in states[ndim//2:]]))
        if input_string != output_string:
            continue
        inds = tuple([info[1] for info in all_info])
        shape = tuple([info[2] for info in all_info])
        if qlabs not in phase_dict:
            phase_dict[qlabs] = np.zeros(shape)
        phase_dict[qlabs][inds] = _compute_swap_phase(*states)
    return phase_dict 
def get_exponential(T, x):
    if setting.DEFAULT_FLAT:
        return get_flat_exponential(T, x)
    else:
        return get_sparse_exponential(T, x)
def get_flat_exponential(T, x):
    symmetry = T.dq.__class__
    if setting.DEFAULT_FERMION:
        state_map = get_state_map(T.symmetry)
        phase_dict = make_phase_dict(state_map, T.ndim)
    def get_phase(qlabels):
        if setting.DEFAULT_FERMION:
            return phase_dict[qlabels]
        else:
            return 1
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
def get_sparse_exponential(T, x):
    symmetry = T.dq.__class__
    if setting.DEFAULT_FERMION:
        state_map = get_state_map(symmetry)
        phase_dict = make_phase_dict(state_map, T.ndim)
    def get_phase(qlabels):
        if setting.DEFAULT_FERMION:
            return phase_dict[qlabels]
        else:
            return 1
    split_ax = T.ndim//2
    left_pattern = T.pattern[:split_ax]
    right_pattern = T.pattern[split_ax:]
    data_map = {}
    from_parent = {}
    to_parent = {}
    blocks = []
    for iblk in T.blocks:
        left_q = iblk.q_labels[:split_ax]
        right_q = iblk.q_labels[split_ax:]
        parent = _fuse_data_map(left_q, right_q, from_parent, to_parent, data_map)
        data_map[parent].append(iblk)

    for slab, datasets in data_map.items():
        row_len = col_len = 0
        row_map = {}
        for iblk in datasets:
            lq = iblk.q_labels[:split_ax]
            rq = iblk.q_labels[split_ax:]
            if lq not in row_map:
                new_row_len = row_len + np.prod(iblk.shape[:split_ax], dtype=int)
                row_map[lq] = (row_len, new_row_len, iblk.shape[:split_ax])
                row_len = new_row_len
            if rq not in row_map:
                new_row_len = row_len + np.prod(iblk.shape[split_ax:], dtype=int)
                row_map[rq] = (row_len, new_row_len, iblk.shape[split_ax:])
                row_len = new_row_len
        data = np.zeros([row_len, row_len], dtype=T.dtype)
        for iblk in datasets:
            lq = iblk.q_labels[:split_ax]
            rq = iblk.q_labels[split_ax:]
            ist, ied = row_map[lq][:2]
            jst, jed = row_map[rq][:2]
            phase = get_phase(iblk.q_labels)
            if isinstance(phase, np.ndarray):
                phase = phase.reshape(ied-ist, jed-jst)
            data[ist:ied,jst:jed] = np.asarray(iblk).reshape(ied-ist, jed-jst) * phase
        if data.size ==0:
            continue
        el, ev = np.linalg.eigh(data)
        s = np.diag(np.exp(el*x))
        tmp = reduce(np.dot, (ev, s, ev.conj().T))
        for lq, (ist, ied, ish) in row_map.items():
            for rq, (jst, jed, jsh) in row_map.items():
                q_labels = lq + rq
                phase = get_phase(q_labels)
                chunk = tmp[ist:ied, jst:jed].reshape(tuple(ish)+tuple(jsh)) * phase
                if abs(chunk).max()<SVD_SCREENING:
                    continue
                blocks.append(SubTensor(reduced=chunk, q_labels=q_labels))
    Texp = T.__class__(blocks=blocks, pattern=T.pattern)
    return Texp 
    
