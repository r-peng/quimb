from itertools import product
from ...utils import pairwise
from ..tensor_2d_tebd import (
    SimpleUpdate, 
    conditioner, 
    LocalHam2D)
from ..tensor_2d import (
    gen_long_range_path, 
    nearest_neighbors, 
    gen_long_range_swap_path, 
    swap_path_to_long_range_path, 
    gen_2d_bonds
)

from .block_interface import eye, Hubbard
from . import block_tools
from .fermion_core import _get_gauge_location
from .fermion_arbgeom_tebd import LocalHamGen

def Hubbard2D(t, u, Lx, Ly, mu=0., symmetry=None):
    """Create a LocalHam2D object for 2D Hubbard Model

    Parameters
    ----------
    t : scalar
        The hopping parameter
    u : scalar
        Onsite columb repulsion
    Lx: int
        Size in x direction
    Ly: int
        Size in y direction
    mu: scalar, optional
        Chemical potential
    symmetry: {"z2",'u1', 'z22', 'u11'}, optional
        Symmetry in the backend

    Returns
    -------
    a LocalHam2D object
    """
    ham = dict()
    def count_neighbour(i, j):
        return (i>0) + (i<Lx-1) + (j>0) + (j<Ly-1)
    for i, j in product(range(Lx), range(Ly)):
        count_ij = count_neighbour(i,j)
        if i+1 != Lx:
            where = ((i,j), (i+1,j))
            count_b = count_neighbour(i+1,j)
            uop = Hubbard(t,u, mu, (1./count_ij, 1./count_b), symmetry=symmetry)
            ham[where] = uop
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            count_b = count_neighbour(i,j+1)
            uop = Hubbard(t,u, mu, (1./count_ij, 1./count_b), symmetry=symmetry)
            ham[where] = uop
    return LocalHam2D(Lx, Ly, ham)

class LocalHam2D(LocalHamGen):
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
    

class SimpleUpdate(SimpleUpdate):
    
    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        self._gauges = dict()
        string_inv = {"+":"-", "-":"+"}
        for ija, ijb in self._psi.gen_bond_coos():
            Tija = self._psi[ija]
            Tijb = self._psi[ijb]
            site_a = Tija.get_fermion_info()[1]
            site_b = Tijb.get_fermion_info()[1]
            if site_a > site_b:
                Tija, Tijb = Tijb, Tija
                ija, ijb = ijb, ija
            bnd = self._psi.bond(ija, ijb)
            sign_ija = Tija.data.pattern[Tija.inds.index(bnd)]
            bond_info = Tija.bond_info(bnd)
            ax = Tija.inds.index(bnd)
            if sign_ija=="-":
                new_bond_info = dict()
                for dq, dim in bond_info.items():
                    new_bond_info[-dq] = dim
                bond_info = new_bond_info
            Tsval = eye(bond_info)
            Tsval.pattern= sign_ija + string_inv[sign_ija]
            self._gauges[(ija, ijb)] = Tsval
    
    def _unpack_gauge(self, ija, ijb):
        Ta = self._psi[ija]
        Tb = self._psi[ijb]
        if (ija, ijb) in self.gauges:
            Tsval = self.gauges[(ija, ijb)]
            loca, locb, flip_pattern = _get_gauge_location(Ta, Tb)
        elif (ijb, ija) in self.gauges:
            Tsval = self.gauges[(ijb, ija)]
            locb, loca, flip_pattern = _get_gauge_location(Tb, Ta)
        else:
            raise KeyError("gauge not found")
        return Tsval, (loca, locb), flip_pattern

    def gate(self, U, where):
        """Like ``TEBD2D.gate`` but absorb and extract the relevant gauges
        before and after each gate application.
        """
        ija, ijb = where

        if callable(self.long_range_path_sequence):
            long_range_path_sequence = self.long_range_path_sequence(ija, ijb)
        else:
            long_range_path_sequence = self.long_range_path_sequence

        if self.long_range_use_swaps:
            path = tuple(gen_long_range_swap_path(
                ija, ijb, sequence=long_range_path_sequence))
            string = swap_path_to_long_range_path(path, ija)
        else:
            # get the string linking the two sites
            string = path = tuple(gen_long_range_path(
                ija, ijb, sequence=long_range_path_sequence))

        def env_neighbours(i, j):
            return tuple(filter(
                lambda coo: self._psi.valid_coo((coo)) and coo not in string,
                nearest_neighbors((i, j))
            ))

        # get the relevant neighbours for string of sites
        neighbours = {site: env_neighbours(*site) for site in string}

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval, (loc_ij, _), flip_pattern = self._unpack_gauge(site, neighbour)
                bnd = self._psi.bond(site, neighbour)
                Tij.multiply_index_diagonal_(
                    ind=bnd, x=Tsval, location=loc_ij, 
                    flip_pattern=flip_pattern, smudge=self.gauge_smudge)

        # absorb the inner bond gauges equally into both sites along string
        for site_a, site_b in pairwise(string):
            Ta, Tb = self._psi[site_a], self._psi[site_b]
            Tsval, (loca, locb), flip_pattern = self._unpack_gauge(site_a, site_b)
            bnd = self._psi.bond(site_a, site_b)
            mult_val = block_tools.sqrt(Tsval)
            Ta.multiply_index_diagonal_(ind=bnd, x=mult_val, 
                        location=loca, flip_pattern=flip_pattern)
            Tb.multiply_index_diagonal_(ind=bnd, x=mult_val, 
                        location=locb, flip_pattern=flip_pattern)

        # perform the gate, retrieving new bond singular values
        info = dict()
        self._psi.gate_(U, where, absorb=None, info=info,
                        long_range_path_sequence=path, **self.gate_opts)

        # set the new singualar values all along the chain
        for site_a, site_b in pairwise(string):
            if ('singular_values', (site_a, site_b)) in info:
                bond_pair = (site_a, site_b)
            else:
                bond_pair = (site_b, site_a)
            s = info['singular_values', bond_pair]
            if self.gauge_renorm:
                s = s / s.norm()
            if bond_pair not in self.gauges:
                self.gauges.pop((bond_pair[1], bond_pair[0]), None)
            self.gauges[bond_pair] = s

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval, (loc_ij, _), flip_pattern = self._unpack_gauge(site, neighbour)
                bnd = self._psi.bond(site, neighbour)
                Tij.multiply_index_diagonal_(
                    ind=bnd, x=Tsval, location=loc_ij, 
                    flip_pattern=flip_pattern, inverse=True)

    def get_state(self, absorb_gauges=True):
        """Return the state, with the diagonal bond gauges either absorbed
        equally into the tensors on either side of them
        (``absorb_gauges=True``, the default), or left lazily represented in
        the tensor network with hyperedges (``absorb_gauges=False``).
        """
        psi = self._psi.copy()

        if not absorb_gauges:
            raise NotImplementedError
        else:
            for (ija, ijb), Tsval in self.gauges.items():
                Ta = psi[ija]
                Tb = psi[ijb]
                bnd, = Ta.bonds(Tb)
                _, (loca, locb), flip_pattern = self._unpack_gauge(ija, ijb)
                mult_val = block_tools.sqrt(Tsval)
                Ta.multiply_index_diagonal_(bnd, mult_val, 
                            location=loca, flip_pattern=flip_pattern)
                Tb.multiply_index_diagonal_(bnd, mult_val, 
                            location=locb, flip_pattern=flip_pattern)

        if self.condition_tensors:
            conditioner(psi, balance_bonds=self.condition_balance_bonds)

        return psi

from ..tensor_2d_tebd import FullUpdate
from ..tensor_2d import calc_plaquette_sizes,calc_plaquette_map,is_lone_coo
from ..tensor_core import rand_uuid,contract_strategy
from itertools import starmap,product
from autoray import conj
from opt_einsum import shared_intermediates
import numpy as np
import functools
inv_map = {'+':'-','-':'+'}
def BGMRES_(A,x,b,max_space=10,singular_thresh=1e-6):
    nvir = len(x.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    r = b-A(x)
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
        return tsr.__class__(q_labels=q_labels,shapes=tsr.shapes,data=tsr.data,
               pattern=pattern,idxs=tsr.idxs,symmetry=tsr.symmetry,shape=tsr.shape)
    def qr(tsr):
        q,r = tsr.tensor_qr(left_idx=axs2,mod='qr') 
        return parse(q,-1),parse(r,0)
    def smin(h):
        s = 1.0
        for i in range(h.n_blocks):
            data = h.data[h.idxs[i]:h.idxs[i+1]].reshape(h.shapes[i])
            for j in range(data.shape[0]):
                s = min(s,abs(data[j,j]))
        return s 
    v,r_ = qr(r)
    V = [v]
    H = []
#    print(v.pattern)
#    print(v)
    for j in range(max_space):
        u = A(V[-1])
        Hj = [np.tensordot(v.dagger,u,axes=(axs1,axs2)) for v in V]
        u = u-sum([np.tensordot(V[l],Hj[l],axes=((-1,),(0,))) 
                   for l in range(len(V))])
        if u.norm()<singular_thresh:
            break
        lhs = [np.tensordot(v.dagger,u,axes=(range(nvir,-1,-1),range(nvir+1))) 
               for v in V]
        if sum(lhs)>singular_thresh:
            print('check V',lhs)
        v,h = qr(u)
        if smin(h)<singular_thresh:
            break
        V.append(v)
        Hj.append(h)
        H.append(Hj)
#        print('Hj,H,V',len(Hj),len(H),len(V))
    # H.T*H
#    print(len(H),len(H[-1]),len(V))
    m = len(H)
    num = np.array([np.tensordot(Hj[0].dagger,r_,axes=((1,0),(0,1))) for Hj in H])
    denom = np.zeros((m,m),dtype=num.dtype)
    for i in range(m):
        for j in range(i+1):
#            print(i,j,len(H[i]),len(H[j]))
            val = sum([np.tensordot(H[i][k].dagger,H[j][k],axes=((1,0),(0,1))) 
                       for k in range(len(H[j]))])
            denom[i,j] = denom[j,i] = val
    y = np.dot(np.linalg.inv(denom),num)
    x = x+sum([V[i]*y[i] for i in range(m)])
#    print('r0 norm=',r.norm())
    r = b-A(x)

    perturb = np.random.rand(m)*1e-3
    y_ = y+perturb
    x_ = x+sum(V[i]*y_[i] for i in range(m))
    r_ = b-A(x_)
    if r.norm()-r_.norm()>0.0:
        print('r  norm=',r.norm())
        print('r_ norm=',r_.norm())
    return x
def BGMRES(A,x0,b,max_space=10,max_iter=50,cutoff=1e-6,singular_thresh=1e-6):
    def get_eo(tsr):
        q_labels = [[],[]]
        shapes = [[],[]]
        data = [[],[]]
        for i in range(tsr.n_blocks):
            q = tsr.q_labels[i,-1]
            q = tsr.symmetry.from_flat(q)
            if q.parity==0:
                q_labels[0].append(tsr.q_labels[i,:])
                shapes[0].append(tsr.shapes[i])
                data[0] += list(tsr.data[tsr.idxs[i]:tsr.idxs[i+1]])
            else:
                q_labels[1].append(tsr.q_labels[i,:])
                shapes[1].append(tsr.shapes[i])
                data[1] += list(tsr.data[tsr.idxs[i]:tsr.idxs[i+1]])
        q_labels = [np.array(qs) for qs in q_labels]
        shapes = [np.array(sh) for sh in shapes]
        data = [np.array(dat) for dat in data]
        return [tsr.__class__(q_labels=q_labels[i],shapes=shapes[i],data=data[i],
                pattern=tsr.pattern,symmetry=tsr.symmetry) for i in [0,1]]
    xs = get_eo(x0)
    bs = get_eo(b)
#    max_iter = 1
    def blk(x,b):
        norm = b.norm()
        b_ = b/norm
        xold = x.copy()/norm
        for i in range(max_iter):
            x = BGMRES_(A,xold,b_,max_space=max_space,singular_thresh=singular_thresh)
            r_norm = (b_-A(x)).norm()
            dx = (x-xold).norm()
#            print('iter={},dx={},r_norm={}'.format(i,dx,r_norm))
            if dx<cutoff:
                break
            xold = x.copy()
        return x*norm
    x = blk(x0,b)
#    xs = [blk(xs[i],bs[i]) for i in [0,1]]
#    q_labels = np.concatenate([x.q_labels for x in xs],axis=0)
#    shapes = np.concatenate([x.shapes for x in xs],axis=0)
#    data = np.concatenate([x.data for x in xs],axis=0)
#    x = x0.__class__(q_labels=q_labels,shapes=shapes,data=data,
#                     pattern=x0.pattern,symmetry=x0.symmetry)
#    print(xs[0])
#    print(xs[1])
#    print(x)
#    exit()
    return x
def gate_full_update_als(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,
    optimize='auto-hq',solver='solve',dense=True,
    enforce_pos=False,pos_smudge=1e-6,init_simple_guess=True,
    condition_tensors=True,
    condition_maintain_norms=True,condition_balance_bonds=True):

    norm_plq = env.copy()
    overlap = env.copy()
    # factorize both local and global phase on the operator tensors
    layer_tags = 'KET','BRA'
    if is_lone_coo(where):
        _where = (where,)
    else:
        _where = tuple(where)
    ng = len(_where)
    site_ix = [bra.site_ind(i, j) for i, j in _where]
#    bnds = [rand_uuid() for _ in range(ng)]
    bnds = [bd+'_' for bd in site_ix]
    TG = overlap[tags_plq[0],'KET'].__class__(G.copy(), 
         inds=site_ix+bnds, left_inds=site_ix)
    TG = bra.fermion_space.move_past(TG)

    reindex_map = dict(zip(site_ix, bnds))
    tids = overlap._get_tids_from_inds(site_ix, which='any')
    for tid_ in tids:
        tsr = overlap.tensor_map[tid_]
        if layer_tags[0] in tsr.tags:
            tsr.reindex_(reindex_map)
    overlap.add_tensor(TG, virtual=True)
#    print('########## norm_plq ###############')
#    fs = norm_plq.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    print('########## overlap ###############')
#    fs = overlap.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    print('########## ket ###############')
#    fs = ket.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    print('########## bra ###############')
#    fs = bra.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    exit()

    print(tags_plq)
    for i,site in enumerate(tags_plq):
        ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
        bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
        norm_plq.fermion_space.move(ket_tid,i)
        norm_plq.fermion_space.move(bra_tid,norm_plq.num_tensors-1-i)

        ket_tid = overlap[site,'KET'].get_fermion_info()[0] 
        bra_tid = overlap[site,'BRA'].get_fermion_info()[0]
        overlap.fermion_space.move(ket_tid,i)
        overlap.fermion_space.move(bra_tid,overlap.num_tensors-1-i)

        ket_tid = ket[site].get_fermion_info()[0] 
        bra_tid = bra[site].get_fermion_info()[0]
        ket.fermion_space.move(ket_tid,i)
        bra.fermion_space.move(bra_tid,bra.num_tensors-1-i)
    for site in tags_plq:
        ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
        bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
        norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))

        ket_tid = overlap[site,'KET'].get_fermion_info()[0] 
        bra_tid = overlap[site,'BRA'].get_fermion_info()[0]
        overlap._refactor_phase_from_tids((ket_tid,bra_tid))

        ket_tid = ket[site].get_fermion_info()[0] 
        bra_tid = bra[site].get_fermion_info()[0]
        ket._refactor_phase_from_tids((ket_tid,))
        bra._refactor_phase_from_tids((bra_tid,))
#    print('########## norm_plq ###############')
#    fs = norm_plq.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    print('########## overlap ###############')
#    fs = overlap.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    print('########## ket ###############')
#    fs = ket.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    print('########## bra ###############')
#    fs = bra.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
    stop = False
    for site in tags_plq:
        ket_1 = norm_plq[site,'KET']
        bra_1 = norm_plq[site,'BRA']
        ket_2 = overlap[site,'KET']
        bra_2 = overlap[site,'BRA']
        ket_3 = ket[site]
        bra_3 = bra[site]
        if (ket_1.data-bra_1.data.dagger).norm() > tol:
            print('check hermitian 1',(ket_1.data-bra_1.data.dagger).norm())
            stop = True
        if (ket_2.data-bra_2.data.dagger).norm() > tol:
            print('check hermitian 2',(ket_2.data-bra_2.data.dagger).norm())
            stop = True
        if (ket_3.data-bra_3.data.dagger).norm() > tol:
            print('check hermitian 3',(ket_3.data-bra_3.data.dagger).norm())
            stop = True
        if (ket_1.data-ket_2.data).norm() > tol:
            print('check ket 2',(ket_1.data-ket_2.data).norm())
            stop = True
        if (bra_1.data-bra_2.data).norm() > tol:
            print('check bra 2',(bra_1.data-bra_2.data).norm())
            stop = True
        if (ket_1.data-ket_3.data).norm() > tol:
            print('check ket 3',(ket_1.data-ket_3.data).norm())
            print(site)
            stop = True
        if (bra_1.data-bra_3.data).norm() > tol:
            print('check bra 3',(bra_1.data-bra_3.data).norm())
            print(site)
            stop = True
    plq = norm_plq.select(tags_plq,which='any',virtual=False)
    ket_plq_tmp1 = plq.select('KET',which='all',virtual=False).contract()
    bra_plq_tmp1 = plq.select('BRA',which='all',virtual=False).contract()
    if (ket_plq_tmp1.data-bra_plq_tmp1.data.dagger).norm()>tol:
        print(ket_plq_tmp1.inds)
        print(bra_plq_tmp1.inds)
        print('hermitian:',(ket_plq_tmp1.data-bra_plq_tmp1.data.dagger).norm())
    ket_plq_tmp2 = ket.select(tags_plq,which='any',virtual=False).contract()
    if (ket_plq_tmp1.data-ket_plq_tmp2.data).norm()>tol:
        print('ket_plq:',(ket_plq_tmp1.data-ket_plq_tmp2.data).norm())
#    if stop:
#        exit()
#    print(ket_plq)
#    print(bra_plq)
#    ket_plq = overlap._select_tids(ket_tids.values())
#    bra_plq = overlap._select_tids(bra_tids.values())
#    print(ket_plq)
#    print(bra_plq)
#    exit()

    init_simple_guess=True
    condition_tensors=True
    if init_simple_guess:
        ket_plq = norm_plq.select(tags_plq,which='any')
        ket_plq = ket_plq.select('KET',which='all')
        ket_plq.view_like_(ket)
        ket_plq.gate_(G,where,contract='reduce-split',max_bond=max_bond)
    if condition_tensors:
        ket_plq = norm_plq.select(tags_plq,which='any')
        ket_plq = ket_plq.select('KET',which='all')
        conditioner(ket_plq,balance_bonds=condition_balance_bonds)
        if condition_maintain_norms:
#            conditioner(ket_plq,value=1.0,balance_bonds=condition_balance_bonds)
            pre_norm = ket_plq[tags_plq[0]].norm()
            print(pre_norm)
    for site in tags_plq:
        data = ket_plq[site].data.dagger
        norm_plq[site,'BRA'].modify(data=data)
        overlap[site,'BRA'].modify(data=data)

    cost_fid = overlap.contract(output_inds=[])
    cost_norm = norm_plq.contract(output_inds=[])
    cost = -2.0*cost_fid+cost_norm
    print('init cost={},norm_plq={}'.format(cost,cost_norm))

    xs = dict()
    x_previous = dict()
    previous_cost = None
    steps = 0
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
#                ket_tid = norm[site,'KET'].get_fermion_info()[0] 
#                bra_tid = norm[site,'BRA'].get_fermion_info()[0]
#                norm.fermion_space.move(ket_tid,0)
#                norm.fermion_space.move(bra_tid,norm_plq.num_tensors-1)
#                norm._refactor_phase_from_tids((ket_tid,bra_tid))

#                ket_tid = ovlp[site,'KET'].get_fermion_info()[0] 
#                bra_tid = ovlp[site,'BRA'].get_fermion_info()[0]
#                ovlp.fermion_space.move(ket_tid,0)
#                ovlp.fermion_space.move(bra_tid,overlap.num_tensors-1)
#                ovlp._refactor_phase_from_tids((ket_tid,bra_tid))

#                print('########## norm_plq ###############')
#                fs = norm_plq.fermion_space
#                for tid, (tsr,tsite) in fs.tensor_order.items():
#                    print(tsite,tid,tsr.tags,tsr.inds,tsr.shape)
#                print('########## overlap ###############')
#                fs = overlap.fermion_space
#                for tid, (tsr,tsite) in fs.tensor_order.items():
#                    print(tsite,tid,tsr.tags,tsr.inds,tsr.shape)
                ket_1 = norm_plq[site,'KET']
                bra_1 = norm_plq[site,'BRA']
                bra_2 = overlap[site,'BRA']
                if (ket_1.data-bra_1.data.dagger).norm() > tol:
                    print('check hermitian 1',(ket_1.data-bra_1.data.dagger).norm())
                if (bra_1.data-bra_2.data).norm() > tol:
                    print('check bra',(bra_1.data-bra_2.data).norm())

                lix = norm_plq[site,'BRA'].inds[1:][::-1]
                rix = norm_plq[site,'KET'].inds[:-1]
                pix = norm_plq[site,'BRA'].inds[:1]

#                def contract(ftn,x):
#                    tid = ftn[site,'BRA'].get_fermion_info()[0]
#                    ctr = ftn.select((site,'BRA'), which='!all')
#                    ctr.add_tag('contract')
#                    ftn[site,'KET'].modify(data=x)
#                    ftn.contract_tags('contract',inplace=True,output_inds=lix+pix)
#                    ftn._refactor_phase_from_tids((tid,))
#                    return ftn['contract'].data
#                b = contract(ftn=ovlp,x=ovlp[site,'KET'].data)
#                A = functools.partial(contract,ftn=norm)

                def A(x):
                    norm = norm_plq.copy()
                    tid = norm[site,'BRA'].get_fermion_info()[0]
                    ctr = norm.select((site,'BRA'), which='!all')
                    ctr.add_tag('contract')
                    norm[site,'KET'].modify(data=x)
                    norm.contract_tags('contract',inplace=True,output_inds=lix+pix)
                    norm._refactor_phase_from_tids((tid,))
                    return norm['contract'].data
                ovlp = overlap.copy()
                tid = ovlp[site,'BRA'].get_fermion_info()[0]
                ctr = ovlp.select((site,'BRA'), which='!all')
                ctr.add_tag('contract')
                ovlp.contract_tags('contract',inplace=True,output_inds=lix+pix)
                ovlp._refactor_phase_from_tids((tid,))
                b = ovlp['contract'].data
                data = BGMRES(A,norm_plq[site,'KET'].data,b)
                norm_plq[site,'KET'].modify(data=data)
                data = norm_plq[site,'KET'].data.dagger
                norm_plq[site,'BRA'].modify(data=data)
                overlap[site,'BRA'].modify(data=data)
          
            cost_fid = overlap.contract(output_inds=[])
            cost_norm = norm_plq.contract(output_inds=[])
            cost = -2.0*cost_fid+cost_norm
            print('iteration={},cost={},norm={}'.format(i,cost,cost_norm))
            print('')
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost

#    for i,site in enumerate(tags_plq):
#        ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
#        bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
#        norm_plq.fermion_space.move(ket_tid,i)
#        norm_plq.fermion_space.move(bra_tid,norm_plq.num_tensors-i-1)
##        print(ket_tid,i)
##        print(bra_tid,norm_plq.num_tensors-i-1)
##    exit()
#    for site in tags_plq:
#        ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
#        bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
#        norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))
    for site in tags_plq:
        ket_1 = norm_plq[site,'KET']
        bra_1 = norm_plq[site,'BRA']
        if (ket_1.data-bra_1.data.dagger).norm() > tol:
            print('check hermitian 1',site,(ket_1.data-bra_1.data.dagger).norm())
    plq = norm_plq.select(tags_plq,which='any',virtual=False)
    ket_plq_tmp = plq.select('KET',which='all',virtual=False).contract()
    bra_plq_tmp = plq.select('BRA',which='all',virtual=False).contract()
    if (ket_plq_tmp.data-bra_plq_tmp.data.dagger).norm()>tol:
        print(ket_plq_tmp.inds)
        print(bra_plq_tmp.inds)
        print((ket_plq_tmp.data-bra_plq_tmp.data.dagger).norm())
#    print('########## norm_plq ###############')
#    fs = norm_plq.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)

    if condition_tensors:
        if condition_maintain_norms:
            conditioner(
                ket_plq, value=pre_norm, balance_bonds=condition_balance_bonds)
#                ket_plq, value=1.0, balance_bonds=condition_balance_bonds)
        else:
            conditioner(
                ket_plq, balance_bonds=condition_balance_bonds)
    for site in tags_plq:
        norm_plq[site,'BRA'].modify(data=norm_plq[site,'KET'].data.dagger)

    for site in tags_plq:
        ket[site].modify(data=norm_plq[site,'KET'].data)
        bra[site].modify(data=norm_plq[site,'BRA'].data)
#    for site in tags_plq:
#        ket_tid = ket[site].get_fermion_info()[0] 
#        bra_tid = bra[site].get_fermion_info()[0]
##        print(ket_tid,bra_tid)
#        ket.fermion_space.move(ket_tid,ket_tid)
#        bra.fermion_space.move(bra_tid,bra.num_tensors-1-bra_tid)
    tid_map = {i:i for i in range(ket.num_tensors)}
    ket._reorder_from_tid(tid_map,inplace=True)
    tid_map = {i:bra.num_tensors-1-i for i in range(bra.num_tensors)}
#    print(tid_map)
    bra._reorder_from_tid(tid_map,inplace=True)
#    exit()
    for site in tags_plq:
        ket_tid = ket[site].get_fermion_info()[0] 
        bra_tid = bra[site].get_fermion_info()[0]
        ket._refactor_phase_from_tids((ket_tid,))
        bra._refactor_phase_from_tids((bra_tid,))
    for site in tags_plq:
        ket_3 = ket[site]
        bra_3 = bra[site]
        if (ket_3.data-bra_3.data.dagger).norm() > tol:
            print('check hermitian 3',(ket_3.data-bra_3.data.dagger).norm())
#    print('########## psi ###############')
#    fs = ket.fermion_space
#    for tid,(tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#    exit()

class FullUpdate(FullUpdate):
    def compute_energy(self):
        return self._psi.compute_local_expectation(self.ham.terms,
               **self.compute_energy_opts)
    def gate(self,G,where,**plaquette_env_options):
        # round 1: recompute environment for each gate
        #          basically copy of compute_local_expectation

        layer_tags = 'KET','BRA' 
        norm, _, self._bra = self._psi.make_norm(
                    return_all=True, layer_tags=layer_tags)
#        print('########## psi ###############')
#        fs = self._psi.fermion_space
#        for tid,(tsr,site) in fs.tensor_order.items():
#            print(site,tid,tsr.tags,tsr.inds,tsr.shape)
       
        max_bond = None
        cutoff = 1e-10
        canonize = True
        mode = 'mps'
        plaquette_env_options["max_bond"] = max_bond
        plaquette_env_options["cutoff"] = cutoff
        plaquette_env_options["canonize"] = canonize
        plaquette_env_options["mode"] = mode
        plaquette_env_options["layer_tags"] = layer_tags

        x_bsz,y_bsz = calc_plaquette_sizes([where], autogroup=True)[0]
        plaquette_envs = norm.compute_plaquette_environments(
            x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options)
        plaquette_map = calc_plaquette_map(plaquette_envs)
        p = plaquette_map[where]
        tn = plaquette_envs[p]
#        print('########## plaquette ###############')
#        fs = tn.fermion_space
#        for tid, (tsr,site) in fs.tensor_order.items():
#            print(site,tid,tsr.tags,tsr.inds,tsr.shape)
#        env = tn.select(tags_plq, which='!any')
#        print('########## env ###############')
#        fs = env.fermion_space
#        for tid,(tsr,site) in fs.tensor_order.items():
#            print(site,tid,tsr.tags,tsr.inds,tsr.shape)

        steps = 10
        tol = 1e-3
        max_bond = 3
        tags_plq = tuple(starmap(self._psi.site_tag, where))
        kets1 = [self._psi[site,'KET'].data.copy() for site in tags_plq]
        gate_full_update_als(ket=self._psi,env=tn,bra=self._bra,
                             G=G,where=where,tags_plq=tags_plq,
                             max_bond=self.D,optimize=self.contract_optimize,
                             condition_balance_bonds=self.condition_balance_bonds,
                             **self._gate_opts)
        self._term_count += 1
        kets2 = [self._psi[site,'KET'].data.copy() for site in tags_plq]
#        for i in [0,1]:
             
