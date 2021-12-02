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
inv_map = {'+':'-','-':'+'}
def BCG1_(A,x0,b,max_iter=10,singular_thresh=1e-6,orthogonalize=True):
    # A(bn,...,b1;k1,...,kn) bi=-ki
    # x(k1,...,kn;p)
    nvir = len(x0.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    def _A(x):
        return np.tensordot(A,x,axes=(range(nvir,2*nvir),range(nvir),))
    def inv(mat):
        mat = mat.to_sparse() 
        blks = []
        for blk in mat.blocks:
            assert blk.q_labels[0]==blk.q_labels[1]
            data = np.asarray(blk)
            if abs(np.linalg.det(data))<singular_thresh:
                return None
            data = np.linalg.inv(data)
            blks.append(blk.__class__(reduced=data,q_labels=blk.q_labels))
        mat_inv = mat.__class__(blocks=blks,pattern=mat.pattern,shape=mat.shape)
        return mat_inv.to_flat()
    def _a(r,p,gamma=None):
        denom = np.tensordot(p.dagger,_A(p),axes=(axs1[::-1],axs2))
        denom = inv(denom)
        if denom is None:
            return None
        num = r.copy() 
        if gamma is not None:
            num = np.tensordot(num,gamma,axes=((-1,),(0,)))
        num = np.tensordot(num.dagger,r,axes=(axs1,axs2))
        return np.tensordot(denom,num,axes=((-1,),(0,)))
    def _b(r_,r,gamma_inv=None):
        denom = np.tensordot(r.dagger,r,axes=(axs1,axs2))
        denom = inv(denom)
        if denom is None:
            return None
        num = np.tensordot(r_.dagger,r_,axes=(axs1,axs2))
        out = np.tensordot(denom,num,axes=((-1,),(0,)))
        if gamma_inv is not None:
            out = np.tensordot(gamma_inv,out,axes=((-1,),(0,)))
        return out
    def _b_(p,r):
        denom = np.tensordot(p.dagger,_A(p),axes=(axs1[::-1],axs2))
        denom = inv(denom)
        num = np.tensordot(r.dagger,_A(p),axes=(axs1,axs2))
        return -np.tensordot(num,denom,axes=((-1,),(0,))).dagger
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
    r0 = b-_A(x0)
    g0 = gi0 = None
    p0 = r0.transpose(tuple(range(nvir-1,-1,-1))+(nvir,))
    xk,rk,pk,gk,gik = x0,r0,p0,g0,gi0
    print('r_norm=',r0.norm())
    for k in range(max_iter):
        pAp = np.tensordot(pk.dagger,_A(pk),axes=(axs1[::-1],axs2))
        if gik is not None:
            pAp = np.tensordot(gik.dagger,pAp,axes=((-1,),(0,)))
        rAp = np.tensordot(rk.dagger,_A(pk),axes=(axs1,axs2))
        if (pAp-rAp).norm()>singular_thresh:
            print('(pAp-rAp)=',(pAp-rAp).norm())
        pr = np.tensordot(pk.dagger,rk,axes=(axs1[::-1],axs2))
        if gik is not None:
            pr = np.tensordot(gik.dagger,pr,axes=((-1,),(0,)))
        rr = np.tensordot(rk.dagger,rk,axes=(axs1,axs2))
        if (pr.dagger-rr).norm()>singular_thresh:
            print('(rp-rr)=',(pr.dagger-rr).norm())
        ak = _a(rk,pk,gamma=gk)
        if ak is None:
            break
        pa = np.tensordot(pk,ak,axes=((-1,),(0,)))
        xk_ = xk+pa
        rk_ = rk-_A(pa)
        lhs = np.tensordot(rk.dagger,rk_,axes=(axs1,axs2))
        if (lhs).norm()>singular_thresh:
            print('check r',(lhs).norm())
        lhs = np.tensordot(rk_.dagger,rk_,axes=(axs1,axs2))
        rhs = -np.tensordot(rk_.dagger,_A(pa),axes=(axs1,axs2))
        if (lhs-rhs).norm()>singular_thresh:
            print('check b',(lhs-rhs).norm())
        bk = _b(rk_,rk,gamma_inv=gik)
        if bk is None:
            break
        bk2 = _b_(pk,rk_)
        if bk2 is None:
            break
        if (bk-bk2).norm()>singular_thresh:
            print('check b',(bk-bk2).norm())
        pk_ = rk_.transpose(tuple(range(nvir-1,-1,-1))+(nvir,))
        pk_ = pk_ + np.tensordot(pk,bk,axes=((-1,),(0,)))
        lhs = np.tensordot(pk.dagger,_A(pk_),axes=(axs1[::-1],axs2))
        if (lhs).norm()>singular_thresh:
            print('check p',(lhs).norm())
        lhs = np.tensordot(pk.dagger,rk_,axes=(axs1[::-1],axs2))
        if (lhs).norm()>singular_thresh:
            print('check pr',(lhs).norm())
        if orthogonalize:
            pk_,gik_ = pk_.tensor_qr(left_idx=axs2,mod='qr')
            gik = parse(gik_,0)
            gk = inv(gik)
            if gk is None:
                xk = xk_
                r = b-_A(xk)
                print('iter={},r_norm={}'.format(k,r.norm()))
                break
            pk_ = parse(pk_,-1)
        xk,rk,pk = xk_,rk_,pk_

        r = b-_A(xk)
        print('iter={},r_norm={}'.format(k,r.norm()))
    return xk
def BCG1(A,x0,b,max_iter=10,singular_thresh=1e-6):
    def get_eo(tsr):
        tsr = tsr.to_sparse()
        even,odd = [],[]
        for blk in tsr.blocks:
            if blk.q_labels[-1].parity==0:
                even.append(blk)
            else:
                odd.append(blk)
        even = tsr.__class__(even,pattern=tsr.pattern)
        odd  = tsr.__class__(odd ,pattern=tsr.pattern)
        return even.to_flat(),odd.to_flat()
    x0_e,x0_o = get_eo(x0)
    b_e,b_o = get_eo(b)
    xe = BCG1_(A,x0_e,b_e,max_iter=max_iter,singular_thresh=singular_thresh) 
    xo = BCG1_(A,x0_o,b_o,max_iter=max_iter,singular_thresh=singular_thresh)
    xe,xo = xe.to_sparse(),xo.to_sparse()
    x = xe.__class__(xe.blocks+xo.blocks,pattern=xe.pattern)
    return x.to_flat()
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
##    exit()

    ket_tids,ket_sites,bra_tids,bra_sites = dict(),dict(),dict(),dict() 
    for site in tags_plq:
        ket_tids[site],ket_sites[site] = norm_plq[site,'KET'].get_fermion_info() 
        bra_tids[site],bra_sites[site] = norm_plq[site,'BRA'].get_fermion_info() 
        norm_plq._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))
        overlap._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))

#        lix = norm_plq[site,'BRA'].inds[1:]
#        rix = norm_plq[site,'KET'].inds[:-1]
#        pix = norm_plq[site,'BRA'].inds[:1]
#        A = (norm_plq.select(site, which='!any',virtual=False)
#             .contract(output_inds=lix+rix))
#        print('check hermitian',(A.data-A.data.dagger).norm())
##        print('A.inds',A.inds)
#    print(ket_tids)
#    print(ket_sites)
#    print(bra_tids)
#    print(bra_sites)

    init_simple_guess=False
    last = norm_plq.num_tensors-1
    if init_simple_guess:
        s0,s1 = tags_plq
#        tid_map = {ket1_tid:bra0_site,bra0_tid:ket1_site}
        tid_map = {ket_tids[s0]:ket_sites[s0],
                   ket_tids[s1]:bra_sites[s0],
                   bra_tids[s0]:bra_sites[s1],
                   bra_tids[s1]:ket_sites[s1]}
        norm_plq._reorder_from_tid(tid_map,inplace=True)
        overlap._reorder_from_tid(tid_map,inplace=True)
        #print('########## norm_plq ###############')
        #fs = norm_plq.fermion_space
        #for tid, (tsr,site) in fs.tensor_order.items():
        #    print(site,tid,tsr.tags,tsr.inds,tsr.shape)
        for site in tags_plq:
            norm_plq._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))
            overlap._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))
#            # check hermitian
#            ket_ = norm_plq[site,'KET']
#            bra_ = norm_plq[site,'BRA']
#            print('check hermitian',(ket_.data-bra_.data.dagger).norm())
##            print('ket.inds',ket_.inds)
##            print('bra.inds',bra_.inds)
##            lix = norm_plq[site,'BRA'].inds[1:]
##            rix = norm_plq[site,'KET'].inds[:-1]
##            pix = norm_plq[site,'BRA'].inds[:1]
##            A = (norm_plq.select(site, which='!any',virtual=False)
##                 .contract(output_inds=lix+rix))
##            print('check hermitian',(A.data-A.data.dagger).norm())
##            print('A.inds',A.inds)
#        ket_plq = norm_plq._select_tids(ket_tids.values(),virtual=False)
#        bra_plq = norm_plq._select_tids(bra_tids.values(),virtual=False)
#        ket_plq = ket_plq.contract()
#        bra_plq = bra_plq.contract()
#        print('check hermitian plq',(ket_plq.data-bra_plq.data.dagger).norm())
##        print('ket.inds',ket_plq.inds)
##        print('bra.inds',bra_plq.inds)
#
        ket_plq = norm_plq._select_tids(ket_tids.values())
        ket_plq.view_like_(ket)
#        ket_plq.gate_(G,where,contract='reduce-split',max_bond=max_bond)
#        if condition_tensors:
#            conditioner(ket_plq,balance_bonds=condition_balance_bonds)
#            if condition_maintain_norms:
#                pre_norm = ket_plq[s0].norm()
        for site in tags_plq:
            data = ket_plq[site].data.dagger
            print(ket_plq[site]._phase)
            print(norm_plq[site,'BRA']._phase)
#            norm_plq[site,'BRA'].modify(data=data)
#            overlap[site,'BRA'].modify(data=data)
            # check hermitian
#            ket_ = norm_plq[site,'KET']
#            ket_ = ket_plq[site]
#            bra_ = norm_plq[site,'BRA']
#            print('check hermitian',(ket_.data-bra_.data.dagger).norm())
##            print('ket.inds',ket_.inds)
##            print('bra.inds',bra_.inds)
##            lix = norm_plq[site,'BRA'].inds[1:]
##            rix = norm_plq[site,'KET'].inds[:-1]
##            pix = norm_plq[site,'BRA'].inds[:1]
##            A = (norm_plq.select(site, which='!any',virtual=False)
##                 .contract(output_inds=lix+rix))
##            print('check hermitian',(A.data-A.data.dagger).norm())
##            print('A.inds',A.inds)
###        tid_map = {ket1_tid:ket1_site,bra0_tid:bra0_site}
        tid_map = {ket_tids[s0]:ket_sites[s0],
                   bra_tids[s0]:bra_sites[s0],
                   ket_tids[s1]:ket_sites[s1],
                   bra_tids[s1]:bra_sites[s1]}
        norm_plq._reorder_from_tid(tid_map,inplace=True)
        overlap._reorder_from_tid(tid_map,inplace=True)
#        print('########## norm_plq ###############')
#        fs = norm_plq.fermion_space
#        for tid, (tsr,site) in fs.tensor_order.items():
#            print(site,tid,tsr.tags,tsr.inds,tsr.shape)
        for site in tags_plq:
            norm_plq._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))
            overlap._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))
        # check hermitian
            ket_ = norm_plq[site,'KET']
            bra_ = norm_plq[site,'BRA']
            print('check hermitian',(ket_.data-bra_.data.dagger).norm())
            lix = norm_plq[site,'BRA'].inds[1:]
            rix = norm_plq[site,'KET'].inds[:-1]
            pix = norm_plq[site,'BRA'].inds[:1]
            A = (norm_plq.select(site, which='!any',virtual=False)
                 .contract(output_inds=lix+rix))
            print('check A hermitian',(A.data-A.data.dagger).norm())
            print('check A hermitian',A.inds)
            print('check A hermitian',A.data.pattern)

    xs = dict()
    x_previous = dict()
    previous_cost = None
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                print(site)
                #tid_map = {ket_tids[site]:0,bra_tids[site]:last}
                tid_map = {ket_tids[site]:0,bra_tids[site]:last}
                norm_plq._reorder_from_tid(tid_map,inplace=True)
                norm_plq._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))

                overlap._reorder_from_tid(tid_map,inplace=True)
                overlap._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))

                lix = norm_plq[site,'BRA'].inds[1:]
                rix = norm_plq[site,'KET'].inds[:-1]
                pix = norm_plq[site,'BRA'].inds[:1]
                b = (overlap.select((site,'BRA'), which='!all',virtual=False)
                     .contract(output_inds=pix+lix)).data

                A = (norm_plq.select(site, which='!any',virtual=False)
                     .contract(output_inds=lix+rix)).data
                if (A-A.dagger).norm()>1e-6:
                    print('check hermitian',(A-A.dagger).norm())
                    exit()
                def _A(x):
                    Ax = norm_plq.select((site,'BRA'), which='!all',virtual=False)
                    Ax[site,'KET'].modify(data=x)
                    return Ax.contract(output_inds=lix+pix).data
                nvir = len(lix)
                x = norm_plq[site,'KET'].data
                y = b.dagger
                y.pattern = x.pattern
                Ax = _A(x)
                Ay = _A(y)
                Ax_  = np.tensordot(A,x,axes=(range(nvir,2*nvir),range(nvir),))
                Ay_  = np.tensordot(A,y,axes=(range(nvir,2*nvir),range(nvir),))
                Ay__ = np.tensordot(A.dagger,y,axes=(range(nvir,2*nvir),range(nvir),))
                if (Ax-Ax_).norm()>1e-6:
                    print('check contraction Ax:',(Ax-Ax_).norm())
                    exit()
                if (Ay-Ay_).norm()>1e-6:
                    print('check contraction:',(Ay-Ay_).norm())
                    exit()
                if (Ay-Ay__).norm()>1e-6:
                    print('check contraction:',(Ay-Ay__).norm())
                    exit()

                lhs = np.tensordot(Ax.dagger,y,axes=(range(1,nvir+1),range(nvir)))
                rhs = np.tensordot(x.dagger,Ay,axes=(range(1,nvir+1),range(nvir)))
                if (lhs-rhs).norm()>1e-6:
                    print('check hermitian',(lhs-rhs).norm())
                    exit()
                b = b.transpose(tuple(range(1,nvir+1))+(0,))                
                data = BCG1(A,norm_plq[site,'KET'].data,b)
                norm_plq[site,'KET'].modify(data=data)

                tid_map = {ket_tids[site]:ket_sites[site],
                           bra_tids[site]:bra_sites[site]}
                norm_plq._reorder_from_tid(tid_map,inplace=True)
                norm_plq._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))

                overlap._reorder_from_tid(tid_map,inplace=True)
                overlap._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))
                data = norm_plq[site,'KET'].data.dagger
                norm_plq[site,'BRA'].modify(data=data)
                overlap[site,'BRA'].modify(data=data)
          
            cost_fid = overlap.contract(output_inds=[])
            cost_norm = norm_plq.contract(output_inds=[])
            cost = -2.0*cost_fid+cost_norm
            print(i,cost)
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost

    condition_tensors=False
    if condition_tensors:
        tid_map = {ket1_tid:bra0_site,bra0_tid:bra1_site,bra1_tid:ket1_site}
        norm_plq._reorder_from_tid(tid_map,inplace=True)
        norm_plq._refactor_phase_from_tids((ket0_tid,bra0_tid))
        norm_plq._refactor_phase_from_tids((ket1_tid,bra1_tid))
        ket_plq = norm_plq._select_tids([ket0_tid,ket1_tid])
        if condition_maintain_norms:
            conditioner(
                ket_plq, value=pre_norm, balance_bonds=condition_balance_bonds)
        else:
            conditioner(
                ket_plq, balance_bonds=condition_balance_bonds)
        norm_plq[tags_plq[0],'BRA'].modify(data=ket_plq[tags_plq[0]].data.dagger)
        norm_plq[tags_plq[1],'BRA'].modify(data=ket_plq[tags_plq[1]].data.dagger)
        tid_map = {ket1_tid:ket1_site,bra0_tid:bra0_site,bra1_tid:bra1_site}
        norm_plq._reorder_from_tid(tid_map,inplace=True)

    for site in tags_plq:
        norm_plq._refactor_phase_from_tids([ket_tids[site],bra_tids[site]])

        tid = ket[site,'KET'].get_fermion_info()[0] 
        ket._refactor_phase_from_tids((tid,))
        ket[site].modify(data=norm_plq[site,'KET'].data)

        tid = bra[site,'BRA'].get_fermion_info()[0] 
        bra._refactor_phase_from_tids((tid,))
        bra[site].modify(data=norm_plq[site,'BRA'].data)

class FullUpdate(FullUpdate):
    def gate(self,G,where,**plaquette_env_options):
        # round 1: recompute environment for each gate
        #          basically copy of compute_local_expectation
        layer_tags = 'KET','BRA' 
        norm, _, self._bra = self._psi.make_norm(
                    return_all=True, layer_tags=layer_tags)
#        print('########## norm ###############')
#        fs = norm.fermion_space
#        for tid, (tsr,site) in fs.tensor_order.items():
#            print(site,tsr.tags,tsr.inds,tsr.shape)
       
        max_bond = None
        cutoff = 1e-10
        canonize = True
        mode = 'mps'
        layer_tags = 'KET','BRA' 
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
#            print(site,tsr.tags,tsr.inds,tsr.shape)
        tags_plq = tuple(starmap(self._psi.site_tag, where))
        steps = 10
        tol = 1e-3
        max_bond = 3
        gate_full_update_als(ket=self._psi,env=tn,bra=self._bra,
                             G=G,where=where,tags_plq=tags_plq,
                             max_bond=self.D,optimize=self.contract_optimize,
                             condition_balance_bonds=self.condition_balance_bonds,
                             **self._gate_opts)
        self._term_count += 1

