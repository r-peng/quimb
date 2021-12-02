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
from opt_einsum import shared_intermediates
import numpy as np
import functools
from pyblock3.algebra.fermion import eye
inv_map = {'+':'-','-':'+'}
def BGMRES_(A,x0,b,max_space=10,tol=1e-4):
    nvir = len(x0.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    r0 = b-A(x0)
    if r0.norm()<tol:
        return x0
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
    def qr(t):
        tq,tr = t.tensor_qr(left_idx=axs2,mod='qr') 
        return parse(tq,-1),parse(tr,0)
    v,r = qr(r0)
    def smin(h):
        s = 1.0
        for i in range(h.n_blocks):
            data = h.data[h.idxs[i]:h.idxs[i+1]].reshape(h.shapes[i])
            for j in range(data.shape[0]):
                s = min(s,abs(data[j,j]))
        return s 
    if smin(r)<tol:
        return x0
    V = [v]
    H = []
    T = []
    R = []
    n_blocks = set(x0.q_labels[:,-1])
    n_blocks = len(n_blocks)
    if n_blocks==1:
        bond_info = {x0.symmetry(1):2} 
    elif n_blocks==2:
        bond_info = {x0.symmetry(0):1,x0.symmetry(2):1} 
    elif n_blocks==3:
        bond_info = {x0.symmetry(0):1,x0.symmetry(1):2,x0.symmetry(2):1}
    eye_ = eye(bond_info,flat=True)
    eye_.pattern = '-+'
    lhs_ = (np.tensordot(v.dagger,v,axes=(axs1,axs2))-eye_).norm()
    assert lhs_<tol
    def col_ovlp(Hi,Hj):
        len_ = min(len(Hi),len(Hj))
        return sum([np.tensordot(Hi[k].dagger,Hj[k],axes=((1,0),(0,1)))  
                    for k in range(len_)])
    def gs(T,Hk):
        # assumes T is orthonormal
        Tk = [h.copy() for h in Hk]
        Rk = []
        for Tj in T:
            Rk.append(col_ovlp(Tj,Hk))
            for i in range(len(Tj)):
                Tk[i] = Tk[i]-Rk[-1]*Tj[i]
        norm = np.sqrt(col_ovlp(Tk,Tk))
        if norm<tol:
            return T,None
        else:
            T.append([h/norm for h in Tk])
            Rk.append(norm)
            assert abs(col_ovlp(T[-1],T[-1])-1.0)<tol
            for i in range(len(Hk)):
                rhs = []
                for j in range(len(T)):
                    if len(T[j])>i:
                        rhs.append(T[j][i]*Rk[j])
                rhs = sum(rhs)
                assert (Hk[i]-rhs).norm()<tol
            return T,Rk
    for j in range(max_space):
        u = A(V[-1])
#        Hj = [np.tensordot(v.dagger,u,axes=(axs1,axs2)) for v in V]
#        u = u-sum([np.tensordot(V[l],Hj[l],axes=((-1,),(0,))) 
#                   for l in range(len(V))])
        Hj = []
        for l in range(len(V)):
            Hj.append(np.tensordot(V[l].dagger,u,axes=(axs1,axs2)))
            u = u-np.tensordot(V[l],Hj[l],axes=((-1,),(0,)))
            assert np.tensordot(V[l].dagger,u,axes=(axs1,axs2)).norm()<tol
        if u.norm()<tol:
            break
        lhs  = [np.tensordot(v.dagger,u,axes=(axs1,axs2)).norm() for v in V]
        lhs_ = [(np.tensordot(v.dagger,v,axes=(axs1,axs2))-eye_).norm() for v in V]
        if sum(lhs)>tol:
            print('u',u.norm())
            print('check V',lhs)
            print('check V',lhs_)
            for k in range(len(V)):
                for l in range(k):
                    print('Vk',k)
                    print(V[k])
                    print('Vl',l)
                    print(V[l])
                    print('k,l',k,l,np.tensordot(V[l].dagger,V[k],axes=(axs1,axs2)).norm())
            exit()
        v,h = qr(u)
        if smin(h)<tol:
            break
        lhs  = [np.tensordot(v_.dagger,v,axes=(axs1,axs2)).norm() for v_ in V]
        lhs_ = (np.tensordot(v.dagger,v,axes=(axs1,axs2))-eye_).norm()
        if sum(lhs)>tol:
            print('v',v.norm())
            print('check V',lhs)
            print('check V',lhs_)
            exit()
#        T,Rk = gs(T,Hj)
#        if Rk is None:
#            break
#        R.append(Rk)
        V.append(v)
        Hj.append(h)
        H.append(Hj)
#        print('Hj,H,V',len(Hj),len(H),len(V))
    for i in range(len(H)):
        lhs = A(V[i])
        assert len(H[i])==i+2
        rhs = sum([np.tensordot(V[j],H[i][j],axes=((-1,),(0,))) for j in range(i+2)])
        assert (lhs-rhs).norm()<tol
    # H.T*H
#    print(len(H),len(H[-1]),len(V))
#    m = len(T)
#    tmp = np.zeros((m,m))
#    for i in range(m):
#        for j in range(m):
#            tmp[i,j] = col_ovlp(T[i],T[j])
#    assert np.linalg.norm(tmp-np.eye(m))<tol
#    num = np.array([np.tensordot(Tj[0].dagger,r,axes=((1,0),(0,1))) for Tj in T])
#    denom = np.zeros((m,m),dtype=num.dtype)
#    for i in range(m):
#        for j in range(i+1):
#            assert len(R[i])==i+1
#            denom[i,j]=R[i][j]
    m = len(H)
    num = np.array([np.tensordot(Hj[0].dagger,r,axes=((1,0),(0,1))) for Hj in H])
    denom = np.zeros((m,m),dtype=num.dtype)
    for i in range(m):
        for j in range(i+1):
            val = sum([np.tensordot(H[i][k].dagger,H[j][k],axes=((1,0),(0,1))) 
                       for k in range(len(H[j]))])
            denom[i,j] = denom[j,i] = val
    denom_ = np.zeros((m,m),dtype=num.dtype)
    for i in range(m):
        for j in range(m):
            denom_[i,j] = col_ovlp(H[i],H[j])
    assert np.linalg.norm(denom-denom_)<tol
    if abs(np.linalg.det(denom))<tol:
        return x0
    y = np.dot(np.linalg.inv(denom),num)
    x = x0+sum([V[i]*y[i] for i in range(m)])
#    print('r0 norm=',r.norm())
    r = b-A(x)

    perturb = np.random.rand(m)*1e-3
    y_ = y+perturb
    x_ = x0+sum(V[i]*y_[i] for i in range(m))
    r_ = b-A(x_)
    assert r.norm()-r_.norm()<1e-6
    return x
def BGMRES(A,x0,b,max_space=10,max_iter=50,cutoff=1e-4):
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
    def blk(x,b):
        norm = 1.0
        b_ = b/norm
        xold = x.copy()/norm
        for i in range(max_iter):
            x = BGMRES_(A,xold,b_,max_space=max_space,tol=cutoff)
            r_norm = (b_-A(x)).norm()
            dx = (x-xold).norm()
#            print('iter={},dx={},r_norm={}'.format(i,dx,r_norm))
            if dx<cutoff:
                break
            xold = x.copy()
        return x*norm
#    x = blk(x0,b)
#    exit()
    xs = [blk(xs[i],bs[i]) for i in [0,1]]
    q_labels = np.concatenate([x.q_labels for x in xs],axis=0)
    shapes = np.concatenate([x.shapes for x in xs],axis=0)
    data = np.concatenate([x.data for x in xs],axis=0)
    x = x0.__class__(q_labels=q_labels,shapes=shapes,data=data,
                     pattern=x0.pattern,symmetry=x0.symmetry)
    return x
def maintain_order(ftn,ref,tags_plq,tol=None):
    # make sure ftn & ref have the same rel order
    # always assumes ref to be phaseless
    ref_order = [ref[site].get_fermion_info()[1] for site in tags_plq]  
    order = [ftn[site,'KET'].get_fermion_info()[1] for site in tags_plq] 
    tids  = [ftn[site,'KET'].get_fermion_info()[0] for site in tags_plq] 
    if (ref_order[0]-ref_order[1])*(order[0]-order[1])<0:
        for i in [0,1]:
            ftn.fermion_space.move(tids[i],order[1-i])
    if tol is not None:
        ftn._refactor_phase_from_tids(tids)
        for site in tags_plq:
            global_flip = ref[site].phase.get('global_flip',False)
            local_inds  = ref[site].phase.get('local_inds',[])
            assert global_flip == False 
            assert len(local_inds) == 0
            global_flip = ftn[site,'KET'].phase.get('global_flip',False)
            local_inds  = ftn[site,'KET'].phase.get('local_inds',[])
            assert global_flip == False 
            assert len(local_inds) == 0
            assert (ref[site].data-ftn[site,'KET'].data).norm()<tol
    return ftn
def gate_full_update_als(norm_plq,overlap,tags_plq,steps,tol,max_bond,
    optimize='auto-hq',enforce_pos=False,pos_smudge=1e-6):
    cost_fid = overlap.contract(output_inds=[])
    cost_norm = norm_plq.contract(output_inds=[])
    cost = -cost_fid+cost_norm
    print('init cost={},norm_plq={}'.format(cost,cost_norm))
    def contract(ftn,site,output_inds):
        bra_pop = ftn[site,'BRA']
        original_phase = bra_pop.phase
        tid = bra_pop.get_fermion_info()[0]
    
        ctr = ftn.select((site,'BRA'), which='!all')
        ctr.add_tag('contract')
        ftn.contract_tags('contract',inplace=True,output_inds=output_inds)
        assert ftn.num_tensors==2
        ftn._refactor_phase_from_tids((tid,))
        global_flip = bra_pop.phase.get('global_flip',False)
        local_inds = bra_pop.phase.get('local_inds',[])
        assert global_flip == False 
        assert len(local_inds) == 0
        return ftn, ftn['contract'].data
        
    xs = dict()
    x_previous = dict()
    previous_cost = None
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
                bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
                norm_plq.fermion_space.move(ket_tid,0)
                norm_plq.fermion_space.move(bra_tid,norm_plq.num_tensors-1)
                norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))
                bra1 = norm_plq[site,'BRA']
                ket1 = norm_plq[site,'KET']
                global_flip = bra1.phase.get('global_flip',False)
                local_inds = bra1.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
                global_flip = ket1.phase.get('global_flip',False)
                local_inds = ket1.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0

                bra_tid = overlap[site,'BRA'].get_fermion_info()[0]
                overlap.fermion_space.move(bra_tid,overlap.num_tensors-1)
                overlap._refactor_phase_from_tids((bra_tid,))
                bra2 = overlap[site,'BRA']
                global_flip = bra2.phase.get('global_flip',False)
                local_inds = bra2.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
                assert (bra2.data-bra1.data).norm()<1e-6 

                output_inds = norm_plq[site,'BRA'].inds[::-1]
                ovlp = overlap.copy()
                ovlp,b = contract(ovlp,site,output_inds)
                def A(x):
                    norm = norm_plq.copy()
                    norm[site,'KET'].modify(data=x)
                    norm,Ax = contract(norm,site,output_inds)
                    return norm['contract'].data
                data = BGMRES(A,norm_plq[site,'KET'].data,b)
                norm_plq[site,'KET'].modify(data=data)
#                data = norm_plq[site,'KET'].data.dagger
#                norm_plq[site,'BRA'].modify(data=data)
#                overlap[site,'BRA'].modify(data=data)
          
            cost_fid = overlap.contract(output_inds=[])
            cost_norm = norm_plq.contract(output_inds=[])
            cost = -cost_fid+cost_norm
            print('iteration={},cost={},norm={}'.format(i,cost,cost_norm))
#            print('')
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost
    return norm_plq

class FullUpdate(FullUpdate):
    def compute_energy(self):
        return self._psi.compute_local_expectation(self.ham.terms,
               **self.compute_energy_opts)
    def gate(self,G,where,**plaquette_env_options):
        layer_tags = 'KET','BRA' 
        canonize = True
        mode = 'mps'
        tags_plq = tuple(starmap(self._psi.site_tag, where))
        steps = 50
        tol = 1e-6

        # form env
        ovlp,_,self._bra = self._psi.make_norm(return_all=True,layer_tags=layer_tags)
        plaquette_env_options["max_bond"] = self.chi
        plaquette_env_options["cutoff"] = 0.0
        plaquette_env_options["canonize"] = canonize
        plaquette_env_options["mode"] = mode
        plaquette_env_options["layer_tags"] = layer_tags
        x_bsz,y_bsz = calc_plaquette_sizes([where], autogroup=True)[0]
        plaquette_envs = ovlp.compute_plaquette_environments(
            x_bsz=x_bsz, y_bsz=y_bsz, **plaquette_env_options)
        plaquette_map = calc_plaquette_map(plaquette_envs)
        env = plaquette_envs[plaquette_map[where]]

        # make TG and add to overlap
        overlap = env.copy()
        bra = self._bra
        if is_lone_coo(where):
            _where = (where,)
        else:
            _where = tuple(where)
        ng = len(_where)
        site_ix = [bra.site_ind(i, j) for i, j in _where]
#        bnds = [rand_uuid() for _ in range(ng)]
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

        # make initial guess
        print(where)
        print('initial energy:',self.compute_energy())
        # assert self._psi is phaseless
        self._psi.add_tag('KET')
        fs = self._psi.fermion_space
        for tid,(tsr,site) in fs.tensor_order.items():
            global_flip = tsr.phase.get('global_flip',False)
            local_inds = tsr.phase.get('local_inds',[])
            assert global_flip == False 
            assert len(local_inds) == 0
        ket_init = self._psi.copy()
        init_simple_guess = True
        condition_tensors = True
        condition_maintain_norms = True
        condition_balance_bonds = True
        if condition_tensors:
            plq = ket_init.select(tags_plq,which='any')
            conditioner(plq,balance_bonds=condition_balance_bonds)
            if condition_maintain_norms:
                pre_norm = plq[tags_plq[0]].norm()
        if init_simple_guess:
            ket_init.gate_(G,where,contract='reduce-split',max_bond=self.D)
            e1 = self.compute_energy()
            if condition_tensors:
                if condition_maintain_norms:
                    conditioner(plq,value=pre_norm,
                                balance_bonds=condition_balance_bonds)
                else:
                    conditioner(plq, balance_bonds=condition_balance_bonds)
            # make sure rel order of involved tsr doesn't change
            ket_init = maintain_order(ket_init,self._psi,tags_plq)
            for site in tags_plq:
                ket_init[site].phase = {}
            # assert: all tsrs are in the same order
            #         all tsrs are phaseless
            #         all uninvolved tsrs data are unchanged 
            for i in range(self._psi.Lx):
                for j in range(self._psi.Ly):
                    site = self._psi.site_tag(i,j)
                    tsr1 = ket_init[site]
                    tsr2 = self._psi[site]
                    order1 = tsr1.get_fermion_info()[1]
                    order2 = tsr2.get_fermion_info()[1]
                    global_flip1 = tsr1.phase.get('global_flip',False)
                    local_inds1 = tsr1.phase.get('local_inds',[])
                    global_flip2 = tsr2.phase.get('global_flip',False)
                    local_inds2 = tsr2.phase.get('local_inds',[])
                    assert order1 == order2
                    assert global_flip1 == False
                    assert global_flip2 == False
                    assert len(local_inds1) == 0 
                    assert len(local_inds2) == 0
                    if site not in tags_plq:
                        assert (tsr1.data-tsr2.data).norm()<1e-6
            e2 = self.compute_energy()
            assert abs(e2-e1)<tol
        norm_plq = env.copy()
        cost_fid = overlap.contract(output_inds=[])
        cost_norm = norm_plq.contract(output_inds=[])
        cost = -cost_fid+cost_norm
#        print('init cost={},norm_plq={}'.format(cost,cost_norm))

        norm_plq = maintain_order(norm_plq,self._psi,tags_plq,tol=1e-6)
        cost_norm1 = norm_plq.contract(output_inds=[])
        assert abs(cost_norm-cost_norm1)<1e-6

        norm_plq2 = norm_plq.copy()
        for site in tags_plq:
            norm_plq2[site,'KET'].modify(data=ket_init[site].data.copy())
        cost_norm2 = norm_plq2.contract(output_inds=[])
        cost2 = -cost_fid+cost_norm2
#        print('init cost={},norm_plq={}'.format(cost2,cost_norm2))
        if abs(cost2)<abs(cost):
            norm_plq = norm_plq2 

        norm_plq = gate_full_update_als(norm_plq,overlap,tags_plq,
                                    steps=steps,tol=tol,max_bond=self.D,
                                    optimize=self.contract_optimize)
        self._term_count += 1
        
        if condition_tensors:
            plq = norm_plq.select(tags_plq,which='any')
            plq = plq.select('KET',which='all')
            if condition_maintain_norms:
                conditioner(plq,value=pre_norm,balance_bonds=condition_balance_bonds)
            else:
                conditioner(plq,balance_bonds=condition_balance_bonds)
        norm_plq = maintain_order(norm_plq,self._psi,tags_plq)
        for site in tags_plq:
            self._psi[site].modify(data=norm_plq[site,'KET'].data.copy())
        print('gated energy:',self.compute_energy())
