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
from ..tensor_2d import (calc_plaquette_sizes,
                         calc_plaquette_map,
                         Rotator2D,
                         plaquette_to_sites)
from ..tensor_core import contract_strategy
from .fermion_core import (FermionSpace,
                           FermionTensor,
                           FermionTensorNetwork,
                           FTNLinearOperator)
from .fermion_2d import FermionTensorNetwork2D
from ...utils import valmap
from itertools import starmap,product
from opt_einsum import shared_intermediates
import numpy as np
import scipy.sparse.linalg as spla
from .gmres import GMRES
from autoray import do
from pyblock3.algebra.fermion import eye,Constructor
def copy(ftn,full=True):
    new_ftn = FermionTensorNetwork([])
    new_order = dict()
    for tid,tsr in ftn.tensor_map.items():
        site = tsr.get_fermion_info()[1]
        new_tsr = FermionTensor(data=tsr.data.copy(),inds=tsr.inds,
                                tags=tsr.tags.copy(),left_inds=tsr.left_inds)
        new_tsr.avoid_phase = tsr.avoid_phase
        new_tsr.phase = tsr.phase.copy()

        # add to fs
        new_order[tid] = (new_tsr,site)
        # add to tn
        new_ftn.tensor_map[tid] = new_tsr
        new_tsr.add_owner(new_ftn,tid)
        new_ftn._link_tags(new_tsr.tags,tid)
        new_ftn._link_inds(new_tsr.inds,tid)
    new_fs = FermionSpace(tensor_order=new_order,virtual=True)
    if full:
        new_ftn.view_like_(ftn)
    return new_ftn
#@profile
def gate_full_update_als(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,rtol=1e-6,
    optimize='auto-hq',init_simple_guess=True,condition_tensors=True,
    condition_maintain_norms=True,condition_balance_bonds=True,atol=1e-10,
    solver='solve',dense=True,enforce_pos=False,pos_smudge=1e-6):
    e0 = ket.compute_local_expectation({where:G.copy()},max_bond=2*max_bond**2,
                                       normalized=True)
    ket1 = copy(ket)
    ket1.gate_(G,where,contract='reduce-split',max_bond=None)
    e1 = ket1.compute_local_expectation({where:G.copy()},max_bond=2*max_bond**2,
                                        normalized=True)

    def add(ftn,G,tags_plq,move_past=None,insert=None):
        site_ix = [ftn[site,'KET'].inds[-1] for site in tags_plq]
        bnds = [bd+'_' for bd in site_ix]
        TG = ftn[tags_plq[0],'KET'].__class__(G.copy(), 
             inds=site_ix+bnds, left_inds=site_ix)
        if move_past is not None:
            TG = move_past.fermion_space.move_past(TG)
        reindex_map = dict(zip(site_ix, bnds))
        tids = ftn._get_tids_from_inds(site_ix, which='any')
        for tid_ in tids:
            tsr = ftn.tensor_map[tid_]
            if 'KET' in tsr.tags:
                tsr.reindex_(reindex_map)
        if insert is None:
            ftn.add_tensor(TG, virtual=True)
        else:
            fs = ftn.fermion_space
            fs.insert(insert,TG,virtual=True)
            tid = TG.get_fermion_info()[0]
            ftn.tensor_map[tid] = TG 
            TG.add_owner(ftn,tid)
            ftn._link_tags(TG.tags,tid)
            ftn._link_inds(TG.inds,tid)
        return ftn
    def reorder(ftn,tags_plq):
        norm0 = ftn.contract()
        for i,site in enumerate(tags_plq):
            tid = ftn[site,'KET'].get_fermion_info()[0]
            ftn.fermion_space.move(tid,i)
            tid = ftn[site,'BRA'].get_fermion_info()[0]
            ftn.fermion_space.move(tid,ftn.num_tensors-1-i)
        norm1 = ftn.contract()
        if abs(norm1-norm0)>atol:
            print('0',norm0,ref)
            print('1',norm1,ref)
            exit()
        tids  = [ftn[site,'KET'].get_fermion_info()[0] for site in tags_plq]
        tids += [ftn[site,'BRA'].get_fermion_info()[0] for site in tags_plq]
        ftn._refactor_phase_from_tids(tids)
        norm1 = ftn.contract()
        if abs(norm1-norm0)>atol:
            print('2',norm1,ref)
            exit()
        return ftn

    print(tags_plq,G.parity)
    #for site in tags_plq: # for compute_envs_every=1
    #    tsr = env[site,'KET']
    #    tsite = tsr.get_fermion_info()[1]
    #    print(tsite,tsr.inds,tsr.phase)
    #print('######## before ################')
    #for tid,(tsr,tsite) in env.fermion_space.tensor_order.items():
    #    assert env.tensor_map[tid] is tsr
    #    print(tid,tsr.inds,tsr.phase)
    #print('######## after ################')
    #for tid,(tsr,tsite) in env.fermion_space.tensor_order.items():
    #    assert env.tensor_map[tid] is tsr
    #    print(tid,tsr.inds,tsr.phase)
    # make initial guess
    ket_init = copy(ket)
    if condition_tensors:
        ket_plq = ket_init.select(tags_plq,which='any')
        ket_plq.view_like_(ket)
        conditioner(ket_plq,balance_bonds=condition_balance_bonds)
        if condition_maintain_norms:
            pre_norm = ket_plq[tags_plq[0]].norm()
    if init_simple_guess:
        ket_plq.gate_(G,where,contract='reduce-split',max_bond=max_bond)
        e2 = ket_init.compute_local_expectation({where:G.copy()},max_bond=2*max_bond**2,
                                                 normalized=True)
        if condition_tensors:
            if condition_maintain_norms:
                conditioner(ket_plq,value=pre_norm,
                            balance_bonds=condition_balance_bonds)
            else:
                conditioner(ket_plq,balance_bonds=condition_balance_bonds)
    env = reorder(env,tags_plq)
    for site in tags_plq:
        assert (env[site,'BRA'].data-env[site,'KET'].data.dagger).norm()<atol 
    norm_plq = env
    overlap = copy(env)
    for site in tags_plq:
        norm_plq[site,'KET'].modify(data=ket[site].data.copy())
        norm_plq[site,'BRA'].modify(data=bra[site].data.copy())
    assert abs(norm_plq.contract()-1.0)<atol
    for site in tags_plq:
        norm_plq[site,'KET'].modify(data=ket_plq[site].data.copy())
        data = ket_plq[site].data.dagger
        norm_plq[site,'BRA'].modify(data=data.copy())
        overlap[site,'BRA'].modify(data=data.copy())
#    overlap = add(overlap,G,tags_plq,move_past=bra)
    overlap = add(overlap,G,tags_plq,insert=2)

#    assert overlap.contract()>0.0
#    assert norm_plq.contract()>0.0
#    if overlap.contract()<0.0:
#        print('ovlp',overlap.contract())
#        exit()
#    if norm_plq.contract()<0.0:
#        print('norm_plq',norm_plq.contract())
#        exit()

    def contract(ftn,site,output_inds):
        pop = ftn[site,'BRA']
        pop.add_tag('pop') 
        ctr = ftn.select((site,'BRA'), which='!all')
        ctr.add_tag('contract')
        ftn.contract_tags('contract',inplace=True,output_inds=output_inds,
                          optimize=optimize)
        assert ftn.num_tensors==2
        assert len(ftn.fermion_space.tensor_order)==2
        tid = pop.get_fermion_info()[0]
        ftn.fermion_space.move(tid,1) 
        ftn._refactor_phase_from_tids((tid,))
        return ftn, ftn['contract'].data
         
    cost_norm = norm_plq.contract()
    cost_fid = overlap.contract()
    cost = -2.0*cost_fid+cost_norm
    print('cost={},norm={},fid={}'.format(cost,cost_norm,cost_fid))
    assert cost_norm>0.0
    assert cost_fid>0.0

#    xs = dict()
#    x_previous = dict() 
    previous_cost = None
    my_gmres = False
#    my_gmres = True
    if not my_gmres: # use spla solver
        bond_info = norm_plq[site,'KET'].data.get_bond_info(ax=-1,flip=False)
        I = eye(bond_info,flat=True)
        if solver in ('lsqr','lsmr'):
            solver_opts = dict(atol=tol,btol=tol)
        elif solver == 'gmres':
            #solver_opts = dict(tol=tol,restart=50,maxiter=50)
            solver_opts = dict(tol=tol*1e-3,atol=tol*1e-3)
        else:
            solver_opts = dict(tol=tol)
        solver = getattr(spla,solver)
#    steps = 1
    info = dict()
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                cost_norm = norm_plq.contract()
                cost_fid = overlap.contract()
                assert cost_norm>0.0
                assert cost_fid>0.0
                norm_plq = reorder(norm_plq,[site])
                overlap  = reorder(overlap ,[site])
                norm0 = norm_plq.contract()
                if abs(norm0-cost_norm)>atol:
                    print('init norm',cost_norm)
                    print('after reorder norm',norm0)
                    exit()
                ovlp0 = overlap.contract()
                if abs(ovlp0-cost_fid)>atol:
                    print('init ovlp',cost_fid)
                    print('after reorder ovlp',ovlp0)
                    exit()
                assert (norm_plq[site,'BRA'].data-overlap[site,'BRA'].data).norm()<atol 
                assert (norm_plq[site,'KET'].data-norm_plq[site,'BRA'].data.dagger).norm()<atol 

                ovlp = copy(overlap)
#                for tid,(tsr,tsite) in ovlp.fermion_space.tensor_order.items():
#                    assert tsr is ovlp.tensor_map[tid]
#                    assert not(tsr is overlap.tensor_map[tid])
#                    #print(tid,tsite,tsr.inds)
                ovlp0 = ovlp.contract()
                if abs(ovlp0-cost_fid)>atol:
                    print('init ovlp',cost_fid)
                    print('copy ovlp0',ovlp0)
                    exit()
                ovlp,b = contract(ovlp,site,ovlp[site,'BRA'].inds[::-1])
                ovlp0 = ovlp.contract()
                if abs(ovlp0-cost_fid)>atol:
                    print('init ovlp',cost_fid)
                    print('after contract ovlp0',ovlp0)
                    ndim = len(b.shape)
                    axes = range(ndim-1,-1,-1),range(ndim)
                    ovlp0 = np.tensordot(overlap[site,'BRA'].data,b,axes=axes)
                    print('after contract ovlp0',ovlp0)
                    for tid,(tsr,tsite) in ovlp.fermion_space.tensor_order.items():
                        assert tsr is ovlp.tensor_map[tid]
                        #print(tid,tsite,tsr.inds,tsr.phase)
                    ovlp_ = copy(overlap)
                    ctr = ovlp_.select((site,'BRA'),which='!all') 
                    ctr.add_tag('ctr')
                    ovlp_.contract_tags('ctr',inplace=True)
                    ovlp0 = ovlp_.contract()
                    print('after contract ovlp0',ovlp0)
                    for tid,(tsr,tsite) in ovlp_.fermion_space.tensor_order.items():
                        assert tsr is ovlp_.tensor_map[tid]
                        #print(tid,tsite,tsr.inds,tsr.phase)
                    exit()

                norm = copy(norm_plq)
#                #print('########## norm #############')
#                for tid,(tsr,tsite) in norm.fermion_space.tensor_order.items():
#                    assert tsr is norm.tensor_map[tid]
#                    #print(tid,tsite,tsr.inds)
                norm0 = norm.contract()
                if abs(norm0-cost_norm)>atol:
                    print('init norm',cost_norm)
                    print('copy norm',norm0)
                    exit()
                if dense:
                    tns = norm.select(site,which='!any')
                    tns.add_tag('contract')
                    norm.contract_tags('contract',inplace=True,optimize=optimize)
#                    for tid,(tsr,tsite) in norm.fermion_space.tensor_order.items():
#                        assert tsr is norm.tensor_map[tid]
#                   #     print(tid,tsite,tsr.inds,tsr.data.pattern)
                    norm0 = norm.contract()
                    if abs(norm0-cost_norm)>atol:
                        print('init norm',cost_norm)
                        print('after contract norm',norm0)
                        print(norm[site,'KET'].inds)
                        print(norm[site,'BRA'].inds)
                        print(norm['contract'].inds)
                        for tid,(tsr,tsite) in norm.fermion_space.tensor_order.items():
                            assert tsr is norm.tensor_map[tid]
                            #print(tid,tsite,tsr.inds)
                        exit()
                if not my_gmres:
                    norm = add(norm,I,[site],insert=1)
#                    norm = add(norm,I,[site])
                    norm0 = norm.contract()
                    if abs(norm0-cost_norm)>atol:
                        print('after add norm',norm0)
                        exit()
                    #print('########## norm #############')
                    #for tid,(tsr,tsite) in norm.fermion_space.tensor_order.items():
                    #    print(tid,tsite,tsr.inds,tsr.data.pattern)
#                x0 = x_previous.get(site,b)
                x0 = norm[site,'KET'].data.copy()
                lix = norm[site,'BRA'].inds[::-1]
                rix = norm[site,'KET'].inds
                if my_gmres:
                    norm = reorder(norm,[site])
                    def A(x):
                        ftn = copy(norm)
                        ftn[site,'KET'].modify(data=x.copy())
                        ftn,Ax = contract(ftn,site,lix)
                        return Ax
                    x = GMRES(A,x0,b,tol=rtol,atol=atol)
                else:
                    tid = norm[site,'BRA'].get_fermion_info()[0]
                    norm._pop_tensor(tid,remove_from_fermion_space='end')
                    tid = norm[site,'KET'].get_fermion_info()[0]
                    norm._pop_tensor(tid,remove_from_fermion_space='front')
                    #print('########## norm #############')
                    #for tid,(tsr,tsite) in norm.fermion_space.tensor_order.items():
                    #    print(tid,tsite,tsr.inds,tsr.data.pattern)
                    dq = x0.dq
                    A = FTNLinearOperator(norm,lix,rix,dq,optimize=optimize)
                    con = A.constructor
                    x0_vec = con.tensor_to_vector(x0)
#                    x0_vec = None
                    b_vec = con.tensor_to_vector(b)
                    x_vec,info = solver(A,b_vec,x0=x0_vec,**solver_opts)
                    if info==0:
                        x = con.vector_to_tensor(x_vec,dq)
                    else:
                        x = x0.copy()
                        x_vec = x0_vec.copy()
                        print('info=',info)
#                    ndim = len(x0.shape)
#                    axes = range(ndim-1,-1,-1),range(ndim)
#                    #cost_fid = np.tensordot(x0.dagger,b,axes=axes) 
#                    cost_fid = np.dot(x0_vec,b_vec) 
#                    #Ax = con.vector_to_tensor(A(x0_vec),dq)
#                    #cost_norm = np.tensordot(x0.dagger,Ax,axes=axes)
#                    cost_norm = np.dot(x0_vec,A(x0_vec))
#                    print('fid  before',cost_fid)
#                    print('norm before',cost_norm)
#                    cost_fid = np.tensordot(x.dagger,b,axes=axes) 
#                    Ax = con.vector_to_tensor(A(x_vec),dq)
#                    cost_norm = np.tensordot(x.dagger,Ax,axes=axes)
#                    print('fid  after ',cost_fid)
#                    print('norm after ',cost_norm)

                norm_plq[site,'KET'].modify(data=x.copy())
                xH = x.dagger
                norm_plq[site,'BRA'].modify(data=xH.copy())
                overlap[site,'BRA'].modify(data=xH.copy())
#                xs[site] = x
          
            ndim = len(x.shape)
            axes = range(ndim-1,-1,-1),range(ndim)
            cost_fid = np.tensordot(xH,b,axes=axes)
            if my_gmres: 
                cost_norm = np.tensordot(xH,A(x),axes=axes)
            else:
                cost_norm = np.dot(x_vec,A(x_vec))
            cost = -2.0*cost_fid+cost_norm
            print('iteration={},cost={},norm={},fid={}'.format(
                   i,cost,cost_norm,cost_fid))
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if previous_cost is not None:
                assert cost-previous_cost<tol*2.0
            if converged:
                break
            previous_cost = cost
#            for site in tags_plq:
#                x_previous[site] = xs[site]

    if condition_tensors:
        plq = norm_plq.select(tags_plq,which='any')
        ket_plq = plq.select('KET',which='all')
        if condition_maintain_norms:
            conditioner(ket_plq,value=pre_norm,balance_bonds=condition_balance_bonds)
        else:
            conditioner(ket_plq,balance_bonds=condition_balance_bonds)
    norm_plq = reorder(norm_plq,tags_plq)
    for site in tags_plq:
        ket[site].modify(data=norm_plq[site,'KET'].data.copy())
        bra[site].modify(data=norm_plq[site,'BRA'].data.copy())
    e3 = ket.compute_local_expectation({where:G.copy()},max_bond=2*max_bond**2,
                                       normalized=True)
    assert e0<e2<e3<e1
#    print(e0)
#    print(e2)
#    print(e3)
#    print(e1)

class FullUpdate(FullUpdate):
    #@profile
    def compute_plaquette_environment(self,where):
        #print(where)
        #print(self.norm)
        x_bsz, y_bsz = calc_plaquette_sizes([where])[0]
        if x_bsz<y_bsz:
            env = self.compute_plaquette_environment_row_first(where,x_bsz,y_bsz)
        else:
            env = self.compute_plaquette_environment_col_first(where,x_bsz,y_bsz)
        if self.pre_normalize:
            nfactor = do('abs',env.contract(all,optimize=self.contract_optimize))
            self._psi.multiply_(nfactor**(-1/2),spread_over='all')
            self._bra.multiply_(nfactor**(-1/2),spread_over='all')

            n = self._psi.num_tensors
            n_missing = x_bsz*y_bsz
            tags_plq = tuple(starmap(self._psi.site_tag,where))
            tns = env.select(tags_plq,which='!any')
            tns.multiply_(nfactor**(n_missing/n-1),spread_over='all')
        return env
    def compute_plaquette_environment_row_first(self,where,x_bsz,y_bsz):
        i0 = min([i for (i,j) in where])
        j0 = min([j for (i,j) in where])
        norm, _, self._bra = self._psi.make_norm(return_all=True,
                                 layer_tags=('KET','BRA'))
        row_envs = norm.compute_row_environments(max_bond=self.chi,
                       layer_tags=('KET','BRA'))
        tn = FermionTensorNetwork((
             row_envs['bottom',i0],
             *[row_envs['mid',i] for i in range(i0,i0+x_bsz)],
             row_envs['top',i0+x_bsz-1],
             )).view_as_(FermionTensorNetwork2D,like=norm)
        col_envs = tn.compute_col_environments(
                   xrange=(max(i0-1,0),min(i0+x_bsz,self._psi.Lx-1)),
                   max_bond=self.chi,layer_tags=('KET','BRA'))
        env = FermionTensorNetwork((
              col_envs['left',j0],
              *[col_envs['mid',j] for j in range(j0,j0+y_bsz)],
              col_envs['right',j0+y_bsz-1]),check_collisions=False)
        return env
    def compute_plaquette_environment_col_first(self,where,x_bsz,y_bsz):
        i0 = min([i for (i,j) in where])
        j0 = min([j for (i,j) in where])
        norm, _, self._bra = self._psi.make_norm(return_all=True,
                                 layer_tags=('KET','BRA'))
        col_envs = norm.compute_col_environments(max_bond=self.chi,
                       layer_tags=('KET','BRA'))
        tn = FermionTensorNetwork((
             col_envs['left',j0],
             *[col_envs['mid',j] for j in range(j0,j0+y_bsz)],
             col_envs['right',j0+y_bsz-1],
             )).view_as_(FermionTensorNetwork2D,like=norm)
        row_envs = tn.compute_row_environments(
                   yrange=(max(j0-1,0),min(j0+y_bsz,self._psi.Ly-1)),
                   max_bond=self.chi,layer_tags=('KET','BRA'))
        env = FermionTensorNetwork((
               row_envs['bottom',i0],
               *[row_envs['mid',i] for i in range(i0,i0+x_bsz)],
               row_envs['top',i0+x_bsz-1]),check_collisions=False)
        return env
    def compute_energy(self):
#        plaquette_envs = dict()
#        for where in self.ham.terms.keys():
#            x_bsz, y_bsz = calc_plaquette_sizes([where])[0]
#            i0 = min([i for (i,j) in where])
#            j0 = min([j for (i,j) in where])
#            plaquette_envs[(i0,j0),(x_bsz,y_bsz)] = self.compute_plaquette_environment(where)
#        print(self._psi.compute_local_expectation(self.ham.terms,max_bond=self.chi,normalized=True,plaquette_envs=plaquette_envs))
#        print(self.state.compute_local_expectation(
#            self.ham.terms,
#            **self.compute_energy_opts,
#        ))
        #exit()
        return self.state.compute_local_expectation(
            self.ham.terms,
            **self.compute_energy_opts,
        )
    #@profile
    def gate(self,G,where):
        """Apply the gate ``G`` at sites where, using a fitting method that
        takes into account the current environment.
        """
        # get the plaquette containing ``where`` and the sites it contains -
        # these will all be fitted
        env = self.compute_plaquette_environment(where)

        tags_plq = tuple(starmap(self._psi.site_tag,where))
        sites = [self._psi[site].get_fermion_info()[1] for site in tags_plq]
        inds = np.argsort(np.array(sites))
        tags_plq = [tags_plq[ind] for ind in inds]

#        print(tags_plq)
#        for site in tags_plq: # for compute_envs_every=1
#            tsr = env[site,'KET']
#            tsite = tsr.get_fermion_info()[1]
#            print(tsite,tsr.inds,tsr.phase)
#        for tid,(tsr,tsite) in self._psi.fermion_space.tensor_order.items():
#            assert self._psi.tensor_map[tid] is tsr
#            print(tid,tsr.inds,tsr.phase)
#        print('initial energy:',self.compute_energy())
        gate_full_update_als(ket=self._psi,env=env,bra=self._bra,G=G,
                             where=where,tags_plq=tags_plq,max_bond=self.D,
                             optimize=self.contract_optimize,
                             condition_balance_bonds=self.condition_balance_bonds,
                             **self._gate_opts)
        self._term_count += 1
#        print('gated energy:',self.compute_energy())
        
