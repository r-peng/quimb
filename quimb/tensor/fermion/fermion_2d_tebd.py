from itertools import product,starmap
import numpy as np
import scipy.sparse.linalg as spla
from autoray import do

from ...utils import pairwise
from ..tensor_core import contract_strategy,bonds
from ..tensor_2d_tebd import (
    SimpleUpdate,
    FullUpdate, 
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
from .fermion_core import (_get_gauge_location,
                           FermionSpace,
                           FermionTensor,
                           FermionTensorNetwork,
                           FTNLinearOperator)
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

from ..tensor_2d import (calc_plaquette_sizes,
                         calc_plaquette_map,
                         Rotator2D,
                         plaquette_to_sites)
from .fermion_2d import FermionTensorNetwork2D
def tensor_copy(tsr):
    new_tsr = FermionTensor(data=tsr.data.copy(),inds=tsr.inds,
                            tags=tsr.tags.copy(),left_inds=tsr.left_inds)
    new_tsr.avoid_phase = tsr.avoid_phase
    new_tsr.phase = tsr.phase.copy()
    return new_tsr
def copy(ftn,full=True):
    new_ftn = FermionTensorNetwork([])
    new_order = dict()
    for tid,tsr in ftn.tensor_map.items():
        site = tsr.get_fermion_info()[1]
        new_tsr = tensor_copy(tsr)
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
def insert(ftn,isite,T):
    fs = ftn.fermion_space
    fs.insert_tensor(isite,T,virtual=True)
    tid = T.get_fermion_info()[0]
    ftn.tensor_map[tid] = T 
    T.add_owner(ftn,tid)
    ftn._link_tags(T.tags,tid)
    ftn._link_inds(T.inds,tid)
    return ftn
def replace(ftn,isite,T):
    fs = ftn.fermion_space
    tid = fs.get_tid_from_site(isite)
    t = ftn.tensor_map.pop(tid)
    ftn._unlink_tags(t.tags,tid)
    ftn._unlink_inds(t.inds,tid)
    t.remove_owner(ftn)

    fs.replace_tensor(isite,T,tid=tid,virtual=True)    
    ftn.tensor_map[tid] = T 
    T.add_owner(ftn,tid)
    ftn._link_tags(T.tags,tid)
    ftn._link_inds(T.inds,tid)
    return ftn
def gate_full_update_als(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,rtol=1e-6,
    optimize='auto-hq',init_simple_guess=True,condition_tensors=True,
    condition_maintain_norms=True,condition_balance_bonds=True,atol=1e-10,
    solver='solve',dense=True,enforce_pos=False,pos_smudge=1e-6):
    print(tags_plq)

    norm_plq = copy(env)
    # move bra/ket plq to last/first positions and refactor
    original_ket = [norm_plq[site,'KET'] for site in tags_plq]
    original_bra = [norm_plq[site,'BRA'] for site in tags_plq]
    fs = norm_plq.fermion_space
    for i,t in enumerate(original_ket):
        t.reindex_({t.inds[-1]:t.inds[-1]+'_'})
        tid = t.get_fermion_info()[0]
        fs.move(tid,i)
    for i,t in enumerate(original_bra):
        tid = t.get_fermion_info()[0]
        fs.move(tid,len(fs.tensor_order)-1-i)
    tids = [t.get_fermion_info()[0] for t in original_ket+original_bra]
    norm_plq._refactor_phase_from_tids(tids)
#    for tid,(t,isite) in fs.tensor_order.items():
#        assert t is norm_plq.tensor_map[tid]
    if condition_tensors:
        ket_plq = norm_plq.select(tags_plq,which='any')
        ket_plq = ket_plq.select('KET',which='any')
        ket_plq.view_like_(ket)
        conditioner(ket_plq,balance_bonds=condition_balance_bonds)
        if condition_maintain_norms:
            pre_norm = ket_plq[tags_plq[0]].norm()

    site_ix = [t.inds[0] for t in original_bra]
    bnds    = [t.inds[-1] for t in original_ket]
    TG = FermionTensor(G.copy(),inds=site_ix+bnds, left_inds=site_ix,tags='G')
    norm_plq = insert(norm_plq,len(original_ket),TG)

#    cost_norm = norm_plq.contract()
#    assert cost_norm>0.0

    # qr
    outer_ket,inner_ket = [],[]
    outer_bra,inner_bra = [],[]
    bond_along = list(bonds(*original_ket))[0]
    tag_map = {'KET':'BRA'}
    for i,t in enumerate(original_ket):
        global_flip = t.phase.get('global_flip',False)
        local_inds  = t.phase.get('local_inds',[])
        assert global_flip==False
        assert local_inds==[]

        tq,tr = t.split(left_inds=None,right_inds=(bond_along,t.inds[-1]),
                        method='qr',get='tensors',absorb='right')
        outer_ket.append(tq)
        inner_ket.append(tr)

        T = tq.H
        ind_map = {ind:ind+'*' for ind in tq.inds}
        T.reindex_(ind_map)
        T.retag_(tag_map)
        outer_bra.append(T)

        T = tr.H
        ind_map = {ind:ind+'*' for ind in tr.inds[:-1]}
        ind_map.update({bnds[i]:site_ix[i]})
        T.reindex_(ind_map)
        T.retag_(tag_map)
        inner_bra.append(T)
    for tq,tr,t in zip(outer_ket,inner_ket,original_ket):
        isite = t.get_fermion_info()[1]
        norm_plq = replace(norm_plq,isite,tr)
        norm_plq = insert(norm_plq,isite+1,tq)
    for tq,tr,t in zip(outer_bra,inner_bra,original_bra):
        isite = t.get_fermion_info()[1]
        norm_plq = replace(norm_plq,isite,tq)
        norm_plq = insert(norm_plq,isite+1,tr)
    tids = []
    for i,tr in enumerate(inner_ket):
        tids.append(tr.get_fermion_info()[0])
        fs.move(tids[-1],i)
    norm_plq._refactor_phase_from_tids(tids)
    for tr in inner_ket+inner_bra:
        tr.add_tag('R')
    saved_ket_plq = norm_plq.select(tags_plq,which='any')
    saved_ket_plq = saved_ket_plq.select('KET',which='any')
    saved_ket_plq = copy(saved_ket_plq,full=False)

    norm_plq.select(('R','G'),which='!any').add_tag('N')
    norm_plq.contract_tags('N',inplace=True,optimize=optimize)
    norm_plq['N'].modify(tags={'N'})

    norm_plq.select(('N','BRA'),which='!any').add_tag('ket_plq')
    norm_plq.contract_tags('ket_plq',inplace=True,optimize=optimize)
    norm_plq['ket_plq'].modify(tags={'ket_plq'})

    overlap = copy(norm_plq)
    overlap.select(('N','ket_plq'),which='any').add_tag('B')
    overlap.contract_tags('B',inplace=True,optimize=optimize)
    overlap['B'].modify(tags={'B'})

    ix = [(list(bonds(outer_ket[i],inner_ket[i]))[0],site_ix[i]) 
         for i in range(len(tags_plq))] 
    norm_plq['ket_plq'].modify(tags={'ket_plq'})
    tid = norm_plq['ket_plq'].get_fermion_info()[0]
    norm_plq._refactor_phase_from_tids([tid])
    tl,tr = norm_plq['ket_plq'].split(left_inds=ix[1],right_inds=ix[0],
                    bond_ind=bond_along,get='tensors',absorb='both',
                    max_bond=max_bond)
    for i,t in enumerate([tr,tl]):
        inds = inner_ket[i].inds[:-1]+(site_ix[i],)
        t.transpose_(*inds)
        t.modify(tags=inner_ket[i].tags)
    isite = norm_plq['ket_plq'].get_fermion_info()[1]
    norm_plq = replace(norm_plq,isite,tr)
    norm_plq = insert(norm_plq,isite+1,tl)

#    cost_fid = overlap.contract()
#    assert abs(cost_fid-cost_norm)<atol
#    cost_norm = norm_plq.contract()
#    cost = -2.0*cost_fid+cost_norm
#    assert cost_norm>0.0
#    assert cost_fid>0.0

    tids = []
    for i,site in enumerate(tags_plq): 
        tids.append(norm_plq[site,'KET'].get_fermion_info()[0])
        fs.move(tids[-1],i) 
        tids.append(norm_plq[site,'BRA'].get_fermion_info()[0])
        fs.move(tids[-1],len(fs.tensor_order)-1-i) 
    norm_plq._refactor_phase_from_tids(tids)
    tids = []
    for i,site in enumerate(tags_plq):
        tids.append(overlap[site,'BRA'].get_fermion_info()[0])
        overlap.fermion_space.move(tids[-1],overlap.num_tensors-1-i)
    overlap._refactor_phase_from_tids(tids)
    for site in tags_plq:
        data = norm_plq[site,'KET'].data.dagger
        norm_plq[site,'BRA'].modify(data=data.copy())
        overlap[site,'BRA'].modify(data=data.copy()) 
 
#    cost_norm = norm_plq.contract()
#    cost_fid = overlap.contract()
#    cost = -2.0*cost_fid+cost_norm
#    print('cost={},norm={},fid={}'.format(cost,cost_norm,cost_fid))
#    assert cost_norm>0.0
#    assert cost_fid>0.0

    previous_cost = None
    bond_info = norm_plq[site,'KET'].data.get_bond_info(ax=-1,flip=False)
    I = eye(bond_info,flat=True)
    if solver in ('lsqr','lsmr'):
        solver_opts = dict(atol=tol,btol=tol)
    else:
        solver_opts = dict(tol=tol)
    solver = getattr(spla,solver)
    invert = True
    for step in range(steps):
        for i,site in enumerate(tags_plq):
            ket_tid = norm_plq[site,'KET'].get_fermion_info()[0]
            fs.move(ket_tid,0) 
            bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
            fs.move(bra_tid,len(fs.tensor_order)-1) 
            norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))
             
            tid = overlap[site,'BRA'].get_fermion_info()[0]
            overlap.fermion_space.move(tid,overlap.num_tensors-1)
            overlap._refactor_phase_from_tids([tid])

            norm = copy(norm_plq)
            norm[site,'KET'].reindex_({site_ix[i]:bnds[i]})
            lix = norm[site,'BRA'].inds[::-1]
            rix = norm[site,'KET'].inds 
            TI = FermionTensor(I.copy(),inds=(site_ix[i],bnds[i]),
                                        left_inds=[site_ix[i]])
            norm = insert(norm,1,TI)
            norm.select(tags_plq[1-i],which='any').add_tag('N')
            norm.contract_tags('N',inplace=True,optimize=optimize,
                               output_inds=lix[:-1]+rix[:-1])
            tid = norm[site,'BRA'].get_fermion_info()[0]
            norm._pop_tensor(tid,remove_from_fermion_space='end')
            tid = norm[site,'KET'].get_fermion_info()[0]
            norm._pop_tensor(tid,remove_from_fermion_space='front')
            dq = norm_plq[site,'KET'].data.dq
            N = FTNLinearOperator(norm,lix,rix,dq,optimize=optimize) 

            ovlp = copy(overlap)
            ovlp[tags_plq[1-i]].add_tag('B')
            ovlp.contract_tags('B',inplace=True,optimize=optimize,output_inds=lix)
            tid = ovlp[site,'BRA'].get_fermion_info()[0]
            ovlp._pop_tensor(tid,remove_from_fermion_space='end')
            b = N.constructor.tensor_to_vector(ovlp['B'].data)

            if invert:
                dim = len(b)
                mat = np.zeros((dim,dim))
                for j in range(dim):
                    vec = np.zeros(dim)
                    vec[j] = 1.0
                    mat[:,j] = N._matvec(vec)
                mat_inv = np.linalg.inv(mat)
                x = np.dot(mat_inv,b)
            else:
                x0 = N.constructor.tensor_to_vector(norm_plq[site,'KET'].data)
                x,info = solver(N,b,x0=x0,**solver_opts)

            x = N.constructor.vector_to_tensor(x,dq) 
            norm_plq[site,'KET'].modify(data=x.copy())
            xH = x.dagger
            norm_plq[site,'BRA'].modify(data=xH.copy())
            overlap[site,'BRA'].modify(data=xH.copy())

        cost_norm = norm_plq.contract()
        cost_fid = overlap.contract()
        cost = -2.0*cost_fid+cost_norm
        print('iteration={},cost={},norm={},fid={}'.format(
               step,cost,cost_norm,cost_fid))
        converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
        if previous_cost is not None:
            assert cost-previous_cost<tol*2.0
        if converged:
            break
        previous_cost = cost

    tids = []
    for i,site in enumerate(tags_plq): 
        tids.append(norm_plq[site,'KET'].get_fermion_info()[0])
        fs.move(tids[-1],i) 
        tids.append(norm_plq[site,'BRA'].get_fermion_info()[0])
        fs.move(tids[-1],len(fs.tensor_order)-1-i) 
    norm_plq._refactor_phase_from_tids(tids)

    for i,site in enumerate(tags_plq):
        saved_ket_plq[site,'R'].modify(data=norm_plq[site,'KET'].data.copy())
    for i,site in enumerate(tags_plq):
        saved_ket_plq.contract_tags(site,inplace=True,optimize=optimize)
        saved_ket_plq[site].transpose_(*original_ket[i].inds)
    if condition_tensors:
        if condition_maintain_norms:
            conditioner(saved_ket_plq,value=pre_norm,
                        balance_bonds=condition_balance_bonds)
        else:
            conditioner(saved_ket_plq,balance_bonds=condition_balance_bonds)
    tids = []
    for i,site in enumerate(tags_plq):
        tids.append(saved_ket_plq[site].get_fermion_info()[0])
        saved_ket_plq.fermion_space.move(tids[-1],i)
    saved_ket_plq._refactor_phase_from_tids(tids)
    for site in tags_plq:
        assert saved_ket_plq[site].inds[:-1]==ket[site].inds[:-1]
        ket[site].modify(data=saved_ket_plq[site].data) 
        bra[site].modify(data=saved_ket_plq[site].data.dagger)

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
#        print(self.compute_energy())
        self._term_count += 1
#        print('gated energy:',self.compute_energy())
        
