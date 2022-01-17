from itertools import product,starmap
import functools
import numpy as np
import scipy.sparse.linalg as spla
from autoray import do
from opt_einsum import shared_intermediates

from ...utils import pairwise
from ...utils import progbar as Progbar
from ..tensor_core import contract_strategy,bonds
from ..tensor_2d_tebd import (
    SimpleUpdate,
    FullUpdate, 
    conditioner, 
    LocalHam2D
)
from ..tensor_2d import (
    gen_long_range_path, 
    nearest_neighbors, 
    gen_long_range_swap_path, 
    swap_path_to_long_range_path, 
    gen_2d_bonds,
    calc_plaquette_sizes,
    calc_plaquette_map,
    Rotator2D,
    plaquette_to_sites
)

from .block_interface import eye, Hubbard
from . import block_tools
from .fermion_core import (
    _get_gauge_location,
    FermionSpace,
    FermionTensor,
    FermionTensorNetwork,
    FTNLinearOperator
)
from .fermion_arbgeom_tebd import LocalHamGen
from .fermion_2d import FermionTensorNetwork2D

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

################## full update #########################
def parse_specific_gate_opts(strategy,fit_opts):
    gate_opts = {
        'tol': fit_opts['tol'],
        'steps': fit_opts['steps'],
        'init_simple_guess': fit_opts['init_simple_guess'],
        'condition_tensors': fit_opts['condition_tensors'],
        'condition_maintain_norms': fit_opts['condition_maintain_norms'],
    }
    gate_opts['solver'] = fit_opts['als_solver']
    gate_opts['dense'] = fit_opts['als_dense']
    gate_opts['enforce_pos'] = fit_opts['als_enforce_pos']
    gate_opts['pos_smudge'] = fit_opts['als_enforce_pos_smudge']
    return gate_opts
def init_benvs(ket,sweep,
    max_bond=None,cutoff=1e-10,canonize=True,dense=False,mode='mps',
    layer_tags=('KET','BRA'),compress_opts=None,**contract_boundary_opts):
    contract_boundary_opts['max_bond'] = max_bond
    contract_boundary_opts['cutoff'] = cutoff
    contract_boundary_opts['canonize'] = canonize
    contract_boundary_opts['mode'] = mode
    contract_boundary_opts['dense'] = dense
    contract_boundary_opts['layer_tags'] = layer_tags
    contract_boundary_opts['compress_opts'] = compress_opts
     
    benvs = {}
    norm,_,bra = ket.make_norm(return_all=True,layer_tags=layer_tags)
    if sweep[0]=='v':
        benvs['left',0] = FermionTensorNetwork([])
        norm.reorder(direction='col',layer_tags=layer_tags,inplace=True)
        benvs['mid',0] = norm.select(norm.col_tag(0)).copy() 
        norm.compute_right_environments(envs=benvs,**contract_boundary_opts)
    else:
        benvs['bottom',0] = FermionTensorNetwork([])
        norm.reorder(direction='row',layer_tags=layer_tags,inplace=True)
        benvs['mid',0] = norm.select(norm.row_tag(0)).copy() 
        norm.compute_top_environments(envs=benvs,**contract_boundary_opts)
    return benvs,bra
def compute_local_envs(bix,benvs,sweep,like,
    max_bond=None,cutoff=1e-10,canonize=True,dense=False,mode='mps',
    layer_tags=('KET','BRA'),compress_opts=None,**contract_boundary_opts):
    contract_boundary_opts['max_bond'] = max_bond
    contract_boundary_opts['cutoff'] = cutoff
    contract_boundary_opts['canonize'] = canonize
    contract_boundary_opts['mode'] = mode
    contract_boundary_opts['dense'] = dense
    contract_boundary_opts['layer_tags'] = layer_tags
    contract_boundary_opts['compress_opts'] = compress_opts
    
    envs = {}
    if sweep[0]=='v':
        envs['bottom',0] = FermionTensorNetwork([])
        tn = FermionTensorNetwork(
             (benvs['left',bix],benvs['mid',bix],benvs['right',bix])
             ).view_as_(FermionTensorNetwork2D,like=like)
        tn.reorder(direction='row',layer_tags=layer_tags,inplace=True)
        envs['mid',0] = tn.select(tn.row_tag(0)).copy() 
        tn.compute_top_environments(envs=envs,
            yrange=(max(bix-1,0),min(bix+1,tn.Ly-1)),**contract_boundary_opts)
    else:
        envs['left',0] = FermionTensorNetwork([])
        tn = FermionTensorNetwork(
             (benvs['bottom',bix],benvs['mid',bix],benvs['top',bix])
             ).view_as_(FermionTensorNetwork2D,like=like)
        tn.reorder(direction='col',layer_tags=layer_tags,inplace=True)
        envs['mid',0] = tn.select(tn.col_tag(0)).copy() 
        tn.compute_right_environments(envs=envs,
            xrange=(max(bix-1,0),min(bix+1,tn.Lx-1)),**contract_boundary_opts)
    return envs
def match_phase(data,T):
    global_flip = T.phase.get('global_flip',False)
    local_inds = T.phase.get('local_inds',[])
    data_new = data.copy()
    if global_flip:
        data_new._global_flip()
    if len(local_inds)>0:
        axes = [T.inds.index(ind) for ind in local_inds]
        data_new._local_flip(axes)
    return data_new
def update_envs(ix,bix,envs,benvs,ket,bra,sweep,like,
    max_bond=None,cutoff=1e-10,canonize=True,dense=False,mode='mps',
    layer_tags=('KET','BRA'),compress_opts=None,**contract_boundary_opts):
    contract_boundary_opts['max_bond'] = max_bond
    contract_boundary_opts['cutoff'] = cutoff
    contract_boundary_opts['canonize'] = canonize
    contract_boundary_opts['mode'] = mode
#    contract_boundary_opts['dense'] = dense
    contract_boundary_opts['layer_tags'] = layer_tags
    contract_boundary_opts['compress_opts'] = compress_opts

    if sweep[0]=='v':
        from_which,to_which = 'bottom','top'
        select_tag = ket.row_tag(ix)
        xrange,yrange = (ix-1,ix),(max(bix-1,0),min(bix+1,ket.Ly-1))
    else:   
        from_which,to_which = 'left','right'
        select_tag = ket.col_tag(ix)
        yrange,xrange = (ix-1,ix),(max(bix-1,0),min(bix+1,ket.Lx-1))

    envs.pop((to_which,ix))
#    for i in [0,1]:
#        tn = envs['mid',ix+i]
#        site = ket.site_tag(ix+i,bix) if sweep[0]=='v' else ket.site_tag(bix,ix+i)
#        data = match_phase(ket[site].data,tn[site,'KET'])
#        tn[site,'KET'].modify(data=data)
#        data = match_phase(bra[site].data,tn[site,'BRA'])
#        tn[site,'BRA'].modify(data=data)
#
#    tn = FermionTensorNetwork(
#         (envs[from_which,ix],
#          *[envs['mid',ix+i] for i in [0,1]],
#          envs[to_which,ix+1])
#         ).view_as_(FermionTensorNetwork2D,like=like)
#    if ix>0:
#        tn.contract_boundary_from_(xrange=xrange,yrange=yrange,
#            from_which=from_which,**contract_boundary_opts)
#    envs[from_which,ix+1] = tn.select(select_tag).copy()
    if sweep[0]=='v':
        direction,select_tag='col',ket.col_tag(bix)
    else:
        direction,select_tag='row',ket.row_tag(bix)
    norm,_,bra = ket.make_norm(return_all=True,layer_tags=layer_tags)
    norm.reorder(direction=direction,layer_tags=layer_tags,inplace=True)
    benvs['mid',bix] = norm.select(select_tag).copy() 
    if sweep[0]=='v':
#        envs['bottom',0] = FermionTensorNetwork([])
        tn = FermionTensorNetwork(
             (benvs['left',bix],benvs['mid',bix],benvs['right',bix])
             ).view_as_(FermionTensorNetwork2D,like=like)
#        tn.reorder(direction='row',layer_tags=layer_tags,inplace=True)
#        envs['mid',0] = tn.select(tn.row_tag(0)).copy() 
        tn.compute_bottom_environments(envs=envs,
            yrange=(max(bix-1,0),min(bix+1,tn.Ly-1)),**contract_boundary_opts)
    else:
#        envs['left',0] = FermionTensorNetwork([])
        tn = FermionTensorNetwork(
             (benvs['bottom',bix],benvs['mid',bix],benvs['top',bix])
             ).view_as_(FermionTensorNetwork2D,like=like)
#        tn.reorder(direction='col',layer_tags=layer_tags,inplace=True)
#        envs['mid',0] = tn.select(tn.col_tag(0)).copy() 
        tn.compute_left_environments(envs=envs,
            xrange=(max(bix-1,0),min(bix+1,tn.Lx-1)),**contract_boundary_opts)
    return
def update_benvs(bix,benvs,ket,bra,sweep,
    max_bond=None,cutoff=1e-10,canonize=True,dense=False,mode='mps',
    layer_tags=('KET','BRA'),compress_opts=None,**contract_boundary_opts):
    contract_boundary_opts['max_bond'] = max_bond
    contract_boundary_opts['cutoff'] = cutoff
    contract_boundary_opts['canonize'] = canonize
    contract_boundary_opts['mode'] = mode
    contract_boundary_opts['layer_tags'] = layer_tags
    contract_boundary_opts['compress_opts'] = compress_opts

    if sweep[0]=='v':
        from_which,to_which,direction = 'left','right','col'
        xrange,yrange = (0,ket.Lx-1),(bix-1,bix) 
        select_tag = ket.col_tag(bix)
    else:
        from_which,to_which,direction = 'bottom','top','row'
        yrange,xrange = (0,ket.Ly-1),(bix-1,bix) 
        select_tag = ket.row_tag(bix)
    if bix>0: 
        benvs.pop((to_which,bix-1))
    norm,_,bra = ket.make_norm(return_all=True,layer_tags=layer_tags)
    norm.reorder(direction=direction,layer_tags=layer_tags,inplace=True)
    benvs['mid',bix] = norm.select(select_tag).copy() 

    tn = FermionTensorNetwork(
         (benvs[from_which,bix],benvs['mid',bix],benvs[to_which,bix])
         ).view_as_(FermionTensorNetwork2D,like=ket)
    if bix>0:
        tn.contract_boundary_from_(xrange=xrange,yrange=yrange,
            from_which=from_which,**contract_boundary_opts)
    benvs[from_which,bix+1] = tn.select(select_tag).copy()
    return
def pre_normalize(fu,env,where):
    if fu.pre_normalize:
        nfactor = do('abs',env.contract(all,optimize=fu.contract_optimize))
        fu._psi.multiply_(nfactor**(-1/2),spread_over='all')
        fu._bra.multiply_(nfactor**(-1/2),spread_over='all')

        n = fu._psi.num_tensors
        n_missing = 2
        tags_plq = tuple(starmap(fu._psi.site_tag,where))
        tns = env.select(tags_plq,which='!any')
        tns.multiply_(nfactor**(n_missing/n-1),spread_over='all')
    return 
def vertical_sweep(fu,tau):
    Lx,Ly = fu._psi.Lx,fu._psi.Ly
    init_benvs_ = functools.partial(init_benvs,sweep='v',max_bond=fu.chi) 
    compute_local_envs_ = functools.partial(compute_local_envs,like=fu._psi,
                          sweep='v',max_bond=fu.chi)
    update_envs_ = functools.partial(update_envs,like=fu._psi,
                   sweep='v',max_bond=fu.chi)
    update_benvs_ = functools.partial(update_benvs,sweep='v',max_bond=fu.chi)

    col_envs,fu._bra = init_benvs_(ket=fu._psi)
    for j in range(Ly):
        row_envs = compute_local_envs_(bix=j,benvs=col_envs)
        for i in range(Lx):
            if i+1!=Lx:
                where = (i,j),(i+1,j)
                U = fu.ham.get_gate_expm(where,-tau) 
                env = FermionTensorNetwork(
                      (row_envs['bottom',i],
                       *[row_envs['mid',i+x] for x in [0,1]],
                       row_envs['top',i+1]),
                      check_collisions=False)
                pre_normalize(fu,env,where) 
                fu.gate(U,where,env=env)
                update_envs_(ix=i,bix=j,envs=row_envs,benvs=col_envs,
                             ket=fu._psi,bra=fu._bra)
        update_benvs_(bix=j,benvs=col_envs,ket=fu._psi,bra=fu._bra)
    return
def horizontal_sweep(fu,tau):
    Lx,Ly = fu._psi.Lx,fu._psi.Ly
    init_benvs_ = functools.partial(init_benvs,sweep='h',max_bond=fu.chi) 
    compute_local_envs_ = functools.partial(compute_local_envs,like=fu._psi,
                          sweep='h',max_bond=fu.chi)
    update_envs_ = functools.partial(update_envs,like=fu._psi,
                   sweep='h',max_bond=fu.chi)
    update_benvs_ = functools.partial(update_benvs,sweep='h',max_bond=fu.chi)

    row_envs,fu._bra = init_benvs_(ket=fu._psi)
    for i in range(Lx):
        col_envs = compute_local_envs_(bix=i,benvs=row_envs)
        for j in range(Ly):
            if j+1!=Ly:
                where = (i,j),(i,j+1)
                U = fu.ham.get_gate_expm(where,-tau) 
                env = FermionTensorNetwork(
                      (col_envs['left',j],
                       *[col_envs['mid',j+y] for y in [0,1]],
                       col_envs['right',j+1]),
                      check_collisions=False)
                pre_normalize(fu,env,where) 
                fu.gate(U,where,env=env)
                update_envs_(ix=j,bix=i,envs=col_envs,benvs=row_envs,
                             ket=fu._psi,bra=fu._bra)
        update_benvs_(bix=i,benvs=row_envs,ket=fu._psi,bra=fu._bra)
    return
def sweep(fu,tau):
    vertical_sweep(fu,tau)
    horizontal_sweep(fu,tau)
    return
def evolve(fu,steps,tau=None,progbar=None,thresh=None):
    if tau is not None:
        fu.tau = tau
    if progbar is None:
        progbar = fu.progbar
    pbar = Progbar(total=steps, disable=fu.progbar is not True)
    try:
        for i in range(steps):
            # anything required by both energy and sweep
            fu.presweep(i)
            # possibly compute the energy
            should_compute_energy = (
                bool(fu.compute_energy_every) and
                (i % fu.compute_energy_every == 0))
            if should_compute_energy:
                fu._check_energy()
                fu._update_progbar(pbar)
                if len(fu.energies)>1 and thresh is not None:
                    if fu.energies[-2]-fu.energies[-1]<thresh:
                        break
            # actually perform the gates
            sweep(fu,fu.tau)
            fu._n += 1
            pbar.update()
            if fu.callback is not None:
                if fu.callback(self):
                    break
        # possibly compute the energy
        if fu.compute_energy_final:
            fu._check_energy()
            fu._update_progbar(pbar)
    except KeyboardInterrupt:
        # allow the user to interupt early
        pass
    finally:
        pbar.close()

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
def gate_full_update_als_qr(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,rtol=1e-6,
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
    invert = False
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
            dim = len(b)

            if invert:
                mat = np.zeros((dim,dim))
                for j in range(dim):
                    vec = np.zeros(dim)
                    vec[j] = 1.0
                    mat[:,j] = N._matvec(vec)
                mat_inv = np.linalg.inv(mat)
                x = np.dot(mat_inv,b)
            else:
                x0 = N.constructor.tensor_to_vector(norm_plq[site,'KET'].data)
                x = solver(N,b,x0=x0,**solver_opts)[0]

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
            if cost-previous_cost>0.0:
                break
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

def reorder(ftn,tags_plq):
    for i,site in enumerate(tags_plq):
        tid = ftn[site,'KET'].get_fermion_info()[0]
        ftn.fermion_space.move(tid,i)
        tid = ftn[site,'BRA'].get_fermion_info()[0]
        ftn.fermion_space.move(tid,ftn.num_tensors-1-i)
    tids  = [ftn[site,'KET'].get_fermion_info()[0] for site in tags_plq]
    tids += [ftn[site,'BRA'].get_fermion_info()[0] for site in tags_plq]
    ftn._refactor_phase_from_tids(tids)
    return ftn
def add(ftn,G,tags_plq,tags=None):
    site_ix = [ftn[site,'KET'].inds[-1] for site in tags_plq]
    bnds = [bd+'_' for bd in site_ix]
    TG = FermionTensor(G.copy(),inds=site_ix+bnds,left_inds=site_ix,tags=tags)
    reindex_map = dict(zip(site_ix, bnds))
    tids = ftn._get_tids_from_inds(site_ix, which='any')
    for tid_ in tids:
        tsr = ftn.tensor_map[tid_]
        if 'KET' in tsr.tags:
            tsr.reindex_(reindex_map)
    ftn = insert(ftn,len(tags_plq),TG)
    return ftn
def gate_full_update_als_site(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,
    rtol=1e-6,
    optimize='auto-hq',init_simple_guess=True,condition_tensors=True,
    condition_maintain_norms=True,condition_balance_bonds=True,atol=1e-10,
    solver='solve',dense=True,enforce_pos=False,pos_smudge=1e-6):
    print(tags_plq)
    norm_plq = copy(env)
    norm_plq = reorder(norm_plq,tags_plq)
    ket_plq = norm_plq.select(tags_plq,which='any')
    ket_plq = ket_plq.select('KET',which='all')
    if condition_tensors:
        conditioner(ket_plq,balance_bonds=condition_balance_bonds)
        if condition_maintain_norms:
            pre_norm = ket_plq[tags_plq[0]].norm()
    if init_simple_guess:
        ket_plq = copy(ket_plq,full=False).view_like_(ket)
        ket_plq.gate_(G,where,contract='reduce-split',max_bond=max_bond)
        if condition_tensors:
            if condition_maintain_norms:
                conditioner(ket_plq,value=pre_norm,
                            balance_bonds=condition_balance_bonds)
            else:
                conditioner(ket_plq,balance_bonds=condition_balance_bonds)
        tids = []
        for i,site in enumerate(tags_plq):
            tids.append(ket_plq[site].get_fermion_info()[0])
            ket_plq.fermion_space.move(tids[-1],i)
        ket_plq._refactor_phase_from_tids(tids)
    overlap = copy(norm_plq)
    for site in tags_plq:
        norm_plq[site,'KET'].modify(data=ket_plq[site].data.copy())
        data = ket_plq[site].data.dagger
        norm_plq[site,'BRA'].modify(data=data.copy())
        overlap[site,'BRA'].modify(data=data.copy())
    overlap = add(overlap,G,tags_plq)
    cost_norm = norm_plq.contract()
    cost_fid = overlap.contract()
    cost = -2.0*cost_fid+cost_norm
    print('cost={},norm={},fid={}'.format(cost,cost_norm,cost_fid))
    assert cost_norm>0.0
    assert cost_fid>0.0
     
    previous_cost = None
    bond_info = norm_plq[site,'KET'].data.get_bond_info(ax=-1,flip=False)
    I = eye(bond_info,flat=True)
    if solver in ('lsqr','lsmr'):
        solver_opts = dict(atol=tol,btol=tol)
    elif solver == 'gmres':
        solver_opts = dict(tol=tol,atol=tol)
    else:
        solver_opts = dict(tol=tol)
    solver = getattr(spla,solver)
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                norm_plq = reorder(norm_plq,[site])
                overlap  = reorder(overlap ,[site])

                norm = copy(norm_plq)
                norm = add(norm,I,[site],tags='I') 
                lix = norm[site,'BRA'].inds[::-1]
                rix = norm[site,'KET'].inds 
                tid = norm[site,'BRA'].get_fermion_info()[0]
                norm._pop_tensor(tid,remove_from_fermion_space='end')
                tid = norm[site,'KET'].get_fermion_info()[0]
                norm._pop_tensor(tid,remove_from_fermion_space='front')
                if dense:
                    norm.select('I',which='!any').add_tag('N')
                    norm.contract_tags('N',inplace=True,optimize=optimize)
                dq = norm_plq[site,'KET'].data.dq
                N = FTNLinearOperator(norm,lix,rix,dq,optimize=optimize) 

                ovlp = copy(overlap)
                ovlp.select((site,'BRA'),which='!all').add_tag('B')
                ovlp.contract_tags('B',inplace=True,optimize=optimize,
                                   output_inds=lix)
                ovlp['B'].modify(tags={'B'})
                tid = ovlp[site,'BRA'].get_fermion_info()[0]
                ovlp._pop_tensor(tid,remove_from_fermion_space='end')
                b = N.constructor.tensor_to_vector(ovlp['B'].data)
                
                x0 = N.constructor.tensor_to_vector(norm_plq[site,'KET'].data)
                x = solver(N,b,x0=x0,**solver_opts)[0]
                x = N.constructor.vector_to_tensor(x,dq) 
                norm_plq[site,'KET'].modify(data=x.copy())
                xH = x.dagger
                norm_plq[site,'BRA'].modify(data=xH.copy())
                overlap[site,'BRA'].modify(data=xH.copy())

            cost_norm = norm_plq.contract()
            cost_fid = overlap.contract()
            cost = -2.0*cost_fid+cost_norm
            print('iteration={},cost={},norm={},fid={}'.format(
                   i,cost,cost_norm,cost_fid))
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if previous_cost is not None:
                if cost-previous_cost>0.0:
                    break
            if converged:
                break
            previous_cost = cost

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

class FullUpdate(FullUpdate):
    def sweep(self,tau):
        Lx,Ly = self._psi.Lx,self._psi.Ly
        ordering = []
        for j in range(Ly):
            for i in range(Lx):
                if i+1!=Lx:
                    where = (i,j),(i+1,j)
                    ordering.append(where)
        for i in range(Lx):
            for j in range(Ly):
                if j+1!=Ly:
                    where = (i,j),(i,j+1)
                    ordering.append(where)
        assert len(ordering)==len(self.ham.terms)
        for where in ordering:
            U = self.ham.get_gate_expm(where,-tau)
            self.gate(U, where)
    def evolve(self, steps, tau=None, progbar=None, thresh=None):
        """Evolve the state with the local Hamiltonian for ``steps`` steps with
        time step ``tau``.
        """
        if tau is not None:
            self.tau = tau
        if progbar is None:
            progbar = self.progbar
        pbar = Progbar(total=steps, disable=self.progbar is not True)
        try:
            for i in range(steps):
                # anything required by both energy and sweep
                self.presweep(i)
                # possibly compute the energy
                should_compute_energy = (
                    bool(self.compute_energy_every) and
                    (i % self.compute_energy_every == 0))
                if should_compute_energy:
                    self._check_energy()
                    self._update_progbar(pbar)
                    if len(self.energies)>1 and thresh is not None:
                        if self.energies[-2]-self.energies[-1]<thresh:
                            break
                # actually perform the gates
                self.sweep(self.tau)
                self._n += 1
                pbar.update()
                if self.callback is not None:
                    if self.callback(self):
                        break
            # possibly compute the energy
            if self.compute_energy_final:
                self._check_energy()
                self._update_progbar(pbar)
        except KeyboardInterrupt:
            # allow the user to interupt early
            pass
        finally:
            pbar.close()
    #@profile
    @property
    def fit_strategy(self):
        return self._fit_strategy
    @fit_strategy.setter
    def fit_strategy(self, fit_strategy):
        self._gate_fit_fn = {
            'qr': gate_full_update_als_qr,
            'site': gate_full_update_als_site,
        }[fit_strategy]
        self._fit_strategy = fit_strategy
    def presweep(self, i):
        """Full update presweep - compute envs and inject gate options.
        """
        # inject the specific gate options required (do
        # here so user can change options between sweeps)
        self._gate_opts = parse_specific_gate_opts(
            self.fit_strategy, self.fit_opts)

        # keep track of number of gates applied, and commutative groups
        self._term_count = 0
        self._group_count = 0
        self._current_group = set()
    def compute_plaquette_environment(self,where):
        x_bsz, y_bsz = calc_plaquette_sizes([where])[0]
        i0 = min([i for (i,j) in where])
        j0 = min([j for (i,j) in where])
        norm,_,self._bra = self._psi.make_norm(return_all=True,
                           layer_tags=('KET','BRA'))
        if x_bsz<y_bsz:
#            plq_envs = norm._compute_plaquette_environments_row_first(
#                       x_bsz,y_bsz,max_bond=self.chi,layer_tags=('KET','BRA')) 
            env = self.compute_plaquette_environment_row_first(where,x_bsz,y_bsz)
        else:
#            plq_envs = norm._compute_plaquette_environments_col_first(
#                       x_bsz,y_bsz,max_bond=self.chi,layer_tags=('KET','BRA')) 
            env = self.compute_plaquette_environment_col_first(where,x_bsz,y_bsz)
#        env = plq_envs[(i0,j0),(x_bsz,y_bsz)]
        pre_normalize(self,env,where) 
#        if self.pre_normalize:
#            nfactor = do('abs',env.contract(all,optimize=self.contract_optimize))
#            self._psi.multiply_(nfactor**(-1/2),spread_over='all')
#            self._bra.multiply_(nfactor**(-1/2),spread_over='all')
#
#            n = self._psi.num_tensors
#            n_missing = x_bsz*y_bsz
#            tags_plq = tuple(starmap(self._psi.site_tag,where))
#            tns = env.select(tags_plq,which='!any')
#            tns.multiply_(nfactor**(n_missing/n-1),spread_over='all')
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
        return self.state.compute_local_expectation(
            self.ham.terms,
            **self.compute_energy_opts,
        )
    #@profile
    def gate(self,G,where,env=None):
        """Apply the gate ``G`` at sites where, using a fitting method that
        takes into account the current environment.
        """
        if env is None:
            env = self.compute_plaquette_environment(where)

        tags_plq = tuple(starmap(self._psi.site_tag,where))
        sites = [self._psi[site].get_fermion_info()[1] for site in tags_plq]
        inds = np.argsort(np.array(sites))
        tags_plq = [tags_plq[ind] for ind in inds]

        self._gate_fit_fn(ket=self._psi,env=env,bra=self._bra,G=G,
                          where=where,tags_plq=tags_plq,max_bond=self.D,
                          optimize=self.contract_optimize,
                          condition_balance_bonds=self.condition_balance_bonds,
                          **self._gate_opts)
        self._term_count += 1
        
