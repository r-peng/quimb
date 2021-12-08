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
from ..tensor_2d import (calc_plaquette_sizes,calc_plaquette_map,is_lone_coo,
                         plaquette_to_sites)
from ..tensor_core import rand_uuid,contract_strategy
from .fermion_core import _tensors_to_constructors,FTNLinearOperator
from itertools import starmap,product
from opt_einsum import shared_intermediates
import numpy as np
import scipy.sparse.linalg as spla
from autoray import do
#import functools
def restart(A,x0,b,max_space=10,tol=1e-6,atol=1e-10):
    ndim = len(x0.shape)
    axes = range(ndim-1,-1,-1),range(ndim)
    r0 = b-A(x0)
    beta = r0.norm()
    if beta<tol:
        return x0,beta
    Q = [r0/beta]
    H = np.zeros((max_space+1,max_space),dtype=np.float64)
    for j in range(max_space):
        q = A(Q[j])
        norm_q = q.norm()
        if norm_q<tol:
            break
        for l in range(j+1):
            H[l,j] = np.tensordot(Q[l].dagger,q,axes=axes)
#        for l in range(j+1):
            q = q-Q[l]*H[l,j]
            assert abs(np.tensordot(Q[l].dagger,q,axes=axes))<atol
        norm_q = q.norm()
        if norm_q<tol:
            break
        q = q/norm_q
        lhs1 = [abs(np.tensordot(q_.dagger,q,axes=axes)) for q_ in Q]
        if sum(lhs1)>atol*len(lhs1):
            print(sum(lhs1),atol)
            print('q',norm_q)
            print('check Q',lhs1)
            lhs2 = [abs(np.tensordot(q_.dagger,q_,axes=axes)-1.0) for q_ in Q]
            if sum(lhs2)>atol:
               print('check Q',lhs2)
            for k in range(j+1):
                for l in range(k):
                    ovlp = np.tensordot(Q[l].dagger,Q[k],axes=axes)
                    if abs(ovlp)>atol:
                        print('k,l',k,l,ovlp)
            exit()
        Q.append(q)
        H[j+1,j] = norm_q

    m = len(Q)-1
    H = H[:m+1,:m]
    for i in range(m):
        lhs = A(Q[i])
        rhs = sum([Q[j]*H[j,i] for j in range(m+1)])
        assert (lhs-rhs).norm()<atol
    # QR on H
    T,R = np.linalg.qr(H)
    y = np.dot(np.linalg.inv(R),T[0,:]*beta)
    x = x0+sum([Q[i]*y[i] for i in range(m)])
    norm_r = (b-A(x)).norm()
    
    perturb = np.random.rand(m)*tol
    y_ = y+perturb
    x_ = x0+sum(Q[i]*y_[i] for i in range(m))
    norm_r_ = (b-A(x_)).norm()
    assert norm_r-norm_r_<atol
    return x,norm_r
def GMRES(A,x0,b,max_space=10,max_iter=50,tol=1e-6,atol=1e-10):
    x = x0.copy()
    r_norm_old = (b-A(x)).norm()
    for i in range(max_iter):
        x,r_norm = restart(A,x,b,max_space=max_space,tol=tol,atol=atol)
        assert r_norm-r_norm_old<atol
        print('iter={},r_norm={}'.format(i,r_norm))
        if r_norm<tol:
            break
        r_norm_old = r_norm
    return x
def maintain_order(ftn,ref,tags_plq,layer_tag='KET',refactor=True):
    # make sure ftn & ref have the same rel order
    # always assumes ref to be phaseless
    ref_order = [ref[site].get_fermion_info()[1] for site in tags_plq]  
    order = [ftn[site,layer_tag].get_fermion_info()[1] for site in tags_plq] 
    tids  = [ftn[site,layer_tag].get_fermion_info()[0] for site in tags_plq] 
    if (ref_order[0]-ref_order[1])*(order[0]-order[1])<0:
        for i in [0,1]:
            ftn.fermion_space.move(tids[i],order[1-i])
    if refactor:
        ftn._refactor_phase_from_tids(tids)
    return ftn
def gate_full_update_als(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,rtol=1e-6,
    optimize='auto-hq',init_simple_guess=True,condition_tensors=True,
    condition_maintain_norms=True,condition_balance_bonds=True,atol=1e-10,
    solver='solve',dense=True,enforce_pos=False,pos_smudge=1e-6):
    layer_tags = 'KET','BRA' 

    # make full target
    overlap = env.copy()
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

    # make initial guess
    ket_init = ket.copy()
    if condition_tensors:
        plq = ket_init.select(tags_plq,which='any')
        conditioner(plq,balance_bonds=condition_balance_bonds)
        if condition_maintain_norms:
            pre_norm = plq[tags_plq[0]].norm()
    if init_simple_guess:
        ket_init.gate_(G,where,contract='reduce-split',max_bond=max_bond)
        if condition_tensors:
            if condition_maintain_norms:
                conditioner(plq,value=pre_norm,
                            balance_bonds=condition_balance_bonds)
            else:
                conditioner(plq,balance_bonds=condition_balance_bonds)
        # make sure rel order of involved tsr doesn't change
        ket_init = maintain_order(ket_init,ket,tags_plq,refactor=False)
        for site in tags_plq:
            ket_init[site].phase = {}
        # assert: all tsrs are in the same order
        #         all tsrs are phaseless
        #         all uninvolved tsrs data are unchanged 
        for i in range(ket.Lx):
            for j in range(ket.Ly):
                site = ket.site_tag(i,j)
                tsr1 = ket_init[site]
                tsr2 = ket[site]
                order1 = tsr1.get_fermion_info()[1]
                order2 = tsr2.get_fermion_info()[1]
                global_flip1 = tsr1.phase.get('global_flip',False)
                local_inds1  = tsr1.phase.get('local_inds',[])
                global_flip2 = tsr2.phase.get('global_flip',False)
                local_inds2  = tsr2.phase.get('local_inds',[])
                assert order1 == order2
                assert global_flip1 == False
                assert global_flip2 == False
                assert len(local_inds1) == 0 
                assert len(local_inds2) == 0
                if site not in tags_plq:
                    assert (tsr1.data-tsr2.data).norm()<atol
    norm_plq = env.copy()
    norm_plq = maintain_order(norm_plq,ket,tags_plq)
    for site in tags_plq:
        norm_plq[site,'KET'].modify(data=ket_init[site].data.copy())

    def contract(ftn,site,output_inds):
        pop = ftn[site,'BRA']
        pop.add_tag('pop') 
        ctr = ftn.select((site,'BRA'), which='!all')
        ctr.add_tag('contract')
        ftn.contract_tags('contract',inplace=True,output_inds=output_inds,
                          optimize=optimize)
        assert ftn.num_tensors==2
        tid = pop.get_fermion_info()[0]
        ftn._refactor_phase_from_tids((tid,))
        return ftn, ftn['contract'].data
        
    previous_cost = None
    cost_fid = overlap.contract(output_inds=[])
    cost_norm = norm_plq.contract(output_inds=[])
    previous_cost = -2.0*cost_fid+cost_norm
    print('init cost={},norm_plq={}'.format(previous_cost,cost_norm))
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
                bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
                norm_plq.fermion_space.move(ket_tid,0)
                norm_plq.fermion_space.move(bra_tid,norm_plq.num_tensors-1)
                norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))

                bra_tid = overlap[site,'BRA'].get_fermion_info()[0]
                overlap.fermion_space.move(bra_tid,overlap.num_tensors-1)
                overlap._refactor_phase_from_tids((bra_tid,))
                assert (norm_plq[site,'BRA'].data-overlap[site,'BRA'].data).norm()<atol 

                output_inds = norm_plq[site,'BRA'].inds[::-1]
                ovlp = overlap.copy()
                ovlp,b = contract(ovlp,site,output_inds)
                def A(x):
                    norm = norm_plq.copy()
                    norm[site,'KET'].modify(data=x.copy())
                    norm,Ax = contract(norm,site,output_inds)
                    return Ax
                data = GMRES(A,norm_plq[site,'KET'].data,b,tol=rtol,atol=atol)
                norm_plq[site,'KET'].modify(data=data)
#                data = norm_plq[site,'KET'].data.dagger
#                norm_plq[site,'BRA'].modify(data=data)
#                overlap[site,'BRA'].modify(data=data)
          
#            cost_fid = overlap.contract(output_inds=[])
            cost_norm = norm_plq.contract(output_inds=[])
            cost = -2.0*cost_fid+cost_norm
            print('iteration={},cost={},norm={}'.format(i,cost,cost_norm))
            print('')
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost

    if condition_tensors:
        plq = norm_plq.select(tags_plq,which='any')
        plq = plq.select('KET',which='all')
        if condition_maintain_norms:
            conditioner(plq,value=pre_norm,balance_bonds=condition_balance_bonds)
        else:
            conditioner(plq,balance_bonds=condition_balance_bonds)
    norm_plq = maintain_order(norm_plq,ket,tags_plq)
    for site in tags_plq:
        ket[site].modify(data=norm_plq[site,'KET'].data.copy())

class FullUpdate(FullUpdate):
    def _maybe_compute_plaquette_envs(self, force=False):
        """Compute and store the plaquette environments for all local terms.
        """
        # first check if we need to compute the envs
        if not self._need_to_recompute_envs() and not force:
            return

        if self.condition_tensors:
            conditioner(self._psi, balance_bonds=self.condition_balance_bonds)

        # useful to store the bra that went into making the norm
        norm, _, self._bra = self._psi.make_norm(return_all=True)

        envs = dict()
        for x_bsz, y_bsz in calc_plaquette_sizes(self.ham.terms):
            envs.update(norm.compute_plaquette_environments(
                x_bsz=x_bsz, y_bsz=y_bsz, max_bond=self.chi, cutoff=0.0, 
                layer_tags=('KET','BRA')))

        if self.pre_normalize:
            # get the first plaquette env and use it to compute current norm
#            p0, env0 = next(iter(envs.items()))
#            sites = plaquette_to_sites(p0)
#            tags_plq = tuple(starmap(norm.site_tag, sites))
#            norm_plq = norm.select_any(tags_plq) | env0
            p0,norm_plq = next(iter(envs.items()))

            # contract the local plaquette norm
            nfactor = do(
                'abs', norm_plq.contract(all, optimize=self.contract_optimize))

            # scale the bra and ket and each of the plaquette environments
            self._psi.multiply_(nfactor**(-1 / 2), spread_over='all')
            self._bra.multiply_(nfactor**(-1 / 2), spread_over='all')

            # scale the envs, taking into account the number of sites missing
            n = self._psi.num_tensors
            for ((_, _), (di, dj)), env in envs.items():
#                n_missing = di * dj
#                env.multiply_(nfactor ** (n_missing / n - 1),
#                              spread_over='all')
                env.multiply_(nfactor,spread_over='all')
#                print(env.contract(all,optimize=self.contract_optimize))

        self.plaquette_envs = envs
        self.plaquette_mapping = calc_plaquette_map(envs)

        self._env_n = self._n
        self._env_group_count = self._group_count
        self._env_term_count = self._term_count

    def gate_simple(self,G,where,**plaquette_env_options):
        layer_tags = 'KET','BRA' 
        canonize = True
        mode = 'mps'
        tags_plq = tuple(starmap(self._psi.site_tag, where))
        steps = 50
        tol = 1e-6

        inplace = True
        print(where)
        print('initial energy:',self.compute_energy())
        if not inplace:
            fs = self._psi.fermion_space
            for tid,(tsr,site) in fs.tensor_order.items():
                global_flip = tsr.phase.get('global_flip',False)
                local_inds = tsr.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
        if inplace:
            ket_init = self._psi
        else:
            ket_init = self._psi.copy()
        condition_tensors = True
        condition_maintain_norms = True
        condition_balance_bonds = True
        if condition_tensors:
            plq = ket_init.select(tags_plq,which='any')
            conditioner(plq,balance_bonds=condition_balance_bonds)
            if condition_maintain_norms:
                pre_norm = plq[tags_plq[0]].norm()
        ket_init.gate_(G,where,contract='reduce-split',max_bond=self.D)
        e1 = self.compute_energy()
        if condition_tensors:
            if condition_maintain_norms:
                conditioner(plq,value=pre_norm,
                            balance_bonds=condition_balance_bonds)
            else:
                conditioner(plq, balance_bonds=condition_balance_bonds)
        if not inplace:
            # make sure rel order of involved tsr doesn't change
            ket_init = maintain_order(ket_init,self._psi,tags_plq,refactor=False)
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
        if not inplace:
            for site in tags_plq:
                self._psi[site].modify(data=ket_init[site].data.copy())
        print('gated energy:',self.compute_energy())
    def gate(self,G,where):
        """Apply the gate ``G`` at sites where, using a fitting method that
        takes into account the current environment.
        """
        # check if the new term commutes with those applied so far, this is to
        #     decide if we need to recompute the environments
        swhere = set(where)
        if self._current_group.isdisjoint(swhere):
            # if so add it to the grouping
            self._current_group |= swhere
        else:
            # else increment and reset the grouping
            self._current_group = swhere
            self._group_count += 1

        # get the plaquette containing ``where`` and the sites it contains -
        # these will all be fitted
        self._maybe_compute_plaquette_envs()
        plq = self.plaquette_mapping[tuple(sorted(where))]
        env = self.plaquette_envs[plq]
        tags_plq = tuple(starmap(self._psi.site_tag, plaquette_to_sites(plq)))

        print(where)
        print('initial energy:',self.compute_energy())
        gate_full_update_als(ket=self._psi,env=env,bra=self._bra,G=G,
                             where=where,tags_plq=tags_plq,max_bond=self.D,
                             optimize=self.contract_optimize,
                             condition_balance_bonds=self.condition_balance_bonds,
                             **self._gate_opts)
        self._term_count += 1
        print('gated energy:',self.compute_energy())
        
