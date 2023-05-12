
import random
from autoray import do, dag, conj, reshape
from itertools import product, cycle, starmap, combinations, count, chain

from ..utils import pairwise
from .tensor_2d_tebd import SimpleUpdate as SimpleUpdate_ 
from .tensor_core import Tensor, contract_strategy
def gen_bond_coos(psi):
    ls = []
    for i in range(psi.Lx):
        for j in range(psi.Ly):
            if j<psi.Ly-1:
                where = (i,j),(i,j+1)
            else:
                where = (i,0),(i,j)
            ls.append(where)

            if i<psi.Lx-1:
                where = (i,j),(i+1,j)
            else:
                where = (0,j),(i,j)
            ls.append(where)
    return ls
def gen_long_range_path(ij_a, ij_b, Lx,Ly, sequence=None):
    ia, ja = ij_a
    ib, jb = ij_b

    if ia==ib:
        assert jb==ja+1 or (ja==0 and jb==Ly-1)
        return ij_a,ij_b
    elif ja==jb:
        assert ib==ia+1 or (ia==0 and ib==Lx-1)
        return ij_a,ij_b
    
def nearest_neighbors(coo, Lx, Ly):
    i,j = coo
    return ((i-1)%Lx,j),(i,(j-1)%Ly),(i,(j+1)%Ly),((i+1)%Lx,j) 
class SimpleUpdate(SimpleUpdate_):
    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        # create the gauges like whatever data array is in the first site.
        data00 = next(iter(self._psi.tensor_map.values())).data

        self._gauges = dict()
        for ija, ijb in gen_bond_coos(self._psi):
            bnd = self._psi.bond(ija, ijb)
            d = self._psi.ind_size(bnd)
            Tsval = Tensor(
                do('ones', (d,), dtype=data00.dtype, like=data00),
                inds=[bnd],
                tags=[
                    self._psi.site_tag(*ija),
                    self._psi.site_tag(*ijb),
                    'SU_gauge',
                ]
            )
            self._gauges[tuple(sorted((ija, ijb)))] = Tsval
    def gate(self, U, where):
        """Like ``TEBD2D.gate`` but absorb and extract the relevant gauges
        before and after each gate application.
        """
        ija, ijb = where
        Lx, Ly =  self._psi.Lx,self._psi.Ly

        if callable(self.long_range_path_sequence):
            long_range_path_sequence = self.long_range_path_sequence(ija, ijb)
        else:
            long_range_path_sequence = self.long_range_path_sequence

        if self.long_range_use_swaps:
            path = tuple(gen_long_range_swap_path(
                ija, ijb, sequence=long_range_path_sequence))
            string = swap_path_to_long_range_path(path, ija)
        else:
            string = path = tuple(gen_long_range_path(
                ija, ijb, Lx, Ly, sequence=long_range_path_sequence))

        def env_neighbours(i, j):
            return tuple(filter(
                lambda coo: self._psi.valid_coo((coo)) and coo not in string,
                nearest_neighbors((i, j),Lx,Ly)
            ))

        # get the relevant neighbours for string of sites
        neighbours = {site: env_neighbours(*site) for site in string}

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval = self.gauges[tuple(sorted((site, neighbour)))]
                Tij.multiply_index_diagonal_(
                    ind=Tsval.inds[0], x=(Tsval.data + self.gauge_smudge))

        # absorb the inner bond gauges equally into both sites along string
        for site_a, site_b in pairwise(string):
            Ta, Tb = self._psi[site_a], self._psi[site_b]
            Tsval = self.gauges[tuple(sorted((site_a, site_b)))]
            bnd, = Tsval.inds
            Ta.multiply_index_diagonal_(ind=bnd, x=Tsval.data**0.5)
            Tb.multiply_index_diagonal_(ind=bnd, x=Tsval.data**0.5)

        # perform the gate, retrieving new bond singular values
        info = dict()
        #print(where)
        #print(path)
        #print(neighbours)
        #print()
        self._psi.gate_(U, where, absorb=None, info=info,
                        long_range_path_sequence=path, **self.gate_opts)

        # set the new singualar values all along the chain
        for site_a, site_b in pairwise(string):
            bond_pair = tuple(sorted((site_a, site_b)))
            s = info['singular_values', bond_pair]
            if self.gauge_renorm:
                # keep the singular values from blowing up
                s = s / s[0]
            Tsval = self.gauges[bond_pair]
            Tsval.modify(data=s)

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval = self.gauges[tuple(sorted((site, neighbour)))]
                Tij.multiply_index_diagonal_(
                    ind=Tsval.inds[0], x=(Tsval.data + self.gauge_smudge)**-1)
