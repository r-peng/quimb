import itertools
import numpy as np
def _remove(T,bd):
    idx = T.inds.index(bd)
    if idx==0:
        T.modify(data=T.data[0],inds=T.inds[1:])
        return T
    if idx==len(T.inds)-1:
        T.modify(data=T.data[...,0],inds=T.inds[:-1])
        return T
    data = np.zeros(T.shape[idx])
    data[0] = 1
    data = np.tensordot(T.data,data,axes=([idx],[0]))
    inds = T.inds[:idx] + T.inds[idx+1:]
    T.modify(data=data,inds=inds)
def _add(T,bd):
    data = np.tensordot(np.ones(1),T.data,axes=0)
    inds = (bd,) + T.inds
    T.modify(data=data,inds=inds)
def convert(peps,typ='v',fname=None):
    if typ=='v':
        irange = range(peps.Lx-1)
        jrange = range(peps.Ly)
        def get_next(i,j):
            return i+1,j
    else:
        irange = range(peps.Lx)
        jrange = range(peps.Ly-1)
        def get_next(i,j):
            return i,j+1
    for i,j in itertools.product(irange,jrange):
        if (i+j)%2==0:
            continue
        i2,j2 = get_next(i,j)
        T1,T2 = peps[i,j],peps[i2,j2]
        bonds = tuple(T1.bonds(T2))
        if len(bonds)==0:
            bd = f'I{i},{j}_I{i2},{j2}' 
            _add(T1,bd)
            _add(T2,bd)
        elif len(bonds)==1:
            bd = bonds[0]
            _remove(T1,bd)
            _remove(T2,bd)
        else:
            raise ValueError
    if fname is None:
        return peps
    import matplotlib.pyplot as plt
    #plt.rcParams.update({'font.size':16})
    fig,ax = plt.subplots(nrows=1,ncols=1)
    fix = {peps.site_tag(i,j):(i,j) for i,j in itertools.product(range(peps.Lx),range(peps.Ly))}
    peps.draw(show_inds=False,show_tags=False,fix=fix,ax=ax)
    #fig.subplots_adjust(left=0.15, bottom=0.15, right=0.99, top=0.95)
    fig.savefig(fname)
    return peps 
from .tensor_core import Tensor
from .tensor_2d_tebd import SimpleUpdate as SimpleUpdate_ 
from autoray import do, dag, conj, reshape
class SimpleUpdate(SimpleUpdate_):
    def _initialize_gauges(self):
        """Create unit singular values, stored as tensors.
        """
        # create the gauges like whatever data array is in the first site.
        data00 = next(iter(self._psi.tensor_map.values())).data

        self._gauges = dict()
        for ija, ijb in self._psi.gen_bond_coos():
            try:
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
            except ValueError:
                pass

        self._old_gauges = {key:val.data for key,val in self._gauges.items()}
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
        print(where,string)

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

            if self.print_conv:
                s_old = self._old_gauges[bond_pair]
                print(np.linalg.norm(s-s_old))
                self._old_gauges[bond_pair] = s

        # absorb the 'outer' gauges from these neighbours
        for site in string:
            Tij = self._psi[site]
            for neighbour in neighbours[site]:
                Tsval = self.gauges[tuple(sorted((site, neighbour)))]
                Tij.multiply_index_diagonal_(
                    ind=Tsval.inds[0], x=(Tsval.data + self.gauge_smudge)**-1)
