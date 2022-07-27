import sys
this = sys.modules[__name__]
this.symmetry = 'u1'
this.flat = True
this.spinless = True
def set_options(symmetry='u1',flat=True,spinless=False):
    this.symmetry=symmetry
    this.flat=flat
    this.spinless=spinless
    
import numpy as np
if not this.spinless:
    from pyblock3.algebra.fermion_ops import (
        bonded_vaccum,
        creation,
        annihilation,
        ParticleNumber,
        onsite_U,
        Hubbard,
        get_exponential,
    )
    from pyblock3.algebra.fermion_encoding import get_state_map
    cre_a = creation(spin='a',symmetry=this.symmetry,flat=this.flat)
    cre_b = creation(spin='b',symmetry=this.symmetry,flat=this.flat)
    cre_sum = creation(spin='sum',symmetry=this.symmetry,flat=this.flat)
    ann_a = annihilation(spin='a',symmetry=this.symmetry,flat=this.flat)
    ann_b = annihilation(spin='b',symmetry=this.symmetry,flat=this.flat)
    ann_sum = annihilation(spin='sum',symmetry=this.symmetry,flat=this.flat)
    pn = ParticleNumber(symmetry=this.symmetry,flat=this.flat)
    pna = np.tensordot(cre_a,ann_a,axes=((-1,),(0,)))
    pnb = np.tensordot(cre_b,ann_b,axes=((-1,),(0,)))
    nanb = onsite_U(u=1.,symmetry=this.symmetry,flat=this.flat)
    
    data_map = {'cre_a':cre_a,'cre_b':cre_b,'ann_a':ann_a,'ann_b':ann_b,
                'cre_sum':cre_sum,'ann_sum':ann_sum,
                'pn':pn,'pna':pna,'pnb':pnb,'nanb':nanb}
    sign_a = (-1)**(cre_a.parity*ann_a.parity)
    sign_b = (-1)**(cre_b.parity*ann_b.parity)

else: 
    from .spinless import (
        bonded_vaccum,
        creation,
        Hubbard,
        get_exponential,
        get_state_map,
    ) 
    cre_a = creation(symmetry=this.symmetry,flat=this.flat)
    ann_a = cre_a.dagger
    pn = np.tensordot(cre_a,ann_a,axes=((-1,),(0,)))
    data_map = {'cre_a':cre_a,'ann_a':ann_a,'pn':pn,'pna':pn}
    sign_a = (-1)**(cre_a.parity*ann_a.parity)
    sign_b = None

##################################################################################
# product state generation 
##################################################################################
from itertools import product
from ..tensor_2d import PEPS
from .fermion_core import FermionTensor,FermionTensorNetwork
from .fermion_2d import FPEPS
from .utils import insert
def get_pattern(inds,ind_to_pattern_map):
    """
    make sure patterns match in input tensors, eg,
    --->A--->B--->
     i    j    k
    pattern for A_ij = +-
    pattern for B_jk = +-
    the pattern of j index must be reversed in two operands
    """
    inv_pattern = {"+":"-", "-":"+"}
    pattern = ""
    for ix in inds[:-1]:
        if ix in ind_to_pattern_map:
            ipattern = inv_pattern[ind_to_pattern_map[ix]]
        else:
            nmin = pattern.count("-")
            ipattern = "-" if nmin*2<len(pattern) else "+"
            ind_to_pattern_map[ix] = ipattern
        pattern += ipattern
    pattern += "+" # assuming last index is the physical index
    return pattern
def get_vaccum(Lx,Ly):
    state_map = get_state_map(this.symmetry)
    _, _, ish = state_map[0]

    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=2)
    ftn = FermionTensorNetwork([])
    ind_to_pattern_map = dict()
    for ix, iy in product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = get_pattern(T.inds,ind_to_pattern_map)
        #put vaccum at site (ix, iy) 
        vac = bonded_vaccum((ish,)*(T.ndim-1), pattern=pattern, 
                            symmetry=this.symmetry, flat=this.flat)
        new_T = FermionTensor(vac, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn
def add_particle(ftn,spin,sites,hole=False):
    cre = data_map['cre_'+spin].copy() 
    if hole:
        cre = cre.dagger
    for site in sites:
        ket = ftn[site]
        pix = ket.inds[-1]
        TG = FermionTensor(data=cre.copy(),inds=(pix,pix+'_'),left_inds=(pix,),
                           tags=ket.tags) 
        inds = ket.inds
        ket.reindex_({pix:pix+'_'})
        ket_tid,ket_site = ket.get_fermion_info()
        ftn = insert(ftn,ket_site+1,TG)
        ftn.contract_tags(ket.tags,which='all',output_inds=inds,inplace=True)
    return ftn
def get_half_filled_product_state(Lx,Ly):
    """
    helper function to generate initial guess from regular PEPS
    |psi> = \prod (|alpha> + |beta>) at each site
    this function only works for half filled case with U1 symmetry
    Note energy of this wfn is 0
    """
    ftn = get_vaccum(Lx,Ly)
    sites = [(i,j) for i in range(Lx) for j in range(Ly)]
    ftn = add_particle(ftn,spin='sum',sites=sites)
    return ftn
def get_product_state(Lx,Ly,sites_a=None,Na=0,sites_b=None,Nb=0):
    ftn = get_vaccum(Lx,Ly)
    if sites_a is None: 
        empty = [(i,j) for i in range(Lx) for j in range(Ly)]
        sites_a = []
        for i in range(Na):
            sites_a.append(empty.pop(np.random.randint(low=0,high=len(empty))))
    ftn = add_particle(ftn,spin='a',sites=sites_a)

    if sites_b is None:
        if Nb == 0:
            return ftn
        empty = [(i,j) for i in range(Lx) for j in range(Ly)]
        sites_b = []
        for i in range(Nb):
            sites_b.append(empty.pop(np.random.randint(low=0,high=len(empty))))
    ftn = add_particle(ftn,spin='b',sites=sites_a)
    return ftn
