import itertools,time
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
from ..tensor_2d_vmc_test import Test as Test_
from .fermion_core import FermionTensor,FermionTensorNetwork
from scipy.optimize import optimize 
class Test(Test_):
    def __init__(self,amp_fac,ham,symmetry):
        super().__init__(amp_fac,ham)
        from pyblock3.algebra.fermion_encoding import get_state_map
        from pyblock3.algebra.fermion import eye
        state_map = get_state_map(symmetry)
        bond_info = {qlab:sh for qlab,_,sh in state_map.values()}
        self.eye_data = eye(bond_info)

    def _get_sector(self,ix_bra,ix_ket,ftn,contract):
        (cons_b,dq_b),size_b,site_bra = self.constructors[ix_bra]
        (cons_k,dq_k),size_k,site_ket = self.constructors[ix_ket]
    
        tsr = ftn[ftn.site_tag(*site_bra),'BRA']
        inds_bra = tsr.inds 
        tid = tsr.get_fermion_info()[0]
        Tb = ftn._pop_tensor(tid,remove_from_fermion_space='end')
         
        tsr = ftn[ftn.site_tag(*site_ket),'KET']
        inds_ket = tsr.inds 
        tid = tsr.get_fermion_info()[0]
        Tk = ftn._pop_tensor(tid,remove_from_fermion_space='front')
    
        skb = ftn.contract()
        ls = [None] * size_b
        for i in range(size_b):
            vec = np.zeros(size_b)
            vec[i] = 1.
            data = cons_b.vector_to_tensor(vec,dq_b).dagger
            Tbi = FermionTensor(data=data,inds=inds_bra)
            si = FermionTensorNetwork([skb,Tbi]).contract(output_inds=inds_ket[::-1])
            ls[i] = cons_k.tensor_to_vector(si.data.dagger)
        norm = None
        if contract:
            norm = FermionTensorNetwork([Tk,skb,Tb]).contract()
        return np.stack(ls,axis=0),norm
    def add_eye(self,ix,ftn,bra):
        site = self.flat2site(ix)
        tsr = ftn[ftn.site_tag(*site),'BRA']
        pix = bra.site_ind(*site)
        tsr.reindex_({pix:pix+'*'})
    
        I = FermionTensor(data=self.eye_data.copy(),inds=(pix+'*',pix))
        I = bra.fermion_space.move_past(I) 
        ftn.add_tensor(I)
        return ftn
    def _get_vector(self,ix,ftn,contract):
        (cons,_),_,site = self.constructors[ix]
        tsr = ftn[ftn.site_tag(*site),'KET']
        inds_ket = tsr.inds 
        tid = tsr.get_fermion_info()[0]
        Tk = ftn._pop_tensor(tid,remove_from_fermion_space='end')
        Nj = ftn.contract(output_inds=Tk.inds[::-1])
        norm = None
        if contract:
            norm = FermionTensorNetwork([Nj,Tk]).contract()
        return ix,cons.tensor_to_vector(Nj.data.dagger),norm
    def _get_gate_tn(self,where):
        peps = self.amp_fac.psi
        ftn,_,bra = peps.make_norm(return_all=True)
        _where = where
        ng = len(_where)
        pixs = [peps.site_ind(i, j) for i, j in _where]
        bnds = [pix+'*' for pix in pixs]
        TG = FermionTensor(self.ham.terms[where].copy(),inds=bnds+pixs)
        TG = bra.fermion_space.move_past(TG)
        ftn.add_tensor(TG)
        for i,site in enumerate(_where):
            tsr = ftn[peps.site_tag(*site),'BRA']
            tsr.reindex_({pixs[i]:bnds[i]})
        return ftn
