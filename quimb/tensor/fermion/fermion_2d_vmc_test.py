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
    def _s1_ix(self,ix):
        peps = self.amp_fac.psi
        ix_bra,ix_ket = self.pairs[ix]
        ftn,_,bra = peps.make_norm(return_all=True)
        if ix_bra == ix_ket:
            site = self.flat2site(ix_ket)
            tsr = ftn[ftn.site_tag(*site),'BRA']
            pix = peps.site_ind(*site)
            tsr.reindex_({pix:pix+'*'})
    
            I = FermionTensor(data=self.eye_data.copy(),inds=(pix+'*',pix))
            I = bra.fermion_space.move_past(I) 
            ftn.add_tensor(I)
        data,norm = self._get_sector(ix_bra,ix_ket,ftn,True) 
        return ix_bra,ix_ket,data/norm,norm
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
    def _vmean(self,ix):
        peps = self.amp_fac.psi
        ftn,_,bra = peps.make_norm(return_all=True)
        ix,vec,norm = self._get_vector(ix,ftn,True)
        return ix,vec/norm,norm
    def S(self,symmetry):
        nsite = self.Lx * self.Ly
        npair = nsite * nsite
        self.pairs = list(itertools.product(range(nsite),repeat=2))
        from pyblock3.algebra.fermion_encoding import get_state_map
        from pyblock3.algebra.fermion import eye
        state_map = get_state_map(symmetry)
        bond_info = {qlab:sh for qlab,_,sh in state_map.values()}
        self.eye_data = eye(bond_info)
        if RANK<nsite:
            result = self._vmean(RANK)
        else:
            result = None
        ls = COMM.gather(result,root=0)
        self.vmean = np.zeros_like(self.x0)
        if RANK==0:
            ls = ls[:nsite]
            vmean = [None] * nsite
            norm = 0.
            for ix,vec,normi in ls:
                vmean[ix] = vec
                norm += normi
            norm /= len(ls)
            self.vmean = np.concatenate(vmean)
        COMM.Bcast(self.vmean,root=0)

        count,disp = self.get_count_disp(npair)
        start = disp[RANK]
        stop = start + count[RANK]
        result = [self._s1_ix(ix) for ix in range(start,stop)]
        ls = COMM.gather(result,root=0)
        self.norm = np.ones(1)
        s = None
        if RANK==0:
            s = np.zeros((self.n,self.n))
            norm = 0.
            ct = 0
            for lsi in ls:
                for ix_bra,ix_ket,data,normij in lsi:
                    start_b,stop_b = self.block_dict[ix_bra]
                    start_k,stop_k = self.block_dict[ix_ket]
                    s[start_b:stop_b,start_k:stop_k] = data
                    ct += 1
                    norm += normij
            norm /= ct
            self.norm *= norm
            s -= np.outer(self.vmean,self.vmean)
        COMM.Bcast(self.norm,root=0)
        return s
    def _get_gate_ftn(self,where):
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
    def _psi_i_hi_psi_0(self,ix_):
        ix,where = self.Hi0_iter[ix_] 
        ftn = self._get_gate_ftn(where)
        ix,vec,_ = self._get_vector(ix,ftn,False)
        return ix,vec        
    def _psi_i_hi_psi_j(self,ix_):
        ix_b,ix_k,where = self.Hij_iter[ix_] 
        ftn = self._get_gate_ftn(where)
        data,_ = self._get_sector(ix_b,ix_k,ftn,False)
        return ix_b,ix_k,data 
    def H(self):
        ham_terms = self.ham.terms
        nsite = self.Lx * self.Ly
        n = self.n 

        ham_keys = list(ham_terms.keys())
        self.Hi0_iter = list(itertools.product(range(nsite),ham_keys))
        nHi0 = len(self.Hi0_iter)
        count,disp = self.get_count_disp(nHi0)
        start = disp[RANK]
        stop = start + count[RANK]
        result = [self._psi_i_hi_psi_0(ix) for ix in range(start,stop)]
        ls = COMM.gather(result,root=0)
        if RANK==0:
            Hvecs = [None] * nsite
            for lsi in ls:
                for ix,vec in lsi:
                    if Hvecs[ix] is None:
                        Hvecs[ix] = vec
                    else:
                        Hvecs[ix] += vec
            Hvecs = np.concatenate(Hvecs) / self.norm[0]
            hi0 = Hvecs - self.E[0] * self.vmean
            print('hi0 error:',np.linalg.norm(2.*hi0-self.g)/np.linalg.norm(self.g))
        
        self.Hij_iter = list(itertools.product(range(nsite),range(nsite),ham_keys))
        nHij = len(self.Hij_iter)
        count,disp = self.get_count_disp(nHij)
        start = disp[RANK]
        stop = start + count[RANK]
        result = [self._psi_i_hi_psi_j(ix) for ix in range(start,stop)]
        ls = COMM.gather(result,root=0)
        if RANK>0:
            return None

        hij = np.zeros((n,n))
        for lsi in ls:
            for ix_bra,ix_ket,data in lsi:
                start_b,stop_b = self.block_dict[ix_bra]
                start_k,stop_k = self.block_dict[ix_ket]
                hij[start_b:stop_b,start_k:stop_k] += data
        hij /= self.norm[0]
        
        #tmp = np.outer(Hvecs,self.vmean)
        #tmp += tmp.T
        #hij += self.E[0] * np.outer(self.vmean,self.vmean) - tmp
        #return hij
        return hij - np.outer(self.vmean,Hvecs) - np.outer(hi0,self.vmean) 
