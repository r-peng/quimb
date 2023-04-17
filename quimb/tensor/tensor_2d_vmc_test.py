import numpy as np
import itertools,time
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
from .tensor_core import Tensor,TensorNetwork
class Test:
    def __init__(self,amp_fac,ham):
        self.amp_fac = amp_fac
        self.ham = ham
        self.x0 = self.amp_fac.get_x()
        self.n = len(self.x0)

        self.block_dict = amp_fac.block_dict
        self.constructors = amp_fac.constructors
        self.Lx,self.Ly = amp_fac.Lx,amp_fac.Ly
        self.flatten = amp_fac.flatten
        self.flat2site = amp_fac.flat2site
    def energy(self):
        self.E = np.ones(1)
        if RANK==0:
            self.E *= self.amp_fac.psi.compute_local_expectation(self.ham.terms,normalized=True) 
            print('energy=',self.E[0])
        COMM.Bcast(self.E,root=0)
        return self.E[0]
    def get_count_disp(self,n):
        batchsize,remain = n//SIZE,n%SIZE
        count = np.array([batchsize]*SIZE)
        if remain > 0:
            count[-remain:] += 1
        disp = [0]
        for batchsize in count[:-1]:
            disp.append(disp[-1]+batchsize)
        return count,disp
    def grad(self,eps=1e-6,root=0):
        x0 = self.x0
        n = self.n

        count,disp = self.get_count_disp(n)
        start = disp[RANK]
        stop = start + count[RANK]
        gi = np.zeros(n)
        self.g = np.zeros(n)
        for i in range(start,stop):
            xnew = x0.copy()
            xnew[i] += eps
            peps_new = self.amp_fac.vec2psi(xnew,inplace=False)
            Enew = peps_new.compute_local_expectation(self.ham.terms,normalized=True)
            gi[i] = (Enew - self.E[0])/eps
        COMM.Allreduce(gi,self.g,op=MPI.SUM)
        return self.g
    def _get_sector(self,ix_bra,ix_ket,tn,contract):
        shape_b,size_b,site_bra = self.constructors[ix_bra]
        shape_k,size_k,site_ket = self.constructors[ix_ket]
    
        tid = tuple(tn._get_tids_from_tags((tn.site_tag(*site_bra),'BRA'),which='all'))[0]
        Tb = tn._pop_tensor(tid)
        inds_bra = Tb.inds 

        tid = tuple(tn._get_tids_from_tags((tn.site_tag(*site_ket),'KET'),which='all'))[0]
        Tk = tn._pop_tensor(tid)
        inds_ket = Tk.inds 
    
        try:
            skb = tn.contract(output_inds=inds_bra+inds_ket)
        except ValueError:
            print(site_bra,site_ket,tn)
        norm = None
        if contract:
            norm = TensorNetwork([Tk,skb,Tb]).contract()
        return skb.data.reshape(size_b,size_k),norm
    def add_eye(self,ix,tn,bra):
        site = self.flat2site(ix)
        tsr = tn[tn.site_tag(*site),'BRA']
        pix = bra.site_ind(*site)
        tsr.reindex_({pix:pix+'*'})
    
        I = Tensor(data=np.eye(2),inds=(pix+'*',pix))
        tn.add_tensor(I)
        return tn
    def _psi_i_psi_j(self,ix):
        peps = self.amp_fac.psi
        ix_bra,ix_ket = self.pairs[ix]
        tn,_,bra = peps.make_norm(return_all=True)
        if ix_bra == ix_ket:
            tn = self.add_eye(ix_ket,tn,bra)
        data,norm = self._get_sector(ix_bra,ix_ket,tn,True) 
        return ix_bra,ix_ket,data/norm,norm
    def _get_vector(self,ix,tn,contract):
        shape,_,site = self.constructors[ix]
        tid = tuple(tn._get_tids_from_tags((tn.site_tag(*site),'KET'),which='all'))[0]
        Tk = tn._pop_tensor(tid)
        Nj = tn.contract(output_inds=Tk.inds)
        norm = None
        if contract:
            norm = TensorNetwork([Nj,Tk]).contract()
        return ix,Nj.data.flatten(),norm
    def _vmean(self,ix):
        peps = self.amp_fac.psi
        tn,_,bra = peps.make_norm(return_all=True)
        ix,vec,norm = self._get_vector(ix,tn,True)
        return ix,vec/norm,norm
    def S(self):
        nsite = self.Lx * self.Ly
        npair = nsite * nsite
        self.pairs = list(itertools.product(range(nsite),repeat=2))
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
        result = [self._psi_i_psi_j(ix) for ix in range(start,stop)]
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
            print('check s symmetry:',np.linalg.norm(s-s.T))
        COMM.Bcast(self.norm,root=0)
        return s
    def _get_gate_tn(self,where):
        peps = self.amp_fac.psi
        tn,_,bra = peps.make_norm(return_all=True)
        _where = where
        ng = len(_where)
        pixs = [peps.site_ind(i, j) for i, j in _where]
        bnds = [pix+'*' for pix in pixs]
        TG = Tensor(self.ham.terms[where].copy(),inds=bnds+pixs)
        tn.add_tensor(TG)
        for i,site in enumerate(_where):
            tsr = tn[peps.site_tag(*site),'BRA']
            tsr.reindex_({pixs[i]:bnds[i]})
        return tn
    def _psi_i_hi_psi_0(self,ix_):
        ix,where = self.Hi0_iter[ix_] 
        tn = self._get_gate_tn(where)
        ix,vec,_ = self._get_vector(ix,tn,False)
        return ix,vec        
    def _psi_i_hi_psi_j(self,ix_):
        ix_b,ix_k,where = self.Hij_iter[ix_] 
        tn = self._get_gate_tn(where)
        if ix_b==ix_k:
            site = self.flat2site(ix_k)
            if site not in where:
                self.add_eye(ix_k,tn,self.amp_fac.psi)
        data,_ = self._get_sector(ix_b,ix_k,tn,False)
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
        print('check hij symmetry=',np.linalg.norm(hij-hij.T))
        
        #tmp = np.outer(Hvecs,self.vmean)
        #tmp += tmp.T
        #hij += self.E[0] * np.outer(self.vmean,self.vmean) - tmp
        #return hij
        return hij - np.outer(self.vmean,Hvecs) - np.outer(hi0,self.vmean) 

