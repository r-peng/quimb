import numpy as np
import itertools,torch,h5py
import scipy.optimize
from pyblock3.algebra.fermion_ops import max_entangled_state,gauss
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid
from .fermion_1d_vmc import FMPS 
from ..tensor_core import Tensor,TensorNetwork
np.set_printoptions(suppress=True,linewidth=1000)
class SpinlessGaussianFPEPS: # for quadratic Hamiltonian only
    def __init__(self,L,M,Ne,pbc=False,occ_b=None,fix_bond=False):
        self.L = L
        self.Ne = Ne
        self.P = 1
        self.M = M
        self.occ_b = self.M if occ_b is None else occ_b 
        self.pbc = pbc
        self.fix_bond = fix_bond

        self.nsite = L
        self.Ptot = self.P * self.nsite
        self.nbond = self.nsite if pbc else self.nsite - 1 
        self.occ_v = self.occ_b * self.nbond
        self.Mtot = self.M * self.nbond * 2 
        print('occ_v=',self.occ_v)
        print('Mtot=',self.Mtot)

        self.get_site_map()
        self.get_Cin()

        self.config = None
        self.nit = 0
        self.terminate = False
        self.fname = None
    def get_site_map(self):
        self.site_map = [None] * self.L 
        self.nparam = 0
        start = 0
        for i in range(self.L):
            nleg = 2 
            if (not self.pbc) and (i==0 or i==self.L-1):
                nleg -= 1
            size = nleg * self.M
            self.site_map[i] = {'start_v':start,'size_v':size,'nleg':nleg}

            start += size
            self.nparam += (size + self.P)**2
        assert self.Mtot == start
        print('nparam=',self.nparam)
        print(self.site_map)
        if self.fix_bond:
            return 
        self.bond_order = []
        self.bond_map = dict()

        site_range = range(self.L) if self.pbc else range(self.L-1)
        for i in site_range:
            where = i,(i+1)%self.L
            self.bond_order.append(where)
            self.bond_map[where] = len(self.bond_order) - 1

        self.nparam1 = self.nparam
        self.nparam2 = (2 * self.M)**2 * len(self.bond_order) 
        self.nparam = self.nparam1 + self.nparam2
    def get_Cin(self):
        if not self.fix_bond:
            return
        U = [] 
        u = 1./np.sqrt(2.) 

        site_range = range(self.L) if self.pbc else range(self.L-1)
        for i in site_range: 
            start1 = self.site_map[i]['start_v']
            ix1 = start1 if (not self.pbc) and i==0 else start1 + self.M # right bond for ix1

            ix2 = self.site_map[(i+1)%self.L]['start_v'] # left bond for ix2
            for mode in range(self.M):
                Ui = np.zeros(self.Mtot) 
                Ui[ix1+mode] = u 
                Ui[ix2+mode] = u 
                U.append(Ui)
        U = np.stack(U,axis=1)
                
        rho = np.dot(U,U.T)
        self.Cin = 2.*rho-np.eye(self.Mtot)
        assert self.occ_v == U.shape[1] 
        assert np.linalg.norm(np.eye(self.Mtot)-np.dot(self.Cin,self.Cin))<1e-6
        self.Cin = torch.tensor(self.Cin,requires_grad=False)
    def get_Q(self,sites=None):
        N = self.occ_v + self.Ne
        Q,R = N // self.nsite, N % self.nsite
        print('mean,remain=',Q,R)
        if sites is None:
            rng = np.random.default_rng()
            sites = rng.choice(self.nsite,size=R,replace=False)
            print('sites Q+1=',sites)
        Qmap = {ix:1 for ix in sites} 
        for i in range(self.L):
            size_v = self.site_map[i]['size_v']
            dim1 = size_v + self.P
            dim2 = Q + Qmap.get(i,0) 
            self.site_map[i]['Ushape'] = dim1,dim2
    def set_ham(self,h1,thresh=1e-5):
        assert h1.shape[0] == self.P * self.nsite # check size 
        assert np.linalg.norm(h1-h1.T)<1e-6 # check hermitian
        w = np.linalg.eigvalsh(h1)
        self.exact = w[:self.Ne].sum()
        self.thresh = thresh 
        print('diagonalization energy=',self.exact)
        h1 = torch.tensor(h1,requires_grad=False)
        def energy(rho):
            return torch.sum(rho * h1)
        self.energy = energy
    def get_K(self,x):
        if self.fix_bond:
            x1 = x
        else:
            x1,x2 = np.split(x,[self.nparam1])
     
        self.K = dict() 
        for i in range(self.L):
            dim,_ = self.site_map[i]['Ushape']
            K,x1 = np.split(x1,[dim**2])
            self.K[i] = K.reshape(dim,dim)
        if self.fix_bond:
            return

        dim = 2 * self.M
        size = dim ** 2
        for where in self.bond_order:
            K,x2 = np.split(x2,[size])
            self.K[where] = K.reshape(dim,dim) 
    def cut_U(self,U,i):
        dim1,dim2 = self.site_map[i]['Ushape']
        if self.config is None:
            return U[:,:dim2]
        ci = self.config[i]
        if ci==1:
            return U[:,:dim2]
        else:
            return U[:,-dim2:]
    def get_Ct(self):
        def cat(size1,start2,stop2,size2,data):
            ls = []
            if start2 > 0:
                ls.append(torch.zeros(size1,start2,requires_grad=False))
            ls.append(data)
            if stop2 < size2:
                ls.append(torch.zeros(size1,size2-stop2,requires_grad=False))
            return torch.cat(ls,dim=1)
            
        # Ct -> A,B,D
        A = [] 
        B = [] 
        D = [] 
        for i in range(self.L):
            K = torch.tensor(self.K[i],requires_grad=True)
            self.K[i] = K 
            U = self.cut_U(torch.linalg.matrix_exp(K-K.T),i)
            Ct = 2.*(U@U.t()) - torch.eye(U.size(0))

            site_map = self.site_map[i] 
            start_v = site_map['start_v']
            size_v = site_map['size_v'] 
            stop_v = start_v + size_v
            start_p = i * self.P
            stop_p = start_p + self.P

            A.append(cat(self.P,start_p,stop_p,self.Ptot,Ct[:self.P,:self.P]))
            B.append(cat(self.P,start_v,stop_v,self.Mtot,Ct[:self.P,self.P:]))
            D.append(cat(size_v,start_v,stop_v,self.Mtot,Ct[self.P:,self.P:]))
        A = torch.cat(A,dim=0)
        B = torch.cat(B,dim=0)
        D = torch.cat(D,dim=0)
        return A,B,D
    def add_Cin(self,D):
        if self.fix_bond:
            return D + self.Cin
        site_range = range(self.L) if self.pbc else range(self.L-1)
        for i in site_range: 
            start1 = self.site_map[i]['start_v']
            ix1 = start1 if (not self.pbc) and i==0 else start1 + self.M

            ix2 = self.site_map[(i+1)%self.L]['start_v']
            where = i,(i+1)%self.L
            K = torch.tensor(self.K[where],requires_grad=True)
            self.K[where] = K 
            U = torch.linalg.matrix_exp(K-K.T)[:,:self.occ_b]
            Cin = 2.*(U@U.t()) - torch.eye(2*self.M)

            D[ix1:ix1+self.M,ix1:ix1+self.M] += Cin[:self.M,:self.M]
            D[ix1:ix1+self.M,ix2:ix2+self.M] += Cin[:self.M,self.M:]
            D[ix2:ix2+self.M,ix1:ix1+self.M] += Cin[self.M:,:self.M]
            D[ix2:ix2+self.M,ix2:ix2+self.M] += Cin[self.M:,self.M:]
        return D
    def get_rho(self,x):
        self.get_K(x)
        A,B,D = self.get_Ct()
        # A,B,D -> Cout 
        Cout = A - B@torch.linalg.inv(self.add_Cin(D))@B.t() 
        return .5 * (Cout + torch.eye(Cout.size(0)))
    def get_grad(self):
        g1 = [None] * self.nsite
        for i in range(self.L): 
            dK = self.K[i].grad.detach().numpy() 
            g1[i] = dK.flatten()
        g1 = np.concatenate(g1)
        if self.fix_bond:
            return g1

        g2 = []
        for where in self.bond_order:
            dK = self.K[where].grad.detach().numpy()
            g2.append(dK.flatten())
        g2 = np.concatenate(g2)
        return np.concatenate([g1,g2])
    def fun(self,x):
        rho = self.get_rho(x)
        E = self.energy(rho)

        if self.terminate:
            g = np.zeros_like(x)
        else:
            E.backward()
            g = self.get_grad() 

        self.x = x
        self.g = g
        self.E = E.detach().numpy()
        self.K = None
        #print(self.E,self.g)
        return self.E,self.g
    def callback(self,xk=None):
        dE = self.exact-self.E
        print(f'niter={self.nit},E={self.E},dE={dE},xnorm={np.linalg.norm(self.x)},gnorm={np.linalg.norm(self.g)}') 
        self.nit += 1
        if self.fname is not None:
            f = h5py.File(self.fname+'.hdf5','w')
            f.create_dataset('x',data=self.x)
            f.close()
        if np.fabs(dE) < self.thresh:
            self.terminate = True
    def run(self,x0=None,method='bfgs',maxiter=100):
        x0 = np.random.rand(self.nparam) if x0 is None else x0
        result = scipy.optimize.minimize(
                     self.fun,
                     x0,
                     method=method,
                     jac=True,
                     callback=self.callback,
                     options={'maxiter':maxiter})
        return result['x']
    def get_fixed_bond_state(self,symmetry='u1',flat=True):
        bond_state1 = max_entangled_state('++',symmetry=symmetry,flat=flat)
        bond_state = bond_state1
        for mode in range(1,self.M):
            bond_state = np.tensordot(bond_state,bond_state1,axes=([],[]))
        left_idx = tuple([2*mode for mode in range(self.M)])
        U,_,V = bond_state.tensor_svd(left_idx)
        tn = FermionTensorNetwork([])
        site_range = range(self.L) if self.pbc else range(self.L-1)
        for i in self.site_order: 
            uix = tuple([f'I{i}_r{mode}' for mode in range(self.M)]) 
            bix = (rand_uuid(),)
            vix = tuple([f'I{(i+1)%L}_l{mode}' for mode in range(self.M)]) 
            utag,vtag = f'I{i}',f'I{(i+1)%L}'
            tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True) 
            tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True) 
        return tn 
    def get_variational_bond_state(self,symmetry='u1',flat=True):
        tn = FermionTensorNetwork([])
        left_inds = tuple(range(self.M))
        site_range = range(self.L) if self.pbc else range(self.L-1)
        for i in site_range: 
            K = self.K[i,(i+1)%self.L]
            U = scipy.linalg.expm(K-K.T)[:,:self.occ_b]
            pattern = ''.join(['+']*U.shape[0])
            data = gauss(pattern,U,symmetry=symmetry,flat=flat)
            U,_,V = data.tensor_svd(left_inds)

            uix = tuple([f'I{i}_r{mode}' for mode in range(self.M)]) 
            bix = (rand_uuid(),)
            vix = tuple([f'I{(i+1)%self.L}_l{mode}' for mode in range(self.M)]) 
            utag,vtag = f'I{i}',f'I{(i+1)%self.L}'
            tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True)
            tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True)
        return tn 
    def get_fpeps(self,x,symmetry='u1',flat=True):
        x = self.x if x is None else x
        self.get_K(x) 
        tn = FermionTensorNetwork([])
        for i in range(self.L): 
            K = self.K[i]
            U = self.cut_U(scipy.linalg.expm(K-K.T),i)
            pattern = ''.join(['+']*U.shape[0])
            data = gauss(pattern,U,symmetry=symmetry,flat=flat)
            data = data.transpose(tuple(range(1,data.ndim))+(0,))

            inds = []
            if self.pbc or i>0:
                inds += [f'I{i}_l{mode}' for mode in range(self.M)]  
            if self.pbc or i<self.L-1:
                inds += [f'I{i}_r{mode}' for mode in range(self.M)]  
            inds += [f'k{i}']
            tags = f'I{i}'
            tn.add_tensor(FermionTensor(data=data,inds=inds,tags=tags),virtual=True) 

        tnI = self.get_fixed_bond_state(symmetry=symmetry,flat=flat) if self.fix_bond else \
              self.get_variational_bond_state(symmetry=symmetry,flat=flat)
        tn.add_tensor_network(tnI.H,virtual=True)
        for i in range(self.L):
            tn.contract_tags(f'I{i}',inplace=True)
        tn.view_as_(FMPS,inplace=True,
                    site_tag_id='I{}',
                    L=self.L,
                    site_ind_id='k{}',
                    )
        print(tn)
        return tn
    def init(self,config,eps=1e-2):
        self.config = config
        x1 = [None] * self.nsite
        for i in range(self.L): 
            dim1,dim2 = self.site_map[i]['Ushape']
            K = np.random.rand(dim1,dim1)
            K[0,:] *= eps
            K[:,0] *= eps
            ix = self.site_map[i]['site_ix']
            x1[ix] = K.flatten()
        x1 = np.concatenate(x1)
        if self.fix_bond:
            return x1
        x2 = np.random.rand(self.nparam2)
        return np.concatenate([x1,x2])
def compute_local_expectation(fpeps,terms):
    norm,_,bra = fpeps.make_norm(return_all=True)
    n = norm.contract()
    e = 0.
    for where,data in terms.items():
        site_ix = [f'k{i}' for i in where]
        bnds = [rand_uuid() for _ in where]
        TG = FermionTensor(data.copy(),inds=site_ix+bnds,left_inds=site_ix) 
        TG = bra.fermion_space.move_past(TG)

        ei = norm.copy()
        for ix,i in enumerate(where):
            T = ei[f'I{i}','KET']
            T.reindex_({site_ix[ix]:bnds[ix]})
        ei.add_tensor(TG)
        e += ei.contract()/n
    return e
#def get_gutzwiller(L,bdim=1,eps=0.,g=1.,normalize=True):
#    arrays = []
#    for i in range(L):
#        shape = [bdim] * 2 
#        if i==0 or i==L-1:
#            shape.pop()
#        shape = tuple(shape) + (4,)
#
#        data = np.ones(shape)
#        data += eps * np.random.rand(*shape)
#        data[...,3] = g * np.random.rand()
#        if normalize:
#            data /= np.linalg.norm(data)
#        arrays.append(data)
#    return PEPS([arrays])
class RestrictedGaussianFPEPS(SpinlessGaussianFPEPS):
    def set_ham(self,h1,eri,exact,thresh=1e-5):
        assert h1.shape[0] == self.P * self.nsite # check size 
        assert np.linalg.norm(h1-h1.T)<1e-6 # check hermitian
        self.exact = exact
        self.thresh = thresh
        if isinstance(eri,np.ndarray):
            # eri = <12|12>
            assert np.linalg.norm(eri-eri.transpose(1,0,3,2))<1e-6 # check symmetry 
            assert np.linalg.norm(eri-eri.transpose(2,3,0,1))<1e-6 # check hermitian
            eri = 2. * eri - eri.transpose(0,1,3,2) 
            def _eri(rho):
                return torch.einsum('pqrs,pr,qs->',eri,rho,rho) 
        elif callable(eri):
            _eri = eri
        else:
            raise NotImplementedError    
        h1 = torch.tensor(h1,requires_grad=False)
        def energy(rho):
            return 2.*torch.sum(rho * h1) + _eri(rho)
        self.energy = energy
class UnrestrictedGaussianFPEPS(SpinlessGaussianFPEPS):
    def __init__(self,L,M,Ne,pbc=False,occ_b=None,fix_bond=False):
        self.psi = [None] * 2
        self.psi[0] = SpinlessGaussianFPEPS2(L,M,Ne[0],pbc=pbc,occ_b=occ_b,fix_bond=fix_bond)
        self.psi[1] = SpinlessGaussianFPEPS2(L,M,Ne[1],pbc=pbc,occ_b=occ_b,fix_bond=fix_bond)

        self.nparam = sum([psi.nparam for psi in self.psi])
        self.P = self.psi[0].P
        self.nsite = self.psi[0].nsite
        self.site_map = self.psi[0].site_map

        self.nit = 0
        self.terminate = False
    def set_ham(self,h1,eri,exact,thresh=1e-5):
        assert h1.shape[0] == self.P * self.nsite # check size 
        assert np.linalg.norm(h1-h1.T)<1e-6 # check hermitian
        self.exact = exact
        self.thresh = thresh
        if isinstance(eri,np.ndarray):
            # eri = <12|12>
            assert np.linalg.norm(eri-eri.transpose(1,0,3,2))<1e-6 # check symmetry 
            assert np.linalg.norm(eri-eri.transpose(2,3,0,1))<1e-6 # check hermitian
            eri_aa = eri - eri.transpose(0,1,3,2) 
            eri = torch.tensor(eri,requires_grad=False) 
            eri_aa = torch.tensor(eri_aa,requires_grad=False) 
            def _eri(rho_a,rho_b):
                return .5 * torch.einsum('pqrs,pr,qs->',eri_aa,rho_a,rho_a) \
                     + .5 * torch.einsum('pqrs,pr,qs->',eri_aa,rho_b,rho_b) \
                     +      torch.einsum('pqrs,pr,qs->',eri_ab,rho_a,rho_b)
        elif callable(eri):
            _eri = eri
        else:
            raise NotImplementedError    
        h1 = torch.tensor(h1,requires_grad=False)
        def energy(rho_a,rho_b):
            return torch.sum((rho_a + rho_b) * h1) + _eri(rho_a,rho_b)
        self.energy = energy
    def fun(self,x):
        x = np.split(x,[self.psi[0].nparam]) 
        rho = [psi.get_rho(xi) for psi,xi in zip(self.psi,x)]
        E = self.energy(*rho)

        if self.terminate:
            g = np.zeros_like(x)
        else:
            E.backward()
            g = np.concatenate([psi.get_grad() for psi in self.psi]) 

        self.x = x
        self.g = g
        self.E = E.detach().numpy()
        return self.E,self.g
