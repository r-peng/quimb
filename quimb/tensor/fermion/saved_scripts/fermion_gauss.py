import numpy as np
import itertools,torch
import scipy.optimize
from pyblock3.algebra.fermion_ops import max_entangled_state,gauss
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid
from .fermion_2d import FPEPS
np.set_printoptions(suppress=True,linewidth=1000)
#def inv_back(A,C,dC):
#    # C = inv(A)
#    # dA = -C.T @ dC @ C.T
#    return -np.linalg.multi_dot([C.T,dC,C.T])
#def test_inv_back(M):
#    A = torch.rand(M,M,requires_grad=True)
#    C = torch.linalg.inv(A)
#    dC = torch.rand(M,M,requires_grad=False)
#    C.backward(gradient=dC)
#    gA = A.grad.detach().numpy()
#
#    A = A.detach().numpy()
#    C = C.detach().numpy()
#    dC = dC.detach().numpy()
#    dA = inv_back(A,C,dC)
#    print(np.linalg.norm(gA-dA))
#def quad_back(A,B,C,dC):
#    # C = B @ A @ B.T
#    # dA = B.T @ dC @ B
#    # dB = dC @ B @ A.T + dC.T @ B @ A
#    dA = np.linalg.multi_dot([B.T,dC,B])
#    dB = np.linalg.multi_dot([dC,B,A.T]) + np.linalg.multi_dot([dC.T,B,A])
#    return dA,dB
#def test_quad_back(M,N):
#    A = torch.rand(M,M,requires_grad=True)
#    B = torch.rand(N,M,requires_grad=True)
#    C = B@A@B.t()
#    dC = torch.rand(N,N,requires_grad=False)
#    C.backward(gradient=dC)
#    gA = A.grad.detach().numpy()
#    gB = B.grad.detach().numpy()
#
#    A = A.detach().numpy()
#    B = B.detach().numpy()
#    C = C.detach().numpy()
#    dC = dC.detach().numpy()
#    dA,dB = quad_back(A,B,C,dC)
#    print(np.linalg.norm(gA-dA))
#    print(np.linalg.norm(gB-dB))

#    def backward(self):
#        # Cout -> E
#        dCout = .5 * self.ham
#
#        # A,B,D -> Cout
#        dA = dCout
#        dT3 = -dCout
#
#        dT2,dB = quad_back(self.T2,self.B,self.T3,dT3) 
# 
#        dT1 = inv_back(self.T1,self.T2,dT2) 
#        dD = dT1
#        print(dA)
#        print(dB)
#        print(dD)
#
#        # A,B,D -> Ct 
#        self.dCt = dict()
#        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
#            start_v,stop_v,start_p,stop_p = self.get_slices(i,j)
#            dA_ = dA[start_p:stop_p,start_p:stop_p]
#            dB_ = dB[start_p:stop_p,start_v:stop_v]
#            dD_ = dD[start_v:stop_v,start_v:stop_v]
#            self.dCt[i,j] = np.block([[dD_,dB_.T],
#                                      [dB_,dA_]])
#        self.T1 = self.T2 = self.T3 = self.B = None
#    def get_grad(self):
#        ls = [None] * self.nsite
#        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
#            K,Ct,_ = self.Ct[i,j]
#            dCt = torch.tensor(self.dCt[i,j],requires_grad=False)
#            Ct.backward(gradient=dCt)
#            dK = K.grad.detach().numpy() 
#
#            ix = self.site_map[i,j]['site_ix']
#            ls[ix] = dK.flatten()
#
#        self.Ct = None
#        return np.concatenate(ls)

class SpinlessGaussianFPEPS: # for quadratic Hamiltonian only
    def __init__(self,Lx,Ly,M,Ne,blks=None):
        self.Lx,self.Ly = Lx,Ly
        self.nsite = Lx * Ly
        self.P,self.M = 1,M
        self.Ptot = self.P * self.nsite
        self.Ne = Ne

        if blks is None:
            blks = [list(itertools.product(range(self.Lx),range(self.Ly)))]
        self.get_site_map(blks)
        self.get_Cin()

        self.nit = 0
        self.terminate = False
    def get_site_map(self,blks):
        self.site_order = []
        for blk in blks:
            self.site_order += blk

        self.site_map = dict()
        self.nparam = 0
        start = 0
        for ix,(i,j) in enumerate(self.site_order):
            nleg = 4
            if i==0 or i==self.Lx-1:
                nleg -= 1
            if j==0 or j==self.Ly-1:
                nleg -= 1
            size = nleg * self.M
            self.site_map[i,j] = {'site_ix':ix,'start_v':start,'size_v':size,'nleg':nleg}

            start += size
            self.nparam += (size + self.P)**2
        self.Mtot = start
        print('Mtot=',self.Mtot)
        print('nparam=',self.nparam)
    def get_Cin(self):
        U = [] 
        u = 1./np.sqrt(2.) 
        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
            start1 = self.site_map[i,j]['start_v']
            if i<self.Lx-1:
                ix1 = start1

                start2 = self.site_map[i+1,j]['start_v']
                bond_ix = 2
                if i+1==self.Lx-1:
                    bond_ix -= 1
                if j==self.Ly-1:
                    bond_ix -= 1
                ix2 = start2 + bond_ix * self.M
                for mode in range(self.M):
                    Ui = np.zeros(self.Mtot) 
                    Ui[ix1+mode] = u 
                    Ui[ix2+mode] = u 
                    U.append(Ui)

            if j<self.Ly-1:
                bond_ix = 1
                if i==self.Lx-1:
                    bond_ix -= 1
                ix1 = start1 + bond_ix * self.M

                start2 = self.site_map[i,j+1]['start_v']
                bond_ix = 3
                if i==self.Lx-1:
                    bond_ix -= 1
                if j+1==self.Ly-1:
                    bond_ix -= 1
                if i==0:
                    bond_ix -= 1
                ix2 = start2 + bond_ix * self.M
                for mode in range(self.M):
                    Ui = np.zeros(self.Mtot) 
                    Ui[ix1+mode] = u 
                    Ui[ix2+mode] = u 
                    U.append(Ui)
        U = np.stack(U,axis=1)
                
        rho = np.dot(U,U.T)
        self.Cin = 2.*rho-np.eye(self.Mtot)
        self.Mocc = U.shape[1] 
        print('Mocc=',self.Mocc)
        assert self.Mocc == self.M * (2*self.nsite - self.Lx - self.Ly)
        assert np.linalg.norm(np.eye(self.Mtot)-np.dot(self.Cin,self.Cin))<1e-6
        self.Cin = torch.tensor(self.Cin,requires_grad=False)
    def get_Q(self,config=None,xstep=-1,ystep=-1,r=0):
        if config is None:
            cnt = 0
            xsweep = range(self.Lx) if xstep==1 else range(self.Lx-1,-1,-1)
            ysweep = range(self.Ly) if ystep==1 else range(self.Ly-1,-1,-1)
            Qmap = {(i,j):0 for i,j in itertools.product(range(self.Lx),range(self.Ly))} 
            for i,j in itertools.product(xsweep,ysweep):
                if (i+j)%2==r:
                    Qmap[i,j] = 1
                    cnt += 1
                if cnt==self.Ne:
                    break
            assert cnt == self.Ne
        else:
            assert sum(config)==self.Ne 
            Qmap = dict() 
            for ix,ci in enumerate(config):
                i,j = ix // self.Ly, ix % self.Ly
                Qmap[i,j] = ci

        cnt = 0
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            size_v = self.site_map[i,j]['size_v']
            dim1 = size_v + self.P
            dim2 = self.M * 2 + Qmap[i,j]
            if i==self.Lx-1:
                dim2 -= self.M 
            if i==0:
                dim2 -= self.M
            assert dim2 > 1 
            cnt += dim2 
            self.site_map[i,j]['Ushape'] = dim1,dim2
        assert cnt == self.Mocc + self.Ne
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
        self.K = dict() 
        for ix,(i,j) in enumerate(self.site_order):
            dim,_ = self.site_map[i,j]['Ushape']
            size = dim**2 
            K,x = x[:size],x[size:]
            self.K[i,j] = K.reshape(dim,dim)
    def get_rho(self,x):
        self.get_K(x)
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
        for ix,(i,j) in enumerate(self.site_order):
            K = self.K[i,j]
            dim1,dim2 = self.site_map[i,j]['Ushape']

            K = torch.tensor(K,requires_grad=True)
            self.K[i,j] = K 
            U = torch.linalg.matrix_exp(K-K.T)[:,:dim2]
            Ct = 2.*(U@U.t()) - torch.eye(dim1)

            site_map = self.site_map[i,j] 
            start_v = site_map['start_v']
            size_v = site_map['size_v'] 
            site_ix = site_map['site_ix']
            stop_v = start_v + size_v
            start_p = site_ix * self.P
            stop_p = start_p + self.P

            A.append(cat(self.P,start_p,stop_p,self.Ptot,Ct[-self.P:,-self.P:]))
            B.append(cat(self.P,start_v,stop_v,self.Mtot,Ct[-self.P:,:-self.P]))
            D.append(cat(size_v,start_v,stop_v,self.Mtot,Ct[:-self.P,:-self.P]))
        A = torch.cat(A,dim=0)
        B = torch.cat(B,dim=0)
        D = torch.cat(D,dim=0)

        # A,B,D -> Cout 
        T1 = D + self.Cin
        T2 = torch.linalg.inv(T1)
        T3 = B@T2@B.t()
        Cout = A - B@torch.linalg.inv(D+self.Cin)@B.t() 
        return .5 * (Cout + torch.eye(Cout.size(0)))
    def get_grad(self):
        ls = [None] * self.nsite
        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
            dK = self.K[i,j].grad.detach().numpy() 
            ix = self.site_map[i,j]['site_ix']
            ls[ix] = dK.flatten()
        self.K = None
        return np.concatenate(ls)
    def init(self,config):
        import scipy.linalg
        ls = [None] * self.nsite
        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
            ci = config[i*self.Ly + j]
            dim1,dim2 = self.site_map[i,j]['Ushape']
            K = np.random.rand(dim1,dim1)
            if ci == 1: # occ
                K[0,:] = 0.
                K[:,0] = 0.
            else:
                pass
            ix = self.site_map[i,j]['site_ix']
            ls[ix] = K.flatten()
            #print(i,j,ci)
            #print(scipy.linalg.expm(K-K.T)) 
        return np.concatenate(ls)
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
        return self.E,self.g
    def callback(self,xk=None):
        dE = self.exact-self.E
        print(f'niter={self.nit},E={self.E},dE={dE},xnorm={np.linalg.norm(self.x)},gnorm={np.linalg.norm(self.g)}') 
        self.nit += 1
        if np.fabs(dE) < self.thresh:
            self.terminate = True
    def run(self,method='bfgs',x0=None,maxiter=100):
        x0 = np.random.rand(self.nparam) if x0 is None else x0
        result = scipy.optimize.minimize(
                     self.fun,
                     x0,
                     method=method,
                     jac=True,
                     callback=self.callback,
                     options={'maxiter':maxiter})
        return result['x']
    def get_bond_state(self,symmetry='u1',flat=True):
        bond_state1 = max_entangled_state('++',symmetry=symmetry,flat=flat)
        bond_state = bond_state1
        for mode in range(1,self.M):
            bond_state = np.tensordot(bond_state,bond_state1,axes=([],[]))
        left_idx = tuple([2*mode for mode in range(self.M)])
        U,_,V = bond_state.tensor_svd(left_idx)
        tn = FermionTensorNetwork([])
        for (i,j) in self.site_order: 
            if i<self.Lx-1:
                uix = tuple([f'I{i},{j}_u{mode}' for mode in range(self.M)]) 
                bix = (rand_uuid(),)
                vix = tuple([f'I{i+1},{j}_d{mode}' for mode in range(self.M)]) 
                utag,vtag = f'I{i},{j}',f'I{i+1},{j}'
                tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True) 
                tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True) 
            if j<self.Ly-1:
                uix = tuple([f'I{i},{j}_r{mode}' for mode in range(self.M)]) 
                bix = (rand_uuid(),)
                vix = tuple([f'I{i},{j+1}_l{mode}' for mode in range(self.M)]) 
                utag,vtag = f'I{i},{j}',f'I{i},{j+1}'
                tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True)
                tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True)
        return tn 
    def get_fpeps(self,x,symmetry='u1',flat=True):
        x = self.x if x is None else x
        self.get_K(x) 
        tn = FermionTensorNetwork([])
        for (i,j) in self.site_order: 
            K = self.K[i,j]
            dim1,dim2 = self.site_map[i,j]['Ushape']
            U = scipy.linalg.expm(K-K.T)[:,:dim2]
            pattern = ''.join(['+']*dim1)
            data = gauss(pattern,U,symmetry=symmetry,flat=flat)

            inds = []
            if i<self.Lx-1:
                inds += [f'I{i},{j}_u{mode}' for mode in range(self.M)]  
            if j<self.Ly-1:
                inds += [f'I{i},{j}_r{mode}' for mode in range(self.M)]  
            if i>0:
                inds += [f'I{i},{j}_d{mode}' for mode in range(self.M)]  
            if j>0:
                inds += [f'I{i},{j}_l{mode}' for mode in range(self.M)]  
            inds += [f'k{i},{j}']
            tags = f'I{i},{j}',f'ROW{i}',f'COL{j}'
            tn.add_tensor(FermionTensor(data=data,inds=inds,tags=tags),virtual=True) 

        tnI = self.get_bond_state(symmetry=symmetry,flat=flat)
        tn.add_tensor_network(tnI.H,virtual=True)
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            tn.contract_tags(f'I{i},{j}',inplace=True)
        tn.view_as_(FPEPS,inplace=True,
                    site_tag_id='I{},{}',
                    row_tag_id='ROW{}',
                    col_tag_id='COL{}',
                    Lx=self.Lx,
                    Ly=self.Ly,
                    site_ind_id='k{},{}')
        return tn
class UnrestrictedGaussianFPEPS(SpinlessGaussianFPEPS):
    def __init__(self,Lx,Ly,M,Ne,blks=None):
        self.psi = [None] * 2
        self.psi[0] = SpinlessGaussianFPEPS(Lx,Ly,M,Ne[0],blks=blks)
        self.psi[1] = SpinlessGaussianFPEPS(Lx,Ly,M,Ne[1],blks=blks)

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
