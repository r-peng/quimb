import numpy as np
import itertools,torch,h5py
import scipy.optimize
from pyblock3.algebra.fermion_ops import max_entangled_state,gauss
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid
from .fermion_2d_vmc import get_bond_state,get_projector_state,get_fpeps_from_bra_ket
from .fermion_2d import FPEPS
from .fermion_vmc import get_data_map
from ..product_vmc import config_to_ab
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
#    def parse_config(self,config,xstep=-1,ystep=-1,r=0):
#        if config is None:
#            cnt = 0
#            xsweep = range(self.Lx) if xstep==1 else range(self.Lx-1,-1,-1)
#            ysweep = range(self.Ly) if ystep==1 else range(self.Ly-1,-1,-1)
#            Qmap = {(i,j):0 for i,j in itertools.product(range(self.Lx),range(self.Ly))} 
#            for i,j in itertools.product(xsweep,ysweep):
#                if (i+j)%2==r:
#                    Qmap[i,j] = 1
#                    cnt += 1
#                if cnt==self.Ne:
#                    break
#            assert cnt == self.Ne
#        else:
#            assert sum(config)==self.Ne 
#            Qmap = dict() 
#            for ix,ci in enumerate(config):
#                i,j = ix // self.Ly, ix % self.Ly
#                Qmap[i,j] = ci
#        return Qmap 

class SpinlessGaussianFPEPS: # for quadratic Hamiltonian only
    def __init__(self,Lx,Ly,M,Ne,blks=None,occ_b=None,fix_bond=False):
        self.Lx,self.Ly = Lx,Ly
        self.Ne = Ne
        self.P = 1
        self.M = M
        self.occ_b = self.M if occ_b is None else occ_b 
        self.fix_bond = fix_bond

        self.nsite = Lx * Ly
        self.Ptot = self.P * self.nsite
        self.nbond = 2 * self.nsite - self.Lx - self.Ly
        self.occ_v = self.occ_b * self.nbond
        self.Mtot = self.M * self.nbond * 2 
        print('occ_v=',self.occ_v)
        print('Mtot=',self.Mtot)

        if blks is None:
            blks = [list(itertools.product(range(self.Lx),range(self.Ly)))]
        self.get_site_map(blks)
        self.get_Cin()

        self.config = None
        self.nit = 0
        self.terminate = False
        self.fname = None
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
        assert self.Mtot == start
        print('nparam=',self.nparam)
        if self.fix_bond:
            return 
        self.bond_order = []
        self.bond_map = dict()
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            if i+1<self.Lx:
                where = (i,j),(i+1,j)
                self.bond_order.append(where)
                self.bond_map[where] = len(self.bond_order) - 1
            if j+1<self.Ly:
                where = (i,j),(i,j+1)
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
        Qmap = {self.site_order[ix]:1 for ix in sites} 
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            size_v = self.site_map[i,j]['size_v']
            dim1 = size_v + self.P
            dim2 = Q + Qmap.get((i,j),0) 
            self.site_map[i,j]['Ushape'] = dim1,dim2
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
        for ix,(i,j) in enumerate(self.site_order):
            dim,_ = self.site_map[i,j]['Ushape']
            K,x1 = np.split(x1,[dim**2])
            self.K[i,j] = K.reshape(dim,dim)
        if self.fix_bond:
            return
        dim = 2 * self.M
        size = dim ** 2
        for where in self.bond_order:
            K,x2 = np.split(x2,[size])
            self.K[where] = K.reshape(dim,dim) 
    def cut_U(self,U,i,j):
        dim1,dim2 = self.site_map[i,j]['Ushape']
        if self.config is None:
            return U[:,:dim2]
        ci = self.config[i*self.Ly+j]
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
        for ix,(i,j) in enumerate(self.site_order):
            K = torch.tensor(self.K[i,j],requires_grad=True)
            self.K[i,j] = K 
            U = self.cut_U(torch.linalg.matrix_exp(K-K.T),i,j)
            Ct = 2.*(U@U.t()) - torch.eye(U.size(0))

            site_map = self.site_map[i,j] 
            start_v = site_map['start_v']
            size_v = site_map['size_v'] 
            site_ix = site_map['site_ix']
            stop_v = start_v + size_v
            start_p = site_ix * self.P
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
 
                where = (i,j),(i+1,j)
                K = torch.tensor(self.K[where],requires_grad=True)
                self.K[where] = K 
                U = torch.linalg.matrix_exp(K-K.T)[:,:self.occ_b]
                Cin = 2.*(U@U.t()) - torch.eye(2*self.M)

                D[ix1:ix1+self.M,ix1:ix1+self.M] += Cin[:self.M,:self.M]
                D[ix1:ix1+self.M,ix2:ix2+self.M] += Cin[:self.M,self.M:]
                D[ix2:ix2+self.M,ix1:ix1+self.M] += Cin[self.M:,:self.M]
                D[ix2:ix2+self.M,ix2:ix2+self.M] += Cin[self.M:,self.M:]
            
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
            
                where = (i,j),(i,j+1)
                K = torch.tensor(self.K[where],requires_grad=True)
                self.K[where] = K 
                U = torch.linalg.matrix_exp(K-K.T)[:,:self.occ_b]
                Cin = 2.*(U@U.t()) - torch.eye(2*self.M)

                D[ix1:ix1+self.M,ix1:ix1+self.M] += Cin[:self.M,:self.M]
                D[ix1:ix1+self.M,ix2:ix2+self.M] += Cin[:self.M,self.M:]
                D[ix2:ix2+self.M,ix1:ix1+self.M] += Cin[self.M:,:self.M]
                D[ix2:ix2+self.M,ix2:ix2+self.M] += Cin[self.M:,self.M:]
        return D
    def get_rho(self,x,plot=False):
        self.get_K(x)
        A,B,D = self.get_Ct()
        #print(B.shape)
        #exit()
        # A,B,D -> Cout 
        Cout = self.add_Cin(D)
        if plot:
            return A,B,Cout

        Cout = torch.linalg.inv(Cout)@B.t() 
        Cout = A - B @ Cout
        #Cout = A - B@torch.linalg.inv(self.add_Cin(D))@B.t() 
        return .5 * (Cout + torch.eye(Cout.size(0)))
    def get_grad(self):
        g1 = [None] * self.nsite
        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
            dK = self.K[i,j].grad.detach().numpy() 
            ix = self.site_map[i,j]['site_ix']
            g1[ix] = dK.flatten()
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
    def get_fixed_bond_state(self,symmetry='u1',flat=True,spin=None):
        #bond_state1 = max_entangled_state('++',symmetry=symmetry,flat=flat,spin=spin)
        #bond_state = bond_state1
        #for mode in range(1,self.M):
        #    bond_state = np.tensordot(bond_state,bond_state1,axes=([],[]))
        #left_idx = tuple([2*mode for mode in range(self.M)])
        #U,_,V = bond_state.tensor_svd(left_idx)
        #tn = FermionTensorNetwork([])
        #for (i,j) in self.site_order: 
        #    if i<self.Lx-1:
        #        uix = tuple([f'I{i},{j}_u{mode}_{spin}' for mode in range(self.M)]) 
        #        bix = (rand_uuid(),)
        #        vix = tuple([f'I{i+1},{j}_d{mode}_{spin}' for mode in range(self.M)]) 
        #        utag,vtag = f'I{i},{j}',f'I{i+1},{j}'
        #        tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True) 
        #        tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True) 
        #    if j<self.Ly-1:
        #        uix = tuple([f'I{i},{j}_r{mode}_{spin}' for mode in range(self.M)]) 
        #        bix = (rand_uuid(),)
        #        vix = tuple([f'I{i},{j+1}_l{mode}_{spin}' for mode in range(self.M)]) 
        #        utag,vtag = f'I{i},{j}',f'I{i},{j+1}'
        #        tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True)
        #        tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True)
        #return tn 
        return get_bond_state(self.Lx,self.Ly,self.M,site_order=self.site_order,
                              symmetry=symmetry,flat=flat,spin=spin)
    def get_variational_bond_state(self,symmetry='u1',flat=True,spin=None):
        tn = FermionTensorNetwork([])
        left_inds = tuple(range(self.M))
        for (i,j) in self.site_order: 
            if i<self.Lx-1:
                K = self.K[(i,j),(i+1,j)]
                U = scipy.linalg.expm(K-K.T)[:,:self.occ_b]
                pattern = ''.join(['+']*U.shape[0])
                data = gauss(pattern,U,symmetry=symmetry,flat=flat,spin=spin)
                U,_,V = data.tensor_svd(left_inds)

                uix = tuple([f'I{i},{j}_u{mode}_{spin}' for mode in range(self.M)]) 
                bix = (rand_uuid(),)
                vix = tuple([f'I{i+1},{j}_d{mode}_{spin}' for mode in range(self.M)]) 
                utag,vtag = f'I{i},{j}',f'I{i+1},{j}'
                tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True) 
                tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True) 
            if j<self.Ly-1:
                K = self.K[(i,j),(i,j+1)]
                U = scipy.linalg.expm(K-K.T)[:,:self.occ_b]
                pattern = ''.join(['+']*U.shape[0])
                data = gauss(pattern,U,symmetry=symmetry,flat=flat,spin=spin)
                U,_,V = data.tensor_svd(left_inds)

                uix = tuple([f'I{i},{j}_r{mode}_{spin}' for mode in range(self.M)]) 
                bix = (rand_uuid(),)
                vix = tuple([f'I{i},{j+1}_l{mode}_{spin}' for mode in range(self.M)]) 
                utag,vtag = f'I{i},{j}',f'I{i},{j+1}'
                tn.add_tensor(FermionTensor(data=V.copy(),inds=bix+vix,tags=vtag),virtual=True)
                tn.add_tensor(FermionTensor(data=U.copy(),inds=uix+bix,tags=utag),virtual=True)
        return tn 
    def get_bond_state(self,symmetry='u1',flat=True,spin=None):
        fxn = self.get_fixed_bond_state if self.fix_bond else self.get_variational_bond_state 
        return fxn(symmetry=symmetry,flat=flat,spin=spin) 
    def get_projector_state(self,symmetry='u1',flat=True,spin=None):
        #tn = FermionTensorNetwork([])
        #for (i,j) in self.site_order: 
        #    K = self.K[i,j]
        #    U = self.cut_U(scipy.linalg.expm(K-K.T),i,j)
        #    pattern = ''.join(['+']*U.shape[0])
        #    data = gauss(pattern,U,symmetry=symmetry,flat=flat,spin=spin)

        #    inds = [f'k{i},{j}'] if spin is None else [f'k{i},{j}_{spin}']
        #    if i<self.Lx-1:
        #        inds += [f'I{i},{j}_u{mode}_{spin}' for mode in range(self.M)]  
        #    if j<self.Ly-1:
        #        inds += [f'I{i},{j}_r{mode}_{spin}' for mode in range(self.M)]  
        #    if i>0:
        #        inds += [f'I{i},{j}_d{mode}_{spin}' for mode in range(self.M)]  
        #    if j>0:
        #        inds += [f'I{i},{j}_l{mode}_{spin}' for mode in range(self.M)]  
        #    tags = f'I{i},{j}',f'ROW{i}',f'COL{j}'
        #    tn.add_tensor(FermionTensor(data=data,inds=inds,tags=tags),virtual=True) 
        #return tn
        tsrs = dict() 
        for (i,j) in self.site_order: 
            K = self.K[i,j]
            U = self.cut_U(scipy.linalg.expm(K-K.T),i,j)
            pattern = ''.join(['+']*U.shape[0])
            tsrs[i,j] = gauss(pattern,U,symmetry=symmetry,flat=flat,spin=spin)
        return get_projector_state(self.Lx,self.Ly,tsrs,self.M,site_order=self.site_order)
    def get_fpeps(self,x,symmetry='u1',flat=True,spin=None):
        self.get_K(x) 
        tn = self.get_projector_state(symmetry=symmetry,flat=flat,spin=spin)
        tnI = self.get_bond_state(symmetry=symmetry,flat=flat,spin=spin) 
        return get_fpeps_from_bra_ket(tn,tnI,self.Lx,self.Ly)
        #tn.add_tensor_network(tnI.H,virtual=True)
        #for i,j in itertools.product(range(self.Lx),range(self.Ly)):
        #    tn.contract_tags(f'I{i},{j}',inplace=True)
        #tn.view_as_(FPEPS,inplace=True,
        #            site_tag_id='I{},{}',
        #            row_tag_id='ROW{}',
        #            col_tag_id='COL{}',
        #            Lx=self.Lx,
        #            Ly=self.Ly,
        #            site_ind_id='k{},{}')
        #print(tn)
        #return tn
    def init(self,config,eps=1e-2):
        self.config = config
        x1 = [None] * self.nsite
        for i,j in itertools.product(range(self.Lx),range(self.Ly)): 
            dim1,dim2 = self.site_map[i,j]['Ushape']
            K = np.random.rand(dim1,dim1)
            K[0,:] *= eps
            K[:,0] *= eps
            ix = self.site_map[i,j]['site_ix']
            x1[ix] = K.flatten()
        x1 = np.concatenate(x1)
        if self.fix_bond:
            return x1
        x2 = np.random.rand(self.nparam2)
        return np.concatenate([x1,x2])
class UnrestrictedGaussianFPEPS(SpinlessGaussianFPEPS):
    def __init__(self,Lx,Ly,M,Ne,blks=None,occ_b=None,fix_bond=False):
        self.psi = [None] * 2
        self.psi[0] = SpinlessGaussianFPEPS(Lx,Ly,M,Ne[0],blks=blks,occ_b=occ_b,fix_bond=fix_bond)
        self.psi[1] = SpinlessGaussianFPEPS(Lx,Ly,M,Ne[1],blks=blks,occ_b=occ_b,fix_bond=fix_bond)

        self.nparam = sum([psi.nparam for psi in self.psi])
        psi = self.psi[0]
        self.P = psi.P
        self.nsite = psi.nsite
        self.site_map = psi.site_map
        self.site_order = psi.site_order
        self.Lx,self.Ly = psi.Lx,psi.Ly

        self.nit = 0
        self.terminate = False
    def get_Q(self,sites=None):
        sites = [None,None] if sites is None else sites
        self.psi[0].get_Q(sites=sites[0])
        self.psi[1].get_Q(sites=sites[1])
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
    def get_rho(self,x):
        x = np.split(x,[self.psi[0].nparam]) 
        return [psi.get_rho(xi) for psi,xi in zip(self.psi,x)]
    def fun(self,x):
        rho = self.get_rho(x)
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
    def get_fpeps_full(self,x,symmetry='u1',flat=True):
        x = np.split(x,[self.psi[0].nparam]) 
        for psi,xi in zip(self.psi,x):
            psi.get_K(xi) 

        tn = FermionTensorNetwork([])
        for spin,psi in zip((0,1),self.psi):
            tnT = psi.get_projector_state(symmetry=symmetry,flat=flat,spin=spin)
            tn.add_tensor_network(tnT,virtual=True)
            tnI = psi.get_bond_state(symmetry=symmetry,flat=flat,spin=spin) 
            tn.add_tensor_network(tnI.H,virtual=True)

        data_map = get_data_map(symmetry=symmetry,flat=flat,spinless=False)
        P = None
        ax = [],[]
        for k1 in (0,1):
            s1 = data_map[k1].dagger
            for k2 in (0,2):
                s2 = data_map[k2].dagger
                s3 = data_map[k1+k2]
                blk = np.tensordot(s3,np.tensordot(s1,s2,axes=ax),axes=ax) 
                if P is None:
                    P = blk
                else:
                    P = P + blk
        P.shape = (4,) * 3
        for (i,j) in self.site_order[::-1]:
            inds = [f'k{i},{j}' + app for app in ['','_0','_1']]
            tags = f'I{i},{j}'
            tn.add_tensor(FermionTensor(data=P.copy(),inds=inds,tags=tags),virtual=True) 
        for i,j in itertools.product(range(self.Lx),range(self.Ly)):
            tn.contract_tags(f'I{i},{j}',inplace=True)
        tn.view_as_(FPEPS,inplace=True,
                    site_tag_id='I{},{}',
                    row_tag_id='ROW{}',
                    col_tag_id='COL{}',
                    Lx=self.Lx,
                    Ly=self.Ly,
                    site_ind_id='k{},{}')
        print(tn)
        return tn
    def init(self,config,eps=1e-2):
        return np.concatenate([psi.init(config_ix,eps=eps) for psi,config_ix in zip(self.psi,config)])
class GaussianFPEPS(SpinlessGaussianFPEPS):
    def __init__(self,Lx,Ly,M,Ne,blks=None,occ_b=None,fix_bond=False):
        self.Lx,self.Ly = Lx,Ly
        self.Ne = Ne
        self.P = 2 
        self.M = M
        self.occ_b = self.M if occ_b is None else occ_b 
        self.fix_bond = fix_bond

        self.nsite = Lx * Ly
        self.Ptot = self.P * self.nsite
        self.nbond = 2 * self.nsite - self.Lx - self.Ly
        self.occ_v = self.occ_b * self.nbond
        self.Mtot = self.M * self.nbond * 2 
        print('occ_v=',self.occ_v)
        print('Mtot=',self.Mtot)

        if blks is None:
            blks = [list(itertools.product(range(self.Lx),range(self.Ly)))]
        self.get_site_map(blks)
        self.get_Cin()

        self.config = None
        self.nit = 0
        self.terminate = False
        self.fname = None
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
