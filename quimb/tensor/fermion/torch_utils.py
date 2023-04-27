
'''
PyTorch has its own implementation of backward function for SVD https://github.com/pytorch/pytorch/blob/291746f11047361100102577ce7d1cfa1833be50/tools/autograd/templates/Functions.cpp#L1577
We reimplement it with a safe inverse function in light of degenerated singular values
'''

import numpy as np
import torch
import os, sys
import scipy.linalg

be_verbose = True
epsilon = 1e-28

def safe_inverse(x):
    return x/(x**2 + epsilon)

class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A):
        if not torch.all(torch.isfinite(A)):
            raise ValueError("input matrix to custom SVD is not finite")
        try:
            U, S, Vh = torch.linalg.svd(A,full_matrices=False)
        except:
            if be_verbose:
                print('trouble in torch gesdd routine, falling back to gesvd')
            U, S, Vh = scipy.linalg.svd(A.detach().numpy(), full_matrices=False, lapack_driver='gesvd')
            U = torch.from_numpy(U)
            S = torch.from_numpy(S)
            Vh = torch.from_numpy(Vh)

        # trim
        ind = S > epsilon
        S = S[ind]            
        U = U[:,ind]
        Vh = Vh[ind,:]

        # make SVD result sign-consistent across multiple runs
        #for idx in range(U.size()[1]):
        #    if max(torch.max(U[:,idx]), torch.min(U[:,idx]), key=abs) < 0.0:
        #        U[:,idx] *= -1.0
        #        Vh[idx,:] *= -1.0

        self.save_for_backward(U, S, Vh)

        return U, S, Vh

    @staticmethod
    def backward(self, dU, dS, dVh):
        if not torch.all(torch.isfinite(dU)):
            raise ValueError("dU is not finite")
        if not torch.all(torch.isfinite(dS)):
            raise ValueError("dS is not finite")
        if not torch.all(torch.isfinite(dVh)):
            raise ValueError("dVh is not finite")
        U, S, Vh = self.saved_tensors

        #Vt = V.t()
        #Ut = U.t()
        M = U.size(0)
        N = Vh.size(1)
        NS = len(S)
        if NS==0:
            return torch.zeros(M,N,dtype=U.dtype)
        for i in range(NS-1):
            if torch.abs(S[i]-S[i+1])<epsilon:
                print('warning! degenerate singular values', S)

        F = (S - S[:, None])
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = (S + S[:, None])
        #G.diagonal().fill_(np.inf)
        #G = 1/G
        G = safe_inverse(G)
        G.diagonal().fill_(0)

        UdU = U.t() @ dU
        VdV = Vh @ dVh.t()

        Su = (F+G)*(UdU-UdU.t())/2
        Sv = (F-G)*(VdV-VdV.t())/2

        dA = U @ (Su + Sv + torch.diag(dS)) @ Vh
        Su = Sv = UdU = VdV = G = F = None   # help with memory
        if (M>NS):
            ##dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU/S) @ Vt
            #dA = dA + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U@Ut) @ (dU*safe_inverse(S)) @ Vt
            # the following is a rewrite of the above one-liner to decrease memory
            tmp1 = (dU*safe_inverse(S)) @ Vh
            tmp2 = U.t() @ tmp1
            tmp2 = U @ tmp2
            U = S = Vh = None
            dA += (tmp1 - tmp2)
            tmp1 = tmp2 = None
        if (N>NS):
            ##dA = dA + (U/S) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
            #dA = dA + (U*safe_inverse(S)) @ dV.t() @ (torch.eye(N, dtype=dU.dtype, device=dU.device) - V@Vt)
            # the following is a rewrite of the above one-liner to decrease memory
            tmp1 = (U*safe_inverse(S)) @ dVh
            tmp2 = tmp1 @ Vh.t()
            tmp2 = tmp2 @ Vh
            U = S = Vh = None
            dA += (tmp1 - tmp2)
            tmp1 = tmp2 = None
        return dA

def test_svd():
    M, N = 50, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, (input), eps=1e-6, atol=1e-4))
    M, N = 20, 50
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(SVD.apply, (input), eps=1e-6, atol=1e-4))

    print("SVD Test Pass!")

def copyltu(A):
    tril0 = A.tril(diagonal=0)
    tril1 = A.tril(diagonal=-1)
    return tril0 + tril1.t()
def QRforward_deep(A):
    if not torch.all(torch.isfinite(A)):
        raise ValueError("input matrix to custom QR is not finite")
    try:
        Q, R = torch.linalg.qr(A,mode='reduced')
    except:
        if be_verbose:
            print('trouble in torch gesdd routine, falling back to scipy')
        Q, R = scipy.linalg.svd(A.detach().numpy(), mode='economic')
        Q = torch.from_numpy(Q)
        R = torch.from_numpy(R)
    return Q,R
def safe_inverse_tri(T,R):
    diag = R.diag()
    full_rank = True
    for i in range(len(diag)):
        if torch.abs(diag[i]) < epsilon:
            full_rank = False
            break
    if full_rank:
        TRinv = T @ torch.linalg.inv(R.t())
        if not torch.all(torch.isfinite(TRinv)):
            raise ValueError('Rinv is not finite')
        return TRinv 
    else:
        TRinv,res,rank,s = torch.linalg.lstsq(R,T.t(),driver='gelsd')
        if not torch.all(torch.isfinite(TRinv)):
            raise ValueError('Rinv is not finite')
        return TRinv.t()
def QRbackward_deep(Q,R,dQ,dR):
    M = R@dR.t() - dQ.t()@Q
    M = copyltu(M)        
    return safe_inverse_tri(dQ + Q@M, R)
def QRforward_wide(A):
    M,N = A.size()
    X,Y = A.split((M,N-M),dim=1)
    Q,U = QRforward_deep(X)
    V = Q.t()@Y
    R = torch.cat((U,V),dim=1)
    return Q,R
def QRbackward_wide(A,Q,R,dQ,dR):
    M,N = A.size()
    X,Y = A.split((M,N-M),dim=1)
    U,V = R.split((M,N-M),dim=1)
    dU,dV = dR.split((M,N-M),dim=1)

    tmp = dQ+Y@dV.t()
    M = U@dU.t() - tmp.t()@Q
    M = copyltu(M)
    dX = safe_inverse_tri(tmp + Q@M,U)
    return torch.cat((dX,Q@dV),dim=1)
class QR(torch.autograd.Function):
    @staticmethod
    def forward(self,A):
        M,N = A.size()
        if M>=N:
            Q,R = QRforward_deep(A)
        else:
            Q,R = QRforward_wide(A)
        self.save_for_backward(A,Q,R)
        return Q,R
    @staticmethod
    def backward(self, dQ, dR):
        if not torch.all(torch.isfinite(dQ)):
            raise ValueError("dQ is not finite")
        if not torch.all(torch.isfinite(dR)):
            raise ValueError("dR is not finite")
        A,Q,R = self.saved_tensors
        M,N = A.size()
        if M>=N:
            return QRbackward_deep(Q,R,dQ,dR)
        else:
            return QRbackward_wide(A,Q,R,dQ,dR)
        
def test_qr():
    M, N = 50, 20
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(QR.apply, (input), eps=1e-6, atol=1e-4))
    M, N = 20, 50
    torch.manual_seed(2)
    input = torch.rand(M, N, dtype=torch.float64, requires_grad=True)
    assert(torch.autograd.gradcheck(QR.apply, (input), eps=1e-6, atol=1e-4))

    print("QR Test Pass!")
if __name__=='__main__':
    test_svd()
    test_qr()



