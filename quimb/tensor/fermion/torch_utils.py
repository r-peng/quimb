
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
        if not torch.all(torch.isfinite(dA)):
            print('p1',dA)
            exit()
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

        if not torch.all(torch.isfinite(dA)):
            print('p1',dA)
            exit()
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

    print("Test Pass!")

if __name__=='__main__':
    test_svd()



