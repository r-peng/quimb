import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
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

