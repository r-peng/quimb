import numpy as np
import scipy
def gs(V,delta,B):
    sh,size = V.shape
    delta -= np.dot(delta,np.dot(B(delta),V))
    Bdelta = B(delta)
    delta_norm = np.dot(delta,Bdelta)**.5
    return delta,delta_norm,Bdelta
def davidson(A,B,x0,t,maxiter=10,tol=1e-4,**kwargs):
    # initialize
    sh = len(x0)
    V = x0.reshape(sh,1)
    VA = A(x0).reshape(sh,1)
    VB = B(x0).reshape(sh,1)
    size = 1

    A_ = np.zeros((1,)*2)
    B_ = np.zeros((1,)*2) 
    A_[0,0] = np.dot(V[:,0],VA[:,0]) 
    B_[0,0] = np.dot(V[:,0],VB[:,0]) 

    for it in range(maxiter):
        theta,s = scipy.linalg.eig(a=A_,b=B_)
        theta -= t
        dist_sq = np.dot(theta,theta)
        idx = np.argmin(dist_sq)
        theta = theta[idx]
        s = s[:,idx]

        u = np.dot(V,s)
        uA = A(u)
        uB = B(u)
        r = uA - theta * uB
        rnorm = np.linalg.norm(r)
        if rnorm < tol:
            return theta,u

        def P(x):
            return x - u * np.dot(uB,x)
        def F(x):
            y = P(x)
            y = A(y) - theta * B(y)
            return P(y)
        LinOp = spla.LinearOperator((sh,sh),matvec=F,dtype=x0.dtype)
        delta,info = spla.gmres(LinOp,-r,tol=tol)
        delta,delta_norm,Bdelta = gs(V,delta,B)
        if delta_norm < tol:
            print('Linear dependence in Davidson subspace! residue norm=',rnorm)
            return theta,u

        delta /= delta_norm
        Bdelta /= delta_norm
        V = np.concatenate([V,delta.reshape(sh,1)],axis=1)
        VA = np.concatenate([VA,A(delta).reshape(sh,1)],axis=1)
        VB = np.concatenate([VB,Bdelta.reshape(sh,1)],axis=1)
        A_ = np.block([[A_,np.arrayV[:,0]],
                       []])
