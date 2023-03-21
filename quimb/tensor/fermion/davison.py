import numpy as np
import scipy
def gs(delta,B,V=None):
    sh,size = V.shape
    delta -= np.dot(delta,np.dot(B(delta),V))
    Bdelta = B(delta)
    delta_norm = np.dot(delta,Bdelta)**.5
    return delta,delta_norm
def davidson(A,B,x0,t,maxiter=10,maxsize=10,tol=1e-4):
    # initialize
    sh = len(x0)
    x0,x0_norm = gs(x0,B)
    if x0_norm < tol:
        raise ValueError(f'xBx norm={x0_norm}')
    x0 /= x0_norm 
    
    V = x0.reshape(sh,1)
    VA = A(x0).reshape(sh,1)

    Am = np.zeros((1,)*2)
    Am[0,0] = np.dot(V[:,0],VA[:,0]) 
    for it in range(maxiter):
        info = davidson_restart(t,V,VA,Am,A,B,maxsize=maxsize,tol=tol)
        if info[-1]:
            return info
        V,VA,_ = info 
         
def davidson_restart(t,V,VA,Am,Af,Bf,maxsize=10,tol=1e-4):
    sh,size = V.shape
    # assume V are B-orthonormal
    for it in range(size,maxsize):
        theta,s = np.linalg.eig(Am)
        theta = theta.real
        s = s.real
        theta -= t
        dist_sq = np.dot(theta,theta)
        idx = np.argmin(dist_sq)
        thetas = np.concatenate([thetas,np.array([theta[idx]])],axis=0)
        s = s[:,idx]

        u = np.dot(V,s)
        uA = Af(u)
        uB = Bf(u)
        r = uA - theta * uB
        rnorm = np.linalg.norm(r)
        if rnorm < tol:
            return theta,u,None,True

        def P(x):
            return x - u * np.dot(uB,x)
        def F(x):
            y = P(x)
            y = Af(y) - theta * Bf(y)
            return P(y)
        LinOp = spla.LinearOperator((sh,sh),matvec=F,dtype=x0.dtype)
        delta,info = spla.lgmres(LinOp,-r,tol=tol)
        delta,delta_norm = gs(delta,B,V=V)
        if delta_norm < tol:
            print('Linear dependence in Davidson subspace! residue norm=',rnorm)
            return theta,u,None,True

        delta /= delta_norm
        Adelta = A(delta)
        AiJ = np.dot(V.T,Adelta)
        AIj = np.dot(delta,VA)
        AIJ = np.dot(delta,Adelta)
        Am = np.block([[Am,AiJ.reshape(size,1),]
                       [AIj.reshape(1,size),np.array([AIJ])]])

        V = np.concatenate([V,delta.reshape(sh,1)],axis=1)
        VA = np.concatenate([VA,Adelta.reshape(sh,1)],axis=1)
    return V,VA,thetas,False
