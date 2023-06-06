import numpy as np
import scipy.linalg
import scipy.sparse.linalg as spla
def davidson(A,B,x0,t,maxsize=25,restart_size=5,maxiter=None,tol=1e-4):
    # initialize
    sh = len(x0)
    size = 1
    x0_norm = np.dot(x0,B(x0))**.5
    if x0_norm < tol:
        raise ValueError(f'Singular B! xBx norm={x0_norm}')
    x0 /= x0_norm 
    
    V = x0.reshape(sh,1)
    VA = A(x0).reshape(sh,1)
    VB = B(x0).reshape(sh,1)

    Am = np.zeros((1,)*2)
    Am[0,0] = np.dot(V[:,0],VA[:,0]) 
    # assume V are B-orthonormal
    maxiter = maxsize if maxiter is None else maxiter
    it = 0
    while it < maxiter:
        # solve subspace eigenvalue problem
        theta,s = np.linalg.eig(Am)
        theta,s = theta.real,s.real
        dist = theta - t
        dist_sq = np.dot(dist,dist)
        if size > maxsize:
            # if size too large, scale down
            idxs = np.argsort(dist_sq)    
            N = np.array([s[:,idx] for idx in idxs[:restart_size]].T)
            V = np.dot(V,N)
            VA = np.dot(VA,N)
            Am = np.dot(N.T,np.dot(Am,N))
            size = restart_size
            continue
        idx = np.argmin(dist_sq)
        theta,s = theta[idx],s[:,idx]
        print(f'it={it},theta={theta}')

        # form residue
        u = np.dot(V,s)
        uA = np.dot(VA,s)
        uB = np.dot(VB,s)
        r = uA - theta * uB
        rnorm = np.linalg.norm(r)
        if rnorm < tol:
            return theta,u
        # solve preconditioned eqn
        def P(x):
            return x - u * np.dot(uB,x)
        def PT(x):
            return x - uB * np.dot(u,x)
        def F(x):
            y = P(x)
            y = A(y) - theta * B(y)
            return PT(y)
        LinOp = spla.LinearOperator((sh,sh),matvec=F,dtype=x0.dtype)
        delta,info = spla.lgmres(LinOp,-r,tol=tol,atol=tol)

        # orthogonalize
        for ix in range(size):
            delta -= V[:,ix] * np.dot(VB[:,ix],delta)
        # normalize
        Bdelta = B(delta)
        delta_norm = np.dot(delta,Bdelta)**.5
        if delta_norm < tol:
            print(f'Linear dependence in Davidson subspace! delta norm={delta_norm}, residue norm={rnorm}')
            return theta,u
        delta /= delta_norm
        Bdelta /= delta_norm

        # append to subspace
        Adelta = A(delta)
        AiJ = np.dot(V.T,Adelta)
        AIj = np.dot(delta,VA)
        AIJ = np.dot(delta,Adelta)
        Am = np.block([[Am,AiJ.reshape(size,1)],
                       [AIj.reshape(1,size),np.array([AIJ])]])

        V = np.concatenate([V,delta.reshape(sh,1)],axis=1)
        VA = np.concatenate([VA,Adelta.reshape(sh,1)],axis=1)
        VB = np.concatenate([VB,Bdelta.reshape(sh,1)],axis=1)
        size += 1
        it += 1
    return theta,u
if __name__ == "__main__":
    sh = 50
    A = np.random.rand(sh,sh)
    Bvals = np.random.rand(sh)
    K = np.random.rand(sh,sh)
    K -= K.T
    U = scipy.linalg.expm(K)
    B = np.einsum('s,si,sj->ij',Bvals,U,U)
    w,v = np.linalg.eigh(B)
    w,v = scipy.linalg.eig(A,b=B)
    w,v = w.real,v.real
    #print('vBv')
    #print(np.dot(v.T,np.dot(B,v)))
    #print('vv')
    #print(np.dot(v.T,v))
    #exit()

    idx = 0 
    t,v = w[idx],v[:,idx]
    v /= v[0]
    print('t=',t)
    x0 = np.random.rand(sh) * 5e-1
    def _A(x):
        return np.dot(A,x)
    def _B(x):
        return np.dot(B,x)
    theta,u = davidson(_A,_B,x0,t)
    print('theta=',theta)
    u /= u[0]
    print('eigenvector err=',np.linalg.norm(v-u)/np.linalg.norm(v))
