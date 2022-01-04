import numpy as np
def restart(A,x0,b,max_space=10,tol=1e-6,atol=1e-10):
    ndim = len(x0.shape)
    axes = range(ndim-1,-1,-1),range(ndim)
    r0 = b-A(x0)
    beta = r0.norm()
    if beta<tol:
        return x0,beta
    Q = [r0/beta]
    H = np.zeros((max_space+1,max_space),dtype=np.float64)
    for j in range(max_space):
        q = A(Q[j])
        norm_q = q.norm()
        if norm_q<tol:
            break
        for l in range(j+1):
            H[l,j] = np.tensordot(Q[l].dagger,q,axes=axes)
#        for l in range(j+1):
            q = q-Q[l]*H[l,j]
#            assert abs(np.tensordot(Q[l].dagger,q,axes=axes))<atol
        norm_q = q.norm()
        if norm_q<tol:
            break
        q = q/norm_q
#        lhs1 = [abs(np.tensordot(q_.dagger,q,axes=axes)) for q_ in Q]
#        if sum(lhs1)>atol*len(lhs1):
#            break
#            print(sum(lhs1),atol)
#            print('q',norm_q)
#            print('check Q',lhs1)
#            lhs2 = [abs(np.tensordot(q_.dagger,q_,axes=axes)-1.0) for q_ in Q]
#            if sum(lhs2)>atol:
#               print('check Q',lhs2)
#            for k in range(j+1):
#                for l in range(k):
#                    ovlp = np.tensordot(Q[l].dagger,Q[k],axes=axes)
#                    if abs(ovlp)>atol:
#                        print('k,l',k,l,ovlp)
#            exit()
        Q.append(q)
        H[j+1,j] = norm_q

    m = len(Q)-1
    H = H[:m+1,:m]
#    for i in range(m):
#        lhs = A(Q[i])
#        rhs = sum([Q[j]*H[j,i] for j in range(m+1)])
#        assert (lhs-rhs).norm()<atol
    # QR on H
    T,R = np.linalg.qr(H)
#    R,T = np.dot(H.T,H),H.copy()
    y = np.dot(np.linalg.inv(R),T[0,:]*beta)
    x = x0+sum([Q[i]*y[i] for i in range(m)])
    norm_r = (b-A(x)).norm()
#    print(y[0]) 
    perturb = np.random.rand(m)*tol
    y_ = y+perturb
    x_ = x0+sum(Q[i]*y_[i] for i in range(m))
    norm_r_ = (b-A(x_)).norm()
#    assert norm_r-norm_r_<atol
    if norm_r>norm_r_:
        print(np.diag(R))
        exit()
    return x,norm_r
def GMRES(A,x0,b,max_space=20,max_iter=100,tol=1e-6,atol=1e-10):
    x = x0.copy()
    r_norm_old = (b-A(x)).norm()
    for i in range(max_iter):
        x,r_norm = restart(A,x,b,max_space=max_space,tol=tol,atol=atol)
        print('iter={},r_norm={}'.format(i,r_norm))
        assert r_norm-r_norm_old<atol
        if r_norm<tol:
            break
        r_norm_old = r_norm
    return x
