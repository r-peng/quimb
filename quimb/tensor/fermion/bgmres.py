import numpy as np
from pyblock3.algebra.fermion import eye
inv_map = {'+':'-','-':'+'}
def parse(tsr,ax):
    q_labels = tsr.q_labels.copy()
    narr = q_labels[:,ax]
    narr = [tsr.symmetry.from_flat(q) for q in narr]
    narr = [-q for q in narr]
    narr = [tsr.symmetry.to_flat(q) for q in narr]
    q_labels[:,ax] = np.array(narr)

    pattern = [c for c in tsr.pattern]
    pattern[ax] = inv_map[pattern[ax]]
    pattern = ''.join(pattern)

    tsr.pattern = pattern
    tsr.q_labels = q_labels
    return tsr
def get_physical_identity(t):
    n_blocks = set(t.q_labels[:,-1])
    n_blocks = len(n_blocks)
    bond_info = t.get_bond_info(-1)
    I = eye(bond_info,flat=True)
    I.pattern = '-+'
    return I
def svd(t,left_idx,cutoff=1e-6):
    qpn_partition = (t.dq,t.symmetry(0))
    u,s,v = t.tensor_svd(left_idx=left_idx,qpn_partition=qpn_partition,absorb=1,
                         cutoff=cutoff,cutoff_mode=1)
    assert s is None
    assert u.dq==t.dq
    assert v.dq==t.symmetry(0)
    u,v = parse(u,-1),parse(v,0)
    assert u.dq==t.dq
    assert v.dq==t.symmetry(0)
#    assert (np.tensordot(u,v,axes=((-1,),(0,)))-t).norm()<cutoff

    nvir = len(t.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    lhs = np.tensordot(u.dagger,u,axes=(axs1,left_idx))
    I = get_physical_identity(u)
    assert (lhs-I).norm()<cutoff
#    if (lhs-I).norm()>cutoff:
#        print((lhs-I).norm())
#        print(lhs)
#        print(I)
    return u,v
def ovlp(D1,D2,c1,c2):
    out = 0.0
    for i in range(min(c1,c2)+2):
        out += np.tensordot(D1[i,c1].dagger,D2[i,c2],axes=((1,0),(0,1)))
    return out
def restart(A,x0,b,max_space=10,tol=1e-4):
    tol = 1e-4

    nvir = len(x0.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    r0 = b-A(x0)
    if r0.norm()<tol:
        return x0
    v,r = svd(r0,left_idx=axs2,cutoff=tol)
    print(r.shape)
    V = [v]
    H = dict()
    for j in range(max_space):
        u = A(V[-1])
#        Hj = [np.tensordot(v.dagger,u,axes=(axs1,axs2)) for v in V]
#        u = u-sum([np.tensordot(V[l],Hj[l],axes=((-1,),(0,))) 
#                   for l in range(len(V))])
        assert len(V)==j+1
        for l in range(j+1):
            H[l,j] = np.tensordot(V[l].dagger,u,axes=(axs1,axs2))
            u = u-np.tensordot(V[l],H[l,j],axes=((-1,),(0,)))
            assert np.tensordot(V[l].dagger,u,axes=(axs1,axs2)).norm()<tol
        print(j,u.norm())
        if u.norm()<tol:
            break
        lhs1 = [np.tensordot(v.dagger,u,axes=(axs1,axs2)).norm() for v in V]
        if sum(lhs1)>tol:
            lhs2 = [(np.tensordot(v.dagger,v,axes=(axs1,axs2))
                     -get_physical_identity(v)).norm() for v in V]
            print('u',u.norm())
            print('check V',lhs1)
            print('check V',lhs2)
            for k in range(j+1):
                for l in range(k):
                    print('Vk',k)
                    print(V[k])
                    print('Vl',l)
                    print(V[l])
                    print('k,l',k,l,np.tensordot(V[l].dagger,V[k],
                                                 axes=(axs1,axs2)).norm())
            exit()
        v,h = svd(u,left_idx=axs2,cutoff=tol)
        lhs = [np.tensordot(v_.dagger,v,axes=(axs1,axs2)).norm() for v_ in V]
        if sum(lhs)>tol:
            print('v',v.norm())
            print('check V',lhs)
            exit()
        V.append(v)
        H[j+1,j] = h

#    for i in range(len(H)):
#        lhs = A(V[i])
#        assert len(H[i])==i+2
#        rhs = sum([np.tensordot(V[j],H[i][j],axes=((-1,),(0,))) for j in range(i+2)])
#        print(lhs.norm(),(lhs-rhs).norm())
#        assert (lhs-rhs).norm()<tol
    # QR on H
    m = len(V)-1
    print(m)
    T = dict()
    R = np.zeros((m,m))
    for key,h in H.items():
        print(key,h.shape)
    exit()
    for k in range(m): # col of H
        for i in range(k+2):
            T[i,k] = H[i,k].copy()
        for j in range(k): # col of T
            R[j,k] = ovlp(T,T,j,k)
            print('j,k',j,k,ovlp(T,T,j,j))
            for i in range(j+2):
                T[i,k] = T[i,k]-T[i,j]*R[j,k]
        R[k,k] = np.sqrt(ovlp(T,T,k,k))
        for i in range(k+2):
            T[i,k] = T[i,k]/R[k,k]
    np.set_printoptions(suppress=True,linewidth=200)
    print(R)
    # assert TT = I
    TT = np.zeros((m,m))
    for i in range(m):
        for j in range(m):
            TT[i,j] = ovlp(T,T,i,j)
    print('TT',np.linalg.norm(TT-np.eye(m)))
    print(TT)
    # assert TR=H 
    for k in range(m): # col of H
        for i in range(k+2): # row of H
            lhs = 0.0
            for j in range(m):
                tij = T.get((i,j),0.0)
                lhs = lhs + tij*R[j,k]
            if (H[i,k]-lhs).norm()>tol:
                print((H[i,k]-lhs).norm())
                print(H[i,k])
                print(lhs)
            
    exit()
    num = np.array([np.tensordot(Hj[0].dagger,r,axes=((1,0),(0,1))) for Hj in H])
    denom = np.zeros((m,m),dtype=num.dtype)
    for i in range(m):
        for j in range(i+1):
            val = sum([np.tensordot(H[i][k].dagger,H[j][k],axes=((1,0),(0,1))) 
                       for k in range(len(H[j]))])
            denom[i,j] = denom[j,i] = val
    denom_ = np.zeros((m,m),dtype=num.dtype)
    for i in range(m):
        for j in range(m):
            denom_[i,j] = col_ovlp(H[i],H[j])
    assert np.linalg.norm(denom-denom_)<tol
    if abs(np.linalg.det(denom))<tol:
        return x0
    y = np.dot(np.linalg.inv(denom),num)
    x = x0+sum([V[i]*y[i] for i in range(m)])
    r = b-A(x)
    
    perturb = np.random.rand(m)*1e-3
    y_ = y+perturb
    x_ = x0+sum(V[i]*y_[i] for i in range(m))
    r_ = b-A(x_)
    assert r.norm()-r_.norm()<1e-6
    return x,r
def BGMRES(A,x0,b,max_space=10,max_iter=50,cutoff=1e-4):
    xold = x0.copy()
    r_norm_old = (b-A(xold)).norm()
    for i in range(max_iter):
        x = restart(A,xold,b,max_space=max_space,tol=cutoff)
        r_norm = (b-A(x)).norm()
        assert r_norm-r_norm_old<1e-6
        dx = (x-xold).norm()
        print('iter={},dx={},r_norm={}'.format(i,dx,r_norm))
        if dx<cutoff:
            break
        xold = x.copy()
        r_norm_old = r_norm
    return x
