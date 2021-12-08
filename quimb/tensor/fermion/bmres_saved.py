import numpy as np
from pyblock3.algebra.fermion import eye
inv_map = {'+':'-','-':'+'}
def BGMRES_(A,x0,b,max_space=10,tol=1e-4):
    nvir = len(x0.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    r0 = b-A(x0)
    if r0.norm()<tol:
        return x0
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
        return tsr.__class__(q_labels=q_labels,shapes=tsr.shapes,data=tsr.data,
               pattern=pattern,idxs=tsr.idxs,symmetry=tsr.symmetry,shape=tsr.shape)
    def qr(t):
        tq,tr = t.tensor_qr(left_idx=axs2,mod='qr') 
        return parse(tq,-1),parse(tr,0)
    v,r = qr(r0)
    def smin(h):
        s = 1.0
        for i in range(h.n_blocks):
            data = h.data[h.idxs[i]:h.idxs[i+1]].reshape(h.shapes[i])
            for j in range(data.shape[0]):
                s = min(s,abs(data[j,j]))
        return s 
    if smin(r)<tol:
        return x0
    V = [v]
    H = []
    T = []
    R = []
    n_blocks = set(x0.q_labels[:,-1])
    n_blocks = len(n_blocks)
    if n_blocks==1:
        bond_info = {x0.symmetry(1):2} 
    elif n_blocks==2:
        bond_info = {x0.symmetry(0):1,x0.symmetry(2):1} 
    elif n_blocks==3:
        bond_info = {x0.symmetry(0):1,x0.symmetry(1):2,x0.symmetry(2):1}
    eye_ = eye(bond_info,flat=True)
    eye_.pattern = '-+'
    lhs_ = (np.tensordot(v.dagger,v,axes=(axs1,axs2))-eye_).norm()
    assert lhs_<tol
    def col_ovlp(Hi,Hj):
        len_ = min(len(Hi),len(Hj))
        return sum([np.tensordot(Hi[k].dagger,Hj[k],axes=((1,0),(0,1)))  
                    for k in range(len_)])
    def gs(T,Hk):
        # assumes T is orthonormal
        Tk = [h.copy() for h in Hk]
        Rk = []
        for Tj in T:
            Rk.append(col_ovlp(Tj,Hk))
            for i in range(len(Tj)):
                Tk[i] = Tk[i]-Rk[-1]*Tj[i]
        norm = np.sqrt(col_ovlp(Tk,Tk))
        if norm<tol:
            return T,None
        else:
            T.append([h/norm for h in Tk])
            Rk.append(norm)
            assert abs(col_ovlp(T[-1],T[-1])-1.0)<tol
            for i in range(len(Hk)):
                rhs = []
                for j in range(len(T)):
                    if len(T[j])>i:
                        rhs.append(T[j][i]*Rk[j])
                rhs = sum(rhs)
                assert (Hk[i]-rhs).norm()<tol
            return T,Rk
    for j in range(max_space):
        u = A(V[-1])
#        Hj = [np.tensordot(v.dagger,u,axes=(axs1,axs2)) for v in V]
#        u = u-sum([np.tensordot(V[l],Hj[l],axes=((-1,),(0,))) 
#                   for l in range(len(V))])
        Hj = []
        for l in range(len(V)):
            Hj.append(np.tensordot(V[l].dagger,u,axes=(axs1,axs2)))
            u = u-np.tensordot(V[l],Hj[l],axes=((-1,),(0,)))
            assert np.tensordot(V[l].dagger,u,axes=(axs1,axs2)).norm()<tol
        if u.norm()<tol:
            break
        lhs  = [np.tensordot(v.dagger,u,axes=(axs1,axs2)).norm() for v in V]
        lhs_ = [(np.tensordot(v.dagger,v,axes=(axs1,axs2))-eye_).norm() for v in V]
        if sum(lhs)>tol:
            print('u',u.norm())
            print('check V',lhs)
            print('check V',lhs_)
            for k in range(len(V)):
                for l in range(k):
                    print('Vk',k)
                    print(V[k])
                    print('Vl',l)
                    print(V[l])
                    print('k,l',k,l,np.tensordot(V[l].dagger,V[k],axes=(axs1,axs2)).norm())
            exit()
        v,h = qr(u)
        if smin(h)<tol:
            break
        lhs  = [np.tensordot(v_.dagger,v,axes=(axs1,axs2)).norm() for v_ in V]
        lhs_ = (np.tensordot(v.dagger,v,axes=(axs1,axs2))-eye_).norm()
        if sum(lhs)>tol:
            print('v',v.norm())
            print('check V',lhs)
            print('check V',lhs_)
            exit()
#        T,Rk = gs(T,Hj)
#        if Rk is None:
#            break
#        R.append(Rk)
        V.append(v)
        Hj.append(h)
        H.append(Hj)
#        print('Hj,H,V',len(Hj),len(H),len(V))
    for i in range(len(H)):
        lhs = A(V[i])
        assert len(H[i])==i+2
        rhs = sum([np.tensordot(V[j],H[i][j],axes=((-1,),(0,))) for j in range(i+2)])
        assert (lhs-rhs).norm()<tol
    # H.T*H
#    print(len(H),len(H[-1]),len(V))
#    m = len(T)
#    tmp = np.zeros((m,m))
#    for i in range(m):
#        for j in range(m):
#            tmp[i,j] = col_ovlp(T[i],T[j])
#    assert np.linalg.norm(tmp-np.eye(m))<tol
#    num = np.array([np.tensordot(Tj[0].dagger,r,axes=((1,0),(0,1))) for Tj in T])
#    denom = np.zeros((m,m),dtype=num.dtype)
#    for i in range(m):
#        for j in range(i+1):
#            assert len(R[i])==i+1
#            denom[i,j]=R[i][j]
    m = len(H)
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
def BGMRES(A,x0,b,max_space=10,max_iter=0,cutoff=1e-4):
    def get_eo(tsr):
        q_labels = [[],[]]
        shapes = [[],[]]
        data = [[],[]]
        for i in range(tsr.n_blocks):
            q = tsr.q_labels[i,-1]
            q = tsr.symmetry.from_flat(q)
            if q.parity==0:
                q_labels[0].append(tsr.q_labels[i,:])
                shapes[0].append(tsr.shapes[i])
                data[0] += list(tsr.data[tsr.idxs[i]:tsr.idxs[i+1]])
            else:
                q_labels[1].append(tsr.q_labels[i,:])
                shapes[1].append(tsr.shapes[i])
                data[1] += list(tsr.data[tsr.idxs[i]:tsr.idxs[i+1]])
        q_labels = [np.array(qs) for qs in q_labels]
        shapes = [np.array(sh) for sh in shapes]
        data = [np.array(dat) for dat in data]
        return [tsr.__class__(q_labels=q_labels[i],shapes=shapes[i],data=data[i],
                pattern=tsr.pattern,symmetry=tsr.symmetry) for i in [0,1]]
    xs = get_eo(x0)
    bs = get_eo(b)
    def blk(x,b):
        norm = 1.0
        b_ = b/norm
        xold = x.copy()/norm
        r_norm_old = (b_-A(x_old)).norm()
        for i in range(max_iter):
            x = BGMRES_(A,xold,b_,max_space=max_space,tol=cutoff)
            r_norm = (b_-A(x)).norm()
            assert r_norm-r_norm_old<1e-6
            dx = (x-xold).norm()
            print('iter={},dx={},r_norm={}'.format(i,dx,r_norm))
            if dx<cutoff:
                break
            xold = x.copy()
            r_norm_old = r_norm
        return x*norm
#    x = blk(x0,b)
#    exit()
    xs = [blk(xs[i],bs[i]) for i in [0,1]]
    q_labels = np.concatenate([x.q_labels for x in xs],axis=0)
    shapes = np.concatenate([x.shapes for x in xs],axis=0)
    data = np.concatenate([x.data for x in xs],axis=0)
    x = x0.__class__(q_labels=q_labels,shapes=shapes,data=data,
                     pattern=x0.pattern,symmetry=x0.symmetry)
    return x
