from pyblock3.algebra.fermion_ops import creation,annihilation

def SpinlessFermion(t,v,Lx,Ly,mu=0.0,symmetry=None): 
    ham = dict()
    cre = creation(spin='a',symmetry=symmetry,flat=True)
    des = cre.dagger
    par = np.tensordot(cre,des,axes=((1,),(0,)))
    op1 = np.tensordot(cre,des,axes=([],[]))
    op1 = op1+op1.transpose([2,3,0,1])

#    cre = creation(spin='b',symmetry=symmetry,flat=True)
#    des = cre.dagger
#    par = np.tensordot(cre,des,axes=((1,),(0,)))
#    op1_ = np.tensordot(cre,des,axes=([],[]))
#    op1_ = op1_+op1_.transpose([2,3,0,1])

#    print(op1.pattern,op1.shape)
    op2 = np.tensordot(par,par,axes=([],[]))
    op = -t*op1+v*op2
    op = op.transpose([0,2,1,3])
#    exit()
    for i, j in product(range(Lx), range(Ly)):
        if i+1 != Lx:
            where = ((i,j), (i+1,j))
            ham[where] = op.copy()
        if j+1 != Ly:
            where = ((i,j), (i,j+1))
            ham[where] = op.copy()
    return LocalHam2D(Lx, Ly, ham)
def match_phase(data,T):
    global_flip = T.phase.get('global_flip',False)
    local_inds = T.phase.get('local_inds',[])
    data_new = data.copy()
    if global_flip:
        data_new._global_flip()
    if len(local_inds)>0:
        axes = [T.inds.index(ind) for ind in local_inds]
        data_new._local_flip(axes)
    return data_new
def sweep(self,tau):
    Lx,Ly = self._psi.Lx,self._psi.Ly
    ordering = []
    for j in range(Ly):
        for i in range(Lx):
            if i+1!=Lx:
                where = (i,j),(i+1,j)
                ordering.append(where)
    for i in range(Lx):
        for j in range(Ly):
            if j+1!=Ly:
                where = (i,j),(i,j+1)
                ordering.append(where)
    assert len(ordering)==len(self.ham.terms)
    for i,where in enumerate(ordering):
        U = self.ham.get_gate_expm(where,-tau)
        self.gate(U, where)
    normalize(self)
