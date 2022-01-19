
#    cre = creation(spin='b',symmetry=symmetry,flat=True)
#    des = cre.dagger
#    par = np.tensordot(cre,des,axes=((1,),(0,)))
#    op1_ = np.tensordot(cre,des,axes=([],[]))
#    op1_ = op1_+op1_.transpose([2,3,0,1])

#    print(op1.pattern,op1.shape)
#    exit()
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
