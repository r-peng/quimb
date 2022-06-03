
def insert_isometry(ftn,site_tag,bnd,new_bnd,new_pattern,phys=False,Hi=None,
                    symmetry='u1',flat=True):
    T = ftn[site_tag]
    ax = T.inds.index(bnd)
    if phys:
        if symmetry=='u1': 
            bond_info = {U1(0):1,U1(1):2,U1(2):1}
        elif symmetry=='z2':
            bond_info = {Z2(0):2,Z2(1):2}
        else:
            raise ValueError(f'{symmetry} symmetry not supported')
        sign = '+'
        dim = 4
    # make identity
    else:
        bond_info = T.data.get_bond_info(ax,flip=False)
        sign = T.data.pattern[ax]
        dim = T.data.shape[ax]
    I = eye(bond_info,flat=flat)
    if sign=="-":
        I.pattern = '-+'
    # make vaccum
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    state_map = get_state_map(symmetry)
    q_label, idx, ish = state_map[0]
    data = np.zeros(ish)
    data[idx] = 1
    blocks = [SubTensor(reduced=data, q_labels=(q_label,))]
    vac = SparseFermionTensor(blocks=blocks, pattern=new_pattern, shape=(dim,))
    if flat:
        vac = vac.to_flat()

    data = np.tensordot(vac,I,axes=([],[]))
    U = FermionTensor(data=data,inds=(new_bnd,bnd,bnd+'_'),left_inds=(new_bnd,bnd),
                      tags=T.tags)
    if phys and (Hi is not None):
        lix = tuple([ix+'*' for ix in U.inds])
        Hi = FermionTensor(data=Hi.copy(),inds=lix[:2]+U.inds[:2],left_inds=lix[:2])
        I.pattern = '-+'
        I = FermionTensor(data=I,inds=(lix[-1],U.inds[-1]),left_inds=(lix[-1],))

        A = FTNLinearOperator([Hi,I],lix,U.inds,U.data.dq)
        maxiter = 100
        tol = 1e-8
        which = 'SA'
        backend = 'lobpcg'
        fallback_to_scipy = True
        ncv = 10
        w,v = eigh(A,B=None,v0=A.tensor_to_vector(U.data),k=1,
                   tol=tol,maxiter=maxiter,ncv=ncv,which=which,
                   backend=backend,fallback_to_scipy=fallback_to_scipy)
        v = A.vector_to_tensor(v,dq=U.data.dq)
        U.modify(data=v)

    T.reindex_({bnd:bnd+'_'})
    site = T.get_fermion_info()[1]
    ftn = insert(ftn,site+1,U)
    out = ftn.contract_tags(T.tags,which='all',inplace=True)
    if isinstance(out,FermionTensor):
        ftn = FermionTensorNetwork([out])
    return ftn
def split(T,lix,rix,D,solver=None,maxiter=10):
    bnd = rand_uuid()
    Tl,Tr = T.split(left_inds=lix,right_inds=rix,method='svd',
                    get='tensors',absorb='both',max_bond=D,bond_ind=bnd)
    Tl = Tl.transpose(bnd,*lix)
    if solver is None:
        return Tl,Tr
    return Tl,Tr
def expand(ftn,ix,step,direction,Hix=None,symmetry='u1',flat=True):
    # expand col/row ix to ix,ix+step
    D = ftn[0,0].shape[0]
    (L0,L1) = (ftn.Lx,ftn.Ly) if direction=='col' else (ftn.Ly,ftn.Lx)
    tag1 = direction.upper()
    pattern_p,pattern_n = '+','-'
    idx_new = {(i,i+1):rand_uuid() for i in range(L0-1)}
    ftn = ftn.reorder(direction=direction,inplace=True) 
    ftn_new = FermionTensorNetwork([])
    for i in range(ix):
        ftn_new.add_tensor_network(ftn.select(tag1+str(i)).copy())
    for i in range(L0):
        site = (i,ix) if direction=='col' else (ix,i)
        site_tag = 'I{},{}'.format(*site)
        
        ftn_new.add_tensor(ftn[site])
        site_p1 = (i,ix-1) if direction=='col' else (ix-1,i) 
        site_n1 = (i,ix+1) if direction=='col' else (ix+1,i) 
        lix = [] if ix==0 else list(ftn[site].bonds(ftn[site_p1]))
        rix = [] if ix==L1-1 else list(ftn[site].bonds(ftn[site_n1]))
        if i>0:
            site_p0 = (i-1,ix) if direction=='col' else (ix,i-1)
            idx = list(ftn[site].bonds(ftn[site_p0]))[0]
            ftn_new = insert_isometry(ftn_new,site_tag,idx,idx_new[i-1,i],
                                      pattern_p,symmetry=symmetry,flat=flat)
            if step>0:
                lix.append(idx)
                rix.append(idx_new[i-1,i])
            else:
                rix.append(idx)
                lix.append(idx_new[i-1,i])
        if i<L0-1:
            site_n0 = (i+1,ix) if direction=='col' else (ix,i+1)
            idx = list(ftn[site].bonds(ftn[site_n0]))[0]
            ftn_new = insert_isometry(ftn_new,site_tag,idx,idx_new[i,i+1],
                                      pattern_n,symmetry=symmetry,flat=flat)
            if step>0:
                lix.append(idx)
                rix.append(idx_new[i,i+1])
            else:
                rix.append(idx)
                lix.append(idx_new[i,i+1])
        pix,pix_new = 'k{},{}'.format(*site),'k{},{}'.format(*site_n1) 
        Hi = None if Hix is None else Hix[i]
        ftn_new = insert_isometry(ftn_new,site_tag,pix,pix_new,'+',
                                  phys=True,Hi=Hi,symmetry=symmetry,flat=flat)
        lix.append(pix)
        rix.append(pix_new)

        Tl,Tr = split(ftn_new[site_tag],lix,rix,D) 
        isite = ftn_new[site_tag].get_fermion_info()[1] 
        Tr.retag_({'I{},{}'.format(*site):'I{},{}'.format(*site_n1),
                   tag1+str(ix):tag1+str(ix+1)})
        ftn_new = replace(ftn_new,isite,Tl) 
        ftn_new = insert(ftn_new,isite+1,Tr)
    for j in range(ix+1,L1):
        for i in range(L0):
            site = (i,j) if direction=='col' else (j,i)
            site_n1 = (i,j+1) if direction=='col' else (j+1,i)
            T = ftn[site].copy()
            T.reindex_({'k{},{}'.format(*site):'k{},{}'.format(*site_n1)})
            T.retag_({'I{},{}'.format(*site):'I{},{}'.format(*site_n1),
                       tag1+str(j):tag1+str(j+1)})
            ftn_new.add_tensor(T)
    (Lx,Ly) = (L0,L1+1) if direction=='col' else (L1+1,L0)
    ftn_new.view_as_(FPEPS, Lx=Lx, Ly=Ly,
                     site_ind_id=ftn._site_ind_id, site_tag_id=ftn._site_tag_id,
                     row_tag_id=ftn._row_tag_id, col_tag_id=ftn._col_tag_id)
    return ftn_new 
###################################################################################
#                            TEBD Hamiltonians 
###################################################################################
def hopping_ham(Nx,Ny,ke1,ke3=None,symmetry='u1',flat=True):
    ham = dict()
    # NN terms
    op = Hubbard(t=-ke1, u=0.0, mu=0., fac=(0.,0.), symmetry=symmetry) 
    for i, j in product(range(Nx), range(Ny)):
        if i+1 != Nx:
            where = ((i,j), (i+1,j))
            ham[where] = op.copy() 
        if j+1 != Ny:
            where = ((i,j), (i,j+1))
            ham[where] = op.copy()
    # 3rd-NN terms
    if ke3 is not None:
        op = Hubbard(t=-ke3, u=0.0, mu=0., fac=(0.,0.), symmetry=symmetry) 
        for i, j in product(range(Nx), range(Ny)):
            if i+3 < Nx:
                where = ((i,j), (i+3,j))
                ham[where] = op.copy() 
            if j+3 < Ny:
                where = ((i,j), (i,j+3))
                ham[where] = op.copy()
    return LocalHam2D(Nx, Ny, ham)
