
def gate_full_update_als(ket,env,bra,G,where,tags_plq,steps,tol,max_bond,
    optimize='auto-hq',solver='solve',dense=True,
    enforce_pos=False,pos_smudge=1e-6,init_simple_guess=True,
    condition_tensors=True,
    condition_maintain_norms=True,condition_balance_bonds=True):
    norm_plq = env.copy()
#    print('########## norm_plq ###############')
#    fs = norm_plq.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tid,tsr.tags,tsr.inds,tsr.shape)
   
    ket0_tid,ket0_site = norm_plq[tags_plq[0],'KET'].get_fermion_info() 
    ket1_tid,ket1_site = norm_plq[tags_plq[1],'KET'].get_fermion_info() 
    bra0_tid,bra0_site = norm_plq[tags_plq[0],'BRA'].get_fermion_info() 
    bra1_tid,bra1_site = norm_plq[tags_plq[1],'BRA'].get_fermion_info() 
    norm_plq._refactor_phase_from_tids((ket0_tid,bra0_tid))
    norm_plq._refactor_phase_from_tids((ket1_tid,bra1_tid))
    ket_plq_old = norm_plq._select_tids([ket0_tid,ket1_tid],virtual=False)

    if init_simple_guess:
#        tid_map = {ket1_tid:bra0_site,bra0_tid:ket1_site}
        tid_map = {ket1_tid:bra0_site,bra0_tid:bra1_site,bra1_tid:ket1_site}
        norm_plq._reorder_from_tid(tid_map,inplace=True)
        norm_plq._refactor_phase_from_tids((ket0_tid,bra0_tid))
        norm_plq._refactor_phase_from_tids((ket1_tid,bra1_tid))
        # check hermitian
#        ket_plq = norm_plq._select_tids([ket0_tid,ket1_tid],virtual=False)
#        bra_plq = norm_plq._select_tids([bra0_tid,bra1_tid],virtual=False)
#        ket_plq = ket_plq.contract()
#        bra_plq = bra_plq.contract()
#        print('plq indices',ket_plq.inds,ket_plq.data.pattern)
#        print('plq indices',bra_plq.inds,ket_plq.data.pattern)
#        print('check hermitian',(ket_plq.data-bra_plq.data.dagger).norm())
        ket_plq = norm_plq._select_tids([ket0_tid,ket1_tid])
        ket_plq.view_like_(ket)
        ket_plq.gate_(G,where,contract='reduce-split',max_bond=max_bond)
        if condition_tensors:
            conditioner(ket_plq,balance_bonds=condition_balance_bonds)
            if condition_maintain_norms:
                pre_norm = ket_plq[tags_plq[0]].norm()
        norm_plq[tags_plq[0],'BRA'].modify(data=ket_plq[tags_plq[0]].data.dagger)
        norm_plq[tags_plq[1],'BRA'].modify(data=ket_plq[tags_plq[1]].data.dagger)
        # check hermitian
#        ket_plq = norm_plq._select_tids([ket0_tid,ket1_tid],virtual=False)
#        bra_plq = norm_plq._select_tids([bra0_tid,bra1_tid],virtual=False)
#        ket_plq = ket_plq.contract()
#        bra_plq = bra_plq.contract()
#        print('plq indices',ket_plq.inds,ket_plq.data.pattern)
#        print('plq indices',bra_plq.inds,ket_plq.data.pattern)
#        print('check hermitian',(ket_plq.data-bra_plq.data.dagger).norm())
#        tid_map = {ket1_tid:ket1_site,bra0_tid:bra0_site}
        tid_map = {ket1_tid:ket1_site,bra0_tid:bra0_site,bra1_tid:bra1_site}
        norm_plq._reorder_from_tid(tid_map,inplace=True)
        norm_plq._refactor_phase_from_tids((ket0_tid,bra0_tid))
        # check hermitian
#        ket = norm_plq[tags_plq[0],'KET']
#        bra = norm_plq[tags_plq[0],'BRA']
#        print('check hermitian',(ket.data-bra.data.dagger).norm())
        norm_plq._refactor_phase_from_tids((ket1_tid,bra1_tid))
#        ket = norm_plq[tags_plq[1],'KET']
#        bra = norm_plq[tags_plq[1],'BRA']
#        print('check hermitian',(ket.data-bra.data.dagger).norm())
    overlap = norm_plq.copy()
    overlap[tags_plq[0],'KET'].modify(data=ket_plq_old[tags_plq[0]].data)
    overlap[tags_plq[1],'KET'].modify(data=ket_plq_old[tags_plq[1]].data)
#    print('check replace',(overlap[tags_plq[0],'KET'].data-norm_plq[tags_plq[0],'KET'].data).norm())
#    print('check replace',(overlap[tags_plq[1],'KET'].data-norm_plq[tags_plq[1],'KET'].data).norm())
#    exit()

    # factorize both local and global phase on the operator tensors
    layer_tags = 'KET','BRA'
    if is_lone_coo(where):
        _where = (where,)
    else:
        _where = tuple(where)
    ng = len(_where)
    site_ix = [bra.site_ind(i, j) for i, j in _where]
#    bnds = [rand_uuid() for _ in range(ng)]
    bnds = [bd+'_' for bd in site_ix]
    TG = overlap[tags_plq[0],'KET'].__class__(G.copy(), 
         inds=site_ix+bnds, left_inds=site_ix)
    TG = bra.fermion_space.move_past(TG)

    reindex_map = dict(zip(site_ix, bnds))
    tids = overlap._get_tids_from_inds(site_ix, which='any')
    for tid_ in tids:
        tsr = overlap.tensor_map[tid_]
        if layer_tags[0] in tsr.tags:
            tsr.reindex_(reindex_map)
    overlap.add_tensor(TG, virtual=True)
#    print('########## overlap ###############')
#    fs = overlap.fermion_space
#    for tid, (tsr,site) in fs.tensor_order.items():
#        print(site,tsr.tags,tsr.inds,tsr.shape)

    xs = dict()
    x_previous = dict()
    previous_cost = None
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
                ket_tid = norm_plq[site,'KET'].get_fermion_info()[0]
                norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))
                overlap._refactor_phase_from_tids((bra_tid,))
                lix = norm_plq[site,'BRA'].inds[1:]
                rix = norm_plq[site,'KET'].inds[:-1]
                pix = norm_plq[site,'BRA'].inds[:1]
                A = (norm_plq.select(site, which='!any',virtual=False)
                     .contract(output_inds=lix+rix))
                print('check hermitian',(A.data-A.data.dagger).norm())
                print('check hermitian',A.data.pattern)
#                def A(x):
#                    Ax = norm_plq.select((site,'BRA'), which='!all',virtual=False)
#                    Ax[site,'KET'].modify(data=x)
#                    return Ax.contract(output_inds=pix+lix).data
                b = (overlap.select((site,'BRA'), which='!all',virtual=False)
                     .contract(output_inds=pix+lix)).data
                print(site)
#                print(norm_plq[site,'KET'].data.pattern[:-1])
#                print(norm_plq[site,'BRA'].data.pattern[1:][::-1])
#                print(overlap[site,'BRA'].data.pattern[1:][::-1])
#                print(Ax.pattern)
#                print(b.pattern[:-1])
#                exit()
                data = BCG1(A.data,norm_plq[site,'KET'].data,b)
                norm_plq[site,'KET'].modify(data=data)
                norm_plq[site,'BRA'].modify(data=data.dagger)
                overlap[site,'BRA'].modify(data=data.dagger)
          
            cost_fid = overlap.contract(output_inds=[])
            cost_norm = norm_plq.contract(output_inds=[])
            cost = -2.0*cost_fid+cost_norm
            print(i,cost)
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost

    if condition_tensors:
        tid_map = {ket1_tid:bra0_site,bra0_tid:bra1_site,bra1_tid:ket1_site}
        norm_plq._reorder_from_tid(tid_map,inplace=True)
        norm_plq._refactor_phase_from_tids((ket0_tid,bra0_tid))
        norm_plq._refactor_phase_from_tids((ket1_tid,bra1_tid))
        ket_plq = norm_plq._select_tids([ket0_tid,ket1_tid])
        if condition_maintain_norms:
            conditioner(
                ket_plq, value=pre_norm, balance_bonds=condition_balance_bonds)
        else:
            conditioner(
                ket_plq, balance_bonds=condition_balance_bonds)
        norm_plq[tags_plq[0],'BRA'].modify(data=ket_plq[tags_plq[0]].data.dagger)
        norm_plq[tags_plq[1],'BRA'].modify(data=ket_plq[tags_plq[1]].data.dagger)
        tid_map = {ket1_tid:ket1_site,bra0_tid:bra0_site,bra1_tid:bra1_site}
        norm_plq._reorder_from_tid(tid_map,inplace=True)

    ket_tids = ket0_tid,ket1_tid
    bra_tids = bra0_tid,bra1_tid
    for i,site in enumerate(tags_plq):
        norm_plq._refactor_phase_from_tids((ket_tids[i],bra_tids[i]))

        tid = ket[site,'KET'].get_fermion_info()[0] 
        ket._refactor_phase_from_tids((tid,))
        ket[site].modify(data=norm_plq[site,'KET'].data)

        tid = bra[site,'BRA'].get_fermion_info()[0] 
        bra._refactor_phase_from_tids((tid,))
        bra[site].modify(data=norm_plq[site,'BRA'].data)

def to_dense(T,match_axes=None):
    # take the first naxes as left indices
    data = T.data.to_sparse()
    print(data)
    dtype = data.blocks[0].dtype
    ndim = len(T.shape)
    # compute total shape
    counted_block = [[] for _ in range(ndim)]
    bond_infos = [[] for _ in range(ndim)]
    for iblk in data.blocks: # for each block
        for ix, iq in enumerate(iblk.q_labels): # for each dimension
            if iq in counted_block[ix]:
                continue
            else:
                counted_block[ix].append(iq)
                bond_infos[ix].append(iblk.shape[ix])
    print(counted_block)
    print(bond_infos)
    counted_block_new = counted_block.copy()
    bond_infos_new = bond_infos.copy()
    if match_axes is not None:
        for tup in match_axes:
            ref_ax = tup[0]
            ref_order = counted_block[ref_ax]
            for ax in tup[1:]:
#                order = counted_block[ax]
#                bond = bond_infos[ax]
#                bond_new = [0 for iq in order]
#                for i,iq in enumerate(order):
#                    if iq in ref_order:
#                        idx = ref_order.index(iq)
#                        bond_new[idx] = bond[i]
#                    else:
#                        raise ValueError
#                counted_block_new[ax] = ref_order
#                bond_infos_new[ax] = bonds_new
    print(counted_block_new)
    print(bond_infos_new)
    shape = tuple([int(sum(ish)) for ish in bond_infos_new])
    x = np.zeros(shape,dtype=dtype)
    for i,blk in enumerate(data.blocks):
        slices = tuple()
        for ix, iq in enumerate(blk.q_labels): # for each dimension
            idx = counted_block_new[ix].index(iq)
            start = sum(bond_infos_new[ix][:idx])
            stop = start+bond_infos_new[ix][idx]
            slices += (slice(start,stop),)
#            print(i,iq,idx,start,stop)
#            exit()
#        print(slices)
        x[slices] = blk
    return x
def to_dense(T,match_axes=None):
    # take the first naxes as left indices
    data = T.data.to_sparse()
    print(data)
    dtype = data.blocks[0].dtype
    ndim = len(T.shape)
    # compute total shape
    counted_block = [[] for _ in range(ndim)]
    bond_infos = [[] for _ in range(ndim)]
    for iblk in data.blocks: # for each block
        for ix, iq in enumerate(iblk.q_labels): # for each dimension
            if iq in counted_block[ix]:
                continue
            else:
                counted_block[ix].append(iq)
                bond_infos[ix].append(iblk.shape[ix])
    print(counted_block)
    print(bond_infos)
    counted_block_new = counted_block.copy()
    bond_infos_new = bond_infos.copy()
    if match_axes is not None:
        for tup in match_axes:
            ref_ax = tup[0]
            ref_order = counted_block[ref_ax]
            ref_bonds = bond_infos[ref_ax]
            for ax in tup[1:]:
            # presumably the dimensions match as well
                counted_block_new[ax] = ref_order
                bond_infos_new[ax] = ref_bonds
    print(counted_block_new)
    print(bond_infos_new)
    shape = tuple([int(sum(ish)) for ish in bond_infos_new])
    x = np.zeros(shape,dtype=dtype)
    for i,blk in enumerate(data.blocks):
        slices = tuple()
        for ix, iq in enumerate(blk.q_labels): # for each dimension
            idx = counted_block_new[ix].index(iq)
            start = sum(bond_infos_new[ix][:idx])
            stop = start+bond_infos_new[ix][idx]
            slices += (slice(start,stop),)
#            print(i,iq,idx,start,stop)
#            exit()
#        print(slices)
        x[slices] = blk
    return x
def inv_u(mat):
    mat = mat.to_sparse() 
    dtype = mat.block[0].dtype
    full = np.zeros((4,4),dtype=dtype)
    for blk in mat.blocks:
        slices = tuple([insert_map[q] for q in blk.q_labels])
        full[slices] = blk
    if abs(np.linalg.det(full))<singular_thresh:
        return None
    full_inv = np.linalg.inv(full)
    blks = []
    for q1,q2 in product(insert_map.keys(),repeat=2):
        if q1.n<q2.n:
            slices = tuple([insert_map[q] for q in [q1,q2]])
            blks.append(SubTensor(reduced=full_inv[slices],q_labels=(q1,q2)))
    mat_inv = SparseFermionTensor(blocks=blks,
              pattern=mat.pattern,shape=mat.shape)
    return mat_inv.to_flat()
def proj(tsr,q_label):
    tsr = tsr.to_sparse()
    dtype = tsr.blocks[0].dtype
    blks = []
    for blk in tsr.blocks:
        if blk.q_labels[-1]==q_label:
            data = np.eye(blk.shape[-1],dtype=dtype)
            blks.append(blk.__class__(reduced=data,q_labels=blk.q_labels))
        else:
            shape = blk.shape[x:-1],dim_map[q_label]
            q_labels = blk.q_labels[-1],q_label
            data = np.zeros(shape,dtype=dtype)
            blks.append(blk.__class__(reduced=data,q_labels=q_labels))
    pattern = inv_map[tsr.pattern[-1]]+tsr.pattern[-1]
    p = tsr.__class__(blks,pattern=pattern)
    return p.to_flat()
def _a(r,p,gamma=1.0):
    denom = np.tensordot(p.dagger,A(p),axes=(axs1,axs2))
    denom = inv(denom)
    if denom is None:
        return None
    num = r.copy() 
    if isinstance(gamma,num.__class__): 
        num = np.tensordot(num,gamma,axes=((-1,),(0,)))
    elif isinstance(gamma,float):
        num *= gamma
    else:
        raise NotImplementedError
    num = np.tensordot(num.dagger,r,axes=(axs1,axs2))
    return np.tensordot(denom,num,axes=((-1,),(0,)))
def _b(r_num,r_denom,gamma_inv=None):
    denom = np.tensordot(r_denom.dagger,r_denom,axes=(axs1,axs2))
    denom = inv(denom)
    if denom is None:
        return None
    num = np.tensordot(r_num.dagger,r_num,axes=(axs1,axs2))
    out = np.tensordot(denom,num,axes=((-1,),(0,)))
    if isinstance(gamma_inv,out.__class__):
        out = np.tensordot(gamma_inv,out,axes=((-1,),(0,)))
    elif isinstance(gamma_inv,float):
        out *= gamma_inv
    else:
        raise NotImplementedError
    return out
def proj(q_target,dim_map,dtype,SPARSE,SUBTENSOR):
    blks = []
    for q in list(dim_map.keys()):
        q_labels = q,q_target
        if q==q_target:
            data = np.eye(dim_map[q],dtype=dtype)
            blks.append(SUBTENSOR(reduced=data,q_labels=q_labels))
        else:
            shape = dim_map[q],dim_map[q_target]
            data = np.zeros(shape,dtype=dtype)
            blks.append(SUBTENSOR(reduced=data,q_labels=q_labels))
    p = SPARSE(blks,pattern='-+')
    print(p)
    return p.to_flat()
def BCG1(A,x0,b,max_iter=1000,singular_thresh=1e-6):
    tmp = x0.to_sparse()
    SYMMETRY = tmp.dq.__class__
    SPARSE = tmp.__class__
    SUBTENSOR = tmp.blocks[0].__class__
    dtype= tmp.blocks[0].dtype
    dim_map = {SYMMETRY(0):1,SYMMETRY(1):2,SYMMETRY(2):1}

    nvir = len(x0.shape)-1
    axes = list(range(nvir,0,-1)),list(range(nvir))
    x = []
    print(x0.dq,b.dq)
    print(x0.pattern,b.pattern)
    for q in list(dim_map.keys()):
        p = proj(q,dim_map,dtype,SPARSE,SUBTENSOR)
        x0_ = np.tensordot(x0,p,axes=((-1,),(0,)))
        b_  = np.tensordot(b ,p,axes=((-1,),(0,)))
        print(p)
#        print(x0_.pattern)
#        print(x0_)
#        print(b_.pattern)
#        print(b_)
#        print()        
        x_ = BCG1_blk(A,x0_,b_,tmp.dq,axes=axes,
             dtype=dtype,SPARSE=SPARSE,SUBTENSOR=SUBTENSOR,
             max_iter=max_iter,singular_thresh=singular_thresh)
#        x += x_.blocks
    exit()
def BCG1_(A,x0,b,max_iter=1000,singular_thresh=1e-6):
    x0 = x0.to_sparse()
    b  = b.to_sparse()
    x_map = dict()
    for blk in x0.blocks:
        if blk.q_labels[-1] in list(x_map.keys()):
            x_map[blk.q_labels[-1]].append(blk)
        else:
            x_map[blk.q_labels[-1]] = [blk]
    b_map = dict()
    for blk in b.blocks:
        if blk.q_labels[-1] in list(b_map.keys()):
            b_map[blk.q_labels[-1]].append(blk)
        else:
            b_map[blk.q_labels[-1]] = [blk]
    print('######### x0 #############')
    print(x0)
    x = []
    for q in b_map:
        x0_ = x0.__class__(blocks=x_map[q],pattern=x0.pattern)
        x0_ = x0_.to_flat()
        b_ = b.__class__(blocks=b_map[q],pattern=b.pattern)
        b_ = b_.to_flat()
        print('####### x0_ #############')
        print(x0_)
#        x_ = BCG1_blk(A,x0_,b_,max_iter=max_iter,singular_thresh=singular_thresh)
#        x += x_.blocks
        exit()
def BCG1(A,x0,b,
         max_iter=1000,singular_thresh=1e-6):
    x0 = x0.to_sparse()
    b  = b.to_sparse()

    dq = b.dq
    SYMMETRY = dq.__class__
    SPARSE = b.__class__
    SUBTENSOR = b.blocks[0].__class__
    dtype= b.blocks[0].dtype

    nvir = len(b.shape)-1
    axes = list(range(nvir,0,-1)),list(range(nvir))
    dim_map = {SYMMETRY(0):1,SYMMETRY(1):2,SYMMETRY(2):1}

    def get_qmap(tsr):
        tsr = tsr.to_sparse()
        q_map = dict()
        for blk in tsr.blocks:
            if blk.q_labels[-1] in list(q_map.keys()):
                q_map[blk.q_labels[-1]].append(blk)
            else:
                q_map[blk.q_labels[-1]] = [blk]
        return q_map
    x_map = get_qmap(x0)
    b_map = get_qmap(b)
    def extract_blks(q_map,q_target,tsr):
        tsr = tsr.to_sparse()
        
def BCG1_blk(A,x0,b,axes=None,
             dq=None,dtype=None,SYMMETRY=None,SPARSE=None,SUBTENSOR=None,
             max_iter=1000,singular_thresh=1e-6):
    if SPARSE is None: 
        tmp = x0.to_sparse()
        dq = tmp.dq
        SYMMETRY = dq.__class__
        SPARSE = tmp.__class__
        SUBTENSOR = tmp.blocks[0].__class__
        dtype= tmp.blocks[0].dtype
    if axes is None:
        nvir = len(x0.shape)-1
        axes = list(range(nvir,0,-1)),list(range(nvir))
    def inv(mat):
        mat = mat.to_sparse()
        data = np.asarray(mat.blocks[0])
        if abs(np.linalg.det(data))<singular_thresh:
            return None
        data = np.linalg.inv(data)
        mat_inv = SUBTENSOR(reduced=data,q_labels=mat.blocks[0].q_labels)
        mat_inv = SPARSE(blocks=[mat_inv],pattern=mat.pattern,shape=mat.shape)
        return mat_inv.to_flat()
    def _a(r,p,gamma=1.0):
        denom = np.tensordot(p.dagger,A(p),axes=(axs1,axs2))
        denom = inv(denom)
        if denom is None:
            return None
        num = r.copy() 
        if isinstance(gamma,num.__class__):
            num = np.tensordot(num,gamma,axes=((-1,),(0,)))
        else:
            num *= gamma
        num = np.tensordot(num.dagger,r,axes=(axs1,axs2))
        return np.tensordot(denom,num,axes=((-1,),(0,)))
    def _b(r_num,r_denom,gamma_inv=1.0):
        denom = np.tensordot(r_denom.dagger,r_denom,axes=(axs1,axs2))
        denom = inv(denom)
        if denom is None:
            return None
        num = np.tensordot(r_num.dagger,r_num,axes=(axs1,axs2))
        out = np.tensordot(denom,num,axes=((-1,),(0,)))
        if isinstance(gamma_inv,out.__class__):
            out = np.tensordot(gamma_inv,out,axes=((-1,),(0,)))
        else:
            out *= gamma_inv
        return out
    def parse(tsr,ax):
        tsr = tsr.to_sparse()
        blks = []
        for blk in tsr.blocks:
            q_labels = list(blk.q_labels)
            q_labels[ax] = -q_labels[ax]
            blks.append(SUBTENSOR(reduced=np.asarray(blk),q_labels=q_labels))
        pattern = [c for c in tsr.pattern]
        pattern[ax] = inv_map[pattern[ax]]
        pattern = ''.join(pattern)
        new = SPARSE(blks,pattern=pattern,shape=tsr.shape)
        return new.to_flat()
    def check_symmetry(tsr,dq):
        tsr = tsr.to_sparse()
        blks = []
        qs = []
        for blk in tsr.blocks:
            dq_i = dq.__class__._compute(tsr.pattern,blk.q_labels) 
            if dq_i==dq:
                blks.append(blk)
            else:
                assert blk.norm()<singular_thresh
                qs.append(blk.q_labels)
        tsr = SPARSE(blks,pattern=tsr.pattern)
        return tsr.to_flat(),qs
#    def qr(pk):
         
    r0 = b-A(x0)
    g0 = gi0 = 1.0
    p0 = r0.copy()
    print('r_norm=',r0.norm())

    xk,rk,pk,gk,gik = x0,r0,p0,g0,gi0
    for k in range(max_iter):
        ak = _a(rk,pk,gamma=gk)
        if ak is None:
            break
        pa = np.tensordot(pk,ak,axes=((-1,),(0,)))
        xk_ = xk+pa
        rk_ = rk-A(pa)
#        rk_ = b-A(x0)
        bk = _b(rk_,rk,gamma_inv=gik)
        if bk is None:
            break
        pk_ = rk_+np.tensordot(pk,bk,axes=((-1,),(0,)))
        print('pk')
        print(pk_)
        exit()
        if pk_.shape[-1]>1:
            print(pk_)
            print(pk_.dq)
            print(pk_.pattern)
            pk_,gik_ = pk_.tensor_qr(left_idx=axs2,right_idx=(2,),mod='qr')
            print(gik_)
            print(gik_.pattern)
            print(gik_.dq)
            print(pk_)
            print(pk_.pattern)
            print(pk_.dq)
            exit()
            pk_  = parse(pk_,-1)
            gik_ = parse(gik_,0)
            gk_ = inv(gik_)
        else:
            gik_ = pk_.norm()
            pk_ /= gik_
            gk_ = 1.0/gik_
        xk,rk,pk,gk,gik = xk_,rk_,pk_,gk_,gik_

        r = b-A(xk)
        print('iter={},r_norm={}'.format(k,r.norm()))
    return xk.to_sparse()
def I(tsr): # invert virtual pattern
    q_labels = tsr.q_labels.copy()
    for i in range(nvir):
        narr = q_labels[:,i]
        narr = [tsr.symmetry.from_flat(q) for q in narr]
        narr = [-q for q in narr]
        narr = [tsr.symmetry.to_flat(q) for q in narr]
        q_labels[:,i] = np.array(narr)
    pattern = [inv_map[c] for c in tsr.pattern]
    pattern[-1] = inv_map[pattern[-1]]
    pattern = ''.join(pattern)
    return x0.__class__(q_labels=q_labels,shapes=tsr.shapes,data=tsr.data,
           pattern=pattern,idxs=tsr.idxs,symmetry=tsr.symmetry,shape=tsr.shape)
def BCG1(A,x0,b,max_iter=10,singular_thresh=1e-6):
    nvir = len(x0.shape)-1
    axs1,axs2 = list(range(nvir,0,-1)),list(range(nvir))
    def inv(mat):
        mat = mat.to_sparse() 
        blks = []
        for blk in mat.blocks:
            assert blk.q_labels[0]==blk.q_labels[1]
            data = np.asarray(blk)
            if abs(np.linalg.det(data))<singular_thresh:
                return None
            data = np.linalg.inv(data)
            blks.append(blk.__class__(reduced=data,q_labels=blk.q_labels))
        mat_inv = mat.__class__(blocks=blks,pattern=mat.pattern,shape=mat.shape)
        return mat_inv.to_flat()
    def _a(r,p,gamma=None):
        denom = np.tensordot(p.dagger,A(p),axes=(axs1,axs2))
        denom = inv(denom)
        if denom is None:
            return None
        num = r.copy() 
        if gamma is not None:
            num = np.tensordot(num,gamma,axes=((-1,),(0,)))
        num = np.tensordot(num.dagger,r,axes=(axs1,axs2))
        return np.tensordot(denom,num,axes=((-1,),(0,)))
    def _b(r_num,r_denom,gamma_inv=None):
        denom = np.tensordot(r_denom.dagger,r_denom,axes=(axs1,axs2))
        denom = inv(denom)
        if denom is None:
            return None
        num = np.tensordot(r_num.dagger,r_num,axes=(axs1,axs2))
        out = np.tensordot(denom,num,axes=((-1,),(0,)))
        if gamma_inv is not None:
            out = np.tensordot(gamma_inv,out,axes=((-1,),(0,)))
        return out
    def parse_gamma(tsr):
        tsr = tsr.to_sparse()
        blks = []
        keep,discard = [],[]
        for blk in tsr.blocks:
            data = np.asarray(blk)
            if abs(np.linalg.det(data))<singular_thresh:
                discard.append(blk.q_labels[-1])
            else:
                keep.append(blk.q_labels[-1])
                q_labels = blk.q_labels[-1],blk.q_labels[-1]
                blks.append(blk.__class__(reduced=np.asarray(blk),q_labels=q_labels))
        if len(keep)>0:
            pattern = inv_map[tsr.pattern[0]]+tsr.pattern[1]
            new = tsr.__class__(blks,pattern=pattern)
            return new.to_flat(),keep,discard
        else:
            return None,keep,discard
    def parse(tsr,neg=False,keep=[]):
        tsr = tsr.to_sparse()
        blks = []
        for blk in tsr.blocks:
            q_labels = list(blk.q_labels)
            if neg:
                q_labels[-1] = -q_labels[-1]
            if q_labels[-1] in keep:
                blks.append(blk.__class__(reduced=np.asarray(blk),q_labels=q_labels))
        pattern = [c for c in tsr.pattern]
        if neg:
            pattern[-1] = inv_map[pattern[-1]]
        pattern = ''.join(pattern)
        new = tsr.__class__(blks,pattern=pattern)
        return new.to_flat()
    def iteration(A,x0,b):
#        print('####### x0 #########')
#        print(x0)
        r0 = b-A(x0)
        g0 = gi0 = None
        p0 = r0.copy()
        xk,rk,pk,gk,gik = x0,r0,p0,g0,gi0
        print('r_norm=',r0.norm())
        keep,discard = [],[]
#        print(r0.pattern)
#        print(r0)
#        print(p0.pattern)
#        print(p0)
#        print(x0.pattern)
#        print(x0)
#        exit()
        for k in range(max_iter):
            ak = _a(rk,pk,gamma=gk)
#            print('ak')
#            print(ak)
            if ak is None:
#                print('ak')
                break
            pa = np.tensordot(pk,ak,axes=((-1,),(0,)))
            xk_ = xk+pa
            rk_ = rk-A(pa)
            bk = _b(rk_,rk,gamma_inv=gik)
#            print('bk')
#            print(bk)
            if bk is None:
#                print('bk')
                break
            pk_ = rk_+np.tensordot(pk,bk,axes=((-1,),(0,)))
#            print('####### pk #########')
#            print(pk_.pattern)
#            print(pk_)
            pk_,gik_ = pk_.tensor_qr(left_idx=axs2,mod='qr')
#            print('####### gik ##############')
#            print(gik_.pattern)
#            print(gik_)
#            print('####### pk #########')
#            print(pk_.pattern)
#            print(pk_)

            gik,keep_,discard_ = parse_gamma(gik_)
            if gik is None:
                xk = xk_
#                r = b-A(xk)
#                print('iter={},r_norm={}'.format(k,r.norm()))
                break
            keep += keep_
            discard += discard_
            gk = inv(gik)
#            print('####### parse gik ##############')
#            print(gik.pattern)
#            print(gik)
            pk = parse(pk_,neg=True,keep=keep_)
#            print('####### parsed pk #########')
#            print(pk.pattern)
#            print(pk)
            if len(discard_)>0:
                xk = parse(xk_,neg=False,keep=keep_)
#                print('####### parsed xk #########')
#                print(xk.pattern)
#                print(xk)
                rk = parse(rk_,neg=False,keep=keep_)
#                print('####### parsed xk #########')
#                print(rk.pattern)
#                print(rk)
            else:
                xk,rk = xk_,rk_
            r = b-A(xk)
            print('iter={},r_norm={}'.format(k,r.norm()))
        return xk.to_sparse(),keep,discard
#    r = b-A(x0)
#    print('r_norm={}'.format(r.norm()))
    blks = []
    xk,keep,discard = iteration(A,x0,b)
    blks += xk.blocks
    while len(discard)>0:
        x0 = parse(x0,neg=False,keep=discard)
        b  = parse(b ,neg=False,keep=discard)
        xk,keep,discard = iteration(A,x0,b)
        blks += xk.blocks
    x = xk.__class__(blks,pattern=x0.pattern)
    x = x.to_flat()
#    print(x)
#    r = b-A(x)
#    print('r_norm={}'.format(r.norm()))
    return x

#                print(site)
                #tid_map = {ket_tids[site]:0,bra_tids[site]:last}
                tid_map = {ket_tids[site]:0,bra_tids[site]:last}
                norm_plq._reorder_from_tid(tid_map,inplace=True)
#                norm_plq._refactor_phase_from_tids((ket_tids[site],))
#                norm_plq._refactor_phase_from_tids((bra_tids[site],))
                norm_plq._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))

                overlap._reorder_from_tid(tid_map,inplace=True)
                overlap._refactor_phase_from_tids((ket_tids[site],bra_tids[site]))

                lix = norm_plq[site,'BRA'].inds[1:]
                rix = norm_plq[site,'KET'].inds[:-1]
                pix = norm_plq[site,'BRA'].inds[:1]
                b = (overlap.select((site,'BRA'), which='!all',virtual=False)
                     .contract(output_inds=pix+lix)).data

                A = (norm_plq.select(site, which='!any',virtual=False)
                     .contract(output_inds=lix+rix)).data
                print('check hermitian',(A-A.dagger).norm())
                def _A(x):
                    Ax = norm_plq.select((site,'BRA'), which='!all',virtual=False)
                    Ax[site,'KET'].modify(data=x)
                    return Ax.contract(output_inds=lix+pix).data
                nvir = len(lix)
                x = norm_plq[site,'KET'].data
                y = b.dagger
                y.pattern = x.pattern
                Ax = _A(x)
                Ay = _A(y)
                Ax_ = np.tensordot(A,x,axes=(range(nvir,2*nvir),range(nvir)))
                Ay_ = np.tensordot(A,y,axes=(range(nvir,2*nvir),range(nvir)))
                Ay__ = np.tensordot(A.dagger,y,axes=(range(nvir,2*nvir),range(nvir)))
                print('check contraction:',(Ax-Ax_).norm())
                print('check contraction:',(Ay-Ay_).norm())
                print('check contraction:',(Ay-Ay__).norm())

                inv_map = {'+':'-','-':'+'}
                Ax.pattern = ''.join([inv_map[c] for c in Ax.pattern[:-1]])+Ax.pattern[-1]
                Ay.pattern = ''.join([inv_map[c] for c in Ay.pattern[:-1]])+Ay.pattern[-1]
                lhs = np.tensordot(Ax,y,axes=(range(nvir-1,-1,-1),range(nvir)))
                print(lhs)
                rhs = np.tensordot(x,Ay,axes=(range(nvir),range(nvir-1,-1,-1)))
                print(rhs)
                print('check hermitian',(lhs-rhs).norm())
                exit()
def get_eo_(tsr):
    tsr = tsr.to_sparse()
    even,odd = [],[]
    for blk in tsr.blocks:
        if blk.q_labels[-1].parity==0:
            even.append(blk)
        else:
            odd.append(blk)
    even = tsr.__class__(even,pattern=tsr.pattern)
    odd  = tsr.__class__(odd ,pattern=tsr.pattern)
    return even.to_flat(),odd.to_flat()

def BGMRES_(A,x,b,max_space=10,singular_thresh=1e-6):
    nvir = len(x.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    r = b-A(x)
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
    def qr(tsr):
        q,r = tsr.tensor_qr(left_idx=axs2,mod='qr') 
        return parse(q,-1),parse(r,0)
    def smin(h):
        s = 1.0
        for i in range(h.n_blocks):
            data = h.data[h.idxs[i]:h.idxs[i+1]].reshape(h.shapes[i])
            for j in range(data.shape[0]):
                s = min(s,abs(data[j,j]))
        return s 
    v,r_ = qr(r)
    V = [v]
    H = []
#    print(v.pattern)
#    print(v)
    for j in range(max_space):
        u = A(V[-1])
        Hj = [np.tensordot(v.dagger,u,axes=(axs1,axs2)) for v in V]
        u = u-sum([np.tensordot(V[l],Hj[l],axes=((-1,),(0,))) 
                   for l in range(len(V))])
        if u.norm()<singular_thresh:
            break
        lhs  = [np.tensordot(v.dagger,u,axes=(range(nvir,-1,-1),range(nvir+1))) 
               for v in V]
        lhs_ = [np.tensordot(v.dagger,v,axes=(range(nvir,0,-1),range(nvir))) 
               for v in V]
        if sum(lhs)>singular_thresh:
            print(u.norm())
            print('check V',lhs)
#            print('check V',lhs_)
            exit()
        v,h = qr(u)
        if smin(h)<singular_thresh:
            break
        V.append(v)
        Hj.append(h)
        H.append(Hj)
#        print('Hj,H,V',len(Hj),len(H),len(V))
    # H.T*H
#    print(len(H),len(H[-1]),len(V))
    m = len(H)
    num = np.array([np.tensordot(Hj[0].dagger,r_,axes=((1,0),(0,1))) for Hj in H])
    denom = np.zeros((m,m),dtype=num.dtype)
    for i in range(m):
        for j in range(i+1):
#            print(i,j,len(H[i]),len(H[j]))
            val = sum([np.tensordot(H[i][k].dagger,H[j][k],axes=((1,0),(0,1))) 
                       for k in range(len(H[j]))])
            denom[i,j] = denom[j,i] = val
    y = np.dot(np.linalg.inv(denom),num)
    x = x+sum([V[i]*y[i] for i in range(m)])
#    print('r0 norm=',r.norm())
    r = b-A(x)

    perturb = np.random.rand(m)*1e-3
    y_ = y+perturb
    x_ = x+sum(V[i]*y_[i] for i in range(m))
    r_ = b-A(x_)
    if r.norm()-r_.norm()>0.0:
        print('r  norm=',r.norm())
        print('r_ norm=',r_.norm())
    return x
def BGMRES(A,x0,b,max_space=10,max_iter=50,cutoff=1e-6,singular_thresh=1e-6):
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
#    max_iter = 1
    def blk(x,b):
        norm = b.norm()
        b_ = b/norm
        xold = x.copy()/norm
        for i in range(max_iter):
            x = BGMRES_(A,xold,b_,max_space=max_space,singular_thresh=singular_thresh)
            r_norm = (b_-A(x)).norm()
            dx = (x-xold).norm()
            print('iter={},dx={},r_norm={}'.format(i,dx,r_norm))
            if dx<cutoff:
                break
            xold = x.copy()
        return x*norm
    x = blk(x0,b)
#    xs = [blk(xs[i],bs[i]) for i in [0,1]]
#    q_labels = np.concatenate([x.q_labels for x in xs],axis=0)
#    shapes = np.concatenate([x.shapes for x in xs],axis=0)
#    data = np.concatenate([x.data for x in xs],axis=0)
#    x = x0.__class__(q_labels=q_labels,shapes=shapes,data=data,
#                     pattern=x0.pattern,symmetry=x0.symmetry)
#    print(xs[0])
#    print(xs[1])
#    print(x)
#    exit()
    return x
#    def match_phase(ref_T,T):
#        ref_global_flip = ref_T.phase.get('global_flip',False)
#        ref_local_inds  = ref_T.phase.get('local_inds',[])
#        global_flip = T.phase.get('global_flip',False)
#        local_inds  = T.phase.get('local_inds',[])
#
#        global_flip = (ref_global_flip!=global_flip)
#        local_inds = set(ref_local_inds).symmetric_difference(set(local_inds))
#        return T.flip(global_flip=global_flip,local_inds=local_inds,inplace=False)

def gate_full_update_als(norm,ovlp,tags_plq,steps,tol,max_bond,
    optimize='auto-hq',solver='solve',
    enforce_pos=False,pos_smudge=1e-6):
    def contract(ftn,site,output_inds):
        pop = ftn[site,'BRA']
        ctr = ftn.select((site,'BRA'), which='!all')
        ctr.add_tag('contract')
        ftn.contract_tags('contract',inplace=True,output_inds=output_inds)
        assert ftn.num_tensors==2
        ctr = ftn['contract']
        tid = ctr.get_fermion_info()[0]
        ftn._refactor_phase_from_tids((tid,))

        global_flip = ctr.phase.get('global_flip',False)
        local_inds = ctr.phase.get('local_inds',[])
        assert global_flip==False
        assert len(local_inds)==0
        return pop,ctr.data
    def flip(data,T):
        # add phase to phaseless data 
        # remove phase from phased data
        global_flip = T.phase.get('global_flip',False)
        local_inds  = T.phase.get('local_inds',[])
        if global_flip:
            data._global_flip()
        if len(local_inds)>0:
            axes = [T.inds.index(ind) for ind in local_inds]
            data._local_flip(axes)
        return data
    cost_fid = ovlp.contract(output_inds=[])
    cost_norm = norm.contract(output_inds=[])
    cost = -cost_fid+cost_norm
    print('init cost={},ovlp={},norm={}'.format(cost,cost_fid,cost_norm))

    xs = dict()
    x_previous = dict()
    previous_cost = None
    steps = 10
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                output_inds = norm[site,'BRA'].inds[::-1]
                ovlp_ = ovlp.copy()
                tsr1,b = contract(ovlp_,site,output_inds)
                tmp = ovlp_.contract(output_inds=[])
                if abs(cost_fid-tmp)>tol:
                    print('check ovlp',abs(cost_fid-tmp))

                def A(x):
                    data = flip(x.copy(),norm[site,'KET'])

                    norm_ = norm.copy()
                    norm_[site,'KET'].modify(data=data)
                    tsr2,Ax = contract(norm_,site,output_inds)
                    return Ax
                x0 = flip(norm[site,'KET'].data.copy(),norm[site,'KET'])
                data = BGMRES(A,x0,b)
                data = flip(data,norm[site,'KET'])

                norm_ = norm.copy()
                norm_[site,'KET'].modify(data=data.copy())
                tsr2,Ax = contract(norm_,site,output_inds)
                cost_norm_ = norm_.contract(output_inds=[])

                norm[site,'KET'].modify(data=data.copy())
                cost_norm = norm.contract(output_inds=[])
                print(cost_norm,cost_norm_)
                assert abs(cost_norm-cost_norm_)<tol

#            cost_fid = ovlp.contract(output_inds=[])
            cost_norm = norm.contract(output_inds=[])
            cost = -cost_fid+cost_norm
            print('iteration={},cost={},norm={}'.format(i,cost,cost_norm))
            print('')
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost
    return norm
def ovlp(Hi,Hj):
    len_ = min(len(Hi),len(Hj))
    return sum([np.tensordot(Hi[k].dagger,Hj[k],axes=((1,0),(0,1)))  
                for k in range(len_)])
def gs(T,Hk,m,tol=1e-6):
    # assumes T is orthonormal
    Tk = [h.copy() for h in Hk]
    Rk = [0.0 for i in range(m)]
    for j,Tj in enumerate(T):
        Rk[j] = ovlp(Tj,Hk)
        for i in range(len(Tj)):
            Tk[i] = Tk[i]-Rk[-1]*Tj[i]
    norm = np.sqrt(ovlp(Tk,Tk))
    Rk[len(T)] = norm
    T.append([h/norm for h in Tk])
    assert abs(ovlp(T[-1],T[-1])-1.0)<tol
    print(Rk)
    for i in range(len(Hk)):
        rhs = []
        for j in range(len(T)):
            if len(T[j])>i:
                rhs.append(T[j][i]*Rk[j])
        rhs = sum(rhs)
        assert (Hk[i]-rhs).norm()<tol
    return T,Rk
def restart(A,x0,b,max_space=10,tol=1e-4):
    tol = 1e-6

    nvir = len(x0.shape)-1
    axs1,axs2 = range(nvir,0,-1),range(nvir)
    r0 = b-A(x0)
    if r0.norm()<tol:
        return x0
    v,r = svd(r0,left_idx=axs2,cutoff=tol)
    V = [v]
    H = []
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
            for k in range(len(V)):
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
        Hj.append(h)
        H.append(Hj)

#    for i in range(len(H)):
#        lhs = A(V[i])
#        assert len(H[i])==i+2
#        rhs = sum([np.tensordot(V[j],H[i][j],axes=((-1,),(0,))) for j in range(i+2)])
#        print(lhs.norm(),(lhs-rhs).norm())
#        assert (lhs-rhs).norm()<tol

    # QR on H
    m = len(H)
    print(m)
    T = []
    R = [[] for i in range(m)]
    for j,Hj in enumerate(H): 
        print(j)
        T,R[j] = gs(T,Hj,m)
    R = np.array(R)
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
def gate_full_update_als(norm_plq,overlap,tags_plq,steps,tol,max_bond,
    optimize='auto-hq',solver='lsqr',pos_smudge=1e-6):
    cost_fid = overlap.contract(output_inds=[])
    cost_norm = norm_plq.contract(output_inds=[])
    cost = -cost_fid+cost_norm
    print('init cost={},norm_plq={}'.format(cost,cost_norm))
    def contract(ftn,site,output_inds):
        bra_pop = ftn[site,'BRA']
        original_phase = bra_pop.phase
        tid = bra_pop.get_fermion_info()[0]
    
        ctr = ftn.select((site,'BRA'), which='!all')
        ctr.add_tag('contract')
        ftn.contract_tags('contract',inplace=True,output_inds=output_inds)
        assert ftn.num_tensors==2
        ftn._refactor_phase_from_tids((tid,))
        global_flip = bra_pop.phase.get('global_flip',False)
        local_inds = bra_pop.phase.get('local_inds',[])
        assert global_flip == False 
        assert len(local_inds) == 0
        return ftn, ftn['contract'].data
        
    xs = dict()
    x_previous = dict()
    previous_cost = None
    with contract_strategy(optimize), shared_intermediates():
        for i in range(steps):
            for site in tags_plq:
                ket_tid = norm_plq[site,'KET'].get_fermion_info()[0] 
                bra_tid = norm_plq[site,'BRA'].get_fermion_info()[0]
                norm_plq.fermion_space.move(ket_tid,0)
                norm_plq.fermion_space.move(bra_tid,norm_plq.num_tensors-1)
                norm_plq._refactor_phase_from_tids((ket_tid,bra_tid))
                bra1 = norm_plq[site,'BRA']
                ket1 = norm_plq[site,'KET']
                global_flip = bra1.phase.get('global_flip',False)
                local_inds = bra1.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
                global_flip = ket1.phase.get('global_flip',False)
                local_inds = ket1.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0

                bra_tid = overlap[site,'BRA'].get_fermion_info()[0]
                overlap.fermion_space.move(bra_tid,overlap.num_tensors-1)
                overlap._refactor_phase_from_tids((bra_tid,))
                bra2 = overlap[site,'BRA']
                global_flip = bra2.phase.get('global_flip',False)
                local_inds = bra2.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
                assert (bra2.data-bra1.data).norm()<1e-6

                # forming b
                ovlp = overlap.copy()
                pop = ovlp[site,'BRA']
                pop.add_tag('pop')
                ctr = ovlp.select((site,'BRA'), which='!all')
                ctr.add_tag('ctr')
                ovlp.contract_tags('ctr',inplace=True,output_inds=pop.inds[::-1])
                assert ovlp.num_tensors==2
                tid = ovlp['pop'].get_fermion_info()[0]
                ovlp._refactor_phase_from_tids((tid,))
                global_flip = ovlp['pop'].phase.get('global_flip',False)
                local_inds  = ovlp['pop'].phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
                b = ovlp['ctr']

                # forming A
                norm = norm_plq.copy()
                if dense:
                    output_inds = norm[site,'BRA'].inds[1:][::-1]
                    ctr = norm.select(site,which='!any')
                    ctr.add_tag('ctr')
                    norm.contrac_tags('ctr',inplace=True,output_inds=output_inds,
                                      optimtize=optimize)
                    assert norm.num_tensors==3
                    tid_bra = norm[site,'BRA'].get_fermion_info()[0]
                    tid_ket = norm[site,'KET'].get_fermion_info()[0]
                    norm._refactor_phase_from_tids((tid_bra,tid_ket))
                    global_flip = norm[site,'BRA'].phase.get('global_flip',False)
                    local_inds  = norm[site,'BRA'].phase.get('local_inds',[])
                    assert global_flip == False 
                    assert len(local_inds) == 0
                    assert (norm[site,'BRA'].data-norm_plq[site,'BRA'].data).norm()<1e-6
                    global_flip = norm[site,'KET'].phase.get('global_flip',False)
                    local_inds  = norm[site,'KET'].phase.get('local_inds',[])
                    assert global_flip == False 
                    assert len(local_inds) == 0
                    assert (norm[site,'KET'].data-norm_plq[site,'KET'].data).norm()<1e-6
                solver_opts = dict(atol=tol,btol=tol)
                data = slice_solve(norm_plq.copy(),b,x0,site,solver,
                                   optimize=optimize,**solver_opts)

                norm_plq[site,'KET'].modify(data=data)
#                data = norm_plq[site,'KET'].data.dagger
#                norm_plq[site,'BRA'].modify(data=data)
#                overlap[site,'BRA'].modify(data=data)
          
            cost_fid = overlap.contract(output_inds=[])
            cost_norm = norm_plq.contract(output_inds=[])
            cost = -cost_fid+cost_norm
            print('iteration={},cost={},norm={}'.format(i,cost,cost_norm))
#            print('')
            converged = (previous_cost is not None)and(abs(cost-previous_cost)<tol)
            if converged:
                break
            previous_cost = cost
    return norm_plq
def flat2slice(tsr):
#    print(tsr)
    tsr = tsr.to_sparse()
    blk_map = dict()
    for blk in tsr.blocks:
        q_lab = blk.q_labels[-1]
        if q_lab in blk_map.keys():
            blk_map[q_lab].append(blk)
        else:
            blk_map[q_lab] = [blk]
#    for q_lab in blk_map:
#        print(q_lab)
#        for blk in blk_map[q_lab]:
#            print(blk)
    tsr_map = {q_lab:[] for q_lab in blk_map.keys()}
    pattern = tsr.pattern[:-1]
    for q_lab,blks in blk_map.items():
        dim = blks[0].shape[-1]
        for i in range(dim):
            blks_i = []
            for blk in blks:
                q_labels = blk.q_labels[:-1]
                data = np.asarray(blk)[...,i]
                blks_i.append(blk.__class__(reduced=data,q_labels=q_labels))
            tsr_i = tsr.__class__(blocks=blks_i,pattern=pattern)
            tsr_map[q_lab].append(tsr_i.to_flat())
#    for q_lab in tsr_map:
#        print(q_lab)
#        for i,tsr in enumerate(tsr_map[q_lab]):
#            print(i,tsr)
#    exit() 
    return tsr_map 
def slice2flat(tsr_map):
    tsr = list(tsr_map.values())[0]
    pattern = tsr.pattern
    sparse = tsr.__class__
    sub = tsr.blocks[0].__class__

    for q_lab in tsr_map:
        print(q_lab)
        for i,tsr in enumerate(tsr_map[q_lab]):
            print(i,tsr)
#    exit() 
    BLKS = []
    for q_lab,tsrs in tsr_map.items():
        dim = len(tsrs)
        blks = dict()
        for i,tsr in enumerate(tsrs):
            for blk in tsr.blocks:
                q_labels = blk.q_labels
                data = np.asarray(blk)
                if i==0:
                    blks[q_labels] = [data]
                else:
                    blks[q_labels].append(data)
        for q_labels in blks.keys():
            data = np.stack(blks[q_labels],axis=-1)
            BLKS.append(sub(reduced=data,q_labels=q_labels+(q_lab,)))
    tsr = sparse(blocks=blks,pattern=pattern+'+')
    print(tsr)
    exit() 
    return tsr.to_flat()
def slice_solve(norm,b,x0,site,solver,optimize='auto',**solver_opts):
    lix = norm[site,'BRA'].inds[1:][::-1]
    rix = norm[site,'KET'].inds[:-1]
    x0_map = flat2slice(norm[site,'KET'].data)
    b_map  = flat2slice(b.data)
    assert set(x0_map.keys())==set(b_map.keys())
    tns = norm.select(site,which='!any')
    const = 
    solver = getattr(spla,solver)
    x_map = dict()
    for q_lab in b_map.keys():
        x_map[q_lab] = []
        symm = x0_map[q_lab][0].dq
        x_cons = _tensors_to_constructors(tns,rix,inv=True) 
        b_cons = _tensors_to_constructors(tns,lix,inv=False)
        A = FTNLinearOperator(tns,lix,rix,symm,constructor=x_cons,optimize=optimize)
#        A = FTNLinearOperator(tns,lix,rix,symm,optimize=optimize)
#        print(A.constructor.pattern)
        for i in range(len(b_map[q_lab])):
            print(x_cons.pattern)
            print(x0_map[q_lab][i].pattern)
            x0_vec = x_cons.tensor_to_vector(x0_map[q_lab][i])
            b_vec  = b_cons.tensor_to_vector(b_map[q_lab][i])
            x,istop,itn,r1norm = solver(A,b_vec,x0=x0_vec,**solver_opts)
            x_map[q_lab].append(A.vector_to_tensor(x))
    return slice2flat(x_map)
    def gate_simple(self,G,where,**plaquette_env_options):
        layer_tags = 'KET','BRA' 
        canonize = True
        mode = 'mps'
        tags_plq = tuple(starmap(self._psi.site_tag, where))
        steps = 50
        tol = 1e-6

        inplace = True
        print(where)
        print('initial energy:',self.compute_energy())
        if not inplace:
            fs = self._psi.fermion_space
            for tid,(tsr,site) in fs.tensor_order.items():
                global_flip = tsr.phase.get('global_flip',False)
                local_inds = tsr.phase.get('local_inds',[])
                assert global_flip == False 
                assert len(local_inds) == 0
        if inplace:
            ket_init = self._psi
        else:
            ket_init = self._psi.copy()
        condition_tensors = True
        condition_maintain_norms = True
        condition_balance_bonds = True
        if condition_tensors:
            plq = ket_init.select(tags_plq,which='any')
            conditioner(plq,balance_bonds=condition_balance_bonds)
            if condition_maintain_norms:
                pre_norm = plq[tags_plq[0]].norm()
        ket_init.gate_(G,where,contract='reduce-split',max_bond=self.D)
        e1 = self.compute_energy()
        if condition_tensors:
            if condition_maintain_norms:
                conditioner(plq,value=pre_norm,
                            balance_bonds=condition_balance_bonds)
            else:
                conditioner(plq, balance_bonds=condition_balance_bonds)
        if not inplace:
            # make sure rel order of involved tsr doesn't change
            ket_init = maintain_order(ket_init,self._psi,tags_plq,refactor=False)
            for site in tags_plq:
                ket_init[site].phase = {}
            # assert: all tsrs are in the same order
            #         all tsrs are phaseless
            #         all uninvolved tsrs data are unchanged 
            for i in range(self._psi.Lx):
                for j in range(self._psi.Ly):
                    site = self._psi.site_tag(i,j)
                    tsr1 = ket_init[site]
                    tsr2 = self._psi[site]
                    order1 = tsr1.get_fermion_info()[1]
                    order2 = tsr2.get_fermion_info()[1]
                    global_flip1 = tsr1.phase.get('global_flip',False)
                    local_inds1 = tsr1.phase.get('local_inds',[])
                    global_flip2 = tsr2.phase.get('global_flip',False)
                    local_inds2 = tsr2.phase.get('local_inds',[])
                    assert order1 == order2
                    assert global_flip1 == False
                    assert global_flip2 == False
                    assert len(local_inds1) == 0 
                    assert len(local_inds2) == 0
                    if site not in tags_plq:
                        assert (tsr1.data-tsr2.data).norm()<1e-6
            e2 = self.compute_energy()
            assert abs(e2-e1)<tol
        if not inplace:
            for site in tags_plq:
                self._psi[site].modify(data=ket_init[site].data.copy())
        print('gated energy:',self.compute_energy())
