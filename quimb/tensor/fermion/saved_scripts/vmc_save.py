
def compute_double_layer_plq(norm,**compress_opts):
    norm.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    Lx,Ly = norm.Lx,norm.Ly

    ftn = norm.copy()
    last_row = ftn.row_tag(Lx-1)
    top = [None] * Lx
    top[-1] = ftn.select(last_row).copy() 
    for i in range(Lx-2,0,-1):
        try:
            ftn.contract_boundary_from_top_(xrange=(i,i+1),yrange=(0,Ly-1),**compress_opts)
            top[i] = ftn.select(last_row).copy()
        except (ValueError,IndexError):
            break

    ftn = norm.copy()
    first_row = ftn.row_tag(0)
    bot = [None] * Lx
    bot[0] = ftn.select(first_row).copy()
    for i in range(1,Lx-1):
        try:
            ftn.contract_boundary_from_bottom_(xrange=(i-1,i),yrange=(0,Ly-1),**compress_opts)
            bot[i] = ftn.select(first_row).copy()
        except (ValueError,IndexError):
            break

    plq = dict()  
    for i in range(Lx):
        ls = []
        if i>0:
            ls.append(bot[i-1])
        ls.append(norm.select(norm.row_tag(i)).copy())
        if i<Lx-1:
            ls.append(top[i+1])
        try:
            ftn = FermionTensorNetwork(ls,virtual=False).view_like_(norm)
        except (AttributeError,TypeError): # top/bot env is None
            break
        plq = update_plq_from_3col(plq,ftn,i,1,1,norm)
    return plq
def get_key_from_qlab(qlab):
    # isinstance(qlab,pyblock3.algebra.symmerty.SZ)
    n = qlab.n 
    if n==0:
        return 0
    if n==2:
        return 3
    sz = qlab.twos  
    if sz==1:
        return 1
    elif sz==-1:
        return 2
    else:
        raise ValueError(f'n={n},sz={sz}')
def build_mpo(self,n,cutoff=1e-9):
    from pyblock3.hamiltonian import Hamiltonian
    from pyblock3.fcidump import FCIDUMP
    fcidump = FCIDUMP(pg='c1',n_sites=n,n_elec=0,twos=0,ipg=0,orb_sym=[0]*n)
    hamil = Hamiltonian(fcidump,flat=False)
    def generate_terms(n_sites,c,d):
        for i in range(0,n_sites):
            for s in (0,1):
                if i-1>=0:
                    yield -self.t*c[i,s]*d[i-1,s]
                if i+1<n_sites:
                    yield -self.t*c[i,s]*d[i+1,s]
    mpo = hamil.build_mpo(generate_terms,cutoff=cutoff).to_sparse()
    mpo,err = mpo.compress(cutoff=cutoff)
    
    from pyblock3.algebra.core import SubTensor
    from pyblock3.algebra.fermion import SparseFermionTensor
    for i,tsr in enumerate(mpo.tensors):
        print(i)
        print(tsr)
        #odd_blks = tsr.odd.to_sparse().blocks
        #even_blks = tsr.even.to_sparse().blocks
        #blk_dict = dict()
        #for blk in odd_blks:
        #    blk_dict = update_blk_dict(blk_dict,blk)
        #for blk in even_blks:
        #    blk_dict = update_blk_dict(blk_dict,blk)
        #blks = [SubTensor(reduced=arr,q_labels=qlabs) for qlabs,arr in blk_dict.items()]
        #tsr_new = SparseFermionTensor(blocks=blks,pattern='++--')
        #print(tsr_new)
def update_blk_dict(blk_dict,blk):
    # isinstance(blk,pyblock3.algebra.core.SubTensor)
    arr = np.asarray(blk)
    assert arr.size==1
    nax = len(blk.q_labels)
    qlabs = [None] * nax
    ixs = [None] * nax 
    shs = [None] * nax
    for ax,qlab in enumerate(blk.q_labels):
        key = get_key_from_qlab(qlab)
        qlabs[ax],ixs[ax],shs[ax] = state_map[key]    
    qlabs = tuple(qlabs)
    if qlabs not in blk_dict:
        blk_dict[qlabs] = np.zeros(shs,dtype=arr.dtype)
    ixs = tuple(ixs)
    blk_dict[qlabs][ixs] += arr[(0,)*nax]
    return blk_dict
def _correlated_local_sampling(info,psi,ham,contract_opts,_compute_g):
    samples,f = info

    amp_fac = get_amplitude_factory(psi,contract_opts)
    v = None
    if _compute_g:
        amp_fac = update_grads(amp_fac,samples)
        c = amp_fac.store
        g = amp_fac.store_grad
        v = [g[x]/c[x] for x in samples]

    e = compute_elocs(ham,amp_fac,samples,f)
    return samples,f,v,e,None 
def cumulate_samples(ls):
    _,f,v,e,hg,config,cx = ls[0]
    for _,fi,vi,ei,hgi in ls[1:]:
        f += fi
        v += vi
        e += ei
        if hg is not None:
            hg += hgi
    if hg is not None: 
        hg = np.array(hg)
    return np.array(f),np.array(v),np.array(e),hg
def _extract_energy_gradient(ls):
    f,v,e,hg = cumulate_samples(ls)

    # mean energy
    _xsum = np.dot(f,e) 
    _xsqsum = np.dot(f,np.square(e))
    n = np.sum(f)
    E,err = _mean_err(_xsum,_xsqsum,n)

    # gradient
    v_mean = np.dot(f,v)/n
    g = np.dot(e*f,v)/n - E*v_mean
    return E,n,err,g,f,v,v_mean,hg
def extract_energy_gradient(ls,optimizer,psi,constructors,tmpdir,contract_opts,ham):
    E,n,err,g,f,v,v_mean,hg = _extract_energy_gradient(ls)
    if optimizer.method not in ['sr','rgn','lin']:
        optimizer._set(E,n,g,None,None)
        return optimizer,err

    # ovlp
    if optimizer.ovlp_matrix:
        S = np.einsum('n,ni,nj,n->ij',f,v,v)/n - np.outer(v_mean,v_mean)
        def _S(x):
            return np.dot(S,x)
    else:
        def _S(x):
    if optimizer.method not in ['rgn','lin']:
        optimizer._set(E,n,g,_S,None)
        return optimizer,err
    
    # hess
    if optimizer.hg:
        hg_mean = np.dot(f,hg)/n
        if optimizer.hess_matrix:
            H = np.einsum('n,ni,nj->ij',f,v,hg)/n - np.outer(v_mean,hg_mean) 
            H -= np.outer(g,v_mean)
            def _H(x):
                return np.dot(H,x)
        else:
            def _H(x):
                Hx1 = np.dot(v.T,f*np.dot(hg,x))/n
                Hx2 = v_mean * np.dot(hg_mean,x)
                Hx3 = g * np.dot(v_mean,x)
                return Hx1-Hx2-Hx3
    else:
        infos = [(samples,fi[:len(samples)]) for samples,fi,_,_,_ in ls]
        eps = optimizer.num_step
        print([(len(si),len(fi)) for si,fi in infos])
        def _H(x):
            psi_ = _update_psi(psi.copy(),-x*eps,constructors)
            psi_ = write_ftn_to_disc(psi_,tmpdir+'tmp',provided_filename=True) 
            args = psi_,ham,contract_opts,True
            ls_ = parallelized_looped_fxn(_correlated_local_sampling,infos,args)
            g_ = _extract_energy_gradient(ls)[3]
            return (g_-g)/eps 
    optimizer._set(E,n,g,_S,_H)
    return optimizer,err
