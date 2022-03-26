
##################### hessp ############################
def compute_p_norm_left_envs_wrapper(site,pfname,xbenvs,directory,**compress_opts):
    # FIX ME
    psip = load_ftn_from_disc(pfname)
    Lx,Ly = psip.Lx,psip.Ly
    ls = [xbenvs['norm','left',site[1]]]
    ls += [xbenvs['norm','mid',j] for j in range(site[1],Ly)]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=psip)
    site_tag = psip.site_tag(*site)
    data = match_phase(psip[site_tag],ftn[site_tag,'KET'])
    ftn[site_tag,'KET'].modify(data=data)
    jmin = max(site[1]-1,0)
    ftn_dict = ftn.compute_left_environments(yrange=(jmin,Ly-1),**compress_opts) 
    term_str = '_'+site_tag+'_norm_'
    fname_dict = dict()
    for (side,j),ftn in ftn_dict.items():
        fname = directory+term_str+side+str(j)
        write_ftn_to_disc(ftn,fname)
        fname_dict['norm',site_tag,side,j] = fname
    return fname_dict
def compute_p_norm_right_envs_wrapper(site,pfname,xbenvs,directory,**compress_opts):
    # FIX ME
    psip = load_ftn_from_disc(pfname)
    Lx,Ly = psip.Lx,psip.Ly
    ls = [xbenvs['norm','mid',j] for j in range(0,site[1]+1)]
    ls += [xbenvs['norm','right',site[1]]] 
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=psip)
    site_tag = psip.site_tag(*site)
    data = match_phase(psip[site_tag],ftn[site_tag,'KET'])
    ftn[site_tag,'KET'].modify(data=data)
    jmax = min(site[1]+1,Ly-1)
    ftn_dict = ftn.compute_right_environments(yrange=(0,jmax),**compress_opts) 
    term_str = '_'+site_tag+'_norm_'
    fname_dict = dict()
    for (side,j),ftn in ftn_dict.items():
        fname = directory+term_str+side+str(j)
        write_ftn_to_disc(ftn,fname)
        fname_dict['norm',site_tag,side,j] = fname
    return fname_dict 
def compute_p_term_left_envs_wrapper(info,H,pfname,pbenvs,xbenvs,directory,
                                     **compress_opts):
    # FIX ME
    site,term_key = info
    psip = load_ftn_from_disc(pfname)
    Lx,Ly = psip.Lx,psip.Ly
    cols = list(set([op_site[-1] for op_site in term_key[0]]))
    cols.sort()
    site_tag = psip.site_tag(*site)
    if site[1]<cols[0]:
        ls = [pbenvs['norm',site_tag,side,site[1]] for side in ['left','mid']]
        ls += [xbenvs['norm','mid',j] for j in range(site[1]+1,cols[0])]
        ls += [xbenvs[term_key,'mid',j] for j in range(cols[0],cols[-1]+1)]
        ls += [xbenvs['norm','mid',j] for j in range(cols[-1]+1,Ly)]
    elif site[1]>cols[-1]:
        ls = [xbenvs[term_key,'left',site[1]]]
        ls += [xbenvs['norm','mid',j] for j in range(site[1],Ly)]
    else:
        ls = [xbenvs[term_key,'left',site[1]]] 
        ls += [xbenvs[term_key,'mid',j] for j in range(site[1],cols[-1]+1)]
        ls += [xbenvs['norm','mid',j] for j in range(cols[-1]+1,Ly)]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=psip)
    if site[1]>=cols[0]:
        if site in term_key[0]:
            op = H[term_key][0][term_key[0].index(site)]
            ket = psip[site_tag]
            ket_site = ket.get_fermion_info()[1]
            pix = ket.inds[-1]
            TG = FermionTensor(op.copy(),inds=(pix,pix+'_'),left_inds=(pix,),
                               tags=ket.tags)
            ket.reindex_({pix:pix+'_'})
            psip = insert(psip,ket_site+1,TG)
            psip.contract_tags(ket.tags,which='all',inplace=True)
        data = match_phase(psip[site_tag],ftn[site_tag,'KET'])
        ftn[site_tag,'KET'].modify(data=data)
    jmin = max(site[1]-1,0)
    ftn_dict = ftn.compute_left_environments(yrange=(jmin,Ly-1),**compress_opts) 
    term_str = '_'+site_tag+'_'+str(term_key)+'_'
    fname_dict = dict()
    for (side,j),ftn in ftn_dict.items():
        fname = directory+term_str+side+str(j)
        write_ftn_to_disc(ftn,fname)
        fname_dict[term_key,site_tag,side,j] = fname
    return fname_dict
def compute_p_term_right_envs_wrapper(info,H,pfname,pbenvs,xbenvs,directory,
                                     **compress_opts):
    # FIX ME
    site,term_key = info
    psip = load_ftn_from_disc(pfname)
    Lx,Ly = psip.Lx,psip.Ly
    cols = list(set([op_site[-1] for op_site in term_key[0]]))
    cols.sort()
    site_tag = psip.site_tag(*site)
    if site[1]>cols[-1]:
        ls = [xbenvs['norm','mid',j] for j in range(cols[0])]
        ls += [xbenvs[term_key,'mid',j] for j in range(cols[0],cols[-1]+1)]
        ls += [xbenvs['norm','mid',j] for j in range(cols[-1]+1,site[1])]
        ls += [pbenvs['norm',site_tag,side,site[1]] for side in ['mid','right']]
    elif site[1]<cols[0]:
        ls = [xbenvs['norm','mid',j] for j in range(site[1]+1)]
        ls += [xbenvs[term_key,'right',site[1]]]
    else:
        ls = [xbenvs['norm','mid',j] for j in range(cols[0])]
        ls += [xbenvs[term_key,'mid',j] for j in range(cols[0],site[1]+1)]
        ls += [xbenvs[term_key,'right',site[1]]] 
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in ls]
          ).view_as_(FermionTensorNetwork2D,like=psip)
    if site[1]<=cols[-1]:
        if site in term_key[0]:
            op = H[term_key][0][term_key[0].index(site)]
            ket = psip[site_tag]
            ket_site = ket.get_fermion_info()[1]
            pix = ket.inds[-1]
            TG = FermionTensor(op.copy(),inds=(pix,pix+'_'),left_inds=(pix,),
                               tags=ket.tags)
            ket.reindex_({pix:pix+'_'})
            psip = insert(psip,ket_site+1,TG)
            psip.contract_tags(ket.tags,which='all',inplace=True)
        data = match_phase(psip[site_tag],ftn[site_tag,'KET'])
        ftn[site_tag,'KET'].modify(data=data)
    jmax = min(site[1]+1,Ly-1)
    ftn_dict = ftn.compute_right_environments(yrange=(0,jmax),**compress_opts) 
    term_str = '_'+site_tag+'_'+str(term_key)+'_'
    fname_dict = dict()
    for (side,j),ftn in ftn_dict.items():
        fname = directory+term_str+side+str(j)
        write_ftn_to_disc(ftn,fname)
        fname_dict[term_key,site_tag,side,j] = fname
    return fname_dict
def compute_p_plq_envs_wrapper(info,pbenvs,xbenvs,directory,**compress_opts):
    # FIX ME
    term_key,site,j = info
    like = load_ftn_from_disc(xbenvs['norm','mid',0])
    Lx,Ly = like.Lx,like.Ly
    psite_tag = like.site_tag(*site)
    if term_key=='norm':
        cols = [j]
    else:
        cols = list(set([op_site[1] for op_site in term_key[0]]))
        cols.sort()
    l = term_key if j>cols[0] else 'norm'
    l = pbenvs[l,psite_tag,'left',j] if site[1]<j else xbenvs[l,'left',j]
    m = term_key if j in cols else 'norm'
    m = pbenvs[m,psite_tag,'mid',j] if site[1]==j else xbenvs[m,'mid',j]
    r = term_key if j<cols[-1] else 'norm'
    r = pbenvs[r,psite_tag,'right',j] if site[1]>j else xbenvs[r,'right',j]
    ftn = FermionTensorNetwork([load_ftn_from_disc(fname) for fname in [l,m,r]]
          ).view_as_(FermionTensorNetwork2D,like=like)
    row_envs = ftn.compute_row_environments(yrange=(max(j-1,0),min(j+1,Ly-1)),
                                            **compress_opts)
    term_str = '_'+psite_tag+'_'+str(term_key)+'_'
    plq_envs = dict()
    for i in range(Lx):
        ftn = FermionTensorNetwork(
              [row_envs[side,i] for side in ['bottom','mid','top']],
              check_collisions=False)
        xsite_tag = like.site_tag(i,j)
        fname = directory+term_str+xsite_tag
        write_ftn_to_disc(ftn,fname)
        plq_envs[term_key,xsite_tag,psite_tag] = fname
    return plq_envs
def compute_p_site_components(site_tag,data_map):
    # FIX ME
    H0,H1,N0,N1 = 0.0,0.0,0.0,0.0 
    for (term_key,xsite_tag,psite_tag),(scal,data) in data_map.items():
        if xsite_tag == site_tag:
            if term_key=='norm':
                N0,N1 = N0+scal,N1+data
            else:
                H0,H1 = H0+scal,H1+data
    return site_tag,H1,N1,H0,N0
def compute_site_hessp(site_tag,H0x,N0x,H1x,N1x,H0p,N0p,H1p,N1p):
    # FIX ME
    E = H0x/N0x
    hp = (H1p[site_tag]-E*N1p[site_tag])/N0x
    hp = hp - (H1x[site_tag]*N0p+N1x[site_tag]*H0p)/N0x**2
    hp = hp + 2.0*E*N0p*N1x[site_tag]/N0x**2
    return site_tag,hp
def compute_hessp(H,xfname,pfname,directory,layer_tags=('KET','BRA'),
    # FIX ME
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    H0x,N0x,H1x,N1x,xbenvs = compute_components(H,xfname,directory,**compress_opts)
    psip = load_ftn_from_disc(pfname)
    Lx,Ly = psip.Lx,psip.Ly
    # get psite norm col envs
    pbenvs = dict()
    iterate_over = [(i,j) for i in range(Lx) for j in range(Ly)]
    args = [pfname,xbenvs,directory]
    kwargs = compress_opts

    fxn = compute_p_norm_left_envs_wrapper
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        pbenvs.update(fname_dict)
    fxn = compute_p_norm_right_envs_wrapper
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        pbenvs.update(fname_dict)
    # get psite term col envs
    iterate_over = [(site,term_key) for site in iterate_over \
                                    for term_key in H.keys()] 
    args = [H,pfname,pbenvs,xbenvs,directory]
    kwargs = compress_opts

    fxn = compute_p_term_left_envs_wrapper
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        pbenvs.update(fname_dict)
    fxn = compute_p_term_right_envs_wrapper
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for fname_dict in ls:
        pbenvs.update(fname_dict)
    # compute plq envs
    fxn = compute_p_plq_envs_wrapper
    iterate_over = [(term_key,site,j) for term_key in list(H.keys())+['norm']
                    for site in [(i,j) for i in range(Lx) for j in range(Ly)] \
                    for j in range(Ly)]
    args = [pbenvs,xbenvs,directory]
    kwargs = compress_opts
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    plq_envs = dict()
    for plq_envs_term in ls:
        plq_envs.update(plq_envs_term)
    # compute site components
    fxn = contract_site
    iterate_over = list(plq_envs.keys())
    args = [plq_envs,H]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    data_map = dict()
    for (key,scal,data) in ls:
        data_map[key] = scal,data

    fxn = compute_p_site_components
    iterate_over = [psip.site_tag(i,j) for i in range(Lx) for j in range(Ly)]
    args = [data_map]
    kwargs = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    site00 = psip.site_tag(0,0)
    H1p,N1p = dict(),dict()
    for (site_tag,H1,N1,H0,N0) in ls:
        if site_tag==site00:
            H0p,N0p = H0,N0
        H1p[site_tag] = H1
        N1p[site_tag] = N1
    
    fxn = compute_site_hessp
    iterate_over = list(H1x.keys())
    args = [H0x,N0x,H1x,N1x,H0p,N0p,H1p,N1p]
    kwargs = dict()
    hessp = dict()
    ls = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    for (site_tag,hp) in ls:
        hessp[site_tag] = hp
    # delete files 
    for _,fname in xbenvs.items():
        delete_ftn_from_disc(fname)
    for _,fname in pbenvs.items():
        delete_ftn_from_disc(fname)
    return hessp 
################### some naive numerical methods #####################
def conjugate_grad(grad,d_old,gnormsq_old):
    gnormsq = sum([np.dot(g.data,g.data) for g in grad.values()])
    if d_old is None:
        d = {key:-g for key,g in grad.items()}
    else:
        beta = gnormsq/gnormsq_old
        d = {key:-g+beta*d_old[key] for key,g in grad.items()}
    return d,gnormsq
def line_search(d,EL,t0,H,psi_name,directory,
    layer_tags=('KET','BRA'),max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    def energy(t):
        psi = load_ftn_from_disc(psi_name)
        for site_tag,data in d.items():
            psi[site_tag].modify(data=psi[site_tag].data+t*data)
        write_ftn_to_disc(psi,psi_name+'_tmp')
        return compute_energy(H,psi_name+'_tmp',directory,**compress_opts)
    tR,ER = 2.0*t0,0.0
    while ER>EL:
        tR = 0.5*tR
        ER = energy(tR)
    tM = 0.5*tR
    EM = energy(tM)
    delete_ftn_from_disc(psi_name+'_tmp')
    if ER<EM:
        return tR,ER
    else:
        return tM,EM
def kernel(H,psi_name,directory,chi,maxiter=50,gtol=1e-5,t0=0.5):
    print('using fpeps update')
    E_old = 0.0
    d_old,gnormsq_old = None,None
    for i in range(maxiter):
        grad,E,N = compute_grad(H,psi_name,directory,max_bond=chi)
        d,gnormsq = conjugate_grad(grad,d_old,gnormsq_old)
#        t,E = line_search(d,E_old,t0,H,psi_name,directory,max_bond=chi) 
        t = t0
        g = [gi.data for _,gi in grad.items()]
        g = np.concatenate(g)
#        E_ = compute_energy(H,psi_name,directory,max_bond=chi)
#        print(abs((E-E_)/E))
        E_ = compute_energy(H,psi_name,directory,max_bond=chi//2)
        print('iter={},E={},N={},Eerr={},gmax={},t={}'.format(
               i,E,N,abs((E-E_)/E),np.amax(abs(g)),t))
        if E-E_old>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
#        d_old,gnormsq_old,E_old = d.copy(),gnormsq,E
        E_old = E
        psi = load_ftn_from_disc(psi_name)
        for site_tag,data in d.items():
            psi[site_tag].modify(data=psi[site_tag].data+t*data)
        write_ftn_to_disc(psi,psi_name)
    print('end iteration')
    return E
def kernel_vec(H,psi_name,directory,chi,maxiter=50,gtol=1e-5,t0=0.5):
    print('using vector update')
    E_old = 0.0

    peps = load_ftn_from_disc(psi_name) 
    constructors = dict()
    for i in range(peps.Lx):
        for j in range(peps.Ly):
            site_tag = peps.site_tag(i,j)
            data = peps[site_tag].data
            bond_infos = [data.get_bond_info(ax,flip=False) \
                          for ax in range(data.ndim)]
            cons = Constructor.from_bond_infos(bond_infos,data.pattern)
            constructors[site_tag] = cons,data.dq
    write_ftn_to_disc(peps,directory+'skeleton')

    psi = peps
    ls = []
    for i in range(psi.Lx):
        for j in range(psi.Ly):
            site_tag = psi.site_tag(i,j)
            cons = constructors[site_tag][0]
            ls.append(cons.tensor_to_vector(psi[site_tag].data))
    x = np.concatenate(ls)
    for it in range(maxiter):
        psi = load_ftn_from_disc(directory+'skeleton')
        start = 0
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j) 
                ket = psi[site_tag]
                stop = start+len(ket.data.data)
                cons,dq = constructors[site_tag]
                ket.modify(data=cons.vector_to_tensor(x[start:stop],dq))
                start = stop
        write_ftn_to_disc(psi,directory+'tmp')
        grad,E,N = compute_grad(H,directory+'tmp',directory,max_bond=chi)
        t = t0
        g = []
        for i in range(psi.Lx):
            for j in range(psi.Ly):
                site_tag = psi.site_tag(i,j)
                cons = constructors[site_tag][0]
                g.append(cons.tensor_to_vector(grad[site_tag]))
        g = np.concatenate(g)
#        E_ = compute_energy(H,directory+'tmp',directory,max_bond=chi)
#        print(abs((E-E_)/E))
        E_ = compute_energy(H,directory+'tmp',directory,max_bond=chi//2)
        print('iter={},E={},N={},Eerr={},gmax={},t={}'.format(
              it,E,N,abs((E-E_)/E),np.amax(abs(g)),t))
        if E-E_old>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
        x = x - t*g  
    print('end iteration')
    return
def compute_hessp(self,x,p):
    psi_x = self.vec2fpeps(x)
    write_ftn_to_disc(psi_x,self.tmp)
    psi_p = self.vec2fpeps(p)
    write_ftn_to_disc(psi_p,self.directory+'psi_p')
    hessp = compute_hessp(self.H,self.tmp,self.directory+'psi_p',self.directory,
                  max_bond=self.chi,cutoff=1e-15)
    hp = []
    for i in range(psi_x.Lx):
        for j in range(psi_x.Ly):
            site_tag = psi_x.site_tag(i,j)
            cons = self.constructors[site_tag][0]
            hp.append(cons.tensor_to_vector(hessp[site_tag]))
    hp = np.concatenate(hp)
    return hp
def line_search(f,xk,pk,gk,old_fval,old_old_fval,c1=1e-4):
    phi0 = old_fval
    dphi0 = np.dot(pk,gk)
    print('dphi0=',dphi0)
    def phi(a):
        return f(xk+a*pk)
    a0 = 1.0 
    phi_a0 = phi(a0)
    print('a0={},phi(ai)={}'.format(a0,phi_a0))
    rhs = phi0+c1*a0*dphi0
    if phi_a0<rhs:
        return a0,phi_a0,old_fval,None
    a1 = -dphi0*a0**2/(2.0*(phi_a0-phi0-dphi0*a0))
    phi_a1 = phi(a1)
    print('a1={},phi(ai)={}'.format(a1,phi_a1))
    i = 2
    while phi_a1>=rhs:
        M = np.array([[1.0/a1**2,-1.0/a0**2],[-a0/a1**2,a1/a0**2]])/(a1-a0)
        v = np.array([phi_a1-phi0-dphi0*a1,phi_a0-phi0-dphi0*a0])
        a,b = np.einsum('ij,j->i',M,v)
        a0,a1 = a1,(-b+np.sqrt(b**2-3.0*a*dphi0))/(3.0*a)
        assert a0>0.0 and a1>0.0
        phi_a0,phi_a1 = phi_a1,phi(a1)
        print('a{}={},phi(ai)={}'.format(i,a1,phi_a1))
        i += 1
    return a1,phi_a1,old_fval,None
def line_search_golden(f,xk,pk,gk,old_fval,old_old_fval,eps=1e-3): 
    def _f(a):
        return f(xk+a*pk)
    a1,a4 = 0.0,1.0
    f1,f4 = old_fval,_f(a4)
    phi = (1.0+np.sqrt(5.0))/2.0
    r = phi-1.0
    c = 1.0-r
    a2,a3 = a1+(a4-a1)*c,a1+(a4-a1)*r
    f2,f3 = _f(a2),_f(a3)
    ls = [f1,f2,f3,f4]
    ls.sort()
    i = 0
    while ls[1]-ls[0]>eps:
        print('it=',i)
        print(f1,f2,f3,f4)
        print(a1,a2,a3,a4)
        min12,min34 = min(f1,f2),min(f3,f4)
        if min12<min34:
            a1,a3,a4 = a1,a2,a3
            f1,f3,f4 = f1,f2,f3
            a2 = a1+(a4-a1)*c
            f2 = _f(a2)
        else:
            a1,a2,a4 = a2,a3,a4
            f1,f2,f4 = f2,f3,f4
            a3 = a1+(a4-a1)*r
            f3 = _f(a3)
        ls = [f1,f2,f3,f4]
        ls.sort()
        i += 1
    idx = np.argmin([f1,f2,f3,f4])
    fmin = list([f1,f2,f3,f4])[idx]
    amin = list([a1,a2,a3,a4])[idx]
    return amin,fmin,old_fval,None
def _minimize_newtoncg(fun, x0, args=(), jac=None, hess=None, hessp=None,
                       callback=None, xtol=1e-5, eps=np.sqrt(np.finfo(float).eps), 
                       maxiter=None, disp=False, return_all=False,
                       bounds=None,constraints=(),
                       **unknown_options):
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.
    Note that the `jac` parameter (Jacobian) is required.
    Options
    -------
    disp : bool
        Set to True to print convergence messages.
    xtol : float
        Average relative error in solution `xopt` acceptable for
        convergence.
    maxiter : int
        Maximum number of iterations to perform.
    eps : float or ndarray
        If `hessp` is approximated, use this value for the step size.
    return_all : bool, optional
        Set to True to return a list of the best solution at each of the
        iterations.
    """
    optimize._check_unknown_options(unknown_options)
    if jac is None:
        raise ValueError('Jacobian is required for Newton-CG method')
    fhess_p = hessp
    fhess = hess
    avextol = xtol
    epsilon = eps
    retall = return_all

    x0 = np.asarray(x0).flatten()
    # TODO: add hessp (callable or FD) to ScalarFunction?
    sf = optimize._prepare_scalar_function(
        fun, x0, jac, args=args, epsilon=eps, hess=hess
    )
    f = sf.fun
    fprime = sf.grad
    _h = sf.hess(x0)

    # Logic for hess/hessp
    # - If a callable(hess) is provided, then use that
    # - If hess is a FD_METHOD, or the output fom hess(x) is a LinearOperator
    #   then create a hessp function using those.
    # - If hess is None but you have callable(hessp) then use the hessp.
    # - If hess and hessp are None then approximate hessp using the grad/jac.

    if (hess in optimize.FD_METHODS or isinstance(_h, scipy.sparse.linalg.LinearOperator)):
        fhess = None

        def _hessp(x, p, *args):
            return sf.hess(x).dot(p)

        fhess_p = _hessp

    def terminate(warnflag, msg):
        if disp:
            print(msg)
            print("         Current function value: %f" % old_fval)
            print("         Iterations: %d" % k)
            print("         Function evaluations: %d" % sf.nfev)
            print("         Gradient evaluations: %d" % sf.ngev)
            print("         Hessian evaluations: %d" % hcalls)
        fval = old_fval
        result = optimize.OptimizeResult(fun=fval, jac=gfk, nfev=sf.nfev,
                                njev=sf.ngev, nhev=hcalls, status=warnflag,
                                success=(warnflag == 0), message=msg, x=xk,
                                nit=k)
        if retall:
            result['allvecs'] = allvecs
        return result

    hcalls = 0
    if maxiter is None:
        maxiter = len(x0)*200
    cg_maxiter = 20*len(x0)

    xtol = len(x0) * avextol
    update = [2 * xtol]
    xk = x0
    if retall:
        allvecs = [xk]
    k = 0
    gfk = None
    old_fval = f(x0)
    old_old_fval = None
    float64eps = np.finfo(np.float64).eps
#    while np.add.reduce(np.abs(update)) > xtol:
    b = np.ones(len(x0))
    while np.amax(np.abs(b)) > avextol:
        if k >= maxiter:
            msg = "Warning: " + _status_message['maxiter']
            return terminate(1, msg)
        # Compute a search direction pk by applying the CG method to
        #  del2 f(xk) p = - grad f(xk) starting from 0.
        b = -fprime(xk)
        maggrad = np.add.reduce(np.abs(b))
        eta = np.min([0.5, np.sqrt(maggrad)])
        termcond = eta * maggrad
        xsupi = np.zeros(len(x0), dtype=x0.dtype)
        ri = -b
        psupi = -ri
        i = 0
        dri0 = np.dot(ri, ri)

        if fhess is not None:             # you want to compute hessian once.
            A = sf.hess(xk)
            hcalls = hcalls + 1

        for k2 in range(cg_maxiter):
            if np.add.reduce(np.abs(ri)) <= termcond:
                break
            if np.amax(abs(ri)) < avextol:
                break
            if fhess is None:
                if fhess_p is None:
                    Ap = optimize.approx_fhess_p(xk, psupi, fprime, epsilon)
                else:
                    Ap = fhess_p(xk, psupi, *args)
                    hcalls = hcalls + 1
            else:
                if isinstance(A, HessianUpdateStrategy):
                    # if hess was supplied as a HessianUpdateStrategy
                    Ap = A.dot(psupi)
                else:
                    Ap = np.dot(A, psupi)
            # check curvature
            Ap = np.asarray(Ap).squeeze()  # get rid of matrices...
            curv = np.dot(psupi, Ap)
            print('curv={},npi={},rmax={}'.format(curv/np.dot(psupi,psupi),np.linalg.norm(psupi),np.amax(abs(ri))))
            if 0 <= curv <= 3 * float64eps:
                break
            elif curv < 0:
                if (i > 0):
                    break
                else:
                    # fall back to steepest descent direction
                    xsupi = dri0 / (-curv) * b
                    break
            alphai = dri0 / curv
            xsupi = xsupi + alphai * psupi
            ri = ri + alphai * Ap
            dri1 = np.dot(ri, ri)
            betai = dri1 / dri0
            psupi = -ri + betai * psupi
            i = i + 1
            dri0 = dri1          # update np.dot(ri,ri) for next time.
        else:
            # curvature keeps increasing, bail out
            msg = ("Warning: CG iterations didn't converge. The Hessian is not "
                   "positive definite.")
            return terminate(3, msg)

        pk = xsupi  # search direction is solution to system.
        gfk = -b    # gradient at xk

        try:
            alphak, fc, gc, old_fval, old_old_fval, gfkp1 = \
                     optimize._line_search_wolfe12(f, fprime, xk, pk, gfk,
                                          old_fval, old_old_fval)
        except optimize._LineSearchError:
            # Line search failed to find a better solution.
            msg = "Warning: " + _status_message['pr_loss']
            return terminate(2, msg)

        update = alphak * pk
        xk = xk + update        # upcast if necessary
        if callback is not None:
            callback(xk)
        if retall:
            allvecs.append(xk)
        k += 1
    else:
        if np.isnan(old_fval) or np.isnan(update).any():
            return terminate(3, _status_message['nan'])

        msg = optimize._status_message['success']
        return terminate(0, msg)
def custom(fun,x0,jac=None,callback=None,gtol=1e-5,maxiter=50,t0=0.5,**options):
    print('using scipy.optimize custom')
    fun,grad = fun,jac
    t = t0
    print('t=',t)
    x = x0
    fold = 0.0
    nit = 0
    while nit < maxiter:
        f,g = fun(x),grad(x) 
        if f-fold>0.0:
            break
        if np.amax(abs(g))<gtol:
            break
        x = x - t*g
        if callback is not None:
            callback(x)
        fold = f
        nit += 1
    return scipy.optimize.OptimizeResult(fun=f,jac=g,x=x,nit=nit,message='')
