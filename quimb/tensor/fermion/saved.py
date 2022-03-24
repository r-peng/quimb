
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
def compute_norm(psi_name,directory,layer_tags=('KET','BRA'),
    # REDUNDANT
    max_bond=None,cutoff=1e-10,canonize=True,mode='mps'):
    compress_opts = dict()
    compress_opts['max_bond'] = max_bond
    compress_opts['cutoff'] = cutoff
    compress_opts['canonize'] = canonize
    compress_opts['mode'] = mode
    compress_opts['layer_tags'] = layer_tags
    psi = load_ftn_from_disc(psi_fname)
    Lx,Ly = psi.Lx,psi.Ly
    norm = psi.make_norm(layer_tags=('KET','BRA'))
    norm.reorder(direction='col',layer_tags=('KET','BRA'),inplace=True)
    ftn_dict = norm.compute_right_environments(**compress_opts)

    ftn = FermionTensorNetwork(
          (norm.select(norm.col_tag(0)).copy(),ftn_dict['right',0])
          ).view_as_(FermionTensorNetwork2D,like=norm)
    ftn.reorder(direction='row',layer_tags=('KET','BRA'),inplace=True)
    top_envs = ftn.compute_top_environments(yrange=(0,1),**compress_opts)
    ftn = FermionTensorNetwork(
          (ftn.select(ftn.row_tag(0)).copy(),top_envs['top',0]))
    return ftn.contract() 
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
