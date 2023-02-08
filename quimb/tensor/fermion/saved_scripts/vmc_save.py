
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
