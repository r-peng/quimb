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

def pre_normalize(fu,env,where):
    if fu.pre_normalize:
        nfactor = do('abs',env.contract(all,optimize=fu.contract_optimize))
        #fu._psi.multiply_(nfactor**(-1/2),spread_over='all')
        #fu._bra.multiply_(nfactor**(-1/2),spread_over='all')

        n = fu._psi.num_tensors
        n_missing = 2
        tags_plq = tuple(starmap(fu._psi.site_tag,where))
        tns = env.select(tags_plq,which='!any')
        tns.multiply_(nfactor**(n_missing/n-1),spread_over='all')
        for site in tags_plq:
            env[site,'KET'].modify(apply=lambda data:data*nfactor**(-1/(2*n)))
            env[site,'BRA'].modify(apply=lambda data:data*nfactor**(-1/(2*n)))
        print('nfactor=',nfactor)
        print('norm=',env.contract())

        fu.env_val *= nfactor**(-1/n)
        fu.benv_val *= nfactor**(-1/n)
    return 
#    if fu.pre_normalize:
#        val = fu.env_val
#        tn = envs.get((from_which,ix),FermionTensorNetwork([]))
#        if tn.num_tensors>0:
#            tn.multiply_(val**(ix*Lbix),spread_over='all')
#        for i in [0,1]:
#            tn = envs['mid',ix+i]
#            site = ket.site_tag(ix+i,bix) if sweep[0]=='v' else \
#                   ket.site_tag(bix,ix+i)
#            tn = tn.select(site,which='!any')
#            tn.multiply_(val**(Lbix-1),spread_over='all')
#
#        tn = envs.get(('mid',ix+2),FermionTensorNetwork([]))
#        if tn.num_tensors>0:
#            tn.multiply_(val**Lbix,spread_over='all')
#        tn = envs.get((to_which,ix+2),FermionTensorNetwork([]))
#        if tn.num_tensors>0:
#            tn.multiply_(val**((Lix-(ix+3))*Lbix),spread_over='all')
#    if fu.pre_normalize:
#        val = fu.benv_val
#        tn = benvs.get((from_which,bix),FermionTensorNetwork([]))
#        if tn.num_tensors>0:
#            tn.multiply_(val**(bix*Lix),spread_over='all')
#        for i in [1]:
#            tn = benvs.get(('mid',bix+i),FermionTensorNetwork([]))
#            if tn.num_tensors>0:
#                tn.multiply_(val**Lix,spread_over='all')
#        tn = benvs.get((to_which,bix+1),FermionTensorNetwork([]))
#        if tn.num_tensors>0:
#            tn.multiply_(val**((Lbix-(bix+2))*Lix),spread_over='all')
def get_grad(args,term_map):
    site,site_map = args
    H_scal,H_arr = 0.0,0.0
    for key,(fac,_) in term_map.items():
        scal,arr = get_component(site,site_map[key],fac=fac)
        if key=='norm':
            N_scal,N_arr = scal,arr
        else:
            H_scal = H_scal + scal
            H_arr  = H_arr + arr
    energy = H_scal/N_scal
    grad = H_arr/N_scal-N_arr*H_scal/N_scal**2
#    for key,fname in site_map.items():
#        delete_ftn_from_disc(fname)
    grad_norm = grad.norm()
    return site,grad,grad_norm,energy 
def get_grads(plq_map,term_map):
    fxn = get_grad
    iterate_over = [(site,site_map) for site,site_map in plq_map.items()]
    args = [term_map]
    kwargs = dict() 
    results = parallelized_looped_function(fxn,iterate_over,args,kwargs)
    grad_map = dict()
    norm = 0.0
    for i in range(len(results)):
        if results[i] is not None:
            site,gradi,normi,energy = results[i]
            grad_map[site] = gradi,normi
            norm += normi
    return grad_map,norm,energy
