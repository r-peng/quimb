
class DMRG(TNVMC):
    def __init__(
	self,
	ham,
	sampler,
	amplitude_factory,
	optimizer='lin',
        ratio=20,
        **kwargs
    ):
	# parse ham
        self.ham = ham

	# parse sampler
        self.config = None
        self.sampler = sampler
        self.exact_sampling = sampler.exact
        self.ratio = ratio
        
        # parse wfn 
        self.amplitude_factory = amplitude_factory         
        self.x = self.amplitude_factory.get_x()
        self.block_dict = self.amplitude_factory.get_block_dict()
        
        # parse gradient optimizer
        self.optimizer = optimizer
        self.ham.initialize_pepo(self.amplitude_factory.psi)
        if self.optimizer=='lin':
            self.xi = kwargs.get('xi',0.5)

    def set_plqs(self,plqs):
        self.plqs = plqs 
        self.nplq = len(plqs)
        self.plq_ix = 0
    def set_batchsize(self):
        plq = self.plqs[self.plq_ix]
        if RANK==0:
            print('\tplq=',plq)
        self.get_plq_info(plq)
        self.batchsize = self.nparam * self.ratio
        self.plq_ix = (self.plq_ix + 1) % self.nplq
    def get_plq_info(self,plq):
        self.plq_info = []
        start_ = 0
        for ix in plq:
            start,stop = self.block_dict[ix]
            sh = stop - start
            stop_ = start_ + sh
            self.plq_info.append((ix,start_,stop_,start,stop)) 
            start_ = stop_
        self.nparam = stop_
        if RANK==0:
            print('\tnparam=',self.nparam)
    def extract_energy_gradient(self):
        t0 = time.time()
        self.extract_energy()
        self.extract_gradient()

        self.S = self.extract_matrix('S')
        self._extract_Hvmean()
        self.H = self.extract_matrix('H')
        if RANK==0:
            print('\tnormalization=',self.n)
            print('\tgradient norm=',np.linalg.norm(self.g))
            print(f'step={self.step},energy={self.E},err={self.Eerr}')
            print('\tcollect data time=',time.time()-t0)
    def _get_Smatrix_stochastic(self,start1,stop1,start2,stop2):
        if RANK==0:
            sh1,sh2 = stop1-start1,stop2-start2
            vvsum_ = np.zeros((sh1,sh2))
        else:
            v1,v2 = self.vlocal[:,start1:stop1],self.vlocal[:,start2:stop2]
            vvsum_ = np.dot(v1.T,v2)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean1,vmean2 = self.vmean[start1:stop1],self.vmean[start2:stop2]
            S = vvsum / self.n - np.outer(vmean1,vmean2)
        return S
    def _get_Smatrix_exact(self,start1,stop1,start2,stop2):
        v1,v2 = self.vlocal[:,start1:stop1],self.vlocal[:,start2:stop2]
        vvsum_ = np.einsum('s,si,sj->ij',self.flocal,v1,v2)
        vvsum = np.zeros_like(vvsum_)
        COMM.Reduce(vvsum_,vvsum,op=MPI.SUM,root=0)
        S = None
        if RANK==0:
            vmean1,vmean2 = self.vmean[start1:stop1],self.vmean[start2:stop2]
            S = vvsum - np.outer(vmean1,vmean2)
        return S
    def _get_Smatrix(self,start1,stop1,start2,stop2):
        if self.exact_sampling:
            return self._get_Smatrix_exact(start1,stop1,start2,stop2)
        else:
            return self._get_Smatrix_stochastic(start1,stop1,start2,stop2)
    def _get_Hmatrix_stochastic(self,start1,stop1,start2,stop2):
        if RANK==0:
            sh1,sh2 = stop1-start1,stop2-start2
            vHvsum_ = np.zeros((sh1,sh2))
        else:
            v1,Hv2 = self.vlocal[:,start1:stop1],self.Hv_local[:,start2:stop2]
            vHvsum_ = np.dot(v1.T,Hv2)
        vHvsum = np.zeros_like(vHvsum_)
        COMM.Reduce(vHvsum_,vHvsum,op=MPI.SUM,root=0)
        H = None
        if RANK==0:
            vmean1,Hvmean2 = self.vmean[start1:stop1],self.Hvmean[start2:stop2]
            g1,vmean2 = self.g[start1:stop1],self.vmean[start2:stop2]
            H = vHvsum / self.n - np.outer(vmean1,Hvmean2) - np.outer(g1,vmean2)
        return H
    def _get_Hmatrix_exact(self,start1,stop1,start2,stop2):
        v1,Hv2 = self.vlocal[:,start1:stop1],self.Hv_local[:,start2:stop2]
        vHvsum_ = np.einsum('s,si,sj->ij',self.flocal,v1,Hv2)
        vHvsum = np.zeros_like(vHvsum_)
        COMM.Reduce(vHvsum_,vHvsum,op=MPI.SUM,root=0)
        H = None
        if RANK==0:
            vmean1,Hvmean2 = self.vmean[start1:stop1],self.Hvmean[start2:stop2]
            g1,vmean2 = self.g[start1:stop1],self.vmean[start2:stop2]
            H = vHvsum - np.outer(vmean1,Hvmean2) - np.outer(g1,vmean2)
        return H
    def _get_Hmatrix(self,start1,stop1,start2,stop2):
        if self.exact_sampling:
            return self._get_Hmatrix_exact(start1,stop1,start2,stop2)
        else:
            return self._get_Hmatrix_stochastic(start1,stop1,start2,stop2)
    def extract_matrix(self,matrix):
        if matrix=='S':
            _get_matrix = self._get_Smatrix
        elif matrix=='H':
            _get_matrix = self._get_Hmatrix
        matrix = np.zeros((self.nparam,self.nparam))
        for ix1 in range(len(self.plq_info)):
            _,start1_,stop1_,start1,stop1 = self.plq_info[ix1]
            for ix2 in range(ix1,len(self.plq_info)):
                _,start2_,stop2_,start2,stop2 = self.plq_info[ix1]
                sub = _get_matrix(start1,stop1,start2,stop2)
                if RANK==0:
                    matrix[start1_:stop1_,start2_:stop2_] = sub    
                    if ix2 > ix2:
                        matrix[start2_:stop2_,start1_:stop1_] = sub.T
        return matrix
    def extract_vector(self,vec_full):
        vec = np.zeros(self.nparam)
        for ix in range(len(self.plq_info)):
            _,start_,stop_,start,stop = self.plq_info[ix]
            vec[start_:stop_] = vec_full[start:stop]
        return vec
    def expand_vector(self,vec):
        vec_full = np.zeros_like(self.x)
        for ix in range(len(self.plq_info)):
            _,start_,stop_,start,stop = self.plq_info[ix]
            vec_full[start:stop] = vec[start_:stop_]
        return vec_full
    def _transform_gradeints_rgn(self,cond):
        t0 = time.time()
        H = self.H - self.E * self.S + cond * np.eye(self.nparam)
        g = self.extract_vector(self.g) 
        deltas = np.linalg.solve(H,g) 
        self.deltas = self.expand_vector(deltas)
        print('\tRGN solver time=',time.time()-t0)
    def _transform_gradients_lin(self,cond):
        t0 = time.time()
        g = self.extract_vector(self.g) 
        Hvmean = self.extract_vector(self.Hvmean)
        vmean = self.extract_vector(self.vmean)
        Hi0 = g
        H0j = Hvmean - self.E * vmean
        A = np.block([[np.ones((1,1))*self.E,H0j.reshape(1,self.nparam)],
                      [Hi0.reshape(self.nparam,1),self.H]])
        B = np.block([[np.ones((1,1)),np.zeros((1,self.nparam))],
                      [np.zeros((self.nparam,1)),self.S+cond*np.eye(self.nparam)]])
        w,v = scipy.linalg.eig(A,b=B) 
        w,deltas,idx = _select_eigenvector(w.real,v.real)
        print('\timaginary norm=',np.linalg.norm(v[:,idx].imag))
        print('\teigenvalue =',w)
        print('\tscale1=',v[0,idx].real)
        self._scale_eigenvector(deltas)
        self.deltas = self.expand_vector(deltas)
        print('\tEIG solver time=',time.time()-t0)
    def _scale_eigenvector(self,deltas):
        if self.xi is None:
            Ns = self.vmean
        else:
            Sp = np.dot(self.S,deltas)
            Ns  = - (1.-self.xi) * Sp 
            Ns /= 1.-self.xi + self.xi * (1.+np.dot(deltas,Sp))**.5
        denom = 1. - np.dot(Ns,deltas)
        deltas /= -denom
        print('\tscale2=',denom)
        return deltas
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
