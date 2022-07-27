import numpy as np
from itertools import product
np.set_printoptions(suppress=True,linewidth=1000,precision=4)
###################################################################################
# FD helper fxn 
###################################################################################
def _dist(site1,site2,h,const=1.):
    dx,dy = site1[0]-site2[0],site1[1]-site2[1]
    return h*np.sqrt(dx**2+dy**2+const)
def _u(site1,sites,h,rho,soft=True):
    if soft:
        I = sum([1./_dist(site1,site2,h,const=1.) for site2 in sites])
    else:
        sites_ = sites.copy()
        sites_.remove(site1)
        I = sum([1./_dist(site1,site2,h,const=0.) for site2 in sites_])
    return -I*rho*h**2 
def _back(sites,h,rho,soft=True):
    if soft:
        I = 0.5*sum([1./_dist(site1,site2,h,const=1.) \
                     for site1 in sites for site2 in sites])
    else:
        I = 0.
        for i,site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                I += 1./_dist(site1,site2,h,const=0.)
    return I*rho**2*h**4
def _ke1(site,eps,N,order=2):
    if order==2:
        return 2./eps**2
    else:
        x,y = site
        if (x==0 and y==0) or (x==0 and y==N-1) or\
           (x==N-1 and y==0) or (x==N-1 and y==N-1):
            return 58./(24.*eps**2)
        elif x==0 or x==N-1 or y==0 or y==N-1:
            return 59./(24.*eps**2)
        else:
            return 60./(24.*eps**2)
def _ke2(eps,order=2):
    if order==2:
        return -1./(2.*eps**2)
    else:
        return -16./(24.*eps**2)
###################################################################################
# FD molecular Hamiltonian 
###################################################################################
def get_U(N,L,Ne,soft=True):
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    n_sites = len(sites)
    rhob = Ne/(N**2*eps**2)

    U = np.zeros((N,)*2)
    for (x,y) in sites:
        U[x,y] = _u((x,y),sites,eps,rhob,soft=soft)
    return U
def get_T(N,L,order=2):
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    n_sites = len(sites)

    T = np.zeros((n_sites,)*2)
    for i in range(n_sites):
        x,y  = sites[i]
        T[i,i] = _ke1(sites[i],eps,N,order=order)
        ke2 = _ke2(eps,order=order)
        if x+1<N:
            j = sites.index((x+1,y))
            T[j,i] = T[i,j] = ke2
        if y+1<N:
            j = sites.index((x,y+1))
            T[j,i] = T[i,j] = ke2
        if order==4:
            ke2 = 1./(24.*eps**2)
            if x+2<N:
                j = sites.index((x+2,y))
                T[j,i] = T[i,j] = ke2
            if y+2<N:
                j = sites.index((x,y+2))
                T[j,i] = T[i,j] = ke2
    return T
def get_fd_qc_ints(N,L,Ne,T=None,order=2,has_coulomb=False,soft=True,symm=1):
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    n_sites = len(sites)

    h1 = get_T(N,L,order=order) if T is None else T.copy()
    eri,Eb = None,None
    if has_coulomb:
        rhob = Ne/(N**2*eps**2)
        Eb = _back(sites,eps,rhob,soft=soft)
        print(f'rhob={rhob},back={Eb}')
        for i in range(n_sites):
            h1[i,i] += _u(sites[i],sites,eps,rhob,soft=soft)
        lambda_ = 1./eps
        const = 1. if soft else 0.
        if symm==1:
            eri = np.zeros((n_sites,)*4)
            for i in range(n_sites): # long range Coulomb
                eri[i,i,i,i] = lambda_ 
                for j in range(i+1,n_sites):
                    ee2 = 1./_dist(sites[i],sites[j],eps,const=const)
                    eri[i,i,j,j] = eri[j,j,i,i] = ee2
        elif symm==4:
            npair = n_sites*(n_sites+1)//2
            diag_idxs = np.cumsum(range(1,n_sites+1))-1
            eri = np.zeros((npair,)*2)
            for i in range(n_sites):
                iix = diag_idxs[i] 
                eri[iix,iix] = lambda_
                for j in range(i+1,n_sites):
                    ee2 = 1./_dist(sites[i],sites[j],eps,const=const)
                    jix = diag_idxs[j]
                    eri[iix,jix] = eri[jix,iix] = ee2
        elif symm==8:
            npair = n_sites*(n_sites+1)//2
            eri = np.zeros(npair*(npair+1)//2)
            diag_idxs = np.cumsum(range(1,npair+1))-1
            for i in range(n_sites):
                iix = diag_idxs[i]
                ix = diag_idxs[iix]
                eri[ix] = lambda_
                for j in range(i+1,n_sites):
                    ee2 = 1./_dist(sites[i],sites[j],eps,const=const)
                    jix = diag_idxs[j]
                    ix = diag_idxs[jix-1]
                    eri[ix+iix+1] = ee2
        else:
            raise NotImplementedError(f'eri symmetry {symm} not implemented!')    
    return h1,eri,Eb
###################################################################################
# FD dmrg Hamiltonian 
###################################################################################
def build_fd_qc(N,L,Ne,order=2,has_coulomb=True,spinless=True,soft=True,cutoff=1e-9):
    from pyblock3.fcidump import FCIDUMP
    from pyblock3.hamiltonian import Hamiltonian
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    norb = len(sites)

    if has_coulomb:
        rhob = Ne/(N**2*eps**2) 
        Eb = _back(sites,eps,rhob,soft=soft) 
        print(f'rhob={rhob},back={Eb}')

    Na = Ne if spinless else (Ne+1) // 2
    Nb = Ne - Na    
    fcidump = FCIDUMP(pg='c1',n_sites=norb,n_elec=Ne,twos=Na-Nb,ipg=0,orb_sym=[0]*norb)
    hamil = Hamiltonian(fcidump,flat=True)
    spins = [0] if spinless else [0,1]
    def generate_terms(n_sites,c,d):
        for i in range(n_sites): # NN hopping
            x,y  = sites[i]
            ke2 = _ke2(eps,order=order)
            if x+1<N:
                j = sites.index((x+1,y))
                for s in spins:
                    yield ke2*c[i,s]*d[j,s]
                    yield ke2*c[j,s]*d[i,s]
            if y+1<N:
                j = sites.index((x,y+1))
                for s in spins:
                    yield ke2*c[i,s]*d[j,s]
                    yield ke2*c[j,s]*d[i,s]
        if order==4:
            for i in range(n_sites): # 3rd-NN hopping
                x,y = sites[i]
                ke2 = 1./(24.*eps**2)
                if x+2<N:
                    j = sites.index((x+2,y))
                    for s in spins:
                        yield ke2*c[i,s]*d[j,s]
                        yield ke2*c[j,s]*d[i,s]
                if y+2<N:
                    j = sites.index((x,y+2))
                    for s in spins:
                        yield ke2*c[i,s]*d[j,s]
                        yield ke2*c[j,s]*d[i,s]
        
        for i in range(n_sites): # onsite
            ke1 = _ke1(sites[i],eps,N,order=order) 
            if has_coulomb:
                ke1 += _u(sites[i],sites,eps,rhob,soft=soft)
                if not spinless:
                    ee1 = 1./eps
                    yield ee1*c[i,0]*c[i,1]*d[i,1]*d[i,0]
            for s in spins:
                yield ke1*c[i,s]*d[i,s]
        if has_coulomb:
            const = 1. if soft else 0.
            for i in range(n_sites): # long range Coulomb
                for j in range(i+1,n_sites):
                    ee2 = 1./_dist(sites[i],sites[j],eps,const=const)
                    for s1 in spins:
                        for s2 in spins:  
                            yield ee2*c[i,s1]*c[j,s2]*d[j,s2]*d[i,s1]
    return hamil,hamil.build_mpo(generate_terms,cutoff=cutoff).to_sparse()
###################################################################################
# FD fpeps Hamiltonian 
###################################################################################
def fd_grad(N,L,Ne,order=2,has_coulomb=True,spinless=True,soft=True,maxdist=1000):
    from .fermion_2d_grad_1 import SumOpGrad
    from .spin_utils import sign_a,sign_b
    eps = L/(N+1.)
    sites = [(x,y) for y in range(N) for x in range(N)]
    norb = len(sites)

    ls1 = []
    ls2 = []
    for (x,y) in sites:
        ke2 = _ke2(eps,order=order) 
        if x+1<N:
            ls1.append((f'cre_a{x}ann_a{x+1}',y,ke2*sign_a)) 
            ls1.append((f'ann_a{x}cre_a{x+1}',y,ke2)) 
            if not spinless:
                ls1.append((f'cre_b{x}ann_b{x+1}',y,ke2*sign_b)) 
                ls1.append((f'ann_b{x}cre_b{x+1}',y,ke2)) 
        if y+1<N:
            ls2.append(('cre_a',x,y,'ann_a',x,y+1,ke2*sign_a)) 
            ls2.append(('ann_a',x,y,'cre_a',x,y+1,ke2)) 
            if not spinless:
                ls2.append(('cre_b',x,y,'ann_b',x,y+1,ke2*sign_b)) 
                ls2.append(('ann_b',x,y,'cre_b',x,y+1,ke2)) 
        if order==4:
            ke2 = 1./(24.*eps**2)
            if x+2<N:
                ls1.append((f'cre_a{x}ann_a{x+2}',y,ke2*sign_a)) 
                ls1.append((f'ann_a{x}cre_a{x+2}',y,ke2)) 
                if not spinless:
                    ls1.append((f'cre_b{x}ann_b{x+2}',y,ke2*sign_b)) 
                    ls1.append((f'ann_b{x}cre_b{x+2}',y,ke2)) 
            if y+2<N:
                ls2.append(('cre_a',x,y,'ann_a',x,y+2,ke2*sign_a)) 
                ls2.append(('ann_a',x,y,'cre_a',x,y+2,ke2)) 
                if not spinless:
                    ls2.append(('cre_b',x,y,'ann_b',x,y+2,ke2*sign_b)) 
                    ls2.append(('ann_b',x,y,'cre_b',x,y+2,ke2)) 
    if has_coulomb:
        rhob = Ne/(N**2*eps**2) 
        Eb = _back(sites,eps,rhob,soft=soft) 
        print(f'rhob={rhob},Eb={Eb}')
    for site in sites:
        x,y = site
        ke1 = _ke1(site,eps,N,order=order)
        if has_coulomb: 
            ke1 += _u(site,sites,eps,rhob,soft=soft)
            if not spinless:
                ee1 = 1./eps 
                ls1.append((f'nanb{x}',y,ee1))
        ls1.append((f'pn{x}',y,ke1))
    if has_coulomb:
        const = 1. if soft else 0.
        for i,site1 in enumerate(sites):
            x1,y1 = site1
            for site2 in sites[i+1:]:
                x2,y2 = site2
                if abs(x1-x2)+abs(y1-y2)<=maxdist:
                    ee2 = 1./_dist(site1,site2,eps,const=const)
                    if y1==y2:
                        ls1.append((f'pn{x1}pn{x2}',y1,ee2))
                    else:
                        ls2.append((f'pn',x1,y1,'pn',x2,y2,ee2))
    tmp = []
    dy = 2 if order==4 else 1
    keys = ['cre_a','ann_a']
    if not spinless:
        keys += ['cre_b','ann_b']
    for y in range(N):
        tmpy = [('pn',x,y,N-1,0) for x in range(N)]
        if has_coulomb:
            tmpy += [('pn',x1,'pn',x2,y,N-1,0) for x1 in range(N) for x2 in range(x1+1,N)]
            if not spinless:
                tmpy += [('nanb',x,y,N-1,0) for x in range(N)]
        for x in range(N-1):
            tmpy.append(('cre_a',x,'ann_a',x+1,y,N-1,0))
            tmpy.append(('ann_a',x,'cre_a',x+1,y,N-1,0))
            if not spinless:
                tmpy.append(('cre_b',x,'ann_b',x+1,y,N-1,0))
                tmpy.append(('ann_b',x,'cre_b',x+1,y,N-1,0))
        if order==4:
            for x in range(N-2):
                tmpy.append(('cre_a',x,'ann_a',x+2,y,N-1,0))
                tmpy.append(('ann_a',x,'cre_a',x+2,y,N-1,0))
                if not spinless:
                    tmpy.append(('cre_b',x,'ann_b',x+2,y,N-1,0))
                    tmpy.append(('ann_b',x,'cre_b',x+2,y,N-1,0))
        tmpy += [(key,x,y,min(N-1,y+dy),max(0,y-dy)) for key in keys for x in range(N)] 
        tmp.append(tmpy)
    return SumOpGrad(ls1,ls2,tmp)
###################################################################################
# FD tebd Hamiltonian 
###################################################################################
def fd_tebd(N,L,order=2,soft=True):
    from .spin_utils import symmetry
    from .fermion_2d_tebd import Hubbard2D
    eps = L/(N+1.)
    t = - _ke2(eps,order=order)
    return Hubbard2D(t,0.,N,N,symmetry=symmetry) 
#def ueg1(ke1,ee1,mu=0.,symmetry='u1', flat=True):
#    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
#    state_map = get_state_map(symmetry)
#    block_dict = dict()
#    for key, pn in pn_dict.items():
#        qlab, ind, dim = state_map[key]
#        irreps = (qlab, qlab)
#        if irreps not in block_dict:
#            block_dict[irreps] = np.zeros([dim, dim])
#        block_dict[irreps][ind, ind] += (pn==2) * ee1 + pn * (mu+ke1)
#    blocks = [SubTensor(reduced=dat, q_labels=q_lab) for q_lab, dat in block_dict.items()]
#    T = SparseFermionTensor(blocks=blocks, pattern="+-")
#    if flat:
#        return T.to_flat()
#    else:
#        return T
#def ueg2(ke1, ee1, ke2, ee2, mu=0., fac=(0.,0.), symmetry='u1', flat=True):
#    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
#    faca, facb = fac
#    (ke1a, ke1b) = ke1 if isinstance(ke1,tuple) else (ke1,ke1)
#    (ee1a, ee1b) = ee1 if isinstance(ee1,tuple) else (ee1,ee1)
#    state_map = get_state_map(symmetry)
#    block_dict = dict()
#    for s1, s2 in product(cre_map.keys(), repeat=2):
#        q1, ix1, d1 = state_map[s1]
#        q2, ix2, d2 = state_map[s2]
#        val = (pn_dict[s1]==2) * faca * ee1a + pn_dict[s1] * faca * (ke1a-mu) +\
#              (pn_dict[s2]==2) * facb * ee1b + pn_dict[s2] * facb * (ke1b-mu)
#        val += pn_dict[s1] * pn_dict[s2] * ee2
#        if (q1, q2, q1, q2) not in block_dict:
#            block_dict[(q1, q2, q1, q2)] = np.zeros([d1, d2, d1, d2])
#        dat = block_dict[(q1, q2, q1, q2)]
#        phase = _compute_swap_phase(s1, s2, s1, s2)
#        dat[ix1, ix2, ix1, ix2] += phase * val
#        # ke2
#        for s3 in hop_map[s1]:
#            q3, ix3, d3 = state_map[s3]
#            input_string = sorted(cre_map[s1]+cre_map[s2])
#            for s4 in hop_map[s2]:
#                q4, ix4, d4 = state_map[s4]
#                output_string = sorted(cre_map[s3]+cre_map[s4])
#                if input_string != output_string:
#                    continue
#                if (q1, q2, q3, q4) not in block_dict:
#                    block_dict[(q1, q2, q3, q4)] = np.zeros([d1, d2, d3, d4])
#                dat = block_dict[(q1, q2, q3, q4)]
#                phase = _compute_swap_phase(s1, s2, s3, s4)
#                dat[ix1, ix2, ix3, ix4] += phase * ke2
#    blocks = [SubTensor(reduced=dat, q_labels=qlab) for qlab, dat in block_dict.items()]
#    T = SparseFermionTensor(blocks=blocks, pattern="++--")
#    if flat:
#        return T.to_flat()
#    else:
#        return T
####################################################################################
## PWDB Hamiltonians 
####################################################################################
#def _pwdb1(imax,jmax,Lx,Ly,has_coulomb=False):
#    Nx,Ny = 2*imax+1,2*jmax+1
#    N,Omega = Nx*Ny,Lx*Ly
#    g,g1,g2 = [],[],[]
#    for gx in range(-imax,imax+1):
#        for gy in range(-jmax,jmax+1):
#            g.append(2.0*np.pi*np.array([gx/Lx,gy/Ly]))
#            g2.append(np.dot(g[-1],g[-1]))
#            g1_inv = 0.0 if (gx==0 and gy==0) else 1.0/np.sqrt(g2[-1])
#            g1.append(g1_inv)
#    ke1 = sum(g2)/(2.0*N)
#    ee1 = sum(g1)*2.0*np.pi/Omega if has_coulomb else 0.0 
#    g1 = np.array(g1)
#    g2 = np.array(g2)
#    return ke1,ee1,g,g1,g2
#def _pwdb2(site1,site2,g,g1,g2,Nx,Ny,Lx,Ly,has_coulomb=False,PBC_NN=False,
#           maxdist=1000):
#    dx,dy = site1[0]-site2[0],site1[1]-site2[1]
#    if PBC_NN:
#        if not (abs(dx)+abs(dy)==1 or \
#                (dx==0 and abs(dy)==Ny-1) or \
#                (dy==0 and abs(dx)==Nx-1)):
#            return 0.,0.
#    if abs(dx)+abs(dy)>maxdist:
#        return 0.,0.
#    N,Omega = Nx*Ny,Lx*Ly
#    r = np.array([dx*Lx/Nx,dy*Ly/Ny])
#    cos = np.array([np.cos(np.dot(gi,r)) for gi in g])
#    ke2 = np.dot(cos,g2)/(2.0*N)
#    ee2 = np.dot(cos,g1)*2.0*np.pi/Omega if has_coulomb else 0.
#    return ke2,ee2
#def get_pwdb_qc_ints(imax,jmax,Lx,Ly,has_coulomb=False,PBC_NN=False,maxdist=1000):
#    Nx,Ny = 2*imax+1,2*jmax+1
#    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
#    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
#
#    h1 = np.zeros((len(sites),)*2)
#    eri = np.zeros((len(sites),)*4)
#    for i in range(len(sites)):
#        h1[i,i] = ke1
#        eri[i,i,i,i] = ee1
#        for j in range(i+1,len(sites)):
#            ke2,ee2 = _pwdb2(sites[i],sites[j],g,g1,g2,Nx,Ny,Lx,Ly,
#                             has_coulomb=has_coulomb,PBC_NN=PBC_NN,maxdist=maxdist)
#            h1[i,j] = ke2
#            h1[j,i] = ke2
#            eri[i,i,j,j] = ee2
#            eri[j,j,i,i] = ee2
#    return h1,eri
#def build_pwdb_qc(imax,jmax,Lx,Ly,Ne,PBC_NN=False,maxdist=1000,cutoff=1e-9):
#    Nx,Ny = 2*imax+1,2*jmax+1
#    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=True)
#    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
#    norb = len(sites)
#
#    fcidump = FCIDUMP(pg='c1',n_sites=norb,n_elec=Ne,twos=0,ipg=0,orb_sym=[0]*norb)
#    hamil = Hamiltonian(fcidump,flat=True)
#    def generate_terms(n_sites,c,d):
#        for i in range(n_sites):
#            yield ee1*c[i,0]*c[i,1]*d[i,1]*d[i,0]
#            for s in [0,1]:
#                yield ke1*c[i,s]*d[i,s]
#            for j in range(i+1,n_sites):
#                ke2,ee2 = _pwdb2(sites[i],sites[j],g,g1,g2,Nx,Ny,Lx,Ly,
#                    has_coulomb=True,PBC_NN=PBC_NN,maxdist=maxdist)
#                if abs(ke2)>cutoff: 
#                    for s in [0,1]:
#                        yield ke2*c[i,s]*d[j,s]
#                        yield ke2*c[j,s]*d[i,s]
#                if abs(ee2)>cutoff: 
#                    for s1 in [0,1]:
#                        for s2 in [0,1]:
#                            yield ee2*(c[i,s1]*c[j,s2]*d[j,s2]*d[i,s1])
#    return hamil,hamil.build_mpo(generate_terms,cutoff=cutoff).to_sparse()
#def pwdb_grad(imax,jmax,Lx,Ly,has_coulomb=False,PBC_NN=False,
#              maxdist=1000,symmetry='u1',flat=True,cutoff=1e-10):
#    Nx,Ny = 2*imax+1,2*jmax+1
#    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
#
#    # operators
#    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
#    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
#    ann_a = cre_a.dagger
#    ann_b = cre_b.dagger
#    pn    = ParticleNumber(symmetry=symmetry,flat=flat)
#    nanb  = onsite_U(u=1.0,symmetry=symmetry)
#    h1 = pn*ke1 + nanb*ee1
#    sign_a = (-1)**(cre_a.parity*ann_a.parity)
#    sign_b = (-1)**(cre_b.parity*ann_b.parity)
#    data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
#                'pn':pn,'onsite':h1}
#
#    _1col_terms = []
#    _2col_terms = []
#    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
#    for i,site1 in enumerate(sites):
#        opis = OPi({'onsite':1.},site1),
#        _1col_terms.append((opis,1.))
#        for site2 in sites[i+1:]:
#            ke2,ee2 = _pwdb2(site1,site2,g,g1,g2,Nx,Ny,Lx,Ly,
#                             has_coulomb=has_coulomb,PBC_NN=PBC_NN,maxdist=maxdist)
#            tmp = []
#            if abs(ke2)>cutoff:
#                opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
#                tmp.append((opis,ke2*sign_a)) 
#                opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
#                tmp.append((opis,ke2)) 
#                opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
#                tmp.append((opis,ke2*sign_b)) 
#                opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
#                tmp.append((opis,ke2))
#            if abs(ee2)>cutoff:
#                opis = OPi({'pn':1.},site1),OPi({'pn':1.},site2)
#                tmp.append((opis,ee2))
#            if site1[1]==site2[1]:
#                _1col_terms += tmp
#            else:
#                _2col_terms += tmp
#
#    reuse = [(OPi({key:1.},(x,y)),min(Ny-1,y+maxdist),max(0,y-maxdist)) \
#             for key in ['cre_a','ann_a','cre_b','ann_b'] for (x,y) in sites] 
#    if has_coulomb:
#        reuse += [(OPi({'pn':1.},(x,y)),min(Ny-1,y+maxdist),max(0,y-maxdist)) \
#                 for (x,y) in sites] 
#    return SumOpGrad(data_map,_1col_terms,_2col_terms,reuse)
#def pwdb_dmrg(imax,jmax,Lx,Ly,mu=0.,has_coulomb=False,PBC_NN=False,
#              maxdist=1000,symmetry='u1',flat=True,cutoff=1e-10):
#    Nx,Ny = 2*imax+1,2*jmax+1
#    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
#    ke1 -= mu
#
#    # operators
#    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
#    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
#    ann_a = cre_a.dagger
#    ann_b = cre_b.dagger
#    pn = ParticleNumber(symmetry=symmetry,flat=flat)
#    nanb = onsite_U(u=1.0,symmetry=symmetry)
#    h1 = pn*ke1 + nanb*ee1
#    op_dict = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
#               'pn':pn,'onsite':h1}
#
#    ham_terms = []
#    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
#    for i,(x1,y1) in enumerate(sites):
#        inds1 = f'b{x1},{y1}',f'k{x1},{y1}'
#        tags1 = f'I{x1},{y1}',f'COL{y1}',f'ROW{x1}','OPS'
#
#        opi1 = 'onsite',inds1,tags1+(f'OPLABEL0,{x1},{y1}',)
#        ham_terms.append(OPTERM([(x1,y1)],[opi1],1.0))
#        for (x2,y2) in sites[i+1:]:
#            ke2,ee2 = _pwdb2((x1,y1),(x2,y2),g,g1,g2,Nx,Ny,Lx,Ly,
#                             has_coulomb=has_coulomb,PBC_NN=PBC_NN,maxdist=maxdist)
#            inds2 = f'b{x2},{y2}',f'k{x2},{y2}'
#            tags2 = f'I{x2},{y2}',f'COL{y2}',f'ROW{x2}','OPS'
#            if abs(ke2)>cutoff:
#                opi1 = 'cre_a',inds1,tags1+(f'OPLABEL1,{x1},{y1}',)
#                opi2 = 'ann_a',inds2,tags2+(f'OPLABEL2,{x2},{y2}',)
#                ham_terms.append(OPTERM([(x2,y2),(x1,y1)],[opi2,opi1],ke2))
#
#                opi1 = 'cre_a',inds2,tags2+(f'OPLABEL1,{x2},{y2}',)
#                opi2 = 'ann_a',inds1,tags1+(f'OPLABEL2,{x1},{y1}',)
#                ham_terms.append(OPTERM([(x1,y1),(x2,y2)],[opi2,opi1],ke2))
#
#                opi1 = 'cre_b',inds1,tags1+(f'OPLABEL3,{x1},{y1}',)
#                opi2 = 'ann_b',inds2,tags2+(f'OPLABEL4,{x2},{y2}',)
#                ham_terms.append(OPTERM([(x2,y2),(x1,y1)],[opi2,opi1],ke2))
#
#                opi1 = 'cre_b',inds2,tags2+(f'OPLABEL3,{x2},{y2}',)
#                opi2 = 'ann_b',inds1,tags1+(f'OPLABEL4,{x1},{y1}',)
#                ham_terms.append(OPTERM([(x1,y1),(x2,y2)],[opi2,opi1],ke2))
#
#            if abs(ee2)>cutoff:
#                opi1 = 'pn',inds1,tags1+(f'OPLABEL5,{x1},{y1}',)
#                opi2 = 'pn',inds2,tags2+(f'OPLABEL5,{x2},{y2}',)
#                ham_terms.append(OPTERM([(x1,y1),(x2,y2)],[opi1,opi2],ee2))
#    print('number of H terms=',len(ham_terms))
#    return SumOpDMRG(op_dict,ham_terms)
#def pwdb_tebd(imax,jmax,Lx,Ly,mu=0.,has_coulomb=False,PBC_NN=False,
#              symmetry='u1',flat=True,cutoff=1e-10):
#    Nx,Ny = 2*imax+1,2*jmax+1
#    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
#    def count_neighbour(i, j):
#        return (i>0) + (i<Nx-1) + (j>0) + (j<Ny-1)
#
#    ham = dict()
#    sites = [(i,j) for i in range(Nx) for j in range(Ny)]
#    # onsite and NN terms
#    for (i,j) in sites: 
#        count_ij = count_neighbour(i,j)
#        if i+1 != Nx:
#            where = (i,j), (i+1,j)
#            count_b = count_neighbour(i+1,j)
#            ke2, ee2 = _pwdb2(*where,g,g1,g2,Nx,Ny,Lx,Ly,has_coulomb=has_coulomb)
#            ham[where] = ueg2(ke1, ee1, ke2, ee2, mu, (1./count_ij, 1./count_b), 
#                              symmetry=symmetry, flat=flat)
#        if j+1 != Ny:
#            where = (i,j), (i,j+1)
#            count_b = count_neighbour(i,j+1)
#            ke2, ee2 = _pwdb2(*where,g,g1,g2,Nx,Ny,Lx,Ly,has_coulomb=has_coulomb)
#            ham[where] = ueg2(ke1, ee1, ke2, ee2, mu, (1./count_ij, 1./count_b),
#                              symmetry=symmetry, flat=flat)
#    print('number of nearest-neighbor terms=',len(ham))
#    return LocalHam2D(Nx,Ny,ham)

