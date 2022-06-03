import numpy as np
from itertools import product
from pyblock3.algebra.fermion_ops import (
    ParticleNumber,
    onsite_U,
    creation,
    bonded_vaccum,
    _compute_swap_phase,
    hop_map,
    pn_dict,
    cre_map,
)
from pyblock3.algebra.fermion_encoding import get_state_map
from pyblock3.algebra.fermion_symmetry import U1,Z2
from pyblock3.algebra.fermion import SparseFermionTensor,Constructor
from pyblock3.algebra.core import SubTensor
from .block_interface import setting,eye

from .fermion_2d_tebd import LocalHam2D
from .fermion_2d import FPEPS 
from .fermion_core import (
    tensor_contract,
    FermionTensor,
    FermionTensorNetwork,
    FTNLinearOperator,
)
from ...linalg.base_linalg import eigh
from .utils import OPi,SumOpGrad,OPTERM,SumOpDMRG,insert,replace

from pyblock3.fcidump import FCIDUMP
from pyblock3.hamiltonian import Hamiltonian
np.set_printoptions(suppress=True,linewidth=1000,precision=4)

def ueg1(ke1,ee1,mu=0.,symmetry='u1', flat=True):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    state_map = get_state_map(symmetry)
    block_dict = dict()
    for key, pn in pn_dict.items():
        qlab, ind, dim = state_map[key]
        irreps = (qlab, qlab)
        if irreps not in block_dict:
            block_dict[irreps] = np.zeros([dim, dim])
        block_dict[irreps][ind, ind] += (pn==2) * ee1 + pn * (mu+ke1)
    blocks = [SubTensor(reduced=dat, q_labels=q_lab) for q_lab, dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="+-")
    if flat:
        return T.to_flat()
    else:
        return T
def ueg2(ke1, ee1, ke2, ee2, mu=0., fac=(0.,0.), symmetry='u1', flat=True):
    symmetry, flat = setting.dispatch_settings(symmetry=symmetry, flat=flat)
    faca, facb = fac
    (ke1a, ke1b) = ke1 if isinstance(ke1,tuple) else (ke1,ke1)
    (ee1a, ee1b) = ee1 if isinstance(ee1,tuple) else (ee1,ee1)
    state_map = get_state_map(symmetry)
    block_dict = dict()
    for s1, s2 in product(cre_map.keys(), repeat=2):
        q1, ix1, d1 = state_map[s1]
        q2, ix2, d2 = state_map[s2]
        val = (pn_dict[s1]==2) * faca * ee1a + pn_dict[s1] * faca * (ke1a-mu) +\
              (pn_dict[s2]==2) * facb * ee1b + pn_dict[s2] * facb * (ke1b-mu)
        val += pn_dict[s1] * pn_dict[s2] * ee2
        if (q1, q2, q1, q2) not in block_dict:
            block_dict[(q1, q2, q1, q2)] = np.zeros([d1, d2, d1, d2])
        dat = block_dict[(q1, q2, q1, q2)]
        phase = _compute_swap_phase(s1, s2, s1, s2)
        dat[ix1, ix2, ix1, ix2] += phase * val
        # ke2
        for s3 in hop_map[s1]:
            q3, ix3, d3 = state_map[s3]
            input_string = sorted(cre_map[s1]+cre_map[s2])
            for s4 in hop_map[s2]:
                q4, ix4, d4 = state_map[s4]
                output_string = sorted(cre_map[s3]+cre_map[s4])
                if input_string != output_string:
                    continue
                if (q1, q2, q3, q4) not in block_dict:
                    block_dict[(q1, q2, q3, q4)] = np.zeros([d1, d2, d3, d4])
                dat = block_dict[(q1, q2, q3, q4)]
                phase = _compute_swap_phase(s1, s2, s3, s4)
                dat[ix1, ix2, ix3, ix4] += phase * ke2
    blocks = [SubTensor(reduced=dat, q_labels=qlab) for qlab, dat in block_dict.items()]
    T = SparseFermionTensor(blocks=blocks, pattern="++--")
    if flat:
        return T.to_flat()
    else:
        return T
###################################################################################
# PWDB Hamiltonians 
###################################################################################
def _pwdb1(imax,jmax,Lx,Ly,has_coulomb=False):
    Nx,Ny = 2*imax+1,2*jmax+1
    N,Omega = Nx*Ny,Lx*Ly
    g,g1,g2 = [],[],[]
    for gx in range(-imax,imax+1):
        for gy in range(-jmax,jmax+1):
            g.append(2.0*np.pi*np.array([gx/Lx,gy/Ly]))
            g2.append(np.dot(g[-1],g[-1]))
            g1_inv = 0.0 if (gx==0 and gy==0) else 1.0/np.sqrt(g2[-1])
            g1.append(g1_inv)
    ke1 = sum(g2)/(2.0*N)
    ee1 = sum(g1)*2.0*np.pi/Omega if has_coulomb else 0.0 
    g1 = np.array(g1)
    g2 = np.array(g2)
    return ke1,ee1,g,g1,g2
def _pwdb2(site1,site2,g,g1,g2,Nx,Ny,Lx,Ly,has_coulomb=False,PBC_NN=False,
           maxdist=1000):
    dx,dy = site1[0]-site2[0],site1[1]-site2[1]
    if PBC_NN:
        if not (abs(dx)+abs(dy)==1 or \
                (dx==0 and abs(dy)==Ny-1) or \
                (dy==0 and abs(dx)==Nx-1)):
            return 0.,0.
    if abs(dx)+abs(dy)>maxdist:
        return 0.,0.
    N,Omega = Nx*Ny,Lx*Ly
    r = np.array([dx*Lx/Nx,dy*Ly/Ny])
    cos = np.array([np.cos(np.dot(gi,r)) for gi in g])
    ke2 = np.dot(cos,g2)/(2.0*N)
    ee2 = np.dot(cos,g1)*2.0*np.pi/Omega if has_coulomb else 0.
    return ke2,ee2
def get_pwdb_qc_ints(imax,jmax,Lx,Ly,has_coulomb=False,PBC_NN=False,maxdist=1000):
    Nx,Ny = 2*imax+1,2*jmax+1
    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
    sites = [(x,y) for y in range(Ny) for x in range(Nx)]

    h1 = np.zeros((len(sites),)*2)
    eri = np.zeros((len(sites),)*4)
    for i in range(len(sites)):
        h1[i,i] = ke1
        eri[i,i,i,i] = ee1
        for j in range(i+1,len(sites)):
            ke2,ee2 = _pwdb2(sites[i],sites[j],g,g1,g2,Nx,Ny,Lx,Ly,
                             has_coulomb=has_coulomb,PBC_NN=PBC_NN,maxdist=maxdist)
            h1[i,j] = ke2
            h1[j,i] = ke2
            eri[i,i,j,j] = ee2
            eri[j,j,i,i] = ee2
    return h1,eri
def build_pwdb_qc(imax,jmax,Lx,Ly,Ne,PBC_NN=False,maxdist=1000,cutoff=1e-9):
    Nx,Ny = 2*imax+1,2*jmax+1
    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=True)
    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
    norb = len(sites)

    fcidump = FCIDUMP(pg='c1',n_sites=norb,n_elec=Ne,twos=0,ipg=0,orb_sym=[0]*norb)
    hamil = Hamiltonian(fcidump,flat=True)
    def generate_terms(n_sites,c,d):
        for i in range(n_sites):
            yield ee1*c[i,0]*c[i,1]*d[i,1]*d[i,0]
            for s in [0,1]:
                yield ke1*c[i,s]*d[i,s]
            for j in range(i+1,n_sites):
                ke2,ee2 = _pwdb2(sites[i],sites[j],g,g1,g2,Nx,Ny,Lx,Ly,
                    has_coulomb=True,PBC_NN=PBC_NN,maxdist=maxdist)
                if abs(ke2)>cutoff: 
                    for s in [0,1]:
                        yield ke2*c[i,s]*d[j,s]
                        yield ke2*c[j,s]*d[i,s]
                if abs(ee2)>cutoff: 
                    for s1 in [0,1]:
                        for s2 in [0,1]:
                            yield ee2*(c[i,s1]*c[j,s2]*d[j,s2]*d[i,s1])
    return hamil,hamil.build_mpo(generate_terms,cutoff=cutoff).to_sparse()
def pwdb_grad(imax,jmax,Lx,Ly,has_coulomb=False,PBC_NN=False,
              maxdist=1000,symmetry='u1',flat=True,cutoff=1e-10):
    Nx,Ny = 2*imax+1,2*jmax+1
    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)

    # operators
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    ann_a = cre_a.dagger
    ann_b = cre_b.dagger
    pn    = ParticleNumber(symmetry=symmetry,flat=flat)
    nanb  = onsite_U(u=1.0,symmetry=symmetry)
    h1 = pn*ke1 + nanb*ee1
    sign_a = (-1)**(cre_a.parity*ann_a.parity)
    sign_b = (-1)**(cre_b.parity*ann_b.parity)
    data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
                'pn':pn,'onsite':h1}

    _1col_terms = []
    _2col_terms = []
    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
    for i,site1 in enumerate(sites):
        opis = OPi({'onsite':1.},site1),
        _1col_terms.append((opis,1.))
        for site2 in sites[i+1:]:
            ke2,ee2 = _pwdb2(site1,site2,g,g1,g2,Nx,Ny,Lx,Ly,
                             has_coulomb=has_coulomb,PBC_NN=PBC_NN,maxdist=maxdist)
            tmp = []
            if abs(ke2)>cutoff:
                opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
                tmp.append((opis,ke2*sign_a)) 
                opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
                tmp.append((opis,ke2)) 
                opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
                tmp.append((opis,ke2*sign_b)) 
                opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
                tmp.append((opis,ke2))
            if abs(ee2)>cutoff:
                opis = OPi({'pn':1.},site1),OPi({'pn':1.},site2)
                tmp.append((opis,ee2))
            if site1[1]==site2[1]:
                _1col_terms += tmp
            else:
                _2col_terms += tmp

    reuse = [(OPi({key:1.},(x,y)),min(Ny-1,y+maxdist),max(0,y-maxdist)) \
             for key in ['cre_a','ann_a','cre_b','ann_b'] for (x,y) in sites] 
    if has_coulomb:
        reuse += [(OPi({'pn':1.},(x,y)),min(Ny-1,y+maxdist),max(0,y-maxdist)) \
                 for (x,y) in sites] 
    return SumOpGrad(data_map,_1col_terms,_2col_terms,reuse)
def pwdb_dmrg(imax,jmax,Lx,Ly,mu=0.,has_coulomb=False,PBC_NN=False,
              maxdist=1000,symmetry='u1',flat=True,cutoff=1e-10):
    Nx,Ny = 2*imax+1,2*jmax+1
    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
    ke1 -= mu

    # operators
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    ann_a = cre_a.dagger
    ann_b = cre_b.dagger
    pn = ParticleNumber(symmetry=symmetry,flat=flat)
    nanb = onsite_U(u=1.0,symmetry=symmetry)
    h1 = pn*ke1 + nanb*ee1
    op_dict = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
               'pn':pn,'onsite':h1}

    ham_terms = []
    sites = [(x,y) for y in range(Ny) for x in range(Nx)]
    for i,(x1,y1) in enumerate(sites):
        inds1 = f'b{x1},{y1}',f'k{x1},{y1}'
        tags1 = f'I{x1},{y1}',f'COL{y1}',f'ROW{x1}','OPS'

        opi1 = 'onsite',inds1,tags1+(f'OPLABEL0,{x1},{y1}',)
        ham_terms.append(OPTERM([(x1,y1)],[opi1],1.0))
        for (x2,y2) in sites[i+1:]:
            ke2,ee2 = _pwdb2((x1,y1),(x2,y2),g,g1,g2,Nx,Ny,Lx,Ly,
                             has_coulomb=has_coulomb,PBC_NN=PBC_NN,maxdist=maxdist)
            inds2 = f'b{x2},{y2}',f'k{x2},{y2}'
            tags2 = f'I{x2},{y2}',f'COL{y2}',f'ROW{x2}','OPS'
            if abs(ke2)>cutoff:
                opi1 = 'cre_a',inds1,tags1+(f'OPLABEL1,{x1},{y1}',)
                opi2 = 'ann_a',inds2,tags2+(f'OPLABEL2,{x2},{y2}',)
                ham_terms.append(OPTERM([(x2,y2),(x1,y1)],[opi2,opi1],ke2))

                opi1 = 'cre_a',inds2,tags2+(f'OPLABEL1,{x2},{y2}',)
                opi2 = 'ann_a',inds1,tags1+(f'OPLABEL2,{x1},{y1}',)
                ham_terms.append(OPTERM([(x1,y1),(x2,y2)],[opi2,opi1],ke2))

                opi1 = 'cre_b',inds1,tags1+(f'OPLABEL3,{x1},{y1}',)
                opi2 = 'ann_b',inds2,tags2+(f'OPLABEL4,{x2},{y2}',)
                ham_terms.append(OPTERM([(x2,y2),(x1,y1)],[opi2,opi1],ke2))

                opi1 = 'cre_b',inds2,tags2+(f'OPLABEL3,{x2},{y2}',)
                opi2 = 'ann_b',inds1,tags1+(f'OPLABEL4,{x1},{y1}',)
                ham_terms.append(OPTERM([(x1,y1),(x2,y2)],[opi2,opi1],ke2))

            if abs(ee2)>cutoff:
                opi1 = 'pn',inds1,tags1+(f'OPLABEL5,{x1},{y1}',)
                opi2 = 'pn',inds2,tags2+(f'OPLABEL5,{x2},{y2}',)
                ham_terms.append(OPTERM([(x1,y1),(x2,y2)],[opi1,opi2],ee2))
    print('number of H terms=',len(ham_terms))
    return SumOpDMRG(op_dict,ham_terms)
def pwdb_tebd(imax,jmax,Lx,Ly,mu=0.,has_coulomb=False,PBC_NN=False,
              symmetry='u1',flat=True,cutoff=1e-10):
    Nx,Ny = 2*imax+1,2*jmax+1
    ke1,ee1,g,g1,g2 = _pwdb1(imax,jmax,Lx,Ly,has_coulomb=has_coulomb)
    def count_neighbour(i, j):
        return (i>0) + (i<Nx-1) + (j>0) + (j<Ny-1)

    ham = dict()
    sites = [(i,j) for i in range(Nx) for j in range(Ny)]
    # onsite and NN terms
    for (i,j) in sites: 
        count_ij = count_neighbour(i,j)
        if i+1 != Nx:
            where = (i,j), (i+1,j)
            count_b = count_neighbour(i+1,j)
            ke2, ee2 = _pwdb2(*where,g,g1,g2,Nx,Ny,Lx,Ly,has_coulomb=has_coulomb)
            ham[where] = ueg2(ke1, ee1, ke2, ee2, mu, (1./count_ij, 1./count_b), 
                              symmetry=symmetry, flat=flat)
        if j+1 != Ny:
            where = (i,j), (i,j+1)
            count_b = count_neighbour(i,j+1)
            ke2, ee2 = _pwdb2(*where,g,g1,g2,Nx,Ny,Lx,Ly,has_coulomb=has_coulomb)
            ham[where] = ueg2(ke1, ee1, ke2, ee2, mu, (1./count_ij, 1./count_b),
                              symmetry=symmetry, flat=flat)
    print('number of nearest-neighbor terms=',len(ham))
    return LocalHam2D(Nx,Ny,ham)

###################################################################################
# FD Hamiltonians 
###################################################################################
def _dist(site1,site2,h):
    dx,dy = site1[0]-site2[0],site1[1]-site2[1]
    return h*np.sqrt(dx**2+dy**2+1.)
def _u(site1,sites,h,rho,has_coulomb=True):
    if has_coulomb: 
        return -rho*h**2*sum([1./_dist(site1,site2,h) for site2 in sites]) 
    else:
        return 0.
def _back(sites,h,rho,has_coulomb=True):
    if has_coulomb:
        return 0.5*rho**2*h**4*sum([1./_dist(site1,site2,h) \
                                    for site1 in sites for site2 in sites])
    else:
        return 0.
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
def get_fd_qc_ints(N,L,rho,const=1.,order=2,has_coulomb=False):
    eps = L/(N+const)
    h = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    n_sites = len(sites)
    rs = 1./np.sqrt(np.pi*rho)
    back = _back(sites,h,rho,has_coulomb=has_coulomb) 
    #print(f'eps={eps},h={h},rho={rho},rs={rs},back={back}')
    h1 = np.zeros((len(sites),)*2)
    eri = np.zeros((len(sites),)*4)
    for i in range(n_sites):
        x,y  = sites[i]
        ke1 = _ke1(sites[i],eps,N,order=order) 
        ke1 += _u(sites[i],sites,h,rho,has_coulomb=has_coulomb)
        h1[i,i] = ke1
        ke2 = _ke2(eps,order=order)
        if x>0:
            j = sites.index((x-1,y))
            h1[j,i] = ke2
        if x<N-1:
            j = sites.index((x+1,y))
            h1[j,i] = ke2
        if y>0:
            j = sites.index((x,y-1))
            h1[j,i] = ke2
        if y<N-1:
            j = sites.index((x,y+1))
            h1[j,i] = ke2
        if order==4:
            ke2 = 1./(24.*eps**2)
            if x>1:
                j = sites.index((x-2,y))
                h1[j,i] = ke2
            if x<N-2:
                j = sites.index((x+2,y))
                h1[j,i] = ke2
            if y>1:
                j = sites.index((x,y-2))
                h1[j,i] = ke2
            if y<N-2:
                j = sites.index((x,y+2))
                h1[j,i] = ke2
    if has_coulomb:
        for i in range(n_sites): # long range Coulomb
            ee1 = 1./h
            eri[i,i,i,i] = ee1
            for j in range(i+1,n_sites):
                ee2 = 1./_dist(sites[i],sites[j],h)
                eri[i,i,j,j] = ee2
                eri[j,j,i,i] = ee2
    #print('check hermitian',np.linalg.norm(h1-h1.T))
    return h1,eri
    
def build_fd_qc(N,L,Ne,order=2,cutoff=1e-9):
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    norb = len(sites)
    rho = Ne/L**2
    rs = 1./np.sqrt(np.pi*rho)
    back = _back(sites,eps,rho,has_coulomb=True) 
    print(f'eps={eps},rho={rho},rs={rs},back={back}')
    
    fcidump = FCIDUMP(pg='c1',n_sites=norb,n_elec=Ne,twos=0,ipg=0,orb_sym=[0]*norb)
    hamil = Hamiltonian(fcidump,flat=True)
    def generate_terms(n_sites,c,d):
        for i in range(n_sites): # NN hopping
            x,y  = sites[i]
            ke2 = _ke2(eps,order=order)
            if x+1<N:
                j = sites.index((x+1,y))
                for s in [0,1]:
                    yield ke2*c[i,s]*d[j,s]
                    yield ke2*c[j,s]*d[i,s]
            if y+1<N:
                j = sites.index((x,y+1))
                for s in [0,1]:
                    yield ke2*c[i,s]*d[j,s]
                    yield ke2*c[j,s]*d[i,s]
        if order==4:
            for i in range(n_sites): # 3rd-NN hopping
                x,y = sites[i]
                ke2 = 1./(24.*eps**2)
                if x+2<N:
                    j = sites.index((x+2,y))
                    for s in [0,1]:
                        yield ke2*c[i,s]*d[j,s]
                        yield ke2*c[j,s]*d[i,s]
                if y+2<N:
                    j = sites.index((x,y+2))
                    for s in [0,1]:
                        yield ke2*c[i,s]*d[j,s]
                        yield ke2*c[j,s]*d[i,s]
        
        for i in range(n_sites): # onsite
            ke1 = _ke1(sites[i],eps,N,order=order) 
            ke1 += _u(sites[i],sites,eps,rho,has_coulomb=True)
            for s in [0,1]:
                yield ke1*c[i,s]*d[i,s]
            ee1 = 1./eps
            yield ee1*c[i,0]*c[i,1]*d[i,1]*d[i,0]
        for i in range(n_sites): # long range Coulomb
            for j in range(i+1,n_sites):
                ee2 = 1./_dist(sites[i],sites[j],eps)
                for s1 in [0,1]:
                    for s2 in [0,1]:  
                        yield ee2*c[i,s1]*c[j,s2]*d[j,s2]*d[i,s1]
    return hamil,hamil.build_mpo(generate_terms,cutoff=cutoff).to_sparse()
def fd_grad(N,L,rho,order=2,has_coulomb=False,maxdist=1000,symmetry='u1',flat=True):
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    norb = len(sites)
    rs = 1./np.sqrt(np.pi*rho)
    back = _back(sites,eps,rho,has_coulomb=has_coulomb) 
    print(f'eps={eps},rho={rho},rs={rs},back={back}')

    # operators
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    ann_a = cre_a.dagger
    ann_b = cre_b.dagger
    pn    = ParticleNumber(symmetry=symmetry,flat=flat)
    nanb  = onsite_U(u=1.0,symmetry=symmetry)
    sign_a = (-1)**(cre_a.parity*ann_a.parity)
    sign_b = (-1)**(cre_b.parity*ann_b.parity)
    data_map = {'cre_a':cre_a,'ann_a':ann_a,'cre_b':cre_b,'ann_b':ann_b,
                'pn':pn,'nanb':nanb}

    _1col_terms = []
    _2col_terms = []
    sites = [(x,y) for y in range(N) for x in range(N)]
    for (x,y) in sites:
        ke2 = _ke2(eps,order=order) 
        if x+1<N:
           site1,site2 = (x,y),(x+1,y)
           opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
           _1col_terms.append((opis,ke2*sign_a)) 
           opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
           _1col_terms.append((opis,ke2)) 
           opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
           _1col_terms.append((opis,ke2*sign_b)) 
           opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
           _1col_terms.append((opis,ke2))
        if y+1<N:
           site1,site2 = (x,y),(x,y+1)
           opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
           _2col_terms.append((opis,ke2*sign_a)) 
           opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
           _2col_terms.append((opis,ke2)) 
           opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
           _2col_terms.append((opis,ke2*sign_b)) 
           opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
           _2col_terms.append((opis,ke2))
        if order==4 and maxdist>=2:
            ke2 = 1./(24.*eps**2)
            if x+2<N:
               site1,site2 = (x,y),(x+2,y)
               opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
               _1col_terms.append((opis,ke2*sign_a)) 
               opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
               _1col_terms.append((opis,ke2)) 
               opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
               _1col_terms.append((opis,ke2*sign_b)) 
               opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
               _1col_terms.append((opis,ke2))
            if y+2<N:
               site1,site2 = (x,y),(x,y+2)
               opis = OPi({'cre_a':1.},site1),OPi({'ann_a':1.},site2)
               _2col_terms.append((opis,ke2*sign_a)) 
               opis = OPi({'ann_a':1.},site1),OPi({'cre_a':1.},site2)
               _2col_terms.append((opis,ke2)) 
               opis = OPi({'cre_b':1.},site1),OPi({'ann_b':1.},site2)
               _2col_terms.append((opis,ke2*sign_b)) 
               opis = OPi({'ann_b':1.},site1),OPi({'cre_b':1.},site2)
               _2col_terms.append((opis,ke2))
    for site in sites:
        ke1 = _ke1(site,eps,N,order=order) 
        ke1 += _u(site,sites,eps,rho,has_coulomb=has_coulomb) 
        ee1 = 1./eps if has_coulomb else 0.0 
        opis = OPi({'pn':ke1,'nanb':ee1},site),
        _1col_terms.append((opis,1.))
    if has_coulomb:
        for i,site1 in enumerate(sites):
            for site2 in sites[i+1:]:
                if abs(site1[0]-site2[0])+abs(site1[1]-site2[1])<=maxdist:
                    ee2 = 1./_dist(site1,site2,eps)
                    opis = OPi({'pn':1.},site1),OPi({'pn':1.},site2)
                    if site1[1]==site2[1]:
                        _1col_terms.append((opis,ee2))
                    else:
                        _2col_terms.append((opis,ee2))
    dy = 1 if order==2 else 2
    reuse = [(OPi({key:1.},(x,y)),min(N-1,y+dy),max(0,y-dy)) \
             for key in ['cre_a','ann_a','cre_b','ann_b'] for (x,y) in sites] 
    if has_coulomb:
        reuse += [(OPi({'pn':1.},(x,y)),min(N-1,y+maxdist),max(0,y-maxdist)) \
                  for (x,y) in sites] 
    return SumOpGrad(data_map,_1col_terms,_2col_terms,reuse)
def fd_tebd(N,L,rho,mu=0.,order=2,has_coulomb=False,symmetry='u1',flat=True):
    eps = L/(N+1.)
    sites = [(x,y) for x in range(N) for y in range(N)]
    norb = len(sites)
    rs = 1./np.sqrt(np.pi*rho)
    back = _back(sites,eps,rho,has_coulomb=has_coulomb) 
    print(f'eps={eps},rho={rho},rs={rs},back={back}')
    def count_neighbour(i, j):
        return (i>0) + (i<N-1) + (j>0) + (j<N-1)

    ham = dict()
    sites = [(i,j) for i in range(N) for j in range(N)]
    # onsite and NN terms
    ke2 = _ke2(eps,order=order)
    ee2 = 1./_dist((0,0),(0,1),eps) if has_coulomb else 0.0
    for (i,j) in sites: 
        count_ij = count_neighbour(i,j)
        ke1a = _ke1((i,j),eps,N,order=order) 
        ke1a += _u((i,j),sites,eps,rho,has_coulomb=has_coulomb)
        ee1 = 1./eps if has_coulomb else 0.0 
        if i+1 != N:
            where = (i,j), (i+1,j)
            count_b = count_neighbour(i+1,j)
            ke1b =  _ke1((i+1,j),eps,N,order=order)
            ke1b += _u((i+1,j),sites,eps,rho,has_coulomb=has_coulomb)
            ham[where] = ueg2((ke1a,ke1b),ee1,ke2,ee2,mu,(1./count_ij,1./count_b), 
                              symmetry=symmetry,flat=flat)
        if j+1 != N:
            where = (i,j), (i,j+1)
            count_b = count_neighbour(i,j+1)
            ke1b = _ke1((i,j+1),eps,N,order=order) 
            ke1b += _u((i,j+1),sites,eps,rho,has_coulomb=has_coulomb)
            ham[where] = ueg2((ke1a,ke1b),ee1,ke2,ee2,mu,(1./count_ij,1./count_b),
                              symmetry=symmetry,flat=flat)
    print('number of nearest-neighbor terms=',len(ham))
    return LocalHam2D(N,N,ham)
