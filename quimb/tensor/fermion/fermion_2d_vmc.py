import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

SYMMETRY = 'u11' # sampler symmetry
# set tensor symmetry
import sys
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True):
    from pyblock3.algebra.fermion_ops import vaccum,creation,H1
    cre_a = creation(spin='a',symmetry=symmetry,flat=flat)
    cre_b = creation(spin='b',symmetry=symmetry,flat=flat)
    vac = vaccum(n=1,symmetry=symmetry,flat=flat)
    occ_a = np.tensordot(cre_a,vac,axes=([1],[0])) 
    occ_b = np.tensordot(cre_b,vac,axes=([1],[0])) 
    occ_db = np.tensordot(cre_a,occ_b,axes=([1],[0]))
    this.data_map = {0:vac,1:occ_a,2:occ_b,3:occ_db,
                     'cre_a':cre_a,
                     'cre_b':cre_b,
                     'ann_a':cre_a.dagger,
                     'ann_b':cre_b.dagger,
                     'h1':H1(symmetry=symmetry,flat=flat)}
    this.symmetry = symmetry
    this.flat = flat
    return this.data_map
pn_map = [0,1,1,2]
config_map = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}
from ..tensor_2d_vmc import flatten,flat2site 
#####################################################################################
# READ/WRITE FTN FUNCS
#####################################################################################
import pickle,uuid
from .fermion_core import FermionTensor,FermionTensorNetwork,rand_uuid,tensor_split
def load_ftn_from_disc(fname, delete_file=False):

    # Get the data
    if type(fname) != str:
        data = fname
    else:
        # Open up the file
        with open(fname, 'rb') as f:
            data = pickle.load(f)

    # Set up a dummy fermionic tensor network
    tn = FermionTensorNetwork([])

    # Put the tensors into the ftn
    tensors = [None,] * data['ntensors']
    for i in range(data['ntensors']):

        # Get the tensor
        ten_info = data['tensors'][i]
        ten = ten_info['tensor']
        ten = FermionTensor(ten.data, inds=ten.inds, tags=ten.tags)

        # Get/set tensor info
        tid, site = ten_info['fermion_info']
        ten.fermion_owner = None
        ten._avoid_phase = False

        # Add the required phase
        ten.phase = ten_info['phase']

        # Add to tensor list
        tensors[site] = (tid, ten)

    # Add tensors to the tn
    for (tid, ten) in tensors:
        tn.add_tensor(ten, tid=tid, virtual=True)

    # Get addition attributes needed
    tn_info = data['tn_info']

    # Set all attributes in the ftn
    extra_props = dict()
    for props in tn_info:
        extra_props[props[1:]] = tn_info[props]

    # Convert it to the correct type of fermionic tensor network
    tn = tn.view_as_(data['class'], **extra_props)

    # Remove file (if desired)
    if delete_file:
        delete_ftn_from_disc(fname)

    # Return resulting tn
    return tn

def rand_fname():
    return str(uuid.uuid4())
def write_ftn_to_disc(tn, tmpdir, provided_filename=False):

    # Create a generic dictionary to hold all information
    data = dict()

    # Save which type of tn this is
    data['class'] = type(tn)

    # Add information relevant to the tensors
    data['tn_info'] = dict()
    for e in tn._EXTRA_PROPS:
        data['tn_info'][e] = getattr(tn, e)

    # Add the tensors themselves
    data['tensors'] = []
    ntensors = 0
    for ten in tn.tensors:
        ten_info = dict()
        ten_info['fermion_info'] = ten.get_fermion_info()
        ten_info['phase'] = ten.phase
        ten_info['tensor'] = ten
        data['tensors'].append(ten_info)
        ntensors += 1
    data['ntensors'] = ntensors

    # If tmpdir is None, then return the dictionary
    if tmpdir is None:
        return data

    # Write fermionic tensor network to disc
    else:
        # Create a temporary file
        if provided_filename:
            fname = tmpdir
            print('saving to ', fname)
        else:
            if tmpdir[-1] != '/': 
                tmpdir = tmpdir + '/'
            fname = tmpdir + rand_fname()

        # Write to a file
        with open(fname, 'wb') as f:
            pickle.dump(data, f)

        # Return the filename
        return fname
####################################################################################
# initialization fxns
####################################################################################
def get_vaccum(Lx,Ly):
    """
    helper function to generate initial guess from regular PEPS
    |psi> = \prod (|alpha> + |beta>) at each site
    this function only works for half filled case with U1 symmetry
    Note energy of this wfn is 0
    """
    from pyblock3.algebra.fermion_ops import bonded_vaccum
    from ..tensor_2d import PEPS
    from .fermion_2d import FPEPS
    tn = PEPS.rand(Lx,Ly,bond_dim=1,phys_dim=1)
    ftn = FermionTensorNetwork([])
    ind_to_pattern_map = dict()
    inv_pattern = {"+":"-", "-":"+"}
    def get_pattern(inds):
        """
        make sure patterns match in input tensors, eg,
        --->A--->B--->
         i    j    k
        pattern for A_ij = +-
        pattern for B_jk = +-
        the pattern of j index must be reversed in two operands
        """
        pattern = ""
        for ix in inds[:-1]:
            if ix in ind_to_pattern_map:
                ipattern = inv_pattern[ind_to_pattern_map[ix]]
            else:
                nmin = pattern.count("-")
                ipattern = "-" if nmin*2<len(pattern) else "+"
                ind_to_pattern_map[ix] = ipattern
            pattern += ipattern
        pattern += "+" # assuming last index is the physical index
        return pattern
    for ix, iy in itertools.product(range(tn.Lx), range(tn.Ly)):
        T = tn[ix, iy]
        pattern = get_pattern(T.inds)
        #put vaccum at site (ix, iy) and apply a^{\dagger}
        data = bonded_vaccum((1,)*(T.ndim-1), pattern=pattern,
                             symmetry=symmetry,flat=flat)
        new_T = FermionTensor(data, inds=T.inds, tags=T.tags)
        ftn.add_tensor(new_T, virtual=False)
    ftn.view_as_(FPEPS, like=tn)
    return ftn
def create_particle(fpeps,site,spin):
    cre = data_map[f'cre_{spin}'].copy()
    T = fpeps[fpeps.site_tag(*site)]
    trans_order = list(range(1,T.ndim))+[0] 
    data = np.tensordot(cre, T.data, axes=((1,), (-1,))).transpose(trans_order)
    T.modify(data=data)
    return fpeps
def get_product_state(Lx,Ly,spin_map):
    fpeps = get_vaccum(Lx,Ly)
    for spin,sites in spin_map.items():
        for site in sites:
            fpeps = create_particle(fpeps,site,spin)
    return fpeps
####################################################################################
# amplitude fxns 
####################################################################################
from ..tensor_2d_vmc import ContractionEngine as ContractionEngine_
class ContractionEngine(ContractionEngine_): 
    def get_bra_tsr(self,fpeps,ci,i,j,append=''):
        inds = fpeps.site_ind(i,j)+append,
        tags = fpeps.site_tag(i,j),fpeps.row_tag(i),fpeps.col_tag(j),'BRA'
        data = data_map[ci].dagger
        return FermionTensor(data=data,inds=inds,tags=tags)
    def site_grad(self,ftn_plq,i,j):
        ket = ftn_plq[ftn_plq.site_tag(i,j),'KET']
        tid = ket.get_fermion_info()[0]
        ket = ftn_plq._pop_tensor(tid,remove_from_fermion_space='end')
        g = ftn_plq.contract(output_inds=ket.inds[::-1])
        return g.data.dagger 
def compute_fpeps_parity(fs,start,stop):
    if start==stop:
        return 0
    tids = [fs.get_tid_from_site(site) for site in range(start,stop)]
    tsrs = [fs.tensor_order[tid][0] for tid in tids] 
    return sum([tsr.parity for tsr in tsrs]) % 2
from ..tensor_2d_vmc import AmplitudeFactory as AmplitudeFactory_
class AmplitudeFactory(ContractionEngine,AmplitudeFactory_):
    def __init__(self,psi,dmrg=False,**contract_opts):
        self.contract_opts=contract_opts
        self.Lx,self.Ly = psi.Lx,psi.Ly
        psi.reorder(direction='row',inplace=True)
        psi.add_tag('KET')
        self.constructors = self.get_constructors(psi)
        self.get_block_dict()
        self.dmrg = dmrg

        self.ix = None
        self.set_psi(psi) # current state stored in self.psi
        self.parity_cum = self.get_parity_cum()
        self.sign = dict()
    def update(self,x,fname=None,root=0):
        psi = self.vec2psi(x,inplace=True)
        self.set_psi(psi) 
        if RANK==root:
            if fname is not None: # save psi to disc
                write_ftn_to_disc(psi,fname,provided_filename=True)
        return psi
    def get_constructors(self,fpeps):
        from .block_interface import Constructor
        constructors = [None] * (fpeps.Lx * fpeps.Ly)
        for i,j in itertools.product(range(fpeps.Lx),range(fpeps.Ly)):
            data = fpeps[fpeps.site_tag(i,j)].data
            bond_infos = [data.get_bond_info(ax,flip=False) \
                          for ax in range(data.ndim)]
            cons = Constructor.from_bond_infos(bond_infos,data.pattern,flat=this.flat)
            dq = data.dq
            size = cons.vector_size(dq)
            ix = flatten(i,j,fpeps.Ly)
            constructors[ix] = (cons,dq),size,(i,j)
        return constructors
    def tensor2vec(self,tsr,ix):
        cons,dq = self.constructors[ix][0]
        return cons.tensor_to_vector(tsr) 
    def vec2tensor(self,x,ix):
        cons,dq = self.constructors[ix][0]
        return cons.vector_to_tensor(x,dq)
    def get_parity_cum(self):
        parity = []
        fs = self.psi.fermion_space
        for i in range(1,self.Lx): # only need parity of row 1,...,Lx-1
            start,stop = i*self.Ly,(i+1)*self.Ly
            parity.append(compute_fpeps_parity(fs,start,stop))
        return np.cumsum(np.array(parity[::-1]))
    def compute_config_parity(self,config):
        parity = [None] * self.Lx
        for i in range(self.Lx):
            parity[i] = sum([pn_map[ci] for ci in config[i*self.Ly:(i+1)*self.Ly]]) % 2
        parity = np.array(parity[::-1])
        parity_cum = np.cumsum(parity[:-1])
        parity_cum += self.parity_cum 
        return np.dot(parity[1:],parity_cum) % 2
    def compute_config_sign(self,config):
        if config in self.sign:
            return self.sign[config]
        sign = (-1) ** self.compute_config_parity(config)
        self.sign[config] = sign
        return sign
####################################################################################
# ham class 
####################################################################################
from ..tensor_2d_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(ContractionEngine,Hamiltonian_):
    def set_pepo_tsrs(self): 
        vac = data_map[0]
        occ = data_map[3]
    
        from pyblock3.algebra.fermion_encoding import get_state_map
        from pyblock3.algebra.fermion import eye 
        state_map = get_state_map(symmetry)
        bond_info = dict()
        for qlab,_,sh in state_map.values():
            if qlab not in bond_info:
                bond_info[qlab] = sh
        I1 = eye(bond_info,flat=flat)
        I2 = np.tensordot(I1,I1,axes=([],[]))
        P0 = np.tensordot(vac,vac.dagger,axes=([],[]))
        P1 = np.tensordot(occ,occ.dagger,axes=([],[]))
    
        ADD = np.tensordot(vac,np.tensordot(vac.dagger,vac.dagger,axes=([],[])),axes=([],[])) \
            + np.tensordot(occ,np.tensordot(occ.dagger,vac.dagger,axes=([],[])),axes=([],[])) \
            + np.tensordot(occ,np.tensordot(vac.dagger,occ.dagger,axes=([],[])),axes=([],[]))
        ADD.shape = (4,)*3
        self.data_map.update({'I1':I1,'I2':I2,'P0':P0,'P1':P1,'ADD':ADD,'occ':occ,'vac':vac})
    def get_data(self,coeff,key='h1'):
        data = np.tensordot(self.data_map['P0'],self.data_map['I2'],axes=([],[])) \
             + np.tensordot(self.data_map['P1'],self.data_map[key],axes=([],[])) * coeff
        data.shape = (4,)*6
        return data
    def get_mpo(self,coeffs,L,key='h1'):
        # add h1
        tsrs = [] 
        pixs = [None] * L
        for ix,(ix1,ix2,coeff) in enumerate(coeffs):
            data = self.get_data(coeff,key=key)
            b1 = f'k{ix1}*' if pixs[ix1] is None else pixs[ix1]
            k1 = rand_uuid()
            pixs[ix1] = k1
            b2 = f'k{ix2}*' if pixs[ix2] is None else pixs[ix2]
            k2 = rand_uuid()
            pixs[ix2] = k2
            inds = f'v{ix}*',f'v{ix}',b1,k1,b2,k2
            tags = f'L{ix}',f'L{ix+1}'
            tsrs.append(FermionTensor(data=data,inds=inds,tags=tags))
        mpo = FermionTensorNetwork(tsrs[::-1],virtual=True)
        mpo.reindex_({pix:f'k{ix}' for ix,pix in enumerate(pixs)})
        # add ADD
        bixs = ['v0']+[rand_uuid() for ix in range(1,len(coeffs))]
        for ix in range(1,len(coeffs)):
            ix1,ix2,_ = coeffs[ix]
            tags = f'L{ix1}',f'L{ix2}'
            inds = bixs[ix]+'*',f'v{ix}*',bixs[ix-1]+'*'
            mpo.add_tensor(FermionTensor(data=self.data_map['ADD'].copy(),inds=inds,tags=tags))
            inds = bixs[ix-1],f'v{ix}',bixs[ix]
            mpo.add_tensor(FermionTensor(data=self.data_map['ADD'].dagger,inds=inds,tags=tags))
    
        # compress
        for ix in range(L-1):
            mpo.contract_tags(f'L{ix}',which='any',inplace=True)
            tid = mpo[f'L{ix}'].get_fermion_info()[0]
            tsr = mpo._pop_tensor(tid,remove_from_fermion_space='end')

            rix = f'k{ix}*',f'k{ix}'
            if ix>0:
                rix = rix+(bix,)
            bix = rand_uuid()
            tl,tr = tensor_split(tsr,left_inds=None,right_inds=rix,bond_ind=bix,method='svd',get='tensors')
            tr.modify(tags=f'L{ix}')
            tl.drop_tags(tags=f'L{ix}')
            mpo.add_tensor(tr,virtual=True)
            mpo.add_tensor(tl,virtual=True)
        mpo[f'L{L-1}'].reindex_({bixs[-1]:'v',bixs[-1]+'*':'v*'})
        return mpo
    def get_comb(self,mpos):
        # add mpo
        pepo = FermionTensorNetwork([])
        L = mpos[0].num_tensors
        tag = f'L{L-1}'
        for ix1,mpo in enumerate(mpos):
            for ix2 in range(L):
                mpo[f'L{ix2}'].reindex_({f'k{ix2}':f'mpo{ix1}_k{ix2}',f'k{ix2}*':f'mpo{ix1}_k{ix2}*'})
            mpo[tag].reindex_({'v':f'v{ix1}','v*':f'v{ix1}*'})
            mpo.add_tag(f'mpo{ix1}')
            pepo.add_tensor_network(mpo,virtual=True,check_collisions=True)
        # add ADD
        nmpo = len(mpos)
        bix = ['v0']+[rand_uuid() for ix in range(1,nmpo)]
        for ix in range(1,nmpo):
            tags = f'mpo{ix}',tag 
            inds = bix[ix]+'*',f'v{ix}*',bix[ix-1]+'*'
            pepo.add_tensor(FermionTensor(data=self.data_map['ADD'].copy(),inds=inds,tags=tags))
            inds = bix[ix-1],f'v{ix}',bix[ix]
            pepo.add_tensor(FermionTensor(data=self.data_map['ADD'].dagger,inds=inds,tags=tags))
    
        # compress
        for ix in range(nmpo-1):
            pepo.contract_tags((f'mpo{ix}',tag),which='all',inplace=True)
            tid = pepo[f'mpo{ix}',tag].get_fermion_info()[0]
            tsr = pepo._pop_tensor(tid,remove_from_fermion_space='end')

            lix = bix[ix],bix[ix]+'*'
            tl,tr = tensor_split(tsr,left_inds=lix,method='svd',get='tensors')
            tl.modify(tags=(f'mpo{ix+1}',tag))
            pepo.add_tensor(tr,virtual=True)
            pepo.add_tensor(tl,virtual=True)
        pepo.contract_tags((f'mpo{nmpo-1}',tag),which='all',inplace=True)
        pepo[f'mpo{nmpo-1}',tag].reindex_({bix[-1]:'v',bix[-1]+'*':'v*'})
        return pepo
    def combine(self,top,bot):
        Lx,Ly = bot.Lx,bot.Ly
        for i in range(Lx):
            for j in range(Ly): 
                pix = top.site_ind(i,j)
                top[i,j].reindex_({pix:pix+'_'})
                bot[i,j].reindex_({pix+'*':pix+'_'})
        top[Lx-1,Ly-1].reindex_({'v':'vt','v*':'vt*'})
        bot[Lx-1,Ly-1].reindex_({'v':'vb','v*':'vb*'})
    
        pepo = bot
        pepo.add_tensor_network(top,virtual=True)
        tags = pepo.site_tag(Lx-1,Ly-1)
        pepo.add_tensor(
            FermionTensor(data=self.data_map['ADD'].copy(),inds=('v*','vt*','vb*'),tags=tags),virtual=True) 
        pepo.add_tensor(
            FermionTensor(data=self.data_map['ADD'].dagger,inds=('vb','vt','v'),tags=tags),virtual=True) 
        for i in range(Lx):
            for j in range(Ly): 
                pepo.contract_tags(pepo.site_tag(i,j),inplace=True) 
        return pepo 
    def trace_virtual(self,pepo):
        tags = pepo.site_tag(pepo.Lx-1,pepo.Ly-1)
        pepo.add_tensor(FermionTensor(data=self.data_map['occ'].dagger,inds=('v*',),tags=tags),virtual=True)
        pepo.add_tensor(FermionTensor(data=self.data_map['occ'].copy(),inds=('v',),tags=tags),virtual=True)
        pepo.contract_tags(tags,inplace=True)
        pepo.add_tag('BRA')
        return pepo
    def pair_tensor(self,bixs,kixs,tags=None):
        data = self.data_map[self.key].copy()
        inds = bixs[0],kixs[0],bixs[1],kixs[1]
        return FermionTensor(data=data,inds=inds,tags=tags) 
    def config_parity(self,config,ix1,ix2):
        return sum([pn_map[ci] for ci in config[ix1+1:ix2]]) % 2
    def complete_plq(self,plq,norm):
        return plq
class Hubbard(Hamiltonian):
    def __init__(self,t,u,Lx,Ly,**kwargs):
        super().__init__(Lx,Ly,**kwargs)
        self.t,self.u = t,u
        self.bsz = 2
        self.data_map = dict()
        self.set_gate()
    def set_gate(self):
        data = data_map['h1']
        data = np.transpose(data,axes=(0,2,1,3))
        self.key = 'h1'
        self.data_map[self.key] = data
    def get_coeffs(self,L):
        coeffs = []
        for i in range(L):
            if i+1 < L:
                coeffs.append((i,i+1,-self.t))
        return coeffs
    def pair_coeff(self,site1,site2):
        return -self.t
    def pair_valid(self,i1,i2):
        if i1==i2:
            return False
        else:
            return True
    def compute_local_energy_eigen(self,config):
        config = np.array(config,dtype=int)
        return self.u*len(config[config==3])
    def pair_terms(self,i1,i2):
        n1,n2 = pn_map[i1],pn_map[i2]
        nsum,ndiff = n1+n2,abs(n1-n2)
        if ndiff==1:
            sign = 1 if nsum==1 else -1
            return [(i2,i1,sign)]
        if ndiff==2:
            return [(1,2,-1),(2,1,1)] 
        if ndiff==0:
            sign = i1-i2
            return [(0,3,sign),(3,0,sign)]
####################################################################################
# sampler 
####################################################################################
from ..tensor_2d_vmc import ExchangeSampler as ExchangeSampler_
class ExchangeSampler(ContractionEngine,ExchangeSampler_):
    def new_pair(self,i1,i2):
        if SYMMETRY=='u11':
            return self.new_pair_u11(i1,i2)
        else:
            raise NotImplementedError
    def new_pair_u11(self,i1,i2):
        n = abs(pn_map[i1]-pn_map[i2])
        if n==1:
            i1_new,i2_new = i2,i1
        else:
            choices = [(i2,i1),(0,3),(3,0)] if n==0 else [(i2,i1),(1,2),(2,1)]
            i1_new,i2_new = self.rng.choice(choices)
        return i1_new,i2_new 
    def new_pair_u1(self,i1,i2):
        return
from ..tensor_2d_vmc import DenseSampler as DenseSampler_
class DenseSampler(DenseSampler_):
    def __init__(self,Lx,Ly,nelec,**kwargs):
        self.nelec = nelec
        super().__init__(Lx,Ly,None,**kwargs)
    def get_all_configs(self):
        if SYMMETRY=='u1':
            return self.get_all_configs_u1()
        elif SYMMETRY=='u11':
            return self.get_all_configs_u11()
        else:
            raise NotImplementedError
    def get_all_configs_u11(self):
        assert isinstance(self.nelec,tuple)
        sites = list(range(self.nsite))
        ls = [None] * 2
        for spin in (0,1):
            occs = list(itertools.combinations(sites,self.nelec[spin]))
            configs = [None] * len(occs) 
            for i,occ in enumerate(occs):
                config = [0] * self.nsite 
                for ix in occ:
                    config[ix] = 1
                configs[i] = tuple(config)
            ls[spin] = configs

        na,nb = len(ls[0]),len(ls[1])
        configs = [None] * (na*nb)
        for ixa,configa in enumerate(ls[0]):
            for ixb,configb in enumerate(ls[1]):
                config = [config_map[configa[i],configb[i]] \
                          for i in range(self.nsite)]
                ix = ixa * nb + ixb
                configs[ix] = tuple(config)
        return configs
    def get_all_configs_u1(self):
        if isinstance(self.nelec,tuple):
            self.nelec = sum(self.nelec)
        sites = list(range(self.nsite*2))
        occs = list(itertools.combinations(sites,self.nelec))
        configs = [None] * len(occs) 
        for i,occ in enumerate(occs):
            config = [0] * (self.nsite*2) 
            for ix in occ:
                config[ix] = 1
            configs[i] = tuple(config)

        for ix in range(len(configs)):
            config = configs[ix]
            configa,configb = config[:self.nsite],config[self.nsite:]
            config = [config_map[configa[i],configb[i]] for i in range(self.nsite)]
            configs[ix] = tuple(config)
        return configs

