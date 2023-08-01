import time,itertools
import numpy as np

from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import sys
this = sys.modules[__name__]
def set_options(symmetry='u1',flat=True,pbc=False,deterministic=False):
    this._PBC = pbc
    this._DETERMINISTIC = deterministic
    this._SYMMETRY = symmetry
    this._FLAT = flat 
    from ..tensor_2d_vmc import set_options 
    set_options(pbc=pbc,deterministic=deterministic)
    from .fermion_2d_vmc import set_options as set_options
    return set_options(symmetry=symmetry,flat=flat,pbc=pbc,deterministic=deterministic)

def config_to_ab(config):
    #print('parse',config)
    config_a = [None] * len(config)
    config_b = [None] * len(config)
    map_a = {0:0,1:1,2:0,3:1}
    map_b = {0:0,1:0,2:1,3:1}
    for ix,ci in enumerate(config):
        config_a[ix] = map_a[ci] 
        config_b[ix] = map_b[ci] 
    return tuple(config_a),tuple(config_b)
def config_from_ab(config_a,config_b):
    map_ = {(0,0):0,(1,0):1,(0,1):2,(1,1):3}
    return tuple([map_[config_a[ix],config_b[ix]] for ix in len(config_a)])
def parse_config(config):
    #if len(config)==2:
    #    config_a,config_b = config
    #    config_full = config_from_ab(config_a,config_b)
    #else:
    #    config_full = config
    #    config_a,config_b = config_to_ab(config)
    config_full = config
    config_a,config_b = config_to_ab(config)
    return config_a,config_b,config_full
from ..tensor_2d_vmc import AmplitudeFactory as BosonAmplitudeFactory
from .fermion_2d_vmc import AmplitudeFactory as FermionAmplitudeFactory
class JastrowAmplitudeFactory(BosonAmplitudeFactory):
    def __init__(self,psi,blks=None,backend='numpy',**compress_opts):
        super().__init__(psi,blks=blks,phys_dim=4,backend=backend,**compress_opts)
    def pair_terms(self,i1,i2,spin):
        if spin=='a':
            map_ = {(0,1):(1,0),(1,0):(0,1),
                    (2,3):(3,2),(3,2):(2,3),
                    (0,3):(1,2),(3,0):(2,1),
                    (1,2):(0,3),(2,1):(3,0)}
        elif spin=='b':
            map_ = {(0,2):(2,0),(2,0):(0,2),
                    (1,3):(3,1),(3,1):(1,3),
                    (0,3):(2,1),(3,0):(1,2),
                    (1,2):(3,0),(2,1):(0,3)}
        else:
            raise ValueError
        return map_.get((i1,i2),(None,)*2)
    def pair_energy_from_plq(self,tn,config,where,model):
        ix1,ix2 = [model.flatten(where[ix]) for ix in (0,1)]
        i1,i2 = config[ix1],config[ix2] 
        if not model.pair_valid(i1,i2): # term vanishes 
            return None 
        cx = [None] * 2
        for ix,spin in zip((0,1),('a','b')):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue
            tn_new = self.replace_sites(tn.copy(),where,(i1_new,i2_new))
            cx[ix] = safe_contract(tn_new)
        return cx 
    def pair_energy_deterministic(self,config,site1,site2,top,bot,model,cache_top=None,cache_bot=None):
        ix1,ix2 = model.flatten(site1),model.flatten(site2)
        i1,i2 = config[ix1],config[ix2]
        if not model.pair_valid(i1,i2): # term vanishes 
            return None 
        imin = min(self.rix1+1,site1[0],site2[0]) 
        imax = max(self.rix2-1,site1[0],site2[0]) 
        cx = [None] * 2 
        for ix,spin in zip((0,1),('a','b')):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue 
            config_new = list(config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_new = tuple(config_new)

            cx_new = self._unsigned_amplitude_deterministic(config_new,imin,imax,bot,top,cache_bot=cache_bot,cache_top=cache_top)
            if cx_new is not None:
                cx[ix] = cx_new
        return cx 
class AmplitudeFactory(FermionAmplitudeFactory):
    def __init__(self,psi,blks=None,backend='numpy',**compress_opts):
        self.psi = [None] * 3
        self.psi[0] = FermionAmplitudeFactory(psi[0],blks=blks,spinless=True,backend=backend,**compress_opts)
        self.psi[1] = FermionAmplitudeFactory(psi[1],blks=blks,spinless=True,backend=backend,**compress_opts)
        self.psi[2] = JastrowAmplitudeFactory(psi[2],blks=blks,backend=backend,**compress_opts)

        self.nparam = [len(amp_fac.get_x()) for amp_fac in self.psi] 
        self.block_dict = self.psi[0].block_dict.copy()
        shift = 0
        for ix in (1,2):
            shift += self.nparam[ix-1]
            self.block_dict += [(start+shift,stop+shift) for start,stop in self.psi[ix].block_dict]

        self.Lx,self.Ly = self.psi[0].Lx,self.psi[0].Ly
        self.pbc = _PBC
        self.deterministic = _DETERMINISTIC
        self.backend = backend
        self.spinless = False
        if self.deterministic:
            self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2
##### wfn methods #####
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        for ix in range(3):
            self.psi[ix].wfn2backend(backend=backend,requires_grad=requires_grad)
    def get_x(self):
        return np.concatenate([amp_fac.get_x() for amp_fac in self.psi])
    def update(self,x,fname=None,root=0):
        x = np.split(x,[self.nparam[0],self.nparam[0]+self.nparam[1]])
        for ix in range(3):
            fname_ = None if fname is None else fname+f'_{ix}' 
            self.psi[ix].update(x[ix],fname=fname_,root=root)
##### sampler methods ######
    def _contract_cols(self,cols,js):
        return [af._contract_cols(cols_,js) for af,cols_ in zip(self.psi,cols)]
    def _get_plq_forward(self,j,y_bsz,cols,renvs):
        return [af._get_plq_forward(j,y_bsz,cols_,renvs_) for af,cols_,renvs_ in zip(self.psi,cols,renvs)]
    def _get_plq_backward(self,j,y_bsz,cols,lenvs):
        return [af._get_plq_backward(j,y_bsz,cols_,lenvs_) for af,cols_,lenvs_ in zip(self.psi,cols,lenvs)]
##### ham methods #####
    def parse_config(self,config):
        return parse_config(config)
    def config_sign(self,config):
        sign = np.ones(3)
        for ix in range(3):
            sign[ix] = self.psi[ix].config_sign(config[ix])
        return sign 
    def get_grad_from_plq(self,plq,cx,to_vec=True):
        vx = [self.psi[ix].get_grad_from_plq(plq[ix],cx[ix],to_vec=to_vec) for ix in range(3)]
        if to_vec:
            vx = np.concatenate(vx)
        return vx
    def _parse_hessian(self,Hvx,cx,ex,vx):
        na,nb,nj = self.nparam
        Hvx = np.split(Hvx,[na,na+nb])
        Hvx[2] = Hvx[2].reshape((2,nj))
        ls = [0.] * 3
        for ix in range(2):
            denom = cx[ix] * cx[2]
            ls[ix] += Hvx[ix] / denom + ex[1-ix] * vx[ix]
            ls[2] += Hvx[2][ix,:] / denom
        Hvx = np.concatenate(ls)
        vx = np.concatenate(vx) 
        return Hvx,vx
    def parse_hessian_from_plq(self,Hvx,vx,ex,eu,cx):
        if isinstance(vx,dict):
            vx = dict2list(vx)
        for ix in range(3):
            if isinstance(vx[ix],dict):
                vx[ix] = self.psi[ix].dict2vec(vx[ix]) 
        if isinstance(cx,dict):
            cx = dict2list(cx)
        cx,err = contraction_error(cx,multiply=False) 
        Hvx,vx = self._parse_hessian(Hvx,cx,ex,vx) 
        return np.prod(cx),np.sum(ex)+eu,vx,Hvx+eu*vx,err 
    def parse_hessian_deterministic(self,Hvx,vx,ex,eu,cx):
        na,nb,nj = self.nparam
        vx = np.split(vx,[na,na+nb])
        for ix in range(2):
            ex[ix] /= cx[ix] * cx[2]
        Hvx,vx = self._parse_hessian(Hvx,cx,ex,vx) 
        return np.prod(cx),np.sum(ex)+eu,vx,Hvx+eu*vx,0. 
    def parse_energy_deterministic(self,ex,cx,_sum=True):
        num = [None] * 2 
        exj = ex[2]
        cxj = cx[2]
        for ix in range(2):
            num_ix = [] 
            exf = ex[ix]
            for where in exf:
                num_ix.append(exf[where] * exj[where][ix])
            num[ix] = None if len(num_ix)==0 else sum(num_ix) / (cx[ix] * cxj)
        if _sum:
            num = sum(num)
            cx = np.prod(cx)
        return num,cx 
    def parse_energy_from_plq(self,ex,cx,_sum=True):
        ratio = [None] * 2
        exj = ex[2]
        cxj = cx[2]
        for ix in range(2):
            ratio_ix = [] 
            exf = ex[ix]
            cxf = cx[ix]
            for where in exf:
                try:
                    ratio_ix.append(exf[where] * exj[where][ix]/ (cxf[where] * cxj[where]))
                except:
                    print(exf[where],exj[where][ix],cxf[where],cxj[where]) 
                    exit()
            ratio[ix] = None if len(ratio_ix)==0 else sum(ratio_ix)
            if ratio[ix] is not None:
                ratio[ix] = tensor2backend(ratio[ix],'numpy')
        ratio = np.array(ratio) 
        if _sum:
            ratio = sum(ratio)
        return ratio
    def parse_derivative(self,ex,cx=None):
        ex_num,_ = self.parse_energy_deterministic(ex,np.ones(3),_sum=False)
        Hvxf = [np.zeros(self.nparam[ix]) for ix in range(2)]  
        Hvxb = [np.zeros(self.nparam[2])] * 2 
        for spin in range(2):
            if ex_num[spin] is None:
                ex_num[spin] = 0.
                continue
            ex_num[spin].backward(retain_graph=True)

            for ix in (spin,2):
                psi = self.psi[ix]
                Hvx = {site:psi.tensor_grad(psi.psi[psi.site_tag(site)].data) for site in psi.sites}
                Hvx = psi.dict2vec(Hvx)  
                if ix == 2:
                    Hvxb[spin] = Hvx
                else:
                    Hvxf[spin] = Hvx

            ex_num[spin] = tensor2backend(ex_num[spin],'numpy')
        if cx is None:
            ex = None
        elif isinstance(cx[2],dict):
            ex = self.parse_energy_from_plq(ex,cx,_sum=False)
        else:
            ex = self.parse_energy_deterministic(ex,cx,_sum=False)
        return ex,np.array(ex_num),np.concatenate(Hvxf + Hvxb)
    def get_grad_deterministic(self,config,unsigned=False):
        cx,vx = np.zeros(3),[None] * 3 
        for ix in range(3):
            cx[ix],vx[ix] = self.psi[ix].get_grad_deterministic(config[ix],unsigned=unsigned)
        vx = np.concatenate(vx)
        return cx,vx
    def _new_prob_from_plq(self,plq,sites,cis):
        plq_new = [None] * 3
        py = np.zeros(3)
        for ix in range(3):
            plq_new[ix],py[ix] = self.psi[ix]._new_prob_from_plq(plq[ix],sites,cis[ix])
        return plq_new,np.prod(py)
    def replace_sites(self,tn,sites,cis):
        return [af.replace_sites(tn_,sites,cis_) for af,tn_,cis_ in zip(self.psi,tn,cis)]
    def get_all_lenvs(self,cols,jmax=None,inplace=False):
        cols_new = [None] * 3
        lenvs = [None] * 3
        for ix in range(3):
            cols_new[ix],lenvs[ix] = self.psi[ix].get_all_lenvs(cols[ix],jmax=jmax,inplace=inplace)
        return cols_new,lenvs
    def get_all_renvs(self,cols,jmin=None,inplace=False):
        cols_new = [None] * 3
        renvs = [None] * 3
        for ix in range(3):
            cols_new[ix],renvs[ix] = self.psi[ix].get_all_renvs(cols[ix],jmin=jmin,inplace=inplace)
        return cols_new,renvs
    def _contract_tags(self,tn,tags):
        for ix in range(3):
            tn[ix] ^= tags
        return tn
    def get_all_bot_envs(self,config,psi=None,cache=None,imin=None,imax=None,append=''):
        env_prev = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_ = self.psi[ix].cache_bot if cache is None else cache[ix]
            env_prev[ix] = self.psi[ix].get_all_bot_envs(config[ix],psi=psi_,cache=cache_,imin=imin,imax=imax,append=append)
        return env_prev
    def get_all_top_envs(self,config,psi=None,cache=None,imin=None,imax=None,append=''):
        env_prev = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_ = self.psi[ix].cache_top if cache is None else cache[ix]
            env_prev[ix] = self.psi[ix].get_all_top_envs(config[ix],psi=psi_,cache=cache_,imin=imin,imax=imax,append=append)
        return env_prev
    def get_all_benvs(self,config,psi=None,cache_bot=None,cache_top=None,x_bsz=1,
                      compute_bot=True,compute_top=True,rix1=None,rix2=None):
        env_top = [None] * 3
        env_bot = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            env_bot[ix],env_top[ix] = self.psi[ix].get_all_benvs(config[ix],psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_,x_bsz=x_bsz,compute_bot=compute_bot,compute_top=compute_top,rix1=rix1,rix2=rix2)
        return env_bot,env_top
    def unsigned_amplitude(self,config,cache_bot=None,cache_top=None,to_numpy=True,multiply=True):
        cx = [None] * 3 
        for ix in range(3):
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            cx[ix] = self.psi[ix].unsigned_amplitude(config[ix],cache_bot=cache_bot_,cache_top=cache_top_,to_numpy=to_numpy)
        cx = np.prod(cx)
        return cx
    def build_3row_tn(self,config,i,x_bsz,psi=None,cache_bot=None,cache_top=None):
        tn = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            tn[ix] = self.psi[ix].build_3row_tn(config[ix],i,x_bsz,psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_) 
        return tn
    def get_plq_from_benvs(self,config,x_bsz,y_bsz,psi=None,cache_bot=None,cache_top=None,imin=0,imax=None):
        plq = [None] * 3
        for ix in range(3):
            psi_ = self.psi[ix].psi if psi is None else psi[ix]
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            plq[ix] = self.psi[ix].get_plq_from_benvs(config[ix],x_bsz,y_bsz,psi=psi_,cache_bot=cache_bot_,cache_top=cache_top_,imin=imin,imax=imax)
        return plq
    def _get_boundary_mps_deterministic(self,config,imin,imax,cache_bot=None,cache_top=None):
        bot = [None] * 3
        top = [None] * 3
        for ix in range(3):
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            bot[ix],top[ix] = self.psi[ix]._get_boundary_mps_deterministic(config[ix],imin,imax,cache_bot=cache_bot_,cache_top=cache_top_)
        return bot,top 
    def _unsigned_amplitude_deterministic(self,config,imin,imax,bot,top,cache_bot=None,cache_top=None):
        cy = np.zeros(3)
        for ix in range(3):
            cache_bot_ = self.psi[ix].cache_bot if cache_bot is None else cache_bot[ix]
            cache_top_ = self.psi[ix].cache_top if cache_top is None else cache_top[ix]
            cy[ix] = self.psi[ix]._unsigned_amplitude_deterministic(config[ix],imin,imax,bot[ix],top[ix],cache_bot=cache_bot_,cache_top=cache_top_) 
        return np.prod(cy)
    def update_cache(self,config):
        for ix in range(3):
            self.psi[ix].update_cache(config[ix])
from ..tensor_2d_vmc import Model as Model2D
from ..tensor_vmc import tensor2backend,contraction_error,safe_contract,dict2list
from ..tensor_2d_vmc import Hamiltonian as Hamiltonian_
class Hamiltonian(Hamiltonian_):
    def batch_pair_energies_from_plq(self,batch_idx,config): # only used for Hessian
        af = self.amplitude_factory 
        bix,tix,plq_types,pairs = self.model.batched_pairs[batch_idx]
        cache_bot = [dict(),dict(),dict()]
        cache_top = [dict(),dict(),dict()]
        af.get_all_bot_envs(config,cache=cache_bot,imax=bix)
        af.get_all_top_envs(config,cache=cache_top,imin=tix)

        # form plqs
        plq = [dict(),dict(),dict()]  
        for imin,imax,x_bsz,y_bsz in plq_types:
            plq_new = af.get_plq_from_benvs(config,x_bsz,y_bsz,cache_bot=cache_bot,cache_top=cache_top,imin=imin,imax=imax)
            for ix in range(3):
                plq[ix].update(plq_new[ix])

        # compute energy numerator 
        ex = [None] * 3
        cx = [None] * 3
        for ix in range(3):
            ex[ix],cx[ix] = self._pair_energies_from_plq(plq[ix],pairs,config[ix],af=af.psi[ix])
        return ex,cx,plq
    def pair_energies_from_plq(self,config): 
        af = self.amplitude_factory 
        x_bsz_min = min([x_bsz for x_bsz,_ in self.model.plq_sz])
        af.get_all_benvs(config,x_bsz=x_bsz_min)

        plq = [dict(),dict(),dict()]  
        for x_bsz,y_bsz in self.model.plq_sz:
            plq_new = af.get_plq_from_benvs(config,x_bsz,y_bsz)
            for ix in range(3):
                plq[ix].update(plq_new[ix])

        # compute energy numerator 
        ex = [None] * 3
        cx = [None] * 3
        for ix in range(3):
            ex[ix],cx[ix] = self._pair_energies_from_plq(plq[ix],self.model.pairs,config[ix],af=af.psi[ix])
        return ex,cx,plq
    def pair_energy_deterministic(self,config,site1,site2):
        af = self.amplitude_factory 
        ix1,ix2 = [self.model.flatten(*site) for site in (site1,site2)]
        i1,i2 = config[2][ix1],config[2][ix2]
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return None 

        cache_bot = [dict(),dict(),dict()]
        cache_top = [dict(),dict(),dict()]
        imin = min(site1[0],site2[0])
        imax = max(site1[0],site2[0])
        bot,top = af._get_boundary_mps_deterministic(config,imin,imax,cache_bot=cache_bot,cache_top=cache_top)

        ex = [None] * 3
        for ix in range(3):
            ex_ix = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,top[ix],bot[ix],self.model,cache_bot=cache_bot[ix],cache_top=cache_top[ix])
            ex[ix] = {(site1,site2):ex_ix}
        return ex
    def batch_pair_energies_deterministic(self,config,batch_imin,batch_imax):
        af = self.amplitude_factory
        cache_bot = [dict(),dict(),dict()]
        cache_top = [dict(),dict(),dict()]
        bot,top = af._get_boundary_mps_deterministic(config,batch_imin,batch_imax,cache_bot=cache_bot,cache_top=cache_top)

        ex = [None] * 3
        for ix in range(3):
            ex_ix = dict() 
            for site1,site2 in self.model.batched_pairs[batch_imin,batch_imax]:
                eij = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,top[ix],bot[ix],self.model,cache_bot=cache_bot[ix],cache_top=cache_top[ix])
                if eij is not None:
                    ex_ix[site1,site2] = eij
            ex[ix] = ex_ix
        return ex
    def pair_energies_deterministic(self,config):
        af = self.amplitude_factory
        af.get_all_benvs(config)

        ex = [None] * 3
        for ix in range(3):
            ex_ix = dict() 
            for (site1,site2) in self.model.pairs:
                imin = min(af.rix1+1,site1[0],site2[0]) 
                imax = max(af.rix2-1,site1[0],site2[0]) 
                bot = af.psi[ix]._get_bot(imin,config[ix])  
                top = af.psi[ix]._get_top(imax,config[ix])  

                eij = af.psi[ix].pair_energy_deterministic(config[ix],site1,site2,top,bot,self.model)
                if eij is not None:
                    ex_ix[site1,site2] = eij
            ex[ix] = ex_ix
        return ex
from ..tensor_2d import PEPS
def get_gutzwiller(Lx,Ly,coeffs,bdim=1,eps=0.,normalize=True):
    if isinstance(coeffs,np.ndarray):
        assert len(coeffs)==4
        coeffs = {(i,j):coeffs for i,j in itertools.product(range(Lx),range(Ly))}
    arrays = []
    for i in range(Lx):
        row = []
        for j in range(Ly):
            shape = [bdim] * 4 
            if i==0 or i==Lx-1:
                shape.pop()
            if j==0 or j==Ly-1:
                shape.pop()
            shape = tuple(shape) + (4,)

            data = np.ones(shape)
            for ix in range(4):
                data[(0,)*(len(shape)-1)+(ix,)] = coeffs[i,j][ix] 
            data += eps * np.random.rand(*shape)
            if normalize:
                data /= np.linalg.norm(data)
            row.append(data)
        arrays.append(row)
    return PEPS(arrays)
