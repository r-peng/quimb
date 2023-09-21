import numpy as np
#####################################################
# for separate ansatz
#####################################################
from ..tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    AmplitudeFactory,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
def _contraction_error(cx,multiply=True):
    n = len(cx)
    cx_,err = np.zeros(n),np.zeros(n)
    for ix in range(n): 
        cx_[ix],err[ix] = contraction_error(cx[ix])
    if multiply:
        cx_ = np.prod(cx_)
    return cx_,np.amax(err)
def config_to_ab(config):
    config = np.array(config)
    return tuple(config % 2), tuple(config // 2)
def config_from_ab(config_a,config_b):
    return tuple(np.array(config_a) + np.array(config_b) * 2)
class ProductAmplitudeFactory:
    def wfn2backend(self,backend=None,requires_grad=False):
        backend = self.backend if backend is None else backend
        for af in self.af:
            af.wfn2backend(backend=backend,requires_grad=requires_grad)
    def get_x(self):
        return np.concatenate([af.get_x() for af in self.af])
    def get_sections(self):
        self.naf = len(self.af)
        self.nparam = np.array([af.nparam for af in self.af])
        self.sections = np.cumsum(self.nparam)[:-1]

        self.block_dict = self.af[0].block_dict.copy()
        for af,shift in zip(self.af[1:],self.sections):
            self.block_dict += [(start+shift,stop+shift) for start,stop in af.block_dict]
    def update(self,x,fname=None,root=0):
        x = np.split(x,self.sections)
        for ix,af in enumerate(self.af):
            fname_ = None if fname is None else fname+f'_{ix}' 
            af.update(x[ix],fname=fname_,root=root)
    def get_grad_deterministic(self,config,unsigned=False):
        cx = [None] * self.naf
        vx = [None] * self.naf 
        for ix,af in enumerate(self.af):
            cx[ix],vx[ix] = af.get_grad_deterministic(config[ix],unsigned=unsigned)
        return np.array(cx),np.concatenate(vx)
    def _new_prob_from_plq(self,plq,sites,cis):
        py = [None] * 3
        plq_new = [None] * 3 
        for af,plq_,cis_ in zip(self.af,plq,cis):
            try:
                plq_new_,py_ = af._new_prob_from_plq(plq_,sites,cis_)
            
            if py_ is not None:
                py.append(py_)
                plq_new.append(plq_new_)
            else:
                py.append(0.)
                plq_new.append(None)
        return plq_new,np.prod(np.array(py))
    def replace_sites(self,tn,sites,cis):
        for 
        return [af.replace_sites(tn_,sites,cis_) for af,tn_,cis_ in zip(self.af,tn,cis)]
##### ham methods #####
    def config_sign(self,config):
        sign = [] 
        for af,config_ in zip(self.af,config):
            sign.append(af.config_sign(config_))
        return np.array(sign) 
    def get_grad_from_plq(self,plq,to_vec=True):
        vx = [af.get_grad_from_plq(plq_,to_vec=to_vec) for af,plq_ in zip(self.af,plq)]
        if to_vec:
            vx = np.concatenate(vx)
        return vx
    def compute_hessian(self,ex_num):
        Hvx = [] 
        for af in self.af:
            Hvx_ = {site:af.tensor_grad(af.psi[af.site_tag(site)].data) for site in af.sites}
            Hvx.append(af.dict2vec(Hvx_)) 
        return np.concatenate(Hvx) 

    def parse_energy_numerator(self,ex):
        keys = set()
        for ex_ in ex:
            keys.union(set(ex_.keys()))
        enum = 0.
        for key in keys:
            term = 1.
            for ex_ in ex:
                if key in ex_:
                    ex_term = ex_.pop(key) 
                    term *= ex_term 
            enum += term
        return enum
    def parse_energy_ratio(self,ex,cx):
        keys = set()
        for ex_ in ex:
            keys.union(set(ex_.keys()))
        eloc = 0.
        for key in keys:
            term = 1.
            for ex_,cx_ in zip(ex,cx):
                if key in ex_:
                    ex_term = ex_.pop(key) 
                    term *= ex_term / cx_[key[0]]
            eloc += term
        return tensor2backend(eloc,'numpy')
    def batch_hessian_from_plq(self,batch_key): # only used for Hessian
        ex = [None] * self.naf
        cx = [None] * self.naf
        vx = [None] * self.naf 
        for ix,af in enumerate(self.af):
            af.wfn2backend(backend='torch',requires_grad=True)
            af.model.gate2backend('torch')
            ex[ix],cx[ix],vx[ix] = af.batch_pair_energies_from_plq(batch_key,new_cache=True)

        ex_num = self.parse_energy_numerator(ex,_sum=False)
        Hvx = af.compute_hessian(ex_num)
        ex = self.parse_energy_ratio(ex,cx,_sum=False)
        af.wfn2backend()
        self.model.gate2backend(af.backend)
        return ex,Hvx,cx,vx
    def compute_local_energy_hessian_from_plq(self): 
        cx = [dict() for _ in range(self.naf)]
        vx = [dict() for _ in range(self.naf)]
        ex,Hvx = 0.,0.
        for batch_key in self.model.batched_pairs:
            ex_,Hvx_,cx_,vx_ = self.batch_hessian_from_plq(config,batch_key)  
            ex += ex_
            Hvx += Hvx_
            for ix in range(af.naf):
                cx[ix].update(cx_[ix])
                vx[ix].update(vx_[ix])
        eu = self.model.compute_local_energy_eigen(config)
        cx,err = _contraction_error(cx,multiply=False) 

        vx = np.concatenate([af.dict2vec(vx_) for af,vx_ in zip(self.af,vx)])

        ex = np.sum(ex) + eu
        Hvx = Hvx / cx + eu * vx
        cx = np.prod(cx)
        return cx,ex,vx,Hvx,err 
    def compute_local_energy_gradient_from_plq(self,compute_v=True):
        ex = [dict() for _ in range(self.naf)]
        cx = [dict() for _ in range(self.naf)]
        vx = [dict() for _ in range(self.naf)]
        for batch_key in self.model.batched_pairs:
            for ix,af in enumerate(self,af):
                ex_,cx_,vx_ = af.batch_pair_energies_from_plq(batch_key,compute_v=compute_v)  
                ex[ix].update(ex_)
                cx[ix].update(cx_)
                vx[ix].update(vx_)

        ex = self.parse_energy_ratio(ex,cx)
        eu = self.model.compute_local_energy_eigen(config)
        ex += eu

        cx,err = _contraction_error(cx)
        if not compute_v:
            return cx,ex,None,None,err 
        af = self.amplitude_factory
        vx = np.concatenate([af_.dict2vec(vx_) for af_,vx_ in zip(af.af,vx)])
        return cx,ex,vx,None,err
    #def batch_hessian_deterministic(self,config,batch_key):
    #    af = self.amplitude_factory
    #    af.wfn2backend(backend='torch',requires_grad=True)
    #    ex = self.batch_pair_energies_deterministic(config,batch_key,new_cache=True)
    #    _,ex,Hvx = af.compute_hessian(ex)
    #    af.wfn2backend()
    #    return ex,Hvx
    #def compute_local_energy_hessian_deterministic(self,config):
    #    af = self.amplitude_factory
    #    cx,vx = af.get_grad_deterministic(config)

    #    ex = 0. 
    #    Hvx = 0.
    #    for key in self.model.batched_pairs:
    #        ex_,Hvx_ = self.batch_hessian_deterministic(config,key) 
    #        ex += ex_
    #        Hvx += Hvx_
    #     
    #    eu = self.model.compute_local_energy_eigen(config)
    #    return af.parse_hessian_deterministic(Hvx,vx,ex,eu,cx)
    #def compute_local_energy_gradient_deterministic(self,config,compute_v=True):
    #    af = self.amplitude_factory
    #    ex = dict() 
    #    for key in self.model.batched_pairs:
    #        ex_ = self.batch_pair_energies_deterministic(config,key)
    #        ex.update(ex_)

    #    if compute_v:
    #        cx,vx = af.get_grad_deterministic(config)
    #    else:
    #        cx = af.unsigned_amplitude(config)
    #        sign = af.config_sign(config)
    #        cx *= sign
    #        vx = None
    #    if cx is None:
    #        return 0.,0.,vx,None,0.
    #    ex,cx = af.parse_energy_deterministic(ex,cx) 
    #    eu = self.model.compute_local_energy_eigen(config)
    #    ex += eu
    #    return cx,ex,vx,None,0.
#######################################################################
# some jastrow forms
#######################################################################
class TNJastrow(AmplitudeFactory):
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
    def update_pair_energy_from_plq(self,tn,where,ex):
        ix1,ix2 = [self.flatten(site) for site in where]
        i1,i2 = self.config[ix1],self.config[ix2] 
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return None 
        for spin in ('a','b'):
            i1_new,i2_new = self.pair_terms(i1,i2,spin)
            if i1_new is None:
                continue
            tn_new = self.replace_sites(tn.copy(),where,(i1_new,i2_new))
            ex_ij = safe_contract(tn_new)
            if ex_ij is not None:
                ex[where,spin] = ex_ij 
        return ex 
    def config_sign(self,config_a,config_b):
        return 1
