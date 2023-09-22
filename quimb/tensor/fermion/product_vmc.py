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
def config_to_ab(config):
    config = np.array(config)
    return tuple(config % 2), tuple(config // 2)
def config_from_ab(config_a,config_b):
    return tuple(np.array(config_a) + np.array(config_b) * 2)
class ProductAmplitudeFactory:
    def parse_config(self,config):
        ca,cb = config_to_ab(config)
        return [{'a':ca,'b':cb,None:config}[af.spin] for af in self.af]
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
        py = [None] * self.naf 
        plq_new = [None] * self.naf
        cis = self.parse_config(cis)
        config_new = self.parse_config(self.config_new)
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq_new[ix],py[ix] = af._new_prob_from_plq(plq[ix],sites,cis[ix])
            else:
                py[ix] = af.prob(config_new[ix])
            if py[ix] is None:
                return plq_new,0. 
        return plq_new,np.prod(np.array(py))
    def prob(self,config):
        try:
            p = np.array([af.prob(config[ix]) for ix,af in enumerate(self.af)])
            return np.prod(p)
        except ValueError:
            return 0.
##### ham methods #####
    def config_sign(self,config):
        sign = [] 
        for af,config_ in zip(self.af,config):
            sign.append(af.config_sign(config_))
        return np.array(sign) 
    def propagate(self,ex_num):
        ex_num.backward()
        Hvx = [af.extract_grad() for af in self.af] 
        return np.concatenate(Hvx) 
    def parse_energy_numerator(self,ex,cx):
        keys = set()
        for ex_ in ex:
            keys.update(set(ex_.keys()))
        enum = 0.
        for key in keys:
            term = 1.
            for ix,ex_ in enumerate(ex):
                if key in ex_:
                    term *= ex_[key] 
                else:
                    term *= cx[ix][key[0]][0]
            enum += term
        return enum
    def batch_quantities_from_plq(self,batch_key,compute_v,compute_Hv): # only used for Hessian
        ex_num = [None] * self.naf
        cx = [None] * self.naf
        plq = [None] * self.naf
        for ix,af in enumerate(self.af):
            if compute_Hv:
                af.wfn2backend(backend='torch',requires_grad=True)
                af.model.gate2backend('torch')
            ex_num[ix],cx[ix],plq[ix] = af.batch_pair_energies_from_plq(batch_key,new_cache=compute_Hv)

        if compute_Hv:
            ex = self.parse_energy_numerator(ex_num,cx)
            Hvx = self.propagate(ex)
        else:
            Hvx = 0.

        keys = set()
        for eix in ex_num:
            keys.update(set(eix.keys()))
        ex = 0.
        for (where,spin) in keys:
            term = 1.
            for ix,eix in enumerate(ex_num):
                cij,plq_key = cx[ix][where]
                if plq_key in self.af[ix].cx:
                    cij = self.af[ix].cx[plq_key]
                else:
                    cij = tensor2backend(cij,'numpy')
                    self.af[ix].cx[plq_key] = cij
                if (where,spin) in eix:
                    term *= eix[where,spin] / cij 
            ex += term
        ex = tensor2backend(ex,'numpy')
        if compute_v: 
            for ix,af in enumerate(self.af):
                af.get_grad_from_plq(plq[ix])
        if compute_Hv: 
            self.wfn2backend()
            self.model.gate2backend(self.backend)
        return ex,Hvx
    def contraction_error(self,multiply=True):
        cx,err = np.zeros(self.naf),np.zeros(self.naf)
        for ix,af in enumerate(self.af): 
            cx[ix],err[ix] = contraction_error(af.cx)
        if multiply:
            cx = np.prod(cx)
        return cx,np.amax(err)
    def compute_local_quantities_from_plq(self,compute_v,compute_Hv): 
        for af in self.af:
            af.cx = dict()
            if af.is_tn:
                af.vx = dict()
        ex,Hvx = 0.,0.
        for batch_key in self.model.batched_pairs:
            ex_,Hvx_ = self.batch_quantities_from_plq(batch_key,compute_v,compute_Hv)  
            ex += ex_
            Hvx += Hvx_
        cx,err = self.contraction_error() 
        eu = self.model.compute_local_energy_eigen(self.config)
        ex += eu

        if compute_v:
            vx = [None] * self.naf
            for ix,af in enumerate(self.af):
                if af.is_tn:
                    vx[ix] = af.dict2vec(af.vx)
                else:
                    vx[ix] = af.vx
            vx = np.concatenate(vx)
        else:
            vx = None

        if compute_Hv:
            Hvx = Hvx / cx + eu * vx
        else:
            Hvx = None
        return cx,ex,vx,Hvx,err 
    def compute_local_energy(self,config,compute_v=True,compute_Hv=False):
        self.config = config 
        for af,config_ in zip(self.af,self.parse_config(config)):
            af.config = config_ 
        return self.compute_local_quantities_from_plq(compute_v,compute_Hv)
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
            return ex 
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
