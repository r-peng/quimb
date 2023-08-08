import numpy as np
#####################################################
# for separate ansatz
#####################################################
from ..tensor_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    AmplitudeFactory,
    Hamiltonian,
)
def _contraction_error(cx,multiply=True):
    cx_,err = np.zeros(3),np.zeros(3)
    for ix in range(3): 
        cx_[ix],err[ix] = contraction_error(cx[ix])
    if multiply:
        cx_ = np.prod(cx_)
    return cx_,np.amax(err)
class JastrowAmplitudeFactory(AmplitudeFactory):
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
class ProductAmplitudeFactory:
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

##### ham methods #####
    def parse_config(self,config):
        return parse_config(config)
    def config_sign(self,config):
        sign = np.ones(3)
        for ix in range(3):
            sign[ix] = self.psi[ix].config_sign(config[ix])
        return sign 
    def get_grad_from_plq(self,plq,to_vec=True):
        vx = [self.psi[ix].get_grad_from_plq(plq[ix],to_vec=to_vec) for ix in range(3)]
        if to_vec:
            vx = np.concatenate(vx)
        return vx
    #def parse_hessian_from_plq(self,Hvx,vx,ex,eu,cx):
    #    if isinstance(vx,dict):
    #        vx = dict2list(vx)
    #    if isinstance(cx,dict):
    #        cx = dict2list(cx)
    #    cx,err = contraction_error(cx,multiply=False) 
    #    Hvx,vx = self._parse_hessian(Hvx,cx,ex,vx) 
    #    return np.prod(cx),np.sum(ex)+eu,vx,Hvx+eu*vx,err 
    #def parse_hessian_deterministic(self,Hvx,vx,ex,eu,cx):
    #    na,nb,nj = self.nparam
    #    vx = np.split(vx,[na,na+nb])
    #    for ix in range(2):
    #        ex[ix] /= cx[ix] * cx[2]
    #    Hvx,vx = self._parse_hessian(Hvx,cx,ex,vx) 
    #    return np.prod(cx),np.sum(ex)+eu,vx,Hvx+eu*vx,0. 
    def compute_hessian(self,ex_num):
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
        return np.array(ex_num),np.concatenate(Hvxf + Hvxb)
    def parse_hessian(self,Hvx,cx,ex,vx):
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
class ProductHamiltonian(Hamiltonian):
    def parse_energy_numerator(self,ex,_sum=True):
        num = [None] * 2 
        exj = ex[2]
        for ix in range(2):
            num_ix = [] 
            exf = ex[ix]
            for where in exf:
                num_ix.append(exf[where] * exj[where][ix])
            num[ix] = None if len(num_ix)==0 else sum(num_ix) 
        if _sum:
            num = sum(num)
        return num 
    def parse_energy_ratio(self,ex,cx,_sum=True):
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
                except TypeError:
                    pass
            ratio[ix] = 0. if len(ratio_ix)==0 else tensor2backend(sum(ratio_ix),'numpy')
        ratio = np.array(ratio) 
        if _sum:
            ratio = sum(ratio)
        return ratio
    def batch_hessian_from_plq(self,batch_key,config): # only used for Hessian
        af = self.amplitude_factory
        af.wfn2backend(backend='torch',requires_grad=True)
        self.model.gate2backend('torch')
        ex,cx,vx = self.batch_pair_energies_from_plq(batch_key,config,new_cache=True)

        ex_num = self.parse_energy_numerator(ex,_sum=False)
        _,Hvx = af.compute_hessian(ex_num)
        ex = self.parse_energy_ratio(ex,cx,_sum=False)
        af.wfn2backend()
        self.model.gate2backend(af.backend)
        return ex,Hvx,cx,vx
    def compute_local_energy_hessian_from_plq(self,config): 
        ex,Hvx = 0.,0.
        cx = [dict(),dict(),dict()]
        vx = [dict(),dict(),dict()]
        for batch_key in self.model.batched_pairs:
            ex_,Hvx_,cx_,vx_ = self.batch_hessian_from_plq(batch_key,config)  
            ex += ex_
            Hvx += Hvx_
            for ix in range(3):
                cx[ix].update(cx_[ix])
                vx[ix].update(vx_[ix])
        eu = self.model.compute_local_energy_eigen(config)
        cx,err = _contraction_error(cx,multiply=False) 

        af = self.amplitude_factory
        vx = [psi.dict2vec(vx_ix) for psi,vx_ix in zip(af.psi,vx)]

        Hvx,vx = af.parse_hessian(Hvx,cx,ex,vx)
        ex = np.sum(ex) + eu
        Hvx += eu * vx
        cx = np.prod(cx)
        return cx,ex,vx,Hvx,err 
    def compute_local_energy_gradient_from_plq(self,config,compute_v=True):
        ex = [dict(),dict(),dict()]
        cx = [dict(),dict(),dict()]
        vx = [dict(),dict(),dict()]
        for batch_key in self.model.batched_pairs:
            ex_,cx_,vx_ = self.batch_pair_energies_from_plq(batch_key,config,compute_v=compute_v)  
            for ix in range(3):
                ex[ix].update(ex_[ix])
                cx[ix].update(cx_[ix])
                vx[ix].update(vx_[ix])

        ex = self.parse_energy_ratio(ex,cx)
        eu = self.model.compute_local_energy_eigen(config)
        ex += eu

        cx,err = _contraction_error(cx)
        if not compute_v:
            return cx,ex,None,None,err 
        af = self.amplitude_factory
        vx = np.concatenate([psi.dict2vec(vx_ix) for psi,vx_ix in zip(af.psi,vx)])
        return cx,ex,vx,None,err
    def batch_hessian_deterministic(self,config,batch_key):
        af = self.amplitude_factory
        af.wfn2backend(backend='torch',requires_grad=True)
        ex = self.batch_pair_energies_deterministic(config,batch_key,new_cache=True)
        _,ex,Hvx = af.compute_hessian(ex)
        af.wfn2backend()
        return ex,Hvx
    def compute_local_energy_hessian_deterministic(self,config):
        af = self.amplitude_factory
        cx,vx = af.get_grad_deterministic(config)

        ex = 0. 
        Hvx = 0.
        for key in self.model.batched_pairs:
            ex_,Hvx_ = self.batch_hessian_deterministic(config,key) 
            ex += ex_
            Hvx += Hvx_
         
        eu = self.model.compute_local_energy_eigen(config)
        return af.parse_hessian_deterministic(Hvx,vx,ex,eu,cx)
    def compute_local_energy_gradient_deterministic(self,config,compute_v=True):
        af = self.amplitude_factory
        ex = dict() 
        for key in self.model.batched_pairs:
            ex_ = self.batch_pair_energies_deterministic(config,key)
            ex.update(ex_)

        if compute_v:
            cx,vx = af.get_grad_deterministic(config)
        else:
            cx = af.unsigned_amplitude(config)
            sign = af.config_sign(config)
            cx *= sign
            vx = None
        if cx is None:
            return 0.,0.,vx,None,0.
        ex,cx = af.parse_energy_deterministic(ex,cx) 
        eu = self.model.compute_local_energy_eigen(config)
        ex += eu
        return cx,ex,vx,None,0.
