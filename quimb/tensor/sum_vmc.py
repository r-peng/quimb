import numpy as np
import itertools
import scipy
from .product_vmc import (
    tensor2backend,
    safe_contract,
    contraction_error,
    CompoundAmplitudeFactory,
)
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
class SumAmplitudeFactory(CompoundAmplitudeFactory):
    def __init__(self,af,fermion=False):
        self.af = af 
        self.get_sections()

        self.Lx,self.Ly = self.af[0].Lx,self.af[0].Ly
        self.sites = self.af[0].sites
        self.model = self.af[0].model
        self.nsite = self.af[0].nsite
        self.backend = self.af[0].backend

        self.pbc = self.af[0].pbc 
        self.deterministic = self.af[0].deterministic 
        #self.rix1,self.rix2 = (self.Lx-1) // 2, (self.Lx+1) // 2

        self.fermion = fermion 
        if self.fermion:
            self.spinless = self.af[0].spinless

        self.flatten = self.af[0].flatten
        self.flat2site = self.af[0].flat2site
        self.intermediate_sign = self.af[0].intermediate_sign
    def get_grad_deterministic(self,config,unsigned=False):
        config = self.parse_config(config)
        cx = [0] * self.naf 
        vx = [None] * self.naf
        for ix,af in enumerate(self.af):
            af.wfn2backend(backend='torch',requires_grad=True)
            if af.is_tn:
                cache_bot = dict() 
                cache_top = dict() 
                cx[ix] = af.unsigned_amplitude(config[ix],cache_bot=cache_bot,cache_top=cache_top)
                cx[ix] *= af.config_sign(config_new[ix])
            else:
                cx[ix] = af.amp(config_new[ix])
            cx[ix],vx[ix] = self.propagate(cx[ix])
        cx = sum(cx)
        return cx,np.concatenate(vx)/cx 
    def unsigned_amplitude(self,config):
        cy = [0] * self.naf 
        for ix,af in enumerate(self.af):
            if af.is_tn:
                cy_ix = af.unsigned_amplitude(config[ix])
                if cy_ix is None:
                    continue
                cy[ix] = cy_ix * af.config_sign(config[ix])
            else:
                cy[ix] = af.amp(config[ix])
        return sum(cy)
    def _new_amp_from_plq(self,plq,sites,config_sites,config_new):
        cy = [0] * self.naf 
        plq_new = [None] * self.naf
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq_new[ix],cy_ix = af._new_amp_from_plq(plq[ix],sites,config_sites[ix])
                if cy_ix is None:
                    continue
                cy[ix] = cy_ix * af.config_sign(config_new[ix])
            else:
                cy[ix] = af.amp(config_new[ix])
        return plq_new,sum(cy)
    def update_pair_energy_from_plq(self,site1,site2,plq):
        ix1,ix2 = [self.flatten(site) for site in (site1,site2)]
        assert ix1<ix2
        i1,i2 = self.config[ix1],self.config[ix2]
        if not self.model.pair_valid(i1,i2): # term vanishes 
            return 0 
        pair_plq = [None] * self.naf 
        for ix,af in enumerate(self.af):
            if not af.is_tn:
                continue
            key = self.model.pair_key(site1,site2)
            if key not in plq[ix]:
                continue
            tn = plq[ix][key].copy()
            b1 = tn[self.site_tag(site1),'BRA']
            b2 = tn[self.site_tag(site2),'BRA'] 
            tid1 = b1.get_fermion_info()[0]
            tid2 = b2.get_fermion_info()[0]
            fs = tn.fermion_space
            site = len(fs.tensor_order)-1
            fs.move(tid1,site)
            fs.move(tid2,site-1)
            fs._refactor_phase_from_tids([tid1,tid2])
            pair_plq[ix] = tn 
        config_sites = (None,) * self.naf
        config = [af.config for af in self.af]
        self.cx[site1,site2] = tensor2backend(self._new_amp_from_plq(pair_plq,None,config_sites,config)[1],'numpy')

        coeff_comm = self.intermediate_sign(self.config,ix1,ix2) * self.model.pair_coeff(site1,site2)
        ex = 0 
        for i1_new,i2_new,coeff,tag in self.model.pair_terms(i1,i2):
            config_new = list(self.config)
            config_new[ix1] = i1_new
            config_new[ix2] = i2_new 
            config_sites = self.parse_config((i1_new,i2_new))
            config_new = self.parse_config(tuple(config_new)) 
            _,cx = self._new_amp_from_plq(pair_plq,(site1,site2),config_sites,config_new)
            if RANK==18:
                print(site1,site2,tag,cx*coeff*coeff_comm)
            ex += coeff * cx 
        return ex * coeff_comm 
    def batch_quantities_from_plq(self,batch_key,compute_v,compute_Hv):
        pairs,plq_types,bix,tix,direction = self.model.batched_pairs[batch_key]
        if compute_Hv:
            for ix,af in enumerate(self.af):
                af.wfn2backend(backend='torch',requires_grad=True)
                af.model.gate2backend('torch')

        plq = [None] * self.naf
        for ix,af in enumerate(self.af):
            if af.is_tn:
                plq[ix],pairs = af.batch_plq_obc(batch_key,compute_Hv)
        
        ex = 0 
        # numerator
        for site1,site2 in pairs:
            ex += self.update_pair_energy_from_plq(site1,site2,plq)
        exit()
        Hvx = self.propagate(ex_num) if compute_Hv else 0
        ex = tensor2backend(ex,'numpy')
        if compute_v:
            for ix,af in enumerate(self.af):
                if af.is_tn:
                    af.get_grad_from_plq(plq[ix],ratio=False) 
        if compute_Hv:
            self.wfn2backend()
            self.model.gate2backend(self.backend)
        return ex,Hvx
    def compute_local_quantities_from_plq(self,compute_v,compute_Hv):
        self.cx = dict()
        for af in self.af:
            if af.is_tn:
                af.vx = dict()
        #if compute_v:
        #    cx,vx = self.get_grad_deterministic(self.config)
        #else:
        #    cx = self.amplitude(self.config)
        #    vx = None

        ex,Hvx = 0.,0.
        for batch_key in self.model.batched_pairs:
            ex_,Hvx_ = self.batch_quantities_from_plq(batch_key,compute_v,compute_Hv)  
            ex += ex_
            Hvx += Hvx_

        cx,err = contraction_error(self.cx) 
        ex /= cx
        print(f'{RANK},{self.config},{ex},{cx}')
        exit()
        eu = self.model.compute_local_energy_eigen(self.config)
        ex += eu

        if compute_v:
            vx = [None] * self.naf
            for ix,af in enumerate(self.af):
                if af.is_tn:
                    vx[ix] = af.dict2vec(af.vx)
                else:
                    af.wfn2backend(backend='torch',requires_grad=True)
                    config = af._input(af.config)
                    cx_ix = af.forward(config) 
                    _,vx[ix] = af.propagate(cx_ix) 
                af.vx = None
            vx = np.concatenate(vx) / cx
        else:
            vx = None
        if compute_Hv: 
            Hvx = Hvx / cx + eu * vx
        else:
            Hvx = None 
        return cx,ex,vx,Hvx,0. 
