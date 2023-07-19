import time,itertools
import numpy as np
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
np.set_printoptions(suppress=True,precision=4,linewidth=2000)

import autoray as ar
import torch
torch.autograd.set_detect_anomaly(False)

from .torch_utils import SVD,QR,SVDforward
ar.register_function('torch','linalg.svd',SVD.apply)
ar.register_function('torch','linalg.qr',QR.apply)

from .vmc_tmp import (
    SpinAmplitudeFactory,
    safe_contract,
    tsr2backend,
    tsr_grad,
    contraction_error,
    write_tn_to_disc,
)
from .tensor_core import (
    _parse_split_opts,
    group_inds,
    rand_uuid,
    tags_to_oset,
    oset,
    prod,
    Tensor,
)
from .decomp import svd,reshape
from .tensor_2d_vmc_tmp import AmplitudeFactory as AmplitudeFactory_ 
from .tensor_2d_vmc_tmp import set_options as set_options_ 
import sys
this = sys.modules[__name__]
def set_options(pbc=False):
    this._PBC = pbc
    set_options_(pbc=pbc,deterministic=True,backend='torch')
class AmplitudeFactory(AmplitudeFactory_):
    def __init__(self,psi,blks=None,phys_dim=2,**compress_opts):
        self.gauge = dict()
        super().__init__(psi,blks=blks,phys_dim=phys_dim,**compress_opts)
    def init_gauge(self,rand=False):
        chi = self.compress_opts['max_bond']
        if rand:
            data = np.random.rand(chi,chi)
        else:
            data = np.zeros((chi,)*2)
        if _PBC:
            raise NotImplementedError
        else:
            self.gauge = {((i,j),(i,j+1)):tsr2backend(data,'torch') for (i,j) in itertools.product(range(1,self.Lx-1),range(self.Ly-1))}
        self.get_gauge_info()
    def gauge_sval(self,s,where):
        sh = len(s)
        return self.gauge[where][:sh,:sh] + torch.diag(s) 
    def get_gauge_info(self):
        self.gauge_order = list(self.gauge.keys())
        self.gauge_map = dict() 
        start = 0
        for ix,where in enumerate(self.gauge_order):
            shape = self.gauge[where].shape
            size = np.prod(np.array(shape))
            stop = start + size
            self.gauge_map[where] = shape,start,stop
            start = stop
        self.gauge_size = stop
        if RANK==0:
            print('gauge_size=',self.gauge_size)
    def gauge2vec(self,gauge=None):
        gauge = self.gauge if gauge is None else gauge
        v = np.zeros(self.gauge_size)  
        for where,(_,start,stop) in self.gauge_map.items():
            g = gauge[where]
            if g is not None:
                v[start:stop] = tsr2backend(g,'numpy').reshape(-1)
        return v 
    def vec2gauge(self,x):
        gauge = dict()
        for where,(shape,start,stop) in self.gauge_map.items():
            gauge[where] = tsr2backend(x[start:stop].reshape(shape),'torch')
        return gauge 
    def get_x(self):
        x1 = self.psi2vec()
        x2 = self.gauge2vec()
        return np.concatenate([x1,x2])
    def update(self,x,fname=None,root=0):
        x1,x2 = np.split(x,[len(x)-self.gauge_size])
        psi = self.vec2psi(x1,inplace=True)
        self.set_psi(psi) 
        self.gauge = self.vec2gauge(x2)
        if RANK==root:
            if fname is not None: # save psi to disc
                write_tn_to_disc(psi,fname,provided_filename=True)
        return psi
    def wfn2backend(self,backend=None,requires_grad=False):
        super().wfn2backend(backend='torch',requires_grad=requires_grad)
        for where in self.gauge:
            self.gauge[where] = tsr2backend(self.gauge[where],'torch',requires_grad=requires_grad)
    def parse_hessian(self,ex,cx=None):
        if len(ex)==0:
            return 0.,0.
        ex_num = sum(ex.values())
        ex_num.backward()

        Hvx1 = {(i,j):tsr_grad(self.psi[i,j].data) for i,j in itertools.product(range(self.Lx),range(self.Ly))}
        Hvx1 = self.dict2vec(Hvx1)

        Hvx2 = {where:tsr_grad(tsr) for where,tsr in self.gauge.items()}
        Hvx2 = self.gauge2vec(Hvx2)
        assert cx is None
        return None,tsr2backend(ex_num,'numpy'),np.concatenate([Hvx1,Hvx2]) 
    def tensor_svd(self,
        T,
        where,
        left_inds,
        absorb=None,
        max_bond=None,
        cutoff=1e-10,
        cutoff_mode='rel',
        renorm=None,
        ltags=None,
        rtags=None,
        stags=None,
        bond_ind=None,
        right_inds=None,
        **kwargs
    ):
    
        if left_inds is None:
            left_inds = oset(T.inds) - oset(right_inds)
        else:
            left_inds = tags_to_oset(left_inds)
    
        if right_inds is None:
            right_inds = oset(T.inds) - oset(left_inds)
    
        TT = T.transpose(*left_inds, *right_inds)
        left_dims = TT.shape[:len(left_inds)]
        right_dims = TT.shape[len(left_inds):]
        array = reshape(TT.data, (prod(left_dims), prod(right_dims)))
    
        opts = _parse_split_opts(
            'svd', cutoff, None, max_bond, cutoff_mode, renorm)
    
        # ``s`` itself will be None unless ``absorb=None`` is specified
        left, s, right = svd(array, **opts)
        s_ = self.gauge_sval(s,where)
        if absorb == 'left':
            left = torch.mm(left,s_)
        elif absorb == 'right':
            right = torch.mm(s_,right)
        else:
            raise NotImplementedError

        left = reshape(left, (*left_dims, -1))
        right = reshape(right, (-1, *right_dims))
    
        bond_ind = rand_uuid() if bond_ind is None else bond_ind
        ltags = T.tags | tags_to_oset(ltags)
        rtags = T.tags | tags_to_oset(rtags)
    
        Tl = Tensor(data=left, inds=(*left_inds, bond_ind), tags=ltags)
        Tr = Tensor(data=right, inds=(bond_ind, *right_inds), tags=rtags)
    
        return Tl, Tr
    def tensor_compress_bond(self,T1,T2,absorb='right',where=None):
        # TODO:check for absorb='left'
        left_env_ix, shared_ix, right_env_ix = group_inds(T1, T2)
        if not shared_ix:
            raise ValueError("The tensors specified don't share an bond.")
        elif len(shared_ix) > 1:
            # fuse multibonds
            T1.fuse_({shared_ix[0]: shared_ix})
            T2.fuse_({shared_ix[0]: shared_ix})
            shared_ix = (shared_ix[0],)
        T1_inds,T2_inds = T1.inds,T2.inds

        tmp_ix = rand_uuid()
        T1.reindex_({shared_ix[0]:tmp_ix})
        T2.reindex_({shared_ix[0]:tmp_ix})
        if absorb=='right': # assume T2 is isometric
            T1_L, T1_R = T1.split(left_inds=left_env_ix, right_inds=(tmp_ix,),
                                  get='tensors', method='qr')
            M,T2_R = T1_R,T2
        elif absorb=='left': # assume T1 is isometric
            T2_L, T2_R = T2.split(left_inds=(tmp_ix,), right_inds=right_env_ix,
                                  get='tensors', method='lq')
            T1_L,M = T1,T2_L
        else:
            raise NotImplementedError(f'absorb={absorb}')
        if where is None:
            M_L, *s, M_R = M.split(left_inds=T1_L.bonds(M), get='tensors',
                               absorb=absorb, **self.compress_opts)
        else:
            M_L, *s, M_R = self.tensor_svd(M,where,left_inds=T1_L.bonds(M), 
                               absorb=absorb, **self.compress_opts)

        # make sure old bond being used
        ns_ix, = M_L.bonds(M_R)
        M_L.reindex_({ns_ix: shared_ix[0]})
        M_R.reindex_({ns_ix: shared_ix[0]})

        T1C = T1_L.contract(M_L, output_inds=T1_inds)
        T2C = M_R.contract(T2_R, output_inds=T2_inds)

        # update with the new compressed data
        T1.modify(data=T1C.data)
        T2.modify(data=T2C.data)

        if absorb == 'right':
            T1.modify(left_inds=left_env_ix)
        else:
            T2.modify(left_inds=right_env_ix)
 
    def compress_row_pbc(self,tn,i):
        for j in range(self.Ly): # compress between j,j+1
            where = (i,j),(i,(j+1)%self.Ly)
            self.tensor_compress_bond(tn[i,j],tn[i,(j+1)%self.Ly],absorb='right',where=where)
        return tn
    def compress_row_obc(self,tn,i):
        tn.canonize_row(i,sweep='left')
        for j in range(self.Ly-1):
            where = (i,j),(i,j+1)
            self.tensor_compress_bond(tn[i,j],tn[i,j+1],absorb='right',where=where)        
        return tn
            
