import time,itertools,functools,h5py
from .nn_core import (
    Dense,NN,AmplitudeNN,TensorNN,
    tensor2backend,
    relu_init_normal,
)
from .tensor_2d_vmc import AmplitudeFactory2D
from mpi4py import MPI
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()
class AmplitudeNN2D(AmplitudeNN,AmplitudeFactory2D):
    def __init__(self,Lx,Ly,af,**kwargs):
        self.Lx = Lx
        self.Ly = Ly 
        super().__init__(af,**kwargs)

def init_gauge_tanh(Lx,Ly,D,scale=.5,const=1,log=False,eps=1e-3):
    nx = Lx * Ly
    ny = D
    afn = 'tanh'
    af = dict()
    for i,j in itertools.product(range(Lx-1),range(Ly)):
        nn = Dense(nx,ny,afn)
        nn.scale = scale 
        nn.init(0,eps)
        nn.init(1,eps)

        nn = TensorNN([nn])
        nn.input_format = (-1,1),None
        nn.shape = (D,)
        nn.const = const 
        nn.log = log
        af[(i,j),(i+1,j)] = nn 
    return af
def init_gauge_relu(Lx,Ly,D,const=1,log=False,eps=1e-3):
    nx = Lx * Ly
    ny = D
    afn = 'relu'
    af = dict()
    for i,j in itertools.product(range(Lx-1),range(Ly)):
        nn = Dense(nx,ny,afn)
        nn = relu_init_normal(nn,-1,1,1/nx,eps)

        nn = TensorNN([nn])
        nn.input_format = (-1,1),None
        nn.shape = (D,)
        nn.const = const 
        af[(i,j),(i+1,j)] = nn 
    return af
class GaugeNN2D(NN):
    def __init__(self,af,psi):
        self.af = af
        self.keys = list(af.keys())
        self.backend = af[self.keys[0]].backend

        self.Lx,self.Ly = psi.Lx,psi.Ly
        self.gauge_inds = dict()
        for i,j in itertools.product(range(psi.Lx-1),range(psi.Ly)):
            self.gauge_inds[(i,j),(i+1,j)] = tuple(psi[i,j].bonds(psi[i+1,j]))[0] 
    def modify_mps_old(self,tn,config,i,step):
        # for forming boundaries
        if tn is None:
            return tn
        for j in range(self.Ly):
            where = ((i,j),(i+1,j)) if step==1 else ((i-1,j),(i,j))
            y = self.af[where].forward(config)
            ind = self.gauge_inds[where]
            tn[i,j].multiply_index_diagonal_(ind,y) 
        return tn 
    def modify_mps_new(self,*args):
        raise NotImplementedError
    def modify_top_mps(self,*args):
        raise NotImplementedError
    def modify_bot_mps(self,tn,config,i):
        # for 2-row contraction into amplitude
        return self.modify_mps_old(tn,config,i,1)
    def save_to_disc(self,fname,root=0):
        if RANK!=root:
            return
        f = h5py.File(fname+'.hdf5','w')
        for i,key in enumerate(self.keys):
            for j,lr in enumerate(self.af[key].af):
                for k,tsr in enumerate(lr.params):
                    f.create_dataset(f'p{key},{j},{k}',data=tensor2backend(tsr,'numpy'))
        f.close()
    def load_from_disc(self,fname):  
        f = h5py.File(fname+'.hdf5','r')
        for i,key in enumerate(keys):
            af = self.af[key]
            for j,lr in enumerate(af.af):
                for k in range(lr.sh):
                    lr.params[k] = f[f'p{key},{j},{k}'][:]
        f.close() 

def init_env_tanh(Lx,Ly,D,scale=.5,const=0,log=False,eps=1e-3):
    afn = 'tanh'
    sh = [None] * Ly
    ny = 0 
    for j in range(Ly):
        sh[j] = (D,) * 2 if j in (0,Ly-1) else (D,)*3
        ny += np.prod(np.array(sh[j]))
    nx = Lx * Ly
    nn = Dense(nx,ny,afn)
    nn.scale = scale 
    nn.init(0,eps)
    nn.init(1,eps)

    nn = TensorNN([nn])
    nn.input_format = (-1,1),None
    nn.const = const 
    nn.log = log
    nn.shape = sh 
    return nn 
class EnvNN2D(GaugeNN2D):
    def __init__(self,af,**kwargs):
        self.nn = af
        self.keys = list(af.keys())
        self.backend = af[self.keys[0]].backend

        self.Lx,self.Ly = psi.Lx,psi.Ly
    def modify_mps_new(self,*args):
        raise NotImplementedError
    def modify_mps_old(self,*args):
        raise NotImplementedError
    def forward(self,config,step):
        y = self._input(config)
        for l,af in enumerate(self.af):
            y = af.forward(y,step=step)
        if self.log:
            y = self.jnp.exp(y)
        y += self.const 

         
        return y  
    def modify_bot_mps(self,tn,config,i,step=1):
        if tn is None:
            return tn
        config = config[:(i+1)*self.Ly] if step==1 else\
                 config[i*self.Ly:]
        y = self.af[step].forward(config)
        for j in range(self.Ly):
            T = tn[i,j]
            lind = [] if j==0 else tuple(T.bonds(tn[i,j-1]))[0]
            rind = [] if j==self.Ly-1 else tuple(T.bonds(tn[i,j+1]))[0]
            inds = lind+rind+[tn.site_ind(i,j)]
            T.transpose_(*inds)
            if len(T.inds)==2:
                dim1,dim2 = T.shape
                y[:dim1,:dim2] += T.data
            else:
                dim1,dim2,dim3 = T.shape
                y[:dim1,:dim2,:dim3] += T.data 
            T.modify(data=y)
        return tn
    def modify_top_mps(self,tn,config,i):
        return self.modify_bot_mps(tn,config,i,step=-1)
