import sklearn
from sklearn.gaussian_process import kernels
import numpy as np

def get_GAP_matrix_v1(Phi, Phi_list,  t, idx, L, beta=1., kernel=None):

    beta = 1.
    y = np.copy(t_e).ravel()
    kernel = kernels.RBF(length_scale=1.)

    X_S = Phi[idx,:]
    C_NpNp = kernel(Phi)
    C_NpS = kernel(Phi,Y=X_S)
    C_SNp = C_NpS.T
    C_SS = kernel(X_S,Y=X_S)
    C_SS_inv = np.linalg.inv(C_SS)

    Lambda = np.diag(np.diag(L.dot(C_NpNp.dot(L.T)) - L.dot(C_NpS.dot(C_SS_inv.dot(C_SNp.dot(L.T))))))
    Q0 = np.linalg.inv(Lambda + 1./beta * np.eye(Lambda.shape[0]))
    Q1 = np.linalg.inv(C_SS + C_SNp.dot(L.T).dot(Q0.dot(L.dot(C_NpS))))
    Q2 = C_SNp.dot(L.T.dot(Q0.dot(y)))
    print("Lambda", Lambda.shape)
    print("Q0",Q0.shape)
    print("Q1",Q1.shape)
    print("Q2",Q2.shape)

    return Q1.dot(Q2)

def get_kvec_v1(single_Phi, idx, Phi, kernel=None):
    X_S = Phi[idx,:]
    _k = kernel(single_Phi, X_S)
    return _k

def get_GAP_matrix_v2(Phi, Phi_list,  t, kernel=None,
                      sigma_E=.001, sigma_W=1.):
    
    # Bartok et al. 2015
    C_SS = np.array([[kernel(_x0,Y=_x1).sum() for _x0 in Phi_list] for _x1 in Phi_list])
    C_SS = sigma_W**2 * C_SS + sigma_E*np.eye(len(Phi_list))

    C_ST = np.copy(C_SS)
    C_TS = C_ST.T

    Lambda_TT = np.diag([len(v)*sigma_E for v in Phi_list])
    Lambda_TT_inv = np.linalg.inv(Lambda_TT)
    
    Q = np.linalg.inv(C_SS + C_ST.dot(Lambda_TT_inv.dot(C_TS))).dot(C_ST.dot(Lambda_TT_inv.dot(t)))
    return Q

def get_kvec_v2(single_Phi, Phi_list, kernel=None):
    _k = np.array([kernel(single_Phi,Y=_x0).sum() for _x0 in Phi_list])
    return _k

def get_atom_contribution_info(Phi, Phi_list, decimals=5):

    decimals = 5
    mod = lambda x: np.around(x, decimals=decimals)
    rPhi = mod(Phi).astype("str")
    #print("rPhi",rPhi)
    
    rPhi_list = [mod(v).astype("str") for v in Phi_list]

    unique, idx, idx_inv = np.unique(rPhi, axis=0, return_inverse=True, return_index=True)
    unique_idx_map = {tuple(unique[v,:]): idx[v] for v in range(unique.shape[0])}

    L = np.zeros((len(Phi_list), idx_inv.shape[0]))

    for i in range(L.shape[0]):
        _ix = np.array([unique_idx_map[tuple(v)] for v in rPhi_list[i]])
        L[i,_ix] = 1.
    return L, unique, idx, idx_inv, unique_idx_map

class GAPRegressor(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    
    Q = None
    X = None
    X_list = None
    kernel = None
    kernel_spec = None
    
    implemented_kernel = set(["rbf",])
    
    def __init__(self, beta=1., sigma_E=0.001, sigma_W=1., kernel=None, 
                 kind="v1", decimals=5):
        
        assert kind in ["v1","v2"], "Unknown 'kind' value (%s)!" %kind
            
        self.kind = kind
        self.beta = beta
        self.sigma_E = sigma_E
        self.sigma_W = sigma_W
        
        self.kernel = kernel
        
        self.decimals = decimals
        
    def _initialize_kernel(self):
        
        assert not self.kernel is None, "'kernel' needs to be provided!"
        assert isinstance(self.kernel, (tuple,list)) and len(self.kernel)==2, "Kernel needs to be provided as a list or tuple of two elements!"
        assert self.kernel[0] in self.implemented_kernel, "Specified kernel is none of the implemented kernels (%s)" %self.implemented_kernel
        
        if self.kernel[0] == "rbf":
            self.k = kernels.RBF(**self.kernel[1])
        else:
            raise NotImplementedError
        
    def fit(self,X_list,y):
        
        self._initialize_kernel()
        
        assert isinstance(X_list, list) and all([isinstance(v,np.ndarray) for v in X_list]),\
            "'X_list' needs to be provided and be be a list of np.ndarrays with a constant number of columns."
        self.X_list = X_list
        self.X = np.vstack(X_list)
        
        if self.kind == "v1":    
            self.L, self.unique, self.idx, self.idx_inv, self.unique_idx_map = get_atom_contribution_info(self.X, self.X_list, decimals=self.decimals)
            self.Q = get_GAP_matrix_v1(self.X, self.X_list, y, self.idx, self.L, 
                                       beta=self.beta, kernel=self.k)
        elif self.kind == "v2":
            self.Q = get_GAP_matrix_v2(self.X, self.X_list, y, 
                                       sigma_W=self.sigma_W, sigma_E=self.sigma_E, 
                                       kernel=self.k)
        
    def predict(self,X,y=None):
        
        if self.kind == "v1":
            k = get_kvec_v1(X, self.idx, self.X, kernel=self.k)
            return k.dot(self.Q).sum()
        
        elif self.kind == "v2":
            if isinstance(X,np.ndarray):
                k = get_kvec_v2(X, self.X_list, kernel=self.k)
                return k.dot(self.Q)
            elif isinstance(X,list):
                ks = [get_kvec_v2(_X, self.X_list, kernel=self.k) for _X in X]
                return np.array([k.dot(self.Q) for k in ks])