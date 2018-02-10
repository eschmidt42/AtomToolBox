import numpy as np
from scipy import spatial, special, misc
import itertools, warnings
from sklearn import utils
import sklearn
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import collections

from .spatial import get_ultracell, get_neighbors

#warnings.filterwarnings("error")

def make_array(val,dtype=float):
    val = val if isinstance(val,(list,tuple,np.ndarray)) else [val]
    return np.array(val,dtype=dtype)

def DistanceCosTapering_basis(kappa_, taper_fun, der=0, dx=1e-6):
    from atomtoolbox.features import make_array
    def cos_wrap(kappa):
        def cos_fun(x):
            if der==0:
                return np.cos(kappa*x)*taper_fun(x)
            elif der==1:
                return -kappa*np.sin(kappa*x)*taper_fun(x) + np.cos(kappa*x)*misc.derivative(taper_fun,x,dx=1,n=1)
            else:
                raise NotImplementedError("der = {} is not implemented!".format(der))
        return cos_fun
    return [cos_wrap(kappa) for kappa in make_array(kappa_)]

def DistanceCosTapering_basis_1stDerivative(kappa,taper_fun):
    def cos_fun(x):
        return -kappa*np.sin(kappa*x)*taper_fun(x) + np.cos(kappa*x)*misc.derivative(taper_fun,x,dx=1,n=1)
    return cos_fun

def DistanceGaussTapering_basis(p_, q_, taper_fun, der=0, dx=1e-6):
    from atomtoolbox.features import make_array
    def gauss_wrap(p,q):
        def gauss_fun(x):
            if der == 0:
                return np.exp(-p*(x-q)**2)*taper_fun(x)
            elif der == 1:
                return -p*2*(x-q)*np.exp(-p*(x-q)**2)*taper_fun(x) + np.exp(-p*(x-q)**2)*misc.derivative(taper_fun, x, dx=dx, n=1)
            else:
                raise NotImplementedError("der = {} is not implemented!".format(der))
        return gauss_fun
    g = []
    for p in make_array(p_):
        for q in make_array(q_):
            g.append(gauss_wrap(p,q))
    return g

def DistanceSineCosineTapering_basis(kappa_, taper_fun, der=0, dx=1e-6):
    from atomtoolbox.features import make_array
    kappa = make_array(kappa_)
    def wrapped_sine(k):
        def sine_fun(x):
            if der == 0:
                return np.sin(k*x)*taper_fun(x)
            elif der == 1:
                return k*np.cos(k*x)*taper_fun(x) + np.sin(k*x)*misc.derivative(taper_fun, x, n=der, dx=dx)
            else:
                raise NotImplementedError("der = {} is not implemented!".format(der))
        return sine_fun
    
    def wrapped_cosine(k):
        def cosine_fun(x):
            if der == 0:
                return np.cos(k*x)*taper_fun(x)
            elif der == 1:
                return -k*np.sin(k*x)*taper_fun(x) + np.cos(k*x)*misc.derivative(taper_fun, x, n=der, dx=dx)
            else:
                raise NotImplementedError("der = {} is not implemented!".format(der))
        return cosine_fun
    return [wrapped_cosine(k) for k in kappa] + [wrapped_sine(k) for k in kappa]

def get_angles(r_vec):
    # theta
    theta = np.arctan2(r_vec[:,1],r_vec[:,0]) + np.pi
    theta = np.absolute(theta)
    # phi
    isminus1 = np.where(r_vec[:,2] - (-1)<0)[0]
    r_vec[isminus1,2] = -1
    is1 = np.where(r_vec[:,2] - 1>0)[0]
    r_vec[is1,2] = 1
    phi = np.arccos(r_vec[:,2])
    phi = np.absolute(phi)
    return theta, phi

def get_q(theta,phi,l): #weighting by number of neighbors
    q = 0
    
    Nneighs = float(len(theta))
    m_range = np.arange(-l,l+1)
    
    for i,m in enumerate(m_range):
        Y = special.sph_harm(m,l,theta,phi)
        sum_Y = np.absolute(np.sum(Y/Nneighs))**2
        q += sum_Y
        
    q *= 4*np.pi/(2.*l+1)
    q = np.sqrt(q)
    return q

def get_q2(theta,phi,l,m_range,factor): #weighting by number of neighbors
        
    Nneighs = float(theta.shape[0])
    q = np.array([(special.sph_harm(m,l,theta,phi)).sum()/Nneighs for m in m_range])
    q = np.absolute(q)**2
    return factor * q.sum()

def get_q3(theta,phi,l,m_range): #weighting by number of neighbors
    Nneighs = float(theta.shape[0])
    q = np.array([(special.sph_harm(m,l,theta,phi)).sum()/Nneighs for m in m_range])
    return q

def get_crystal_design_matrix(positions=None, species=None, cell=None, atoms=None, 
                              r_cut=None, features_class=None, params_features=None, 
                              return_force=False, num_neigh=None,
                              emb_density_funs=None, tol0=1e-6, return_mapper=False,
                              check_bounds=True):
    """Creates the design matrix for the given crystal.

    Parameters
    ----------

    r_cut : float, optional, default None

    num_neigh : int, optional, default None

    Notes
    -----
    r_cut always needs to be different than None, even if the num_neigh parameter is
    not equal to None.
    """
    assert not r_cut is None and isinstance(r_cut,(float,int)), "Error 'r_cut' has to be given as either a float or int value."
    assert all([not positions is None, not species is None, not cell is None]) or not atoms is None, "Either 'cell', 'positions' and 'species' need to be given or 'atoms'."
    if not atoms is None:
        positions = atoms.get_positions(wrap=True)
        species = atoms.get_chemical_symbols()
        invcell = np.linalg.inv(atoms.get_cell())
        fpos = np.dot(positions,invcell)
    else:
        positions = np.array(positions)
        species = np.array(species)
        invcell = np.linalg.inv(cell)
        fpos = np.dot(positions,invcell)
        
    upositions, uspecies, uindices = get_ultracell(atoms=atoms,fpos=fpos,cell=cell,
                                                   species=species,r_cut=r_cut,show=False,
                                                   verbose=False,max_iter=20,ftol=1e-6,
                                                   check_bounds=check_bounds)
    
    # getting the ultracell neighboring indices
    uindices_neigh = get_neighbors(positions,upositions=upositions, r_cut=r_cut, 
                                   num_neigh=num_neigh, tol=tol0)
    if any([len(v)==0 for v in uindices_neigh]):
        warnings.warn("uindices_neigh contains empty arrays! Consider changing r_cut ({}) or num_neigh ({}).".format(r_cut,num_neigh))
    
    if not emb_density_funs is None:
        from atomtoolbox.eam import get_rhos
        rhos, urhos = get_rhos(positions, species, upositions, uspecies, 
                               uindices_neigh, uindices, emb_density_funs)
    else:
        rhos, urhos = None, None

    mapper, c = dict(), 0
    
    if isinstance(features_class,(list,tuple)):
        for i,f in enumerate(features_class):
            
            if isinstance(params_features,dict):
                featurer = f(return_force=return_force,**params_features)
            elif isinstance(params_features,list) and all([isinstance(v,dict) for v in params_features]):
                featurer = f(return_force=return_force,**params_features[i])
            else:
                raise ValueError("WTF!")

            if return_force:
                if i==0:
                    Phi, Phi_f = featurer.fit_transform(positions, species, upositions, uspecies, 
                                        uindices_neigh, rhos=rhos, urhos=urhos)
                    incr = Phi.shape[1]
                else:
                    _Phi, _Phi_f = featurer.fit_transform(positions, species, upositions, uspecies, \
                                        uindices_neigh, rhos=rhos, urhos=urhos)
                    Phi = np.hstack((Phi, _Phi))
                    Phi_f = np.hstack((Phi_f, _Phi_f))
                    incr = _Phi.shape[1]
            else:
                if i==0:
                    Phi = featurer.fit_transform(positions, species, upositions, uspecies, 
                                        uindices_neigh)
                    incr = Phi.shape[1]
                else:
                    _Phi = featurer.fit_transform(positions, species, upositions, uspecies, \
                                        uindices_neigh)
                    Phi = np.hstack((Phi, _Phi))
                    incr = _Phi.shape[1]
            mapper[i] = np.arange(c, c+incr)
            c += incr
    else:
        featurer = features_class(return_force=return_force,**params_features)
        if return_force:
            Phi, Phi_f = featurer.fit_transform(positions, species, upositions, uspecies, 
                                        uindices_neigh, rhos=rhos, urhos=urhos)
        else:
            Phi = featurer.fit_transform(positions, species, upositions, uspecies, 
                                        uindices_neigh, rhos=rhos,urhos=urhos)

        mapper[0] = np.arange(Phi.shape[1])

    if return_force:
        if return_mapper:
            return Phi, Phi_f, mapper
        else:
            return Phi, Phi_f
    else:
        if return_mapper:
            return Phi, mapper
        else:
            return Phi

class ThreeBodyAngleFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate three-body angle features.

    Generate a design matrix of k bins between 0 and np.pi containing the
    three-body angle distribution for individual atoms.

    Parameters
    ----------
    k : int, optional, default 10
        The number of bins.
        
    _range : tuple, optional, default (0,pi)
        The range for the histograms.
    
    normed : boolean, optional, default True
        Whether or not to normalize the histograms.
        
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> TBAF_all = ThreeBodyAngleFeatures(k=20, _range=(0,np.pi), normed=True, 
                                          element_filter=all_filter)
    >>> Phi_all = TBAF_all.fit_transform(positions, species, upositions, uspecies, 
                                         uindices_neigh)
    
    """

    name = "ThreeBodyAngleFeatures"

    def __init__(self, k=10, _range=(0,np.pi), normed=True,\
                 element_filter=None, tol0=1e-6, return_force=False, **kwargs):
        self.k = k
        self._range = _range
        self.normed = normed
        self.element_filter = element_filter
        self.tol0 = tol0
        self.return_force = return_force
        
    @staticmethod
    def _get_histograms(i,X, species, uX, uspecies, idx_neigh, k=10, _range=(0,np.pi),
                        normed=True, element_filter=None, tol0=1e-6, return_force=False):
        """Computes the 3-body angle histograms.
        """
        
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
                
        uatoms = uX[uidx]
        
        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        # in case no neighbors remain
        if len(uidx) == 0:
            return np.zeros(k), None
        uatoms = uatoms[r_valid]

        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        i0,i1 = np.triu_indices(n_neigh,k=1)        
        dr0 = uatoms[i0]-atom
        dr1 = uatoms[i1]-atom
        r0 = np.linalg.norm(dr0,axis=1)
        r1 = np.linalg.norm(dr1,axis=1)
        r0dotr1 = np.einsum("nd,nd->n",dr0,dr1)
        cos_theta = r0dotr1/(r0*r1)          
        
        # taking care of floating precision errors where theta_arg may be -1 - 1e-42 causing NaNs
        isminus1 = np.where(cos_theta - (-1)<0)[0]
        cos_theta[isminus1] = -1
        # taking care of floating precision errors where theta_arg may be 1 + 1e-42 causing NaNs
        is1 = np.where(cos_theta - 1>0)[0]
        cos_theta[is1] = 1
        
        theta = np.arccos(cos_theta)
        
        if return_force:
            dr0_n = (dr0.T / r0).T
            dr1_n = (dr1.T / r1).T
            
            c0 = 1. / (r0*r1)
            c1 = (-r0/r1 * dr1.T - r1/r0 * dr0.T).T
            theta_der = (c0 * (-dr0.T - dr1.T - cos_theta * c1.T)).T
            
        idx_nan = np.where(np.isnan(theta))[0]
        if len(idx_nan)>0:
            print("isminus1 ",isminus1)
            print("is1 ",is1)
            print("idx_nan ",idx_nan)
            print("cos_theta ",cos_theta[idx_nan])
            print("r0dotr1 ",r0dotr1[idx_nan])
            print("r0 ",r0[idx_nan])
            print("r1 ",r1[idx_nan])
            raise ValueError("Found NaNs!")
        
        h, e = np.histogram(theta,bins=k,normed=normed,range=_range)
        if return_force:
            hf = []
            for i in range(3):
                _hf, _ = np.histogram(theta_der[:,i], bins=k, normed=normed, range=_range)
                hf.append(_hf)
            hf = np.array(hf).T
        else:
            hf = None
        return h, hf
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.k
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((self.n_output_features_, n_samples*3), dtype=X.dtype)
            print("XP ",XP.shape," XP_f ",XP_f.shape)
        
        for i in range(n_samples):
            h, hf = self._get_histograms(i, X, species, uX, uspecies, uidx_neigh, k=self.k,
                                        _range=self._range, normed=self.normed, 
                                        element_filter=self.element_filter, tol0=self.tol0,
                                        return_force=self.return_force)
            XP[i,:] = h
            if self.return_force:
                XP_f[:,3*i:3*(i+1)] = hf
        if self.return_force:
            return XP, XP_f.T
        return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class BondOrderParameterFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate bond order parameter features.

    Generate a design matrix of bond order parameters for individual atoms.

    Parameters
    ----------
    k : iterable of int, optional, default [4,6]
        The order of the bond order parameter.
                
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.
    
    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> BOPF_all = BondOrderParameterFeatures(k=[4,6], element_filter=all_filter)
    >>> Phi_all = BOPF.fit_transform(positions, species, upositions, uspecies, uindices_neigh)
    """

    name = "BondOrderParameterFeatures"
    dt_q = 0
    
    def __init__(self, k=[4,6], element_filter=None, tol0=1e-6, kind="3", **kwargs):
        self.k = k
        self.element_filter = element_filter
        self.tol0 = tol0
        
        self.m_range_dict = {l:np.arange(-l,l+1) for l in self.k}
        self.factor_dict = {l:4*np.pi/(2.*l+1) for l in self.k}
        
        assert kind in ["1", "2", "3"], "'kind' ({}) not understood!".format(kind)
        self.kind = kind
        
    @staticmethod
    def _get_bops(i, X, species, uX, uspecies, idx_neigh, k=[4,6],
                  element_filter=None, tol0=1e-6, m_range_dict=None, 
                  factor_dict=None):
        """Computes the bond order paramters.
        """
        
        n_q = len(k)
        qs = np.zeros(n_q)
                          
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]

        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        if len(uidx) == 0:
            return qs
        uatoms = uatoms[r_valid]

        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        dr = np.divide(dr.T, r).T
        theta, phi = get_angles(dr)
        
        for j,l in enumerate(k):
            qs[j] = get_q(theta,phi,l)
        
        return qs
    
    @staticmethod
    def _get_bops2(i, X, species, uX, uspecies, idx_neigh, k=[4,6],
                  element_filter=None, tol0=1e-6, m_range_dict=None, 
                  factor_dict=None):
        """Computes the bond order paramters.
        """
        
        n_q = len(k)
        qs = np.zeros(n_q)
                          
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]

        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        if len(uidx) == 0:
            return qs
        uatoms = uatoms[r_valid]

        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        dr = np.divide(dr.T, r).T
        theta, phi = get_angles(dr)
        
        
        for j,l in enumerate(k):
            qs[j] = get_q2(theta,phi,l,m_range_dict[l],factor_dict[l])
        qs = np.sqrt(qs)        
        return qs
    
    @staticmethod
    def _get_bops3(X, species, uX, uspecies, idx_neigh, k=[4,6],
                  element_filter=None, tol0=1e-6, m_range_dict=None, 
                  factor_dict=None):
        """Computes the bond order paramters.
        """
        
        n_q = len(k)
        qs = np.zeros((X.shape[0],n_q),dtype=float)
        qs_dict = {l: np.zeros((X.shape[0], len(m_range_dict[l])),dtype=complex) for l in k}
                        
        for i in range(X.shape[0]):
            # current atom
            atom = X[i,:]
            spec = species[i]

            #neighboring atoms
            uidx = idx_neigh[i]

            if callable(element_filter):
                uidx = uidx[element_filter(uspecies[uidx],spec)]

            uatoms = uX[uidx]

            r = np.linalg.norm(uatoms-atom,axis=1)
            r_valid = np.where(r>tol0)[0]
            uidx = uidx[r_valid]
            if len(uidx) == 0:
                continue
            uatoms = uatoms[r_valid]

            uspecs = uspecies[uidx]
            n_neigh = len(uidx)

            dr = uatoms-atom
            r = np.linalg.norm(dr,axis=1)
            dr = np.divide(dr.T, r).T
            theta, phi = get_angles(dr)
            
            for j,l in enumerate(k):
                qs_dict[l][i,:] = get_q3(theta,phi,l,m_range_dict[l])
        
        for j,l in enumerate(k):
            assert np.isfinite(qs_dict[l]).all(), "qs_dict[{}] contains non-finite values.".format(l)
            qs[:,j] = (np.absolute(qs_dict[l])**2).sum(axis=1)
            qs[:,j] = np.sqrt(factor_dict[l]*qs[:,j])
         
        return qs
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = len(self.k)
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
                
        if self.kind == "1":
            for i in range(n_samples):
                qs = self._get_bops(i, X, species, uX, uspecies, uidx_neigh, 
                                    k=self.k, tol0=self.tol0, m_range_dict=self.m_range_dict,
                                    factor_dict=self.factor_dict, element_filter=self.element_filter)
                XP[i,:] = qs
          
        elif self.kind == "2":
            for i in range(n_samples):
                qs = self._get_bops2(i, X, species, uX, uspecies, uidx_neigh, 
                                    k=self.k, tol0=self.tol0, m_range_dict=self.m_range_dict,
                                    factor_dict=self.factor_dict, element_filter=self.element_filter)
                XP[i,:] = qs
        
        elif self.kind == "3":
            XP = self._get_bops3(X, species, uX, uspecies, uidx_neigh, 
                                    k=self.k, tol0=self.tol0, m_range_dict=self.m_range_dict,
                                    factor_dict=self.factor_dict, element_filter=self.element_filter)
        else:
            raise ValueError("Do not understand 'kind' = '{}'".format(kind))
        assert np.isfinite(XP).all(), "XP contains non-finite values."
        return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

def taper_fun_wrapper(_type="x4ge",**kwargs):
    if _type=="x4ge":
        def x4_fun(x):
            x4 = ((x-kwargs["a"])/float(kwargs["b"]))**4
            x4[x>=kwargs["a"]] = 0
            return x4/(1.+x4)
        return x4_fun
    
    elif _type=="x4le":
        def x4_fun(x):
            x4 = ((x-kwargs["a"])/float(kwargs["b"]))**4
            x4[x<=kwargs["a"]] = 0
            return x4/(1.+x4)
        return x4_fun
    
    elif _type=="Behlerge":
        def Behler_fun(x):
            y = .5 * (np.cos(np.pi*x/float(kwargs["a"])) + 1.)
            y[x>=kwargs["a"]] = 0
            return y
        return Behler_fun
    
    elif _type=="Behlerle":
        def Behler_fun(x):
            y = .5 * (np.cos(np.pi*x/float(kwargs["a"])) + 1.)
            y[x<=kwargs["a"]] = 0
            return y
        return Behler_fun
    
    else:
        raise NotImplementedError("_type '{}' unknown.".format(_type))

class DistanceTaperingFeatures_2body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate features which taper based on 2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = sum_j f(r_ij)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DTF_all = DistanceTaperingFeatures_2body(taper_fun=taper_fun,
                                          element_filter=all_filter)
    >>> Phi_all = DTF_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)

                                         
    Literature
    ----------
    Behler, J., Constructing high-dimensional neural network potentials: A tutorial review,
        International Journal of Quantum Chemistry, 2015, 115 (16), 1032-1050
    
    """
    
    name = "DistanceTaperingFeatures_2body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None, 
                 return_force=False, tol0=1e-6, **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
            
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
        self.return_force = return_force

        self.tol0 = tol0
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, return_force=False,
                    tol0=1e-6):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]

        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        uatoms = uatoms[r_valid]

        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        if return_force:
            f = misc.derivative(taper_fun,r,dx=1.,n=1) * 1./r * dr.T
            return taper_fun(r).sum(), f.sum(axis=1)
        else:
            return taper_fun(r).sum()
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = 1
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((self.n_output_features_, n_samples*3), dtype=X.dtype)
            
        for i in range(n_samples):
            if self.return_force:
                G, G_f = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, taper_fun=self.taper_fun,
                                     return_force=self.return_force, tol0=self.tol0)
                XP_f[:,3*i:3*(i+1)] = G_f
            else:
                G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, taper_fun=self.taper_fun,
                                     return_force=self.return_force, tol0=self.tol0)
            
            XP[i] = G
        if self.return_force:
            return XP, XP_f.T
        else:
            return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class DistanceExpTaperingFeatures_2body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate exponential features which taper based on 2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = sum_j exp(-eta*(r_ij-r_s)**2)f(r_ij)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    eta_ : int, float, list, tuple or np.ndarray of length N
        
    rs_ : int, float, list, tuple or np.ndarray of length N
        
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DeTF_all = DistanceExpTaperingFeatures_2body(element_filter=all_filter,
                                            taper_fun=taper_fun, rs_=[0.,1.], eta_=[1.,1.])

    >>> Phi_all = DeTF_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)

                                         
    Literature
    ----------
    Behler, J., Constructing high-dimensional neural network potentials: A tutorial review,
        International Journal of Quantum Chemistry, 2015, 115 (16), 1032-1050
    
    """
    
    name = "DistanceExpTaperingFeatures_2body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None, eta_=[1], 
                 rs_=[0], return_force=False, tol0=1e-6, **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
            
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
        
        self.eta_ = make_array(eta_)
        self.rs_ = make_array(rs_)
        assert len(self.eta_)==len(self.rs_), "eta_ and rs_ need to be the same length {}!={}".format(len(self.eta_),len(self.rs_))
        self.N = self.eta_.shape[0]
        self.return_force = return_force

        self.tol0 = tol0
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, rs_=[0], eta_=[1],
                    return_force=False, tol0=1e-6):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]

        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        uatoms = uatoms[r_valid]

        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        
        f = taper_fun(r)
        N = len(rs_)
        vals = np.zeros(N)
        
        if return_force:
            fp = misc.derivative(taper_fun,r,dx=1,n=1)
            forces = np.zeros((3,N))
            dr_normed = (dr.T/r)
                
        for i in range(N):
            e = np.exp(-eta_[i]*(r-rs_[i])**2)
            vals[i] = (e*f).sum()
            
            if return_force:
                ep = -2.*eta_[i]*(r-rs_[i]) * e
                _f = (e*fp + ep*f) * dr_normed
                forces[:,i] = _f.sum(axis=1)
        if return_force:
            return vals, forces
        else:
            return vals
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((3*n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):            
            
            if self.return_force:
                G, G_f = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                          element_filter=self.element_filter, rs_=self.rs_, eta_=self.eta_,
                                          taper_fun=self.taper_fun, return_force=self.return_force,
                                          tol0=self.tol0)
                XP_f[3*i:3*(i+1),:] = G_f
            else:
                G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, rs_=self.rs_, eta_=self.eta_, 
                                     taper_fun=self.taper_fun,
                                     return_force=self.return_force, tol0=self.tol0)
            XP[i,:] = G
            
        if self.return_force:
            return XP, XP_f
        else:
            return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class DistanceCosTaperingFeatures_2body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate cosine features which taper based on 2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = sum_j cos(kappa*r_ij)*f(r_ij)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    kappa_ : int, float, list, tuple or np.ndarray of length N
                
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.
    
    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DcTF_all = DistanceCosTaperingFeatures_2body(element_filter=all_filter,
                                            taper_fun=taper_fun, kappa_=[1.,2.])

    >>> Phi_all = DcTF_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)

                                         
    Literature
    ----------
    Behler, J., Constructing high-dimensional neural network potentials: A tutorial review,
        International Journal of Quantum Chemistry, 2015, 115 (16), 1032-1050
    
    """
    
    name = "DistanceCosTaperingFeatures_2body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None,\
                 kappa_=1, return_force=False, emb_density_funs=None,\
                 taper_fun_emb=None, tol0=1e-6, **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
        
        self.tol0 = tol0
        
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
        
        self.kappa_ = make_array(kappa_)
        self.N = self.kappa_.shape[0]
        self.return_force = return_force  
        
        self.emb_density_funs = emb_density_funs
        self.taper_fun_emb = taper_fun_emb
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, kappa_=[1.], 
                    return_force=False, emb_density_funs=None, taper_fun_emb=None,
                    rhos=None, urhos=None, tol0=1e-6):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        do_emb = False # whether or not to compute the embedding density (in the case of EAM)
        do_taper_emb = False # whether or not to taper the embedding density
        if not emb_density_funs is None:
            do_emb = True
            do_taper_emb = not taper_fun_emb is None
            assert isinstance(emb_density_funs,dict), "Expected the provided 'emb_density_funs' parameter to be a dict."
            assert all([callable(v) for v in emb_density_funs.values()]), "Expected the provided the values of the 'emb_density_funs' parameter to be callable."
            assert all([not rhos is None, not urhos is None]), "In order to compute embedding density based features 'rhos' and 'urhos' need to be provided."
            
            species_order = sorted(list(emb_density_funs))
            
        if all([not rhos is None, not urhos is None]) and not do_emb:
            raise ValueError("'rhos' and 'urhos' are given but not 'emb_density_funs'!")
                            
        # current atom
        atom = X[i,:]
        spec = species[i]

        #neighboring atoms
        uidx = idx_neigh[i]
        assert len(uidx) > 0, "No neighbors available for current atom!"
            
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
        if do_emb:
            assert isinstance(element_filter,str), "For embedding density features the parameter 'element_filter' needs to be a string! Given: {}".format(element_filter)
            emb_spec = element_filter # in case of embedding density the element_filter is just a string

        uatoms = uX[uidx]
        uspecs = uspecies[uidx]
        if do_emb:
            udens = urhos[uidx]
        n_neigh = len(uidx)

        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        
        # re-do neighborhood for valid r (just to be safe)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        
        dr = dr[r_valid]
        r = r[r_valid]
        if do_emb:
            udens = udens[r_valid]
        n_neigh = len(uidx)
        
        if do_emb:
            dens = rhos[i]
            
        if do_emb:
            
            if do_taper_emb:
                udens_f = taper_fun_emb(udens)
                dens_f = taper_fun_emb(dens)
                
            else:
                taper_fun_emb = lambda x: 1. if isinstance(x,float) else np.ones(x.shape)
                udens_f = taper_fun_emb(udens)
                dens = taper_fun_emb(dens)
        
        f = taper_fun(r)
        N = len(kappa_)
                
        if return_force:
            
            if do_emb:
                dens_fp = misc.derivative(taper_fun_emb, dens, dx=1, n=1)
                udens_fp = misc.derivative(taper_fun_emb, udens, dx=1, n=1)
            
            fp = misc.derivative(taper_fun, r, dx=1, n=1)
            dr_normed = (dr.T/r)
        
        if do_emb: # embedding density version
            #Nspec = len(species_order)
            #vals = np.zeros(N*Nspec)
            #forces = np.zeros((3,N*Nspec))        
            #idx_dict_species = {k:np.arange(N*v,N*(v+1)) for v,k in enumerate(species_order)}
            
            _forces = np.zeros((3,N))
            _vals = np.zeros(N)
            forces = np.zeros((3,N))
            vals = np.zeros(N)
            
            if spec == emb_spec:
                # current atom - looping all basis functions
                for _i in range(N):

                    c = np.cos(kappa_[_i]*dens)
                    _vals[_i] = c*dens_f

                    if return_force:

                        cp = -kappa_[_i]*np.sin(kappa_[_i]*dens)
                        _f = (cp*dens_f+c*dens_fp) * fp * dr_normed

                        _forces[:,_i] = _f.sum(axis=1)
            
                if return_force:
                    #forces[:,idx_dict_species[spec]] += _forces
                    forces += _forces
                #vals[idx_dict_species[spec]] += _vals
                vals += _vals
            
            if return_force:
                
                # looping all neighbors
                for i_el, _spec in enumerate(uspecs):
                    if _spec != emb_spec:
                        continue
                    _forces[:,:] = 0
                    
                    _dens = udens[i_el]
                    _dens_f = udens_f[i_el]
                    
                    _dens_fp = udens_fp[i_el]
                    _fp = fp[i_el]
                    
                    # looping all basis functions
                    for _i in range(N):

                        c = np.cos(kappa_[_i]*_dens)
                        cp = -kappa_[_i]*np.sin(kappa_[_i]*_dens)
                        
                        _f = (cp*_dens_f+c*_dens_fp) * _fp * dr_normed[:,i_el]
                        _forces[:,_i] = _f
                        
                    forces[:,idx_dict_species[_spec]] += _forces
                    
        else: # pair distance version
            
            vals = np.zeros(N)
            forces = np.zeros((3,N))        
            
            for i in range(N):

                c = np.cos(kappa_[i]*r)
                vals[i] = (c*f).sum()

                if return_force:

                    cp = -kappa_[i]*np.sin(kappa_[i]*r)
                    _f = (cp*f + c*fp) * dr_normed

                    forces[:,i] = _f.sum(axis=1)
        
        if return_force:
            return vals, forces
        else:
            return vals
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        #if not self.emb_density_funs is None:
        #    self.n_output_features_ = self.N*len(self.emb_density_funs)
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh, rhos=None, urhos=None):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((3*n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            if self.return_force:
                G, G_f = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, 
                                     taper_fun=self.taper_fun, kappa_=self.kappa_, 
                                     return_force=self.return_force, 
                                     emb_density_funs=self.emb_density_funs, 
                                     taper_fun_emb=self.taper_fun_emb, rhos=rhos,
                                     urhos=urhos, tol0=self.tol0)
                XP_f[3*i:3*(i+1),:] = G_f
            else:
                G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, 
                                     taper_fun=self.taper_fun, kappa_=self.kappa_, 
                                     return_force=self.return_force,
                                     emb_density_funs=self.emb_density_funs, 
                                     taper_fun_emb=self.taper_fun_emb, rhos=rhos,
                                     urhos=urhos, tol0=self.tol0)
            
            XP[i,:] = G
        if self.return_force:
            return XP, XP_f
        else:
            return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, rhos=None,\
                      urhos=None, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh, rhos=rhos, urhos=urhos)

class DistanceCosExpTaperingFeatures_3body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate cosine exponential 3-body features which taper based on products of
    2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = 2^(1-xi) sum_{j,k \ne i} (1 + lambda*cos(theta_{ijk})^xi
        * exp(-eta*(r_ij^2+r_jk^2+r_ik^2)) * f(r_ij)
        * f(r_ij)*f(r_ik)*f(r_jk)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    xi_ : int, float, list, tuple or np.ndarray of length N
    
    lambda_ : int, float, list, tuple or np.ndarray of length N
    
    eta_ : int, float, list, tuple or np.ndarray of length N
            
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DceTF3_all = DistanceCosExpTaperingFeatures_3body(element_filter=all_filter,
                                            taper_fun=taper_fun, xi_=[0,1], lambda_=[1,1],
                                            eta_=[.5,.5])

    >>> Phi_all = DceTF3_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)

                                         
    Literature
    ----------
    Behler, J., Constructing high-dimensional neural network potentials: A tutorial review,
        International Journal of Quantum Chemistry, 2015, 115 (16), 1032-1050
    
    """
    
    name = "DistanceCosExpTaperingFeatures_3body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None, xi_=0.,
                                             lambda_=1., eta_=1., **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
            
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
                
        self.lambda_ = make_array(lambda_)
        self.eta_ = make_array(eta_)
        self.xi_ = make_array(xi_)
        assert len(self.lambda_) == len(self.eta_) == len(self.xi_), "lambda_ ({}), eta_ ({}) and xi_ ({}) need to be of the same length.".format(self.lambda_.shape[0],self.eta_.shape[0],self.xi_.shape[0])
        self.N = self.lambda_.shape[0]
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, xi_=[0.],
                    lambda_=[1.], eta_=[1.]):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]
        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        i0,i1 = np.triu_indices(n_neigh,k=1)
        
        dr0 = uatoms[i0]-atom
        dr1 = uatoms[i1]-atom
        dr = uatoms[i0]-uatoms[i0]
        
        r = np.linalg.norm(dr,axis=1)
        r0 = np.linalg.norm(dr0,axis=1)
        r1 = np.linalg.norm(dr1,axis=1)
        
        r0dotr1 = np.einsum("nd,nd->n",dr0,dr1)
        costheta = (r0dotr1/(r0*r1))        
        
        N = len(lambda_)
        
        f = taper_fun(r)
        f0 = taper_fun(r0)
        f1 = taper_fun(r1)
        fall = f*f0*f1
        
        vals = np.zeros(N)
        for i in range(N):
            g = (1. + lambda_[i]*costheta)**xi_[i]
            e = np.exp(-eta_[i]*(r**2+r0**2+r1**2))
            vals[i] = 2**(1-xi_[i])*(fall*g*e).sum()
        
        return vals
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                 element_filter=self.element_filter, 
                                 taper_fun=self.taper_fun, xi_=self.xi_, 
                                 eta_=self.eta_, lambda_=self.lambda_)
            
            XP[i,:] = G
        return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class DistanceCosExpTaperingFeatures_3body2(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate cosine exponential 3-body features which taper based on products of
    2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = 2^(1-xi) sum_{j,k \ne i} (1 + lambda*cos(theta_{ijk})^xi
        * exp(-eta*(r_ij^2+r_ik^2)) * f(r_ij)
        * f(r_ij)*f(r_ik)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    xi_ : int, float, list, tuple or np.ndarray of length N
    
    lambda_ : int, float, list, tuple or np.ndarray of length N
    
    eta_ : int, float, list, tuple or np.ndarray of length N
            
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DceTF3_2_all = DistanceCosExpTaperingFeatures_3body2(element_filter=all_filter,
                                            taper_fun=taper_fun, xi_=[0,1], lambda_=[1,1],
                                            eta_=[.5,.5])

    >>> Phi_all = DceTF3_2_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)

                                         
    Literature
    ----------
    Behler, J., Constructing high-dimensional neural network potentials: A tutorial review,
        International Journal of Quantum Chemistry, 2015, 115 (16), 1032-1050
    
    """
    
    name = "DistanceCosExpTaperingFeatures_3body2"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None, xi_=0.,
                                             lambda_=1., eta_=1., **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
            
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
        self.lambda_ = make_array(lambda_)
        self.eta_ = make_array(eta_)
        self.xi_ = make_array(xi_)
        assert len(self.lambda_) == len(self.eta_) == len(self.xi_), "lambda_ ({}), eta_ ({}) and xi_ ({}) need to be of the same length.".format(self.lambda_.shape[0],self.eta_.shape[0],self.xi_.shape[0])
        self.N = self.lambda_.shape[0]
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, xi_=[0.],
                    lambda_=[1.], eta_=[1.]):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]
        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        i0,i1 = np.triu_indices(n_neigh,k=1)
        
        dr0 = uatoms[i0]-atom
        dr1 = uatoms[i1]-atom
        
        r0 = np.linalg.norm(dr0,axis=1)
        r1 = np.linalg.norm(dr1,axis=1)
        
        r0dotr1 = np.einsum("nd,nd->n",dr0,dr1)
        costheta = (r0dotr1/(r0*r1))        
        
        N = len(xi_)
        
        f0 = taper_fun(r0)
        f1 = taper_fun(r1)
        fall = f0*f1
        
        vals = np.zeros(N)
        
        for i in range(N):
            g = (1. + lambda_[i]*costheta)**xi_[i]
            e = np.exp(-eta_[i]*(r0**2+r1**2))
            vals[i] = 2**(1-xi_[i])*(fall*g*e).sum()
        
        return vals
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                 element_filter=self.element_filter, 
                                 taper_fun=self.taper_fun, xi_=self.xi_, 
                                 eta_=self.eta_, lambda_=self.lambda_)
            
            XP[i,:] = G
        return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class CentroSymmetryParameterFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate centro symmetry parameter features.

    Generate a design matrix of centro symmetry parameters for individual atoms.

    Parameters
    ----------
    N : iterable of int, optional, default [12,]
        The number of pairs for the CSP.
                
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> CSP_all = CentroSymmetryParameterFeatures(N=[12,18], element_filter=all_filter)
    >>> Phi_all = CSP_all.fit_transform(positions, species, upositions, uspecies, uindices_neigh)
    """

    name = "CentroSymmetryParameterFeatures"

    def __init__(self, N=[12], element_filter=None, **kwargs):
        self.N = make_array(N,dtype=int)
        self.element_filter = element_filter
        
    @staticmethod
    def _get_csps(i, X, species, uX, uspecies, idx_neigh, N=[12],
                        element_filter=None):
        """Computes the bond order paramters.
        """
                                  
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]
        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        
        # find pairs of nearest neighbours
        i0,i1 = np.triu_indices(n_neigh,k=1)
        
        dr_sum = dr[i0]+dr[i1]
        r_sum = np.linalg.norm(dr_sum,axis=1)
        near_zero = np.where(np.isclose(r_sum,0,atol=1e-5))[0]
        
        csps = np.zeros(len(N))
        for i in range(len(N)):
            idx_nearzero_sorted = np.argsort(r_sum[near_zero])[:N[i]]
            csps[i] = r_sum[near_zero[idx_nearzero_sorted]].sum() 
        return csps
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N.shape[0]
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            csps = self._get_csps(0,X, species, uX, uspecies, uidx_neigh, N=self.N)
            
            XP[i,:] = csps
        return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class ElementCountFeatures(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate features which count the number of neighboring atoms of each type.

    Generates an integer design matrix for a given set of potential neighboring 
    elements.

    Parameters
    ----------
    elements : list, tuple or np.ndarray of str
        The elements to look for.
        
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> TBAF_all = ThreeBodyAngleFeatures(k=20, _range=(0,np.pi), normed=True, 
                                          element_filter=all_filter)
    >>> Phi_all = TBAF_all.fit_transform(positions, species, upositions, uspecies, 
                                         uindices_neigh)
    
    """

    name = "ElementCountFeatures"

    def __init__(self, elements, normed,\
                 tol0=1e-6, **kwargs):
        
        self.elements = np.array(sorted(list(elements)),dtype=str)
        self.normed = normed
        self.tol0 = tol0
        
    @staticmethod
    def _count_elements(i,X, species, uX, uspecies, idx_neigh, elements, tol0=1e-6, 
                        normed=False):
        """Counts the neighboring elements.
        """
        
        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
                
        uatoms = uX[uidx]
        
        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        # in case no neighbors remain
        if len(uidx) == 0:
            return np.zeros(len(elements))
        uatoms = uatoms[r_valid]

        uspecs = uspecies[uidx]
        counted_elements = collections.Counter(uspecs)
        counts = np.zeros(elements.shape[0])
        
        for _i,el in enumerate(elements):
            if el in counted_elements:
                counts[_i] = counted_elements[el]
        if normed:
            Z = counts.sum()
            if Z>0:
                counts /= float(counts.sum())
        return counts
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = len(self.elements)
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            h = self._count_elements(i, X, species, uX, uspecies, uidx_neigh, 
                                     elements=self.elements, tol0=self.tol0, 
                                     normed=self.normed)
            
            XP[i,:] = h
        return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class DistanceCosTaperingFeatures_3body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate cosine 3-body features which taper based on products of
    2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = sum_{j,k \ne i} cos(kappa_t * theta_{ijk}) * cos(kappa_r*r_{ji})*f(r_ji)
          cos(kappa_r*r_{ki})*f(r_ki)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    kappa_t : int, float, list, tuple or np.ndarray of length N
    
    kappa_r : int, float, list, tuple or np.ndarray of length N
                
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DcTF3_all = DistanceCosTaperingFeatures_3body(element_filter=all_filter,
                                            taper_fun=taper_fun, kappa_t=[0,1], kappa_r=[1,1],)

    >>> Phi_all = DcTF3_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)

                                         
    Literature
    ----------
    Behler, J., Constructing high-dimensional neural network potentials: A tutorial review,
        International Journal of Quantum Chemistry, 2015, 115 (16), 1032-1050
    
    """
    
    name = "DistanceCosTaperingFeatures_3body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None, kappa_t=0.,
                 kappa_r=1., return_force=False, tol0=1e-6, **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
            
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
                
        self.kappa_t = make_array(kappa_t)
        self.kappa_r = make_array(kappa_r)
        
        self.N = self.kappa_t.shape[0] * self.kappa_r.shape[0]**2
        self.tol0 = tol0
        self.return_force = return_force
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, kappa_t=[0.],
                    kappa_r=[1.], tol0=1e-6, return_force=False):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        Nt, Nr = kappa_t.shape[0], kappa_r.shape[0]

        # current atom
        atom = X[i,:]
        spec = species[i]
                    
        #neighboring atoms
        uidx = idx_neigh[i]
        
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
            
        uatoms = uX[uidx]
        
        # filter for valid distances
        r = np.linalg.norm(uatoms-atom,axis=1)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        # in case no neighbors remain
        if len(uidx) == 0:
            if return_force:
                return np.zeros(Nt*Nr*Nr), np.zeros((Nt*Nr*Nr,3))
            return np.zeros(Nt*Nr*Nr), None
        
        uspecs = uspecies[uidx]
        n_neigh = len(uidx)
        
        i0,i1 = np.triu_indices(n_neigh,k=1)
        
        dr0 = uatoms[i0]-atom
        dr1 = uatoms[i1]-atom
        dr = uatoms[i0]-uatoms[i0]
        
        r = np.linalg.norm(dr,axis=1)
        r0 = np.linalg.norm(dr0,axis=1)
        r1 = np.linalg.norm(dr1,axis=1)
        
        r0dotr1 = np.einsum("nd,nd->n",dr0,dr1)
        assert not (np.isclose(r0,0)).any() and not (np.isclose(r1,0)).any(), "DAMMIT!"
        cos_theta = r0dotr1/(r0*r1)          
        
        # taking care of floating precision errors where theta_arg may be -1 - 1e-42 causing NaNs
        isminus1 = np.where(cos_theta - (-1)<0)[0]
        cos_theta[isminus1] = -1
        # taking care of floating precision errors where theta_arg may be 1 + 1e-42 causing NaNs
        is1 = np.where(cos_theta - 1>0)[0]
        cos_theta[is1] = 1 
        theta = np.arccos(cos_theta)
        
        if return_force:
            dr0_n = (dr0.T / r0)
            dr1_n = (dr1.T / r1)
            
            c0 = 1. / (r0*r1)
            c1 = (-r0/r1 * dr1.T - r1/r0 * dr0.T)
            #print("c0 ",c0.shape," dr0.T ",dr0.T.shape," dr1.T ",dr1.T.shape," cos_theta ",cos_theta.shape," c1 ",c1.shape)
            theta_der = c0 * (-dr0.T - dr1.T - cos_theta * c1)
                
        f0 = taper_fun(r0)
        f1 = taper_fun(r1)
        fall = f0*f1
        if return_force:
            f0_der = misc.derivative(taper_fun, r0, dx=1., n=1) * 1./r0 * dr0.T
            f1_der = misc.derivative(taper_fun, r1, dx=1., n=1) * 1./r1 * dr1.T
            vals_f = np.zeros((3,Nt*Nr*Nr))
            
        vals = np.zeros(Nt*Nr*Nr)
        
        ijk = itertools.product(range(Nt),range(Nr),range(Nr))
        for i, (_i, _j, _k) in enumerate(ijk):
            g0 = np.cos(kappa_t[_i]*theta) 
            g1 = np.cos(kappa_r[_j]*r0) 
            g2 = np.cos(kappa_r[_k]*r1)
            g = g0*g1*g2 * fall
            vals[i] = g.sum()            
            
            if return_force:
                
                g0_der = - kappa_t[_i] * np.sin(kappa_t[_i]*theta) * theta_der
                g1_der = - kappa_r[_j] * np.sin(kappa_r[_j]*r0) * dr0_n
                g2_der = - kappa_r[_k] * np.sin(kappa_r[_k]*r1) * dr1_n
                
                g_f = g0_der * g1*g2*f0*f1 + g0 * g1_der * g2*f0*f1 +\
                    g0*g1 * g2_der *f0*f1 + g0*g1*g2 * f0_der *f1 +\
                    g0*g1*g2*f0 * f1_der
                
                vals_f[:,i] = g_f.sum(axis=1)
        
        if return_force:
            return vals, vals_f.T
        return vals, None
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((self.n_output_features_, n_samples*3), dtype=X.dtype)
            print("XP ",XP.shape," XP_f ",XP_f.shape)
        
        for i in range(n_samples):
            G, Gf = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                 element_filter=self.element_filter, 
                                 taper_fun=self.taper_fun, kappa_t=self.kappa_t, 
                                 kappa_r=self.kappa_r, tol0=self.tol0, return_force=self.return_force)
            
            assert np.isfinite(G).all(), "G contains infs or NaNs!"
            XP[i,:] = G

            if self.return_force:
                assert np.isfinite(Gf).all(), "Gf contains infs or NaNs!"
            
            if self.return_force:
                XP_f[:,3*i:3*(i+1)] = Gf
        if self.return_force:
            return XP, XP_f.T
        return XP, None
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh)

class DistanceGaussTaperingFeatures_2body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate cosine features which taper based on 2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = sum_j np.exp(p_i*(r_ij-g_i))*f(r_ij)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    p_ : int, float, list, tuple or np.ndarray of length N
    
    q_ : int, float, list, tuple or np.ndarray of length N
                
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.
    
    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DGTF_all = DistanceGaussTaperingFeatures_2body(element_filter=all_filter,
                                            taper_fun=taper_fun, p_=[1., .5], q_=[0., 1.])

    >>> Phi_all = DGTF_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)
    
    """
    
    name = "DistanceGaussTaperingFeatures_2body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None,\
                 p_=.5, q_=0., return_force=False, emb_density_funs=None,\
                 taper_fun_emb=None, tol0=1e-6, **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
        
        self.tol0 = tol0
        
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
        
        self.p_ = make_array(p_)
        self.q_ = make_array(q_)
        self.N = self.p_.shape[0]*self.q_.shape[0]
        self.return_force = return_force  
        
        self.emb_density_funs = emb_density_funs
        self.taper_fun_emb = taper_fun_emb
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, p_=[.5], q_=[0.],
                    return_force=False, emb_density_funs=None, taper_fun_emb=None,
                    rhos=None, urhos=None, tol0=1e-6):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        do_emb = False # whether or not to compute the embedding density (in the case of EAM)
        do_taper_emb = False # whether or not to taper the embedding density
        if not emb_density_funs is None:
            do_emb = True
            do_taper_emb = not taper_fun_emb is None
            assert isinstance(emb_density_funs,dict), "Expected the provided 'emb_density_funs' parameter to be a dict."
            assert all([callable(v) for v in emb_density_funs.values()]), "Expected the provided the values of the 'emb_density_funs' parameter to be callable."
            assert all([not rhos is None, not urhos is None]), "In order to compute embedding density based features 'rhos' and 'urhos' need to be provided."
            
            species_order = sorted(list(emb_density_funs))
            
        if all([not rhos is None, not urhos is None]) and not do_emb:
            raise ValueError("'rhos' and 'urhos' are given but not 'emb_density_funs'!")
                            
        # current atom
        atom = X[i,:]
        spec = species[i]

        #neighboring atoms
        uidx = idx_neigh[i]
        assert len(uidx) > 0, "No neighbors available for current atom!"
            
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
        if do_emb:
            assert isinstance(element_filter,str), "For embedding density features the parameter 'element_filter' needs to be a string! Given: {}".format(element_filter)
            emb_spec = element_filter # in case of embedding density the element_filter is just a string

        uatoms = uX[uidx]
        uspecs = uspecies[uidx]
        if do_emb:
            udens = urhos[uidx]
        n_neigh = len(uidx)

        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        
        # re-do neighborhood for valid r (just to be safe)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        
        dr = dr[r_valid]
        r = r[r_valid]
        if do_emb:
            udens = udens[r_valid]
        n_neigh = len(uidx)
        
        if do_emb:
            dens = rhos[i]
            
        if do_emb:
            
            if do_taper_emb:
                udens_f = taper_fun_emb(udens)
                dens_f = taper_fun_emb(dens)
                
            else:
                taper_fun_emb = lambda x: 1. if isinstance(x,float) else np.ones(x.shape)
                udens_f = taper_fun_emb(udens)
                dens = taper_fun_emb(dens)
        
        f = taper_fun(r)
        Np = len(p_)
        Nq = len(q_)
                
        if return_force:
            
            if do_emb:
                dens_fp = misc.derivative(taper_fun_emb, dens, dx=1, n=1)
                udens_fp = misc.derivative(taper_fun_emb, udens, dx=1, n=1)
            
            fp = misc.derivative(taper_fun, r, dx=1, n=1)
            dr_normed = (dr.T/r)
        
        if do_emb: # embedding density version
            #Nspec = len(species_order)
            #vals = np.zeros(Np*Nq*Nspec)
            #forces = np.zeros((3,Np*Nq*Nspec))        
            #idx_dict_species = {k:np.arange(Np*Nq*v, Np*Nq*(v+1)) for v,k in enumerate(species_order)}
            
            forces = np.zeros((3,Np*Nq))
            vals = np.zeros(Np*Nq)
            _forces = np.zeros((3,Np*Nq))
            _vals = np.zeros(Np*Nq)
            
            if spec == emb_spec:
                # current atom - looping all basis functions
                for _ip in range(Np):
                    for _iq in range(Nq):

                        c = np.exp(-p_[_ip]*(dens-q_[_iq])**2)
                        _vals[_ip*Nq+_iq] = c*dens_f

                        if return_force:

                            cp = -p_[_ip]*2*(dens-q_[_iq])*np.exp(-p_[_ip]*(dens-q_[_iq])**2)
                            _f = (cp*dens_f+c*dens_fp) * fp * dr_normed

                            _forces[:,_ip*Nq+_iq] = _f.sum(axis=1)
            
            if return_force:
                #forces[:,idx_dict_species[spec]] += _forces
                forces += _forces
            #vals[idx_dict_species[spec]] += _vals
            vals += _vals
            
            if return_force: # neighborhood contributions
                
                # looping all neighbors
                for i_el, _spec in enumerate(uspecs):
                    if _spec != emb_spec:
                        continue
                    _forces[:,:] = 0
                    
                    _dens = udens[i_el]
                    _dens_f = udens_f[i_el]
                    
                    _dens_fp = udens_fp[i_el]
                    _fp = fp[i_el]
                    
                    # looping all basis functions
                    for _ip in range(Np):
                        for _iq in range(Nq):

                            c = np.exp(-p_[_ip]*(_dens-q_[_iq])**2)
                            cp = -p_[_ip]*2*(_dens-q_[_iq])*np.exp(-p_[_ip]*(_dens-q_[_iq])**2)

                            _f = (cp*_dens_f+c*_dens_fp) * _fp * dr_normed[:,i_el]
                            _forces[:,_ip*Nq+_iq] = _f
                        
                    #forces[:,idx_dict_species[_spec]] += _forces
                    forces += _forces
                    
        else: # pair distance version
            
            vals = np.zeros(Np*Nq)
            forces = np.zeros((3,Np*Nq))        
            
            for _ip in range(Np):
                for _iq in range(Nq):
                    c = np.exp(-p_[_ip]*(r-q_[_iq])**2)
                    vals[_ip*Nq+_iq] = (c*f).sum()

                    if return_force:

                        cp = -2*p_[_ip]*(r-q_[_iq])*np.exp(-p_[_ip]*(r-q_[_iq])**2)
                        _f = (cp*f + c*fp) * dr_normed

                        forces[:,_ip*Nq+_iq] = _f.sum(axis=1)
        
        if return_force:
            return vals, forces
        else:
            return vals
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        #if not self.emb_density_funs is None:
        #    self.n_output_features_ = self.N*len(self.emb_density_funs)
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh, rhos=None, urhos=None):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((3*n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            if self.return_force:
                G, G_f = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, 
                                     taper_fun=self.taper_fun, p_=self.p_, q_=self.q_,
                                     return_force=self.return_force, 
                                     emb_density_funs=self.emb_density_funs, 
                                     taper_fun_emb=self.taper_fun_emb, rhos=rhos,
                                     urhos=urhos, tol0=self.tol0)
                XP_f[3*i:3*(i+1),:] = G_f
            else:
                G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, 
                                     taper_fun=self.taper_fun, p_=self.p_, q_=self.q_,
                                     return_force=self.return_force,
                                     emb_density_funs=self.emb_density_funs, 
                                     taper_fun_emb=self.taper_fun_emb, rhos=rhos,
                                     urhos=urhos, tol0=self.tol0)
            
            XP[i,:] = G
        if self.return_force:
            return XP, XP_f
        else:
            return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, rhos=None,\
                      urhos=None, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh, rhos=rhos, urhos=urhos)

class DistanceSineCosineTaperingFeatures_2body(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Generate sine + cosine features which taper based on 2-body distances.

    Generate a design matrix using a distance tapering function f
    for individual atoms:
    
    G_i = sum_j sin(kappa*r_ij)*f(r_ij) + cos(kappa*r_ij)*f(r_ij)

    Parameters
    ----------
    taper_fun : callable
        Custom tapering function.
        
    taper_type : str
        Type of tapering function to be used.
        
    kappa_ : int, float, list, tuple or np.ndarray of length N
                
    element_filter : callable, optional, default None
        Function to filter neighboring elements. Has to return True
        to keep the element and False to remove it.
    
    tol0 : float, optional, default 1e-6
        Minimum distance for r>0 checks to prevent the appearance of
        unphysical distances of, for example, 1e-22.

    Example
    --------
    >>> all_filter = lambda s,s_ref: np.array([True for v in range(s.shape[0])])
    >>> taper_fun_params = dict(a=r_cut,b=1)
    >>> taper_fun = taper_fun_wrapper(_type="x4le",**taper_fun_params)

    >>> DscTF_all = DistanceSineCosineTaperingFeatures_2body(element_filter=all_filter,
                                            taper_fun=taper_fun, kappa_=[1.,2.])

    >>> Phi_all = DscTF_all.fit_transform(positions, species, upositions, uspecies, 
                                   uindices_neigh)
    
    """
    
    name = "DistanceSineCosineTaperingFeatures_2body"
    implemented_taper_fun_types = dict()
    
    def __init__(self, taper_type=None, taper_fun=None, element_filter=None,\
                 kappa_=1, return_force=False, emb_density_funs=None,\
                 taper_fun_emb=None, tol0=1e-6, **kwargs):
        
        assert not all([taper_type is None, taper_fun is None]), "Either taper_type or taper_fun need to be provided."
        
        if not taper_fun is None:
            assert callable(taper_fun), "Given taper_fun parameter needs to be callable."
            taper_type = "custom"
        elif not taper_type is None:
            assert taper_type in self.implemented_taper_fun_types, "Given taper_type ('{}') not understood.".format(taper_type)
            taper_fun = self.implemented_taper_fun_types[taper_type]
        
        self.tol0 = tol0
        
        self.taper_type = taper_type
        self.taper_fun = taper_fun
        self.element_filter = element_filter
        
        self.kappa_ = make_array(kappa_)
        self.kappa_ = np.hstack((self.kappa_, self.kappa_))
        self.N = self.kappa_.shape[0]
        self.return_force = return_force  
        
        self.emb_density_funs = emb_density_funs
        self.taper_fun_emb = taper_fun_emb
        
    @staticmethod
    def _get_Gvalue(i, X, species, uX, uspecies, idx_neigh, 
                    element_filter=None, taper_fun=None, kappa_=[1.], 
                    return_force=False, emb_density_funs=None, taper_fun_emb=None,
                    rhos=None, urhos=None, tol0=1e-6):
        """Computes G values.
        """
        assert callable(taper_fun), "taper_fun needs to be callable."
        
        do_emb = False # whether or not to compute the embedding density (in the case of EAM)
        do_taper_emb = False # whether or not to taper the embedding density
        if not emb_density_funs is None:
            do_emb = True
            do_taper_emb = not taper_fun_emb is None
            assert isinstance(emb_density_funs,dict), "Expected the provided 'emb_density_funs' parameter to be a dict."
            assert all([callable(v) for v in emb_density_funs.values()]), "Expected the provided the values of the 'emb_density_funs' parameter to be callable."
            assert all([not rhos is None, not urhos is None]), "In order to compute embedding density based features 'rhos' and 'urhos' need to be provided."
            
            species_order = sorted(list(emb_density_funs))
            
        if all([not rhos is None, not urhos is None]) and not do_emb:
            raise ValueError("'rhos' and 'urhos' are given but not 'emb_density_funs'!")
                            
        # current atom
        atom = X[i,:]
        spec = species[i]

        #neighboring atoms
        uidx = idx_neigh[i]
        assert len(uidx) > 0, "No neighbors available for current atom!"
            
        if callable(element_filter):
            uidx = uidx[element_filter(uspecies[uidx],spec)]
        if do_emb:
            assert isinstance(element_filter,str), "For embedding density features the parameter 'element_filter' needs to be a string! Given: {}".format(element_filter)
            emb_spec = element_filter # in case of embedding density the element_filter is just a string

        uatoms = uX[uidx]
        uspecs = uspecies[uidx]
        if do_emb:
            udens = urhos[uidx]
        n_neigh = len(uidx)

        dr = uatoms-atom
        r = np.linalg.norm(dr,axis=1)
        
        # re-do neighborhood for valid r (just to be safe)
        r_valid = np.where(r>tol0)[0]
        uidx = uidx[r_valid]
        
        dr = dr[r_valid]
        r = r[r_valid]
        if do_emb:
            udens = udens[r_valid]
        n_neigh = len(uidx)
        
        if do_emb:
            dens = rhos[i]
            
        if do_emb:
            
            if do_taper_emb:
                udens_f = taper_fun_emb(udens)
                dens_f = taper_fun_emb(dens)
                
            else:
                taper_fun_emb = lambda x: 1. if isinstance(x,float) else np.ones(x.shape)
                udens_f = taper_fun_emb(udens)
                dens = taper_fun_emb(dens)
        
        f = taper_fun(r)
        N = len(kappa_)
        Nc = Ns = N//2
                
        if return_force:
            
            if do_emb:
                dens_fp = misc.derivative(taper_fun_emb, dens, dx=1, n=1)
                udens_fp = misc.derivative(taper_fun_emb, udens, dx=1, n=1)
            
            fp = misc.derivative(taper_fun, r, dx=1, n=1)
            dr_normed = (dr.T/r)
        
        if do_emb: # embedding density version
            #Nspec = len(species_order)
            #vals = np.zeros(N*Nspec)
            #forces = np.zeros((3,N*Nspec))        
            #idx_dict_species = {k:np.arange(N*v,N*(v+1)) for v,k in enumerate(species_order)}
            
            #_forces = np.zeros((3,N))
            #_vals = np.zeros(N)
            forces = np.zeros((3,N))
            vals = np.zeros(N)
            _forces = np.zeros((3,N))
            
            if not spec == emb_spec:
                # current atom - looping all basis functions
                for _i in range(N):
                    
                    if _i<Nc:
                        c = np.cos(kappa_[_i]*dens)
                    else:
                        c = np.sin(kappa_[_i]*dens)
                    vals[_i] = c*dens_f

                    if return_force:
                        
                        if _i<Nc:
                            cp = -kappa_[_i]*np.sin(kappa_[_i]*dens)
                        else:
                            cp = kappa_[_i]*np.cos(kappa_[_i]*dens)
                            
                        _f = (cp*dens_f+c*dens_fp) * fp * dr_normed

                        _forces[:,_i] = _f.sum(axis=1)
            
            if return_force:
               #forces[:,idx_dict_species[spec]] += _forces
               forces += _forces
            #vals[idx_dict_species[spec]] += _vals
            
            if return_force:
                
                # looping all neighbors
                for i_el, _spec in enumerate(uspecs):
                    if _spec != emb_spec:
                        continue
                    _forces[:,:] = 0
                    
                    _dens = udens[i_el]
                    _dens_f = udens_f[i_el]
                    
                    _dens_fp = udens_fp[i_el]
                    _fp = fp[i_el]
                    
                    # looping all basis functions
                    for _i in range(N):
                        
                        if _i<Nc:
                            c = np.cos(kappa_[_i]*_dens)
                            cp = -kappa_[_i]*np.sin(kappa_[_i]*_dens)
                        else:
                            c = np.sin(kappa_[_i]*_dens)
                            cp = kappa_[_i]*np.cos(kappa_[_i]*_dens)                        
                        
                        _f = (cp*_dens_f+c*_dens_fp) * _fp * dr_normed[:,i_el]
                        _forces[:,_i] = _f
                        
                    #forces[:,idx_dict_species[_spec]] += _forces
                    forces += _forces
                    
        else: # pair distance version
            
            vals = np.zeros(N)
            forces = np.zeros((3,N))        
            
            for i in range(N):
                
                if i<Nc:
                    c = np.cos(kappa_[i]*r)
                else:
                    c = np.sin(kappa_[i]*r)
                vals[i] = (c*f).sum()

                if return_force:
                    
                    if i<Nc:
                        cp = -kappa_[i]*np.sin(kappa_[i]*r)
                    else:
                        cp = kappa_[i]*np.cos(kappa_[i]*r)
                    _f = (cp*f + c*fp) * dr_normed

                    forces[:,i] = _f.sum(axis=1)
        
        if return_force:
            return vals, forces
        else:
            return vals
        
    def fit(self, X, species, uX, uspecies, idx_neigh, y=None):
        """Compute number of output features.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX
            
        Returns
        -------
        self : instance
        """
        n_samples, n_features = utils.check_array(X).shape
        self.n_input_features_ = n_features
        
        self.n_output_features_ = self.N
        #if not self.emb_density_funs is None:
        #    self.n_output_features_ = self.N*len(self.emb_density_funs)
        return self
    
    def transform(self, X, species, uX, uspecies, uidx_neigh, rhos=None, urhos=None):
        """Computes the histograms.

        Parameters
        ----------
        X : array-like, shape (n_atoms, 3)
            The supercell atom positions.
        
        species : np.ndarray of str of shape (n_atoms,)
            atom species in the supercell
        
        uX : array-like, shape (n_ultra_atoms, 3)
            The ultracell atom positions.
            
        uspecies : np.ndarray of str of shape (n_atoms,)
            atom species in the ultracell
        
        idx_neigh : list of np.ndarrays of int
            indices connecting each atom in X with its neighbors in uX

        Returns
        -------
        XP : np.ndarray of shape (n_samples, n_output_features)
            The design matrix.

        Note
        ----
        Requires prior execution of self.fit.
        """
        sklearn.utils.validation.check_is_fitted(self, ['n_input_features_', 'n_output_features_'])

        X = sklearn.utils.validation.check_array(X, dtype=sklearn.utils.validation.FLOAT_DTYPES)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape")

        # allocate output data
        XP = np.empty((n_samples, self.n_output_features_), dtype=X.dtype)
        if self.return_force:
            XP_f = np.empty((3*n_samples, self.n_output_features_), dtype=X.dtype)
        
        for i in range(n_samples):
            if self.return_force:
                G, G_f = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, 
                                     taper_fun=self.taper_fun, kappa_=self.kappa_, 
                                     return_force=self.return_force, 
                                     emb_density_funs=self.emb_density_funs, 
                                     taper_fun_emb=self.taper_fun_emb, rhos=rhos,
                                     urhos=urhos, tol0=self.tol0)
                XP_f[3*i:3*(i+1),:] = G_f
            else:
                G = self._get_Gvalue(i, X, species, uX, uspecies, uidx_neigh,  
                                     element_filter=self.element_filter, 
                                     taper_fun=self.taper_fun, kappa_=self.kappa_, 
                                     return_force=self.return_force,
                                     emb_density_funs=self.emb_density_funs, 
                                     taper_fun_emb=self.taper_fun_emb, rhos=rhos,
                                     urhos=urhos, tol0=self.tol0)
            
            XP[i,:] = G
        if self.return_force:
            return XP, XP_f
        else:
            return XP
    
    def fit_transform(self, X, species, uX, uspecies, uidx_neigh, rhos=None,\
                      urhos=None, **kwargs):
        """Calls fit and transform on X, species, uX, uspecies, uidx_neigh.
        """
        self.fit(X, species, uX, uspecies, uidx_neigh)
        return self.transform(X, species, uX, uspecies, uidx_neigh, rhos=rhos, urhos=urhos)