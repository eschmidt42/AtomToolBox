import copy, os, pickle, collections, warnings

from scipy import stats, optimize
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import KMeans
from scipy.cluster import hierarchy
from scipy import spatial
import matplotlib.pylab as plt
import matplotlib as mpl

import tensorflow as tf
import sklearn
from time import time
from functools import partial
from datetime import datetime

def scipy_gmm_wrapper(gmm=None, means_=None, covariances_=None, weights_=None):
    """Wraps the gmm so it's pdf can be accessed.
    """
    gmm_given = (not gmm is None)
    parameters_given = not (means_ is None and covariances_ is None and weights_ is None)
    assert gmm_given or parameters_given,\
        "Either 'gmm' needs to be given or 'means_', 'covariances_' and 'weights_'!"
    
    if gmm_given:
        means_ = gmm.means_
        covariances_ = gmm.covariances_
        weights_ = gmm.weights_
    elif parameters_given:
        assert len(means_)==len(covariances_)==len(weights_), "'means_', 'covariances_' and 'weights_' all need to have the same length!"
    else:
        raise ValueError("WTF!")
    
    gaussians = [stats.multivariate_normal(mean=means_[i],cov=covariances_[i])
                 for i in range(weights_.shape[0])]

    def scipy_gmm(x):
        
        if len(x.shape)==1:
            Phi = np.array([g.pdf(x) for g in gaussians])
        elif len(x.shape)==2:
            Phi = np.array([[g.pdf(_x) for g in gaussians] for _x in x])
        else:
            raise ValueError("Cannot handle 'x' of shape {}!".format(x.shape))
        return np.dot(Phi,weights_)

    return scipy_gmm

def get_decomposed_models(in_models, verbose=False):
    """Decomposes 'models'.

    Parameters
    ----------
    in_models : dict of lists
        Example: {"a":[0,2], "b":[0,1]}
    verbose : boolean, optional, default False
    
    Returns
    -------
    out_models : list of np.ndarrays of int
        Each np.ndarray represents a maximally decomposed model
        breaking down in_models.
    
    Example:
    >>> in_model = {"a":[0,2], "b":[0,1]}
    >>> print(get_decomposed_models(in_model))
    [array([2]), array([1]), array([0])]
    """
    if verbose:
        print("in_models ",in_models)
    out_models = []
    num_models = len(in_models)
    assert num_models > 0, "number of models = {}!".format(num_models)
    
    out_models = collections.deque([set(v) for v in in_models.values()])
    
    if num_models > 1:
        unique_components = np.sort(np.unique(np.hstack(in_models.values())))
        if verbose:
            print("unique_components ",unique_components)
        for i, uc in enumerate(unique_components):
            if verbose:
                print("\nunique component: ",uc)
                print("out_models ",out_models)
            
            # splits out_models into relevant and relevant
            # relevant models will be decomposed and the remainders
            # added with relevant_models to proce new out_models
            relevant_models = []
            irrelevant_models = []
            while len(out_models)>0:
                m = out_models.popleft()
                if verbose:
                    print("    m ",m)
                if uc in m:
                    relevant_models.append(m)
                else:
                    irrelevant_models.append(m)
            if verbose:
                print("    > relevant ",relevant_models)
                print("    > irrelevant ",irrelevant_models)
                
            # sort relevant models by length
            relevant_models = sorted(relevant_models, key = lambda x: len(x))
            
            if len(relevant_models)==0 and len(irrelevant_models)==0: # error
                raise ValueError("Relevant and irrelevant models are surprisingly empty!")
            
            elif len(relevant_models)==0: # no model was relevant
                out_models = collections.deque(irrelevant_models)
            
            elif len(relevant_models)==1: # a single model was relevant
                out_models = collections.deque(relevant_models+irrelevant_models)
            
            else: # there were some relevant models
                i0, i1 = np.triu_indices(len(relevant_models),k=1)
                
                # all intersections have the unique component in common
                intersections = [relevant_models[v0].intersection(relevant_models[v1]) for v0,v1 in zip(i0,i1)]
                assert len(intersections)>0, "The intersections are surprisingly empty!"
                
                # sorting by number of model components
                intersections = sorted(intersections, key=lambda x: len(x))
                if verbose:
                    print("    > intersections ",intersections)
                
                # the smallest possible set of components
                intersection = intersections[0]
                assert len(intersection)>0, "The intersection is surprisingly empty!"
                if verbose:
                    print("    > intersection ",intersection)
                
                #retrieving remainders
                remainders = [m.difference(intersection) for m in relevant_models]
                remainders = [v for v in remainders if len(v)>0]
                if verbose:
                    print("    > intersection ",intersection)
                
                # composing out_models again
                if verbose:
                    print("    > irrelevant_models ",irrelevant_models)
                    print("    > remainders ",remainders)
                out_models = collections.deque(irrelevant_models + [intersection] + remainders)
                if verbose:
                    print("    > new out_models ",out_models)
            
    out_models = [np.array(list(v),dtype=int) for v in out_models]
    
    assert set([v0 for v1 in in_models.values() for v0 in v1]) == set(np.hstack(out_models)), "Mismatching sets, lost some components!"
    return out_models

def fit_gmm_weights(w0,_gaussians,X,method="Nelder-Mead"):
    def wbound(w):
        w = np.absolute(w)
        return w/w.sum()

    def _wrap(_gaussians,X):
        _g = np.array([[g.pdf(x) for g in _gaussians] for x in X])
        def fun(w):
            return - (np.log(np.dot(_g,wbound(w)))).sum()
        return fun
    
    res = optimize.minimize(_wrap(_gaussians,X),w0,method=method)
    return wbound(res["x"])

class GaussianMixtureClassifier:
    """Approximates the sample distribution for classification.
    
    GaussianMixture and BayesianGaussianMixture as implemented in sklearn
    classify by individual Gaussian components found during the density 
    regression of the given samples. In this class the training set is 
    split by class and approximates the entire density distributions for 
    each class. The resulting pdfs are then used for classification.
    This class can also do decomposition of Gaussian Mixture Models (GMMs).
    
    The decomposition is triggered when GaussianMixtureClassifier.fit
    is passed an X that is a list of np.ndarrays. It is then assumed that each
    array contains a superposition of underlying distributions (which may each be 
    larger than a single Gaussian), e.g. comparison of LAMMPS trajectories 
    of different crystals.
    
    Parameters
    ----------

    gmm : instance of GaussianMixture or BayesianGaussianMixture, optional, default None
        Required to regress data.
        
    load_path : str, optional default None
        To load results of a previous regression.

    check_labels : boolean, optional, default True
        Checks upon loading the classifier from disk whether the number of 
        GMM models matches the number of known labels or not.

    Methods
    -------

    fit : fit(X, y)
        Approximates the density distributions using the provided gmm.
        X : float np.ndarray of shape (N, M)
        y : int np.ndarray of shape (N,)
    
    predict_proba : predict_proba(X)
        Requires previous execution of 'fit'. Returns the probability 
        for the given samples to belong to either of the classes.
        
    predict : predict(X, show=False, axes=[0,1])
        Requires previous execution of 'fit'. Returns most probably class
        for all samples. Optionally can also plot the classification along
        two axes.

    pdf : pdf(X, label)
        Requires previous execution of 'fit'. Allows access to the 
        pdf for a specified class.
        label : int
        
    cluster : dict
        Settings for the clustering/decomposition of Gaussian. See self.decompos_gmms 
        for more detail.
        
    weights_fit_method : str, optional, default "Nelder-Mead"
        Specifies the scipy.optimize.minimize method to use for the optimization of
        GMM weights.
    """
    
    fitted_gmms = dict()
    labels = None
    idx_class = None
    
    weights_ = dict() # keys are integers. correspond to order of 'appearances' (a list)
    covariances_ = dict() # keys are integers. correspond to order of 'appearances' (a list)
    means_ = dict() # keys are integers. correspond to order of 'appearances' (a list)
    label_map = None
    check_labels = True
    appearances = None # in which original given crystals centroids were observed
    
    default_cluster_kwargs = {"method":"average","metric":"euclidean","cluster_parameters":"mu",
                            "threshold":1e-6,"criterion":"distance","combine":"mean"}
    
    def __init__(self, gmm=None, load_path=None, tol0=1e-6, cluster = default_cluster_kwargs,
                 weights_fit_method="Nelder-Mead", verbose=False, check_labels=False):
        if not gmm is None:
            assert isinstance(gmm,(GaussianMixture, BayesianGaussianMixture)), "'gmm' needs to be an instance of sklearn.mixture.{GaussianMixture, BayesianGaussianMixture}!"
        elif isinstance(load_path,str):
            assert os.path.exists(load_path), "Given 'load_path' ({}) invalid!".format(load_path)
        else:
            raise ValueError("Either 'gmm' or 'load_path' needs to be given!")
            
        self.gmm = gmm
        self.check_labels = check_labels
                
        if gmm is None:
            self._load_parameters(load_path)
            
        # decomposition related parameters
        self.tol0 = tol0
        for _k, _v in self.default_cluster_kwargs.items():
            if not _k in cluster:
                cluster[_k] = _v
        self.cluster = cluster
        self.weights_fit_method = weights_fit_method
        
        # misc
        self.verbose = verbose
    
    def _load_parameters(self,load_path):
        with open(load_path,"rb") as f:
            params = pickle.load(f)
        
        for label in sorted(params["weights_"]):
            self.fitted_gmms[label] = scipy_gmm_wrapper(weights_=params["weights_"][label],\
                                                        covariances_=params["covariances_"][label],\
                                                        means_=params["means_"][label])
        
        self.weights_ = params["weights_"]
        self.covariances_ = params["covariances_"]
        self.means_ = params["means_"]
        self.label_map = params["label_map"]
        if "appearances" in params:
            self.appearances = params["appearances"]
        if self.check_labels:
            assert not self.label_map is None, "No labels are given in the stored file!"
            assert len(self.fitted_gmms) == len(self.label_map), "The number of GMMs (%i) does not match the number of available labels (%i)!" % (len(self.fitted_gmms),len(self.label_map))
            
    def save(self,save_path):
        params = {"weights_":self.weights_,
                  "covariances_":self.covariances_,
                  "means_":self.means_,
                  "label_map":self.label_map,
                  "appearances":self.appearances}
        with open(save_path,"wb") as f:
            pickle.dump(params,f)
    
    def fit(self, X, y=None, label_map=None):
        """Fits.
        
        Parameters
        ----------
        X : np.ndarray of floats or list of np.ndarrays
        
        Notes
        -----
        If X is an array then y needs to be given and the classifier is developed
        directly over all the samples. If, otherwise, X is a list of arrays then
        y is not needed and labels are generated by decomposing the arrays in X by 
        comparison of their computed pdfs.
        """
        X_is_array = isinstance(X,np.ndarray)
        X_is_list = isinstance(X,list)
        if X_is_array:
            assert isinstance(y,np.ndarray), "'X' is an array and thus 'y' needs to be an array!"
            assert y.shape[0]==X.shape[0], "The array 'X' of shape {} is not matched by 'y' of shape {}!".format(X.shape,y.shape)
        elif X_is_list:
            assert all([isinstance(x,np.ndarray) for x in X]), "'X' is a list and all its entries need to be np.ndarrays! Got: {}".format([type(x) for x in X])
            assert len(set([x.shape[1] for x in X]))==1, "All arrays in 'X' need to have the same number of features! Got: {}".format([x.shape[1] for x in X])
            n_features = X[0].shape[1]
        else:
            raise ValueError("'X' input not understood. Needs to be an array of list of arrays!")
        
        if X_is_array:
            self.label_map=label_map
            self.labels = np.unique(y)
            self.idx_class = {k: np.where(y==k)[0] for k in self.labels}
            for label in sorted(self.idx_class):

                _gmm = copy.deepcopy(self.gmm)
                _gmm.fit(X[self.idx_class[label],:])

                self.fitted_gmms[label] = scipy_gmm_wrapper(gmm=_gmm)   

                self.weights_[label] = _gmm.weights_
                self.covariances_[label] = _gmm.covariances_
                self.means_[label] = _gmm.means_
        
        elif X_is_list:
            
            Nstructures = len(X)
            structure_ids = ["structure%s"%i for i in range(Nstructures)]
            
            # fit structure pdfs
            all_gmms = [copy.deepcopy(self.gmm).fit(X[i]) for i in range(Nstructures)]
            
            # compressed and decomposed (cd) model            
            mus_cd, covs_cd, appearances = self.decompose_gmms(all_gmms, structure_ids, n_features,
                                                  method=self.cluster["method"], metric=self.cluster["metric"], 
                                                  threshold=self.cluster["threshold"], criterion=self.cluster["criterion"],
                                                  cluster_parameters=self.cluster["cluster_parameters"], combine=self.cluster["combine"],
                                                  verbose=self.verbose)
            N_cd = len(mus_cd)
            self.appearances = appearances.copy()
                        
            # re-fit models
            gmms_cd = []
            _X = np.vstack(X)
            
            if self.verbose:
                print("number of resulting models: ",N_cd)
                print("number of components in total: ",sum(m.shape[0] for m in mus_cd))
            
            for i in range(N_cd):
                
                gaussians = [stats.multivariate_normal(mean=mus_cd[i][j],cov=covs_cd[i][j])
                             for j in range(mus_cd[i].shape[0])]
                
                #check closeness to original models
                is_new_model = True
                for _gmm in all_gmms:
                    
                    if _gmm.means_.shape == mus_cd[i].shape and _gmm.covariances_.shape == covs_cd[i].shape:
                        
                        mu_close = np.allclose(_gmm.means_,mus_cd[i])
                        cov_close = np.allclose(_gmm.covariances_,covs_cd[i])
                        
                        if mu_close and cov_close:
                            is_new_model = False
                            weights_cd = _gmm.weights_
                            break
                                    
                Ng = len(gaussians)
                if is_new_model: # in case it's a new model get new weights
                    if Ng>1:

                        w0 = np.ones(Ng)/float(Ng)
                        weights_cd = fit_gmm_weights(w0, gaussians, _X, method=self.weights_fit_method)

                        assert weights_cd.sum() < 1+1e-6, "Weights invalid!"
                        if np.linalg.norm(w0-weights_cd)<self.tol0:
                            warnings.warn("Weights were not optimized! Changes are smaller than {}.".format(self.tol0))
                    else:
                        weights_cd = np.array([1.])
                                
                # finalize and store models                
                self.fitted_gmms[i] = scipy_gmm_wrapper(means_=mus_cd[i], covariances_=covs_cd[i], weights_=weights_cd)

                self.weights_[i] = weights_cd
                self.covariances_[i] = covs_cd[i]
                self.means_[i] = mus_cd[i]
            
        else:
            raise ValueError("Boink!")
            
    @staticmethod
    def decompose_gmms(gmms, structure_ids, n_features, method="average",
                       metric="euclidean", threshold=1e-6, criterion="distance",
                       cluster_parameters="mu", combine="mean", verbose=False):
        """Decomposes GMMs.
        
        Parameters
        ----------
        gmms : list of GaussianMixtureModel instances
        
        structure_ids : list of str
            Names for the individual structures.
            
        n_features : int
            Number of features.
            
        method : str, optional, default "average"
            Method to use in scipy.cluster.hierarchy.linkage
        
        metric: str, optional, default, "euclidean"
            Metric to use in scipy.cluster.hierarchy.linkage.
        
        threshold : float, optional, default 1e-6
            Threshold to use in scipy.cluster.hierarchy.fcluster. The
            smaller the value the more clusters will be found.
        
        criterion : str, optional, default "distance"
            Criterion to use in scipy.cluster.hierarchy.fcluster.
        
        cluster_parameters : str, optional, default "mu"
            Defines what is to be clustered:
                "mu" : Gaussians are clustered by their mean/mu values.
                "cov": Gaussians are clustered by their covariance values.
                "mu+cov": Gaussians are clustered by both their mean/mu and covariance values.
        
        combine : str, optional, default "mean"
            Specifies how Gaussians found to belong to the same cluster are to be 
            combined. Currently "mean" is the only recognized option.
        """
        
        model_reference = {}
        N = 0
        all_covs = []
        all_mus = []
        
        for i,sid in enumerate(structure_ids):
            model_reference[sid] = np.arange(N,N+gmms[i].weights_.shape[0])
            
            N += gmms[i].weights_.shape[0]
            covs = np.array([c.ravel() for c in gmms[i].covariances_])
            mus = np.array([m for m in gmms[i].means_])
            
            all_covs.append(covs)
            all_mus.append(mus)
            
        all_covs = np.vstack(all_covs)
        all_mus = np.vstack(all_mus)
                
        # find approximately unique components
        if cluster_parameters == "mu":
            p = all_mus
        elif cluster_parameters == "cov":
            p = all_covs
        elif cluster_parameters == "mu+cov":
            p = np.hstack((all_mus, all_covs))
        else:
            raise ValueError("Unexpected 'cluster_parameters' value.")
            
        Z = hierarchy.linkage(p, method=method, metric=metric)
        T = hierarchy.fcluster(Z, threshold, criterion=criterion) -1
        
        # relabel clusters to keep original parameter ordering
        T_map, _c = {}, 0
        for _t in T:
            if not _t in T_map:
                T_map[_t] = _c
                _c += 1
        
        T = np.array([T_map[_t] for _t in T])
        T_set = np.sort(np.unique(T))
        
        # combine parameters and thus compress models
        all_mus_clustered = []
        all_covs_clustered = []
        for t in T_set:
            if combine == "mean":
                _mu = all_mus[T==t,:].mean(axis=0)
                _cov = all_covs[T==t,:].mean(axis=0).reshape((n_features,n_features))
            else:
                raise ValueError("'combine' ({}) not understood!".format(combine))
            all_mus_clustered.append(_mu)
            all_covs_clustered.append(_cov)
        
        all_mus_clustered = np.array(all_mus_clustered)
        all_covs_clustered = np.array(all_covs_clustered)
                
        compressed_model_reference = {sid:np.sort(np.unique(T[vals])) for sid, vals in model_reference.items()}
        if verbose:
            print("model_reference:")
            for sid in sorted(model_reference):
                print("    initial ",model_reference[sid])
                print("    compressed ",compressed_model_reference[sid])
            
        # decompose by comparison
        compressed_decomposed_models = get_decomposed_models(compressed_model_reference, verbose=verbose)
        
        if verbose:
            print("compressed_decomposed_models ",compressed_decomposed_models)
            print("all_mus_clustered ",all_mus_clustered.shape)
            print("all_covs_clustered ",all_covs_clustered.shape)
        
        # compressed and decomposed (cd) parameters
        mus_cd = [all_mus_clustered[m,:] for m in compressed_decomposed_models]
        covs_cd = [all_covs_clustered[m,:] for m in compressed_decomposed_models]
        
        # model origins
        if verbose:
            print("\nMaximally decomposed models:")
            print("Components -> component appearances")

        appearances = []

        for _decomposed_model in compressed_decomposed_models:
            _appearances = [_label for _label,_vals in compressed_model_reference.items()\
                           if any([_val in _decomposed_model for _val in _vals])]
            if verbose:
                print("%s -> %s"%(str(_decomposed_model), ", ".join(sorted(_appearances))))
            appearances.append(_appearances)
            
        return mus_cd, covs_cd, appearances
    
    def predict_proba(self, X):
        p = np.zeros((X.shape[0],len(self.fitted_gmms)))
        
        for i,label in enumerate(sorted(self.fitted_gmms)):
            p[:,i] = self.fitted_gmms[label](X)
            
        Z = p.sum(axis=1)
        return (p.T/Z).T
    
    def show(self, X, y, axes=[0,1], title=None, xlim=(0,1), ylim=(0,1),
             xlabel=None, ylabel=None, labelfs=16, tickfs=14, legendfs=12,
             titlefs=18, data_labels=None, cmap=plt.cm.jet, figsize=(10,5)):
        """Plots."""
        isarray = isinstance(X,np.ndarray)
        islist = isinstance(X,list) and all([isinstance(x,np.ndarray) for x in X])
        if islist:
            _X = np.vstack(X)
        else:
            _X = np.copy(X)

        uy = np.unique(y) if isarray else np.unique(np.hstack(y))
        _y = np.copy(y) if isarray else np.hstack(y)
        
        norm = mpl.colors.Normalize(vmin=uy.min(), vmax=uy.max())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(121) if islist else fig.add_subplot(111)
        ax.set_aspect("equal")
        hs = [None for _uy in uy]
        for i,_uy in enumerate(uy):
            idx = np.where(_y==_uy)[0]
            ax.scatter(_X[idx,axes[0]], _X[idx,axes[1]], label="class %i"%_uy, 
                       alpha=.5, color=cmap(norm(_uy*np.ones(len(idx)))))
            ax.scatter(self.means_[i][:,axes[0]], self.means_[i][:,axes[1]],
                       marker="+", color=cmap(norm(_uy*np.ones(len(self.means_[i])))))
        if xlabel is None:
            ax.set_xlabel("Feature {}".format(axes[0]), fontsize=labelfs)
        else:
            ax.set_xlabel(xlabel, fontsize=labelfs)
        if ylabel is None:
            ax.set_ylabel("Feature {}".format(axes[1]), fontsize=labelfs)
        else:
            ax.set_ylabel(ylabel, fontsize=labelfs)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title("Inferred classes",fontsize=labelfs)
        ax.tick_params(labelsize=tickfs)
        ax.legend(loc=0, fontsize=legendfs)

        if islist:
            ax2 = fig.add_subplot(122)
            ax2.set_aspect("equal")
            norm = mpl.colors.Normalize(vmin=0, vmax=len(X))
            hs = [None for _X in X]
            for i,_X in enumerate(X):
                if data_labels is None:
                    label = "trajectory #%i"%i
                else:
                    label = data_labels[i]
                ax2.scatter(_X[:,axes[0]], _X[:,axes[1]], label=label, c=cmap(norm(np.ones(_X.shape[0])*i)),
                            alpha=.5)
            if xlabel is None:
                ax2.set_xlabel("Feature {}".format(axes[0]), fontsize=labelfs)
            else:
                ax2.set_xlabel(xlabel, fontsize=labelfs)
            if ylabel is None:
                ax2.set_ylabel("Feature {}".format(axes[1]), fontsize=labelfs)
            else:
                ax2.set_ylabel(ylabel, fontsize=labelfs)
            ax2.tick_params(labelsize=tickfs)
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            ax2.set_title("Trajectories", fontsize=labelfs)
            ax2.legend(loc=0, fontsize=legendfs)
        if not title is None:
            plt.suptitle(title, fontsize=titlefs)
        plt.tight_layout()
        plt.show()
                
    
    def predict(self, X, show=False, show_kwargs={}):
        """Predicts.
        """
        
        isarray = isinstance(X,np.ndarray)
        islist = isinstance(X,list) and all([isinstance(x,np.ndarray) for x in X])
        if isarray:
            p = self.predict_proba(X)
            y = np.argmax(p,axis=1)
        elif islist:
            p = [self.predict_proba(x) for x in X]
            y = [np.argmax(_p,axis=1) for _p in p]
        else:
            raise ValueError("X needs to be an array of a list of arrays!")
            
        if show:
            self.show(X,y, **show_kwargs)
        
        return y
    
    def pdf(self, X, label):
        return self.fitted_gmms[label](X)
    
    def explain_inference(self, abbreviate=True):
        assert not any([self.appearances is None, len(self.means_)==0]), "Class instance does not appear to have self.appearances and or self.centroids_ set."
        
        print("\nMaximally decomposed models:")
        print("Components -> component appearances") 
        for i, _appearances in enumerate(self.appearances):
            _decomposed_model = self.means_[i].shape if abbreviate else self.means_[i]
            print("%s -> %s"%(", ".join(sorted(_appearances)),_decomposed_model))
            
    def set_labels(self,labels):
        """Sets labels for the label_map.
        
        Associates model identifier (integer value of self.centroids_)
        with a string, e.g. name for a type of crystal.
        
        Parameters
        ----------
        labels : dict
            Example: {"fcc":0, "fccVac":[1,2]}
            
        Note: 'labels' can also just set strings for a subset of model identifiers
        since not always all names which should be assigned are immediately obvious.
        The remaining labels are set to string versions of integer values.
        """
        assert isinstance(labels,dict)
        assert all([isinstance(_k,str) for _k in labels.keys()])
        assert all([isinstance(_v,(int,list,tuple)) for _v in labels.values()])
        
        if self.label_map is None:
            self.label_map = dict()
        
        values = set()    
        for _k, _v in labels.items():
            self.label_map[_k] = _v
            if isinstance(_v,int):
                values.add(_v)
            else:
                for _v2 in _v:
                    values.add(_v2)
            
        for _k in set(self.means_.keys()).difference(values):
            self.label_map[str(_k)] = _k

def kmeans_wrapper(kmeans=None, centroids_=None):
    """Wraps the kmeans so it can be accessed.
    """
    kmeans_given = (not kmeans is None)
    parameters_given = not (centroids_ is None)
    assert kmeans_given or parameters_given,\
        "Either 'kmeans' needs to be given or 'centroids_'!"
    
    if kmeans_given:
        centroids_ = kmeans.cluster_centers_.copy()
    elif parameters_given:
        pass
    else:
        raise ValueError("WTF!")
    
    def _kmeans(x):
        
        Phi = spatial.distance.cdist(x, centroids_)
        assert Phi.shape == (x.shape[0], centroids_.shape[0])
        
        return Phi.min(axis=1)

    return _kmeans

class KMeansClassifier:
    """Approximates the sample distribution for classification.
    
    KMeans as implemented in sklearn classifies by individual centroids found during the  
    regression of the given samples. In this class the training set is 
    split by class and approximates the sample distributions for 
    each class. The resulting groups of centroids are then used for classification.
    This class can also do decomposition of KMeans models.
    
    The decomposition is triggered when KMeansClasifier.fit
    is passed an X that is a list of np.ndarrays. It is then assumed that each
    array contains a superposition of underlying distributions, e.g. comparison of 
    LAMMPS trajectories of different crystals.
    
    Parameters
    ----------

    kmeans : instance of Kmeans, optional, default None
        Required to regress data.
        
    load_path : str, optional default None
        To load results of a previous regression.

    check_labels : boolean, optional, default True
        Checks upon loading the classifier from disk whether the number of 
        Kmeans models matches the number of known labels or not.

    Methods
    -------

    fit : fit(X, y)
        Approximates the density distributions using the provided centroids.
        X : float np.ndarray of shape (N, M)
        y : int np.ndarray of shape (N,)
    
        
    predict : predict(X, show=False, axes=[0,1])
        Requires previous execution of 'fit'. Returns most probably class
        for all samples. Optionally can also plot the classification along
        two axes.

    pdf : pdf(X, label)
        Requires previous execution of 'fit'. Allows access to the 
        pdf for a specified class.
        label : int
        
    cluster : dict
        Settings for the clustering/decomposition of centroids. See self.decompos_kmeans 
        for more detail.
    
    Example
    -------
    >>> kmc = atb.KMeansClassifier(KMeans(n_clusters=4), cluster={"threshold":.5})
    >>> _D = 2
    >>> _X = [np.vstack((stats.norm.rvs(size=(220,_D), loc=1, scale=1), stats.norm.rvs(size=(200,_D), loc=0, scale=1.))),
        stats.norm.rvs(size=(400,_D), loc=3, scale=1.),stats.norm.rvs(size=(200,_D), loc=3, scale=1.)]
    >>> kmc.fit(_X)
    >>> kmc.explain_inference()
    Maximally decomposed models:
    Components -> component appearances
    structure2 -> [[4.1626868  3.06136932]
    [2.16839552 2.92352057]
    [2.98831573 4.29232913]
    [2.92131459 1.68174912]]
    structure1 -> [[3.61949901 4.03556476]
    [2.38409816 2.06630056]
    [4.01029769 2.31575259]
    [2.15251654 3.65100976]]
    structure0 -> [[-0.32768532  0.96976257]
    [ 1.39818763  0.33646176]
    [-0.3366716  -0.75370139]
    [ 1.2962009   2.09468514]]
    """
    
    fitted_kmeans = dict()
    labels = None
    idx_class = None
    
    centroids_ = dict() # keys are integers. correspond to order of 'appearances' (a list)
    label_map = None # needs to be assigned manually after understanding the meaning of the centroids relative to the given structures
    check_labels = True
    appearances = None # in which original given crystals centroids were observed
    
    default_cluster_kwargs = {"method":"average","metric":"euclidean",
                            "threshold":1e-6,"criterion":"distance","combine":"mean"}
    
    def __init__(self, kmeans=None, load_path=None, tol0=1e-6, 
                 cluster = default_cluster_kwargs,
                 verbose=False, check_labels=False):
        if not kmeans is None:
            assert isinstance(kmeans,KMeans), "'kmeans' needs to be an instance of sklearn.cluster.KMeans!"
        elif isinstance(load_path,str):
            assert os.path.exists(load_path), "Given 'load_path' ({}) invalid!".format(load_path)
        else:
            raise ValueError("Either 'kmeans' or 'load_path' needs to be given!")
            
        self.kmeans = kmeans
        self.check_labels = check_labels
                
        if kmeans is None:
            self._load_parameters(load_path)
            
        # decomposition related parameters
        self.tol0 = tol0
        for _k, _v in self.default_cluster_kwargs.items():
            if not _k in cluster:
                cluster[_k] = _v
        self.cluster = cluster
                
        # misc
        self.verbose = verbose
    
    def _load_parameters(self,load_path):
        with open(load_path,"rb") as f:
            params = pickle.load(f)
        
        for label in sorted(params["centroids_"]):
            self.fitted_kmeans[label] = kmeans_wrapper(centroids_=params["centroids_"][label])
        
        self.centroids_ = params["centroids_"]
        self.label_map = params["label_map"]
        if "appearances" in params:
            self.appearances = params["appearances"]
        if self.check_labels:
            assert not self.label_map is None, "No labels are given in the stored file!"
            assert len(self.fitted_kmeans) == len(self.label_map), "The number of KMeans models (%i) does not match the number of available labels (%i)!" % (len(self.fitted_gmms),len(self.label_map))
            
    def save(self,save_path):
        params = {"centroids_":self.centroids_,
                  "label_map":self.label_map,
                  "appearances":self.appearances}
        with open(save_path,"wb") as f:
            pickle.dump(params,f)
    
    def fit(self, X, y=None, label_map=None):
        """Fits.
        
        Parameters
        ----------
        X : np.ndarray of floats or list of np.ndarrays
        
        Notes
        -----
        If X is an array then y needs to be given and the classifier is developed
        directly over all the samples. If, otherwise, X is a list of arrays then
        y is not needed and labels are generated by decomposing the arrays in X by 
        comparison of their computed pdfs.
        """
        X_is_array = isinstance(X,np.ndarray)
        X_is_list = isinstance(X,list)
        if X_is_array:
            assert isinstance(y,np.ndarray), "'X' is an array and thus 'y' needs to be an array!"
            assert y.shape[0]==X.shape[0], "The array 'X' of shape {} is not matched by 'y' of shape {}!".format(X.shape,y.shape)
        elif X_is_list:
            assert all([isinstance(x,np.ndarray) for x in X]), "'X' is a list and all its entries need to be np.ndarrays! Got: {}".format([type(x) for x in X])
            assert len(set([x.shape[1] for x in X]))==1, "All arrays in 'X' need to have the same number of features! Got: {}".format([x.shape[1] for x in X])
            n_features = X[0].shape[1]
        else:
            raise ValueError("'X' input not understood. Needs to be an array of list of arrays!")
        
        if X_is_array:
            self.label_map=label_map
            self.labels = np.unique(y)
            self.idx_class = {k: np.where(y==k)[0] for k in self.labels}
            for label in sorted(self.idx_class):

                _km = copy.deepcopy(self.kmeans)
                _km.fit(X[self.idx_class[label],:])

                self.fitted_kmeans[label] = kmeans_wrapper(kmeans=_km)   

                self.centroids_[label] = _km.cluster_centers_
        
        elif X_is_list:
            
            Nstructures = len(X)
            structure_ids = ["structure%s"%i for i in range(Nstructures)]
            
            # fit structure pdfs
            all_kmeans = [copy.deepcopy(self.kmeans).fit(X[i]) for i in range(Nstructures)]
            
            # compressed and decomposed (cd) model            
            centroids_cd, appearances = self.decompose_kmeans(all_kmeans, structure_ids, n_features,
                                                  method=self.cluster["method"], metric=self.cluster["metric"], 
                                                  threshold=self.cluster["threshold"], criterion=self.cluster["criterion"],
                                                  combine=self.cluster["combine"],
                                                  verbose=self.verbose)
            N_cd = len(centroids_cd)
            self.appearances = appearances.copy()
                        
            # re-fit models
            kmeans_cd = []
            _X = np.vstack(X)
            
            if self.verbose:
                print("number of resulting models: ",N_cd)
                print("number of components in total: ",sum(c.shape[0] for c in centroids_cd))
            
            for i in range(N_cd):
                self.fitted_kmeans[i] = kmeans_wrapper(centroids_=centroids_cd[i])
                self.centroids_[i] = centroids_cd[i]
                            
        else:
            raise ValueError("Boink!")
            
    @staticmethod
    def decompose_kmeans(kmeans, structure_ids, n_features, method="average",
                       metric="euclidean", threshold=1e-6, criterion="distance",
                       combine="mean", verbose=False):
        """Decomposes K-Means models.

        Parameters
        ----------
        kmeans : list of KMeans instances

        structure_ids : list of str
            Names for the individual structures.

        n_features : int
            Number of features.

        method : str, optional, default "average"
            Method to use in scipy.cluster.hierarchy.linkage

        metric: str, optional, default, "euclidean"
            Metric to use in scipy.cluster.hierarchy.linkage.

        threshold : float, optional, default 1e-6
            Threshold to use in scipy.cluster.hierarchy.fcluster. The
            smaller the value the more clusters will be found.

        criterion : str, optional, default "distance"
            Criterion to use in scipy.cluster.hierarchy.fcluster.

        combine : str, optional, default "mean"
            Specifies how Gaussians found to belong to the same cluster are to be 
            combined. Currently "mean" is the only recognized option.
            
        Returns
        -------
        mus_cd : list of np.ndarrays of int
            Each np.ndarray represents the collection of clusters
            for a maximally decomposed set of centroids.
        appearances : list of lists of str
            Same order as mus_cd, indicating the origin of the collection
            of clusters.
            
        Example: Assume mus_cd = [np.array([0,1]), np.array([2])] and
        appearances = [["structure0"], ["structure1", "structure2"]]. 
        This means that structure1 and structure2 shared the same centroid 
        number 2 and structure0 was the only one for which centroids number
        0 and 1 were found. For crystals that could mean that structure1 and 
        structure2 were really two files containing the same type of crystal.
        """

        model_reference = {}
        N = 0
        all_covs = []
        all_mus = []

        for i, sid in enumerate(structure_ids):
            model_reference[sid] = np.arange(N,N+kmeans[i].cluster_centers_.shape[0])

            N += kmeans[i].cluster_centers_.shape[0]
            mus = np.array([m for m in kmeans[i].cluster_centers_])

            all_mus.append(mus)

        all_mus = np.vstack(all_mus)

        # find approximately unique components
        p = all_mus

        Z = hierarchy.linkage(p, method=method, metric=metric)
        T = hierarchy.fcluster(Z, threshold, criterion=criterion) -1

        # relabel clusters to keep original parameter ordering
        T_map, _c = {}, 0
        for _t in T:
            if not _t in T_map:
                T_map[_t] = _c
                _c += 1

        T = np.array([T_map[_t] for _t in T])
        T_set = np.sort(np.unique(T))

        # combine parameters and thus compress models
        all_mus_clustered = []

        for t in T_set:
            if combine == "mean":
                _mu = all_mus[T==t,:].mean(axis=0)
            else:
                raise ValueError("'combine' ({}) not understood!".format(combine))
            all_mus_clustered.append(_mu)

        all_mus_clustered = np.array(all_mus_clustered)

        compressed_model_reference = {sid:np.sort(np.unique(T[vals])) for sid, vals in model_reference.items()}
        if verbose:
            print("model_reference:")
            for sid in sorted(model_reference):
                print("    initial ",model_reference[sid])
                print("    compressed ",compressed_model_reference[sid])

        # decompose by comparison
        compressed_decomposed_models = get_decomposed_models(compressed_model_reference, verbose=verbose)

        if verbose:
            print("compressed_decomposed_models ",compressed_decomposed_models)
            print("all_mus_clustered ",all_mus_clustered.shape)

        # compressed and decomposed (cd) parameters
        mus_cd = [all_mus_clustered[m,:] for m in compressed_decomposed_models]

        # model origins
        if verbose:
            print("\nMaximally decomposed models:")
            print("Components -> component appearances")

        appearances = []

        for _decomposed_model in compressed_decomposed_models:
            _appearances = [_label for _label,_vals in compressed_model_reference.items()\
                           if any([_val in _decomposed_model for _val in _vals])]
            if verbose:
                print("%s -> %s"%(str(_decomposed_model), ", ".join(sorted(_appearances))))
            appearances.append(_appearances)

        return mus_cd, appearances                
    
    def predict(self, X, show=False, show_kwargs={}):
        """Predicts.
        """
        
        isarray = isinstance(X,np.ndarray)
        islist = isinstance(X,list) and all([isinstance(x,np.ndarray) for x in X])
        if isarray:
            p = np.array([self.fitted_kmeans[_k](X) for _k in sorted(self.fitted_kmeans)]).T
            y = np.argmin(p,axis=1)
        elif islist:
            p = [np.array([self.fitted_kmeans[_k](x) for _k in sorted(self.fitted_kmeans)]).T for x in X]
            y = [np.argmin(_p,axis=1) for _p in p]
        else:
            raise ValueError("X needs to be an array of a list of arrays!")
        
        if show:
            self.show(X,y, **show_kwargs)
            
        return y
    
    def show(self, X, y, axes=[0,1], title=None, xlim=(0,1), ylim=(0,1),
             xlabel=None, ylabel=None, labelfs=16, tickfs=14, legendfs=12,
             titlefs=18, data_labels=None, cmap=plt.cm.jet, figsize=(10,5)):
        """Plots."""
        isarray = isinstance(X,np.ndarray)
        islist = isinstance(X,list) and all([isinstance(x,np.ndarray) for x in X])
        if islist:
            _X = np.vstack(X)
        else:
            _X = np.copy(X)

        uy = np.unique(y) if isarray else np.unique(np.hstack(y))
        _y = np.copy(y) if isarray else np.hstack(y)
        
        norm = mpl.colors.Normalize(vmin=uy.min(), vmax=uy.max())

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(121) if islist else fig.add_subplot(111)
        ax.set_aspect("equal")
        hs = [None for _uy in uy]
        for i,_uy in enumerate(uy):
            idx = np.where(_y==_uy)[0]
            ax.scatter(_X[idx,axes[0]], _X[idx,axes[1]], label="class %i"%_uy, 
                       alpha=.5, color=cmap(norm(_uy*np.ones(len(idx)))))
            ax.scatter(self.centroids_[i][:,axes[0]], self.centroids_[i][:,axes[1]],
                       marker="+", edgecolor="r", color=cmap(norm(_uy*np.ones(len(self.centroids_[i])))),
                       )
        if xlabel is None:
            ax.set_xlabel("Feature {}".format(axes[0]), fontsize=labelfs)
        else:
            ax.set_xlabel(xlabel, fontsize=labelfs)
        if ylabel is None:
            ax.set_ylabel("Feature {}".format(axes[1]), fontsize=labelfs)
        else:
            ax.set_ylabel(ylabel, fontsize=labelfs)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title("Inferred classes",fontsize=labelfs)
        ax.tick_params(labelsize=tickfs)
        ax.legend(loc=0, fontsize=legendfs)

        if islist:
            ax2 = fig.add_subplot(122)
            ax2.set_aspect("equal")
            norm = mpl.colors.Normalize(vmin=0, vmax=len(X))
            hs = [None for _X in X]
            for i,_X in enumerate(X):
                if data_labels is None:
                    label = "trajectory #%i"%i
                else:
                    label = data_labels[i]
                ax2.scatter(_X[:,axes[0]], _X[:,axes[1]], label=label, c=cmap(norm(np.ones(_X.shape[0])*i)),
                            alpha=.5)
            if xlabel is None:
                ax2.set_xlabel("Feature {}".format(axes[0]), fontsize=labelfs)
            else:
                ax2.set_xlabel(xlabel, fontsize=labelfs)
            if ylabel is None:
                ax2.set_ylabel("Feature {}".format(axes[1]), fontsize=labelfs)
            else:
                ax2.set_ylabel(ylabel, fontsize=labelfs)
            ax2.tick_params(labelsize=tickfs)
            ax2.set_xlim(xlim)
            ax2.set_ylim(ylim)
            ax2.set_title("Trajectories", fontsize=labelfs)
            ax2.legend(loc=0, fontsize=legendfs)
        if not title is None:
            plt.suptitle(title, fontsize=titlefs)
        plt.tight_layout()
        plt.show()
        
    def explain_inference(self, abbreviate=True):
        assert not any([self.appearances is None, len(self.centroids_)==0]), "Class instance does not appear to have self.appearances and or self.centroids_ set."
        
        print("\nMaximally decomposed models:")
        print("Components -> component appearances") 
        for i, _appearances in enumerate(self.appearances):
            _decomposed_model = self.centroids_[i].shape if abbreviate else self.centroids_[i]
            print("%s -> %s"%(", ".join(sorted(_appearances)),_decomposed_model))
            
    def set_labels(self,labels):
        """Sets labels for the label_map.
        
        Associates model identifier (integer value of self.centroids_)
        with a string, e.g. name for a type of crystal.
        
        Parameters
        ----------
        labels : dict
            Example: {"fcc":0, "fccVac":[1,2]}
            
        Note: 'labels' can also just set strings for a subset of model identifiers
        since not always all names which should be assigned are immediately obvious.
        The remaining labels are set to string versions of integer values.
        """
        assert isinstance(labels,dict)
        assert all([isinstance(_k,str) for _k in labels.keys()])
        assert all([isinstance(_v,(int,list,tuple)) for _v in labels.values()])
        
        if self.label_map is None:
            self.label_map = dict()
        
        values = set()    
        for _k, _v in labels.items():
            self.label_map[_k] = _v
            if isinstance(_v,int):
                values.add(_v)
            else:
                for _v2 in _v:
                    values.add(_v2)
            
        for _k in set(self.centroids_.keys()).difference(values):
            self.label_map[str(_k)] = _k

def assign_chemical_disorder_labels(atoms_dict, t_l_flat, Phi, mapper, species_flat, mapper_key=3,
                                    count_params={"elements":["Al","Ni"]}, 
                                    dis_elements=set(["Al","Ni"]), dis_label="gamma"):
    
    idx_dis_t = np.array([i for i,v in enumerate(t_l_flat) \
                          if 'gamma' in v and not 'prime' in v],dtype=int)
        
    idx_dis_elements = np.array([i for i,v in enumerate(count_params["elements"]) if v in dis_elements])
    sorted_dis_elements = np.array(sorted(list(dis_elements)))
    idx_Phi = np.where((Phi[:,mapper[mapper_key][idx_dis_elements]]>0).all(axis=1))[0]
    
    idx_gamma = np.intersect1d(idx_dis_t,idx_Phi)
    idx_rest = np.setdiff1d(idx_dis_t, idx_gamma) # also labeled 'gamma' but disobey element condition
    
    t_l_flat[idx_gamma] = dis_label
        
    for ix in idx_rest:
        _phi = Phi[ix, mapper[mapper_key][idx_dis_elements]]
        _phi /= _phi.sum()
        _ix = np.argsort(_phi)[-1]
        if np.isclose(_phi[_ix],1):
            t_l_flat[ix] = sorted_dis_elements[_ix]
    return t_l_flat

def batcher(X, t, n=1):
    for i in range(1,len(X),n):
        yield X[i:i+n,:], t[i:i+n]

def tf_softmax_dnn(X_in, t_in=None, mode="fit", path_ckpt="/tmp/dnn-softmax_model.ckpt",
               learning_rate=.01, l1_scale=0.001, n_epochs=5, batch_size=75, n_print=5,
               verbose=False, n_hidden1=300, n_hidden2=100, n_hidden3=75, n_hidden4=50,
               n_hidden5=25, n_outputs=10, **kwargs):
    
    tf.reset_default_graph()
    n_samples, n_inputs = X_in.shape
    
    # input and output nodes
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    t = tf.placeholder(tf.int64, shape=(None,) , name="t")

    # weights initialization (to prevent vanishing gradient) - Xavier (logistic activation) initialization is the default
    w_init = tf.contrib.layers.variance_scaling_initializer() # He initialization (ReLU activation)

    # activation
    activation = tf.nn.relu

    # L1 regularization
    my_dense_layer = partial(tf.layers.dense, activation=activation, 
                             kernel_regularizer=tf.contrib.layers.l1_regularizer(l1_scale),
                             kernel_initializer=w_init)

    # pre-packaged neural layer version
    with tf.name_scope("dnn"):
        # no batch normalization
        hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=activation,
                                  kernel_initializer=w_init)
        hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=activation,
                                  kernel_initializer=w_init)
        hidden3 = tf.layers.dense(hidden2, n_hidden3, name="hidden3", activation=activation,
                                  kernel_initializer=w_init)
        hidden4 = tf.layers.dense(hidden3, n_hidden4, name="hidden4", activation=activation,
                                  kernel_initializer=w_init)
        hidden5 = tf.layers.dense(hidden4, n_hidden5, name="hidden5", activation=activation,
                                  kernel_initializer=w_init)
        logits = tf.layers.dense(hidden4, n_outputs, name="outputs")

    # loss
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=logits)

        # without regularization
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # constant learning rate
        update = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, t, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # logging node
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = "tf_softmax/dnn-%s/" % (now,)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    if mode == "fit":
        
        t0 = time()
        with tf.Session() as sess:

            init.run()

            for epoch in range(n_epochs):
                
                for X_batch, t_batch in batcher(X_in, t_in, n=batch_size):
                    sess.run(update, feed_dict={X: X_batch, t: t_batch})
                    
                acc_train = accuracy.eval(feed_dict={X: X_batch, t:t_batch})
                if verbose and (epoch%n_print==0 or epoch+1==n_epochs):
                    print("Epoch", epoch, "Train accuracy", acc_train)

            save_path = saver.save(sess, path_ckpt)
        if verbose:
            print("training took %.3f s" % (time()-t0))
    
    elif mode == "predict":
        with tf.Session() as sess:
            
            saver.restore(sess, path_ckpt)
            return logits.eval(feed_dict={X:X_in})
    else:
        raise NotImplementedError
        
def tf_softmax_dnn_cnn(X_in, t_in=None, mode="fit", path_ckpt="/tmp/dnn-softmax_model.ckpt",
               learning_rate=.01, l1_scale=0.001, n_epochs=5, batch_size=75, n_print=5,
               verbose=False, n_hidden=300, n_outputs=10, height=28, width=28, channels=1, **kwargs):
    
    tf.reset_default_graph()
    n_samples, n_inputs = X_in.shape
    
    # input and output nodes
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
    t = tf.placeholder(tf.int64, shape=(None,) , name="t")

    # weights initialization (to prevent vanishing gradient) - Xavier (logistic activation) initialization is the default
    w_init = tf.contrib.layers.variance_scaling_initializer() # He initialization (ReLU activation)

    # activation
    activation = tf.nn.relu

    # L1 regularization
    my_dense_layer = partial(tf.layers.dense, activation=activation, 
                             kernel_regularizer=tf.contrib.layers.l1_regularizer(l1_scale),
                             kernel_initializer=w_init)

    with tf.name_scope("cnn"):
        conv = tf.layers.conv2d(X_reshaped, filters=32, kernel_size=3, strides=[1,1], padding="SAME", name="conv") # strides=[1,2,2,1]
        conv1 = tf.layers.conv2d(conv, filters=64, kernel_size=3, strides=[2,2], padding="SAME", name="conv1")
        
        pool = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
        pool_flat = tf.reshape(pool, shape=[-1, pool.get_shape()[1:4].num_elements()])
        assert np.prod(pool.shape[1:]) == pool_flat.shape[1], "Shape mismatch pool (%s) != pool_flat (%s)" % (pool.shape, pool_flat.shape)

        hidden = tf.layers.dense(pool_flat, n_hidden, activation=tf.nn.relu, name="hidden")

        logits = tf.layers.dense(hidden, n_outputs, name="outputs")

    # loss
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=logits)

        # without regularization
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # constant learning rate
        update = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, t, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # logging node
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = "tf_softmax/dnn-%s/" % (now,)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    if mode == "fit":
        
        t0 = time()
        with tf.Session() as sess:

            init.run()

            for epoch in range(n_epochs):
                
                for X_batch, t_batch in batcher(X_in, t_in, n=batch_size):
                    sess.run(update, feed_dict={X: X_batch, t: t_batch})
                    
                acc_train = accuracy.eval(feed_dict={X: X_batch, t:t_batch})
                if verbose and (epoch%n_print==0 or epoch+1==n_epochs):
                    print("Epoch", epoch, "Train accuracy", acc_train)

            save_path = saver.save(sess, path_ckpt)
        if verbose:
            print("training took %.3f s" % (time()-t0))
    
    elif mode == "predict":
        with tf.Session() as sess:
            
            saver.restore(sess, path_ckpt)
            return logits.eval(feed_dict={X:X_in})
    else:
        raise NotImplementedError
        
def tf_softmax_dnn_rnn(X_in, t_in=None, mode="fit", path_ckpt="/tmp/dnn-softmax_model.ckpt",
               learning_rate=.01, l1_scale=0.001, n_epochs=5, batch_size=75, n_print=5,
               verbose=False, n_neurons=300, n_outputs=10, n_steps=28, n_inputs=28, **kwargs):
    
    tf.reset_default_graph()
    n_samples, _n_inputs = X_in.shape
    
    # input and output nodes
    assert n_inputs*n_steps == _n_inputs
    X_in = X_in.reshape((-1, n_steps, n_inputs))
    X = tf.placeholder(tf.float32, shape=(None, n_steps, n_inputs), name="X")
    #X_reshaped = tf.reshape(X, shape=[-1, n_steps, n_inputs,])
    #X = tf.placeholder(tf.float32, [None, n_steps, n_inputs], name="X")
    t = tf.placeholder(tf.int64, shape=(None,) , name="t")    
    
    with tf.name_scope("rnn"):
        basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
        outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

        logits = tf.layers.dense(states, n_outputs, name="outputs")

    # loss
    with tf.name_scope("loss"):
        xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=t, logits=logits)

        # without regularization
        loss = tf.reduce_mean(xentropy, name="loss")

    with tf.name_scope("train"):
        optimizer = tf.train.AdamOptimizer(learning_rate)

        # constant learning rate
        update = optimizer.minimize(loss)

    with tf.name_scope("eval"):
        correct = tf.nn.in_top_k(logits, t, 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    # logging node
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    logdir = "tf_softmax/dnn-%s/" % (now,)

    init = tf.global_variables_initializer()

    saver = tf.train.Saver()
    
    if mode == "fit":
        
        t0 = time()
        with tf.Session() as sess:

            init.run()

            for epoch in range(n_epochs):
                
                for X_batch, t_batch in batcher(X_in, t_in, n=batch_size):
                    sess.run(update, feed_dict={X: X_batch, t: t_batch})
                    
                acc_train = accuracy.eval(feed_dict={X: X_batch, t:t_batch})
                if verbose and (epoch%n_print==0 or epoch+1==n_epochs):
                    print("Epoch", epoch, "Train accuracy", acc_train)

            save_path = saver.save(sess, path_ckpt)
        if verbose:
            print("training took %.3f s" % (time()-t0))
    
    elif mode == "predict":
        with tf.Session() as sess:
            
            saver.restore(sess, path_ckpt)
            return logits.eval(feed_dict={X:X_in})
    else:
        raise NotImplementedError
            
class DNNSoftmaxClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):

    dnn_params = dict(n_hidden1 = 300, 
                      n_hidden2 = 100, 
                      n_hidden3 = 75, 
                      n_hidden4 = 50, 
                      n_hidden5 = 25, 
                      learning_rate = .01,
                      path_ckpt = "/tmp/dnn-softmax_model.ckpt",
                      l1_scale = 0.001, 
                      n_epochs = 5, 
                      batch_size = 75, 
                      n_print = 5,
                      verbose = False,)
    
    dnn_type = "simple"
    
    dnn_impl = {"simple":tf_softmax_dnn,
                "cnn":tf_softmax_dnn_cnn,
                "rnn":tf_softmax_dnn_rnn}
    
    def __init__(self, dnn_type=None, dnn_params=None):

        if  (not dnn_type is None) and (dnn_type in self.dnn_impl):
            self.dnn_type = dnn_type
        
        if (not dnn_params is None) and isinstance(dnn_params,dict):
            self.dnn_params = dnn_params

    def fit(self, X, y, sample_weight=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape (n_samples,)
            Target vector relative to X.

        sample_weight : array-like, shape (n_samples,) optional
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

            .. versionadded:: 0.17
               *sample_weight* support to LogisticRegression.

        Returns
        -------
        self : object
            Returns self.
        """
                
        X, y = sklearn.utils.validation.check_X_y(X, y, accept_sparse='csr', dtype=np.float64,
                         order="C")
        
        sklearn.utils.multiclass.check_classification_targets(y)
        self.classes_ = np.unique(y)
        
        n_samples, n_features = X.shape

        n_classes = len(self.classes_)
        classes_ = self.classes_
        self.dnn_params["n_outputs"] = n_classes
        if n_classes < 2:
            raise ValueError("This solver needs samples of at least 2 classes"
                             " in the data, but the data contains only one"
                             " class: %r" % classes_[0])

        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        self.dnn_impl[self.dnn_type](X, t_in=y, mode="fit", **self.dnn_params)

        return self
    
    def decision_function(self, X):
        return self.dnn_impl[self.dnn_type](X, mode="predict", **self.dnn_params)

    def predict_proba(self, X):
        """Probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        For a multi_class problem, if multi_class is set to be "multinomial"
        the softmax function is used to find the predicted probability of
        each class.
        Else use a one-vs-rest approach, i.e calculate the probability
        of each class assuming it to be positive using the logistic function.
        and normalize these values across all the classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the probability of the sample for each class in the model,
            where classes are ordered as they are in ``self.classes_``.
        """
        
        return sklearn.utils.extmath.softmax(self.decision_function(X), copy=False)

    def predict_log_proba(self, X):
        """Log of probability estimates.

        The returned estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        T : array-like, shape = [n_samples, n_classes]
            Returns the log-probability of the sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.
        """
        return np.log(self.predict_proba(X))
    
    def predict(self, X):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        C : array, shape = [n_samples]
            Predicted class label per sample.
        """
        scores = self.decision_function(X)
        if len(scores.shape) == 1:
            indices = (scores > 0).astype(np.int)
        else:
            indices = scores.argmax(axis=1)
        return self.classes_[indices]