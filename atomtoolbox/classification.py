import copy, os, pickle, collections, warnings

from scipy import stats, optimize
import numpy as np
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.cluster import hierarchy
import matplotlib.pylab as plt

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
    This class can also do decomposition auf Gaussian Mixture Models (GMMs).
    
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
    
    weights_ = dict()
    covariances_ = dict()
    means_ = dict()
    label_map = None
    check_labels = True
    
    def __init__(self, gmm=None, load_path=None, tol0=1e-6, cluster = {"method":"average","metric":"euclidean",
                       "threshold":1e-6,"criterion":"distance","cluster_parameters":"mu","combine":"mean"},
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
        if self.check_labels:
            assert not self.label_map is None, "No labels are given in the stored file!"
            assert len(self.fitted_gmms) == len(self.label_map), "The number of GMMs (%i) does not match the number of available labels (%i)!" % (len(self.fitted_gmms),len(self.label_map))
            
    def save(self,save_path):
        params = {"weights_":self.weights_,
                  "covariances_":self.covariances_,
                  "means_":self.means_,
                  "label_map":self.label_map}
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
            mus_cd, covs_cd = self.decompose_gmms(all_gmms, structure_ids, n_features,
                                                  method=self.cluster["method"], metric=self.cluster["metric"], 
                                                  threshold=self.cluster["threshold"], criterion=self.cluster["criterion"],
                                                  cluster_parameters=self.cluster["cluster_parameters"], combine=self.cluster["combine"],)
            N_cd = len(mus_cd)
                        
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
        compressed_decomposed_models = get_decomposed_models(compressed_model_reference, verbose=False)
        
        if verbose:
            print("compressed_decomposed_models ",compressed_decomposed_models)
            print("all_mus_clustered ",all_mus_clustered.shape)
            print("all_covs_clustered ",all_covs_clustered.shape)
        
        # compressed and decomposed (cd) parameters
        mus_cd = [all_mus_clustered[m,:] for m in compressed_decomposed_models]
        covs_cd = [all_covs_clustered[m,:] for m in compressed_decomposed_models]
        return mus_cd, covs_cd
    
    def predict_proba(self, X):
        p = np.zeros((X.shape[0],len(self.fitted_gmms)))
        
        for i,label in enumerate(sorted(self.fitted_gmms)):
            p[:,i] = self.fitted_gmms[label](X)
            
        Z = p.sum(axis=1)
        return (p.T/Z).T
    
    def show(self, X, y, axes=[0,1], title=None, xlim=(0,1), ylim=(0,1),
             xlabel=None, ylabel=None, labelfs=16, tickfs=14, legendfs=12,
             titlefs=18, data_labels=None):
        """Plots."""
        isarray = isinstance(X,np.ndarray)
        islist = isinstance(X,list) and all([isinstance(x,np.ndarray) for x in X])
        if islist:
            _X = np.vstack(X)
        else:
            _X = np.copy(X)

        uy = np.unique(y) if isarray else np.unique(np.hstack(y))
        _y = np.copy(y) if isarray else np.hstack(y)

        fig = plt.figure()
        ax = fig.add_subplot(121) if islist else fig.add_subplot(111)
        ax.set_aspect("equal")
        for i,_uy in enumerate(uy):
            idx = np.where(_y==_uy)[0]
            ax.plot(_X[idx,axes[0]],_X[idx,axes[1]],'o',label="class "+str(_uy),markerfacecolor="None",alpha=.5)
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
        plt.legend(loc=0,fontsize=legendfs)

        if islist:
            ax2 = fig.add_subplot(122)
            ax2.set_aspect("equal")
            for i,_X in enumerate(X):
                if data_labels is None:
                    label = "crystal#"+str(i)
                else:
                    label = data_labels[i]
                ax2.plot(_X[:,axes[0]],_X[:,axes[1]],'o',label=label,markerfacecolor="None",alpha=.5)
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
            ax2.set_title("Crystals")
            plt.legend(loc=0, fontsize=legendfs)
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