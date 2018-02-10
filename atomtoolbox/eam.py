import sys, time, itertools, copy, pickle
from pottytrainer import fitenergy as fe

import numpy as np
from ase.calculators.eam import EAM
from scipy.interpolate import InterpolatedUnivariateSpline as spline
import matplotlib.pylab as plt
from .spatial import get_ultracell, get_neighbors
from .features import get_crystal_design_matrix


def get_rhos(positions, species, upositions, uspecies, 
             uindices_neigh, uindices, emb_density_funs,
             tol0=1e-6):
    """Computes embedding densitites 'rhos' using the given emb_density_funs.
    
    Parameters
    ----------
    positions : np.ndarray of float of shape (Natoms,3)
        Atom positions in the supercell (Natoms = number of atoms 
        in the supercell).
    
    species : np.ndarray of str of shape (Natoms,)
    
    upositions : np.ndarray of float of shape (Nultraatoms,3)
        Atom positions in the supercell (Nultraatoms = number of atoms 
        in the ultracell).
    
    uspecies : np.ndarray of str of shape (Nultraatoms,)
    
    uindices_neigh : list of length Nultraatoms of np.ndarrays of int
        Each np.ndarray corresponds to an atom (same order as 'positions') 
        containing the indices for each neighboring atom for the given 
        'positions' parameter. If uindices and upositions are given then neigh_idx
        contains indices for the ultracell.
    
    uindices : np.ndarray of int of shape (Nultraatoms,)
        Indices to the original supercell atoms for each ultracell atom.
        
    emb_density_funs : dict
        Contains embedding density functions for individual elements.
        Example: emb_density_funs = {"Al":lambda x: x**2, "Ni":lambda x: np.zeros(x.shape[0])}
    
    Returns
    -------
    rhos : np.ndarray of shape (Natoms,)
        Embedding density values for the supercell.
    urhos : np.ndarray of shape (Nultratoms,)
        Embedding density values for the ultracell.
        
    Note
    ----
    There is a subtle distinction between uindices_neigh and uindices. Please read their 
    description carefully.
    """
        
    if emb_density_funs is None:
        return None, None
    element_order = sorted(emb_density_funs.keys())
    
    Na = positions.shape[0]
    rhos = np.zeros(Na)
    
    for i in range(Na):
        uidx = uindices_neigh[i]
        dr = positions[i,:] - upositions[uidx]
        r = np.linalg.norm(dr,axis=1)
        r_valid = np.where(r>tol0)[0]
        _rhos = np.array([emb_density_funs[uspecies[_i]](r[_i]) for _i in r_valid])
        rhos[i] = _rhos.sum()
    
    if (rhos<0).any():
        print("positions ",positions)
        print("\nmin {:.2f} max {:.2f} mean {:.2f} std {:.2f}".format(rhos.min(),
                                                                    rhos.max(),
                                                                    np.mean(rhos),
                                                                    np.std(rhos)))
        print(rhos)
    urhos = rhos[uindices]
    return rhos, urhos

def get_r_and_dens_values(rpositions, cells, species, r_cut, emb_density_funs=None,
                 num_neigh=None, element_filter=None, check_bounds=True, verbose=False):
    
    all_r, all_dens = [], []
    print("structure processed:")
    N = len(rpositions)
    for i in range(N):
        if (i+1)%10==0:
            print("\n\n{}/{}...".format(i+1,N))
        _pos = rpositions[i]
        _cell = cells[i]
        _spec = species[i]
        
        _pos = np.array(_pos)
        _spec = np.array(_spec)
        _invcell = np.linalg.inv(_cell)
        _fpos = np.dot(_pos,_invcell)
        
        upositions, uspecies, uindices = get_ultracell(fpos=_fpos,cell=_cell,
                                                       species=_spec,r_cut=r_cut,show=False,
                                                       verbose=verbose,max_iter=20,ftol=1e-6,
                                                       check_bounds=check_bounds)

        # getting the ultracell neighboring indices
        uindices_neigh = get_neighbors(_pos,upositions=upositions, r_cut=r_cut, 
                                       num_neigh=num_neigh)
        if not emb_density_funs is None:
            dens, udens = get_rhos(_pos, _spec, upositions, uspecies, 
                            uindices_neigh, uindices, emb_density_funs)
            all_dens.append(dens)
        else:
            dens, udens = None, None
        
        _Natoms = _pos.shape[0]
        for j in range(_Natoms):
        
            # current atom
            atom = _pos[j,:]
            spec = _spec[j]

            #neighboring atoms
            uidx = uindices_neigh[j]

            if callable(element_filter):
                uidx = uidx[element_filter(uspecies[uidx],spec)]

            uatoms = upositions[uidx]
            uspecs = uspecies[uidx]
            
            dr = uatoms-atom
            r = np.linalg.norm(dr,axis=1)
            all_r.append(r)
    return all_r, all_dens

def get_embedding_density_functions(bonds, rho_dict, ignore_negatives=True,
                                    show=True, return_rho_bounds=True, r_lb=0, r_ub=1, Nsteps_iso=100, 
                                    scale="std"):
    
    _bonds = copy.deepcopy(bonds)
    _rho_dict = copy.deepcopy(rho_dict)
    X, dens_lb, dens_ub = fe.get_all_sources(_bonds, return_rho_bounds=return_rho_bounds)
    
    # rescale rho(r) and embedding densities by the largest observed embedding density
    dens = np.hstack(X["density"])
    _scale = None
    if scale == "std":
        _scale = np.std(dens)
        
    if not ignore_negatives:
        assert (np.hstack(X["density"])>0).all(), "Found negative embedding densities..."

        if rho_conv_type is not None:
            obs_emb = np.amax([v for v2 in X["density"] for v in v2])
            X["density"] = [v/obs_emb for v in X["density"]]
            for _s in rho_dict["rhos"].keys():
                _rho_dict["rhos"][_s] /= obs_emb
                dens_min = min([np.amin(v) for v in X["density"]])
                dens_lb[_s] = 0 if dens_min > 0 else rho_min
                dens_ub[_s] = 2*max([np.amax(v) for v in X["density"]])

            print("\nEmbedding densities after re-scaling {}".format([list(v) for v in X["density"]]))
            print("\na0 {}".format([bond.box[0,0] for bond in bonds]))
            print("\nnames {}".format([bond.name for bond in bonds]))
            print("\nBonds with negative densities:")
            for i,bond in enumerate(bonds):
                if any([v<0 for v in X["density"][i]]):
                    print("bond {}".format(bond.name))
                    print("densities {}".format(X["density"][i]))

            print("rho_lb {}\nrho_ub {}".format(dens_lb, dens_ub))
            if not ignore_negatives:
                assert all ([not v<0 for v in dens_lb.values()]), "Assertion failed - found density functions with negative values!"
        else:
            print(X["species"])
            for _s in rho_dict["rhos"].keys():
                _density = [[v2 for iv2,v2 in enumerate(v) if X["species"][iv][iv2]==_s] for iv,v in enumerate(X["density"]) if _s in X["species"][iv]]
                print(_s,": ",_density)
                dens_min = min([np.amin(v) for v in _density])
                dens_max = max([np.amax(v) for v in _density])
                ddens = dens_max-dens_min
                dens_min -= .1*ddens
                dens_lb[_s] = dens_min if dens_min <= 0 else 0
                dens_ub[_s] = dens_max + .1*ddens
            print("dens_lb {}\ndens_ub {}".format(dens_lb,dens_ub))
    
    if not scale is None:
        X["density"] = [v/_scale for v in X["density"]]
        for _s in rho_dict["rhos"].keys():
            _rho_dict["rhos"][_s] /= _scale
    
    r = _rho_dict["r"]
    emb_density_funs = {el: spline(r,_rho_dict["rhos"][el]) for el in _rho_dict["rhos"]}
    
    if show:
        
        ks = sorted(_rho_dict["rhos"].keys())
        for k in ks:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(_rho_dict["r"],_rho_dict["rhos"][k],label="rho_dict "+k)
            ax.plot(r,emb_density_funs[k](r),label="spline "+k)
            ax.set_xlabel("r [AA]",fontsize=14)
            ax.set_ylabel("rho (r)",fontsize=14)
            ax.grid()
            ax.set_title(k)
            plt.legend(loc=0)
            plt.tight_layout()
        plt.show()
    
    return emb_density_funs, X, _scale

def element_emb_filters():
    """Filter for embedding density calculations.
    
    Due to the nature of embedding density energies all neighbors need to be passed
    regardless of element.
    """
    def _filter(s,s_ref):
        N = len(s)
        return np.array([True for v in range(N)])
    return _filter

def element_pair_filters(el_ref="Al",el_neigh="Al",kind="symmetric"):
    
    if kind=="symmetric":
        def _filter(s,s_ref):
            _el_neigh = el_neigh
            _el_ref = el_ref
            
            N = len(s)
            
            _curr = s_ref==_el_ref
            if not _curr: # in case the elements need to be swapped to match the pattern
                _el_neigh, _el_ref = el_ref, el_neigh
                _curr = s_ref==_el_ref
            
            if not _curr: # the current element is neither el_ref nor el_neigh
                return np.array([False for v in range(N)])
            
            curr = np.array([True for v in range(N)],dtype=bool)
            neigh = s==_el_neigh
            return np.logical_and(curr,neigh)
        
        return _filter
    
    elif kind=="asymmetric":
        def _filter(s,s_ref):
            _curr = s_ref==el_ref
            N = len(s)
            curr = np.array([s for v in range(N)],dtype=bool)
            neigh = s==el_neigh
            return np.logical_and(curr,neigh)
    else:
        raise NotImplementedError("'kind' parameter needs to be 'symmetric' or 'asymmetric'.")

def get_mapper(params_emb, params_pair, features_class, densgiven=False):
    """Computes the mapping inherent to the Design matrix.

    Parameters
    ----------
    params_emb : dict
        Contains setting for features to compute of embedding densities.
    
    params_pair : dict
        Contains setting for features to compute of pair distances.        
    
    features_class : list or features_class reference
        Class or list of classes to use to compute features. 
        Example: [atb.DistanceGaussTaperingFeatures_2body,atb.DistanceSineCosineTaperingFeatures_2body]
    
    densgiven : boolean, optional, default False
        Wether or not the design matrix contains embedding density features.

    Returns
    -------
    mapper : dict
        Contains indices grouped by embedding density features "emb", pair distance features
        "pair" and by the feature types used as "features_class".
        Example: {"emb":{"Al":..., "Ni":...}, "pair":{"AlAl":..., "AlNi":...},
                  "features_class":{"emb":{"Al":{DistanceCosTaperingFeatures_2body:..., DistanceGaussTaperingFeatures_2body:...}, "Ni":...}, 
                                    "pair":{"AlAl":{...}, "AlNi":{...}, ...}}}

    Example
    -------
    >>> import itertools
    >>> import atomtoolbox as atb
     
    >>> elements = ["Al","Ni"]
    >>> pairs = [(elements[i], elements[j]) for i,j in \
            itertools.product(range(len(elements)), range(len(elements))) if i<=j]
    
    >>> kappa_ = np.arange(0,50)/float(r_cut)
    >>> kappa_emb_ = None#np.arange(0,50)/float(1.)
    >>> q_ = np.linspace(0, r_cut, 50)
    >>> p_ = np.array([.1, .5, 1.])
    >>> taper_fun = atb.taper_fun_wrapper(_type="x4ge",a=r_cut,b=.05)
    >>> taper_fun_emb = atb.taper_fun_wrapper(_type="Ones")
    >>> emb_density_funs = {k: None for k in elements}

    >>> gauss_params = lambda p0,p1: {"element_filter":atb.element_pair_filters(el_ref=p0,el_neigh=p1,kind="symmetric"),\
                        "p_":p_, "q_":q_, "taper_fun":taper_fun}
    >>> sc_params = lambda p0, p1: {"element_filter":atb.element_pair_filters(el_ref=p0,el_neigh=p1,kind="symmetric"),\
                    "kappa_":kappa_, "taper_fun":taper_fun,"kappa_t":kappa_t,
                    "kappa_r":kappa_r}

    >>> params_pair = {p0+p1: [gauss_params(p0,p1), sc_params(p0,p1)] for (p0,p1) in pairs}

    >>> sc_params_emb = {"element_filter":emb_filter, "kappa_":kappa_emb_, 
                        "taper_fun":taper_fun_emb, "emb_density_funs":{k:None for k in elements}}
    >>> gauss_params_emb = {"element_filter":emb_filter, "taper_fun":taper_fun_emb, "p_":p_, 
                            "q_":q_, "emb_density_funs":{k:None for k in elements}}

    >>> params_emb = {el: [sc_params_emb, gauss_params_emb] for el in elements}

    >>> features_class = [atb.DistanceGaussTaperingFeatures_2body,atb.DistanceSineCosineTaperingFeatures_2body] #atb.DistanceSineCosineTaperingFeatures_2body,atb.DistanceCosTaperingFeatures_2body,
    
    >>> mapper = get_mapper(params_emb, params_pair, features_class, densgiven=False)
    """

    from atomtoolbox.features import make_array
    
    def get_N_pair(fc, pair, i=None):
        if fc.name == "DistanceCosTaperingFeatures_2body":
            return make_array(params_pair[pair]["kappa_"]).shape[0] \
                if i is None else make_array(params_pair[pair][i]["kappa_"]).shape[0]

        elif fc.name == "DistanceSineCosineTaperingFeatures_2body":
            return make_array(params_pair[pair]["kappa_"]).shape[0]*2 \
                if i is None else make_array(params_pair[pair][i]["kappa_"]).shape[0]*2

        elif fc.name == "DistanceGaussTaperingFeatures_2body":
            return make_array(params_pair[pair]["q_"]).shape[0]*make_array(params_pair[pair]["p_"]).shape[0] \
                if i is None else make_array(params_pair[pair][i]["q_"]).shape[0]*make_array(params_pair[pair][i]["p_"]).shape[0]
        else:
            raise NotImplementedError("features_class '{}' not implemented!".format(fc.name))
    
    def get_N_emb(fc, el, i=None):
        if fc.name == "DistanceCosTaperingFeatures_2body":
            return make_array(params_emb[el]["kappa_"]).shape[0] \
                if i is None else make_array(params_emb[el][i]["kappa_"]).shape[0]

        elif fc.name == "DistanceSineCosineTaperingFeatures_2body":
            return make_array(params_emb[el]["kappa_"]).shape[0]*2 \
                if i is None else make_array(params_emb[el][i]["kappa_"]).shape[0]*2

        elif fc.name == "DistanceGaussTaperingFeatures_2body":
            return make_array(params_emb[el]["q_"]).shape[0]*make_array(params_emb[el]["p_"]).shape[0] \
                if i is None else make_array(params_emb[el][i]["q_"]).shape[0]*make_array(params_emb[el][i]["p_"]).shape[0]

        else:
            raise NotImplementedError("features_class '{}' not implemented!".format(fc.name))
    
    implemented = set(["DistanceSineCosineTaperingFeatures_2body","DistanceCosTaperingFeatures_2body",
                       "DistanceGaussTaperingFeatures_2body"])
    islist = isinstance(features_class,list)
    if islist:
        for fc in features_class:
            assert fc.name in implemented, "features_class '{}' is not yet implemented!".format(fc)
    else:
        assert features_class.name in implemented, "features_class '{}' is not yet implemented!".format(features_class)
    
    elements = sorted(params_emb.keys())
    pairs = sorted(params_pair.keys())
    
    mapper = {"emb":{el:None for el in elements}, "pair":{pair:None for pair in pairs},
              "features_class":{"emb":{el:{} for el in elements}, "pair":{pair:{} for pair in pairs}}}
    
    N = 0

    if densgiven:
        for el in elements:
            if islist:
                _N = 0
                for i, fc in enumerate(features_class):
                    mapper["features_class"]["emb"][el][fc.name] = np.arange(N+_N,N+_N+get_N_emb(fc, el, i=i))
                    _N += get_N_emb(fc, el, i=i)
                mapper["emb"][el] = np.arange(N,N+_N)
            else:
                _N = get_N_emb(features_class, el)
                mapper["features_class"]["emb"][el][features_class.name] = np.arange(N,N+_N)
                mapper["emb"][el] = np.arange(N,N+_N)
            N += _N
            
            #mapper["emb"][el] = np.arange(N,N+params_emb["kappa_"].shape[0])
            #N += params_emb["kappa_"].shape[0]

    for pair in pairs:
        if islist:
            _N = 0
            for i,fc in enumerate(features_class):
                mapper["features_class"]["pair"][pair][fc.name] = np.arange(N+_N,N+_N+get_N_pair(fc, pair, i=i))
                _N += get_N_pair(fc,pair, i=i)
            mapper["pair"][pair] = np.arange(N,N+_N)
        else:
            _N = get_N_pair(features_class,pair)
            mapper["features_class"]["pair"][pair][features_class.name] = np.arange(N,N+_N)
            mapper["pair"][pair] = np.arange(N,N+_N)
        N += _N
    return mapper

def collect_EAM_design_matrices(r_cut, params_emb, params_pair,  
                                rpositions, species, cells, return_force=True,
                                features_class=None, tol0=1e-6, return_mapper=True, 
                                verbose=False, check_bounds=True, densgiven=False):
    """Computes the design matrices for EAM regression.

    Parameters
    ----------
    r_cut : float
        Cut-off radius.
    params_emb : dict
        Specifies parameters for embedding energy related features.
    params_pair :
        Specifies parameters for pair energy related features.
    rpositions : 
    species : 
    cells : 
    return_force : 
    features_class : 
    tol0 : 
    return_mapper :
    verbose : 
    check_bounds : 
    densgiven : 
    """
    if densgiven:
        for el in sorted(params_emb):
            for i,fc in enumerate(features_class):
                assert "emb_density_funs" in params_emb[el][i], "Computing the EAM design matrix with densgiven = True requires that all elements ({}) have a specified density function! However, 'emb_density_funs' is not even a key for {} -> {}!".format(list(params_emb.keys()),el,fc.name)
                assert all([not params_emb[el][i]["emb_density_funs"][k] is None for k in params_emb[el][i]["emb_density_funs"]]), "Computing the EAM design matrix with densgiven = True requires that all elements ({}) have a specified density function! {}, however, does not for {}!".format(list(params_emb.keys()),el,fc.name)
    #densgiven = not all([emb_density_funs[k] is None for k in emb_density_funs])

    Phi_e_emb, Phi_f_emb = None, None
    Phi_e_pair_dict, Phi_f_pair_dict = {}, {}
    Phi_e_emb_dict, Phi_f_emb_dict = {}, {}
        
    t0 = time.time()

    if return_mapper:
        mapper = get_mapper(params_emb, params_pair, features_class, densgiven=densgiven)
    
    N = len(rpositions)
    for i in range(N): # looping through crystals
        if verbose:
            print("\n{}/{}".format(i+1,N))
        _pos = rpositions[i]
        _cell = cells[i]
        _spec = species[i]
        
        # embedding part
        if densgiven:
            for el in sorted(params_emb):
                _params_emb = params_emb[el]
                if return_force:
                    _Phi_e_emb, _Phi_f_emb = get_crystal_design_matrix(positions=_pos, 
                                                    species=_spec, cell=_cell, r_cut=r_cut, 
                                                    features_class=features_class,
                                                    params_features=_params_emb, 
                                                    return_force=True,
                                                    emb_density_funs=_params_emb[0]["emb_density_funs"], 
                                                    tol0=tol0, check_bounds=check_bounds)
                else:
                    _Phi_e_emb = get_crystal_design_matrix(positions=_pos, 
                                                    species=_spec, cell=_cell, r_cut=r_cut, 
                                                    features_class=features_class,
                                                    params_features=_params_emb, 
                                                    return_force=False,
                                                    emb_density_funs=_params_emb[0]["emb_density_funs"], 
                                                    tol0=tol0, check_bounds=check_bounds)
                    _Phi_f_emb = None
                _Phi_e_emb = _Phi_e_emb.sum(axis=0)
                
                if not el in Phi_e_emb_dict:
                    Phi_e_emb_dict[el] = _Phi_e_emb
                    Phi_f_emb_dict[el] = _Phi_f_emb
                else:
                    Phi_e_emb_dict[el] = np.vstack((Phi_e_emb_dict[el],_Phi_e_emb))
                    Phi_f_emb_dict[el] = np.vstack((Phi_f_emb_dict[el],_Phi_f_emb))

        # pair parts
        for sname in sorted(params_pair):
            
            if verbose:
                print("\n",sname)
            _params_pair = params_pair[sname]
            if return_force:
                _Phi_e_pair, _Phi_f_pair = get_crystal_design_matrix(positions=_pos, species=_spec, cell=_cell, r_cut=r_cut, 
                                                features_class=features_class,
                                                params_features=_params_pair, return_force=True, check_bounds=check_bounds)
            else:
                _Phi_f_pair = None                                                                                        
                _Phi_e_pair = get_crystal_design_matrix(positions=_pos, species=_spec, cell=_cell, r_cut=r_cut, 
                                                features_class=features_class,
                                                params_features=_params_pair, return_force=False, check_bounds=check_bounds)
            _Phi_e_pair = _Phi_e_pair.sum(axis=0)
            
            if not sname in Phi_e_pair_dict:
                Phi_e_pair_dict[sname] = _Phi_e_pair
                if return_force:
                    Phi_f_pair_dict[sname] = _Phi_f_pair
            else:
                Phi_e_pair_dict[sname] = np.vstack((Phi_e_pair_dict[sname],_Phi_e_pair))
                if return_force:
                    Phi_f_pair_dict[sname] = np.vstack((Phi_f_pair_dict[sname],_Phi_f_pair))
            
    # double counting of pair contributions
    for sname in Phi_e_pair_dict:
        Phi_e_pair_dict[sname] /= 2.
    
    if densgiven:
        if return_force:
            Phi_emb_dict = {el:np.vstack((Phi_e_emb_dict[el],Phi_f_emb_dict[el])) \
                            for el in params_emb}
        else:
            Phi_emb_dict = Phi_e_emb_dict.copy()

        Phi_emb = np.hstack((Phi_emb_dict[el] for el in sorted(Phi_emb_dict)))
        #Phi_emb = np.vstack((Phi_e_emb,Phi_f_emb))
    if return_force:
        Phi_pair_dict = {sname:np.vstack((Phi_e_pair_dict[sname],Phi_f_pair_dict[sname])) \
                        for sname in params_pair}
    else:
        Phi_pair_dict = Phi_e_pair_dict.copy()
    Phi_pair = np.hstack((Phi_pair_dict[sname] for sname in sorted(Phi_pair_dict)))
    #print("Phi_emb ",Phi_emb.shape," Phi_pair ",Phi_pair.shape)
    
    if densgiven:
        Phi_e = np.hstack((np.hstack((Phi_e_emb_dict[el] for el in sorted(params_emb))),
                           np.hstack((Phi_e_pair_dict[sname] for sname in sorted(params_pair)))))
        if return_force:
            Phi_f = np.hstack((np.hstack((Phi_f_emb_dict[el] for el in sorted(params_emb))),
                            np.hstack((Phi_f_pair_dict[sname] for sname in sorted(params_pair)))))
    else:
        Phi_e = np.hstack((Phi_e_pair_dict[sname] for sname in sorted(params_pair)))
        if return_force:
            Phi_f = np.hstack((Phi_f_pair_dict[sname] for sname in sorted(params_pair)))
        Phi_emb = None
    if return_force:
        print("Phi_e",Phi_e.shape,"Phi_f",Phi_f.shape)
    else:
        print("Phi_e",Phi_e.shape)
    print("\ntime spent {} s...".format(time.time()-t0))

    if return_mapper:
        if return_force:
            return Phi_e, Phi_f, Phi_emb, Phi_pair, mapper
        else:
            return Phi_e, Phi_emb, Phi_pair, mapper
    else:
        if return_force:
            return Phi_e, Phi_f, Phi_emb, Phi_pair
        else:
            return Phi_e, Phi_emb, Phi_pair


# def generate_EAM_calculator_from_linear_models(mapper, model, emb_density_funs, r_bounds, Nr,
#                                                dens_bounds, Ndens, r_cut, atom_info, basis_r,
#                                                basis_dens, save_path=None, show=False,
#                                                ylim_emb=None, ylim_pair=None,
#                                                basis_r_1stder=None, figsize=(5,5),
#                                                basis_dens_1stder=None):
#     densgiven = not all([emb_density_funs[k] is None for k in emb_density_funs])

#     elements = sorted(list(emb_density_funs))
#     N_ele = len(elements)
#     pair_map = {el1+el2:"".join(sorted([el1,el2])) for el1,el2 in itertools.product(elements,elements)}
    
#     element_pairs_sorted = sorted(list(set(pair_map.values())))
    
#     r = np.linspace(r_bounds[0],r_bounds[1],Nr)
#     dens = np.linspace(dens_bounds[0],dens_bounds[1],Ndens)
#     dr = r[1]-r[0]
#     ddens = dens[1]-dens[0]
    
#     Nb_pair = sum([v.shape[0] for v in mapper["pair"].values()]) # number of basis terms for pair energies
#     Nb_emb = sum([v.shape[0] for v in mapper["emb"].values()]) if densgiven else 0 # number of basis terms for embedding energies
#     N_pair = len(element_pairs_sorted) # number of element pairs
    
#     # generate design matrices
#     zero_Phi_emb = np.zeros((Ndens, Nb_emb + Nb_pair))
#     zero_Phi_pair = np.zeros((Nr, Nb_emb + Nb_pair))
#     print(zero_Phi_emb.shape,zero_Phi_pair.shape)
    
#     _Phi_emb = np.array([b(dens) for b in basis_dens]).T
#     _Phi_pair = np.array([b(r) for b in basis_r]).T
    
#     print("emb: min = {} max = {}".format(np.amin(_Phi_emb),np.amax(_Phi_emb)))
#     print("pair: min = {} max = {}".format(np.amin(_Phi_pair),np.amax(_Phi_pair)))
    
#     Phi_emb, Phi_pair = {}, {}
#     Phi_emb_f, Phi_pair_f = {}, {}
    
#     E_emb, E_pair = {}, {}
    
#     if show:
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111)
    
#     for el,m in mapper["emb"].items():
#         if densgiven:
#             Phi_emb[el] = np.copy(zero_Phi_emb)
#             Phi_emb[el][:,m] = _Phi_emb
        
#         # compute energy functions
#         _y = model.predict(Phi_emb[el]) if densgiven else np.zeros(dens.shape[0])
#         E_emb[el] = _y.ravel()
        
#         if show:
#             ax.plot(dens, _y, label=el)
    
#     if show:
#         ax.set_title("Emb")
#         ax.set_ylim(ylim_emb)
#         plt.legend(loc=0)
#         plt.show()
        
#         fig = plt.figure(figsize=figsize)
#         ax = fig.add_subplot(111)
    
#     for pair,m in mapper["pair"].items():
#         Phi_pair[pair] = np.copy(zero_Phi_pair)
#         Phi_pair[pair][:,m] = _Phi_pair
        
#         # compute energy functions
#         _y = model.predict(Phi_pair[pair])
#         E_pair[pair] = _y.ravel()
        
#         if show:
#             ax.plot(r, _y, label=pair)
        
#     if show:
#         ax.set_title("Pair")
#         ax.set_ylim(ylim_pair)
#         plt.legend(loc=0)
#         plt.show()
        
#     # embedding energy derivatives
#     E_emb_f = dict()
#     if not basis_r_1stder is None:
#         _Phi_emb_f = np.array([b(dens) for b in basis_dens_1stder]).T
#         for el,m in mapper["emb"].items():
#             Phi_emb_f[el] = np.copy(zero_Phi_emb)
#             Phi_emb_f[el][:,m] = _Phi_emb_f
#             E_emb_f[el] = model.predict(Phi_emb_f[el]).ravel() if densgiven else np.zeros(dens.shape[0])
    
#     # pair energy derivatives
#     E_pair_f = dict()
#     if not basis_dens_1stder is None:
#         _Phi_pair_f = np.array([b(r) for b in basis_r_1stder]).T
#         for pair,m in mapper["pair"].items():
#             Phi_pair_f[pair] = np.copy(zero_Phi_pair)
#             Phi_pair_f[pair][:,m] = _Phi_pair_f
#             E_pair_f[pair] = model.predict(Phi_pair_f[pair]).ravel()
        
#     # store functions in ase EAM calculator friendly format
#     embedded_energy = np.array([spline(dens,E_emb[el]) for el in elements])
#     phi = np.array([[spline(r,E_pair[(pair_map[el2+el1])]) for el2 in elements] for el1 in elements])
#     electron_density = np.array([spline(r,emb_density_funs[el](r)) for el in elements])
        
#     if not basis_r_1stder is None:
#         d_phi = np.array([[spline(r,E_pair_f[(pair_map[el2+el1])]) for el2 in elements] for el1 in elements])
#     else:
#         d_phi = np.array([[phi[v0,v1].derivative() for v1 in range(N_ele)] for v0 in range(N_ele)])
        
#     if not basis_dens_1stder is None:
#         d_embedded_energy = np.array([spline(dens,E_emb_f[el]) for el in elements])
#     else:    
#         d_embedded_energy = np.array([v.derivative() for v in embedded_energy])
        
#     d_electron_density = np.array([v.derivative() for v in electron_density])
        
#     EAM_obj = EAM(elements=elements, embedded_energy=embedded_energy,
#                     electron_density=electron_density,
#                     phi=phi, 
#                     d_embedded_energy=d_embedded_energy,
#                     d_electron_density=d_electron_density,
#                     d_phi=d_phi,
#                     cutoff=r_cut, form='alloy',
#                     Z=[atom_info[el]["number"] for el in elements], nr=Nr, nrho=Ndens, 
#                     dr=dr, drho=ddens,
#                     lattice=[atom_info[el]["lattice"] for el in elements], 
#                     mass=[atom_info[el]["mass"] for el in elements], 
#                     a=[atom_info[el]["a0"] for el in elements])
#     setattr(EAM_obj,"Nelements",N_ele)
    
#     if show:
#         fig = plt.figure()
#         EAM_obj.plot()
#         plt.show()
#     if not save_path is None:
#         print("Writing setfl file to {}...".format(save_path))
#         EAM_obj.write_potential(save_path)
#         return None
#     else:
#         return EAM_obj

def generate_EAM_calculator_from_linear_models(mapper, model, r_bounds, Nr,
                                               dens_bounds, Ndens, r_cut, atom_info, basis_r,
                                               basis_dens, save_path=None, show=False,
                                               ylim_emb=None, ylim_pair=None,
                                               basis_r_1stder=None, figsize=(5,5),
                                               basis_dens_1stder=None, densgiven=True,
                                               emb_density_funs=None, params_emb=None):
    if densgiven:
        assert not emb_density_funs is None, "'densgiven' = True, hence 'emb_density_funs' need to be provided as well!"
        assert not params_emb is None, "'densgiven' = True, hence 'params_emb' need to be provided as well!"
        assert all([not emb_density_funs[k] is None for k in params_emb]), "Computing the EAM design matrix with densgiven = True requires that all elements ({}) have a specified density function!".format(list(params_emb.keys()))

    elements = sorted(list(emb_density_funs))
    N_ele = len(elements)
    pair_map = {el1+el2:"".join(sorted([el1,el2])) for el1,el2 in itertools.product(elements,elements)}
    
    element_pairs_sorted = sorted(list(set(pair_map.values())))
    
    r = np.linspace(r_bounds[0],r_bounds[1],Nr)
    dens = np.linspace(dens_bounds[0],dens_bounds[1],Ndens)
    dr = r[1]-r[0]
    ddens = dens[1]-dens[0]
    
    Nb_pair = sum([v.shape[0] for v in mapper["pair"].values()]) # number of basis terms for pair energies
    Nb_emb = sum([v.shape[0] for v in mapper["emb"].values()]) if densgiven else 0 # number of basis terms for embedding energies
    N_pair = len(element_pairs_sorted) # number of element pairs
    print("Number of basis functions: pair =",Nb_pair,"embedding =",Nb_emb,"\nnumber of element pairs =",N_pair)
    
    # generate design matrices
    zero_Phi_emb = np.zeros((Ndens, Nb_emb + Nb_pair))
    zero_Phi_pair = np.zeros((Nr, Nb_emb + Nb_pair))
    print("shapes",zero_Phi_emb.shape,zero_Phi_pair.shape)
    
    _Phi_emb = None
    if densgiven:
        _Phi_emb = np.array([b(dens) for b in basis_dens]).T
    
    _Phi_pair = np.array([b(r) for b in basis_r]).T
    
    print("emb: min = {} max = {}".format(np.amin(_Phi_emb),np.amax(_Phi_emb)))
    print("pair: min = {} max = {}".format(np.amin(_Phi_pair),np.amax(_Phi_pair)))
    
    # design matrices for energies
    Phi_emb, Phi_pair = {}, {}
    # design matrices for forces
    Phi_emb_f, Phi_pair_f = {}, {}
    # energy predictions
    E_emb, E_pair = {}, {}
    
    if show:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    for el,m in mapper["emb"].items():
        
        Phi_emb[el] = np.copy(zero_Phi_emb)
        if densgiven:
            Phi_emb[el][:,m] = _Phi_emb
        
        # compute energy functions
        _y = model.predict(Phi_emb[el]) # if densgiven else np.zeros(dens.shape[0])
        E_emb[el] = _y.ravel()
        if not densgiven:
            assert np.allclose(_y,0), "Something went wrong. densgiven = {} and still some embedding energy values are nonzero!".format(densgiven)
        
        if show:
            ax.plot(dens, _y, label=el)
    
    if show:
        ax.set_title("Emb")
        ax.set_ylim(ylim_emb)
        plt.legend(loc=0)
        plt.show()
        
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    
    for pair,m in mapper["pair"].items():
        Phi_pair[pair] = np.copy(zero_Phi_pair)
        Phi_pair[pair][:,m] = _Phi_pair
        
        # compute energy functions
        _y = model.predict(Phi_pair[pair])
        E_pair[pair] = _y.ravel()
        
        if show:
            ax.plot(r, _y, label=pair)
        
    if show:
        ax.set_title("Pair")
        ax.set_ylim(ylim_pair)
        plt.legend(loc=0)
        plt.show()
        
    # embedding energy derivatives
    E_emb_f = dict()
    if not basis_r_1stder is None:
        _Phi_emb_f = None
        if densgiven:
            _Phi_emb_f = np.array([b(dens) for b in basis_dens_1stder]).T
            
        for el,m in mapper["emb"].items():
            Phi_emb_f[el] = np.copy(zero_Phi_emb)
            if densgiven:
                Phi_emb_f[el][:,m] = _Phi_emb_f
                E_emb_f[el] = model.predict(Phi_emb_f[el]).ravel()
            else:
                E_emb_f[el] = np.zeros(dens.shape[0])
    
    # pair energy derivatives
    E_pair_f = dict()
    if not basis_dens_1stder is None:
        _Phi_pair_f = np.array([b(r) for b in basis_r_1stder]).T
        for pair,m in mapper["pair"].items():
            Phi_pair_f[pair] = np.copy(zero_Phi_pair)
            Phi_pair_f[pair][:,m] = _Phi_pair_f
            E_pair_f[pair] = model.predict(Phi_pair_f[pair]).ravel()
        
    # store functions in ase EAM calculator friendly format
    embedded_energy = np.array([spline(dens,E_emb[el]) for el in elements])
    phi = np.array([[spline(r,E_pair[(pair_map[el2+el1])]) for el2 in elements] for el1 in elements])
    if densgiven:
        electron_density = np.array([spline(r,emb_density_funs[el](r)) for el in elements])
    else:
        electron_density = np.array([spline(r,np.zeros(r.shape[0])) for el in elements])
        
    if not basis_r_1stder is None:
        d_phi = np.array([[spline(r,E_pair_f[(pair_map[el2+el1])]) for el2 in elements] for el1 in elements])
    else:
        d_phi = np.array([[phi[v0,v1].derivative() for v1 in range(N_ele)] for v0 in range(N_ele)])
        
    if not basis_dens_1stder is None:
        d_embedded_energy = np.array([spline(dens,E_emb_f[el]) for el in elements])
    else:    
        d_embedded_energy = np.array([v.derivative() for v in embedded_energy])
        
    d_electron_density = np.array([v.derivative() for v in electron_density])
        
    EAM_obj = EAM(elements=elements, embedded_energy=embedded_energy,
                    electron_density=electron_density,
                    phi=phi, 
                    d_embedded_energy=d_embedded_energy,
                    d_electron_density=d_electron_density,
                    d_phi=d_phi,
                    cutoff=r_cut, form='alloy',
                    Z=[atom_info[el]["number"] for el in elements], nr=Nr, nrho=Ndens, 
                    dr=dr, drho=ddens,
                    lattice=[atom_info[el]["lattice"] for el in elements], 
                    mass=[atom_info[el]["mass"] for el in elements], 
                    a=[atom_info[el]["a0"] for el in elements])
    setattr(EAM_obj,"Nelements",N_ele)
    
    if show:
        fig = plt.figure()
        EAM_obj.plot()
        plt.show()
    if not save_path is None:
        print("Writing setfl file to {}...".format(save_path))
        EAM_obj.write_potential(save_path)
        return None
    else:
        return EAM_obj

def band_filter_wrapper(lb, ub, f=0.01):
    
    def band_filter(rho_dict):
        r = rho_dict["r"]
        m = np.ones(r.shape[0])
        m[r<lb] = 0.
        m[r>ub] = 0.
        
        x = (r-lb)/float(f)
        x4 = x**4
        x = (r-ub)/float(f)
        xp4 = x**4
        
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] *= m * x4/(1+x4) * xp4/(1+xp4)
        
        return rho_dict
    return band_filter

def load_regressed_rho(path_rhos,operations=[],conv=None,params=None,
        return_bounds=False,show=False):
    """Loads a file containing information about previously regressed rho(r) functions.

    Parameters
    ----------
    path_rhos : str, list of str or tuple of str
        path(s) to ".rhos" file(s)
    operations : list of str
        can contain "normalize", "absolute", "shift". does the respective operations to
        the rho(r) functions in the order they are put in the list.
    return_bounds : boolean
        return min and max density value observed after applying the operations
    """
    print("Loading regression data from {}...".format(path_rhos))

    if isinstance(path_rhos,str):
        with open(path_rhos,"rb") as f:
            rho_dict = pickle.load(f)
    elif isinstance(path_rhos,(list,tuple)):
        for i,p in enumerate(path_rhos):
            with open(p,"rb") as f:
                if i==0:
                    rho_dict = pickle.load(f)
                else:
                    rho_dict.update(pickle.load(f))

    def _normalize(rho_dict):
        rho_min = min([min(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        rho_max = max([max(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        drho = rho_max - rho_min
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] /= drho
        return rho_dict

    def _absolute(rho_dict):
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] = np.absolute(rho_dict["rhos"][s])
        return rho_dict

    def _shift(rho_dict):
        rho_min = min([min(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        for s in rho_dict["rhos"]:
            rho_dict["rhos"][s] -= rho_min
        return rho_dict
            
    def _convolute(params,rho,fun_type="exp"):
        if fun_type == "exp":
            def fun(r):
                return (rho-params[0])*np.exp(-r)
        elif fun_type == "psi":
            def fun(r):
                x = (r-params[1])/float(params[2])
                x4 = x**4
                return (rho-params[0])* x4/(1.+x4)
        elif fun_type == "psi2":
            def fun(r):
                x = (r-params[1])/float(params[2])
                x4 = x**4
                x = (r-params[3])/float(params[4])
                xp4 = x**4
                return (rho-params[0])* x4/(1+x4) * xp4/(1+xp4)
        else:
            raise NotImplementedError
        return fun

    implemented_ops = {"normalize":_normalize,"absolute":_absolute,"shift":_shift}
    implemented_convs = {"exp","psi","psi2"}

    for op in operations:
        if op in implemented_ops:
            rho_dict = implemented_ops[op](rho_dict)
        elif callable(op):
            rho_dict = op(rho_dict)
        else:
            raise NotImplementedError

    if conv is not None:
        assert conv in implemented_convs, "Assertion failed - conv '{}' is not one of the implemented convolutions: {}".format(conv,implemented_convs)
        
        lb = min([min(rho_dict["rhos"][v]) for v in rho_dict["rhos"]])
        for s in rho_dict["rhos"]:
            
            if conv == "exp":
                _params = [0]#[lb]
            elif conv == "psi":
                _params = [0]+list(params)#[lb]+list(params)
            else:
                raise NotImplementedError

            fun = _convolute(_params,rho_dict["rhos"][s],fun_type=conv)
            rho_dict["rhos"][s] = fun(rho_dict["r"])
            
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ks = sorted(rho_dict["rhos"].keys())
        for k in ks:
            ax.plot(rho_dict["r"],rho_dict["rhos"][k],label=k)
        ax.set_xlabel("r [AA]",fontsize=14)
        ax.set_ylabel("rho (r)",fontsize=14)
        ax.grid()
        plt.legend(loc=0)
        plt.tight_layout()
        plt.show()

    if return_bounds:
        rho_lb = min([min(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        rho_ub = max([max(rho_dict["rhos"][s]) for s in rho_dict["rhos"]])
        return rho_dict, rho_lb, rho_ub
    else:
        return rho_dict