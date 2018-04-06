import numpy as np
import pandas as pd
from scipy import misc, optimize
import copy, itertools
from ase.lattice.cubic import SimpleCubicFactory
from ase.spacegroup import crystal as ase_crystal
import matplotlib.pylab as plt

from .spatial import get_ultracell, get_neighbors

from .misc import show_atoms, print_error_distributions_properties,\
    show_rattled_atoms, show_Efuns, show_predict_vs_true,\
    show_deviation_distributions, parse_LAMMPS_custom_dump,\
    wrap_positions, load_regressed_rho, create_LAMMPS_traj,\
    cell2box, box2cell, wrap_lammps_pos, lammps2ase,\
    lammpstrj_to_design_matrices, periodic_distances, \
    fclusterdata, get_GB_position_DataFrame, get_GB_positions_from_histograms,\
    square_cluster_statistics


from .features import make_array, get_crystal_design_matrix, get_angles, get_q, get_q2, get_q3,\
    ThreeBodyAngleFeatures, BondOrderParameterFeatures,\
    taper_fun_wrapper, DistanceTaperingFeatures_2body, DistanceExpTaperingFeatures_2body,\
    DistanceCosTaperingFeatures_2body, DistanceCosExpTaperingFeatures_3body,\
    DistanceCosExpTaperingFeatures_3body2, CentroSymmetryParameterFeatures,\
    DistanceCosTapering_basis, DistanceCosTapering_basis_1stDerivative,\
    ElementCountFeatures, DistanceCosTaperingFeatures_3body, DistanceGaussTaperingFeatures_2body,\
    DistanceSineCosineTaperingFeatures_2body, DistanceSineCosineTapering_basis,\
    DistanceGaussTapering_basis, DistancePolynomialTapering_basis, DistancePolynomialTaperingFeatures_2body,\
    DistanceGenericTaperingFeatures_2body

from .eam import get_rhos, get_r_and_dens_values,\
    get_embedding_density_functions, element_emb_filters, element_pair_filters,\
    collect_EAM_design_matrices, generate_EAM_calculator_from_linear_models,\
    load_regressed_rho, band_filter_wrapper, get_mapper

from .electron_density import rescale_and_shift_densities

from .classification import scipy_gmm_wrapper, GaussianMixtureClassifier,\
    assign_chemical_disorder_labels, get_decomposed_models

from .crystal import introduce_vacancies, thermal_vacancy_concentration,\
    create_chemical_disorder, insert_spherical_precipitate, get_bicrystal_directions,\
    get_crystals

from .diffusion import diffusion_solver, pde_measure_wrapper

from .gap import get_GAP_matrix_v1, get_GAP_matrix_v2, get_kvec_v1, get_kvec_v2,\
    GAPRegressor

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.datasets import make_moons, make_circles, make_classification
# from sklearn.neural_network import MLPClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from sklearn.gaussian_process import GaussianProcessClassifier
# from sklearn.gaussian_process.kernels import RBF
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def wrapper_optimize_cell(atoms, kind="direct"):
    """Wrapper for the optimization of crystal supercell boxes (cells).

    Parameters
    ----------
    atoms : ase Atoms instance
    kind : str, optional, default "direct"
        Determines how "x" is interpreted by optimize_cell.
        If kind = "direct" then x is interpreted as the diagional of the cell.
        If kind = "factor" then x is interpreted as a factor of the individual
        basis vectors.
    """
    assert kind in ["direct","factor"], "'kind' parameter not understood!"
    _atoms = copy.deepcopy(atoms)
    cell = _atoms.get_cell()
    fpos = _atoms.get_scaled_positions(wrap=True)
    
    diag = (np.array([0,1,2],dtype=int), np.array([0,1,2],dtype=int))
    I = np.eye(3)
    def optimize_cell(x):
        #cell[diag] = x
        _I = I.copy()
        if kind == "factor":
            if isinstance(x,(float,int)) or (isinstance(x,(list,tuple,np.ndarray)) and len(x)==1):
                _I[diag] = x
            elif isinstance(x,(list,tuple,np.ndarray)) and len(x) == 3:
                _I = np.diag(x)
            else:
                raise ValueError("'x' (%s) not understood!"%x)
            _cell = _I.dot(cell)
        elif kind == "direct":
            _I[diag] = x
            _cell = _I
        
        _atoms.set_cell(_cell)
        _atoms.set_positions(np.dot(fpos,_cell))
        _atoms.set_scaled_positions(fpos)
        return _atoms.get_potential_energy()
    return optimize_cell

class L12Factory(SimpleCubicFactory):
    "A factory for creating AuCu3 (L1_2) lattices."
    bravais_basis = [[0, 0, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]
    element_basis = (0, 1, 1, 1)

class CrystalRattler:
    """Rattles a crystal.
    
    Parameters
    ----------
    
    rattle_rv : instance continuous scipy.stats random variable
        Random variable to generate atom displacements.
    """
    
    def __init__(self, rattle_rv=None):
        self.rattle_rv = rattle_rv
        
    @staticmethod
    def rattle(positions, rattle_rv, N=1):
        """Rattle rattle...

        Parameters
        ----------

        positions : float np.ndarray of shape (Na,3)
            Positions of Na atoms in direct space.

        rattle_rv : instance continuous scipy.stats random variable
            Random variable to generate atom displacements.

        N : int, optional, default 1
            Number of perturbed crystals to produce.

        Returns
        -------

        rattled_positions : float np.ndarray of shape (N,Na,3)
        """
        Na, d = positions.shape
        dx = rattle_rv.rvs(size=(N,Na,d))
        return positions + dx
    
    def run(self, atoms, N=1):
        """Runs the rattler.
        
        Parameters
        ----------

        atoms : ase Atoms object.

        N : int, optional, default 1
            Number of perturbed crystals to produce.

        Returns
        -------

        rattled_positions : float np.ndarray of shape (N,Na,3)
        """
        positions = atoms.get_positions(wrap=True)
        return self.rattle(positions, self.rattle_rv, N=N)

def rattle_crystal_multiple_rvs(Nrattle, rattle_rvs, atoms, rpositions=None, 
                            Natoms=None, t_e=None, t_f=None, species=None,
                            cells=None, relax=False, relax_kwargs=dict(method="BFGS"),
                            opt_cell_kind="iso", calculator=None):
    """Rattling a crystal with multiple rattle rvs.

    Can be called once or multiple times passing Natoms, t_e, t_f, species
    and cells to accumulate info.

    Parameters
    ----------

    Nrattle : int
        Number of perturbed crystals to produce for each given rattle rv.
    
    rattle_rvs : list of stats.rv_continuous
        Random variables (rvs) to use for sampling perturbations 
        to ideal crystals.
    
    atoms : ase Atoms instance
        Crystal to perturb.
    
    rpositions : None or list, optional, default None
        List of positions.

    Natoms : None or list, optional, default None
        List of numbers of atoms.
    
    t_e : None or list, optional, default None
        List of per atom energies.

    t_f : None or list, optional, default None
        List of atom forces.
    
    species : None or list, optional, default None
        List of species.
    
    cells : None or list, optional, default None
        List of crystal cells.

    Returns
    -------

    rpositions : list 
        Atom positions. List of len(rattle_rvs) np.ndarrays 
        of shape (Nrattle,Na,3). Each np.ndarray contains multiple 
        perturbed versions of the original crystal given one 
        of the rattle_rvs.
        Example: [np.ndarray(...), np.ndarray(...), ...]
    
    Natoms : list
        Numbers of atoms. List of len(rattle_rvs) lists of 
        length Nrattle.
        Example: [[2, 2, 2], [2, 2, 2], ...]
        
    cells : list
        Simulation boxes. List of len(rattle_rvs) lists each 
        containing Nrattle cell np.ndarrays of shape (3,3).
    
    species : list
        Atom species. List of len(rattle_rvs) lists each 
        containing Nrattle species np.ndarrays of shape (Na,)

    t_e : list
        Returned only if the atoms object has a calculator. 
        Energies of individual atom. List of len(rattle_rvs) 
        lists of length Nrattle each of lenth Na.
    
    t_f : list
        Returned only if the atoms object has a calculator. 
        Forces of individual atom. List of len(rattle_rvs) 
        lists of length Nrattle each containing an np.ndarray
        of shape (Na*3,).
    Notes
    -----
    
    Na referred to above is the number of atoms in the given 
    ase's atoms object.

    Example
    -------
    >>> import atomtoolbox as atb
    >>> Nrattle = 5
    >>> rattle_rvs = [stats.norm(loc=0,scale=.001), stats.norm(loc=0,scale=.01),
    ...                  stats.norm(loc=0,scale=.015), stats.norm(loc=0,scale=.05)]
    >>> CRs = [atb.CrystalRattler(rattle_rv=rattle_rv) for rattle_rv in rattle_rvs]

    >>> cell = atoms.get_cell()

    >>> rpositions, Natoms, t_e, t_f, cells, species = None, None, None, None, None, None

    >>> for sname in sorted(atoms_dict):
    >>>     print("rattling ",sname)
    >>>     rpositions, Natoms, t_e, t_f, cells, species = atb.rattle_multiple_crystal(Nrattle, rattle_rvs,\
    ...                                atoms_dict[sname], rpositions=rpositions, Natoms=Natoms, t_e=t_e, t_f=t_f,\
    ...                                species=species, cells=cells)
    """
    if not calculator is None:
        try:
            atoms.set_calculator(calculator)
            e = atoms.get_potential_energy()
            has_calc = True    
        except:
            has_calc = False
    try:
        e = atoms.get_potential_energy()
        has_calc = True
    except RuntimeError:
        has_calc = False
    except:
        print("WTF!")
        raise

    if has_calc:
        all_None = all([rpositions is None, Natoms is None, t_e is None,\
                        t_f is None, species is None, cells is None])
        all_list = all([isinstance(rpositions,list), isinstance(Natoms,list),\
                        isinstance(cells,list), isinstance(t_e,list),\
                        isinstance(t_f,list), isinstance(species,list)])
    else:
        all_None = all([rpositions is None, Natoms is None, species is None,\
                        cells is None])
        all_list = all([isinstance(rpositions,list), isinstance(Natoms,list),\
                        isinstance(cells,list), isinstance(species,list)])

    if all_None:
        rpositions = []
        Natoms = []
        t_e = []
        t_f = []
        species = []
        cells = []
    else:
        assert all_list, "All parameters rpositions, Natoms, t_e, cells, t_f and species either need to be None or lists."
    
    CRs = [CrystalRattler(rattle_rv=rattle_rv) for rattle_rv in rattle_rvs]

    cell = atoms.get_cell()
    invcell = np.linalg.inv(cell)

    for i,CR in enumerate(CRs):
        
        _rpositions = CR.run(atoms,N=Nrattle)
        _rpositions = get_wrapped_positions(_rpositions,cell)

        if relax: # relaxing the cell
            _cells = [None for v in range(Nrattle)]
            for j in range(Nrattle):
                _atoms = copy.deepcopy(atoms)
                _atoms.set_positions(_rpositions[j])
                opt_fun = wrapper_optimize_cell(_atoms,kind="factor")
                if opt_cell_kind == "iso":
                    _res = optimize.minimize(opt_fun,[1.],**relax_kwargs)
                    _cells[j] = _res["x"]*cell
                elif opt_cell_kind == "diag":
                    _res = optimize.minimize(opt_fun,np.ones(3),**relax_kwargs)
                    _cells[j] = np.diag(_res["x"]).dot(cell)
                else:
                    raise NotImplementedError
                _atoms.set_cell(_cells[j])
                _rpositions[j] = _atoms.get_positions(wrap=True)
            #rpositions.append([_rpositions[j] for j in range(Nrattle)])
            #cells.append(_cells)
        else:
            _cells = [cell for j in range(Nrattle)]
        rpositions.append(_rpositions)
        cells.append(_cells)
        
        _species = [atoms.get_chemical_symbols() for v in range(Nrattle)]
        species.append(_species)
        Natoms.append([atoms.positions.shape[0] for v in range(Nrattle)])
        
        if has_calc:
            # for i in range(len(CRs)):
            #     print("E:i",i)
            _t_e = [None for v in range(_rpositions.shape[0])]
            _t_f = [None for v in range(_rpositions.shape[0])]
            for j in range(len(_rpositions)):
                _pos = _rpositions[j]
                _cell = _cells[j]
                _spec = _species[j]
                
                _inv_cell = np.linalg.inv(_cell)

                _atoms = ase_crystal(_spec, _pos.dot(_inv_cell), cell=_cell,
                                    pbc=(1,1,1)) #copy.deepcopy(atoms)
                if not calculator is None:
                    _atoms.set_calculator(calculator)
                else:
                    _atoms.set_calculator(atoms.get_calculator())
                #_fpos = np.dot(_pos,invcell)
                #_atoms.set_scaled_positions(_fpos)
                #_atoms.set_positions(_pos)
                _t_e[j] = _atoms.get_potential_energy()#/float(_fpos.shape[0])
                
                _t_f[j] = _atoms.get_forces().ravel()
            #t_e.append([_t for v in range(_fpos.shape[0])])
            t_e.append(_t_e)
            t_f.append(_t_f)

    if has_calc:
        return rpositions, Natoms, cells, species, t_e, t_f
    else:
        return rpositions, Natoms, cells, species
    
def flattening_rattled_crystal_data(rpositions, Natoms, cells, species, energy=None, force=None):
    """Flattens data obtained from rattle_crystal_multiple_rvs.
    
    Parameters
    ----------
    
    rpositions : list 
        Atom positions. List of len(rattle_rvs) np.ndarrays 
        of shape (Nrattle,Na,3). Each np.ndarray contains multiple 
        perturbed versions of the original crystal given one 
        of the rattle_rvs.
        Example: [np.ndarray(...), np.ndarray(...), ...]
    
    Natoms : list
        Numbers of atoms. List of len(rattle_rvs) lists of 
        length Nrattle.
        Example: [[2, 2, 2], [2, 2, 2], ...]
        
    cells : list
        Simulation boxes. List of len(rattle_rvs) lists each 
        containing Nrattle cell np.ndarrays of shape (3,3).
    
    species : list
        Atom species. List of len(rattle_rvs) lists each 
        containing Nrattle species np.ndarrays of shape (Na,)

    energy : list
        Energies of individual atom. List of len(rattle_rvs) 
        lists of length Nrattle each of lenth Na.
    
    force : list
        Forces of individual atom. List of len(rattle_rvs) 
        lists of length Nrattle each containing an np.ndarray
        of shape (Na*3,).
        
    Returns
    -------
    
    rpositions_flat : list of Np float np.ndarrays each of shape (Na,3)
    
    Natoms_flat : list of int of length Np
        
    cells_flat : list of Np float np.ndarrays of shape (3,3)
    
    species_flat : list of Np lists each containing Na strings.

    energy_flat : float np.ndarray of shape (Nn,) or nothing
    
    force_flat : float np.ndarray of shape (Nn*3,) or nothing
    
    Notes
    -----
    Na is a shorthand for the number of atoms in a crystal 
    (may vary by crystal).
    Np is a shorthand for len(rattle_rvs)*Nrattle representing all
    perturbed crystals.
    Nn is a shorthand for Np*Na (assuming all crystals have the same
    numbers of atoms)
    
    """
    rpositions_flat = list(itertools.chain(*[[v for v in v2] for v2 in rpositions]))
    cells_flat = list(itertools.chain(*cells))
    species_flat = [x for y in species for x in y]
    Natoms_flat = [x for y in Natoms for x in y]
    
    if not energy is None:
        energy_flat = np.hstack(energy)
    
    if not force is None:
        force_flat = np.hstack(force).ravel()
    
    if energy is None and force is None:
        return rpositions_flat, Natoms_flat, cells_flat, species_flat

    elif (not energy is None) and (not force is None):
        return rpositions_flat, Natoms_flat, cells_flat, species_flat, energy_flat, force_flat

    elif (energy is None) and (not force is None):
        return rpositions_flat, Natoms_flat, cells_flat, species_flat, force_flat

    elif (not energy is None) and (force is None):
        return rpositions_flat, Natoms_flat, cells_flat, species_flat, energy_flat

    else:
        raise


def get_wrapped_positions(positions,cell):
    """Wraps real space positions by converting to fractional space.
    """
    invcell = np.linalg.inv(cell)
    fpos = np.dot(positions,invcell)
    fpos = np.mod(fpos,1)
    return np.dot(fpos,cell)

def show_rvm_performance(niter,logbook,log=True):
    fig = plt.figure()
    x_iter = np.arange(len(logbook["L"]))
    ax = fig.add_subplot(131)
    
    ax.plot(x_iter,logbook["L"],'-',linewidth=2,alpha=0.8)
    ax.set_xlabel("#iteration")
    ax.set_ylabel("Log likelihood")
    ax.grid()

    ax2 = fig.add_subplot(132)
    ax2.plot(x_iter,logbook["beta"],'-')
    ax2.set_xlabel('#iteration')
    if log: ax2.set_yscale("log")
    ax2.set_ylabel("beta")
    ax2.grid()

    ax3 = fig.add_subplot(133)
    ax3.plot(x_iter,logbook["mse"],'-') 
    ax3.set_xlabel('#iteration')
    if log: ax3.set_yscale("log")
    ax3.set_ylabel("mse")
    ax3.grid()

    plt.suptitle("RVM approach")
    plt.tight_layout()
    plt.show()

def collect_multiple_crystal_info(gip, bonds, rpositions=None, 
                            Natoms=None, t_e=None, t_f=None, species=None, cells=None):
    
    if all([rpositions is None, Natoms is None, t_e is None, t_f is None, species is None,
            cells is None]):
        rpositions = []
        Natoms = []
        t_e = []
        t_f = []
        species = []
        cells = []
    else:
        assert all([isinstance(rpositions,list), isinstance(Natoms,list), isinstance(cells,list),
                    isinstance(t_e,list), isinstance(t_f,list), isinstance(species,list)]), "All parameters rpositions, Natoms, t_e, cells, t_f and species either need to be None or lists."
    
    assert len(bonds)==len(gip), "bonds ({}) and gip ({}) instances need to contain the same number of supercells!".format(len(bonds),len(gip))
    Nbonds = len(bonds)
    
    for i in range(Nbonds):
        
        _rpositions = np.copy(gip[i].get_positions(wrap=True))
        _Natoms = _rpositions.shape[0]
        _cell = np.copy(gip[i].get_cell())
        #_rpositions = get_wrapped_positions(np.dot(_rpositions,_cell),_cell)
        rpositions.append(np.dot(_rpositions,_cell))
        
        Natoms.append(_Natoms)
        cells.append(_cell)
        species.append(gip[i].species)
        
        t_e.append([bonds[i].t["energy"]*len(gip[i].species)])# for v in range(_Natoms)])
        t_f.append(gip[i].forces.ravel())
        
    return rpositions, Natoms, cells, species, t_e, t_f

def taper_fun_wrapper(_type="x4ge",**kwargs):
    if _type=="x4ge":
        def x4_fun(x):
            x4 = ((x-kwargs["a"])/float(kwargs["b"]))**4
            if isinstance(x,(int,float)):
                x4 = 0 if x>=kwargs["a"] else x4
            else:
                x4[x>=kwargs["a"]] = 0
            return x4/(1.+x4)
        return x4_fun
    
    elif _type=="x4le":
        def x4_fun(x):
            x4 = ((x-kwargs["a"])/float(kwargs["b"]))**4
            if isinstance(x,(int,float)):
                x4 = 0 if x<=kwargs["a"] else x4
            else:
                x4[x<=kwargs["a"]] = 0
            return x4/(1.+x4)
        return x4_fun
    
    elif _type=="Behlerge":
        def Behler_fun(x):
            y = .5 * (np.cos(np.pi*x/float(kwargs["a"])) + 1.)
            if isinstance(x,(int,float)):
                y = 0 if x>=kwargs["a"] else y
            else:
                y[x>=kwargs["a"]] = 0
            return y
        return Behler_fun
    
    elif _type=="Behlerle":
        def Behler_fun(x):
            y = .5 * (np.cos(np.pi*x/float(kwargs["a"])) + 1.)
            if isinstance(x,(int,float)):
                y = 0 if x<=kwargs["a"] else y
            else:
                y[x<=kwargs["a"]] = 0
            
            return y
        return Behler_fun
    
    elif _type=="Ones":
        def ones_fun(x):
            if isinstance(x,np.ndarray):
                return np.ones(x.shape)
            elif isinstance(x,(float,int)):
                return 1
            else:
                raise
        return ones_fun
    
    else:
        raise NotImplementedError("_type '{}' unknown.".format(_type))

def get_feature_DataFrame(feature_class, taper_fun, taper_type, element_filter,\
                positions, species, upositions, uspecies, uindices_neigh, **feature_params):
    
    fc = feature_class(taper_fun=taper_fun, taper_type=taper_type, \
        element_filter=element_filter,**feature_params)
    Phi = fc.fit_transform(positions, species, upositions, uspecies, 
                                    uindices_neigh)

    df = pd.DataFrame(data=Phi)
    df.columns = [str(v) for v in df.columns]
    return df

def distribution_wrapper(dis,size=None,single=True):
    """Wraps scipy.stats distributions for RVM initialization.

    Parameters
    ----------
    size : int
        How many samples to draw (if given, see 'single').
    single : boolean
        Whether or not a single float value is to be returned or an array of values.
        If single == False then either 'size' samples are drawn or otherwise if the
        design matrix is provided as an argument of the wrapped function 'samples'
        then as M samples are drawn (N, M = X.shape).
    """
    def samples(X=None):
        if single:
            return dis.rvs(size=1)[0]
        else:
            if isinstance(size,int):
                return dis.rvs(size=size)
            elif isinstance(X,np.ndarray):
                return dis.rvs(size=X.shape[1])
            else:
                raise ValueError("size is not properly specified")
    return samples

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