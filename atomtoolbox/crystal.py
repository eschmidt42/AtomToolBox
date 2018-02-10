from __future__ import absolute_import, print_function, division

import ase, copy
from ase.lattice.spacegroup import crystal as ase_crystal
import numpy as np

from ase.lattice.cubic import FaceCenteredCubic
import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

import os
import inspect
import pickle
import itertools
import copy

from math import gcd
from ase.visualize import view
from ase import build

import sys # to be removed
sys.path.append("E:/PetProjects/GBpy/") # to be removed
import GBpy
import GBpy.find_csl_dsc as gb_csl
import GBpy.integer_manipulations as int_man
import GBpy.bp_basis as plb

def thermal_vacancy_concentration(Hf, Sf, T):
    """Computes the concentration of thermal vacancies.
    
    Parameters
    ----------
    Hf : float
        Vacancy formation enthalpy [eV].
    Sf : float
        Vacancy formation entropy [k_B].
    T : float
        Temperature [K].
        
    Returns
    -------
    c_vac : float
        Concentration of vacancies [1].
    """
    kb = 8.6173324e-5 #[eV/K]
    assert T >= 0, "Temperature needs to be at least 0 K!"

    c_vac = np.exp(Sf) * np.exp(-Hf/(kb*T))
    return c_vac 

def introduce_vacancies(atoms, c_vac=None, N_vac=None, select="Ni", 
                        verbose=False): 
    """Introduces vacancies into a crystal.
    
    Parameters
    ----------
    atoms : ase crystal object
    
    c_vac : float, optional, default None
        Concentration of vacancies in [0,1].
        
    N_vac : int, optional, default None
        Number of vacancies.
    
    select : str, optional, default "Ni"
        Specifies which atom types to select for the introduction
        of vacancies.
        
    Returns
    -------
    atoms : ase crystal object
        Modified ase crystal object.
    """
    _atoms = copy.deepcopy(atoms)
        
    elements = np.array(_atoms.get_chemical_symbols(), dtype=str)
    elements_set = set(elements)
    assert select in elements_set, "'select' ({}) is not one of the elements in the atoms object: {}".format(select,elements_set)
    
    idx_el = np.where(elements==select)[0]
    Na = idx_el.shape[0]
    
    do_c = not c_vac is None
    do_N = not N_vac is None
    assert do_c != do_N, "Either 'c_vac' or 'N_vac' needs to be provided (but not both)!"
    if do_c:
        assert 0.<=c_vac<=1., "'c_vac' needs to be between 0 and 1!"
    elif do_N:
        assert 0<=N_vac<Na and isinstance(N_vac,int), "'N_vac' needs to be positive, smaller than the total number of {} atoms and an integer!".format(select)
    
    
    if do_c:
        N_vac = int(Na*c_vac)
    
    if verbose:
        print('Introducing {} thermal vacancies...'.format(N_vac))
    
    idx_vac = np.random.choice(idx_el, size=N_vac, replace=False)
    del _atoms[idx_vac]
    
    return _atoms

def create_chemical_disorder(atoms, concentrations):
    """Creates chemical disorder.
    
    Parameters
    ----------
    atoms : ase Atoms instance
        Contains only a single element.
        
    concentractions : dict
        Keys are chemical elements. Values are concentrations in [1].
        Assuming the single element in 'atoms' is Ni. If a chemical disorder 
        is supposed to be generated using only one more element then 
        'concenctrations' could look like:
            {"Al":.20}
        For two elements similarly: {"Al":.12, "Nb":.002}
        The remainder is the base element.
        
    Returns
    -------
    atoms : copy of given atoms object
        Has a modified element composition.
    """
    assert isinstance(concentrations,dict), "'concentrations' needs to be a dictionary."
    assert sum(concentrations.values())<1., "The sum of concentrations cannot exceed 1!"
    assert all([v>0 for v in concentrations.values()]), "All concentrations need to be larger than zero!"
    
    _atoms = copy.deepcopy(atoms)
    elements = np.array(_atoms.get_chemical_symbols(),dtype=str)
    N = len(elements)
    idx = np.arange(N)
    for el,pct in concentrations.items():
        _idx = np.random.choice(idx,size=int(N*pct), replace=False)
        elements[_idx] = el
        idx = np.setdiff1d(idx,_idx)
    _atoms.set_chemical_symbols(list(elements))
    return _atoms

def insert_spherical_precipitate(atoms, atoms_ppt, ppt_pos=(.5,.5,.5), ppt_r=5, verbose=False,
                                 shift_type="origin"):
    """Inserts a L12 precipitate into the crystal.
    
    Parameters
    ----------
    ppt_pos : tuple of float, optional, default (.5,.5,.5)
        Precipitate position in fractional coordinates of simulation box.
    
    ppt_r : float or int, optional, defautl 5
        Precipitate radius in Angstrom.
        
    Returns
    -------
    _atoms : 
        Modified copy of the atoms object.
    """
    from ase import Atoms
    from atomtoolbox import get_ultracell
    
    assert len(ppt_pos)==3 and all([isinstance(v,float) and 0<=v<=1 for v in ppt_pos]), "'ppt_pos' ({}) needs to be given in fractional coordinates in 3d!"
    assert isinstance(ppt_r,(float,int)), "ppt_r needs to be an int or float value!"
    
    _atoms = copy.deepcopy(atoms)
    Na = _atoms.positions.shape[0]
    cell = _atoms.get_cell()
    
    ppt_pos = np.dot(ppt_pos,cell)
    if verbose:
        print('Inserting L12 prectipitate with r = {} A at {}'.format(ppt_r,' A, '.join(map(str,ppt_pos))))
    
    ## expand precipitate crystal
    ppt_upos, ppt_uspecies, ppt_uidx = get_ultracell(atoms=atoms_ppt, r_cut=1.5*ppt_r)
    
    ## select host atoms for removal
    host_positions = _atoms.get_positions(wrap=True)
    host_species = np.array(_atoms.get_chemical_symbols(), dtype=str)
    idx_host_survive = np.where(np.linalg.norm(host_positions - ppt_pos, axis=1) > ppt_r)[0]
    
    ## select precipitate atoms for survival
    if shift_type == "origin":
        ppt_shift = np.zeros(3)
    elif shift_type == "mean":
        ppt_shift = ppt_upos.mean(axis=0)
        
    idx_ppt_survive = np.where(np.linalg.norm(ppt_upos - ppt_shift, axis=1) <= ppt_r)[0]
    ppt_upos = ppt_upos - ppt_shift + ppt_pos
    
    host_positions = np.vstack((host_positions[idx_host_survive,:], ppt_upos[idx_ppt_survive,:]))
    host_species = np.hstack((host_species[idx_host_survive], ppt_uspecies[idx_ppt_survive]))
    
    new_atoms = Atoms(host_species, positions=host_positions, cell=cell, pbc=True)
            
    return new_atoms

def convert_array(x, dtype=np.int64, check=True):
    assert isinstance(x,np.ndarray), "x is not an array!"
    _x = np.array(x,dtype=dtype)
    assert np.allclose(_x,x), "Given x and type cast ({}) of x (_x) diverge! x ({}) is not close to _x ({}).".format(dtype, x.astype(str), _x.astype(str))
    return _x

def get_gcded_vector(x):
    x = convert_array(x, dtype=np.int64, check=True)
    not_zero = np.where(np.logical_not(np.isclose(x,0)))[0]
    if not_zero.shape[0] == 3:
        x_gcds = [gcd(x[0],x[1]), gcd(x[1],x[2]), gcd(x[0],x[2])]
        x_gcd = min(x_gcds)
        #x_gcd = gcd(x[0],x[1])
        
        x = x/x_gcd
    elif not_zero.shape[0] == 2:
        x_gcd = gcd(x[not_zero[0]],x[not_zero[1]])
        x = x/x_gcd
    elif not_zero.shape[0] == 1:
        x[not_zero[0]] = 1.
    else:
        raise ValueError("x = {} consists only of zeros.".format(x))
        
    return convert_array(x, dtype=np.int64, check=True)

def find_rectangular_plane(int_L_2d_csl_po1, n=10):
    """Finds an approximately rectangular plane.
    
    Iterates through [1,n] x [1,n] to find the GB CSL
    plane vectors which are approximately orthogonal
    spanning the smallest possible area (to minimize the 
    size of the simulation).
    
    Parameters
    ----------
    int_L_2d_csl_po1 : int np.ndarray of shape (3,2)
        Contains the integer form of the CSL basis vectors
        describing the GB plane in the po1 reference system.
        po1 = orthogonal version of the primitive basis of the 
        lower crystal.
    
    n : int, optional, default 10
        Beware that the search is brute force and thus scales n^4.
        
    Returns
    -------
    xy1_plane_po1 : int np.ndarray of shape (3,2)
        The plane, also in po1.
    """
    int_range = np.arange(-n,n+1) #np.arange(n*n).reshape((n,n))
    
    _n = int_range.shape[0]
    vec_prod = np.ones(_n**4)
    vec_prod[:] = np.inf
    vec_cros  = np.ones(_n**4)
    vec_cros[:] = np.inf

    _i = 0
    all_lm_a = np.zeros((_n**4,2))
    all_lm_c = np.zeros((_n**4,2))
    
    def lm_invalid(_l_a, _m_a, _l_c, _m_c):
        if _l_c == _m_c == 0:
            return True
        elif _l_a == _l_c and _m_a == _m_c:
            return True
        else:
            return False
    
    for (_l_a, _m_a) in itertools.product(int_range, int_range):
        if _l_a == _m_a == 0: 
            continue
        
        a = int_L_2d_csl_po1[:,0]*_l_a + int_L_2d_csl_po1[:,1]*_m_a
        
        for (_l_c, _m_c) in itertools.product(int_range,int_range):
            if lm_invalid(_l_a, _m_a, _l_c, _m_c): 
                continue
            
            c = int_L_2d_csl_po1[:,0]*_l_c + int_L_2d_csl_po1[:,1]*_m_c
            vec_prod[_i] = np.absolute(a.dot(c))
            vec_cros[_i] = np.linalg.norm(np.cross(a,c))

            all_lm_a[_i,:] = [_l_a,_m_a]
            all_lm_c[_i,:] = [_l_c,_m_c]

            _i += 1

    idx_prod = np.argsort(vec_prod)[:10]
    idx_cros = np.argsort(vec_cros[idx_prod])
    
    lm_a = all_lm_a[idx_prod[idx_cros[0]],:]
    lm_c = all_lm_c[idx_prod[idx_cros[0]],:]
        
    xy1_plane_po1 = np.array([ int_man.int_finder(int_L_2d_csl_po1.dot(lm_a)),
                             int_man.int_finder(int_L_2d_csl_po1.dot(lm_c))]).T
    
    xy1_plane_po1 = np.array([ get_gcded_vector(xy1_plane_po1[:,0]),
                             get_gcded_vector(xy1_plane_po1[:,1])]).T
    
    return xy1_plane_po1

def get_bicrystal_directions(inds_po, bravais_lattice="fcc",lattice_constant=1.,sigma='3',sigma_number=0,
                             verbose=False):
    """Generates both directions vectors for both components of a bicrystal using GBpy.
    
    Using both arrays of directions ase's FaceCenteredCubic function can be used to 
    set up the bicrystal components and then generate the complete bicrystal.
    
    Parameters
    ----------
    inds_po : int np.ndarray of shape (3,) or tuple or list of ints
        Orthogonal lattice direct space indices for the GB normal of the
        lower crystal.
    
    bravais_lattice : str, optional, default "fcc"
        The Bravais lattice.
        
    lattice_constant: float, optional, default 1.
    
    sigma : str, optional, default '3'
        The Sigma value.
        
    sigma_number : int, optional, 0
        A given Sigma may have more than one rotation matrix associated with it. This
        value is the list index allowing further selection.
        
    verbose : boolean, optional, default False
    
    Returns
    -------
    directions : dict of dicts of float np.ndarrays of shape (3,)
        Contains "upper" and "lower" keys with respective values containing
        the directions indicated as "x", "y" and "z". "z" is parallel or anti-parallel
        to inds_po for the lower and upper values respectively. np.array
        forms of "upper" and "lower", with directions as row vectors, can directly be 
        used in ase's FaceCenteredCubic function.
    """
    
    lattices = {"fcc":np.array([[0., .5, .5],
                                [.5, 0., .5],
                                [.5, .5, 0.]])}
    
    assert bravais_lattice in lattices, "bravais_lattice '{}' not known/implemented, did you mean any of the following: {}?".format(bravais_lattice,lattices.keys())
    assert isinstance(sigma,str) and int(sigma), "Parameter 'sigma' needs to be given in form of a str which contains an integer value (given '{}').".format(sigma)
    if not isinstance(inds_po,np.ndarray):
        assert all([isinstance(v,int) for v in inds_po]), "inds_po has to contain only integer values (given '{}').".format(inds_po)
        inds_po = np.array(inds_po,dtype=np.int64)
    else:
        assert inds_po.dtype in [np.int, np.int32, np.int64], "inds_po.dtype has to be one of np.int, np.int32 or np.int64 (given '{}').".format(inds_po.dtype)
        inds_po = inds_po.astype(np.int64)
    
    # load data to generate the rotation matrix
    gbpy_dir = os.path.dirname((inspect.getfile(GBpy)))
    pkl_dir = gbpy_dir + '/pkl_files'
    
    pkl_file = pkl_dir + '/cF_Id_csl_common_rotations.pkl'
    if os.name != "posix":
        pkl_file = pkl_file.replace("\\","/")
    print("pkl_file ",pkl_file)
    with open(pkl_file, "rb") as f:
        pkl_content = pickle.load(f, encoding='latin1')
        
    # matrix transform for primitive -> orthogonal reference frame
    # L_p_po: defines components of the basis of the primitive lattice in an orthogonal reference frame for the same lattice
    a = lattice_constant #[Ang]
    
    L_p_po = a*lattices[bravais_lattice] # primitive -> primitive orthogonal lattice transform
    L_po_p = np.linalg.inv(L_p_po) # primitive orthogonal -> primitive lattice transform

    # reciprocal lattice transforms
    L_rp_po = gb_csl.reciprocal_mat(L_p_po) # the same as np.lingalg.inv(L_p_po)
    L_po_rp = np.linalg.inv(L_rp_po)
    
    # GB normal general orthogonal coordinates -> reciprocal of primitive lattice coordinates
    inds_rp = L_po_rp.dot(inds_po)
    n_rp_integral_p = convert_array(int_man.int_finder(inds_rp)) # GB normal -> primitive (hkl)
    
    if verbose:
        print("\nGB normal ({}):".format(bravais_lattice))
        print("    lower crystal orthogonal indices: {}".format(inds_po))
        print("    lower crystal primitive miller indices: {}".format(n_rp_integral_p))
    
    # Define the Sigma matrix of interest
    Sigma = pkl_content[sigma]['N'][sigma_number]/pkl_content[sigma]['D'][sigma_number]
    # transform matrix p1->p2 in orthogonal reference frame
    T_p1top2_po = L_p_po.dot(Sigma.dot(L_po_p))
    
    if verbose:
        print("\nTransformation matrix (sigma = {}):".format(sigma))
        print("    Sigma = {}".format(Sigma))
        print("    T_p1->p2 = {}".format(T_p1top2_po))
        
    # finding the CSL and DSC lattices - expressed in the primitive lattice
    L_csl_p1, L_dsc_p1 = gb_csl.find_csl_dsc(L_p_po, Sigma)

    # GB in-plane vectors for both crystals - expressed in the primitive lattice
    index_type  = 'normal_go'
    T_reference = 'g1'

    L_2d_csl_p1, L_pl1_p1, L_pl2_p1 = plb.gb_2d_csl(inds_po, Sigma, L_p_po, index_type, T_reference)
        
    # transform from primitive lower crystal basis to orthogonal version
    L_pl1_po1 = L_p_po.dot(L_pl1_p1)
    L_pl2_po1 = np.around(L_p_po.dot(L_pl2_p1), decimals=14)
    L_2d_csl_po1 = L_p_po.dot(L_2d_csl_p1)
        
    # get integer forms
    int_L_pl1_po1 = np.array([int_man.int_finder(L_pl1_po1[:,0]),int_man.int_finder(L_pl1_po1[:,1])]).T
    int_L_pl2_po1 = np.array([int_man.int_finder(L_pl2_po1[:,0]),int_man.int_finder(L_pl2_po1[:,1])]).T
    int_L_2d_csl_po1 = np.array([int_man.int_finder(L_2d_csl_po1[:,0]),int_man.int_finder(L_2d_csl_po1[:,1])]).T

    # re-compute the lower crystal normal and the upper crystal normal
    xy1_normal_po1 = np.cross(int_L_pl1_po1[:,0],int_L_pl1_po1[:,1])
    xy1_normal_po1 = get_gcded_vector(int_man.int_finder(xy1_normal_po1))
    assert np.isclose(np.dot(xy1_normal_po1/np.linalg.norm(xy1_normal_po1), inds_po/np.linalg.norm(inds_po)), 1),\
        "Normal versions of re-computed GB plane normal ({}) given normal ({}) are not aligned: {}*{} != 1.".format(\
                                xy1_normal_po1/np.linalg.norm(xy1_normal_po1), inds_po/np.linalg.norm(inds_po),\
                                np.dot(xy1_normal_po1/np.linalg.norm(xy1_normal_po1), inds_po/np.linalg.norm(inds_po)))
    xy2_normal_po1 = get_gcded_vector(int_man.int_finder(T_p1top2_po.dot(xy1_normal_po1)))
    
    # plane vectors in orthogonal reference frame
    xy1_plane_po1 = find_rectangular_plane(int_L_2d_csl_po1, n=10)
    xy2_plane_po1 = T_p1top2_po.dot(xy1_plane_po1)

    xy2_plane_po1[:,0] = int_man.int_finder(xy2_plane_po1[:,0])
    xy2_plane_po1[:,1] = int_man.int_finder(xy2_plane_po1[:,1])
    
    xy2_plane_po1[:,0] = get_gcded_vector(xy2_plane_po1[:,0])
    xy2_plane_po1[:,1] = get_gcded_vector(xy2_plane_po1[:,1])
    
    # sanity check
    xdz = xy1_plane_po1[:,0].dot(xy1_normal_po1)
    ydz = xy1_plane_po1[:,1].dot(xy1_normal_po1)
    xdy = xy1_plane_po1[:,0].dot(xy1_plane_po1[:,1])
    if not np.allclose([xdz, ydz, xdy], [0,0,0]):
        print("Not all lower crystal vectors are approximately orthogonal: x*z = {}, y*z = {}, x*y = {}".format(xdz, ydz, xdy))
              
    xdz = xy2_plane_po1[:,0].dot(xy2_normal_po1)
    ydz = xy2_plane_po1[:,1].dot(xy2_normal_po1)
    xdy = xy2_plane_po1[:,0].dot(xy2_plane_po1[:,1])
    if not np.allclose([xdz, ydz, xdy], [0,0,0]):
        print("Not all lower crystal vectors are approximately orthogonal: x*z = {}, y*z = {}, x*y = {}".format(xdz, ydz, xdy))
    
    # done
    directions = {"lower":{"x":xy1_plane_po1[:,0],
                           "y":xy1_plane_po1[:,1],
                           "z":xy1_normal_po1,},
                  "upper":{"x":xy2_plane_po1[:,0],
                           "y":xy2_plane_po1[:,1],
                           "z":xy2_normal_po1,}}
        
    return directions

def get_crystals(directions, symbol="Cu", scale_up=True, verbose=False, lattice_constant=1.):
    """Create crystals using 'directions' obtained from get_bicrystal_directions.
    
    Parameters
    ----------
    directions : dict of dicts of int np.ndarrays of shape (3,)
        Each vector contains lattice directions in the orthogonal reference frame.
        These directions can directly be used with ase's FaceCenteredCubic function
        to create atom object instances.
    
    symbol : str, optional, default "Cu"
        Which element the atoms in the crystals shall have.
        
    scale_up : boolean, optional, default True
        Scales up the x and y dimension of the lower and upper crystal
        so that they match.
        
    Returns
    -------
    crystals : dict of ase atom instances.
    """
    crystals = {}
    for k in sorted(directions):
        if verbose:
            print("\n{}".format(k))
        _d = np.array([directions[k][_k] for _k in ["x", "y", "z"]], dtype=np.int64)
        
        atoms = FaceCenteredCubic(directions=_d, symbol=symbol,pbc=True,\
            latticeconstant=lattice_constant)
        if atoms.get_cell()[2,2]<0:
            print("dealing with it...")
            atoms = FaceCenteredCubic(directions=-_d, symbol=symbol,pbc=True,\
                latticeconstant=lattice_constant)
        crystals[k] = copy.deepcopy(atoms)
        
    if scale_up:
        upper_cell = crystals["upper"].get_cell()
        lower_cell = crystals["lower"].get_cell()
        x2 = upper_cell[0,0]
        x1 = lower_cell[0,0]
        
        if x2>=x1:
            fx1 = x2/x1
            fx2 = 1.
        elif x1>x2:
            fx1 = 1.
            fx2 = x1/x2
        
        assert np.allclose(np.array([fx1, fx2], dtype=float), np.array([fx1+.5,fx2+.5], dtype=int)), "x factors are not integers! fx1 = {} fx2 = {}".format(fx1,fx2)
                
        y2 = np.linalg.norm(upper_cell[1,:])
        y1 = np.linalg.norm(lower_cell[1,:])
        
        if y2>=y1:
            fy1 = y2/y1
            fy2 = 1
        elif y1>y2:
            fy1 = 1
            fy2 = y1/y2
            
        assert np.allclose(np.array([fy1, fy2], dtype=float), np.array([fy1+.5,fy2+.5], dtype=int)), "y factors are not integers! fy1 = {} fy2 = {}".format(fy1,fy2)
                
        z2 = upper_cell[2,2]
        z1 = lower_cell[2,2]
        if y2>=y1:
            fz1 = int(z2/z1+.5)
            fz2 = 1
        elif y1>y2:
            fz1 = 1
            fz2 = int(z1/z2+.5)
        
        size1 = tuple([int(fx1+.5), int(fy1+.5), fz1])
        size2 = tuple([int(fx2+.5), int(fy2+.5), fz2])
        if verbose:
            print("size1 = ",size1)
            print("size2 = ",size2)
        crystals["lower"] *= size1
        crystals["upper"] *= size2
            
    return crystals