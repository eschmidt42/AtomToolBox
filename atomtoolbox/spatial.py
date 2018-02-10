import numpy as np
import itertools
import scipy as sp
import warnings

from .misc import show_atoms

def get_ultracell(atoms=None, fpos=None, cell=None, species=None, r_cut=5., show=False,
                  verbose=False, max_iter=20, ftol=1e-6, return_fractional=False, check_bounds=True):
    """Generates an ultracell of a supercell.
    
    Either 'atoms' or 'fpos' + 'cell' + 'species' need to be given.
    
    Parameters
    ----------
    atoms : ase atom instance, optional, default None
    
    fpos : np.ndarray of float of shape (n_atoms,3), optional, default None
        fractional atom positions
        
    cell : np.ndarray of float of shape (3,3), optional, default None
        cell of the simulation box
        
    species : list of str, optional, default None
        list of chemical elements
    
    r_cut : float, optional
        cutoff radius (NOT fractional)
        
    show : boolean, optional, default False
        to show the supercell constraints and the ultracell using show_atoms
        
    verbose: boolean, optional, default False
    
    max_iter : int, optional, default 20
        number of iterations to expand the ultracell searching for
        completeness
        
    ftol : float, optional, default 1e-6
        tolerance for fractional position check
        
    return_fractional : boolean, optional, default False
        True: returns position in fractional space coordinates
        False: returns position in real space coordinates

    check_bounds : boolean, optional, default True
        True : tests whether all fractional positions are within 0 and 1
        False : does nothing
        
    Returns
    -------
    ultra_pos : np.ndarray of float of shape (n_ultra_atoms,3)
        atom positions in the ultracell in real or fractional 
        coordinates depending on 'return_fractional'
        
    ultra_species : np.ndarray of str of shape (n_ultra_atoms,)
        atom species of the ultracell atoms
    
    ultracell_idx : np.ndarray of int of shape (n_ultra_atoms,)
        indices to the original supercell atoms for each ultracell atom
        
    Notes
    -----
    This algorithm creates an ultracell adding at least one layer of supercells 
    around the original supercell. This may be very slow for structures of 
    millions of atoms.
    """
    do_ase = not atoms is None
    assert do_ase or not any([fpos is None, cell is None, species is None]), \
        "Either 'atoms' or 'fpos' + 'cell' + 'species' need to be given."
    
    if do_ase:
        fpos = atoms.get_scaled_positions(wrap=True)
        cell = atoms.get_cell()
        species = atoms.get_chemical_symbols()
        
    if check_bounds:
        assert (fpos - 1. < ftol).all() and (fpos+ftol > 0).all(), "'fpos' does not contain fractional coordinates!"
    r_cut = abs(r_cut)
    if r_cut < 1.:
        warnings.warn("The ultracell is generated using NON-FRACTIONAL cutoff distance r_cut. Just sayin' since you chose r_cut = {}".format(r_cut))
    if verbose:
        print("\nGenerating ultracell with r_cut = {:.2f}".format(r_cut))
    Vcell = np.absolute(np.linalg.det(cell))
    
    # find center and corners of the cell
    center = .5 * cell.sum(axis=0)
    
    fcorners = np.array([[0,0,0],
                         [1,0,0],
                         [0,1,0],
                         [0,0,1],
                         [1,1,0],
                         [1,0,1],
                         [0,1,1],
                         [1,1,1]])
    corners = fcorners.dot(cell)
    
    # plot to make sure
    if show:        
        _pos = np.vstack((corners,center))
        _species = ["corner" for v in range(len(corners))] + ["center"]
        show_atoms(pos=_pos,species=_species,markers={"corner":"bd","center":"ro"},
                   facecolor={"corner":"None","center":None},title="Supercell geometry")
        
        _pos = np.dot(fpos,cell)
        _pos = np.vstack((_pos,corners))
        _species = list(species)+["corner" for v in range(len(corners))]
        show_atoms(pos=_pos, species=_species, title="Supercell", cell=cell,)
    
    r_center2corner = np.linalg.norm(corners - center,axis=1).max()
    if verbose:
        print ('rcut = {} rcenter21corner = {}'.format(r_cut,r_center2corner))
    r_search = (r_cut + r_center2corner) * 1.5
    
    Vsphere = 4./3.*np.pi*r_search**3
    if verbose:
        print("approx. num of required cells = {}".format(Vsphere/float(Vcell)))
    
    start = list(itertools.product(*[[-1,0,1] for v in range(3)]))
    ijks_accepted = set(start) # contains all ijks ever accepted
    ijks_to_test = set(start) # contains all ijks which should be tested
    ijks_saturated = set() # contains all ijks which have max number of neighbors
    
    allowed_moves = [v for v in itertools.product(*[[-1,0,1] for v in range(3)]) if not (v[0]==0 and v[1]==0 and v[2]==0)]
    if verbose: 
        print("allowed moves {}".format(allowed_moves))
    
    i = 0
    while i<max_iter:
        if verbose:
            print("\n{}/{}".format(i+1,max_iter))
            print("cells: current = {} estimate for final = {}".format(len(ijks_accepted),Vsphere/float(Vcell)))
        
        # generate possible ijks by going through ijks_to_test comparing to ijks_saturated
        ijks_possible = [(i0+m0,i1+m1,i2+m2) for (i0,i1,i2) in ijks_to_test \
            for (m0,m1,m2) in allowed_moves if (i0+m0,i1+m1,i2+m2) not in ijks_saturated]
        if verbose: 
            print("possible new cells: {}".format(len(ijks_possible)))
        
        # check which ijks are within the specified search radius and add those to ijks_accpeted
        ijks_possible = [(i0,i1,i2) for (i0,i1,i2) in ijks_possible if np.linalg.norm(i0*cell[0,:]+i1*cell[1,:]+i2*cell[2,:])<=r_search]
        if verbose: print("cells after r filter {}".format(len(ijks_possible)))
        if len(ijks_possible) == 0:
            if verbose:
                print("Found all cells for r_cut {} => r_search = {:.2f} Ang, terminating after {} iterations".format(r_cut,r_search,i+1))
            break

        # add all ijks_possible points to ijks_accepted
        ijks_accepted.update(ijks_possible)
        if verbose:
            print("accepted new cells: {}".format(len(ijks_accepted)))
        
        # all ijks_to_test points now are saturated, hence add to ijks_saturated
        ijks_saturated.update(ijks_to_test)
        if verbose:
            print("stored cells so far: {}".format(len(ijks_saturated)))
        
        # remove all previously tested points
        ijks_to_test.clear()
        
        # add all points which were not already known to ijks_to_test
        ijks_to_test.update(ijks_possible)
        if verbose:
            print("cell to test next round: {}".format(len(ijks_to_test)))
        
        i += 1
    if i == max_iter:
        warnings.warn("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
        raise ValueError("max_iter reached in the ultracell generation! Generated {}/{} cells. Consider increasing max_iter.".format(len(ijks_accepted),Vsphere/float(Vcell)))
    
    # calculating the fractional atom positions
    fbasis = np.eye(3)
    idx_atoms = np.arange(len(fpos))
    
    for h,(i,j,k) in enumerate(ijks_accepted):
        new_fpos = fpos + i*fbasis[0,:] + j*fbasis[1,:] + k*fbasis[2,:]
        if h == 0:
            ultra_fpos = new_fpos
            ultra_species = np.array(species)
            ultracell_idx = idx_atoms
        else:
            ultra_fpos = np.vstack((ultra_fpos,new_fpos))
            ultra_species = np.hstack((ultra_species,species))
            ultracell_idx = np.hstack((ultracell_idx,idx_atoms))
                
    # converting atom positions into umm... non-fractional ...
    if not return_fractional:
        ultra_pos = np.dot(ultra_fpos,cell)
    
    if show:
        show_atoms(pos=ultra_pos,species=ultra_species,title="Ultracell")
    
    return ultra_pos, ultra_species, ultracell_idx

def get_neighbors(positions, upositions=None, r_cut=None,\
                  num_neigh=None, tol=1e-6):
    """Find neighors for each atom.
    
    Parameters
    ----------
    positions : np.ndarray of float of shape (n_atoms,3)
        Supercell atom positions (if using fractional positions consider the impact of the 
        cutoff radius not being equal along every axis).
        
    upositions : np.ndarray of float of shape (n_ultra_atoms,3), optional, default None
        Ultracell atom positions.
                
    r_cut : float, optional, default None
        Cutoff radius for the neighborhood search.
        
    num_neigh : int, optional, default None
        Number of neighboring atoms for the search. 
    
    Returns
    -------
    neigh_idx : list of np.ndarrays of ints
        Each np.ndarray corresponds to an atom (same order as 'positions') 
        containing the indices for each neighboring atom for the given 
        'positions' parameter. If uindices and upositions are given then neigh_idx
        contains indices for the ultracell.
    
    Notes
    -----
    If both r_cut and num_neigh are not None then num_neigh takes precedence.
    
    """  
    
    # setting up KDTrees for the neighborhood search
    if upositions is None:
        skd = sp.spatial.cKDTree(positions)
    else:
        skd = sp.spatial.cKDTree(upositions)
        
    # doing the search
    if num_neigh is not None:
        r, neigh_idx = skd.query(positions, k=num_neigh+1) #+1 because the current atom itself is also found as its own neighbour
        if isinstance(neigh_idx,np.ndarray): # convert to list
            neigh_idx = [v for v in neigh_idx]
    elif r_cut is not None:
        neigh_idx = skd.query_ball_point(positions, r_cut)
    else:
        raise ValueError("WTF!")
    
    # removing self reference
    for i,_idx in enumerate(neigh_idx):
        _idx = np.array(_idx)
                
        if upositions is None:
            r = np.linalg.norm(positions[_idx] - positions[i], axis=1)
        else:
            r = np.linalg.norm(upositions[_idx] - positions[i], axis=1)
        
        nonzero = np.where(r > tol)[0]
        _idx = _idx[nonzero]
        neigh_idx[i] = _idx
                
    return neigh_idx