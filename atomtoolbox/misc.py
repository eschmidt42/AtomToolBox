import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy import stats
import warnings
import os
import time
import collections
import numpy as np
from ase import Atoms
from scipy.cluster import hierarchy
import pandas as pd

def wrap_positions(pos, scaled=True, cell=None):
    _pos = np.copy(pos)
    d = pos.shape[1]
    
    if scaled:
        for i in range(d):
            _pos[:,i] = _pos[:,i] % 1.
    else:
        assert not cell is None, "If 'scaled' is False then 'cell' needs to be provided!"
        assert isinstance(cell, np.ndarray) and cell.shape==(d,d), "'cell' needs to be a (3,3) numpy array."
        for i in range(d):
            _pos[:,i] = _pos[:,i] % cell[i,i]
    return _pos
    

def show_atoms(atoms=None, pos=None, species=None, markers=None, facecolor=None,\
               title=None, cell=None):
    
    if atoms is None:
        assert (not pos is None) and (not species is None), "Either 'atoms' or 'pos', 'species' and 'cell' need to be given."
    else:
        pos = atoms.get_positions(wrap=True)
        species = atoms.get_chemical_symbols()
        cell = atoms.get_cell()
        
    species_set = set(species)
    species = np.array(species)
    
    fig = plt.figure()
    ax = Axes3D(fig)
    
    for s in species_set:
        idx = species == s
        x,y,z = pos[idx].T
        m = "o" if not isinstance(markers,dict) else markers[s]
        mc = "None" if not isinstance(facecolor,dict) else facecolor[s]
        ax.plot(x,y,z,m,label=s,markerfacecolor=mc)
    if not cell is None:
        xlim = (cell[:,0].min(), cell[:,0].max())
        ax.set_xlim(xlim)
        ylim = (cell[:,1].min(), cell[:,1].max())
        ax.set_ylim(ylim)
        zlim = (cell[:,2].min(), cell[:,2].max())
        ax.set_zlim(zlim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.legend(loc=0)
    plt.show()

def print_error_distributions_properties(true,pred,title=None,unit=None):
    dt = true-pred
    if not title is None:
        print("\n{}:".format(title))
    else:
        print("\n")
    if not unit is None:
        print("    mean = {:.2g} {unit},\n    2*std = {:.2g} {unit},\n    se = {:.2g} {unit},\n    min = {:.2g} {unit},\n    max = {:.2g} {unit}".format(
            np.mean(dt),np.std(dt),stats.sem(dt),np.min(dt),np.max(dt),unit=unit))
    else:
        print("    mean = {:.2g},\n    2*std = {:.2g},\n    se = {:.2g},\n    min = {:.2g},\n    max = {:.2g}".format(
            np.mean(dt),np.std(dt),stats.sem(dt),np.min(dt),np.max(dt)))

def show_rattled_atoms(rpositions, species=None, cells=None, markers=None, 
                       facecolor=None, title=None, markersizes=None, 
                       figsize=(5,5), alpha=0.2):
    """Show multiple rattled crystals.
    
    rpositions and such are obtained with rattle_crystal_multiple_rvs and 
    then flattened with flattening_rattled_crystal_data.
    """    
    fig = plt.figure(figsize=figsize)
    ax = Axes3D(fig)
    all_cells = np.vstack(np.concatenate(cells, axis=0))
    xlim = (all_cells[:,0].min(), all_cells[:,0].max())
    ylim = (all_cells[:,1].min(), all_cells[:,1].max())
    zlim = (all_cells[:,2].min(), all_cells[:,2
                                           ].max())
    N = len(rpositions)
    
    _species = np.hstack(species)
    species_set = set(list(np.unique(_species)))
    pos = np.vstack(rpositions)
            
    for s in species_set:
        idx = _species == s
        x,y,z = pos[idx,:].T
        m = "o" if not isinstance(markers,dict) else markers[s]
        mc = None if not isinstance(facecolor,dict) else facecolor[s]
        ms = None if not isinstance(markersizes,dict) else markersizes[s]
        
        ax.plot(x,y,z,m,label=s,markerfacecolor=mc,ms=ms,alpha=alpha)
        
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_title(title)
    plt.legend(loc=0)
    plt.show()
    
    fig = plt.figure()
    xlim = (np.amin(pos),np.amax(pos))
    bins = 50
    delta = (xlim[1]-xlim[0])/float(bins)
    ind = np.arange(bins)*delta + xlim[0]
    width = (ind[1]-ind[0])/3.
    
    ax = fig.add_subplot(111)
    labels = {0:"X", 1:"Y", 2:"Z"}
    for i in range(3):
        h,e = np.histogram(pos[:,i],bins=bins,range=xlim)
        ax.bar(ind+width*i,h,width,label=labels[i])
    ax.set_xlim(xlim)
    ax.set_title(title)
    ax.set_xlabel("Position [Ang]")
    ax.set_ylabel("Count")
    plt.legend(loc=0)
    plt.show()

def show_Efuns(regressors, regressor_names, x, plot_Phi,\
               xlabel="embedding density", ylabel="embedding energy",\
               title="Al", tol=1e5, ylim=None):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    for reg_name in regressor_names:
        reg = regressors[reg_name]
        try:
            _y, _yerr = reg.predict(plot_Phi,return_std=True)
            err_bool = True
        except TypeError:
            _y = reg.predict(plot_Phi)
            err_bool = False
        except:
            print("WTF!")
            raise

        if tol is None or (np.absolute(_y)<tol).all():

            ax.plot(x,_y,label=reg_name,lw=2)

            if err_bool:
                ax.fill_between(x,_y-2*_yerr,_y+2*_yerr,
                                alpha=.3,label=reg_name)
        else:
            warnings.warn("{} contained values which exceeded {:.2g}, not plotted..".format(reg_name,tol))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_ylim(ylim)
    plt.subplots_adjust(right=.7)
    plt.legend(loc=0,bbox_to_anchor=(1.,1.))
    plt.show()

def show_predict_vs_true(t_train, Phi_train, regressors, regressor_names, t_test=None, Phi_test=None,\
                         xlabel="observation", ylabel="Force [eV/Ang]", tol=None, title=None):
    test_and_train = (not t_test is None) and (not t_train is None)
    fig = plt.figure()
    
    if test_and_train:
        ax = fig.add_subplot(211)
    else:
        ax = fig.add_subplot(111)

    ax.plot(t_train,'ko',label="rattling")
    for reg_name in regressor_names:
        reg = regressors[reg_name]
        y = reg.predict(Phi_train)
        if tol is None or (np.absolute(y)<tol).all():
            ax.plot(y,'.',label=reg_name,markerfacecolor="None",alpha=.5)
        else:
            warnings.warn("{} contained values which exceeded {:.2g}, not plotted..".format(reg_name,tol))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    if test_and_train:
        ax.set_title("Train vs Reference")
    else:
        ax.set_title(title)
    
    if test_and_train:
        ax1 = fig.add_subplot(212)
        ax1.plot(t_test,'ko',label="rattling")
        for reg, reg_name in zip(regressors,regressor_names):
            y = reg.predict(Phi_test)
            if tol is None or (np.absolute(y)<tol).all():
                ax1.plot(y,'.',label=reg_name,markerfacecolor="None",alpha=.5)
            else:
                warnings.warn("{} contained values which exceeded {:.2g}, not plotted..".format(reg_name,tol))
        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(ylabel)
        ax1.set_title("Test vs Reference")

    plt.tight_layout()
    
    if test_and_train:
        plt.subplots_adjust(right=.65)
        plt.legend(loc=0, bbox_to_anchor=(1.,1.5))
    else:
        plt.legend(loc=0)

    plt.show()

def show_deviation_distributions(t_train, t_test, Phi_train, Phi_test, regressors, regressor_names, _range=(-.01,.01),\
                                 bins=50, xlabel="Force [eV/Ang]", ylabel="Normed frequency",
                                 normed=True):
    
    delta = (_range[1]-_range[0])/float(bins)
    width = delta * .5
    ind = np.arange(bins)*delta + _range[0]
    
    axs = []
    ylim = [0,None]
    for reg_name in regressor_names:
        reg = regressors[reg_name]
        fig = plt.figure(figsize=(7,3))

        ax = fig.add_subplot(111)

        y = reg.predict(Phi_train)
        h, e = np.histogram(y-t_train, normed=normed, range=_range, bins=bins)
        ax.bar(ind, h, width,label="train", alpha=.5)

        y = reg.predict(Phi_test)
        h, e = np.histogram(y-t_test, normed=normed, range=_range, bins=bins)
        ax.bar(ind+width, h, width,label="test", alpha=.5)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        axs.append(ax)

        plt.tight_layout()
        plt.suptitle(reg_name)
        plt.legend(loc=0)

        _ylim = ax.get_ylim()
        if ylim[1] is None or _ylim[1]>ylim[1]:
            ylim[1] = _ylim[1]

    for ax in axs:
        ax.set_ylim(ylim)

    plt.show()

def process_time(chunk, markers):
    return np.int64(chunk[1])

def process_Natoms(chunk, markers):
    return np.int64(chunk[1])

def process_box(chunk, markers):
    box = chunk[1:]
    stuff = []
    for b in box:
        stuff.append([float(_b) for _b in b.split(" ")])
    return np.array(stuff,dtype=np.float64)

def process_atoms(chunk, markers, default_type=str):
    converters = {"x":np.float64, "y":np.float64, "z":np.float64, "id":np.int64, "element":str, 
                  "type":np.int64, "vx":np.float64, "vy":np.float64, "vz":np.float64}
    
    N = len(chunk)-1
    keys = list(filter(None,chunk[0].lstrip(markers["atoms"]).split(" ")))
    atoms = {k: [None for v in range(N)] for k in keys}
    
    for i,line in enumerate(chunk[1:]):
        line = line.split()
        for j,k in enumerate(keys):
            atoms[k][i] = line[j]
            
    for k in atoms:
        if k in converters:
            atoms[k] = np.array(atoms[k],dtype=converters[k])
        else:
            atoms[k] = np.array(atoms[k],dtype=default_type)
    return atoms

def interpet_LAMMPS_lines(lines,markers):    
    processing_dict = {"time":process_time,
                       "N":process_Natoms,
                       "box":process_box,
                       "atoms":process_atoms}
    
    N = len(lines)
    data = {"time":None,
            "N":N,
            "box":None,
            "atoms":dict()}
    
    idx_dict = {k: [i for i,v in enumerate(lines) if v[:len(markers[k])]==markers[k]][0]\
                for k in markers}
    
    idx_tuples = [[k,idx_dict[k]] for k in sorted(idx_dict, key=lambda v: idx_dict[v])]
    idx_tuples = [[v[0],v[1],idx_tuples[i+1][1]] if i<len(idx_tuples)-1 else [v[0],v[1],-1] for i,v in enumerate(idx_tuples)]
    
    for m, start, end in idx_tuples:
        if end == -1:
            chunk = lines[start:]
        else:
            chunk = lines[start:end]
        data[m] = processing_dict[m](chunk, markers)
    return data

def parse_LAMMPS_custom_dump(path, verbose=False, return_1by1=True):
    """Parses a LAMMPS custom dump file.
    
    Parameters
    ----------
    path : str
        Path to the dump file.

    verbose : boolean, optional, default False
    
    return_1by1 : boolean, optional, default True
        If True this function acts as a generator function.
        If False a list of all dicts is returned.
    
    Returns
    -------
    output : generator or list of dicts
        if return_1by1: Generator which produces a data dict for each timestep:
            Example: {"x":np.array([...]), "element":np.array([...]), ...}
        else:
            List of data dicts, one dict for each timestep.
    """
    if verbose:
        print("Parsing {} ...".format(path))
        
    markers = {"time":"ITEM: TIMESTEP",
               "atoms":"ITEM: ATOMS",
               "N":"ITEM: NUMBER OF ATOMS",
               "box":"ITEM: BOX BOUNDS"}
    
    # find the lines limiting individual frames
    idx_step = []
    t0 = time.time()
    with open(path,"r") as f:
        for i,line in enumerate(f):
            if markers["time"] == line[:len(markers["time"])]:
                idx_step.append(i)
    if verbose:
        print("finding frames took {:.2f} s".format(time.time()-t0))
    
    Nlines = idx_step[1]-idx_step[0] # number of lines for each time step (assuming conservation of the number of atoms)
    
    if not return_1by1:
        all_data = []
    with open(path,"r") as f:
        for i,ix in enumerate(idx_step):
            lines = [next(f).strip() for c in range(Nlines)]
            data = interpet_LAMMPS_lines(lines,markers)
            if return_1by1:
                yield data
            else:
                all_data.append(data)
    if not return_1by1:
        return all_data

def lammpstrj_to_design_matrices(path, featurers, featurer_params,
                                 r_cut=6., num_neigh=18, 
                                 return_lammpstrj_data=False):
    """Turns a *.lammpstrj file into a list of design matrices.
    
    Parameters
    ----------
    path : str
        Path to *.lammpstrj file.
        
    featurers : list of atomtoolbox.features classes
        Example of a class to pass is: BondOrderParameterFeatures
        
    featurer_params : list of dicts
        Contains the parameters for each class in 'featurers' in the
        same order.
        
    r_cut : float, optional, default 6.
        Cutoff distance for neighbourhood search and ultracell generation.
        In the case num_neigh is not None num_neigh will take precedence
        over r_cut in the neighbourhood search but r_cut will still be used
        to set up ultracells and thus should approximately equal the search
        radius as implicitly defined by num_neigh.
        
    num_neigh : int, optional, default 18
        Number of nearest neighbours to select. Takes precedence over r_cut
        if num_neigh != None.
        
    return_lammpstrj_data : boolean, optional, default False
        Wether to only return a list of design matrices or also
        return lists of parsed LAMMPS data.
        
    Returns
    -------
    all_Phis : list of float np.ndarrays of shapes (Natoms, Nfeatures)
    
    all_pos, all_spec, all_box : lists of np.ndarrays
        Information parsed from the *.lammpstrj file. Returned in that
        order if return_lammpstrj is True.
    """
    from atomtoolbox.features import get_crystal_design_matrix
    # parse lammpstrj
    path_lammps = r"%s" % path

    if os.name != "posix":
        path_lammps = path_lammps.replace("\\","/")

    parser = parse_LAMMPS_custom_dump(path_lammps, verbose=True)
    
    # compute Phis
    all_Phis = []
    if return_lammpstrj_data:
        all_pos, all_spec, all_box = [], [], []
    for i,data in enumerate(parser): # looping rattle rv & crystal type

        _pos = np.array([data["atoms"]["x"], data["atoms"]["y"], data["atoms"]["z"]]).T
        _pos = wrap_lammps_pos(_pos,data["box"])
        _cell = box2cell(data["box"])
        _spec = data["atoms"]["element"]
        print("Species stats: ",collections.Counter(_spec).most_common())
        _fpos = np.dot(_pos,np.linalg.inv(_cell))
        
        if return_lammpstrj_data:
            all_pos.append(_pos)
            all_spec.append(_spec)
            all_box.append(data["box"])

        
        _Phi, mapper = get_crystal_design_matrix(positions=_pos, species=_spec, cell=_cell, r_cut=r_cut, 
                                                     num_neigh=num_neigh, features_class=featurers, 
                                                     params_features=featurer_params, return_force=False, 
                                                     return_mapper=True)
        assert np.isfinite(_Phi).all(), "_Phi contains non-finite values."
        all_Phis.append(_Phi)
        
    if return_lammpstrj_data:
        return all_Phis, all_pos, all_spec, all_box
    else:
        return all_Phis

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

def parse_LAMMPS_custom_dump(path, verbose=False):
    """Parses LAMMPS custom dump files.
    
    The LAMMPS files are supposed to contain trajectories.
    
    Returns
    -------
    
    Generator which produces a data dict:
        Example: {"x":np.array([...]), "element":np.array([...]), ...}
    """
    if verbose:
        print("Parsing {} ...".format(path))
        
    markers = {"time":"ITEM: TIMESTEP",
               "atoms":"ITEM: ATOMS",
               "N":"ITEM: NUMBER OF ATOMS",
               "box":"ITEM: BOX BOUNDS"}
    
    # find the lines limiting individual frames
    idx_step = []
    t0 = time.time()
    with open(path,"r") as f:
        for i,line in enumerate(f):
            if markers["time"] == line[:len(markers["time"])]:
                idx_step.append(i)
    if verbose:
        print("finding frames took {:.2f} s".format(time.time()-t0))
    
    Nlines = idx_step[1]-idx_step[0] # number of lines for each time step (assuming conservation of the number of atoms)
    
    with open(path,"r") as f:
        for i,ix in enumerate(idx_step):
            lines = [next(f).strip() for c in range(Nlines)]
            data = interpet_LAMMPS_lines(lines,markers)
            yield data

def cell2box(cell):
    d = cell.shape[0]
    box = np.zeros((d,2))
    for i,c in enumerate(cell):
        box[i,1] = c[i]
    return box

def box2cell(box):
    d = len(box)
    cell = np.zeros((d,d))
    for i,b in enumerate(box):
        cell[i,i] = b[1]-b[0]
    return cell

def wrap_lammps_pos(pos,box):
    for i,b in enumerate(box):
        pos[:,i] = (pos[:,i]+b[0]) % (b[1]-b[0])
    return pos

def create_LAMMPS_traj(path, rpositions, Natoms, species, cells, verbose=False):
    """Creates a LAMMPS lammpstrj file.
    
    Parameters
    ----------
    path : str
        lammpstrj file path.
    
    rpositions : list of np.ndarray of floats of shape (Na,3)
        Atom positions.
    
    Natoms : list of int
        Number of atoms.
        
    species : list of lists of str
        Atom species.
        
    cells : list of np.ndarrays of shape (3,3)
        List of supercell dimensions.
        
    verbose : boolean, optional, default False
    """
    from atomtoolbox import flattening_rattled_crystal_data
    _rpositions, _Natoms, _cells, _species = flattening_rattled_crystal_data(rpositions, Natoms, cells, species)
    
    Nt = len(_rpositions)
    elements = sorted(list(set(_species[0])))
    
    lammpstrj_header = ['ITEM: TIMESTEP','ITEM: NUMBER OF ATOMS',
                        'ITEM: BOX BOUNDS pp pp pp','ITEM: ATOMS id element type x y z ']
    
    type_map = {el:v+1 for v,el in enumerate(elements)}
    
    lines = []
    for i in range(Nt):
        pos = _rpositions[i]
        Na = _Natoms[i]
        cell = _cells[i]
        spec = _species[i]
        
        box = cell2box(cell)
        box_str = [" ".join([str(v) for v in b]) for b in box]
        
        atom_lines = ["{} {} {} {:.4f} {:.4f} {:.4f}".format(j, spec[j], type_map[spec[j]], pos[j][0], pos[j][1], pos[j][2])\
                      for j in range(Na)]
        
        lines.extend([lammpstrj_header[0],str(i)])
        lines.extend([lammpstrj_header[1],str(Na)])
        lines.extend([lammpstrj_header[2]]+box_str)
        lines.extend([lammpstrj_header[3]]+atom_lines)
        
    if verbose:
        print("Writing to {}".format(path))
    with open(path,"w") as f:
        for line in lines:
            f.write("{}\n".format(line))

def lammps2ase(data=None, pos=None, spec=None, box=None):
    """Creates an ase Atoms instance from parsed LAMMPS data.
    
    Parameters
    ----------
    data : dict
        Generated by parse_LAMMPS_custom_dump.
    
    pos : float np.ndarrays of shape (Natoms,3)
        Atom positions.
        
    spec : str np.ndarrays of shape (Natoms,)
        Atom elements.
        
    box : float np.ndarrays of shape (3,2)
        Simulation box.
        
    Returns
    -------
    atoms : ase.Atoms instance
    """
    if not data is None:
        pos = np.array([data["atoms"]["x"], data["atoms"]["y"], data["atoms"]["z"]]).T
        pos = wrap_lammps_pos(_pos,data["box"])
        cell = box2cell(data["box"])
        spec = data["atoms"]["element"]
        
    elif not all([pos is None, spec is None, box is None]):
        cell = box2cell(box)
        atoms = Atoms(spec,positions=pos, cell=cell, pbc=True)
    
    else:
        raise ValueError("Either 'data' or 'pos', 'spec' and 'box' need to be provided!")
    
    return atoms

def periodic_distances(X,box_lengths):
    """Cluster distance.
    
    Computes periodic atom-atom distances of atoms wrapped into the
    simulation box.
    
    Parameters
    ----------
    X : float np.ndarray of shape (N,d)
        Atom positions. N is the number of atoms
        and d the dimension.
    
    box_lengths : float np.ndarray of shape (d,)
        Contains supercell vector sizes, one for each
        dimenion.
    
    Returns
    -------
    
    Y : float np.narray of shape (N,)
        Periodic atom-atom distances.
    
    """
    b = box_lengths
    
    i,j = np.triu_indices(X.shape[0],k=1)
    dz = X[i,:] - X[j,:]
    dz = np.where(dz<-.5*b, dz+b, np.where(dz>.5*b, dz-b, dz))
        
    return np.linalg.norm(dz,axis=1)
    
def fclusterdata(X, t, box, criterion='distance', metric='euclidean', 
                 method='single', compute_inconsistency=True, depth=2):
    """
    This function is a costumized version of scipy.cluster.hierarchy.fclusterdata.
    The change made is towards a periodic boundary conditon calculation of 
    distances using the cluster_distance function. 
    
    Parameters
    ----------
    X : float np.ndarray of shape (N,d)
        Position of N atoms in d dimensional space.
    
    t : float
        Clustering distance.
    
    box : float of np.ndarray of shape (d,2)
        Simulation box specifications (assuming orthogonal vectors).
    
    criterion : str, optional, default "distance"
        scipy.cluster.hierarchy.fcluster 'criterion' parameter.
    
    metric : str, optional, default "euclidean"
        scipy.cluster.hierarchy.linkage 'metric' parameter.
        
    method : str, optional, default "single"
        scipy.cluster.hierarchy.linkage 'method' parameter.
    
    compute_inconsistency : boolean, optional, default True
    
    depth : int, optional, default 2
        Parameter for the inconsistency calculation.
            
    Returns
    -------
    fcluster : int np.ndarray of shape (N,)
        Integers indicate to which cluster each original atom with the same
        positional index belongs.
    
    """
    X = np.asarray(X, order='c', dtype=float)
    assert len(X.shape) == 2, "X has to be a rectangular numpy array!"
    
    X = wrap_lammps_pos(X,box)
    box_lengths = box[:,1] - box[:,0]
    
    Y = periodic_distances(X, box_lengths)
    Z = hierarchy.linkage(Y, method=method, metric=metric)
    
    if compute_inconsistency:
        R = hierarchy.inconsistent(Z, d=depth)
    else:
        R = None
    
    return hierarchy.fcluster(Z, criterion=criterion, depth=depth, R=R, t=t)

def get_GB_positions_from_histograms(all_pos, all_box, gb_class, bin_size=1., hist_threshold=1e-4,
                                     peak_threshold=1., pbc=(1,1,1), axis=2, show=False):
    """Computes atom occurrence histograms for all timesteps and from that GB positions.
    
    Parameters
    ----------
    all_pos : list of float np.ndarrays of shape (N,3)
    
    all_box : list of float np.ndarrays of shape (3,2)
    
    bin_size : float, optional, default 1.
    
    hist_threshold : float, optional, default .01
        In atoms per Ang^3.
    
    peak_threshold : float, optional, default 1.
    
    pbc : tuple of int of len(3), optional, default (1,1,1)
    
    axis : int, optional, default 2
    
    show : boolean, optional, default False
    
    Returns
    -------
    gb_positions : list of float np.ndarrays of shape (N,)
    
    Note: assumes orthogonal simulation box.
    """
    
    # loop timesteps and get histograms
    not_axis = np.setdiff1d([0,1,2],[axis])
    Nt = len(all_pos)
    
    all_h, all_e = [None for v in range(Nt)], [None for v in range(Nt)]
    for t in range(Nt):
        
        pos = all_pos[t][:,axis]
        lb, ub = all_box[t][axis,:]
        bins = int((ub-lb)/bin_size+.5)
        d_bin = (ub-lb)/float(bins)
        
        d_bin_off0 = all_box[t][not_axis[0],1] - all_box[t][not_axis[0],0]
        d_bin_off1 = all_box[t][not_axis[1],1] - all_box[t][not_axis[1],0]
        
        V_bin = d_bin*d_bin_off0*d_bin_off1
                
        h, e = np.histogram(pos, bins=bins,range=(lb,ub),normed=False)
        e = .5*(e[1:]+e[:-1])
        h = h.astype(np.float64)
        h /= V_bin
        
        all_h[t] = h
        all_e[t] = e
                
    # loop histograms and manipulate them
    all_gb_positions = [None for v in range(Nt)]
    
    for t in range(Nt):
        idx = np.where(all_h[t]>hist_threshold)[0]
        e = all_e[t][idx].reshape((-1,1))
        h = all_h[t][idx]
        box = all_box[t][axis,:].reshape((1,-1))
        
        clustering_distance = peak_threshold
        
        fdata = fclusterdata(e, clustering_distance, box, criterion='distance', 
                             metric='euclidean', method='single')
                
        fdata = np.array(fdata,dtype=int)
        
        cluster_labels = np.unique(fdata)
        b = all_box[t][axis,1] - all_box[t][axis,0]
        Nc = len(cluster_labels)
        peaks = np.zeros(Nc)
        
        for i,cl in enumerate(cluster_labels):
        
            _idx = np.where(fdata==cl)[0]
            _e = e[_idx].ravel()
            _h = h[_idx].ravel()
            _h /= _h.sum()
            
            Delta_e = _e[-1]-_e[0]
            gb_pbc_split = Delta_e > .5*b
            if gb_pbc_split: # GB is split on periodic boundary
                _e += .5*b
                _e = _e % b
            
            avg = np.average(_e,weights=_h)
            
            if gb_pbc_split:
                avg -= .5*b
                avg = avg % b
                
            peaks[i] = avg
            
        all_gb_positions[t] = peaks
        
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for t in range(Nt):
            # atom frequencies
            ax.plot(all_e[t],all_h[t],'-',color=plt.cm.seismic((t+1)/float(Nt)),alpha=.5)
            
            idx = np.where(all_h[t]>hist_threshold)[0]
            e = all_e[t][idx].reshape((-1,1))
            h = all_h[t][idx]
            ax.plot(e,h,'o',markerfacecolor="None",color=plt.cm.seismic((t+1)/float(Nt)),alpha=.5)
            
            # gb pos
            x = all_gb_positions[t]
            y = np.zeros(len(all_gb_positions[t]))
            ax.plot(x,y,'x',color=plt.cm.seismic((t+1)/float(Nt)), ms=10)
            
        xlim = ax.get_xlim()
        # bin selection threshold
        ax.plot(xlim,(hist_threshold,hist_threshold),'-k',label="hist_threshold")
        ax.set_xlabel("Axis {} [Ang]".format(axis))
        ax.set_ylabel("Atoms found [Atoms/Ang^3]")
        plt.legend(loc=0)
        plt.show()
        
    return all_gb_positions

def get_GB_position_DataFrame(all_gb_positions):
    """Creates a pandas DataFrame for trackpy.
    """
    print(all_gb_positions)
    x,y,frame = [], [], []
    
    Nt = len(all_gb_positions)
    for t in range(Nt):
        _x = all_gb_positions[t]
        x.extend(list(_x))
        y.extend([0 for v in range(_x.shape[0])])
        frame.extend([t for v in range(_x.shape[0])])
    x = np.array(x,dtype=np.float)
    y = np.array(y,dtype=np.float)
    frame = np.array(frame,dtype=np.int)
    df = pd.DataFrame(data=np.array([x,y,frame], dtype=object).T, columns=["x","y","frame"])
    return df

def square_cluster_statistics(all_cluster_statistics, min_size=1, Natoms=None):
    """Creates a rectangular cluster distribution array.
    
    Parameters
    ----------
    all_cluster_statistics : list of lists
        Each sublist contains tuples of cluster label and number of 
        atoms in that cluster.
        
    min_size : int, optional, default 1
        Minimum cluster size to output.
        
    Natoms : int, optional, default None
        if min_size == 0 and Natoms isn't None then Natoms
        is used to assign the number of atoms not participating
        in the clustering to 0-size clusters.
        
    Returns
    -------
    cluster_size : np.ndarray of shape (J,)
    
    atom_distributions : np.ndarray of shape (N,J)
    
    cluster_distributions : np.ndarray of shape (N,J)
    """
    assert isinstance(min_size, int), "min_size needs to be an integer!"
    
    size = [np.array(v)[:,1] for v in all_cluster_statistics]
    
    Nsteps = len(all_cluster_statistics)
    biggest_cluster = max([s.max() for s in size])
    
    cluster_size = np.arange(min_size, biggest_cluster+1)
    Nbins = cluster_size.shape[0]
    
    cluster_distributions = np.zeros((Nsteps,Nbins))
    atom_distributions = np.zeros((Nsteps,Nbins))
    
    for i,c in enumerate(size):
        _c = collections.Counter(c).most_common()
        
        for (f,s) in _c:
            if min_size == 1:
                cluster_distributions[i,f-1] = s
                atom_distributions[i,f-1] = s*f
            elif min_size == 0:
                cluster_distributions[i,f] = s
                atom_distributions[i,f] = s*f
            else:
                raise ValueError("min_size has to be either 0 or 1! Given min_size = {}".format(min_size))
        
        if min_size == 0:
            atom_distributions[i,0] = Natoms - atom_distributions[i,1:].sum()
            
    return cluster_size, cluster_distributions, atom_distributions

def plot_single_feature_distribution(Phi_df,feature="feature0"):

    import seaborn as sns
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(Phi_df, row="cloudtype", hue="cloudtype", aspect=15, size=.5, palette=pal)
    
    bw = .01
    g.map(sns.kdeplot, feature, shade=True, alpha=1, lw=1.5, bw=bw, clip_on=False)
    g.map(sns.kdeplot, feature, color="w", lw=2, bw=bw, clip_on=False)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)

    def fun_label(x, color, label):
        ax = plt.gca()
        ax.text(0, .4, label, fontweight="bold", color=color, 
                ha="left", va="center", transform=ax.transAxes)
    g.map(fun_label, feature)

    g.fig.subplots_adjust(hspace=-.1)
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)

    plt.show()

def get_stratified(y, Ndraws, flat=True):
    """Returns indices for stratified samples.
    
    Parameters
    ----------
    y : np.ndarray of str or int and shape (Nsamples,)
        Labels to sample.
    Ndraws : int
        Number of samples to draw for each label
    flat : boolean, optional, default True
        Whether or not to flatten the returned indices.
        
    Returns
    -------
    idx_stratified : np.ndarray of int and shape (len(np.unique(y)),Ndraws) or (len(np.unique(y)),)
    """
    
    labels = np.unique(y)
    labels.sort()
    idx_stratified = np.array([np.random.choice(np.where(y==label)[0], size=Ndraws, replace=False)\
                               for label in labels])
    if flat:
        return idx_stratified.ravel()
    return idx_stratified

def precision_barplot_classifier_cloudtype(all_trained_classifiers,
                                           Phi_by_cloud, int2str_map, Ndraws=None, all_Ndraws=None):
    import seaborn as sns
    
    if Ndraws is None:
        idx_draws = -1
    elif isinstance(Ndraws,int) and not (all_Ndraws is None):
        idx_draws = all_Ndraws.index(Ndraws)
    else:
        raise NotImplementedError
    
    data = {}
    _Phis = {_l: np.vstack(Phi_by_cloud[_l]) for _l in Phi_by_cloud}
    for classifier in sorted(all_trained_classifiers):
        data[classifier] = {}
        for label in sorted(Phi_by_cloud):
        
            _Phi = _Phis[label]
            pred_new = all_trained_classifiers[classifier][idx_draws].predict(_Phi)
            counts = collections.Counter(pred_new)
            
            try:
                data[classifier][label] = counts[label]/_Phi.shape[0]*100.
            except KeyError:
                data[classifier][label] = 0.
    
    sns.set(style="white", context="talk")
    f, axs = plt.subplots(len(data), 1, figsize=(10,2*len(data)), sharex=True)
    if not isinstance(axs, (list, np.ndarray)):
        axs = [axs]
    
    x = sorted(Phi_by_cloud.keys())
    x_str = [int2str_map[v] for v in x]
    for i,classifier in enumerate(sorted(data)):
        sns.barplot(x_str, [data[classifier][_k] for _k in x], ax=axs[i])
        axs[i].set_ylabel(classifier)
    sns.despine(bottom=True)
    plt.tight_layout()
    plt.show()