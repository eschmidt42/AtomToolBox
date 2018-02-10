"""Markov Chain Monte Carlo diffusion example.

In this script diffusion of atoms between clusters of different 
sizes is first simulated using Markov Chain Monte Carlo. Then the
obtained distribution are fitted using the Heat / Diffusion equation
and finite differences.

Eric Schmidt
2017-11-28
"""

import matplotlib.pylab as plt
from scipy.linalg import toeplitz
import numpy as np
from scipy import integrate, interpolate, optimize

def odefun(u, t, UM):
    return np.dot(UM,u)

def diffusion_solver(cluster_sizes, atom_distributions, c=0, d=0.02, pde_form="centred explicit", 
                     Nt=1, Ncluster_pad=0, return_normalized=False, bc_left="Neumann", 
                     bc_right="Neumann", odeint_params={"rtol":1e-8},  do_optimization=False, 
                     opt_params={"method":"Nelder-Mead"}, measure=None, verbose=False, show=False):
    """Solves the diffusion / heat equation.
    
    Assumes insulating boundary conditions.
    
    Parameters
    ----------
    
    cluster_sizes : int np.ndarray of shape (J,)
        Cluster sizes.
        
    atom_distributions : float np.ndarray of shape (N,J)
        How many atoms in which cluster size bin.
        
    c : float or float np.ndarray of shape (J,), optional, default 0
        Convection parameter.
        
    d : float or float np.ndarray of shape (J,), optional, default 0.02
        Diffusion constant(s).
    
    pde_form : str, optional, default "centred explicit"
        Which finite difference form to use for the first difference.
        Options: "centred explicit", "upwind convection", "implicit diffusion"
            
    Nt : int, optional, default 1
        Number of PDE time steps between atom_distributions timesteps.
        If Nt is not equal to 1 interpolation is carried out.
    
    Ncluster_pad : int, optional, default 0
        Padding for cluster_sizes.
        
    return_normalized : boolean, optional, default False
        Wether to return normalized distributions of atoms or not.
        
    bc_left : str, optional, default "Neumann"
        The kind of boundary condition on the left. Options are: "Neumann" and "Dirichlet".
        
    bc_right : str, optional, default "Neumann"
        The kind of boundary condition on the right. Options are: "Neumann" and "Dirichlet".
        
    odeint_params : dict, optional, default {"rtol":1e-8}
        Parameters for scipy.integrate.odeint.
        
    do_optimization : boolean, optional, default False
        Wether to carry out optimization of diffusion equation parameters or not.
        Note that opt_params needs to be provided if True as well as 'measure'.
        
    opt_params : dict, optional, default {"method":"Nelder-Mead"}        
        scipy.optimize.minimize parameters.
        
    measure : callable or None, optional, default None
        Needs to be a callable for optimization. The pde_predictions will be 
        passed to 'measure' in full and 'measure' is required to return
        a fitness value for the optimizer.
        
    verbose : boolean, optional, default False
    
    show : boolean, optional, default False
        
    Returns
    -------
    x : int np.ndarray of shape (cluster_sizes.shape[0] + Ncluster_pad,)
        Cluster sizes.
        
    solutions : float np.ndarray of shape (cluster_sizes.shape[0] + Ncluster_pad, cluster_sizes.shape[0]*Nt)
        Normalized or not normalized number of atoms in any given cluster size.
        
    (c, d) : 
        Diffusion equation parameters. Only returned if do_optimization is True

    Notes
    -----
    Assumes that the distributions are given in equal time spacings. Choosing Nt = 1
    means that the time intervals of the pde solver are the same as for the distributions. 
    If Nt > 1 then the time intervals of the pde solver are shorter than those of the distributions.
    """
    implemented_pde_forms = set(["implicit diffusion", "centred explicit", "upwind convection"])
    assert pde_form in implemented_pde_forms, "pde_form '{}' is not one of the implemented forms: {}".format(pde_form, ", ".join(["'{}'".format(v) for v in implemented_pde_forms]))
    assert (Ncluster_pad is None) or isinstance(Ncluster_pad,int), "Ncluster_pad needs to be 'None' or an integer, given: {}".format(Ncluster_pad)
    if do_optimization:
        assert callable(measure), "'measure' needs to be callable for optimization to work."
        
    if not Ncluster_pad is None and Ncluster_pad < 0:
        Ncluster_pad = int(abs(Ncluster_pad))
        
    if (not isinstance(c,(float,int)) or (not isinstance(d,(int,float)))):
        assert isinstance(c,(list,tuple,np.ndarray)) and isinstance(c,(list,tuple,np.ndarray)) \
            and len(c)==len(d), "If c or d is a list/tuple/np.ndarray the other needs to be one one of the same length as well! c = {} d = {}".format(c,d)
        if not isinstance(c,np.ndarray):
            c = np.array(c, dtype=np.float64).reshape((-1,1))
        if not isinstance(d,np.ndarray):
            d = np.array(d, dtype=np.float64).reshape((-1,1))
    
    Natoms = atom_distributions[0,:].sum()
    u_ref = np.copy(atom_distributions) / float(Natoms)
    u_ref = np.array(u_ref, dtype=np.float64)
    
    # temporal mesh
    t = np.linspace(0, atom_distributions.shape[0],Nt*atom_distributions.shape[0], dtype=np.float64)
    dt = t[1]-t[0]
    N = t.shape[0]
    
    # spatial mesh
    x = np.copy(cluster_sizes) #np.linspace(0, Lx, Nx, dtype=np.float64) #np.copy(cluster_sizes)
    if isinstance(Ncluster_pad, int):
        x = np.hstack((x,np.arange(x.shape[0],x.shape[0]+Ncluster_pad)))
    dx = x[1]-x[0]
    J = x.shape[0]
    
    if isinstance(Ncluster_pad, int):
        u_ref = np.hstack((u_ref,np.zeros((u_ref.shape[0],Ncluster_pad))))
        
    u0 = np.copy(u_ref[0,:])
        
    # first difference matrix
    if pde_form == "centred explicit": ### Centred explicit
        D_minus = - np.diag(np.ones(J-1, dtype=np.float64), k=-1)
        D_plus = np.diag(np.ones(J-1, dtype=np.float64), k=1)
        D = D_plus + D_minus
        D[0,:] = 0.
        D[-1,:] = 0.

    elif pde_form in ["upwind convection", "implicit diffusion"]: ### Upwind convection or implicit diffusion
        D_minus = - np.diag(np.ones(J, dtype=np.float64), k=0)
        D_plus = np.diag(np.ones(J-1, dtype=np.float64), k=1)
        D = D_plus + D_minus

    else:
        raise ValueError("pde_form '{}' unknown!".format(pde_form))
    
    # second difference matrix
    T = toeplitz(c=[-2, 1] + [0 for v in range(J-2)])
    if bc_left == "Neumann":
        T[0,0] = -1
    elif bc_left == "Dirichlet":
        pass
    else:
        raise ValueError("bc_left ('{}') not understood!".format(bc_left))
    
    if bc_right == "Neumann":
        T[-1,-1] = -1
    elif bc_right == "Dirichlet":
        pass
    else:
        raise ValueError("bc_right ('{}') not understood!".format(bc_right))
    
    T = np.array(T, dtype=np.float64)
    if verbose:
        print("\nToeplitz:\n",T)
    
    def _solving_fun(_x):
        if _x.shape[0] == 2:
            c, d = _x
        else:
            c, d = _x[:int(_x.shape[0]*.5)], _x[int(_x.shape[0]*.5):]
        
        # factors
        f_D = c/(2.*dx)
        f_T = d/(dx**2)
        
        # update matrix
        if pde_form == "implicit diffusion":
            UM = (-np.linalg.inv(T*f_T - np.identity(J)/dt).dot(D*f_D + np.identity(J)/dt) - np.identity(J))/dt
        elif pde_form in ["centred explicit", "upwind convection"]: # centred explicit or upwind convection
            UM = D*f_D + T*f_T
        else:
            raise ValueError("pde_form '{}' unknown!".format(pde_form))

        UM = np.array(UM, dtype=np.float64)

        # running the solver
        return integrate.odeint(odefun, u0, t, args=(UM,), **odeint_params)
    
    def _opt_fun(_x):
        return measure(_solving_fun(_x))
    
    if do_optimization:
        if verbose:
            print("Performing optimization...")
        x0 = np.array([c,d]) if all([isinstance(c,float), isinstance(d,float)]) else np.hstack((c,d))
        s = x0.shape[0]
        res = optimize.minimize(_opt_fun, x0, **opt_params)
        c, d = res["x"][:int(s*.5)], res["x"][int(s*.5):]
    
    x0 = np.array([c,d]) if all([isinstance(c,float), isinstance(d,float)]) else np.hstack((c,d))
    solutions = _solving_fun(x0)
    
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for i,y in enumerate(solutions):
            if verbose:
                print("sum = {} sum(||u||^2) = {}".format(y.sum(), (np.absolute(y)**2).sum()))
            ax.plot(x, y,'-', label=str(i), color=plt.cm.jet(i/float(N)))
        ax.set_xlabel("Cluster sizes")
        ax.set_ylabel("Number of atoms")
        ax.set_title("PDE solutions")
        plt.show()
    
    if return_normalized:
        if do_optimization:
            return x, solutions, (c, d)
        return x, solutions
    if do_optimization:
        return x, solutions*Natoms, (c, d)
    return x, solutions*Natoms

def pde_measure_wrapper(atom_distributions):
    """Wraps functions which compute distance measures.
    
    Intended for the use withing the diffusion_solver function. The function
    returned by pde_measure_wrapper computes the distance between reference
    and modeled (pde) data.
    
    Parameters
    ----------
    atom_distributions : float np.ndarray of shape (N,J)
    """
    # normalized reference (diffusion_solver works internally with the same normalization)
    u_ref = np.copy(atom_distributions)/(atom_distributions[0,:].sum())
    
    # indices required since diffusion_solver may pad along the x-axis
    idx = np.arange(u_ref.shape[1])
    
    def measure(u):
        return np.linalg.norm(u[-1,idx]-u_ref[-1,:])**2
    
    return measure