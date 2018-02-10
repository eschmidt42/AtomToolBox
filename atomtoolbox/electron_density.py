import numpy as np

def rescale_and_shift_densities(bonds, verbose=False, offset=1e-6, rescale=True):
    """Searches for negative densities in bonds and shifts them accordingly.
    
    This function is used for the regression of EAM potentials, which
    can not have negative embedding densities.

    Parameters
    ----------
    bonds : list of supercell instances
    
    Returns
    -------
    bonds : list of supercell instances
        with modified electron densities
    min_dens : float or None
        smallest negative density value found or None if no densities were negative
    """
    
    dens = np.array([bond.t["density"] for bond in bonds])
    
    if rescale:
        scale = np.absolute(dens.max()-dens.min())
        dens /= scale
                
    idx_neg = np.where(dens<0.)[0]
    neg_dens = np.copy(dens[idx_neg])
    shift = offset
        
    if len(neg_dens)>0:
        min_dens = abs(neg_dens.min())
        shift += min_dens
    else:
        min_dens = None
                
    for i in range(len(bonds)):
        bonds[i].t["density"] = dens[i] + shift
    
    assert all([bond.t["density"]>offset for bond in bonds]), "Bummer, still incorrect densities present."
    return bonds, min_dens