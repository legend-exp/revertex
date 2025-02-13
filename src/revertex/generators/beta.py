from __future__ import annotations



def generate_beta_spectrum(energies:ArrayLike,phase_space:ArrayLike, *,seed:int | None = None,mode:str = "left")->ak.Array:
    """Generate samples from a beta spectrum defined by a list of energies and phase space
    values.
    
    This function interprets the energies and phase_space as a histogram and samples 
    energies from this. These are then converted into momenta.

    Parameters
    ----------
    energies
        the energy values
    spectrum
        the phase space values
    mode
        Whether the energies are interpreted as the "left","right" or "center" of the bins.
    seed
        random seed.

    Returns
    -------
    An awkward array with the sampled kinematics.
    """

    pass