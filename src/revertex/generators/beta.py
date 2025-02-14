from __future__ import annotations

import logging

import awkward as ak
import hist
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as rot

from revertex.core import sample_histogram

log = logging.getLogger(__name__)


def generate_beta_spectrum(
    energies: ArrayLike,
    phase_space: ArrayLike,
    size: int,
    *,
    seed: int | None = None,
    eunit: str = "keV",
) -> ak.Array:
    """Generate samples from a beta spectrum defined by a list of energies and phase space
    values.

    This function interprets the energies and phase_space as a histogram and samples
    energies from this. These are then converted into momenta. The energies should be
    the left edge of the bin of a histogram

    Parameters
    ----------
    energies
        the energy values
    spectrum
        the phase space values
    size
        number of events to generate
    seed
        random seed.
    eunit
        the unit for energy in the input file, default keV.

    Returns
    -------
    An awkward array with the sampled kinematics, in keV.
    """
    # convert energies to bin edges
    if eunit == "MeV":
        factor = 1 / 1000.0
    elif eunit == "keV":
        factor = 1
    elif eunit == "eV":
        factor = 1000
    else:
        msg = f"Only eunits keV, MeV or eV are supported not {eunit}"
        raise ValueError(msg)

    histo = hist.Hist(hist.axis.Variable(energies * factor))

    for b in range(histo.size - 2):
        histo[b] = phase_space[b]

    energy_samples = sample_histogram(histo, size, seed=seed)
    matrix = np.vstack(
        [[energy_samples, np.zeros_like(energy_samples), np.zeros_like(energy_samples)]]
    )

    mass = 511.0  # keV
    momentum = np.sqrt(energy_samples**2 + 2 * mass * energy_samples)

    rng = np.random.default_rng(seed)  # Set seed
    rand_rot = rot.random(random_state=rng)  # Random rotation

    matrix = np.vstack([[momentum, np.zeros_like(momentum), np.zeros_like(momentum)]])
    momenta = rand_rot.as_matrix() @ matrix

    return ak.Array(
        {
            "px": momenta[0, :],
            "py": momenta[1, :],
            "pz": momenta[2, :],
            "ekin": energy_samples,
            "particle": np.full_like(energy_samples, 11),
        }
    )
