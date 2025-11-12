from __future__ import annotations

import logging

import legendhpges
import numpy as np
from numpy.typing import ArrayLike, NDArray

from revertex import core, utils

log = logging.getLogger(__name__)


def generate_hpge_borehole_points(
    n_tot: int,
    seed: int | None = None,
    *,
    hpges: dict[str, legendhpges.HPGe],
    positions: dict[str, ArrayLike],
) -> NDArray:
    """Generate events on many HPGe boreholes weighting by the volume.

    Parameters
    ----------
    n_tot
        total number of events to generate
    seed
        random seed for the RNG.
    hpges
        List of :class:`legendhpges.HPGe` objects.
    positions
        List of the origin position of each HPGe.

    Returns
    -------
    Array of global coordinates.
    """

    weights = utils.get_borehole_weights(hpges)

    rng = np.random.default_rng(seed=seed)

    out = np.full((n_tot, 3), np.nan)

    det_index = rng.choice(np.arange(len(hpges)), size=n_tot, p=weights)

    # loop over n_det maybe could be faster
    for idx, (name, hpge) in enumerate(hpges.items()):
        n = np.sum(det_index == idx)

        out[det_index == idx] = (
            generate_hpge_borehole(n, hpge, seed=seed) + positions[name]
        )

    return out


def generate_hpge_borehole(
    size: int,
    hpge: legendhpges.HPGe,
    seed: int | None = None,
) -> NDArray:
    """Generate events on the surface of the HPGe.

    Parameters
    ----------
    n
        number of vertexs to generate.
    hpge
        legendhpges object describing the detector geometry.
    surface_type
        Which surface to generate events on either `nplus`, `pplus`, `passive` or None (generate on all surfaces).
    seed
        seed for random number generator.

    Returns
    -------
    Array with shape `(n,3)` describing the local `(x,y,z)` positions for every vertex
    """
    r, z = hpge.get_profile()

    height = max(z)
    radius = max(r)

    output = None

    # sampling efficiency is not necessarily high but hopefully this is not a big limitation
    seed_tmp = seed

    while output is None or (len(output) < size):
        # adjust seed
        seed_tmp = seed_tmp * 7 if seed_tmp is not None else seed

        # get some proposed points
        proposals = core.sample_cylinder(
            r_range=(0, radius),
            z_range=(0, height),
            size=size,
            seed=seed_tmp,
        )

        is_good = hpge.is_inside_borehole(proposals)

        sel = proposals[is_good]

        # extend
        output = np.vstack((output, sel)) if output is not None else sel

    # now cut to the right size
    return output[:size]
