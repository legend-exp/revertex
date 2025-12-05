from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pygeomhpges
from numpy.typing import ArrayLike, NDArray

from revertex import core, utils

log = logging.getLogger(__name__)


def sample_hpge_borehole(
    n_tot: int,
    *,
    seed: int | None = None,
    hpges: dict[str, pygeomhpges.HPGe] | pygeomhpges.HPGe,
    positions: dict[str, ArrayLike] | ArrayLike,
) -> NDArray:
    """Generate events on many HPGe boreholes weighting by the volume.

    Parameters
    ----------
    n_tot
        total number of events to generate
    seed
        random seed for the RNG.
    hpges
        List of :class:`pygeomhpges.HPGe` objects.
    positions
        List of the origin position of each HPGe.

    Returns
    -------
    Array of global coordinates.
    """
    rng = np.random.default_rng(seed=seed)

    out = np.full((n_tot, 3), np.nan)

    # loop over n_det maybe could be faster
    if isinstance(hpges, Mapping):
        weights = utils.get_borehole_weights(hpges)

        det_index = rng.choice(np.arange(len(hpges)), size=n_tot, p=weights)

        for idx, (name, hpge) in enumerate(hpges.items()):
            n = np.sum(det_index == idx)

            out[det_index == idx] = (
                _sample_hpge_borehole_impl(n, hpge, seed=seed) + positions[name]
            )
    else:
        out = _sample_hpge_borehole_impl(n_tot, hpges, seed=seed) + positions

    return out


def _sample_hpge_borehole_impl(
    size: int,
    hpge: pygeomhpges.HPGe,
    seed: int | None = None,
) -> NDArray:
    """Generate events on the surface of a single HPGe.

    Parameters
    ----------
    n
        number of vertexs to generate.
    hpge
        pygeomhpges object describing the detector geometry.
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
