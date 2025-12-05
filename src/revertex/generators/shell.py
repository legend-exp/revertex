from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pygeomhpges
from numpy.typing import ArrayLike, NDArray

from revertex import core, utils

log = logging.getLogger(__name__)


def sample_hpge_shell(
    n_tot: int,
    *,
    seed: int | None = None,
    hpges: dict[str, pygeomhpges.HPGe] | pygeomhpges.HPGe,
    positions: dict[str, ArrayLike] | ArrayLike,
    distance: float,
    surface_type: str | None = None,
) -> NDArray:
    """Generate events on many HPGe's shells weighting by the surface area.

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
    surface_type
        Which surface to generate events on either `nplus`, `pplus`, `passive` or None (generate on all surfaces).

    Returns
    -------
    Array of global coordinates.
    """

    rng = np.random.default_rng(seed=seed)

    out = np.full((n_tot, 3), np.nan)

    if isinstance(hpges, Mapping):
        p_det = utils.get_surface_weights(hpges, surface_type=surface_type)
        det_index = rng.choice(np.arange(len(hpges)), size=n_tot, p=p_det)

        for idx, (name, hpge) in enumerate(hpges.items()):
            n = np.sum(det_index == idx)

            out[det_index == idx] = (
                _sample_hpge_shell_impl(
                    n, hpge, distance=distance, surface_type=surface_type, seed=seed
                )
                + positions[name]
            )

    else:
        out = (
            _sample_hpge_shell_impl(
                n_tot, hpges, distance=distance, surface_type=surface_type, seed=seed
            )
            + positions
        )
    return out


def _sample_hpge_shell_impl(
    size: int,
    hpge: pygeomhpges.HPGe,
    surface_type: str | None,
    distance: float,
    seed: int | None = None,
) -> NDArray:
    """Generate events on a shell around a single HPGe. This uses rejection sampling.

    Parameters
    ----------
    size
        number of vertexs to generate.
    hpge
        pygeomhpges object describing the detector geometry.
    surface_type
        Which surface to generate events on either `nplus`, `pplus`, `passive` or None (generate on all surfaces).
    distance
        Size of the hpge shell to generate in.
    seed
        seed for random number generator.

    Returns
    -------
    Array with shape `(n,3)` describing the local `(x,y,z)` positions for every vertex
    """

    # get the surface indices (which sides to use)
    surface_indices = utils.get_surface_indices(hpge, surface_type)

    # the bounding box should be in x +/- (radius+distance)
    # and in y -distance to height + distance

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
            r_range=(0, radius + distance),
            z_range=(-distance, height + distance),
            size=size * 5,
            seed=seed_tmp,
        )

        distances = hpge.distance_to_surface(proposals, surface_indices, signed=True)

        # should be negative (outside) and > -distance
        is_good = (distances < 0) & (abs(distances) < distance)
        sel = proposals[is_good]

        # extend
        output = np.vstack((output, sel)) if output is not None else sel

    # now cut to the right size
    return output[:size]
