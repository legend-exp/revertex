from __future__ import annotations

import logging
from collections.abc import Mapping

import numpy as np
import pygeomhpges
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rv_continuous

from revertex import core, utils

log = logging.getLogger(__name__)


def sample_hpge_surface(
    n_tot: int,
    seed: int | None = None,
    *,
    hpges: dict[str, pygeomhpges.HPGe] | pygeomhpges.HPGe,
    positions: dict[str, ArrayLike] | ArrayLike,
    surface_type: str | None = None,
) -> NDArray:
    """Generate events on many HPGe's weighting by the surface area.

    Parameters
    ----------
    n_tot
        total number of events to generate
    hpges
        List of :class:`pygeomhpges.HPGe` objects.
    positions
        List of the origin position of each HPGe.
    surface_type
        Which surface to generate events on either `nplus`, `pplus`, `passive` or None (generate on all surfaces).
    seed
        seed for random number generator.

    Returns
    -------
    Array of global coordinates.
    """

    rng = np.random.default_rng(seed=seed)

    out = np.full((n_tot, 3), np.nan)

    # loop over n_det maybe could be faster
    if isinstance(hpges, Mapping):
        # index of the surfaces per detector
        p_det = utils.get_surface_weights(hpges, surface_type=surface_type)
        det_index = rng.choice(np.arange(len(hpges)), size=n_tot, p=p_det)

        for idx, (name, hpge) in enumerate(hpges.items()):
            n = np.sum(det_index == idx)

            out[det_index == idx] = (
                _sample_hpge_surface_impl(n, hpge, surface_type=surface_type, seed=seed)
                + positions[name]
            )
    else:
        out = (
            _sample_hpge_surface_impl(
                n_tot, hpges, surface_type=surface_type, seed=seed
            )
            + positions
        )

    return out


def _sample_hpge_surface_impl(
    n: int,
    hpge: pygeomhpges.HPGe,
    surface_type: str | None,
    depth: rv_continuous | None = None,
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
    depth
        scipy `rv_continuous` object describing the depth profile, if None events are generated directly on the surface.
    seed
        seed for random number generator.

    Returns
    -------
    Array with shape `(n,3)` describing the local `(x,y,z)` positions for every vertex
    """
    rng = (
        np.random.default_rng(seed=seed)
        if seed is not None
        else np.random.default_rng()
    )

    surface_indices = utils.get_surface_indices(hpge, surface_type)

    # surface areas
    areas = hpge.surface_area(surface_indices).magnitude

    # get the sides
    sides = rng.choice(surface_indices, size=n, p=areas / np.sum(areas))
    # get thhe detector geometry
    r, z = hpge.get_profile()
    s1, s2 = pygeomhpges.utils.get_line_segments(r, z)

    # compute random coordinates
    r1 = s1[sides][:, 0]
    r2 = s2[sides][:, 0]

    frac = core.sample_proportional_radius(r1, r2, size=(len(sides)))

    rz_coords = s1[sides] + (s2[sides] - s1[sides]) * frac[:, np.newaxis]

    phi = rng.uniform(low=0, high=2 * np.pi, size=(len(sides)))

    # convert to random x,y
    x = rz_coords[:, 0] * np.cos(phi)
    y = rz_coords[:, 0] * np.sin(phi)

    if depth is not None:
        msg = "depth profile is not yet implemented "
        raise NotImplementedError(msg)

    return np.vstack([x, y, rz_coords[:, 1]]).T
