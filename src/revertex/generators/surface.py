from __future__ import annotations

import logging

import legendhpges
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import rv_continuous

from revertex import core

log = logging.getLogger(__name__)


def generate_hpge_surface_points(
    n_tot: int,
    seed: int | None = None,
    *,
    hpges: dict[str, legendhpges.HPGe],
    positions: dict[str, ArrayLike],
    surface_type: str | None = None,
) -> NDArray:
    """Generate events on many HPGe's weighting by the surface area.

    Parameters
    ----------
    n_tot
        total number of events to generate
    hpges
        List of :class:`legendhpges.HPGe` objects.
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

    # index of the surfaces per detector
    surf_ids_tot = [
        np.array(hpge.surfaces) == surface_type
        if surface_type is not None
        else np.arange(len(hpge.surfaces))
        for name, hpge in hpges.items()
    ]

    # total surface area per detector
    surf_tot = [
        np.sum(hpge.surface_area(surf_ids).magnitude)
        for (name, hpge), surf_ids in zip(hpges.items(), surf_ids_tot)
    ]

    p_det = surf_tot / np.sum(surf_tot)

    det_index = rng.choice(np.arange(len(hpges)), size=n_tot, p=p_det)

    # loop over n_det maybe could be faster
    for idx, (name, hpge) in enumerate(hpges.items()):
        n = np.sum(det_index == idx)

        out[det_index == idx] = (
            generate_hpge_surface(n, hpge, surface_type=surface_type, seed=seed)
            + positions[name]
        )

    return out


def generate_hpge_surface(
    n: int,
    hpge: legendhpges.HPGe,
    surface_type: str | None,
    depth: rv_continuous | None = None,
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
    depth
        scipy `rv_continuous` object describing the depth profile, if None events are generated directly on the surface.
    seed
        seed for random number generator.

    Returns
    -------
    NDArray with shape `(n,3)` describing the local `(x,y,z)` positions for every vertex
    """
    rng = (
        np.random.default_rng(seed=seed)
        if seed is not None
        else np.random.default_rng()
    )

    # get the indices (r,z) pairs

    surf = np.array(hpge.surfaces)
    surface_indices = (
        np.where(surf == surface_type)[0]
        if (surface_type is not None)
        else np.arange(len(hpge.surfaces))
    )

    # surface areas
    areas = hpge.surface_area(surface_indices).magnitude

    # get the sides
    sides = rng.choice(surface_indices, size=n, p=areas / np.sum(areas))
    # get thhe detector geometry
    r, z = hpge.get_profile()
    s1, s2 = legendhpges.utils.get_line_segments(r, z)

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
