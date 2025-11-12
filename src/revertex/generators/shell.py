from __future__ import annotations

import logging

import legendhpges
from numpy.typing import NDArray

log = logging.getLogger(__name__)


def generate_hpge_shell(
    n: int,
    hpge: legendhpges.HPGe,
    surface_type: str | None,
    distance: float,
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
    distance
        Size of the hpge shell to generate in.
    seed
        seed for random number generator.

    Returns
    -------
    NDArray with shape `(n,3)` describing the local `(x,y,z)` positions for every vertex
    """
    raise NotImplementedError
