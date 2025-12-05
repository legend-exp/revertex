from __future__ import annotations

import logging
import re

import colorlog
import numpy as np
import pyg4ometry.geant4 as pg4
import pygeomhpges
import pygeomtools

from revertex import core

log = logging.getLogger(__name__)


def expand_regex(inputs: list, patterns: list) -> list:
    """Get a list of detectors from regex

    This matches any wildcars with * or ? in the patterns.

    Parameters
    ----------
    inputs
        list of input strings to find matches in.
    patterns
        list of patterns to search for.
    """
    regex_patterns = [
        re.compile(
            "^" + p.replace(".", r"\.").replace("*", ".*").replace("?", ".") + "$"
        )
        for p in patterns
    ]
    return [v for v in inputs if any(r.fullmatch(v) for r in regex_patterns)]


def read_input_beta_csv(path: str, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """Reads a CSV file into numpy arrays.

    The file should have the following format:

        energy_1, phase_space_1
        energy_2, phase_space_2
        energy_3, phase_space_3

    Parameters
    ----------
    path
        filepath to the csv file.
    kwargs
        keyword arguments to pass to `np.genfromtxt`
    """
    return np.genfromtxt(path, **kwargs).T[0], np.genfromtxt(path, **kwargs).T[1]


def get_hpges(
    reg: pg4.geant4.registry, detectors: str | list[str]
) -> tuple[dict, dict]:
    """Extract the objects for each HPGe detector in `reg` and in the list of `detectors`"""

    phy_vol_dict = reg.physicalVolumeDict
    det_list = expand_regex(list(phy_vol_dict.keys()), list(detectors))

    hpges = {
        name: pygeomhpges.make_hpge(
            pygeomtools.get_sensvol_metadata(reg, name), registry=None
        )
        for name in det_list
    }

    pos = {name: phy_vol_dict[name].position.eval() for name in det_list}

    return hpges, pos


def get_surface_indices(hpge: pygeomhpges.base.HPGe, surface_type: str | None) -> tuple:
    """Get which surface index corresponds to the desired surface type"""

    surf = np.array(hpge.surfaces)
    return (
        np.where(surf == surface_type)[0]
        if (surface_type is not None)
        else np.arange(len(hpge.surfaces))
    )


def get_surface_weights(hpges: dict, surface_type: str | None) -> list:
    """Get a weighting for each hpge in the `hpges` based on surface area
    for a given `surface_type`
    """

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
        for (name, hpge), surf_ids in zip(hpges.items(), surf_ids_tot, strict=True)
    ]

    return surf_tot / np.sum(surf_tot)


def get_borehole_volume(hpge: pygeomhpges.HPGe, size=1000000):
    """Estimate the borehole volume (with MC)"""

    r, z = hpge.get_profile()
    height = max(z)
    radius = max(r)

    points = core.sample_cylinder(
        r_range=(0, radius), z_range=(0, height), seed=None, size=size
    )
    vol = np.pi * radius**2 * height

    is_good = len(points[hpge.is_inside_borehole(points)])

    return (is_good / size) * vol


def get_borehole_weights(hpges: dict) -> list:
    """Get a weighting for each hpge in the `hpges` based on borehole volume"""

    vol_tot = [get_borehole_volume(hpge, size=int(1e6)) for _, hpge in hpges.items()]

    return vol_tot / np.sum(vol_tot)


def setup_log(level: int | None = None) -> None:
    """Setup a colored logger for this package.

    Parameters
    ----------
    level
        initial log level, or ``None`` to use the default.
    """
    fmt = "%(log_color)s%(name)s [%(levelname)s]"
    fmt += " %(message)s"

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(fmt))

    logger = logging.getLogger("revertex")
    # logger.addHandler(handler)

    if level is not None:
        logger.setLevel(level)
