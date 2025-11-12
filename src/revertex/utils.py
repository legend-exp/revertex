from __future__ import annotations

import logging
import re

import colorlog
import legendhpges
import numpy as np
import pyg4ometry.geant4 as pg4
import pygeomtools

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


def get_hpges(reg: pg4.geant4.registry, detectors: str | list[str]):
    """Extract the objects for each HPGe detector in `ref` and in the list of `detectors`"""

    phy_vol_dict = reg.physicalVolumeDict
    det_list = expand_regex(list(phy_vol_dict.keys()), list(detectors))

    hpges = {
        name: legendhpges.make_hpge(
            pygeomtools.get_sensvol_metadata(reg, name), registry=None
        )
        for name in det_list
    }

    pos = {name: phy_vol_dict[name].position.eval() for name in det_list}

    return hpges, pos


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
