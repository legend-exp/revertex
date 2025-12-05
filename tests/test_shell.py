from __future__ import annotations

import numpy as np
import pygeomhpges
from pyg4ometry import geant4

from revertex.generators.shell import (
    _sample_hpge_shell_impl,
    sample_hpge_shell,
)


def test_shell_gen(test_data_configs):
    hpge = pygeomhpges.make_hpge(test_data_configs + "/V99000A.yaml", registry=None)

    coords = _sample_hpge_shell_impl(100, hpge, surface_type=None, distance=1)

    assert np.shape(coords) == (100, 3)


def test_many_shell_gen(test_data_configs):
    reg = geant4.Registry()
    hpge_IC = pygeomhpges.make_hpge(test_data_configs + "/V99000A.yaml", registry=reg)
    hpge_BG = pygeomhpges.make_hpge(test_data_configs + "/B99000A.yaml", registry=reg)
    hpge_SC = pygeomhpges.make_hpge(test_data_configs + "/C99000A.yaml", registry=reg)

    coords = sample_hpge_shell(
        1000,
        seed=None,
        distance=2,
        hpges={"V99000A": hpge_IC, "B99000A": hpge_BG, "C99000A": hpge_SC},
        positions={"V99000A": [0, 0, 0], "B99000A": [0, 0, 0], "C99000A": [0, 0, 0]},
    )

    assert np.shape(coords) == (1000, 3)

    # should also work for one hpge

    coords = sample_hpge_shell(
        1000, seed=None, distance=2, hpges=hpge_IC, positions=[0, 0, 0]
    )

    assert np.shape(coords) == (1000, 3)
