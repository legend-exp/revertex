from __future__ import annotations

import legendhpges
import numpy as np
import pytest
from legendtestdata import LegendTestData
from pyg4ometry import geant4

from revertex.generators.shell import generate_hpge_shell, generate_hpge_shell_points


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_shell_gen(test_data_configs):
    hpge = legendhpges.make_hpge(test_data_configs + "/V99000A.json", registry=None)

    coords = generate_hpge_shell(100, hpge, surface_type=None, distance=1)

    assert np.shape(coords) == (100, 3)


def test_many_surface_gen(test_data_configs):
    reg = geant4.Registry()
    hpge_IC = legendhpges.make_hpge(test_data_configs + "/V99000A.json", registry=reg)
    hpge_BG = legendhpges.make_hpge(test_data_configs + "/B99000A.json", registry=reg)
    hpge_SC = legendhpges.make_hpge(test_data_configs + "/C99000A.json", registry=reg)

    coords = generate_hpge_shell_points(
        1000,
        seed=None,
        distance=2,
        hpges={"V99000A": hpge_IC, "B99000A": hpge_BG, "C99000A": hpge_SC},
        positions={"V99000A": [0, 0, 0], "B99000A": [0, 0, 0], "C99000A": [0, 0, 0]},
    )

    assert np.shape(coords) == (1000, 3)
