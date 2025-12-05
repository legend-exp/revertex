from __future__ import annotations

from pathlib import Path

import numpy as np
import pyg4ometry
import pygeomhpges
from pyg4ometry import geant4

from revertex import utils
from revertex.utils import read_input_beta_csv


def test_read():
    path = f"{Path(__file__).parent}/test_files/beta.csv"

    energy, spec = read_input_beta_csv(path, delimiter=",")

    assert np.all(
        energy == np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    )
    assert np.all(
        spec == np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.2, 0.1, 0.05, 0.07])
    )


def test_regex():
    assert utils.expand_regex(["B0", "V0"], ["B*"]) == ["B0"]


def test_hpges():
    test_file_dir = Path(__file__).parent
    gdml = f"{test_file_dir}/test_files/geom.gdml"
    reg = pyg4ometry.gdml.Reader(gdml).getRegistry()

    hpges, pos = utils.get_hpges(reg, ["B*"])

    assert isinstance(hpges["B99000A"], pygeomhpges.base.HPGe)
    assert isinstance(pos["B99000A"], list)

    # all
    assert len(utils.get_surface_indices(hpges["B99000A"], None)) == 8

    # just nplus
    assert len(utils.get_surface_indices(hpges["B99000A"], "nplus")) == 3

    assert utils.get_surface_weights(hpges, None)[0] == 1.0
    assert utils.get_surface_weights(hpges, "nplus")[0] == 1.0


def test_borehole(test_data_configs):
    reg = geant4.Registry()
    hpge_IC = pygeomhpges.make_hpge(test_data_configs + "/V99000A.yaml", registry=reg)

    borehole_vol = utils.get_borehole_volume(hpge_IC)
    assert isinstance(borehole_vol, float)

    assert utils.get_borehole_weights({"V99000A": hpge_IC})[0] == 1.0
