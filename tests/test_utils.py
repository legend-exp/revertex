from __future__ import annotations

from pathlib import Path

import legendhpges
import numpy as np
import pyg4ometry

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


def test_get_hpges():
    test_file_dir = Path(__file__).parent
    gdml = f"{test_file_dir}/test_files/geom.gdml"
    reg = pyg4ometry.gdml.Reader(gdml).getRegistry()

    hpges, pos = utils.get_hpges(reg, ["B*"])

    assert isinstance(hpges["BEGe"], legendhpges.base.HPGe)
    assert isinstance(pos["BEGe"], list)
