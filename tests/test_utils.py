from __future__ import annotations

from pathlib import Path

import numpy as np

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
