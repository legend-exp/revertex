from __future__ import annotations

import awkward as ak
import numpy as np

from revertex.generators import beta


def test_beta():
    e = np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    p = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.2, 0.1, 0.05, 0.07])

    assert ak.all(beta.generate_beta_spectrum(100, energies=e, phase_space=p).ekin < 15)
