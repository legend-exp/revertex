from __future__ import annotations

import awkward as ak
import hist
import numpy as np
from scipy import stats

from revertex.generators import beta


def test_beta():
    e =  np.array([ 0.,  1.,  2.,  3.,  5.,  7.,  9., 11., 13., 15.])
    p = np.array([0.1 , 0.2 , 0.3 , 0.4 , 0.6 , 0.5 , 0.2 , 0.1 , 0.05, 0.07])

    assert ak.all(beta.generate_beta_spectrum(e,p,size=100).ekin < 15)
