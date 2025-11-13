from __future__ import annotations

import legendhpges
import numpy as np

from revertex.generators.borehole import (
    _sample_hpge_borehole_impl,
    sample_hpge_borehole,
)


def test_borehole_gen(test_data_configs):
    hpge = legendhpges.make_hpge(test_data_configs + "/V99000A.yaml", registry=None)

    coords = _sample_hpge_borehole_impl(100, hpge)

    assert np.shape(coords) == (100, 3)

    assert np.shape(
        sample_hpge_borehole(
            1000, seed = None, hpges={"IC": hpge}, positions={"IC": [0, 0, 0]}
        )
    ) == (1000, 3)

    # should also work for one hpge

    coords = sample_hpge_borehole(
        1000, seed=None, hpges=hpge, positions=[0, 0, 0]
    )

    assert np.shape(coords) == (1000, 3)
