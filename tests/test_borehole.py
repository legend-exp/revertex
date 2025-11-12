from __future__ import annotations

import legendhpges
import numpy as np
import pytest
from legendtestdata import LegendTestData

from revertex.generators.borehole import (
    _hpge_borehole_points_impl,
    generate_hpge_borehole_points,
)


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("5f9b368")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


def test_borehole_gen(test_data_configs):
    hpge = legendhpges.make_hpge(test_data_configs + "/V99000A.json", registry=None)

    coords = _hpge_borehole_points_impl(100, hpge)

    assert np.shape(coords) == (100, 3)

    assert np.shape(
        generate_hpge_borehole_points(
            1000, None, hpges={"IC": hpge}, positions={"IC": [0, 0, 0]}
        )
    ) == (1000, 3)

    # should also work for one hpge

    coords = generate_hpge_borehole_points(
        1000, seed=None, hpges=hpge, positions=[0, 0, 0]
    )

    assert np.shape(coords) == (1000, 3)
