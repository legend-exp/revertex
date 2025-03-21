from __future__ import annotations

import awkward as ak
import hist
import numpy as np
from scipy import stats

from revertex import core


def test_hist_sample_one_dim():
    rng = np.random.default_rng()

    # basic case - only one bin with entries

    h = hist.Hist.new.Reg(10, 0, 10).Double().fill([0.1, 0.1])

    samples = core.sample_histogram(h, size=1000)

    # all samples in the right bin
    assert np.all((samples > 0) & (samples < 1))
    assert len(samples) == 1000

    # fill with 100 events
    n_tot = 100000
    n = 10
    h = hist.Hist.new.Reg(10, 0, n).Double().fill(rng.uniform(0, 10, size=n_tot))

    samples = core.sample_histogram(h, size=n_tot)
    h2 = hist.Hist.new.Reg(10, 0, n).Double().fill(samples)

    expected_fraction, _ = h.to_numpy()
    fraction, _ = h2.to_numpy()
    expected_fraction /= n_tot
    fraction /= n_tot

    test_stat = 2 * np.sum(
        n_tot * expected_fraction
        - (fraction * n_tot) * (1 - np.log(fraction / expected_fraction))
    )
    p = stats.chi2.sf(test_stat, n - 1)
    sigma = stats.norm.ppf(1 - p)

    assert sigma < 5


def test_hist_sample_two_dim():
    rng = np.random.default_rng()

    # basic case - only one bin with entries

    h = (
        hist.Hist.new.Reg(10, 0, 10)
        .Reg(10, 0, 10)
        .Double()
        .fill([0.1, 0.1], [0.1, 0.1])
    )

    samples_x, samples_y = core.sample_histogram(h, size=1000)

    # all samples in the right bin
    assert np.all((samples_x > 0) & (samples_x < 1))
    assert np.all((samples_y > 0) & (samples_y < 1))

    assert len(samples_x) == 1000
    assert len(samples_y) == 1000

    # fill with 100 events
    n_tot = 1000000
    n = 10
    h = (
        hist.Hist.new.Reg(10, 0, n)
        .Reg(10, 0, n)
        .Double()
        .fill(rng.uniform(0, 10, size=n_tot), rng.uniform(0, 10, size=n_tot))
    )
    sample_x, sample_y = core.sample_histogram(h, size=n_tot)
    h2 = hist.Hist.new.Reg(10, 0, n).Reg(10, 0, n).Double().fill(sample_x, sample_y)
    expected_fraction, _, _ = h.to_numpy()
    fraction, _, _ = h2.to_numpy()

    # flatten
    fraction = fraction.flatten()
    expected_fraction = expected_fraction.flatten()

    # normalise
    expected_fraction /= n_tot
    fraction /= n_tot

    test_stat = 2 * np.sum(
        n_tot * expected_fraction
        - (fraction * n_tot) * (1 - np.log(fraction / expected_fraction))
    )
    p = stats.chi2.sf(test_stat, n * n - 1)
    sigma = stats.norm.ppf(1 - p)

    assert sigma < 5


def test_convert():
    arr = ak.Array({"xloc": [1, 2, 3], "yloc": [1, 2, 3], "zloc": [1, 2, 3]})

    for f in arr.fields:
        assert ak.all(core.convert_output(arr, mode="pos").view_as("ak")[f] == arr[f])

    arr = ak.Array(
        {
            "px": [1, 2, 3],
            "py": [1, 2, 3],
            "pz": [1, 2, 3],
            "ekin": [1, 2, 3],
            "g4_pid": [11, 11, 11],
        }
    )

    for f in arr.fields:
        assert ak.all(core.convert_output(arr, mode="kin").view_as("ak")[f] == arr[f])


def test_sample_proportional_radius():
    samples = core.sample_proportional_radius(
        np.zeros(10000), np.ones(10000), size=10000
    )
    assert len(samples) == 10000
