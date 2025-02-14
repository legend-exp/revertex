from __future__ import annotations

import logging

import awkward as ak
import hist
import numpy as np
from lgdo.types import Table

log = logging.getLogger(__name__)


def sample_histogram(
    histo: hist.Hist, size: int, *, seed: int | None = None
) -> np.ndarray:
    """Generate samples from a 1D or 2D histogram.

    Based on approximating the histogram as a piecewise uniform,
    probability distribution.

    Parameters
    ----------
    histo
        The histogram to generate samples from.
    size
        The number of samples to generate.
    seed
        Random seed.

    Returns
    -------
    an array of the samples (1D case) of a tuple of x, y samples (2D case)
    """
    # create rng
    rng = np.random.default_rng(seed=seed)

    if not isinstance(histo, hist.Hist):
        msg = f"sample histogram needs hist.Hist object not {type(histo)}"
        raise TypeError(msg)

    ndim = histo.ndim

    if ndim == 1:
        # convert to numpy
        probs, bins = histo.to_numpy()

        # compute the binwidth
        binwidths = np.diff(bins)

        # normalise
        probs = probs / sum(probs)
        bin_idx = rng.choice(np.arange(len(probs)), size=size, p=probs)
        start = bins[bin_idx]

        # the value within the bin
        delta = rng.uniform(low=0, high=1, size=size) * binwidths[bin_idx]
        return start + delta

    if ndim == 2:
        probs, bins_x, bins_y = histo.to_numpy()

        binwidths_x = np.diff(bins_x)
        binwidths_y = np.diff(bins_y)

        # get the flattened bin number
        probs = probs / np.sum(probs)
        bin_idx = rng.choice(np.arange(np.size(probs)), size=size, p=probs.flatten())

        rows, cols = probs.shape

        # extract unflattened indices
        row = bin_idx // cols
        col = bin_idx % cols

        # get the delta within the bin
        delta_x = rng.uniform(low=0, high=1, size=size) * binwidths_x[row]
        delta_y = rng.uniform(low=0, high=1, size=size) * binwidths_y[col]

        # returned values
        values_x = delta_x + bins_x[row]
        values_y = delta_y + bins_y[col]

        return values_x, values_y

    msg = f"It is only supported to sample from 1D or 2D histograms not {ndim}"
    raise ValueError(msg)


def convert_output(arr: ak.Array, *, mode: str = "pos") -> Table:
    """Converts the vertices to the correct output format."""
