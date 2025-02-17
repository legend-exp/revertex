from __future__ import annotations

import logging

import awkward as ak
import hist
import numpy as np
from lgdo.types import Array, Table

log = logging.getLogger(__name__)


def _get_chunks(n: int, m: int) -> np.ndarray:
    return (
        np.full(n // m, m, dtype=int)
        if n % m == 0
        else np.append(np.full(n // m, m, dtype=int), n % m)
    )


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


def convert_output(
    arr: ak.Array, *, mode: str = "pos", lunit: str = "mm", eunit: str = "keV"
) -> Table:
    """Converts the vertices to the correct output format.

    This function creates a table of either `kin` or `pos` information.

    Parameters
    ----------
    arr
        The input data to convert
    mode
        The mode either 'pos' or 'kin'
    lunit
        Unit for distances, by default mm.
    eunit
        Unit for energy, by default keV.

    Returns
    -------
    The output table.
    """
    out = Table(size=len(arr))

    if mode == "pos":
        for field in ["xloc", "yloc", "zloc"]:
            out.add_field(field, Array(arr[field].to_numpy(), attrs={"units": lunit}))

    elif mode == "kin":
        for field in ["px", "py", "pz", "ekin"]:
            unit = eunit if field == "ekin" else f""
            out.add_field(field, Array(arr[field].to_numpy(), attrs={"units": unit}))

        out.add_field("g4_pid", Array(arr["g4_pid"].to_numpy().astype(np.int64),dtype=np.int64))
    else:
        msg = f"Only modes pos or kin are supported for converting outputs not {mode}"
        raise ValueError(msg)

    return out
