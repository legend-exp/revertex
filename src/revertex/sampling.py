from __future__ import annotations

import logging

import hist
import numpy as np
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def sample_cylinder(
    r_range: float,
    z_range: tuple,
    size: int,
    seed: int | None,
    phi_range: tuple = (0, 2 * np.pi),
):
    """Generate points in a cylinder, returns the points as a 2D array

    Parameters
    ----------
    r_range
        The range of `r` to sample.
    z_range
        The range of `z` to sample.
    phi
        The range of angles to sample.
    size
        The number of points to generate.
    seed
        The random seed for the rng.
    """

    rng = np.random.default_rng(seed=seed)

    r2 = rng.uniform(low=r_range[0] ** 2, high=r_range[1] ** 2, size=size)
    r = np.sqrt(r2)

    z = rng.uniform(low=z_range[0], high=z_range[1], size=size)
    phi = rng.uniform(low=phi_range[0], high=phi_range[1], size=size)

    x = r * np.cos(phi)
    y = r * np.sin(phi)

    return np.column_stack((x, y, z))


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

        _rows, cols = probs.shape

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


def sample_proportional_radius(
    r0: ArrayLike, r1: ArrayLike, size: int = 10000, seed: int | None = None
):
    r"""Sample from a distribution weighted by the radius. This is used for the surface sampling og shapes.

    Based on sampling from a distribution:

    .. math::

        P(r) \propto r

    restricted to the range min(r0,r1) to max(r0,r1).


    Parameters
    ----------
    r0
        list of first radius, must have the same length as size.
    r1
        list of second, must have the same length as size.
    size
        number of samples.
    seed
        random seed for rng.
    """
    rng = (
        np.random.default_rng(seed=seed)
        if seed is not None
        else np.random.default_rng()
    )
    if len(r0) != size or len(r1) != size:
        msg = (
            f"r0 and r1 must have {size} elements not {len(r0)} (r0) or {len(r1)} (r1)"
        )
        raise ValueError(msg)

    # Ensure r0 and r1 are numpy arrays
    r0, r1 = np.asarray(r0), np.asarray(r1)

    # Get min and max for each pair
    sign = r1 > r0
    a = np.minimum(r0, r1)
    b = np.maximum(r0, r1)

    # Generate uniform samples for each pair
    u = rng.uniform(size=a.shape)  # Same shape as r0 and r1

    # Apply inverse transform sampling element-wise
    result = u
    mask = a != b

    result[mask] = (
        np.sqrt(u[mask] * (b[mask] ** 2 - a[mask] ** 2) + a[mask] ** 2) - a[mask]
    ) / (b[mask] - a[mask])

    result[~sign] = 1 - result[~sign]

    return result
