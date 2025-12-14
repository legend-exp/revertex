from __future__ import annotations

import logging
from collections.abc import Callable

import awkward as ak
import hist
import numpy as np
from lgdo import lh5
from lgdo.types import Array, Table
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def _get_chunks(n: int, m: int) -> np.ndarray:
    return (
        np.full(n // m, m, dtype=int)
        if n % m == 0
        else np.append(np.full(n // m, m, dtype=int), n % m)
    )


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


def convert_output_pos(
    arr: ak.Array,
    *,
    lunit: str = "mm",
) -> Table:
    """Converts the vertices to the correct output format for `pos` information.

    Parameters
    ----------
    arr
        The input data to convert
    lunit
        Unit for distances, by default mm.

    Returns
    -------
    The output table.
    """
    out = Table(size=len(arr))

    for field in ["xloc", "yloc", "zloc"]:
        assert arr[field].ndim == 1
        col = arr[field].to_numpy().astype(np.float64, copy=False)
        out.add_field(field, Array(col, attrs={"units": lunit}))

    return out


def convert_output_kin(
    arr: ak.Array,
    *,
    eunit: str = "keV",
    tunit: str = "ns",
) -> Table:
    """Converts the vertices to the correct output format for `kin` information.

    This follows the convention `defined by remage <remage:manual-input-kinetics>`__

    Parameters
    ----------
    arr
        The input data to convert
    eunit
        Unit for energy, by default keV.
    tunit
        Unit for time, by default ns.

    Returns
    -------
    The output table.
    """
    lens = []
    for field in ak.fields(arr):
        lens.append(ak.count(arr[field], axis=None))
    assert all(x == lens[0] for x in lens)
    out = Table(size=lens[0])

    for field in ["px", "py", "pz", "ekin", "time"]:
        assert arr[field].ndim in (1, 2)
        unit = eunit if field == "ekin" else ""
        unit = tunit if field == "time" else ""
        col = ak.flatten(arr[field]) if arr[field].ndim > 1 else arr[field]
        assert col.ndim == 1
        col = col.to_numpy().astype(np.float64, copy=False)
        out.add_field(field, Array(col, attrs={"units": unit}))

    for field in ["g4_pid"]:
        assert arr[field].ndim in (1, 2)
        col = ak.flatten(arr[field]) if arr[field].ndim > 1 else arr[field]
        assert col.ndim == 1
        col = col.to_numpy().astype(np.int64, copy=False)
        out.add_field(field, Array(col, dtype=np.int64))

    # derive the number of particles in each event.
    n_part = np.zeros(lens[0], dtype=np.int64)
    part_idx = 0
    for x in arr["px"]:
        part_evt = len(x) if isinstance(x, ak.Array) else 1
        n_part[part_idx] = part_evt
        part_idx += part_evt

    out.add_field("n_part", Array(n_part, dtype=np.int64))

    return out


def write_remage_vtx(
    n: int,
    out_file: str,
    seed: int | None,
    generator: Callable,
    lunit: str = "mm",
    **kwargs,
) -> None:
    """Save the vertices generatored by a particular vertex generator function.

    This follows the convention :ref:`defined by remage <remage:manual-input-vertex>`.

    Parameters
    ----------
    n
        The number of vertices to generate
    out_file
        The path to the file to save the results.
    seed
        The seed to the random number generator
    generator
        A function generating the vertices (following the revertex specifications)
    kwargs
        The keyword arguments to the function

    """

    chunks = _get_chunks(n, 1000_000)

    for idx, chunk in enumerate(chunks):
        positions = generator(chunk, **kwargs)

        pos_ak = ak.Array(
            {"xloc": positions[:, 0], "yloc": positions[:, 1], "zloc": positions[:, 2]}
        )

        msg = f"Generated vertices {pos_ak}"
        log.debug(msg)

        # update the seed
        seed = seed * 7 if seed is not None else None

        # convert
        pos_lh5 = convert_output_pos(pos_ak, lunit=lunit)

        msg = f"Output {pos_lh5}"
        log.debug(msg)

        # write
        mode = "of" if idx == 0 else "append"
        lh5.write(pos_lh5, "vtx/pos", out_file, wo_mode=mode)


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
