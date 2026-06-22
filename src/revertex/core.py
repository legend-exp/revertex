from __future__ import annotations

import logging
from collections.abc import Callable

import awkward as ak
import lh5
import numpy as np
from lgdo.types import Array, Table
from numpy.typing import ArrayLike

log = logging.getLogger(__name__)


def convert_output_pos(
    arr: ak.Array,
    *,
    lunit: str = "mm",
) -> Table:
    """Converts the vertices to the correct output format for `pos` information.

    Parameters
    ----------
    arr
        The input data to convert.
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
    lunit: str = "mm",
    include_positions: bool = False,
) -> Table:
    """Converts the vertices to the correct output format for `kin` information.

    This follows the convention `defined by remage <remage:manual-input-kinetics>`__
    and optionally can also include positions into a combined table.

    Parameters
    ----------
    arr
        The input data to convert
    eunit
        Unit for energy, by default keV.
    tunit
        Unit for time, by default ns.
    lunit
        Unit for distances, by default mm.
    include_positions
        If positions (xloc/yloc/zloc)

    Returns
    -------
    The output table.
    """
    lens = []
    for field in ak.fields(arr):
        lens.append(ak.count(arr[field], axis=None))
    assert all(x == lens[0] for x in lens)
    out = Table(size=lens[0])

    def _flatten_col(arr: ak.Array, field: str, dtype) -> ArrayLike:
        assert arr[field].ndim in (1, 2)
        col = ak.flatten(arr[field]) if arr[field].ndim > 1 else arr[field]
        assert col.ndim == 1
        return col.to_numpy().astype(dtype, copy=False)

    for field in ["px", "py", "pz", "ekin", "time"]:
        col = _flatten_col(arr, field, np.float64)
        unit = eunit if field == "ekin" else ""
        unit = tunit if field == "time" else ""
        out.add_field(field, Array(col, attrs={"units": unit}))

    for field in ["g4_pid"]:
        col = _flatten_col(arr, field, np.int64)
        out.add_field(field, Array(col, dtype=np.int64))

    # optionally include positions into table.
    if include_positions:
        fields = ak.fields(arr)
        if "xloc" not in fields or "yloc" not in fields or "zloc" not in fields:
            msg = "no position columns available to include in output file."
            raise ValueError(msg)

        for field in ["xloc", "yloc", "zloc"]:
            col = _flatten_col(arr, field, np.float64)
            out.add_field(field, Array(col, attrs={"units": lunit}))

    # derive the number of particles in each event.
    n_part = np.zeros(lens[0], dtype=np.int64)
    part_idx = 0
    for x in arr["px"]:
        part_evt = len(x) if isinstance(x, ak.Array) else 1
        n_part[part_idx] = part_evt
        part_idx += part_evt

    out.add_field("n_part", Array(n_part, dtype=np.int64))

    return out


def _get_chunks(n: int, m: int) -> np.ndarray:
    return (
        np.full(n // m, m, dtype=int)
        if n % m == 0
        else np.append(np.full(n // m, m, dtype=int), n % m)
    )


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
