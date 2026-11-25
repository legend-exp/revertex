from __future__ import annotations

import logging

import awkward as ak
import hist
import lh5
import numpy as np
from numpy.typing import ArrayLike, NDArray

from revertex import utils
from revertex.core import _get_chunks, convert_output_kin
from revertex.sampling import rotate_to_axis, sample_histogram, sample_isotropic

log = logging.getLogger(__name__)

# electron rest mass in keV
_ME = 510.99895

# PDG code of the electron
_PID_ELECTRON = 11

# default chunk size (events per write)
_CHUNK_SIZE = 1_000_000


def save_double_beta_kinematics(
    n_gen: int,
    in_file: str,
    out_file: str,
    *,
    seed: int | None = None,
    eunit: str = "MeV",
) -> None:
    """Save the two electrons of double-beta decays to a file.

    The electron energies are sampled from the 2D spectrum in `in_file`, see
    :func:`.utils.read_input_2d_spectrum` for the format.

    Parameters
    ----------
    n_gen
        number of decays to generate.
    in_file
        path to the text file with the 2D spectrum, can be gzip compressed.
    out_file
        path to the output file.
    seed
        random seed.
    eunit
        the unit of the energies in `in_file`, default MeV.
    """
    spectrum = utils.read_input_2d_spectrum(in_file, eunit=eunit)

    msg = f"Read 2D double-beta spectrum from {in_file}: {spectrum}"
    log.info(msg)

    # one random number generator for all chunks, so that they are independent
    rng = np.random.default_rng(seed)

    for idx, chunk in enumerate(_get_chunks(n_gen, _CHUNK_SIZE)):
        kin_ak = generate_double_beta_kinematics(chunk, spectrum=spectrum, seed=rng)

        msg = f"Generated double-beta kinematics {kin_ak}"
        log.debug(msg)

        kin_lh5 = convert_output_kin(kin_ak, eunit="keV")

        mode = "of" if idx == 0 else "append"
        lh5.write(kin_lh5, "vtx/kin", out_file, wo_mode=mode)


def generate_double_beta_kinematics(
    size: int,
    *,
    spectrum: hist.Hist,
    seed: int | np.random.Generator | None = None,
) -> ak.Array:
    r"""Generate the two electrons of double-beta decays.

    The kinetic energies :math:`(T_1, T_2)` are sampled from `spectrum`. The
    direction of the first electron is isotropic. The angle :math:`\theta`
    between the two electrons follows

    .. math::

        \frac{dN}{d\cos\theta} \propto 1 - \beta_1 \beta_2 \cos\theta,

    with :math:`\beta_i` the velocity of electron :math:`i` (in units of
    :math:`c`), see :func:`sample_cos_theta`. The azimuth of the second electron
    around the first is uniform.

    Parameters
    ----------
    size
        number of decays to generate.
    spectrum
        2D histogram of :math:`(T_1, T_2)` in keV, for example from
        :func:`.utils.read_input_2d_spectrum`.
    seed
        random seed or random number generator.

    Returns
    -------
    An awkward array with two electrons per decay, energies in keV.
    """
    rng = np.random.default_rng(seed)

    t1, t2 = sample_histogram(spectrum, size, seed=rng)

    cos_theta = sample_cos_theta(-electron_beta(t1) * electron_beta(t2), seed=rng)
    sin_theta = np.sqrt(1 - cos_theta**2)
    phi = rng.uniform(low=0, high=2 * np.pi, size=size)

    dir1 = sample_isotropic(size, seed=rng)
    dir2 = rotate_to_axis(
        np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta)),
        dir1,
    )

    directions = np.stack((dir1, dir2), axis=1)
    ekin = np.column_stack((t1, t2))

    out = {
        "px": directions[..., 0],
        "py": directions[..., 1],
        "pz": directions[..., 2],
        "ekin": ekin,
        "time": np.zeros_like(ekin),
        "g4_pid": np.full(ekin.shape, _PID_ELECTRON, dtype=np.int64),
    }
    # variable-length lists, so every decay gets its own n_part entry
    return ak.Array({key: ak.from_regular(val, axis=1) for key, val in out.items()})


def electron_beta(ekin: ArrayLike) -> NDArray:
    """Velocity of an electron in units of c, for kinetic energy `ekin` in keV."""
    ekin = np.asarray(ekin, dtype=float)
    return np.sqrt(ekin * (ekin + 2 * _ME)) / (ekin + _ME)


def sample_cos_theta(
    a: ArrayLike, seed: int | np.random.Generator | None = None
) -> NDArray:
    r"""Sample :math:`\cos\theta` from :math:`dN/d\cos\theta \propto 1 + a\cos\theta`.

    Uses the inverse of the cumulative distribution,

    .. math::

        \cos\theta = \frac{-1 + \sqrt{(1 - a)^2 + 4 a u}}{a},

    with :math:`u` uniform in [0, 1], and :math:`\cos\theta = 2u - 1` for
    :math:`a = 0`.

    Parameters
    ----------
    a
        the coefficient, one value per sample, with :math:`|a| \le 1`.
    seed
        random seed or random number generator.
    """
    a = np.asarray(a, dtype=float)
    if np.any(np.abs(a) > 1):
        msg = "the coefficient a must be between -1 and 1"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    u = rng.uniform(size=a.shape)

    # avoid the division by zero, the formula tends to 2u - 1 for a -> 0
    small = np.abs(a) < 1e-6
    a_safe = np.where(small, 1.0, a)
    cos_theta = (-1 + np.sqrt((1 - a_safe) ** 2 + 4 * a_safe * u)) / a_safe

    return np.clip(np.where(small, 2 * u - 1, cos_theta), -1, 1)
