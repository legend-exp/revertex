from __future__ import annotations

import gzip

import lh5
import numpy as np
import pytest

from revertex import utils
from revertex.cli import cli
from revertex.generators import double_beta

ME = 510.99895


def _write_spectrum(path, energies_mev, func, endpoint):
    """Write a spectrum in the Kotila-Iachello *_2ds.txt format, with only the
    points with energy sum below `endpoint` (MeV)."""
    lines = []
    for i, e1 in enumerate(energies_mev, start=1):
        for j, e2 in enumerate(energies_mev, start=1):
            if e1 + e2 < endpoint:
                lines.append(f"{i:5d} {j:5d} {e1:7.4f} {e2:7.4f} {func(e1, e2):12.4E}")
    text = "\n".join(lines) + "\n"

    if str(path).endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(text)
    else:
        path.write_text(text)


def _directions(arr, i):
    return np.column_stack(
        [arr.px[:, i].to_numpy(), arr.py[:, i].to_numpy(), arr.pz[:, i].to_numpy()]
    )


@pytest.mark.parametrize("name", ["spec_2ds.txt", "spec_2ds.txt.gz"])
def test_read_input_2d_spectrum(tmp_path, name):
    path = tmp_path / name
    energies = np.array([0.1, 0.2, 0.3, 0.4])  # MeV
    _write_spectrum(path, energies, lambda e1, e2: 1 + e1 + 2 * e2, 0.55)

    histo = utils.read_input_2d_spectrum(str(path))
    values = histo.values()

    # grid points (in keV) are the bin centres
    assert np.allclose(histo.axes[0].centers, [100, 200, 300, 400])
    assert np.allclose(histo.axes[0].edges, [50, 150, 250, 350, 450])

    # points in the file
    assert values[0, 1] == pytest.approx(1 + 0.1 + 2 * 0.2)
    assert values[1, 0] == pytest.approx(1 + 0.2 + 2 * 0.1)
    assert values[2, 1] == pytest.approx(1 + 0.3 + 2 * 0.2)
    # points above the endpoint are not in the file
    assert values[2, 2] == 0
    assert values[3, 3] == 0

    histo_kev = utils.read_input_2d_spectrum(str(path), eunit="keV")
    assert np.allclose(histo_kev.axes[0].centers, energies)


def test_read_input_2d_spectrum_comments_and_single_point(tmp_path):
    path = tmp_path / "single_2ds.txt"
    path.write_text("# a comment\n    3     2  0.0030  0.0020   1.0E+00\n")
    histo = utils.read_input_2d_spectrum(str(path))

    # grid of three 1 keV bins per axis, starting at 0.5 keV
    assert np.allclose(histo.axes[0].edges, [0.5, 1.5, 2.5, 3.5])
    assert histo.values()[2, 1] == 1
    assert histo.values().sum() == 1


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("1 1 0.1 0.1\n", "5 columns"),
        ("1 1 0.1 0.1 nan\n", "not finite"),
        ("0 1 0.0 0.1 1\n", "positive integers"),
        ("1.5 1 0.15 0.1 1\n", "positive integers"),
        # not a uniform grid
        ("1 1 0.1 0.1 1\n2 1 0.25 0.1 1\n", "index \\* step"),
        # different steps for the two particles
        ("1 1 0.1 0.2 1\n", "index \\* step"),
        ("1 1 0.1 0.1 1\n2 1 0.2 0.1 -1\n", "negative"),
        ("1 1 0.1 0.1 0\n2 1 0.2 0.1 0\n", "no positive"),
        ("1 1 0.1 0.1 1\n1 1 0.1 0.1 2\n", "more than once"),
    ],
)
def test_read_input_2d_spectrum_invalid(tmp_path, text, match):
    path = tmp_path / "bad_2ds.txt"
    path.write_text(text)
    with pytest.raises(ValueError, match=match) as err:
        utils.read_input_2d_spectrum(str(path))
    # the message names the file
    assert str(path) in str(err.value)


def test_read_input_2d_spectrum_invalid_unit(tmp_path):
    path = tmp_path / "spec_2ds.txt"
    path.write_text("1 1 0.1 0.1 1\n")
    with pytest.raises(ValueError, match="eunits"):
        utils.read_input_2d_spectrum(str(path), eunit="GeV")


def test_validate_2d_spectrum():
    # valid: 1 keV grid in MeV, with a missing point
    utils.validate_2d_spectrum(
        np.array([[1, 1, 0.001, 0.001, 0.0], [2, 1, 0.002, 0.001, 1.0]])
    )

    with pytest.raises(ValueError, match="5 columns"):
        utils.validate_2d_spectrum(np.array([1, 1, 0.001, 0.001, 1.0]))

    with pytest.raises(ValueError, match="negative"):
        utils.validate_2d_spectrum(np.array([[1, 1, 0.001, 0.001, -1.0]]))


def test_electron_beta():
    assert double_beta.electron_beta(0) == 0
    # T = m_e gives gamma = 2
    assert double_beta.electron_beta(ME) == pytest.approx(np.sqrt(3) / 2)


@pytest.mark.parametrize("a", [-1.0, -0.6, 0.0, 1e-8, 0.5, 1.0])
def test_sample_cos_theta(a):
    cos_theta = double_beta.sample_cos_theta(np.full(400_000, a), seed=3)
    assert np.all(np.abs(cos_theta) <= 1)

    # for dN/dcos ~ 1 + a cos: <cos> = a/3 and <cos^2> = 1/3
    assert np.mean(cos_theta) == pytest.approx(a / 3, abs=0.003)
    assert np.mean(cos_theta**2) == pytest.approx(1 / 3, abs=0.003)

    with pytest.raises(ValueError, match="between -1 and 1"):
        double_beta.sample_cos_theta([1.5])


def test_generate_double_beta_kinematics(tmp_path):
    # spectrum that is zero except at (T1, T2) = (1000, 500) keV
    path = tmp_path / "point_2ds.txt"
    energies = np.arange(0.1, 2.05, 0.1)  # MeV
    _write_spectrum(
        path,
        energies,
        lambda e1, e2: float(np.isclose(e1, 1.0) and np.isclose(e2, 0.5)),
        2.15,
    )
    histo = utils.read_input_2d_spectrum(str(path))

    n = 200_000
    arr = double_beta.generate_double_beta_kinematics(n, spectrum=histo, seed=5)

    assert len(arr) == n
    assert np.all(arr.g4_pid.to_numpy() == 11)
    assert np.all(arr.time.to_numpy() == 0)

    # energies uniform within the bin around the grid point
    t1, t2 = arr.ekin[:, 0].to_numpy(), arr.ekin[:, 1].to_numpy()
    assert np.all((t1 >= 950) & (t1 < 1050))
    assert np.all((t2 >= 450) & (t2 < 550))

    d1, d2 = _directions(arr, 0), _directions(arr, 1)
    assert np.allclose(np.linalg.norm(d1, axis=1), 1)
    assert np.allclose(np.linalg.norm(d2, axis=1), 1)

    # first electron isotropic
    assert np.all(np.abs(d1.mean(axis=0)) < 0.01)

    # angular correlation 1 - beta1 beta2 cos: <cos> = -beta1 beta2 / 3
    beta12 = double_beta.electron_beta(t1) * double_beta.electron_beta(t2)
    cos_theta = np.sum(d1 * d2, axis=1)
    assert np.mean(cos_theta) == pytest.approx(-np.mean(beta12) / 3, abs=0.005)

    # uniform azimuth: the component of d2 perpendicular to d1 averages to zero
    perp = d2 - cos_theta[:, None] * d1
    assert np.all(np.abs(perp.mean(axis=0)) < 0.01)


def test_energy_spectrum(tmp_path):
    # summed energy follows the input spectrum
    path = tmp_path / "spec_2ds.txt"
    energies = np.arange(0.01, 2.04, 0.01)  # MeV
    _write_spectrum(path, energies, lambda e1, e2: max(2.04 - e1 - e2, 0) ** 5, 2.045)
    histo = utils.read_input_2d_spectrum(str(path))

    arr = double_beta.generate_double_beta_kinematics(200_000, spectrum=histo, seed=2)
    t_sum = arr.ekin[:, 0].to_numpy() + arr.ekin[:, 1].to_numpy()

    centres = histo.axes[0].centers
    expected = np.sum(histo.values() * (centres[:, None] + centres[None, :])) / np.sum(
        histo.values()
    )
    assert np.mean(t_sum) == pytest.approx(expected, abs=2)


def test_reproducible(tmp_path):
    path = tmp_path / "spec_2ds.txt"
    _write_spectrum(path, np.arange(0.1, 2.05, 0.1), lambda e1, e2: e1 * e2, 2.15)
    histo = utils.read_input_2d_spectrum(str(path))

    a = double_beta.generate_double_beta_kinematics(100, spectrum=histo, seed=7)
    b = double_beta.generate_double_beta_kinematics(100, spectrum=histo, seed=7)
    assert np.all(a.ekin.to_numpy() == b.ekin.to_numpy())
    assert np.all(a.px.to_numpy() == b.px.to_numpy())


def test_save_and_cli(tmp_path):
    spec = tmp_path / "spec_2ds.txt.gz"
    _write_spectrum(spec, np.arange(0.1, 2.05, 0.1), lambda e1, e2: e1 * e2, 2.15)

    out = tmp_path / "double_beta.lh5"
    cli(
        [
            "--seed",
            "1",
            "double-beta-kin",
            "-i",
            str(spec),
            "-o",
            str(out),
            "-n",
            "1000",
        ]
    )
    tbl = lh5.read("vtx/kin", str(out))

    # two rows per decay: n_part = 2 then 0
    n_part = tbl["n_part"].nda
    assert len(n_part) == 2000
    assert np.all(n_part[0::2] == 2)
    assert np.all(n_part[1::2] == 0)
    assert tbl["ekin"].attrs["units"] == "keV"
