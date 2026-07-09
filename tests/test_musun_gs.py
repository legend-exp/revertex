from __future__ import annotations

import textwrap
from unittest.mock import patch

import awkward as ak
import lh5
import numpy as np
import pytest

from revertex.cli import cli
from revertex.generators.musun_gs import _MUON_MASS_KEV, _parse_output

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Two synthetic musun-gs output lines in the native Fortran format:
#   format(i9,i3,f10.1,3f8.1,3f10.6)
# Line 1: mu+ (id=10), E=100 GeV, downward vertical muon
# Line 2: mu- (id=11), E=200 GeV, diagonal muon
_SAMPLE_OUTPUT = textwrap.dedent("""\
        1 10     100.0  -1000.0    500.0   1750.0   0.000000   0.000000  -1.000000
        2 11     200.0   2000.0  -1000.0  -1750.0  -0.500000   0.500000  -0.707107
""")


@pytest.fixture
def sample_output_file(tmp_path):
    p = tmp_path / "muons_output.dat"
    p.write_text(_SAMPLE_OUTPUT)
    return p


# ---------------------------------------------------------------------------
# _parse_output unit tests
# ---------------------------------------------------------------------------


def test_parse_output_fields(sample_output_file):
    kin, pos = _parse_output(sample_output_file)
    assert set(ak.fields(kin)) == {"px", "py", "pz", "ekin", "time", "g4_pid"}
    assert set(ak.fields(pos)) == {"xloc", "yloc", "zloc"}


def test_parse_output_count(sample_output_file):
    kin, pos = _parse_output(sample_output_file)
    assert len(kin) == 2
    assert len(pos) == 2


def test_parse_output_particle_ids(sample_output_file):
    kin, _ = _parse_output(sample_output_file)
    pids = ak.to_numpy(kin.g4_pid)
    # GEANT3 10 (mu+) → PDG -13
    assert pids[0] == -13
    # GEANT3 11 (mu-) → PDG 13
    assert pids[1] == 13


def test_parse_output_energy_conversion(sample_output_file):
    """musun outputs GeV; revertex expects keV."""
    kin, _ = _parse_output(sample_output_file)
    ekin = ak.to_numpy(kin.ekin)
    assert ekin[0] == pytest.approx(100.0e6, rel=1e-6)  # 100 GeV → 100e6 keV
    assert ekin[1] == pytest.approx(200.0e6, rel=1e-6)


def test_parse_output_position_conversion(sample_output_file):
    """musun outputs cm; revertex uses mm (factor 10)."""
    _, pos = _parse_output(sample_output_file)
    assert ak.to_numpy(pos.xloc)[0] == pytest.approx(
        -10000.0, rel=1e-6
    )  # -1000 cm → mm
    assert ak.to_numpy(pos.yloc)[0] == pytest.approx(5000.0, rel=1e-6)
    assert ak.to_numpy(pos.zloc)[0] == pytest.approx(17500.0, rel=1e-6)


def test_parse_output_momenta_direction(sample_output_file):
    """Direction cosines must be preserved in the momentum vector."""
    kin, _ = _parse_output(sample_output_file)

    # muon 1: straight down → px=py=0, pz<0
    assert ak.to_numpy(kin.px)[0] == pytest.approx(0.0, abs=1.0)
    assert ak.to_numpy(kin.py)[0] == pytest.approx(0.0, abs=1.0)
    assert ak.to_numpy(kin.pz)[0] < 0

    # muon 2: diagonal — verify ratios match cosines
    px2 = ak.to_numpy(kin.px)[1]
    py2 = ak.to_numpy(kin.py)[1]
    pz2 = ak.to_numpy(kin.pz)[1]
    assert py2 / px2 == pytest.approx(-1.0, rel=1e-4)  # cy/cx = 0.5/-0.5 = -1
    assert pz2 < 0


def test_parse_output_momentum_magnitude(sample_output_file):
    """Relativistic momentum: |p| = sqrt(ekin*(ekin + 2*m))."""
    kin, _ = _parse_output(sample_output_file)
    ekin = ak.to_numpy(kin.ekin)
    px = ak.to_numpy(kin.px)
    py = ak.to_numpy(kin.py)
    pz = ak.to_numpy(kin.pz)

    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    p_expected = np.sqrt(ekin * (ekin + 2 * _MUON_MASS_KEV))
    np.testing.assert_allclose(p_mag, p_expected, rtol=1e-5)


def test_parse_output_time_is_zero(sample_output_file):
    kin, _ = _parse_output(sample_output_file)
    assert np.all(ak.to_numpy(kin.time) == 0.0)


# ---------------------------------------------------------------------------
# CLI integration test (container mocked)
# ---------------------------------------------------------------------------


def _fake_run_container(
    n_muons,
    seed,
    dx_cm,
    dy_cm,
    dz_cm,
    center_x_cm,
    center_y_cm,
    center_z_cm,
    runtime,  # noqa: ARG001
    image,  # noqa: ARG001
):
    """Return synthetic kinematics, positions and a fake rate without calling Docker."""
    rng = np.random.default_rng(seed)
    ekin = rng.uniform(1e5, 1e9, n_muons)
    p_mag = np.sqrt(ekin * (ekin + 2 * _MUON_MASS_KEV))
    cx = rng.uniform(-1, 1, n_muons)
    cy = rng.uniform(-1, 1, n_muons)
    cz = -np.sqrt(np.clip(1 - cx**2 - cy**2, 0, 1))  # downward

    kin_ak = ak.Array(
        {
            "px": cx * p_mag,
            "py": cy * p_mag,
            "pz": cz * p_mag,
            "ekin": ekin,
            "time": np.zeros(n_muons),
            "g4_pid": np.where(rng.integers(0, 2, n_muons) == 0, -13, 13),
        }
    )
    pos_ak = ak.Array(
        {
            "xloc": rng.uniform(
                center_x_cm - dx_cm * 10, center_x_cm + dx_cm * 10, n_muons
            ),
            "yloc": rng.uniform(
                center_y_cm - dy_cm * 10, center_y_cm + dy_cm * 10, n_muons
            ),
            "zloc": rng.uniform(
                center_z_cm - dz_cm * 10, center_z_cm + dz_cm * 10, n_muons
            ),
        }
    )
    return kin_ak, pos_ak, 4.419e-1


def test_cli_musun_gs(tmptestdir):
    out_file = str(tmptestdir / "test_musun.lh5")

    with (
        patch("revertex.generators.musun_gs._detect_runtime", return_value="docker"),
        patch("revertex.generators.musun_gs._check_image"),
        patch(
            "revertex.generators.musun_gs._run_container",
            side_effect=_fake_run_container,
        ),
    ):
        cli(
            [
                "-s",
                "42",
                "musun-gs",
                "-o",
                out_file,
                "-n",
                "500",
            ]
        )
    # print(lh5.show(out_file))
    kin = lh5.read("vtx/kin", out_file).view_as("ak")

    assert set(kin.fields) == {
        "px",
        "py",
        "pz",
        "ekin",
        "time",
        "g4_pid",
        "n_part",
        "xloc",
        "yloc",
        "zloc",
    }
    assert len(kin) == 500

    # all particles must be muons
    pids = ak.to_numpy(kin.g4_pid)
    assert np.all((pids == -13) | (pids == 13))

    # one particle per event
    assert np.all(ak.to_numpy(kin.n_part) == 1)
