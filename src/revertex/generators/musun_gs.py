from __future__ import annotations

import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import awkward as ak
import lh5
import numpy as np
from lgdo.types import Scalar
from numpy.typing import NDArray

from revertex.core import _get_chunks, convert_output_kin

log = logging.getLogger(__name__)

# muon rest mass in keV
_MUON_MASS_KEV = 105_658.4

# GEANT3 id -> PDG id
_GEANT3_TO_PDG: dict[int, int] = {10: -13, 11: 13}

# default chunk size (muons per container run)
_CHUNK_SIZE = 1_000_000

# regex to extract global muon intensity from container stdout
_RATE_RE = re.compile(r"Global intensity\s*=\s*([\d.E+\-]+)", re.IGNORECASE)

DEFAULT_CONTAINER_IMAGE = "ghcr.io/legend-exp/musun-gs:latest"


def generate_musun_primaries(
    n_muons: int,
    out_file: str,
    seed: int | None = None,
    *,
    dx_cm: float = 4000.0,
    dy_cm: float = 2000.0,
    dz_cm: float = 3500.0,
    container_image: str = DEFAULT_CONTAINER_IMAGE,
    container_runtime: str | None = None,
) -> None:
    """Generate atmospheric muon kinematics using musun-gs and save to LH5.

    Runs the musun-gs Fortran code inside a container, parses its ASCII output,
    and writes kinematic and position data to an LH5 file in the remage vtx
    format.

    Generates *n_muons* total by running the container in chunks of up to
    1,000,000 muons. Each chunk uses a different seed so the results are
    statistically independent.

    The global muon intensity reported by musun-gs at startup is stored as the
    scalar dataset ``/vtx/rate`` (units: ``(s)^-1``).

    Parameters
    ----------
    n_muons
        Total number of muons to generate.
    out_file
        Path to the output LH5 file.
    seed
        Base RANLUX seed. Successive chunks multiply the seed by 7 (same
        convention as other revertex generators). If ``None``, chunks use
        deterministic seeds ``1, 2, 3, ...`` based on their index.
    dx_cm
        Full width of the sampling cuboid along x [cm], centred at the
        origin. Default corresponds to the LNGS Hall A geometry (40 m).
    dy_cm
        Full width of the sampling cuboid along y [cm] (default 20 m).
    dz_cm
        Full height of the sampling cuboid along z [cm] (default 35 m).
    container_image
        Docker/Shifter image reference.
    container_runtime
        ``"docker"`` or ``"shifter"``. If ``None``, the first available
        runtime on ``PATH`` is used.
    """
    runtime = _detect_runtime(container_runtime)
    _check_image(runtime, container_image)

    Path(out_file).parent.mkdir(parents=True, exist_ok=True)

    chunks = _get_chunks(n_muons, _CHUNK_SIZE)
    chunk_seed = seed
    global_rate: float | None = None

    for idx, chunk in enumerate(chunks):
        kin_ak, pos_ak, rate = _run_container(
            int(chunk),
            chunk_seed if chunk_seed is not None else idx + 1,
            dx_cm,
            dy_cm,
            dz_cm,
            runtime,
            container_image,
        )

        if global_rate is None and rate is not None:
            global_rate = rate

        chunk_seed = chunk_seed * 7 if chunk_seed is not None else None

        combined_ak = ak.Array(
            {
                **{f: kin_ak[f] for f in ak.fields(kin_ak)},
                **{f: pos_ak[f] for f in ak.fields(pos_ak)},
            }
        )
        kin_lh5 = convert_output_kin(combined_ak, include_positions=True, lunit="mm")
        lh5.write(kin_lh5, "vtx/kin", out_file, wo_mode="append")

        msg = "Chunk %d/%d: wrote %d muons to %s"
        log.info(msg, idx + 1, len(chunks), int(chunk), out_file)

    if global_rate is not None:
        rate_scalar = Scalar(value=global_rate, attrs={"units": "(s)^-1"})
        lh5.write(rate_scalar, "vtx/rate", out_file, wo_mode="append")
        log.info("Global muon intensity: %.4e (s)^-1", global_rate)
    else:
        log.warning("Could not parse global muon intensity from container log.")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _detect_runtime(requested: str | None) -> str:
    if requested is not None:
        if requested not in ("docker", "shifter"):
            msg = f"Unsupported container runtime '{requested}'. Use 'docker' or 'shifter'."
            raise RuntimeError(msg)
        if shutil.which(requested) is None:
            msg = f"Requested container runtime '{requested}' not found on PATH."
            raise RuntimeError(msg)
        return requested
    for rt in ("docker", "shifter"):
        if shutil.which(rt):
            return rt
    msg = (
        "No container runtime found. Install Docker or Shifter and make sure "
        "it is available on PATH."
    )
    raise RuntimeError(msg)


def _check_image(runtime: str, image: str) -> None:
    if runtime == "docker":
        result = subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            msg = (
                f"Docker image '{image}' not found locally. "
                f"Follow the quick start instructions on https://github.com/legend-exp/MUSUN-gs."
            )
            raise RuntimeError(msg)
    if runtime == "shifter":
        result = subprocess.run(
            ["shifterimg", "lookup", image],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            msg = (
                f"Shifter image '{image}' not found. "
                f"Follow the quick start instructions on https://github.com/legend-exp/MUSUN-gs."
            )
            raise RuntimeError(msg)


def _write_namelist(
    path: Path, n_muons: int, seed: int, dx_cm: float, dy_cm: float, dz_cm: float
) -> None:
    path.write_text(
        f"&musun_config\n"
        f"  n_muons = {n_muons}\n"
        f"  iranlux = {seed}\n"
        f"  dx_cm   = {dx_cm}\n"
        f"  dy_cm   = {dy_cm}\n"
        f"  dz_cm   = {dz_cm}\n"
        f"/\n"
    )


def _parse_global_intensity(log_text: str) -> float | None:
    m = _RATE_RE.search(log_text)
    return float(m.group(1)) if m else None


def _run_container(
    n_muons: int,
    seed: int,
    dx_cm: float,
    dy_cm: float,
    dz_cm: float,
    runtime: str,
    image: str,
) -> tuple[ak.Array, ak.Array, float | None]:
    with tempfile.TemporaryDirectory(prefix=".revertex_musun_") as tmpdir:
        tmp = Path(tmpdir)
        _write_namelist(tmp / "input.nml", n_muons, seed, dx_cm, dy_cm, dz_cm)

        if runtime == "docker":
            cmd = ["docker", "run", "--rm", "-v", f"{tmpdir}:/data", image]
        elif runtime == "shifter":
            cmd = [
                "shifter",
                f"--image={image}",
                f"--volume={tmpdir}:/data",
                "--entrypoint",
            ]
        else:
            msg = f"Unknown container runtime: {runtime}"
            raise RuntimeError(msg)

        log.debug("Running: %s", " ".join(cmd))

        log_path = tmp / "musun.log"
        with log_path.open("w") as lf:
            result = subprocess.run(
                cmd, stdout=lf, stderr=subprocess.STDOUT, check=False
            )

        log_content = log_path.read_text()

        if result.returncode != 0:
            msg = f"musun container failed (exit {result.returncode}):\n{log_content}"
            raise RuntimeError(msg)

        output_path = tmp / "muons_output.dat"
        if not output_path.exists():
            msg = (
                "Container finished successfully but /data/muons_output.dat is missing."
            )
            raise RuntimeError(msg)

        rate = _parse_global_intensity(log_content)
        kin_ak, pos_ak = _parse_output(output_path)
        return kin_ak, pos_ak, rate


def _parse_output(path: Path) -> tuple[ak.Array, ak.Array]:
    """Parse musun-gs ASCII output into kinematic and position ak.Arrays.

    Output columns: muon_num  id_part  energy_GeV  x_cm  y_cm  z_cm  cx  cy  cz
    id_part: GEANT3 (10 = mu+, 11 = mu-)
    """
    data: NDArray = np.loadtxt(path)
    if data.ndim == 1:
        data = data[np.newaxis, :]

    # particle IDs: GEANT3 → PDG
    geant3_id = data[:, 1].astype(np.int64)
    g4_pid = np.empty_like(geant3_id, dtype=np.int64)
    mask_muplus = geant3_id == 10
    mask_muminus = geant3_id == 11
    g4_pid[mask_muplus] = -13
    g4_pid[mask_muminus] = 13
    if not np.all(mask_muplus | mask_muminus):
        unknown = np.unique(geant3_id[~(mask_muplus | mask_muminus)])
        msg = f"Unsupported GEANT3 particle IDs in musun output: {unknown.tolist()}"
        raise ValueError(msg)
    # energy: musun GeV (ultrarelativistic: E_total ≈ E_kin) → keV
    ekin_kev = data[:, 2] * 1e6

    # momenta: |p| = sqrt(ekin*(ekin + 2*m)), direction from cosines
    p_mag = np.sqrt(ekin_kev * (ekin_kev + 2 * _MUON_MASS_KEV))
    cx, cy, cz = data[:, 6], data[:, 7], data[:, 8]

    kin_ak = ak.Array(
        {
            "px": cx * p_mag,
            "py": cy * p_mag,
            "pz": cz * p_mag,
            "ekin": ekin_kev,
            "time": np.zeros(len(ekin_kev)),
            "g4_pid": g4_pid,
        }
    )

    # positions: cm → mm
    pos_ak = ak.Array(
        {
            "xloc": data[:, 3] * 10.0,
            "yloc": data[:, 4] * 10.0,
            "zloc": data[:, 5] * 10.0,
        }
    )

    return kin_ak, pos_ak
