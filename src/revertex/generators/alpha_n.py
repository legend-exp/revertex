from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pyg4ometry as pyg4
from lgdo import Array, Table, lh5, types

from revertex.utils import collect_isotopes

log = logging.getLogger(__name__)


SAG4N_INPUT_TEMPLATE = """
################################################################################
###                               GEOMETRY                                   ###
################################################################################

WORLDSIZE 1000 # OPTIONAL


# VOLUMES:

# A 4x4x4 cm3 cube placed in the center:
VOLUME 1 Vol01 1 2 0 DummyWord 8 8 8 0 0 0


# MATERIALS:

{sub_material}

################################################################################
###                                SOURCE                                    ###
################################################################################

{sub_source_chain}

################################################################################
###                                PHYSICS                                   ###
################################################################################

MAXSTEPSIZE 0.001     # OPTIONAL
BIASFACTOR 10000   # OPTIONAL

################################################################################
###                                OTHER                                     ###
################################################################################

NEVENTS {sub_n_events}       # OPTIONAL
OUTPUTFILE /data/{sub_output_file} # without the extension
OUTPUTTYPE 1 1 2 #OPTIONAL

# if defined, interactive running mode:
# INTERACTIVE

# if defined, secondary particles are not killed when created:
# DONOTKILLSECONDARIES

# New seed for the MC (default is 1234567):
SEED {sub_seed} #OPTIONAL

END
"""

SAG4N_MATERIAL_TEMPLATE = """
MATERIAL 1 {sub_name} {sub_density} {sub_n_isotopes}
{sub_isotopes}
ENDMATERIAL
"""

SAG4N_SOURCES = {
    "Th232": """
SOURCE 1 0 0 0 0
1.00 1
Chain_Th232 0 100.0
ENDSOURCE
""",
    "U238_lower": """
SOURCE 1 0 0 0 0
1.00 24
86222  5.48948  99.92
86222  4.986  0.078
86222  4.826  0.0005
84218  6.00235  99.97890022
84218  5.181  0.00109978
85218  6.756  0.00071928
85218  6.693  0.017982
85218  6.653  0.00127872
86218  7.1292  1.9974e-05
86218  6.5311  2.54e-08
83214  5.516  0.009408
83214  5.452  0.012936
83214  5.273  0.001392
83214  5.184  0.0001464
83214  5.023  5.04e-05
83214  4.941  6e-05
84214  7.68682  99.96550252
84214  6.9022  0.010397504
84214  6.6098  5.99856e-05
82210  3.72  1.9e-06
83210  4.694  5.2e-05
83210  4.656  7.8e-05
84210  5.30433  99.99987
84210  4.51658  0.001039998648
ENDSOURCE
""",
    "U238_upper": """
SOURCE 1 0 0 0 0
1.00 23
92238  4.198  79.0
92238  4.151  20.9
92238  4.038  0.078
92234  4.7746  71.38
92234  4.7224  28.42
92234  4.6035  0.2
92234  4.2773  4e-05
92234  4.1506  2.6e-05
92234  4.1086  7e-06
90230  4.687  76.3
90230  4.6205  23.4
90230  4.4798  0.12
90230  4.4384  0.03
90230  4.3718  0.00097
90230  4.2783  8e-06
90230  4.2485  1.03e-05
90230  3.8778  3.4e-06
90230  3.8294  1.4e-06
88226  4.78434  93.84
88226  4.601  6.16
88226  4.34  0.0065
88226  4.191  0.001
88226  4.16  0.00027
ENDSOURCE
""",
}

alphas_per_decay = {"Th232": 6, "U238_lower": 4, "U238_upper": 4}


def _detect_container_runtime(input_data: dict) -> str:
    """Select container runtime, preferring docker and falling back to shifter."""
    requested_runtime = input_data.get("container_runtime")
    if requested_runtime is not None:
        runtime = str(requested_runtime).strip().lower()
        if runtime not in {"docker", "shifter"}:
            msg = (
                "Unsupported container runtime "
                f"'{requested_runtime}'. Supported runtimes are 'docker' and 'shifter'."
            )
            raise ValueError(msg)
        if shutil.which(runtime) is None:
            msg = (
                f"Requested container runtime '{runtime}' was not found in PATH. "
                "Please install it or choose a different runtime."
            )
            raise RuntimeError(msg)
        return runtime

    if shutil.which("docker") is not None:
        return "docker"
    if shutil.which("shifter") is not None:
        return "shifter"

    msg = (
        "No supported container runtime found. Install Docker or Shifter and "
        "ensure the executable is available in PATH."
    )
    raise RuntimeError(msg)


def _check_container_image(runtime: str, image: str) -> None:
    """Validate container image availability for the selected runtime when supported."""
    if runtime == "docker":
        try:
            subprocess.run(
                ["docker", "image", "inspect", image],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError:
            msg = f"Docker image '{image}' not found. Please pull it with 'docker pull {image}'."
            raise RuntimeError(msg) from None
    elif runtime == "shifter":
        try:
            proc = subprocess.run(
                ["shifterimg", "images"],
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            details = (exc.stderr or exc.stdout or str(exc)).strip()
            msg = f"Failed to query Shifter images via 'shifterimg images': {details}"
            raise RuntimeError(msg) from exc

        candidates = {image}
        if image.startswith("docker:"):
            candidates.add(image.split("docker:", 1)[1])
        else:
            candidates.add(f"docker:{image}")

        available = False
        for line in proc.stdout.splitlines():
            tokens = line.split()
            if not tokens:
                continue
            if any(token in candidates for token in tokens):
                available = True
                break

        if not available:
            image_hint = image if image.startswith("docker:") else f"docker:{image}"
            msg = (
                f"Shifter image '{image}' is not available. "
                f"Pull it first with 'shifterimg -v pull {image_hint}' and wait until status is READY."
            )
            raise RuntimeError(msg)


def _build_container_run_command(runtime: str, image: str, mount_dir: str) -> list[str]:
    """Build the runtime-specific command used to run SaG4n."""
    if runtime == "docker":
        return [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{mount_dir}:/data",
            "-w",
            "/data",
            image,
            "input.txt",
        ]

    if runtime == "shifter":
        shifter_image = image
        if ":" not in shifter_image.split("/")[0]:
            shifter_image = f"docker:{shifter_image}"
        return [
            "shifter",
            f"--image={shifter_image}",
            f"--volume={mount_dir}:/data",
            "--workdir=/data",
            "input.txt",
        ]

    msg = f"Unsupported container runtime '{runtime}'."
    raise ValueError(msg)


def calculate_integral_yield(
    weights: np.ndarray, particle: np.ndarray, n_events: int, decay_chain: str
) -> float:
    """Helper function to calculate the integral neutron yield per emitted alpha."""
    mask = particle == "neutron"
    return np.sum(weights[mask]) / n_events * alphas_per_decay.get(decay_chain, 1)


def read_sag4n_output(input_data: dict) -> dict:
    """Helper function to read the SaG4n output files."""
    fields = ["evtid", "particle", "ekin", "weight", "x", "y", "z", "px", "py", "pz"]
    output = {field: [] for field in fields}
    with Path(input_data.get("output_file_sag4n")).open() as f:
        for line in f:
            if line.startswith("#") or line.strip().startswith("EventNumber"):
                continue
            evtid, particle, ekin, weight, x, y, z, px, py, pz = line.split()
            output["evtid"].append(int(evtid))
            output["particle"].append(particle)
            output["ekin"].append(float(ekin))
            output["weight"].append(float(weight))
            output["x"].append(float(x))
            output["y"].append(float(y))
            output["z"].append(float(z))
            output["px"].append(float(px))
            output["py"].append(float(py))
            output["pz"].append(float(pz))
    return {
        "evts": ak.Array(output),
        "integral_yield": calculate_integral_yield(
            np.array(output["weight"]),
            np.array(output["particle"]),
            input_data.get("n_events", 10000000),
            input_data["source_chain"],
        ),
    }


def prepare_sag4n_output_for_lh5(ak_array: ak.Array) -> ak.Array:
    """Helper function to format the SaG4n output to remage readable output."""
    if len(ak_array["evtid"]) == 0:
        return ak.Array(
            {
                "px": np.array([], dtype=float),
                "py": np.array([], dtype=float),
                "pz": np.array([], dtype=float),
                "ekin": np.array([], dtype=float),
                "time": np.array([], dtype=float),
                "g4_pid": np.array([], dtype=int),
                "n_part": np.array([], dtype=float),
            }
        )
    _, idx, counts = np.unique(ak_array["evtid"], return_counts=True, return_index=True)
    n_part = np.zeros(len(ak_array["evtid"]))
    n_part[idx] = counts
    time = np.zeros(len(ak_array["evtid"]))
    g4_pid = np.zeros(len(ak_array["evtid"]), dtype=int)
    particles = {"neutron": 2112, "gamma": 22}
    for part, value in particles.items():
        mask = ak_array["particle"] == part
        g4_pid[mask] = value

    return ak.Array(
        {
            "px": ak_array["px"],
            "py": ak_array["py"],
            "pz": ak_array["pz"],
            "ekin": ak_array["ekin"],
            "time": time,
            "g4_pid": g4_pid,
            "n_part": n_part,
        }
    )


def save_sag4n_output_to_lh5(
    ak_array: ak.Array,
    integral_yield: float,
    output_file: str | Path,
    eunit: str = "keV",
    tunit: str = "ns",
) -> None:
    """Helper function to save the SaG4n generated events and integral yield to lh5 file."""

    kin_lh5 = Table(size=len(ak_array))

    for field in ["px", "py", "pz", "ekin", "time"]:
        assert ak_array[field].ndim in (1, 2)
        unit = eunit if field == "ekin" else ""
        unit = tunit if field == "time" else ""
        col = ak_array[field].to_numpy().astype(np.float64, copy=False)
        kin_lh5.add_field(field, Array(col, attrs={"units": unit}))

    for field in ["g4_pid", "n_part"]:
        col = ak_array[field].to_numpy().astype(np.int64, copy=False)
        kin_lh5.add_field(field, Array(col, dtype=np.int64))

    lh5.write(kin_lh5, "vtx/kin", output_file, wo_mode="of")
    lh5.write(
        types.Scalar(integral_yield),
        "misc/integral_yield",
        output_file,
        wo_mode="append",
    )


def generate_material_input(gdml_file: str | Path, part: str) -> str:
    """Helper function to generate material definition for the input file."""

    reg = pyg4.gdml.Reader(str(gdml_file)).getRegistry()

    if part not in reg.logicalVolumeDict:
        volumes = ", ".join(reg.logicalVolumeDict.keys())
        msg = f"Logical volume '{part}' not found in GDML. Available: {volumes}"
        raise KeyError(msg)

    material = reg.logicalVolumeDict[part].material
    if material is None:
        msg = f"Logical volume '{part}' has no material attached."
        raise ValueError(msg)

    nist_element_z_to_name = pyg4.geant4.getNistElementZToName()
    if getattr(material, "type", None) == "nist":
        material = pyg4.geant4.nist_material_2geant4Material(material.name, reg)

    isotopes: dict[int, float] = {}
    collect_isotopes(
        material,
        1.0,
        isotopes,
        reg,
        nist_element_z_to_name,
        pyg4,
    )

    total_mass_fraction = float(np.sum(list(isotopes.values())))
    isotope_lines = []
    for zaid, mass_fraction in sorted(isotopes.items()):
        isotope_lines.append(
            f"{zaid} -{mass_fraction / total_mass_fraction:.12g}"
        )  # negative sign for mass fraction in sag4n

    return SAG4N_MATERIAL_TEMPLATE.format(
        sub_name=material.name,
        sub_density=f"{float(material.density):.12g}",
        sub_n_isotopes=len(isotope_lines),
        sub_isotopes="\n".join(isotope_lines),
    )


def generate_sag4n_input_file(input_data: dict) -> str:
    """Helper function to generate valid SaG4n input files."""

    if "sub_material" not in input_data:
        if "gdml_file" not in input_data or "part" not in input_data:
            msg = "Either 'sub_material' or both 'gdml_file' and 'part' must be provided in input_data."
            raise ValueError(msg)
        input_data["sub_material"] = generate_material_input(
            input_data["gdml_file"], input_data["part"]
        )
    output_stem = input_data["output_file_sag4n"].stem

    compiled_input = SAG4N_INPUT_TEMPLATE.format(
        sub_material=input_data["sub_material"],
        sub_source_chain=SAG4N_SOURCES[input_data["source_chain"]],
        sub_n_events=input_data.get("n_events", 10000000),
        sub_output_file=output_stem,
        sub_seed=input_data.get("seed", input_data.get("sub_seed", 1234567)),
    )

    # log output of the generated input for debugging
    log.info("Generated SaG4n input file content:")
    log.info(compiled_input)

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(compiled_input.encode())
        return tmp_file.name


def run_sag4n(input_data: dict) -> None:
    """Wrapper for SaG4n."""
    with tempfile.TemporaryDirectory(prefix=".revertex_sag4n_") as tmpdir:
        input_path = Path(tmpdir) / "input.txt"
        runtime = input_data["container_runtime"]
        image = input_data["container_image"]
        output_stem = input_data["sag4n_output_stem"]

        #        Path(tmpdir).chmod(0o777)

        input_path.write_text(
            Path(input_data["input_file_sag4n"]).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        #        Path(input_path).chmod(0o644)

        cmd = _build_container_run_command(runtime, image, tmpdir)

        try:
            proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
            (Path(tmpdir) / f"{output_stem}.log").write_text(
                (
                    (proc.stdout or "")
                    + ("\n" if proc.stdout and proc.stderr else "")
                    + (proc.stderr or "")
                ),
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            msg = (
                f"Container runtime executable '{runtime}' not found. "
                "Please install it and ensure it is in PATH."
            )
            raise RuntimeError(msg) from exc
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            stdout = (exc.stdout or "").strip()
            details = stderr or stdout or str(exc)
            (Path(tmpdir) / f"{output_stem}.log").write_text(
                (
                    (exc.stdout or "")
                    + ("\n" if exc.stdout and exc.stderr else "")
                    + (exc.stderr or "")
                ),
                encoding="utf-8",
            )
            if (
                "permission denied" in details.lower()
                and "docker daemon socket" in details.lower()
            ):
                msg = "Cannot access Docker daemon socket. Ensure your user has permission to run Docker (e.g. join the 'docker' group or use rootless Docker), then retry."
                raise RuntimeError(msg) from exc
            if runtime == "shifter" and (
                "bind mount failed" in details.lower()
                or "failed to setup user-requested mounts" in details.lower()
                or "unclean exit from bind-mount routine" in details.lower()
            ):
                msg = (
                    "Shifter failed to bind-mount the temporary working directory. "
                    "Set TMPDIR environment variable to a Shifter-accessible path "
                    "(e.g., TMPDIR=/tmp or TMPDIR=$SCRATCH) and retry."
                )
                raise RuntimeError(msg) from exc
            if "/data/input.txt" in details and "permission denied" in details.lower():
                msg = f"{runtime.capitalize()} could not read /data/input.txt inside the container. This is usually caused by host filesystem permission restrictions on the bind mount."
                raise RuntimeError(msg) from exc
            # Display generated input file content for debugging
            input_file_content = input_path.read_text(encoding="utf-8")
            msg = f"SaG4n execution failed with {runtime}: {details}\n\nGenerated input file:\n{input_file_content}"
            raise RuntimeError(msg) from exc

        log.info("SaG4n output:")
        log.info(proc.stdout)

        input_data["output_file_sag4n"].parent.mkdir(parents=True, exist_ok=True)

        main_output = Path(tmpdir) / f"{output_stem}.out"
        if not main_output.exists():
            msg = f"SaG4n output file not found at '{main_output}'."
            raise FileNotFoundError(msg)
        shutil.copy2(main_output, input_data["output_file_sag4n"])

        for suffix in (".root", ".log"):
            companion = Path(tmpdir) / f"{output_stem}{suffix}"
            if companion.exists():
                shutil.copy2(
                    companion, input_data["output_file_sag4n"].with_suffix(suffix)
                )


def generate_alpha_n_spectrum(input_data: dict) -> None:
    """
    Generate an (alpha, n) spectrum using SaG4n and save it in LH5 format.

    There are several ways one can use this wrapper:

    1. pre-prepared `input_file_sag4n`:
        The user provides a path `input_file_sag4n` to a valid SaG4n input file. Then only `output_file` has to be provided.

    2. pre-prepared `sub_material` string:
        The user provides a `sub_material` string substituted into the template input file. In addition, the user has to provide `source_chain` and `output_file`.

    3. material read from a gdml file:
        The user provides a `gdml_file` and `part` name for a logical volume contained in the gdml file. The script will identify the material definition of this part and automatically generate the `sub_material`. In addition, the user has to provide `source_chain` and `output_file`.

    Additional optional input is:
    - `n_events`: Number of events to simulate in SaG4n. Default is 10 million.
    - `seed`: Random seed for the SaG4n simulation. Default is 1234567.
    - `container_runtime`: Container runtime to use (`docker` or `shifter`). If omitted, the script auto-detects (`docker` first, then `shifter`).
    - `container_image`: Name of the container image used to run SaG4n. Default is 'moritzneuberger/sag4n-for-revertex:latest'.
    - `output_file_sag4n`: Folder and stem path of SaG4n output files (.out, .root, .log). These are usually temporary files deleted after processing.
    """

    if "output_file" not in input_data:
        msg = "'output_file' must be provided in input_data."
        raise ValueError(msg)

    input_data["container_runtime"] = _detect_container_runtime(input_data)
    _check_container_image(
        input_data["container_runtime"], input_data["container_image"]
    )

    if "output_file_sag4n" in input_data:
        input_data["output_file_sag4n"] = Path(input_data["output_file_sag4n"])
    else:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        output_file = tmp_file.name
        tmp_file.close()
        input_data["output_file_sag4n"] = Path(output_file)
    input_data["sag4n_output_stem"] = input_data["output_file_sag4n"].stem

    if "input_file_sag4n" not in input_data:
        if "sub_material" not in input_data:
            if "gdml_file" not in input_data or "part" not in input_data:
                msg = "Either 'sub_material' or both 'gdml_file' and 'part' must be provided in input_data."
                raise ValueError(msg)
            input_data["sub_material"] = generate_material_input(
                input_data["gdml_file"], input_data["part"]
            )

        if "source_chain" not in input_data:
            msg = (
                "'source_chain' must be provided in input_data.\n Available chains: "
                + ", ".join(SAG4N_SOURCES.keys())
            )
            raise ValueError(msg)

        input_file = generate_sag4n_input_file(input_data)
        input_data["input_file_sag4n"] = Path(input_file)
    else:
        # Extract the OUTPUTFILE stem from the input file early for validation
        input_text = Path(input_data["input_file_sag4n"]).read_text(encoding="utf-8")
        output_stem = None
        for line in input_text.split("\n"):
            line_strip = line.strip()
            if line_strip.startswith("OUTPUTFILE"):
                # Extract filename from "OUTPUTFILE /data/filename" or "OUTPUTFILE filename"
                parts = line_strip.split()
                if len(parts) >= 2:
                    filepath = parts[1]
                    output_stem = Path(filepath).name
                    break
        if output_stem is None:
            output_stem = input_data["output_file_sag4n"].stem

        source_chain_in_file = None
        for chain_name, chain_def in SAG4N_SOURCES.items():
            if chain_def.strip() in input_text:
                source_chain_in_file = chain_name
                break
        if source_chain_in_file is None:
            msg = (
                "Could not detect source chain in the provided input file. "
                "Please ensure it contains one of the following source definitions:\n"
                + "\n".join(f"- {name}" for name in SAG4N_SOURCES)
            )
            raise ValueError(msg)
        input_data["source_chain"] = source_chain_in_file

        input_data["sag4n_output_stem"] = output_stem

    run_sag4n(input_data)
    sag4n_output = read_sag4n_output(input_data)

    evt_data = sag4n_output["evts"]
    integral_yield = sag4n_output["integral_yield"]
    msg = f"Integral yield: {integral_yield:.3e} (n/alpha)"
    log.info(msg)

    save_sag4n_output_to_lh5(
        prepare_sag4n_output_for_lh5(evt_data),
        integral_yield,
        input_data["output_file"],
    )
