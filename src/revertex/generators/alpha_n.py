from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pyg4ometry as pyg4
from lgdo import lh5, types

from revertex.core import convert_output_kin
from revertex.utils import collect_isotopes

log = logging.getLogger(__name__)


sag4n_input_template = """
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

sag4n_material_template = """
MATERIAL 1 {sub_name} {sub_density} {sub_n_isotopes}
{sub_isotopes}
ENDMATERIAL
"""

sag4n_sources = {
    "th232": """
SOURCE 1 0 0 0 0
1.00 1
Chain_Th232 0 100.0
ENDSOURCE
""",
    "u238_lower": """
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
    "u238_upper": """
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

alphas_per_decay = {"th232": 6, "u238_lower": 4, "u238_upper": 4}


def calculate_integral_yield(
    weights: np.ndarray, particle: np.ndarray, n_events: int, decay_chain: str
) -> float:
    """ Helper function to calculate the integral neutron yield per emitted alpha. """
    mask = particle == "neutron"
    return np.sum(weights[mask]) / n_events * alphas_per_decay.get(decay_chain, 1)


def read_sag4n_output(input_data: dict) -> dict:
    """ Helper function to read the SaG4n output files. """
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
    """ Helper function to format the SaG4n output to remage readable output. """
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

    n_part = np.zeros(len(ak_array["evtid"]))
    time = np.zeros(len(ak_array["evtid"]))
    g4_pid = np.zeros(len(ak_array["evtid"]), dtype=int)
    particles = {"neutron": 2112, "photon": 22}
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
) -> None:
    """ Helper function to save the SaG4n generated events and integral yield to lh5 file.  """
    kin_lh5 = convert_output_kin(ak_array, eunit=eunit)
    lh5.write(kin_lh5, "vtx/kin", output_file, wo_mode="of")
    lh5.write(
        types.Scalar(integral_yield),
        "misc/integral_yield",
        output_file,
        wo_mode="append",
    )


def generate_material_input(gdml_file: str | Path, part: str) -> str:
    """    Helper function to generate material definition for the input file.    """

    reg = pyg4.gdml.Reader(str(gdml_file)).getRegistry()

    if part not in reg.logicalVolumeDict:
        volumes = ", ".join(reg.logicalVolumeDict.keys())
        msg = f"Logical volume '{part}' not found in GDML. Available: {volumes}"
        raise KeyError(msg)

    material = reg.logicalVolumeDict[part].material
    if material is None:
        msg = f"Logical volume '{part}' has no material attached."
        raise ValueError(msg)

    nist_registry = pyg4.geant4.Registry()
    nist_element_z_to_name = pyg4.geant4.getNistElementZToName()
    if getattr(material, "type", None) == "nist":
        material = pyg4.geant4.nist_material_2geant4Material(
            material.name, nist_registry
        )

    isotopes: dict[int, float] = {}
    collect_isotopes(
        material,
        1.0,
        isotopes,
        nist_registry,
        nist_element_z_to_name,
        pyg4,
    )

    total_mass_fraction = float(np.sum(list(isotopes.values())))
    isotope_lines = []
    for zaid, mass_fraction in sorted(isotopes.items()):
        isotope_lines.append(
            f"{zaid} -{mass_fraction / total_mass_fraction:.12g}"
        )  # negative sign for mass fraction in sag4n

    return sag4n_material_template.format(
        sub_name=material.name,
        sub_density=f"{float(material.density):.12g}",
        sub_n_isotopes=len(isotope_lines),
        sub_isotopes="\n".join(isotope_lines),
    )


def generate_sag4n_input_file(input_data: dict) -> str:
    """ Helper function to generate valid SaG4n input files. """

    if "sub_material" not in input_data:
        if "gdml_file" not in input_data or "part" not in input_data:
            msg = "Either 'sub_material' or both 'gdml_file' and 'part' must be provided in input_data."
            raise ValueError(msg)
        input_data["sub_material"] = generate_material_input(
            input_data["gdml_file"], input_data["part"]
        )

    output_stem = Path(input_data["output_file_sag4n"]).stem

    compiled_input = sag4n_input_template.format(
        sub_material=input_data["sub_material"],
        sub_source_chain=sag4n_sources[input_data["source_chain"]],
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


def run_sag4n(
        input_data: dict
) -> None:
    """ Wrapper for SaG4n. """
    with tempfile.TemporaryDirectory(
        prefix=".revertex_sag4n_", dir=str(Path.cwd())
    ) as tmpdir:
        input_path = Path(tmpdir) / "input.txt"
        output_stem = input_data["output_file_sag4n"].stem

#        Path(tmpdir).chmod(0o777)

        input_path.write_text(
            Path(input_data["input_file_sag4n"]).read_text(encoding="utf-8"), encoding="utf-8"
        )
#        Path(input_path).chmod(0o644)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tmpdir}:/data",
            "-w",
            "/data",
            input_data["docker_image"],
            "input.txt",
        ]

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
            msg = "Docker executable not found. Please install Docker and ensure 'docker' is in PATH."
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
            if "/data/input.txt" in details and "permission denied" in details.lower():
                msg = "Docker could not read /data/input.txt inside the container. This is usually caused by host filesystem permission restrictions on the bind mount."
                raise RuntimeError(msg) from exc
            msg = f"SaG4n Docker execution failed: {details}"
            raise RuntimeError(msg) from exc

        input_data["output_file_sag4n"].parent.mkdir(parents=True, exist_ok=True)

        main_output = Path(tmpdir) / f"{output_stem}.out"
        if not main_output.exists():
            msg = f"SaG4n output file not found at '{main_output}'."
            raise FileNotFoundError(msg)
        shutil.copy2(main_output, input_data["output_file_sag4n"])

        for suffix in (".root", ".log"):
            companion = Path(tmpdir) / f"{output_stem}{suffix}"
            if companion.exists():
                shutil.copy2(companion, input_data["output_file_sag4n"].with_suffix(suffix))


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
    - `docker_image`: Name of the Docker image to use for running SaG4n. Default is 'moritzneuberger/sag4n-for-revertex:latest'.
    - `output_file_sag4n`: Folder and stemp path of SaG4n output files (.out, .root, .log). These are usually temporary files deleted after processing.
    """

    if "output_file" not in input_data:
        msg = "'output_file' must be provided in input_data."
        raise ValueError(msg)

    if "docker_image" not in input_data:
        input_data["docker_image"] = "moritzneuberger/sag4n-for-revertex:latest"

    is_docker_available = shutil.which("docker") is not None
    if not is_docker_available:
        msg = "Docker is required to run SaG4n but was not found. Please install Docker and ensure 'docker' is in PATH."
        raise RuntimeError(msg)

    try:
        subprocess.run(
            ["docker", "image", "inspect", input_data["docker_image"]],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        msg = f"Docker image '{input_data['docker_image']}' not found. Please pull it with 'docker pull {input_data['docker_image']}'."
        raise RuntimeError(msg) from None


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
                + ", ".join(sag4n_sources.keys())
            )
            raise ValueError(msg)

        input_file = generate_sag4n_input_file(input_data)
        input_data["input_file_sag4n"] = input_file
    input_data["input_file_sag4n"] = Path(input_data["input_file_sag4n"])

    if "output_file_sag4n" in input_data:
        input_data["output_file_sag4n"] = Path(input_data["output_file_sag4n"])
    else:
        tmp_file = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
        output_file = tmp_file.name
        tmp_file.close()
        input_data["output_file_sag4n"] = Path(output_file)

    run_sag4n(input_data)
    sag4n_output = read_sag4n_output(input_data)

    evt_data = sag4n_output["evts"]
    integral_yield = sag4n_output["integral_yield"]

    save_sag4n_output_to_lh5(
        prepare_sag4n_output_for_lh5(evt_data),
        integral_yield,
        input_data["output_file"],
    )
