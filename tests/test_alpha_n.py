from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pyg4ometry as pyg4
import pytest

from revertex.cli import cli
from revertex.generators import alpha_n
from revertex.generators.alpha_n import (
    generate_material_input,
    generate_sag4n_input_file,
    prepare_sag4n_output_for_lh5,
    run_sag4n,
)

# Check if Docker is available
_DOCKER_AVAILABLE = shutil.which("docker") is not None

# Layout of tests:
# - cli tests
# - test container runtime
# - generate material input
# - generate sag4n inputs
# - generate alpha_n spectra
# - read sag4n output
# - prepare sag4n output to lh5
# - post-proc (integral yield)

### cli test


def test_cli_gdml_pathway(monkeypatch, capsys):

    def _fake_generate_alpha_n_spectrum(input_data):
        print(input_data)

    monkeypatch.setattr(
        alpha_n, "generate_alpha_n_spectrum", _fake_generate_alpha_n_spectrum
    )

    args = [
        "alpha-n-kin",
        "--gdml-file",
        "./test_files/geom.gdml",
        "--part",
        "V99000A",
        "--output-file",
        ".tmp.lh5",
        "--source-chain",
        "Th232",
    ]

    cli(args)

    captured = capsys.readouterr()
    generated_input_data = eval(captured.out)  # nosec: eval is used here

    assert generated_input_data["output_file"] == ".tmp.lh5"
    assert generated_input_data["gdml_file"] == "./test_files/geom.gdml"
    assert generated_input_data["part"] == "V99000A"
    assert generated_input_data["source_chain"] == "Th232"
    assert "n_events" in generated_input_data
    assert "seed" in generated_input_data
    assert "container_image" in generated_input_data


def test_cli_substitution_pathway(monkeypatch, capsys):

    def _fake_generate_alpha_n_spectrum(input_data):
        print(input_data)

    monkeypatch.setattr(
        alpha_n, "generate_alpha_n_spectrum", _fake_generate_alpha_n_spectrum
    )

    args = [
        "alpha-n-kin",
        "--sub-material",
        '"MATERIAL 1 MyMat 1.0 1\\n1001 -1\\nENDMATERIAL\\n"',
        "--output-file",
        ".tmp.lh5",
        "--source-chain",
        "Th232",
    ]

    cli(args)

    captured = capsys.readouterr()
    generated_input_data = eval(captured.out)  # nosec: eval is used here

    assert generated_input_data["output_file"] == ".tmp.lh5"
    assert (
        generated_input_data["sub_material"]
        == '"MATERIAL 1 MyMat 1.0 1\\n1001 -1\\nENDMATERIAL\\n"'
    )
    assert generated_input_data["source_chain"] == "Th232"
    assert "n_events" in generated_input_data
    assert "seed" in generated_input_data
    assert "container_image" in generated_input_data


def test_cli_input_sag4n_file_pathway(monkeypatch, capsys, tmp_path):

    input = """
SOURCE 1 0 0 0 0
1.00 1
Chain_Th232 0 100.0
ENDSOURCE
"""

    input_file = tmp_path / "input.txt"
    input_file.write_text(input, encoding="utf-8")

    def _fake_run_sag4n(input_data):
        input_data["output_file_sag4n"] = str(input_data["output_file_sag4n"])
        print(input_data)

    def _fake_check_for_container_runtime(runtime, image):  # noqa: ARG001
        return None

    def _fake_save_sag4n_output_to_lh5(output_data, output_file):  # noqa: ARG001
        return None

    monkeypatch.setattr(alpha_n, "run_sag4n", _fake_run_sag4n)
    monkeypatch.setattr(
        alpha_n,
        "_check_for_container_runtime_and_image",
        _fake_check_for_container_runtime,
    )
    monkeypatch.setattr(
        alpha_n, "save_sag4n_output_to_lh5", _fake_save_sag4n_output_to_lh5
    )

    args = [
        "alpha-n-kin",
        "--input-file-sag4n",
        str(input_file),
        "--output-file",
        ".tmp.lh5",
        "--source-chain",
        "Th232",
    ]

    cli(args)
    captured = capsys.readouterr()
    generated_input_data = eval(captured.out)  # nosec: eval is used here

    assert generated_input_data["output_file"] == ".tmp.lh5"
    assert generated_input_data["input_file_sag4n"] == str(input_file)
    assert generated_input_data["source_chain"] == "Th232"
    assert "n_events" in generated_input_data
    assert "seed" in generated_input_data
    assert "container_image" in generated_input_data

    input_file.unlink()  # Clean up the temporary input file


def test_cli_fails_when_more_than_one_pathway_specified():
    args = [
        "alpha-n-kin",
        "--input-file-sag4n",
        "input.txt",
        "--sub-material",
        '"MATERIAL 1 MyMat 1.0 1\\n1001 -1\\nENDMATERIAL\\n"',
        "--output-file",
        ".tmp.lh5",
        "--source-chain",
        "Th232",
    ]

    with pytest.raises(
        RuntimeError,
        match="You can only provide one of the following options to specify the target material for the alpha-n interaction",
    ):
        cli(args)


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker is not available")
def test_container_runtime_docker_available():
    runtime = alpha_n._detect_container_runtime({})
    assert runtime == "docker"


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker is not available")
def test_container_runtime_and_image_docker():
    # This test assumes that 'docker' is available in the test environment. If not, it should be skipped or adapted.
    alpha_n._check_for_container_runtime_and_image(
        "docker", "moritzneuberger/sag4n-for-revertex:latest"
    )


def test_container_run_sag4n_raises_runtime_error_when_container_executable_missing(
    monkeypatch, tmp_path
):
    input_file = tmp_path / "input.txt"
    input_file.write_text("OUTPUTFILE /data/test\n", encoding="utf-8")

    def _raise_file_not_found(*_args, **_kwargs):
        msg = "docker"
        raise FileNotFoundError(msg)

    monkeypatch.setattr(alpha_n.subprocess, "Popen", _raise_file_not_found)

    with pytest.raises(RuntimeError, match="runtime executable 'docker' not found"):
        alpha_n.run_sag4n(
            {
                "container_runtime": "docker",
                "container_image": "repo/image:tag",
                "sag4n_output_stem": "test_output",
                "input_file_sag4n": input_file,
                "output_file_sag4n": tmp_path / "test_output.out",
            }
        )


### generate material


def test_generate_material_input_for_element_only_material(tmp_path):
    reg = pyg4.geant4.Registry()

    world_s = pyg4.geant4.solid.Box("world_s", 100, 100, 100, reg, "mm")
    world_l = pyg4.geant4.LogicalVolume(world_s, "G4_Galactic", "world", reg)
    reg.setWorld(world_l)

    argon = pyg4.geant4.ElementSimple("argon", "Ar", 18, 39.95, registry=reg)
    mat = pyg4.geant4.MaterialCompound("liquid_argon", 1.39, 1, registry=reg)
    mat.add_element_massfraction(argon, 1.0)

    part_s = pyg4.geant4.solid.Box("part_s", 10, 10, 10, reg, "mm")
    part_l = pyg4.geant4.LogicalVolume(part_s, mat, "part", reg)
    pyg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], part_l, "part_pv", world_l, reg)

    gdml_file = Path(tmp_path) / "argon.gdml"
    writer = pyg4.gdml.Writer()
    writer.addDetector(reg)
    writer.write(str(gdml_file))

    material_input = generate_material_input(gdml_file, "part")

    assert "MATERIAL 1 liquid_argon 1.39 3" in material_input
    assert "18036 0.003365" in material_input
    assert "18038 0.000632" in material_input
    assert "18040 0.996003" in material_input


def test_generate_material_input_for_natoms_material(tmp_path):
    reg = pyg4.geant4.Registry()

    world_s = pyg4.geant4.solid.Box("world_s_natoms", 100, 100, 100, reg, "mm")
    world_l = pyg4.geant4.LogicalVolume(world_s, "G4_Galactic", "world_natoms", reg)
    reg.setWorld(world_l)

    isotope_h1 = pyg4.geant4.Isotope("H1", 1, 1, 1.007825, registry=reg)
    isotope_h2 = pyg4.geant4.Isotope("H2", 1, 2, 2.014102, registry=reg)
    isotope_o16 = pyg4.geant4.Isotope("O16", 8, 16, 15.9949, registry=reg)

    element_h = pyg4.geant4.ElementIsotopeMixture("Hydrogen", "H", 2, registry=reg)
    element_h.add_isotope(isotope_h1, 0.999885)
    element_h.add_isotope(isotope_h2, 0.000115)

    element_o = pyg4.geant4.ElementIsotopeMixture("Oxygen", "O", 1, registry=reg)
    element_o.add_isotope(isotope_o16, 1.0)

    material = pyg4.geant4.MaterialCompound("water_nat", 1.0, 2, registry=reg)
    material.add_element_natoms(element_h, 2)
    material.add_element_natoms(element_o, 1)

    part_s = pyg4.geant4.solid.Box("part_s_natoms", 10, 10, 10, reg, "mm")
    part_l = pyg4.geant4.LogicalVolume(part_s, material, "part_natoms", reg)
    pyg4.geant4.PhysicalVolume(
        [0, 0, 0], [0, 0, 0], part_l, "part_pv_natoms", world_l, reg
    )

    gdml_file = Path(tmp_path) / "water_nat.gdml"
    writer = pyg4.gdml.Writer()
    writer.addDetector(reg)
    writer.write(str(gdml_file))

    material_input = generate_material_input(gdml_file, "part_natoms")

    assert "MATERIAL 1 water_nat 1 3" in material_input
    assert "1001 0.66659" in material_input
    assert "1002 7.66666666667e-05" in material_input
    assert "8016 0.333333333333" in material_input


def test_generate_material_input_for_nist_material(test_gdml):
    material_input = generate_material_input(test_gdml, "LAr_l")

    assert "MATERIAL 1 G4_lAr 1.396 3" in material_input
    assert "18036 0.003365" in material_input
    assert "18038 0.000632" in material_input
    assert "18040 0.996003" in material_input


def test_generate_material_input_fails_for_unknown_part(test_gdml):
    with pytest.raises(KeyError, match="Logical volume 'does_not_exist' not found"):
        generate_material_input(test_gdml, "does_not_exist")


### generate input file


def test_generate_sag4n_input_file_uses_gdml_part(test_gdml):
    tmp_file = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
    output_file = tmp_file.name
    tmp_file.close()

    # Generate material input first (now responsibility of caller)
    sub_material = generate_material_input(test_gdml, "V99000A")

    input_data = {
        "sub_material": sub_material,
        "source_chain": "Th232",
        "n_events": 1000,
        "output_file_sag4n": Path(output_file),
    }

    input_file = generate_sag4n_input_file(input_data)

    assert Path(input_file).exists()
    content = Path(input_file).read_text(encoding="utf-8")
    assert "MATERIAL 1 EnrichedGermanium0.750" in content
    assert "Chain_Th232" in content
    assert "NEVENTS 1000" in content


def test_generate_sag4n_input_file_uses_sub_material_without_gdml(tmp_path):
    input_data = {
        "sub_material": "MATERIAL 1 MyMat 1.0 1\n1001 -1\nENDMATERIAL\n",
        "source_chain": "U238_upper",
        "n_events": 42,
        "seed": 99,
        "output_file_sag4n": tmp_path / "my_result.out",
    }

    input_file = Path(generate_sag4n_input_file(input_data))
    content = input_file.read_text(encoding="utf-8")

    assert "MATERIAL 1 MyMat 1.0 1" in content
    assert "NEVENTS 42" in content
    assert "SEED 99" in content
    assert "OUTPUTFILE /data/my_result" in content


def test_generate_sag4n_input_file_fails_when_source_chain_missing(tmp_path):
    with pytest.raises(KeyError, match="source_chain"):
        generate_sag4n_input_file(
            {
                "sub_material": "MATERIAL 1 MyMat 1.0 1\n1001 1\nENDMATERIAL\n",
                "output_file_sag4n": tmp_path / "missing_chain.out",
            }
        )


### generate alpha n spectrum


def _install_generate_spectrum_mocks(monkeypatch):
    """Install mocks for generate_alpha_n_spectrum to avoid container dependencies."""

    calls = {"save": {}, "run_count": 0}

    monkeypatch.setattr(
        alpha_n, "_detect_container_runtime", lambda _input_data: "docker"
    )
    monkeypatch.setattr(alpha_n.subprocess, "run", lambda *_args, **_kwargs: None)

    def _fake_run_sag4n(input_data):
        calls["run_count"] += 1
        output_file = Path(input_data["output_file_sag4n"])
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(
            "# comment\n"
            "EventNumber particle ekin weight x y z px py pz\n"
            "0 neutron 1.0 0.25 0 0 0 1 0 0\n"
            "1 neutron 2.0 0.75 0 0 0 0 1 0\n",
            encoding="utf-8",
        )

    def _fake_save(output_data, output_file, eunit="keV", tunit="ns"):
        calls["save"] = {
            "output_data": output_data,
            "output_file": Path(output_file),
            "eunit": eunit,
            "tunit": tunit,
        }

    monkeypatch.setattr(alpha_n, "run_sag4n", _fake_run_sag4n)
    monkeypatch.setattr(alpha_n, "save_sag4n_output_to_lh5", _fake_save)
    return calls


def test_generate_alpha_n_spectrum_requires_output_file():
    with pytest.raises(ValueError, match="'output_file' must be provided"):
        alpha_n.generate_alpha_n_spectrum({})


def test_generate_alpha_n_spectrum_with_input_file_pathway(monkeypatch, tmp_path):
    calls = _install_generate_spectrum_mocks(monkeypatch)

    input_file = tmp_path / "input.txt"
    input_file.write_text(
        f"OUTPUTFILE /data/my_chain_out\n{alpha_n.SAG4N_SOURCES['Th232']}\n",
        encoding="utf-8",
    )

    alpha_n.generate_alpha_n_spectrum(
        {
            "output_file": tmp_path / "result_input_file.lh5",
            "input_file_sag4n": input_file,
            "container_image": "repo/image:tag",
            "n_events": 10,
        }
    )

    assert calls["run_count"] == 1
    assert calls["save"] is not None
    assert calls["save"]["output_file"] == tmp_path / "result_input_file.lh5"


def test_generate_alpha_n_spectrum_with_sub_material_pathway(monkeypatch, tmp_path):
    calls = _install_generate_spectrum_mocks(monkeypatch)

    alpha_n.generate_alpha_n_spectrum(
        {
            "output_file": tmp_path / "result_submat.lh5",
            "sub_material": "MATERIAL 1 MyMat 1.0 1\n1001 1\nENDMATERIAL\n",
            "source_chain": "U238_upper",
            "container_image": "repo/image:tag",
            "n_events": 10,
        }
    )

    assert calls["run_count"] == 1
    assert calls["save"] is not None
    assert calls["save"]["output_data"]["n_simulated_events"] == 10


def test_generate_alpha_n_spectrum_with_gdml_material_pathway(
    monkeypatch, tmp_path, test_gdml
):
    calls = _install_generate_spectrum_mocks(monkeypatch)

    alpha_n.generate_alpha_n_spectrum(
        {
            "output_file": tmp_path / "result_gdml.lh5",
            "gdml_file": test_gdml,
            "part": "V99000A",
            "source_chain": "Th232",
            "container_image": "repo/image:tag",
            "n_events": 10,
        }
    )

    assert calls["run_count"] == 1
    assert calls["save"] is not None
    assert calls["save"]["output_data"]["integral_yield"] == pytest.approx(
        (0.25 + 0.75) / 10 * 6
    )


def test_generate_alpha_n_spectrum_fails_when_source_chain_missing_in_input_file(
    monkeypatch, tmp_path
):
    bad_input_file = tmp_path / "bad_input.txt"
    bad_input_file.write_text("OUTPUTFILE /data/no_chain\n", encoding="utf-8")

    monkeypatch.setattr(
        alpha_n, "_detect_container_runtime", lambda _input_data: "docker"
    )
    # Mock subprocess to avoid actual Docker calls during image validation
    monkeypatch.setattr(alpha_n.subprocess, "run", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="Could not detect source chain"):
        alpha_n.generate_alpha_n_spectrum(
            {
                "output_file": tmp_path / "result.lh5",
                "input_file_sag4n": bad_input_file,
                "container_image": "repo/image:tag",
            }
        )


@pytest.mark.parametrize(
    "extra_input",
    [
        {},
        {"gdml_file": "USE_TEST_GDML"},
        {"part": "V99000A"},
    ],
)
def test_generate_alpha_n_spectrum_fails_for_incomplete_material_input_combinations(
    monkeypatch, tmp_path, test_gdml, extra_input
):
    monkeypatch.setattr(
        alpha_n, "_detect_container_runtime", lambda _input_data: "docker"
    )
    monkeypatch.setattr(alpha_n.subprocess, "run", lambda *_args, **_kwargs: None)

    payload = {
        "output_file": tmp_path / "result.lh5",
        "container_image": "repo/image:tag",
    }
    payload.update(extra_input)
    if payload.get("gdml_file") == "USE_TEST_GDML":
        payload["gdml_file"] = test_gdml

    with pytest.raises(
        ValueError, match="Either 'sub_material' or both 'gdml_file' and 'part'"
    ):
        alpha_n.generate_alpha_n_spectrum(payload)


def test_generate_alpha_n_spectrum_fails_when_source_chain_missing_for_sub_material(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(
        alpha_n, "_detect_container_runtime", lambda _input_data: "docker"
    )
    monkeypatch.setattr(alpha_n.subprocess, "run", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="'source_chain' must be provided"):
        alpha_n.generate_alpha_n_spectrum(
            {
                "output_file": tmp_path / "result.lh5",
                "sub_material": "MATERIAL 1 MyMat 1.0 1\n1001 1\nENDMATERIAL\n",
                "container_image": "repo/image:tag",
            }
        )


@pytest.mark.skipif(not _DOCKER_AVAILABLE, reason="Docker is not available")
def test_call_to_sag4n(tmp_path):
    input_data = {
        "output_file": tmp_path / "final_output.lh5",
        "sub_material": "MATERIAL 1 MyMat 1.0 1\n8016 1\nENDMATERIAL\n",
        "source_chain": "Th232",
        "container_image": "moritzneuberger/sag4n-for-revertex:latest",
        "container_runtime": "docker",
        "n_events": 1000000,
        "sag4n_output_stem": "test_alpha_n_spectrum",
        "output_file_sag4n": tmp_path / "test_alpha_n_spectrum.out",
        "seed": 12345,
    }
    input_file = generate_sag4n_input_file(input_data)
    input_data["input_file_sag4n"] = Path(input_file)
    run_sag4n(input_data)

    assert input_data["output_file_sag4n"].exists()

    result = alpha_n.read_sag4n_output(input_data)

    assert len(result["evts"]["evtid"]) == 1
    assert result["evts"]["evtid"][0] == 548822
    assert result["evts"]["particle"][0] == "gamma"


### read sag4n output


def test_read_sag4n_output_parses_event_file(tmp_path):
    output_file = tmp_path / "sag4n.out"
    output_file.write_text(
        "# comment\n"
        "EventNumber particle ekin weight x y z px py pz\n"
        "0 neutron 1.0 0.25 0 0 0 1 0 0\n"
        "0 gamma 2.0 0.50 0 0 0 0 1 0\n"
        "1 neutron 3.0 0.75 0 0 0 0 0 1\n",
        encoding="utf-8",
    )

    result = alpha_n.read_sag4n_output(
        {
            "output_file_sag4n": output_file,
            "n_events": 10,
            "source_chain": "U238_lower",
        }
    )

    assert len(result["evts"]["evtid"]) == 3
    assert result["evts"]["particle"][0] == "neutron"
    assert result["integral_yield"] == pytest.approx((0.25 + 0.75) / 10 * 4)


### prepare output for lh5


def test_prepare_sag4n_output_for_lh5_handles_empty_array():
    empty = ak.Array(
        {
            "evtid": [],
            "particle": [],
            "ekin": [],
            "weight": [],
            "x": [],
            "y": [],
            "z": [],
            "px": [],
            "py": [],
            "pz": [],
        }
    )

    prepared = prepare_sag4n_output_for_lh5(empty)
    assert len(prepared["px"]) == 0
    assert len(prepared["g4_pid"]) == 0


def test_prepare_sag4n_output_for_lh5_non_empty_mapping():
    evts = ak.Array(
        {
            "evtid": [10, 10, 11],
            "particle": ["neutron", "gamma", "neutron"],
            "ekin": [1.0, 2.0, 3.0],
            "weight": [1.0, 1.0, 1.0],
            "x": [0.0, 0.0, 0.0],
            "y": [0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0],
            "px": [1.0, 0.0, 0.0],
            "py": [0.0, 1.0, 0.0],
            "pz": [0.0, 0.0, 1.0],
        }
    )

    prepared = prepare_sag4n_output_for_lh5(evts)

    assert prepared["g4_pid"].to_list() == [2112, 22, 2112]
    assert prepared["n_part"].to_list() == [2, 0.0, 1]
    assert prepared["time"].to_list() == [0.0, 0.0, 0.0]


### calculate integral yield


def test_calculate_integral_yield_scales_with_alphas_per_decay():
    weights = np.array([0.5, 2.0, 1.5, 5.0])
    particles = np.array(["neutron", "gamma", "neutron", "gamma"])

    integral_yield = alpha_n.calculate_integral_yield(
        weights, particles, n_events=20, decay_chain="Th232"
    )

    assert integral_yield == pytest.approx((0.5 + 1.5) / 20 * 6)
