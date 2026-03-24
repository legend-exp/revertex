from __future__ import annotations

import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import pyg4ometry as pyg4
import pytest

from revertex.generators import alpha_n
from revertex.generators.alpha_n import (
    generate_material_input,
    generate_sag4n_input_file,
    prepare_sag4n_output_for_lh5,
)


def test_generate_material_input(test_gdml):
    material_input = generate_material_input(test_gdml, "V99000A")

    assert "MATERIAL 1 EnrichedGermanium0.750" in material_input
    assert " 5.5281676971 " in material_input
    assert " 2\n" in material_input
    assert "32074 -0.245027909999" in material_input
    assert "32076 -0.754972090001" in material_input
    assert "ENDMATERIAL" in material_input


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
    assert "18036 -0.00302972781022" in material_input
    assert "18038 -0.000600596039154" in material_input
    assert "18040 -0.996369676151" in material_input


def test_generate_material_input_for_nist_material(test_gdml):
    material_input = generate_material_input(test_gdml, "LAr_l")

    assert "MATERIAL 1 G4_lAr 1.396 3" in material_input
    assert "18036 -0.00302972781022" in material_input
    assert "18038 -0.000600596039154" in material_input
    assert "18040 -0.996369676151" in material_input


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
    assert "1001 -0.111900652759" in material_input
    assert "1002 -2.57203420878e-05" in material_input
    assert "8016 -0.888073626899" in material_input


def test_generate_sag4n_input_file(test_gdml):
    from revertex.generators.alpha_n import generate_material_input

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
    content = Path(input_file).read_text()
    assert "MATERIAL 1 EnrichedGermanium0.750" in content
    assert "Chain_Th232" in content
    assert "NEVENTS 1000" in content


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


def test_detect_container_runtime_uses_shifter_when_docker_missing(monkeypatch):
    def _fake_which(cmd):
        if cmd == "docker":
            return None
        if cmd == "shifter":
            return "/usr/bin/shifter"
        return None

    monkeypatch.setattr(alpha_n.shutil, "which", _fake_which)

    assert alpha_n._detect_container_runtime({}) == "shifter"



def test_detect_container_runtime_requested_runtime_missing(monkeypatch):
    monkeypatch.setattr(alpha_n.shutil, "which", lambda _cmd: None)

    with pytest.raises(RuntimeError, match="was not found in PATH"):
        alpha_n._detect_container_runtime({"container_runtime": "docker"})


def test_detect_container_runtime_raises_when_no_runtime_available(monkeypatch):
    monkeypatch.setattr(alpha_n.shutil, "which", lambda _cmd: None)

    with pytest.raises(RuntimeError, match="No supported container runtime found"):
        alpha_n._detect_container_runtime({})





def test_calculate_integral_yield_scales_with_alphas_per_decay():
    weights = np.array([0.5, 2.0, 1.5, 5.0])
    particles = np.array(["neutron", "gamma", "neutron", "gamma"])

    integral_yield = alpha_n.calculate_integral_yield(
        weights, particles, n_events=20, decay_chain="Th232"
    )

    assert integral_yield == pytest.approx((0.5 + 1.5) / 20 * 6)


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
    assert prepared["n_part"].to_list() == [2.0, 0.0, 1.0]
    assert prepared["time"].to_list() == [0.0, 0.0, 0.0]


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


def test_generate_alpha_n_spectrum_requires_output_file():
    with pytest.raises(ValueError, match="'output_file' must be provided"):
        alpha_n.generate_alpha_n_spectrum({})


def test_generate_alpha_n_spectrum_from_input_file_without_container_execution(
    monkeypatch, tmp_path
):
    input_text = (
        "HEADER\n"
        "OUTPUTFILE /data/from_input_file\n"
        f"{alpha_n.SAG4N_SOURCES['Th232'].strip()}\n"
    )
    input_file = tmp_path / "input.txt"
    input_file.write_text(input_text, encoding="utf-8")

    captured: dict = {}

    def _fake_detect_runtime(_input_data):
        return "docker"

    def _fake_check_image(_runtime, _image):
        return None

    def _fake_run_sag4n(input_data):
        captured["run_source_chain"] = input_data["source_chain"]
        captured["run_output_stem"] = input_data["sag4n_output_stem"]

    def _fake_read(_input_data):
        return {
            "evts": ak.Array(
                {
                    "evtid": [0],
                    "particle": ["neutron"],
                    "ekin": [1.0],
                    "weight": [1.0],
                    "x": [0.0],
                    "y": [0.0],
                    "z": [0.0],
                    "px": [1.0],
                    "py": [0.0],
                    "pz": [0.0],
                }
            ),
            "integral_yield": 1.23,
        }

    def _fake_prepare(evt_data):
        return evt_data

    def _fake_save(ak_array, integral_yield, output_file, eunit="keV", tunit="ns"):
        captured["saved_integral_yield"] = integral_yield
        captured["saved_output_file"] = Path(output_file)
        captured["saved_entries"] = len(ak_array["evtid"])
        captured["saved_units"] = (eunit, tunit)

    monkeypatch.setattr(alpha_n, "_detect_container_runtime", _fake_detect_runtime)
    # Mock subprocess to avoid actual Docker calls during image validation
    monkeypatch.setattr(alpha_n.subprocess, "run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(alpha_n, "run_sag4n", _fake_run_sag4n)
    monkeypatch.setattr(alpha_n, "read_sag4n_output", _fake_read)
    monkeypatch.setattr(alpha_n, "prepare_sag4n_output_for_lh5", _fake_prepare)
    monkeypatch.setattr(alpha_n, "save_sag4n_output_to_lh5", _fake_save)

    alpha_n.generate_alpha_n_spectrum(
        {
            "output_file": tmp_path / "result.lh5",
            "input_file_sag4n": input_file,
            "container_image": "repo/image:tag",
        }
    )

    assert captured["run_source_chain"] == "Th232"
    assert captured["run_output_stem"] == "from_input_file"
    assert captured["saved_integral_yield"] == pytest.approx(1.23)
    assert captured["saved_output_file"].name == "result.lh5"
    assert captured["saved_entries"] == 1
    assert captured["saved_units"] == ("keV", "ns")


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
