from __future__ import annotations

import tempfile
from pathlib import Path

import awkward as ak
import pyg4ometry as pyg4
import pytest

from revertex.generators import alpha_n
from revertex.generators.alpha_n import (
    generate_material_input,
    generate_sag4n_input_file,
    prepare_sag4n_output_for_lh5,
)


def test_detect_container_runtime_uses_shifter_when_docker_missing(monkeypatch):
    def _fake_which(cmd):
        if cmd == "docker":
            return None
        if cmd == "shifter":
            return "/usr/bin/shifter"
        return None

    monkeypatch.setattr(alpha_n.shutil, "which", _fake_which)

    assert alpha_n._detect_container_runtime({}) == "shifter"


def test_detect_container_runtime_rejects_unsupported_runtime(monkeypatch):
    monkeypatch.setattr(alpha_n.shutil, "which", lambda _cmd: "/usr/bin/anything")

    with pytest.raises(ValueError, match="Unsupported container runtime"):
        alpha_n._detect_container_runtime({"container_runtime": "podman"})


def test_build_container_run_command_shifter_adds_docker_source_prefix():
    cmd = alpha_n._build_container_run_command(
        "shifter", "moritzneuberger/sag4n-for-revertex:latest", "/tmp/work"
    )

    assert cmd[0] == "shifter"
    assert "--image=docker:moritzneuberger/sag4n-for-revertex:latest" in cmd


def test_check_container_image_shifter_accepts_ready_image(monkeypatch):
    class _Completed:
        def __init__(self, stdout):
            self.stdout = stdout

    def _fake_run(*_args, **_kwargs):
        return _Completed(
            "perlmutter docker READY 9682fadb8f 2026-03-24T03:05:41 moritzneuberger/sag4n-for-revertex:latest\n"
        )

    monkeypatch.setattr(alpha_n.subprocess, "run", _fake_run)

    alpha_n._check_container_image(
        "shifter", "moritzneuberger/sag4n-for-revertex:latest"
    )


def test_check_container_image_shifter_raises_when_missing(monkeypatch):
    class _Completed:
        def __init__(self, stdout):
            self.stdout = stdout

    def _fake_run(*_args, **_kwargs):
        return _Completed("")

    monkeypatch.setattr(alpha_n.subprocess, "run", _fake_run)

    with pytest.raises(RuntimeError, match="Shifter image"):
        alpha_n._check_container_image(
            "shifter", "moritzneuberger/sag4n-for-revertex:latest"
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
    tmp_file = tempfile.NamedTemporaryFile(delete=False)  # noqa: SIM115
    output_file = tmp_file.name
    tmp_file.close()

    input_data = {
        "gdml_file": test_gdml,
        "part": "V99000A",
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
