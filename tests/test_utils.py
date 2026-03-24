from __future__ import annotations

from pathlib import Path

import numpy as np
import pyg4ometry
import pygeomhpges
from pyg4ometry import geant4

from revertex import utils
from revertex.utils import read_input_beta_csv


def test_read():
    path = f"{Path(__file__).parent}/test_files/beta.csv"

    energy, spec = read_input_beta_csv(path, delimiter=",")

    assert np.all(
        energy == np.array([0.0, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
    )
    assert np.all(
        spec == np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.5, 0.2, 0.1, 0.05, 0.07])
    )


def test_regex():
    assert utils.expand_regex(["B0", "V0"], ["B*"]) == ["B0"]


def test_hpges():
    test_file_dir = Path(__file__).parent
    gdml = f"{test_file_dir}/test_files/geom.gdml"
    reg = pyg4ometry.gdml.Reader(gdml).getRegistry()

    hpges, pos = utils.get_hpges(reg, ["B*"])

    assert isinstance(hpges["B99000A"], pygeomhpges.base.HPGe)
    assert isinstance(pos["B99000A"], list)

    # all
    assert len(utils.get_surface_indices(hpges["B99000A"], None)) == 8

    # just nplus
    assert len(utils.get_surface_indices(hpges["B99000A"], "nplus")) == 3

    assert utils.get_surface_weights(hpges, None)[0] == 1.0
    assert utils.get_surface_weights(hpges, "nplus")[0] == 1.0


def test_position():
    test_file_dir = Path(__file__).parent
    gdml = f"{test_file_dir}/test_files/geom.gdml"

    reg = pyg4ometry.gdml.Reader(gdml).getRegistry()

    # B99000A is placed at (-5, 0, -3) cm in LAr, which is at the origin
    assert utils._get_position("B99000A", reg) == [-50.0, 0.0, -30.0]
    # V99000A is placed at (5, 0, -3) cm in LAr, which is at the origin
    assert utils._get_position("V99000A", reg) == [50.0, 0.0, -30.0]


def test_borehole(test_data_configs):
    reg = geant4.Registry()
    hpge_IC = pygeomhpges.make_hpge(test_data_configs + "/V99000A.yaml", registry=reg)

    borehole_vol = utils.get_borehole_volume(hpge_IC)
    assert isinstance(borehole_vol, float)

    assert utils.get_borehole_weights({"V99000A": hpge_IC})[0] == 1.0


def test_collect_isotopes(test_gdml):
    reg = pyg4ometry.gdml.Reader(str(test_gdml)).getRegistry()

    material = reg.logicalVolumeDict["V99000A"].material
    isotopes: dict[int, float] = {}
    utils.collect_isotopes(
        material,
        1.0,
        isotopes,
        pyg4ometry.geant4.Registry(),
        pyg4ometry.geant4.getNistElementZToName(),
        pyg4ometry,
    )

    assert len(isotopes) == 2
    assert 32074 in isotopes
    assert 32076 in isotopes
    assert np.isclose(isotopes[32074], 0.25)
    assert np.isclose(isotopes[32076], 0.75)


def test_collect_isotopes_resolves_string_component_names():
    reg = pyg4ometry.geant4.Registry()

    _manganese = pyg4ometry.geant4.ElementSimple(
        "Manganese", "Mn", 25, 54.938, registry=reg
    )

    class DummyMaterial:
        components = [("Manganese", 1.0, "massfraction")]  # noqa: RUF012

    isotopes: dict[int, float] = {}
    utils.collect_isotopes(
        DummyMaterial(),
        1.0,
        isotopes,
        reg,
        pyg4ometry.geant4.getNistElementZToName(),
        pyg4ometry,
    )

    assert 25055 in isotopes
    assert np.isclose(sum(isotopes.values()), 1.0)
