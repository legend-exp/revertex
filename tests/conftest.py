from __future__ import annotations

import shutil
import uuid
from getpass import getuser
from pathlib import Path
from tempfile import gettempdir

import dbetto
import pyg4ometry as pg4
import pygeomhpges as hpges
import pygeomtools
import pytest
from legendtestdata import LegendTestData

_tmptestdir = Path(gettempdir()) / f"revertex-tests-{getuser()}-{uuid.uuid4()!s}"


@pytest.fixture(scope="session")
def tmptestdir_global():
    _tmptestdir.mkdir(exist_ok=False)
    return _tmptestdir


@pytest.fixture(scope="module")
def tmptestdir(tmptestdir_global, request):
    p = tmptestdir_global / request.module.__name__
    p.mkdir(exist_ok=True)  # note: will be cleaned up globally.
    return p


def pytest_sessionfinish(exitstatus):
    if exitstatus == 0 and Path.exists(_tmptestdir):
        shutil.rmtree(_tmptestdir)


@pytest.fixture(scope="session")
def test_data_configs():
    ldata = LegendTestData()
    ldata.checkout("8247690")
    return ldata.get_path("legend/metadata/hardware/detectors/germanium/diodes")


@pytest.fixture(scope="session", autouse=True)
def test_gdml(test_data_configs):
    test_file_dir = Path(__file__).parent

    reg = pg4.geant4.Registry()

    # get metadata and make hpges
    meta_IC = dbetto.utils.load_dict(test_data_configs + "/V99000A.yaml")
    meta_BG = dbetto.utils.load_dict(test_data_configs + "/B99000A.yaml")

    # make HPGe
    hpge_IC = hpges.make_hpge(meta_IC, registry=reg)
    hpge_BG = hpges.make_hpge(meta_BG, registry=reg)

    # create a world volume
    world_s = pg4.geant4.solid.Orb("World_s", 20, registry=reg, lunit="cm")
    world_l = pg4.geant4.LogicalVolume(world_s, "G4_Galactic", "World", registry=reg)
    reg.setWorld(world_l)

    # let's make a liquid argon balloon
    lar_s = pg4.geant4.solid.Orb("LAr_s", 15, registry=reg, lunit="cm")
    lar_l = pg4.geant4.LogicalVolume(lar_s, "G4_lAr", "LAr_l", registry=reg)
    pg4.geant4.PhysicalVolume([0, 0, 0], [0, 0, 0], lar_l, "LAr", world_l, registry=reg)

    # now place the two HPGe detectors in the argon
    hpge_IC_pv = pg4.geant4.PhysicalVolume(
        [0, 0, 0], [5, 0, -3, "cm"], hpge_IC, "V99000A", lar_l, registry=reg
    )
    hpge_BG_pv = pg4.geant4.PhysicalVolume(
        [0, 0, 0], [-5, 0, -3, "cm"], hpge_BG, "B99000A", lar_l, registry=reg
    )

    # register them as sensitive in remage
    # this also saves the metadata into the files for later use
    hpge_IC_pv.pygeom_active_detector = pygeomtools.RemageDetectorInfo(
        "germanium",
        1,
        meta_IC,
    )
    hpge_BG_pv.pygeom_active_detector = pygeomtools.RemageDetectorInfo(
        "germanium",
        2,
        meta_BG,
    )

    pygeomtools.write_pygeom(reg, f"{test_file_dir}/test_files/geom.gdml")

    return f"{test_file_dir}/test_files/geom.gdml"
