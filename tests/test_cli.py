from __future__ import annotations

from pathlib import Path

from lgdo import lh5

from revertex.cli import cli


def test_cli(tmptestdir, test_gdml):
    test_file_dir = Path(__file__).parent

    # test cli for betas
    cli(
        [
            "beta-kin",
            "-i",
            f"{test_file_dir}/test_files/beta.csv",
            "-o",
            f"{tmptestdir}/test_beta.lh5",
            "-n",
            "2000",
            "-e",
            "keV",
        ]
    )

    kin = lh5.read("vtx/kin", f"{tmptestdir}/test_beta.lh5").view_as("ak")
    assert set(kin.fields) == {"px", "py", "pz", "ekin", "time", "g4_pid", "n_part"}
    assert len(kin) == 2000

    cli(
        [
            "hpge-surf-pos",
            "-g",
            test_gdml,
            "-t",
            "nplus",
            "-d",
            "B*",
            "-o",
            f"{tmptestdir}/test_surf.lh5",
            "-n",
            "1000",
        ]
    )

    pos = lh5.read("vtx/pos", f"{tmptestdir}/test_surf.lh5").view_as("ak")
    assert set(pos.fields) == {"xloc", "yloc", "zloc"}

    assert len(pos) == 1000

    cli(
        [
            "hpge-shell-pos",
            "-g",
            test_gdml,
            "-t",
            "nplus",
            "-d",
            "B*",
            "-r",
            "2",
            "-o",
            f"{tmptestdir}/test_shell.lh5",
            "-n",
            "1000",
        ]
    )

    pos = lh5.read("vtx/pos", f"{tmptestdir}/test_shell.lh5").view_as("ak")
    assert set(pos.fields) == {"xloc", "yloc", "zloc"}

    assert len(pos) == 1000

    cli(
        [
            "hpge-borehole-pos",
            "-g",
            test_gdml,
            "-d",
            "V*",
            "-o",
            f"{tmptestdir}/test_bh.lh5",
            "-n",
            "1000",
        ]
    )

    pos = lh5.read("vtx/pos", f"{tmptestdir}/test_bh.lh5").view_as("ak")
    assert set(pos.fields) == {"xloc", "yloc", "zloc"}

    assert len(pos) == 1000
