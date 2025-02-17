from __future__ import annotations

from pathlib import Path

from lgdo import lh5

from revertex.cli import cli


def test_cli(tmptestdir):
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
    assert kin.fields == ["px", "py", "pz", "ekin", "g4_pid"]
