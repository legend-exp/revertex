from __future__ import annotations

import argparse
import logging
import random

import pyg4ometry

from revertex import core, utils
from revertex.generators import alpha_n, beta, borehole, shell, surface
from revertex.utils import setup_log

log = logging.getLogger(__name__)


def cli(args=None) -> None:
    parser = argparse.ArgumentParser(
        prog="revertex",
        description="%(prog)s command line interface",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=1,
        help="""Increase the program verbosity""",
    )
    parser.add_argument(
        "--seed",
        "-s",
        default=None,
        type=int,
        help="Seed for rng",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # beta spectra
    beta_parser = subparsers.add_parser(
        "beta-kin", help="Generate beta kinematics from a csv file."
    )

    beta_parser.add_argument(
        "--input-file",
        "-i",
        required=True,
        type=str,
        help="Path to the file with the spectrum to sample",
    )
    beta_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    beta_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )
    beta_parser.add_argument(
        "--eunit",
        "-e",
        required=True,
        type=str,
        help="Unit for the energy",
    )

    hpge_surface_parser = subparsers.add_parser(
        "hpge-surf-pos", help="Generate samples from the surface of the HPGes"
    )

    hpge_surface_parser.add_argument(
        "--gdml",
        "-g",
        required=True,
        type=str,
        help="Path to the GDML file of the geometry",
    )
    hpge_surface_parser.add_argument(
        "--surface-type",
        "-t",
        required=True,
        type=str,
        help="Type of surface",
    )
    hpge_surface_parser.add_argument(
        "--detectors",
        "-d",
        required=True,
        nargs="+",
        type=str,
        help="Name of a detector, list of detectors or regex's.",
    )
    hpge_surface_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    hpge_surface_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )

    hpge_shell_parser = subparsers.add_parser(
        "hpge-shell-pos", help="Generate samples from the shell of the HPGes"
    )

    hpge_shell_parser.add_argument(
        "--gdml",
        "-g",
        required=True,
        type=str,
        help="Path to the GDML file of the geometry",
    )
    hpge_shell_parser.add_argument(
        "--surface-type",
        "-t",
        required=True,
        type=str,
        help="Type of surface",
    )
    hpge_shell_parser.add_argument(
        "--detectors",
        "-d",
        required=True,
        nargs="+",
        type=str,
        help="Name of a detector, list of detectors or regex's.",
    )
    hpge_shell_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    hpge_shell_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )
    hpge_shell_parser.add_argument(
        "--radius",
        "-r",
        required=True,
        type=float,
        help="Radius of the shell to generate in",
    )

    hpge_borehole_parser = subparsers.add_parser(
        "hpge-borehole-pos", help="Generate samples from the borehole of the HPGes"
    )

    hpge_borehole_parser.add_argument(
        "--gdml",
        "-g",
        required=True,
        type=str,
        help="Path to the GDML file of the geometry",
    )

    hpge_borehole_parser.add_argument(
        "--detectors",
        "-d",
        required=True,
        nargs="+",
        type=str,
        help="Name of a detector, list of detectors or regex's.",
    )
    hpge_borehole_parser.add_argument(
        "--out-file",
        "-o",
        required=True,
        type=str,
        help="Path to the output file. ",
    )
    hpge_borehole_parser.add_argument(
        "--n-events",
        "-n",
        required=True,
        type=int,
        help="Number of events to generate",
    )

    alpha_n_parser = subparsers.add_parser(
        "alpha-n-kin",
        help="Generate neutron and gamma kinematics from SaG4n.",
        description="""
        There are three ways to use this option:
        (1) providing a valid SaG4n input file in `input-file-sag4n` (requires `output-file`).
        (2) providing a substitution string for the material in `sub-material` (requires `source-chain` and `output-file`).
        (3, recommended) providing a gdml file `gdml-file` and name of a logical volume `part` from which to take the material information (requires `source-chain` and `output-file`).""",
    )
    alpha_n_parser.add_argument(
        "--output-file", "-o", required=True, type=str, help="Path to output vtx file."
    )
    alpha_n_parser.add_argument(
        "--input-file-sag4n",
        type=str,
        default="",
        help="Path to valid SaG4n input file.",
    )
    alpha_n_parser.add_argument(
        "--sub-material",
        type=str,
        default="",
        help="String to substitute into the template SaG4n input file defining the target material.",
    )
    alpha_n_parser.add_argument(
        "--gdml-file",
        type=str,
        default="",
        help="Path to gdml file to read the material of `part`.",
    )
    alpha_n_parser.add_argument(
        "--part",
        type=str,
        default="",
        help="Part to extract the material information in `gdml-file` from.",
    )
    alpha_n_parser.add_argument(
        "--source-chain",
        "-s",
        type=str,
        default="",
        help="Name of the radiogenic chain to be used as alpha source. Options are [Th232, U238_lower, U238_upper]",
    )
    alpha_n_parser.add_argument(
        "--n-events",
        "-n",
        type=int,
        default=10000000,
        help="Number of primary alphas to simulate. Default is 10000000.",
    )
    alpha_n_parser.add_argument(
        "--seed",
        type=int,
        default=-1,
        help="Option for users to use custom seeds. If not set, a random seed will be used.",
    )
    alpha_n_parser.add_argument(
        "--container-runtime",
        type=str,
        default="",
        help="Container runtime for SaG4n ('docker' or 'shifter'). If omitted, runtime is auto-detected.",
    )
    alpha_n_parser.add_argument(
        "--container-image",
        type=str,
        default="moritzneuberger/sag4n-for-revertex:latest",
        help="Container image used to run SaG4n.",
    )
    alpha_n_parser.add_argument(
        "--output-file-sag4n",
        type=str,
        default="",
        help="Folder and stem path of SaG4n output files (.out, .root, .log). These are usually temporary files deleted after processing.",
    )

    args = parser.parse_args(args)

    log_level = (None, logging.INFO, logging.DEBUG)[min(args.verbose, 2)]
    setup_log(log_level)

    if args.command == "beta-kin":
        msg = f"Generating beta kinematics from {args.input_file} to {args.out_file} and seed {args.seed}"
        log.info(msg)

        beta.save_beta_spectrum(
            n_gen=args.n_events,
            out_file=args.out_file,
            in_file=args.input_file,
            seed=args.seed,
            eunit=args.eunit,
        )

    elif args.command == "hpge-surf-pos":
        msg = "Generating points on the HPGes for \n"
        msg += f"gdml:      {args.gdml} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"detectors: {args.detectors} ({args.surface_type})"
        log.info(msg)

        # read the registry
        reg = pyg4ometry.gdml.Reader(args.gdml).getRegistry()

        hpges, pos = utils.get_hpges(reg, args.detectors)

        core.write_remage_vtx(
            args.n_events,
            args.out_file,
            args.seed,
            surface.sample_hpge_surface,
            hpges=hpges,
            positions=pos,
            surface_type=args.surface_type,
        )
    elif args.command == "hpge-shell-pos":
        msg = "Generating points on the HPGes shells for \n"
        msg += f"gdml:      {args.gdml} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"detectors: {args.detectors} ({args.surface_type})"
        msg += f"radius : {args.radius}"
        log.info(msg)

        # read the registry
        reg = pyg4ometry.gdml.Reader(args.gdml).getRegistry()

        hpges, pos = utils.get_hpges(reg, args.detectors)

        core.write_remage_vtx(
            args.n_events,
            args.out_file,
            args.seed,
            shell.sample_hpge_shell,
            hpges=hpges,
            positions=pos,
            distance=args.radius,
            surface_type=args.surface_type,
        )
    elif args.command == "hpge-borehole-pos":
        msg = "Generating points on the HPGes boreholes for \n"
        msg += f"gdml:      {args.gdml} \n"
        msg += f"output:    {args.out_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"detectors: {args.detectors}"
        log.info(msg)

        # read the registry
        reg = pyg4ometry.gdml.Reader(args.gdml).getRegistry()

        hpges, pos = utils.get_hpges(reg, args.detectors)

        core.write_remage_vtx(
            args.n_events,
            args.out_file,
            args.seed,
            borehole.sample_hpge_borehole,
            hpges=hpges,
            positions=pos,
        )
    elif args.command == "alpha-n-kin":
        if args.seed == -1:
            random.seed()
            args.seed = random.randint(0, 1e5)

        input_data = {
            "output_file": args.output_file,
            "n_events": args.n_events,
            "seed": args.seed,
            "container_image": args.container_image,
        }

        if args.container_runtime != "":
            input_data["container_runtime"] = args.container_runtime

        if args.input_file_sag4n != "":
            input_data["input_file_sag4n"] = args.input_file_sag4n
        elif args.sub_material != "":
            input_data["sub_material"] = args.sub_material
            if args.source_chain == "":
                msg = "When using the 'sub-material' option, you also need to provide a 'source-chain' (with --source-chain) to specify the radiogenic chain to be used as alpha source."
                raise RuntimeError(msg)
            input_data["source_chain"] = args.source_chain
        elif args.gdml_file != "" and args.part != "":
            input_data["gdml_file"] = args.gdml_file
            input_data["part"] = args.part
            if args.source_chain == "":
                msg = "When using the 'gdml-file' and 'part' options, you also need to provide a 'source-chain' (with --source-chain) to specify the radiogenic chain to be used as alpha source."
                raise RuntimeError(msg)
            input_data["source_chain"] = args.source_chain
        else:
            msg = "For the alpha-n-kin option, you need to provide either a valid SaG4n input file in `input-file-sag4n`, or a substitution string for the material in `sub-material` (with `source-chain`) or a gdml file `gdml-file` and name of a logical volume `part` from which to take the material information (with `source-chain`)."
            raise RuntimeError(msg)

        if args.output_file_sag4n != "":
            input_data["output_file_sag4n"] = args.output_file_sag4n

        msg = "Generating neutrons and gammas using SaG4n: \n"
        if args.gdml_file != "":
            msg += f"gdml:      {args.gdml_file} \n"
        if args.part != "":
            msg += f"part:      {args.part} \n"
        if args.sub_material != "":
            msg += f"sub-material: \n{args.sub_material} \n"
        if args.input_file_sag4n != "":
            msg += f"input-file-sag4n: {args.input_file_sag4n} \n"
        if args.source_chain != "":
            msg += f"source-chain: {args.source_chain} \n"
        msg += f"output_file:    {args.output_file} \n"
        msg += f"seed:      {args.seed} \n"
        msg += f"n_events:      {args.n_events} \n"
        if args.container_runtime != "":
            msg += f"container-runtime: {args.container_runtime} \n"
        if args.container_image != "":
            msg += f"container-image: {args.container_image} \n"
        log.info(msg)

        alpha_n.generate_alpha_n_spectrum(input_data)
