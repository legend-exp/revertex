from __future__ import annotations

import argparse
import logging

import pyg4ometry

from revertex import core, utils
from revertex.generators import beta, borehole, shell, surface
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
