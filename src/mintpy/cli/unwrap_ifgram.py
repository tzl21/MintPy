#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################

import logging
import sys

from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

from pathlib import Path

from mintpy.stdproc.unwrap_ifgram import (
    _DEFAULT_CONNCOMP_THRESH,
    _DEFAULT_COST_MODE,
    _DEFAULT_DEFO_LAY_CONST,
    _DEFAULT_DEFO_MAX_CYCLES,
    _DEFAULT_DEFO_THRESH_FACTOR,
    _DEFAULT_INIT_METHOD,
    _DEFAULT_LAMBDA,
    _DEFAULT_MAX_NCOMPS,
    _DEFAULT_MIN_CONNCOMP_FRAC,
    _DEFAULT_MIN_REGION_SIZE,
    _DEFAULT_PHASE_GRAD_WINDOW,
    _DEFAULT_TILE_COST_THRESH,
    find_matching_files,
    unwrap_batch,
)
from mintpy.stdproc.utils.log_utils import setup_logging

EXAMPLE = """example:
  unwrap_ifgram.py --processor isce3 --ifg-dir ./ifgs --cor-dir ./cors --nlooks 20.0 --max-workers 4
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Phase unwrapping via the SNAPHU binary'
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE, subparsers=subparsers)

    # ---- Processor ----
    parser.add_argument("--processor", type=str, choices=["isce2", "isce3"],
                        required=True,
                        help="Processor type")

    # ---- I/O ----
    parser.add_argument("--ifg-dir", type=Path, required=True)
    parser.add_argument("--cor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./unwrapped_output"))
    parser.add_argument("--ifg-pattern", type=str)
    parser.add_argument("--cor-pattern", type=str)

    # ---- SNAPHU core ----
    parser.add_argument("--nlooks", type=float, default=5.0)
    parser.add_argument("--cost-mode", choices=["topo", "defo", "smooth"],
                        default=_DEFAULT_COST_MODE)
    parser.add_argument("--init-method", choices=["mst", "mcf"],
                        default=_DEFAULT_INIT_METHOD)

    # ---- Deformation-mode params ----
    parser.add_argument("--defo-max-cycles", type=float,
                        default=_DEFAULT_DEFO_MAX_CYCLES,
                        help=f"DEFOMAX_CYCLE (default: {_DEFAULT_DEFO_MAX_CYCLES})")
    parser.add_argument("--defo-thresh-factor", type=float,
                        default=_DEFAULT_DEFO_THRESH_FACTOR,
                        help=f"DEFOTHRESHFACTOR (default: {_DEFAULT_DEFO_THRESH_FACTOR})")
    parser.add_argument("--defo-lay-const", type=float,
                        default=_DEFAULT_DEFO_LAY_CONST,
                        help=f"DEFOCONST for defo mode (default: {_DEFAULT_DEFO_LAY_CONST})")
    parser.add_argument("--lambda", type=float, dest="wavelength",
                        default=_DEFAULT_LAMBDA,
                        help=f"Radar wavelength in meters (default: {_DEFAULT_LAMBDA})")
    parser.add_argument("--snaphu-binary", type=str, default=None,
                        help="Path to snaphu executable (auto-detected if not set)")

    # ---- Tiling ----
    parser.add_argument("--ntiles", type=int, nargs=2, default=[1, 1],
                        metavar=("ROW", "COL"))
    parser.add_argument("--tile-overlap", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=1,
                        help="Processors per snaphu call (default: 1)")
    parser.add_argument("--tile-cost-thresh", type=int,
                        default=_DEFAULT_TILE_COST_THRESH)
    parser.add_argument("--min-region-size", type=int,
                        default=_DEFAULT_MIN_REGION_SIZE)

    # ---- Connected components ----
    parser.add_argument("--min-conncomp-frac", type=float,
                        default=_DEFAULT_MIN_CONNCOMP_FRAC)
    parser.add_argument("--conncomp-thresh", type=float,
                        default=_DEFAULT_CONNCOMP_THRESH)
    parser.add_argument("--max-ncomps", type=int, default=_DEFAULT_MAX_NCOMPS)
    parser.add_argument("--no-conncomp-out", action="store_true",
                        help="Do NOT write connected component file")

    # ---- Phase gradient window ----
    parser.add_argument("--phase-grad-window", type=int, nargs=2,
                        default=list(_DEFAULT_PHASE_GRAD_WINDOW),
                        metavar=("PSI", "DPSI"))

    # ---- Optional files ----
    parser.add_argument("--mask-file", type=Path)
    parser.add_argument("--init-phase", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--keep-scratch", action="store_true",
                        help="Keep scratch directory after processing")

    # ---- Parallelism ----
    parser.add_argument("--max-workers", type=int, default=None)

    # ---- Other ----
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.ifg_pattern is None:
        inps.ifg_pattern = "**/*.int.tif" if inps.processor == "isce3" else "**/*.int"
    if inps.cor_pattern is None:
        inps.cor_pattern = "**/*.phsig.coh.tif" if inps.processor == "isce3" else "**/*.phsig.coh"
    inps.ntiles = tuple(inps.ntiles)
    inps.phase_grad_window = tuple(inps.phase_grad_window)
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    setup_logging(verbose=getattr(inps, "verbose", False))

    try:
        if not inps.ifg_dir.exists():
            raise FileNotFoundError(f"ifg-dir not found: {inps.ifg_dir}")
        if not inps.cor_dir.exists():
            raise FileNotFoundError(f"cor-dir not found: {inps.cor_dir}")

        ifg_files, cor_files = find_matching_files(
            inps.ifg_dir, inps.cor_dir,
            inps.ifg_pattern, inps.cor_pattern, inps.processor,
        )

        logger.info("Found %d interferogram/correlation pairs", len(ifg_files))
        logger.info("Cost mode: %s, Init: %s, Nlooks: %s",
                     inps.cost_mode, inps.init_method, inps.nlooks)
        logger.info("DEFOMAX: %.2f cycles, Ntiles: %s, Nproc: %s",
                     inps.defo_max_cycles, inps.ntiles, inps.nproc)

        inps.output_dir.mkdir(parents=True, exist_ok=True)

        unwrapped, conncomps = unwrap_batch(
            ifg_paths=ifg_files, cor_paths=cor_files,
            nlooks=inps.nlooks, output_dir=inps.output_dir,
            processor=inps.processor,
            max_workers=inps.max_workers if inps.max_workers else 1,
            snaphu_bin=getattr(inps, "snaphu_binary", None),
            cost_mode=inps.cost_mode,
            init_method=inps.init_method,
            mask_paths=[inps.mask_file] * len(ifg_files) if inps.mask_file else None,
            init_phase_paths=([inps.init_phase] * len(ifg_files)
                              if inps.init_phase else None),
            ntiles=inps.ntiles, tile_overlap=inps.tile_overlap,
            nproc=inps.nproc,
            tile_cost_thresh=inps.tile_cost_thresh,
            min_region_size=inps.min_region_size,
            min_conncomp_frac=inps.min_conncomp_frac,
            conncomp_thresh=inps.conncomp_thresh,
            max_ncomps=inps.max_ncomps,
            phase_grad_window=inps.phase_grad_window,
            defo_max_cycles=inps.defo_max_cycles,
            defo_thresh_factor=inps.defo_thresh_factor,
            defo_lay_const=inps.defo_lay_const,
            wavelength=inps.wavelength,
            conncomp_out=not inps.no_conncomp_out,
            keep_scratch=inps.keep_scratch,
        )

        logger.info("Generated %d unwrapped interferograms", len(unwrapped))
        logger.info("Generated %d connected component files",
                     sum(1 for c in conncomps if c is not None))

        return 0

    except FileNotFoundError as e:
        logger.error("File error: %s", e)
        return 1
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except Exception as e:
        logger.error("Unwrapping failed: %s", e)
        return 2


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
