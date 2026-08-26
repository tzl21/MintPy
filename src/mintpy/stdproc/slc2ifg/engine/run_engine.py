#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
run_engine.py — Dask-scheduled SLC-to-unwrapped engine (see docs/engine_design.md).

Examples:
    # Full pipeline (default: keep only unw/conncomp/phsig)
    run_engine.py slc2ifg.cfg

    # Plan only: print the DAG and write graph.dot / graph.svg
    run_engine.py slc2ifg.cfg --dry-run

    # Run a single tool standalone (inputs must exist from a previous run)
    run_engine.py slc2ifg.cfg --tool unwrap

    # Keep all intermediates / use distributed scheduler
    run_engine.py slc2ifg.cfg --keep-intermediates all
    run_engine.py slc2ifg.cfg --scheduler distributed
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional

# Ensure src/ is in sys.path when run directly
_src_root = Path(__file__).resolve().parents[3]
if str(_src_root) not in sys.path:
    sys.path.insert(0, str(_src_root))

# libnetcdf 4.9.x probes every netCDF-4/HDF5 open for DAOS containers via the
# external `getfattr` command; if the attr package is missing, each probe
# spams "sh: 1: getfattr: not found" into the log.  Disabling HDF5's own
# xattr use is a harmless belt-and-braces default (must happen before any
# hdf5/netcdf import).
os.environ.setdefault('HDF5_USE_XATTR', 'NO')

logger = logging.getLogger('mintpy.stdproc.slc2ifg.engine')


def parse_arguments(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Sarforge engine: Dask-scheduled SLC-to-unwrapped processing.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('config_file', type=str, nargs='?', default=None,
                        help='MintPy-style configuration file')
    parser.add_argument('--tool', type=str, default=None,
                        help='Run only this tool standalone (e.g. unwrap, filter)')
    parser.add_argument('--restore', action='store_true',
                        help='Rebuild products deleted by a previous cleanup '
                             '(idempotent full re-run, nothing deleted afterwards)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Build & print the DAG plan, do not execute')
    parser.add_argument('--scheduler', choices=['threaded', 'distributed'],
                        default=None, help='Override engine.scheduler')
    parser.add_argument('--max-workers', type=int, default=None,
                        help='Override engine.max_workers')
    parser.add_argument('--gpu', choices=['auto', 'true', 'false'], default=None,
                        help='Override engine.gpu')
    parser.add_argument('--keep-intermediates', default=None,
                        help='Override engine.keep_intermediates '
                             '(none | all | tool1,tool2,...)')
    parser.add_argument('--list-tools', action='store_true',
                        help='List available tools and exit')
    parser.add_argument('--list-stages', action='store_true',
                        help='Show the effective processing chain and exit')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose logging')
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_arguments()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Route Python warnings (e.g. numpy RuntimeWarnings) through logging so
    # they get the same timestamped format instead of raw stderr lines.
    logging.captureWarnings(True)
    # numexpr prints 2-3 "Note: detected N cores" INFO lines at import.
    logging.getLogger('numexpr.utils').setLevel(logging.WARNING)

    from mintpy.stdproc.slc2ifg.engine.config import load_engine_config
    from mintpy.stdproc.slc2ifg.engine.engine import Engine
    from mintpy.stdproc.slc2ifg.engine.tool import available_tools

    if args.list_tools:
        print("Available tools:", ', '.join(available_tools()))
        return 0

    config = load_engine_config(args.config_file)

    if args.list_stages:
        from mintpy.stdproc.slc2ifg.engine.chain import format_chain, resolve_chain
        print(format_chain(resolve_chain(config.stages, config.tools)))
        return 0

    # CLI overrides
    if args.scheduler:
        config.scheduler = args.scheduler
    if args.max_workers:
        config.max_workers = args.max_workers
    if args.gpu:
        config.gpu = args.gpu
    if args.keep_intermediates:
        config.keep_intermediates = args.keep_intermediates

    engine = Engine(config)
    try:
        if args.restore:
            ok = engine.restore(dry_run=args.dry_run)
        elif args.tool:
            ok = engine.run_tool(args.tool, dry_run=args.dry_run)
        else:
            ok = engine.run(dry_run=args.dry_run)
    except Exception as e:
        logger.error("Engine failed: %s", e)
        if args.verbose:
            raise
        return 1
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
