#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Command line interface for the MintPy slc2ifg pipeline (mintpy slc2ifg).

This is the single entry point for the SLC -> interferogram pre-processing
workflow.  The execution backend is chosen by ``slc2ifg.engine`` in the config
(``none`` = MintPy's built-in executor, ``engine`` = the Dask-scheduled
``mintpy.stdproc.engine``, ``auto`` = engine when importable else built-in).
"""

import logging
import os
import sys

import mintpy
from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

NOTE = """
example:
  mintpy slc2ifg slc2ifg.cfg
  mintpy slc2ifg slc2ifg.cfg --dry-run
  mintpy slc2ifg slc2ifg.cfg --tool unwrap
  mintpy slc2ifg --list-tools
  mintpy slc2ifg --generate-template
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'SLC-to-interferogram pre-processing (engine or built-in executor)'
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=NOTE,
        subparsers=subparsers)

    parser.add_argument('cfg_file', nargs='?', default=None,
                        help='MintPy-style config file (flat key=value, no section header).')
    parser.add_argument('--template', dest='template', default=None,
                        help='Template string / dict passed programmatically (API use).')

    engine = parser.add_argument_group('engine backend options',
                                       'Only meaningful with slc2ifg.engine = engine')
    engine.add_argument('--dry-run', action='store_true',
                        help='Build and print the DAG plan, do not execute')
    engine.add_argument('--tool', help='Run only this stage standalone')
    engine.add_argument('--restore', action='store_true',
                        help='Rebuild products deleted by a previous cleanup')
    engine.add_argument('--scheduler', choices=['threaded', 'distributed'],
                        help='Override engine.scheduler')
    engine.add_argument('--max-workers', type=int,
                        help='Override engine.max_workers')
    engine.add_argument('--gpu', choices=['auto', 'true', 'false'],
                        help='Override engine.gpu')
    engine.add_argument('--keep-intermediates',
                        help='Override engine.keep_intermediates (none | all | tool1,tool2,...)')

    parser.add_argument('--list-tools', action='store_true',
                        help='List the available processing stages and exit.')
    parser.add_argument('--list-stages', action='store_true',
                        help='Show the effective processing chain and exit.')
    parser.add_argument('--generate-template', action='store_true',
                        help='Print the slc2ifg default template and exit.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print verbose output messages')
    parser.add_argument('--version', action='store_true',
                        help='Print the MintPy version and exit.')
    return parser


def _template_file():
    return os.path.join(os.path.dirname(mintpy.__file__),
                        'stdproc/template/slc2ifg.cfg')


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if inps.version:
        print(mintpy.version.version_description)
        raise SystemExit(0)

    if inps.generate_template:
        with open(_template_file()) as f:
            print(f.read())
        raise SystemExit(0)

    if inps.list_tools:
        from mintpy.stdproc.engine.chain import DEFAULT_CHAIN
        print('available stages:', ', '.join(s.name for s in DEFAULT_CHAIN))
        raise SystemExit(0)

    if not inps.cfg_file and not inps.template:
        raise SystemExit(
            'ERROR: no config file given!\n'
            '   mintpy slc2ifg <config.cfg>\n'
            '   mintpy slc2ifg --generate-template   # print the template')

    return inps


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    if inps.list_stages:
        from mintpy.stdproc.engine.chain import format_chain, resolve_chain
        from mintpy.stdproc.engine.config import load_engine_config
        cfg = load_engine_config(str(inps.cfg_file))
        print(format_chain(resolve_chain(cfg.stages, cfg.tools)))
        return 0

    from mintpy import slc2ifg
    slc2ifg.run_slc2ifg(cfg_file=inps.cfg_file,
                        template=None if inps.cfg_file else inps.template,
                        overrides=vars(inps))
    return 0


if __name__ == '__main__':
    sys.exit(main())
