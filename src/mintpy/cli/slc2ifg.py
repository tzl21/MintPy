#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Command line interface for the MintPy slc2ifg pipeline (mintpy slc2ifg)."""

import argparse

import mintpy

NOTE = """
example:
  mintpy slc2ifg slc2ifg.cfg
  mintpy slc2ifg slc2ifg.cfg --list-tools
  # without a config file: print the slc2ifg default template and exit
  mintpy slc2ifg --generate-template
"""


def create_parser(subparsers=None):
    """Create the argparse parser for the 'mintpy slc2ifg' subcommand."""
    if subparsers is not None:
        parser = subparsers.add_parser(
            'slc2ifg',
            description='SLC-to-interferogram pre-processing (engine or basic).',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=NOTE,
        )
    else:
        parser = argparse.ArgumentParser(
            prog='slc2ifg',
            description='SLC-to-interferogram pre-processing (engine or basic).',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog=NOTE,
        )
    parser.add_argument('cfg_file', nargs='?', default=None,
                        help='MintPy-style config file (flat key=value, '
                             'no section header).')
    parser.add_argument('--template', dest='template', default=None,
                        help='Template string / dict passed programmatically '
                             '(for API use).')
    parser.add_argument('--list-tools', action='store_true',
                        help='List the available processing stages and exit.')
    parser.add_argument('--generate-template', action='store_true',
                        help='Print the slc2ifg default template and exit.')
    parser.add_argument('-v', '--version', action='store_true',
                        help='Print the MintPy version and exit.')
    return parser


def parse_arguments(iargs=None):
    """Parse command line arguments (list of strings or sys.argv[1:])."""
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    if inps.version:
        print(mintpy.version.version_description)
        raise SystemExit(0)

    if inps.generate_template:
        import os
        tfile = os.path.join(os.path.dirname(mintpy.__file__),
                             'stdproc/template/slc2ifg.cfg')
        with open(tfile) as f:
            print(f.read())
        raise SystemExit(0)

    if inps.list_tools:
        print('available stages: crop_slc, ifgram_list, generate_ifgram, '
              'stitch, multilook, filter, phsig_coh, complex_coh, unwrap, '
              'atmosphere')
        raise SystemExit(0)

    if not inps.cfg_file and not inps.template:
        raise SystemExit(
            'ERROR: no config file given!\n'
            '   mintpy slc2ifg <config.cfg>\n'
            '   mintpy slc2ifg --generate-template   # print the template')

    return inps


def main(iargs=None):
    """Run the slc2ifg pipeline (CLI entry)."""
    from mintpy import slc2ifg

    inps = parse_arguments(iargs)
    slc2ifg.run_slc2ifg(cfg_file=inps.cfg_file, template=inps.template)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
