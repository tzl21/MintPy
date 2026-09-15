#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################

import logging
import sys

from mintpy.stdproc.utils.log_utils import setup_logging
from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

from mintpy.stdproc.stitch import stitch_all

EXAMPLE = """example:
  stitch.py --processor isce3 --burst-dir ./ifgrams --output-dir ./stitched
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Stitch per-burst interferograms and coherence maps'
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE, subparsers=subparsers)
    parser.add_argument('--processor', type=str, choices=['isce2', 'isce3'],
                        default='isce3',
                        help="Processor type (informational: the stitched output "
                             "is GeoTIFF for both; default: %(default)s)")
    parser.add_argument('--burst-dir', type=str, required=True, nargs='+',
                        help="Root directories containing per-burst results")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory for stitched products")
    parser.add_argument('--file-types', type=str, nargs='+',
                        default=['.int.tif', '.cpx.coh.tif'],
                        help="File extensions to stitch")
    parser.add_argument('--out-bounds', type=float, nargs=4,
                        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'), default=None,
                        help="Final crop bounds in EPSG:4326. "
                             "If not set, the full union extent of all bursts is stitched.")
    parser.add_argument('--output-prefix', type=str, default='',
                        help="Prefix for output subdirectories")
    parser.add_argument('--max-workers', type=int, default=1,
                        help="Number of parallel stitch workers")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing output files")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose output")
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    setup_logging(verbose=getattr(inps, "verbose", False))
    return stitch_all(inps)


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
