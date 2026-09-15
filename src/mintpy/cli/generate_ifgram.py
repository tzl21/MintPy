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

from mintpy.stdproc.generate_ifgram import generate_ifgram

EXAMPLE = """example:
  generate_ifgram.py --processor isce3 --pairs-file ifgram_list.txt --slc-dir ./slc --output-dir ./ifgrams
  generate_ifgram.py --processor isce2 --pairs-file pairs.txt --slc-dir ./slc --output-dir ./ifgs --only-vrt
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Generate VRT interferograms and materialise them'
    """Parse command line arguments (compatible with engine invocation)."""
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE, subparsers=subparsers)
    parser.add_argument('--processor', required=True, choices=['isce2', 'isce3'],
                        help="Processor type: 'isce2' or 'isce3'")
    parser.add_argument('--pairs-file', required=True,
                        help="Path to interferometric pairs file (e.g., ifgram_list.txt)")
    parser.add_argument('--slc-dir', nargs='+', required=True,
                        help="One or more directories containing SLC files (wildcards allowed)")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for interferogram files")
    parser.add_argument('--only-vrt', action='store_true',
                        help="Only create VRT interferograms, do not materialise them")
    parser.add_argument('--max-workers', type=int, default=None,
                        help="Maximum number of parallel workers (default: auto)")
    parser.add_argument('--bbox', type=float, nargs=4, metavar=('W', 'S', 'E', 'N'),
                        default=None,
                        help="Read-time crop: materialise each interferogram "
                             "only over this WSEN bbox (EPSG:4326, degrees); "
                             "no cropped SLC files are written")
    parser.add_argument('--bbox-buffer', type=float, default=0.0,
                        help="Buffer in degrees around --bbox (default: 0.0)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose logging")
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    setup_logging(verbose=getattr(inps, "verbose", False))

    logger.info("Processor: %s", inps.processor)

    ok = generate_ifgram(
        pairs_file=inps.pairs_file,
        slc_dir_patterns=inps.slc_dir,
        output_dir=inps.output_dir,
        processor=inps.processor,
        slc_pattern=None,      # inferred from the input files
        subdataset=None,       # auto-detected (/data/[VV,VH,HH], preferring VV)
        no_verify=False,       # always verify SLC dimensions
        only_vrt=inps.only_vrt,
        max_workers=inps.max_workers,
        verbose=inps.verbose,
        bbox=tuple(inps.bbox) if inps.bbox else None,
        bbox_buffer=inps.bbox_buffer,
    )
    return 0 if ok else 1


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
