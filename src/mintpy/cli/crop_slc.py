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

EXAMPLE = """example:
  # ISCE3 (geocoded) input
  crop_slc.py --processor isce3 --input-dir ./slc --output-dir ./cropped --bbox 102.8 27.3 103.3 27.6
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Crop SLCs to a geographic bbox (isce2 radar / isce3 geocoded)'
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE,
        subparsers=subparsers)

    parser.add_argument('--processor', choices=['isce2', 'isce3'], required=True,
                        help="Processor type: 'isce2' (radar coordinates) or 'isce3' (geocoded)")
    parser.add_argument('--input-dir', nargs='+', required=True,
                        help='Input directory/pattern(s) containing SLC files')
    parser.add_argument('--output-dir', required=True,
                        help='Directory for the cropped SLCs (yyyymmdd.slc.tif)')
    parser.add_argument('--bbox', '--wsen', dest='bbox', type=float, nargs=4,
                        metavar=('W', 'S', 'E', 'N'), required=True,
                        help='Crop bounds in WSEN, EPSG:4326 degrees')
    parser.add_argument('--pattern', default=None,
                        help='Input SLC glob (default: processor raw pattern)')
    parser.add_argument('--buffer', type=float, default=0.0,
                        help='Buffer in degrees around the bbox (default: %(default)s)')
    parser.add_argument('--max-workers', type=int, default=1,
                        help='Number of parallel crop workers (default: %(default)s)')
    parser.add_argument('--by-burst', action='store_true',
                        help='Per-burst output subdirectories')
    parser.add_argument('--no-burst-dirs', action='store_true',
                        help='Disable per-burst output subdirectories')
    parser.add_argument('--file-list', help='Explicit input file list (overrides --input-dir)')
    parser.add_argument('--no-skip-existing', action='store_true',
                        help='Recompute existing outputs')
    parser.add_argument('--dest-epsg', type=int, default=None,
                        help='Reproject geocoded outputs to this EPSG')
    parser.add_argument('--fill-nan', action='store_true', help='Replace NaN with 0')
    parser.add_argument('--compress-level', type=int, default=6, choices=range(0, 10),
                        help='Output GeoTIFF DEFLATE compression level (default: %(default)s)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Only report what would be done')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.processor == 'isce2':
        parser.error('bbox cropping requires geocoded (isce3) SLCs; isce2 '
                     'radar-coordinate SLCs have no georeferencing to crop by bbox')
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)

    from mintpy.stdproc.crop_slc import crop_slc
    from mintpy.stdproc.utils.log_utils import setup_logging

    setup_logging(verbose=inps.verbose)

    return crop_slc(
        input_dir=inps.input_dir,
        output_dir=inps.output_dir,
        bbox=tuple(inps.bbox),
        processor=inps.processor,
        pattern=inps.pattern,
        buffer=inps.buffer,
        by_burst=inps.by_burst,
        file_list=inps.file_list,
        subdataset=None,   # auto-detected (/data/[VV,VH,HH], preferring VV)
        workers=inps.max_workers,
        no_skip_existing=inps.no_skip_existing,
        no_burst_dirs=inps.no_burst_dirs,
        dest_epsg=inps.dest_epsg,
        fill_nan=inps.fill_nan,
        compress_level=inps.compress_level,
        dry_run=inps.dry_run,
    )


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
