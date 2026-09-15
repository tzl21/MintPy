#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Zhang Yunjun, Aug 2022        #
############################################################


import logging
import os
import sys
from pathlib import Path

from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

##################################################################################################
EXAMPLE = """example:
  multilook.py velocity.h5 -r 15 -a 15
  multilook.py srtm30m.dem -x 10 -y 10 -o srtm300m.dem
  multilook.py filt_fine.int -r 2 -a 2 -o filt_fine_mli.int

  # support GDAL VRT file from ISCE2 as input
  multilook.py lat.rdr.full.vrt lon.rdr.full.vrt -x 9 -y 3

  # --off-file option: use as reference to adjust for the irregular size from isce2 dense offsets
  multilook.py lat.rdr.full.vrt -x 128 -y 64 -o lat.rdr.mli --off-file dense_offsets.bil -m nearest

  # slc2ifg batch mode: multilook every interferogram in a directory
  multilook.py --processor isce3 --input-dir ./ifgrams --pattern '**/*.int.tif' \\
      --output-dir ./mli --lks-y 4 --lks-x 4

  # slc2ifg batch mode: multilook the isce2 geometry files as well
  multilook.py --processor isce2 --input-dir ./ifgrams --pattern '**/*.int' \\
      --output-dir ./mli --lks-y 4 --lks-x 4 --geom-dir ./geom_reference
"""


def create_parser(subparsers=None):
    synopsis = 'Multilook the input file'
    epilog = EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    # basic (MintPy single-file mode)
    parser.add_argument('file', nargs='*', help='File(s) to multilook')
    parser.add_argument('-r', '--range', '-x', '--lks-x', dest='lks_x', type=int, default=1,
                        help='number of multilooking in range /x direction (default: %(default)s).')
    parser.add_argument('-a', '--azimuth', '-y', '--lks-y', dest='lks_y', type=int, default=1,
                        help='number of multilooking in azimuth/y direction (default: %(default)s).')
    parser.add_argument('-o', '--outfile',
                        help='Output file name. Disabled when more than 1 input files')
    parser.add_argument('-m', '--method', dest='method', type=str, default='mean',
                        choices=['mean', 'median', 'nearest'],
                        help='downsampling method (default: %(default)s) \n'
                             'e.g. nearest for geometry, average for observations')

    # offset
    ampcor = parser.add_argument_group('Ampcor options', 'Ampcor options for dense offsets to account for the extra margin')
    ampcor.add_argument('--search', '--search-win', dest='search_win', type=int, nargs=2, metavar=('X', 'Y'),
                        help='Ampcor (half) search window in (width, height) in pixel, e.g. 20 x 20.')
    ampcor.add_argument('--xcorr', '--xcorr-win', dest='xcorr_win', type=int, nargs=2, metavar=('X', 'Y'),
                        help='Ampcor cross-correlation window in (width, height) in pixel e.g. 32 x 32.')
    ampcor.add_argument('--margin', dest='margin', type=int, default=0,
                        help='Ampcor margin offset (default: %(default)s).')
    ampcor.add_argument('--off-file', dest='off_file', type=str,
                        help='Ampcor offset file as reference for the size.')

    # slc2ifg batch mode
    batch = parser.add_argument_group('slc2ifg batch mode',
                                      'Directory-driven multilooking for the slc2ifg pipeline')
    batch.add_argument('--processor', choices=['isce2', 'isce3'],
                       help="Processor type (batch mode; default: isce3)")
    batch.add_argument('--input-dir', help='Directory containing files to multilook (enables batch mode)')
    batch.add_argument('--pattern', default=None,
                       help='File glob for batch mode (default: processor pattern, e.g. **/*.int[.tif])')
    batch.add_argument('--output-dir', help='Output directory for batch mode')
    batch.add_argument('--max-workers', type=int, default=4,
                       help='Number of parallel workers in batch mode (default: %(default)s)')
    batch.add_argument('--geom-dir', help='Geometry directory (.full files) to multilook as well')
    batch.add_argument('--output-geom-dir', help='Output directory for the multilooked geometry')
    batch.add_argument('--ref-file', help='Reference file for geometry dimension matching')
    batch.add_argument('--geom-only', action='store_true',
                       help='Only multilook the geometry files, skip the interferograms')
    batch.add_argument('--no-skip-existing', action='store_true',
                       help='Recompute existing outputs in batch mode')
    batch.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # batch mode: --input-dir drives the slc2ifg directory workflow
    inps.batch = bool(inps.input_dir)
    if inps.batch:
        if inps.processor is None:
            inps.processor = 'isce3'
        if inps.pattern is None:
            inps.pattern = '**/*.int.tif' if inps.processor == 'isce3' else '**/*.int'
        if inps.lks_x == 1 and inps.lks_y == 1:
            raise SystemExit('ERROR: no multilooking specified: lks_x/y=1!')
        return inps

    # MintPy single-file mode
    from mintpy.utils import utils1 as ut
    inps.file = ut.get_file_list(inps.file)
    if inps.lks_x == 1 and inps.lks_y == 1:
        raise SystemExit('ERROR: no multilooking specified: lks_x/y=1!')
    if len(inps.file) > 1 and inps.outfile:
        inps.outfile = None
        print('more than one file is input, disable custom output filename.')
    return inps


##################################################################################################
def _run_batch(inps):
    """Directory-driven multilooking for the slc2ifg pipeline."""
    from mintpy.stdproc.multilook import process_batch_files, process_geometry_files

    if inps.geom_dir:
        if not os.path.isdir(inps.geom_dir):
            raise FileNotFoundError(f'Geometry directory not found: {inps.geom_dir}')
        if inps.ref_file:
            reference_file = inps.ref_file
        else:
            input_files = sorted(Path(inps.input_dir).glob(inps.pattern))
            if not input_files:
                raise FileNotFoundError(
                    f'No files found in {inps.input_dir} matching pattern {inps.pattern}')
            reference_file = str(input_files[0])
        geom_output_dir = process_geometry_files(
            geom_dir=inps.geom_dir, input_file=reference_file,
            lks_y=inps.lks_y, lks_x=inps.lks_x,
            output_geom_dir=inps.output_geom_dir, processor=inps.processor)
        logger.info('geometry multilook output: %s', geom_output_dir)

    if inps.geom_only:
        return 0
    if not inps.output_dir:
        raise SystemExit('--output-dir is required in batch mode')
    process_batch_files(inps)
    return 0


def main(iargs=None):
    inps = cmd_line_parse(iargs)

    from mintpy.stdproc.utils.log_utils import setup_logging
    setup_logging(verbose=getattr(inps, 'verbose', False))

    if inps.batch:
        return _run_batch(inps)

    from mintpy.multilook import multilook_file
    for infile in inps.file:
        multilook_file(
            infile,
            lks_y=inps.lks_y,
            lks_x=inps.lks_x,
            outfile=inps.outfile,
            method=inps.method,
            search_win=inps.search_win,
            xcorr_win=inps.xcorr_win,
            margin=inps.margin,
            off_file=inps.off_file,
        )
    print('Done.')
    return 0


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
