#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Antonio Valentino, Forrest Williams, Aug 2022    #
############################################################


import sys

from mintpy.utils.arg_utils import create_argument_parser

EXAMPLE = """example:
  save_gdal.py geo/geo_velocity.h5
  save_gdal.py geo/geo_timeseries_ERA5_demErr.h5  -d 20200505_20200517 --of ENVI
  save_gdal.py geo/geo_ifgramStack.h5 -d unwrapPhase-20101120_20110220 --of ISCE
  save_gdal.py geo/geo_ifgramStack.h5 -d   coherence-20101120_20110220 --of ISCE
  save_gdal.py geo_20230225.slc
  save_gdal.py geo/timeseries.h5 -d 20230217 -m maskTempCoh.h5
  save_gdal.py geo/timeseries.h5 -d 20230217 --zero-mask
"""


def create_parser(subparsers=None):
    synopsis = 'Generate GDAL raster from MintPy h5 file.'
    epilog = EXAMPLE
    name = __name__.split('.')[-1]
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=epilog, subparsers=subparsers)

    parser.add_argument('file', help='file to be converted, in geo coordinate.')
    parser.add_argument('-d', '--dset', '--dataset', dest='dset',
                        help='date of timeseries, or date12 of interferograms to be converted')
    parser.add_argument('-o', '--output', dest='outfile',
                        help='output file base name. Extension is fixed by GDAL driver')
    parser.add_argument('-m','--mask', dest='mask_file', metavar='FILE',
                        help='mask file (default: %(default)s).')
    parser.add_argument('--zm','--zero-mask', dest='zero_mask', action='store_true',
                        help='Mask pixels with zero value (default: %(default)s)..')
    parser.add_argument('--of', '--out-format', '--output-format', dest='out_format', default='GTiff',
                        help='file format as defined by GDAL driver name, e.g. GTiff, ENVI (default: %(default)s).\n'
                             'GDAL driver names can be found at https://gdal.org/drivers/raster/index.html')
    parser.add_argument('--geo', dest='geo', choices=['auto', 'yes', 'no'], default='auto',
                        help='write georeferencing (geotransform + projection):\n'
                             'auto - write it when the input metadata has a complete geotransform (default)\n'
                             'yes  - require it, error out when the input is not geocoded\n'
                             'no   - never write it (radar-coordinate / plain raster)')
    parser.add_argument('--compress', dest='compress', default=None,
                        help='GDAL COMPRESS creation option for GTiff, e.g. LZW, DEFLATE (default: %(default)s)')
    return parser


def cmd_line_parse(iargs=None):
    # parse
    parser = create_parser()
    inps = parser.parse_args(args=iargs)

    # import
    from mintpy.utils import readfile

    # check: input file coordinate system
    atr = readfile.read_attribute(inps.file)
    is_geocoded = 'Y_FIRST' in atr.keys() and 'X_FIRST' in atr.keys()

    # --geo is translated to the write_gdal() geo argument
    if inps.geo == 'yes' and not is_geocoded:
        raise ValueError(f'ERROR: Input file ({inps.file}) is not geocoded!')
    elif inps.geo == 'no':
        inps.geo = False
    else:
        inps.geo = None

    return inps


##############################################################################
def main(iargs=None):
    # parse
    inps = cmd_line_parse(iargs)

    # import
    from mintpy.save_gdal import save_gdal

    # run
    save_gdal(inps)


##############################################################################
if __name__ == "__main__":
    main(sys.argv[1:])
