############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Forrest Williams, Zhang Yunjun, Jun 2020         #
############################################################


import os
import warnings

import numpy as np
from osgeo import gdal, osr

gdal.UseExceptions()

from mintpy.utils import plot as pp, readfile, utils0 as ut

# link: https://gdal.org/drivers/raster/index.html
GDAL_DRIVER2EXT = {
    'GTiff' : '.tif',
    'ENVI'  : '',
    'GMT'   : '.grd',
    'GRIB'  : '.grb',
    'JPEG'  : '.jpg',
    'PNG'   : '.png',
}


##############################################################################
def _is_missing(value):
    """Return True when a metadata value is absent/empty (incl. the string 'None')."""
    return value is None or str(value).strip().lower() in ('', 'none')


def resolve_geo(meta, geo=None):
    """Decide whether georeferencing can/should be written for the given metadata.

    Parameters: meta - dict, metadata with optional X/Y_FIRST/STEP and EPSG/UTM_ZONE
                geo  - None / True / False
                       None  - auto: write geo when a complete geotransform is given
                       True  - require geo (raise ValueError when incomplete)
                       False - never write geo (radar-coordinate products)
    Returns:    write_geo - bool, whether to write the geotransform / projection
                epsg      - int or None, EPSG code to use (only when write_geo)
    """
    gt_keys = ('X_FIRST', 'X_STEP', 'Y_FIRST', 'Y_STEP')
    has_gt = all(not _is_missing(meta.get(k)) for k in gt_keys)
    has_epsg = not _is_missing(meta.get('EPSG'))
    has_utm = not _is_missing(meta.get('UTM_ZONE'))
    has_crs = has_epsg or has_utm

    if geo is False:
        return False, None

    if geo is True:
        if not has_gt:
            raise ValueError(f'geo=True but incomplete geotransform in metadata: '
                             f'{[k for k in gt_keys if _is_missing(meta.get(k))]}')
        if not has_crs:
            raise ValueError('geo=True but no EPSG / UTM_ZONE in metadata')
        write_geo = True

    else:   # None - auto
        if not has_gt:
            # plain/raster-coordinate product: no georeferencing, no warning
            return False, None
        # keep the legacy behaviour for a complete transform without any CRS
        write_geo = True

    if has_epsg:
        epsg = int(float(meta['EPSG']))
    elif has_utm:
        epsg = int(ut.utm_zone2epsg_code(meta['UTM_ZONE']))
    else:
        epsg = 4326
        warnings.warn('No EPSG or UTM_ZONE metadata found! '
                      'Assume EPSG = 4326 (WGS84) and continue.')

    return write_geo, epsg


def _gdal_creation_options(out_fmt, compress=None, tiled=False):
    """Build the GDAL creation-option list for the given driver."""
    if out_fmt != 'GTiff':
        return []

    options = []
    if compress:
        options.append(f'COMPRESS={compress}')
    if tiled:
        options.append('TILED=YES')
    options.append('BIGTIFF=IF_SAFER')
    return options


##############################################################################
def write_gdal(data, meta, out_file, out_fmt='GTiff', geo=None, compress=None,
               tiled=False, nodata=None, atomic=False, gdal_metadata=None):
    """Write 2D matrix into GDAL raster.

    Parameters: data      - 2D np.ndarray
                meta      - dict, metadata used to calculate the transform / epsg info:
                            X/Y_FIRST/STEP, EPSG / UTM_ZONE
                out_file  - str, output file path
                out_fmt   - str, output GDAL file format (driver name)
                            https://gdal.org/drivers/raster/index.html
                geo       - None / True / False, georeferencing control:
                            None  - auto: write geo only when a complete geotransform
                                    is provided; otherwise write a plain raster
                            True  - require geo, raise ValueError when incomplete
                            False - never write geotransform/projection (radar products)
                compress  - str or None, GTiff COMPRESS creation option, e.g. 'LZW'
                tiled     - bool, GTiff TILED=YES creation option
                nodata    - float or None, band no-data value
                atomic    - bool, write to '<out_file>.tmp' then os.replace() on success
                gdal_metadata - dict or None, extra items written via SetMetadataItem()
    Returns:    out_file  - str, output file path
    """

    # decide the georeferencing to write (and validate it) BEFORE creating files
    write_geo, epsg = resolve_geo(meta, geo=geo)

    if write_geo:
        # link: https://gdal.org/tutorials/geotransforms_tut.html
        transform = (
            float(meta['X_FIRST']), float(meta['X_STEP']), 0,
            float(meta['Y_FIRST']), 0, float(meta['Y_STEP']),
        )

    # convert boolean to uint8, as GDAL does not have a direct analogue to boolean
    if data.dtype == 'bool':
        print('convert data from boolean to uint8, as GDAL does not support boolean')
        data = np.array(data, dtype=np.uint8)

    # write to a temporary path first when atomic is requested
    out_dir = os.path.dirname(os.path.abspath(out_file))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    write_file = out_file + '.tmp' if atomic else out_file

    # write file
    driver = gdal.GetDriverByName(out_fmt)
    print(f'initiate GDAL driver: {driver.LongName}')

    rows, cols = data.shape
    dtype = readfile.DATA_TYPE_NUMPY2GDAL[str(data.dtype)]
    print('create raster band:')
    print(f'  raster row / column number: {rows}, {cols}')
    print(f'  raster data type: {dtype} ({data.dtype})')
    options = _gdal_creation_options(out_fmt, compress=compress, tiled=tiled)
    raster = driver.Create(
        write_file,
        xsize=cols,
        ysize=rows,
        bands=1,
        eType=dtype,
        options=options,
    )

    try:
        if write_geo:
            print(f'set transform info: {transform}')
            raster.SetGeoTransform(transform)

            print(f'set projection as: EPSG {epsg}')
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(epsg)
            raster.SetProjection(srs.ExportToWkt())
        else:
            print('no georeferencing: writing a plain raster (no transform/projection)')

        if gdal_metadata:
            for key, value in gdal_metadata.items():
                if value is not None:
                    raster.SetMetadataItem(str(key), str(value))

        print('write data to raster band')
        band = raster.GetRasterBand(1)
        band.WriteArray(data)
        if nodata is not None:
            band.SetNoDataValue(float(nodata))

        band.FlushCache()
        raster = None
    except BaseException:
        raster = None
        for stray in (write_file, write_file + '.hdr', write_file + '.aux.xml'):
            if os.path.isfile(stray):
                os.remove(stray)
        raise

    if atomic:
        # rename the data file and the driver's companion .hdr, if any
        os.replace(write_file, out_file)
        if os.path.isfile(write_file + '.hdr'):
            os.replace(write_file + '.hdr', out_file + '.hdr')

    print(f'finished writing to {out_file}')

    return out_file


def save_gdal(inps):

    ## read data
    ftype = readfile.read_attribute(inps.file)['FILE_TYPE']

    # grab ref_date from dset
    if ftype == 'timeseries' and inps.dset and '_' in inps.dset:
        inps.ref_date, inps.dset = inps.dset.split('_')
    else:
        inps.ref_date = None

    ds_name = inps.dset if inps.dset else 'data'
    print(f'read {ds_name} from file: {inps.file}')
    data, meta = readfile.read(inps.file, datasetName=inps.dset)

    if ftype == 'timeseries' and inps.ref_date:
        print(f'read {inps.ref_date} from file: {inps.file}')
        data -= readfile.read(inps.file, datasetName=inps.ref_date)[0]

    # mask
    if inps.mask_file:
        mask = pp.read_mask(
            inps.file,
            mask_file=inps.mask_file,
            datasetName=inps.dset)[0]
        if mask is not None:
            print(f'masking out pixels with zero value in file: {inps.mask_file}')
            data[mask == 0] = np.nan
        del mask
    if inps.zero_mask:
        print('masking out pixels with zero value')
        data[data == 0] = np.nan

    ## write data to file
    # output file name
    if not inps.outfile:
        fbase = pp.auto_figure_title(inps.file, inps.dset, vars(inps))
        fext = GDAL_DRIVER2EXT.get(inps.out_format, '')
        inps.outfile = fbase + fext
    inps.outfile = os.path.abspath(inps.outfile)

    write_gdal(data, meta, out_file=inps.outfile, out_fmt=inps.out_format,
               geo=getattr(inps, 'geo', None),
               compress=getattr(inps, 'compress', None))

    return
