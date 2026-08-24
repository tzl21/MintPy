############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Sep 2024                           #
############################################################
# multilook_tif merged from insarflow.slc2ifg (2026, Zhenli Tang);
# multilook_gdal removed -- multilook_tif is the single file-level
# implementation (true averaging, nodata/complex support, dual output).


import logging
import os
import warnings
from pathlib import Path

import numpy as np

# suppress numpy.RuntimeWarning message
np_logger = logging.getLogger('numpy')
np_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


def multilook_data(data, lks_y=1, lks_x=1, method='mean'):
    """Apply multilooking (spatial averaging/resampling) to a multi-dimensional array.

    Link: https://stackoverflow.com/questions/34689519

    Parameters: data     - 2D / 3D np.array in real or complex
                lks_y    - int, number of multilook in y/azimuth direction
                lks_x    - int, number of multilook in x/range direction
                method   - str, multilook method, mean, median or nearest
    Returns:    out_data - 2D / 3D np.array after multilooking in last two dimension
    """
    # check - method
    method_list = ['mean', 'median', 'nearest']
    if method not in method_list:
        raise ValueError(f'Un-supported multilook method: {method}! Available methods: {method_list}.')

    # check - number of looks: do nothing if no multilook is applied
    lks_y = int(lks_y)
    lks_x = int(lks_x)
    if lks_y * lks_x == 1:
        return data

    shape = np.array(data.shape, dtype=float)
    if len(shape) == 2:
        # prepare - crop data to the exact multiple of the multilook number
        new_shape = np.floor(shape / (lks_y, lks_x)).astype(int) * (lks_y, lks_x)
        crop_data = data[:new_shape[0], :new_shape[1]]

        if method in ['mean', 'median']:
            # approach 1: reshape to higher dimensions, then collapse the extra dimensions
            #   with desired math operation
            # a. reshape to more dimensions
            temp = crop_data.reshape(
                (new_shape[0] // lks_y, lks_y,
                 new_shape[1] // lks_x, lks_x,
                ),
            )

            # b. collapse the extra dimensions with mean / median
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if method == 'mean':
                    out_data = np.nanmean(temp, axis=(1, 3))
                elif method == 'median':
                    out_data = np.nanmedian(temp, axis=(1, 3))

        elif method == 'nearest':
            out_data = crop_data[
                int(lks_y/2)::lks_y,
                int(lks_x/2)::lks_x,
            ]

    elif len(shape) == 3:
        # prepare - crop data to the exact multiple of the multilook number
        new_shape = np.floor(shape / (1, lks_y, lks_x)).astype(int) * (1, lks_y, lks_x)
        crop_data = data[
            :new_shape[0],
            :new_shape[1],
            :new_shape[2],
        ]

        if method in ['mean', 'median']:
            # a. reshape to more dimensions
            temp = crop_data.reshape(
                (new_shape[0],
                 new_shape[1] // lks_y, lks_y,
                 new_shape[2] // lks_x, lks_x,
                ),
            )

            # b. collapse the extra dimensions with mean / median
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if method == 'mean':
                    out_data = np.nanmean(temp, axis=(2, 4))
                elif method == 'median':
                    out_data = np.nanmedian(temp, axis=(2, 4))

        elif method == 'nearest':
            out_data = crop_data[
                :,
                int(lks_y/2)::lks_y,
                int(lks_x/2)::lks_x,
            ]

    else:
        raise ValueError(f'Un-supported data dimension: {shape} --> {len(shape)}!')

    # ensure output data type
    out_data = np.array(out_data, dtype=data.dtype)

    return out_data


def multilook_tif(input_tif, output_tif=None, lks_y=1, lks_x=1, method='mean',
                  processor='isce3', box=None):
    """Apply multilooking (spatial averaging/resampling) to a GDAL-readable file.

    Parameters
    ----------
    input_tif  : str
        Path to input file.
    output_tif : str, optional
        Path to output file (if None, auto-generate as ``multilooked_*``).
    lks_y      : int
        Number of multilook in y/azimuth direction.
    lks_x      : int
        Number of multilook in x/range direction.
    method     : str
        Multilook method, mean, median or nearest.
    processor  : str
        Processor type ('isce2' for ENVI output, 'isce3' for GeoTIFF).
    box        : tuple(int), optional
        Area of interest in (x0, y0, x1, y1) in input pixel coordinates.

    Returns
    -------
    output_tif : str
        Path to output file.
    """
    from osgeo import gdal

    # Auto-generate output filename if not provided
    if output_tif is None:
        input_path = Path(input_tif)
        suffixes = input_path.suffixes
        if suffixes:
            base = input_path.name[:-len(''.join(suffixes))]
            new_name = "multilooked_" + base + ''.join(suffixes)
            output_tif = str(input_path.parent / new_name)
        else:
            output_tif = str(input_path.parent / f"multilooked_{input_path.stem}{input_path.suffix}")

    # Validate method
    method_list = ['mean', 'median', 'nearest']
    if method not in method_list:
        raise ValueError(f'Un-supported multilook method: {method}! Available methods: {method_list}.')

    lks_y = int(lks_y)
    lks_x = int(lks_x)

    # One short line per file: include the parent (pair) dir so interleaved
    # engine logs stay attributable; full paths go to DEBUG.
    src_dir = os.path.basename(os.path.dirname(input_tif))
    src = f"{src_dir}/{os.path.basename(input_tif)}" if src_dir \
        else os.path.basename(input_tif)
    logger.info("multilook %sx%s: %s -> %s",
                lks_y, lks_x, src, os.path.basename(output_tif))
    logger.debug("multilook: %s -> %s (method=%s, processor=%s)",
                 input_tif, output_tif, method, processor)

    if os.path.exists(output_tif):
        logger.info(f"Skipping existing output: {output_tif}")
        return output_tif

    # Warn if file extension does not match processor expectations
    input_ext = Path(input_tif).suffix.lower()
    if processor == 'isce2':
        expected_exts = ['.slc', '.int', '.rdr', '.full', '.vrt']
        if input_ext not in expected_exts:
            logger.warning(
                f"Processor 'isce2' expects ENVI-style files (extensions {expected_exts}), "
                f"but got '{input_ext}'. Processing will continue but may fail."
            )
    elif processor == 'isce3':
        expected_exts = ['.tif', '.tiff', '.h5', '.hdf5']
        if input_ext not in expected_exts:
            logger.warning(
                f"Processor 'isce3' expects GeoTIFF or HDF5 files (extensions {expected_exts}), "
                f"but got '{input_ext}'. Processing will continue but may fail."
            )

    try:
        # Open input dataset
        ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
        if ds is None:
            error_msg = gdal.GetLastErrorMsg()
            raise ValueError(f"Could not open input file: {input_tif}. GDAL error: {error_msg}")

        band_count = ds.RasterCount
        if band_count == 0:
            raise ValueError(f"No raster bands found in file: {input_tif}")

        # Get geotransform and projection
        geotransform = ds.GetGeoTransform()
        projection = ds.GetProjection()

        # IO box: (x0, y0, x1, y1) in input pixels; None = whole image
        if box is not None:
            box = tuple(int(i) for i in box)
            x0, y0, x1, y1 = box
            win = (x0, y0, x1 - x0, y1 - y0)
        else:
            win = None
            x0, y0 = 0, 0

        # Read all bands data (avoid np.stack copy when a single band)
        all_band_data = []
        band_no_data_values = []
        band_dtypes = []

        for band_idx in range(1, band_count + 1):
            band = ds.GetRasterBand(band_idx)
            if band is None:
                raise RuntimeError(f"Failed to get raster band {band_idx}")

            no_data_value = band.GetNoDataValue()
            data = band.ReadAsArray(*win) if win else band.ReadAsArray()
            if data is None:
                raise RuntimeError(f"Failed to read data from band {band_idx}")

            # If no_data_value exists, convert to NaN for processing
            if no_data_value is not None:
                if data.dtype not in (np.complex64, np.complex128):
                    data = data.astype(np.float32)
                data[data == no_data_value] = np.nan

            all_band_data.append(data)
            band_no_data_values.append(no_data_value)
            band_dtypes.append(data.dtype)

        ds = None

        # Apply multilooking
        if band_count == 1:
            data = all_band_data[0]
            data = multilook_data(data, lks_y, lks_x, method)
        else:
            data = np.stack(all_band_data, axis=0)
            data = multilook_data(data, lks_y, lks_x, method)

        # Update geotransform: shift origin to the box corner, then scale by looks
        new_geotransform = (
            geotransform[0] + x0 * geotransform[1] + y0 * geotransform[2],
            geotransform[1] * lks_x,
            geotransform[2],
            geotransform[3] + x0 * geotransform[4] + y0 * geotransform[5],
            geotransform[4],
            geotransform[5] * lks_y,
        )

        # Create output directory
        output_dir = os.path.dirname(output_tif)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Determine output data type
        original_dtype = band_dtypes[0]
        if original_dtype in (np.complex64, np.complex128):
            output_dtype = gdal.GDT_CFloat32
        elif original_dtype in (np.float32, np.float64):
            output_dtype = gdal.GDT_Float32
        elif original_dtype in (np.int16, np.uint16):
            output_dtype = gdal.GDT_Int16
        elif original_dtype in (np.int32, np.uint32):
            output_dtype = gdal.GDT_Int32
        elif original_dtype in (np.int8, np.uint8):
            output_dtype = gdal.GDT_Byte
        else:
            output_dtype = gdal.GDT_Float32

        # Get output dimensions
        if band_count == 1:
            height, width = data.shape
        else:
            height, width = data.shape[1], data.shape[2]

        # Select driver based on processor
        if processor == 'isce2':
            driver = gdal.GetDriverByName('ENVI')
            options = []
        else:  # isce3
            driver = gdal.GetDriverByName('GTiff')
            options = ['COMPRESS=LZW', 'TILED=YES']

        out_ds = driver.Create(
            output_tif,
            width,
            height,
            band_count,
            output_dtype,
            options=options
        )

        if out_ds is None:
            error_msg = gdal.GetLastErrorMsg()
            raise RuntimeError(f"Failed to create output file: {output_tif}. GDAL error: {error_msg}")

        out_ds.SetGeoTransform(new_geotransform)
        if projection:
            out_ds.SetProjection(projection)

        # Write data for each band
        for band_idx in range(band_count):
            out_band = out_ds.GetRasterBand(band_idx + 1)

            if band_count == 1:
                band_data = data
            else:
                band_data = data[band_idx]

            # Convert back to original data type if no no-data
            if band_no_data_values[band_idx] is None:
                band_data = band_data.astype(band_dtypes[band_idx])

            if band_no_data_values[band_idx] is not None:
                output_data = np.where(np.isnan(band_data), band_no_data_values[band_idx], band_data)
                out_band.SetNoDataValue(band_no_data_values[band_idx])
            else:
                output_data = band_data

            out_band.WriteArray(output_data)
            out_band.FlushCache()

        out_ds = None
        logger.debug(f"Successfully created: {output_tif}")

        if processor == 'isce2':
            ext = Path(output_tif).suffix.lower()
            family = 'intimage' if ext == '.int' else 'image'
            from mintpy.stdproc.slc2ifg.utils.slc2ifg_utils import create_xml_for_binary
            create_xml_for_binary(output_tif, family=family,
                                  description=f'Multilooked {lks_y}x{lks_x}',
                                  alooks=lks_y, rlooks=lks_x)

        return output_tif

    except Exception as e:
        logger.error(f"Error processing {input_tif}: {str(e)}")
        raise
