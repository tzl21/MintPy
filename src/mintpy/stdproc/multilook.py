############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhang Yunjun, Sep 2024; Zhenli Tang, 2026        #
############################################################
"""
Multilooking (spatial averaging/resampling) for MintPy and the slc2ifg
pipeline — single merged implementation:

- :func:`multilook_data`  — pure-numpy multilooking of an in-memory array
  (2D/3D, real or complex; used by ``mintpy/multilook.py`` and 8+ modules).
- :func:`multilook_tif`   — file-level multilooking of a GDAL-readable
  raster (mean/median/nearest, nodata/complex support, GeoTIFF/ENVI output,
  idempotent).  Merged from insarflow.slc2ifg (2026); ``multilook_gdal``
  was removed — this is the single file-level implementation.
- CLI / batch driver — single-file, directory-pattern batch (thread
  parallel), and geometry-file (``.full``) multilooking with canonical
  ``output_dir/{date1}_{date2}/{variant}.int[.tif]`` naming (see
  ``stdproc.utils.naming``).

Importing this module does **not** import osgeo / h5py (lazy inside the
functions) — ``mintpy/multilook.py`` and other core modules import it
without pulling GDAL.
"""

import argparse
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .utils.naming import (
    int_ext,
    is_date_pair_dir,
    next_variant,
    variant_of,
)

# suppress numpy.RuntimeWarning message
np_logger = logging.getLogger('numpy')
np_logger.setLevel(logging.WARNING)

# numpy warns "Mean of empty slice" / "All-NaN slice encountered" when a
# multilook block is entirely NaN (normal at scene edges).  Suppress these
# process-wide: a global filter is thread-safe, unlike the local
# warnings.catch_warnings() used inside the workers (which races under the
# threaded engine scheduler and lets the warning leak into the log).
warnings.filterwarnings("ignore", message="Mean of empty slice",
                        category=RuntimeWarning)
warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                        category=RuntimeWarning)

logger = logging.getLogger(__name__)


def _gdal():
    """Lazily import GDAL (keeps ``import mintpy.stdproc.multilook`` osgeo-free)."""
    from osgeo import gdal
    gdal.UseExceptions()
    return gdal


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


def _output_matches_looks(input_tif, output_tif, lks_y, lks_x):
    """True when the existing output's size equals the expected multilooked
    size for the given look factors (cheap metadata-only check)."""
    gdal = _gdal()
    try:
        ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
        if ds is None:
            return False
        in_rows, in_cols = ds.RasterYSize, ds.RasterXSize
        ds = None
        ods = gdal.Open(output_tif, gdal.GA_ReadOnly)
        if ods is None:
            return False
        out_rows, out_cols = ods.RasterYSize, ods.RasterXSize
        ods = None
    except Exception:
        return False
    return (out_rows == in_rows // int(lks_y)
            and out_cols == in_cols // int(lks_x))


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
    gdal = _gdal()

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
        # A re-run with different look factors must not silently keep a
        # stale product of the old size: validate before skipping.
        if _output_matches_looks(input_tif, output_tif, lks_y, lks_x):
            logger.info(f"Skipping existing output: {output_tif}")
            return output_tif
        logger.warning(
            "Existing output %s does not match the requested multilook "
            "%dx%d — regenerating", output_tif, lks_y, lks_x)

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
            from .utils.slc2ifg_utils import create_xml_for_binary
            create_xml_for_binary(output_tif, family=family,
                                  description=f'Multilooked {lks_y}x{lks_x}',
                                  alooks=lks_y, rlooks=lks_x)

        return output_tif

    except Exception as e:
        logger.error(f"Error processing {input_tif}: {str(e)}")
        raise


# ------------------------------------------------------------------------
# CLI / batch driver (merged from insarflow.slc2ifg, 2026)
# ------------------------------------------------------------------------
def get_file_info(input_tif):
    """Get detailed file information for debugging"""
    gdal = _gdal()
    try:
        ds = gdal.Open(input_tif, gdal.GA_ReadOnly)
        if ds is None:
            return f"Cannot open file: {gdal.GetLastErrorMsg()}"

        info = []
        info.append(f"File: {os.path.basename(input_tif)}")
        info.append(f"  Size: {ds.RasterXSize} x {ds.RasterYSize}")
        info.append(f"  Band count: {ds.RasterCount}")
        info.append(f"  Projection: {ds.GetProjection()[:50]}..." if ds.GetProjection() else "  Projection: None")

        geotransform = ds.GetGeoTransform()
        if geotransform:
            info.append(f"  Geotransform: {geotransform}")

        # Check each band
        for i in range(1, ds.RasterCount + 1):
            try:
                band = ds.GetRasterBand(i)
                if band is None:
                    info.append(f"  Band {i}: Cannot access")
                    continue

                info.append(f"  Band {i}:")
                info.append(f"    Data type: {gdal.GetDataTypeName(band.DataType)}")
                info.append(f"    NoData value: {band.GetNoDataValue()}")
                info.append(f"    Block size: {band.GetBlockSize()}")
                info.append(f"    Statistics: Min={band.GetMinimum()}, Max={band.GetMaximum()}")

            except Exception as e:
                info.append(f"  Band {i} error: {str(e)}")

        ds = None
        return "\n".join(info)

    except Exception as e:
        return f"Error getting file information: {str(e)}"


def auto_output_name(input_tif: str, output_dir: Path, processor: str) -> Path:
    """Derive the canonical multilook output path for an input product.

    If the input is a wrapped interferogram (``.int[.tif]``) with a canonical
    variant stored inside a ``{date1}_{date2}`` directory, output the next
    variant (``fullres -> mli``, ``filt -> filt_mli``) under the same date-pair
    directory in ``output_dir``.  Otherwise fall back to the legacy
    ``multilooked_`` prefix (standalone single-file usage).
    """
    input_path = Path(input_tif)
    date_pair = input_path.parent.name
    if is_date_pair_dir(date_pair) and input_path.name.endswith(int_ext(processor)):
        variant = variant_of(input_path, processor)
        out_variant = next_variant(variant, "multilook")
        if out_variant == variant:
            # already multilooked: overwrite in place (idempotent re-run)
            return output_dir / date_pair / input_path.name
        return output_dir / date_pair / f"{out_variant}{int_ext(processor)}"

    # Legacy fallback
    suffixes = input_path.suffixes
    if suffixes:
        base = input_path.name[:-len(''.join(suffixes))]
        new_name = "multilooked_" + base + ''.join(suffixes)
        return output_dir / new_name
    return output_dir / f"multilooked_{input_path.name}"


def get_file_size(input_file):
    """Get the size (width, height) of a file using GDAL."""
    gdal = _gdal()
    ds = gdal.Open(input_file, gdal.GA_ReadOnly)
    if ds is None:
        error_msg = gdal.GetLastErrorMsg()
        raise ValueError(f"Could not open file: {input_file}. GDAL error: {error_msg}")

    width = ds.RasterXSize
    height = ds.RasterYSize
    ds = None
    return width, height


def process_geometry_files(geom_dir, input_file, lks_y, lks_x, output_geom_dir=None, processor='isce3'):
    """Process geometry files in the specified directory."""
    input_width, input_height = get_file_size(input_file)

    geom_files = []
    for root, dirs, files in os.walk(geom_dir):
        for file in files:
            if file.endswith('.full'):
                geom_files.append(os.path.join(root, file))

    if not geom_files:
        logger.warning(f"No .full files found in geometry directory: {geom_dir}")
        return

    logger.info(f"Found {len(geom_files)} geometry files in {geom_dir}")

    if output_geom_dir:
        os.makedirs(output_geom_dir, exist_ok=True)
    else:
        output_geom_dir = geom_dir

    # Ensure full-resolution geometry files have .xml companions.
    # MintPy's extract_multilook_number() reads {multilooked}.full.xml to compare
    # full-resolution vs multilooked dimensions and compute ALOOKS/RLOOKS.
    for geom_file in geom_files:
        xml_file = geom_file + '.xml'
        if not os.path.isfile(xml_file):
            logger.info(f"Creating .xml for full-resolution geometry: {os.path.basename(geom_file)}")
            from .utils.slc2ifg_utils import create_xml_for_binary
            create_xml_for_binary(geom_file, family='image',
                                  description='Full-resolution geometry')

    logger.info(f"Processing geometry files with multilook {lks_y}x{lks_x}, method=nearest, processor={processor}")
    logger.info(f"Output directory for geometry files: {output_geom_dir}")

    processed_files = 0
    for geom_file in geom_files:
        try:
            geom_path = Path(geom_file)
            output_name = geom_path.stem  # removes .full
            output_file = os.path.join(output_geom_dir, output_name)

            # Remove stale output so multilook_tif's skip-existing check
            # doesn't prevent regeneration with correct dimensions.
            for ext in ['', '.xml', '.hdr', '.aux.xml', '.rsc']:
                stale = output_file + ext
                if os.path.isfile(stale):
                    os.remove(stale)

            multilook_tif(
                input_tif=geom_file,
                output_tif=output_file,
                lks_y=lks_y,
                lks_x=lks_x,
                method='nearest',
                processor=processor
            )
            processed_files += 1
        except Exception as e:
            logger.error(f"Error processing geometry file {geom_file}: {str(e)}")

    logger.info(f"Processed {processed_files} geometry files")
    return output_geom_dir


def parse_arguments(args_list=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Multilook files using multilook_tif function.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ISCE3 (GeoTIFF) single file
  multilook.py --processor isce3 --input data.tif --lks-y 4 --lks-x 4

  # ISCE2 (ENVI) batch processing
  multilook.py --processor isce2 --input-dir ./interferograms --pattern *.int --lks-y 4 --lks-x 4 --output-dir ./multilooked

  # Process geometry files along with input files
  multilook.py --processor isce2 --input-dir ./interferograms --pattern *.int --lks-y 4 --lks-x 4 --geom-dir ./geom
        """
    )

    parser.add_argument(
        '--processor',
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF)"
    )

    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--input', '-i', help='Path to single input file to be multilooked')
    input_group.add_argument('--input-dir', help='Directory containing files to be multilooked')

    parser.add_argument('--ref-file', help='Reference IFG file for geometry dimension matching (requires --geom-dir)')
    parser.add_argument('--geom-only', action='store_true', help='Only process geometry files, skip IFG multilooking')

    parser.add_argument('--pattern', default="*.tif", help='File pattern for batch processing (default: *.tif)')
    parser.add_argument('--lks-y', type=int, required=True, help='Number of looks in y / row direction')
    parser.add_argument('--lks-x', type=int, required=True, help='Number of looks in x / column direction')
    parser.add_argument('--output', '-o', default=None, help='Path to output multilooked file (for single file processing)')
    parser.add_argument('--output-dir', default=None, help='Output directory for batch processing')
    parser.add_argument('--method', choices=['mean', 'median', 'nearest'], default='mean', help='Multilook method')
    parser.add_argument('--max-workers', type=int, default=4, help='Number of parallel workers (default: 4)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output for debugging')
    parser.add_argument('--geom-dir', help='Directory containing geometry files (.full) to be multilooked')
    parser.add_argument('--output-geom-dir', help='Output directory for multilooked geometry files')

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def process_single_file_wrapper(task):
    """Wrapper function for parallel processing of single files."""
    input_file, lks_y, lks_x, output_file, method, processor = task
    try:
        output_path = multilook_tif(
            input_tif=input_file,
            lks_y=lks_y,
            lks_x=lks_x,
            output_tif=output_file,
            method=method,
            processor=processor
        )
        return (input_file, True, f"Output: {Path(output_path).name}")
    except Exception as e:
        return (input_file, False, f"Error: {str(e)}")


def process_single_file(args):
    """Process a single input file."""
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    try:
        output_file = multilook_tif(
            input_tif=args.input,
            output_tif=args.output,
            lks_y=args.lks_y,
            lks_x=args.lks_x,
            method=args.method,
            processor=args.processor
        )
        logger.info("SUCCESS Multilooking completed successfully!")
        logger.info(f"  Input:  {args.input}")
        logger.info(f"  Output: {output_file}")
    except Exception as e:
        logger.error(f"FAILED Error processing {args.input}: {e}")
        raise


def process_batch_files(args):
    """Process multiple files using directory and pattern."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Recursive glob: supports date-pair subdirectories (e.g. **/*.int.tif)
    input_files = sorted(input_dir.glob(args.pattern))

    if not input_files:
        logger.warning(f"No files found matching pattern: {args.pattern} in {input_dir}")
        return

    logger.info(f"Found {len(input_files)} files matching pattern: {args.pattern}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = input_dir

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Using {args.max_workers} parallel workers (threads)")

    tasks = []
    for input_file in input_files:
        if args.output_dir:
            output_file = auto_output_name(input_file, output_dir, args.processor)
        else:
            output_file = None
        tasks.append((str(input_file), args.lks_y, args.lks_x, str(output_file), args.method, args.processor))

    successful_count = 0
    error_count = 0

    with ThreadPoolExecutor(max_workers=max(args.max_workers, 1)) as executor:
        futures = [executor.submit(process_single_file_wrapper, task) for task in tasks]
        for future in futures:
            input_file, success, message = future.result()
            if success:
                logger.info(f"SUCCESS {Path(input_file).name}: {message}")
                successful_count += 1
            else:
                logger.error(f"FAILED {Path(input_file).name}: {message}")
                error_count += 1

    logger.info("\nBatch processing complete:")
    logger.info(f"  Successfully processed: {successful_count} files")
    logger.info(f"  Errors: {error_count} files")
    logger.info(f"  Output directory: {output_dir}")


def main(args=None):
    """Main function to handle command line arguments and execute the script."""
    if args is None:
        args = parse_arguments()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.debug("Verbose debugging enabled")

    if args.lks_y <= 0 or args.lks_x <= 0:
        raise ValueError("Look numbers (--lks-y and --lks-x) must be positive integers")

    # Process geometry files if specified
    if args.geom_dir:
        if not os.path.isdir(args.geom_dir):
            raise FileNotFoundError(f"Geometry directory not found: {args.geom_dir}")

        if args.ref_file:
            reference_file = args.ref_file
        elif args.input:
            reference_file = args.input
        elif args.input_dir:
            input_files = sorted(Path(args.input_dir).glob(args.pattern))
            if not input_files:
                raise FileNotFoundError(f"No files found in {args.input_dir} matching pattern {args.pattern}")
            reference_file = str(input_files[0])
        else:
            raise ValueError("--ref-file, --input, or --input-dir is required with --geom-dir")

        logger.info(f"Processing geometry files with reference to: {reference_file}")
        geom_output_dir = process_geometry_files(
            geom_dir=args.geom_dir,
            input_file=reference_file,
            lks_y=args.lks_y,
            lks_x=args.lks_x,
            output_geom_dir=args.output_geom_dir,
            processor=args.processor
        )
        if geom_output_dir:
            logger.info(f"Geometry files processed and saved to: {geom_output_dir}")

    # Process input files (skip if --geom-only)
    if args.geom_only:
        pass
    elif args.input:
        process_single_file(args)
    elif args.input_dir:
        process_batch_files(args)


if __name__ == '__main__':
    main()
