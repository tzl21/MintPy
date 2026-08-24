#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Script for multilooking files using multilook_tif function.
Supports both single file and batch processing with file patterns.
Also supports multilooking geometry files.

Fixed output structure (see ``utils.naming``):
    ``output_dir/{date1}_{date2}/mli.int[.tif]``        (from fullres)
    ``output_dir/{date1}_{date2}/filt_mli.int[.tif]``   (from filt)

Parallelism uses threads (``ThreadPoolExecutor``) so GDAL/numpy shared
libraries are reused instead of spawning processes.
"""

import argparse
import logging
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from osgeo import gdal

from .utils.naming import (
    int_ext,
    is_date_pair_dir,
    next_variant,
    variant_of,
)
from .utils.slc2ifg_utils import create_xml_for_binary

from mintpy.stdproc.multilook import multilook_tif


# Enable GDAL exception handling
gdal.UseExceptions()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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


def get_file_info(input_tif):
    """Get detailed file information for debugging"""
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
