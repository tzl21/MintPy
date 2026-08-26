#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Script for applying Goldstein and long-wavelength filters to interferometric data.
Supports batch processing with file patterns in directories.

Fixed output structure for wrapped interferograms (see ``utils.naming``):
    ``output_dir/{date1}_{date2}/filt.int[.tif]``       (from fullres)
    ``output_dir/{date1}_{date2}/filt_mli.int[.tif]``   (from mli)

Parallelism uses threads (``ThreadPoolExecutor``).
"""

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import repeat
from pathlib import Path

import numpy as np

from .utils.filter_utils import goldstein, filter_rasters
from .utils.naming import (
    int_ext,
    is_date_pair_dir,
    next_variant,
    variant_of,
)
from .utils.slc2ifg_utils import create_xml_for_binary
from .utils.stitching_utils import load_gdal, write_arr, DEFAULT_TIFF_OPTIONS, DEFAULT_ENVI_OPTIONS

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments(args_list=None):
    """Parse command line arguments.

    Args:
        args_list: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Apply Goldstein or long-wavelength filters to interferometric data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Goldstein filter for ISCE3 (GeoTIFF)
  filter.py --processor isce3 --filter-type goldstein --input-dir ./interferograms --pattern *.int.tif --output-dir ./filtered --alpha 0.8 --psize 32

  # Long-wavelength filter for ISCE2 (ENVI)
  filter.py --processor isce2 --filter-type long_wavelength --input-dir ./unwrapped --pattern *.unw --output-dir ./filtered --wavelength-cutoff 25000 --pixel-spacing 30

  # Long-wavelength filter with correlation mask for ISCE3
  filter.py --processor isce3 --filter-type long_wavelength --input-dir ./unwrapped --pattern *.unw.tif --cor-dir ./correlation --cor-pattern *_phsig.coh.tif --wavelength-cutoff 50000 --correlation-cutoff 0.3
        """
    )

    parser.add_argument(
        '--processor',
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF/HDF5)"
    )

    parser.add_argument(
        '--filter-type',
        choices=['goldstein', 'long_wavelength'],
        default='goldstein',
        help='Type of filter to apply: goldstein or long_wavelength'
    )

    parser.add_argument(
        '--input-dir',
        required=True,
        help='Directory containing input files to be filtered'
    )

    parser.add_argument(
        '--pattern',
        help='File pattern for input files. If not set, uses processor default: '
             '".int.tif" for isce3, ".int" for isce2.'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for filtered files'
    )

    # Common filter parameters
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    # Goldstein filter specific parameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.8,
        help='Goldstein filter parameter (0=no filtering, 1=max filtering) (default: 0.8)'
    )

    parser.add_argument(
        '--psize',
        type=int,
        default=32,
        help='Patch size for Goldstein filter (default: 32)'
    )

    # Long-wavelength filter specific parameters
    parser.add_argument(
        '--wavelength-cutoff',
        type=float,
        default=25000,
        help='Spatial wavelength cutoff in meters (default: 25000)'
    )

    parser.add_argument(
        '--pixel-spacing',
        type=float,
        default=30,
        help='Pixel spacing in meters (default: 30)'
    )

    parser.add_argument(
        '--correlation-cutoff',
        type=float,
        default=0.5,
        help='Correlation threshold for masking (default: 0.5)'
    )

    # Masking files for long-wavelength filter (directory + pattern)
    parser.add_argument(
        '--cor-dir',
        default=None,
        help='Directory containing correlation files for masking'
    )

    parser.add_argument(
        '--cor-pattern',
        help='Pattern for correlation files. If not set, uses processor default: '
             '"*_phsig.coh.tif" for isce3, "*_phsig.coh" for isce2.'
    )

    parser.add_argument(
        '--conncomp-dir',
        default=None,
        help='Directory containing connected component files for masking'
    )

    parser.add_argument(
        '--conncomp-pattern',
        help='Pattern for connected component files. If not set, uses processor default: '
             '".unw.conncomp.tif" for isce3, ".unw.conncomp" for isce2.'
    )

    parser.add_argument(
        '--temporal-coherence-file',
        default=None,
        help='Temporal coherence file for masking'
    )

    parser.add_argument(
        '--fill-value',
        type=float,
        default=None,
        help='Value to place in masked output pixels (default: interpolate)'
    )

    parser.add_argument(
        '--scratch-dir',
        default=None,
        help='Directory for temporary files (default: system temp)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    if args_list is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_list)

    # Set default patterns based on processor if not provided
    if args.pattern is None:
        args.pattern = "**/*.int.tif" if args.processor == "isce3" else "**/*.int"
    if args.cor_pattern is None:
        args.cor_pattern = "**/*_phsig.coh.tif" if args.processor == "isce3" else "**/*_phsig.coh"
    if args.conncomp_pattern is None:
        args.conncomp_pattern = "**/*.unw.conncomp.tif" if args.processor == "isce3" else "**/*.unw.conncomp"

    return args


def find_files_by_pattern(directory, pattern, processor):
    """
    Find files in directory matching the pattern (recursive).
    Also warns if file extensions do not match processor expectations.

    Parameters:
    -----------
    directory : Path
        Directory to search in
    pattern : str
        File pattern (e.g., "*.tif", "*.h5")
    processor : str
        'isce2' or 'isce3' for extension validation

    Returns:
    --------
    list: List of matching file paths
    """
    directory = Path(directory)
    files = sorted(directory.glob(pattern))

    # Expected extensions based on processor
    expected_exts = ['.tif', '.tiff', '.h5', '.hdf5'] if processor == 'isce3' else ['.int', '.unw', '.coh', '.conncomp']

    for f in files:
        ext = Path(f).suffix.lower()
        if ext not in expected_exts:
            logger.warning(
                f"Processor '{processor}' expects extensions {expected_exts}, "
                f"but found '{ext}' for file {Path(f).name}"
            )

    return files


def auto_filter_output_name(input_file, output_dir, processor):
    """Derive the canonical filtered output path.

    Wrapped interferograms (``.int[.tif]``) inside ``{date1}_{date2}``
    directories get the next filter variant (``fullres -> filt``,
    ``mli -> filt_mli``).  Other products keep the legacy ``filtered_``
    prefix (standalone usage, e.g. long-wavelength filtering of .unw).
    """
    input_path = Path(input_file)
    date_pair = input_path.parent.name
    if is_date_pair_dir(date_pair) and input_path.name.endswith(int_ext(processor)):
        variant = variant_of(input_path, processor)
        out_variant = next_variant(variant, "filter")
        if out_variant == variant:
            # already filtered: overwrite in place (idempotent re-run)
            return output_dir / date_pair / input_path.name
        return output_dir / date_pair / f"{out_variant}{int_ext(processor)}"
    if is_date_pair_dir(date_pair):
        # include the date pair so different pairs never collide on the
        # same flat output name (e.g. unwrapped inputs of several pairs)
        return output_dir / f"filtered_{date_pair}_{input_path.stem}{input_path.suffix}"
    return output_dir / f"filtered_{input_path.stem}{input_path.suffix}"


def process_goldstein_files(input_files, output_dir, args):
    """Process files using Goldstein filter."""
    logger.info(f"Processing {len(input_files)} files with {args.max_workers} workers (threads)...")

    with ThreadPoolExecutor(max_workers=max(args.max_workers, 1)) as executor:
        results = list(executor.map(
            process_single_goldstein,
            input_files,
            repeat(output_dir),
            repeat(args.alpha),
            repeat(args.psize),
            repeat(args.processor)
        ))

    logger.info(f"Goldstein filtering completed. Processed {len(results)} files.")


def process_single_goldstein(input_file, output_dir, alpha, psize, processor,
                             output_file=None, gpu=False):
    """Process single file with Goldstein filter (for parallel processing).

    If ``output_file`` is not given it is derived canonically from
    ``output_dir`` (the root containing ``{date1}_{date2}/`` directories).
    ``gpu`` selects the CuPy batched-FFT path (identical results).
    """
    input_path = Path(input_file)
    if output_file is None:
        output_file = auto_filter_output_name(input_file, output_dir, processor)
    output_file = Path(output_file)

    if output_file.exists():
        logger.info(f"Skipping existing output: {output_file.name}")
        return output_file

    logger.info(f"Applying Goldstein filter to: {input_path.name}")

    data = load_gdal(str(input_file))

    # Determine driver and options based on processor
    if processor == 'isce3':
        driver = "GTiff"
        opts = list(DEFAULT_TIFF_OPTIONS)
    else:
        driver = "ENVI"
        opts = list(DEFAULT_ENVI_OPTIONS)

    nodata_mask = np.abs(data) < 1e-6
    filtered_data = goldstein(data, alpha, psize, nodata_mask=nodata_mask,
                              gpu=gpu)
    write_arr(
        arr=filtered_data,
        like_filename=input_file,
        output_name=output_file,
        driver=driver,
        options=opts,
    )

    if processor == 'isce2':
        ext = input_path.suffix.lower()
        family = 'intimage' if ext in ('.int', '.unw') else 'image'
        create_xml_for_binary(output_file, family=family,
                              description='Filtered interferogram')

    return output_file


def process_long_wavelength_files(input_files, output_dir, args, cor_files=None, conncomp_files=None):
    """Process files using long-wavelength filter."""
    # Convert file lists to Path objects
    input_files = [Path(f) for f in input_files]
    cor_files = [Path(f) for f in cor_files] if cor_files else None
    conncomp_files = [Path(f) for f in conncomp_files] if conncomp_files else None
    temporal_coherence_file = Path(args.temporal_coherence_file) if args.temporal_coherence_file else None

    # Use the filter_rasters function for batch processing
    output_files = filter_rasters(
        unw_filenames=input_files,
        cor_filenames=cor_files,
        conncomp_filenames=conncomp_files,
        temporal_coherence_filename=temporal_coherence_file,
        wavelength_cutoff=args.wavelength_cutoff,
        correlation_cutoff=args.correlation_cutoff,
        pixel_spacing=args.pixel_spacing,
        fill_value=args.fill_value,
        output_dir=output_dir,
        max_workers=args.max_workers
    )

    logger.info(f"Long-wavelength filtering completed. Processed {len(output_files)} files.")


def main(args=None):
    """Main function to handle command line arguments and execute the script."""
    if args is None:
        args = parse_arguments()

    setup_logging(args.verbose)

    # Validate inputs
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files
    input_files = find_files_by_pattern(input_dir, args.pattern, args.processor)
    if not input_files:
        logger.error(f"No files found matching pattern '{args.pattern}' in {input_dir}")
        return 1

    logger.info(f"Found {len(input_files)} files matching pattern '{args.pattern}' in {input_dir}")

    # Validate filter-specific parameters
    if args.filter_type == 'goldstein':
        if args.alpha < 0 or args.alpha > 1:
            logger.error("Alpha must be between 0 and 1")
            return 1
        if args.psize <= 0:
            logger.error("Patch size must be positive")
            return 1

    elif args.filter_type == 'long_wavelength':
        if args.wavelength_cutoff <= 0:
            logger.error("Wavelength cutoff must be positive")
            return 1
        if args.pixel_spacing <= 0:
            logger.error("Pixel spacing must be positive")
            return 1

        # Find correlation files if directory provided
        cor_files = None
        if args.cor_dir:
            cor_dir = Path(args.cor_dir)
            if not cor_dir.exists():
                logger.error(f"Correlation directory not found: {cor_dir}")
                return 1
            cor_files = find_files_by_pattern(cor_dir, args.cor_pattern, args.processor)
            if not cor_files:
                logger.warning(f"No correlation files found matching pattern '{args.cor_pattern}' in {cor_dir}")
            else:
                logger.info(f"Found {len(cor_files)} correlation files")

        # Find connected component files if directory provided
        conncomp_files = None
        if args.conncomp_dir:
            conncomp_dir = Path(args.conncomp_dir)
            if not conncomp_dir.exists():
                logger.error(f"Connected component directory not found: {conncomp_dir}")
                return 1
            conncomp_files = find_files_by_pattern(conncomp_dir, args.conncomp_pattern, args.processor)
            if not conncomp_files:
                logger.warning(f"No connected component files found matching pattern '{args.conncomp_pattern}' in {conncomp_dir}")
            else:
                logger.info(f"Found {len(conncomp_files)} connected component files")

    # Process files based on filter type
    try:
        if args.filter_type == 'goldstein':
            process_goldstein_files(input_files, output_dir, args)
        else:
            process_long_wavelength_files(input_files, output_dir, args, cor_files, conncomp_files)
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        return 1

    logger.info("All filtering tasks completed successfully.")
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.error("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"\nUnexpected error: {e}")
        raise
