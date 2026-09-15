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

import logging
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
from .utils.stitching_utils import load_gdal, write_arr, DEFAULT_TIFF_OPTIONS

logger = logging.getLogger(__name__)


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

    # All slc2ifg products are GeoTIFF for both processors
    driver = "GTiff"
    opts = list(DEFAULT_TIFF_OPTIONS)

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


