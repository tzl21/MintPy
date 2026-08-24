#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Coherence Estimator from SLC pairs - Complex correlation magnitude for InSAR.

This module computes the magnitude of the complex correlation coefficient
(coherence) from pairs of Single Look Complex (SLC) images using a boxcar
sliding window, fully vectorized with ``scipy.ndimage``.

Fixed output structure (see ``utils.naming``):
    ``output_dir/{date1}_{date2}/fullres_cpx.coh``      (isce2)
    ``output_dir/{date1}_{date2}/fullres_cpx.coh.tif``  (isce3)

Parallelism uses threads (``ThreadPoolExecutor``).
"""

import argparse
import glob
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from osgeo import gdal
from scipy.ndimage import correlate

from .utils.naming import coh_path
from .utils.slc2ifg_utils import create_xml_for_binary, open_gdal

gdal.UseExceptions()

DEFAULT_PARAMS = {
    'window_size': 5,
    'use_amplitude': True,
}


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with console and optional file output."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.getLogger('osgeo').setLevel(logging.WARNING)
    return logger


class CoherenceEstimator:
    """
    Coherence estimator using complex correlation magnitude from two SLCs.

    The boxcar coherence is computed with scipy.ndimage.correlate (C level),
    which is O(N) per pixel instead of O(window^2) — several orders of
    magnitude faster than the previous per-pixel Python loop.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the estimator with given parameters.

        Args:
            params: Dictionary of processing parameters. Defaults to DEFAULT_PARAMS.
        """
        self.logger = logging.getLogger(__name__)
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.win_size = self.params['window_size']

        # Ensure odd window size
        if self.win_size % 2 == 0:
            self.win_size += 1

        self.half_win = self.win_size // 2

        self._precompute_kernel()

        self.logger.info("Initialized: window_size=%dx%d", self.win_size, self.win_size)

    def _precompute_kernel(self) -> None:
        """Precompute the spatial averaging kernel (normalized boxcar)."""
        self.kernel = np.ones((self.win_size, self.win_size), dtype=np.float32)
        self.kernel /= np.sum(self.kernel)

    def compute_coherence(self, slc1: np.ndarray, slc2: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute coherence map from two SLC images.

        Args:
            slc1: Complex SLC array (reference).
            slc2: Complex SLC array (secondary).
            mask: Optional boolean mask of pixels to process.

        Returns:
            2D float32 array of coherence values in [0,1].
        """
        if slc1.shape != slc2.shape:
            raise ValueError(f"SLC shape mismatch: {slc1.shape} vs {slc2.shape}")

        rows, cols = slc1.shape
        self.logger.info("Computing coherence with window %dx%d...", self.win_size, self.win_size)
        start_time = time.time()

        # Form complex interferogram: S1 * conj(S2)
        interferogram = slc1 * np.conj(slc2)

        # Boxcar-averaged numerator and denominator (vectorized, C-level)
        mag_sq1 = (np.abs(slc1) ** 2).astype(np.float32)
        mag_sq2 = (np.abs(slc2) ** 2).astype(np.float32)

        sum_ifg = correlate(interferogram, self.kernel, mode='constant', cval=0.0)
        sum_mag1 = correlate(mag_sq1, self.kernel, mode='constant', cval=0.0)
        sum_mag2 = correlate(mag_sq2, self.kernel, mode='constant', cval=0.0)

        denominator = np.sqrt(sum_mag1 * sum_mag2)
        coherence = np.zeros((rows, cols), dtype=np.float32)
        valid = denominator > 0
        coherence[valid] = np.abs(sum_ifg[valid]) / denominator[valid]
        coherence = np.clip(coherence, 0.0, 1.0)

        # Match original behaviour: borders of half_win are not estimated
        if self.half_win > 0:
            coherence[:self.half_win, :] = 0.0
            coherence[-self.half_win:, :] = 0.0
            coherence[:, :self.half_win] = 0.0
            coherence[:, -self.half_win:] = 0.0

        if mask is not None:
            coherence[~mask] = 0.0

        self.logger.info("Coherence computation completed in %.1fs", time.time() - start_time)
        return coherence

    def process(self, slc1: np.ndarray, slc2: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Public interface for coherence estimation from two SLCs."""
        return self.compute_coherence(slc1, slc2, mask)


def read_complex_image(filename: str, processor: str,
                       subdataset: str = '/data/VV') -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a complex SLC image with processor-specific format awareness.

    ``subdataset`` selects the HDF5 dataset for ``.h5`` inputs (OPERA CSLC
    style, e.g. ``/data/VV``).
    """
    ext = Path(filename).suffix.lower()
    if processor == 'isce2':
        expected_exts = ['.slc', '.int', '.rdr', '.full']
        if ext not in expected_exts:
            logging.warning(
                f"Processor 'isce2' expects ENVI-style files (extensions {expected_exts}), "
                f"but got '{ext}'. Processing will continue but may fail."
            )
    elif processor == 'isce3':
        expected_exts = ['.tif', '.tiff', '.h5', '.hdf5']
        if ext not in expected_exts:
            logging.warning(
                f"Processor 'isce3' expects GeoTIFF or HDF5 files (extensions {expected_exts}), "
                f"but got '{ext}'. Processing will continue but may fail."
            )

    ds = open_gdal(filename, subdataset if ext in ('.h5', '.hdf5') else None)
    if ds is None:
        raise ValueError(f"Cannot open file: {filename}")

    metadata = {
        'transform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'rows': ds.RasterYSize,
        'cols': ds.RasterXSize,
        'band_count': ds.RasterCount,
    }

    if metadata['band_count'] == 1:
        data = ds.GetRasterBand(1).ReadAsArray()
        if data.dtype not in (np.complex64, np.complex128):
            data = data.astype(np.complex64)
    elif metadata['band_count'] == 2:
        real = ds.GetRasterBand(1).ReadAsArray()
        imag = ds.GetRasterBand(2).ReadAsArray()
        data = (real + 1j * imag).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported band count: {metadata['band_count']}")

    ds = None
    return data, metadata


def write_coherence_image(filename: str, coherence: np.ndarray,
                          metadata: Dict[str, Any], processor: str) -> None:
    """Write a single-band coherence image (0-1) to disk."""
    rows, cols = coherence.shape
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if processor == 'isce2':
        driver = 'ENVI'
        options = []
    else:  # isce3
        driver = 'GTiff'
        options = ['COMPRESS=LZW', 'TILED=YES']

    driver_obj = gdal.GetDriverByName(driver)
    out_ds = driver_obj.Create(filename, cols, rows, 1, gdal.GDT_Float32, options)
    out_ds.SetGeoTransform(metadata['transform'])
    out_ds.SetProjection(metadata['projection'])

    band = out_ds.GetRasterBand(1)
    band.WriteArray(coherence)
    band.SetDescription('coherence')
    band.SetNoDataValue(0.0)

    if driver == 'GTiff':
        band.SetColorInterpretation(gdal.GCI_GrayIndex)

    out_ds.FlushCache()
    out_ds = None


def extract_date_from_slc(filename: str) -> str:
    """Extract YYYYMMDD date from SLC filename."""
    name = Path(filename).name
    # Remove known SLC extensions
    if name.endswith('.slc.tif'):
        name = name[:-8]   # remove '.slc.tif'
    elif name.endswith('.slc.h5'):
        name = name[:-7]   # remove '.slc.h5'
    elif name.endswith('.slc'):
        name = name[:-4]   # remove '.slc'
    elif name.endswith('.tif') or name.endswith('.tiff'):
        name = Path(filename).stem
    # Try to find an 8-digit date pattern
    match = re.search(r'(\d{8})', name)
    if match:
        return match.group(1)
    # Fallback to the cleaned name
    return name


def expand_directories(directory_patterns):
    """Expand directory patterns into a list of directory paths."""
    expanded_dirs = []
    for pattern in directory_patterns:
        matches = glob.glob(pattern, recursive=False)
        if not matches:
            logging.warning(f"No directories found matching pattern: {pattern}")
        expanded_dirs.extend(Path(match) for match in matches)
    return expanded_dirs


def find_slc_file_by_date(slc_directories, target_date, slc_pattern):
    """Find SLC file that contains the target date in its filename."""
    # First, try exact pattern match
    for slc_dir in slc_directories:
        pattern = f"*{target_date}{slc_pattern}"
        matching_files = list(slc_dir.glob(pattern)) + list(slc_dir.glob(f"*/{pattern}"))
        if matching_files:
            return matching_files[0]

    # If not found, try a more flexible search with subdirectories
    for slc_dir in slc_directories:
        for slc_file in list(slc_dir.glob(slc_pattern)) + list(slc_dir.glob(f"*/{slc_pattern}")):
            if target_date in slc_file.name:
                return slc_file

    return None


def process_slc_pair(slc1_file: str, slc2_file: str, output_dir: Path,
                     params: Dict[str, Any], processor: str) -> Tuple[str, str, bool, str]:
    """Process a single SLC pair and write coherence output."""
    logger = logging.getLogger(__name__)

    try:
        date1 = extract_date_from_slc(slc1_file)
        date2 = extract_date_from_slc(slc2_file)

        # Fixed structure: output_dir/{date1}_{date2}/fullres_cpx.coh[.tif]
        output_file = coh_path(output_dir, date1, date2, variant='fullres',
                               kind='cpx', processor=processor)

        if output_file.exists():
            logger.info("Skipping existing file: %s", output_file)
            return date1, date2, True, f"Exists: {output_file.name}"

        logger.info("Processing pair: %s - %s", date1, date2)
        start_time = time.time()

        slc1, meta1 = read_complex_image(slc1_file, processor)
        slc2, meta2 = read_complex_image(slc2_file, processor)

        if slc1.shape != slc2.shape:
            raise ValueError(f"Dimension mismatch: {slc1.shape} vs {slc2.shape}")

        logger.info("Data shape: %s", slc1.shape)

        estimator = CoherenceEstimator(params)
        coherence = estimator.process(slc1, slc2)
        coherence = np.clip(coherence, 0.0, 1.0)

        write_coherence_image(str(output_file), coherence, meta1, processor)

        if processor == 'isce2':
            create_xml_for_binary(output_file, family='image',
                                  description='Complex correlation magnitude')

        elapsed = time.time() - start_time
        logger.info("Completed in %.1fs: %s", elapsed, output_file.name)

        valid = coherence > 0
        if np.any(valid):
            mean_val = np.mean(coherence[valid])
            min_val = np.min(coherence[valid])
            max_val = np.max(coherence[valid])
            logger.info("  Coherence stats: mean=%.3f, min=%.3f, max=%.3f",
                        mean_val, min_val, max_val)

        return date1, date2, True, f"Done in {elapsed:.1f}s"

    except Exception as exc:
        logger.error("Error processing %s - %s: %s", slc1_file, slc2_file, exc)
        date1 = extract_date_from_slc(slc1_file) if slc1_file else "unknown"
        date2 = extract_date_from_slc(slc2_file) if slc2_file else "unknown"
        return date1, date2, False, f"Error: {exc}"


def parse_arguments(args_list=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Coherence Estimator from SLC pairs - Computes complex correlation magnitude (0-1)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default parameters:
  window_size:   {DEFAULT_PARAMS['window_size']}

Input: Pairs file (ifgram_list.txt) and SLC directory.
Output: Single-band raster with coherence values in range [0,1] for each pair.
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
        '--pairs-file',
        required=True,
        help='Path to interferometric pairs list file (e.g., ifgram_list.txt)'
    )
    parser.add_argument(
        '--slc-dir',
        required=True,
        nargs='+',
        help='Directory patterns containing SLC files (can use wildcards like ../t124*iw2/*/)'
    )
    parser.add_argument(
        '--slc-pattern',
        default='*.slc.*',
        help='Pattern for SLC files (e.g., "*.slc.tif", "*.slc", "*.slc.h5") (default: "*.slc.*")'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='.',
        help='Output directory (default: current directory)'
    )

    param_group = parser.add_argument_group('Parameters')
    param_group.add_argument(
        '--window-size',
        type=int,
        default=DEFAULT_PARAMS['window_size'],
        help=f'Sliding window size for coherence estimation (default: {DEFAULT_PARAMS["window_size"]})'
    )

    proc_group = parser.add_argument_group('Processing options')
    proc_group.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Number of parallel workers (default: 1)'
    )

    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    log_group.add_argument(
        '--log-file',
        type=str,
        help='Write log to file'
    )
    log_group.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress non-error messages'
    )

    return parser.parse_args(args_list) if args_list else parser.parse_args()


def _detect_burst_dirs(slc_dirs):
    """Detect burst subdirectories under the given SLC directory list."""
    burst_pattern = re.compile(r'^t\d+_\d+_iw\d+$')
    burst_map = {}
    for slc_dir in slc_dirs:
        if slc_dir.is_dir():
            for entry in sorted(slc_dir.iterdir()):
                if entry.is_dir() and burst_pattern.match(entry.name):
                    burst_map[entry.name] = entry
    if burst_map:
        logging.info("Detected %d burst(s): %s", len(burst_map), list(burst_map.keys()))
        return burst_map
    # If slc_dirs themselves ARE burst directories, use flat mode
    # (caller already handles burst separation via --output-dir)
    burst_dirs = [d for d in slc_dirs if burst_pattern.match(d.name)]
    if burst_dirs and len(burst_dirs) == len(slc_dirs):
        logging.info("Input directories are already burst-specific — flat output mode")
        return {None: slc_dirs}
    return {None: slc_dirs}


def main(args=None):
    """Command-line entry point."""
    if args is None:
        args = parse_arguments()

    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    setup_logging(verbose=(log_level == logging.DEBUG), log_file=args.log_file)
    logging.getLogger().setLevel(log_level)
    logger = logging.getLogger(__name__)
    logger.info("Coherence Estimator from SLC pairs started")
    logger.info(f"Processor: {args.processor}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Expand directory patterns
    slc_directories = expand_directories(args.slc_dir)
    if not slc_directories:
        logger.error("No SLC directories found.")
        sys.exit(1)

    logger.info(f"Found {len(slc_directories)} SLC directories:")
    for d in slc_directories:
        logger.info(f"  {d}")

    # Read pairs file
    try:
        pairs_df = pd.read_csv(args.pairs_file, comment='#', sep=r'\s+', names=['date12'])
        logger.info(f"Found {len(pairs_df)} pairs in {args.pairs_file}")
    except Exception as e:
        logger.error(f"Error reading pairs file {args.pairs_file}: {e}")
        sys.exit(1)

    # Detect burst subdirectories for per-burst processing
    burst_map = _detect_burst_dirs(slc_directories)

    # Build list of SLC pairs — per-burst if multiple bursts detected
    slc_pairs = []
    for burst_id, burst_dirs in burst_map.items():
        b_dirs = [burst_dirs] if burst_id else burst_dirs
        b_out = output_dir / burst_id if burst_id else output_dir
        b_out.mkdir(parents=True, exist_ok=True)
        for _, row in pairs_df.iterrows():
            date1, date2 = row['date12'].split('-')
            slc1 = find_slc_file_by_date(b_dirs, date1, args.slc_pattern)
            slc2 = find_slc_file_by_date(b_dirs, date2, args.slc_pattern)
            if slc1 is None:
                logger.warning(f"SLC file for date {date1} not found, skipping pair {date1}_{date2}")
                continue
            if slc2 is None:
                logger.warning(f"SLC file for date {date2} not found, skipping pair {date1}_{date2}")
                continue
            slc_pairs.append((str(slc1), str(slc2), b_out))

    if not slc_pairs:
        logger.error("No valid SLC pairs found. Exiting.")
        sys.exit(1)

    logger.info(f"Found {len(slc_pairs)} SLC pair(s) to process")

    params = {
        'window_size': args.window_size,
        'use_amplitude': DEFAULT_PARAMS['use_amplitude'],
    }

    logger.info("Parameters: window_size=%d", params['window_size'])

    successful = 0
    failed = 0

    if args.max_workers > 1 and len(slc_pairs) > 1:
        logger.info("Using parallel processing with %d workers (threads)", args.max_workers)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = [
                (Path(slc1).name, Path(slc2).name,
                 executor.submit(process_slc_pair, slc1, slc2, b_out, params, args.processor))
                for slc1, slc2, b_out in slc_pairs
            ]

            for name1, name2, future in results:
                try:
                    n1, n2, success, msg = future.result()
                    if success:
                        logger.info("SUCCESS %s - %s: %s", n1, n2, msg)
                        successful += 1
                    else:
                        logger.error("FAILED %s - %s: %s", n1, n2, msg)
                        failed += 1
                except Exception as exc:
                    logger.error("FAILED %s - %s: %s", name1, name2, exc)
                    failed += 1
    else:
        logger.info("Using sequential processing")
        for slc1, slc2, b_out in slc_pairs:
            n1, n2, success, msg = process_slc_pair(slc1, slc2, b_out, params, args.processor)
            if success:
                logger.info("SUCCESS %s - %s: %s", n1, n2, msg)
                successful += 1
            else:
                logger.error("FAILED %s - %s: %s", n1, n2, msg)
                failed += 1

    logger.info("Processing complete: Successful=%d, Failed=%d", successful, failed)
    logger.info("Output directory: %s", output_dir.absolute())
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
