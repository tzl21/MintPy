#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Phase-sigma correlation estimator — matches ISCE2 Fortran ph_slope.F / ph_sigma.F.

Uses scipy.ndimage.correlate for GPU-free C-level gradient smoothing and
``numpy.lib.stride_tricks.sliding_window_view`` for vectorized window
extraction in row batches (bounded memory, no per-pixel Python loops).

Fixed output structure (see ``utils.naming``):
    ``output_dir/{date1}_{date2}/xxx_phsig.coh``      (isce2)
    ``output_dir/{date1}_{date2}/xxx_phsig.coh.tif``  (isce3)

Parallelism uses threads (``ThreadPoolExecutor``).
"""

import argparse
import glob
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal
from scipy.ndimage import correlate

from .utils.naming import (
    coh_ext,
    coh_path,
    extract_date_pair,
    is_date_pair_dir,
    variant_of,
)
from .utils.slc2ifg_utils import create_xml_for_binary

gdal.UseExceptions()

#: Approximate number of window-pixels processed per row-batch
_BATCH_WINDOW_PIXELS = 2_000_000


def setup_logging(verbose: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger('insarflow.phsig')
    for h in list(logger.handlers):
        try:
            h.close()
        except Exception:
            pass
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(
        '%(asctime)s %(levelname).4s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    if log_file:
        fh = logging.FileHandler(log_file, mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    logging.getLogger('osgeo').setLevel(logging.WARNING)
    return logger


# ---------------------------------------------------------------------------
# Gaussian kernel (matches ISCE2 Fortran ph_slope.F / ph_sigma.F)
# ---------------------------------------------------------------------------
def _gaussian_kernel(size):
    half = size // 2
    s1 = 0.0
    kernel = np.zeros((size, size), dtype=np.float64)
    for k in range(size):
        for j in range(size):
            w1 = (k - half) ** 2 + (j - half) ** 2
            kernel[k, j] = np.exp(-w1 / (size / 2.0))
            s1 += kernel[k, j]
    return (kernel / s1).astype(np.float32)


# ---------------------------------------------------------------------------
# Phase-sigma correlation (ISCE2 ph_slope.F + ph_sigma.F)
# ---------------------------------------------------------------------------
def estimate_phsig_correlation(
    ifg_arr: np.ndarray,
    ps_win: int = 5,
    grad_win: int = 5,
    nlks: float = 1.0,
) -> np.ndarray:
    """Estimate phase-sigma correlation from a complex interferogram.

    Matches ISCE2 Fortran ``ph_slope.F`` + ``ph_sigma.F`` algorithm:
    Gaussian-weighted phase gradient estimation via scipy.ndimage.correlate,
    local window deramping, weighted phase variance, and NLKS-based
    correlation conversion.

    Parameters
    ----------
    ifg_arr : np.ndarray (complex64)
        Complex interferogram ``[rows, cols]``.
    ps_win : int
        Phase-sigma estimation window size (odd, default 5).
    grad_win : int
        Gradient estimation window size (odd, default 5).
    nlks : float
        Number of looks parameter (default 1.0).

    Returns
    -------
    coh_phsig : np.ndarray (float32)
        Phase-sigma correlation array, clipped to [0, 1].
    """
    rows, cols = ifg_arr.shape

    if ps_win % 2 == 0:
        ps_win += 1
    if grad_win < 2:
        raise ValueError(
            f"ps_gradient_window must be >= 2, got {grad_win} "
            "(grad_win=1 degenerates the slope estimation)")
    if grad_win % 2 == 0:
        grad_win += 1
    ps_half = ps_win // 2
    grad_half = grad_win // 2

    # Gradient: complex product with neighbour
    padded = np.pad(ifg_arr,
                    ((grad_half, grad_half), (grad_half, grad_half)),
                    mode='constant')

    rg_diff = (
        padded[grad_half:grad_half + rows, grad_half:grad_half + cols] *
        np.conj(padded[grad_half:grad_half + rows,
                       grad_half - 1:grad_half + cols - 1])
    )
    az_diff = (
        padded[grad_half:grad_half + rows, grad_half:grad_half + cols] *
        np.conj(padded[grad_half - 1:grad_half + rows - 1,
                       grad_half:grad_half + cols])
    )

    # Gaussian-weighted smoothing of gradient (scipy C-level correlate)
    gk = _gaussian_kernel(grad_win)
    rg_smooth = correlate(rg_diff, gk)
    az_smooth = correlate(az_diff, gk)

    rg_slope = np.arctan2(rg_smooth.imag, rg_smooth.real)
    az_slope = np.arctan2(az_smooth.imag, az_smooth.real)
    rg_slope[np.abs(rg_smooth) == 0] = 0.0
    az_slope[np.abs(az_smooth) == 0] = 0.0

    # ISCE2 Fortran valid range: [half+1, size-half-1] → zero edges
    if grad_half > 0:
        for sl in (rg_slope, az_slope):
            sl[:grad_half + 1, :] = 0.0
            sl[-(grad_half):, :] = 0.0
            sl[:, :grad_half + 1] = 0.0
            sl[:, -(grad_half):] = 0.0

    # Valid (interior) output region
    i0, i1 = ps_half, rows - ps_half
    j0, j1 = ps_half, cols - ps_half
    if i1 <= i0 or j1 <= j0:
        return np.zeros((rows, cols), dtype=np.float32)

    offsets = np.arange(-ps_half, ps_half + 1)
    di_mesh, dj_mesh = np.meshgrid(offsets, offsets, indexing='ij')
    ps_weights = _gaussian_kernel(ps_win)

    # All windows as a strided *view*: (rows-ps_win+1, cols-ps_win+1, ps_win, ps_win)
    windows = sliding_window_view(ifg_arr, (ps_win, ps_win))

    n_rows_out = i1 - i0
    n_cols_out = j1 - j0

    # Per-pixel deramping ramps (broadcast over the window axes) are built
    # PER BATCH, never for the whole image at once: their shape is
    # (rows, cols, ps_win, ps_win) and they materialize as float64 (int64
    # mesh * float32 slope promotes), so a full-image ramp is O(N * ps_win^2
    # * 8) bytes — ~5.4 GB for a 3000x9000 scene at ps_win=5.  Building it
    # inside the row loop keeps the peak at O(batch * ps_win^2 * 8) with
    # bit-identical values (same float64 ops, same shapes).
    coh = np.zeros((rows, cols), dtype=np.float32)

    # Row-batched processing to bound peak memory
    win_pixels = n_cols_out * ps_win * ps_win
    batch_rows = max(1, _BATCH_WINDOW_PIXELS // max(win_pixels, 1))

    for r0 in range(0, n_rows_out, batch_rows):
        r1 = min(r0 + batch_rows, n_rows_out)

        # Window position (r, c) maps to image pixel (r+ps_half, c+ps_half), so
        # the valid window region covers ALL window rows/cols (n_rows_out x n_cols_out).
        win_block = windows[r0:r1]                   # (b, ncols_out, ps_win, ps_win) view
        ramp_block = (
            di_mesh[None, None, :, :] *
            az_slope[i0 + r0:i0 + r1, j0:j1][:, :, None, None] +
            dj_mesh[None, None, :, :] *
            rg_slope[i0 + r0:i0 + r1, j0:j1][:, :, None, None]
        )                                            # (b, ncols_out, ps_win, ps_win)

        exp_ramp = np.cos(ramp_block) - 1j * np.sin(ramp_block)
        comp = win_block * exp_ramp

        wsum = np.sum(comp, axis=(2, 3))             # (b, ncols)
        mag = np.abs(wsum)

        mag_valid = mag > 1e-10
        if np.any(mag_valid):
            norm_sum = np.zeros_like(wsum)
            norm_sum[mag_valid] = wsum[mag_valid] / mag[mag_valid]
            deramped = comp * np.conj(norm_sum[:, :, None, None])

            phases = np.arctan2(deramped.imag, deramped.real)
            wt = ps_weights[None, None, :, :]
            mean_ph = np.sum(wt * phases, axis=(2, 3))
            mean_ph2 = np.sum(wt * phases * phases, axis=(2, 3))
            var = mean_ph2 - mean_ph * mean_ph

            with np.errstate(divide='ignore', invalid='ignore'):
                val = np.where(var > 0,
                               1.0 / np.sqrt(2.0 * nlks * var + 1.0),
                               1.0)
            val = np.where(mag_valid, val, 0.0)
            coh[i0 + r0:i0 + r1, j0:j1] = val

    return np.clip(coh, 0.0, 1.0)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------
def read_complex_image(filename: str, processor: str) -> Tuple[np.ndarray, Dict]:
    ext = Path(filename).suffix.lower()
    expected_exts = ['.int', '.slc', '.rdr', '.full'] if processor == 'isce2' else ['.tif', '.tiff', '.h5', '.hdf5']
    if ext not in expected_exts:
        logging.warning(f"Processor '{processor}' expects {expected_exts}, got '{ext}'.")

    ds = gdal.Open(filename, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f"Cannot open file: {filename}")

    meta = {
        'transform': ds.GetGeoTransform(),
        'projection': ds.GetProjection(),
        'rows': ds.RasterYSize, 'cols': ds.RasterXSize,
        'band_count': ds.RasterCount,
    }
    if meta['band_count'] == 1:
        data = ds.GetRasterBand(1).ReadAsArray()
        if data.dtype not in (np.complex64, np.complex128):
            data = data.astype(np.complex64)
    elif meta['band_count'] == 2:
        data = (ds.GetRasterBand(1).ReadAsArray() + 1j * ds.GetRasterBand(2).ReadAsArray()).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported band count: {meta['band_count']}")
    ds = None
    return data, meta


def _write_band(filename, arr, meta, processor, description=''):
    rows, cols = arr.shape
    out_path = Path(filename)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    driver = 'ENVI' if processor == 'isce2' else 'GTiff'
    options = [] if driver == 'ENVI' else ['COMPRESS=LZW', 'TILED=YES']
    drv = gdal.GetDriverByName(driver)
    # Atomic write: create at a temp path, rename only after a successful
    # close — an interrupted run leaves no partial product at `filename`.
    tmp = f"{filename}.tmp"
    try:
        ds = drv.Create(tmp, cols, rows, 1, gdal.GDT_Float32, options)
        ds.SetGeoTransform(meta['transform'])
        ds.SetProjection(meta['projection'])
        band = ds.GetRasterBand(1)
        band.WriteArray(arr)
        band.SetNoDataValue(0.0)
        if description:
            band.SetDescription(description)
        ds.FlushCache()
        ds = None
        os.replace(tmp, filename)
        # GDAL's ENVI driver writes a companion '<tmp>.hdr' — rename it along
        # so the final product keeps its header (else the file is unreadable).
        tmp_hdr = f"{tmp}.hdr"
        if os.path.exists(tmp_hdr):
            os.replace(tmp_hdr, f"{filename}.hdr")
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
            if os.path.exists(f"{tmp}.hdr"):
                os.unlink(f"{tmp}.hdr")
        except OSError:
            pass
        raise


def _output_path(input_file, output_dir, processor):
    """Canonical phase-sigma coherence output path.

    ``output_dir/{date1}_{date2}/{variant}_phsig.coh[.tif]`` when the input
    lives in a date-pair directory; ``output_dir/{base}_phsig.coh[.tif]``
    (base = stem without .int[.tif]) otherwise (flat/standalone usage).
    """
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    date_pair = input_path.parent.name
    if is_date_pair_dir(date_pair):
        dp = extract_date_pair(date_pair)
        date1, date2 = dp.split('_')
        variant = variant_of(input_path, processor)
        return coh_path(output_dir, date1, date2, variant=variant,
                        kind='phsig', processor=processor)

    # Flat fallback (standalone usage): preserve the original stem
    name = input_path.name
    if name.endswith('.int.tif'):
        base = name[:-8]
    elif name.endswith('.int'):
        base = name[:-4]
    elif name.endswith('.unw.tif'):
        base = name[:-8]
    elif name.endswith('.unw'):
        base = name[:-4]
    else:
        base = input_path.stem
    ext = coh_ext(processor, 'phsig')
    return output_dir / f"{base}{ext}"


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
def process_single_file(input_file, output_dir, params, processor):
    logger = logging.getLogger('insarflow.phsig')
    try:
        input_path = Path(input_file)

        if params.get('keep_sigma'):
            logger.warning(
                "keep_sigma is set but the sigma (phase standard deviation) "
                "product is NOT implemented in the current backend — only "
                "the coherence product is written")

        output_file = _output_path(input_file, output_dir, processor)

        # skip on the coherence product only (sigma is not produced)
        if output_file.exists():
            logger.info("Skipping existing: %s", output_file)
            return input_file, True, f"Exists: {output_file.name}"

        logger.info("Processing: %s", input_path.name)
        start_time = time.time()

        ifg, meta = read_complex_image(input_file, processor)
        logger.info("Data shape: %s", ifg.shape)

        coh = estimate_phsig_correlation(
            ifg,
            ps_win=params['phase_sigma_window'],
            grad_win=params['gradient_window'],
            nlks=params['nlks'],
        )

        _write_band(str(output_file), coh, meta, processor, 'phase-sigma correlation')
        if processor == 'isce2':
            create_xml_for_binary(output_file, family='image',
                                  description='Phase-sigma correlation')


        elapsed = time.time() - start_time
        logger.info("Completed in %.1fs: %s", elapsed, output_file.name)

        valid = coh > 0
        if np.any(valid):
            logger.info("  Stats: mean=%.3f, min=%.3f, max=%.3f",
                        np.mean(coh[valid]), np.min(coh[valid]), np.max(coh[valid]))

        return input_file, True, f"Done in {elapsed:.1f}s"
    except Exception as exc:
        logger.error("Error processing %s: %s", input_file, exc)
        return input_file, False, f"Error: {exc}"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_arguments(args_list=None):
    parser = argparse.ArgumentParser(
        description='Phase-sigma correlation estimator (ISCE2 ph_slope.F + ph_sigma.F)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  generate_coh_phsig.py --processor isce3 --input ./filtered/*.int.tif --output-dir ./coh
  generate_coh_phsig.py --processor isce3 --input ./ifgs/*.int.tif --ps-window 5 --nlks 3.0
        """)
    parser.add_argument('--processor', choices=['isce2', 'isce3'], required=True)
    parser.add_argument('--input', type=str, nargs='+', required=True,
                        help='Interferogram file(s) or glob pattern')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory')
    parser.add_argument('--window-size', '--ps-window-size', type=int, default=5,
                        help='Phase-sigma window size (default: 5)')
    parser.add_argument('--gradient-window', '--ps-gradient-window', type=int, default=5,
                        help='Gradient estimation window size (default: 5)')
    parser.add_argument('--nlks', '--ps-nlks', type=float, default=1.0,
                        help='Number of looks (default: 1.0)')
    parser.add_argument('--keep-sigma', action='store_true',
                        help='Also write phase standard deviation')
    parser.add_argument('--max-workers', type=int, default=1)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--log-file', type=str, help='Write log to file')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_arguments()

    log_level = logging.ERROR if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    setup_logging(verbose=(log_level == logging.DEBUG), log_file=args.log_file)
    logger = logging.getLogger('insarflow.phsig')
    logger.setLevel(log_level)

    input_files = (list(args.input)
                   if not any(c in str(args.input) for c in "*?")
                   else sum((glob.glob(p, recursive=True) for p in args.input), []))
    if not input_files:
        logger.error("No files match pattern: %s", args.input)
        return 1

    logger.info("Found %d input file(s)", len(input_files))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params = {
        'phase_sigma_window': args.window_size,
        'gradient_window': args.gradient_window,
        'nlks': args.nlks,
        'keep_sigma': args.keep_sigma,
    }

    successful = 0
    failed = 0

    if args.max_workers > 1 and len(input_files) > 1:
        logger.info("Using %d parallel workers (threads)", args.max_workers)
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            results = [
                (Path(f).name, executor.submit(process_single_file, f, output_dir, params, args.processor))
                for f in input_files
            ]
            for name, future in results:
                try:
                    _, success, msg = future.result()
                    if success:
                        logger.info("SUCCESS %s: %s", name, msg)
                        successful += 1
                    else:
                        logger.error("FAILED %s: %s", name, msg)
                        failed += 1
                except Exception as exc:
                    logger.error("FAILED %s: %s", name, exc)
                    failed += 1
    else:
        for f in input_files:
            _, success, msg = process_single_file(f, output_dir, params, args.processor)
            if success:
                successful += 1
            else:
                failed += 1

    logger.info("Complete: %d OK, %d failed", successful, failed)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
