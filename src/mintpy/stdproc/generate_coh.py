#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Coherence products for the slc2ifg pipeline - one module, two estimators.

* **complex coherence (cpx)** - magnitude of the complex correlation
  coefficient from a pair of SLCs, with an ISCE2-compatible Bartlett window;
* **phase-sigma coherence (phsig)** - phase standard deviation estimator
  matching the ISCE2 Fortran ``ph_slope.F`` / ``ph_sigma.F``, computed from a
  complex interferogram; optionally also writes the phase std-dev raster
  (``keep_sigma``).

Fixed product structure (see ``utils.naming``)::

    output_dir/{date1}_{date2}/xxx.cpx.coh.tif
    output_dir/{date1}_{date2}/xxx.phsig.coh.tif
    output_dir/{date1}_{date2}/xxx.phsig.sigma.tif   (keep_sigma only)

All products are GeoTIFF for both processors; isce2 radar-coordinate products
carry no georeferencing.  Raster I/O goes through :mod:`mintpy.stdproc.io`.

This is a plain implementation library: no CLI and no logging configuration
(the command line lives in ``mintpy/cli/generate_coh.py``).
"""

from __future__ import annotations

import glob
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import correlate

from . import io as sio
from .utils.naming import (
    coh_path,
    extract_date_pair,
    is_date_pair_dir,
    variant_of,
)

logger = logging.getLogger(__name__)

#: complex-coherence defaults
DEFAULT_PARAMS = {
    'window_size': 5,
    #: spatial weighting of the estimation window:
    #:   'triangular' (default) - Bartlett weights, identical to the ISCE2
    #:       `cchz_wave.cpp` estimator: w(i) = 1 - |2*(i - n//2)/(n+1)|
    #:   'uniform'    - plain boxcar (equal weights)
    'window_type': 'triangular',
    'use_amplitude': True,
}

#: supported coherence window weightings
WINDOW_TYPES = ('triangular', 'uniform')

#: approximate number of window-pixels processed per row-batch (phsig)
_BATCH_WINDOW_PIXELS = 2_000_000


# ===========================================================================
# complex coherence (SLC pair)
# ===========================================================================
def coherence_kernel(window_size: int, window_type: str = 'triangular'):
    """Normalized 2-D spatial weighting kernel for coherence estimation.

    ``triangular`` reproduces the ISCE2 ``mroipac/correlation`` Bartlett
    weighting ``w(i) = 1 - |2*(i - n//2)/(n+1)|`` as a separable outer
    product (the coherence is a ratio, so the normalization cancels; it is
    applied here to keep the sums numerically well behaved).
    """
    if window_type not in WINDOW_TYPES:
        raise ValueError(
            f"Unknown window_type '{window_type}', expected one of {WINDOW_TYPES}")
    n = int(window_size)
    if n % 2 == 0:
        n += 1
    if window_type == 'uniform':
        kernel = np.ones((n, n), dtype=np.float32)
    else:
        i = np.arange(n, dtype=np.float64)
        w = 1.0 - np.abs(2.0 * (i - n // 2) / (n + 1.0))
        kernel = np.outer(w, w)
    return (kernel / kernel.sum()).astype(np.float32)


class CoherenceEstimator:
    """Coherence estimator from two SLCs (complex correlation magnitude).

    The boxcar coherence is computed with ``scipy.ndimage.correlate`` (C level),
    which is O(N) per pixel instead of O(window^2).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        self.params = DEFAULT_PARAMS.copy()
        if params:
            self.params.update(params)

        self.win_size = self.params['window_size']
        self.window_type = self.params.get('window_type', 'triangular')
        if self.win_size % 2 == 0:
            self.win_size += 1
        self.half_win = self.win_size // 2
        self.kernel = coherence_kernel(self.win_size, self.window_type)

    def compute_coherence(self, slc1: np.ndarray, slc2: np.ndarray,
                          mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Coherence map from two SLC images (2D float32 in [0, 1])."""
        if slc1.shape != slc2.shape:
            raise ValueError(f"SLC shape mismatch: {slc1.shape} vs {slc2.shape}")

        rows, cols = slc1.shape
        start_time = time.time()

        interferogram = slc1 * np.conj(slc2)
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

        # borders of half_win are not estimated
        if self.half_win > 0:
            coherence[:self.half_win, :] = 0.0
            coherence[-self.half_win:, :] = 0.0
            coherence[:, :self.half_win] = 0.0
            coherence[:, -self.half_win:] = 0.0

        if mask is not None:
            coherence[~mask] = 0.0

        self.logger.debug("coherence computed in %.1fs", time.time() - start_time)
        return coherence

    def process(self, slc1: np.ndarray, slc2: np.ndarray,
                mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Public interface for coherence estimation from two SLCs."""
        return self.compute_coherence(slc1, slc2, mask)


# ===========================================================================
# phase-sigma coherence (interferogram)
# ===========================================================================
def _gaussian_kernel(size):
    """Gaussian kernel matching the ISCE2 Fortran ph_slope.F / ph_sigma.F."""
    half = size // 2
    s1 = 0.0
    kernel = np.zeros((size, size), dtype=np.float64)
    for k in range(size):
        for j in range(size):
            w1 = (k - half) ** 2 + (j - half) ** 2
            kernel[k, j] = np.exp(-w1 / (size / 2.0))
            s1 += kernel[k, j]
    return (kernel / s1).astype(np.float32)


def estimate_phsig_correlation(
    ifg_arr: np.ndarray,
    ps_win: int = 5,
    grad_win: int = 5,
    nlks: float = 1.0,
    return_sigma: bool = False,
):
    """Estimate phase-sigma correlation from a complex interferogram.

    Matches ISCE2 Fortran ``ph_slope.F`` + ``ph_sigma.F``: Gaussian-weighted
    phase gradient estimation, local window deramping, weighted phase variance
    and the NLKS-based correlation conversion.

    Parameters
    ----------
    ifg_arr : np.ndarray (complex)
        Complex interferogram ``[rows, cols]``.
    ps_win, grad_win : int
        Phase-sigma / gradient estimation window sizes (odd).
    nlks : float
        Number of looks.
    return_sigma : bool
        Also return the phase standard deviation raster (``sqrt(variance)``).

    Returns
    -------
    coh : np.ndarray (float32), clipped to [0, 1]
    sigma : np.ndarray (float32), only when ``return_sigma`` is True
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

    # ISCE2 Fortran valid range: [half+1, size-half-1] -> zero edges
    if grad_half > 0:
        for sl in (rg_slope, az_slope):
            sl[:grad_half + 1, :] = 0.0
            sl[-(grad_half):, :] = 0.0
            sl[:, :grad_half + 1] = 0.0
            sl[:, -(grad_half):] = 0.0

    # Valid (interior) output region
    i0, i1 = ps_half, rows - ps_half
    j0, j1 = ps_half, cols - ps_half
    coh = np.zeros((rows, cols), dtype=np.float32)
    sigma = np.zeros((rows, cols), dtype=np.float32) if return_sigma else None
    if i1 <= i0 or j1 <= j0:
        return (coh, sigma) if return_sigma else coh

    offsets = np.arange(-ps_half, ps_half + 1)
    di_mesh, dj_mesh = np.meshgrid(offsets, offsets, indexing='ij')
    ps_weights = _gaussian_kernel(ps_win)

    # All windows as a strided *view*: (rows-ps_win+1, cols-ps_win+1, ps_win, ps_win)
    windows = sliding_window_view(ifg_arr, (ps_win, ps_win))

    n_rows_out = i1 - i0
    n_cols_out = j1 - j0

    # Row-batched processing to bound peak memory: the per-pixel deramping
    # ramps broadcast over the window axes are materialized per batch only.
    win_pixels = n_cols_out * ps_win * ps_win
    batch_rows = max(1, _BATCH_WINDOW_PIXELS // max(win_pixels, 1))

    for r0 in range(0, n_rows_out, batch_rows):
        r1 = min(r0 + batch_rows, n_rows_out)

        win_block = windows[r0:r1]
        ramp_block = (
            di_mesh[None, None, :, :] *
            az_slope[i0 + r0:i0 + r1, j0:j1][:, :, None, None] +
            dj_mesh[None, None, :, :] *
            rg_slope[i0 + r0:i0 + r1, j0:j1][:, :, None, None]
        )

        exp_ramp = np.cos(ramp_block) - 1j * np.sin(ramp_block)
        comp = win_block * exp_ramp

        wsum = np.sum(comp, axis=(2, 3))
        mag = np.abs(wsum)

        mag_valid = mag > 1e-10
        if np.any(mag_valid):
            norm_sum = np.zeros_like(wsum)
            norm_sum[mag_valid] = wsum[mag_valid] / mag[mag_valid]
            deramped = comp * np.conj(norm_sum[:, :, None, None])

            phases = np.arctan2(deramped.imag, deramped.real)
            wt = ps_weights[None, None, :, :]
            # accumulate the variance sums in float64, matching the GPU kernel
            phases64 = phases.astype(np.float64)
            wt64 = wt.astype(np.float64)
            mean_ph = np.sum(wt64 * phases64, axis=(2, 3))
            mean_ph2 = np.sum(wt64 * phases64 * phases64, axis=(2, 3))
            var = mean_ph2 - mean_ph * mean_ph

            with np.errstate(divide='ignore', invalid='ignore'):
                val = np.where(var > 0,
                               1.0 / np.sqrt(2.0 * nlks * var + 1.0),
                               1.0)
            val = np.where(mag_valid, val, 0.0)
            coh[i0 + r0:i0 + r1, j0:j1] = val

            if sigma is not None:
                sig = np.where(mag_valid, np.sqrt(np.maximum(var, 0.0)), 0.0)
                sigma[i0 + r0:i0 + r1, j0:j1] = sig

    coh = np.clip(coh, 0.0, 1.0)
    if return_sigma:
        return coh, sigma
    return coh


# ===========================================================================
# file I/O
# ===========================================================================
def read_complex_image(filename: str, processor: str,
                       subdataset: Optional[str] = None,
                       window: Optional[Tuple[int, int, int, int]] = None,
                       ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Read a complex SLC / interferogram with processor-specific awareness.

    ``subdataset`` selects the HDF5 dataset for ``.h5`` inputs (OPERA CSLC
    style).  ``window`` ``(x0, y0, w, h)`` reads *only* that pixel window (the
    read-time AOI crop); the returned metadata then carries the window's own
    size and geotransform.
    """
    ext = Path(filename).suffix.lower()
    if processor == 'isce2':
        expected = ['.slc', '.int', '.rdr', '.full', '.tif', '.tiff']
    else:
        expected = ['.tif', '.tiff', '.h5', '.hdf5']
    if ext not in expected:
        logger.warning("Processor '%s' expects %s, but got '%s'",
                       processor, expected, ext)

    ds = sio.open_raster(filename, subdataset if sio.is_hdf5_file(filename) else None)
    if ds is None:
        raise ValueError(f"Cannot open file: {filename}")

    gt = sio.get_geotransform(ds)
    full_rows, full_cols = ds.RasterYSize, ds.RasterXSize
    if window is not None:
        x0, y0, w, h = (int(v) for v in window)
        if x0 < 0 or y0 < 0 or x0 >= full_cols or y0 >= full_rows:
            raise ValueError(
                f"window {window} is outside {filename} ({full_cols}x{full_rows})")
        w = max(1, min(w, full_cols - x0))
        h = max(1, min(h, full_rows - y0))
        if gt is not None:
            # shift the geotransform to the window origin (x0, y0)
            gt = (gt[0] + x0 * gt[1] + y0 * gt[2], gt[1], gt[2],
                  gt[3] + x0 * gt[4] + y0 * gt[5], gt[4], gt[5])
    else:
        x0 = y0 = 0
        w, h = full_cols, full_rows

    def _read(band):
        if window is None:
            return band.ReadAsArray()
        return band.ReadAsArray(x0, y0, w, h)

    metadata = {
        'transform': gt,
        'projection': ds.GetProjection() if gt is not None else '',
        'rows': h,
        'cols': w,
        'band_count': ds.RasterCount,
        'window': None if window is None else (x0, y0, w, h),
    }

    if metadata['band_count'] == 1:
        data = _read(ds.GetRasterBand(1))
        if data.dtype not in (np.complex64, np.complex128):
            logger.warning(
                "%s: band 1 is real-valued (dtype %s) - casting to complex "
                "with zero imaginary part; verify the product is really "
                "complex data", filename, data.dtype)
            data = data.astype(np.complex64)
    elif metadata['band_count'] == 2:
        real = _read(ds.GetRasterBand(1))
        imag = _read(ds.GetRasterBand(2))
        data = (real + 1j * imag).astype(np.complex64)
    else:
        raise ValueError(f"Unsupported band count: {metadata['band_count']}")

    ds = None
    return data, metadata


def write_coherence_image(filename: str, coherence: np.ndarray,
                          metadata: Dict[str, Any], processor: str,
                          description: str = '') -> str:
    """Write a single-band coherence (or sigma) raster as a GeoTIFF.

    isce2 radar-coordinate products are written WITHOUT georeferencing.
    """
    gt = metadata.get('transform')
    epsg = None
    if processor != 'isce2' and gt is not None:
        proj = metadata.get('projection') or ''
        epsg = sio.epsg_from_projection(proj) if proj else None

    meta: Dict[str, Any] = {'FILE_TYPE': '.cor'}
    geo = False
    if epsg:
        meta.update({
            'X_FIRST': gt[0], 'Y_FIRST': gt[3],
            'X_STEP': abs(gt[1]), 'Y_STEP': gt[5], 'EPSG': epsg,
        })
        geo = True

    return sio.write_product(
        np.asarray(coherence, dtype=np.float32), filename, processor=processor,
        meta=meta, geo=geo, compress='LZW', tiled=True, nodata=0.0)


# ===========================================================================
# discovery / naming helpers
# ===========================================================================
def extract_date_from_slc(filename: str) -> str:
    """Extract a YYYYMMDD date from an SLC filename (validated when possible)."""
    name = Path(filename).name
    for suffix in ('.slc.tif', '.slc.h5', '.slc', '.tif', '.tiff'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break

    match = re.search(r'(\d{8})', name)
    if match:
        candidate = match.group(1)
        try:
            from datetime import datetime as _dt
            _dt.strptime(candidate, '%Y%m%d')
            return candidate
        except ValueError:
            logger.warning("8-digit sequence %r in %s is not a valid date",
                           candidate, Path(filename).name)
    return name


def expand_directories(directory_patterns) -> List[Path]:
    """Expand directory patterns into a list of directories."""
    expanded: List[Path] = []
    for pattern in directory_patterns:
        matches = glob.glob(pattern, recursive=False)
        if not matches:
            logger.warning("No directories found matching pattern: %s", pattern)
        expanded.extend(Path(m) for m in matches)
    return expanded


def find_slc_file_by_date(slc_directories, target_date, slc_pattern,
                          processor: str = 'isce3') -> Optional[Path]:
    """Find the SLC file containing ``target_date``.

    ``slc_pattern`` may be a single glob or a sequence of globs (tried in
    order), so a directory holding either raw inputs or the pipeline's own
    ``*.slc.tif`` products can be searched with one call.
    """
    patterns = (slc_pattern,) if isinstance(slc_pattern, str) else tuple(slc_pattern)
    for slc_dir in slc_directories:
        for pat in patterns:
            glob_pat = f"*{target_date}{pat}"
            matches = list(Path(slc_dir).glob(glob_pat)) + \
                list(Path(slc_dir).glob(f"*/{glob_pat}"))
            if matches:
                return sorted(matches)[0]
    return None


def _detect_burst_dirs(slc_dirs) -> Dict[Optional[str], List[Path]]:
    """Detect burst subdirectories under the given SLC directory list."""
    burst_pattern = re.compile(r'^t\d+_\d+_iw\d+$')
    burst_map: Dict[Optional[str], List[Path]] = {}
    for slc_dir in slc_dirs:
        if Path(slc_dir).is_dir():
            for entry in sorted(Path(slc_dir).iterdir()):
                if entry.is_dir() and burst_pattern.match(entry.name):
                    burst_map[entry.name] = [entry]
    if burst_map:
        logger.info("Detected %d burst(s): %s", len(burst_map), list(burst_map.keys()))
        return burst_map
    return {None: list(slc_dirs)}


def cpx_coh_path(output_dir, slc1_file: str, slc2_file: str, processor: str) -> Path:
    """Canonical ``fullres.cpx.coh.tif`` path for an SLC pair."""
    date1 = extract_date_from_slc(slc1_file)
    date2 = extract_date_from_slc(slc2_file)
    return coh_path(output_dir, date1, date2, variant='fullres',
                    kind='cpx', processor=processor)


def phsig_coh_path(input_file, output_dir, processor: str) -> Path:
    """Canonical phase-sigma coherence output path.

    ``output_dir/{date1}_{date2}/{variant}.phsig.coh.tif`` when the input lives
    in a date-pair directory, else a flat ``output_dir/{base}.phsig.coh.tif``
    (standalone usage).
    """
    input_path = Path(input_file)
    output_dir = Path(output_dir)
    date_pair = input_path.parent.name
    if is_date_pair_dir(date_pair):
        date1, date2 = extract_date_pair(date_pair).split('_')
        variant = variant_of(input_path, processor)
        return coh_path(output_dir, date1, date2, variant=variant,
                        kind='phsig', processor=processor)

    name = input_path.name
    for suffix in ('.int.tif', '.int', '.unw.tif', '.unw', '.tif', '.tiff'):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    base = name or input_path.stem
    # include the parent directory name: same-stemmed inputs from different
    # directories (e.g. per-burst flat layouts) must not collide
    parent = input_path.parent.name
    prefix = f"{parent}_" if parent else ""
    return output_dir / f"{prefix}{base}.phsig.coh.tif"


def phsig_sigma_path(input_file, output_dir, processor: str) -> Path:
    """Canonical phase-sigma (std-dev) path, mirroring :func:`phsig_coh_path`."""
    coh_out = phsig_coh_path(input_file, output_dir, processor)
    return coh_out.with_name(coh_out.name.replace('.phsig.coh.tif', '.phsig.sigma.tif'))


# ===========================================================================
# per-pair / per-file processing
# ===========================================================================
def process_slc_pair(slc1_file: str, slc2_file: str, output_dir: Path,
                     params: Dict[str, Any], processor: str,
                     subdataset: Optional[str] = None) -> Tuple[str, str, bool, str]:
    """Process one SLC pair and write the complex-coherence product."""
    date1 = extract_date_from_slc(slc1_file)
    date2 = extract_date_from_slc(slc2_file)

    try:
        output_file = cpx_coh_path(output_dir, slc1_file, slc2_file, processor)
        if output_file.exists():
            return date1, date2, True, f"Exists: {output_file.name}"

        start_time = time.time()
        slc1, meta1 = read_complex_image(slc1_file, processor, subdataset=subdataset)
        slc2, _ = read_complex_image(slc2_file, processor, subdataset=subdataset)
        if slc1.shape != slc2.shape:
            raise ValueError(f"Dimension mismatch: {slc1.shape} vs {slc2.shape}")

        estimator = CoherenceEstimator(params)
        coherence = np.clip(estimator.process(slc1, slc2), 0.0, 1.0)
        write_coherence_image(str(output_file), coherence, meta1, processor,
                              'complex correlation magnitude')

        elapsed = time.time() - start_time
        valid = coherence > 0
        if np.any(valid):
            logger.info("  %s: mean=%.3f min=%.3f max=%.3f",
                        output_file.name, float(np.mean(coherence[valid])),
                        float(np.min(coherence[valid])), float(np.max(coherence[valid])))
        return date1, date2, True, f"Done in {elapsed:.1f}s"
    except Exception as exc:                                  # noqa: BLE001
        logger.error("Error processing %s - %s: %s", slc1_file, slc2_file, exc)
        return date1, date2, False, f"Error: {exc}"


def process_phsig_file(input_file: str, output_dir: Path,
                       params: Dict[str, Any], processor: str,
                       subdataset: Optional[str] = None) -> Tuple[str, bool, str]:
    """Process one interferogram and write the phase-sigma coherence (+ sigma)."""
    try:
        output_file = phsig_coh_path(input_file, output_dir, processor)
        sigma_file = phsig_sigma_path(input_file, output_dir, processor)
        keep_sigma = bool(params.get('keep_sigma'))

        want = [output_file] + ([sigma_file] if keep_sigma else [])
        if all(p.exists() for p in want):
            return input_file, True, f"Exists: {output_file.name}"

        start_time = time.time()
        ifg, meta = read_complex_image(input_file, processor, subdataset=subdataset)
        sigma = None
        if keep_sigma:
            coh, sigma = estimate_phsig_correlation(
                ifg,
                ps_win=params['phase_sigma_window'],
                grad_win=params['gradient_window'],
                nlks=params['nlks'],
                return_sigma=True,
            )
        else:
            coh = estimate_phsig_correlation(
                ifg,
                ps_win=params['phase_sigma_window'],
                grad_win=params['gradient_window'],
                nlks=params['nlks'],
            )
        write_coherence_image(str(output_file), coh, meta, processor,
                              'phase-sigma correlation')
        if keep_sigma:
            write_coherence_image(str(sigma_file), sigma, meta, processor,
                                  'phase standard deviation')

        elapsed = time.time() - start_time
        valid = coh > 0
        if np.any(valid):
            logger.info("  %s: mean=%.3f min=%.3f max=%.3f",
                        output_file.name, float(np.mean(coh[valid])),
                        float(np.min(coh[valid])), float(np.max(coh[valid])))
        return input_file, True, f"Done in {elapsed:.1f}s"
    except Exception as exc:                                  # noqa: BLE001
        logger.error("Error processing %s: %s", input_file, exc)
        return input_file, False, f"Error: {exc}"


def _run_parallel(tasks, worker, max_workers, what):
    """Run ``worker(task)`` over tasks, returning (n_ok, n_failed)."""
    n_workers = max(1, min(int(max_workers or 1), 8, max(1, len(tasks))))
    results = []
    if n_workers == 1 or len(tasks) <= 1:
        results = [worker(t) for t in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            results = list(pool.map(worker, tasks))

    n_ok = sum(1 for r in results if (r[2] if len(r) == 4 else r[1]))
    n_bad = len(results) - n_ok
    logger.info("%s complete: %d ok, %d failed", what, n_ok, n_bad)
    return n_ok, n_bad


# ===========================================================================
# public entry points
# ===========================================================================
def generate_complex_coherence(pairs_file, slc_dir, output_dir, processor='isce3',
                               slc_pattern=None, window_size=5,
                               window_type='triangular', max_workers=1,
                               subdataset=None) -> int:
    """Generate complex coherence for every pair in ``pairs_file``."""
    patterns = slc_pattern if slc_pattern else ('*.slc.tif', '*.slc.*', '*.slc')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    slc_directories = expand_directories(
        slc_dir if isinstance(slc_dir, (list, tuple)) else [slc_dir])
    if not slc_directories:
        logger.error("No SLC directories found.")
        return 1

    try:
        pairs_df = pd.read_csv(pairs_file, comment='#', sep=r'\s+', names=['date12'])
    except Exception as exc:                                  # noqa: BLE001
        logger.error("Error reading pairs file %s: %s", pairs_file, exc)
        return 1
    logger.info("Found %d pairs in %s", len(pairs_df), pairs_file)

    params = {'window_size': window_size, 'window_type': window_type,
              'use_amplitude': DEFAULT_PARAMS['use_amplitude']}

    tasks = []
    for burst_id, burst_dirs in _detect_burst_dirs(slc_directories).items():
        b_out = output_dir / burst_id if burst_id else output_dir
        b_out.mkdir(parents=True, exist_ok=True)
        for _, row in pairs_df.iterrows():
            date1, date2 = row['date12'].split('-')
            slc1 = find_slc_file_by_date(burst_dirs, date1, patterns, processor)
            slc2 = find_slc_file_by_date(burst_dirs, date2, patterns, processor)
            if slc1 is None or slc2 is None:
                logger.warning("SLC not found for pair %s_%s, skipping", date1, date2)
                continue
            tasks.append((str(slc1), str(slc2), b_out))

    if not tasks:
        logger.error("No valid SLC pairs found.")
        return 1

    _, n_bad = _run_parallel(
        tasks,
        lambda t: process_slc_pair(t[0], t[1], t[2], params, processor, subdataset),
        max_workers, 'complex coherence')
    return 0 if n_bad == 0 else 1


def generate_phsig_coherence(input_files, output_dir, processor='isce3',
                             phase_sigma_window=5, gradient_window=5,
                             nlks=1.0, keep_sigma=False, max_workers=1,
                             subdataset=None) -> int:
    """Generate the phase-sigma coherence (and optionally the sigma raster)."""
    patterns = input_files if isinstance(input_files, (list, tuple)) else [input_files]
    files: List[str] = []
    for pat in patterns:
        matched = sorted(glob.glob(str(pat), recursive=True))
        files.extend(matched if matched else [str(pat)])

    files = [f for f in files if Path(f).is_file()]
    if not files:
        logger.error("No files match pattern: %s", input_files)
        return 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    params = {
        'phase_sigma_window': phase_sigma_window,
        'gradient_window': gradient_window,
        'nlks': nlks,
        'keep_sigma': keep_sigma,
    }

    _, n_bad = _run_parallel(
        files,
        lambda f: process_phsig_file(f, output_dir, params, processor, subdataset),
        max_workers, 'phase-sigma coherence')
    return 0 if n_bad == 0 else 1


def generate_coh(processor='isce3', input_files=None, output_dir='.',
                 pairs_file=None, slc_dir=None, slc_pattern=None,
                 ps_window_size=5, ps_gradient_window=5, ps_nlks=1.0,
                 cc_window_size=5, cc_window_type='triangular',
                 keep_sigma=False, max_workers=1, subdataset=None) -> int:
    """Unified coherence generation: phase-sigma and/or complex coherence.

    Each estimator runs when its inputs are given: the phase-sigma estimator
    works on interferograms (``input_files``), the complex estimator needs the
    SLC pairs (``pairs_file`` + ``slc_dir``).  *Which* coherence products are
    kept is decided by the caller (engine.stages / engine.keep_intermediates),
    not by flags here.
    """
    if not input_files and not (pairs_file and slc_dir):
        logger.error(
            "nothing to do: pass input_files (phase-sigma) and/or "
            "pairs_file+slc_dir (complex coherence)")
        return 1

    exit_code = 0
    if input_files:
        logger.info("--- phase-sigma coherence ---")
        ret = generate_phsig_coherence(
            input_files, output_dir, processor=processor,
            phase_sigma_window=ps_window_size, gradient_window=ps_gradient_window,
            nlks=ps_nlks, keep_sigma=keep_sigma, max_workers=max_workers,
            subdataset=subdataset)
        exit_code = exit_code or ret

    if pairs_file and slc_dir:
        logger.info("--- complex coherence ---")
        ret = generate_complex_coherence(
            pairs_file, slc_dir, output_dir, processor=processor,
            slc_pattern=slc_pattern, window_size=cc_window_size,
            window_type=cc_window_type, max_workers=max_workers,
            subdataset=subdataset)
        exit_code = exit_code or ret

    return exit_code
