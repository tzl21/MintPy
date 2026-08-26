#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
GPU compute kernels for the MintPy slc2ifg engine (design M3).

Each kernel is a drop-in replacement for the corresponding CPU
implementation with an identical interface (numpy arrays in, numpy arrays
out) and an automatic CPU fallback when CuPy is unavailable or the caller
disables GPU.  Only the window/FFT-based numerics are GPU-accelerated —
file I/O stays on the CPU.

Kernels:
- ``complex_coh_block`` — boxcar complex coherence (windowed sums)
- ``estimate_phsig_block`` — phase-sigma correlation (gradients + windowed
  deramping), mirroring ``generate_coh_phsig.estimate_phsig_correlation``
- ``goldstein_block`` — Goldstein filter on one padded block with a patch
  grid anchored to the *full-image* coordinate system (bit-identical to
  ``filter_utils.goldstein``), with batched GPU FFTs.

All kernels allocate in **bounded batches** (per-row-batch deramping ramps,
per-chunk patch FFTs sized from the free VRAM), so peak device memory is a
small multiple of the block size rather than scaling with
``image_pixels * window^2``.  GPU failures — including ``OutOfMemoryError`` —
are counted (``gpu_fallback_count()``), reset the CuPy pools
(``_reset_cuda_pools()``, otherwise a bloated pool poisons every later GPU
task) and fall back to the CPU implementation.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _ensure_cuda_path() -> None:
    """Set ``CUDA_PATH`` for CuPy when running inside a conda env that ships
    the CUDA runtime libraries (CuPy needs it for cuFFT/cuBLAS plans)."""
    if os.environ.get('CUDA_PATH'):
        return
    candidates = []
    if getattr(sys, 'prefix', None):
        candidates.append(sys.prefix)
    if getattr(sys, 'base_prefix', None):
        candidates.append(sys.base_prefix)
    exe_dir = os.path.dirname(os.path.dirname(sys.executable))
    if exe_dir not in candidates:
        candidates.append(exe_dir)
    for c in candidates:
        libdir = os.path.join(c, 'lib')
        try:
            if any(f.startswith('libcufft') for f in os.listdir(libdir)):
                os.environ['CUDA_PATH'] = c
                logger.debug("CUDA_PATH set to %s", c)
                return
        except OSError:
            continue


try:
    import cupy as cp
    _ensure_cuda_path()
    from cupy.lib.stride_tricks import sliding_window_view as cp_sliding_window_view
    import cupyx.scipy.ndimage as cp_ndimage
    _CUPY = True
except ImportError:  # pragma: no cover - CPU-only environments
    _CUPY = False


def cupy_available() -> bool:
    """True if CuPy is importable and a device is usable."""
    if not _CUPY:
        return False
    try:
        _ensure_cuda_path()
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


#: Number of GPU kernel calls that failed and fell back to CPU (observability).
_FALLBACK_COUNT = 0


def gpu_fallback_count() -> int:
    """Number of GPU kernel calls that failed and fell back to CPU."""
    return _FALLBACK_COUNT


def _reset_cuda_pools() -> None:
    """Free CuPy memory pools after a GPU failure.

    An OOM can leave the pool holding every block it ever allocated; without
    a reset every later GPU task keeps failing and the whole run silently
    degrades to CPU.  Safe to call from any thread: only cached (unreferenced)
    blocks are freed, live arrays are untouched.
    """
    try:
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _note_fallback(name: str, exc: Exception) -> None:
    """Record one GPU->CPU fallback, reset CuPy pools, log with a counter."""
    global _FALLBACK_COUNT
    _FALLBACK_COUNT += 1
    _reset_cuda_pools()
    logger.warning("GPU %s failed (%s) — falling back to CPU "
                   "(GPU fallbacks so far: %d)", name, exc, _FALLBACK_COUNT)


def _gaussian_kernel_cp(size: int):
    """Gaussian kernel matching ``generate_coh_phsig._gaussian_kernel``."""
    half = size // 2
    idx = cp.arange(size, dtype=cp.float64)
    w1 = (idx[:, None] - half) ** 2 + (idx[None, :] - half) ** 2
    kernel = cp.exp(-w1 / (size / 2.0))
    return (kernel / kernel.sum()).astype(cp.float32)


# ------------------------------------------------------------------------
# Complex coherence
# ------------------------------------------------------------------------
def complex_coh_block(slc1: np.ndarray, slc2: np.ndarray, window: int,
                      gpu: bool = False) -> np.ndarray:
    """Boxcar complex coherence of two complex SLC blocks -> float32.

    Bit-identical semantics to ``CoherenceEstimator.compute_coherence``
    (borders of ``window//2`` are zeroed by the caller via ``zero_margin``).
    """
    if gpu and cupy_available():
        try:
            return _complex_coh_gpu(slc1, slc2, window)
        except Exception as e:
            _note_fallback('complex coherence', e)
    return _complex_coh_cpu(slc1, slc2, window)


def _complex_coh_cpu(slc1: np.ndarray, slc2: np.ndarray, window: int):
    from scipy.ndimage import correlate
    win = window if window % 2 else window + 1
    kernel = np.ones((win, win), dtype=np.float32) / (win * win)
    ifg = slc1 * np.conj(slc2)
    mag_sq1 = (np.abs(slc1) ** 2).astype(np.float32)
    mag_sq2 = (np.abs(slc2) ** 2).astype(np.float32)
    sum_ifg = correlate(ifg, kernel, mode='constant', cval=0.0)
    denom = np.sqrt(correlate(mag_sq1, kernel, mode='constant', cval=0.0) *
                    correlate(mag_sq2, kernel, mode='constant', cval=0.0))
    coh = np.zeros(slc1.shape, dtype=np.float32)
    valid = denom > 0
    coh[valid] = np.abs(sum_ifg[valid]) / denom[valid]
    return np.clip(coh, 0.0, 1.0)


def _complex_coh_gpu(slc1: np.ndarray, slc2: np.ndarray, window: int):
    win = window if window % 2 else window + 1
    kernel = cp.ones((win, win), dtype=cp.float32) / (win * win)
    s1 = cp.asarray(slc1)
    s2 = cp.asarray(slc2)
    ifg = s1 * cp.conj(s2)
    sum_ifg = cp_ndimage.correlate(ifg, kernel, mode='constant', cval=0)
    mag1 = cp_ndimage.correlate((cp.abs(s1) ** 2).astype(cp.float32), kernel,
                                mode='constant', cval=0)
    mag2 = cp_ndimage.correlate((cp.abs(s2) ** 2).astype(cp.float32), kernel,
                                mode='constant', cval=0)
    denom = cp.sqrt(mag1 * mag2)
    coh = cp.zeros(slc1.shape, dtype=cp.float32)
    valid = denom > 0
    coh[valid] = cp.abs(sum_ifg[valid]) / denom[valid]
    return cp.asnumpy(cp.clip(coh, 0.0, 1.0))


# ------------------------------------------------------------------------
# Phase-sigma correlation
# ------------------------------------------------------------------------
def estimate_phsig_block(ifg_arr: np.ndarray, ps_win: int = 5,
                         grad_win: int = 5, nlks: float = 1.0,
                         gpu: bool = False) -> np.ndarray:
    """Phase-sigma correlation of a complex block -> float32.

    Mirrors ``generate_coh_phsig.estimate_phsig_correlation``.  Border
    zeroing matches the full-image run via the caller's ``zero_margin``.
    """
    if gpu and cupy_available():
        try:
            return _phsig_gpu(ifg_arr, ps_win, grad_win, nlks)
        except Exception as e:
            _note_fallback('phsig', e)
    from mintpy.stdproc.slc2ifg.generate_coh_phsig import estimate_phsig_correlation
    return estimate_phsig_correlation(ifg_arr, ps_win, grad_win, nlks)


def _phsig_gpu(ifg_arr: np.ndarray, ps_win: int, grad_win: int,
               nlks: float) -> np.ndarray:
    rows, cols = ifg_arr.shape
    if ps_win % 2 == 0:
        ps_win += 1
    if grad_win % 2 == 0:
        grad_win += 1
    ps_half = ps_win // 2
    grad_half = grad_win // 2

    ifg = cp.asarray(ifg_arr)

    # gradient smoothing (scipy-compatible correlate)
    padded = cp.pad(ifg, ((grad_half, grad_half), (grad_half, grad_half)),
                    mode='constant')
    rg_diff = (padded[grad_half:grad_half + rows, grad_half:grad_half + cols] *
               cp.conj(padded[grad_half:grad_half + rows,
                              grad_half - 1:grad_half + cols - 1]))
    az_diff = (padded[grad_half:grad_half + rows, grad_half:grad_half + cols] *
               cp.conj(padded[grad_half - 1:grad_half + rows - 1,
                              grad_half:grad_half + cols]))
    gk = _gaussian_kernel_cp(grad_win)
    rg_smooth = cp_ndimage.correlate(rg_diff, gk)
    az_smooth = cp_ndimage.correlate(az_diff, gk)
    rg_slope = cp.arctan2(rg_smooth.imag, rg_smooth.real)
    az_slope = cp.arctan2(az_smooth.imag, az_smooth.real)
    rg_slope[cp.abs(rg_smooth) == 0] = 0.0
    az_slope[cp.abs(az_smooth) == 0] = 0.0
    if grad_half > 0:
        for sl in (rg_slope, az_slope):
            sl[:grad_half + 1, :] = 0.0
            sl[-(grad_half):, :] = 0.0
            sl[:, :grad_half + 1] = 0.0
            sl[:, -(grad_half):] = 0.0

    i0, i1 = ps_half, rows - ps_half
    j0, j1 = ps_half, cols - ps_half
    coh = cp.zeros((rows, cols), dtype=cp.float32)
    if i1 <= i0 or j1 <= j0:
        return cp.asnumpy(coh)

    offsets = cp.arange(-ps_half, ps_half + 1)
    di_mesh, dj_mesh = cp.meshgrid(offsets, offsets, indexing='ij')
    ps_weights = _gaussian_kernel_cp(ps_win)

    windows = cp_sliding_window_view(ifg, (ps_win, ps_win))
    n_rows_out = i1 - i0
    n_cols_out = j1 - j0

    # NOTE: the deramping ramp is built PER BATCH, never for the whole image
    # at once.  Its shape is (rows, cols, ps_win, ps_win) and it materializes
    # as float64 (int64 mesh * float32 slope promotes), so a full-image ramp
    # is O(N * ps_win^2 * 8) bytes — e.g. ~5.4 GB for a 3000x9000 scene at
    # ps_win=5, the single biggest GPU OOM source in the engine.  Building it
    # inside the row loop keeps the peak at O(batch * ps_win^2 * 8) (~32 MB)
    # with bit-identical values (same float64 ops, same shapes).
    win_pixels = n_cols_out * ps_win * ps_win
    batch_rows = max(1, 4_000_000 // max(win_pixels, 1))

    for r0 in range(0, n_rows_out, batch_rows):
        r1 = min(r0 + batch_rows, n_rows_out)
        win_block = windows[r0:r1]                    # (b, ncols, ps, ps) view
        ramp_block = (
            di_mesh[None, None, :, :] *
            az_slope[i0 + r0:i0 + r1, j0:j1][:, :, None, None] +
            dj_mesh[None, None, :, :] *
            rg_slope[i0 + r0:i0 + r1, j0:j1][:, :, None, None]
        )
        exp_ramp = cp.cos(ramp_block) - 1j * cp.sin(ramp_block)
        comp = win_block * exp_ramp
        wsum = comp.sum(axis=(2, 3))
        mag = cp.abs(wsum)
        mag_valid = mag > 1e-10
        if not bool(mag_valid.any()):
            continue
        norm_sum = cp.zeros_like(wsum)
        norm_sum[mag_valid] = wsum[mag_valid] / mag[mag_valid]
        deramped = comp * cp.conj(norm_sum[:, :, None, None])
        # float64 variance accumulation (avoids float32 cancellation)
        phases = cp.arctan2(deramped.imag, deramped.real).astype(cp.float64)
        wt = ps_weights.astype(cp.float64)[None, None, :, :]
        mean_ph = (wt * phases).sum(axis=(2, 3))
        mean_ph2 = (wt * phases * phases).sum(axis=(2, 3))
        var = mean_ph2 - mean_ph * mean_ph
        # safe denominator (avoids sqrt(negative) / div-by-zero warnings)
        safe = cp.maximum(2.0 * nlks * var + 1.0, 1e-12)
        val = cp.where(var > 0, 1.0 / cp.sqrt(safe), 1.0)
        val = cp.where(mag_valid, val, 0.0)
        coh[i0 + r0:i0 + r1, j0:j1] = val.astype(cp.float32)

    return cp.asnumpy(cp.clip(coh, 0.0, 1.0))


# ------------------------------------------------------------------------
# Goldstein filter (anchored patch grid, batched GPU FFT)
# ------------------------------------------------------------------------
def goldstein_block(block: np.ndarray, nodata_mask: np.ndarray, alpha: float,
                    psize: int, wf: np.ndarray, origin: Tuple[int, int],
                    gpu: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Goldstein-filter one padded block with full-image-anchored patches.

    Parameters
    ----------
    block : np.ndarray (complex64)
        Padded data block (already nan->0 converted), in padded full-image
        coordinates.
    nodata_mask : np.ndarray (bool)
        Padded nodata mask of the block.
    alpha, psize : float, int
        Goldstein parameters (psize must equal the padded full-image pad*2).
    wf : np.ndarray
        Triangle window (``psize x psize``), precomputed.
    origin : (int, int)
        Global padded-coordinate offset of ``block`` (top-left).
    gpu : bool
        Use the CuPy batched-FFT path when available.

    Returns
    -------
    (filtered, norm) numpy arrays of ``block.shape`` (padded coordinates).
    """
    if gpu and cupy_available():
        try:
            return _goldstein_gpu(block, nodata_mask, alpha, psize, wf, origin)
        except Exception as e:
            _note_fallback('goldstein', e)
    return _goldstein_cpu(block, nodata_mask, alpha, psize, wf, origin)


def _goldstein_cpu(block, nodata_mask, alpha, psize, wf, origin):
    b_rows, b_cols = block.shape
    step = psize // 2
    filtered = np.zeros_like(block)
    norm = np.zeros((b_rows, b_cols), dtype=np.float32)
    r0, c0 = origin
    # global patch starts are multiples of step from the padded image origin
    i_first = max(0, ((r0 - psize + 1 + step - 1) // step) * step)
    j_first = max(0, ((c0 - psize + 1 + step - 1) // step) * step)
    # patch grid extent: full padded image size is unknown here, so patches
    # are bounded by the block itself (callers ensure full coverage)
    for i in range(i_first, r0 + b_rows, step):
        li = i - r0
        if li + psize > b_rows:
            break
        for j in range(j_first, c0 + b_cols, step):
            lj = j - c0
            if lj + psize > b_cols:
                break
            nd = nodata_mask[li:li + psize, lj:lj + psize]
            if np.all(nd):
                continue
            patch = block[li:li + psize, lj:lj + psize].copy()
            patch[nd] = 0
            S = np.fft.fft2(patch, s=(psize, psize))
            H = np.power(np.abs(S), alpha)
            pf = np.fft.ifft2(H * S, s=(psize, psize))
            w = wf[:psize, :psize]
            filtered[li:li + psize, lj:lj + psize] += pf * w
            norm[li:li + psize, lj:lj + psize] += w
    return filtered, norm


def _goldstein_gpu(block, nodata_mask, alpha, psize, wf, origin):
    b_rows, b_cols = block.shape
    step = psize // 2
    r0, c0 = origin

    i_first = max(0, ((r0 - psize + 1 + step - 1) // step) * step)
    j_first = max(0, ((c0 - psize + 1 + step - 1) // step) * step)
    i_starts = list(range(i_first, r0 + b_rows - psize + 1, step))
    j_starts = list(range(j_first, c0 + b_cols - psize + 1, step))
    if not i_starts or not j_starts:
        return np.zeros_like(block), np.zeros((b_rows, b_cols), dtype=np.float32)

    blk = cp.asarray(block)
    nd = cp.asarray(nodata_mask)
    wf_cp = cp.asarray(wf[:psize, :psize])

    # Deterministic CPU scatter accumulators (see NOTE below): kept on the
    # host and updated once per patch batch.
    filt_re = np.zeros((b_rows, b_cols), dtype=np.float32)
    filt_im = np.zeros((b_rows, b_cols), dtype=np.float32)
    norm = np.zeros((b_rows, b_cols), dtype=np.float32)
    wf_np = wf[:psize, :psize]

    li = cp.asarray([i - r0 for i in i_starts])
    lj = cp.asarray([j - c0 for j in j_starts])
    ii, jj = cp.meshgrid(li, lj, indexing='ij')          # (ni, nj)
    ii_f = ii.ravel()
    jj_f = jj.ravel()
    n_patches = len(ii_f)

    rr = ii_f[:, None] + cp.arange(psize)[None, :]        # (n, psize)
    cc = jj_f[:, None] + cp.arange(psize)[None, :]        # (n, psize)

    # NOTE: FFT the patches in bounded batches.  Materializing every patch of
    # the whole image at once costs ~7 full-sized complex64 copies
    # (patches/S/H/pf/contrib...) ≈ 224 * N bytes — 5-6 GB for a 3000x9000
    # scene, another GPU OOM source.  Batches of ~4096 patches (~200 MB)
    # bound the peak.  The CPU scatter accumulates in the same per-pixel
    # order as a single all-patch call (np.add.at walks the flat index array
    # sequentially and each batch is a contiguous slice of it), so the
    # output stays bit-identical.
    patch_batch = _goldstein_patch_batch(psize)

    for k0 in range(0, n_patches, patch_batch):
        k1 = min(k0 + patch_batch, n_patches)
        rrk = rr[k0:k1]
        cck = cc[k0:k1]
        patches = blk[rrk[:, :, None], cck[:, None, :]]   # (b, psize, psize)
        patches_nd = nd[rrk[:, :, None], cck[:, None, :]]
        all_nd = patches_nd.all(axis=(1, 2))

        patches = patches.copy()
        patches[patches_nd] = 0
        S = cp.fft.fft2(patches, axes=(1, 2))
        H = cp.power(cp.abs(S), alpha)
        pf = cp.fft.ifft2(H * S, axes=(1, 2))
        pf[all_nd] = 0
        contrib = pf * wf_cp[None, :, :]

        # scatter target for patch pixel (a, b): (ii_f[k]+a, jj_f[k]+b)
        nb = k1 - k0
        out_r = cp.broadcast_to(rrk[:, :, None], (nb, psize, psize))
        out_c = cp.broadcast_to(cck[:, None, :], (nb, psize, psize))
        # NOTE: cupy.add.at with duplicate indices is byte-nondeterministic
        # (atomic-add order, amplified by cancellation on large values), which
        # breaks reproducibility of the filtered output.  The scatter is cheap
        # compared to the batched FFT, so accumulate deterministically on the
        # CPU (numpy add.at is sequential and stable).
        contrib_np = cp.asnumpy(contrib)
        out_r_np = cp.asnumpy(out_r.ravel())
        out_c_np = cp.asnumpy(out_c.ravel())
        np.add.at(filt_re, (out_r_np, out_c_np), contrib_np.real.ravel())
        np.add.at(filt_im, (out_r_np, out_c_np), contrib_np.imag.ravel())
        # all-nodata patches contribute NEITHER the filtered value (already 0)
        # NOR the norm weight — the CPU kernels skip them entirely, so the
        # GPU norm must not be inflated by their triangle window either.
        wf_scatter = np.broadcast_to(wf_np, (nb, psize, psize)).copy()
        wf_scatter[all_nd] = 0.0
        np.add.at(norm, (out_r_np, out_c_np), wf_scatter.ravel())

    filtered = (filt_re + 1j * filt_im).astype(np.complex64)

    return filtered, norm


def _goldstein_patch_batch(psize: int) -> int:
    """Number of Goldstein patches to FFT per GPU batch.

    Sizes a batch from the free VRAM so its device working set (≈8 complex64
    copies of ``batch * psize^2``) stays within 1/4 of what is currently
    free, with a floor that keeps small scenes in a single call.
    """
    px = max(1, psize * psize)
    try:
        free = cp.cuda.runtime.memGetInfo()[0]            # bytes free right now
        batch = int(free // 4 // (px * 16))
        return int(max(256, min(batch, 16384)))
    except Exception:
        return 4096
