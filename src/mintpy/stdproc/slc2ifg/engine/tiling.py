#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Intra-interferogram tiling for the MintPy slc2ifg engine (design M4).

Streaming tiled computation with bounded memory:
- the input raster is read block-by-block (each block padded by ``overlap``
  pixels so local window computations are identical to the full-image run);
- the interior of each computed block is written to the output;
- image-border margins that the full-image algorithm zeroes are zeroed at
  the end (``zero_margin``), giving **bit-identical** results to the
  full-image computation;
- tiles are processed in parallel with a thread pool (each worker opens its
  own GDAL handles — datasets are never shared across threads).

Goldstein filtering is patch-anchored: the patch grid is aligned to the
*full-image* coordinate system (starts at multiples of ``psize//2`` from the
image origin), so overlapping-tile processing reproduces the exact
accumulation of the full-image run.
"""

from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

logger = logging.getLogger(__name__)

#: Rows processed per tile when tiling is enabled (config override possible)
DEFAULT_TILE_SIZE = 1024

#: Serializes GDAL block-cache I/O across tile workers (GTiff's global block
#: cache is not safe for concurrent writes, even to disjoint regions).
_IO_LOCK = threading.Lock()

_COMPANION_XML = False  # set by callers for isce2 products


def _run_ordered(jobs, compute: Callable, write: Callable, workers: int) -> None:
    """Run ``compute(job)`` in parallel but call ``write(job, payload)`` in
    deterministic job order.

    The byte layout of a tiled GTiff depends on the order its blocks are
    first written, so writing from worker threads makes the file non-
    reproducible across runs (pixels are unaffected, but md5 differs).
    Decoupling compute (parallel) from write (ordered) restores byte-level
    reproducibility.  A bounded in-flight window (``2 * workers``) keeps the
    memory of completed-but-unwritten tiles in check.
    """
    n = len(jobs)
    if workers <= 1 or n <= 1:
        for i, job in enumerate(jobs):
            write(i, job, compute(job))
        return

    window = max(2, workers * 2)
    sem = threading.Semaphore(window)
    pending: Dict[int, Any] = {}
    cv = threading.Condition()
    errors: List[BaseException] = []

    def run(i: int, job) -> None:
        try:
            payload = compute(job)
        except BaseException as e:  # noqa: BLE001 - propagate to the writer
            with cv:
                errors.append(e)
        else:
            with cv:
                pending[i] = payload
        finally:
            sem.release()
            with cv:
                cv.notify_all()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, job in enumerate(jobs):
            sem.acquire()
            ex.submit(run, i, job)
        # writer: consume results strictly in job order
        for i in range(n):
            with cv:
                while i not in pending:
                    if errors:
                        raise errors[0]
                    cv.wait()
                payload = pending.pop(i)
            write(i, jobs[i], payload)


# ------------------------------------------------------------------------
# Tile geometry
# ------------------------------------------------------------------------
def tile_jobs(rows: int, cols: int, tile_size: int, overlap: int):
    """Yield ``(r0, r1, pr0, pr1, c0, c1, pc0, pc1)`` per tile.

    ``(r0, r1, c0, c1)`` is the tile's own region (output pixels);
    ``(pr0, pr1, pc0, pc1)`` is the padded block region (input pixels).
    """
    t = max(1, int(tile_size))
    for r0 in range(0, rows, t):
        r1 = min(r0 + t, rows)
        pr0, pr1 = max(0, r0 - overlap), min(rows, r1 + overlap)
        for c0 in range(0, cols, t):
            c1 = min(c0 + t, cols)
            pc0, pc1 = max(0, c0 - overlap), min(cols, c1 + overlap)
            yield r0, r1, pr0, pr1, c0, c1, pc0, pc1


def _read_block(ds, pr0: int, pr1: int, pc0: int, pc1: int) -> np.ndarray:
    band = ds.GetRasterBand(1)
    return band.ReadAsArray(pc0, pr0, pc1 - pc0, pr1 - pr0)


def _gdal_dtype_for(arr: np.ndarray) -> int:
    if np.issubdtype(arr.dtype, np.complexfloating):
        return gdal.GDT_CFloat32
    if arr.dtype == np.float64:
        return gdal.GDT_Float64
    if arr.dtype == np.float32:
        return gdal.GDT_Float32
    if arr.dtype == np.uint16:
        return gdal.GDT_UInt16
    if arr.dtype == np.int16:
        return gdal.GDT_Int16
    return gdal.GDT_Float32


def _open_like(input_file, output_file, rows: int, cols: int, gdt: int,
               processor: str, subdataset: Optional[str] = None):
    """Create an output dataset with the input's georeferencing."""
    from mintpy.stdproc.slc2ifg.utils.slc2ifg_utils import open_gdal
    src = open_gdal(input_file, subdataset)
    gt = src.GetGeoTransform()
    proj = src.GetProjection()
    src = None
    driver_name = 'GTiff' if processor == 'isce3' else 'ENVI'
    options = (['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
               if processor == 'isce3' else [])
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    drv = gdal.GetDriverByName(driver_name)
    ds = drv.Create(str(out_path), cols, rows, 1, gdt, options)
    if gt is not None:
        ds.SetGeoTransform(gt)
    if proj:
        ds.SetProjection(proj)
    return ds


def _atomic_write(output_file: str, fn: Callable[[str], None]) -> str:
    """Run ``fn(tmp_path)`` then atomically rename onto ``output_file``.

    An interrupted run leaves only a ``.tmp`` file — the final path never holds
    a partial/zero product, so the skip-if-exists logic can trust it.
    """
    tmp = f"{output_file}.tmp"
    try:
        fn(tmp)
        os.replace(tmp, output_file)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise
    return output_file


# ------------------------------------------------------------------------
# Generic tiled compute (single input file)
# ------------------------------------------------------------------------
def compute_tiled_file(
    input_file: str,
    output_file: str,
    tile_size: int,
    overlap: int,
    compute_block: Callable[[np.ndarray], np.ndarray],
    processor: str,
    zero_margin: Tuple[int, int] = (0, 0),
    tile_workers: int = 1,
    sample_dtype: Optional[np.dtype] = None,
) -> str:
    """Streaming tiled compute: read padded blocks, compute, write interiors.

    ``compute_block(padded_block) -> result_block`` (same shape).  The result
    must be bit-identical to the full-image computation for interior pixels.
    The output is written atomically (temp file + rename).
    """
    src = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    rows, cols = src.RasterYSize, src.RasterXSize
    src = None

    # sample dtype: from the caller or read the first block
    if sample_dtype is None:
        ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
        band = ds.GetRasterBand(1)
        from osgeo import gdal_array
        sample_dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType))
        ds = None

    probe = np.zeros((1, 1), dtype=sample_dtype)
    gdt = _gdal_dtype_for(probe)
    jobs = list(tile_jobs(rows, cols, tile_size, overlap))

    def compute(job) -> np.ndarray:
        r0, r1, pr0, pr1, c0, c1, pc0, pc1 = job
        with _IO_LOCK:
            src_ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
            try:
                block = _read_block(src_ds, pr0, pr1, pc0, pc1)
            finally:
                src_ds = None
        return compute_block(block)

    def run(target: str) -> None:
        out_ds = _open_like(input_file, target, rows, cols, gdt, processor)
        out_ds = None  # noqa: F841 -- closes the create handle (GTiff stays 0 bytes until closed)

        def write(_i: int, job, res: np.ndarray) -> None:
            r0, r1, pr0, pr1, c0, c1, pc0, pc1 = job
            h = r1 - r0
            w = c1 - c0
            off_r = r0 - pr0
            off_c = c0 - pc0
            with _IO_LOCK:
                out_ds_w = gdal.Open(str(target), gdal.GA_Update)
                try:
                    out_ds_w.GetRasterBand(1).WriteArray(
                        res[off_r:off_r + h, off_c:off_c + w], c0, r0)
                finally:
                    out_ds_w = None

        _run_ordered(jobs, compute, write, tile_workers)
        _zero_margins(target, rows, cols, zero_margin)

    return _atomic_write(output_file, run)


def _zero_margins(output_file: str, rows: int, cols: int,
                  zero_margin: Tuple[int, int]) -> None:
    """Zero the image-border margins (matches full-image edge behaviour)."""
    mr, mc = zero_margin
    if mr <= 0 and mc <= 0:
        return
    out_ds = gdal.Open(str(output_file), gdal.GA_Update)
    if out_ds is None:
        return
    try:
        band = out_ds.GetRasterBand(1)
        if mr > 0:
            top = np.zeros((min(mr, rows), cols), dtype=np.float32)
            band.WriteArray(top, 0, 0)
            if mr < rows:
                bot = np.zeros((min(mr, rows), cols), dtype=np.float32)
                band.WriteArray(bot, 0, rows - mr)
        if mc > 0:
            mid_h = max(0, rows - 2 * mr)
            if mid_h > 0:
                left = np.zeros((mid_h, min(mc, cols)), dtype=np.float32)
                band.WriteArray(left, 0, mr)
                if mc < cols:
                    right = np.zeros((mid_h, min(mc, cols)), dtype=np.float32)
                    band.WriteArray(right, cols - mc, mr)
        band.FlushCache()
    finally:
        out_ds = None  # noqa: F841 -- release the GDAL write handle


# ------------------------------------------------------------------------
# Complex coherence (two input files, boxcar window)
# ------------------------------------------------------------------------
def complex_coh_tiled(
    slc1_file: str,
    slc2_file: str,
    output_file: str,
    window: int,
    tile_size: int,
    processor: str,
    tile_workers: int = 1,
    gpu: bool = False,
    subdataset: str = '/data/VV',
) -> str:
    """Tiled boxcar complex coherence between two SLC files.

    ``subdataset`` selects the HDF5 dataset for ``.h5`` SLC inputs.
    """
    from mintpy.stdproc.slc2ifg.engine.gpu_kernels import complex_coh_block
    from mintpy.stdproc.slc2ifg.utils.slc2ifg_utils import open_gdal

    half = window // 2
    overlap = half

    def compute_block(block: np.ndarray) -> np.ndarray:
        # block is the interferogram of the two SLC blocks is not enough:
        # we need both SLCs.  Handled by a specialized path below.
        raise NotImplementedError

    # Two-file variant: read both blocks inside the worker
    src1 = open_gdal(slc1_file, subdataset)
    rows, cols = src1.RasterYSize, src1.RasterXSize
    src1 = None

    jobs = list(tile_jobs(rows, cols, tile_size, overlap))

    def compute(job) -> np.ndarray:
        r0, r1, pr0, pr1, c0, c1, pc0, pc1 = job
        with _IO_LOCK:
            a = open_gdal(slc1_file, subdataset)
            b = open_gdal(slc2_file, subdataset)
            try:
                s1 = _read_block(a, pr0, pr1, pc0, pc1)
                s2 = _read_block(b, pr0, pr1, pc0, pc1)
            finally:
                a = b = None
            if s1.dtype not in (np.complex64, np.complex128):
                s1 = s1.astype(np.complex64)
            if s2.dtype not in (np.complex64, np.complex128):
                s2 = s2.astype(np.complex64)
        return complex_coh_block(s1, s2, window, gpu=gpu)

    def run(target: str) -> None:
        out_ds = _open_like(slc1_file, target, rows, cols,
                            gdal.GDT_Float32, processor, subdataset=subdataset)
        out_ds = None  # noqa: F841 -- closes the create handle (GTiff stays 0 bytes until closed)

        def write(_i: int, job, res: np.ndarray) -> None:
            r0, r1, pr0, pr1, c0, c1, pc0, pc1 = job
            h = r1 - r0
            w = c1 - c0
            with _IO_LOCK:
                o = gdal.Open(str(target), gdal.GA_Update)
                try:
                    o.GetRasterBand(1).WriteArray(
                        res[r0 - pr0:r0 - pr0 + h, c0 - pc0:c0 - pc0 + w], c0, r0)
                finally:
                    o = None

        _run_ordered(jobs, compute, write, tile_workers)
        _zero_margins(target, rows, cols, (half, half))

    return _atomic_write(output_file, run)


# ------------------------------------------------------------------------
# Goldstein filter (patch-grid anchored to the full image)
# ------------------------------------------------------------------------
def goldstein_tiled(
    input_file: str,
    output_file: str,
    alpha: float,
    psize: int,
    tile_size: int,
    processor: str,
    tile_workers: int = 1,
    gpu: bool = False,
) -> str:
    """Tiled Goldstein filter, bit-identical to the full-image run.

    The patch grid (starts at multiples of ``psize//2`` in the *padded*
    full-image coordinates) is preserved across tiles, so the triangle-window
    accumulation matches the non-tiled ``goldstein()`` exactly.
    """

    pad = psize // 2

    # triangle window (same as filter_utils.goldstein)
    half = pad
    wx = (1.0 - np.abs(np.arange(half) - (psize / 2.0 - 1.0)) / (psize / 2.0 - 1.0))
    wy = wx
    q = np.outer(wy, wx)
    wf = np.block([[q, np.flip(q, 1)],
                   [np.flip(q, 0), np.flip(np.flip(q, 0), 1)]])

    src = gdal.Open(str(input_file), gdal.GA_ReadOnly)
    rows, cols = src.RasterYSize, src.RasterXSize
    band = src.GetRasterBand(1)
    from osgeo import gdal_array
    in_dtype = np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(band.DataType))
    src = None
    is_complex = np.issubdtype(in_dtype, np.complexfloating)

    p_rows, p_cols = rows + 2 * pad, cols + 2 * pad
    jobs = list(tile_jobs(p_rows, p_cols, tile_size, psize))

    def read_padded_block(pr0, pr1, pc0, pc1) -> Tuple[np.ndarray, np.ndarray]:
        """Padded (complex, nodata) block in padded full-image coords.

        Padded coord ``p`` maps to file coord ``p - pad``; file reads are
        clamped to the raster extent (padded margins read as zero/nodata).
        """
        block = np.zeros((pr1 - pr0, pc1 - pc0), dtype=np.complex64)
        nodata = np.ones((pr1 - pr0, pc1 - pc0), dtype=bool)
        # file region intersecting the block (clamped)
        f_r0, f_r1 = max(0, pr0 - pad), min(rows, pr1 - pad)
        f_c0, f_c1 = max(0, pc0 - pad), min(cols, pc1 - pad)
        if f_r1 > f_r0 and f_c1 > f_c0:
            with _IO_LOCK:
                ds = gdal.Open(str(input_file), gdal.GA_ReadOnly)
                try:
                    arr = ds.GetRasterBand(1).ReadAsArray(
                        f_c0, f_r0, f_c1 - f_c0, f_r1 - f_r0)
                finally:
                    ds = None
            if arr is not None:
                if not np.issubdtype(arr.dtype, np.complexfloating):
                    arr = arr.astype(np.complex64)
                arr = np.nan_to_num(arr).astype(np.complex64)
                # file coords -> padded coords -> block-local coords
                lr0, lr1 = f_r0 + pad - pr0, f_r1 + pad - pr0
                lc0, lc1 = f_c0 + pad - pc0, f_c1 + pad - pc0
                block[lr0:lr1, lc0:lc1] = arr
                nodata[lr0:lr1, lc0:lc1] = np.abs(arr) < 1e-6
        return block, nodata

    def compute(job) -> Optional[Tuple[np.ndarray, int, int, int, int, int, int]]:
        """Read padded block + kernel; returns (out_val, slice coords, out
        origin) or None when the tile lies fully outside the original image."""
        r0, r1, pr0, pr1, c0, c1, pc0, pc1 = job
        block, nodata = read_padded_block(pr0, pr1, pc0, pc1)

        # shared anchored kernel (CPU or GPU batched FFT); origin = block's
        # global padded coordinates so the patch grid matches the full image
        from mintpy.stdproc.slc2ifg.engine.gpu_kernels import goldstein_block
        filtered, norm = goldstein_block(
            block, nodata, alpha, psize, wf, (pr0, pc0), gpu=gpu)

        # tile's own region (block-local coords)
        t0 = r0 - pr0
        t1 = r1 - pr0
        u0 = c0 - pc0
        u1 = c1 - pc0
        seg = filtered[t0:t1, u0:u1]
        nseg = norm[t0:t1, u0:u1]
        valid = nseg > 0
        seg[valid] /= nseg[valid]
        seg[~valid] = 0 + 0j

        # original-coordinate nodata masking (same as goldstein())
        tile_nd = nodata[t0:t1, u0:u1]
        seg[tile_nd] = 0 + 0j

        out_val = seg.astype(np.complex64) if is_complex else np.angle(seg).astype(np.float32)

        # write only the part of the tile inside the original image
        o_r0, o_r1 = max(0, r0 - pad), min(rows, r1 - pad)
        o_c0, o_c1 = max(0, c0 - pad), min(cols, c1 - pad)
        if o_r1 <= o_r0 or o_c1 <= o_c0:
            return None
        # seg coords for original rows/cols: seg row s = padded row (r0 + s)
        # -> original row (r0 + s - pad)
        s_r0 = o_r0 + pad - r0
        s_r1 = o_r1 + pad - r0
        s_c0 = o_c0 + pad - c0
        s_c1 = o_c1 + pad - c0
        return out_val, s_r0, s_r1, s_c0, s_c1, o_c0, o_r0

    def run(target: str) -> None:
        out_ds = _open_like(input_file, target, rows, cols,
                            gdal.GDT_CFloat32 if is_complex else gdal.GDT_Float32,
                            processor)
        out_ds = None  # noqa: F841 -- closes the create handle (GTiff stays 0 bytes until closed)

        def write(_i: int, job, payload) -> None:
            if payload is None:
                return
            out_val, s_r0, s_r1, s_c0, s_c1, o_c0, o_r0 = payload
            with _IO_LOCK:
                o = gdal.Open(str(target), gdal.GA_Update)
                try:
                    o.GetRasterBand(1).WriteArray(
                        out_val[s_r0:s_r1, s_c0:s_c1], o_c0, o_r0)
                finally:
                    o = None

        _run_ordered(jobs, compute, write, tile_workers)
        out_ds = None  # noqa: F841 -- release the GDAL write handle

    return _atomic_write(output_file, run)
