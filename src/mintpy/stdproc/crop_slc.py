#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Crop SLCs (and, for isce2, the geometry products) to a geographic bbox.

Single implementation for both processors:

* **isce3** (geocoded): the AOI is mapped to a pixel window with
  :func:`mintpy.stdproc.io.bbox_to_window`, only that window is read, and the
  result is written as a georeferenced ``yyyymmdd.slc.tif``.
* **isce2** (radar coordinates): the radar window is derived from the
  ``lat.rdr.full`` / ``lon.rdr.full`` lookup tables, the SLC is written as a
  **georeferenced-free** ``yyyymmdd.slc.tif`` (a plain GeoTIFF: same container
  for both processors, no fake coordinates), and the geometry products are kept
  in their original ENVI + ``.hdr`` / ``.xml`` form so that ``prep_isce`` and
  ``load_data`` keep working unchanged.

The module is a plain implementation library: it has no CLI and no logging
configuration.  The command line interface lives in ``mintpy/cli/crop_slc.py``.
"""

from __future__ import annotations

import os
import re
import glob
import logging
import concurrent.futures
from pathlib import Path
from typing import List, Optional

import numpy as np

from . import io as sio
from .utils import naming

logger = logging.getLogger(__name__)

#: burst id pattern used by the per-burst output layout, e.g. t124_264305_iw2
BURST_RE = re.compile(r'^t\d+_\d+_iw\d+$')

SLC_NOT_FOUND_MSG = 'no input files found'


# ---------------------------------------------------------------------------
# input discovery / naming
# ---------------------------------------------------------------------------
def extract_date_from_filename(file_path) -> str:
    """Extract a YYYYMMDD date string from a filename (mtime fallback)."""
    filename = Path(file_path).name

    match = re.search(r'^(\d{8})', filename)
    if match:
        return match.group(1)

    match = re.search(r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])', filename)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"

    import time
    return time.strftime('%Y%m%d', time.localtime(os.path.getmtime(file_path)))


def extract_burst_id(file_path) -> Optional[str]:
    """Burst id from the direct parent (or grandparent) directory name."""
    parent = Path(file_path).parent
    if BURST_RE.match(parent.name):
        return parent.name
    if BURST_RE.match(parent.parent.name):
        return parent.parent.name
    return None


def get_input_files(input_dir, pattern=None, file_list=None) -> List[str]:
    """Resolve the input SLC file list from patterns and/or an explicit list."""
    if file_list:
        with open(file_list) as f:
            files = [line.strip() for line in f if line.strip()]
        files = [f for f in files if os.path.isfile(f) and 'static' not in f]
        logger.info('read %d files from file list: %s', len(files), file_list)
        return files

    patterns = input_dir if isinstance(input_dir, (list, tuple)) else [input_dir]
    found: List[str] = []
    for pat in patterns:
        for path in glob.glob(str(pat), recursive=True):
            if os.path.isdir(path):
                for f in glob.glob(os.path.join(path, pattern or '*'), recursive=True):
                    if os.path.isfile(f) and 'static' not in f:
                        found.append(f)
            elif os.path.isfile(path) and 'static' not in path:
                found.append(path)

    unique = sorted(set(found))
    logger.info('found %d unique input file(s)', len(unique))
    return unique


def _output_paths(input_files, output_dir, prefix='', by_burst=False,
                  no_burst_dirs=False) -> List[str]:
    """Canonical ``[prefix]yyyymmdd.slc.tif`` output path per input file."""
    paths = []
    for f in input_files:
        out_dir = Path(output_dir)
        if by_burst and not no_burst_dirs:
            burst_id = extract_burst_id(f)
            if burst_id:
                out_dir = out_dir / burst_id
        paths.append(str(out_dir / f'{prefix}{extract_date_from_filename(f)}.slc.tif'))
    return paths


# ---------------------------------------------------------------------------
# isce2 radar window from the lon/lat lookup tables
# ---------------------------------------------------------------------------
def _default_nodata(dtype):
    return np.nan if np.issubdtype(np.dtype(dtype), np.inexact) else 0


def _maybe_fill_nan(data, fill_nan):
    if fill_nan and np.issubdtype(data.dtype, np.inexact) and np.isnan(data).any():
        data = np.where(np.isnan(data), 0 + 0j if np.iscomplexobj(data) else 0, data)
    return data


def _warp_to_epsg(path, dest_epsg, compress_level=6):
    """Reproject a raster in place with gdal.Warp (only when really needed)."""
    from osgeo import gdal
    ds = gdal.Open(str(path))
    if ds is None:
        raise IOError(f'cannot open {path}')
    epsg = sio.epsg_from_projection(ds.GetProjection())
    ds = None
    if not dest_epsg or epsg == int(dest_epsg):
        return

    tmp = f'{path}.warp'
    options = gdal.WarpOptions(
        dstSRS=f'EPSG:{int(dest_epsg)}',
        creationOptions=sio.DEFAULT_TIFF_OPTIONS if compress_level > 0 else [],
    )
    gdal.Warp(tmp, str(path), options=options)
    os.replace(tmp, str(path))


def crop_geocoded(src, out_file, wsen, subdataset=None, buffer=0.0,
                  dest_epsg=None, fill_nan=False, compress_level=6) -> bool:
    """Crop one geocoded SLC (GeoTIFF or HDF5 with coordinates) to ``wsen``.

    The HDF5 subdataset is auto-detected (preferring VV) when not given.
    """
    src = str(src)
    compress = 'DEFLATE' if compress_level > 0 else None

    if sio.is_hdf5_file(src):
        subdataset = sio.detect_hdf5_subdataset(src, subdataset)
        win_info = sio.hdf5_window(src, _expand(wsen, buffer), subdataset)
        if win_info is None:
            logger.info('skipping %s: no overlap with bbox', os.path.basename(src))
            return True
        row0, row1, col0, col1 = win_info['window']
        import h5py
        with h5py.File(src, 'r') as h5:
            if subdataset not in h5:
                raise ValueError(f'subdataset {subdataset} not found in {src}')
            data = h5[subdataset][row0:row1, col0:col1]
        sub_x = win_info['x_coords'][col0:col1]
        sub_y = win_info['y_coords'][row0:row1]
        x_res = abs(sub_x[1] - sub_x[0]) if len(sub_x) > 1 else 1.0
        y_res = abs(sub_y[1] - sub_y[0]) if len(sub_y) > 1 else 1.0
        y_step = y_res if win_info['y_descending'] else -y_res
        meta = {
            'X_FIRST': float(sub_x[0]), 'Y_FIRST': float(sub_y[0]),
            'X_STEP': x_res, 'Y_STEP': y_step,
        }
        epsg = win_info.get('epsg')
        if epsg:
            meta['EPSG'] = int(epsg)

    else:
        window = sio.bbox_to_window(src, wsen, subdataset=subdataset, buffer=buffer)
        if window is None:
            logger.info('skipping %s: no overlap with bbox', os.path.basename(src))
            return True
        x0, y0, w, h = window
        data, smeta = sio.read_raster(src, box=(x0, y0, x0 + w, y0 + h), geo=True)
        gt = sio.get_geotransform(sio.open_raster(src))
        if gt is None:
            raise ValueError(f'{src} is not geocoded; cannot crop by bbox')
        meta = {
            'X_FIRST': gt[0] + x0 * gt[1] + y0 * gt[2],
            'Y_FIRST': gt[3] + x0 * gt[4] + y0 * gt[5],
            'X_STEP': abs(gt[1]), 'Y_STEP': gt[5],
        }
        if smeta.get('EPSG'):
            meta['EPSG'] = smeta['EPSG']

    data = _maybe_fill_nan(data, fill_nan)
    data = np.asarray(data)

    sio.write_raster(data, out_file, meta=meta, geo=True, processor='isce3',
                     compress=compress, tiled=True,
                     nodata=_default_nodata(data.dtype))

    _warp_to_epsg(out_file, dest_epsg, compress_level=compress_level)
    return True


def _expand(wsen, buffer):
    w, s, e, n = (float(v) for v in wsen)
    return (w - buffer, s - buffer, e + buffer, n + buffer)


def _summary(results, what):
    ok = [r for r in results if r[2]]
    bad = [r for r in results if not r[2]]
    logger.info('%s summary: %d successful, %d failed', what, len(ok), len(bad))
    for _src, _out, _ok, msg in bad:
        logger.error('  failed: %s', msg)
    return len(bad)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------
def crop_slc(input_dir, output_dir, bbox, processor='isce3', pattern=None,
             buffer=0.0, by_burst=False,
             file_list=None, subdataset=None, workers=1,
             no_skip_existing=False, no_burst_dirs=False,
             dest_epsg=None, fill_nan=False, compress_level=6,
             dry_run=False) -> int:
    """Crop SLCs to a geographic bbox.

    Parameters
    ----------
    input_dir : str or list of str
        Directory / glob pattern(s) holding the input SLCs.
    output_dir : str
        Directory for the cropped SLCs (``yyyymmdd.slc.tif``).
    bbox : tuple of 4 floats
        (west, south, east, north) in EPSG:4326.
    processor : {'isce3', 'isce2'}
    pattern : str, optional
        Input glob; defaults to the processor's raw SLC pattern.
    buffer : float
        Extra margin in degrees around ``bbox``.
    by_burst, no_burst_dirs : bool
        Per-burst output subdirectory (id detected from the input path).
    file_list : str, optional
        Explicit input file list (takes precedence over ``input_dir``).
    subdataset : str, optional
        HDF5 subdataset for HDF5 SLCs; auto-detected when omitted.
    workers : int
        Thread pool size for the per-file crop.
    no_skip_existing : bool
        Recompute existing outputs.
    dest_epsg : int, optional
        Reproject geocoded outputs to this EPSG (isce3 only).
    dry_run : bool
        Only report what would be done.

    Returns
    -------
    int, 0 on success, 1 when at least one file failed
    """
    if not bbox:
        raise ValueError('crop_slc requires a bbox (slc2ifg.bbox)')

    if processor not in ('isce2', 'isce3'):
        raise ValueError(f'unsupported processor: {processor}')

    if processor == 'isce2':
        raise ValueError(
            'bbox cropping requires geocoded (isce3) SLCs; isce2 '
            'radar-coordinate SLCs have no georeferencing to crop by bbox')
    if processor == 'isce2' and dest_epsg:
        logger.warning('dest_epsg is ignored for isce2 radar-coordinate products')

    if pattern is None:
        pattern = naming.slc_pattern(processor)

    input_files = get_input_files(input_dir, pattern=pattern, file_list=file_list)
    if not input_files:
        logger.error(SLC_NOT_FOUND_MSG)
        return 1

    output_files = _output_paths(input_files, output_dir,
                                 by_burst=by_burst, no_burst_dirs=no_burst_dirs)

    todo = []
    for src, out in zip(input_files, output_files):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        if not no_skip_existing and os.path.exists(out):
            logger.info('skipping existing output: %s', out)
            continue
        todo.append((src, out))

    if dry_run:
        for src, out in todo:
            logger.info('DRY RUN: %s -> %s', src, out)
        return 0

    def _work(task):
        src, out = task
        try:
            return (src, out, crop_geocoded(
                src, out, bbox, subdataset=subdataset, buffer=buffer,
                dest_epsg=dest_epsg, fill_nan=fill_nan,
                compress_level=compress_level), '')
        except Exception as exc:                                  # noqa: BLE001
            logger.error('error cropping %s: %s', src, exc)
            return (src, out, False, str(exc))

    results = []
    n_workers = max(1, min(int(workers or 1), 4, max(1, len(todo))))
    if todo:
        if n_workers == 1:
            results = [_work(t) for t in todo]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                results = list(pool.map(_work, todo))

    failed = _summary(results, 'SLC')
    return 0 if failed == 0 else 1
