#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Phase unwrapping via SNAPHU binary with full parameter control.

Calls the ``snaphu`` executable directly via subprocess, writing a full
configuration file that exposes ALL SNAPHU parameters (DEFOMAX_CYCLE,
DEFOTHRESHFACTOR, etc.) — including deformation-mode cost parameters that
snaphu-py does not expose.

Supports ISCE2 (ENVI) and ISCE3 (GeoTIFF) input/output formats.
Every ISCE2 output binary is accompanied by an ISCE2 <imageFile> XML.

Examples:
    # ISCE3 (GeoTIFF) basic unwrapping
    unwrap_ifgram.py --processor isce3 --ifg-dir ./ifgrams --cor-dir ./cors \\
        --nlooks 20.0 --max-workers 4

    # ISCE2 (ENVI) unwrapping in deformation mode with custom DEFOMAX
    unwrap_ifgram.py --processor isce2 --ifg-dir ./ifgs --cor-dir ./cors \\
        --ifg-pattern *.int --cor-pattern *_phsig.coh --cost-mode defo \\
        --defo-max-cycles 2.0 --nlooks 16.0 --ntiles 2 2 --max-workers 8
"""

import re
import argparse
import logging
import os
import shutil
import subprocess
import threading
import sys
import tempfile
import textwrap
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from osgeo import gdal

from importlib.resources import files

from .utils.naming import (
    conncomp_path,
    extract_date_pair,
    unw_path,
    variant_of,
)
from .utils.slc2ifg_utils import (
    create_xml_for_binary,
    tqdm_progress,
)

# Packaged snaphu binary path
_SNAPHU_PACKAGED = str(files('mintpy.stdproc.slc2ifg').joinpath('bin', 'snaphu'))

gdal.UseExceptions()

logger = logging.getLogger(__name__)


def _open_wbd(wbd_path):
    """Open a .wbd water-body raster as a WGS84 GDAL dataset.

    Supports three companion-metadata layouts:
    - ``.wbd.rsc`` (ISCE-style; the format produced by ``sardem ... -o
      swbd.wbd`` on the Guam run): binary + RSC text metadata;
    - ``.wbd.vrt`` (sardem GeoTIFF/ENVI companion): GDAL-readable directly;
    - ``.wbd.json`` (earthscope-style): binary + JSON metadata.

    Returns an open GDAL dataset (caller closes it) or ``None``.
    """
    from osgeo import gdal, osr

    p = str(wbd_path)
    base = os.path.splitext(p)[0]

    # 1) GDAL-readable companion (.vrt)
    vrt = base + '.vrt'
    if os.path.isfile(vrt):
        ds = gdal.Open(vrt, gdal.GA_ReadOnly)
        if ds is not None:
            return ds

    # 2) ISCE-style .rsc companion: the binary is `xxx.wbd` and its
    #    metadata is `xxx.wbd.rsc` (NOT `xxx.rsc` — keep the .wbd stem).
    rsc = p + '.rsc'
    if os.path.isfile(rsc):
        meta = {}
        with open(rsc) as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    meta[parts[0]] = parts[1]
        required = ('WIDTH', 'FILE_LENGTH', 'X_FIRST', 'Y_FIRST',
                    'X_STEP', 'Y_STEP')
        missing_keys = [k for k in required if k not in meta]
        if missing_keys:
            raise ValueError(
                f"wbd .rsc {rsc} missing required key(s): {missing_keys}")
        width = int(float(meta['WIDTH']))
        length = int(float(meta['FILE_LENGTH']))
        x_first = float(meta['X_FIRST'])
        y_first = float(meta['Y_FIRST'])
        x_step = float(meta['X_STEP'])
        y_step = float(meta['Y_STEP'])
        raw = np.fromfile(p, dtype=np.uint8)
        if raw.size != width * length:
            raise ValueError(
                f"wbd binary size {raw.size} != WIDTH*FILE_LENGTH "
                f"{width}x{length} from {rsc}")
        data = raw.reshape(length, width)
        ds = gdal.GetDriverByName('MEM').Create(
            '', width, length, 1, gdal.GDT_Byte)
        ds.SetGeoTransform((x_first, x_step, 0, y_first, 0, y_step))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(data)
        return ds

    # 3) earthscope-style .json companion
    js = base + '.json'
    if os.path.isfile(js):
        import json
        with open(js) as f:
            meta = json.load(f)
        raw = np.fromfile(p, dtype=np.uint8)
        data = raw.reshape(meta['height'], meta['width'])
        ds = gdal.GetDriverByName('MEM').Create(
            '', meta['width'], meta['height'], 1, gdal.GDT_Byte)
        ds.SetGeoTransform(
            (meta['lon0'], meta['dlon'], 0, meta['lat0'], 0, meta['dlat']))
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)
        ds.SetProjection(srs.ExportToWkt())
        ds.GetRasterBand(1).WriteArray(data)
        return ds

    logger.warning("wbd %s has no .rsc/.vrt/.json companion — skipped", p)
    return None


def wbd_to_mask_array(wbd_path, ifg_path, invert=True):
    """Load a .wbd water mask warped onto the interferogram's grid.

    Parameters
    ----------
    wbd_path : str
        Path to the .wbd water-body raster (with .rsc / .vrt / .json).
    ifg_path : str
        Path to the (wrapped) interferogram raster defining the target grid
        (geotransform + projection, typically UTM).
    invert : bool
        True -> return 1 = valid/land, 0 = water (SNAPHU mask semantics);
        False -> return 1 = water, 0 = land (raw .wbd semantics).

    Returns
    -------
    np.ndarray (uint8) on the interferogram grid, or None on any failure.
    """
    from osgeo import gdal

    ifg_ds = gdal.Open(str(ifg_path), gdal.GA_ReadOnly)
    if ifg_ds is None:
        logger.warning("wbd: cannot open interferogram %s", ifg_path)
        return None
    rows, cols = ifg_ds.RasterYSize, ifg_ds.RasterXSize
    gt = ifg_ds.GetGeoTransform()
    proj = ifg_ds.GetProjection()
    ifg_ds = None

    src = _open_wbd(wbd_path)
    if src is None:
        return None

    dst = gdal.GetDriverByName('MEM').Create('', cols, rows, 1, gdal.GDT_Byte)
    dst.SetGeoTransform(gt)
    dst.SetProjection(proj)

    try:
        gdal.ReprojectImage(
            src, dst, src.GetProjection(), proj,
            gdal.GRA_NearestNeighbour)
        warped = dst.GetRasterBand(1).ReadAsArray()
    except Exception as e:
        logger.warning("wbd reprojection failed: %s", e)
        warped = None
    finally:
        src = None
        dst = None

    if warped is None:
        return None
    water = (warped > 0).astype(np.uint8)
    return (1 - water).astype(np.uint8) if invert else water

# ------------------------------------------------------------------------
# Active SNAPHU subprocess registry
# ------------------------------------------------------------------------
# SNAPHU runs can take minutes; when the engine aborts we terminate the
# in-flight ones so the process exits promptly instead of waiting out every
# unwrap (and leaving the workers busy).
_ACTIVE_SNAPHU: set = set()
_ACTIVE_SNAPHU_LOCK = threading.Lock()


def terminate_active_snaphu() -> int:
    """Terminate all currently running SNAPHU subprocesses.

    Returns the number of subprocesses terminated.  Safe to call from any
    thread; used by the engine's failure path to release CPU quickly.
    """
    with _ACTIVE_SNAPHU_LOCK:
        procs = list(_ACTIVE_SNAPHU)
    n = 0
    for p in procs:
        try:
            p.terminate()
            n += 1
        except Exception:  # noqa: BLE001 - already dead / race
            pass
    if n:
        logger.warning("Terminated %d in-flight SNAPHU subprocess(es)", n)
    return n

# ------------------------------------------------------------------------
# Defaults (from ISCE2/SNAPHU defaults)
# ------------------------------------------------------------------------
_DEFAULT_COST_MODE = "smooth"
_DEFAULT_INIT_METHOD = "mcf"
_DEFAULT_DEFO_MAX_CYCLES = 1.2
_DEFAULT_DEFO_THRESH_FACTOR = 1.2
_DEFAULT_DEFO_LAY_CONST = 0.9
_DEFAULT_MIN_CONNCOMP_FRAC = 0.01
_DEFAULT_CONNCOMP_THRESH = 300
_DEFAULT_MAX_NCOMPS = 32
_DEFAULT_TILE_COST_THRESH = 500
_DEFAULT_MIN_REGION_SIZE = 100
_DEFAULT_PHASE_GRAD_WINDOW = (7, 7)
_DEFAULT_LAMBDA = 0.05546576  # S1 C-band

UNW_SUFFIX_ISCE3 = ".unw.tif"
CONNCOMP_SUFFIX_ISCE3 = ".unw.conncomp.tif"
UNW_SUFFIX_ISCE2 = ".unw"
CONNCOMP_SUFFIX_ISCE2 = ".unw.conncomp"


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def find_snaphu_binary() -> str:
    """Find the snaphu executable.

    Searches in priority order:
      1. Packaged binary at ``slc2ifg/bin/snaphu``
      2. ``$PATH`` (``shutil.which("snaphu")``)
    """
    if os.path.isfile(_SNAPHU_PACKAGED) and os.access(_SNAPHU_PACKAGED, os.X_OK):
        return _SNAPHU_PACKAGED
    found = shutil.which("snaphu")
    if found:
        return found
    raise FileNotFoundError(
        f"snaphu binary not found at {_SNAPHU_PACKAGED} or in PATH. "
        "Install snaphu or set --snaphu-binary."
    )


# ------------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------------
def parse_arguments(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase unwrapping via SNAPHU binary with full parameter control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ISCE3 (GeoTIFF)
  %(prog)s --processor isce3 --ifg-dir ./ifgs --cor-dir ./cors \\
      --nlooks 20.0 --max-workers 4

  # ISCE2 (ENVI) deformation mode with custom DEFOMAX
  %(prog)s --processor isce2 --ifg-dir ./ifgs --cor-dir ./cors \\
      --ifg-pattern *.int --cor-pattern *_phsig.coh --cost-mode defo \\
      --defo-max-cycles 2.0 --nlooks 16.0 --ntiles 2 2 --max-workers 8
        """,
    )

    # ---- Processor ----
    parser.add_argument("--processor", type=str, choices=["isce2", "isce3"],
                        required=True,
                        help="Processor type")

    # ---- I/O ----
    parser.add_argument("--ifg-dir", type=Path, required=True)
    parser.add_argument("--cor-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("./unwrapped_output"))
    parser.add_argument("--ifg-pattern", type=str)
    parser.add_argument("--cor-pattern", type=str)

    # ---- SNAPHU core ----
    parser.add_argument("--nlooks", type=float, default=5.0)
    parser.add_argument("--cost-mode", choices=["topo", "defo", "smooth"],
                        default=_DEFAULT_COST_MODE)
    parser.add_argument("--init-method", choices=["mst", "mcf"],
                        default=_DEFAULT_INIT_METHOD)

    # ---- Deformation-mode params ----
    parser.add_argument("--defo-max-cycles", type=float,
                        default=_DEFAULT_DEFO_MAX_CYCLES,
                        help=f"DEFOMAX_CYCLE (default: {_DEFAULT_DEFO_MAX_CYCLES})")
    parser.add_argument("--defo-thresh-factor", type=float,
                        default=_DEFAULT_DEFO_THRESH_FACTOR,
                        help=f"DEFOTHRESHFACTOR (default: {_DEFAULT_DEFO_THRESH_FACTOR})")
    parser.add_argument("--defo-lay-const", type=float,
                        default=_DEFAULT_DEFO_LAY_CONST,
                        help=f"DEFOCONST for defo mode (default: {_DEFAULT_DEFO_LAY_CONST})")
    parser.add_argument("--lambda", type=float, dest="wavelength",
                        default=_DEFAULT_LAMBDA,
                        help=f"Radar wavelength in meters (default: {_DEFAULT_LAMBDA})")
    parser.add_argument("--snaphu-binary", type=str, default=None,
                        help="Path to snaphu executable (auto-detected if not set)")

    # ---- Tiling ----
    parser.add_argument("--ntiles", type=int, nargs=2, default=[1, 1],
                        metavar=("ROW", "COL"))
    parser.add_argument("--tile-overlap", type=int, default=0)
    parser.add_argument("--nproc", type=int, default=1,
                        help="Processors per snaphu call (default: 1)")
    parser.add_argument("--tile-cost-thresh", type=int,
                        default=_DEFAULT_TILE_COST_THRESH)
    parser.add_argument("--min-region-size", type=int,
                        default=_DEFAULT_MIN_REGION_SIZE)

    # ---- Connected components ----
    parser.add_argument("--min-conncomp-frac", type=float,
                        default=_DEFAULT_MIN_CONNCOMP_FRAC)
    parser.add_argument("--conncomp-thresh", type=float,
                        default=_DEFAULT_CONNCOMP_THRESH)
    parser.add_argument("--max-ncomps", type=int, default=_DEFAULT_MAX_NCOMPS)
    parser.add_argument("--no-conncomp-out", action="store_true",
                        help="Do NOT write connected component file")

    # ---- Phase gradient window ----
    parser.add_argument("--phase-grad-window", type=int, nargs=2,
                        default=list(_DEFAULT_PHASE_GRAD_WINDOW),
                        metavar=("PSI", "DPSI"))

    # ---- Optional files ----
    parser.add_argument("--mask-file", type=Path)
    parser.add_argument("--init-phase", type=Path)
    parser.add_argument("--scratch-dir", type=Path)
    parser.add_argument("--keep-scratch", action="store_true",
                        help="Keep scratch directory after processing")

    # ---- Parallelism ----
    parser.add_argument("--max-workers", type=int, default=None)

    # ---- Other ----
    parser.add_argument("-v", "--verbose", action="store_true")

    args, _ = parser.parse_known_args(args_list) if args_list is not None \
        else (parser.parse_args(), None)

    if args.ifg_pattern is None:
        args.ifg_pattern = "**/*.int.tif" if args.processor == "isce3" else "**/*.int"
    if args.cor_pattern is None:
        args.cor_pattern = "**/*_phsig.coh.tif" if args.processor == "isce3" else "**/*_phsig.coh"

    args.ntiles = tuple(args.ntiles)
    args.phase_grad_window = tuple(args.phase_grad_window)

    return args


# ------------------------------------------------------------------------
# SNAPHU config file builder
# ------------------------------------------------------------------------
def _build_snaphu_config(
    ifg_file: str,
    width: int,
    corr_file: str,
    out_file: str,
    nlooks: float,
    cost_mode: str = "smooth",
    init_method: str = "mcf",
    ntiles: Tuple[int, int] = (1, 1),
    tile_overlap: int = 0,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
    min_conncomp_frac: float = 0.01,
    conncomp_thresh: float = 300,
    max_ncomps: int = 32,
    phase_grad_window: Tuple[int, int] = (7, 7),
    defo_max_cycles: float = 1.2,
    defo_thresh_factor: float = 1.2,
    defo_lay_const: float = 0.9,
    wavelength: float = 0.05546576,
    mask_file: Optional[str] = None,
    conncomp_file: Optional[str] = None,
    init_phase_file: Optional[str] = None,
    conncomp_out: bool = True,
) -> str:
    """
    Build a SNAPHU configuration file content string.

    Parameters match the SNAPHU config file specification (see
    ``snaphu.conf.brief``).  All values are written as keyword/value pairs.
    """
    cost_upper = cost_mode.upper()
    init_upper = init_method.upper()

    config = textwrap.dedent(f"""\
    INFILE        {ifg_file}
    INFILEFORMAT  COMPLEX_DATA
    CORRFILE      {corr_file}
    CORRFILEFORMAT FLOAT_DATA
    OUTFILE       {out_file}
    OUTFILEFORMAT FLOAT_DATA
    LINELENGTH    {width}
    STATCOSTMODE  {cost_upper}
    INITMETHOD    {init_upper}
    NCORRLOOKS    {nlooks:.6f}
    NTILEROW      {ntiles[0]}
    NTILECOL      {ntiles[1]}
    ROWOVRLP      {tile_overlap}
    COLOVRLP      {tile_overlap}
    NPROC         {nproc}
    TILECOSTTHRESH {tile_cost_thresh}
    MINREGIONSIZE  {min_region_size}
    MINCONNCOMPFRAC {min_conncomp_frac:.6f}
    CONNCOMPTHRESH  {int(conncomp_thresh)}
    MAXNCOMPS       {max_ncomps}
    KPARDPSI        {phase_grad_window[0]}
    KPERPDPSI       {phase_grad_window[1]}
    LAMBDA          {wavelength:.8f}
    """)

    if cost_mode == "defo":
        config += textwrap.dedent(f"""\
        DEFOMAX_CYCLE     {defo_max_cycles:.6f}
        DEFOTHRESHFACTOR  {defo_thresh_factor:.6f}
        DEFOCONST         {defo_lay_const:.6f}
        """)

    if mask_file:
        config += f"BYTEMASKFILE   {mask_file}\n"
    if init_phase_file:
        config += f"ESTIMATEFILE   {init_phase_file}\n"

    if conncomp_out and conncomp_file:
        config += f"CONNCOMPFILE   {conncomp_file}\n"
        config += "CONNCOMPOUTTYPE UINT\n"

    return config


# ------------------------------------------------------------------------
# Non-finite input sanitization (SNAPHU-only)
# ------------------------------------------------------------------------
def _sanitize_nonfinite(ifg_data: np.ndarray, corr_data: np.ndarray,
                        mask_array: Optional[np.ndarray]):
    """Zero NaN/Inf pixels and return an exclusion mask for SNAPHU.

    SNAPHU aborts on non-finite float input ("NaN or infinity found in
    input float data").  Pixels that are NaN/Inf in the wrapped phase or in
    the correlation are zeroed and excluded via a mask so SNAPHU never sees
    non-finite data.  A caller-provided mask is combined (kept as 0 where the
    input is non-finite).

    Returns ``(ifg_clean, corr_clean, mask)`` — ``mask`` is ``None`` when the
    input is fully finite (no mask file is then written).
    """
    invalid = (np.isnan(ifg_data.real) | np.isnan(ifg_data.imag)
               | np.isinf(ifg_data.real) | np.isinf(ifg_data.imag))
    if corr_data is not None:
        invalid |= np.isnan(corr_data) | np.isinf(corr_data)

    if not np.any(invalid):
        return ifg_data, corr_data, mask_array

    ifg_clean = np.where(invalid, 0j, ifg_data)
    corr_clean = (np.where(invalid, 0.0, corr_data)
                  if corr_data is not None else None)
    if mask_array is None:
        mask = (~invalid).astype(np.uint8)
    else:
        mask = (mask_array & ~invalid).astype(np.uint8)
    return ifg_clean, corr_clean, mask


# ------------------------------------------------------------------------
# Core unwrapping function
# ------------------------------------------------------------------------
def _snaphu_unwrap(
    ifg_data: np.ndarray,
    corr_data: np.ndarray,
    width: int,
    nlooks: float,
    snaphu_bin: str,
    scratch_dir: Path,
    cost_mode: str = "smooth",
    init_method: str = "mcf",
    ntiles: Tuple[int, int] = (1, 1),
    tile_overlap: int = 0,
    nproc: int = 1,
    tile_cost_thresh: int = 500,
    min_region_size: int = 100,
    min_conncomp_frac: float = 0.01,
    conncomp_thresh: float = 300,
    max_ncomps: int = 32,
    phase_grad_window: Tuple[int, int] = (7, 7),
    defo_max_cycles: float = 1.2,
    defo_thresh_factor: float = 1.2,
    defo_lay_const: float = 0.9,
    wavelength: float = 0.05546576,
    mask_array: Optional[np.ndarray] = None,
    init_phase_array: Optional[np.ndarray] = None,
    conncomp_out: bool = True,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Run SNAPHU binary on a single interferogram.

    Returns (unwrapped_float32_array, conncomp_uint32_array_or_None).
    """
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # SNAPHU aborts on non-finite input: zero NaN/Inf pixels (wrapped phase
    # and correlation) and exclude them via a mask file.
    ifg_data, corr_data, mask_array = _sanitize_nonfinite(
        ifg_data, corr_data, mask_array)
    if mask_array is not None and np.any(mask_array == 0):
        logger.info("snaphu: %d non-finite pixel(s) masked out",
                    int(np.count_nonzero(mask_array == 0)))

    # Write raw binary files for SNAPHU
    igram_file = str(scratch_dir / "snaphu_ifg.c8")
    corr_file = str(scratch_dir / "snaphu_corr.f4")
    out_file = str(scratch_dir / "snaphu_unw.f4")
    conncomp_file = str(scratch_dir / "snaphu_conncomp.u4") if conncomp_out else None
    config_file = str(scratch_dir / "snaphu.conf")

    igram_flat = ifg_data.ravel().astype(np.complex64)
    with open(igram_file, "wb") as f:
        f.write(igram_flat.tobytes())

    corr_flat = corr_data.ravel().astype(np.float32)
    with open(corr_file, "wb") as f:
        f.write(corr_flat.tobytes())

    # Optional mask
    mask_file = None
    if mask_array is not None:
        mask_file = str(scratch_dir / "snaphu_mask.u1")
        mask_flat = mask_array.ravel().astype(np.uint8)
        with open(mask_file, "wb") as f:
            f.write(mask_flat.tobytes())

    # Optional initial phase estimate
    init_phase_file = None
    if init_phase_array is not None:
        init_phase_file = str(scratch_dir / "snaphu_init.f4")
        init_flat = init_phase_array.ravel().astype(np.float32)
        with open(init_phase_file, "wb") as f:
            f.write(init_flat.tobytes())

    # Build & write config
    config = _build_snaphu_config(
        ifg_file=igram_file, width=width, corr_file=corr_file,
        out_file=out_file, nlooks=nlooks, cost_mode=cost_mode,
        init_method=init_method, ntiles=ntiles, tile_overlap=tile_overlap,
        nproc=nproc, tile_cost_thresh=tile_cost_thresh,
        min_region_size=min_region_size,
        min_conncomp_frac=min_conncomp_frac,
        conncomp_thresh=conncomp_thresh, max_ncomps=max_ncomps,
        phase_grad_window=phase_grad_window,
        defo_max_cycles=defo_max_cycles,
        defo_thresh_factor=defo_thresh_factor,
        defo_lay_const=defo_lay_const, wavelength=wavelength,
        mask_file=mask_file, init_phase_file=init_phase_file,
        conncomp_file=conncomp_file, conncomp_out=conncomp_out,
    )
    with open(config_file, "w") as f:
        f.write(config)

    # Build & run command:  snaphu -f config
    cmd = [snaphu_bin, "-f", config_file]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=str(scratch_dir),
    )
    # Register so a failing engine run can terminate this subprocess
    # (terminate_active_snaphu) instead of waiting out the whole unwrap.
    with _ACTIVE_SNAPHU_LOCK:
        _ACTIVE_SNAPHU.add(proc)
    try:
        out, err = proc.communicate()
    finally:
        with _ACTIVE_SNAPHU_LOCK:
            _ACTIVE_SNAPHU.discard(proc)
    if proc.returncode != 0:
        raise RuntimeError(
            f"snaphu failed (exit {proc.returncode}):\n"
            f"STDOUT: {out[-500:]}\n"
            f"STDERR: {err[-500:]}"
        )

    # Read output: unwrapped (float32)
    total_pixels = ifg_data.size
    unw = np.fromfile(out_file, dtype=np.float32, count=total_pixels)
    unw = unw.reshape(ifg_data.shape)

    # Masked (non-finite / user-excluded) pixels get a deterministic 0 output
    if mask_array is not None:
        excluded = mask_array == 0
        unw[excluded] = 0.0

    conncomp = None
    if conncomp_out and conncomp_file and os.path.exists(conncomp_file):
        conncomp = np.fromfile(conncomp_file, dtype=np.uint32, count=total_pixels)
        conncomp = conncomp.reshape(ifg_data.shape)
        # Kept as uint32: casting to uint16 would silently wrap above 65535
        # connected components (GDAL GDT_UInt32 supports uint32 natively).
        if mask_array is not None:
            conncomp[excluded] = 0

    return unw, conncomp


# ------------------------------------------------------------------------
# Single interferogram wrapper (reads rasters, calls SNAPHU, writes outputs)
# ------------------------------------------------------------------------
def _unwrap_single(
    ifg_path: Path,
    output_dir: Path,
    processor: str,
    snaphu_bin: str,
    cor_path: Optional[Path] = None,
    nlooks: float = 1.0,
    cost_mode: str = _DEFAULT_COST_MODE,
    init_method: str = _DEFAULT_INIT_METHOD,
    mask_path: Optional[Path] = None,
    init_phase_path: Optional[Path] = None,
    ntiles: Tuple[int, int] = (1, 1),
    tile_overlap: int = 0,
    nproc: int = 1,
    tile_cost_thresh: int = _DEFAULT_TILE_COST_THRESH,
    min_region_size: int = _DEFAULT_MIN_REGION_SIZE,
    min_conncomp_frac: float = _DEFAULT_MIN_CONNCOMP_FRAC,
    conncomp_thresh: float = _DEFAULT_CONNCOMP_THRESH,
    max_ncomps: int = _DEFAULT_MAX_NCOMPS,
    phase_grad_window: Tuple[int, int] = _DEFAULT_PHASE_GRAD_WINDOW,
    defo_max_cycles: float = _DEFAULT_DEFO_MAX_CYCLES,
    defo_thresh_factor: float = _DEFAULT_DEFO_THRESH_FACTOR,
    defo_lay_const: float = _DEFAULT_DEFO_LAY_CONST,
    wavelength: float = _DEFAULT_LAMBDA,
    conncomp_out: bool = True,
    keep_scratch: bool = False,
) -> Tuple[Path, Path]:
    """
    Unwrap a single interferogram using SNAPHU binary.

    ``cor_path`` may be ``None`` — SNAPHU then runs with a uniform weight of
    1 (no coherence constraint).

    Returns paths to (unwrapped_file, connected_component_file).
    """
    if not ifg_path.exists():
        raise FileNotFoundError(f"Interferogram not found: {ifg_path}")
    if cor_path is not None and not cor_path.exists():
        raise FileNotFoundError(f"Correlation file not found: {cor_path}")

    if snaphu_bin is None:
        snaphu_bin = find_snaphu_binary()

    # Fixed structure: output_dir/{date1}_{date2}/{variant}.unw[.tif]
    variant = variant_of(ifg_path, processor)
    dp = extract_date_pair(ifg_path.parent.name) or extract_date_pair(ifg_path.name)
    if dp:
        date1, date2 = dp.split('_')
        unw_path_out = unw_path(output_dir, date1, date2, variant=variant, processor=processor)
        conn_path_out = conncomp_path(output_dir, date1, date2, variant=variant, processor=processor)
    else:
        # Flat fallback (standalone usage): keep the input stem
        stem = ifg_path.stem
        for ext in [".int", ".tif"]:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
        if processor == "isce2":
            unw_path_out = output_dir / f"{stem}{UNW_SUFFIX_ISCE2}"
            conn_path_out = output_dir / f"{stem}{CONNCOMP_SUFFIX_ISCE2}"
        else:
            unw_path_out = output_dir / f"{stem}{UNW_SUFFIX_ISCE3}"
            conn_path_out = output_dir / f"{stem}{CONNCOMP_SUFFIX_ISCE3}"

    # Skip if output already exists
    if unw_path_out.exists() and (not conncomp_out or conn_path_out.exists()):
        logger.info("Skipping existing output: %s", unw_path_out.name)
        return unw_path_out, conn_path_out

    # Read input rasters via GDAL
    ifg_ds = gdal.Open(str(ifg_path), gdal.GA_ReadOnly)
    if ifg_ds is None:
        raise RuntimeError(f"Cannot open {ifg_path}")
    ifg_data = ifg_ds.GetRasterBand(1).ReadAsArray()
    if ifg_data.dtype not in (np.complex64, np.complex128):
        ifg_data = ifg_data.astype(np.complex64)
    rows, cols = ifg_data.shape
    # Cache metadata to avoid redundant GDAL re-opens in output writers
    _ref_gt = ifg_ds.GetGeoTransform()
    _ref_proj = ifg_ds.GetProjection()
    ifg_ds = None

    if cor_path is not None:
        corr_ds = gdal.Open(str(cor_path), gdal.GA_ReadOnly)
        if corr_ds is None:
            raise RuntimeError(f"Cannot open {cor_path}")
        corr_data = corr_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        corr_ds = None
        if corr_data.shape != (rows, cols):
            raise ValueError(
                f"Coherence raster {cor_path} has shape {corr_data.shape}, "
                f"expected {(rows, cols)} matching the interferogram — "
                "refusing to run SNAPHU with misaligned weights")
    else:
        # No coherence file: SNAPHU weight = 1 (uniform) — write an
        # all-ones correlation raster so the cost weighting is deterministic.
        logger.info("No coherence file — SNAPHU weight set to 1 (uniform)")
        corr_data = np.ones((rows, cols), dtype=np.float32)

    # Zero-valued magnitude → invalid
    mask_zeros = (ifg_data.real == 0) & (ifg_data.imag == 0)

    # External mask: .wbd water masks are warped onto the ifg grid (SNAPHU
    # semantics: 1 = valid/land, 0 = water/excluded); any other GDAL-readable
    # file is used as-is.
    mask_array = None
    if mask_path is not None and mask_path.exists():
        if str(mask_path).lower().endswith('.wbd'):
            mask_array = wbd_to_mask_array(
                str(mask_path), str(ifg_path), invert=True)
            if mask_array is not None:
                logger.info("wbd water mask applied: %d water pixel(s) excluded",
                            int(np.count_nonzero(mask_array == 0)))
        else:
            mask_ds = gdal.Open(str(mask_path), gdal.GA_ReadOnly)
            if mask_ds is None:
                raise RuntimeError(f"Cannot open mask {mask_path}")
            mask_array = mask_ds.GetRasterBand(1).ReadAsArray().astype(np.uint8)
            mask_ds = None
            if mask_array.shape != (rows, cols):
                raise ValueError(
                    f"Mask raster {mask_path} has shape {mask_array.shape}, "
                    f"expected {(rows, cols)} matching the interferogram")

    # Initial phase
    init_phase_array = None
    if init_phase_path is not None and init_phase_path.exists():
        ip_ds = gdal.Open(str(init_phase_path), gdal.GA_ReadOnly)
        if ip_ds is None:
            raise RuntimeError(f"Cannot open init-phase {init_phase_path}")
        init_phase_array = ip_ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
        ip_ds = None
        if init_phase_array.shape != (rows, cols):
            raise ValueError(
                f"Init-phase raster {init_phase_path} has shape "
                f"{init_phase_array.shape}, expected {(rows, cols)} "
                "matching the interferogram")

    # Scratch directory
    scratch_dir = Path(tempfile.mkdtemp(prefix="snaphu_scratch_"))

    try:
        logger.debug(
            "Unwrapping %s (cost=%s init=%s ntiles=%s nproc=%s defomax=%.2f)",
            ifg_path.name, cost_mode, init_method, ntiles, nproc, defo_max_cycles,
        )

        unw_array, conncomp_array = _snaphu_unwrap(
            ifg_data=ifg_data, corr_data=corr_data, width=cols,
            nlooks=nlooks, snaphu_bin=snaphu_bin,
            scratch_dir=scratch_dir, cost_mode=cost_mode,
            init_method=init_method, ntiles=ntiles, tile_overlap=tile_overlap,
            nproc=nproc, tile_cost_thresh=tile_cost_thresh,
            min_region_size=min_region_size,
            min_conncomp_frac=min_conncomp_frac,
            conncomp_thresh=conncomp_thresh, max_ncomps=max_ncomps,
            phase_grad_window=phase_grad_window,
            defo_max_cycles=defo_max_cycles,
            defo_thresh_factor=defo_thresh_factor,
            defo_lay_const=defo_lay_const, wavelength=wavelength,
            mask_array=mask_array, init_phase_array=init_phase_array,
            conncomp_out=conncomp_out,
        )

        # Mask invalid pixels
        if np.any(mask_zeros):
            unw_array[mask_zeros] = 0.0
            if conncomp_array is not None:
                conncomp_array[mask_zeros] = 0

        # Write outputs
        if processor == "isce2":
            _write_envi_output(unw_path_out, unw_array, rows, cols, _ref_gt, _ref_proj,
                               nodata=0.0, dtype=np.float32)
            create_xml_for_binary(unw_path_out, family="intimage",
                                  description="Unwrapped interferogram phase")

            if conncomp_out and conncomp_array is not None:
                _write_envi_output(conn_path_out, conncomp_array, rows, cols, _ref_gt, _ref_proj,
                                   nodata=0, dtype=np.uint16)
                create_xml_for_binary(conn_path_out, family="image",
                                      description="Connected components")
        else:
            _write_geotiff_output(unw_path_out, unw_array, rows, cols, _ref_gt, _ref_proj, nodata=0.0)
            if conncomp_out and conncomp_array is not None:
                _write_geotiff_output(conn_path_out, conncomp_array, rows, cols, _ref_gt, _ref_proj,
                                      nodata=0, dtype=gdal.GDT_UInt16)

    finally:
        if not keep_scratch:
            shutil.rmtree(scratch_dir, ignore_errors=True)

    return unw_path_out, conn_path_out


def _write_envi_output(
    output_path: Path, array: np.ndarray,
    rows: int, cols: int, gt: tuple, proj: str,
    nodata: float = 0.0, dtype: np.dtype = np.float32,
) -> None:

    if array.shape != (rows, cols):
        raise ValueError(f"Array shape {array.shape} != ({rows}, {cols})")

    gdal_dtype = gdal.GDT_Float32 if np.issubdtype(dtype, np.floating) else gdal.GDT_UInt16
    driver = gdal.GetDriverByName("ENVI")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: create at a temp path, rename only after a successful
    # close — an interrupted run leaves no partial product at `output_path`.
    tmp = f"{output_path}.tmp"
    try:
        ds_out = driver.Create(tmp, cols, rows, 1, gdal_dtype)
        if ds_out is None:
            raise RuntimeError(f"Failed to create {output_path}")
        if gt is not None:
            ds_out.SetGeoTransform(gt)
        if proj:
            ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.WriteArray(array)
        if np.isfinite(nodata):
            band.SetNoDataValue(nodata)
        band.FlushCache()
        ds_out = None
        os.replace(tmp, str(output_path))
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


def _write_geotiff_output(
    output_path: Path, array: np.ndarray,
    rows: int, cols: int, gt: tuple, proj: str,
    nodata: float = 0.0, dtype: Optional[int] = None,
) -> None:

    if array.shape != (rows, cols):
        raise ValueError(f"Array shape {array.shape} != ({rows}, {cols})")

    if dtype is None:
        gdal_dtype = gdal.GDT_Float32
    else:
        gdal_dtype = dtype
    driver = gdal.GetDriverByName("GTiff")
    opts = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: create at a temp path, rename only after a successful
    # close — an interrupted run leaves no partial product at `output_path`.
    tmp = f"{output_path}.tmp"
    try:
        ds_out = driver.Create(tmp, cols, rows, 1, gdal_dtype, opts)
        if ds_out is None:
            raise RuntimeError(f"Failed to create {output_path}")
        ds_out.SetGeoTransform(gt)
        if proj:
            ds_out.SetProjection(proj)
        band = ds_out.GetRasterBand(1)
        band.WriteArray(array)
        if np.isfinite(nodata):
            band.SetNoDataValue(nodata)
        band.FlushCache()
        ds_out = None
        os.replace(tmp, str(output_path))
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------------
# Batch unwrapping
# ------------------------------------------------------------------------
def unwrap_batch(
    ifg_paths: List[Path],
    cor_paths: List[Path],
    nlooks: float,
    output_dir: Path,
    processor: str,
    max_workers: int = 1,
    snaphu_bin: Optional[str] = None,
    cost_mode: str = _DEFAULT_COST_MODE,
    init_method: str = _DEFAULT_INIT_METHOD,
    mask_paths: Optional[List[Optional[Path]]] = None,
    init_phase_paths: Optional[List[Optional[Path]]] = None,
    ntiles: Tuple[int, int] = (1, 1),
    tile_overlap: int = 0,
    nproc: int = 1,
    tile_cost_thresh: int = _DEFAULT_TILE_COST_THRESH,
    min_region_size: int = _DEFAULT_MIN_REGION_SIZE,
    min_conncomp_frac: float = _DEFAULT_MIN_CONNCOMP_FRAC,
    conncomp_thresh: float = _DEFAULT_CONNCOMP_THRESH,
    max_ncomps: int = _DEFAULT_MAX_NCOMPS,
    phase_grad_window: Tuple[int, int] = _DEFAULT_PHASE_GRAD_WINDOW,
    defo_max_cycles: float = _DEFAULT_DEFO_MAX_CYCLES,
    defo_thresh_factor: float = _DEFAULT_DEFO_THRESH_FACTOR,
    defo_lay_const: float = _DEFAULT_DEFO_LAY_CONST,
    wavelength: float = _DEFAULT_LAMBDA,
    conncomp_out: bool = True,
    keep_scratch: bool = False,
) -> Tuple[List[Path], List[Path]]:
    if len(ifg_paths) != len(cor_paths):
        raise ValueError(f"Mismatched counts: {len(ifg_paths)} vs {len(cor_paths)}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if snaphu_bin is None:
        snaphu_bin = find_snaphu_binary()
    logger.info("Using snaphu binary: %s", snaphu_bin)

    if mask_paths is None:
        mask_paths = [None] * len(ifg_paths)
    if init_phase_paths is None:
        init_phase_paths = [None] * len(ifg_paths)

    unwrapped = [None] * len(ifg_paths)
    conncomps = [None] * len(ifg_paths)

    common_kw = dict(
        output_dir=output_dir, processor=processor, snaphu_bin=snaphu_bin,
        cost_mode=cost_mode, init_method=init_method,
        ntiles=ntiles, tile_overlap=tile_overlap, nproc=nproc,
        tile_cost_thresh=tile_cost_thresh, min_region_size=min_region_size,
        min_conncomp_frac=min_conncomp_frac, conncomp_thresh=conncomp_thresh,
        max_ncomps=max_ncomps, phase_grad_window=phase_grad_window,
        defo_max_cycles=defo_max_cycles, defo_thresh_factor=defo_thresh_factor,
        defo_lay_const=defo_lay_const, wavelength=wavelength,
        conncomp_out=conncomp_out, keep_scratch=keep_scratch,
    )

    if max_workers <= 1:
        for i in tqdm_progress(range(len(ifg_paths)), desc="Unwrapping", unit="ifg"):
            unw, conn = _unwrap_single(
                ifg_path=ifg_paths[i], cor_path=cor_paths[i], nlooks=nlooks,
                mask_path=mask_paths[i], init_phase_path=init_phase_paths[i],
                **common_kw,
            )
            unwrapped[i] = unw
            conncomps[i] = conn
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {}
            for i in range(len(ifg_paths)):
                fut = executor.submit(
                    _unwrap_single,
                    ifg_path=ifg_paths[i], cor_path=cor_paths[i], nlooks=nlooks,
                    mask_path=mask_paths[i], init_phase_path=init_phase_paths[i],
                    **common_kw,
                )
                futures[fut] = i

            for fut in tqdm_progress(as_completed(futures), total=len(ifg_paths),
                                     desc="Unwrapping", unit="ifg"):
                idx = futures[fut]
                unw, conn = fut.result()
                unwrapped[idx] = unw
                conncomps[idx] = conn

    if any(p is None for p in unwrapped):
        raise RuntimeError("Some interferograms were not unwrapped successfully.")

    return unwrapped, conncomps


# ------------------------------------------------------------------------
# File matching
# ------------------------------------------------------------------------
def find_matching_files(
    ifg_dir: Path, cor_dir: Path,
    ifg_pattern: str, cor_pattern: str, processor: str,
) -> Tuple[List[Path], List[Path]]:
    # Recursive globs support the date-pair directory structure
    # (e.g. **/*.int.tif, **/*_phsig.coh.tif)
    ifg_files = sorted(ifg_dir.glob(ifg_pattern))
    cor_files = sorted(cor_dir.glob(cor_pattern))

    if not ifg_files:
        raise FileNotFoundError(
            f"No interferogram files in {ifg_dir} matching '{ifg_pattern}'")
    if not cor_files:
        raise FileNotFoundError(
            f"No correlation files in {cor_dir} matching '{cor_pattern}'")
    # Pair by the date-pair key extracted from each filename instead of by
    # positional order: a single extra/renamed file in either directory used
    # to silently shift every subsequent pairing.
    def _dp_key(p: Path):
        m = re.search(r'(\d{8})_(\d{8})', p.name)
        return m.group(0) if m else p.name
    cor_by_dp = {_dp_key(c): c for c in cor_files}
    paired = []
    missing = []
    for f in ifg_files:
        c = cor_by_dp.get(_dp_key(f))
        if c is None:
            missing.append(f.name)
            continue
        paired.append((f, c))
    if missing:
        raise ValueError(
            f"{len(missing)} interferogram(s) have no matching coherence "
            f"file: {', '.join(missing[:8])}")
    ifg_out, cor_out = zip(*paired) if paired else ([], [])
    return list(ifg_out), list(cor_out)


# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_arguments()

    setup_logging(args.verbose)

    try:
        if not args.ifg_dir.exists():
            raise FileNotFoundError(f"ifg-dir not found: {args.ifg_dir}")
        if not args.cor_dir.exists():
            raise FileNotFoundError(f"cor-dir not found: {args.cor_dir}")

        ifg_files, cor_files = find_matching_files(
            args.ifg_dir, args.cor_dir,
            args.ifg_pattern, args.cor_pattern, args.processor,
        )

        logger.info("Found %d interferogram/correlation pairs", len(ifg_files))
        logger.info("Cost mode: %s, Init: %s, Nlooks: %s",
                     args.cost_mode, args.init_method, args.nlooks)
        logger.info("DEFOMAX: %.2f cycles, Ntiles: %s, Nproc: %s",
                     args.defo_max_cycles, args.ntiles, args.nproc)

        args.output_dir.mkdir(parents=True, exist_ok=True)

        unwrapped, conncomps = unwrap_batch(
            ifg_paths=ifg_files, cor_paths=cor_files,
            nlooks=args.nlooks, output_dir=args.output_dir,
            processor=args.processor,
            max_workers=args.max_workers if args.max_workers else 1,
            snaphu_bin=getattr(args, "snaphu_binary", None),
            cost_mode=args.cost_mode,
            init_method=args.init_method,
            mask_paths=[args.mask_file] * len(ifg_files) if args.mask_file else None,
            init_phase_paths=([args.init_phase] * len(ifg_files)
                              if args.init_phase else None),
            ntiles=args.ntiles, tile_overlap=args.tile_overlap,
            nproc=args.nproc,
            tile_cost_thresh=args.tile_cost_thresh,
            min_region_size=args.min_region_size,
            min_conncomp_frac=args.min_conncomp_frac,
            conncomp_thresh=args.conncomp_thresh,
            max_ncomps=args.max_ncomps,
            phase_grad_window=args.phase_grad_window,
            defo_max_cycles=args.defo_max_cycles,
            defo_thresh_factor=args.defo_thresh_factor,
            defo_lay_const=args.defo_lay_const,
            wavelength=args.wavelength,
            conncomp_out=not args.no_conncomp_out,
            keep_scratch=args.keep_scratch,
        )

        logger.info("Generated %d unwrapped interferograms", len(unwrapped))
        logger.info("Generated %d connected component files",
                     sum(1 for c in conncomps if c is not None))

        return 0

    except FileNotFoundError as e:
        logger.error("File error: %s", e)
        return 1
    except ValueError as e:
        logger.error("Configuration error: %s", e)
        return 1
    except Exception as e:
        logger.error("Unwrapping failed: %s", e)
        return 2


if __name__ == "__main__":
    sys.exit(main())
