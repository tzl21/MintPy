#!/usr/bin/env python3
"""GDAL-based raster stitching utilities — replaces dolphin.stitching.merge_images."""

import logging
import math
import os
import sys
# MUST be set before any osgeo import
for _pj in (
    os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'proj'),
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'proj'),
    '/home/tangzhenli/tools/bash/envs/ISCE3/share/proj',
):
    if os.path.isdir(_pj) and 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = _pj
        break


from pathlib import Path  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
from typing import Any, Dict, List, Optional, Tuple, Union  # noqa: E402  (imports follow the mandatory PROJ_LIB block)

import numpy as np  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
from osgeo import gdal, osr  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import shutil  # noqa: E402  (imports follow the mandatory PROJ_LIB block)

gdal.UseExceptions()

logger = logging.getLogger(__name__)

DEFAULT_TIFF_OPTIONS = [
    "COMPRESS=LZW",
    "TILED=YES",
    "BIGTIFF=IF_SAFER",
    "NUM_THREADS=ALL_CPUS",
]

DEFAULT_ENVI_OPTIONS: List[str] = []


def load_gdal(filepath: str, band: int = 1) -> np.ndarray:
    """Read a GDAL-readable raster band into a numpy array."""
    ds = gdal.Open(filepath, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open {filepath}: {gdal.GetLastErrorMsg()}")
    data = ds.GetRasterBand(band).ReadAsArray()
    ds = None
    return data


def write_arr(
    arr: np.ndarray,
    like_filename: str,
    output_name: str,
    driver: str = "GTiff",
    options: Optional[List[str]] = None,
    nodata: Optional[float] = None,
    band: int = 1,
) -> str:
    """Write a numpy array to a GDAL raster, copying geo-metadata from a reference file."""
    ref_ds = gdal.Open(like_filename, gdal.GA_ReadOnly)
    if ref_ds is None:
        raise RuntimeError(f"Cannot open reference {like_filename}")
    rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
    gt = ref_ds.GetGeoTransform()
    proj = ref_ds.GetProjection()
    ref_ds = None

    if arr.shape != (rows, cols):
        raise ValueError(f"Array shape {arr.shape} != ({rows}, {cols})")

    if options is None:
        options = []

    if np.issubdtype(arr.dtype, np.complexfloating):
        arr = arr.astype(np.complex64)
        gdal_dtype = gdal.GDT_CFloat32
    elif np.issubdtype(arr.dtype, np.floating):
        if arr.dtype == np.float64:
            gdal_dtype = gdal.GDT_Float64
        else:
            arr = arr.astype(np.float32)
            gdal_dtype = gdal.GDT_Float32
    elif np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.int16:
            gdal_dtype = gdal.GDT_Int16
        elif arr.dtype == np.int32:
            gdal_dtype = gdal.GDT_Int32
        elif arr.dtype == np.uint8:
            gdal_dtype = gdal.GDT_Byte
        elif arr.dtype == np.uint16:
            gdal_dtype = gdal.GDT_UInt16
        elif arr.dtype == np.uint32:
            gdal_dtype = gdal.GDT_UInt32
        else:
            arr = arr.astype(np.int32)
            gdal_dtype = gdal.GDT_Int32
    else:
        arr = arr.astype(np.float32)
        gdal_dtype = gdal.GDT_Float32

    drv = gdal.GetDriverByName(driver)
    if drv is None:
        raise RuntimeError(f"GDAL driver '{driver}' not available")

    out_dir = os.path.dirname(output_name)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    out_ds = drv.Create(output_name, cols, rows, 1, gdal_dtype, options)
    if out_ds is None:
        raise RuntimeError(f"Failed to create {output_name}")
    if gt is not None:
        out_ds.SetGeoTransform(gt)
    if proj:
        out_ds.SetProjection(proj)
    out_band = out_ds.GetRasterBand(band)
    out_band.WriteArray(arr)
    if nodata is not None:
        out_band.SetNoDataValue(nodata)
    out_band.FlushCache()
    out_ds = None
    return output_name


def _write_geotiff(out_path: Path, arr: np.ndarray, geotransform: tuple,
                   projection: str, overwrite: bool):
    """Write a numpy array as GeoTIFF with proper compression."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        logger.info(f"Skipping existing: {out_path}")
        return

    rows, cols = arr.shape
    if np.issubdtype(arr.dtype, np.complexfloating):
        gdt = gdal.GDT_CFloat32
    elif arr.dtype == np.float64:
        gdt = gdal.GDT_Float64
    else:
        gdt = gdal.GDT_Float32

    drv = gdal.GetDriverByName('GTiff')
    ds = drv.Create(str(out_path), cols, rows, 1, gdt,
                    ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'])
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection)
    ds.GetRasterBand(1).WriteArray(arr)
    ds.GetRasterBand(1).SetNoDataValue(0)
    ds = None


# ---------------------------------------------------------------------------
# stitch_arrays — pure numpy pixel-offset copy (gdal_merge style)
# ---------------------------------------------------------------------------
def stitch_arrays(
    file_list: List[Path],
    bbox_wsen: Optional[Tuple[float, float, float, float]] = None,
    epsg_utm: int = 32605,
    method: str = 'last',
    margin_px: int = 0,
) -> Tuple[np.ndarray, Tuple[float, ...], str]:
    """Stitch geocoded GeoTIFF arrays via gdal_merge-style pixel-offset copy.

    Pure numpy implementation — no intermediate VRT, no GDAL Warp.
    Each input file is placed at its correct geographic position based on
    its geotransform, clipped to the output bbox.

    Memory-efficient two-pass design: the first pass reads only the
    geotransforms (no raster data), the second pass streams each source
    array into the pre-allocated output — only one source array is ever
    held in memory at a time.

    Parameters
    ----------
    file_list : list of Path
        GeoTIFF files to stitch. Each file must have same CRS and resolution.
    bbox_wsen : tuple or None
        ``(west, south, east, north)`` in EPSG:4326 for output clipping.
        None for union of all source extents (default: full-burst stitch).
    epsg_utm : int
        UTM EPSG code of input files (default 32605).
    method : {'last', 'first'}
        Overlap resolution: 'last' = later sources overwrite earlier.
    margin_px : int
        Extra input pixels to include on each side of the bbox.  The window
        is first snapped to the input pixel grid (so the output origin is
        grid-aligned — matching the HDF5 crop path instead of landing on a
        fractional bbox corner), then extended by ``margin_px`` pixels.
        ``1`` reproduces the HDF5 crop path's interpolation margin exactly.

    Returns
    -------
    stitched : np.ndarray  ``[rows, cols]``
    out_gt : tuple  GDAL geotransform
    proj_wkt : str  CRS WKT
    """
    try:
        from pyproj import Transformer
        tf = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg_utm}', always_xy=True)
        _use_pyproj = True
    except ImportError:
        _use_pyproj = False

    if not file_list:
        raise ValueError("file_list is empty")

    # ---- Pass 1: metadata only (geotransforms, no raster data) ----
    pieces = []
    proj_wkt = None
    sample_dtype = None

    for fp in file_list:
        ds = gdal.Open(str(fp))
        if ds is None:
            logger.warning(f"Cannot open {fp}, skipping")
            continue
        gt = ds.GetGeoTransform()
        p_wkt = ds.GetProjection()
        if proj_wkt is None:
            proj_wkt = p_wkt
        elif p_wkt and p_wkt != proj_wkt:
            raise ValueError(
                f"stitch_arrays: source {fp} has a different CRS than the "
                "first file (docstring contract: all sources must share "
                "CRS and resolution)")
        if sample_dtype is None:
            band = ds.GetRasterBand(1)
            if band is not None:
                sample_dtype = gdal_to_numpy_dtype(band.DataType)
        rows, cols = ds.RasterYSize, ds.RasterXSize
        x0, dx, _, y0, _, dy = gt
        pieces.append({
            'path': str(fp),
            'rows': rows, 'cols': cols,
            'x0': x0, 'y0': y0,
            'x1': x0 + cols * dx, 'y1': y0 + rows * dy,
            'dx': dx, 'dy': dy,
        })
        ds = None

    if not pieces:
        raise ValueError("No valid input files")

    dx = pieces[0]['dx']
    dy = pieces[0]['dy']
    if sample_dtype is None:
        sample_dtype = np.float32
    # validate uniform pixel resolution across sources (docstring contract)
    bad_res = [p['path'] for p in pieces
               if abs(p['dx'] - dx) > 1e-6 or abs(p['dy'] - dy) > 1e-6]
    if bad_res:
        raise ValueError(
            "stitch_arrays: %d source(s) have a different pixel resolution "
            "than the first file: %s", len(bad_res), bad_res[:3])

    # Union extent of all sources
    # Union extent of all sources (used to clamp a bbox to the data)
    union_x0 = min(p['x0'] for p in pieces)
    union_x1 = max(p['x1'] for p in pieces)
    union_y0 = max(p['y0'] for p in pieces)
    union_y1 = min(p['y1'] for p in pieces)

    # Clip to bbox (EPSG:4326 → UTM)
    if bbox_wsen is not None:
        if _use_pyproj:
            xs, ys = tf.transform(
                [bbox_wsen[0], bbox_wsen[2], bbox_wsen[2], bbox_wsen[0]],
                [bbox_wsen[1], bbox_wsen[1], bbox_wsen[3], bbox_wsen[3]],
            )
        else:
            # osgeo.osr fallback: EPSG:4326 → UTM (swap lon/lat for axis order)
            src_srs = osr.SpatialReference()
            src_srs.ImportFromEPSG(4326)
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(epsg_utm)
            t_4326 = osr.CoordinateTransformation(src_srs, dst_srs)
            pts = [
                t_4326.TransformPoint(bbox_wsen[1], bbox_wsen[0]),
                t_4326.TransformPoint(bbox_wsen[1], bbox_wsen[2]),
                t_4326.TransformPoint(bbox_wsen[3], bbox_wsen[2]),
                t_4326.TransformPoint(bbox_wsen[3], bbox_wsen[0]),
            ]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]

        bbox_xmin, bbox_ymin = min(xs), min(ys)
        bbox_xmax, bbox_ymax = max(xs), max(ys)

        # Snap the window to the input pixel grid (matching the HDF5 crop
        # path, which aligns to the input grid): take the input pixels whose
        # coordinates fall inside the (data-clamped) bbox, then extend by
        # ``margin_px`` on each side.  Anchoring to the raw bbox corner
        # instead would land the output origin on a fractional pixel and
        # shift the grid by up to 1 px vs grid-aligned crops of the same
        # data (the old behaviour).
        x0, y0 = pieces[0]['x0'], pieces[0]['y0']
        bx0 = max(bbox_xmin, union_x0)
        bx1 = min(bbox_xmax, union_x1)
        by0 = max(bbox_ymin, union_y1)
        by1 = min(bbox_ymax, union_y0)
        m = int(margin_px)
        w = math.ceil((bx0 - x0) / dx - 1e-9)
        e = math.floor((bx1 - x0) / dx + 1e-9)
        n = math.ceil((by1 - y0) / dy - 1e-9)
        s = math.floor((by0 - y0) / dy + 1e-9)
        # Output pixels span indices [w-m .. e+m] (and [n-m .. s+m] in y);
        # the east/south edge of the last pixel is one pixel beyond, so the
        # extent uses (index + m + 1) on the max side (pixel-count parity
        # with the HDF5 crop path's exclusive-end slices).
        ulx = x0 + (w - m) * dx
        lrx = x0 + (e + m + 1) * dx
        uly = y0 + (n - m) * dy
        lry = y0 + (s + m + 1) * dy
    else:
        ulx, lrx, uly, lry = union_x0, union_x1, union_y0, union_y1

    if lrx <= ulx:
        raise ValueError(f"Invalid extent: ulx={ulx}, lrx={lrx}")
    if (dy < 0 and lry >= uly) or (dy > 0 and lry <= uly):
        raise ValueError(f"Invalid y extent: uly={uly}, lry={lry}")

    # Output grid (gdal_merge style: int((extent / pixel_size) + 0.5))
    out_cols = int((lrx - ulx) / dx + 0.5)
    out_rows = int((lry - uly) / dy + 0.5)
    out_gt = (ulx, dx, 0, uly, 0, dy)

    stitched = np.zeros((out_rows, out_cols), dtype=sample_dtype)
    is_complex = np.issubdtype(sample_dtype, np.complexfloating)

    items = pieces if method == 'last' else list(reversed(pieces))

    # ---- Pass 2: stream each source array into the output ----
    for p in items:
        ds = gdal.Open(p['path'])
        if ds is None:
            continue
        arr = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        if arr is None:
            continue
        arr = arr.astype(sample_dtype, copy=False)
        src_x0, src_y0 = p['x0'], p['y0']

        xoff = int((src_x0 - ulx) / dx)
        yoff = int((src_y0 - uly) / dy)

        src_r0 = max(0, -yoff)
        src_r1 = min(arr.shape[0], out_rows - yoff)
        src_c0 = max(0, -xoff)
        src_c1 = min(arr.shape[1], out_cols - xoff)
        dst_r0 = max(0, yoff)
        dst_r1 = min(out_rows, yoff + arr.shape[0])
        dst_c0 = max(0, xoff)
        dst_c1 = min(out_cols, xoff + arr.shape[1])

        h = min(src_r1 - src_r0, dst_r1 - dst_r0)
        w = min(src_c1 - src_c0, dst_c1 - dst_c0)
        if h <= 0 or w <= 0:
            continue

        src = arr[src_r0:src_r0 + h, src_c0:src_c0 + w]
        valid = np.isfinite(src) & (np.abs(src) > 1e-6)

        dst = stitched[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w]
        if method == 'last':
            dst[valid] = src[valid]
        else:
            empty = ~np.isfinite(dst) | (np.abs(dst) < 1e-6) if is_complex else (dst == 0)
            write_mask = valid & empty
            dst[write_mask] = src[write_mask]
        stitched[dst_r0:dst_r0 + h, dst_c0:dst_c0 + w] = dst

    return stitched, out_gt, proj_wkt


def gdal_to_numpy_dtype(gdal_type: int) -> np.dtype:
    """Map a GDAL data type constant to a numpy dtype."""
    from osgeo import gdal_array
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


# ---------------------------------------------------------------------------
# merge_images — GDAL-based merge with pixel-aligned numpy backend for GTiff
# ---------------------------------------------------------------------------
def merge_images(
    file_list: List[Union[str, Path]],
    outfile: Union[str, Path],
    driver: str = "GTiff",
    target_aligned_pixels: bool = True,
    out_bounds: Optional[Tuple[float, float, float, float]] = None,
    out_bounds_epsg: Optional[int] = None,
    dest_epsg: Optional[int] = None,
    out_nodata: Optional[float] = None,
    in_nodata: Optional[float] = None,
    resample_alg: str = "lanczos",
    overwrite: bool = False,
    options: Optional[List[str]] = None,
    create_only: bool = False,
    strides: Optional[Dict[str, int]] = None,
    out_dtype: Optional[Any] = None,
    margin_px: int = 0,
) -> Path:
    """Merge multiple rasters into a single output — drop-in replacement for dolphin.

    For GeoTIFF (driver='GTiff'), uses pure-numpy pixel-offset copy (``stitch_arrays``).
    For other drivers (ENVI etc.), falls back to GDAL BuildVRT + Warp.

    Parameters
    ----------
    file_list : list of str/Path  Input file paths.
    outfile : str/Path           Output file path.
    driver : str                 GDAL driver (default "GTiff").
    target_aligned_pixels : bool For GDAL fallback only — align to source grid.
    out_bounds : tuple           (left, bottom, right, top) in EPSG given by out_bounds_epsg.
    out_bounds_epsg : int        EPSG of out_bounds (default 4326).
    dest_epsg : int              Override output EPSG.
    out_nodata : float           Output nodata value.
    in_nodata : float            For GDAL fallback — input nodata override.
    resample_alg : str           For GDAL fallback — resampling algorithm.
    overwrite : bool             Overwrite existing output.
    options : list of str        GDAL creation options.
    create_only : bool           Not used by numpy path.
    strides : dict               Subsampling (numpy path uses source resolution).
    out_dtype : numpy.dtype      Output data type (numpy path uses source dtype).

    Returns
    -------
    Path  Output file path.
    """
    outfile = Path(outfile)

    if outfile.exists() and not overwrite:
        logger.info("Output %s exists, skipping.", outfile)
        return outfile

    outfile.parent.mkdir(parents=True, exist_ok=True)

    # Use numpy-based stitch for GeoTIFF (primary path)
    if driver == "GTiff":
        epsg_utm = 32605
        if dest_epsg:
            epsg_utm = dest_epsg
        elif file_list:
            # Auto-detect EPSG from first input
            ds = gdal.Open(str(file_list[0]))
            if ds:
                srs = osr.SpatialReference()
                srs.ImportFromWkt(ds.GetProjection())
                code = srs.GetAuthorityCode(None)
                if code:
                    epsg_utm = int(code)
                ds = None

        stitched, out_gt, proj = stitch_arrays(
            [Path(f) for f in file_list],
            bbox_wsen=out_bounds,
            epsg_utm=epsg_utm,
            margin_px=margin_px,
        )

        if out_dtype is not None:
            stitched = stitched.astype(out_dtype)

        _write_geotiff(outfile, stitched, out_gt, proj, True)
        return outfile

    # --- GDAL fallback for non-GeoTIFF drivers (ENVI etc.) ---
    import tempfile
    tmp_dir = tempfile.mkdtemp(prefix="stitch_vrt_")
    vrt_path = os.path.join(tmp_dir, "mosaic.vrt")

    try:
        str_files = [str(f) for f in file_list]
        vrt_opts = gdal.BuildVRTOptions()
        if in_nodata is not None:
            vrt_opts = gdal.BuildVRTOptions(srcNodata=in_nodata, VRTNodata=in_nodata)
        gdal.BuildVRT(vrt_path, str_files, options=vrt_opts)

        warp_kwargs = {"format": driver, "resampleAlg": resample_alg}

        if target_aligned_pixels:
            warp_kwargs["targetAlignedPixels"] = True
            vrt_ds = gdal.Open(vrt_path)
            if vrt_ds:
                vrt_gt = vrt_ds.GetGeoTransform()
                warp_kwargs["xRes"] = abs(vrt_gt[1])
                warp_kwargs["yRes"] = abs(vrt_gt[5])
                vrt_ds = None

        if options:
            warp_kwargs["creationOptions"] = options
        elif driver == "GTiff":
            warp_kwargs["creationOptions"] = DEFAULT_TIFF_OPTIONS

        if dest_epsg is not None:
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(int(dest_epsg))
            warp_kwargs["dstSRS"] = dst_srs.ExportToWkt()

        if out_bounds is not None:
            warp_kwargs["outputBounds"] = (
                out_bounds[0], out_bounds[1], out_bounds[2], out_bounds[3]
            )
            if out_bounds_epsg is not None:
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(int(out_bounds_epsg))
                warp_kwargs["outputBoundsSRS"] = srs.ExportToWkt()

        if out_nodata is not None:
            warp_kwargs["dstNodata"] = out_nodata

        if strides and "x" in strides and "y" in strides:
            warp_kwargs["xRes"] = strides["x"]
            warp_kwargs["yRes"] = strides["y"]

        if out_dtype is not None:
            dtype_map = {
                np.float32: gdal.GDT_Float32, np.float64: gdal.GDT_Float64,
                np.complex64: gdal.GDT_CFloat32, np.complex128: gdal.GDT_CFloat64,
                np.int16: gdal.GDT_Int16, np.int32: gdal.GDT_Int32,
                np.uint8: gdal.GDT_Byte, np.uint16: gdal.GDT_UInt16, np.uint32: gdal.GDT_UInt32,
            }
            warp_kwargs["outputType"] = dtype_map.get(out_dtype, gdal.GDT_Float32)

        warp_options = gdal.WarpOptions(**warp_kwargs)
        ds = gdal.Warp(str(outfile), vrt_path, options=warp_options)
        if ds is None:
            raise RuntimeError(f"gdal.Warp failed: {gdal.GetLastErrorMsg()}")
        ds = None
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return outfile
