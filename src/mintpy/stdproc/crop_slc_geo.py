#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Script for cropping individual GeoTIFF and HDF5 files with parallel processing.
Each file is cropped independently (no cross-file stitching). Per-burst output
directories are created when burst IDs are detected in the input paths.

The buffer parameter should be set large enough so that coherence computation
windows (complex coherence and phase-sigma) fully cover the target bbox after
multilooking.

Outputs always as GeoTIFF (.slc.tif) regardless of input format.
"""

import os
import sys
import argparse
import glob
import concurrent.futures
import logging
# MUST be set before any osgeo import — PROJ database path for conda envs
for _pj in (
    os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'proj'),
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'proj'),
    '/home/tangzhenli/tools/bash/envs/ISCE3/share/proj',
):
    if os.path.isdir(_pj) and 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = _pj
        break


from pathlib import Path  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import numpy as np  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import h5py  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import hashlib  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import tempfile  # noqa: E402  (imports follow the mandatory PROJ_LIB block)

from osgeo import gdal, osr  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import math  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import re  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
import traceback  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
from typing import Dict, List, Optional, Any  # noqa: E402  (imports follow the mandatory PROJ_LIB block)

try:
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    print("ERROR: rasterio is not installed. Install with: pip install rasterio")

from .utils.slc2ifg_utils import (  # noqa: E402  (imports follow the mandatory PROJ_LIB block)
    is_hdf5_file,
    tqdm_progress,
)
from .utils.stitching_utils import merge_images, DEFAULT_TIFF_OPTIONS  # noqa: E402  (imports follow the mandatory PROJ_LIB block)


class SLCProcessingError(Exception):
    """Custom exception for SLC processing errors."""
    pass


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def parse_arguments(args_list=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crop individual GeoTIFF and HDF5 files (no cross-file stitching)"
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        nargs='+',
        required=True,
        help="Directory/pattern containing input GeoTIFF/HDF5 files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where cropped files will be saved"
    )
    parser.add_argument(
        "--wsen",
        type=float,
        nargs=4,
        required=True,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help="Crop bounds in WSEN format (EPSG:4326)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.slc.tif",
        help="File pattern for input files"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Buffer to add around the crop area in degrees. "
             "Set large enough so coherence windows cover bbox after multilooking."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to output filenames"
    )
    parser.add_argument(
        "--file-list",
        type=str,
        help="File containing list of input files to process"
    )
    parser.add_argument(
        "--no-burst-dirs",
        action="store_true",
        default=False,
        help="Do not create per-burst output subdirectories (flat output)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        default=False,
        help="Do not skip existing files (reprocess everything)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    parser.add_argument(
        "--dest-epsg",
        type=int,
        default=None,
        help="Target EPSG code for output"
    )
    parser.add_argument(
        "--subdataset",
        type=str,
        default="/data/VV",
        help="HDF5 subdataset to read"
    )
    parser.add_argument(
        "--fill-nan",
        action="store_true",
        help="Fill NaN values with zeros"
    )
    parser.add_argument(
        "--compress-level",
        type=int,
        default=6,
        choices=range(0, 10),
        help="Compression level for output GeoTIFF"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Chunk size for reading/writing data"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually processing"
    )

    return parser.parse_args(args_list) if args_list else parser.parse_args()


def is_geotiff_file(file_path: str) -> bool:
    """Check if file is GeoTIFF format."""
    extensions = ['.tif', '.tiff', '.geotiff']
    return any(str(file_path).lower().endswith(ext) for ext in extensions)


def read_hdf5_metadata(hdf5_path: str, subdataset: str) -> Dict[str, Any]:
    """Read metadata from HDF5 file."""
    metadata = {}
    try:
        with h5py.File(hdf5_path, 'r') as h5file:
            if '/data/projection' in h5file:
                proj_group = h5file['/data/projection']
                metadata['projection'] = dict(proj_group.attrs)
                if 'epsg_code' in proj_group.attrs:
                    metadata['epsg'] = int(proj_group.attrs['epsg_code'])
                elif 'spatial_ref' in proj_group.attrs:
                    spatial_ref = proj_group.attrs['spatial_ref']
                    match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', spatial_ref)
                    if match:
                        metadata['epsg'] = int(match.group(1))
            for coord in ['x_coordinates', 'y_coordinates']:
                if f'/data/{coord}' in h5file:
                    data = h5file[f'/data/{coord}'][:]
                    metadata[coord] = data
                    if len(data) > 1:
                        metadata[f'{coord}_spacing'] = abs(data[1] - data[0])
            for spacing in ['x_spacing', 'y_spacing']:
                if f'/data/{spacing}' in h5file:
                    metadata[spacing] = h5file[f'/data/{spacing}'][()]
            if subdataset in h5file:
                dataset = h5file[subdataset]
                metadata['shape'] = dataset.shape
                metadata['dtype'] = dataset.dtype
                metadata['dataset_attrs'] = dict(dataset.attrs)
            if '/data' in h5file:
                data_group = h5file['/data']
                if 'GeoTransform' in data_group.attrs:
                    metadata['GeoTransform'] = data_group.attrs['GeoTransform']
                if 'SRS' in data_group.attrs:
                    metadata['SRS'] = data_group.attrs['SRS']
    except Exception as e:
        raise SLCProcessingError(f"Failed to read HDF5 metadata: {str(e)}")
    return metadata


def read_hdf5_data_chunked(hdf5_path: str, subdataset: str, chunk_size: tuple = None) -> np.ndarray:
    """Read HDF5 data efficiently with optional chunking."""
    with h5py.File(hdf5_path, 'r') as h5file:
        dataset = h5file[subdataset]
        shape = dataset.shape

        dtype_size = dataset.dtype.itemsize
        total_memory_bytes = shape[0] * shape[1] * dtype_size

        if total_memory_bytes < 4 * 1024**3:
            return dataset[:]

        return read_hdf5_chunked_optimized(dataset, chunk_size)


def read_hdf5_chunked_optimized(dataset: h5py.Dataset, chunk_size: tuple = None) -> np.ndarray:
    """Optimized chunked reading with better memory access patterns."""
    shape = dataset.shape

    if chunk_size is None and dataset.chunks:
        chunk_size = dataset.chunks

    if chunk_size is None:
        chunk_size = (min(1024, shape[0]), shape[1])

    data = np.empty(shape, dtype=dataset.dtype)

    row_chunk = chunk_size[0]

    for i in range(0, shape[0], row_chunk):
        i_end = min(i + row_chunk, shape[0])
        data[i:i_end, :] = dataset[i:i_end, :]

    return data


# ---------------------------------------------------------------------------
# New helper functions (replacing the old stitching logic)
# ---------------------------------------------------------------------------

def extract_date_from_filename(file_path: str) -> str:
    """Extract a YYYYMMDD date string from the filename.

    Returns the first match found. If no date pattern is found, falls back to
    the file's modification time.
    """
    filename = Path(file_path).name

    match = re.search(r'^(\d{8})', filename)
    if match:
        return match.group(1)

    match = re.search(r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])', filename)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"

    mtime = os.path.getmtime(file_path)
    import time
    return time.strftime("%Y%m%d", time.localtime(mtime))


def extract_burst_id(file_path: str) -> Optional[str]:
    """Extract the burst ID from the file path.

    Looks for a parent directory name matching the Sentinel-1 TOPS burst ID
    pattern ``tXXXX_XXXXXX_iwX``. Returns None if no burst ID is found.
    """
    burst_pattern = re.compile(r'^t\d+_\d+_iw\d+$')

    parent = Path(file_path).parent
    parent_name = parent.name
    if burst_pattern.match(parent_name):
        return parent_name

    grandparent_name = parent.parent.name
    if burst_pattern.match(grandparent_name):
        return grandparent_name

    return None


def prepare_crop_params(
    input_files: List[str],
    crop_bounds: tuple,
    dest_epsg: Optional[int] = None,
) -> dict:
    """Build merge_images parameters for cropping a single file or group of files."""
    params = {
        "driver": "GTiff",
        "target_aligned_pixels": False,
        "out_nodata": 0,
        "in_nodata": None,
        "resample_alg": "lanczos",
        "overwrite": True,
        "options": DEFAULT_TIFF_OPTIONS,
        "create_only": False,
    }

    # NOTE: crop bounds stay in EPSG:4326 — merge_images/stitch_arrays
    # transform them into the input file CRS internally (the ``bbox_wsen``
    # contract).  Pre-converting here double-transformed non-4326 inputs
    # (4326 -> input CRS -> input CRS again), producing
    # "Invalid extent: ulx=inf" when cropping GeoTIFF inputs.
    if crop_bounds:
        params["out_bounds"] = tuple(crop_bounds)
        params["out_bounds_epsg"] = 4326
        # Snap the output to the input pixel grid + keep a 1-px margin on
        # each side, exactly like the HDF5 direct crop path — identical
        # input grids then produce identical crops (h5 vs tif).
        params["margin_px"] = 1

    if dest_epsg:
        params["dest_epsg"] = dest_epsg

    return {k: v for k, v in params.items() if v is not None}


# ---------------------------------------------------------------------------
# HDF5 conversion utilities (kept from original)
# ---------------------------------------------------------------------------

def convert_hdf5_to_temp_geotiff(
    hdf5_path: str,
    subdataset: str,
    temp_dir: str,
    fill_nan: bool = False,
    compress_level: int = 6,
    chunk_size: tuple = (256, 256)
) -> tuple:
    """Convert HDF5 file to temporary GeoTIFF for cropping."""
    logger = logging.getLogger(__name__)

    if not RASTERIO_AVAILABLE:
        raise SLCProcessingError("rasterio not installed")

    try:
        temp_path_obj = Path(temp_dir)
        temp_path_obj.mkdir(parents=True, exist_ok=True)

        metadata = read_hdf5_metadata(hdf5_path, subdataset)

        with h5py.File(hdf5_path, 'r') as h5file:
            if subdataset not in h5file:
                raise SLCProcessingError(f"Subdataset {subdataset} not found")
            data = read_hdf5_data_chunked(hdf5_path, subdataset, chunk_size)

        if fill_nan and np.isnan(data).any():
            if np.iscomplexobj(data):
                data = np.where(np.isnan(data), 0+0j, data)
            else:
                data = np.nan_to_num(data, nan=0)

        file_hash = hashlib.md5(f"{hdf5_path}_{subdataset}".encode()).hexdigest()[:8]
        h5_path_obj = Path(hdf5_path)
        subdataset_slug = subdataset.replace('/', '_').strip('_')
        temp_filename = f"{h5_path_obj.stem}_{subdataset_slug}_{file_hash}.tif"
        temp_path = temp_path_obj / temp_filename

        if 'x_coordinates' in metadata and 'y_coordinates' in metadata:
            x_coords = metadata['x_coordinates']
            y_coords = metadata['y_coordinates']

            if len(y_coords) > 1:
                y_is_descending = y_coords[0] > y_coords[-1]
                x_res = abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0
                y_res = abs(y_coords[1] - y_coords[0])
                if y_is_descending:
                    transform = from_origin(x_coords[0], y_coords[0], x_res, y_res)
                else:
                    transform = from_origin(x_coords[0], y_coords[0], x_res, -y_res)
            else:
                transform = from_origin(x_coords[0], y_coords[0], 1.0, 1.0)
        else:
            transform = from_origin(0, 0, 1, 1)

        crs = None
        if 'epsg' in metadata:
            try:
                crs = CRS.from_epsg(metadata['epsg'])
            except Exception:
                pass

        compression = 'DEFLATE' if compress_level > 0 else None

        with rasterio.open(
            temp_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=crs,
            transform=transform,
            compress=compression,
            tiled=True,
            blockxsize=chunk_size[1],
            blockysize=chunk_size[0],
            nodata=np.nan
        ) as dst:
            dst.write(data, 1)

        return str(temp_path), metadata

    except Exception as e:
        logger.error(f"Error converting HDF5 to GeoTIFF: {e}")
        return None, None


# ---------------------------------------------------------------------------
def _get_geotiff_extent_4326(file_path: str) -> Optional[tuple]:
    """Get the extent of a GeoTIFF file in EPSG:4326 as (west, south, east, north).

    Returns None if the file cannot be opened or its extent cannot be determined.
    """
    ds = gdal.Open(file_path)
    if ds is None:
        return None
    try:
        gt = ds.GetGeoTransform()
        if gt is None:
            return None
        w = gt[0]
        n = gt[3]
        e = w + gt[1] * ds.RasterXSize
        s = n + gt[5] * ds.RasterYSize
        extent = (min(w, e), min(s, n), max(w, e), max(s, n))

        proj = ds.GetProjection()
        if proj:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(proj)
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(4326)
            if not src_srs.IsSame(dst_srs):
                # Force traditional GIS axis order (x=lon, y=lat) on both
                # SRS objects so TransformPoint deterministically returns
                # (lon, lat) — the previous hemisphere heuristic broke for
                # |lon| < 90° scenes (Europe / eastern US / ...).
                src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
                transform = osr.CoordinateTransformation(src_srs, dst_srs)
                try:
                    ll = transform.TransformPoint(extent[0], extent[1])
                    ur = transform.TransformPoint(extent[2], extent[3])
                    extent = (ll[0], ll[1], ur[0], ur[1])
                except Exception:
                    pass
        return extent
    finally:
        ds = None


def _hdf5_window(hdf5_path, crop_bounds_4326, subdataset="/data/VV"):
    """Pixel window of the bbox intersection inside an HDF5 SLC.

    Uses the file's ``x_coordinates`` / ``y_coordinates`` metadata: computes
    the file extent in EPSG:4326, intersects with ``crop_bounds_4326``,
    transforms the intersection to the file CRS, masks the coordinate arrays
    to pixel indices, then applies the same 1 px margin as the write-to-disk
    crop path and clamps to the image.

    Returns None when the bbox does not overlap the file, else a dict::

        {'window': (row_start, row_end, col_start, col_end),
         'x_coords': ..., 'y_coords': ..., 'epsg': ..., 'y_descending': ...}

    (the coordinate slices are re-used by ``_crop_hdf5_direct`` to
    georeference the written subset).  Raises ValueError when the file has no
    x/y_coordinates — the caller decides whether to fall back to a full read.
    """
    meta = read_hdf5_metadata(hdf5_path, subdataset)
    if 'x_coordinates' not in meta or 'y_coordinates' not in meta:
        raise ValueError(f"No x/y coordinates in {hdf5_path}")

    x_coords = meta['x_coordinates']
    y_coords = meta['y_coordinates']
    epsg = meta.get('epsg', 4326)

    # Step 1: file extent in EPSG:4326
    if epsg and epsg != 4326:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(int(epsg))
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(4326)
        t_native_to_4326 = osr.CoordinateTransformation(src_srs, dst_srs)
        f_W, f_E = float(x_coords[0]), float(x_coords[-1])
        f_S, f_N = float(y_coords[-1]), float(y_coords[0])
        file_corners_4326 = [
            t_native_to_4326.TransformPoint(f_W, f_S),
            t_native_to_4326.TransformPoint(f_E, f_S),
            t_native_to_4326.TransformPoint(f_E, f_N),
            t_native_to_4326.TransformPoint(f_W, f_N),
        ]
        lon_vals = [c[1] for c in file_corners_4326]
        lat_vals = [c[0] for c in file_corners_4326]
        f_ext_4326_W = min(lon_vals)
        f_ext_4326_E = max(lon_vals)
        f_ext_4326_S = min(lat_vals)
        f_ext_4326_N = max(lat_vals)
    else:
        f_ext_4326_W = float(x_coords[0])
        f_ext_4326_E = float(x_coords[-1])
        f_ext_4326_S = float(y_coords[-1])
        f_ext_4326_N = float(y_coords[0])

    # Step 2: intersection in EPSG:4326
    inter_W = max(f_ext_4326_W, crop_bounds_4326[0])
    inter_S = max(f_ext_4326_S, crop_bounds_4326[1])
    inter_E = min(f_ext_4326_E, crop_bounds_4326[2])
    inter_N = min(f_ext_4326_N, crop_bounds_4326[3])
    if inter_W >= inter_E or inter_S >= inter_N:
        return None

    # Step 3: intersection -> native CRS
    if epsg and epsg != 4326:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(epsg))
        t_4326_to_native = osr.CoordinateTransformation(src_srs, dst_srs)
        # EPSG:4326 axis order is (lat, lon) — swap inputs
        corners = [
            t_4326_to_native.TransformPoint(inter_S, inter_W),
            t_4326_to_native.TransformPoint(inter_S, inter_E),
            t_4326_to_native.TransformPoint(inter_N, inter_E),
            t_4326_to_native.TransformPoint(inter_N, inter_W),
        ]
        native_W = min(c[0] for c in corners)
        native_E = max(c[0] for c in corners)
        native_S = min(c[1] for c in corners)
        native_N = max(c[1] for c in corners)
    else:
        native_W, native_S, native_E, native_N = inter_W, inter_S, inter_E, inter_N

    # Step 4: mask coordinate arrays -> pixel range
    col_mask = (x_coords >= native_W) & (x_coords <= native_E)
    if not np.any(col_mask):
        return None
    cols = np.where(col_mask)[0]
    col_start, col_end = int(cols[0]), int(cols[-1]) + 1

    row_mask = (y_coords >= native_S) & (y_coords <= native_N)
    if not np.any(row_mask):
        return None
    rows = np.where(row_mask)[0]
    row_start, row_end = int(rows[0]), int(rows[-1]) + 1

    # 1-pixel margin (mirrors the write-to-disk crop) + clamp
    col_start = max(0, col_start - 1)
    col_end = min(len(x_coords), col_end + 1)
    row_start = max(0, row_start - 1)
    row_end = min(len(y_coords), row_end + 1)

    y_descending = len(y_coords) > 1 and y_coords[0] > y_coords[-1]
    return {
        'window': (row_start, row_end, col_start, col_end),
        'x_coords': x_coords,
        'y_coords': y_coords,
        'epsg': epsg,
        'y_descending': y_descending,
    }


def _crop_hdf5_direct(hdf5_path, output_path, crop_bounds_4326, subdataset,
                      fill_nan, compress_level):
    """Crop HDF5 directly: read only the intersection region, write GeoTIFF.

    Reads metadata to find coordinates, computes intersection in native CRS,
    reads only the needed subset from the HDF5 file, and writes a compact
    GeoTIFF covering only the intersection area.

    Returns True on success, False on failure, or None when the file has no
    coordinates (caller falls back to the temp+warp path).
    """
    logger = logging.getLogger(__name__)

    try:
        win_info = _hdf5_window(hdf5_path, crop_bounds_4326, subdataset)
    except ValueError:
        logger.warning(f"No coordinates in {hdf5_path}, falling back to full read")
        return None  # signal: fall back to old path
    if win_info is None:
        logger.info(f"Skipping {Path(hdf5_path).name}: no overlap with bbox")
        return True

    row_start, row_end, col_start, col_end = win_info['window']
    x_coords = win_info['x_coords']
    y_coords = win_info['y_coords']
    epsg = win_info['epsg']
    y_descending = win_info['y_descending']

    # Read subset from HDF5
    try:
        with h5py.File(hdf5_path, 'r') as h5file:
            if subdataset not in h5file:
                logger.error(f"Subdataset {subdataset} not found in {hdf5_path}")
                return False
            dataset = h5file[subdataset]
            data = dataset[row_start:row_end, col_start:col_end]
    except Exception as e:
        logger.error(f"Failed to read HDF5 subset {hdf5_path}: {e}")
        return False

    if fill_nan and np.isnan(data).any():
        if np.iscomplexobj(data):
            data = np.where(np.isnan(data), 0 + 0j, data)
        else:
            data = np.nan_to_num(data, nan=0)

    # Compute geotransform for the subset
    sub_x = x_coords[col_start:col_end]
    sub_y = y_coords[row_start:row_end]
    x_res = abs(sub_x[1] - sub_x[0]) if len(sub_x) > 1 else             (abs(x_coords[1] - x_coords[0]) if len(x_coords) > 1 else 1.0)
    y_res = abs(sub_y[1] - sub_y[0]) if len(sub_y) > 1 else             (abs(y_coords[1] - y_coords[0]) if len(y_coords) > 1 else 1.0)

    if y_descending:
        transform = from_origin(float(sub_x[0]), float(sub_y[0]), x_res, y_res)
    else:
        transform = from_origin(float(sub_x[0]), float(sub_y[0]), x_res, -y_res)

    crs = None
    if epsg:
        try:
            crs = CRS.from_epsg(int(epsg))
        except Exception:
            pass

    # Write directly as GeoTIFF
    compression = 'DEFLATE' if compress_level > 0 else None
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        compress=compression,
        tiled=True,
        # TIFF block dimensions must be multiples of 16 (rasterio raises
        # RasterBlockError otherwise).  A small crop subset can be < 256 px,
        # where min(256, n) would yield e.g. 220 -> invalid; round down to
        # the nearest multiple of 16 instead.
        blockxsize=min(data.shape[1], max(16, (data.shape[1] // 16) * 16)),
        blockysize=min(data.shape[0], max(16, (data.shape[0] // 16) * 16)),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)

    logger.debug(f"HDF5 subset crop: {data.shape} -> {output_path}")
    return True


def parse_wsen(raw):
    """Parse a WSEN bbox ('W S E N' or 'W,S,E,N') into a tuple of 4 floats."""
    tokens = re.split(r'[\s,]+', str(raw).strip())
    if len(tokens) != 4:
        raise ValueError(f"expected 4 numbers (W S E N), got {raw!r}")
    try:
        return tuple(float(t) for t in tokens)
    except ValueError:
        raise ValueError(f"expected 4 numbers (W S E N), got {raw!r}")


def bbox_to_window(slc_path, wsen, subdataset="/data/VV", buffer=0.0):
    """Map a WSEN bbox (EPSG:4326, degrees) to a pixel window in an SLC file.

    Mirrors the write-to-disk crop (``crop_single_input``): ``wsen`` is
    expanded by ``buffer`` (degrees), intersected with the SLC extent,
    transformed to the SLC CRS and snapped to its pixel grid with a 1 px
    margin, then clamped to the image.  This powers the engine/basic
    "read-time crop" (``slc2ifg.bbox`` without the ``crop_slc`` stage):
    ``generate_ifgram`` reads and materialises only this window, so no
    intermediate cropped SLC files are stored.

    Parameters
    ----------
    slc_path  : str, path to a geocoded SLC (GeoTIFF or HDF5 with
                x/y_coordinates).
    wsen      : tuple of 4 floats (west, south, east, north) in EPSG:4326.
    subdataset: str, HDF5 subdataset path (used for HDF5 SLCs only).
    buffer    : float, extra margin in degrees added around ``wsen``.

    Returns
    -------
    (x0, y0, w, h) in the SLC pixel grid, or None when the bbox does not
    intersect the SLC.  Raises ValueError for non-geocoded (isce2 radar)
    inputs — read-time cropping requires georeferenced SLCs.
    """
    w, s, e, n = (float(v) for v in wsen)
    crop_bounds = (w - buffer, s - buffer, e + buffer, n + buffer)

    if is_hdf5_file(slc_path):
        try:
            win_info = _hdf5_window(slc_path, crop_bounds, subdataset)
        except ValueError as ex:
            raise ValueError(
                f"bbox read-time crop unsupported for {slc_path}: {ex}") from ex
        if win_info is None:
            return None
        r0, r1, c0, c1 = win_info['window']
        return (c0, r0, c1 - c0, r1 - r0)

    if not is_geotiff_file(slc_path):
        raise ValueError(
            f"bbox read-time crop requires geocoded SLCs (isce3 GeoTIFF/HDF5), "
            f"got {slc_path} — enable the crop_slc stage for isce2 radar")

    # GeoTIFF: inverse geotransform of the bbox corners
    ds = gdal.Open(str(slc_path), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open SLC: {slc_path}")
    try:
        gt = ds.GetGeoTransform()
        cols, rows = ds.RasterXSize, ds.RasterYSize
        proj = ds.GetProjection()
        if not proj:
            raise ValueError(f"No projection in {slc_path}; "
                             "cannot map bbox to pixels")
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(proj)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        t = osr.CoordinateTransformation(src_srs, dst_srs)
        # (x, y) = (lon, lat) with the traditional GIS axis order forced on
        # both SRS, so the result is deterministic across GDAL/PROJ versions
        pts = [t.TransformPoint(lon, lat)
               for lon, lat in ((crop_bounds[0], crop_bounds[3]),
                                (crop_bounds[2], crop_bounds[3]),
                                (crop_bounds[2], crop_bounds[1]),
                                (crop_bounds[0], crop_bounds[1]))]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        px_x0 = int(math.floor((min(xs) - gt[0]) / gt[1]))
        px_x1 = int(math.ceil((max(xs) - gt[0]) / gt[1]))
        if gt[5] < 0:      # north-up (standard): row grows southwards
            px_y0 = int(math.floor((gt[3] - max(ys)) / (-gt[5])))
            px_y1 = int(math.ceil((gt[3] - min(ys)) / (-gt[5])))
        else:              # south-up (unusual): row grows northwards
            px_y0 = int(math.floor((min(ys) - gt[3]) / gt[5]))
            px_y1 = int(math.ceil((max(ys) - gt[3]) / gt[5]))
        # 1 px margin (mirrors the crop path) + clamp to the image
        x0 = max(0, px_x0 - 1)
        x1 = min(cols, px_x1 + 1)
        y0 = max(0, px_y0 - 1)
        y1 = min(rows, px_y1 + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        return (x0, y0, x1 - x0, y1 - y0)
    finally:
        ds = None


# Core: crop a single file (or set of files) to bbox+buffer
# ---------------------------------------------------------------------------

def crop_single_input(
    input_files: List[str],
    output_path: str,
    crop_bounds: tuple,
    dest_epsg: Optional[int] = None,
    subdataset: str = "/data/VV",
    fill_nan: bool = False,
    compress_level: int = 6,
    verbose: bool = False,
) -> bool:
    """Crop one or more input files to the given bounds+buffer.

    If the input is an HDF5 file, it is first converted to a temporary GeoTIFF.
    The output is always a GeoTIFF covering the crop_bounds region.

    Parameters
    ----------
    input_files : list of str
        Input file paths (typically a single file per burst/date).
    output_path : str
        Where to write the cropped GeoTIFF.
    crop_bounds : tuple
        (west, south, east, north) in EPSG:4326.
    """
    logger = logging.getLogger(__name__)

    if not input_files:
        logger.error("No input files provided")
        return False

    missing = [f for f in input_files if not os.path.exists(f)]
    if missing:
        logger.error(f"Input files not found: {missing}")
        return False

    hdf5_files = [f for f in input_files if is_hdf5_file(f)]
    geotiff_files = [f for f in input_files if is_geotiff_file(f)]

    if hdf5_files and geotiff_files:
        logger.error("Mixed HDF5 and GeoTIFF files in the same group")
        return False

    try:
        if hdf5_files:
            for hdf5_path in hdf5_files:
                result = _crop_hdf5_direct(
                    hdf5_path, output_path, crop_bounds, subdataset,
                    fill_nan, compress_level
                )
                if result is None:
                    # Fall back to old temp+warp path
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_path, _ = convert_hdf5_to_temp_geotiff(
                            hdf5_path, subdataset, temp_dir,
                            fill_nan, compress_level
                        )
                        if not temp_path:
                            return False
                        file_extent = _get_geotiff_extent_4326(temp_path)
                        if file_extent is not None:
                            actual_bounds = (
                                max(file_extent[0], crop_bounds[0]),
                                max(file_extent[1], crop_bounds[1]),
                                min(file_extent[2], crop_bounds[2]),
                                min(file_extent[3], crop_bounds[3]),
                            )
                            if actual_bounds[0] >= actual_bounds[2] or actual_bounds[1] >= actual_bounds[3]:
                                return True
                        else:
                            actual_bounds = crop_bounds
                        params = prepare_crop_params([temp_path], actual_bounds, dest_epsg)
                        merge_images(file_list=[temp_path], outfile=output_path, **params)
                elif not result:
                    return False
                # result=True means success or skipped
        else:
            working_files = input_files

            file_extent = _get_geotiff_extent_4326(working_files[0])
            if file_extent is not None:
                actual_bounds = (
                    max(file_extent[0], crop_bounds[0]),
                    max(file_extent[1], crop_bounds[1]),
                    min(file_extent[2], crop_bounds[2]),
                    min(file_extent[3], crop_bounds[3]),
                )
                if actual_bounds[0] >= actual_bounds[2] or actual_bounds[1] >= actual_bounds[3]:
                    if verbose:
                        logger.info(f"Skipping {output_path}: file outside bbox+buffer")
                    return True
            else:
                actual_bounds = crop_bounds

            params = prepare_crop_params(working_files, actual_bounds, dest_epsg)
            merge_images(file_list=working_files, outfile=output_path, **params)

        if verbose:
            logger.info(f"Cropped → {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error cropping to {output_path}: {e}")
        return False


# ---------------------------------------------------------------------------
# Input file discovery
# ---------------------------------------------------------------------------

def get_input_files(args) -> List[str]:
    """Retrieve a list of input files to process (unchanged from original)."""
    logger = logging.getLogger(__name__)

    if args.file_list:
        with open(args.file_list, 'r') as f:
            file_list = [line.strip() for line in f if line.strip()]
        file_list = [f for f in file_list if os.path.exists(f) and "static" not in f]
        logger.info(f"Read {len(file_list)} files from file list")
        return file_list
    else:
        file_list = []

        for input_pattern in args.input_dir:
            expanded_paths = glob.glob(input_pattern, recursive=True)

            if not expanded_paths:
                logger.warning(f"No matches found for pattern: {input_pattern}")
                continue

            for path in expanded_paths:
                if os.path.isdir(path):
                    search_pattern = os.path.join(path, args.pattern)
                    dir_files = glob.glob(search_pattern, recursive=True)
                    dir_files = [f for f in dir_files if os.path.isfile(f) and "static" not in f]
                    file_list.extend(dir_files)
                elif os.path.isfile(path):
                    if Path(path).match(args.pattern) and "static" not in path:
                        file_list.append(path)

        seen = set()
        unique_file_list = []
        for f in file_list:
            if f not in seen:
                seen.add(f)
                unique_file_list.append(f)

        unique_file_list.sort()
        logger.info(f"Found {len(unique_file_list)} unique files")
        return unique_file_list


# ---------------------------------------------------------------------------
# Task wrapper for parallel processing
# ---------------------------------------------------------------------------

def _crop_task(task_dict: dict) -> bool:
    """Wrapper for parallel execution of crop_single_input."""
    return crop_single_input(**task_dict)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args=None):
    """Main function — crops each input file independently, no cross-file stitching."""
    if args is None:
        args = parse_arguments()

    logger = setup_logging(args.verbose or args.debug)



    crop_bounds = list(args.wsen)
    if args.buffer > 0:
        crop_bounds[0] -= args.buffer
        crop_bounds[1] -= args.buffer
        crop_bounds[2] += args.buffer
        crop_bounds[3] += args.buffer

    logger.info("=" * 60)
    logger.info("SLC Cropping Script (Geocoded — per-file, no stitching)")
    logger.info("=" * 60)
    logger.info(f"Crop bounds: West={crop_bounds[0]:.6f}, South={crop_bounds[1]:.6f}")
    logger.info(f"            East={crop_bounds[2]:.6f}, North={crop_bounds[3]:.6f}")
    logger.info(f"Buffer: {args.buffer} degrees")
    logger.info(f"Subdataset: {args.subdataset}")
    logger.info(f"Destination EPSG: {args.dest_epsg}")
    logger.info("=" * 60)

    if not RASTERIO_AVAILABLE:
        logger.error("rasterio is required but not installed.")
        return 1

    input_files = get_input_files(args)

    if not input_files:
        logger.error("No input files found.")
        return 1

    logger.info(f"Found {len(input_files)} input files")

    tasks = []
    for file_path in input_files:
        date_str = extract_date_from_filename(file_path)
        burst_id = extract_burst_id(file_path)

        if burst_id and not args.no_burst_dirs:
            out_dir = Path(args.output_dir) / burst_id
        else:
            out_dir = Path(args.output_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

        output_filename = f"{args.prefix}{date_str}.slc.tif"
        output_path = str(out_dir / output_filename)

        if not args.no_skip_existing and os.path.exists(output_path):
            logger.info(f"Skipping existing: {output_path}")
            continue

        tasks.append({
            'input_files': [file_path],
            'output_path': output_path,
            'crop_bounds': tuple(crop_bounds),
            'dest_epsg': args.dest_epsg,
            'subdataset': args.subdataset,
            'fill_nan': args.fill_nan,
            'compress_level': args.compress_level,
            'verbose': args.verbose,
        })

    if not tasks:
        logger.info("All output files already exist. Nothing to process.")
        return 0

    if args.dry_run:
        logger.info(f"DRY RUN: Would process {len(tasks)} files")
        for task in tasks:
            logger.info(f"  {task['input_files'][0]} → {task['output_path']}")
        return 0

    logger.info(f"Processing {len(tasks)} files...")
    max_workers = args.max_workers if args.max_workers else min(4, len(tasks))

    success_count = 0
    fail_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_crop_task, t): t for t in tasks}
        for future in tqdm_progress(
            concurrent.futures.as_completed(futures),
            total=len(tasks),
            desc="Cropping files"
        ):
            try:
                ok = future.result(timeout=7200)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except concurrent.futures.TimeoutError:
                logger.error("Task timed out")
                fail_count += 1
            except Exception as exc:
                logger.error(f"Task exception: {exc}")
                fail_count += 1

    logger.info("\n" + "=" * 60)
    logger.info("Processing Summary")
    logger.info("=" * 60)
    logger.info(f"Successful: {success_count} files")
    logger.info(f"Failed: {fail_count} files")

    return 0 if fail_count == 0 else 1


# Keep backward-compatible alias for any external callers
stitch_files = None  # removed — use per-burst crop + stitch.py instead

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.error("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"\nUnexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
