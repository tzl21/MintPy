#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Unified raster I/O for the slc2ifg (stdproc) pipeline.

This module is the single read/write entry point for every step of the SLC ->
interferogram pipeline.  It replaces the several ad-hoc GDAL / rasterio / h5py
code paths that used to live in the individual step modules and in
``utils/slc2ifg_utils.py`` / ``utils/stitching_utils.py``.

Conventions
-----------
* **Georeferencing is explicit.**  ``write_raster(..., geo=None)`` writes the
  geotransform / projection only when the metadata actually carries one;
  ``geo=False`` never writes it (radar-coordinate ISCE2 products), ``geo=True``
  requires it.  Readers use ``GetGeoTransform(can_return_null=True)`` so that a
  non-georeferenced raster is never mistaken for one at the origin with 1-pixel
  spacing.
* **Atomic writes.**  ``write_raster`` writes to ``<path>.tmp`` and renames on
  success, so an interrupted run never leaves a partial product behind.
* **Lazy heavy imports.**  ``osgeo`` / ``h5py`` are imported inside the
  functions on purpose: ``import mintpy.stdproc`` must not require GDAL (see
  ``stdproc/__init__.py``).  The default pixel-read path does not touch the
  georeferencing API at all, so it also works with the lightweight GDAL stubs
  used by the unit tests.
* **Complex data stays complex.**  ``read_raster`` returns complex64/128 for a
  complex band (unlike ``mintpy.utils.readfile.read``, whose default is the
  phase band).

Product metadata (``PROCESSOR``, ``FILE_TYPE``, ``DATE12``, ...) is embedded in
the GeoTIFF itself through :func:`write_raster` and recovered by
``mintpy.utils.readfile.read_attribute`` from the GDAL metadata domain, so a
radar-coordinate product remains self-describing without any sidecar file.
"""

from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)

#: extensions GDAL can read as a raster (keep in sync with readfile.GDAL_FILE_EXTS)
GDAL_FILE_EXTS: Tuple[str, ...] = ('.tif', '.tiff', '.grd', '.cos', '.vrt')
#: extensions treated as HDF5 / NetCDF containers
HDF5_EXTS: Tuple[str, ...] = ('.h5', '.hdf5', '.hdf', '.he5', '.nc')

#: metadata keys embedded into the GeoTIFF by write_raster()
EMBED_KEYS: Tuple[str, ...] = (
    'FILE_TYPE',
    'DATE12',
    'P_BASELINE_TOP_HDR',
    'P_BASELINE_BOTTOM_HDR',
    'SLC2IFG_PROCESSOR',
)

#: GTiff creation options used by default
DEFAULT_TIFF_OPTIONS = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']


# ---------------------------------------------------------------------------
# small path helpers
# ---------------------------------------------------------------------------
def is_hdf5_file(file_path: Union[str, Path]) -> bool:
    """Return True when the path has an HDF5 / NetCDF extension."""
    return Path(file_path).suffix.lower() in HDF5_EXTS


def is_gdal_file(file_path: Union[str, Path]) -> bool:
    """Return True when the path has a GDAL-readable raster extension."""
    return Path(file_path).suffix.lower() in GDAL_FILE_EXTS


def parse_wsen(raw) -> Tuple[float, float, float, float]:
    """Parse a WSEN bbox ('W S E N' or 'W,S,E,N') into a tuple of 4 floats."""
    tokens = re.split(r'[\s,]+', str(raw).strip())
    if len(tokens) != 4:
        raise ValueError(f'expected 4 numbers (W S E N), got {raw!r}')
    try:
        return tuple(float(t) for t in tokens)
    except ValueError:
        raise ValueError(f'expected 4 numbers (W S E N), got {raw!r}')


# ---------------------------------------------------------------------------
# GDAL dataset helpers
# ---------------------------------------------------------------------------
#: HDF5 SLC subdataset polarization priority (first match wins)
POLARIZATIONS: Tuple[str, ...] = ('VV', 'VH', 'HH', 'HV')

#: SLC-derived data groups (not polarization layers) under /data
_NON_SLC_DATA_GROUPS = {
    'projection', 'x_coordinates', 'y_coordinates', 'x_spacing', 'y_spacing',
}


def detect_hdf5_subdataset(hdf5_path: Union[str, Path],
                           subdataset: Optional[str] = None
                           ) -> Optional[str]:
    """Resolve the HDF5 subdataset of an SLC, auto-detecting the polarization.

    An explicit ``subdataset`` is returned unchanged.  Otherwise the
    ``/data/<POL>`` groups present in the file are inspected and the first by
    priority (``VV > VH > HH > HV``) is returned.  When the file cannot be
    opened, the polarization token in the filename is used as a fallback.

    Returns ``None`` for a non-HDF5 file or when no polarization is found.
    """
    if subdataset:
        return subdataset
    path = str(hdf5_path)
    if not is_hdf5_file(path):
        return None
    try:
        import h5py
        with h5py.File(path, 'r') as h5file:
            if '/data' in h5file:
                names = set(h5file['/data'].keys())
                for pol in POLARIZATIONS:
                    if pol in names:
                        return f'/data/{pol}'
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug('subdataset detection failed for %s: %s', path, exc)
    match = re.search(r'_(VV|VH|HH|HV)[_.]', Path(path).name)
    if match:
        return f'/data/{match.group(1)}'
    return None


def open_raster(file_path: Union[str, Path], subdataset: Optional[str] = None):
    """Open a raster with GDAL, resolving HDF5 subdatasets.

    For ``.h5`` files the ``NETCDF`` driver prefix is used (same as dolphin's
    VRT sources), so GDAL's netCDF driver reads OPERA-style metadata
    (``x_coordinates``/``y_coordinates``/``projection``) and returns real
    georeferencing instead of the HDF5 driver's default identity transform.
    When ``subdataset`` is not given it is auto-detected
    (:func:`detect_hdf5_subdataset`, preferring VV).

    Parameters
    ----------
    file_path : str or Path
        Path to the raster (GeoTIFF / ENVI / HDF5 / VRT ...).
    subdataset : str, optional
        HDF5 subdataset path, e.g. ``/data/VV``; auto-detected when omitted.

    Returns
    -------
    gdal.Dataset or None
    """
    from osgeo import gdal

    path = str(file_path)
    if is_hdf5_file(path):
        subdataset = detect_hdf5_subdataset(path, subdataset)
    if subdataset and is_hdf5_file(path):
        path = f'NETCDF:"{path}":"//{str(subdataset).lstrip("/")}"'
    return gdal.Open(path, gdal.GA_ReadOnly)


def get_geotransform(ds) -> Optional[Tuple[float, ...]]:
    """Return the geotransform of a GDAL dataset, or None when not georeferenced.

    GDAL returns the default identity transform ``(0, 1, 0, 0, 0, 1)`` for a
    raster without a geotransform; ``can_return_null=True`` returns ``None``
    instead, which is what makes a radar-coordinate product distinguishable
    from a geocoded one.
    """
    try:
        return ds.GetGeoTransform(can_return_null=True)
    except TypeError:
        # older GDAL python bindings without the can_return_null keyword
        gt = ds.GetGeoTransform()
        identity = (0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
        return None if (not ds.GetProjection() and tuple(gt) == identity) else gt


def has_geo(ds_or_path) -> bool:
    """Return True when the given dataset/path carries a geotransform."""
    if isinstance(ds_or_path, (str, Path)):
        ds = open_raster(ds_or_path)
        if ds is None:
            raise IOError(f'cannot open raster: {ds_or_path}')
        return get_geotransform(ds) is not None
    return get_geotransform(ds_or_path) is not None


def epsg_from_projection(proj: str) -> Optional[int]:
    """Return the EPSG code of a WKT projection string, or None."""
    if not proj:
        return None
    from osgeo import osr
    code = osr.SpatialReference(wkt=proj).GetAuthorityCode(None)
    return int(code) if code else None


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def read_metadata(file_path: Union[str, Path], subdataset: Optional[str] = None,
                  geo: bool = True) -> Dict:
    """Read raster metadata into a MintPy-flavoured dictionary.

    Parameters
    ----------
    file_path : str or Path
    subdataset : str, optional
        HDF5 subdataset path (``/data/VV``); ignored for plain rasters.
    geo : bool
        When False, skip GetGeoTransform/GetProjection entirely (the default
        pixel-only read path must work with a minimal GDAL stub).

    Returns
    -------
    dict with WIDTH, LENGTH, BANDS, DATA_TYPE, NO_DATA_VALUE and, when
    available, X_FIRST/Y_FIRST/X_STEP/Y_STEP, PROJ and EPSG.
    """
    from mintpy.utils import readfile

    ds = open_raster(file_path, subdataset=subdataset)
    if ds is None:
        raise IOError(f'cannot open raster: {file_path}')

    meta: Dict = {
        'WIDTH': ds.RasterXSize,
        'LENGTH': ds.RasterYSize,
        'BANDS': ds.RasterCount,
    }

    band = ds.GetRasterBand(1)
    data_type = readfile.DATA_TYPE_GDAL2NUMPY.get(band.DataType)
    if data_type:
        meta['DATA_TYPE'] = data_type.replace('>', '').replace('<', '')

    ndv = band.GetNoDataValue()
    if ndv is not None:
        meta['NO_DATA_VALUE'] = 'nan' if ndv != ndv else float(ndv)

    if geo:
        gt = get_geotransform(ds)
        if gt is not None:
            meta['X_FIRST'] = gt[0]
            meta['Y_FIRST'] = gt[3]
            meta['X_STEP'] = abs(gt[1])
            meta['Y_STEP'] = gt[5]
            proj = ds.GetProjection()
            if proj:
                meta['PROJ'] = proj
                epsg = epsg_from_projection(proj)
                if epsg:
                    meta['EPSG'] = epsg

    return meta


def read_raster(file_path: Union[str, Path], band: Optional[int] = None,
                box: Optional[Tuple[int, int, int, int]] = None,
                subdataset: Optional[str] = None,
                masked: bool = False, geo: bool = False,
                steps: int = 1) -> Tuple:
    """Read a raster into a numpy array.

    Parameters
    ----------
    file_path : str or Path
    band : int, optional
        1-based band index; None reads every band (a single band is squeezed).
    box : tuple of 4 int, optional
        ``(x0, y0, x1, y1)`` pixel window in the input grid.
    subdataset : str, optional
        HDF5 subdataset path (``/data/VV``).
    masked : bool
        Return a masked array using the band no-data value.
    geo : bool
        Also return the geo metadata (default False keeps the read free of any
        georeferencing call).
    steps : int
        Sub-sampling step for both directions (1 = every pixel).

    Returns
    -------
    data : np.ndarray, shape (LENGTH, WIDTH) or (BANDS, ...)
    meta : dict, see read_metadata(); always includes WIDTH/LENGTH/DATA_TYPE of
        the RETURNED array plus the read window.
    """
    import numpy as np

    ds = open_raster(file_path, subdataset=subdataset)
    if ds is None:
        raise IOError(f'cannot open raster: {file_path}')

    full_cols, full_rows = ds.RasterXSize, ds.RasterYSize
    if box is None:
        x0, y0 = 0, 0
        w, h = full_cols, full_rows
    else:
        x0, y0, x1, y1 = (int(v) for v in box)
        w, h = x1 - x0, y1 - y0

    def _read(b):
        arr = b.ReadAsArray(x0, y0, w, h)
        if steps > 1:
            off = steps // 2
            arr = arr[off::steps, off::steps]
        return arr

    band_count = ds.RasterCount
    if band is None:
        data = _read(ds.GetRasterBand(1)) if band_count == 1 else np.stack(
            [_read(ds.GetRasterBand(i)) for i in range(1, band_count + 1)])
        meta_band = 1
    else:
        meta_band = int(band)
        data = _read(ds.GetRasterBand(meta_band))

    meta = read_metadata(file_path, subdataset=subdataset, geo=geo)
    meta['WINDOW'] = (x0, y0, w, h)
    meta['WIDTH'] = data.shape[-1]
    meta['LENGTH'] = data.shape[-2]
    meta['DATA_TYPE'] = str(data.dtype)

    if masked:
        ndv = ds.GetRasterBand(meta_band).GetNoDataValue()
        if ndv is not None:
            data = np.ma.masked_invalid(data) if ndv != ndv else np.ma.masked_equal(data, ndv)
        else:
            data = np.ma.masked_array(data)

    return data, meta


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def build_gdal_metadata(meta: Optional[Dict], processor: Optional[str] = None) -> Dict:
    """Build the GDAL metadata items embedded into a written product.

    ``PROCESSOR`` is stored as a GDAL-readable family: isce2 radar products are
    written as ``gdal`` (they ARE GeoTIFFs and must be read with GDAL) with the
    real provenance kept in ``SLC2IFG_PROCESSOR``.
    """
    items: Dict = {}
    if meta:
        for key in EMBED_KEYS:
            value = meta.get(key)
            if value not in (None, ''):
                items[key] = value

    if processor == 'isce2':
        items['PROCESSOR'] = 'gdal'
        items['SLC2IFG_PROCESSOR'] = 'isce2'
    elif processor:
        items['PROCESSOR'] = processor

    return items


def write_raster(data, out_file: Union[str, Path], meta: Optional[Dict] = None,
                 like: Optional[Union[str, Path]] = None, driver: str = 'GTiff',
                 geo: Optional[bool] = None, compress: Optional[str] = 'LZW',
                 tiled: bool = True, nodata: Optional[float] = None,
                 dtype=None, atomic: bool = True,
                 gdal_metadata: Optional[Dict] = None,
                 processor: Optional[str] = None,
                 overwrite: bool = True):
    """Write a numpy array to a GDAL raster through the single stdproc writer.

    Parameters
    ----------
    data : np.ndarray, 2D
    out_file : str or Path
    meta : dict, optional
        Metadata; geo keys (X/Y_FIRST/STEP + EPSG/UTM_ZONE) drive the
        georeferencing, see ``geo``.
    like : str or Path, optional
        Reference raster whose metadata/georeferencing is copied (only for keys
        missing from ``meta``).
    driver : str
        GDAL driver name (``GTiff`` for both processors in slc2ifg).
    geo : None / True / False
        None  - write geo only when the metadata has a complete geotransform
                (and a CRS; see save_gdal.resolve_geo)
        True  - require geo
        False - never write geo (radar-coordinate products)
    compress, tiled : GTiff creation options.
    nodata : float, optional
    dtype : numpy dtype, optional
        Cast ``data`` before writing.
    atomic : bool
        Write to ``<out_file>.tmp`` then rename (default True).
    gdal_metadata : dict, optional
        Explicit GDAL metadata items; when None they are derived from ``meta``
        and ``processor`` via :func:`build_gdal_metadata`.
    processor : {'isce2', 'isce3'}, optional
        Provenance stamped into the product (see build_gdal_metadata).
    overwrite : bool
        When False and ``out_file`` exists, skip the write.

    Returns
    -------
    out_file : str
    """
    import numpy as np

    out_file = str(out_file)
    if not overwrite and os.path.isfile(out_file):
        logger.info('Skipping existing output: %s', out_file)
        return out_file

    full_meta: Dict = dict(meta or {})
    if like is not None:
        for key, value in read_metadata(like, geo=True).items():
            full_meta.setdefault(key, value)

    if dtype is not None and np.dtype(dtype) != data.dtype:
        data = np.asarray(data, dtype=dtype)

    if gdal_metadata is None:
        gdal_metadata = build_gdal_metadata(full_meta, processor=processor)

    out_dir = os.path.dirname(os.path.abspath(out_file))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    from mintpy.save_gdal import write_gdal
    write_gdal(data, full_meta, out_file=out_file, out_fmt=driver, geo=geo,
               compress=compress, tiled=tiled, nodata=nodata, atomic=atomic,
               gdal_metadata=gdal_metadata)

    return out_file


def write_product(data, out_file: Union[str, Path], processor: str = 'isce3',
                  meta: Optional[Dict] = None, like=None, **kwargs):
    """Write an slc2ifg product: always GeoTIFF, for both processors.

    isce2 products are in radar coordinates, so they are written WITHOUT any
    georeferencing; isce3 products keep the georeferencing carried by ``meta``
    or copied from ``like``.  An explicit ``geo=`` passes through unchanged.
    """
    geo = kwargs.pop('geo', None)
    if geo is None and processor == 'isce2':
        geo = False
    return write_raster(data, out_file, meta=meta, like=like, driver='GTiff',
                        geo=geo, processor=processor, **kwargs)


# ---------------------------------------------------------------------------
# bbox -> pixel window (AOI helpers, shared by crop / generate_ifgram /
# complex_coh / select quick-coherence)
# ---------------------------------------------------------------------------
def read_hdf5_metadata(hdf5_path: Union[str, Path],
                       subdataset: Optional[str] = None) -> Dict:
    """Read the geolocation metadata of an HDF5 (OPERA GSLC style) SLC.

    ``subdataset`` is auto-detected (preferring VV) when not given.
    """
    import h5py

    subdataset = detect_hdf5_subdataset(hdf5_path, subdataset)
    metadata: Dict = {}
    with h5py.File(hdf5_path, 'r') as h5file:
        if '/data/projection' in h5file:
            proj_group = h5file['/data/projection']
            metadata['projection'] = dict(proj_group.attrs)
            if 'epsg_code' in proj_group.attrs:
                metadata['epsg'] = int(proj_group.attrs['epsg_code'])
            elif 'spatial_ref' in proj_group.attrs:
                match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]',
                                  proj_group.attrs['spatial_ref'])
                if match:
                    metadata['epsg'] = int(match.group(1))
        for coord in ('x_coordinates', 'y_coordinates'):
            if f'/data/{coord}' in h5file:
                data = h5file[f'/data/{coord}'][:]
                metadata[coord] = data
                if len(data) > 1:
                    metadata[f'{coord}_spacing'] = abs(data[1] - data[0])
        for spacing in ('x_spacing', 'y_spacing'):
            if f'/data/{spacing}' in h5file:
                metadata[spacing] = h5file[f'/data/{spacing}'][()]
        if subdataset in h5file:
            dataset = h5file[subdataset]
            metadata['shape'] = dataset.shape
            metadata['dtype'] = dataset.dtype
    return metadata


def hdf5_window(hdf5_path: Union[str, Path], crop_bounds_4326,
                subdataset: Optional[str] = None) -> Optional[Dict]:
    """Pixel window of the bbox intersection inside an HDF5 SLC.

    Uses the file's ``x_coordinates``/``y_coordinates`` metadata: computes the
    file extent in EPSG:4326, intersects it with ``crop_bounds_4326``,
    transforms the intersection to the file CRS, masks the coordinate arrays to
    pixel indices, then applies the same 1-px margin as the write-to-disk crop
    and clamps to the image.

    Returns None when the bbox does not overlap the file, else a dict with
    ``window`` ``(row_start, row_end, col_start, col_end)`` and the coordinate
    slices / epsg / y_descending.  Raises ValueError when the file has no
    x/y_coordinates.
    """
    import numpy as np
    from osgeo import osr

    meta = read_hdf5_metadata(hdf5_path, subdataset)
    if 'x_coordinates' not in meta or 'y_coordinates' not in meta:
        raise ValueError(f'No x/y coordinates in {hdf5_path}')

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
        f_w, f_e = float(x_coords[0]), float(x_coords[-1])
        f_s, f_n = float(y_coords[-1]), float(y_coords[0])
        corners = [
            t_native_to_4326.TransformPoint(f_w, f_s),
            t_native_to_4326.TransformPoint(f_e, f_s),
            t_native_to_4326.TransformPoint(f_e, f_n),
            t_native_to_4326.TransformPoint(f_w, f_n),
        ]
        lon_vals = [c[1] for c in corners]
        lat_vals = [c[0] for c in corners]
        ext_w, ext_e = min(lon_vals), max(lon_vals)
        ext_s, ext_n = min(lat_vals), max(lat_vals)
    else:
        ext_w, ext_e = float(x_coords[0]), float(x_coords[-1])
        ext_s, ext_n = float(y_coords[-1]), float(y_coords[0])

    # Step 2: intersection in EPSG:4326
    inter_w = max(ext_w, crop_bounds_4326[0])
    inter_s = max(ext_s, crop_bounds_4326[1])
    inter_e = min(ext_e, crop_bounds_4326[2])
    inter_n = min(ext_n, crop_bounds_4326[3])
    if inter_w >= inter_e or inter_s >= inter_n:
        return None

    # Step 3: intersection -> native CRS
    if epsg and epsg != 4326:
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromEPSG(int(epsg))
        t_4326_to_native = osr.CoordinateTransformation(src_srs, dst_srs)
        # EPSG:4326 axis order is (lat, lon) - swap inputs
        corners = [
            t_4326_to_native.TransformPoint(inter_s, inter_w),
            t_4326_to_native.TransformPoint(inter_s, inter_e),
            t_4326_to_native.TransformPoint(inter_n, inter_e),
            t_4326_to_native.TransformPoint(inter_n, inter_w),
        ]
        native_w = min(c[0] for c in corners)
        native_e = max(c[0] for c in corners)
        native_s = min(c[1] for c in corners)
        native_n = max(c[1] for c in corners)
    else:
        native_w, native_s, native_e, native_n = inter_w, inter_s, inter_e, inter_n

    # Step 4: mask coordinate arrays -> pixel range
    col_mask = (x_coords >= native_w) & (x_coords <= native_e)
    if not np.any(col_mask):
        return None
    cols = np.where(col_mask)[0]
    col_start, col_end = int(cols[0]), int(cols[-1]) + 1

    row_mask = (y_coords >= native_s) & (y_coords <= native_n)
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


def bbox_to_window(slc_path: Union[str, Path], wsen,
                   subdataset: Optional[str] = None, buffer: float = 0.0
                   ) -> Optional[Tuple[int, int, int, int]]:
    """Map a WSEN bbox (EPSG:4326, degrees) to a pixel window in an SLC file.

    Three cases, all mapping the AOI to the SLC's own pixel grid with a 1-px
    margin and clamping to the image:

    * **HDF5** (OPERA-style): via the file's ``x/y_coordinates``;
    * **geocoded raster**: via its geotransform/projection;
    * **isce2 radar coordinates** (no georeferencing): via the standard ISCE2
      ``<merged>/geom_reference`` lookup tables (``lat.rdr.full`` /
      ``lon.rdr.full``), so read-time cropping works for isce2 too.

    Parameters
    ----------
    slc_path : str or Path
        SLC file (GeoTIFF, HDF5, or the radar-coordinate ISCE2 SLC).
    wsen : tuple of 4 floats
        (west, south, east, north) in EPSG:4326.
    subdataset : str, optional
        HDF5 subdataset path (HDF5 SLCs only); auto-detected when omitted.
    buffer : float
        Extra margin in degrees added around ``wsen``.

    Returns
    -------
    (x0, y0, w, h) in the SLC pixel grid, or None when the bbox does not
    intersect the SLC.
    """
    import math

    slc_path = str(slc_path)
    w, s, e, n = (float(v) for v in wsen)
    crop_bounds = (w - buffer, s - buffer, e + buffer, n + buffer)

    if is_hdf5_file(slc_path):
        try:
            win_info = hdf5_window(slc_path, crop_bounds, subdataset)
        except ValueError as ex:
            raise ValueError(
                f'bbox read-time crop unsupported for {slc_path}: {ex}') from ex
        if win_info is None:
            return None
        r0, r1, c0, c1 = win_info['window']
        return (c0, r0, c1 - c0, r1 - r0)

    gt = proj = None
    cols = rows = 0
    if is_gdal_file(slc_path):
        from osgeo import gdal
        ds = gdal.Open(slc_path, gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f'Cannot open SLC: {slc_path}')
        try:
            gt = get_geotransform(ds)
            proj = ds.GetProjection() if gt is not None else ''
            cols, rows = ds.RasterXSize, ds.RasterYSize
        finally:
            ds = None

    if gt is not None:
        from osgeo import osr
        if not proj:
            raise ValueError(f'No projection in {slc_path}; cannot map bbox to pixels')
        src_srs = osr.SpatialReference()
        src_srs.ImportFromEPSG(4326)
        src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        dst_srs = osr.SpatialReference()
        dst_srs.ImportFromWkt(proj)
        dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        t = osr.CoordinateTransformation(src_srs, dst_srs)
        # (x, y) = (lon, lat) with the traditional GIS axis order forced on both
        # SRS, so the result is deterministic across GDAL/PROJ versions
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

    # isce2 radar coordinates (no georeferencing): the SLC shares the radar
    # grid of the standard merged geometry, so the AOI maps to rows/cols via
    # lat.rdr.full / lon.rdr.full under <merged>/geom_reference.
    from .utils.slc_input import standard_geom_dir
    geom = standard_geom_dir(slc_path)
    if geom is None:
        raise ValueError(
            f'cannot map bbox to {slc_path}: not georeferenced and no standard '
            f'ISCE2 <merged>/geom_reference (lat.rdr.full / lon.rdr.full) found')
    from .crop_slc import find_crop_window_from_full_files
    min_row, max_row, min_col, max_col = find_crop_window_from_full_files(
        str(geom / 'lon.rdr.full'), str(geom / 'lat.rdr.full'), crop_bounds)
    return (min_col, min_row, max_col - min_col + 1, max_row - min_row + 1)
