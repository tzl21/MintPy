#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Common utility functions for SLC to interferogram processing.

This module provides:
- Coordinate system conversions and EPSG handling
- GDAL/rasterio I/O helpers
- XML metadata creation for ISCE2/ISCE3 products
- HDF5 metadata extraction
"""

import re
import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any

import h5py
import numpy as np
from osgeo import gdal, osr

try:
    import rasterio
    from rasterio.warp import transform_bounds
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Progress bars
# ------------------------------------------------------------------------
def tqdm_progress(iterable=None, **kwargs):
    """``tqdm`` wrapper that stays silent in log files.

    tqdm >= 4.46 defaults to ``disable=False``, so progress bars are written
    even when stderr is redirected to a file (each ``\\r`` update becomes a
    separate log line).  This wrapper disables the bar whenever stderr is
    not a terminal, keeping redirected logs clean while interactive runs
    keep their live progress display.
    """
    import sys

    from tqdm import tqdm

    if 'disable' not in kwargs:
        kwargs['disable'] = not sys.stderr.isatty()
    return tqdm(iterable, **kwargs)


# ------------------------------------------------------------------------
# Coordinate System Utilities
# ------------------------------------------------------------------------
def get_input_epsg(file_list: List[Union[str, Path]]) -> Optional[int]:
    """
    Determine the EPSG code from a list of GDAL-readable files.

    Parameters
    ----------
    file_list : list of str or Path
        Input raster files.

    Returns
    -------
    Optional[int]
        EPSG code if consistent across all files, otherwise None.
    """
    if not file_list:
        return None

    from collections import Counter
    counts = Counter()
    for file_path in file_list:
        ds = gdal.Open(str(file_path))
        if ds is None:
            continue
        proj = ds.GetProjection()
        if proj:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(proj)
            epsg = srs.GetAuthorityCode(None)
            if epsg:
                counts[int(epsg)] += 1
        ds = None

    if len(counts) == 1:
        return next(iter(counts))
    elif len(counts) > 1:
        logger.warning(f"Multiple EPSG codes found: {dict(counts)}, using most common")
        return counts.most_common(1)[0][0]
    return None


def convert_bounds_to_target_epsg(
    bounds: Tuple[float, float, float, float],
    src_epsg: int,
    dest_epsg: int
) -> Tuple[float, float, float, float]:
    """
    Convert bounding box from source EPSG to target EPSG.

    Parameters
    ----------
    bounds : tuple (left, bottom, right, top)
        Bounds in source CRS.
    src_epsg : int
        Source EPSG code.
    dest_epsg : int
        Target EPSG code.

    Returns
    -------
    tuple
        Bounds in target CRS.
    """
    if src_epsg == dest_epsg:
        return bounds

    if not RASTERIO_AVAILABLE:
        raise ImportError("rasterio is required for bounds conversion")

    src_crs = rasterio.crs.CRS.from_epsg(src_epsg)
    dst_crs = rasterio.crs.CRS.from_epsg(dest_epsg)
    converted = transform_bounds(src_crs, dst_crs, *bounds)
    logger.debug(f"Bounds converted EPSG:{src_epsg} -> EPSG:{dest_epsg}: {bounds} -> {converted}")
    return converted


# ------------------------------------------------------------------------
# GDAL / Raster I/O Helpers
# ------------------------------------------------------------------------
def load_gdal(
    filename: Union[str, Path],
    band: Optional[int] = None,
    subsample_factor: Union[int, Tuple[int, int]] = 1,
    overview: Optional[int] = None,
    rows: Optional[slice] = None,
    cols: Optional[slice] = None,
    masked: bool = False
) -> np.ndarray:
    """
    Load a GDAL file into a NumPy array.

    Parameters
    ----------
    filename : str or Path
        Path to the raster file.
    band : int, optional
        Band index (1-based). If None, load all bands.
    subsample_factor : int or tuple (y, x)
        Subsampling factor (default 1).
    overview : int, optional
        If provided, read from overview level.
    rows, cols : slice, optional
        Subset rows/columns to read.
    masked : bool
        If True, return a masked array using the nodata value.

    Returns
    -------
    np.ndarray or np.ma.MaskedArray
        Array of shape (bands, height, width) or (height, width) if band specified.
    """
    ds = gdal.Open(str(filename))
    if ds is None:
        raise IOError(f"Cannot open {filename}")

    nrows, ncols = ds.RasterYSize, ds.RasterXSize

    # Handle overview
    if overview is not None:
        bnd = ds.GetRasterBand(band or 1)
        ovr_count = bnd.GetOverviewCount()
        if ovr_count > 0:
            idx = ovr_count + overview if overview < 0 else overview
            out = bnd.GetOverview(idx).ReadAsArray()
            ds = None
            return out
        logger.warning(f"Requested overview {overview} not found in {filename}")

    # Parse row/col slices
    rows = slice(0, nrows) if rows in (None, slice(None)) else rows
    cols = slice(0, ncols) if cols in (None, slice(None)) else cols
    xoff, yoff = int(cols.start), int(rows.start)
    xsize = min(cols.stop, ncols) - cols.start
    ysize = min(rows.stop, nrows) - rows.start

    # Subsampling factor
    if isinstance(subsample_factor, int):
        subsample_factor = (subsample_factor, subsample_factor)
    out_rows = ysize // subsample_factor[0]
    out_cols = xsize // subsample_factor[1]

    dt = gdal_to_numpy_type(ds.GetRasterBand(1).DataType)
    resamp = gdal.GRA_NearestNeighbour

    if band is None:
        count = ds.RasterCount
        out = np.empty((count, out_rows, out_cols), dtype=dt)
        ds.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)
        if count == 1:
            out = out[0]
    else:
        out = np.empty((out_rows, out_cols), dtype=dt)
        bnd = ds.GetRasterBand(band)
        bnd.ReadAsArray(xoff, yoff, xsize, ysize, buf_obj=out, resample_alg=resamp)

    ds = None

    if masked:
        nd = get_raster_nodata(filename)
        if nd is not None and np.isnan(nd):
            return np.ma.masked_invalid(out)
        else:
            return np.ma.masked_equal(out, nd)
    return out


def gdal_to_numpy_type(gdal_type: Union[str, int]) -> np.dtype:
    """Convert GDAL data type to NumPy dtype."""
    from osgeo import gdal_array
    if isinstance(gdal_type, str):
        gdal_type = gdal.GetDataTypeByName(gdal_type)
    return np.dtype(gdal_array.GDALTypeCodeToNumericTypeCode(gdal_type))


def get_raster_nodata(filename: Union[str, Path], band: int = 1) -> Optional[float]:
    """Get nodata value from a raster band."""
    ds = gdal.Open(str(filename))
    if ds is None:
        return None
    nd = ds.GetRasterBand(band).GetNoDataValue()
    ds = None
    return nd


def get_raster_bounds(filename: Union[str, Path]) -> Tuple[float, float, float, float]:
    """Return (left, bottom, right, top) bounds of a raster."""
    ds = gdal.Open(str(filename))
    if ds is None:
        raise IOError(f"Cannot open {filename}")
    gt = ds.GetGeoTransform()
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    left, top = gt[0], gt[3]
    right = left + xsize * gt[1] + ysize * gt[2]
    bottom = top + xsize * gt[4] + ysize * gt[5]
    ds = None
    return (left, bottom, right, top)


def get_raster_crs(filename: Union[str, Path]) -> Optional[str]:
    """Get WKT representation of the raster's CRS."""
    ds = gdal.Open(str(filename))
    if ds is None:
        return None
    wkt = ds.GetProjection()
    ds = None
    return wkt if wkt else None


def get_raster_geotransform(filename: Union[str, Path]) -> Tuple[float, ...]:
    """Return the 6-element geotransform tuple."""
    ds = gdal.Open(str(filename))
    gt = ds.GetGeoTransform()
    ds = None
    return gt


# ------------------------------------------------------------------------
# XML Metadata Handling (ISCE2 / ENVI format)
# ------------------------------------------------------------------------
def create_xml_file(
    input_xml_path: Union[str, Path],
    output_xml_path: Union[str, Path],
    crop_window: Optional[Tuple[int, int, int, int]],
    output_file_path: Union[str, Path],
    dtype: Optional[str] = None
) -> bool:
    """
    Create an updated XML metadata file for a cropped SLC/geometry file.

    Parameters
    ----------
    input_xml_path : str or Path
        Path to original XML file.
    output_xml_path : str or Path
        Path to write updated XML.
    crop_window : tuple (min_row, max_row, min_col, max_col) or None
        Cropping indices in radar coordinates.
    output_file_path : str or Path
        Path to the new data file.
    dtype : str, optional
        Override data type (e.g., 'cfloat').

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    input_xml_path = Path(input_xml_path)
    output_xml_path = Path(output_xml_path)

    if not input_xml_path.exists():
        logger.warning(f"XML file not found: {input_xml_path}")
        return False

    try:
        tree = ET.parse(input_xml_path)
        root = tree.getroot()

        # Update file name
        for tag in [".//property[@name='file_name']/value", ".//property[@name='FILE_NAME']/value"]:
            elem = root.find(tag)
            if elem is not None:
                elem.text = str(output_file_path)

        if dtype is not None:
            for tag in [".//property[@name='data_type']/value", ".//property[@name='DATA_TYPE']/value"]:
                elem = root.find(tag)
                if elem is not None:
                    elem.text = dtype

        if crop_window is not None:
            min_row, max_row, min_col, max_col = crop_window
            width = max_col - min_col + 1
            length = max_row - min_row + 1

            # Update width and length
            for tag, val in [("width", width), ("length", length)]:
                elem = root.find(f".//property[@name='{tag}']/value")
                if elem is not None:
                    elem.text = str(val)

            # Update xmax
            elem = root.find(".//property[@name='xmax']/value")
            if elem is not None:
                elem.text = str(float(width))

            # Update coordinate1 (range)
            coord1 = root.find(".//component[@name='coordinate1']")
            if coord1 is not None:
                size_elem = coord1.find(".//property[@name='size']/value")
                if size_elem is not None:
                    size_elem.text = str(width)
                start_elem = coord1.find(".//property[@name='startingvalue']/value")
                if start_elem is not None:
                    start_elem.text = str(float(min_col))
                end_elem = coord1.find(".//property[@name='endingvalue']/value")
                if end_elem is not None:
                    end_elem.text = str(float(max_col))

            # Update coordinate2 (azimuth)
            coord2 = root.find(".//component[@name='coordinate2']")
            if coord2 is not None:
                size_elem = coord2.find(".//property[@name='size']/value")
                if size_elem is not None:
                    size_elem.text = str(length)
                start_elem = coord2.find(".//property[@name='startingvalue']/value")
                if start_elem is not None:
                    start_elem.text = str(float(min_row))
                end_elem = coord2.find(".//property[@name='endingvalue']/value")
                if end_elem is not None:
                    end_elem.text = str(float(max_row))

        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
        return True

    except Exception as e:
        logger.error(f"Error creating XML file {output_xml_path}: {e}")
        return False


def create_xml_ifg(
    slc_xml_path: Union[str, Path],
    output_xml_path: Union[str, Path],
    ifg_file_path: Union[str, Path]
) -> bool:
    """
    Create XML metadata for an interferogram based on an SLC template.

    Parameters
    ----------
    slc_xml_path : str or Path
        Path to reference SLC XML file.
    output_xml_path : str or Path
        Output path for interferogram XML.
    ifg_file_path : str or Path
        Path to the interferogram data file.

    Returns
    -------
    bool
        True if successful, False otherwise.
    """
    slc_xml_path = Path(slc_xml_path)
    if not slc_xml_path.exists():
        logger.warning(f"SLC XML template not found: {slc_xml_path}")
        return False

    try:
        tree = ET.parse(slc_xml_path)
        root = tree.getroot()

        # Update file name
        elem = root.find(".//property[@name='file_name']/value")
        if elem is not None:
            elem.text = str(ifg_file_path)

        # Set interferogram-specific attributes
        for tag, val in [("data_type", "cfloat"), ("image_type", "cpx"), ("family", "intimage")]:
            elem = root.find(f".//property[@name='{tag}']/value")
            if elem is not None:
                elem.text = val

        tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
        return True

    except Exception as e:
        logger.error(f"Error creating interferogram XML {output_xml_path}: {e}")
        return False


def create_xml_for_binary(
    binary_path: Union[str, Path],
    family: Optional[str] = None,
    description: Optional[str] = None,
    template_xml_path: Optional[Union[str, Path]] = None,
    alooks: Optional[int] = None,
    rlooks: Optional[int] = None,
) -> bool:
    """
    Create a proper ISCE2 <imageFile> XML companion for any binary data file.

    Opens the binary via GDAL to determine actual dimensions, band count, data type,
    and interleave scheme, then writes an ISCE2-compatible XML file that MintPy can
    parse via read_isce_xml().

    Parameters
    ----------
    binary_path : str or Path
        Path to the binary data file (e.g., *.int, *.phsig.coh, *.unw, *.rdr, *.slc).
    family : str, optional
        ISCE2 family tag. Auto-detected from extension if not provided:
        '.int' → 'intimage', '.slc' → 'slcimage', other → 'image'.
    description : str, optional
        Human-readable description of the data.
    template_xml_path : str or Path, optional
        Path to an existing SLC/rdr XML to copy coordinate1/coordinate2 from,
        when dimensions match. If not provided, standard pixel-based coordinates
        are used (delta=1.0, startingvalue=0.0).
    alooks : int, optional
        Number of azimuth looks for multilooked data.
    rlooks : int, optional
        Number of range looks for multilooked data.

    Returns
    -------
    bool
        True if XML was written successfully, False otherwise.
    """
    binary_path = Path(binary_path)
    xml_path = Path(str(binary_path) + '.xml')

    if not binary_path.exists():
        logger.warning(f"Binary file not found, cannot create XML: {binary_path}")
        return False

    try:
        ds = gdal.Open(str(binary_path), gdal.GA_ReadOnly)
        if ds is None:
            logger.warning(f"GDAL cannot open {binary_path}, skipping XML creation")
            return False

        width = ds.RasterXSize
        length = ds.RasterYSize
        band_count = ds.RasterCount
        gdal_dtype = ds.GetRasterBand(1).DataType
        ds = None
    except Exception as e:
        logger.warning(f"GDAL read failed for {binary_path}: {e}")
        return False

    ext = binary_path.suffix.lower()

    if family is None:
        _family_map = {'.int': 'intimage', '.slc': 'slcimage'}
        family = _family_map.get(ext, 'image')

    isce_dtype = _gdal_dtype_to_isce2(gdal_dtype)
    if isce_dtype is None:
        logger.warning(f"Unsupported GDAL data type {gdal_dtype} for {binary_path}")
        return False

    scheme = 'BSQ'

    root = ET.Element('imageFile')

    _add_property(root, 'ISCE_VERSION', 'Release: 2.6.3 (20230418)')
    _add_property(root, 'access_mode', 'read')
    _add_property(root, 'byte_order', 'l')
    _add_property(root, 'data_type', isce_dtype)
    _add_property(root, 'file_name', str(binary_path))
    _add_property(root, 'family', family)
    _add_property(root, 'length', str(length))
    _add_property(root, 'name', 'image_name')
    _add_property(root, 'number_bands', str(band_count))
    _add_property(root, 'scheme', scheme)
    _add_property(root, 'width', str(width))
    _add_property(root, 'xmin', '0.0')
    _add_property(root, 'xmax', str(float(width)))

    if description:
        _add_property(root, 'description', description)

    if family == 'intimage':
        _add_property(root, 'image_type', 'cpx')

    # Multilook factors (for MintPy isce_utils.extract_multilook_number)
    if alooks is not None:
        _add_property(root, 'numberAzimuthLooks', str(alooks))
    if rlooks is not None:
        _add_property(root, 'numberRangeLooks', str(rlooks))

    _add_coordinate(root, 'coordinate1', width)
    _add_coordinate(root, 'coordinate2', length)

    if template_xml_path:
        template_path = Path(template_xml_path)
        if template_path.exists():
            try:
                tmpl_tree = ET.parse(template_path)
                tmpl_root = tmpl_tree.getroot()
                tmpl_width_elem = tmpl_root.find(".//property[@name='width']/value")
                tmpl_length_elem = tmpl_root.find(".//property[@name='length']/value")
                tmpl_w = int(tmpl_width_elem.text) if tmpl_width_elem is not None else None
                tmpl_l = int(tmpl_length_elem.text) if tmpl_length_elem is not None else None
                if tmpl_w == width and tmpl_l == length:
                    _copy_coordinate(root, tmpl_root, 'coordinate1')
                    _copy_coordinate(root, tmpl_root, 'coordinate2')
            except Exception:
                pass

    tree = ET.ElementTree(root)
    ET.indent(tree, space='  ')
    tree.write(xml_path, encoding='utf-8', xml_declaration=True)
    logger.debug(f"Created ISCE2 XML: {xml_path}")
    return True


def _gdal_dtype_to_isce2(gdal_dtype: int) -> Optional[str]:
    mapping = {
        getattr(gdal, 'GDT_CFloat32', 10): 'CFLOAT',
        getattr(gdal, 'GDT_CFloat64', 11): 'CFLOAT',
        getattr(gdal, 'GDT_Float32', 6): 'FLOAT',
        getattr(gdal, 'GDT_Float64', 7): 'DOUBLE',
        getattr(gdal, 'GDT_Byte', 1): 'BYTE',
        getattr(gdal, 'GDT_Int16', 3): 'SHORT',
        getattr(gdal, 'GDT_Int32', 5): 'INT',
        getattr(gdal, 'GDT_UInt16', 2): 'SHORT',
        getattr(gdal, 'GDT_UInt32', 4): 'INT',
    }
    return mapping.get(gdal_dtype)


def _add_property(root: ET.Element, name: str, value: str) -> None:
    prop = ET.SubElement(root, 'property', name=name)
    val = ET.SubElement(prop, 'value')
    val.text = value


def _add_coordinate(root: ET.Element, coord_name: str, size: int) -> None:
    comp = ET.SubElement(root, 'component', name=coord_name)
    _add_property(comp, 'factorymodule', 'isceobj.Image')
    _add_property(comp, 'factoryname', 'createCoordinate')
    _add_property(comp, 'family', 'imagecoordinate')
    _add_property(comp, 'name', 'imagecoordinate_name')
    _add_property(comp, 'delta', '1.0')
    _add_property(comp, 'size', str(size))
    _add_property(comp, 'startingvalue', '0.0')
    _add_property(comp, 'endingvalue', str(float(size - 1)))


def _copy_coordinate(root: ET.Element, src_root: ET.Element, coord_name: str) -> None:
    src_comp = src_root.find(f"./component[@name='{coord_name}']")
    if src_comp is None:
        return
    dst_comp = root.find(f"./component[@name='{coord_name}']")
    if dst_comp is not None:
        root.remove(dst_comp)
    root.append(src_comp)


# ------------------------------------------------------------------------
# HDF5 Metadata Extraction (ISCE3 / OPERA CSLC)
# ------------------------------------------------------------------------
def read_burst_metadata_h5(
    h5_file: Union[str, Path],
    group_path: str = "/metadata/processing_information/input_burst_metadata/"
) -> Dict[str, Any]:
    """
    Read burst metadata from an OPERA CSLC HDF5 file.

    Parameters
    ----------
    h5_file : str or Path
        Path to HDF5 file.
    group_path : str
        Internal group containing burst attributes.

    Returns
    -------
    dict
        Dictionary of burst metadata values.
    """
    metadata = {}
    h5_file = Path(h5_file)

    try:
        with h5py.File(h5_file, 'r') as f:
            if group_path not in f:
                logger.warning(f"Group {group_path} not found in {h5_file}")
                return metadata

            group = f[group_path]
            for name, item in group.items():
                if not isinstance(item, h5py.Dataset):
                    continue
                value = item[()]
                # Handle byte strings
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                elif isinstance(value, np.ndarray):
                    if value.dtype.kind in ('S', 'O'):
                        value = value.astype(str).tolist()
                    else:
                        value = value.tolist()
                elif hasattr(value, 'item'):
                    value = value.item()
                metadata[name] = value

                # Special handling for shape -> length/width
                if name == 'shape' and isinstance(value, list) and len(value) == 2:
                    metadata['length'] = value[0]
                    metadata['width'] = value[1]

        # Compute sensing_mid if possible
        if 'sensing_start' in metadata and 'sensing_stop' in metadata:
            try:
                from datetime import datetime
                start = datetime.strptime(metadata['sensing_start'], '%Y-%m-%d %H:%M:%S.%f')
                stop = datetime.strptime(metadata['sensing_stop'], '%Y-%m-%d %H:%M:%S.%f')
                mid = start + (stop - start) / 2
                metadata['sensing_mid'] = mid.strftime('%Y-%m-%d %H:%M:%S.%f')
            except Exception:
                pass

        # Compute mid_range
        if all(k in metadata for k in ['starting_range', 'width', 'range_pixel_spacing']):
            metadata['mid_range'] = (metadata['starting_range'] +
                                     (metadata['width'] / 2) * metadata['range_pixel_spacing'])

        return metadata

    except Exception as e:
        logger.error(f"Error reading burst metadata from {h5_file}: {e}")
        return {}


def read_geogrid_from_metadata_h5(
    h5_file: Union[str, Path],
    group_path: str = "/data/"
) -> Dict[str, Any]:
    """
    Extract geocoding grid information from an OPERA CSLC HDF5 file.

    Returns keys: x_start, y_start, x_spacing, y_spacing, width, length, epsg_code.
    """
    metadata = {}
    h5_file = Path(h5_file)

    try:
        with h5py.File(h5_file, 'r') as f:
            if group_path not in f:
                return metadata
            group = f[group_path]

            # Spacing
            for key in ['x_spacing', 'y_spacing']:
                if key in group:
                    metadata[key] = float(group[key][()])

            # Coordinates and size
            for coord_key, start_key in [('x_coordinates', 'x_start'), ('y_coordinates', 'y_start')]:
                if coord_key in group:
                    data = group[coord_key]
                    metadata[start_key] = float(data[0])
                    metadata['width' if coord_key == 'x_coordinates' else 'length'] = len(data)

            # EPSG from projection
            if 'projection' in group:
                proj_ds = group['projection']
                if 'epsg_code' in proj_ds.attrs:
                    metadata['epsg_code'] = int(proj_ds.attrs['epsg_code'])
                elif 'spatial_ref' in proj_ds.attrs:
                    ref = proj_ds.attrs['spatial_ref']
                    if isinstance(ref, bytes):
                        ref = ref.decode()
                    match = re.search(r'AUTHORITY\["EPSG","(\d+)"\]', ref)
                    if match:
                        metadata['epsg_code'] = int(match.group(1))

        return metadata

    except Exception as e:
        logger.error(f"Error reading geogrid from {h5_file}: {e}")
        return {}


# ------------------------------------------------------------------------
# Miscellaneous Helpers
# ------------------------------------------------------------------------
def ensure_slices(rows: Union[slice, int, None], cols: Union[slice, int, None]) -> Tuple[slice, slice]:
    """Convert integer or Ellipsis to slice objects."""
    def _parse(idx):
        if isinstance(idx, int):
            return slice(idx, idx + 1)
        elif idx is ... or idx is None:
            return slice(None)
        return idx
    return _parse(rows), _parse(cols)

def is_hdf5_file(file_path: Union[str, Path]) -> bool:
    """Check if a file is in HDF5 format based on its extension.

    Parameters
    ----------
    file_path : str or Path
        Path to the file.

    Returns
    -------
    bool
        True if the file has a common HDF5 or NetCDF extension, False otherwise.
    """
    extensions = ['.h5', '.hdf5', '.hdf', '.he5', '.nc']
    return Path(file_path).suffix.lower() in extensions


def open_gdal(file_path: Union[str, Path], subdataset: Optional[str] = None):
    """Open a raster with GDAL, resolving HDF5 subdatasets.

    For ``.h5`` files the ``NETCDF`` driver prefix is used (same as
    dolphin's VRT sources), so GDAL's netCDF driver reads OPERA-style
    metadata (``x_coordinates``/``y_coordinates``/``projection``) and
    returns real georeferencing instead of the HDF5 driver's default
    ``(0, 1, 0, 0, 0, 1)`` identity transform.

    Parameters
    ----------
    file_path : str or Path
        Path to the raster (GeoTIFF / ENVI / HDF5 ...).
    subdataset : str, optional
        HDF5 subdataset path, e.g. ``/data/VV`` (default: open the root).

    Returns
    -------
    gdal.Dataset or None
    """
    from osgeo import gdal as gdal_mod

    path = str(file_path)
    if subdataset and is_hdf5_file(path):
        path = f'NETCDF:"{path}":"//{str(subdataset).lstrip("/")}"'
    return gdal_mod.Open(path, gdal_mod.GA_ReadOnly)
