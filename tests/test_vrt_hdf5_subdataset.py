#!/usr/bin/env python3
"""HDF5 source connection strings and the georeferencing of VRT interferograms.

Regression guard: GDAL's ``HDF5`` driver ignores the CF metadata
(``x_coordinates`` / ``y_coordinates`` / ``grid_mapping``) of an OPERA-style
CSLC and reports NO geotransform, while the ``NETCDF`` driver reports the real
one.  Building the VRT sources with the HDF5 driver therefore stripped the
georeferencing from every product of the pipeline.
"""

import numpy as np
import pytest

pytest.importorskip('h5py')
gdal = pytest.importorskip('osgeo.gdal')

from mintpy.stdproc import io as sio  # noqa: E402
from mintpy.stdproc.utils.vrt_utils import (  # noqa: E402
    VRTInterferogram,
    _format_nc_filename,
)

EPSG = 32605
X_FIRST, X_STEP = 258530.0, 5.0
Y_FIRST, Y_STEP = 2149910.0, -10.0
ROWS, COLS = 4, 4


def _write_h5(path):
    """A plain polarization HDF5: no CF coordinates, hence no georeferencing."""
    import h5py
    data = (np.ones((ROWS, COLS)) + 1j).astype(np.complex64)
    with h5py.File(path, 'w') as f:
        f.create_dataset('/data/VV', data=data)
    return path


def _write_cf_h5(path):
    """An OPERA-style CSLC: CF x/y coordinates + a grid_mapping projection."""
    import h5py
    from osgeo import osr

    x_coords = np.array([X_FIRST + X_STEP * i for i in range(COLS)])
    y_coords = np.array([Y_FIRST + Y_STEP * i for i in range(ROWS)])
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(EPSG)

    with h5py.File(path, 'w') as f:
        x = f.create_dataset('/data/x_coordinates', data=x_coords)
        x.attrs['standard_name'] = 'projection_x_coordinate'
        x.attrs['units'] = 'meters'
        y = f.create_dataset('/data/y_coordinates', data=y_coords)
        y.attrs['standard_name'] = 'projection_y_coordinate'
        y.attrs['units'] = 'meters'

        data = (np.ones((ROWS, COLS)) + 2j).astype(np.complex64)
        vv = f.create_dataset('/data/VV', data=data, dtype=np.complex64)
        vv.attrs['grid_mapping'] = 'projection'
        vv.dims[0].attach_scale(y)
        vv.dims[1].attach_scale(x)

        proj = f.create_dataset('/data/projection', data=np.int32(0))
        proj.attrs['grid_mapping_name'] = np.bytes_(
            b'universal_transverse_mercator')
        proj.attrs['epsg_code'] = np.int64(EPSG)
        proj.attrs['spatial_ref'] = np.bytes_(srs.ExportToWkt().encode())
    return path


def _open_quiet(path):
    """gdal.Open without the driver probing errors on stderr."""
    gdal.PushErrorHandler('CPLQuietErrorHandler')
    try:
        return gdal.Open(path)
    finally:
        gdal.PopErrorHandler()


def test_format_and_open_hdf5_subdataset(tmp_path):
    """The subdataset is addressed through the CF-aware NETCDF driver."""
    h5 = _write_h5(tmp_path / 'slc.h5')
    conn = _format_nc_filename(h5, '/data/VV')
    assert conn == f'NETCDF:"{h5}":"//data/VV"'
    ds = _open_quiet(conn)
    assert ds is not None
    assert (ds.RasterXSize, ds.RasterYSize) == (ROWS, COLS)
    ds = None


def test_format_auto_detects_subdataset(tmp_path):
    h5 = _write_h5(tmp_path / 'slc.h5')
    assert _format_nc_filename(h5) == f'NETCDF:"{h5}":"//data/VV"'


def test_format_non_hdf5_unchanged(tmp_path):
    tif = tmp_path / 'a.tif'
    tif.write_bytes(b'')
    assert _format_nc_filename(tif, '/data/VV') == str(tif)


def test_netcdf_driver_keeps_cf_georeferencing(tmp_path):
    """The netCDF driver reports the geotransform the HDF5 driver drops."""
    h5 = _write_cf_h5(tmp_path / 'cf.h5')

    ds = _open_quiet(_format_nc_filename(h5, '/data/VV'))
    assert ds is not None
    gt = ds.GetGeoTransform(can_return_null=True)
    assert gt is not None
    assert gt[1] == X_STEP and gt[5] == Y_STEP
    assert ds.GetProjection()
    ds = None

    ds = _open_quiet(f'HDF5:"{h5}"://data/VV')
    assert ds.GetGeoTransform(can_return_null=True) is None
    ds = None


def test_read_metadata_keeps_cf_georeferencing(tmp_path):
    h5 = _write_cf_h5(tmp_path / 'cf.h5')
    meta = sio.read_metadata(h5, geo=True)
    assert float(meta['X_STEP']) == X_STEP
    assert float(meta['Y_STEP']) == Y_STEP
    assert int(meta['EPSG']) == EPSG


def test_vrt_interferogram_keeps_georeferencing(tmp_path):
    """A VRT built from CF SLCs carries SRS/GeoTransform and passes it on."""
    from mintpy.stdproc.generate_ifgram import create_interferogram_from_vrt

    ref = _write_cf_h5(tmp_path / 'ref.h5')
    sec = _write_cf_h5(tmp_path / 'sec.h5')

    vrt_path = tmp_path / 'fullres.int.vrt'
    VRTInterferogram(ref, sec, vrt_path, subdataset='/data/VV')

    text = vrt_path.read_text()
    assert '<GeoTransform>' in text
    assert '<SRS>' in text

    ds = _open_quiet(str(vrt_path))
    assert ds.GetGeoTransform(can_return_null=True) is not None
    assert ds.GetProjection()
    ds = None

    out_tif = tmp_path / 'fullres.int.tif'
    assert create_interferogram_from_vrt(vrt_path, out_tif, 'isce3')

    ds = _open_quiet(str(out_tif))
    gt = ds.GetGeoTransform(can_return_null=True)
    assert gt is not None
    assert gt[1] == X_STEP and gt[5] == Y_STEP
    assert ds.GetProjection()
    ds = None
