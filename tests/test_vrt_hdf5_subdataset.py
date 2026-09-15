#!/usr/bin/env python3
"""HDF5 subdataset connection string used by the VRT interferogram sources."""

import numpy as np
import pytest

pytest.importorskip('h5py')
gdal = pytest.importorskip('osgeo.gdal')

from mintpy.stdproc.utils.vrt_utils import _format_nc_filename  # noqa: E402


def _write_h5(path):
    import h5py
    data = (np.ones((4, 4)) + 1j).astype(np.complex64)
    with h5py.File(path, 'w') as f:
        f.create_dataset('/data/VV', data=data)
    return path


def test_format_and_open_hdf5_subdataset(tmp_path):
    """The HDF5 driver needs '://data/VV' (double slash) to open the dataset."""
    h5 = _write_h5(tmp_path / 'slc.h5')
    conn = _format_nc_filename(h5, '/data/VV')
    assert conn == f'HDF5:"{h5}"://data/VV'
    ds = gdal.Open(conn)
    assert ds is not None
    assert (ds.RasterXSize, ds.RasterYSize) == (4, 4)


def test_format_auto_detects_subdataset(tmp_path):
    h5 = _write_h5(tmp_path / 'slc.h5')
    assert _format_nc_filename(h5) == f'HDF5:"{h5}"://data/VV'


def test_format_non_hdf5_unchanged(tmp_path):
    tif = tmp_path / 'a.tif'
    tif.write_bytes(b'')
    assert _format_nc_filename(tif, '/data/VV') == str(tif)
