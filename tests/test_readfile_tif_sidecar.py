#!/usr/bin/env python3
"""Regression tests for reading a GeoTIFF that carries sidecar metadata files.

Covers the failure modes that made an slc2ifg GeoTIFF unreadable before:

  * ``<file>.tif`` + ``<file>.tif.rsc`` used to set PROCESSOR='roipac' and read
    the GeoTIFF as a raw binary, returning garbage;
  * ``<file>.tif`` + ``<file>.tif.xml`` used to be discarded (and an ISCE2 xml
    next to a tif silently returned the wrong FILE_TYPE);
  * a radar-coordinate ``fullres.int.tif`` inside a ``{date1}_{date2}``
    directory used to be mis-detected as a geocoded ISCE3 product and got
    hardcoded X_UNIT='meters'.
"""

import os
import sys

import numpy as np
import pytest

for _pj in (
    os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'proj'),
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'proj'),
):
    if os.path.isdir(_pj) and 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = _pj
        break

gdal = pytest.importorskip('osgeo.gdal')
gdal.UseExceptions()

from mintpy.stdproc import io as sio  # noqa: E402
from mintpy.utils import readfile  # noqa: E402

GEO_META = {
    'X_FIRST': 500000.0,
    'Y_FIRST': 4000000.0,
    'X_STEP': 30.0,
    'Y_STEP': -30.0,
    'EPSG': 32605,
}


def _values(rows=3, cols=4):
    return np.arange(rows * cols, dtype=np.float32).reshape(rows, cols)


def test_tif_with_rsc_is_read_as_gdal(tmp_path):
    """A .rsc next to a GeoTIFF must supplement metadata, not switch to raw read."""
    arr = _values()
    path = tmp_path / 'fullres.int.tif'
    sio.write_raster(arr, path, meta={'FILE_TYPE': '.int'}, geo=False, processor='isce2')

    # the ROI_PAC sidecar (as written by prep_isce) carries baselines
    (tmp_path / 'fullres.int.tif.rsc').write_text(
        'WIDTH 4\n'
        'FILE_LENGTH 3\n'
        'DATA_TYPE float32\n'
        'P_BASELINE_TOP_HDR 123.4\n'
        'P_BASELINE_BOTTOM_HDR 120.1\n'
    )

    atr = readfile.read_attribute(path)
    assert atr['PROCESSOR'] == 'gdal' or atr['PROCESSOR'] == 'isce3'
    assert atr['FILE_TYPE'] == '.int'
    assert float(atr['P_BASELINE_TOP_HDR']) == 123.4

    data, _ = readfile.read(path)
    assert data.shape == arr.shape
    assert np.allclose(data, arr), 'the GeoTIFF must not be read as a raw binary'


def test_tif_with_xml_sidecar_still_geo_less(tmp_path):
    """An ISCE2-style .xml next to a GeoTIFF must not break the GDAL read path."""
    arr = _values()
    path = tmp_path / '20200101.slc.tif'
    sio.write_raster(arr, path, meta={'FILE_TYPE': '.slc'}, geo=False, processor='isce2')

    (tmp_path / '20200101.slc.tif.xml').write_text('<imageFile/>\n')

    atr = readfile.read_attribute(path)
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['FILE_TYPE'] == '.slc'
    assert 'X_FIRST' not in atr

    data, _ = readfile.read(path)
    assert np.allclose(data, arr)


def test_geoless_datepair_int_tif_is_not_isce3(tmp_path):
    arr = _values()
    path = tmp_path / '20200101_20200113' / 'fullres.int.tif'
    sio.write_raster(arr, path, meta={'FILE_TYPE': '.int'}, geo=False, processor='isce2')

    atr = readfile.read_attribute(path)
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['FILE_TYPE'] == '.int'
    for key in ('X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP', 'EPSG'):
        assert key not in atr, f'{key} must not be fabricated'
    assert atr['X_UNIT'] == 'pixel'


def test_geocoded_datepair_int_tif_is_isce3(tmp_path):
    arr = _values()
    path = tmp_path / '20200101_20200113' / 'fullres.int.tif'
    meta = dict(GEO_META, FILE_TYPE='.int')
    sio.write_raster(arr, path, meta=meta, geo=True, processor='isce3')

    atr = readfile.read_attribute(path)
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['FILE_TYPE'] == '.int'
    assert float(atr['X_FIRST']) == 500000.0
    assert float(atr['X_STEP']) == 30.0
    assert 'EPSG' in atr


def test_product_file_type_inference(tmp_path):
    cases = {
        '20200101.slc.tif': '.slc',
        'fullres.int.tif': '.int',
        'mli.unw.tif': '.unw',
        'fullres.unw.conncomp.tif': '.unw.conncomp',
        'fullres.cpx.coh.tif': '.cor',
        'fullres.phsig.coh.tif': '.cor',
        'random.tif': None,
    }
    for name, expected in cases.items():
        assert readfile.product_file_type(tmp_path / name) == expected


def test_coh_tif_file_type(tmp_path):
    arr = _values()
    path = tmp_path / '20200101_20200113' / 'fullres.phsig.coh.tif'
    sio.write_raster(arr, path, meta={'FILE_TYPE': '.cor'}, geo=False, processor='isce2')
    atr = readfile.read_attribute(path)
    assert atr['FILE_TYPE'] == '.cor'
    assert atr['PROCESSOR'] == 'gdal'
