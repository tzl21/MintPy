#!/usr/bin/env python3
"""Tests for the unified stdproc io layer and geo-less GeoTIFF support.

Covers:
  * ``io.write_raster(geo=False)`` writes NO georeferencing and the reader does
    not fabricate X_FIRST/X_STEP/EPSG for it;
  * ``io.write_raster(geo=True)`` preserves the geotransform / EPSG;
  * embedded GDAL metadata keeps a radar product self-describing
    (PROCESSOR / FILE_TYPE / SLC2IFG_PROCESSOR);
  * ``io.read_raster`` returns complex data as complex (not phase);
  * ``save_gdal.write_gdal`` no longer crashes on ``EPSG=None`` and never writes
    an assumed EPSG:4326 for a plain raster.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

# PROJ data dir is not exported in a non-interactive shell; set it before the
# first osgeo/PROJ call (mirrors stdproc/crop_slc_geo.py).
for _pj in (
    os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'proj'),
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'proj'),
):
    if os.path.isdir(_pj) and 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = _pj
        break

gdal = pytest.importorskip('osgeo.gdal')
gdal.UseExceptions()

from mintpy.save_gdal import write_gdal  # noqa: E402
from mintpy.stdproc import io as sio  # noqa: E402
from mintpy.utils import readfile  # noqa: E402

GEO_META = {
    'X_FIRST': 500000.0,
    'Y_FIRST': 4000000.0,
    'X_STEP': 30.0,
    'Y_STEP': -30.0,
    'EPSG': 32605,
}

#: keys that must never exist on a radar-coordinate (geo-less) product
FABRICATED_GEO_KEYS = ('X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP', 'EPSG', 'UTM_ZONE')


def _complex_arr(rows=8, cols=16):
    return (np.ones((rows, cols), dtype=np.complex64) * (1 + 2j)).astype(np.complex64)


def test_write_geoless_product_has_no_fabricated_geo(tmp_path):
    out = tmp_path / '20200101.slc.tif'
    sio.write_raster(_complex_arr(), out, meta={'FILE_TYPE': '.slc'},
                     geo=False, processor='isce2')
    atr = readfile.read_attribute(out)

    for key in FABRICATED_GEO_KEYS:
        assert key not in atr, f'{key} must not be fabricated for a geo-less product'

    assert atr['X_UNIT'] == 'pixel'
    assert atr['Y_UNIT'] == 'pixel'
    assert atr['FILE_TYPE'] == '.slc'
    # a radar GeoTIFF is read with GDAL; its real provenance is kept separately
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['SLC2IFG_PROCESSOR'] == 'isce2'
    assert int(atr['WIDTH']) == 16
    assert int(atr['LENGTH']) == 8


def test_geo_product_preserves_georeferencing(tmp_path):
    out = tmp_path / '20200101_20200113' / 'fullres.unw.tif'
    meta = dict(GEO_META, FILE_TYPE='.unw')
    sio.write_raster(np.ones((8, 16), dtype=np.float32), out, meta=meta,
                     geo=True, processor='isce3')
    atr = readfile.read_attribute(out)

    assert float(atr['X_FIRST']) == 500000.0
    assert float(atr['Y_FIRST']) == 4000000.0
    assert float(atr['X_STEP']) == 30.0
    assert float(atr['Y_STEP']) == -30.0
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['FILE_TYPE'] == '.unw'
    assert 'EPSG' in atr and str(atr['EPSG']) == '32605'


def test_geo_auto_does_not_invent_epsg_for_plain_raster(tmp_path):
    """geo=None + no geotransform -> a plain raster, without a 4326 assumption."""
    out = tmp_path / 'plain.tif'
    sio.write_raster(np.ones((4, 5), dtype=np.float32), out, meta={}, geo=None)
    ds = gdal.Open(str(out))
    try:
        assert ds.GetGeoTransform(can_return_null=True) is None
        assert not ds.GetProjection()
    finally:
        ds = None

    atr = readfile.read_attribute(out)
    for key in FABRICATED_GEO_KEYS:
        assert key not in atr


def test_read_raster_keeps_complex(tmp_path):
    arr = _complex_arr(4, 6)
    out = tmp_path / 'x.slc.tif'
    sio.write_raster(arr, out, meta={'FILE_TYPE': '.slc'}, geo=False, processor='isce2')

    data, meta = sio.read_raster(out)
    assert np.iscomplexobj(data)
    assert data.dtype == np.complex64
    assert np.allclose(data, 1 + 2j)
    assert meta['WIDTH'] == 6
    assert meta['LENGTH'] == 4


def test_read_raster_window_and_box(tmp_path):
    arr = np.arange(32, dtype=np.float32).reshape(4, 8)
    out = tmp_path / 'd.tif'
    sio.write_raster(arr, out, meta={}, geo=False, processor='isce2')

    data, _ = sio.read_raster(out, box=(2, 1, 6, 3))
    assert data.shape == (2, 4)
    assert np.allclose(data, arr[1:3, 2:6])


def test_write_raster_atomic_and_nodata(tmp_path):
    out = tmp_path / 'c.tif'
    sio.write_raster(np.array([[1, 2], [3, 4]], dtype=np.uint16), out,
                     geo=False, processor='isce2', nodata=0)

    assert out.is_file()
    assert not (tmp_path / 'c.tif.tmp').exists()

    data, meta = sio.read_raster(out)
    assert data.dtype == np.uint16
    assert np.allclose(data, [[1, 2], [3, 4]])
    assert float(meta['NO_DATA_VALUE']) == 0.0


def test_write_gdal_tolerates_epsg_none(tmp_path):
    """Regression: read_attribute sets EPSG=None; int(None) used to raise."""
    meta = dict(X_FIRST=0.0, Y_FIRST=0.0, X_STEP=1.0, Y_STEP=-1.0, EPSG=None)
    out = tmp_path / 'epsg_none.tif'
    with pytest.warns(UserWarning):
        write_gdal(np.zeros((2, 2), dtype=np.float32), meta, out_file=str(out))
    assert out.is_file()


def test_write_gdal_geo_false_skips_georeferencing(tmp_path):
    meta = dict(GEO_META, EPSG=None)
    out = tmp_path / 'geo_false.tif'
    write_gdal(np.zeros((2, 2), dtype=np.float32), meta, out_file=str(out), geo=False)

    ds = gdal.Open(str(out))
    try:
        assert ds.GetGeoTransform(can_return_null=True) is None
        assert not ds.GetProjection()
    finally:
        ds = None


def test_write_gdal_geo_true_requires_georeferencing(tmp_path):
    out = tmp_path / 'missing_geo.tif'
    with pytest.raises(ValueError):
        write_gdal(np.zeros((2, 2), dtype=np.float32), {}, out_file=str(out), geo=True)


def test_bbox_to_window_geoless_raises(tmp_path):
    out = tmp_path / '20200101.slc.tif'
    sio.write_raster(np.ones((8, 8), dtype=np.complex64), out,
                     meta={'FILE_TYPE': '.slc'}, geo=False, processor='isce2')
    with pytest.raises(ValueError):
        sio.bbox_to_window(out, (0.0, 0.0, 1.0, 1.0))


def test_bbox_to_window_geo(tmp_path):
    out = tmp_path / 'burst' / '20200101.slc.tif'
    # geographic CRS so the WSEN bbox and the raster share the same units
    meta = {
        'X_FIRST': -120.0, 'Y_FIRST': 40.0,
        'X_STEP': 0.01, 'Y_STEP': -0.01, 'EPSG': 4326,
        'FILE_TYPE': '.slc',
    }
    sio.write_raster(np.ones((20, 20), dtype=np.complex64), out, meta=meta,
                     geo=True, processor='isce3')

    # a bbox covering the whole raster snaps to the full grid (with clamping)
    window = sio.bbox_to_window(
        out, (-120.01, 40.0 - 0.20, -120.0 + 0.20, 40.01))
    assert window is not None
    x0, y0, w, h = window
    assert (x0, y0) == (0, 0)
    assert (w, h) == (20, 20)


def test_parse_wsen():
    assert sio.parse_wsen('1 2 3 4') == (1.0, 2.0, 3.0, 4.0)
    assert sio.parse_wsen('1,2,3,4') == (1.0, 2.0, 3.0, 4.0)
    with pytest.raises(ValueError):
        sio.parse_wsen('1 2 3')


def test_import_stdproc_does_not_require_osgeo(monkeypatch):
    """The io module must stay lazy: no osgeo at import time."""
    import subprocess

    code = (
        'import sys; import mintpy.stdproc; import mintpy.stdproc.io; '
        "sys.exit(1 if 'osgeo' in sys.modules else 0)"
    )
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[1] / 'src')
    env['PYTHONPATH'] = src + os.pathsep + env.get('PYTHONPATH', '')
    env.pop('PROJ_LIB', None)
    proc = subprocess.run([sys.executable, '-c', code], env=env,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
