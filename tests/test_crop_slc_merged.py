#!/usr/bin/env python3
"""Tests for the merged ``stdproc.crop_slc``.

Covers:
  * isce3: a geocoded GeoTIFF is cropped to the bbox window, keeping its
    (shifted) georeferencing and writing ``yyyymmdd.slc.tif``;
  * a bbox with no overlap is skipped rather than mis-written;
  * isce2 (radar coordinates, no georeferencing) is rejected;
  * ``dry_run`` writes nothing.
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
from osgeo import osr  # noqa: E402

gdal.UseExceptions()

from mintpy.stdproc import io as sio  # noqa: E402
from mintpy.stdproc.crop_slc import crop_slc  # noqa: E402
from mintpy.utils import readfile  # noqa: E402


def _write_tif(path, arr, gt=None, epsg=None):
    path = str(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ds = gdal.GetDriverByName('GTiff').Create(path, arr.shape[1], arr.shape[0], 1,
                                              gdal.GDT_CFloat32)
    if gt is not None:
        ds.SetGeoTransform(gt)
    if epsg:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(epsg)
        ds.SetProjection(srs.ExportToWkt())
    ds.GetRasterBand(1).WriteArray(np.asarray(arr, dtype=np.complex64))
    ds = None
    return path


def test_crop_isce3_geocoded(tmp_path):
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    arr = (np.arange(400).reshape(20, 20) + 1j).astype(np.complex64)
    _write_tif(slc_dir / '20200101.slc.tif', arr,
               gt=(-120.0, 0.01, 0.0, 40.0, 0.0, -0.01), epsg=4326)

    out_dir = tmp_path / 'cropped'
    ret = crop_slc(input_dir=str(slc_dir), output_dir=str(out_dir),
                   bbox=(-120.01, 39.79, -119.94, 40.01), processor='isce3',
                   pattern='*.slc.tif')
    assert ret == 0

    out = out_dir / '20200101.slc.tif'
    assert out.is_file()
    atr = readfile.read_attribute(out)
    assert atr['FILE_TYPE'] == '.slc'
    assert float(atr['X_FIRST']) == -120.0
    assert float(atr['X_STEP']) == 0.01
    assert str(atr['EPSG']) == '4326'

    data, _ = sio.read_raster(out)
    assert np.iscomplexobj(data)
    assert data.shape[0] == 20


def test_crop_isce3_no_overlap_is_skipped(tmp_path):
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    _write_tif(slc_dir / '20200101.slc.tif', np.ones((10, 10), dtype=np.complex64),
               gt=(-120.0, 0.01, 0.0, 40.0, 0.0, -0.01), epsg=4326)

    out_dir = tmp_path / 'cropped'
    ret = crop_slc(input_dir=str(slc_dir), output_dir=str(out_dir),
                   bbox=(10.0, 10.0, 10.1, 10.1), processor='isce3',
                   pattern='*.slc.tif')
    assert ret == 0
    assert not (out_dir / '20200101.slc.tif').exists()


def test_crop_requires_bbox(tmp_path):
    with pytest.raises(ValueError):
        crop_slc(input_dir=str(tmp_path), output_dir=str(tmp_path / 'o'),
                 bbox=None, processor='isce3')


def test_crop_isce2_rejected(tmp_path):
    """Bbox cropping needs geocoded SLCs; isce2 radar products are rejected."""
    with pytest.raises(ValueError):
        crop_slc(input_dir=str(tmp_path), output_dir=str(tmp_path / 'o'),
                 bbox=(1.0, 2.0, 3.0, 4.0), processor='isce2')


def test_crop_dry_run_writes_nothing(tmp_path):
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    _write_tif(slc_dir / '20200101.slc.tif', np.ones((10, 10), dtype=np.complex64),
               gt=(-120.0, 0.01, 0.0, 40.0, 0.0, -0.01), epsg=4326)
    out_dir = tmp_path / 'cropped'
    ret = crop_slc(input_dir=str(slc_dir), output_dir=str(out_dir),
                   bbox=(-120.01, 39.79, -119.94, 40.01), processor='isce3',
                   pattern='*.slc.tif', dry_run=True)
    assert ret == 0
    assert not (out_dir / '20200101.slc.tif').exists()
