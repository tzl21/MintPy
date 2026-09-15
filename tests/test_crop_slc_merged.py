#!/usr/bin/env python3
"""Tests for the merged ``stdproc.crop_slc``.

Covers:
  * isce3: a geocoded GeoTIFF is cropped to the bbox window, keeping its
    (shifted) georeferencing and writing ``yyyymmdd.slc.tif``;
  * isce2: the radar window comes from the standard ``<merged>/geom_reference``
    ``lon.rdr.full`` / ``lat.rdr.full`` lookup tables, the SLC is written as a
    georeferenced-free GeoTIFF, and the geometry products are cropped in their
    original ENVI + ``.hdr`` / ``.xml`` form;
  * a bbox with no overlap is skipped rather than mis-written;
  * isce2 without the standard geometry tree is rejected;
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


def _make_radar_inputs(root):
    """Standard ISCE2 merged tree.

    ``<root>/merged/SLC/<date>/*.slc.full`` plus the geometry in
    ``<root>/merged/geom_reference`` (``lat.rdr.full`` / ``lon.rdr.full``),
    which ``crop_slc`` derives automatically from the SLC input.
    """
    merged = root / 'merged'
    geom = merged / 'geom_reference'
    geom.mkdir(parents=True)
    lat = (30.0 - np.arange(20)[:, None] * 0.001) + np.zeros((20, 20))
    lon = (100.0 + np.arange(20)[None, :] * 0.001) + np.zeros((20, 20))
    for name, arr in (('lat.rdr.full', lat), ('lon.rdr.full', lon)):
        ds = gdal.GetDriverByName('GTiff').Create(str(geom / name), 20, 20, 1,
                                                  gdal.GDT_Float32)
        ds.GetRasterBand(1).WriteArray(arr.astype(np.float32))
        ds = None

    slc_dir = merged / 'SLC'
    d = slc_dir / '20200102'
    d.mkdir(parents=True)
    arr = np.full((20, 20), 2j, dtype=np.complex64)
    _write_tif(d / '20200102.slc.full', arr)
    return merged, slc_dir


def test_crop_isce2_geoless_tif_and_envi_geometry(tmp_path):
    merged, slc_dir = _make_radar_inputs(tmp_path)

    out_dir = tmp_path / 'cropped'
    ret = crop_slc(input_dir=str(slc_dir / '*'), output_dir=str(out_dir),
                   bbox=(100.004, 29.990, 100.008, 29.996), processor='isce2',
                   pattern='*.slc.full')
    assert ret == 0

    # SLC -> georeferenced-free GeoTIFF
    out = out_dir / '20200102.slc.tif'
    assert out.is_file()
    atr = readfile.read_attribute(out)
    for key in ('X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP', 'EPSG'):
        assert key not in atr, f'{key} must not exist on a radar-coordinate product'
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['SLC2IFG_PROCESSOR'] == 'isce2'
    assert atr['FILE_TYPE'] == '.slc'

    data, _ = sio.read_raster(out)
    assert np.iscomplexobj(data)
    assert np.allclose(data, 2j)

    # geometry -> still ENVI + .hdr / .xml, cropped next to the output dir
    out_geom = tmp_path / 'geom'
    assert (out_geom / 'lat.rdr.full').is_file()
    assert (out_geom / 'lat.rdr.full.hdr').is_file()
    assert (out_geom / 'lat.rdr.full.xml').is_file()


def test_bbox_to_window_isce2_radar(tmp_path):
    """A geoless isce2 SLC maps the bbox through <merged>/geom_reference."""
    merged, slc_dir = _make_radar_inputs(tmp_path)
    slc = slc_dir / '20200102' / '20200102.slc.full'
    # bbox over cols 3..5 / rows 4..6 -> 1-px margin -> (x0, y0, w, h)
    window = sio.bbox_to_window(slc, (100.003, 29.994, 100.005, 29.996))
    assert window == (2, 3, 5, 5)


def test_crop_isce2_out_of_bbox(tmp_path):
    merged, slc_dir = _make_radar_inputs(tmp_path)
    out_dir = tmp_path / 'cropped'
    with pytest.raises(ValueError):
        crop_slc(input_dir=str(slc_dir / '*'), output_dir=str(out_dir),
                 bbox=(10.0, 10.0, 10.1, 10.1), processor='isce2',
                 pattern='*.slc.full')


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


def test_crop_isce2_without_geometry_errors(tmp_path):
    """isce2 radar crop needs the standard <merged>/geom_reference tree."""
    slc_dir = tmp_path / 'SLC'
    slc_dir.mkdir()
    with pytest.raises(ValueError):
        crop_slc(input_dir=str(slc_dir), output_dir=str(tmp_path / 'o'),
                 bbox=(1.0, 2.0, 3.0, 4.0), processor='isce2', pattern='*.slc')


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
