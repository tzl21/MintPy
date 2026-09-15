#!/usr/bin/env python3
"""Tests for the native ``utils.vrt_utils.VRTInterferogram``.

``generate_ifgram`` used to hard-depend on ``dolphin.interferogram`` while the
native drop-in sat unused.  These tests lock the native behaviour:

  * the VRT materialises to ``ref * conj(sec)``;
  * a geocoded SLC keeps its georeferencing in the VRT, a radar-coordinate SLC
    produces a VRT with no georeferencing at all;
  * the materialised array matches dolphin's implementation when dolphin is
    installed (skipped otherwise).
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

from mintpy.stdproc.utils.vrt_utils import VRTInterferogram  # noqa: E402

GT = (-120.0, 0.01, 0.0, 40.0, 0.0, -0.01)


def _write_slc(path, arr, gt=GT, epsg=4326):
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


def _materialise(vrt_path):
    ds = gdal.Open(str(vrt_path), gdal.GA_ReadOnly)
    try:
        return ds.GetRasterBand(1).ReadAsArray()
    finally:
        ds = None


def _inputs(tmp_path):
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(24, 32)) + 1j * rng.normal(size=(24, 32))).astype(np.complex64)
    b = (rng.normal(size=(24, 32)) + 1j * rng.normal(size=(24, 32))).astype(np.complex64)
    # date-bearing names: dolphin derives the pair dates from the filenames
    return (_write_slc(tmp_path / '20200101.slc.tif', a),
            _write_slc(tmp_path / '20200113.slc.tif', b), a, b)


def test_native_vrt_materialises_interferogram(tmp_path):
    pa, pb, a, b = _inputs(tmp_path)
    vrt = tmp_path / 'fullres.int.vrt'
    VRTInterferogram(ref_slc=pa, sec_slc=pb, path=vrt, outdir=vrt.parent,
                     verify_slcs=True, write=True)

    assert vrt.is_file()
    data = _materialise(vrt)
    assert data.shape == a.shape
    assert np.allclose(data, a * np.conj(b), rtol=1e-6, atol=1e-6)

    # georeferencing is carried through the VRT
    ds = gdal.Open(str(vrt))
    try:
        assert ds.GetGeoTransform(can_return_null=True) == pytest.approx(GT)
        assert ds.GetProjection()
    finally:
        ds = None


def test_native_vrt_geoless_has_no_georeferencing(tmp_path):
    rng = np.random.default_rng(1)
    a = (rng.normal(size=(16, 16)) + 1j).astype(np.complex64)
    b = (rng.normal(size=(16, 16)) + 2j).astype(np.complex64)
    pa = _write_slc(tmp_path / 'a.slc', a, gt=None, epsg=None)
    pb = _write_slc(tmp_path / 'b.slc', b, gt=None, epsg=None)

    vrt = tmp_path / 'fullres.int.vrt'
    VRTInterferogram(ref_slc=pa, sec_slc=pb, path=vrt, verify_slcs=True, write=True)

    text = vrt.read_text()
    assert 'GeoTransform' not in text
    assert '<SRS>' not in text

    ds = gdal.Open(str(vrt))
    try:
        assert ds.GetGeoTransform(can_return_null=True) is None
        assert not ds.GetProjection()
        data = ds.GetRasterBand(1).ReadAsArray()
    finally:
        ds = None
    assert np.allclose(data, a * np.conj(b), rtol=1e-6, atol=1e-6)


def test_native_vrt_matches_dolphin(tmp_path):
    dolphin = pytest.importorskip('dolphin.interferogram')
    pa, pb, _a, _b = _inputs(tmp_path)

    vrt_native = tmp_path / 'native.int.vrt'
    VRTInterferogram(ref_slc=pa, sec_slc=pb, path=vrt_native,
                     outdir=vrt_native.parent, verify_slcs=True, write=True)

    vrt_dolphin = tmp_path / 'dolphin.int.vrt'
    dolphin.VRTInterferogram(ref_slc=pa, sec_slc=pb, path=vrt_dolphin,
                             outdir=vrt_dolphin.parent, verify_slcs=True,
                             write=True)

    native = _materialise(vrt_native)
    ref = _materialise(vrt_dolphin)
    assert native.shape == ref.shape
    np.testing.assert_array_equal(native, ref)
