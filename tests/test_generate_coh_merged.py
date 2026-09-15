#!/usr/bin/env python3
"""Tests for the merged ``stdproc.generate_coh`` (cpx + phsig in one module).

Covers:
  * complex coherence naming/shape, for a geocoded input (geo preserved) and a
    radar-coordinate isce2 input (written geo-less);
  * phase-sigma coherence + the ``keep_sigma`` phase std-dev raster;
  * ``generate_coh`` dispatching to both estimators.
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
from mintpy.stdproc.generate_coh import (  # noqa: E402
    generate_coh,
    generate_complex_coherence,
    generate_phsig_coherence,
)
from mintpy.utils import readfile  # noqa: E402

GT = (-120.0, 0.01, 0.0, 40.0, 0.0, -0.01)


def _write(path, arr, gt=GT, epsg=4326):
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


def test_complex_coherence_isce3(tmp_path):
    slc_dir = tmp_path / 'slc'
    rng = np.random.default_rng(0)
    a = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(np.complex64)
    _write(slc_dir / '20200101.slc.tif', a)
    _write(slc_dir / '20200113.slc.tif', a)   # identical -> coherence ~1

    pairs = tmp_path / 'ifgram_list.txt'
    pairs.write_text('20200101-20200113\n')

    out_dir = tmp_path / 'coh'
    ret = generate_complex_coherence(pairs, str(slc_dir), out_dir, processor='isce3')
    assert ret == 0

    out = out_dir / '20200101_20200113' / 'fullres.cpx.coh.tif'
    assert out.is_file()
    atr = readfile.read_attribute(out)
    assert atr['FILE_TYPE'] == '.cor'
    assert float(atr['X_FIRST']) == -120.0
    assert str(atr['EPSG']) == '4326'

    data, _ = sio.read_raster(out)
    assert data.dtype == np.float32
    # interior pixels of identical scenes are highly coherent
    assert float(data[10:22, 10:22].mean()) > 0.9


def test_complex_coherence_isce2_geo_less(tmp_path):
    slc_dir = tmp_path / 'slc'
    arr = np.ones((16, 16), dtype=np.complex64)
    # radar-coordinate input: no georeferencing at all
    _write(slc_dir / '20200101.slc', arr, gt=None, epsg=None)
    _write(slc_dir / '20200113.slc', arr, gt=None, epsg=None)

    pairs = tmp_path / 'pairs.txt'
    pairs.write_text('20200101-20200113\n')

    out_dir = tmp_path / 'coh'
    ret = generate_complex_coherence(pairs, str(slc_dir), out_dir, processor='isce2')
    assert ret == 0

    out = out_dir / '20200101_20200113' / 'fullres.cpx.coh.tif'
    assert out.is_file()
    atr = readfile.read_attribute(out)
    for key in ('X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP', 'EPSG'):
        assert key not in atr
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['SLC2IFG_PROCESSOR'] == 'isce2'


def test_phsig_with_sigma(tmp_path):
    rng = np.random.default_rng(1)
    ifg = (rng.normal(size=(40, 40)) + 1j * rng.normal(size=(40, 40))).astype(np.complex64)
    ifg_dir = tmp_path / 'ifgrams' / '20200101_20200113'
    path = _write(ifg_dir / 'fullres.int.tif', ifg)

    out_dir = tmp_path / 'coh'
    ret = generate_phsig_coherence([str(path)], out_dir, processor='isce3',
                                   phase_sigma_window=5, gradient_window=5,
                                   nlks=1.0, keep_sigma=True)
    assert ret == 0

    coh = out_dir / '20200101_20200113' / 'fullres.phsig.coh.tif'
    sig = out_dir / '20200101_20200113' / 'fullres.phsig.sigma.tif'
    assert coh.is_file() and sig.is_file()

    coh_data, _ = sio.read_raster(coh)
    sig_data, _ = sio.read_raster(sig)
    assert coh_data.shape == ifg.shape
    assert sig_data.shape == ifg.shape
    assert coh_data.min() >= 0.0 and coh_data.max() <= 1.0
    assert sig_data.min() >= 0.0


def test_phsig_without_sigma_writes_only_coh(tmp_path):
    ifg = np.ones((24, 24), dtype=np.complex64)
    path = _write(tmp_path / 'ifgrams' / '20200101_20200113' / 'fullres.int.tif', ifg)
    out_dir = tmp_path / 'coh'
    assert generate_phsig_coherence([str(path)], out_dir, processor='isce3') == 0
    assert (out_dir / '20200101_20200113' / 'fullres.phsig.coh.tif').is_file()
    assert not (out_dir / '20200101_20200113' / 'fullres.phsig.sigma.tif').exists()


def test_generate_coh_dispatch(tmp_path):
    rng = np.random.default_rng(2)
    arr = (rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))).astype(np.complex64)
    slc_dir = tmp_path / 'slc'
    _write(slc_dir / '20200101.slc.tif', arr)
    _write(slc_dir / '20200113.slc.tif', arr)
    pairs = tmp_path / 'pairs.txt'
    pairs.write_text('20200101-20200113\n')

    ifg_dir = tmp_path / 'ifgrams' / '20200101_20200113'
    ifg_path = _write(ifg_dir / 'fullres.int.tif', arr)

    out_dir = tmp_path / 'coh'
    ret = generate_coh(processor='isce3', input_files=[str(ifg_path)],
                       output_dir=out_dir, pairs_file=str(pairs),
                       slc_dir=[str(slc_dir)], keep_sigma=False)
    assert ret == 0
    assert (out_dir / '20200101_20200113' / 'fullres.phsig.coh.tif').is_file()
    assert (out_dir / '20200101_20200113' / 'fullres.cpx.coh.tif').is_file()


def test_generate_coh_requires_inputs(tmp_path):
    # no interferogram input and no pairs -> nothing to do
    assert generate_coh(processor='isce3', output_dir=tmp_path) == 1
