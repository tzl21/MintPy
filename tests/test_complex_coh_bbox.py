"""Regression tests for the complex_coh read-time AOI crop (slc2ifg.bbox).

Covers:
  * ``read_complex_image(window=...)`` shape + windowed geotransform;
  * ``complex_coh_tiled(crop_window=...)`` == untiled windowed computation;
  * engine wiring: ``complex_coh`` carries bbox when crop_slc is off and
    drops it when the SLCs are already cropped on disk.
"""

from pathlib import Path

import numpy as np
import pytest

gdal = pytest.importorskip('osgeo.gdal')
from osgeo import gdal  # noqa: E402

gdal.UseExceptions()

#: EPSG-free geotransform (no PROJ needed): 0.001 deg-ish pixels
GT = (100.0, 0.001, 0.0, 10.0, 0.0, -0.001)


def _write_slc(path, arr, gt=GT):
    rows, cols = arr.shape
    ds = gdal.GetDriverByName('GTiff').Create(
        str(path), cols, rows, 1, gdal.GDT_CFloat32)
    ds.SetGeoTransform(gt)
    ds.GetRasterBand(1).WriteArray(np.asarray(arr, dtype=np.complex64))
    ds = None
    return path


def test_read_complex_image_window(tmp_path):
    """A window read returns only the window and shifts its geotransform."""
    from mintpy.stdproc.generate_coh_complex import read_complex_image

    arr = (np.arange(40 * 60).reshape(40, 60)
           + 1j * np.arange(40 * 60).reshape(40, 60)).astype(np.complex64)
    p = _write_slc(tmp_path / '20230105.slc.tif', arr)

    data, meta = read_complex_image(str(p), 'isce3', window=(10, 5, 20, 30))
    assert data.shape == (30, 20)
    assert meta['rows'] == 30 and meta['cols'] == 20
    assert meta['window'] == (10, 5, 20, 30)
    np.testing.assert_array_equal(data, arr[5:35, 10:30])
    # window origin shift
    assert meta['transform'][0] == pytest.approx(100.0 + 10 * 0.001)
    assert meta['transform'][3] == pytest.approx(10.0 - 5 * 0.001)

    # no window -> full scene, unchanged behaviour
    full, meta_full = read_complex_image(str(p), 'isce3')
    assert full.shape == (40, 60)
    assert meta_full['window'] is None
    assert meta_full['transform'] == GT


def test_read_complex_image_window_outside(tmp_path):
    from mintpy.stdproc.generate_coh_complex import read_complex_image

    p = _write_slc(tmp_path / 'a.slc.tif',
                   np.ones((8, 8), dtype=np.complex64))
    with pytest.raises(ValueError):
        read_complex_image(str(p), 'isce3', window=(8, 0, 4, 4))


def test_complex_coh_tiled_crop_window_matches_untiled(tmp_path):
    """Tiled + crop_window equals the untiled windowed computation."""
    from mintpy.stdproc.engine.gpu_kernels import complex_coh_block
    from mintpy.stdproc.engine.tiling import complex_coh_tiled
    from mintpy.stdproc.generate_coh_complex import read_complex_image

    rng = np.random.default_rng(0)
    rows, cols = 48, 64
    a = (rng.normal(size=(rows, cols))
         + 1j * rng.normal(size=(rows, cols))).astype(np.complex64)
    b = (rng.normal(size=(rows, cols))
         + 1j * rng.normal(size=(rows, cols))).astype(np.complex64)
    pa = _write_slc(tmp_path / 'a.slc.tif', a)
    pb = _write_slc(tmp_path / 'b.slc.tif', b)

    win = (13, 7, 30, 25)   # x0, y0, w, h
    out = tmp_path / 'crop.cpx.coh.tif'
    complex_coh_tiled(str(pa), str(pb), str(out), 5, 16, 'isce3',
                      tile_workers=2, gpu=False, crop_window=win)

    ds = gdal.Open(str(out))
    assert (ds.RasterXSize, ds.RasterYSize) == (30, 25)
    got = ds.GetRasterBand(1).ReadAsArray()
    gt = ds.GetGeoTransform()
    ds = None

    s1, _ = read_complex_image(str(pa), 'isce3', window=win)
    s2, _ = read_complex_image(str(pb), 'isce3', window=win)
    ref = complex_coh_block(s1, s2, 5, gpu=False)
    ref[:2, :] = 0
    ref[-2:, :] = 0
    ref[:, :2] = 0
    ref[:, -2:] = 0

    np.testing.assert_allclose(got, ref, atol=1e-6)
    assert gt[0] == pytest.approx(100.0 + 13 * 0.001)
    assert gt[3] == pytest.approx(10.0 - 7 * 0.001)


def _engine(tmp_path, stages):
    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    root = tmp_path / stages.replace(',', '_')
    inp = root / 'input'
    inp.mkdir(parents=True)
    for d in ('20220105', '20220117'):
        (inp / f'{d}.slc.tif').touch()
    cfg = root / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {root}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        f'engine.tools = {stages}\n'
        'engine.gpu = false\n'
        'slc2ifg.bbox = 1 2 3 4\n'
        'slc2ifg.bbox_buffer = 0.01\n')
    return Engine(load_engine_config(str(cfg)))


def test_complex_coh_engine_read_crop_wiring(tmp_path):
    """Read-time crop: complex_coh gets bbox; with crop_slc it is dropped."""
    eng = _engine(tmp_path, 'ifgram_list,generate_ifgram,complex_coh,unwrap')
    g = eng.plan(dry_run=True)
    p = g.nodes['complex_coh#single#20220105_20220117'].ctx.params
    assert str(p['bbox']).strip() == '1 2 3 4'
    assert p['bbox_buffer'] == pytest.approx(0.01)

    eng2 = _engine(tmp_path,
                   'crop_slc,ifgram_list,generate_ifgram,complex_coh,unwrap')
    g2 = eng2.plan(dry_run=True)
    p2 = g2.nodes['complex_coh#single#20220105_20220117'].ctx.params
    assert 'bbox' not in p2 and 'bbox_buffer' not in p2
    pg = g2.nodes['generate_ifgram#single#20220105_20220117'].ctx.params
    assert 'bbox' not in pg and 'bbox_buffer' not in pg


def test_complex_coh_tool_crop_window(tmp_path, monkeypatch):
    """The tool writes a window-shaped coherence for a bbox, on both the
    untiled and the tiled path (the bug behind unwrap's shape mismatch)."""
    import mintpy.stdproc.crop_slc_geo as csg

    from mintpy.stdproc.engine.tool import ToolContext
    from mintpy.stdproc.engine.tools.complex_coh import ComplexCohTool

    # avoid a PROJ-dependent transform: the bbox maps to a fixed window
    monkeypatch.setattr(
        csg, 'bbox_to_window',
        lambda path, wsen, subdataset='/data/VV', buffer=0.0: (13, 7, 30, 25))

    rng = np.random.default_rng(1)
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    for d in ('20220105', '20220117'):
        arr = (rng.normal(size=(48, 64))
               + 1j * rng.normal(size=(48, 64))).astype(np.complex64)
        _write_slc(slc_dir / f'{d}.slc.tif', arr)

    pair_file = tmp_path / 'ifgram_list.txt'
    pair_file.write_text('20220105-20220117\n')

    for tile_size, tag in ((0, 'untiled'), (16, 'tiled')):
        out = tmp_path / f'{tag}.cpx.coh.tif'
        ctx = ToolContext(
            tool_name='complex_coh',
            inputs={'pairs_file': pair_file, 'slc_dir': slc_dir,
                    'date1': '20220105', 'date2': '20220117', 'burst': None},
            outputs={'coh': out},
            params={'processor': 'isce3', 'slc_pattern': '*.slc.tif',
                    'subdataset': '/data/VV', 'window_size': 5,
                    'window_type': 'triangular', 'bbox': '1 2 3 4',
                    'bbox_buffer': 0.01, 'tile_size': tile_size,
                    'use_gpu': False, 'max_workers': 1},
            work_dir=tmp_path)
        ComplexCohTool().run(ctx)
        ds = gdal.Open(str(out))
        assert (ds.RasterXSize, ds.RasterYSize) == (30, 25), tag
        ds = None
