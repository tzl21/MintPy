"""Unit tests for the ISCE3/Dolphin interface."""

import glob
import os

import numpy as np
import pytest
from osgeo import gdal, osr

from mintpy.utils import isce3_utils, readfile


def _write_baseline_file(dir_path, date_pair, bperp, fmt='Bperp (m)'):
    fname = os.path.join(str(dir_path), f'{date_pair}.txt')
    with open(fname, 'w') as f:
        f.write(f'{fmt}: {bperp}\n')
        f.write('Bpar average (m): 12.345\n')
    return fname


def test_read_baseline_timeseries_isce3(tmp_path):
    # plain directory layout
    _write_baseline_file(tmp_path, '20210104_20210110', -85.65046462265644)
    _write_baseline_file(tmp_path, '20210104_20210116', -19.081015753493432,
                         fmt='Bperp average (m)')
    # a pair NOT starting with the reference date must be ignored
    _write_baseline_file(tmp_path, '20210110_20210116', 99.9)

    bdict = isce3_utils.read_baseline_timeseries_isce3(str(tmp_path))
    assert bdict['20210104'] == [0.0, 0.0]
    assert bdict['20210110'] == pytest.approx([-85.65046462265644] * 2)
    assert bdict['20210116'] == pytest.approx([-19.081015753493432] * 2)
    assert len(bdict) == 3


def test_read_baseline_timeseries_isce3_nested_glob(tmp_path):
    # Dolphin layout: baselines/t124_xxx/YYYYMMDD_YYYYMMDD.txt
    burst_dir = tmp_path / 'baselines' / 't124_264305_iw2'
    burst_dir.mkdir(parents=True)
    _write_baseline_file(burst_dir, '20210104_20210110', -85.65)
    _write_baseline_file(burst_dir, '20210104_20210116', -19.08)

    # glob in the baseline dir itself must be expanded
    bdict = isce3_utils.read_baseline_timeseries_isce3(
        str(tmp_path / 'baselines' / 't124_*/'))
    assert bdict['20210104'] == [0.0, 0.0]
    assert bdict['20210110'] == pytest.approx([-85.65] * 2)

    # plain dir without glob must fall back to a recursive search
    bdict2 = isce3_utils.read_baseline_timeseries_isce3(str(tmp_path / 'baselines'))
    assert bdict2['20210110'] == pytest.approx([-85.65] * 2)

    # a non-existent dir must NOT crash, just return an empty dict
    assert isce3_utils.read_baseline_timeseries_isce3(
        str(tmp_path / 'nope')) == {}


def test_read_baseline_missing_dir_warns(tmp_path, capsys):
    isce3_utils.read_baseline_timeseries_isce3(str(tmp_path / 'empty'))
    assert 'no baseline text files found' in capsys.readouterr().out


#########################################################################
# read_attribute on ISCE3/Dolphin GeoTIFF products
#########################################################################


def _write_synthetic_geotiff(fname, shape=(8, 10), gt=(500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0),
                             epsg=32605, nodata=0.0):
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(fname), shape[1], shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    data = np.ones(shape, dtype=np.float32) * 1.5
    data[0, :] = nodata
    band.WriteArray(data)
    if nodata is not None:
        band.SetNoDataValue(nodata)
    ds = None
    return fname


def test_read_attribute_isce3_unw_tif(tmp_path):
    # Dolphin-style name with the date pair in the filename
    fname = str(tmp_path / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['DATE12'] == '220105-220117'
    assert atr['FILE_TYPE'] == '.unw'
    assert atr['LENGTH'] == 8
    assert atr['WIDTH'] == 10
    assert atr['EPSG'] == '32605'
    assert atr['NO_DATA_VALUE'] == '0.0'


def test_read_attribute_isce3_fullres_parent_dir(tmp_path):
    # Dolphin "fullres.unw.tif" layout: date pair only in the parent dir name
    ifg_dir = tmp_path / '20220105_20220117'
    ifg_dir.mkdir()
    fname = str(ifg_dir / 'fullres.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['DATE12'] == '220105-220117'
    assert atr['FILE_TYPE'] == '.unw'


def test_read_attribute_isce3_rsc_priority(tmp_path):
    # a .rsc sidecar written by prep_isce3 must take priority over the
    # attributes derived from the GeoTIFF itself
    fname = str(tmp_path / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)
    with open(fname + '.rsc', 'w') as f:
        f.write('PROCESSOR isce3\n')
        f.write('DATE12 220105-220117\n')
        f.write('P_BASELINE_TOP_HDR -2.5048807779724456\n')
        f.write('P_BASELINE_BOTTOM_HDR -2.5048807779724456\n')

    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['P_BASELINE_TOP_HDR'] == '-2.5048807779724456'
    assert 'FILE_PATH' in atr


def test_read_isce3_geotiff_no_nodata(tmp_path):
    # a band without nodata must not produce the string 'None'
    fname = str(tmp_path / '20220105_20220117.int.tif')
    _write_synthetic_geotiff(fname, nodata=None)
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'isce3'
    assert 'NO_DATA_VALUE' not in atr or atr['NO_DATA_VALUE'] != 'None'


#########################################################################
# water mask auto-align (geometryDict._warp_water_mask)
#########################################################################


def _write_baseline_pair(dir_path, date1, date2, bperp, bpar=None):
    os.makedirs(str(dir_path), exist_ok=True)
    fname = os.path.join(str(dir_path), f'{date1}_{date2}.txt')
    with open(fname, 'w') as f:
        f.write(f'Bperp (m): {bperp}\n')
        if bpar is not None:
            f.write(f'Bpar (m): {bpar}\n')
    return fname


def _write_cslc_files(tmp_path, burst_ids, dates):
    files = []
    for burst_id in burst_ids:
        for date in dates:
            date_dir = tmp_path / 'CSLC' / burst_id / date
            date_dir.mkdir(parents=True, exist_ok=True)
            fname = date_dir / f'{burst_id}_{date}.h5'
            fname.write_text('')
            files.append(str(fname))
    return files


def test_scan_and_group_cslc_files(tmp_path):
    _write_cslc_files(tmp_path, ['t124_264305_iw2'], ['20210104', '20210110', '20210116'])

    pattern = str(tmp_path / 'CSLC' / 't124*' / '2021*' / 't124*.h5')
    files = isce3_utils.scan_cslc_h5_files(pattern)
    assert len(files) == 3
    assert all(f.endswith('.h5') for f in files)

    groups = isce3_utils.group_cslc_files_by_burst(files)
    assert list(groups) == ['t124_264305_iw2']
    assert sorted(groups['t124_264305_iw2']['dates']) == ['20210104', '20210110', '20210116']

    assert isce3_utils.is_cslc_baseline_input(pattern) is True
    assert isce3_utils.get_burst_ids_from_cslc(pattern) == ['t124_264305_iw2']


def test_is_cslc_baseline_input_false_for_baselines_dir(tmp_path):
    _write_baseline_pair(tmp_path / 'baselines' / 't124_264305_iw2',
                         '20210104', '20210110', -85.6)
    assert isce3_utils.is_cslc_baseline_input(str(tmp_path / 'baselines')) is False


def test_compute_baselines_output_layout(tmp_path, monkeypatch):
    _write_cslc_files(tmp_path, ['t124_264305_iw2'], ['20210104', '20210110', '20210116'])
    pattern = str(tmp_path / 'CSLC' / 't124*' / '2*' / 't124*.h5')

    calls = []

    def fake_reference(h5_ref):
        return {'h5_ref': h5_ref}

    def fake_pair(ref_cache, h5_secondary):
        calls.append(h5_secondary)
        return 12.5, -34.5

    monkeypatch.setattr(isce3_utils, '_prepare_baseline_reference', fake_reference)
    monkeypatch.setattr(isce3_utils, '_compute_baseline_pair', fake_pair)

    baseline_dirs = isce3_utils.compute_baselines_isce3(pattern, update_mode=False)
    assert baseline_dirs == [str(tmp_path / 'CSLC' / 'baselines')]

    burst_dir = tmp_path / 'CSLC' / 'baselines' / 't124_264305_iw2'
    for pair in ['20210104_20210110', '20210104_20210116']:
        fname = burst_dir / f'{pair}.txt'
        assert fname.is_file()
        content = fname.read_text()
        assert 'Bperp (m): -34.5' in content
        assert 'Bpar (m): 12.5' in content
    assert len(calls) == 2

    # update_mode=True must skip the pairs that already exist
    calls.clear()
    isce3_utils.compute_baselines_isce3(pattern, update_mode=True)
    assert len(calls) == 0


def test_read_baseline_multiburst_average(tmp_path):
    dates = ['20210104', '20210110', '20210116']
    for burst_id, bperp in [('t124_264304_iw2', -10.0), ('t124_264305_iw2', -20.0)]:
        for date2 in dates[1:]:
            _write_baseline_pair(tmp_path / 'baselines' / burst_id,
                                 dates[0], date2, bperp, bpar=1.0)

    bdict = isce3_utils.read_baseline_timeseries_isce3(str(tmp_path / 'baselines'))
    assert bdict['20210104'] == [0.0, 0.0]
    assert bdict['20210110'] == pytest.approx([-15.0, -15.0])
    assert bdict['20210116'] == pytest.approx([-15.0, -15.0])


def test_read_baseline_multiburst_inconsistent(tmp_path, capsys):
    # burst A has 3 dates, burst B only 2 -> use burst A and warn
    dates = ['20210104', '20210110', '20210116']
    for date2 in dates[1:]:
        _write_baseline_pair(tmp_path / 'baselines' / 't124_264304_iw2',
                             dates[0], date2, -10.0)
    _write_baseline_pair(tmp_path / 'baselines' / 't124_264305_iw2',
                         dates[0], dates[1], -20.0)

    bdict = isce3_utils.read_baseline_timeseries_isce3(str(tmp_path / 'baselines'))
    assert bdict['20210110'] == pytest.approx([-10.0, -10.0])
    assert bdict['20210116'] == pytest.approx([-10.0, -10.0])
    assert 'different' in capsys.readouterr().out


def test_read_baseline_burst_id_filter(tmp_path, capsys):
    dates = ['20210104', '20210110']
    _write_baseline_pair(tmp_path / 'baselines' / 't124_264304_iw2',
                         dates[0], dates[1], -10.0)
    _write_baseline_pair(tmp_path / 'baselines' / 't124_264305_iw2',
                         dates[0], dates[1], -20.0)

    bdict = isce3_utils.read_baseline_timeseries_isce3(
        str(tmp_path / 'baselines'), burst_ids=['t124_264304_iw2'])
    assert bdict['20210110'] == pytest.approx([-10.0, -10.0])
    assert 'not in the CSLC stack' in capsys.readouterr().out


def test_get_burst_ids_from_geom_dirs(tmp_path):
    for burst_id in ['t124_264304_iw2', 't124_264305_iw2']:
        geom_dir = tmp_path / 'geom' / burst_id / '20210104'
        geom_dir.mkdir(parents=True)
        (geom_dir / 'static_layers.h5').write_text('')

    for geom_dir in [str(tmp_path / 'geom'), str(tmp_path / 'geom' / 't124*/20210104')]:
        geom_dirs = sorted(glob.glob(geom_dir)) if '*' in geom_dir else [geom_dir]
        assert isce3_utils.get_burst_ids_from_geom_dirs(geom_dirs) == [
            't124_264304_iw2', 't124_264305_iw2']
