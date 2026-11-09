"""Unit tests for the ISCE3/Dolphin interface."""

import glob
import os

import numpy as np
import pytest
from osgeo import gdal, osr

from mintpy.objects.stackDict import geometryDict
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
    atr = readfile.read_attribute(fname, processor='isce3')
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['DATE12'] == '220105-220117'
    assert atr['FILE_TYPE'] == '.unw'
    assert atr['LENGTH'] == 8
    assert atr['WIDTH'] == 10
    assert atr['EPSG'] == '32605'
    assert atr['NO_DATA_VALUE'] == '0.0'


def test_read_attribute_isce3_per_pair_dir(tmp_path):
    # per-pair-directory layout (the date pair is in the directory name):
    # ifgrams/YYYYMMDD_YYYYMMDD/fullres.unw.tif, date pair in the dir name.
    # (Dolphin itself writes flat YYYYMMDD_YYYYMMDD.int.tif, covered above.)
    ifg_dir = tmp_path / '20220105_20220117'
    ifg_dir.mkdir()
    fname = str(ifg_dir / 'fullres.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname, processor='isce3')
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['DATE12'] == '220105-220117'
    assert atr['FILE_TYPE'] == '.unw'


def test_read_attribute_isce3_rsc_priority(tmp_path):
    # a .rsc sidecar written by prep_isce3 must take priority over the
    # attributes derived from the GeoTIFF itself: write sidecar values that
    # CONFLICT with what the GeoTIFF/filename would produce, so that a
    # reversed precedence order fails this test
    fname = str(tmp_path / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)
    with open(fname + '.rsc', 'w') as f:
        f.write('PROCESSOR isce3\n')
        f.write('DATE12 990101-990102\n')   # the filename says 220105-220117
        f.write('LENGTH 8\n')
        f.write('WIDTH 10\n')
        f.write('P_BASELINE_TOP_HDR -2.5048807779724456\n')
        f.write('P_BASELINE_BOTTOM_HDR -2.5048807779724456\n')

    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['DATE12'] == '990101-990102'   # sidecar wins over the filename
    assert atr['P_BASELINE_TOP_HDR'] == '-2.5048807779724456'
    assert 'FILE_PATH' in atr


def test_read_isce3_geotiff_no_nodata(tmp_path):
    # a band without nodata must not invent a NO_DATA_VALUE: check the
    # GeoTIFF reader itself (so that an invented '0.0'/'nan'/'none' fails),
    # plus the public no-data-value lookup used by the readers.
    fname = str(tmp_path / '20220105_20220117.int.tif')
    _write_synthetic_geotiff(fname, nodata=None)
    assert 'NO_DATA_VALUE' not in readfile.read_isce3_geotiff(fname)
    assert readfile.read_attribute(fname, processor='isce3')['PROCESSOR'] == 'isce3'
    assert readfile.get_no_data_value(fname) is None


def test_read_attribute_isce3_date12_priority(tmp_path):
    # the interferogram date pair must win over a date pair in an ancestor dir
    root = tmp_path / '20190101_20191231_project'
    ifg_dir = root / 'ifgrams'
    ifg_dir.mkdir(parents=True)

    fname = str(ifg_dir / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)
    assert readfile.read_attribute(fname, processor='isce3')['DATE12'] == '220105-220117'

    # per-pair-directory layout: date pair in the immediate parent dir
    pair_dir = ifg_dir / '20220205_20220217'
    pair_dir.mkdir()
    fname = str(pair_dir / 'fullres.unw.tif')
    _write_synthetic_geotiff(fname)
    assert readfile.read_attribute(fname, processor='isce3')['DATE12'] == '220205-220217'

    # Dolphin flat .int.tif layout, again under an ancestor with a different
    # date pair
    int_dir = tmp_path / 'S1_20190101_20191231' / 'dolphin' / 'interferograms'
    int_dir.mkdir(parents=True)
    fname = str(int_dir / '20220305_20220317.int.tif')
    _write_synthetic_geotiff(fname)
    assert readfile.read_attribute(fname, processor='isce3')['DATE12'] == '220305-220317'

    # without the hint the file keeps the generic GDAL attributes; with the
    # hint it is read as isce3, but no date pair is invented for it
    fname = str(ifg_dir / 'fullres.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'gdal'
    assert 'DATE12' not in atr

    atr = readfile.read_attribute(fname, processor='isce3')
    assert atr['PROCESSOR'] == 'isce3'
    assert 'DATE12' not in atr


def test_read_attribute_tif_default_is_gdal(tmp_path):
    # GeoTIFFs are not promoted to isce3 based on their name: without an
    # explicit processor hint they keep the generic GDAL attributes
    fname = str(tmp_path / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['FILE_TYPE'] != '.unw'
    assert 'DATE12' not in atr

    # the same for a LiCSAR-style name
    fname = str(tmp_path / '20220105_20220117.geo.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'gdal'
    assert atr['FILE_TYPE'] != '.unw'

    # an explicitly configured non-isce3 processor also disables isce3 handling
    fname2 = str(tmp_path / '20220205_20220217.unw.tif')
    _write_synthetic_geotiff(fname2)
    atr = readfile.read_attribute(fname2, processor='roipac')
    assert atr['PROCESSOR'] == 'gdal'
    assert 'DATE12' not in atr


def test_read_attribute_isce3_processor_hint(tmp_path):
    # an explicit processor='isce3' from the configuration is authoritative
    fname = str(tmp_path / '20220105_20220117.geo.unw.tif')
    _write_synthetic_geotiff(fname)
    atr = readfile.read_attribute(fname, processor='isce3')
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['FILE_TYPE'] == '.unw'
    assert atr['DATE12'] == '220105-220117'

    # read() forwards the hint to the attribute reader used for the data
    _, atr = readfile.read(fname, processor='isce3')
    assert atr['PROCESSOR'] == 'isce3'


def test_read_attribute_rsc_sidecar_short_circuit(tmp_path):
    # with a .rsc sidecar, read_attribute() takes the .rsc branch and does not
    # call read_isce3_geotiff(); this is the pre-existing upstream
    # ".rsc > file itself" priority, so a partial sidecar is NOT completed
    # from the GeoTIFF (same behavior as main for any .tif + .rsc)
    fname = str(tmp_path / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)
    with open(fname + '.rsc', 'w') as f:
        f.write('LENGTH 8\n')
        f.write('WIDTH 10\n')
    atr = readfile.read_attribute(fname)
    assert atr['PROCESSOR'] == 'roipac'
    assert 'DATE12' not in atr


def test_warp_water_mask(tmp_path):
    # reference grid: 10 x 8 pixels at 10 m
    ref_fname = str(tmp_path / 'ref.tif')
    _write_synthetic_geotiff(ref_fname, shape=(8, 10))

    # water mask on a coarser grid: 5 x 4 pixels at 20 m, same CRS/extent
    src_fname = str(tmp_path / 'water_mask.tif')
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(src_fname, 5, 4, 1, gdal.GDT_Float32)
    ds.SetGeoTransform((500000.0, 20.0, 0.0, 4000000.0, 0.0, -20.0))
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(32605)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    mask = np.ones((4, 5), dtype=np.float32)   # 1 = land
    mask[0, :] = 0.0                            # 0 = water
    band.WriteArray(mask)
    ds = None

    geom_obj = geometryDict(
        processor='isce3',
        datasetDict={'height': ref_fname, 'waterMask': src_fname},
        extraMetadata={'dummy': 'x'},
    )
    result = geom_obj._warp_water_mask('waterMask', 8, 10)
    assert result.shape == (8, 10)
    assert set(np.unique(result)).issubset({0.0, 1.0})
    # the top row of the source (water) stays water after the warp
    assert np.all(result[0, :] == 0.0)
    assert np.all(result[-1, :] == 1.0)


#########################################################################
# geometry merging / angle computation (synthetic data, no real data needed)
#########################################################################


def _write_geotiff(fname, data, gt=(500000.0, 10.0, 0.0, 4000000.0, 0.0, -10.0), epsg=32605):
    driver = gdal.GetDriverByName('GTiff')
    data = np.asarray(data, dtype=np.float32)
    ds = driver.Create(str(fname), data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(gt)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(epsg)
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    ds = None
    return fname


def test_warp_water_mask_box_and_multilook(tmp_path):
    # reference grid 8 x 10 at 10 m; water mask on the same grid: left half water, right half land
    ref_fname = str(tmp_path / 'ref.tif')
    _write_synthetic_geotiff(ref_fname, shape=(8, 10))
    mask = np.zeros((8, 10), dtype=np.float32)
    mask[:, 5:] = 1.0
    mask_fname = str(tmp_path / 'water_mask.tif')
    _write_geotiff(mask_fname, mask)

    geom_obj = geometryDict(
        processor='isce3',
        datasetDict={'height': ref_fname, 'waterMask': mask_fname},
        extraMetadata={'dummy': 'x'},
    )

    # box=(x0, y0, x1, y1) = the lower-right land half, at half resolution
    result = geom_obj._warp_water_mask('waterMask', 4, 5, box=(5, 4, 10, 8), xstep=1, ystep=1)
    assert result.shape == (4, 5)
    assert np.all(result == 1.0)

    # full grid is returned unchanged when no box/step is given
    result_full = geom_obj._warp_water_mask('waterMask', 8, 10)
    assert np.all(result_full[:, :5] == 0.0)
    assert np.all(result_full[:, 5:] == 1.0)


def test_build_vrt_from_h5_requires_coordinate_arrays(tmp_path):
    import h5py

    # missing x/y coordinates
    h5_file = tmp_path / 'static_layers_missing.h5'
    with h5py.File(h5_file, 'w') as f:
        grp = f.create_group('data')
        grp.create_dataset('los_east', data=np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(KeyError):
        isce3_utils.build_vrt_from_h5(h5_file, tmp_path / 'vrt', ['los_east.tif'])

    # degenerate 1-pixel coordinate arrays
    h5_file = tmp_path / 'static_layers_1px.h5'
    with h5py.File(h5_file, 'w') as f:
        grp = f.create_group('data')
        grp.create_dataset('x_coordinates', data=np.array([0.0], dtype=np.float64))
        grp.create_dataset('y_coordinates', data=np.array([0.0], dtype=np.float64))
        grp.create_dataset('los_east', data=np.zeros((1, 1), dtype=np.float32))
    with pytest.raises(ValueError):
        isce3_utils.build_vrt_from_h5(h5_file, tmp_path / 'vrt', ['los_east.tif'])


def test_compute_angles_with_finite_nodata(tmp_path):
    nodata = -9999.0
    east = np.full((4, 5), 0.1, dtype=np.float32)
    north = np.full((4, 5), 0.2, dtype=np.float32)
    east[0, :] = nodata
    north[0, :] = nodata
    east_file = _write_geotiff(tmp_path / 'los_east.tif', east)
    north_file = _write_geotiff(tmp_path / 'los_north.tif', north)

    inc_file = tmp_path / 'incidenceAngle.tif'
    isce3_utils.compute_incidence_angle(east_file, north_file, inc_file, nodata=nodata)
    inc = gdal.Open(str(inc_file)).ReadAsArray()
    assert np.all(inc[0, :] == nodata)
    assert np.all(np.isfinite(inc[1:, :]))

    az_file = tmp_path / 'azimuthAngle.tif'
    isce3_utils.compute_azimuth_angle(east_file, north_file, az_file, nodata=nodata)
    az = gdal.Open(str(az_file)).ReadAsArray()
    assert np.all(az[0, :] == nodata)
    assert np.all((az[1:, :] >= 0) & (az[1:, :] < 360))


#########################################################################
# prep_isce3.py: RSC generation and geometry preparation
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


def test_geometry_dict_processor_hint(tmp_path):
    # the processor from the configuration decides on the ISCE3/Dolphin
    # GeoTIFF detection: an explicit non-isce3 processor must not promote the
    # file, while an explicit isce3 one must
    fname = str(tmp_path / '20220105_20220117.unw.tif')
    _write_synthetic_geotiff(fname)

    geom_obj = geometryDict(processor='roipac', datasetDict={'height': fname},
                            extraMetadata={'dummy': 'x'})
    assert geom_obj.get_metadata()['PROCESSOR'] == 'gdal'

    geom_obj = geometryDict(processor='isce3', datasetDict={'height': fname},
                            extraMetadata={'dummy': 'x'})
    atr = geom_obj.get_metadata()
    assert atr['PROCESSOR'] == 'isce3'
    assert atr['DATE12'] == '220105-220117'
