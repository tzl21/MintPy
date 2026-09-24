#!/usr/bin/env python3
"""geometryDict: a raster that is NOT on the observation grid must be warped.

Regression for 5f34216a: ``geometryDict.read()`` resized every 2-D geometry /
mask array to the observation shape, so a water mask with a different CRS
(e.g. a global 1-arcsec mask covering 4x4 degrees) was squashed into the AOI
and the geo-aware warp in ``write2hdf5`` never ran.  The mask is now
reprojected whenever the CRS or the footprint differs, and the block-mean /
resize path is kept for the same-grid (resolution-only) case.
"""
import os
import sys

import numpy as np
import pytest

# PROJ data dir is not exported in a non-interactive shell; set it before the
# first osgeo/PROJ call (mirrors tests/test_io_geoless_tiff.py).
for _pj in (
    os.path.join(os.path.dirname(os.path.dirname(sys.executable)), 'share', 'proj'),
    os.path.join(os.environ.get('CONDA_PREFIX', ''), 'share', 'proj'),
):
    if os.path.isdir(_pj) and 'PROJ_LIB' not in os.environ:
        os.environ['PROJ_LIB'] = _pj
        os.environ['PROJ_DATA'] = _pj
        break

gdal = pytest.importorskip('osgeo.gdal')
osr = pytest.importorskip('osgeo.osr')

from mintpy.objects.stackDict import geometryDict  # noqa: E402
from mintpy.utils import readfile                 # noqa: E402

#: synthetic AOI: the SE quadrant of a 4x4 deg mask tile around Guam
MASK_LON_W, MASK_LON_E = 144.60, 145.00
MASK_LAT_S, MASK_LAT_N = 13.40, 13.80
AOI_LON_W, AOI_LON_E = 144.80, 145.00
AOI_LAT_S, AOI_LAT_N = 13.40, 13.60
UTM_EPSG = 32655


def _write_raster(path, data, geotransform, epsg, dtype=None):
    """Write a single-band GeoTIFF with an explicit grid."""
    dtype = dtype or data.dtype
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(path), data.shape[1], data.shape[0], 1,
                       gdal.GDT_Byte if dtype == np.uint8 else gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(epsg))
    ds.SetProjection(srs.ExportToWkt())
    band = ds.GetRasterBand(1)
    band.WriteArray(data)
    if dtype != np.uint8:
        # a real nodata value: read_attribute would otherwise carry
        # NoDataValue=None, which h5py cannot write as an attribute
        band.SetNoDataValue(-9999.0)
    ds = None
    return str(path)


def _grid_from_lonlat(lon_w, lat_s, lon_e, lat_n, epsg, res=10.0):
    """(geotransform, width, height) of a target grid covering a lon/lat box."""
    src = osr.SpatialReference()
    src.ImportFromEPSG(4326)
    dst = osr.SpatialReference()
    dst.ImportFromEPSG(int(epsg))
    src.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src, dst)
    x0, y0 = transform.TransformPoint(lon_w, lat_s)[:2]
    x1, y1 = transform.TransformPoint(lon_e, lat_n)[:2]
    width = max(2, int(round((x1 - x0) / res)))
    height = max(2, int(round((y1 - y0) / res)))
    return (x0, res, 0.0, y1, 0.0, -res), width, height


def _mask_tile(path):
    """4x4 deg EPSG:4326 mask; land (1) only in the NW quadrant -> 25% land."""
    res = 0.01
    n = int(round((MASK_LON_E - MASK_LON_W) / res))
    lons = MASK_LON_W + (np.arange(n) + 0.5) * res
    lats = MASK_LAT_N - (np.arange(n) + 0.5) * res
    land = (lons[None, :] < 144.80) & (lats[:, None] > 13.60)
    data = land.astype(np.uint8)
    assert abs(data.mean() - 0.25) < 0.01
    return _write_raster(path, data, (MASK_LON_W, res, 0.0, MASK_LAT_N, 0.0, -res),
                         4326, dtype=np.uint8)


def _target_grid_files(tmp_path, epsg=UTM_EPSG, lon_w=AOI_LON_W, lat_s=AOI_LAT_S,
                       lon_e=AOI_LON_E, lat_n=AOI_LAT_N, res=100.0):
    """Reference geometry raster on the target grid + its metadata."""
    geotransform, width, height = _grid_from_lonlat(lon_w, lat_s, lon_e, lat_n,
                                                    epsg, res=res)
    height_file = _write_raster(tmp_path / 'height.tif',
                                np.ones((height, width), dtype=np.float32) * 100.0,
                                geotransform, epsg)
    return height_file, readfile.read_attribute(height_file)


def test_mask_on_a_different_grid_is_warped(tmp_path, capsys):
    """The AOI is entirely in the mask's ocean quadrant -> warp gives ~0% land.

    The old shape-only resize would have squeezed the whole 4x4 deg tile
    (25% land) into the AOI.
    """
    mask = _mask_tile(tmp_path / 'swbd_watermask.tif')
    height_file, obs_atr = _target_grid_files(tmp_path)

    geom = geometryDict(processor='isce3',
                        datasetDict={'height': height_file, 'waterMask': mask},
                        extraMetadata=obs_atr,
                        ref_size=(int(obs_atr['LENGTH']), int(obs_atr['WIDTH'])))
    data, _ = geom.read('waterMask')

    out = capsys.readouterr().out
    assert 'reproject waterMask' in out
    assert data.shape == (int(obs_atr['LENGTH']), int(obs_atr['WIDTH']))
    assert data.mean() < 0.05, f'land fraction {data.mean():.3f} (warp expected ~0)'

    # the previous shape-only resize kept the whole tile's 25% land fraction
    ds = gdal.Open(mask)
    raw = ds.GetRasterBand(1).ReadAsArray()
    ds = None
    resized = geom._downsample_to(raw, data.shape, 'waterMask')
    assert resized.mean() > 0.2, 'sanity: the two paths must differ'


def test_same_grid_mask_is_still_downsampled(tmp_path, capsys):
    """Same CRS/origin/extent, finer sampling -> block mean, not a warp."""
    epsg = UTM_EPSG
    geotransform, width, height = _grid_from_lonlat(AOI_LON_W, AOI_LAT_S,
                                                    AOI_LON_E, AOI_LAT_N, epsg, 100.0)
    height_file = _write_raster(tmp_path / 'height.tif',
                                np.ones((height, width), dtype=np.float32) * 100.0,
                                geotransform, epsg)
    obs_atr = readfile.read_attribute(height_file)

    # same grid, 2x finer: left half land, right half water
    coarse = np.zeros((height, width), dtype=np.uint8)
    coarse[:, :width // 2] = 1
    fine = np.repeat(np.repeat(coarse, 2, axis=0), 2, axis=1)
    fine_gt = (geotransform[0], geotransform[1] / 2, 0.0,
               geotransform[3], 0.0, geotransform[5] / 2)
    mask = _write_raster(tmp_path / 'mask_fine.tif', fine, fine_gt, epsg,
                         dtype=np.uint8)

    geom = geometryDict(processor='isce3',
                        datasetDict={'height': height_file, 'waterMask': mask},
                        extraMetadata=obs_atr,
                        ref_size=(height, width))
    assert geom._same_grid_as_target(mask)
    data, _ = geom.read('waterMask')

    out = capsys.readouterr().out
    assert 'downsample waterMask' in out and 'reproject' not in out
    assert data.shape == (height, width)
    assert abs(float(data.mean()) - 0.5) < 0.02


def test_disjoint_mask_warns(tmp_path):
    """A mask from another region cannot overlap -> loud warning, not silence."""
    mask = _mask_tile(tmp_path / 'swbd_watermask.tif')          # around Guam
    # target grid somewhere else entirely (UTM 12N, SW USA)
    height_file, obs_atr = _target_grid_files(tmp_path, epsg=32612,
                                              lon_w=-112.6, lat_s=41.5,
                                              lon_e=-112.4, lat_n=41.7)
    geom = geometryDict(processor='isce3',
                        datasetDict={'height': height_file, 'waterMask': mask},
                        extraMetadata=obs_atr,
                        ref_size=(int(obs_atr['LENGTH']), int(obs_atr['WIDTH'])))
    with pytest.warns(UserWarning, match='does not overlap'):
        data, _ = geom.read('waterMask')
    assert data.shape == (int(obs_atr['LENGTH']), int(obs_atr['WIDTH']))
    assert data.mean() == 0.0          # nothing overlaps -> all water


def test_write2hdf5_water_mask_matches_the_warp(tmp_path):
    """End-to-end: the written geometry file carries the reprojected mask."""
    mask = _mask_tile(tmp_path / 'swbd_watermask.tif')
    height_file, obs_atr = _target_grid_files(tmp_path)
    out_file = tmp_path / 'geometryGeo.h5'

    obs_meta = {k: obs_atr[k] for k in
                ('LENGTH', 'WIDTH', 'X_FIRST', 'Y_FIRST', 'X_STEP', 'Y_STEP', 'EPSG')
                if k in obs_atr}
    geom = geometryDict(processor='isce3',
                        datasetDict={'height': height_file, 'waterMask': mask},
                        extraMetadata=dict(obs_meta, FILE_TYPE='geometry'),
                        ref_size=(int(obs_atr['LENGTH']), int(obs_atr['WIDTH'])))
    geom.write2hdf5(outputFile=str(out_file))

    written, atr = readfile.read(str(out_file), datasetName='waterMask')
    assert written.shape == (int(obs_atr['LENGTH']), int(obs_atr['WIDTH']))
    assert float(written.mean()) < 0.05

    # independent reference: GDAL Warp straight onto the recorded grid
    reference = gdal.Warp(
        '', mask, format='MEM',
        outputBounds=(float(atr['X_FIRST']),
                      float(atr['Y_FIRST']) + written.shape[0] * float(atr['Y_STEP']),
                      float(atr['X_FIRST']) + written.shape[1] * float(atr['X_STEP']),
                      float(atr['Y_FIRST'])),
        width=written.shape[1], height=written.shape[0],
        dstSRS=f'EPSG:{int(atr["EPSG"])}', resampleAlg='near')
    expected = reference.GetRasterBand(1).ReadAsArray() > 0
    reference = None
    assert np.array_equal(np.asarray(written).astype(bool), expected)


def test_no_georeferencing_keeps_the_legacy_path(tmp_path, capsys):
    """A raster without a CRS cannot be warped -> keep the old resize path."""
    height_file, obs_atr = _target_grid_files(tmp_path)
    ny, nx = int(obs_atr['LENGTH']) * 2, int(obs_atr['WIDTH']) * 2

    plain = tmp_path / 'plain.tif'
    driver = gdal.GetDriverByName('GTiff')
    ds = driver.Create(str(plain), nx, ny, 1, gdal.GDT_Byte)
    ds.GetRasterBand(1).WriteArray(np.ones((ny, nx), dtype=np.uint8))
    ds = None

    geom = geometryDict(processor='isce3',
                        datasetDict={'height': height_file, 'waterMask': str(plain)},
                        extraMetadata=obs_atr,
                        ref_size=(int(obs_atr['LENGTH']), int(obs_atr['WIDTH'])))
    assert geom._same_grid_as_target(str(plain))
    data, _ = geom.read('waterMask')
    assert data.shape == (int(obs_atr['LENGTH']), int(obs_atr['WIDTH']))
    assert 'downsample waterMask' in capsys.readouterr().out
