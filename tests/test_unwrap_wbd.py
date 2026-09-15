#!/usr/bin/env python3
"""Tests for the .wbd water-body companion resolution (unwrap masking)."""

import numpy as np
import pytest

pytest.importorskip('osgeo.gdal')

from mintpy.stdproc.unwrap_ifgram import _open_wbd  # noqa: E402


def _write_wbd(tmp_path, companion):
    """Write ``test.wbd`` + its VRT companion; return (wbd, vrt) paths."""
    data = np.zeros((4, 4), dtype=np.uint8)
    data[0, 0] = 1
    wbd = tmp_path / 'test.wbd'
    wbd.write_bytes(data.tobytes())
    vrt = tmp_path / ('test.wbd.vrt' if companion == 'wbd.vrt' else 'test.vrt')
    vrt.write_text(
        '<VRTDataset rasterXSize="4" rasterYSize="4">'
        '<SRS>EPSG:4326</SRS>'
        '<GeoTransform>-156.0 0.0002777777777777778 0 20.0 0 -0.0002777777777777778</GeoTransform>'
        '<VRTRasterBand dataType="Byte" band="1" subClass="VRTRawRasterBand">'
        '<SourceFilename relativeToVRT="1">test.wbd</SourceFilename>'
        '<ByteOrder>LSB</ByteOrder><ImageOffset>0</ImageOffset>'
        '<PixelOffset>1</PixelOffset><LineOffset>4</LineOffset>'
        '</VRTRasterBand></VRTDataset>')
    return wbd, vrt


def test_open_wbd_wbd_vrt_companion(tmp_path):
    """The sardem layout is ``xxx.wbd`` + ``xxx.wbd.vrt``."""
    wbd, _ = _write_wbd(tmp_path, 'wbd.vrt')
    ds = _open_wbd(str(wbd))
    assert ds is not None
    assert (ds.RasterXSize, ds.RasterYSize) == (4, 4)


def test_open_wbd_plain_vrt_companion(tmp_path):
    """``xxx.wbd`` + ``xxx.vrt`` is also accepted."""
    wbd, _ = _write_wbd(tmp_path, 'vrt')
    ds = _open_wbd(str(wbd))
    assert ds is not None


def test_open_wbd_without_companion(tmp_path):
    wbd = tmp_path / 'lonely.wbd'
    wbd.write_bytes(b'\x00' * 16)
    assert _open_wbd(str(wbd)) is None
