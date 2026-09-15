#!/usr/bin/env python3
"""geometryDict reference-size downsampling (full-res geometry -> ifgram grid)."""

import numpy as np

from mintpy.objects import stackDict


def _geom(ref_size, monkeypatch, arr, family='latitude'):
    """Build a geometryDict whose single raster read returns ``arr``."""
    monkeypatch.setattr(
        stackDict.readfile, 'read',
        lambda *a, **k: (arr, {}))
    monkeypatch.setattr(
        stackDict.readfile, 'read_attribute',
        lambda *a, **k: {'LENGTH': str(arr.shape[0]), 'WIDTH': str(arr.shape[1])})
    return stackDict.geometryDict(
        datasetDict={family: 'fake.rdr'},
        extraMetadata={'LENGTH': str(arr.shape[0]), 'WIDTH': str(arr.shape[1])},
        ref_size=ref_size)


def test_get_size_uses_ref_size(monkeypatch):
    arr = np.zeros((8, 8), dtype=np.float32)
    obj = _geom((2, 2), monkeypatch, arr)
    assert obj.get_size() == (2, 2)


def test_read_downsamples_to_ref_size(monkeypatch):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    obj = _geom((2, 2), monkeypatch, arr)
    data, _ = obj.read('latitude')
    assert data.shape == (2, 2)
    assert np.allclose(data, [[13.5, 17.5], [45.5, 49.5]])


def test_read_keeps_mask_boolean(monkeypatch):
    arr = np.zeros((8, 8), dtype=np.uint8)
    arr[:4, :4] = 1
    obj = _geom((2, 2), monkeypatch, arr, family='waterMask')
    data, _ = obj.read('waterMask')
    assert data.dtype == bool
    assert data.tolist() == [[True, False], [False, False]]


def test_read_height_nodata_becomes_nan(monkeypatch):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    arr[0, 0] = -32768.0
    obj = _geom((2, 2), monkeypatch, arr, family='height')
    data, _ = obj.read('height')
    # the no-data sentinel is excluded from the block mean
    assert np.isclose(data[0, 0], (216.0 - 0.0) / 15.0)


def test_no_ref_size_keeps_native_shape(monkeypatch):
    arr = np.arange(64, dtype=np.float32).reshape(8, 8)
    obj = _geom(None, monkeypatch, arr)
    data, _ = obj.read('latitude')
    assert data.shape == (8, 8)
