#!/usr/bin/env python3
"""Tests for slc2ifg.slc_input resolution.

Covers glob expansion, SLC pattern inference, multi-burst extraction and the
standard ISCE2 geometry directory derivation.
"""

from mintpy.stdproc.utils.slc_input import (
    extract_date,
    infer_slc_pattern,
    is_burst_id,
    resolve_slc_input,
    standard_geom_dir,
)


def _write(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def test_flat_glob_pattern_inference(tmp_path):
    for d in ('20200101', '20200113'):
        _write(tmp_path / 'y1' / d / f'{d}.slc.tif')

    s = resolve_slc_input(str(tmp_path / 'y*' / '*'))
    assert s.pattern == '*.slc.tif'
    assert s.bursts == [None]
    assert not s.is_multi_burst
    assert s.date_list() == ['20200101', '20200113']
    assert s.file_for(None, '20200113').name == '20200113.slc.tif'


def test_multi_burst_glob(tmp_path):
    bursts = ('t001_264305_iw1', 't002_264305_iw1')
    for b in bursts:
        for d in ('20200101', '20200113'):
            _write(tmp_path / b / d / f'{b}_{d}.h5')

    s = resolve_slc_input(str(tmp_path / 't*' / '*'))
    assert s.pattern == '*.h5'
    assert s.is_multi_burst
    assert s.bursts == list(bursts)
    assert s.date_list('t001_264305_iw1') == ['20200101', '20200113']
    assert s.file_for('t002_264305_iw1', '20200101').name == (
        't002_264305_iw1_20200101.h5')
    assert len(s.files) == 4


def test_isce2_slc_full_pattern(tmp_path):
    _write(tmp_path / 'SLC' / '20210104' / '20210104.slc.full')
    s = resolve_slc_input(str(tmp_path / 'SLC' / '*'))
    assert s.pattern == '*.slc.full'
    assert s.date_list() == ['20210104']


def test_standard_geom_dir(tmp_path):
    (tmp_path / 'merged' / 'geom_reference').mkdir(parents=True)
    _write(tmp_path / 'merged' / 'SLC' / '20210104' / '20210104.slc.full')
    geom = tmp_path / 'merged' / 'geom_reference'
    # from the slc_input glob ...
    assert standard_geom_dir(str(tmp_path / 'merged' / 'SLC' / '*')) == geom
    # ... and from a date directory (the engine's input_dirs)
    assert standard_geom_dir(str(tmp_path / 'merged' / 'SLC' / '20210104')) == geom
    # absent
    assert standard_geom_dir(str(tmp_path / 'nowhere')) is None


def test_small_helpers():
    assert is_burst_id('t124_264305_iw2')
    assert not is_burst_id('20210104')
    assert extract_date('t124_264305_iw2_20210104.h5') == '20210104'
    assert infer_slc_pattern(['a.h5'], 'isce3') == '*.h5'
    assert infer_slc_pattern(['a.slc.full'], 'isce2') == '*.slc.full'
