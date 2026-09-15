#!/usr/bin/env python3
"""Unit tests for the slc2ifg ifgram_list ``exclude_date`` support.

Excluding a bad SLC acquisition date must drop it from the date list
*before* pairing so no interferogram involves it — in every mode
(sequential / reference / select) and through every entry point
(standalone CLI, engine eager planning, basic executor).

Run:  python -m pytest tests/test_ifgram_list_exclude_date.py -v
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mintpy.stdproc.ifgram_list import (
    filter_date_list,
    generate_pairs,
    parse_exclude_dates,
)
from mintpy.stdproc.select_ifgrams import check_connected

DATES = ['20230105', '20230117', '20230129', '20230210', '20230222',
         '20230306', '20230318', '20230330', '20230411', '20230423',
         '20230505', '20230517']


def _read_pairs(pair_file):
    return [tuple(line.strip().split('-')) for line in
            pair_file.read_text().splitlines()
            if line.strip() and not line.startswith('#')]


# ------------------------------------------------------------------------
# parse_exclude_dates
# ------------------------------------------------------------------------
def test_parse_exclude_dates_none_and_auto():
    assert parse_exclude_dates(None) == []
    assert parse_exclude_dates('auto') == []
    assert parse_exclude_dates('none') == []
    assert parse_exclude_dates('') == []
    assert parse_exclude_dates('  ') == []


def test_parse_exclude_dates_string_forms():
    assert parse_exclude_dates('20230129') == ['20230129']
    assert parse_exclude_dates('20230129,20230411') == ['20230129', '20230411']
    assert parse_exclude_dates('20230129 20230411') == ['20230129', '20230411']
    assert parse_exclude_dates('20230129, 20230411') == ['20230129', '20230411']
    # dedup + sorted
    assert parse_exclude_dates('20230411,20230129,20230129') == \
        ['20230129', '20230411']


def test_parse_exclude_dates_list_and_invalid():
    assert parse_exclude_dates(['20230129', '20230411']) == \
        ['20230129', '20230411']
    # invalid entries are dropped, valid kept
    assert parse_exclude_dates('20230129,99999999') == ['20230129']
    assert parse_exclude_dates('not-a-date') == []


# ------------------------------------------------------------------------
# filter_date_list
# ------------------------------------------------------------------------
def test_filter_date_list_exclude():
    out = filter_date_list(DATES, exclude_date='20230129')
    assert out == [d for d in DATES if d != '20230129']


def test_filter_date_list_exclude_multiple():
    out = filter_date_list(DATES, exclude_date='20230129,20230411')
    assert out == [d for d in DATES if d not in ('20230129', '20230411')]


def test_filter_date_list_exclude_combined_with_range():
    out = filter_date_list(DATES, start_date='20230210', end_date='20230423',
                           exclude_date='20230318')
    assert out == ['20230210', '20230222', '20230306', '20230330',
                   '20230411', '20230423']


def test_filter_date_list_exclude_missing_warns(caplog):
    with caplog.at_level(logging.WARNING):
        out = filter_date_list(DATES, exclude_date='20230101')
    assert out == DATES
    assert any('20230101' in r.message for r in caplog.records)


# ------------------------------------------------------------------------
# pair generation (the excluded date never appears in any pair)
# ------------------------------------------------------------------------
def test_generate_pairs_sequential_exclude():
    kept = [d for d in DATES if d != '20230129']
    pairs = generate_pairs(kept, mode='sequential', num_connections=3)
    flat = [d for p in pairs for d in p]
    assert '20230129' not in flat
    assert check_connected(kept, pairs)


def test_generate_pairs_select_exclude(tmp_path):
    """select mode: the excluded date never appears in any selected pair."""
    from osgeo import gdal
    kept = ['20230105', '20230117', '20230210']   # 20230129 excluded

    # real (small) coherence rasters for every candidate pair
    for a, b in (('20230105', '20230117'), ('20230105', '20230210'),
                 ('20230117', '20230210')):
        d = tmp_path / f'{a}_{b}'
        d.mkdir(parents=True, exist_ok=True)
        ds = gdal.GetDriverByName('GTiff').Create(
            str(d / 'filt_mli.phsig.coh.tif'), 8, 8, 1, gdal.GDT_Float32)
        ds.GetRasterBand(1).Fill(0.6)
        ds = None

    pairs = generate_pairs(kept, mode='select', num_connections=3,
                           select_params={'coh_root': str(tmp_path),
                                          'min_degree': 2})
    flat = [d for p in pairs for d in p]
    assert '20230129' not in flat
    assert check_connected(kept, pairs)


# ------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------
def test_ifgram_list_cli_exclude_date(tmp_path):
    from mintpy.cli import ifgram_list
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    for d in DATES:
        (slc_dir / d).mkdir()
    out = tmp_path / 'ifg'
    rc = ifgram_list.main([
        '--slc', str(slc_dir), '--outdir', str(out), '--mode', 'sequential',
        '-n', '3', '--exclude-date', '20230129', '--exclude-date', '20230411',
    ])
    assert rc == 0
    pairs = _read_pairs(out / 'ifgram_list.txt')
    flat = [d for p in pairs for d in p]
    assert '20230129' not in flat and '20230411' not in flat
    kept = [d for d in DATES if d not in ('20230129', '20230411')]
    assert check_connected(kept, pairs)


# ------------------------------------------------------------------------
# engine eager planning (cfg-driven)
# ------------------------------------------------------------------------
def test_engine_exclude_date_plan(tmp_path):
    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    inp = tmp_path / 'input'
    inp.mkdir()
    dates = ['20220105', '20220117', '20220129', '20220210',
             '20220222', '20220306']
    for d in dates:
        (inp / f'{d}.slc.tif').touch()
    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        'engine.max_workers = 2\n'
        'engine.gpu = false\n'
        'slc2ifg.ifgram_list.mode = sequential\n'
        'slc2ifg.ifgram_list.num_connections = 3\n'
        'slc2ifg.ifgram_list.exclude_date = 20220129\n'
    )
    eng = Engine(load_engine_config(str(cfg)))
    eng.plan(dry_run=True)
    pair_file = eng.ifgram_dir / 'ifgram_list.txt'
    assert pair_file.exists()
    pairs = eng._read_pairs(pair_file)
    flat = [d for p in pairs for d in p]
    assert '20220129' not in flat
    kept = [d for d in dates if d != '20220129']
    assert check_connected(kept, pairs)


# ------------------------------------------------------------------------
# basic executor (cfg-driven)
# ------------------------------------------------------------------------
def test_basic_executor_exclude_date(tmp_path):
    from mintpy.stdproc.executor import BasicExecutor

    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    dates = ['20220105', '20220117', '20220129', '20220210', '20220222']
    for d in dates:
        (slc_dir / f'{d}.slc.tif').touch()
    out = tmp_path / 'ifg'
    ex = BasicExecutor()
    ex.cfg = {
        'slc2ifg.ifgram_list.mode': 'sequential',
        'slc2ifg.ifgram_list.num_connections': '3',
        'slc2ifg.ifgram_list.exclude_date': '20220129',
    }
    from mintpy.stdproc.utils.slc_input import resolve_slc_input
    state = resolve_slc_input(str(slc_dir))
    all_dates = state.date_list()
    slc_files = {d: state.file_for(None, d) for d in all_dates}
    ex._run_ifgram_list(all_dates, out, slc_files=slc_files)
    pairs = _read_pairs(out / 'ifgram_list.txt')
    flat = [d for p in pairs for d in p]
    assert '20220129' not in flat
    kept = [d for d in dates if d != '20220129']
    assert check_connected(kept, pairs)


# ------------------------------------------------------------------------
# crop file list: only the excluded date is dropped
# (see engine._add_crop_node)
# ------------------------------------------------------------------------
def test_engine_crop_drops_excluded_date(tmp_path):
    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    inp = tmp_path / 'input'
    inp.mkdir()
    dates = ['20220105', '20220117', '20220129', '20220210', '20220222']
    for d in dates:
        dd = inp / d
        dd.mkdir()
        (dd / f't124_264305_iw2_{d}.h5').touch()

    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        'engine.max_workers = 2\n'
        'engine.gpu = false\n'
        'engine.tools = crop_slc,ifgram_list,generate_ifgram\n'
        'slc2ifg.ifgram_list.mode = sequential\n'
        'slc2ifg.ifgram_list.num_connections = 3\n'
        'slc2ifg.ifgram_list.exclude_date = 20220129\n'
    )
    eng = Engine(load_engine_config(str(cfg)))
    eng.plan(dry_run=True)

    lst = tmp_path / 'engine' / 'crop_file_list.txt'
    assert lst.exists()
    lines = [line for line in lst.read_text().splitlines() if line.strip()]
    # 5 date SLCs - 1 excluded = 4 files kept
    assert len(lines) == len(dates) - 1
    names = [Path(line).name for line in lines]
    assert 't124_264305_iw2_20220129.h5' not in names
    assert 't124_264305_iw2_20220105.h5' in names
