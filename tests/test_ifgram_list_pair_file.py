#!/usr/bin/env python3
"""ifgram_list mode='file': an explicit pair list used verbatim.

Covers the reader (parsing / dedup / errors), the missing-SLC-date guard and
the engine path (`generate_ifgram` for exactly the pairs in the file, without
running any pair planning).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mintpy.stdproc.engine.config import load_engine_config
from mintpy.stdproc.engine.engine import Engine
from mintpy.stdproc.ifgram_list import generate_pairs, read_pair_list


def test_read_pair_list_parsing(tmp_path):
    p = tmp_path / 'pairs.txt'
    p.write_text(
        '# Interferometric pairs\n'
        '\n'
        '20240107-20240119\n'
        '20240119_20240131   # trailing comment\n'
        '20240131 20240212 123.4\n'
        '20240107-20240119\n'          # duplicate
        '20240119-20240107\n')         # reversed duplicate
    assert read_pair_list(str(p)) == [
        ('20240107', '20240119'),
        ('20240119', '20240131'),
        ('20240131', '20240212'),
    ]


def test_read_pair_list_rejects_bad_input(tmp_path):
    for text in ('20240107\n', '20240107-20240107\n', '# only a comment\n'):
        p = tmp_path / 'bad.txt'
        p.write_text(text)
        try:
            read_pair_list(str(p))
        except ValueError as e:
            assert 'bad.txt' in str(e)
        else:
            raise AssertionError(f'expected ValueError for {text!r}')

    try:
        read_pair_list(str(tmp_path / 'missing.txt'))
    except ValueError as e:
        assert 'not found' in str(e)
    else:
        raise AssertionError('expected ValueError for a missing pair file')


def test_generate_file_pairs_requires_pair_file():
    try:
        generate_pairs(['20240107', '20240119'], 'file', None)
    except ValueError as e:
        assert 'pair_file' in str(e)
    else:
        raise AssertionError('expected ValueError without pair_file')


def test_generate_file_pairs_rejects_unknown_date(tmp_path):
    p = tmp_path / 'pairs.txt'
    p.write_text('20240107-20240119\n20210104-20210116\n')
    try:
        generate_pairs(['20240107', '20240119'], 'file', None,
                       select_params={'pair_file': str(p)})
    except ValueError as e:
        assert '20210104' in str(e) and 'no SLC' in str(e)
    else:
        raise AssertionError('expected ValueError for a date without SLC')


def _pair_file_engine(tmp_path, pairs_text, extra=''):
    """SLC dir (flat) + pair file + generate_ifgram-only chain."""
    slc = tmp_path / 'slcs'
    slc.mkdir()
    for d in ('20220105', '20220117', '20220210'):
        (slc / f'{d}.slc.tif').touch()
    pf = tmp_path / 'pairs.txt'
    pf.write_text(pairs_text)
    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        'engine.stages = ifgram_list,generate_ifgram\n'
        f'slc2ifg.ifgram_list.pair_file = {pf}\n'
        'engine.gpu = false\n' + extra)
    return Engine(load_engine_config(str(cfg)))


def test_engine_generate_ifgram_from_pair_file(tmp_path):
    """The engine builds generate_ifgram nodes for exactly the file's pairs."""
    eng = _pair_file_engine(
        tmp_path, '20220105-20220210\n20220117-20220210\n',
        extra='slc2ifg.ifgram_list.mode = file\n')
    g = eng.plan(dry_run=True)
    assert sorted(k for k in g.nodes if k.startswith('generate_ifgram')) == [
        'generate_ifgram#single#20220105_20220210',
        'generate_ifgram#single#20220117_20220210',
    ]
    # the canonical pair list is the verbatim copy
    written = read_pair_list(str(eng.ifgram_dir / 'ifgram_list.txt'))
    assert written == [('20220105', '20220210'), ('20220117', '20220210')]


def test_pair_file_wins_without_mode_file(tmp_path):
    """A set pair_file switches to file mode even if mode=sequential."""
    eng = _pair_file_engine(
        tmp_path, '20220105-20220117\n',
        extra='slc2ifg.ifgram_list.mode = sequential\n'
              'slc2ifg.ifgram_list.num_connections = 1\n')
    g = eng.plan(dry_run=True)
    assert sorted(k for k in g.nodes if k.startswith('generate_ifgram')) == [
        'generate_ifgram#single#20220105_20220117',
    ]


def test_mode_file_without_pair_file_raises(tmp_path):
    slc = tmp_path / 'slcs'
    slc.mkdir()
    (slc / '20220105.slc.tif').touch()
    (slc / '20220117.slc.tif').touch()
    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        'engine.stages = ifgram_list,generate_ifgram\n'
        'slc2ifg.ifgram_list.mode = file\n'
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))
    try:
        eng.plan(dry_run=True)
    except ValueError as e:
        assert 'pair_file' in str(e)
    else:
        raise AssertionError('expected ValueError for mode=file without pair_file')


def test_entry_mode_honours_pair_file(tmp_path):
    """Entry mode (no generate_ifgram): the pair file still selects the pairs.

    Regression guard: without this, the product tree always won and the
    explicit pair list was silently ignored in mid-chain entry mode.
    """
    slc = tmp_path / 'slcs'
    for d in ('20220105', '20220117', '20220210'):
        (slc / d).mkdir(parents=True)
        (slc / d / f't124_264305_iw2_{d}.h5').touch()
    inp = tmp_path / 'products'
    inp.mkdir()
    (inp / 'ifgram_list.txt').write_text(
        '20220105-20220117\n20220105-20220210\n20220117-20220210\n')
    pf = tmp_path / 'pairs.txt'
    pf.write_text('20220105-20220210\n')
    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        'slc2ifg.slc_pattern = **/t*.h5\n'
        f'engine.input_dir = {inp}\n'
        f'slc2ifg.ifgram_list.pair_file = {pf}\n'
        'engine.stages = complex_coh\n'      # no generate_ifgram -> entry mode
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == ['complex_coh#20220105_20220210']
