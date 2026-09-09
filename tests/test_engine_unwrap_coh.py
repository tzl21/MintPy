"""Tests for external coherence input to unwrap (slc2ifg.unwrap.coh_dir / coh_pattern).

Covers:
  * pair-subdir lookup  coh_dir/{date1}_{date2}/{pattern}
  * flat fallback      coh_dir/{pattern}
  * custom pattern filtering (only matching files are picked)
  * graceful fallback when no external file matches (coh_type stays 'none')
"""

from mintpy.stdproc.engine.chain import resolve_chain
from mintpy.stdproc.engine.config import load_engine_config
from mintpy.stdproc.engine.dag import TaskGraph
from mintpy.stdproc.engine.engine import Engine


def _write_cfg(tmp_path, **opts):
    """Write a minimal user config (no section header, template merged in)."""
    lines = [
        'slc2ifg.work_dir = ' + str(tmp_path / 'work'),
        'slc2ifg.slc_input = ' + str(tmp_path / 'slc'),
    ]
    for k, v in opts.items():
        lines.append(f'slc2ifg.unwrap.{k} = {v}')
    f = tmp_path / 'test.cfg'
    f.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(f)


def _build_unwrap_coh(tmp_path, coh_dir, coh_pattern=None):
    """Build the engine unwrap node and return (coh_input, coh_type)."""
    opts = {'coh_dir': str(coh_dir)}
    if coh_pattern is not None:
        opts['coh_pattern'] = coh_pattern
    cfg_file = _write_cfg(tmp_path, **opts)
    config = load_engine_config(cfg_file)
    engine = Engine(config)
    graph = TaskGraph()
    chain = resolve_chain(None, config.tools)
    engine._add_uniform_nodes(
        graph,
        base_dir=tmp_path / 'work' / 'ifgrams',
        pair_file=None,
        chain=chain,
        entry_variant='fullres',   # mid-chain entry: no upstream node needed
        pairs=[('20240101', '20240113')],
    )
    node = graph.nodes['unwrap#20240101_20240113']
    return node.ctx.inputs['coh'], node.ctx.params['coh_type']


def test_external_coh_pair_subdir(tmp_path):
    """External coherence found under coh_dir/{date1}_{date2}/ (default pattern)."""
    coh_dir = tmp_path / 'ext_coh'
    pair_dir = coh_dir / '20240101_20240113'
    pair_dir.mkdir(parents=True)
    coh_file = pair_dir / 'filt_mli.phsig.coh.tif'
    coh_file.write_bytes(b'coh')

    coh_input, coh_type = _build_unwrap_coh(tmp_path, coh_dir)
    assert coh_input == coh_file
    assert coh_type == 'external'


def test_external_coh_flat_fallback(tmp_path):
    """No pair subdir -> flat fallback coh_dir/{pattern} is used."""
    coh_dir = tmp_path / 'ext_coh'
    coh_dir.mkdir(parents=True)
    coh_file = coh_dir / '20240101_20240113.phsig.coh.tif'
    coh_file.write_bytes(b'coh')

    coh_input, coh_type = _build_unwrap_coh(tmp_path, coh_dir)
    assert coh_input == coh_file
    assert coh_type == 'external'


def test_external_coh_custom_pattern(tmp_path):
    """Only files matching coh_pattern are considered."""
    coh_dir = tmp_path / 'ext_coh'
    pair_dir = coh_dir / '20240101_20240113'
    pair_dir.mkdir(parents=True)
    target = pair_dir / 'filt_mli.phsig.coh.tif'
    target.write_bytes(b'coh')
    other = pair_dir / 'unrelated.txt'
    other.write_bytes(b'nope')

    coh_input, coh_type = _build_unwrap_coh(tmp_path, coh_dir,
                                            coh_pattern='*.phsig.coh.tif')
    assert coh_input == target
    assert coh_type == 'external'

    # wrong pattern -> no match -> fallback to normal resolution ('none')
    coh_input, coh_type = _build_unwrap_coh(tmp_path, coh_dir,
                                            coh_pattern='*_wrong.coh.tif')
    assert coh_input is None
    assert coh_type == 'none'


def test_external_coh_missing_dir_fallback(tmp_path):
    """coh_dir set but empty/missing -> falls back (no crash, coh stays None)."""
    coh_dir = tmp_path / 'no_such_coh'
    coh_input, coh_type = _build_unwrap_coh(tmp_path, coh_dir)
    assert coh_input is None
    assert coh_type == 'none'
