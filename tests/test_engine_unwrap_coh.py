"""Tests for automatic coherence discovery for unwrap weighting.

Coherence rasters are auto-discovered under ``<work_dir>/ifgrams`` as
``{date1}_{date2}/*.coh.tif`` (kind/variant inferred from the filename);
when none exists the unwrap node runs with uniform weights (``coh_type``
``none``).
"""

from mintpy.stdproc.engine.chain import resolve_chain
from mintpy.stdproc.engine.config import load_engine_config
from mintpy.stdproc.engine.dag import TaskGraph
from mintpy.stdproc.engine.engine import Engine


def _write_cfg(tmp_path):
    """Write a minimal user config (no section header, template merged in).

    The chain is just ``unwrap`` so the unwrapped variant stays ``fullres``
    (a multilook/filter stage would make it ``filt_mli``, where complex
    coherence no longer applies).
    """
    f = tmp_path / 'test.cfg'
    f.write_text(
        'slc2ifg.work_dir = ' + str(tmp_path / 'work') + '\n'
        'slc2ifg.slc_input = ' + str(tmp_path / 'slc') + '\n'
        'engine.stages = unwrap\n',
        encoding='utf-8')
    return str(f)


def _build_unwrap_coh(tmp_path):
    """Build the engine unwrap node and return (coh_input, coh_type)."""
    config = load_engine_config(_write_cfg(tmp_path))
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


def _put_coh(tmp_path, name):
    pair_dir = tmp_path / 'work' / 'ifgrams' / '20240101_20240113'
    pair_dir.mkdir(parents=True, exist_ok=True)
    coh_file = pair_dir / name
    coh_file.write_bytes(b'coh')
    return coh_file


def test_auto_coh_phsig(tmp_path):
    """A {variant}.phsig.coh.tif under <work_dir>/ifgrams is picked up."""
    coh_file = _put_coh(tmp_path, 'filt_mli.phsig.coh.tif')
    coh_input, coh_type = _build_unwrap_coh(tmp_path)
    assert coh_input == coh_file
    assert coh_type == 'phsig'


def test_auto_coh_variant_inferred(tmp_path):
    """Kind/variant are inferred from the filename (mli.phsig here)."""
    coh_file = _put_coh(tmp_path, 'mli.phsig.coh.tif')
    coh_input, coh_type = _build_unwrap_coh(tmp_path)
    assert coh_input == coh_file
    assert coh_type == 'phsig'


def test_auto_coh_complex_preferred_for_fullres(tmp_path):
    """At fullres, a complex coherence raster is used as 'complex'."""
    coh_file = _put_coh(tmp_path, 'fullres.cpx.coh.tif')
    coh_input, coh_type = _build_unwrap_coh(tmp_path)
    assert coh_input == coh_file
    assert coh_type == 'complex'


def test_auto_coh_complex_wins_at_fullres(tmp_path):
    """At fullres with both kinds present, complex coherence takes priority."""
    cpx = _put_coh(tmp_path, 'fullres.cpx.coh.tif')
    _put_coh(tmp_path, 'fullres.phsig.coh.tif')
    coh_input, coh_type = _build_unwrap_coh(tmp_path)
    assert coh_input == cpx
    assert coh_type == 'complex'


def test_auto_coh_missing(tmp_path):
    """No coherence raster -> uniform weights (coh_type 'none')."""
    coh_input, coh_type = _build_unwrap_coh(tmp_path)
    assert coh_input is None
    assert coh_type == 'none'
