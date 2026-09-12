#!/usr/bin/env python3
"""Pure-Python unit tests for the processing chain (Plan B, first cut).

No osgeo/numpy required — run locally with any Python 3.10+:
    python -m pytest tests/test_chain.py -v
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mintpy.stdproc.engine.chain import DEFAULT_CHAIN, format_chain, resolve_chain
from mintpy.stdproc.engine.config import AUTO_TOOLS, load_engine_config


def test_default_chain_matches_auto_tools():
    """Default chain + auto tools gating == AUTO_TOOLS (explicit list only)."""
    chain = resolve_chain(None, list(AUTO_TOOLS))
    names = [s.name for s in chain]
    assert names == ['ifgram_list', 'generate_ifgram', 'stitch',
                     'multilook', 'filter', 'unwrap']
    # Nothing is force-enabled: phsig_coh / complex_coh / crop_slc /
    # atmosphere are NOT part of the auto default.
    assert 'phsig_coh' not in names
    assert 'complex_coh' not in names
    assert 'crop_slc' not in names
    assert 'atmosphere' not in names


def test_default_chain_with_explicit_tools():
    """Explicit tools (e.g. stitch config, no unwrap) gate consistently with the old behavior."""
    tools = ['ifgram_list', 'generate_ifgram', 'complex_coh', 'stitch',
             'multilook', 'filter', 'phsig_coh']
    chain = resolve_chain(None, tools)
    names = [s.name for s in chain]
    assert 'unwrap' not in names
    assert 'complex_coh' in names


def test_explicit_stages_enable_optional_stage():
    """engine.stages explicitly listing atmosphere enables it (even if tools=auto omits it)."""
    stages = ['generate_ifgram', 'multilook', 'filter', 'phsig_coh',
              'unwrap', 'atmosphere']
    chain = resolve_chain(stages, list(AUTO_TOOLS))
    assert [s.name for s in chain] == stages
    assert any(s.name == 'atmosphere' for s in chain)


def test_unwrap_no_static_phsig_requirement():
    """unwrap no longer force-requires phsig_coh at the chain level — the
    coherence input is resolved at graph build via slc2ifg.unwrap.coh_type."""
    chain = resolve_chain(['generate_ifgram', 'unwrap'], list(AUTO_TOOLS))
    assert [s.name for s in chain] == ['generate_ifgram', 'unwrap']


def test_atmosphere_requires_unwrap_still_enforced():
    """Static requires stays for atmosphere -> unwrap."""
    try:
        resolve_chain(['generate_ifgram', 'atmosphere'], list(AUTO_TOOLS))
    except ValueError as e:
        assert 'requires' in str(e) and 'unwrap' in str(e)
    else:
        raise AssertionError("expected ValueError for missing unwrap")


def test_unknown_stage_rejected():
    try:
        resolve_chain(['generate_ifgram', 'not_a_stage'], [])
    except ValueError as e:
        assert 'Unknown stage' in str(e)
    else:
        raise AssertionError("expected ValueError for unknown stage")


def test_format_chain_smoke():
    chain = resolve_chain(None, list(AUTO_TOOLS))
    text = format_chain(chain)
    assert 'Processing chain:' in text
    assert 'unwrap' in text


def test_config_stages_parsing():
    """engine.stages is parsed from a config file."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'input').mkdir()
        cfg = tmp / 'test.cfg'
        cfg.write_text(
            'slc2ifg.work_dir = {work}\n'
            'slc2ifg.slc_input = {work}/input\n'
            'slc2ifg.processor = isce3\n'
            'engine.stages = generate_ifgram,multilook,filter,phsig_coh,unwrap,atmosphere\n'
            'engine.tools = auto\n'.format(work=tmp))
        config = load_engine_config(str(cfg))
        assert config.stages == ['generate_ifgram', 'multilook', 'filter',
                                 'phsig_coh', 'unwrap', 'atmosphere']
        chain = resolve_chain(config.stages, config.tools)
        assert [s.name for s in chain] == config.stages


def test_config_stages_auto_is_none():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'input').mkdir()
        cfg = tmp / 'test.cfg'
        cfg.write_text(
            'slc2ifg.work_dir = {work}\n'
            'slc2ifg.slc_input = {work}/input\n'
            'slc2ifg.processor = isce3\n'.format(work=tmp))
        config = load_engine_config(str(cfg))
        assert config.stages is None


def test_default_chain_has_core_and_optional_mix():
    names = [s.name for s in DEFAULT_CHAIN]
    assert names == ['crop_slc', 'ifgram_list', 'generate_ifgram',
                     'complex_coh', 'stitch', 'multilook', 'filter',
                     'phsig_coh', 'unwrap', 'atmosphere']
    core = {s.name for s in DEFAULT_CHAIN if s.core}
    # phsig_coh is no longer core (nothing is force-enabled anymore)
    assert core == {'ifgram_list', 'generate_ifgram'}
    optional = {s.name for s in DEFAULT_CHAIN if s.optional}
    assert optional == {'crop_slc', 'complex_coh', 'atmosphere'}


# ------------------------------------------------------------------------
# Engine-level (no osgeo needed: slc2ifg package init is lazy now)
# ------------------------------------------------------------------------
def _mini_engine(stages_cfg: str = '', extra: str = '', tools: str = 'auto'):
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    inp = tmp / 'input'
    inp.mkdir()
    for d in ('20220105', '20220117'):
        (inp / f'{d}.slc.tif').touch()
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        'engine.max_workers = 4\n'
        f'engine.tools = {tools}\n'
        'engine.gpu = false\n'
        f'{stages_cfg}\n'
        f'{extra}\n')
    return Engine(load_engine_config(str(cfg)))


def _entry_engine(tools: str = 'phsig_coh,unwrap',
                  variant: str = 'filt_mli',
                  extra: str = ''):
    """Engine configured for mid-chain entry mode (input = existing products).

    Builds an input tree with ``{d1}_{d2}/`` subdirectories; the engine reads
    pairs from those directories (no ifgram_list.txt).
    """
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    inp = tmp / 'products'
    for d1, d2 in (('20220105', '20220117'), ('20220117', '20220210')):
        (inp / f'{d1}_{d2}').mkdir(parents=True, exist_ok=True)
    (tmp / 'input').mkdir(exist_ok=True)
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {tmp}/input\n'
        'slc2ifg.processor = isce3\n'
        'engine.max_workers = 4\n'
        f'engine.tools = {tools}\n'
        f'engine.input_dir = {inp}\n'
        f'engine.input_variant = {variant}\n'
        'engine.gpu = false\n'
        f'{extra}\n')
    return Engine(load_engine_config(str(cfg)))


def test_engine_build_graph_mini():
    """Default chain (auto): phsig_coh is NOT computed; unwrap runs with no
    coherence (weight 1) since no coherence stage is enabled."""
    eng = _mini_engine()
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'filter#20220105_20220117',
        'generate_ifgram#single#20220105_20220117',
        'multilook#20220105_20220117',
        'unwrap#20220105_20220117',
    ]
    # unwrap without any coherence stage -> coh input None (SNAPHU weight 1)
    assert g.nodes['unwrap#20220105_20220117'].ctx.inputs['coh'] is None
    # chain version written to manifest
    assert eng._manifest._meta['chain'] == [
        'ifgram_list', 'generate_ifgram', 'stitch', 'multilook',
        'filter', 'unwrap']


def test_engine_build_graph_atmosphere():
    """Explicit stages insert atmosphere: after unwrap, producing filt_mli.atm.unw.tif."""
    eng = _mini_engine(
        'engine.stages = generate_ifgram,multilook,filter,phsig_coh,unwrap,atmosphere')
    g = eng.plan(dry_run=True)
    atm = g.nodes['atmosphere#20220105_20220117']
    assert atm.ctx.outputs['unw'].name == 'filt_mli.atm.unw.tif'
    deps = g.nodes['atmosphere#20220105_20220117'].deps
    assert 'unwrap#20220105_20220117' in deps


def test_engine_tool_params():
    """params_spec data-driven: values/types/special cases (ps_nlks, ntiles, algorithm) correct."""
    eng = _mini_engine(extra='engine.tile_size = 512\n'
                             'slc2ifg.multilook.lks_y = 4\n'
                             'slc2ifg.multilook.lks_x = 8\n'
                             'slc2ifg.filter.goldstein.alpha = 0.5\n'
                             'slc2ifg.unwrap.snaphu.nproc = 2\n')
    p = eng._tool_params('multilook')
    assert p['lks_y'] == 4 and p['lks_x'] == 8 and p['method'] == 'mean'
    p = eng._tool_params('filter')
    assert p['alpha'] == 0.5 and p['psize'] == 32 and p['use_gpu'] is False
    p = eng._tool_params('phsig_coh')
    assert p['ps_nlks'] == 32.0 and p['window_size'] == 5
    p = eng._tool_params('unwrap')
    assert p['nlooks'] == 32.0 and p['ntiles'] == (1, 1)
    assert p['coh_type'] == 'auto'
    assert p['nproc'] == 2 and p['algorithm'] == 'snaphu'
    p = eng._tool_params('atmosphere')
    assert p['method'] == 'none'


def test_gpu_tools_declared():
    """GPU tool set derived from Resource.device declarations (replacing the old hard-coded set)."""
    from mintpy.stdproc.engine.tool import gpu_tools
    assert gpu_tools() == ['complex_coh', 'filter', 'phsig_coh']


def test_all_source_modules_compile():
    """Every .py under src/ must at least compile (catches e.g. a def whose
    defaulted parameter precedes a non-defaulted one — a SyntaxError that only
    surfaces when the module is imported)."""
    import py_compile
    import tempfile

    src = Path(__file__).resolve().parents[1] / 'src'
    tmp = tempfile.TemporaryDirectory()
    bad = []
    for p in sorted(src.rglob('*.py')):
        try:
            py_compile.compile(str(p), cfile=str(Path(tmp.name) / 'x.pyc'),
                               doraise=True)
        except Exception as e:  # noqa: BLE001 - collect all failures
            bad.append(f"{p.relative_to(src)}: {e}")
    tmp.cleanup()
    assert not bad, "compile failures:\n" + '\n'.join(bad)


# ------------------------------------------------------------------------
# Backend-scoped config naming (slc2ifg.<stage>.<backend>.<param>)
# ------------------------------------------------------------------------
def test_unwrap_backend_scoped_keys():
    """snaphu-specific params live in slc2ifg.unwrap.snaphu.*; algorithm stays stage-level."""
    eng = _mini_engine(extra='slc2ifg.unwrap.snaphu.cost_mode = defo\n'
                             'slc2ifg.unwrap.snaphu.nproc = 4\n'
                             'slc2ifg.unwrap.snaphu.nlooks = 16\n'
                             'slc2ifg.unwrap.algorithm = snaphu\n')
    p = eng._tool_params('unwrap')
    assert p['cost_mode'] == 'defo'
    assert p['nproc'] == 4
    assert p['nlooks'] == 16.0
    assert p['algorithm'] == 'snaphu'
    # a parallel backend namespace must not leak into snaphu params
    assert p['init_method'] == 'mcf'


def test_unwrap_legacy_keys_fallback():
    """Legacy flat keys (slc2ifg.unwrap.cost_mode, ...) still work with a warning."""
    eng = _mini_engine(extra='slc2ifg.unwrap.cost_mode = defo\n'
                             'slc2ifg.unwrap.nproc = 3\n'
                             'slc2ifg.unwrap.ntiles_row = 2\n')
    p = eng._tool_params('unwrap')
    assert p['cost_mode'] == 'defo'   # legacy fallback
    assert p['nproc'] == 3
    assert p['ntiles'] == (2, 1)


def test_unwrap_new_key_wins_over_legacy():
    """When both the new and the legacy key are set, the new one wins."""
    eng = _mini_engine(extra='slc2ifg.unwrap.snaphu.cost_mode = smooth\n'
                             'slc2ifg.unwrap.cost_mode = defo\n')
    assert eng._tool_params('unwrap')['cost_mode'] == 'smooth'


def test_unwrap_explicit_false_bool_honored():
    """An explicit false for a bool param is honored (not clobbered by the 0->default rule)."""
    eng = _mini_engine(extra='slc2ifg.unwrap.snaphu.keep_scratch = true\n')
    assert eng._tool_params('unwrap')['keep_scratch'] is True
    eng2 = _mini_engine(extra='slc2ifg.unwrap.snaphu.keep_scratch = false\n')
    assert eng2._tool_params('unwrap')['keep_scratch'] is False


def test_filter_backend_scoped_keys():
    """goldstein/long_wavelength params live in their own namespaces; new keys win."""
    eng = _mini_engine(extra='slc2ifg.filter.filter_type = goldstein\n'
                             'slc2ifg.filter.goldstein.alpha = 0.5\n'
                             'slc2ifg.filter.goldstein.psize = 16\n')
    p = eng._tool_params('filter')
    assert p['filter_type'] == 'goldstein'
    assert p['alpha'] == 0.5 and p['psize'] == 16
    # long_wavelength namespace untouched
    assert p['wavelength_cutoff'] == 25000

    eng2 = _mini_engine(extra='slc2ifg.filter.goldstein.alpha = 0.6\n'
                              'slc2ifg.filter.alpha = 0.3\n')
    assert eng2._tool_params('filter')['alpha'] == 0.6  # new key wins

    eng3 = _mini_engine(extra='slc2ifg.filter.alpha = 0.3\n')
    assert eng3._tool_params('filter')['alpha'] == 0.3  # legacy fallback


def test_unwrap_mask_file_param():
    """slc2ifg.unwrap.snaphu.mask_file flows into unwrap params (new + legacy key)."""
    eng = _mini_engine(extra='slc2ifg.unwrap.snaphu.mask_file = /tmp/mask_a.tif\n')
    assert eng._tool_params('unwrap')['mask_file'] == '/tmp/mask_a.tif'
    eng2 = _mini_engine(extra='slc2ifg.unwrap.mask_file = /tmp/mask_b.tif\n')
    assert eng2._tool_params('unwrap')['mask_file'] == '/tmp/mask_b.tif'


def test_snaphu_sanitize_nonfinite():
    """SNAPHU-only NaN/Inf sanitization: zeroed + mask generated, user mask combined."""
    try:
        from mintpy.stdproc.unwrap_ifgram import _sanitize_nonfinite
    except Exception:
        return  # osgeo unavailable locally — exercised on the server
    import numpy as np

    ifg = np.array([1 + 2j, np.nan + 1j, 3 + np.inf * 1j, 0 + 0j])
    corr = np.array([0.9, np.nan, 0.5, 0.0], dtype=np.float32)
    ifg2, corr2, mask = _sanitize_nonfinite(ifg, corr, None)
    assert not np.isnan(ifg2).any() and not np.isinf(ifg2).any()
    assert np.allclose(ifg2, [1 + 2j, 0j, 0j, 0 + 0j])
    # corr is zeroed wherever the phase OR corr is non-finite
    assert np.allclose(corr2, [0.9, 0.0, 0.0, 0.0])
    assert mask.tolist() == [1, 0, 0, 1]       # NaN/Inf excluded, zeros kept
    # all-finite input -> no mask
    _, _, m2 = _sanitize_nonfinite(np.array([1 + 1j]), np.array([0.5]), None)
    assert m2 is None
    # user-provided mask is combined (0 where non-finite)
    user = np.array([1, 1, 1, 1], dtype=np.uint8)
    _, _, m3 = _sanitize_nonfinite(ifg, corr, user)
    assert m3.tolist() == [1, 0, 0, 1]


def test_cleanup_keep_policy_stage_names():
    """keep_intermediates lists tool names; products of listed nodes are kept
    (node keys are 'tool#...' — matching must use the tool part)."""
    import tempfile

    from mintpy.stdproc.engine.manifest import Manifest, plan_cleanup

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        m = Manifest(tmp / 'm.json')   # one shared instance (as in the engine)
        # produce fake products per node
        def mk(node, name):
            p = tmp / name
            p.write_text('x')
            m.record(node, [p])
            return p

        ifg = mk('generate_ifgram#single#20220105_20220117', 'fullres.int.tif')
        ml = mk('multilook#20220105_20220117', 'mli.int.tif')
        ph = mk('phsig_coh#20220105_20220117', 'filt_mli.phsig.coh.tif')
        unw = mk('unwrap#20220105_20220117', 'filt_mli.unw.tif')

        # keep generate_ifgram + phsig products -> multilook output deleted,
        # final products always kept
        to_delete = plan_cleanup(m, keep_policy='generate_ifgram,phsig_coh')
        assert ifg not in to_delete, 'listed stage product must be kept'
        assert ph not in to_delete
        assert unw not in to_delete      # final product always kept
        assert ml in to_delete, 'unlisted stage product must be deleted'

        # default 'none': everything intermediate is deleted
        to_delete2 = plan_cleanup(m, keep_policy='none')
        assert ifg in to_delete2 and ml in to_delete2
        assert unw not in to_delete2


def test_cleanup_keeps_pair_dir_containing_final_products():
    """A recorded DIRECTORY (e.g. the generate_ifgram pair_dir) that
    physically contains a kept final product (unw/conncomp/phsig) must not be
    rmtree'd by the default cleanup — otherwise the final products are
    destroyed along with the intermediates (regression)."""
    import tempfile

    from mintpy.stdproc.engine.manifest import (
        Manifest,
        execute_cleanup,
        plan_cleanup,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        m = Manifest(tmp / 'm.json')
        pair_dir = tmp / '20220105_20220117'
        pair_dir.mkdir()
        ifg = pair_dir / 'fullres.int.tif'
        ifg.write_text('x')
        unw = pair_dir / 'fullres.unw.tif'
        unw.write_text('x')
        # record the pair DIRECTORY (as GenerateIfgramTool returns pair_dir)
        m.record('generate_ifgram#single#20220105_20220117', [ifg, pair_dir])
        m.record('unwrap#20220105_20220117', [unw])

        to_delete = plan_cleanup(m, keep_policy='none')
        assert unw not in to_delete          # final product always kept
        assert ifg in to_delete              # intermediate file deleted
        assert pair_dir not in to_delete, \
            'pair dir containing the kept unw must survive cleanup'
        # executing the plan must not delete the final product
        execute_cleanup(to_delete)
        assert unw.exists()
        assert not ifg.exists()


def test_execute_cleanup_deletes_directories():
    """Cleanup must remove directory products (e.g. crop_slc output dir)
    recursively — unlink alone fails with 'Is a directory'."""
    import tempfile

    from mintpy.stdproc.engine.manifest import execute_cleanup

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        d = tmp / 'cropped_slc'
        d.mkdir()
        (d / '20220105.slc.tif').touch()
        (d / '20220117.slc.tif').touch()
        f = tmp / 'mid.int.tif'
        f.touch()
        n = execute_cleanup([d, f])
        assert n == 2
        assert not d.exists() and not f.exists()


def test_output_ready_rejects_empty_and_corrupt_files():
    """skip_if_exists must not treat zero-byte or truncated-but-non-empty
    (interrupted-run) outputs as valid: raster outputs must also be GDAL-
    openable with non-trivial dimensions."""
    import tempfile

    from mintpy.stdproc.engine.tool import Tool, ToolContext

    class _DummyTool(Tool):
        name = 'dummy'
        def run(self, ctx):  # pragma: no cover - not executed in this test
            return {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        good = tmp / 'good.txt'
        good.write_bytes(b'x')          # non-empty, non-raster
        corrupt = tmp / 'bad.tif'
        corrupt.write_bytes(b'x')       # non-empty but NOT a valid raster
        empty = tmp / 'empty.tif'
        empty.write_bytes(b'')          # zero-byte — interrupted write
        missing = tmp / 'missing.tif'
        d = tmp / 'outdir'
        d.mkdir()                       # dir output: existence is enough

        assert Tool.output_ready([good]) is True
        assert Tool.output_ready([good, d]) is True
        assert Tool.output_ready([corrupt]) is False,             'non-empty but unreadable raster must NOT be ready'
        assert Tool.output_ready([good, empty]) is False
        assert Tool.output_ready([good, missing]) is False
        assert Tool.output_ready([empty]) is False

        # skip_if_exists skips only when every output is ready
        ctx = ToolContext(tool_name='t', inputs={}, outputs={'f': good},
                          params={}, work_dir=tmp)
        assert _DummyTool().skip_if_exists(ctx) is not None
        ctx2 = ToolContext(tool_name='t', inputs={}, outputs={'f': empty},
                           params={}, work_dir=tmp)
        assert _DummyTool().skip_if_exists(ctx2) is None
        ctx3 = ToolContext(tool_name='t', inputs={}, outputs={'f': corrupt},
                           params={}, work_dir=tmp)
        assert _DummyTool().skip_if_exists(ctx3) is None


# ------------------------------------------------------------------------
# Config consolidation: engine.max_workers is the single knob
# ------------------------------------------------------------------------
def test_config_max_workers_single_source():
    """engine.max_workers is honored; the legacy slc2ifg.max_workers is ignored."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'input').mkdir()
        cfg = tmp / 'test.cfg'
        cfg.write_text(
            'slc2ifg.work_dir = {work}\n'
            'slc2ifg.slc_input = {work}/input\n'
            'slc2ifg.processor = isce3\n'
            'slc2ifg.max_workers = 8\n'
            'engine.max_workers = 32\n'.format(work=tmp))
        config = load_engine_config(str(cfg))
        assert config.max_workers == 32

        # without engine.max_workers, the legacy key no longer leaks through
        cfg2 = tmp / 'test2.cfg'
        cfg2.write_text(
            'slc2ifg.work_dir = {work}\n'
            'slc2ifg.slc_input = {work}/input\n'
            'slc2ifg.processor = isce3\n'
            'slc2ifg.max_workers = 8\n'.format(work=tmp))
        config2 = load_engine_config(str(cfg2))
        assert config2.max_workers is None  # falls back to n_cpu at runtime


def test_config_scheduler_auto_is_threaded():
    """engine.scheduler = auto maps to 'threaded'."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'input').mkdir()
        for sch, expect in (('auto', 'threaded'), ('threaded', 'threaded'),
                            ('distributed', 'distributed')):
            cfg = tmp / f'{sch}.cfg'
            cfg.write_text(
                'slc2ifg.work_dir = {work}\n'
                'slc2ifg.slc_input = {work}/input\n'
                'slc2ifg.processor = isce3\n'
                f'engine.scheduler = {sch}\n'.format(work=tmp))
            assert load_engine_config(str(cfg)).scheduler == expect


# ------------------------------------------------------------------------
# Mid-chain entry mode (start from an arbitrary stage's products)
# ------------------------------------------------------------------------
def test_engine_entry_mode_tools_based():
    """engine.tools without generate_ifgram + input_variant: build only the
    requested stages, consuming existing products in the canonical layout."""
    eng = _entry_engine(tools='phsig_coh,unwrap', variant='filt_mli')
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'phsig_coh#20220105_20220117',
        'phsig_coh#20220117_20220210',
        'unwrap#20220105_20220117',
        'unwrap#20220117_20220210',
    ]
    # entry stage consumes the configured variant from the input root
    ph = g.nodes['phsig_coh#20220105_20220117']
    assert Path(ph.ctx.inputs['ifg']).name == 'filt_mli.int.tif'
    assert ph.deps == []                      # no upstream node
    assert Path(ph.ctx.outputs['coh']).name == 'filt_mli.phsig.coh.tif'
    # unwrap derives from the phsig node and writes into the canonical tree
    uw = g.nodes['unwrap#20220105_20220117']
    assert uw.deps == ['phsig_coh#20220105_20220117']
    assert Path(uw.ctx.inputs['ifg']).name == 'filt_mli.int.tif'
    assert Path(uw.ctx.outputs['unw']).name == 'filt_mli.unw.tif'


def test_engine_entry_mode_fullres_input():
    """Entry at filter with fullres input: variant chain fullres->filt->coh/unw."""
    eng = _entry_engine(tools='filter,phsig_coh,unwrap', variant='fullres')
    g = eng.plan(dry_run=True)
    f = g.nodes['filter#20220105_20220117']
    assert Path(f.ctx.inputs['ifg']).name == 'fullres.int.tif'
    assert Path(f.ctx.outputs['ifg']).name == 'filt.int.tif'
    ph = g.nodes['phsig_coh#20220105_20220117']
    assert Path(ph.ctx.inputs['ifg']).name == 'filt.int.tif'
    assert Path(ph.ctx.outputs['coh']).name == 'filt.phsig.coh.tif'
    uw = g.nodes['unwrap#20220105_20220117']
    assert Path(uw.ctx.inputs['ifg']).name == 'filt.int.tif'
    assert Path(uw.ctx.outputs['unw']).name == 'filt.unw.tif'


def test_engine_entry_mode_via_stages():
    """Explicit engine.stages without generate_ifgram also enters mid-chain mode."""
    eng = _entry_engine(extra='engine.stages = phsig_coh,unwrap')
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'phsig_coh#20220105_20220117',
        'phsig_coh#20220117_20220210',
        'unwrap#20220105_20220117',
        'unwrap#20220117_20220210',
    ]


def test_engine_entry_mode_invalid_variant():
    try:
        _entry_engine(variant='not_a_variant').plan(dry_run=True)
    except ValueError as e:
        assert 'input_variant' in str(e)
    else:
        raise AssertionError("expected ValueError for invalid input_variant")


def test_engine_entry_mode_crop_rejected():
    try:
        _entry_engine(tools='crop_slc,phsig_coh,unwrap').plan(dry_run=True)
    except ValueError as e:
        assert 'crop_slc' in str(e)
    else:
        raise AssertionError("expected ValueError when crop_slc is used in entry mode")


def test_engine_entry_mode_pairs_from_pairfile():
    """When the input root has an ifgram_list.txt, pairs come from it."""
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    inp = tmp / 'products'
    inp.mkdir()
    (inp / 'ifgram_list.txt').write_text('20220105-20220117\n20220210-20220306\n')
    (tmp / 'input').mkdir(exist_ok=True)
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {tmp}/input\n'
        'slc2ifg.processor = isce3\n'
        f'engine.tools = unwrap\n'
        f'engine.input_dir = {inp}\n'
        'engine.input_variant = filt_mli\n'
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'unwrap#20220105_20220117',
        'unwrap#20220210_20220306',
    ]
    # no coherence stage enabled -> coh input None (SNAPHU weight = 1)
    assert g.nodes['unwrap#20220105_20220117'].ctx.inputs['coh'] is None


# ------------------------------------------------------------------------
# Unwrap coherence input (slc2ifg.unwrap.coh_type: auto|complex|phsig|none)
# ------------------------------------------------------------------------
def test_unwrap_auto_uses_complex_coh():
    """fullres + complex_coh + unwrap: unwrap consumes the complex coherence
    (fullres.cpx.coh.tif) and depends on the complex_coh node."""
    eng = _mini_engine(tools='ifgram_list,generate_ifgram,complex_coh,unwrap')
    g = eng.plan(dry_run=True)
    uw = g.nodes['unwrap#20220105_20220117']
    assert uw.ctx.inputs['coh'].name == 'fullres.cpx.coh.tif'
    assert 'complex_coh#single#20220105_20220117' in uw.deps
    assert 'phsig_coh' not in g.nodes
    # unwrap has both its ifg source and the coherence node as deps
    assert 'generate_ifgram#single#20220105_20220117' in uw.deps


def test_unwrap_explicit_phsig_missing_errors():
    """coh_type=phsig without the phsig_coh stage -> error naming the stage."""
    eng = _mini_engine(tools='ifgram_list,generate_ifgram,unwrap',
                       extra='slc2ifg.unwrap.coh_type = phsig\n')
    try:
        eng.plan(dry_run=True)
    except ValueError as e:
        assert "'phsig_coh'" in str(e)
    else:
        raise AssertionError("expected ValueError for missing phsig_coh")


def test_unwrap_explicit_complex_missing_errors():
    """coh_type=complex without the complex_coh stage -> error naming it."""
    eng = _mini_engine(tools='ifgram_list,generate_ifgram,unwrap',
                       extra='slc2ifg.unwrap.coh_type = complex\n')
    try:
        eng.plan(dry_run=True)
    except ValueError as e:
        assert "'complex_coh'" in str(e)
    else:
        raise AssertionError("expected ValueError for missing complex_coh")


def test_unwrap_explicit_complex_fullres_only():
    """complex coherence is fullres-only — unwrapping a looked variant errors."""
    eng = _mini_engine(tools='ifgram_list,generate_ifgram,complex_coh,multilook,unwrap',
                       extra='slc2ifg.unwrap.coh_type = complex\n')
    try:
        eng.plan(dry_run=True)
    except ValueError as e:
        assert 'fullres-only' in str(e)
    else:
        raise AssertionError("expected ValueError: complex coherence is fullres-only")


def test_unwrap_invalid_coh_type_errors():
    eng = _mini_engine(tools='ifgram_list,generate_ifgram,unwrap',
                       extra='slc2ifg.unwrap.coh_type = bananas\n')
    try:
        eng.plan(dry_run=True)
    except ValueError as e:
        assert 'auto|complex|phsig|none' in str(e)
    else:
        raise AssertionError("expected ValueError for invalid coh_type")


def test_complex_coh_rejected_in_entry_mode():
    """complex_coh needs SLC inputs — rejected in mid-chain entry mode."""
    eng = _entry_engine(tools='complex_coh,unwrap')
    try:
        eng.plan(dry_run=True)
    except ValueError as e:
        assert 'complex_coh' in str(e) and 'SLC' in str(e)
    else:
        raise AssertionError("expected ValueError for complex_coh in entry mode")


def _entry_engine_with_slcs(tools: str = 'complex_coh,unwrap'):
    """Mid-chain entry engine whose slc_input is a flat dir with SLC files."""
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    inp = tmp / 'products'
    inp.mkdir()
    (inp / 'ifgram_list.txt').write_text('20220105-20220117\n20220210-20220306\n')
    slc = tmp / 'slcs'
    slc.mkdir()
    for d in ('20220105', '20220117', '20220210', '20220306'):
        (slc / f'{d}.slc.tif').touch()
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        f'engine.tools = {tools}\n'
        f'engine.input_dir = {inp}\n'
        'engine.input_variant = fullres\n'
        'engine.gpu = false\n')
    return Engine(load_engine_config(str(cfg)))


def test_entry_complex_coh_from_slcs():
    """Entry mode + SLCs: complex_coh computed without generate_ifgram."""
    eng = _entry_engine_with_slcs(tools='complex_coh')
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'complex_coh#20220105_20220117',
        'complex_coh#20220210_20220306',
    ]
    n = g.nodes['complex_coh#20220105_20220117']
    assert n.ctx.outputs['coh'].name == 'fullres.cpx.coh.tif'
    assert n.ctx.inputs['slc_dir'] == Path(eng.slc_input)


def test_entry_complex_coh_slc_pairs_fallback():
    """complex_coh-only entry with no product tree: pairs from the SLC dates.

    Regression: an operator CSLC layout (``<date>/tXXX_..._yyyymmdd.h5``)
    plus a configured ``slc2ifg.slc_pattern`` must pass the entry-mode SLC
    check and yield the SLC-derived pairs instead of failing with
    "no date pairs found under input_dir".
    """
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    slc = tmp / 'slcs'
    for d in ('20220105', '20220117', '20220210'):
        (slc / d).mkdir(parents=True)
        (slc / d / f't124_264305_iw2_{d}.h5').touch()
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        'slc2ifg.slc_pattern = **/t*.h5\n'
        'engine.stages = ifgram_list,complex_coh\n'   # no generate_ifgram -> entry
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))

    assert eng._entry_slcs_available()          # honours the configured pattern
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'complex_coh#20220105_20220117',
        'complex_coh#20220105_20220210',
        'complex_coh#20220117_20220210',
    ]


def test_entry_ifgram_list_replans_pairs():
    """'ifgram_list' in engine.stages re-plans the pairs from the SLCs.

    Regression: a stale pair list left in the product tree (e.g. a complete
    graph from an earlier run) used to win in entry mode, so ifgram_list.*
    settings (mode / select.* / date filters) had no effect at all.
    """
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    slc = tmp / 'slcs'
    for d in ('20220105', '20220117', '20220210'):
        (slc / d).mkdir(parents=True)
        (slc / d / f't124_264305_iw2_{d}.h5').touch()
    inp = tmp / 'products'
    inp.mkdir()
    (inp / 'ifgram_list.txt').write_text(
        '# Interferometric pairs generated by ifgram_list.py\n'
        '# Date12\n'
        '    20190101-20190113\n')
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        'slc2ifg.slc_pattern = **/t*.h5\n'
        f'engine.input_dir = {inp}\n'
        f'slc2ifg.generate_ifgram.output_dir = {inp}\n'
        'engine.stages = ifgram_list,complex_coh\n'   # explicit pair planning
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))
    g = eng.plan(dry_run=True)

    assert sorted(g.nodes) == [
        'complex_coh#20220105_20220117',
        'complex_coh#20220105_20220210',
        'complex_coh#20220117_20220210',
    ]
    # the stale pair list was re-planned in place
    txt = (inp / 'ifgram_list.txt').read_text()
    assert '20190101-20190113' not in txt
    assert '20220105-20220117' in txt


def test_entry_select_forwards_slc_pattern():
    """select-mode quick coherence must receive the configured slc_pattern.

    Regression: 'slc_pattern' was missing from the engine's select-parameter
    list, so the on-the-fly coherence screener kept the processor default
    (``*.slc.*``) and found no SLC in an OPERA ``<date>/tXXX_....h5`` layout.
    """
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    slc = tmp / 'slcs'
    for d in ('20220105', '20220117'):
        (slc / d).mkdir(parents=True)
        (slc / d / f't124_264305_iw2_{d}.h5').touch()
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.processor = isce3\n'
        'slc2ifg.slc_pattern = **/t*.h5\n'
        'slc2ifg.ifgram_list.mode = select\n'
        'slc2ifg.ifgram_list.select.weight_source = coherence\n'
        'engine.stages = ifgram_list,complex_coh\n'
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))
    p = eng._select_params(eng.config.raw)
    assert p['slc_pattern'] == '**/t*.h5'
    assert p['weight_source'] == 'coherence'

    # legacy select.slc_pattern is still honoured when the unified key is 'auto'
    cfg2 = tmp / 'legacy.cfg'
    cfg2.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {slc}\n'
        'slc2ifg.slc_pattern = auto\n'
        'slc2ifg.ifgram_list.mode = select\n'
        'slc2ifg.ifgram_list.select.slc_pattern = *.h5\n'
        'engine.stages = ifgram_list,complex_coh\n'
        'engine.gpu = false\n')
    eng2 = Engine(load_engine_config(str(cfg2)))
    assert eng2._select_params(eng2.config.raw)['slc_pattern'] == '*.h5'


def test_entry_ifgram_list_without_slcs_keeps_tree():
    """ifgram_list listed, but no SLCs: warn and keep the existing pair list."""
    import tempfile

    from mintpy.stdproc.engine.config import load_engine_config
    from mintpy.stdproc.engine.engine import Engine

    tmp = Path(tempfile.mkdtemp())
    inp = tmp / 'products'
    inp.mkdir()
    (inp / 'ifgram_list.txt').write_text('20220105-20220117\n20220210-20220306\n')
    cfg = tmp / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp}\n'
        f'slc2ifg.slc_input = {tmp / "no_such_slcs"}\n'
        'slc2ifg.processor = isce3\n'
        f'engine.input_dir = {inp}\n'
        'engine.stages = ifgram_list,unwrap\n'        # no SLCs needed by unwrap
        'engine.gpu = false\n')
    eng = Engine(load_engine_config(str(cfg)))
    g = eng.plan(dry_run=True)
    assert sorted(g.nodes) == [
        'unwrap#20220105_20220117', 'unwrap#20220210_20220306',
    ]


def test_entry_complex_coh_feeds_unwrap():
    """Entry mode + SLCs + unwrap: unwrap consumes the complex coherence."""
    eng = _entry_engine_with_slcs(tools='complex_coh,unwrap')
    g = eng.plan(dry_run=True)
    uw = g.nodes['unwrap#20220105_20220117']
    assert uw.ctx.inputs['coh'].name == 'fullres.cpx.coh.tif'
    assert 'complex_coh#20220105_20220117' in uw.deps
    assert 'phsig_coh' not in g.nodes


if __name__ == '__main__':
    import traceback

    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith('test_') and callable(fn):
            try:
                fn()
                print(f'PASS {name}')
            except Exception:
                failed += 1
                print(f'FAIL {name}')
                traceback.print_exc()
    sys.exit(1 if failed else 0)
