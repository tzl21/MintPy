#!/usr/bin/env python3
"""engine.no_skip_existing: recompute and overwrite existing stage outputs.

Default behaviour is an idempotent resume (a stage whose outputs are ready is
skipped); ``engine.no_skip_existing = True`` removes those output files so the
stage recomputes them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mintpy.stdproc.engine.config import (
    get_bool_opt,
    load_engine_config,
    read_config,
)
from mintpy.stdproc.engine.engine import Engine
from mintpy.stdproc.engine.tool import Tool, ToolContext, get_tool


class _DummyTool(Tool):
    name = 'dummy'

    def run(self, ctx):  # pragma: no cover - never executed here
        return {}


def _ctx(tmp, outputs, flag):
    return ToolContext(tool_name='dummy', inputs={}, outputs=outputs,
                       params={'no_skip_existing': flag}, work_dir=tmp)


def test_skip_if_exists_default_keeps_the_output(tmp_path):
    out = tmp_path / 'a.txt'
    out.write_bytes(b'x')
    ctx = _ctx(tmp_path, {'f': out}, False)
    assert _DummyTool().skip_if_exists(ctx) is not None
    assert ctx.skipped is True and out.exists()


def test_skip_if_exists_with_flag_removes_and_reruns(tmp_path):
    out = tmp_path / 'a.txt'
    out.write_bytes(b'x')
    ctx = _ctx(tmp_path, {'f': out}, True)
    assert _DummyTool().skip_if_exists(ctx) is None
    assert out.exists() is False and ctx.skipped is False


def test_overwrite_keeps_directory_outputs(tmp_path):
    """A pair directory may hold other stages' products — never delete it."""
    from osgeo import gdal

    gdal.UseExceptions()
    f = tmp_path / 'fullres.int.tif'
    gdal.GetDriverByName('GTiff').Create(
        str(f), 4, 4, 1, gdal.GDT_Float32)          # a real (tiny) raster
    d = tmp_path / '20220105_20220117'
    d.mkdir()
    other = d / 'fullres.phsig.coh.tif'
    other.write_bytes(b'x')                 # another stage's product
    ctx = _ctx(tmp_path, {'ifg': f, 'pair_dir': d}, True)
    assert _DummyTool().skip_ready_outputs(ctx, 'generate_ifgram') is False
    assert not f.exists()
    assert d.is_dir() and other.exists()


def test_real_tool_skips_ready_output(tmp_path):
    """Registered tools share the helper (skip path only: no processing)."""
    tool = get_tool('multilook')
    out = tmp_path / 'm.txt'                # non-raster: non-empty is enough
    out.write_bytes(b'x')
    ctx = _ctx(tmp_path, {'ifg': out}, False)
    assert tool.skip_ready_outputs(ctx, 'multilook') is True
    assert ctx.skipped is True and out.exists()


def _engine(tmp_path, extra=''):
    inp = tmp_path / 'input'
    inp.mkdir(exist_ok=True)
    for d in ('20220105', '20220117'):
        (inp / f'{d}.slc.tif').touch()
    cfg_file = tmp_path / 'mini.cfg'
    cfg_file.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        'engine.stages = generate_ifgram,complex_coh\n'
        'engine.gpu = false\n' + extra)
    return Engine(load_engine_config(str(cfg_file)))


def test_engine_injects_the_param_into_every_tool(tmp_path):
    eng = _engine(tmp_path, 'engine.no_skip_existing = true\n')
    for tool in ('generate_ifgram', 'complex_coh', 'multilook', 'filter',
                 'phsig_coh', 'unwrap', 'crop_slc'):
        assert eng._tool_params(tool)['no_skip_existing'] is True, tool
    # the crop stage also reads its stage-specific alias
    assert eng._tool_params('crop_slc')['crop_no_skip_existing'] is False


def test_engine_default_is_false(tmp_path):
    eng = _engine(tmp_path)
    for tool in ('generate_ifgram', 'complex_coh', 'multilook', 'unwrap'):
        assert eng._tool_params(tool)['no_skip_existing'] is False, tool


def test_crop_alias_alone_also_works(tmp_path):
    eng = _engine(tmp_path, 'slc2ifg.crop_slc.no_skip_existing = true\n')
    assert eng._tool_params('crop_slc')['crop_no_skip_existing'] is True
    # ... while the global flag stays off for the other stages
    assert eng._tool_params('generate_ifgram')['no_skip_existing'] is False


def test_template_declares_the_key_as_auto():
    """The key ships in the template and 'auto' resolves to False."""
    cfg = read_config(None)
    for key in ('engine.no_skip_existing', 'slc2ifg.crop_slc.no_skip_existing'):
        assert cfg.has_option('slc2ifg', key), key
        assert get_bool_opt(cfg, key, fallback=False) is False, key
