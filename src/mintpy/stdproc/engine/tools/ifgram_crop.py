#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Engine tools: ifgram_list (pair generation) and crop_slc."""

from __future__ import annotations

import re

from pathlib import Path
from typing import Dict

from mintpy.stdproc.engine.tool import (
    ParamSpec,
    Port,
    Resource,
    Tool,
    ToolContext,
    register,
)


@register
class IfgramListTool(Tool):
    """Generate the interferometric pair list (also run eagerly by the engine)."""

    name = 'ifgram_list'
    inputs = [Port('slc_dir', 'dir')]
    outputs = [Port('pairs_file', 'pairs_file')]
    resource = Resource(device='cpu', mem_estimate_gb=0.1)
    #: ``default=None`` keeps a key out of ``ctx.params`` unless explicitly
    #: configured, so each mode's built-in fallback applies.
    params_spec = [
        ParamSpec('mode', cfg='slc2ifg.ifgram_list.mode'),
        ParamSpec('num_connections', cfg='slc2ifg.ifgram_list.num_connections',
                  kind='int'),
        ParamSpec('start_date', cfg='slc2ifg.ifgram_list.start_date'),
        ParamSpec('end_date', cfg='slc2ifg.ifgram_list.end_date'),
        ParamSpec('exclude_date', cfg='slc2ifg.ifgram_list.exclude_date'),
        # unified SLC pattern; legacy select.slc_pattern as fallback
        ParamSpec('slc_pattern', cfg='slc2ifg.slc_pattern',
                  legacy_cfg='slc2ifg.ifgram_list.select.slc_pattern'),
    ] + [
        ParamSpec(name, cfg=f'slc2ifg.ifgram_list.select.{name}',
                  kind=kind)
        for name, kind in [
            ('annual_windows', 'str'),
            ('temp_baseline_max', 'int'),
            ('perp_baseline_max', 'float'),
            ('perp_baseline_file', 'str'),
            ('weight_source', 'str'),
            ('model_tau_days', 'float'),
            ('model_gamma0', 'float'),
            ('coh_dir', 'str'),
            ('coh_kind', 'str'),
            ('coh_variant', 'str'),
            ('coh_stat', 'str'),
            ('coh_usable_threshold', 'float'),
            ('quick_window', 'int'),
            ('quick_nlks', 'int'),
            ('quick_max_pixels', 'int'),
            ('quick_max_workers', 'int'),
            ('quick_debias', 'bool'),
            ('min_degree', 'int'),
            ('max_pairs', 'int'),
            ('quality_threshold', 'float'),
            ('robust', 'bool'),
            ('verify', 'bool'),
        ]
    ] + [
        # unified pipeline AOI (not a select.* key): quick coherence reads
        # only the bbox+buffer window of each SLC instead of the whole scene
        ParamSpec('bbox', cfg='slc2ifg.bbox',
                  legacy_cfg='slc2ifg.crop_slc.wsen'),
        ParamSpec('bbox_buffer', cfg='slc2ifg.bbox_buffer', kind='float',
                  legacy_cfg='slc2ifg.crop_slc.buffer'),
        # config keys are select.report / select.dot, param names report_file/dot_file
        ParamSpec('report_file', cfg='slc2ifg.ifgram_list.select.report'),
        ParamSpec('dot_file', cfg='slc2ifg.ifgram_list.select.dot'),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        from mintpy.stdproc.ifgram_list import (
            filter_date_list,
            generate_pairs,
            get_date_list,
            write_pair_list,
        )

        out = ctx.output('pairs_file')
        out.parent.mkdir(parents=True, exist_ok=True)
        dates = get_date_list(str(ctx.input('slc_dir')))
        dates = filter_date_list(
            dates,
            start_date=ctx.param('start_date'),
            end_date=ctx.param('end_date'),
            exclude_date=ctx.param('exclude_date'),
        )
        mode = ctx.param('mode', 'sequential')

        if mode == 'select':
            from mintpy.stdproc.select_ifgrams import (
                SELECT_PARAM_KEYS,
                select_pairs,
            )
            params = {
                k: ctx.param(k) for k in SELECT_PARAM_KEYS
                if ctx.param(k) is not None
            }
            # k-NN skeleton default for select mode is 3 (vs 5 sequential)
            params.setdefault('num_connections', 3)
            params['slc_dir'] = str(ctx.input('slc_dir'))
            params['processor'] = ctx.param('processor', 'isce3')
            if params.get('report_file'):
                rp = Path(params['report_file'])
                if not rp.is_absolute():
                    rp = (ctx.work_dir / rp).resolve()
                params['report_file'] = str(rp)
            pairs, report = select_pairs(dates, params=params)
            ctx.logger.info(
                "ifgram_list (select): %d dates -> %d candidate(s) -> %d "
                "selected pair(s) (connected=%s, rank=%s, min_degree=%s, "
                "weight_source=%s)",
                report['n_dates'], report['n_candidates'],
                report['n_selected'], report.get('connected'),
                report.get('rank'), report.get('min_degree_actual'),
                report.get('weight_source'))
        else:
            # annual_windows applies to ALL modes (canonical knob)
            aw = ctx.param('annual_windows')
            pairs = generate_pairs(
                dates,
                mode=mode,
                num_connections=ctx.param('num_connections', 5),
                oneyear_range=None,
                select_params={'annual_windows': aw} if aw is not None else None,
            )
        write_pair_list(pairs, out)
        return {'pairs_file': out}


@register
class CropSlcTool(Tool):
    """Crop SLCs to a geographic bbox (isce3 geocoded / isce2 radar)."""

    name = 'crop_slc'
    inputs = [Port('slc_dir', 'dir')]
    outputs = [Port('slc_dir', 'dir')]
    resource = Resource(device='cpu', mem_estimate_gb=2.0)
    params_spec = [
        # AOI keys: slc2ifg.bbox is the single source of truth; the legacy
        # crop_slc.wsen/buffer keys keep working with a deprecation warning.
        ParamSpec('wsen', cfg='slc2ifg.bbox',
                  legacy_cfg='slc2ifg.crop_slc.wsen'),
        ParamSpec('pattern', cfg='slc2ifg.slc_pattern',
                  legacy_cfg='slc2ifg.crop_slc.pattern'),
        ParamSpec('buffer', cfg='slc2ifg.bbox_buffer',
                  legacy_cfg='slc2ifg.crop_slc.buffer', kind='float',
                  default=0.0),
        ParamSpec('geom_dir', cfg='slc2ifg.crop_slc.geom_dir'),
        ParamSpec('prefix', cfg='slc2ifg.crop_slc.prefix', default=''),
        ParamSpec('by_burst', cfg='slc2ifg.crop_slc.by_burst', kind='bool',
                  default=False),
        # stage-specific alias; engine.no_skip_existing (injected globally)
        # also turns this on
        ParamSpec('crop_no_skip_existing',
                  cfg='slc2ifg.crop_slc.no_skip_existing', kind='bool',
                  default=False),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:

        from mintpy.stdproc.utils import naming
        from mintpy.stdproc.crop_slc_geo import main as geo_main
        from mintpy.stdproc.crop_slc_geo import parse_arguments as geo_parse
        from mintpy.stdproc.crop_slc_rdr import main as rdr_main
        from mintpy.stdproc.crop_slc_rdr import parse_arguments as rdr_parse

        processor = ctx.param('processor')
        in_dir = Path(ctx.input('slc_dir'))
        # cap the crop-internal worker pool (the engine injects its own
        # max_workers here — reusing n_cpu inside n_cpu dask threads would
        # oversubscribe the node)
        workers = max(1, min(int(ctx.param('max_workers', 1) or 1), 4))
        out_dir = ctx.output('slc_dir')
        out_dir.mkdir(parents=True, exist_ok=True)

        wsen = ctx.param('wsen')
        if not wsen:
            raise ValueError("crop_slc requires slc2ifg.bbox in config (legacy alias: slc2ifg.crop_slc.wsen)")

        file_list = ctx.inputs.get('file_list')
        if file_list is not None:
            # date-filtered file list provided by the engine
            # (--input-dir is still required by the parser; the file list
            # takes precedence in crop_slc_geo's discovery)
            args_list = [
                '--input-dir', str(in_dir),
                '--file-list', str(file_list),
                '--output-dir', str(out_dir),
                '--wsen'] + [x for x in re.split(r'[\s,]+', str(wsen)) if x.strip()] + [
                '--buffer', str(ctx.param('buffer', 0.0)),
                '--prefix', str(ctx.param('prefix', '')),
                '--max-workers', str(workers),
            ]
        else:
            args_list = [
                '--input-dir', str(in_dir),
                '--output-dir', str(out_dir),
                '--wsen'] + [x for x in re.split(r'[\s,]+', str(wsen)) if x.strip()] + [
                '--pattern', ctx.param('pattern', naming.slc_pattern(processor)),
                '--buffer', str(ctx.param('buffer', 0.0)),
                '--prefix', str(ctx.param('prefix', '')),
                '--max-workers', str(workers),
            ]
        if ctx.param('no_skip_existing', False) \
                or ctx.param('crop_no_skip_existing', False):
            args_list.append('--no-skip-existing')
        if ctx.param('geom_dir'):
            args_list += ['--geom-dir', str(ctx.param('geom_dir'))]
        if ctx.param('by_burst', False):
            args_list.append('--by-burst')
        if ctx.inputs.get('no_burst_dirs', False):
            args_list.append('--no-burst-dirs')

        if processor == 'isce2':
            ns = rdr_parse(args_list)
            ret = rdr_main(ns)
        else:
            ns = geo_parse(args_list)
            ret = geo_main(ns)
        if ret not in (0, None):
            raise RuntimeError(f"crop_slc failed with exit code {ret}")
        return {'slc_dir': out_dir}
