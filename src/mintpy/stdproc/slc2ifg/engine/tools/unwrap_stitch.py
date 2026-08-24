#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Engine tools: unwrap (SNAPHU) and stitch (multi-burst merge)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mintpy.stdproc.slc2ifg.engine.tool import (
    ParamSpec,
    Port,
    Resource,
    Tool,
    ToolContext,
    register,
)


@register
class UnwrapTool(Tool):
    name = 'unwrap'
    inputs = [Port('ifg', 'file'), Port('coh', 'file'),
              Port('date1', 'value'), Port('date2', 'value')]
    outputs = [Port('unw', 'file'), Port('conncomp', 'file')]
    resource = Resource(device='cpu', mem_estimate_gb=2.0)
    # Backend-scoped naming: stage-level `algorithm` selects the backend; the
    # snaphu-specific parameters live in their own `slc2ifg.unwrap.snaphu.*`
    # namespace (a future `phass`/`icu` backend gets its own namespace).
    # `legacy_cfg` keeps the old flat keys working with a deprecation warning.
    params_spec = [
        ParamSpec('algorithm', cfg='slc2ifg.unwrap.algorithm', default='snaphu'),
        #: Coherence input type: auto | complex | phsig | none.
        #: auto = phsig if the phsig_coh stage is enabled, else complex if
        #: complex_coh is enabled (fullres only), else none; none = SNAPHU
        #: runs with weight 1 (uniform, no coherence file).
        ParamSpec('coh_type', cfg='slc2ifg.unwrap.coh_type', default='auto'),
        ParamSpec('cost_mode', cfg='slc2ifg.unwrap.snaphu.cost_mode',
                  legacy_cfg='slc2ifg.unwrap.cost_mode', default='smooth'),
        ParamSpec('init_method', cfg='slc2ifg.unwrap.snaphu.init_method',
                  legacy_cfg='slc2ifg.unwrap.init_method', default='mcf'),
        ParamSpec('snaphu_binary', cfg='slc2ifg.unwrap.snaphu.binary',
                  legacy_cfg='slc2ifg.unwrap.snaphu_binary'),
        ParamSpec('nproc', cfg='slc2ifg.unwrap.snaphu.nproc',
                  legacy_cfg='slc2ifg.unwrap.nproc', kind='int', default=1),
        ParamSpec('keep_scratch', cfg='slc2ifg.unwrap.snaphu.keep_scratch',
                  legacy_cfg='slc2ifg.unwrap.keep_scratch',
                  kind='bool', default=False),
        ParamSpec('mask_file', cfg='slc2ifg.unwrap.snaphu.mask_file',
                  legacy_cfg='slc2ifg.unwrap.mask_file'),
        # Unwrap algorithm switch: snaphu (built-in) | phass | icu (requires
        # third-party packages; skeleton)
    ]
    # nlooks (default = ps_nlks) and ntiles (two-key tuple) are injected by
    # the engine as special cases

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        unw, cc = ctx.output('unw'), ctx.output('conncomp')
        if self.output_ready([unw, cc]):
            ctx.logger.info("skip unwrap: %s exists", unw.name)
            ctx.skipped = True
            return {'unw': unw, 'conncomp': cc}

        algorithm = ctx.param('algorithm', 'snaphu')
        if algorithm != 'snaphu':
            raise NotImplementedError(
                f"unwrap algorithm '{algorithm}' not available in this build "
                f"(supported: snaphu; see docs/engine_stages_design.md §4.3 "
                f"for phass/icu integration points)")

        from mintpy.stdproc.slc2ifg.unwrap_ifgram import _unwrap_single

        # _unwrap_single derives {dp}/{variant}.unw under output_dir,
        # so pass the stage root (parent of the date-pair directory).
        unw_root = unw.parent.parent
        unw_root.mkdir(parents=True, exist_ok=True)
        mask = ctx.param('mask_file')
        # 'coh' may be None when coh_type resolves to 'none' — SNAPHU then
        # runs with weight 1 (uniform), handled inside _unwrap_single.
        cor_path = ctx.input('coh')
        _unwrap_single(
            ifg_path=Path(ctx.input('ifg')),
            cor_path=Path(cor_path) if cor_path else None,
            nlooks=ctx.param('nlooks', 1.0),
            output_dir=unw_root,
            processor=ctx.param('processor'),
            snaphu_bin=ctx.param('snaphu_binary'),
            cost_mode=ctx.param('cost_mode', 'smooth'),
            init_method=ctx.param('init_method', 'mcf'),
            mask_path=Path(mask) if mask else None,
            ntiles=tuple(ctx.param('ntiles', (1, 1))),
            nproc=ctx.param('nproc', 1),
            keep_scratch=ctx.param('keep_scratch', False),
        )
        # Same one-line-per-task INFO style as the other tools, with the
        # task's actual processing time appended.
        ctx.logger.info("unwrap (%s, coh_type=%s): %s/%s %s", algorithm,
                        ctx.param('coh_type', 'auto'),
                        unw.parent.name, unw.name, ctx.elapsed_str())
        return {'unw': unw, 'conncomp': cc}


@register
class StitchTool(Tool):
    name = 'stitch'
    inputs = [Port('file_list', 'file_list'),
              Port('date1', 'value'), Port('date2', 'value')]
    outputs = [Port('stitched', 'file')]
    resource = Resource(device='cpu', mem_estimate_gb=3.0)
    params_spec = [
        ParamSpec('out_bounds', cfg='slc2ifg.stitch.out_bounds'),
        ParamSpec('overwrite', cfg=None, kind='bool', default=True),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        out = ctx.output('stitched')
        overwrite = ctx.param('overwrite', True)
        if self.output_ready([out]) and not overwrite:
            ctx.logger.info("skip stitch: %s exists", out.name)
            ctx.skipped = True
            return {'stitched': out}

        from mintpy.stdproc.slc2ifg.stitch import get_file_epsg, stitch_date_pair

        file_list: List[Path] = [Path(f) for f in ctx.input('file_list')]
        if not file_list:
            raise ValueError(f"stitch {ctx.tool_name}: empty file list")

        # Detect EPSG from the first readable input (default UTM zone 5N)
        epsg_utm = 32605
        for f in file_list:
            epsg = get_file_epsg(f)
            if epsg:
                epsg_utm = epsg
                break

        out_bounds = self._parse_bounds(ctx.param('out_bounds'))
        ok = stitch_date_pair(file_list, out, out_bounds, overwrite, epsg_utm)
        if not ok:
            raise RuntimeError(f"stitch failed for {out}")
        return {'stitched': out}

    @staticmethod
    def _parse_bounds(raw: Optional[str]) -> Optional[Tuple[float, float, float, float]]:
        if not raw:
            return None
        parts = [float(x) for x in str(raw).split()]
        if len(parts) != 4:
            raise ValueError(f"Invalid out_bounds '{raw}', expected 'W S E N'")
        return tuple(parts)  # type: ignore[return-value]
