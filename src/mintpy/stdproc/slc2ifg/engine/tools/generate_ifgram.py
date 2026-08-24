#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Engine tool: generate_ifgram (one node per date pair)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from mintpy.stdproc.slc2ifg.engine.tool import (
    ParamSpec,
    Port,
    Resource,
    Tool,
    ToolContext,
    register,
)


@register
class GenerateIfgramTool(Tool):
    name = 'generate_ifgram'
    inputs = [
        Port('pairs_file', 'pairs_file'),
        Port('slc_dir', 'dir'),
        Port('date1', 'value'),
        Port('date2', 'value'),
        Port('burst', 'value'),
    ]
    outputs = [Port('ifg', 'file', ext='.int[.tif]'),
               Port('pair_dir', 'dir')]
    resource = Resource(device='cpu', mem_estimate_gb=2.0)
    params_spec = [
        ParamSpec('slc_pattern', cfg='slc2ifg.generate_ifgram.slc_pattern'),
        ParamSpec('subdataset', cfg='slc2ifg.generate_ifgram.subdataset',
                  default='/data/VV'),
        ParamSpec('no_verify', cfg='slc2ifg.generate_ifgram.no_verify',
                  kind='bool', default=False),
        ParamSpec('only_vrt', cfg='slc2ifg.generate_ifgram.only_vrt',
                  kind='bool', default=False),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        skipped = self.skip_if_exists(ctx)
        if skipped is not None:
            return skipped

        from mintpy.stdproc.slc2ifg.generate_ifgram import (
            find_slc_file_by_date,
            process_single_pair,
        )
        from mintpy.stdproc.slc2ifg.utils import naming

        processor = ctx.param('processor')
        slc_pattern = ctx.param('slc_pattern', naming.slc_pattern(processor))
        slc_dir = Path(ctx.input('slc_dir'))
        d1, d2 = str(ctx.input('date1')), str(ctx.input('date2'))

        slc1 = find_slc_file_by_date([slc_dir], d1, slc_pattern, processor)
        slc2 = find_slc_file_by_date([slc_dir], d2, slc_pattern, processor)
        if slc1 is None:
            raise FileNotFoundError(f"SLC for date {d1} not found in {slc_dir}")
        if slc2 is None:
            raise FileNotFoundError(f"SLC for date {d2} not found in {slc_dir}")

        pair_dir = ctx.output('pair_dir')
        pair_dir.mkdir(parents=True, exist_ok=True)
        ifg_path = ctx.output('ifg')
        vrt_path = pair_dir / 'fullres.int.vrt'

        pair_info = (
            f"{d1}-{d2}", d1, d2, slc1, slc2, vrt_path, ifg_path,
            not ctx.param('no_verify', False),
            ctx.param('subdataset', '/data/VV'),
            processor,
            ctx.param('only_vrt', False),
        )
        date12, ok, msg = process_single_pair(pair_info)
        if not ok:
            raise RuntimeError(f"[{date12}] {msg}")
        ctx.logger.info("generate_ifgram %s: %s %s", date12, msg,
                        ctx.elapsed_str())
        return {'ifg': ifg_path, 'pair_dir': pair_dir}
