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

from mintpy.stdproc.engine.tool import (
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
        # The HDF5 subdataset is auto-detected (/data/[VV,VH,HH], preferring
        # VV) and the SLC pattern is inferred from slc2ifg.slc_input.
        ParamSpec('only_vrt', cfg='slc2ifg.generate_ifgram.only_vrt',
                  kind='bool', default=False),
        # Read-time crop (AOI): with the crop_slc stage absent, materialise
        # each interferogram only over this bbox — no cropped SLC files are
        # stored.  The engine pops these params when crop_slc is enabled
        # (the SLCs are then already cropped).
        ParamSpec('bbox', cfg='slc2ifg.bbox'),
        ParamSpec('bbox_buffer', cfg='slc2ifg.bbox_buffer',
                  kind='float', default=0.0),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        skipped = self.skip_if_exists(ctx)
        if skipped is not None:
            return skipped

        from mintpy.stdproc.generate_ifgram import (
            find_slc_file_by_date,
            process_single_pair,
        )

        processor = ctx.param('processor')
        raw_dirs = ctx.input('slc_dir')
        slc_dirs = [Path(d) for d in (raw_dirs if isinstance(raw_dirs, (list, tuple))
                                      else [raw_dirs])]
        slc_pattern = ctx.param('slc_pattern') or self._infer_pattern(slc_dirs)
        d1, d2 = str(ctx.input('date1')), str(ctx.input('date2'))

        slc1 = find_slc_file_by_date(slc_dirs, d1, slc_pattern, processor)
        slc2 = find_slc_file_by_date(slc_dirs, d2, slc_pattern, processor)
        if slc1 is None:
            raise FileNotFoundError(f"SLC for date {d1} not found in {slc_dirs}")
        if slc2 is None:
            raise FileNotFoundError(f"SLC for date {d2} not found in {slc_dirs}")

        pair_dir = ctx.output('pair_dir')
        pair_dir.mkdir(parents=True, exist_ok=True)
        ifg_path = ctx.output('ifg')
        vrt_path = pair_dir / 'fullres.int.vrt'

        # engine.no_skip_existing => recompute everything.  The VRT is an
        # intermediate (not a declared output port), so refresh it as well:
        # a stale VRT would otherwise keep the sources / metadata of an older
        # code path (e.g. a geoless VRT built with GDAL's HDF5 driver) for
        # every product derived from it.
        if ctx.param('no_skip_existing', False) and vrt_path.exists():
            try:
                vrt_path.unlink()
            except OSError:
                pass

        # Read-time crop: map the bbox to this pair's SLC pixel window
        window = None
        bbox = ctx.param('bbox')
        if bbox:
            from mintpy.stdproc import io as sio
            window = sio.bbox_to_window(
                slc1, sio.parse_wsen(str(bbox)),
                None,  # subdataset auto-detected
                float(ctx.param('bbox_buffer', 0.0) or 0.0))
            if window is None:
                raise RuntimeError(
                    f"bbox {bbox} does not intersect SLC {slc1} "
                    f"(pair {d1}-{d2})")

        pair_info = (
            f"{d1}-{d2}", d1, d2, slc1, slc2, vrt_path, ifg_path,
            True,           # verify SLCs (dimension check)
            None,           # subdataset auto-detected from the HDF5 file
            processor,
            ctx.param('only_vrt', False),
            window,
        )
        date12, ok, msg = process_single_pair(pair_info)
        if not ok:
            raise RuntimeError(f"[{date12}] {msg}")
        ctx.logger.info("generate_ifgram %s: %s %s", date12, msg,
                        ctx.elapsed_str())
        return {'ifg': ifg_path, 'pair_dir': pair_dir}

    @staticmethod
    def _infer_pattern(slc_dirs) -> str:
        """Infer the SLC filename pattern from the files in ``slc_dirs``."""
        from mintpy.stdproc.utils.slc_input import (infer_slc_pattern,
                                                    _walk_slc_files)
        names = [f.name for d in slc_dirs for f in _walk_slc_files(Path(d))]
        return infer_slc_pattern(names) if names else '*.slc.*'
