#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Engine tools: multilook, filter, phsig_coh (uniform stage, per pair).

``filter`` and ``phsig_coh`` support:
- GPU kernels (``engine.gpu``) via ``gpu_kernels`` with CPU fallback;
- intra-interferogram tiling (``engine.tile_size``) via ``tiling`` with
  bit-identical results to the full-image computation.
"""

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

#: complex_coh / phsig_coh share the slc2ifg.generate_coh.* config section
_GENERATE_COH_PARAMS = [
    ParamSpec('slc_pattern', cfg='slc2ifg.generate_coh.slc_pattern'),
    ParamSpec('subdataset', cfg='slc2ifg.generate_coh.subdataset',
              default='/data/VV'),
    ParamSpec('window_size', cfg='slc2ifg.generate_coh.cc_window_size',
              kind='int', default=5),
    ParamSpec('ps_window_size', cfg='slc2ifg.generate_coh.ps_window_size',
              kind='int', default=5),
    ParamSpec('ps_gradient_window',
              cfg='slc2ifg.generate_coh.ps_gradient_window',
              kind='int', default=5),
    ParamSpec('keep_sigma', cfg='slc2ifg.generate_coh.keep_sigma',
              kind='bool', default=False),
]


def _tile_workers(ctx: ToolContext) -> int:
    return max(1, min(int(ctx.param('max_workers', 1) or 1), 4))


def _tag(path: Path) -> str:
    """Short 'pair_dir/file' tag so interleaved engine logs stay attributable."""
    return f"{path.parent.name}/{path.name}"


@register
class MultilookTool(Tool):
    name = 'multilook'
    inputs = [Port('ifg', 'file'), Port('date1', 'value'), Port('date2', 'value')]
    outputs = [Port('ifg', 'file')]
    resource = Resource(device='cpu', mem_estimate_gb=1.5)
    params_spec = [
        ParamSpec('lks_y', cfg='slc2ifg.multilook.lks_y', kind='int', default=1),
        ParamSpec('lks_x', cfg='slc2ifg.multilook.lks_x', kind='int', default=1),
        ParamSpec('method', cfg='slc2ifg.multilook.method', default='mean'),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        out = ctx.output('ifg')
        if self.output_ready([out]):
            ctx.logger.info("skip multilook: %s exists", out.name)
            ctx.skipped = True
            return {'ifg': out}

        from mintpy.stdproc.slc2ifg.multilook import multilook_tif

        out.parent.mkdir(parents=True, exist_ok=True)
        multilook_tif(
            input_tif=str(ctx.input('ifg')),
            output_tif=str(out),
            lks_y=ctx.param('lks_y', 1),
            lks_x=ctx.param('lks_x', 1),
            method=ctx.param('method', 'mean'),
            processor=ctx.param('processor'),
        )
        return {'ifg': out}


@register
class FilterTool(Tool):
    name = 'filter'
    inputs = [Port('ifg', 'file'), Port('date1', 'value'), Port('date2', 'value')]
    outputs = [Port('ifg', 'file')]
    resource = Resource(device='gpu', mem_estimate_gb=1.5,
                        gpu_mem_estimate_gb=1.0)
    params_spec = [
        # stage-level backend switch; each backend owns its namespace
        # (slc2ifg.filter.goldstein.* / slc2ifg.filter.long_wavelength.*).
        # legacy_cfg keeps the old flat keys working with a warning.
        ParamSpec('filter_type', cfg='slc2ifg.filter.filter_type',
                  default='goldstein'),
        ParamSpec('alpha', cfg='slc2ifg.filter.goldstein.alpha',
                  legacy_cfg='slc2ifg.filter.alpha', kind='float',
                  default=0.8),
        ParamSpec('psize', cfg='slc2ifg.filter.goldstein.psize',
                  legacy_cfg='slc2ifg.filter.psize', kind='int', default=32),
        ParamSpec('wavelength_cutoff',
                  cfg='slc2ifg.filter.long_wavelength.wavelength_cutoff',
                  legacy_cfg='slc2ifg.filter.wavelength_cutoff',
                  kind='float', default=25000),
        ParamSpec('pixel_spacing',
                  cfg='slc2ifg.filter.long_wavelength.pixel_spacing',
                  legacy_cfg='slc2ifg.filter.pixel_spacing',
                  kind='float', default=30),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        out = ctx.output('ifg')
        if self.output_ready([out]):
            ctx.logger.info("skip filter: %s exists", out.name)
            ctx.skipped = True
            return {'ifg': out}

        filter_type = ctx.param('filter_type', 'goldstein')
        if filter_type != 'goldstein':
            raise NotImplementedError(
                f"engine 'filter' tool currently supports only 'goldstein' "
                f"(got '{filter_type}')")

        from mintpy.stdproc.slc2ifg.engine.gpu_kernels import cupy_available
        use_gpu = bool(ctx.param('use_gpu', False)) and cupy_available()
        tile_size = int(ctx.param('tile_size', 0) or 0)

        out.parent.mkdir(parents=True, exist_ok=True)

        if tile_size > 0:
            from mintpy.stdproc.slc2ifg.engine.tiling import goldstein_tiled
            goldstein_tiled(
                str(ctx.input('ifg')), str(out),
                ctx.param('alpha', 0.8), ctx.param('psize', 32),
                tile_size, ctx.param('processor'),
                tile_workers=_tile_workers(ctx), gpu=use_gpu)
            ctx.logger.info(
                "filter (tiled %d, gpu=%s): %s %s", tile_size, use_gpu,
                _tag(out), ctx.elapsed_str())
        else:
            from mintpy.stdproc.slc2ifg.filter import process_single_goldstein
            process_single_goldstein(
                str(ctx.input('ifg')), out.parent,
                ctx.param('alpha', 0.8), ctx.param('psize', 32),
                ctx.param('processor'),
                output_file=out, gpu=use_gpu,
            )
            ctx.logger.info("filter (gpu=%s): %s %s", use_gpu, _tag(out),
                            ctx.elapsed_str())
        return {'ifg': out}


@register
class PhsigCohTool(Tool):
    name = 'phsig_coh'
    inputs = [Port('ifg', 'file'), Port('date1', 'value'), Port('date2', 'value')]
    outputs = [Port('coh', 'file')]
    resource = Resource(device='gpu', mem_estimate_gb=2.0,
                        gpu_mem_estimate_gb=1.0)
    params_spec = list(_GENERATE_COH_PARAMS)

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        out = ctx.output('coh')
        if self.output_ready([out]):
            ctx.logger.info("skip phsig_coh: %s exists", out.name)
            ctx.skipped = True
            return {'coh': out}

        from mintpy.stdproc.slc2ifg.engine.gpu_kernels import cupy_available
        use_gpu = bool(ctx.param('use_gpu', False)) and cupy_available()
        tile_size = int(ctx.param('tile_size', 0) or 0)
        ps_win = int(ctx.param('ps_window_size', 5))
        grad_win = int(ctx.param('ps_gradient_window', 5))
        nlks = float(ctx.param('ps_nlks', 1.0))

        out.parent.mkdir(parents=True, exist_ok=True)

        if tile_size > 0:
            self._run_tiled(ctx, out, tile_size, use_gpu, ps_win, grad_win, nlks)
        else:
            self._run_whole(ctx, out, use_gpu, ps_win, grad_win, nlks)
        return {'coh': out}

    # ------------------------------------------------------------------
    def _run_whole(self, ctx, out, use_gpu, ps_win, grad_win, nlks) -> None:
        from mintpy.stdproc.slc2ifg.engine.gpu_kernels import estimate_phsig_block
        from mintpy.stdproc.slc2ifg.generate_coh_phsig import (
            _write_band,
            read_complex_image,
        )
        from mintpy.stdproc.slc2ifg.utils.slc2ifg_utils import create_xml_for_binary

        processor = ctx.param('processor')
        ifg, meta = read_complex_image(str(ctx.input('ifg')), processor)
        coh = estimate_phsig_block(ifg, ps_win, grad_win, nlks, gpu=use_gpu)
        _write_band(str(out), coh, meta, processor, 'phase-sigma correlation')
        if processor == 'isce2':
            create_xml_for_binary(out, family='image',
                                  description='Phase-sigma correlation')

    def _run_tiled(self, ctx, out, tile_size, use_gpu, ps_win, grad_win,
                   nlks) -> None:
        import numpy as np

        from mintpy.stdproc.slc2ifg.engine.gpu_kernels import estimate_phsig_block
        from mintpy.stdproc.slc2ifg.engine.tiling import compute_tiled_file
        from mintpy.stdproc.slc2ifg.utils.slc2ifg_utils import create_xml_for_binary

        processor = ctx.param('processor')
        ps_half = ps_win // 2
        grad_half = grad_win // 2
        overlap = grad_half + ps_half + 1

        def compute_block(block):
            return estimate_phsig_block(block, ps_win, grad_win, nlks,
                                        gpu=use_gpu)

        # The block kernel already reproduces the full-image slope zeroing at
        # image borders, so no output margin zeroing is needed.
        compute_tiled_file(
            str(ctx.input('ifg')), str(out), tile_size, overlap,
            compute_block, processor,
            tile_workers=_tile_workers(ctx),
            sample_dtype=np.float32,   # coherence output is real float32
        )
        ctx.logger.info("phsig_coh (tiled %d, gpu=%s): %s %s",
                        tile_size, use_gpu, _tag(out), ctx.elapsed_str())
        if processor == 'isce2':
            create_xml_for_binary(out, family='image',
                                  description='Phase-sigma correlation')
