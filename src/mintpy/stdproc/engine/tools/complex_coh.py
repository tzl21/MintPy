#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Engine tool: complex coherence (optional, fullres variant only).

Supports GPU kernels (``engine.gpu``) and intra-interferogram tiling
(``engine.tile_size``) with bit-identical results.
"""

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

#: complex_coh / phsig_coh share the slc2ifg.generate_coh.* config section
_GENERATE_COH_PARAMS = [
    ParamSpec('slc_pattern', cfg='slc2ifg.slc_pattern',
                  legacy_cfg='slc2ifg.generate_coh.slc_pattern'),
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


@register
class ComplexCohTool(Tool):
    name = 'complex_coh'
    inputs = [
        Port('pairs_file', 'pairs_file'),
        Port('slc_dir', 'dir'),
        Port('date1', 'value'),
        Port('date2', 'value'),
        Port('burst', 'value'),
    ]
    outputs = [Port('coh', 'file', variant='fullres', ext='.cpx.coh[.tif]')]
    resource = Resource(device='gpu', mem_estimate_gb=2.0,
                        gpu_mem_estimate_gb=1.0)
    params_spec = list(_GENERATE_COH_PARAMS)

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        skipped = self.skip_if_exists(ctx)
        if skipped is not None:
            return skipped

        from mintpy.stdproc.engine.gpu_kernels import cupy_available
        from mintpy.stdproc.generate_coh_complex import (
            find_slc_file_by_date,
        )
        from mintpy.stdproc.utils import naming

        processor = ctx.param('processor')
        slc_pattern = ctx.param('slc_pattern', naming.slc_pattern(processor))
        slc_dir = Path(ctx.input('slc_dir'))
        d1, d2 = str(ctx.input('date1')), str(ctx.input('date2'))

        slc1 = find_slc_file_by_date([slc_dir], d1, slc_pattern)
        slc2 = find_slc_file_by_date([slc_dir], d2, slc_pattern)
        if slc1 is None:
            raise FileNotFoundError(f"SLC for date {d1} not found in {slc_dir}")
        if slc2 is None:
            raise FileNotFoundError(f"SLC for date {d2} not found in {slc_dir}")

        use_gpu = bool(ctx.param('use_gpu', False)) and cupy_available()
        tile_size = int(ctx.param('tile_size', 0) or 0)
        window = int(ctx.param('window_size', 5))
        out = ctx.output('coh')
        out.parent.mkdir(parents=True, exist_ok=True)

        if tile_size > 0:
            from mintpy.stdproc.engine.tiling import complex_coh_tiled
            workers = max(1, min(int(ctx.param('max_workers', 1) or 1), 4))
            complex_coh_tiled(str(slc1), str(slc2), str(out), window,
                              tile_size, processor,
                              tile_workers=workers, gpu=use_gpu,
                              subdataset=ctx.param('subdataset', '/data/VV'))
            ctx.logger.info("complex_coh (tiled %d, gpu=%s): %s/%s %s",
                            tile_size, use_gpu, out.parent.name, out.name,
                            ctx.elapsed_str())
        else:
            from mintpy.stdproc.engine.gpu_kernels import complex_coh_block
            from mintpy.stdproc.generate_coh_complex import (
                read_complex_image,
                write_coherence_image,
            )
            from mintpy.stdproc.utils.slc2ifg_utils import create_xml_for_binary

            sub = ctx.param('subdataset', '/data/VV')
            s1, meta = read_complex_image(str(slc1), processor, subdataset=sub)
            s2, _ = read_complex_image(str(slc2), processor, subdataset=sub)
            if s1.shape != s2.shape:
                raise ValueError(f"Dimension mismatch: {s1.shape} vs {s2.shape}")
            coh = complex_coh_block(s1, s2, window, gpu=use_gpu)
            # Zero the outer window//2 border, matching the tiled path's
            # margin zeroing and the reference CoherenceEstimator (partial
            # windows at the border are biased and must not be emitted).
            half = window // 2
            if half > 0:
                coh[:half, :] = 0
                coh[-half:, :] = 0
                coh[:, :half] = 0
                coh[:, -half:] = 0
            write_coherence_image(str(out), coh, meta, processor)
            if processor == 'isce2':
                create_xml_for_binary(out, family='image',
                                      description='Complex correlation magnitude')

        return {'coh': out}
