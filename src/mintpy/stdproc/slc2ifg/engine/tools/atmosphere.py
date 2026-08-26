#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Engine tool: atmosphere correction (skeleton).

Example of the ``atmosphere`` stage in the processing chain: it consumes
the unwrap output and produces the corrected ``{variant}.atm.unw[.tif]``.
Currently a **skeleton implementation** (``method=none`` is a pass-through
copy, only verifying the chain wiring and product naming); a real algorithm
(ERA5/GACOS, etc.) can be integrated by lazy-loading a third-party package
inside ``run()``, with no changes needed outside this file.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Dict

from mintpy.stdproc.slc2ifg.engine.tool import (
    ParamSpec,
    Port,
    Resource,
    StageSpec,
    Tool,
    ToolContext,
    register,
)


@register
class AtmosphereTool(Tool):
    name = 'atmosphere'
    inputs = [Port('unw', 'file'), Port('date1', 'value'), Port('date2', 'value')]
    outputs = [Port('unw', 'file')]
    resource = Resource(device='cpu', mem_estimate_gb=4.0, tileable=False)
    # Tool self-declared chain fragment (demonstration): the engine merges
    # it with the built-in default chain; this tool is already in the
    # built-in chain, so the declaration mainly serves as a template for
    # third-party tools.
    stage = StageSpec(
        name='atmosphere', tool='atmosphere', scope='global',
        optional=True, requires=('unwrap',), out_dir='unwrap',
        desc='Atmospheric correction (skeleton: method=none is a pass-through copy)',
    )
    params_spec = [
        ParamSpec('method', cfg='slc2ifg.atmosphere.method', default='none'),
        ParamSpec('tile_size', cfg='slc2ifg.atmosphere.tile_size',
                  kind='int', default=0),
    ]

    def run(self, ctx: ToolContext) -> Dict[str, Path]:
        from osgeo import gdal

        skipped = self.skip_if_exists(ctx)
        if skipped is not None:
            return skipped

        src = Path(ctx.input('unw'))
        out = ctx.output('unw')
        out.parent.mkdir(parents=True, exist_ok=True)
        method = ctx.param('method', 'none')

        ds = gdal.Open(str(src))
        if ds is None:
            raise RuntimeError(f"atmosphere: cannot open {src}: "
                               f"{gdal.GetLastErrorMsg()}")
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray()

        if method == 'none':
            # Skeleton: pass-through copy (placeholder to keep the chain
            # runnable end-to-end)
            corrected = data.copy()
        else:
            # Real-algorithm integration point: lazy-load the third-party
            # package and dispatch by method
            raise NotImplementedError(
                f"atmosphere method '{method}' not implemented yet "
                f"(skeleton supports 'none'); plug your correction package "
                f"into AtmosphereTool.run")

        processor = ctx.param('processor', 'isce3')
        driver_name = 'GTiff' if processor == 'isce3' else 'ENVI'
        drv = gdal.GetDriverByName(driver_name)
        # atomic write: create at '<out>.tmp' and rename on success
        tmp = f"{out}.{os.getpid()}.tmp"
        try:
            out_ds = drv.Create(tmp, ds.RasterXSize, ds.RasterYSize, 1,
                                band.DataType,
                                options=([] if processor == 'isce2'
                                         else ['COMPRESS=LZW', 'TILED=YES']))
            if out_ds is None:
                raise RuntimeError(
                    f"atmosphere: failed to create {out}: "
                    f"{gdal.GetLastErrorMsg()}")
            out_ds.SetGeoTransform(ds.GetGeoTransform())
            out_ds.SetProjection(ds.GetProjection())
            out_ds.GetRasterBand(1).WriteArray(corrected)
            out_ds.FlushCache()
            out_ds = None
            os.replace(tmp, str(out))
            tmp_hdr = f"{tmp}.hdr"
            if os.path.exists(tmp_hdr):
                os.replace(tmp_hdr, f"{out}.hdr")
            if processor == 'isce2':
                from mintpy.stdproc.slc2ifg.utils.slc2ifg_utils import (
                    create_xml_for_binary)
                create_xml_for_binary(out, family='image',
                                      description='Atmosphere-corrected phase')
        except BaseException:
            for stray in (tmp, f"{tmp}.hdr"):
                try:
                    if os.path.exists(stray):
                        os.unlink(stray)
                except OSError:
                    pass
            raise
        ds = None

        ctx.logger.info("atmosphere (%s): %s/%s", method,
                        out.parent.name, out.name)
        return {'unw': out}
