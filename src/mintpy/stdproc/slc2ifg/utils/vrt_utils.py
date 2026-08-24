#!/usr/bin/env python3
"""Native VRT interferogram creation — replaces dolphin.interferogram.VRTInterferogram.

Uses GDAL's built-in ``VRTDerivedRasterBand`` with ``PixelFunctionType="cmul"``
to compute ``ref_slc * conj(sec_slc)`` on-the-fly.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from osgeo import gdal
import re

gdal.UseExceptions()

logger = logging.getLogger(__name__)

DEFAULT_SUFFIX = ".int.vrt"
DEFAULT_DATETIME_FORMAT = "%Y%m%d"

# VRT template matching dolphin's VRTDerivedRasterBand format
_VRT_TEMPLATE = """\
<VRTDataset rasterXSize="{xsize}" rasterYSize="{ysize}">
    <SRS>{srs}</SRS>
    <GeoTransform>{geotransform}</GeoTransform>
    <VRTRasterBand dataType="CFloat32" band="1" subClass="VRTDerivedRasterBand">
        <PixelFunctionType>cmul</PixelFunctionType>
        <SimpleSource>
            <SourceFilename relativeToVRT="{rel}">{ref_slc}</SourceFilename>
        </SimpleSource>
        <SimpleSource>
            <SourceFilename relativeToVRT="{rel}">{sec_slc}</SourceFilename>
        </SimpleSource>
    </VRTRasterBand>
</VRTDataset>
"""


def _format_nc_filename(filepath: Union[str, Path], subdataset: Optional[str] = None):
    """Format a filepath for GDAL, handling HDF5/NetCDF subdatasets."""
    if subdataset and _is_hdf5(str(filepath)):
        return f'HDF5:"{filepath}":{subdataset}'
    return str(filepath)


def _is_hdf5(path: str) -> bool:
    """Check if a file is HDF5/NetCDF."""
    low = path.lower()
    return low.endswith(('.h5', '.hdf5', '.nc', '.nc4'))


def _get_raster_xysize(path: str) -> tuple:
    """Get (xsize, ysize) of a raster."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f"Cannot open: {path}")
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    ds = None
    return xsize, ysize


def _get_geotransform(path: str) -> str:
    """Get geotransform as a comma-separated string."""
    ds = gdal.Open(path, gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds = None
    return ', '.join(str(x) for x in gt), proj


class VRTInterferogram:
    """Native VRT interferogram — replacement for dolphin.interferogram.VRTInterferogram.

    Uses GDAL's ``VRTDerivedRasterBand`` with ``PixelFunctionType="cmul"`` to
    compute the complex interferogram from two SLC files.

    Supports the same interface used by generate_ifgram.py:
      - Constructor: creates a VRT from two SLC files.
      - ``from_vrt_file()``: loads from an existing VRT.
      - ``load()``: materialises the VRT into a numpy array.
    """

    def __init__(
        self,
        ref_slc: Union[str, Path],
        sec_slc: Union[str, Path],
        path: Union[str, Path],
        outdir: Optional[Union[str, Path]] = None,
        verify_slcs: bool = True,
        write: bool = True,
        subdataset: Optional[str] = None,
        resolve_paths: bool = True,
        use_relative: bool = False,
        **kwargs,
    ):
        self.ref_slc = Path(ref_slc)
        self.sec_slc = Path(sec_slc)
        self.vrt_path = Path(path)
        self.subdataset = subdataset
        self.use_relative = use_relative

        # Format for GDAL (handle HDF5 subdatasets)
        self._ref_gdal_str = _format_nc_filename(self.ref_slc, subdataset)
        self._sec_gdal_str = _format_nc_filename(self.sec_slc, subdataset)

        if verify_slcs:
            xsize1, ysize1 = _get_raster_xysize(self._ref_gdal_str)
            xsize2, ysize2 = _get_raster_xysize(self._sec_gdal_str)
            if xsize1 != xsize2 or ysize1 != ysize2:
                raise ValueError(
                    f"SLC dimensions mismatch: ref {xsize1}x{ysize1} "
                    f"vs sec {xsize2}x{ysize2}"
                )

        if write:
            self._write_vrt()

    def _write_vrt(self):
        """Write the VRT file to disk."""
        self.vrt_path.parent.mkdir(parents=True, exist_ok=True)
        if self.vrt_path.exists():
            self.vrt_path.unlink()

        xsize, ysize = _get_raster_xysize(self._ref_gdal_str)
        gt, srs = _get_geotransform(self._ref_gdal_str)

        rel = "1" if self.use_relative else "0"

        content = _VRT_TEMPLATE.format(
            xsize=xsize,
            ysize=ysize,
            srs=srs,
            geotransform=gt,
            rel=rel,
            ref_slc=self._ref_gdal_str,
            sec_slc=self._sec_gdal_str,
        )

        with open(self.vrt_path, 'w') as f:
            f.write(content)

        logger.info("Created VRT interferogram: %s", self.vrt_path)

    def load(self) -> np.ndarray:
        """Materialise the VRT into a numpy complex array."""
        ds = gdal.Open(str(self.vrt_path), gdal.GA_ReadOnly)
        if ds is None:
            raise RuntimeError(f"Cannot open VRT: {self.vrt_path}")
        data = ds.GetRasterBand(1).ReadAsArray()
        ds = None
        if data.dtype not in (np.complex64, np.complex128):
            data = data.view(np.complex64)
        return data

    @property
    def shape(self):
        """Return (rows, cols) of the VRT raster."""
        xsize, ysize = _get_raster_xysize(str(self.vrt_path))
        return (ysize, xsize)

    @staticmethod
    def from_vrt_file(vrt_path: str) -> "VRTInterferogram":
        """Load an existing VRT file as a VRTInterferogram instance."""
        vrt_path = Path(vrt_path)
        ref_slc = None
        sec_slc = None
        subdataset = None

        try:
            with open(vrt_path) as f:
                content = f.read()
            # Parse SimpleSource elements to extract source filenames
            sources = re.findall(
                r'<SourceFilename[^>]*>([^<]+)</SourceFilename>',
                content,
            )
            if len(sources) >= 2:
                ref_slc = sources[0]
                sec_slc = sources[1]
                # Check for HDF5 subdataset pattern
                m = re.match(r'HDF5:"([^"]+)":(.+)', ref_slc)
                if m:
                    ref_slc = m.group(1)
                    subdataset = m.group(2)
        except Exception:
            pass

        inst = object.__new__(VRTInterferogram)
        inst.vrt_path = vrt_path
        inst.ref_slc = Path(ref_slc) if ref_slc else None
        inst.sec_slc = Path(sec_slc) if sec_slc else None
        inst._ref_gdal_str = ref_slc or str(vrt_path)
        inst._sec_gdal_str = sec_slc or str(vrt_path)
        inst.subdataset = subdataset
        inst.use_relative = False
        return inst
