#!/usr/bin/env python3
"""Native Goldstein and long-wavelength phase filters — replaces dolphin.goldstein + dolphin.filtering."""

import logging
from pathlib import Path
from typing import List, Optional

import re

import numpy as np


logger = logging.getLogger(__name__)


def goldstein(
    phase: np.ndarray,
    alpha: float = 0.5,
    psize: int = 32,
    nodata_mask: Optional[np.ndarray] = None,
    gpu: bool = False,
    rescale_magnitude: bool = True,
) -> np.ndarray:
    """Goldstein-Werner power-spectral filter, ISCE2-compatible.

    Reproduces ISCE2 ``mroipac/filter/src/psfilt.c`` + ``rescale_magnitude.c``:
    FFT patches of ``psize`` pixels stepped by ``psize // 2``, anchored at the
    image origin, Bartlett weighting ``w(i) = 1 - |2*(i - psize//2)/(psize+1)|``
    and spectral exponent ``alpha``; overlapping patches are summed with the
    ISCE2 normalization (window already divided by ``psize**2``, no
    per-pixel re-normalization).

    Parameters
    ----------
    phase : np.ndarray
        2D array of complex data (complex64) or float phase to be filtered.
    alpha : float
        Filter exponent, must be >= 0 (0 = no filtering).
    psize : int
        Edge length of the square FFT patch (ISCE2 ``NFFT``, default 32;
        the patch step is ``psize // 2``).
    nodata_mask : np.ndarray (bool), optional
        Boolean mask where data is invalid. Masked pixels are zeroed before
        the FFT and restored as zero in the output (slc2ifg extension, not
        present in ISCE2).
    gpu : bool
        Use the CuPy batched-FFT path when available (results identical to
        the CPU path).
    rescale_magnitude : bool
        ISCE2 ``rescale_magnitude`` behaviour (default True): keep the
        filtered phase but restore the *input* magnitude. Set False for the
        pure power-spectral filter output.

    Returns
    -------
    np.ndarray
        2D filtered array, same shape and dtype as input.
    """
    if phase.ndim != 2:
        raise ValueError("Input must be a 2D array")
    if alpha < 0:
        raise ValueError(f"alpha must be >= 0, got {alpha}")
    if alpha == 0 or psize < 1:
        return phase.copy()

    orig_rows, orig_cols = phase.shape
    step = max(1, psize // 2)

    # ISCE2 Bartlett window (psfilt.c), normalized by psize**2 exactly as the
    # C routine does; overlapping patches are summed without re-normalizing.
    idx = np.arange(psize, dtype=np.float64)
    w1d = 1.0 - np.abs(2.0 * (idx - psize // 2) / (psize + 1.0))
    wf = (np.outer(w1d, w1d) / float(psize * psize)).astype(np.float32)

    # --- complex conversion ---
    if np.iscomplexobj(phase):
        data = np.nan_to_num(phase, copy=True).astype(np.complex64)
    else:
        data = np.nan_to_num(np.exp(1j * np.nan_to_num(phase)), copy=True,
                             nan=np.complex64(0+0j)).astype(np.complex64)

    # --- no-data handling ---
    if nodata_mask is None:
        orig_nodata = np.zeros((orig_rows, orig_cols), dtype=bool)
    else:
        orig_nodata = nodata_mask.copy()

    # Zero-extension of psize pixels past the image edge (psfilt.c reads
    # zeros past EOF).  The patch grid itself is anchored at the image
    # origin, so no half-patch offset is introduced.
    pad = psize
    padded = np.pad(data, ((0, pad), (0, pad)), mode='constant')
    nodata = np.pad(orig_nodata, ((0, pad), (0, pad)),
                    mode='constant', constant_values=True)
    p_rows, p_cols = padded.shape

    filtered = np.zeros((p_rows, p_cols), dtype=np.complex64)

    if gpu:
        # Optional engine accelerator (lazy import); falls back to CPU when
        # the CuPy kernel is unavailable.
        try:
            from mintpy.stdproc.engine.gpu_kernels import goldstein_block
        except ImportError:
            gpu = False
        else:
            filtered = goldstein_block(
                padded, nodata, alpha, psize, wf, (0, 0), gpu=True)
    if not gpu:
        for i in range(0, orig_rows, step):
            for j in range(0, orig_cols, step):
                ri, rj = slice(i, i + psize), slice(j, j + psize)
                patch = padded[ri, rj].copy()

                if np.all(nodata[ri, rj]):
                    continue

                patch[nodata[ri, rj]] = 0
                S = np.fft.fft2(patch, s=(psize, psize))
                H = np.power(np.abs(S), alpha)
                pf = np.fft.ifft2(H * S, s=(psize, psize))
                filtered[ri, rj] += pf * wf

    # Crop back to original size
    filtered = filtered[:orig_rows, :orig_cols]
    filtered[orig_nodata] = 0 + 0j

    if rescale_magnitude:
        # ISCE2 rescale_magnitude.c: keep the filtered phase, restore |input|
        mag = np.abs(data[:orig_rows, :orig_cols])
        filtered = (mag * np.exp(1j * np.angle(filtered))).astype(np.complex64)

    if np.iscomplexobj(phase):
        return filtered.astype(np.complex64)
    else:
        return np.angle(filtered).astype(phase.dtype)


def long_wavelength_filter(
    unwrapped_phase: np.ndarray,
    bad_pixel_mask: np.ndarray,
    wavelength_cutoff: float = 25000,
    pixel_spacing: float = 30.0,
    workers: int = 1,
    fill_value: Optional[float] = None,
    scratch_dir: Optional[Path] = None,
) -> np.ndarray:
    """Filter out signals with spatial wavelength longer than a threshold.

    Drop-in replacement for ``dolphin.filtering.filter_long_wavelength``.

    Parameters
    ----------
    unwrapped_phase : np.ndarray
        Unwrapped interferogram phase to filter.
    bad_pixel_mask : np.ndarray
        Boolean array with same shape as `unwrapped_phase`, True = ignore.
    wavelength_cutoff : float
        Spatial wavelength threshold in meters (default: 25 km).
    pixel_spacing : float
        Pixel spatial spacing in meters (default: 30).
    workers : int
        Number of FFT workers for scipy (default: 1).
    fill_value : float, optional
        Value to place in masked output pixels.
    scratch_dir : Path, optional
        Directory for temporary files.

    Returns
    -------
    np.ndarray
        Filtered interferogram without long-wavelength signals.
    """
    from scipy import fft, ndimage

    sigma = _compute_filter_sigma(wavelength_cutoff, pixel_spacing, cutoff_value=0.5)
    rows, cols = unwrapped_phase.shape

    if sigma > rows or sigma > cols:
        msg = f"{wavelength_cutoff=} too large for image. Shape={(rows, cols)}, {pixel_spacing=}"
        raise ValueError(msg)

    displacement = np.nan_to_num(unwrapped_phase.astype(np.float64))

    nodata_mask = displacement == 0
    in_bounds_pixels = np.logical_not(nodata_mask)

    displacement[bad_pixel_mask] = np.nan

    from scipy.ndimage import distance_transform_edt
    mask_nan = np.isnan(displacement)
    indices = distance_transform_edt(mask_nan, return_distances=False, return_indices=True)
    displacement = displacement[tuple(indices)]

    lowpass = fft.fft2(displacement, workers=workers)
    lowpass = ndimage.fourier_gaussian(lowpass, sigma=sigma)
    lowpass = fft.ifft2(lowpass, workers=workers).real.astype(unwrapped_phase.dtype)

    filtered = (displacement - lowpass) * in_bounds_pixels

    if fill_value is not None:
        good_pixel_mask = np.logical_not(bad_pixel_mask)
        total_valid = in_bounds_pixels & good_pixel_mask
        return np.where(total_valid, filtered, fill_value)

    return filtered


def _compute_filter_sigma(
    wavelength_cutoff: float, pixel_spacing: float, cutoff_value: float = 0.25
) -> float:
    """Compute Gaussian sigma (in pixels) matching the given wavelength cutoff."""
    sigma_spatial = (
        wavelength_cutoff * np.sqrt(-np.log(cutoff_value)) / (np.sqrt(2) * np.pi)
    )
    return sigma_spatial / pixel_spacing


def filter_rasters(
    unw_filenames: List[Path],
    cor_filenames: Optional[List[Path]] = None,
    conncomp_filenames: Optional[List[Path]] = None,
    temporal_coherence_filename: Optional[Path] = None,
    wavelength_cutoff: float = 25000,
    correlation_cutoff: float = 0.5,
    pixel_spacing: float = 30.0,
    output_dir: Optional[Path] = None,
    max_workers: int = 1,
    fill_value: Optional[float] = None,
) -> List[Path]:
    """Batch long-wavelength filter with optional correlation/conncomp masking.

    Drop-in replacement for ``dolphin.filtering.filter_rasters``.

    Parameters
    ----------
    unw_filenames : list of Path
        Input unwrapped phase files.
    cor_filenames : list of Path, optional
        Correlation files for masking.
    conncomp_filenames : list of Path, optional
        Connected component files for masking.
    temporal_coherence_filename : Path, optional
        Temporal coherence file for masking.
    wavelength_cutoff : float
        Spatial wavelength cutoff in meters (default: 25 km).
    correlation_cutoff : float
        Threshold below which to mask pixels.
    pixel_spacing : float
        Pixel spacing in meters.
    output_dir : Path, optional
        Output directory (defaults to same as input).
    max_workers : int
        Number of parallel workers (threads).
    fill_value : float, optional
        Value to fill masked pixels.

    Returns
    -------
    list of Path
        Output file paths.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .naming import int_ext, is_date_pair_dir, next_variant, variant_of
    from .stitching_utils import load_gdal, write_arr
    from .slc2ifg_utils import create_xml_for_binary

    if output_dir is None:
        output_dir = unw_filenames[0].parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bad_pixel_mask = np.zeros(
        load_gdal(str(unw_filenames[0])).shape, dtype=bool
    )
    if temporal_coherence_filename and temporal_coherence_filename.exists():
        tc = load_gdal(str(temporal_coherence_filename))
        bad_pixel_mask = bad_pixel_mask | (tc < 0.5)

    def output_name_for(unw_path: Path) -> Path:
        """Canonical filtered output name (mirrors filter.py helper)."""
        date_pair = unw_path.parent.name
        if is_date_pair_dir(date_pair) and unw_path.name.endswith(int_ext('isce3')):
            variant = variant_of(unw_path, 'isce3')
            out_variant = next_variant(variant, "filter")
            if out_variant == variant:
                return output_dir / date_pair / unw_path.name
            return output_dir / date_pair / f"{out_variant}.int.tif"
        if is_date_pair_dir(date_pair):
            # include the date pair: the flat fallback name would otherwise
            # collide across pairs (e.g. every pair's fullres.unw.tif)
            return output_dir / f"filtered_{date_pair}_{unw_path.stem}{unw_path.suffix}"
        return output_dir / f"filtered_{unw_path.stem}{unw_path.suffix}"

    # match coherence/conncomp by date-pair key rather than positional order
    def _dp_key(path: Path) -> str:
        m = re.search(r'(\d{8})_(\d{8})', str(path))
        return m.group(0) if m else path.name
    cor_by_key = {_dp_key(Path(c)): Path(c) for c in (cor_filenames or [])}
    cc_by_key = {_dp_key(Path(c)): Path(c) for c in (conncomp_filenames or [])}

    def process_one(idx: int) -> Path:
        unw_path = Path(unw_filenames[idx])
        output_name = output_name_for(unw_path)

        if output_name.exists():
            logger.info("Skipping existing: %s", output_name.name)
            return output_name

        logger.info("Long-wavelength filtering: %s", unw_path.name)

        unw_data = load_gdal(str(unw_path))

        this_mask = bad_pixel_mask.copy()
        cor_path = cor_by_key.get(_dp_key(unw_path))
        if cor_path is not None and cor_path.exists():
            cor_data = load_gdal(str(cor_path))
            this_mask = this_mask | (cor_data < correlation_cutoff)
        cc_path = cc_by_key.get(_dp_key(unw_path))
        if cc_path is not None and cc_path.exists():
            cc_data = load_gdal(str(cc_path))
            this_mask = this_mask | (cc_data == 0)

        from osgeo import gdal
        ds = gdal.Open(str(unw_path))
        gt = ds.GetGeoTransform()
        ds = None
        px_spacing = (abs(gt[1]) + abs(gt[5])) / 2.0 if pixel_spacing <= 0 else pixel_spacing

        filtered = long_wavelength_filter(
            unwrapped_phase=unw_data,
            bad_pixel_mask=this_mask,
            wavelength_cutoff=wavelength_cutoff,
            pixel_spacing=px_spacing,
            fill_value=fill_value,
        )

        ext = unw_path.suffix.lower()
        if ext in ('.int', '.unw', '.coh', '.conncomp'):
            driver = "ENVI"
            opts = []
        else:
            driver = "GTiff"
            opts = ["COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"]

        write_arr(filtered, str(unw_path), str(output_name), driver=driver, options=opts)

        if driver == "ENVI":
            create_xml_for_binary(output_name, family='intimage',
                                  description=f'Filtered (λ>{wavelength_cutoff}m)')

        return output_name

    if max_workers > 1 and len(unw_filenames) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            output_files = list(executor.map(process_one, range(len(unw_filenames))))
    else:
        output_files = [process_one(i) for i in range(len(unw_filenames))]

    return output_files
