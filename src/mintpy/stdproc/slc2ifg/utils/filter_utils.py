#!/usr/bin/env python3
"""Native Goldstein and long-wavelength phase filters — replaces dolphin.goldstein + dolphin.filtering."""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np


logger = logging.getLogger(__name__)


def goldstein(
    phase: np.ndarray,
    alpha: float = 0.5,
    psize: int = 32,
    nodata_mask: Optional[np.ndarray] = None,
    gpu: bool = False,
) -> np.ndarray:
    """Apply the Goldstein adaptive phase filter with 50% overlapping patches.

    Drop-in replacement for ``dolphin.goldstein.goldstein``.

    Parameters
    ----------
    phase : np.ndarray
        2D array of complex data (complex64) or float phase to be filtered.
    alpha : float
        Filter exponent, must be in [0, 1].
    psize : int
        Edge length of square FFT patch (power of 2 recommended).
    nodata_mask : np.ndarray (bool), optional
        Boolean mask where data is invalid. Masked pixels are zeroed
        before FFT and restored as zero in output.
    gpu : bool
        Use the CuPy batched-FFT path when available (results identical
        to the CPU path).

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
    pad = psize // 2
    step = pad
    half = pad  # psize // 2

    # --- triangle window (50% overlap add) ---
    wx = (1.0 - np.abs(np.arange(half) - (psize / 2.0 - 1.0))
          / (psize / 2.0 - 1.0))
    wy = (1.0 - np.abs(np.arange(half) - (psize / 2.0 - 1.0))
          / (psize / 2.0 - 1.0))
    q = np.outer(wy, wx)
    wf = np.block([[q, np.flip(q, 1)],
                   [np.flip(q, 0), np.flip(np.flip(q, 0), 1)]])

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

    # Zero-pad image and no-data mask
    padded = np.pad(data, ((pad, pad), (pad, pad)), mode='constant')
    nodata = np.pad(orig_nodata, ((pad, pad), (pad, pad)),
                    mode='constant', constant_values=True)
    p_rows, p_cols = padded.shape

    filtered = np.zeros((p_rows, p_cols), dtype=np.complex64)
    norm = np.zeros((p_rows, p_cols), dtype=np.float32)

    if gpu:
        # Optional engine accelerator (lazy import): when the insarflow
        # engine is installed it provides GPU kernels; otherwise fall back
        # to the CPU path below.
        try:
            from insarflow.engine.gpu_kernels import goldstein_block
        except ImportError:
            gpu = False
        else:
            filtered, norm = goldstein_block(
                padded, nodata, alpha, psize, wf, (0, 0), gpu=True)
    else:
        for i in range(0, p_rows - psize + 1, step):
            for j in range(0, p_cols - psize + 1, step):
                ri, rj = slice(i, i + psize), slice(j, j + psize)
                patch = padded[ri, rj].copy()

                if np.all(nodata[ri, rj]):
                    continue

                patch[nodata[ri, rj]] = 0
                S = np.fft.fft2(patch, s=(psize, psize))
                H = np.power(np.abs(S), alpha)
                S = H * S
                pf = np.fft.ifft2(S, s=(psize, psize))

                w = wf[:patch.shape[0], :patch.shape[1]]
                filtered[ri, rj] += pf * w
                norm[ri, rj] += w

    valid = norm > 0
    filtered[valid] /= norm[valid]

    # Crop back to original size
    filtered = filtered[pad:pad + orig_rows, pad:pad + orig_cols]
    filtered[orig_nodata] = 0 + 0j

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
        return output_dir / f"filtered_{unw_path.stem}{unw_path.suffix}"

    def process_one(idx: int) -> Path:
        unw_path = Path(unw_filenames[idx])
        output_name = output_name_for(unw_path)

        if output_name.exists():
            logger.info("Skipping existing: %s", output_name.name)
            return output_name

        logger.info("Long-wavelength filtering: %s", unw_path.name)

        unw_data = load_gdal(str(unw_path))

        this_mask = bad_pixel_mask.copy()
        if cor_filenames is not None and idx < len(cor_filenames):
            cor_path = cor_filenames[idx]
            if cor_path is not None and cor_path.exists():
                cor_data = load_gdal(str(cor_path))
                this_mask = this_mask | (cor_data < correlation_cutoff)
        if conncomp_filenames is not None and idx < len(conncomp_filenames):
            cc_path = conncomp_filenames[idx]
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
