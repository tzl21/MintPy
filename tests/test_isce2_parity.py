#!/usr/bin/env python3
"""ISCE2 parity tests for the slc2ifg coherence / Goldstein implementations.

Guards the two behaviours aligned with the ISCE2 reference code:

- complex coherence uses the ISCE2 Bartlett window by default
  (``mroipac/correlation/src/cchz_wave.cpp``);
- Goldstein filter reproduces ``mroipac/filter/src/psfilt.c`` +
  ``rescale_magnitude.c`` (magnitude restored to the input).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def _isce2_psfilt(img, alpha=0.5, n=32):
    """Inline ISCE2 psfilt.c reference (patches from the image origin)."""
    rows, cols = img.shape
    step = n // 2
    idx = np.arange(n, dtype=np.float64)
    w1d = 1.0 - np.abs(2.0 * (idx - n // 2) / (n + 1.0))
    wf = (np.outer(w1d, w1d) / float(n * n)).astype(np.float32)
    out = np.zeros((rows + n, cols + n), dtype=np.complex64)
    padded = np.pad(img.astype(np.complex64), ((0, n), (0, n)), mode='constant')
    for i in range(0, rows, step):
        for j in range(0, cols, step):
            patch = padded[i:i + n, j:j + n]
            S = np.fft.fft2(patch, s=(n, n))
            H = np.power(np.abs(S), alpha)
            pf = np.fft.ifft2(H * S, s=(n, n))
            out[i:i + n, j:j + n] += pf * wf
    return out[:rows, :cols]


def test_coherence_kernel_triangular_matches_isce2():
    """coherence_kernel('triangular') == ISCE2 Bartlett outer product."""
    from mintpy.stdproc.generate_coh_complex import coherence_kernel

    for n in (5, 7, 9):
        k = coherence_kernel(n, 'triangular')
        idx = np.arange(n, dtype=np.float64)
        w = 1.0 - np.abs(2.0 * (idx - n // 2) / (n + 1.0))
        ref = np.outer(w, w)
        ref = (ref / ref.sum()).astype(np.float32)
        assert np.allclose(k, ref, rtol=0, atol=1e-7), f"n={n}"


def test_coherence_kernel_uniform_is_boxcar():
    from mintpy.stdproc.generate_coh_complex import coherence_kernel

    k = coherence_kernel(5, 'uniform')
    assert np.allclose(k, np.full((5, 5), 1.0 / 25.0))


def test_coherence_estimator_default_is_triangular():
    from mintpy.stdproc.generate_coh_complex import CoherenceEstimator

    est = CoherenceEstimator({'window_size': 5})
    assert est.window_type == 'triangular'
    ref = np.ones((5, 5), dtype=np.float32)
    assert not np.allclose(est.kernel, ref / ref.sum())


def test_goldstein_magnitude_rescale():
    """ISCE2 rescale_magnitude: |filtered| == |input| (non-nodata pixels)."""
    from mintpy.stdproc.utils.filter_utils import goldstein

    rng = np.random.default_rng(0)
    ifg = (rng.standard_normal((96, 80)) + 1j * rng.standard_normal((96, 80)))
    ifg = ifg.astype(np.complex64)

    out = goldstein(ifg, alpha=0.5, psize=32)
    assert np.allclose(np.abs(out), np.abs(ifg), rtol=1e-5, atol=1e-6)

    # phase is preserved where the filtered result is well defined
    assert np.isfinite(np.angle(out)).all()


def test_goldstein_matches_isce2_psfilt():
    """The filter core reproduces ISCE2 psfilt.c (before magnitude rescale)."""
    from mintpy.stdproc.utils.filter_utils import goldstein

    rng = np.random.default_rng(1)
    ifg = (rng.standard_normal((96, 80)) + 1j * rng.standard_normal((96, 80)))
    ifg = ifg.astype(np.complex64)

    out = goldstein(ifg, alpha=0.5, psize=32, rescale_magnitude=False)
    ref = _isce2_psfilt(ifg, alpha=0.5, n=32)
    assert np.allclose(out, ref, rtol=1e-5, atol=1e-6)


def test_goldstein_alpha_zero_is_noop():
    from mintpy.stdproc.utils.filter_utils import goldstein

    rng = np.random.default_rng(2)
    ifg = (rng.standard_normal((64, 64)) + 1j * rng.standard_normal((64, 64)))
    ifg = ifg.astype(np.complex64)
    out = goldstein(ifg, alpha=0.0)
    assert np.array_equal(out, ifg)
