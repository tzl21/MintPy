#!/usr/bin/env python3
"""Unit tests for the GPU-memory improvements (P0-P2).

Covers:
- phsig deramping-ramp batching is bit-identical to the old full-image ramp
  (CPU reference; the GPU kernel mirrors the same structure);
- Goldstein patch batching sizing;
- GPU-fallback counter / pool reset safety;
- VRAM-aware ``gpu_workers`` accounting;
- device round-robin ``GpuPool``;
- auto tile_size when GPU is enabled and ``engine.tile_size`` is unset.

Pure-Python: osgeo is stubbed for the phsig test (the module imports it at
top level but only uses it in I/O functions); no CuPy required.
"""

import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def _install_osgeo_stub():
    """Minimal osgeo stub so ``generate_coh_phsig`` imports without GDAL."""
    if 'osgeo' in sys.modules:
        return
    gdal = types.ModuleType('osgeo.gdal')
    gdal.UseExceptions = lambda: None
    osr = types.ModuleType('osgeo.osr')
    osgeo = types.ModuleType('osgeo')
    sys.modules['osgeo'] = osgeo
    sys.modules['osgeo.gdal'] = gdal
    sys.modules['osgeo.osr'] = osr


# ------------------------------------------------------------------------
# P0: phsig ramp batching (CPU reference vs new implementation)
# ------------------------------------------------------------------------
def _phsig_reference_full_ramp(ifg_arr, ps_win=5, grad_win=5, nlks=1.0):
    """The pre-batching algorithm (full-image ramp) — regression reference."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.ndimage import correlate

    from mintpy.stdproc.slc2ifg.generate_coh_phsig import _gaussian_kernel

    rows, cols = ifg_arr.shape
    if ps_win % 2 == 0:
        ps_win += 1
    if grad_win % 2 == 0:
        grad_win += 1
    ps_half = ps_win // 2
    grad_half = grad_win // 2

    padded = np.pad(ifg_arr, ((grad_half, grad_half), (grad_half, grad_half)),
                    mode='constant')
    rg_diff = (padded[grad_half:grad_half + rows, grad_half:grad_half + cols] *
               np.conj(padded[grad_half:grad_half + rows,
                              grad_half - 1:grad_half + cols - 1]))
    az_diff = (padded[grad_half:grad_half + rows, grad_half:grad_half + cols] *
               np.conj(padded[grad_half - 1:grad_half + rows - 1,
                              grad_half:grad_half + cols]))
    gk = _gaussian_kernel(grad_win)
    rg_smooth = correlate(rg_diff, gk)
    az_smooth = correlate(az_diff, gk)
    rg_slope = np.arctan2(rg_smooth.imag, rg_smooth.real)
    az_slope = np.arctan2(az_smooth.imag, az_smooth.real)
    rg_slope[np.abs(rg_smooth) == 0] = 0.0
    az_slope[np.abs(az_smooth) == 0] = 0.0
    if grad_half > 0:
        for sl in (rg_slope, az_slope):
            sl[:grad_half + 1, :] = 0.0
            sl[-(grad_half):, :] = 0.0
            sl[:, :grad_half + 1] = 0.0
            sl[:, -(grad_half):] = 0.0

    i0, i1 = ps_half, rows - ps_half
    j0, j1 = ps_half, cols - ps_half
    if i1 <= i0 or j1 <= j0:
        return np.zeros((rows, cols), dtype=np.float32)

    offsets = np.arange(-ps_half, ps_half + 1)
    di_mesh, dj_mesh = np.meshgrid(offsets, offsets, indexing='ij')
    ps_weights = _gaussian_kernel(ps_win)
    windows = sliding_window_view(ifg_arr, (ps_win, ps_win))
    n_rows_out = i1 - i0
    n_cols_out = j1 - j0

    # OLD behaviour: the full-image ramp materializes once (O(N*w^2) float64)
    ramp = (
        di_mesh[None, None, :, :] * az_slope[i0:i1, j0:j1][:, :, None, None] +
        dj_mesh[None, None, :, :] * rg_slope[i0:i1, j0:j1][:, :, None, None]
    )

    coh = np.zeros((rows, cols), dtype=np.float32)
    win_pixels = n_cols_out * ps_win * ps_win
    batch_rows = max(1, 2_000_000 // max(win_pixels, 1))
    for r0 in range(0, n_rows_out, batch_rows):
        r1 = min(r0 + batch_rows, n_rows_out)
        win_block = windows[r0:r1]
        ramp_block = ramp[r0:r1]
        exp_ramp = np.cos(ramp_block) - 1j * np.sin(ramp_block)
        comp = win_block * exp_ramp
        wsum = np.sum(comp, axis=(2, 3))
        mag = np.abs(wsum)
        mag_valid = mag > 1e-10
        if np.any(mag_valid):
            norm_sum = np.zeros_like(wsum)
            norm_sum[mag_valid] = wsum[mag_valid] / mag[mag_valid]
            deramped = comp * np.conj(norm_sum[:, :, None, None])
            phases = np.arctan2(deramped.imag, deramped.real)
            wt = ps_weights[None, None, :, :]
            mean_ph = np.sum(wt * phases, axis=(2, 3))
            mean_ph2 = np.sum(wt * phases * phases, axis=(2, 3))
            var = mean_ph2 - mean_ph * mean_ph
            with np.errstate(divide='ignore', invalid='ignore'):
                val = np.where(var > 0, 1.0 / np.sqrt(2.0 * nlks * var + 1.0),
                               1.0)
            val = np.where(mag_valid, val, 0.0)
            coh[i0 + r0:i0 + r1, j0:j1] = val
    return np.clip(coh, 0.0, 1.0)


def test_phsig_batched_ramp_bit_identical():
    """Per-batch ramp must be bit-identical to the old full-image ramp."""
    pytest.importorskip('scipy')
    import numpy as np

    _install_osgeo_stub()
    from mintpy.stdproc.slc2ifg.generate_coh_phsig import estimate_phsig_correlation

    rng = np.random.default_rng(7)
    for shape in ((37, 53), (100, 200), (33, 300)):
        for ps, gd in ((5, 5), (7, 3), (3, 7)):
            ifg = (rng.standard_normal(shape) +
                   1j * rng.standard_normal(shape)).astype(np.complex64)
            new = estimate_phsig_correlation(ifg, ps, gd, 4.0)
            ref = _phsig_reference_full_ramp(ifg, ps, gd, 4.0)
            assert new.shape == ref.shape
            assert np.array_equal(new, ref), f"mismatch {shape} {ps} {gd}"


def test_phsig_batch_size_invariant():
    """Forcing one output row per batch must not change the result."""
    pytest.importorskip('scipy')
    import numpy as np

    _install_osgeo_stub()
    import mintpy.stdproc.slc2ifg.generate_coh_phsig as phsig

    rng = np.random.default_rng(3)
    ifg = (rng.standard_normal((40, 60)) +
           1j * rng.standard_normal((40, 60))).astype(np.complex64)

    default = phsig.estimate_phsig_correlation(ifg, 5, 5, 4.0)
    assert default.shape == (40, 60) and default.dtype == np.float32
    assert default.min() >= 0.0 and default.max() <= 1.0

    old = phsig._BATCH_WINDOW_PIXELS
    try:
        phsig._BATCH_WINDOW_PIXELS = 1       # one output row per batch
        tiny = phsig.estimate_phsig_correlation(ifg, 5, 5, 4.0)
    finally:
        phsig._BATCH_WINDOW_PIXELS = old
    assert np.array_equal(default, tiny)


# ------------------------------------------------------------------------
# P0: Goldstein patch batching + fallback counter
# ------------------------------------------------------------------------
def test_goldstein_patch_batch_default():
    """Without CuPy the patch batch falls back to a fixed 4096."""
    from mintpy.stdproc.slc2ifg.engine.gpu_kernels import _goldstein_patch_batch
    assert _goldstein_patch_batch(32) == 4096


def test_goldstein_patch_batch_uses_free_vram(monkeypatch):
    """With CuPy the batch is sized from the free VRAM (capped at 16384)."""
    from mintpy.stdproc.slc2ifg.engine import gpu_kernels

    class _FakeMemInfo:
        @staticmethod
        def memGetInfo():
            return (1_000_000_000, 8_000_000_000)

    class _FakeRuntime:
        runtime = _FakeMemInfo

    class _FakeCp:
        cuda = _FakeRuntime

    monkeypatch.setattr(gpu_kernels, 'cp', _FakeCp(), raising=False)
    # px = 32*32 = 1024; free=1e9 -> 1e9//4//(1024*16) = 15258 (< 16384 cap)
    assert gpu_kernels._goldstein_patch_batch(32) == 15258
    # a tiny card forces the floor of 256
    monkeypatch.setattr(gpu_kernels, 'cp', type('X', (), {
        'cuda': type('Y', (), {'runtime': type('Z', (), {
            'memGetInfo': staticmethod(lambda: (10_000, 8_000_000_000))})()})()})())
    assert gpu_kernels._goldstein_patch_batch(32) == 256


def test_gpu_fallback_counter_increments():
    """_note_fallback counts fallbacks and is safe without CuPy installed."""
    from mintpy.stdproc.slc2ifg.engine.gpu_kernels import _note_fallback, gpu_fallback_count
    before = gpu_fallback_count()
    _note_fallback('phsig', RuntimeError('boom'))
    assert gpu_fallback_count() == before + 1


# ------------------------------------------------------------------------
# P1: VRAM-aware gpu_workers accounting
# ------------------------------------------------------------------------
def test_gpu_workers_accounted_from_estimates(monkeypatch):
    """gpu_workers = min(gpu_count, gpu_mem_limit_gb // per-task estimate)."""
    from mintpy.stdproc.slc2ifg.engine import resources

    monkeypatch.setattr(resources, 'gpu_available', lambda: True)
    monkeypatch.setattr(resources, 'max_gpu_mem_estimate_gb', lambda: 2.0)

    # 1 GPU, 8 GB -> limit 5.6 GB -> min(1, 5.6//2.0) = 1
    monkeypatch.setattr(resources, 'gpu_memory_gb', lambda: 8.0)
    monkeypatch.setattr(resources, 'gpu_count', lambda: 1)
    plan = resources.build_resource_plan(max_workers=8, gpu='auto')
    assert plan.gpu_enabled and plan.gpu_workers == 1
    assert plan.gpu_mem_estimate_gb == 2.0

    # 4 GPUs, 24 GB each -> limit 16.8 GB -> min(4, 16.8//2.0) = 4
    monkeypatch.setattr(resources, 'gpu_memory_gb', lambda: 24.0)
    monkeypatch.setattr(resources, 'gpu_count', lambda: 4)
    plan = resources.build_resource_plan(max_workers=8, gpu='auto')
    assert plan.gpu_workers == 4


def test_gpu_workers_budget_warns_when_overcommitted(monkeypatch, caplog):
    """A budget below one task's estimate warns and floors at 1 worker."""
    import logging

    from mintpy.stdproc.slc2ifg.engine import resources

    monkeypatch.setattr(resources, 'gpu_available', lambda: True)
    monkeypatch.setattr(resources, 'gpu_memory_gb', lambda: 24.0)
    monkeypatch.setattr(resources, 'gpu_count', lambda: 4)
    monkeypatch.setattr(resources, 'max_gpu_mem_estimate_gb', lambda: 2.0)

    with caplog.at_level(logging.WARNING, logger='mintpy.stdproc.slc2ifg.engine.resources'):
        plan = resources.build_resource_plan(
            max_workers=8, gpu='auto', gpu_mem_limit_gb=1.0)
    assert plan.gpu_workers == 1
    assert 'GPU budget 1.0 GB' in caplog.text


# ------------------------------------------------------------------------
# P1: device round-robin GpuPool
# ------------------------------------------------------------------------
def test_gpu_pool_round_robin_devices():
    """Slots are pinned to devices round-robin; acquire/release gate."""
    from mintpy.stdproc.slc2ifg.engine.resources import GpuPool

    pool = GpuPool(4, device_ids=[0, 1, 0, 1])
    assert [pool.acquire() for _ in range(4)] == [0, 1, 0, 1]
    pool.release()
    assert pool.acquire() == 0

    # default pool: single device 0
    pool2 = GpuPool(2)
    assert [pool2.acquire() for _ in range(2)] == [0, 0]


def test_gpu_pool_context_manager_returns_device():
    from mintpy.stdproc.slc2ifg.engine.resources import GpuPool

    with GpuPool(1, device_ids=[7]) as dev:
        assert dev == 7


def test_gpu_pool_blocks_when_exhausted():
    """A slot is not handed out until released (bounded semaphore)."""
    import threading
    import time

    from mintpy.stdproc.slc2ifg.engine.resources import GpuPool

    pool = GpuPool(1, device_ids=[0])
    assert pool.acquire() == 0
    result = []

    def try_acquire():
        result.append(pool.acquire())

    t = threading.Thread(target=try_acquire)
    t.start()
    time.sleep(0.05)
    assert result == []                 # blocked on the semaphore
    pool.release()
    t.join(timeout=2)
    assert result == [0]


# ------------------------------------------------------------------------
# P2: auto tile_size when GPU is enabled
# ------------------------------------------------------------------------
def _gpu_engine_config(tmp_path):
    from mintpy.stdproc.slc2ifg.engine.config import load_engine_config

    inp = tmp_path / 'input'
    inp.mkdir()
    for d in ('20220105', '20220117'):
        (inp / f'{d}.slc.tif').touch()
    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        'engine.max_workers = 4\n'
        'engine.tools = auto\n')        # engine.gpu defaults to auto
    return load_engine_config(str(cfg))


def test_auto_tile_size_when_gpu_enabled(tmp_path, monkeypatch):
    """GPU on + tile_size unset -> a VRAM-budgeted tile size is picked."""
    from mintpy.stdproc.slc2ifg.engine import gpu_kernels, resources
    from mintpy.stdproc.slc2ifg.engine.engine import Engine

    monkeypatch.setattr(gpu_kernels, 'cupy_available', lambda: True)
    monkeypatch.setattr(resources, 'gpu_memory_gb', lambda: 8.0)

    config = _gpu_engine_config(tmp_path)
    assert config.tile_size is None
    eng = Engine(config)
    # 8 GB -> budget 5.6 GB -> phsig ramp (w=5) fits at 3584 (512..8192)
    assert eng.config.tile_size is not None
    assert 512 <= eng.config.tile_size <= 8192
    assert eng.config.tile_size % 256 == 0
    assert eng._tool_params('phsig_coh')['tile_size'] == eng.config.tile_size


def test_no_auto_tile_when_gpu_disabled(tmp_path, monkeypatch):
    """GPU off -> engine.tile_size stays None (whole-image CPU as before)."""
    from mintpy.stdproc.slc2ifg.engine.engine import Engine

    monkeypatch.setattr('mintpy.stdproc.slc2ifg.engine.gpu_kernels.cupy_available',
                        lambda: True)
    config = _gpu_engine_config(tmp_path)
    config.gpu = 'false'
    eng = Engine(config)
    assert eng.config.tile_size is None
