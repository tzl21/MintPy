#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Resource management for the MintPy slc2ifg engine.

- GPU detection (cupy / numba.cuda) with graceful CPU fallback;
- host memory / GPU memory budget computation;
- a thread-safe GPU semaphore pool for the local threaded backend.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# GPU detection
# ------------------------------------------------------------------------
def gpu_available() -> bool:
    """Return True if a usable GPU device is actually present."""
    return gpu_count() > 0


def gpu_count() -> int:
    """Number of usable GPUs (0 when none — a CPU box must never report 1)."""
    try:
        import cupy
        return int(cupy.cuda.runtime.getDeviceCount())
    except Exception:
        pass
    try:
        from numba import cuda
        return int(len(cuda.gpus))
    except Exception:
        return 0


# ------------------------------------------------------------------------
# Memory budgets
# ------------------------------------------------------------------------
def host_memory_gb() -> float:
    """Total host memory in GB (fallback 16 GB)."""
    try:
        import psutil
        return psutil.virtual_memory().total / 1e9
    except ImportError:
        pass
    if hasattr(os, 'sysconf'):
        try:
            return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1e9
        except (ValueError, OSError):
            pass
    return 16.0


def gpu_memory_gb() -> float:
    """Total GPU memory in GB (first device, fallback 8 GB)."""
    try:
        import cupy
        return float(cupy.cuda.runtime.memGetInfo()[1]) / 1e9
    except Exception:
        pass
    try:
        from numba import cuda
        dev = cuda.get_current_device()
        return float(dev.total_memory) / 1e9
    except Exception:
        return 8.0


@dataclass
class ResourcePlan:
    """Concurrency limits derived from machine + engine configuration."""

    max_workers: int            # task-level concurrency (host threads)
    mem_limit_gb: float         # engine host-memory budget
    gpu_enabled: bool
    gpu_limit_gb: float         # engine GPU-memory budget
    gpu_workers: int            # number of concurrent GPU tasks
    gpu_mem_estimate_gb: float = 1.0   # per-task VRAM estimate used to size gpu_workers

    def concurrency_for(self, mem_estimate_gb: float) -> int:
        """Task concurrency allowed given a per-task memory estimate."""
        if mem_estimate_gb <= 0:
            return self.max_workers
        by_mem = max(1, int(self.mem_limit_gb // mem_estimate_gb))
        return max(1, min(self.max_workers, by_mem))


def max_gpu_mem_estimate_gb() -> float:
    """Largest declared ``gpu_mem_estimate_gb`` across registered GPU tools.

    This is the number GPU concurrency is budgeted against: launching
    ``gpu_workers`` tasks each needing this much must stay inside
    ``gpu_mem_limit_gb``.  Falls back to 1.0 GB when the tool registry cannot
    be inspected (no GPU tools / import failure).
    """
    try:
        from mintpy.stdproc.slc2ifg.engine.tool import TOOL_REGISTRY
        ests = [
            getattr(cls.resource, 'gpu_mem_estimate_gb', 0.0)
            for cls in TOOL_REGISTRY.values()
            if getattr(cls.resource, 'device', 'cpu') == 'gpu'
        ]
        if ests:
            return max(ests)
    except Exception:
        pass
    return 1.0


def build_resource_plan(
    max_workers: Optional[int] = None,
    mem_limit_gb: Optional[float] = None,
    gpu: Optional[str] = None,          # 'auto' | 'true' | 'false'
    gpu_mem_limit_gb: Optional[float] = None,
) -> ResourcePlan:
    """Build a resource plan from engine configuration values."""
    total_mem = host_memory_gb()
    if max_workers is None:
        max_workers = os.cpu_count() or 1
    if mem_limit_gb is None:
        mem_limit_gb = 0.8 * total_mem

    gpu_on = False
    if gpu == 'true':
        gpu_on = gpu_available()
        if not gpu_on:
            logger.warning("engine.gpu=true but no GPU backend found — falling back to CPU")
    elif gpu == 'auto':
        gpu_on = gpu_available()
        if gpu_on:
            logger.info("GPU backend detected — GPU tasks enabled")
    # 'false' -> off

    if gpu_mem_limit_gb is None:
        gpu_mem_limit_gb = 0.7 * gpu_memory_gb() if gpu_on else 0.0

    gpu_workers = 1
    gpu_est = 1.0
    if gpu_on:
        # Size GPU concurrency against the tools' declared per-task VRAM
        # estimate (the tools declare gpu_mem_estimate_gb=2.0), not a flat
        # 1 GB/task: overcommitting the budget is what makes concurrent GPU
        # tasks OOM.  Tiled runs peak *below* the whole-image estimate, so
        # this is conservative in the tiled case, which is intended.
        gpu_est = max_gpu_mem_estimate_gb()
        gpu_workers = max(1, min(gpu_count(),
                                 int(gpu_mem_limit_gb // max(gpu_est, 0.5))))
        if gpu_workers * gpu_est > gpu_mem_limit_gb:
            logger.warning(
                "GPU budget %.1f GB < %d worker(s) x %.1f GB/task — GPU "
                "concurrency capped at %d", gpu_mem_limit_gb, gpu_workers,
                gpu_est, gpu_workers)

    return ResourcePlan(
        max_workers=max_workers,
        mem_limit_gb=mem_limit_gb,
        gpu_enabled=gpu_on,
        gpu_limit_gb=gpu_mem_limit_gb,
        gpu_workers=gpu_workers,
        gpu_mem_estimate_gb=gpu_est,
    )


# ------------------------------------------------------------------------
# CuPy memory pool cap
# ------------------------------------------------------------------------
def configure_gpu_memory_limit(limit_gb: float) -> None:
    """Bound CuPy's device memory pool to ``limit_gb``.

    Without a cap, CuPy's pool grows towards the whole card; a spike then
    risks exhausting VRAM to the point of disturbing other processes or the
    display driver.  With a cap the allocation fails deterministically at the
    configured budget and the kernel's CPU fallback takes over (see
    ``gpu_kernels``).  No-op when ``limit_gb <= 0`` or CuPy is unavailable.
    """
    if limit_gb <= 0:
        return
    try:
        import cupy as cp
        cp.cuda.set_allocator(cp.cuda.MemoryPool(limit=int(limit_gb * 1e9)))
        logger.info("CuPy device memory pool capped at %.1f GB", limit_gb)
    except Exception as e:
        logger.debug("Could not cap the CuPy memory pool (%s)", e)


# ------------------------------------------------------------------------
# GPU semaphore pool (local threaded backend)
# ------------------------------------------------------------------------
class GpuPool:
    """Thread-safe GPU slot pool for the local threaded scheduler.

    Each slot is bound to one physical device (round-robin over the available
    GPUs), so concurrent GPU tasks spread VRAM across devices instead of all
    landing on device 0.  ``acquire()`` returns the device id assigned to the
    slot; callers should ``cp.cuda.Device(dev).use()`` before running.
    """

    def __init__(self, slots: int = 1,
                 device_ids: Optional[List[int]] = None):
        self._sem = threading.BoundedSemaphore(max(1, slots))
        self._devices = list(device_ids) if device_ids else [0] * max(1, slots)
        self._idx = 0
        self._lock = threading.Lock()

    def acquire(self) -> int:
        """Block until a slot is free; return the slot's device id."""
        self._sem.acquire()
        with self._lock:
            dev = self._devices[self._idx % len(self._devices)]
            self._idx += 1
        return dev

    def release(self) -> None:
        self._sem.release()

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False
