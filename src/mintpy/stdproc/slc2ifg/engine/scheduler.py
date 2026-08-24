#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Dask scheduler adapter for the MintPy slc2ifg engine.

Converts a :class:`~mintpy.stdproc.slc2ifg.engine.dag.TaskGraph` into a Dask task graph
and executes it:

- ``threaded`` (default): ``dask.threaded`` — zero extra dependencies,
  task-level parallelism across interferograms;
- ``distributed``: a ``dask.distributed.Client`` with optional GPU worker
  resources (``resources={'GPU': 1}`` per GPU task).

Dependencies between nodes are encoded as *dummy* delayed arguments so Dask
guarantees a node only runs after its prerequisites.  GPU tasks acquire a
slot from :class:`~mintpy.stdproc.slc2ifg.engine.resources.GpuPool` on the local threaded
backend; in distributed mode GPU concurrency is enforced by dask worker
*resources* instead (the semaphore cannot cross process boundaries).

Note: the ``distributed`` backend is implemented but currently validated only
against a local ``dask.distributed.LocalCluster`` (multi-node clusters are
untested; task accounting is disabled there).
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional

from mintpy.stdproc.slc2ifg.engine.dag import TaskGraph
from mintpy.stdproc.slc2ifg.engine.resources import GpuPool, ResourcePlan

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------
# Task accounting (threaded backend only)
# ------------------------------------------------------------------------
# The threaded scheduler executes every node inside this process, so a
# module-level counter gives exact progress and per-tool statistics.
# Distributed workers keep their own copy of this state, so accounting is
# disabled there and the summary falls back to "N/N tasks OK".
_STATS_LOCK = threading.Lock()
_STATS: Dict = {
    'tracking': False,
    'scheduler': None,
    'total': 0,
    'completed': 0,
    'by_tool': {},
    'seconds_by_tool': {},
    'skipped_by_tool': {},
    'last_bucket': 0,
}


def _init_stats(total: int, scheduler: str) -> None:
    with _STATS_LOCK:
        _STATS.update(
            tracking=(scheduler == 'threaded'),
            scheduler=scheduler,
            total=total,
            completed=0,
            by_tool={},
            seconds_by_tool={},
            skipped_by_tool={},
            last_bucket=0,
        )


def run_stats() -> Dict:
    """Snapshot of task accounting (thread-safe)."""
    with _STATS_LOCK:
        return dict(_STATS)


def _record_completion(node, t0: float) -> None:
    """Count one finished node; log progress at every ~10% milestone."""
    with _STATS_LOCK:
        if not _STATS['tracking']:
            return
        elapsed = time.monotonic() - t0
        _STATS['completed'] += 1
        tool = node.tool.name
        _STATS['by_tool'][tool] = _STATS['by_tool'].get(tool, 0) + 1
        _STATS['seconds_by_tool'][tool] = (
            _STATS['seconds_by_tool'].get(tool, 0.0) + elapsed)
        if getattr(node.ctx, 'skipped', False):
            _STATS['skipped_by_tool'][tool] = (
                _STATS['skipped_by_tool'].get(tool, 0) + 1)
        total, done = _STATS['total'], _STATS['completed']
        step = max(1, total // 10)
        bucket = done // step
        if bucket > _STATS['last_bucket']:
            _STATS['last_bucket'] = bucket
            logger.info("Progress: %d/%d task(s) (%d%%)",
                        done, total, 100 * done // total)


def _fmt_dur(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_avg(seconds: float) -> str:
    """Per-task average: sub-second precision for fast tools."""
    if seconds < 10:
        return f"{seconds:.1f}s"
    return _fmt_dur(seconds)


def _log_run_summary(t0: float, failed: bool) -> None:
    stats = run_stats()
    total = stats['total']
    if stats['tracking']:
        done = stats['completed']
        tools = sorted(stats['by_tool'])
    else:
        # distributed: the local process does not run the nodes
        done = 0 if failed else total
        tools = []
    status = "FAILED" if failed else "finished"
    lines = [
        f"Run {status}: {done}/{total} task(s) in "
        f"{_fmt_dur(time.monotonic() - t0)} (scheduler='{stats['scheduler']}')",
    ]
    if tools:
        parts = []
        for t in tools:
            n = stats['by_tool'][t]
            secs = stats['seconds_by_tool'][t]
            skipped = stats['skipped_by_tool'].get(t, 0)
            ran = n - skipped
            extra = []
            if skipped:
                extra.append(f"{skipped} skipped")
            if ran > 1:
                extra.append(f"avg {_fmt_avg(secs / ran)}/task")
            parts.append(f"{t}={n} ({_fmt_dur(secs)}"
                         + (", " + ", ".join(extra) if extra else "")
                         + ")")
        lines.append(
            "  per tool (actual processing time; tasks run in parallel, so "
            "the sum can exceed wall time):")
        lines.append("    " + ", ".join(parts))
    logger.info("\n%s", "\n".join(lines))


def _select_gpu(dev: int) -> None:
    """Make ``dev`` the calling thread's current CUDA device (thread-local)."""
    try:
        import cupy as cp
        cp.cuda.Device(dev).use()
    except Exception:
        pass


def _gpu_preflight(node) -> bool:
    """VRAM pre-flight check for one GPU node.

    Returns True when the free VRAM covers the tool's declared
    ``gpu_mem_estimate_gb`` (the CuPy pool is reset first so the number is
    real).  On failure the node is degraded to CPU in place (``use_gpu=False``)
    and the caller runs it without holding a GPU slot.

    Advisory only: two tasks can pass the check concurrently and still
    overcommit — the CuPy pool cap (``configure_gpu_memory_limit``) and the
    kernels' own OOM -> CPU fallback catch that race.
    """
    est = getattr(node.tool.resource, 'gpu_mem_estimate_gb', 0.0) or 0.0
    if est <= 0:
        return True
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
        free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
        if free_gb < est:
            node.tool.resource.device = 'cpu'
            node.ctx.params['use_gpu'] = False
            logger.warning(
                "GPU node '%s' degraded to CPU: free VRAM %.1f GB < declared "
                "estimate %.1f GB", node.label, free_gb, est)
            return False
    except Exception as e:
        logger.debug("GPU pre-flight check skipped (%s)", e)
    return True


def _run_node(node_key: str, graph: TaskGraph, gpu_pool: Optional[GpuPool],
              *dep_results) -> Dict:
    """Execute one graph node (used inside the Dask task graph).

    Timing starts only after a GPU slot is acquired, so ``took X`` and the
    per-tool averages reflect actual processing time (not queue wait).  GPU
    nodes first pass the VRAM pre-flight check (degrade to CPU when the free
    VRAM cannot cover the declared estimate, without holding a slot), then
    acquire a slot pinned to one physical device (threaded backend).
    """
    node = graph.nodes[node_key]
    if node.tool.resource.device == 'gpu':
        if not _gpu_preflight(node):
            return _run_timed(node)
        if gpu_pool is not None:
            dev = gpu_pool.acquire()
            try:
                _select_gpu(dev)
                return _run_timed(node)
            finally:
                gpu_pool.release()
    return _run_timed(node)


def _run_timed(node) -> Dict:
    t0 = time.monotonic()
    node.ctx.started = t0  # lets tools append their actual processing time
    try:
        return node.tool.run(node.ctx)
    finally:
        _record_completion(node, t0)


def _terminate_active_work() -> None:
    """Best-effort release of in-flight resources after a failed run.

    The threaded pool's still-queued tasks are cancelled by
    ``_execute_threaded``; this terminates the *running* SNAPHU subprocesses
    so the remaining pool threads finish quickly and the process exits —
    instead of lingering for every multi-minute unwrap.
    """
    try:
        from mintpy.stdproc.slc2ifg.unwrap_ifgram import terminate_active_snaphu
        n = terminate_active_snaphu()
        if n:
            logger.warning("Terminated %d in-flight SNAPHU subprocess(es) "
                           "after run failure", n)
    except Exception:  # noqa: BLE001 - best-effort only
        logger.debug("Failed to terminate in-flight SNAPHU subprocesses",
                     exc_info=True)


def execute_graph(
    graph: TaskGraph,
    scheduler: str = 'threaded',
    plan: Optional[ResourcePlan] = None,
    client_kwargs: Optional[Dict] = None,
) -> Dict[str, Dict]:
    """Execute the whole DAG with Dask.

    Parameters
    ----------
    graph : TaskGraph
        Validated task graph (call ``graph.validate()`` first).
    scheduler : str
        ``'threaded'`` or ``'distributed'``.
    plan : ResourcePlan, optional
        Resource plan controlling concurrency and GPU slots.
    client_kwargs : dict, optional
        Extra kwargs for ``dask.distributed.Client`` (distributed backend).

    Returns
    -------
    dict
        ``{node_key: tool.run() result}`` for every node.
    """
    order = graph.topo_order()
    if not order:
        logger.info("Empty graph — nothing to execute")
        return {}

    if scheduler in (None, '', 'auto'):
        scheduler = 'threaded'

    if plan is None:
        from mintpy.stdproc.slc2ifg.engine.resources import build_resource_plan
        plan = build_resource_plan()

    gpu_pool = None
    if plan.gpu_enabled:
        from mintpy.stdproc.slc2ifg.engine.resources import (
            configure_gpu_memory_limit,
            gpu_count,
        )
        # Cap CuPy's pool so a VRAM spike fails deterministically at the
        # configured budget instead of exhausting the whole card.
        configure_gpu_memory_limit(plan.gpu_limit_gb)
        # Each slot is pinned to one physical device (round-robin), so
        # concurrent GPU tasks spread VRAM across GPUs instead of all landing
        # on device 0.
        n_dev = max(1, gpu_count())
        gpu_pool = GpuPool(
            plan.gpu_workers,
            device_ids=[i % n_dev for i in range(max(1, plan.gpu_workers))])
    logger.info("Executing %d tasks with scheduler='%s', max_workers=%d%s",
                len(order), scheduler, plan.max_workers,
                f", gpu_workers={plan.gpu_workers} "
                f"(devices={gpu_pool._devices})" if gpu_pool else "")

    _init_stats(len(order), scheduler)
    t0 = time.monotonic()
    try:
        if scheduler == 'distributed':
            results = _execute_distributed(graph, order, gpu_pool, client_kwargs)
        else:
            results = _execute_threaded(graph, order, plan, gpu_pool)
    except Exception:
        _log_run_summary(t0, failed=True)
        # Release in-flight resources (running SNAPHU subprocesses; queued
        # dask tasks were already cancelled) so the process can exit promptly.
        _terminate_active_work()
        raise
    _log_run_summary(t0, failed=False)
    return results


# ------------------------------------------------------------------------
# Local threaded backend
# ------------------------------------------------------------------------
def _execute_threaded(graph: TaskGraph, order: List[str], plan: ResourcePlan,
                      gpu_pool: Optional[GpuPool]) -> Dict[str, Dict]:
    from concurrent.futures import ThreadPoolExecutor
    from dask import compute, delayed

    tasks: Dict[str, object] = {}
    for key in order:
        node = graph.nodes[key]
        deps = [tasks[d] for d in node.deps]
        tasks[key] = delayed(_run_node)(key, graph, gpu_pool, *deps)

    # Concurrency: bounded by memory budget & worker count
    max_mem = max(
        (graph.nodes[k].tool.resource.mem_estimate_gb for k in order), default=0.0)
    concurrency = plan.concurrency_for(max_mem)

    futures = [tasks[k] for k in order]
    # Use OUR OWN thread pool instead of dask's process-global one: on a task
    # failure we can cancel the still-queued tasks (shutdown(cancel_futures=True))
    # so the engine exits promptly instead of draining the whole graph —
    # including multi-minute unwraps — in the background (the old behaviour
    # left the process alive for hours after a failure).
    pool = ThreadPoolExecutor(max_workers=concurrency)
    try:
        results = compute(*futures, scheduler='threads',
                          num_workers=concurrency, pool=pool)
        return dict(zip(order, results))
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


# ------------------------------------------------------------------------
# Distributed backend
# ------------------------------------------------------------------------
def _execute_distributed(graph: TaskGraph, order: List[str],
                         gpu_pool: Optional[GpuPool],
                         client_kwargs: Optional[Dict]) -> Dict[str, Dict]:
    from dask.distributed import Client, wait

    kwargs = client_kwargs or {}
    client = kwargs.pop('client', None)
    own_client = client is None
    if own_client:
        client = Client(**kwargs)
    try:
        tasks: Dict[str, object] = {}
        for key in order:
            node = graph.nodes[key]
            deps = [tasks[d] for d in node.deps]
            # GPU slots are enforced by dask worker resources
            # (``resources={'GPU': 1}``), NOT by the client-side GpuPool —
            # the pool is a threading semaphore that cannot be pickled to
            # the workers, so ``None`` is passed to the remote task.
            if node.tool.resource.device == 'gpu':
                tasks[key] = client.submit(
                    _run_node, key, graph, None, *deps,
                    resources={'GPU': 1})
            else:
                tasks[key] = client.submit(_run_node, key, graph, None, *deps)

        wait(list(tasks.values()))
        results = client.gather(list(tasks.values()))
        return dict(zip(order, results))
    finally:
        if own_client:
            client.close()
