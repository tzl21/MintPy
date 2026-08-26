#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Coherence-aware interferogram selection for the SBAS network.

Theory (see ``docs/ifgram_selection.md`` for the full derivation)
-----------------------------------------------------------------
Let ``T = {t_1 < ... < t_N}`` be the sorted acquisition dates and
``G' = (T, E')`` the graph of the selected interferograms.  The SBAS
observation equation is

    psi_ij = phi_j - phi_i + eps_ij,

whose design matrix ``H`` is the oriented incidence matrix of ``G'``
(+1 on the slave column, -1 on the master column).

1. **Rank <=> connectivity.**  ``rank(H) = N - c(G')`` where ``c`` is the
   number of connected components.  After removing the reference-date
   column, ``H^T H`` is invertible **iff** ``G'`` is connected.  A unique
   SBAS solution therefore exists iff the interferogram network is
   connected — connectivity is the only hard constraint.

2. **Closure => zero residual, whatever the edge count.**  With temporal
   phase closure (e.g. the tinsaru guarantee) the observed phases are
   cycle-consistent, i.e. ``psi = H phi`` holds exactly for some ``phi``.
   Then the least-squares residual is **zero for any connected** ``G'``:
   redundant interferograms never "fix" anything.  Redundancy buys only
   (a) noise reduction — the covariance of the estimate is
   ``(H^T W H)^-1`` with information weight ``w_ij ~ gamma_ij^2/(1-gamma_ij^2)``,
   so a high-coherence edge reduces variance far more than a low-coherence
   one — and (b) robustness to a single failed (unwrapped) edge.

3. **Consequence.**  The near-optimal strategy is to *span the network
   with the highest-quality edges and then greedily augment to maximise
   the **mean** (average) quality*: this is what this module implements
   (maximum spanning tree + mean-maximising augmentation, with optional
   per-date minimum degree, an edge budget, a quality floor, and bridge
   repair for 2-edge robustness where the candidate set allows).

Algorithm (``select_ifgrams``)
------------------------------
1. ``generate_candidates`` — k temporal nearest neighbours + pairs in
   user-specified temporal windows (e.g. half-year / one-year) + optional
   all pairs within a temporal-baseline cap, optionally filtered by
   perpendicular baseline.  This covers both "sparse k-NN + annual pairs"
   and "fully connected within a window" flavours.
2. Weight every candidate (``temporal_model_weights``,
   ``weights_from_coherence_rasters``, ``quick_coherence_weights`` or a
   mixture).
3. ``select_ifgrams`` —
   a. maximum spanning tree (Kruskal, highest weight first) — guarantees
      connectivity with exactly ``N-1`` edges;
   b. mean-maximising augmentation (every date in >= ``min_degree``
      interferograms; keeps edges at or above the running mean so the
      average coherence is maximised);
   c. optional budget fill with the highest-quality remaining edges;
   d. optional bridge repair (2-edge robustness where possible).
4. ``verify_selection`` — re-checks connectivity and the full-rank
   condition ``rank(H_{-ref}) == N-1`` and reports the result.

Everything below the weight-source functions is pure Python + numpy (no
GDAL required); GDAL/scipy are imported lazily only for the coherence
raster / quick-coherence paths.
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)

DATE_FMT = '%Y%m%d'

# ------------------------------------------------------------------------
# Defaults / parameter registry
# ------------------------------------------------------------------------
#: Default selection parameters (keys mirror ``slc2ifg.ifgram_list.select.*``)
DEFAULT_PARAMS: Dict[str, object] = {
    # candidate generation
    'num_connections': 3,            # k temporal nearest neighbours (0 = off)
    'annual_windows': 'auto',        # 'auto' = derive from the repeat cycle;
                                     # or 'center:tol' pairs, or 'none'/'off'
    'temp_baseline_max': None,       # also include ALL pairs within this many days
    'perp_baseline_max': None,       # metres; requires perp_baselines
    'perp_baseline_file': None,      # two-column 'date bperp' text file
    # weight source: model | coherence | mixed
    'weight_source': 'model',
    # model weights
    'model_tau_days': 90.0,          # temporal decorrelation time constant
    'model_gamma0': 1.0,             # coherence at zero temporal baseline
    # measured weights from existing coherence rasters
    'coh_dir': None,
    'coh_kind': 'phsig',             # phsig | cpx
    'coh_variant': 'filt_mli',       # fullres | mli | filt | filt_mli
    'coh_stat': 'mean',              # mean | median | usable_frac | fisher
    'coh_usable_threshold': 0.3,     # for stat='usable_frac'
    # measured weights from on-the-fly complex coherence on grid-sampled SLCs
    # (sampling: grid x grid windows of block x block px; no whole-image read)
    'quick_window': 5,               # coherence estimation window
    'quick_grid': 12,                # sampling grid per side (12x12 windows)
    'quick_block': 16,               # sampled window size in pixels (16x16)
    'quick_debias': True,            # Touzi (1999) bias correction
    'quick_stat': 'mean',            # mean | median | usable_frac | fisher
    'quick_usable_threshold': 0.3,
    'quick_max_workers': 1,          # threads for on-the-fly coherence
    # selection
    'min_degree': 2,                 # minimum interferograms per date
    'max_pairs': None,               # global edge budget (None = no cap)
    'quality_threshold': 0.0,        # augmentation floor on the weight
    'robust': False,                 # repair bridges -> 2-edge robustness
    'verify': True,                  # rank/connectivity verification
    'reference': None,               # reference date (None = earliest)
    # output / plumbing (set by the caller)
    'slc_dir': None,
    'processor': 'isce3',
    'slc_pattern': None,
    'report_file': None,             # write a JSON report here when set
    'dot_file': None,                # write the network as GraphViz DOT here
}

#: Parameter names read by the engine / tool layer (order = doc order)
SELECT_PARAM_KEYS: Tuple[str, ...] = tuple(DEFAULT_PARAMS)


# ------------------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------------------
def _days_between(d1: str, d2: str) -> int:
    return (datetime.strptime(d2, DATE_FMT)
            - datetime.strptime(d1, DATE_FMT)).days


def _validate_pairs(dates: Sequence[str], pairs: Iterable[Tuple[str, str]]) -> None:
    known = set(dates)
    for a, b in pairs:
        if a not in known or b not in known:
            raise ValueError(
                f"pair {a}-{b} references a date outside the date list")


def parse_annual_windows(value: object) -> Optional[Tuple[Tuple[int, int], ...]]:
    """Parse an explicit ``annual_windows`` spec into ``((center, tol), ...)``.

    ``None`` -> ``None`` (the caller resolves it to ``auto``);
    a tuple/list of ``(center, tol)`` pairs -> tuple;
    a string like ``"182:10,365:15"`` -> tuple;
    a "disable" token (``""`` / ``"none"`` / ``"off"`` / ``"0"``) -> ``()``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('', 'none', 'off', '0'):
            return ()
        out = []
        for chunk in v.split(','):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                center, tol = chunk.split(':')
                out.append((int(center), int(tol)))
            except ValueError:
                logger.warning("ignoring malformed annual window '%s'", chunk)
        if not out:
            raise ValueError(f"no valid annual windows parsed from '{value}'")
        return tuple(out)
    return tuple((int(c), int(t)) for c, t in value)


def _dominant_repeat_interval(dates: Sequence[str]) -> int:
    """Most frequent acquisition gap in days (fallback: median, then 12)."""
    dts = sorted(datetime.strptime(d, DATE_FMT) for d in set(dates))
    gaps = [(b - a).days for a, b in zip(dts, dts[1:]) if (b - a).days > 0]
    if not gaps:
        return 12
    return max(set(gaps), key=gaps.count)


def auto_annual_windows(dates: Sequence[str]) -> Tuple[Tuple[int, int], ...]:
    """Derive half-year / one-year windows from the dominant repeat cycle.

    Centers are snapped to the multiples of the repeat interval nearest to
    182.5 and 365.25 days; the tolerance covers +/- one repeat interval, so
    irregular sampling, missing acquisitions and leap years stay inside the
    window.  Example (12-day cycle): ``((180, 12), (360, 12))``.
    """
    r = _dominant_repeat_interval(dates)
    out = []
    for target in (182.5, 365.25):
        k = max(1, int(round(target / r)))
        out.append((k * r, max(r, 8)))
    return tuple(sorted(set(out)))


def _resolve_annual_windows(
    value: object, dates: Sequence[str]
) -> Tuple[Tuple[int, int], ...]:
    """Resolve the ``annual_windows`` spec against a date list.

    ``None`` / ``'auto'`` -> windows derived from the repeat cycle
    (:func:`auto_annual_windows`); disable tokens -> no windows; otherwise
    the explicit ``center:tol`` spec.
    """
    if value is None:
        return auto_annual_windows(dates)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ('', 'none', 'off', '0'):
            return ()
        if v == 'auto':
            return auto_annual_windows(dates)
    parsed = parse_annual_windows(value)
    return parsed if parsed is not None else auto_annual_windows(dates)


def read_perp_baselines(path: str) -> Dict[str, float]:
    """Read a two-column ``date bperp`` text file into ``{date: bperp}``."""
    out: Dict[str, float] = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                logger.warning("skipping malformed perp-baseline line: %s", line)
    if not out:
        logger.warning("no perpendicular baselines parsed from %s", path)
    return out


# ------------------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------------------
def generate_candidates(
    dates: Sequence[str],
    num_connections: int = 3,
    annual_windows: Optional[Sequence[Tuple[int, int]]] = None,
    temp_baseline_max: Optional[int] = None,
    perp_baseline_max: Optional[float] = None,
    perp_baselines: Optional[Dict[str, float]] = None,
) -> List[Tuple[str, str]]:
    """Build the candidate interferogram set (undirected, ``d1 < d2``).

    The candidate set is the union of:

    * **k temporal nearest neighbours** (``num_connections``) — the sparse
      skeleton that keeps short temporal baselines;
    * **window pairs** (``annual_windows``) — ``None``/``'auto'`` derives
      half-year / one-year windows from the dominant repeat cycle
      (:func:`auto_annual_windows`); an explicit spec like
      ``((182, 10), (365, 15))`` or ``"182:10,365:15"`` overrides it;
      ``'none'`` disables the rule;
    * **all pairs within ``temp_baseline_max`` days** — the "fully
      connected within a window" flavour (small scenes / small N).

    Optionally filtered by a maximum perpendicular baseline.

    Returns a sorted, deduplicated list of ``(date1, date2)``.
    """
    dates = sorted(set(dates))
    n = len(dates)
    if n < 2:
        return []
    windows = _resolve_annual_windows(annual_windows, dates)
    dts = [datetime.strptime(d, DATE_FMT) for d in dates]

    cand: Set[Tuple[str, str]] = set()
    # k temporal nearest neighbours
    if num_connections:
        k = min(int(num_connections), n - 1)
        for i in range(n - 1):
            for j in range(i + 1, min(i + k + 1, n)):
                cand.add((dates[i], dates[j]))
    # window pairs (annual / semi-annual / ...)
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = (dts[j] - dts[i]).days
            if any(abs(dt - c) <= t for c, t in windows):
                cand.add((dates[i], dates[j]))
    # all pairs within a maximum temporal baseline
    if temp_baseline_max:
        for i in range(n - 1):
            for j in range(i + 1, n):
                if (dts[j] - dts[i]).days <= int(temp_baseline_max):
                    cand.add((dates[i], dates[j]))
    # perpendicular baseline filter (applies to every rule above)
    if perp_baseline_max and perp_baselines:
        cand = {
            p for p in cand
            if abs(perp_baselines.get(p[0], 0.0) - perp_baselines.get(p[1], 0.0))
            <= float(perp_baseline_max)
        }
    return sorted(cand)


# ------------------------------------------------------------------------
# Graph primitives
# ------------------------------------------------------------------------
class _UnionFind:
    def __init__(self, nodes: Iterable[str]) -> None:
        self.parent = {d: d for d in nodes}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


def check_connected(dates: Sequence[str], pairs: Iterable[Tuple[str, str]]) -> bool:
    """True if the graph ``(dates, pairs)`` is connected."""
    dates = sorted(set(dates))
    if not dates:
        return True
    uf = _UnionFind(dates)
    for a, b in pairs:
        uf.union(a, b)
    root = uf.find(dates[0])
    return all(uf.find(d) == root for d in dates)


def design_matrix(
    dates: Sequence[str],
    pairs: Iterable[Tuple[str, str]],
    reference: Optional[str] = None,
) -> np.ndarray:
    """SBAS design matrix ``H`` with the reference-date column removed.

    Row for pair ``(master, slave)`` (ordered ``date1 < date2``) encodes
    ``psi = phi_slave - phi_master``: ``-1`` on the master column, ``+1``
    on the slave column.  The reference-date column is dropped, so
    ``H`` has shape ``(M, N-1)`` and, by the rank-connectivity theorem,
    full column rank ``N-1`` iff the interferogram graph is connected.
    """
    dates = sorted(set(dates))
    if reference is None:
        reference = dates[0]
    cols = [d for d in dates if d != reference]
    col_idx = {d: i for i, d in enumerate(cols)}
    pairs = list(pairs)
    H = np.zeros((len(pairs), len(cols)), dtype=np.float64)
    for r, (a, b) in enumerate(pairs):
        if a in col_idx:
            H[r, col_idx[a]] -= 1.0
        if b in col_idx:
            H[r, col_idx[b]] += 1.0
    return H


def _kruskal_max(
    dates: Sequence[str],
    order: Sequence[Tuple[str, str]],
    weights: Dict[Tuple[str, str], float],
) -> List[Tuple[str, str]]:
    """Maximum-weight spanning tree via Kruskal (edges in descending weight).

    ``order`` must already be sorted by descending weight; ``weights`` is
    used only for the summary.  Returns ``N-1`` edges when the candidate
    graph is connected.
    """
    uf = _UnionFind(dates)
    tree: List[Tuple[str, str]] = []
    remaining = len(set(dates)) - 1
    for a, b in order:
        if remaining == 0:
            break
        if uf.union(a, b):
            tree.append((a, b))
            remaining -= 1
    return tree


def _bridges(dates: Sequence[str], edges: Iterable[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Bridges (cut edges) of the graph, iterative Tarjan, O(V + E)."""
    adj: Dict[str, List[str]] = {d: [] for d in dates}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    disc = {d: -1 for d in dates}
    low = {d: -1 for d in dates}
    parent = {d: None for d in dates}
    bridges: List[Tuple[str, str]] = []
    timer = 0
    for root in dates:
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        stack = [(root, iter(adj[root]))]
        while stack:
            u, it = stack[-1]
            descended = False
            for v in it:
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, iter(adj[v])))
                    descended = True
                    break
                elif v != parent[u]:
                    low[u] = min(low[u], disc[v])
            if descended:
                continue
            stack.pop()
            if parent[u] is not None:
                p = parent[u]
                low[p] = min(low[p], low[u])
                if low[u] > disc[p]:
                    bridges.append(tuple(sorted((p, u))))
    return bridges


def _components(
    dates: Sequence[str], edges: Iterable[Tuple[str, str]]
) -> Dict[str, str]:
    """Map each node to a component representative (union-find)."""
    uf = _UnionFind(dates)
    for a, b in edges:
        uf.union(a, b)
    return {d: uf.find(d) for d in dates}


def _repair_bridges(
    dates: Sequence[str],
    selected: Set[Tuple[str, str]],
    order: Sequence[Tuple[str, str]],
) -> Tuple[Set[Tuple[str, str]], int]:
    """Replace each bridge by the highest-weight candidate crossing its cut.

    A bridge is a single point of failure: if its interferogram fails to
    unwrap, the network splits and the SBAS system becomes rank-deficient.
    For every bridge, add the highest-weight *unselected* candidate edge
    whose endpoints lie on opposite sides of the bridge cut.  Iterates
    until no bridge remains or no replacement exists.
    """
    selected = set(selected)
    repaired = 0
    for _ in range(len(selected)):
        bridges = _bridges(dates, selected)
        if not bridges:
            break
        fixed_any = False
        for u, v in bridges:
            comp = _components(dates, selected - {(u, v)})
            if comp[u] == comp[v]:
                continue
            best: Optional[Tuple[str, str]] = None
            for p in order:
                if p in selected:
                    continue
                a, b = p
                if (comp[a] == comp[u] and comp[b] == comp[v]) or (
                        comp[a] == comp[v] and comp[b] == comp[u]):
                    best = p
                    break
            if best is not None:
                selected.add(best)
                repaired += 1
                fixed_any = True
                break
        if not fixed_any:
            break
    return selected, repaired


# ------------------------------------------------------------------------
# Quality weights
# ------------------------------------------------------------------------
def temporal_model_weights(
    dates: Sequence[str],
    pairs: Iterable[Tuple[str, str]],
    tau_days: float = 90.0,
    gamma0: float = 1.0,
    perp_baselines: Optional[Dict[str, float]] = None,
    perp_baseline_max: Optional[float] = None,
) -> Dict[Tuple[str, str], float]:
    """Model-based quality ``w_ij = gamma0 * exp(-dt/tau)``.

    Optionally multiplied by a perpendicular-baseline factor
    ``max(0, 1 - |dB|/B_max)``.  Used as a cold start when no coherence
    data is available; values are clipped to ``[0, 1]``.
    """
    dts = {d: datetime.strptime(d, DATE_FMT) for d in dates}
    tau = float(tau_days)
    out: Dict[Tuple[str, str], float] = {}
    for a, b in pairs:
        dt = (dts[b] - dts[a]).days
        w = float(gamma0) * math.exp(-dt / tau)
        if perp_baselines is not None and perp_baseline_max:
            dB = abs(perp_baselines.get(a, 0.0) - perp_baselines.get(b, 0.0))
            w *= max(0.0, 1.0 - dB / float(perp_baseline_max))
        out[(a, b)] = float(np.clip(w, 0.0, 1.0))
    return out


def aggregate_coherence(
    values: np.ndarray,
    stat: str = 'mean',
    usable_threshold: float = 0.3,
) -> float:
    """Aggregate a coherence raster into a scalar pair quality.

    ``stat``:

    * ``mean``        — mean over valid (``> 0``) pixels
    * ``median``      — robust median over valid pixels
    * ``usable_frac`` — fraction of valid pixels with coherence
      ``>= usable_threshold`` (correlates with the unwrappable area)
    * ``fisher``      — mean Fisher information ``gamma^2/(1-gamma^2)``
      (the natural information weight of an interferogram; unbounded
      range ``>= 0``, so ``quality_threshold`` is in these units too)
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[v > 0]
    if v.size == 0:
        return 0.0
    if stat == 'mean':
        return float(np.mean(v))
    if stat == 'median':
        return float(np.median(v))
    if stat == 'usable_frac':
        return float(np.mean(v >= float(usable_threshold)))
    if stat == 'fisher':
        w = v ** 2 / np.maximum(1e-12, 1.0 - v ** 2)
        return float(np.mean(w))
    raise ValueError(
        f"unknown coherence stat '{stat}', expected mean|median|usable_frac|fisher")


def weights_from_coherence_rasters(
    pairs: Iterable[Tuple[str, str]],
    coh_dir: str,
    kind: str = 'phsig',
    variant: str = 'filt_mli',
    processor: str = 'isce3',
    stat: str = 'mean',
    usable_threshold: float = 0.3,
) -> Dict[Tuple[str, str], Optional[float]]:
    """Pair quality from existing coherence rasters.

    Looks up ``coh_dir/{date1}_{date2}/{variant}_{kind}.coh[.tif]`` (see
    ``mintpy.stdproc.slc2ifg.utils.naming.coh_path``).  Missing rasters yield
    ``None`` (the caller decides: model fallback or weight 0).
    """
    from osgeo import gdal  # lazy

    from .utils.naming import coh_path

    gdal.UseExceptions()
    out: Dict[Tuple[str, str], Optional[float]] = {}
    for a, b in pairs:
        p = coh_path(coh_dir, a, b, variant=variant, kind=kind,
                     processor=processor)
        if not p.exists():
            logger.warning("coherence raster missing for %s_%s: %s", a, b, p)
            out[(a, b)] = None
            continue
        try:
            ds = gdal.Open(str(p))
            arr = ds.GetRasterBand(1).ReadAsArray()
            ds = None
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("failed to read coherence raster %s: %s", p, exc)
            out[(a, b)] = None
            continue
        out[(a, b)] = aggregate_coherence(arr, stat, usable_threshold)
    return out


def _read_sampled_slc(
    path: str,
    grid: int = 12,
    block: int = 16,
    subdataset: Optional[str] = None,
) -> np.ndarray:
    """Read a *grid sample* of an SLC for coherence estimation (no full read).

    The old implementation resampled the WHOLE image down to a coarse grid
    (GDAL ``ReadAsArray(buf_xsize=..., buf_ysize=...)`` still reads every
    source pixel — ~25 s for a 20592x4656 scene).  Coherence-based *pair
    ordering* does not need the full scene, so we read ``grid x grid``
    uniformly-spaced ``block x block`` windows instead: 0.04% of the pixels
    at ~9000x less I/O, with the same ordering power for selection.

    Returns a ``(grid*block, grid*block)`` complex array (the sampled tiles
    are tiled in the same row/col layout as the source grid).

    For OPERA-style GSLC ``.h5`` files, ``subdataset`` (e.g. ``/data/VV``)
    is opened through the NETCDF driver — the same convention as
    ``utils.slc2ifg_utils.open_gdal`` and dolphin's VRT sources.
    """
    from osgeo import gdal  # lazy

    gdal.UseExceptions()
    p = str(path)
    if subdataset and p.lower().endswith(('.h5', '.hdf5')):
        ds = gdal.Open(f'NETCDF:"{p}":"//{str(subdataset).lstrip("/")}"')
    else:
        ds = gdal.Open(p)
    rows, cols = ds.RasterYSize, ds.RasterXSize
    grid = max(1, int(grid))
    # a window may not exceed the source extent (tiny test rasters)
    block = max(1, min(int(block), rows, cols))

    def _read_block(band, y0, x0):
        return band.ReadAsArray(x0, y0, block, block)

    if ds.RasterCount == 2:
        out = np.zeros((grid * block, grid * block), dtype=np.complex64)
        b1, b2 = ds.GetRasterBand(1), ds.GetRasterBand(2)
        for gy in range(grid):
            y0 = int((rows - block) * gy / (grid - 1)) if grid > 1 else 0
            for gx in range(grid):
                x0 = int((cols - block) * gx / (grid - 1)) if grid > 1 else 0
                real = _read_block(b1, y0, x0)
                imag = _read_block(b2, y0, x0)
                out[gy * block:(gy + 1) * block,
                    gx * block:(gx + 1) * block] = real + 1j * imag
        ds = None
        return out.astype(np.complex64)

    out = np.zeros((grid * block, grid * block), dtype=np.complex64)
    b1 = ds.GetRasterBand(1)
    for gy in range(grid):
        y0 = int((rows - block) * gy / (grid - 1)) if grid > 1 else 0
        for gx in range(grid):
            x0 = int((cols - block) * gx / (grid - 1)) if grid > 1 else 0
            win = _read_block(b1, y0, x0)
            if win.dtype not in (np.complex64, np.complex128):
                win = win.astype(np.complex64)
            out[gy * block:(gy + 1) * block,
                gx * block:(gx + 1) * block] = win
    ds = None
    return out


def _boxcar_coherence(slc1: np.ndarray, slc2: np.ndarray,
                      window: int) -> np.ndarray:
    """Complex coherence magnitude (boxcar, scipy correlate) of two SLCs."""
    from scipy.ndimage import correlate  # lazy

    kernel = np.ones((window, window), dtype=np.float32)
    kernel /= kernel.size
    ifg = slc1 * np.conj(slc2)
    s1 = correlate(np.abs(slc1) ** 2, kernel, mode='constant', cval=0.0)
    s2 = correlate(np.abs(slc2) ** 2, kernel, mode='constant', cval=0.0)
    denom = np.sqrt(s1 * s2)
    coh = np.zeros_like(s1)
    valid = denom > 0
    coh[valid] = np.abs(correlate(ifg, kernel, mode='constant',
                                  cval=0.0)[valid]) / denom[valid]
    return np.clip(coh, 0.0, 1.0)


def _debias_coherence(coh: np.ndarray, looks: int) -> np.ndarray:
    """Touzi (1999) magnitude bias correction.

    ``E[gamma_hat] ~= gamma + (1 - gamma)/(2L)``, so
    ``gamma ~= (gamma_hat - 1/(2L)) / (1 - 1/(2L))`` for ``gamma_hat > 1/(2L)``.
    """
    L = max(1.0, float(looks))
    thr = 1.0 / (2.0 * L)
    out = np.zeros_like(coh)
    m = coh > thr
    out[m] = (coh[m] - thr) / (1.0 - thr)
    return out


def quick_coherence_weights(
    dates: Sequence[str],
    pairs: Iterable[Tuple[str, str]],
    slc_dir: str,
    processor: str = 'isce3',
    slc_pattern: Optional[str] = None,
    nlks: int = 8,
    window: int = 5,
    max_pixels: int = 1_048_576,
    grid: int = 12,
    block: int = 16,
    debias: bool = True,
    stat: str = 'mean',
    usable_threshold: float = 0.3,
    subdataset: str = '/data/VV',
    max_workers: int = 1,
) -> Dict[Tuple[str, str], Optional[float]]:
    """Pair quality from complex coherence on *grid-sampled* SLCs.

    This is the "fully connected on a coarse sample" screening: each SLC is
    read as a regular ``grid x grid`` sample of ``block x block`` windows
    (0.04% of the pixels, ~9000x less I/O than the old whole-image
    resampled read), the complex correlation magnitude is estimated with a
    small boxcar, and the per-pair weight is a robust statistic of the map.
    Cheap enough to run over every candidate pair, accurate enough to order
    them by coherence.

    ``grid``/``block`` control the sampling density (default 12x12 windows
    of 16x16 px -> 192x192 sample).  ``nlks``/``max_pixels`` are accepted
    for backward compatibility and no longer drive a whole-image read.

    ``max_workers > 1`` parallelises the per-pair coherence with a thread
    pool (the sampled SLCs are loaded once up front, so the parallel
    section is read-only).
    """
    from osgeo import gdal  # lazy

    from .generate_coh_complex import find_slc_file_by_date
    from .utils.naming import slc_pattern as default_pattern

    gdal.UseExceptions()
    pattern = slc_pattern or default_pattern(processor)
    slc_dirs = [Path(slc_dir)]
    first = next(
        (d for d in dates
         if find_slc_file_by_date(slc_dirs, d, pattern) is not None), None)
    if first is None:
        logger.warning("no SLC files found under %s (pattern %s)",
                       slc_dir, pattern)
        return {(a, b): None for a, b in pairs}
    f0 = find_slc_file_by_date(slc_dirs, first, pattern)
    ds0 = gdal.Open(str(f0))
    rows, cols = ds0.RasterYSize, ds0.RasterXSize
    ds0 = None
    grid = max(1, int(grid))
    block = max(1, int(block))
    logger.info("quick coherence: %dx%d SLC -> %dx%d sample (%dx%d grid of %dx%d blocks, workers=%d)",
                rows, cols, grid * block, grid * block, grid, grid,
                block, block, int(max_workers))

    # load every sampled SLC once (serial, cheap); missing -> None
    cache: Dict[str, Optional[np.ndarray]] = {}
    for d in sorted({x for p in pairs for x in p}):
        f = find_slc_file_by_date(slc_dirs, d, pattern)
        if f is None:
            cache[d] = None
            continue
        cache[d] = _read_sampled_slc(str(f), grid=grid, block=block,
                                     subdataset=subdataset)

    looks = (block * window) ** 2

    def _one(pair: Tuple[str, str]) -> Tuple[Tuple[str, str], Optional[float]]:
        a, b = pair
        s1, s2 = cache.get(a), cache.get(b)
        if s1 is None or s2 is None:
            logger.warning("SLC missing for pair %s-%s", a, b)
            return pair, None
        coh = _boxcar_coherence(s1, s2, int(window))
        if debias:
            coh = _debias_coherence(coh, looks)
        return pair, aggregate_coherence(coh, stat, usable_threshold)

    pairs = list(pairs)
    if int(max_workers) > 1 and len(pairs) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=int(max_workers)) as ex:
            results = list(ex.map(_one, pairs))
    else:
        results = [_one(p) for p in pairs]
    return dict(results)


# ------------------------------------------------------------------------
# Selection
# ------------------------------------------------------------------------
def _mean_weight(selected: Iterable[Tuple[str, str]],
                 weights: Dict[Tuple[str, str], float]) -> float:
    """Mean quality weight of ``selected`` (0.0 when empty)."""
    if not selected:
        return 0.0
    return float(np.mean([weights[p] for p in selected]))


def _max_mean_augment(
    dates: Sequence[str],
    selected: Set[Tuple[str, str]],
    degree: Dict[str, int],
    order: Sequence[Tuple[str, str]],
    weights: Dict[Tuple[str, str], float],
    min_degree: int,
    max_pairs: Optional[int],
    thresh: float,
) -> Tuple[Set[Tuple[str, str]], Dict[str, int]]:
    """Augment the tree to a min-degree, mean-maximising network.

    Two phases, both bounded by the hard ``max_pairs`` edge budget:

    1. **Phase B1 — satisfy ``min_degree`` first** (a hard guarantee): edges
       are added in descending weight whenever an endpoint still has degree
       ``< min_degree``, until every date reaches the target (or the budget /
       the above-threshold candidates run out).  This guarantees the degree
       lower bound is met before anything else.
    2. **Phase B2 — maximise the mean with the remaining budget**: with any
       budget left over, add edges at or above the running mean (they can
       never lower the average), repeating to a fixed point.

    Because degree-filling is done first, a tight ``max_pairs`` budget cannot
    be exhausted by above-mean edges on already-dense dates before
    low-coherence dates are lifted to ``min_degree``.
    """
    selected = set(selected)
    degree = dict(degree)

    def budget_left() -> bool:
        return max_pairs is None or len(selected) < max_pairs

    # --- Phase B1: lift every date to min_degree (highest weight first) ---
    # Two tiers to avoid over-filling already-satisfied dates: first add edges
    # where BOTH endpoints are deficient (raises two dates at once, never
    # overshoots a satisfied date), then edges where one endpoint is deficient.
    for tier in (0, 1):
        while budget_left():
            added_any = False
            for a, b in order:
                if not budget_left():
                    break
                if (a, b) in selected:
                    continue
                if weights[(a, b)] < thresh:
                    continue
                da = degree[a] < min_degree
                db = degree[b] < min_degree
                if not (da or db):
                    continue            # not needed for the degree constraint
                if tier == 0 and not (da and db):
                    continue            # tier 0: both endpoints deficient
                selected.add((a, b))
                degree[a] += 1
                degree[b] += 1
                added_any = True
            if not added_any:
                break

    # --- Phase B2: fill the remaining budget with above-mean edges ---
    while budget_left():
        mu = _mean_weight(selected, weights)
        added_any = False
        for a, b in order:
            if not budget_left():
                break
            if (a, b) in selected:
                continue
            if weights[(a, b)] < thresh:
                continue
            if weights[(a, b)] < mu:
                continue            # below the running mean -> would lower the average
            selected.add((a, b))
            degree[a] += 1
            degree[b] += 1
            added_any = True
        if not added_any:
            break
    return selected, degree


def select_ifgrams(
    dates: Sequence[str],
    candidates: Sequence[Tuple[str, str]],
    weights: Dict[Tuple[str, str], float],
    min_degree: int = 2,
    max_pairs: Optional[int] = None,
    quality_threshold: float = 0.0,
    robust: bool = False,
) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    """Select a connected, coherence-prioritised interferogram network.

    Guarantees
    ----------
    * **Connectivity** — the selected graph spans all dates, so the SBAS
      design matrix ``H_{-ref}`` has full column rank ``N-1`` and
      ``H^T H`` is invertible.  Raised: ``ValueError`` if the candidate
      graph itself is disconnected (no connected subset exists).
    * **Quality priority** — the ``N-1`` tree edges form the
      *maximum-weight spanning tree* (highest total weight among all
      spanning trees).  Augmentation then maximises the *mean* (average)
      edge weight, not the sum: it adds any edge that is at or above the
      running mean coherence, and only the few below-mean edges that are
      forced by the ``min_degree`` lower bound.
    * **min_degree** — every date ends up in at least ``min_degree``
      selected interferograms *when the (above-threshold) candidates allow
      it*; otherwise a warning lists the unmet dates.

    Parameters
    ----------
    dates : sorted ``YYYYMMDD`` list.
    candidates : candidate pairs (see :func:`generate_candidates`).
    weights : ``{(d1, d2): quality}``; missing weights are treated as 0.
    min_degree : minimum interferograms per date (1 = pure spanning tree).
    max_pairs : global edge budget; ``None`` = unbounded.
    quality_threshold : augmentation floor on the weight (the spanning tree
        may still use below-threshold edges — connectivity is a hard
        guarantee, a warning is logged).
    robust : repair bridges with the highest-weight crossing candidate
        (2-edge robustness where the candidate set allows).

    Returns ``(sorted selected pairs, report dict)``.
    """
    dates = sorted(set(dates))
    n = len(dates)
    if n < 2:
        raise ValueError(
            f"select_ifgrams requires at least 2 acquisition dates, got {n}")
    cand = sorted(set(candidates))
    _validate_pairs(dates, cand)
    if len(cand) < n - 1:
        raise ValueError(
            f"candidate network has {len(cand)} edges but at least {n - 1} "
            f"are required for connectivity")

    # resolve weights (accept either orientation of the key)
    w: Dict[Tuple[str, str], float] = {}
    missing = 0
    for p in cand:
        v = weights.get(p, weights.get((p[1], p[0])))
        if v is None:
            missing += 1
            v = 0.0
        w[p] = float(v)
    if missing:
        logger.warning("%d candidate pair(s) have no quality weight; "
                       "treated as 0.0", missing)

    if not check_connected(dates, cand):
        raise ValueError(
            "candidate interferogram network is disconnected — no connected "
            "subset exists. Relax the temporal/perpendicular constraints or "
            "raise num_connections / temp_baseline_max.")

    # deterministic order: weight desc, then lexicographic
    order = sorted(cand, key=lambda p: (-w[p], p[0], p[1]))

    # --- Phase 1: maximum spanning tree (guarantees connectivity) ---------
    tree = _kruskal_max(dates, order, w)
    selected = set(tree)
    degree = {d: 0 for d in dates}
    for a, b in selected:
        degree[a] += 1
        degree[b] += 1

    min_degree = max(0, int(min_degree))
    max_pairs = int(max_pairs) if max_pairs else None
    thresh = float(quality_threshold) if quality_threshold is not None else 0.0

    # --- Phase 2a: max-mean augmentation (degree >= min_degree) -----------
    # Objective: maximise the *average* (mean) coherence of the selected
    # network while keeping every date at degree >= min_degree.  Starting
    # from the max-weight spanning tree, edges are added in descending
    # weight when they either lift a node to min_degree or are not below the
    # current running mean (they cannot lower the average).  This keeps the
    # high-coherence edges and only the few low edges forced by the degree
    # lower bound — see _max_mean_augment.
    if min_degree > 1:
        selected, degree = _max_mean_augment(
            dates, selected, degree, order, w, min_degree, max_pairs, thresh)
        unmet = [d for d in dates if degree[d] < min_degree]
        if unmet:
            logger.warning(
                "min_degree=%d not fully met for %d date(s): %s "
                "(candidate set too sparse or below quality_threshold)",
                min_degree, len(unmet), ', '.join(unmet[:8]))

    # --- Phase 2b: fill the remaining budget with the best edges ----------
    if max_pairs is not None and len(selected) < max_pairs:
        budget = max_pairs - len(selected)
        for p in order:
            if budget <= 0:
                break
            if p in selected:
                continue
            if w[p] < thresh:
                continue
            selected.add(p)
            a, b = p
            degree[a] += 1
            degree[b] += 1
            budget -= 1

    # --- Phase 3: bridge repair (2-edge robustness) -----------------------
    repaired = 0
    if robust and len(selected) < len(cand):
        selected, repaired = _repair_bridges(dates, selected, order)

    # final accounting + verification
    degree = {d: 0 for d in dates}
    for a, b in selected:
        degree[a] += 1
        degree[b] += 1
    if not check_connected(dates, selected):
        raise AssertionError("internal error: selection lost connectivity")

    sw = [w[p] for p in selected]
    report: Dict[str, object] = {
        'n_dates': n,
        'n_candidates': len(cand),
        'n_selected': len(selected),
        'n_tree_edges': len(tree),
        'tree_edges': [f"{a}_{b}" for a, b in sorted(tree)],
        'n_bridge_repairs': repaired,
        'tree_weight_sum': round(float(sum(w[p] for p in tree)), 6),
        'selected_weight_sum': round(float(sum(sw)), 6),
        'selected_weight_mean': round(float(np.mean(sw)), 6) if sw else None,
        'min_selected_weight': round(float(min(sw)), 6) if sw else None,
        'median_selected_weight': round(float(np.median(sw)), 6) if sw else None,
        'min_degree_target': min_degree,
        'min_degree_actual': min(degree.values()),
        'max_degree_actual': max(degree.values()),
        'degree_unmet': [d for d in dates if degree[d] < min_degree],
        'max_pairs': max_pairs,
        'quality_threshold': thresh,
        'robust': bool(robust),
    }
    return sorted(selected), report


def verify_selection(
    dates: Sequence[str],
    pairs: Iterable[Tuple[str, str]],
    reference: Optional[str] = None,
    rank_cap: int = 2000,
    check_rank: bool = True,
) -> Dict[str, object]:
    """Verify connectivity and the SBAS full-rank condition.

    ``rank(H_{-ref}) == N - 1`` iff the graph is connected (rank-connectivity
    theorem).  The rank check is bounded to ``rank_cap`` dates to keep it
    cheap on very large networks (connectivity is always checked) and can be
    disabled entirely with ``check_rank=False``.
    """
    dates = sorted(set(dates))
    pairs = list(pairs)
    connected = check_connected(dates, pairs)
    rank: Optional[int] = None
    full_rank: Optional[bool] = None
    if check_rank and connected and len(dates) <= rank_cap:
        H = design_matrix(dates, pairs, reference=reference)
        rank = int(np.linalg.matrix_rank(H))
        full_rank = rank == len(dates) - 1
    return {
        'n_dates': len(dates),
        'n_pairs': len(pairs),
        'connected': connected,
        'rank': rank,
        'full_rank': full_rank,
    }


# ------------------------------------------------------------------------
# Orchestration
# ------------------------------------------------------------------------
def select_pairs(
    dates: Sequence[str],
    slc_dir: Optional[str] = None,
    processor: str = 'isce3',
    params: Optional[Dict[str, object]] = None,
) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    """Full pipeline: candidates -> weights -> selection -> verification.

    Parameters mirror ``DEFAULT_PARAMS`` (the engine reads the same keys
    from ``slc2ifg.ifgram_list.select.*``).  ``slc_dir`` / ``processor``
    may be given directly or inside ``params``.

    Returns ``(sorted selected pairs, report dict)``.  When
    ``params['report_file']`` is set, the report (with ``{d1}_{d2}`` pair
    keys) is also written to that path as JSON.
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    slc_dir = slc_dir or p.get('slc_dir')
    processor = processor or p.get('processor') or 'isce3'

    dates = sorted(set(dates))
    if len(dates) < 2:
        raise ValueError(
            f"select_pairs requires at least 2 acquisition dates, got {len(dates)}")

    # perpendicular baselines (optional)
    perp_baselines: Optional[Dict[str, float]] = None
    if p.get('perp_baseline_file'):
        perp_baselines = read_perp_baselines(str(p['perp_baseline_file']))

    # 1. candidates
    candidates = generate_candidates(
        dates,
        num_connections=int(p.get('num_connections') or 0),
        annual_windows=p.get('annual_windows'),
        temp_baseline_max=p.get('temp_baseline_max'),
        perp_baseline_max=p.get('perp_baseline_max'),
        perp_baselines=perp_baselines,
    )
    if len(candidates) < len(dates) - 1:
        raise ValueError(
            f"candidate generation produced only {len(candidates)} pair(s) "
            f"for {len(dates)} dates — need at least {len(dates) - 1}. "
            f"Relax num_connections / annual_windows / temp_baseline_max.")

    # 2. weights
    source = str(p.get('weight_source') or 'model').lower()
    if source not in ('model', 'coherence', 'mixed'):
        raise ValueError(
            f"weight_source '{source}' invalid, expected model|coherence|mixed")

    model_w = temporal_model_weights(
        dates, candidates,
        tau_days=float(p.get('model_tau_days') or 90.0),
        gamma0=float(p.get('model_gamma0') or 1.0),
        perp_baselines=perp_baselines,
        perp_baseline_max=p.get('perp_baseline_max'),
    )

    measured_w: Optional[Dict[Tuple[str, str], Optional[float]]] = None
    if source in ('coherence', 'mixed'):
        if p.get('coh_dir'):
            measured_w = weights_from_coherence_rasters(
                candidates,
                coh_dir=str(p['coh_dir']),
                kind=str(p.get('coh_kind') or 'phsig'),
                variant=str(p.get('coh_variant') or 'filt_mli'),
                processor=processor,
                stat=str(p.get('coh_stat') or 'mean'),
                usable_threshold=float(p.get('coh_usable_threshold') or 0.3),
            )
        elif slc_dir:
            measured_w = quick_coherence_weights(
                dates, candidates, str(slc_dir),
                processor=processor,
                slc_pattern=p.get('slc_pattern'),
                nlks=int(p.get('quick_nlks') or 8),
                window=int(p.get('quick_window') or 5),
                max_pixels=int(p.get('quick_max_pixels') or 1_048_576),
                grid=int(p.get('quick_grid') or 12),
                block=int(p.get('quick_block') or 16),
                debias=bool(p.get('quick_debias', True)),
                stat=str(p.get('quick_stat') or 'mean'),
                usable_threshold=float(p.get('quick_usable_threshold') or 0.3),
                max_workers=int(p.get('quick_max_workers') or 1),
            )
        else:
            raise ValueError(
                "weight_source='coherence'/'mixed' requires either "
                "select.coh_dir (existing coherence rasters) or an SLC "
                "directory (on-the-fly quick coherence)")

    if source == 'model':
        weights: Dict[Tuple[str, str], float] = dict(model_w)
        weight_source_used = 'model'
    elif source == 'coherence':
        weights = {
            (a, b): (v if v is not None else 0.0)
            for (a, b), v in (measured_w or {}).items()
        }
        weight_source_used = ('coherence_rasters' if p.get('coh_dir')
                              else 'quick_coherence')
    else:  # mixed: measured takes precedence, model fills the gaps
        weights = {}
        for (a, b), v in (measured_w or {}).items():
            weights[(a, b)] = v if v is not None else model_w[(a, b)]
        weight_source_used = 'mixed'

    # 3. selection
    md = p.get('min_degree')
    min_degree = int(md) if md is not None else DEFAULT_PARAMS['min_degree']
    selected, report = select_ifgrams(
        dates,
        candidates,
        weights,
        min_degree=min_degree,
        max_pairs=p.get('max_pairs'),
        quality_threshold=float(p.get('quality_threshold') or 0.0),
        robust=bool(p.get('robust', False)),
    )

    # 4. verification
    ver = verify_selection(dates, selected, reference=p.get('reference'),
                           check_rank=bool(p.get('verify', True)))
    report['weight_source'] = weight_source_used
    report['connected'] = ver['connected']
    report['rank'] = ver['rank']
    report['full_rank'] = ver['full_rank']
    # parameter snapshot (reproducibility)
    report['params'] = dict(p)
    if not ver['connected']:
        raise AssertionError("selection failed connectivity verification")

    logger.info(
        "select: %d dates, %d candidates -> %d selected (connected=%s, "
        "rank=%s, min_degree=%s, weight_source=%s)",
        report['n_dates'], report['n_candidates'], report['n_selected'],
        ver['connected'], ver['rank'], report['min_degree_actual'],
        weight_source_used)

    # optional JSON report + network DOT
    if p.get('report_file'):
        write_report(report, str(p['report_file']))
    if p.get('dot_file'):
        write_network_dot(
            dates, selected,
            weights={k: v for k, v in weights.items() if v is not None},
            tree_edges=report.get('tree_edges', ()),
            path=str(p['dot_file']),
        )
    return sorted(selected), report


def write_report(report: Dict[str, object], path: str) -> None:
    """Write the selection report as JSON (tuple keys -> ``{d1}_{d2}``)."""
    import os

    safe: Dict[str, object] = {}
    for k, v in report.items():
        if isinstance(k, tuple):
            safe[f"{k[0]}_{k[1]}"] = v
        else:
            safe[str(k)] = v
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(safe, f, indent=2, sort_keys=True)
    logger.info("selection report written to %s", path)


def _edge_key(edge) -> str:
    a, b = edge.split('_', 1) if isinstance(edge, str) else edge
    return f"{a}_{b}"


def write_network_dot(
    dates: Sequence[str],
    pairs: Iterable[Tuple[str, str]],
    weights: Optional[Dict[Tuple[str, str], float]] = None,
    tree_edges: Optional[Iterable] = None,
    path: str = 'network.dot',
) -> str:
    """Write the selected network as a GraphViz DOT file.

    Dates are laid out left-to-right in time order; spanning-tree edges are
    drawn bold, augmentation edges solid, and edge labels show the quality
    weight rounded to 3 decimals when ``weights`` is provided.  ``tree_edges``
    may carry ``(d1, d2)`` tuples or ``"d1_d2"`` strings (as in the report).
    """
    import os

    tree = {_edge_key(e) for e in (tree_edges or ())}
    dates = sorted(dates)
    lines = [
        'digraph ifgram_network {',
        '  rankdir=LR;',
        '  node [shape=point, width=0.05];',
    ]
    for d in dates:
        lines.append(f'  "{d}";')
    for a, b in sorted(pairs):
        style = 'bold' if _edge_key((a, b)) in tree else 'solid'
        label = ''
        if weights is not None and (a, b) in weights:
            label = f', label="{weights[(a, b)]:.3f}"'
        lines.append(f'  "{a}" -> "{b}" [style={style}{label}];')
    lines.append('}')
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    logger.info("network DOT written to %s (%d nodes, %d edges)",
                path, len(dates), len(list(pairs)))
    return path
