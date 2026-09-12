#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
The MintPy slc2ifg engine — DAG construction and orchestration.

Builds a :class:`~mintpy.stdproc.engine.dag.TaskGraph` from an
:class:`~mintpy.stdproc.engine.config.EngineConfig`, covering the same phases as
the legacy sequential pipeline but as a dependency graph that Dask executes
with task-level parallelism (one node per date pair / burst).

Phases:
    Phase 1 (per burst): ifgram_list (eager) -> generate_ifgram [+ complex_coh]
    Phase 2:             stitch (multi-burst) or direct use (single burst)
    Phase 3 (uniform):   multilook -> filter -> phsig_coh -> unwrap

Notes
-----
- ``ifgram_list`` runs eagerly during graph construction because the graph
  topology (date pairs) depends on its output. It is fast and idempotent.
- Complex coherence is an *optional* tool (default off); when enabled it is
  only produced for the ``fullres`` variant and never consumed downstream.
- Single-burst mode skips the stitch/copy step entirely (more efficient).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mintpy.stdproc.engine.chain import resolve_chain
from mintpy.stdproc.engine.config import EngineConfig
from mintpy.stdproc.engine.dag import TaskGraph, TaskNode
from mintpy.stdproc.engine.manifest import (
    Manifest,
    execute_cleanup,
    plan_cleanup,
)
from mintpy.stdproc.engine.scheduler import execute_graph
from mintpy.stdproc.engine.tool import (
    ToolContext,
    available_tools,
    get_tool,
    gpu_tools,
)
from mintpy.stdproc.utils import naming

logger = logging.getLogger(__name__)

_BURST_RE = re.compile(r'^t\d+_\d+_iw\d+$')


def _legacy_opt(cfg, new_key: str, legacy_key: str, kind: str,
                default=None):
    """Read a config option, falling back to a deprecated legacy key.

    Used for engine-injected parameters (e.g. ``unwrap.nlooks``) that are not
    declared in a tool's ``params_spec``: reads ``new_key`` (backend-scoped,
    e.g. ``slc2ifg.unwrap.snaphu.nlooks``) and falls back to ``legacy_key``
    with a deprecation warning when the new key is absent.
    """
    from mintpy.stdproc.engine.config import (
        get_float_opt,
        get_int_opt,
        get_opt,
        has_real_value,
    )

    key = new_key
    if not has_real_value(cfg, new_key) and has_real_value(cfg, legacy_key):
        logger.warning("config key '%s' is deprecated — use '%s' instead",
                       legacy_key, new_key)
        key = legacy_key
    if kind == 'float':
        return get_float_opt(cfg, key, fallback=default)
    if kind == 'int':
        return get_int_opt(cfg, key, fallback=default)
    return get_opt(cfg, key, fallback=default)


def discover_bursts(slc_dir: Path) -> List[Optional[str]]:
    """Scan ``slc_dir`` for burst subdirectories (``tXXX_XXXXXX_iwX``).

    Returns ``[None]`` for single-burst (flat) mode.
    """
    if not slc_dir.is_dir():
        return [None]
    bursts = sorted(
        e.name for e in slc_dir.iterdir()
        if e.is_dir() and _BURST_RE.match(e.name))
    if bursts:
        logger.info("Discovered %d burst(s): %s", len(bursts), bursts)
        return bursts
    return [None]


class Engine:
    """Builds and executes the processing DAG."""

    def __init__(self, config: EngineConfig):
        self.config = config
        self.slc_input = Path(config.slc_input)
        self.processor = config.processor

        self.ifgram_dir = config.ifgram_out_dir
        self.stitched_dir = config.stitched_dir
        self.ml_dir = config.ml_out_dir
        self.filter_dir = config.filter_out_dir
        self.unwrap_dir = config.unwrap_out_dir
        self.engine_dir = config.engine_work_dir

        self.run_multilook = 'multilook' in config.tools
        self.run_filter = 'filter' in config.tools
        self.run_stitch = 'stitch' in config.tools
        self.run_unwrap = 'unwrap' in config.tools

        self._manifest = Manifest(self.engine_dir / 'manifest.json',
                                  use_hash=config.manifest_hash)

        # GPU + engine.tile_size unset: auto-tile so the dominant per-tile GPU
        # peak (the phsig deramping ramp) fits the VRAM budget — the
        # whole-image default would OOM small cards (see gpu_kernels._phsig_gpu).
        if self.config.tile_size is None and self._gpu_enabled():
            ts = self._auto_tile_size()
            if ts:
                self.config.tile_size = ts
                logger.info("GPU enabled & engine.tile_size unset: "
                            "auto-tiling at %d (VRAM-budgeted)", ts)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def build_graph(self) -> TaskGraph:
        g = TaskGraph()

        # --- processing chain (engine.stages / default chain) ---
        chain = resolve_chain(self.config.stages, self.config.tools)
        chain_names = [s.name for s in chain]

        # Mid-chain entry mode: the run starts from *existing* products of a
        # later stage (e.g. filtered interferograms -> unwrap).  It is active
        # whenever the user's selection omits generate_ifgram (engine.stages
        # without it).
        if self.config.stages:
            entry_mode = 'generate_ifgram' not in chain_names
        else:
            entry_mode = 'generate_ifgram' not in self.config.tools

        if entry_mode:
            # crop operates on SLCs and requires generate_ifgram — it cannot
            # be part of a mid-chain entry run (inputs are existing products).
            if 'crop_slc' in chain_names:
                raise ValueError(
                    "engine: 'crop_slc' cannot run in mid-chain entry mode "
                    "(it transforms SLCs and requires generate_ifgram)")
            # complex_coh reads SLCs directly — allowed in entry mode when a
            # usable SLC directory is configured (honours slc2ifg.slc_pattern).
            if 'complex_coh' in chain_names and not self._entry_slcs_available():
                raise ValueError(
                    "engine: 'complex_coh' in mid-chain entry mode needs SLC "
                    "inputs — point slc2ifg.slc_input at a directory "
                    "containing SLC files matching slc2ifg.slc_pattern, or "
                    "drop complex_coh")
            # Pair planning is explicit when the user lists 'ifgram_list' in
            # engine.stages, or hands in a pair list
            # (slc2ifg.ifgram_list.pair_file): its output then defines the
            # pairs (mode / select.* / date filters / verbatim file) even
            # though it is not a DAG node — an explicit pair list is never
            # silently overridden by the product tree.
            from mintpy.stdproc.engine.config import get_opt
            plan_pairs = (
                'ifgram_list' in chain_names
                or bool(get_opt(self.config.raw,
                                'slc2ifg.ifgram_list.pair_file')))
            # Drop the upstream infrastructure stages (SLC crop / pair
            # planning / stitching) — the input products already exist on
            # disk.  complex_coh (per_burst) is kept: in entry mode it
            # computes fullres complex coherence directly from the SLCs.
            chain = [s for s in chain
                     if (s.scope not in ('source', 'plan', 'stitch')
                         or s.name == 'complex_coh')]
            if not chain:
                raise ValueError(
                    "engine: mid-chain entry selected ('generate_ifgram' not "
                    "in engine.stages) but no processing stage remains — "
                    "include at least one of: multilook, filter, phsig_coh, "
                    "unwrap, complex_coh")
            chain_names = [s.name for s in chain]
            logger.info("Processing chain (mid-chain entry): %s",
                        ' -> '.join(chain_names))
        else:
            if 'generate_ifgram' not in chain_names:
                raise ValueError(
                    "The processing chain must include the 'generate_ifgram' "
                    f"stage (got: {chain_names})")
            logger.info("Processing chain: %s", ' -> '.join(chain_names))

        if entry_mode:
            return self._build_entry_graph(g, chain, plan_pairs=plan_pairs)

        # --- optional crop step: transforms the input SLC directory ---
        slc_base = self.slc_input
        has_crop = 'crop_slc' in chain_names
        if has_crop:
            slc_base = self._add_crop_node(g)

        # Read-time crop mode (default): with crop_slc absent, generate_ifgram
        # reads only the slc2ifg.bbox region from the SLCs — no cropped SLC
        # files are stored.  isce2 (radar) SLCs are not georeferenced, so this
        # mode requires the crop_slc stage.
        if not has_crop:
            from mintpy.stdproc.engine.config import get_opt
            bbox_raw = (get_opt(self.config.raw, 'slc2ifg.bbox')
                        or get_opt(self.config.raw, 'slc2ifg.crop_slc.wsen'))
            if bbox_raw and self.processor == 'isce2':
                raise ValueError(
                    "engine: slc2ifg.bbox read-time crop requires geocoded "
                    "SLCs (isce3 GeoTIFF/HDF5) — for isce2 enable the "
                    "'crop_slc' stage instead")

        bursts = discover_bursts(slc_base)
        multi = len(bursts) > 1
        burst_ids: List[Optional[str]] = bursts if multi else [None]
        logger.info("Mode: %s", "multi-burst" if multi else "single-burst")

        # ---------------- Phase 1: per-burst ----------------
        pair_files: Dict[str, Path] = {}
        ifg_nodes: Dict[Tuple[Optional[str], str], str] = {}  # (burst, dp) -> node key
        cpx_nodes: Dict[Tuple[Optional[str], str], str] = {}
        do_cpx = 'complex_coh' in chain_names

        for b in burst_ids:
            slc_dir = slc_base / b if b else slc_base
            ifg_out = self.ifgram_dir / b if b else self.ifgram_dir
            ifg_out.mkdir(parents=True, exist_ok=True)

            # ifgram_list is a graph-construction prerequisite (cheap & idempotent).
            # When cropping is enabled, discover dates from the *input* SLC dir
            # (the cropped dir only exists after the crop step executes; the
            # date sets are identical after filtering).
            scan_dir = self.slc_input if has_crop else slc_dir
            pair_file = self._run_ifgram_list(scan_dir, ifg_out)
            pair_files[b] = pair_file

            pairs = self._read_pairs(pair_file)
            for d1, d2 in pairs:
                dp = f"{d1}_{d2}"
                key = f"generate_ifgram#{b or 'single'}#{dp}"
                node = self._make_generate_ifgram_node(key, b, slc_dir, ifg_out,
                                                       d1, d2, pair_file,
                                                       read_crop=not has_crop)
                g.add_node(node)
                ifg_nodes[(b, dp)] = key
                if has_crop:
                    g.add_edge('crop_slc', key)

                if do_cpx:
                    ckey = f"complex_coh#{b or 'single'}#{dp}"
                    cnode = self._make_complex_coh_node(ckey, b, slc_dir, ifg_out,
                                                        d1, d2, pair_file,
                                                        read_crop=not has_crop)
                    g.add_node(cnode)
                    cpx_nodes[(b, dp)] = ckey
                    if has_crop:
                        g.add_edge('crop_slc', ckey)

        # ---------------- Phase 2: stitch / direct ----------------
        if multi:
            if 'stitch' not in chain_names:
                raise ValueError(
                    "Multi-burst mode requires the 'stitch' stage "
                    "(add 'stitch' to engine.stages)")
            phase3_base = self.stitched_dir
            self._add_stitch_nodes(g, ifg_nodes, cpx_nodes, do_cpx)
        else:
            phase3_base = self.ifgram_dir

        # Pair list for Phase 3: the UNION of all per-burst pair lists.
        # Per-burst date sets can legitimately differ (an acquisition missing
        # in one burst); using only the first burst's list would silently
        # skip those pairs' multilook/filter/phsig/unwrap products.
        phase3_pair_file = self.ifgram_dir / 'ifgram_list.txt'
        if pair_files:
            all_pairs = set()
            for pf in pair_files.values():
                if pf is not None:
                    all_pairs.update(self._read_pairs(pf))
            if not all_pairs:
                all_pairs = set(self._read_pairs(phase3_pair_file))
            if pair_files and len(all_pairs) != len(
                    self._read_pairs(pair_files.get(bursts[0])
                                     or phase3_pair_file)):
                logger.info("Phase 3 uses the union of %d burst pair list(s): "
                            "%d pair(s) (per-burst date sets differ)",
                            len(pair_files), len(all_pairs))
            phase3_pair_file = self.engine_dir / 'phase3_pairs.txt'
            phase3_pair_file.write_text(
                ''.join(f"{d1}-{d2}\n" for d1, d2 in sorted(all_pairs)))

        # ---------------- Phase 3: global chain ----------------
        self._add_uniform_nodes(g, phase3_base, phase3_pair_file, chain)

        self._apply_gpu(g)
        g.validate()
        # Record the effective chain for this run (reproducibility: products
        # can be traced back to the chain version that produced them)
        self._manifest.set_meta(
            chain=chain_names,
            chain_config=self.config.stages,
            engine_tools=self.config.tools,
        )
        return g

    # ------------------------------------------------------------------
    # Mid-chain entry (start from an arbitrary stage's products)
    # ------------------------------------------------------------------
    def _build_entry_graph(self, g: TaskGraph, chain,
                           plan_pairs: bool = False) -> TaskGraph:
        """Build the DAG for mid-chain entry mode (inputs are existing products).

        The date pairs come from, in order of precedence:

        1. **``ifgram_list`` in engine.stages** (or an explicit
           ``ifgram_list.pair_file``) — pair planning is explicit, so it is
           re-run from the SLC acquisition dates and its output defines the
           pairs (``mode`` / ``select.*`` / date filters are honoured),
           even though it is not a DAG node.
        2. **the product tree** — ``input_dir/ifgram_list.txt`` or the
           ``input_dir/{date1}_{date2}/`` directories.  The entry stages then
           consume ``{input_variant}.int[.tif]`` per pair, so the first stage
           depends only on files already on disk.
        3. **the SLC dates** — only for a ``complex_coh``-only chain whose
           product tree is empty/absent: the coherence is computed directly
           from the SLCs, so the pair list is generated from the SLC
           acquisition dates (``ifgram_list``), exactly as in the standard
           pipeline.
        """
        chain_names = [s.name for s in chain]
        variant = self._entry_variant()
        slc_only = all(name == 'complex_coh' for name in chain_names)
        input_root = self._entry_input_root()
        pair_file = None
        pairs: List[Tuple[str, str]] = []
        slc_pairs = False

        # 1) explicit pair planning ('ifgram_list' in engine.stages) wins over
        #    whatever pair list may be left over in the product tree.
        if plan_pairs and self._entry_slcs_available():
            pair_file = self._plan_entry_pairs()
            pairs = self._read_pairs(pair_file)
            if not pairs:
                raise ValueError(
                    f"engine: 'ifgram_list' (engine.stages) selected no pairs "
                    f"from the SLC directory {self.slc_input} — check "
                    f"slc2ifg.ifgram_list.*, slc2ifg.slc_pattern and the date "
                    f"filters")
        elif plan_pairs:
            logger.warning(
                "engine: engine.stages lists 'ifgram_list' but slc_input=%s "
                "has no SLC matching slc2ifg.slc_pattern — keeping the "
                "existing pair list under input_dir=%s", self.slc_input,
                input_root)

        # 2) product tree: an existing {root}/{date1}_{date2}/ tree (or an
        #    ifgram_list.txt) defines the pairs to process.
        if not pairs:
            pair_file = self._find_input_pairs(input_root)
            pairs = self._read_pairs(pair_file) if pair_file else \
                self._pairs_from_dirs(input_root)
            if pairs:
                logger.info("Mid-chain entry: using the existing pair list "
                            "under input_dir=%s (add 'ifgram_list' to "
                            "engine.stages to re-plan the pairs)", input_root)
            # 3) no product tree: for a complex_coh-only chain fall back to
            #    the SLC acquisition dates, since the coherence is computed
            #    from the SLCs and there is nothing else to take them from.
            elif slc_only and self._entry_slcs_available():
                pair_file = self._plan_entry_pairs()
                pairs = self._read_pairs(pair_file)
                if not pairs:
                    raise ValueError(
                        f"engine: no date pairs found in the SLC directory "
                        f"{self.slc_input} (check slc2ifg.slc_pattern / the "
                        f"ifgram_list date filters)")
                input_root = self.ifgram_dir
                slc_pairs = True
            else:
                raise ValueError(
                    f"engine: no date pairs found under input_dir={input_root} "
                    f"(expected {input_root}/{{date1}}_{{date2}}/ directories "
                    f"or an ifgram_list.txt)")

        if slc_pairs:
            logger.info("Mid-chain entry (SLC-based complex coherence): "
                        "slc_input=%s, %d pair(s)", self.slc_input, len(pairs))
        else:
            logger.info("Mid-chain entry: input_dir=%s, input_variant=%s, "
                        "%d pair(s)", input_root, variant, len(pairs))
        self._add_uniform_nodes(g, input_root, pair_file, chain,
                                entry_variant=variant, pairs=pairs)
        self._apply_gpu(g)
        g.validate()
        self._manifest.set_meta(
            chain=[s.name for s in chain],
            chain_config=self.config.stages,
            engine_tools=self.config.tools,
        )
        return g

    def _plan_entry_pairs(self) -> Path:
        """Run ``ifgram_list`` eagerly for mid-chain entry mode.

        Writes ``ifgram_list.txt`` under the engine's ifgram root (the
        canonical pair-list location) and returns its path, so the entry
        stages consume exactly the planned pairs.
        """
        out_dir = self.ifgram_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        return self._run_ifgram_list(self.slc_input, out_dir)

    def _entry_input_root(self) -> Path:
        """Input root for mid-chain entry; default = the unified ifgram tree."""
        raw = self.config.input_dir
        if raw:
            p = Path(raw)
            if not p.is_absolute():
                p = (self.config.work_dir / p).resolve()
            return p
        return self.config.ifgram_out_dir

    def _entry_variant(self) -> str:
        """Input ifg variant for mid-chain entry (default 'fullres')."""
        from mintpy.stdproc.utils.naming import IFG_VARIANTS
        v = self.config.input_variant or 'fullres'
        if v not in IFG_VARIANTS:
            raise ValueError(
                f"engine.input_variant '{v}' is invalid, expected one of "
                f"{IFG_VARIANTS}")
        return v

    @staticmethod
    def _find_input_pairs(input_root: Path) -> Optional[Path]:
        """Pair list inside the input root, or None when absent."""
        pf = input_root / 'ifgram_list.txt'
        return pf if pf.exists() else None

    @staticmethod
    def _pairs_from_dirs(input_root: Path) -> List[Tuple[str, str]]:
        """Discover ``(date1, date2)`` pairs from ``{d1}_{d2}`` subdirectories."""
        from mintpy.stdproc.utils import naming
        pairs = set()
        for d in naming.glob_date_pair_dirs(input_root):
            dp = naming.extract_date_pair(d.name)
            if dp:
                d1, d2 = dp.split('_')
                pairs.add((d1, d2))
        return sorted(pairs)

    # ------------------------------------------------------------------
    # GPU marking
    # ------------------------------------------------------------------
    def _gpu_enabled(self) -> bool:
        from mintpy.stdproc.engine.gpu_kernels import cupy_available
        if self.config.gpu == 'false':
            return False
        return cupy_available()

    def _entry_slcs_available(self) -> bool:
        """True when ``slc_input`` holds SLC files usable by entry-mode
        complex coherence.

        Uses the *configured* ``slc2ifg.slc_pattern`` (falling back to the
        processor default) and the same lookup as
        :func:`generate_ifgram.find_slc_file_by_date` — the top level plus one
        nested level (e.g. ``<date>/yyyymmdd.slc.tif`` or the OPERA GSLC
        ``<date>/tXXX_..._yyyymmdd.h5`` layout).
        """
        d = Path(self.slc_input)
        if not d.is_dir():
            return False
        from mintpy.stdproc.engine.config import get_opt
        pattern = (get_opt(self.config.raw, 'slc2ifg.slc_pattern')
                   or naming.slc_pattern(self.processor))
        return bool(list(d.glob(pattern)) or list(d.glob(f"*/{pattern}")))

    def _auto_tile_size(self) -> Optional[int]:
        """Pick a tile size so the dominant per-tile GPU peak fits the budget.

        The phsig deramping ramp is the largest per-tile device array —
        roughly ``(tile + 2*overlap)^2 * ps_win^2 * 8`` bytes (float64, see
        ``gpu_kernels._phsig_gpu``).  Keep it within 50% of the GPU memory
        budget; the rest covers the gradient arrays, exp_ramp/comp copies and
        the other GPU tools (Goldstein, complex coherence).
        """
        from mintpy.stdproc.engine.config import get_int_opt
        from mintpy.stdproc.engine.resources import gpu_memory_gb
        try:
            limit_gb = self.config.gpu_mem_limit_gb or (0.7 * gpu_memory_gb())
            if limit_gb <= 0:
                return None
            ps_win = get_int_opt(self.config.raw,
                                 'slc2ifg.generate_coh.ps_window_size',
                                 fallback=5) or 5
            if ps_win % 2 == 0:
                ps_win += 1
            overlap = ps_win // 2 + 3      # grad_half + ps_half + 1 (defaults)
            per_px = ps_win * ps_win * 8.0  # float64 ramp bytes per window px
            budget = 0.5 * limit_gb * 1e9
            side = int((budget / per_px) ** 0.5) - 2 * overlap
            tile = max(512, min(8192, side // 256 * 256))
            return tile
        except Exception:
            return None

    def _apply_gpu(self, g: TaskGraph) -> None:
        """Resolve GPU marking from each tool's declared Resource.device.

        Tools opt in with ``Resource(device='gpu')`` (see
        :func:`mintpy.stdproc.engine.tool.gpu_tools`); when the backend is
        unavailable or disabled, they are degraded to CPU at runtime.
        """
        if not self._gpu_enabled():
            for node in g.nodes.values():
                if node.tool.resource.device == 'gpu':
                    node.tool.resource.device = 'cpu'
                    node.ctx.params['use_gpu'] = False
            return
        for node in g.nodes.values():
            if node.tool.resource.device == 'gpu':
                node.ctx.params['use_gpu'] = True
        logger.info("GPU enabled: %d task(s) marked for GPU execution",
                    sum(1 for n in g.nodes.values()
                        if n.tool.resource.device == 'gpu'))

    # ------------------------------------------------------------------
    # Phase 1 builders
    # ------------------------------------------------------------------
    def _add_crop_node(self, g: TaskGraph) -> Path:
        """Add the optional crop_slc node; returns the cropped SLC dir.

        When ``slc2ifg.ifgram_list.start_date`` / ``end_date`` /
        ``exclude_date`` are set, only date-specific files passing the
        filter are cropped (via ``--file-list``); non-date assets (e.g.
        ``static_layers_*.h5``) are always kept.
        """
        cfg = self.config.raw
        from mintpy.stdproc.engine.config import get_opt
        from mintpy.stdproc.ifgram_list import parse_exclude_dates
        from mintpy.stdproc.utils import naming as _naming

        crop_out = get_opt(cfg, 'slc2ifg.crop_slc.output_dir',
                           fallback=str(self.config.work_dir / 'cropped_slc'))
        crop_dir = Path(crop_out)
        if not crop_dir.is_absolute():
            crop_dir = (self.config.work_dir / crop_dir).resolve()

        start_date = get_opt(cfg, 'slc2ifg.ifgram_list.start_date')
        end_date = get_opt(cfg, 'slc2ifg.ifgram_list.end_date')
        exclude_date = get_opt(cfg, 'slc2ifg.ifgram_list.exclude_date')
        ex_dates = parse_exclude_dates(exclude_date)

        inputs = {'slc_dir': self.slc_input}
        # single-burst input: force flat cropped output (no burst nesting)
        if len(discover_bursts(self.slc_input)) <= 1:
            inputs['no_burst_dirs'] = True
        if start_date or end_date or ex_dates:
            # unified SLC pattern; legacy crop_slc.pattern as fallback
            pattern = (get_opt(cfg, 'slc2ifg.slc_pattern')
                       or get_opt(cfg, 'slc2ifg.crop_slc.pattern')
                       or _naming.slc_pattern(self.processor))
            candidates = sorted(self.slc_input.glob(f"**/{pattern}"))
            keep = []
            skipped = []
            for p in candidates:
                m = re.search(r'(20\d{6})', p.name)
                if not m:
                    # Non-date assets (e.g. static_layers_*.h5 / geometry):
                    # the date filter only applies to date-specific SLCs,
                    # so always keep them.
                    keep.append(str(p))
                    continue
                d = m.group(1)
                if start_date and d < str(start_date):
                    skipped.append(str(p))
                    continue
                if end_date and d > str(end_date):
                    skipped.append(str(p))
                    continue
                if d in ex_dates:
                    skipped.append(str(p))
                    continue
                keep.append(str(p))
            if not keep:
                raise ValueError(
                    f"No SLC files match the crop date range "
                    f"[{start_date or '-inf'}, {end_date or '+inf'}] "
                    f"excl {ex_dates} under {self.slc_input}")
            list_file = self.engine_dir / 'crop_file_list.txt'
            list_file.write_text('\n'.join(keep) + '\n')
            inputs['file_list'] = list_file
            logger.info("crop: %d file(s) in date range [%s, %s] excl %s "
                        "(%d skipped)",
                        len(keep), start_date or '-inf', end_date or '+inf',
                        ','.join(ex_dates) or 'none', len(skipped))

        tool = get_tool('crop_slc')
        ctx = ToolContext(
            tool_name=tool.name,
            inputs=inputs,
            outputs={'slc_dir': crop_dir},
            params=self._tool_params('crop_slc'),
            work_dir=self.engine_dir,
        )
        g.add_node(TaskNode(key='crop_slc', tool=tool,
                            label='crop_slc', ctx=ctx))
        return crop_dir

    def _run_ifgram_list(self, slc_dir: Path, out_dir: Path) -> Path:
        """Eagerly generate the pair list (graph topology depends on it)."""
        from mintpy.stdproc.ifgram_list import (
            filter_date_list,
            generate_pairs,
            get_date_list,
            write_pair_list,
        )
        cfg = self.config.raw
        from mintpy.stdproc.engine.config import get_int_opt, get_opt

        mode = get_opt(cfg, 'slc2ifg.ifgram_list.mode', fallback='sequential') or 'sequential'
        start_date = get_opt(cfg, 'slc2ifg.ifgram_list.start_date')
        end_date = get_opt(cfg, 'slc2ifg.ifgram_list.end_date')
        exclude_date = get_opt(cfg, 'slc2ifg.ifgram_list.exclude_date')

        # An explicit pair file always wins, so that setting it is never
        # silently ignored (mode='file' is the documented way).
        pair_file_cfg = get_opt(cfg, 'slc2ifg.ifgram_list.pair_file')
        if pair_file_cfg:
            if mode != 'file':
                logger.info(
                    "ifgram_list: slc2ifg.ifgram_list.pair_file is set — using "
                    "it instead of mode=%s (set "
                    "slc2ifg.ifgram_list.mode = file to make this explicit); "
                    "select.* / num_connections do not apply", mode)
            mode = 'file'
        elif mode == 'file':
            raise ValueError(
                "ifgram_list mode=file requires slc2ifg.ifgram_list.pair_file "
                "(a text file with one 'YYYYMMDD-YYYYMMDD' pair per line)")

        all_dates = get_date_list(str(slc_dir))
        dates = filter_date_list(all_dates, start_date=start_date,
                                 end_date=end_date, exclude_date=exclude_date)
        pair_file = out_dir / 'ifgram_list.txt'

        if mode == 'file':
            # Explicit pair list, used verbatim: no date filtering, no
            # candidate generation.  Every referenced date must have an SLC.
            path = Path(pair_file_cfg)
            if not path.is_absolute():
                path = (self.config.work_dir / path).resolve()
            if start_date or end_date or exclude_date:
                logger.warning(
                    "ifgram_list: start_date/end_date/exclude_date are ignored "
                    "in mode=file — the pair file is used verbatim")
            pairs = generate_pairs(all_dates, 'file', None,
                                   select_params={'pair_file': str(path)})
        elif mode == 'select':
            # Coherence-aware connected selection (see select_ifgrams.py).
            # Runs eagerly because the graph topology depends on it; the
            # selection guarantees connectivity, hence full SBAS rank.
            from mintpy.stdproc.select_ifgrams import select_pairs
            params = self._select_params(cfg)
            params['slc_dir'] = str(slc_dir)
            params['processor'] = self.processor
            pairs, report = select_pairs(dates, params=params)
            logger.info(
                "ifgram_list (select): %d dates -> %d candidate(s) -> %d "
                "selected pair(s) (connected=%s, rank=%s, min_degree=%s, "
                "weight_source=%s)",
                report['n_dates'], report['n_candidates'],
                report['n_selected'], report.get('connected'),
                report.get('rank'), report.get('min_degree_actual'),
                report.get('weight_source'))
        else:
            nconn = get_int_opt(cfg, 'slc2ifg.ifgram_list.num_connections',
                                fallback=5) or 5
            # annual_windows is the canonical knob for annual/one-year pairs
            # in ALL modes (the legacy slc2ifg.ifgram_list.oneyear_interferograms
            # key was removed)
            aw = get_opt(cfg, 'slc2ifg.ifgram_list.select.annual_windows')
            pairs = generate_pairs(
                dates, mode, nconn, oneyear_range=None,
                select_params={'annual_windows': aw} if aw is not None else None)

        write_pair_list(pairs, pair_file)
        return pair_file

    def _select_params(self, cfg) -> dict:
        """Read the select-mode params (``select.*`` keys + pipeline AOI)."""
        from mintpy.stdproc.engine.config import (
            get_bool_opt,
            get_float_opt,
            get_int_opt,
            get_opt,
            has_real_value,
        )

        p: dict = {}
        nconn = get_int_opt(cfg, 'slc2ifg.ifgram_list.num_connections',
                            fallback=None)
        # select-mode k-NN skeleton default is 3 (vs 5 for sequential)
        p['num_connections'] = nconn if nconn is not None else 3
        readers = {
            'str': get_opt,
            'int': get_int_opt,
            'float': get_float_opt,
            'bool': get_bool_opt,
        }
        for name, kind in [
                ('annual_windows', 'str'),
                ('temp_baseline_max', 'int'),
                ('perp_baseline_max', 'float'),
                ('perp_baseline_file', 'str'),
                ('weight_source', 'str'),
                ('slc_pattern', 'str'),
                ('model_tau_days', 'float'),
                ('model_gamma0', 'float'),
                ('coh_dir', 'str'),
                ('coh_kind', 'str'),
                ('coh_variant', 'str'),
                ('coh_stat', 'str'),
                ('coh_usable_threshold', 'float'),
                ('quick_window', 'int'),
                ('quick_nlks', 'int'),
                ('quick_max_pixels', 'int'),
                ('quick_grid', 'int'),
                ('quick_block', 'int'),
                ('quick_max_workers', 'int'),
                ('quick_debias', 'bool'),
                ('quick_stat', 'str'),
                ('quick_usable_threshold', 'float'),
                ('min_degree', 'int'),
                ('max_pairs', 'int'),
                ('quality_threshold', 'float'),
                ('robust', 'bool'),
                ('verify', 'bool'),
        ]:
            if name == 'slc_pattern':
                # unified pipeline key; legacy select.slc_pattern as fallback
                key = ('slc2ifg.slc_pattern'
                       if has_real_value(cfg, 'slc2ifg.slc_pattern')
                       else 'slc2ifg.ifgram_list.select.slc_pattern')
            else:
                key = f'slc2ifg.ifgram_list.select.{name}'
            val = readers[kind](cfg, key, fallback=None)
            if val is not None:
                p[name] = val
        # AOI: the unified pipeline bbox is not a select.* key, but the quick
        # coherence uses it to read ONLY the AOI window of each SLC (the AOI
        # is already cropped) instead of the whole scene.
        wsen = (get_opt(cfg, 'slc2ifg.bbox', fallback=None)
                or get_opt(cfg, 'slc2ifg.crop_slc.wsen', fallback=None))
        if wsen:
            p['bbox'] = wsen
            p['bbox_buffer'] = get_float_opt(
                cfg, 'slc2ifg.bbox_buffer',
                fallback=get_float_opt(cfg, 'slc2ifg.crop_slc.buffer',
                                       fallback=0.0)) or 0.0
        # config keys select.report / select.dot (param names report_file/dot_file)
        rpt = get_opt(cfg, 'slc2ifg.ifgram_list.select.report', fallback=None)
        if rpt:
            p['report_file'] = rpt
        dot = get_opt(cfg, 'slc2ifg.ifgram_list.select.dot', fallback=None)
        if dot:
            p['dot_file'] = dot
        for out_key in ('report_file', 'dot_file'):
            if p.get(out_key):
                rp = Path(p[out_key])
                if not rp.is_absolute():
                    rp = (self.config.work_dir / rp).resolve()
                p[out_key] = str(rp)
        return p

    @staticmethod
    def _read_pairs(pair_file: Path) -> List[Tuple[str, str]]:
        pairs = []
        for line in pair_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('-')
            if len(parts) != 2 or not (
                    len(parts[0]) == 8 and len(parts[1]) == 8
                    and parts[0].isdigit() and parts[1].isdigit()):
                logger.warning("Skipping malformed pair line: %r", line)
                continue
            pairs.append((parts[0], parts[1]))
        return pairs

    def _make_generate_ifgram_node(self, key: str, burst: Optional[str],
                                   slc_dir: Path, out_dir: Path,
                                   d1: str, d2: str, pair_file: Path,
                                   read_crop: bool = True) -> TaskNode:
        """Build the per-pair generate_ifgram node.

        ``read_crop`` enables the read-time crop: the slc2ifg.bbox params are
        kept so the tool materialises only the bbox region of the SLCs.  When
        False (crop_slc stage active) the SLCs are already cropped and the
        bbox params are dropped to avoid double windowing.
        """
        tool = get_tool('generate_ifgram')
        pair_out = out_dir / f"{d1}_{d2}"
        ifg_path = naming.ifg_path(out_dir, d1, d2, 'fullres', self.processor)
        params = self._tool_params('generate_ifgram')
        if not read_crop:
            params.pop('bbox', None)
            params.pop('bbox_buffer', None)
        ctx = ToolContext(
            tool_name=tool.name,
            inputs={
                'pairs_file': pair_file,
                'slc_dir': slc_dir,
                'date1': d1,
                'date2': d2,
                'burst': burst,
            },
            outputs={'ifg': ifg_path, 'pair_dir': pair_out},
            params=params,
            work_dir=self.engine_dir,
        )
        return TaskNode(key=key, tool=tool, label=f"generate_ifgram {d1}_{d2}", ctx=ctx)

    def _make_complex_coh_node(self, key: str, burst: Optional[str],
                               slc_dir: Path, out_dir: Path,
                               d1: str, d2: str, pair_file: Path,
                               read_crop: bool = True) -> TaskNode:
        """Build the per-pair complex-coherence node.

        ``read_crop`` enables the read-time AOI crop (same convention as
        ``_make_generate_ifgram_node``): when False (crop_slc stage active)
        the SLCs are already cropped and the bbox params are dropped.
        """
        tool = get_tool('complex_coh')
        coh_path = naming.coh_path(out_dir, d1, d2, 'fullres', 'cpx', self.processor)
        params = self._tool_params('complex_coh')
        if not read_crop:
            params.pop('bbox', None)
            params.pop('bbox_buffer', None)
        ctx = ToolContext(
            tool_name=tool.name,
            inputs={
                'pairs_file': pair_file,
                'slc_dir': slc_dir,
                'date1': d1,
                'date2': d2,
                'burst': burst,
            },
            outputs={'coh': coh_path},
            params=params,
            work_dir=self.engine_dir,
        )
        return TaskNode(key=key, tool=tool, label=f"complex_coh {d1}_{d2}", ctx=ctx)

    # ------------------------------------------------------------------
    # Phase 2 builder
    # ------------------------------------------------------------------
    def _add_stitch_nodes(self, g: TaskGraph,
                          ifg_nodes: Dict[Tuple[Optional[str], str], str],
                          cpx_nodes: Dict[Tuple[Optional[str], str], str],
                          do_cpx: bool) -> None:
        """One stitch node per (date pair, file type); collects all bursts."""
        # Sorting guarantees a deterministic fan-in order: set iteration order
        # is randomized by the process hash, while stitch's overlap-region
        # overwrite (method='last') depends on input order → must be fixed
        burst_keys = sorted({b for (b, _dp) in ifg_nodes})
        file_types: List[str] = [naming.int_ext(self.processor)]
        if do_cpx:
            file_types.append(naming.coh_ext(self.processor, 'cpx'))

        # group by date pair
        dp_set = sorted({dp for (_b, dp) in ifg_nodes})
        for dp in dp_set:
            d1, d2 = dp.split('_')
            for ftype in file_types:
                key = f"stitch#{dp}#{ftype.lstrip('.').replace('.', '_')}"
                tool = get_tool('stitch')
                if '.cpx' in ftype:
                    src_keys = [cpx_nodes[(b, dp)] for b in burst_keys if (b, dp) in cpx_nodes]
                    src_paths = [g.nodes[k].ctx.outputs['coh'] for k in src_keys]
                else:
                    missing = [b for b in burst_keys if (b, dp) not in ifg_nodes]
                    if missing:
                        logger.warning(
                            "stitch %s: missing burst(s) %s — the stitched "
                            "product will be PARTIAL (per-burst date sets "
                            "differ)", dp, ','.join(str(b) for b in missing))
                    src_keys = [ifg_nodes[(b, dp)] for b in burst_keys if (b, dp) in ifg_nodes]
                    src_paths = [g.nodes[k].ctx.outputs['ifg'] for k in src_keys]
                out_path = self.stitched_dir / dp / f"fullres{ftype}"
                ctx = ToolContext(
                    tool_name=tool.name,
                    inputs={'file_list': src_paths, 'date1': d1, 'date2': d2},
                    outputs={'stitched': out_path},
                    params=self._tool_params('stitch'),
                    work_dir=self.engine_dir,
                )
                node = TaskNode(key=key, tool=tool,
                                label=f"stitch {dp} {ftype}", ctx=ctx)
                g.add_node(node)
                for src in src_keys:
                    g.add_edge(src, key)

    # ------------------------------------------------------------------
    # Phase 3 builder
    # ------------------------------------------------------------------
    def _add_uniform_nodes(self, g: TaskGraph, base_dir: Path,
                           pair_file: Optional[Path],
                           chain,
                           entry_variant: Optional[str] = None,
                           pairs: Optional[List[Tuple[str, str]]] = None) -> None:
        """Global per-pair chain: walk the stages in order (multilook →
        filter → phsig_coh → unwrap → atmosphere ...), threading the
        interferogram variant through the chain.

        Parameters
        ----------
        entry_variant : str or None
            When set (mid-chain entry mode), the chain starts from this input
            variant and the first stage has no upstream node — its input file
            ``{base_dir}/{d1}_{d2}/{entry_variant}.int[.tif]`` is expected to
            already exist.  ``None`` = the standard full pipeline (the first
            stage consumes the generate_ifgram/stitch node output).
        pairs : list of (date1, date2) or None
            Explicit pair list (used by mid-chain entry when the input tree
            has no ``ifgram_list.txt``).
        """
        # Date pairs come from the pair list (per-burst for multi-burst mode)
        if pairs is None:
            if pair_file is None:
                pair_file = base_dir / 'ifgram_list.txt'
            if not pair_file.exists():
                pair_file = self.ifgram_dir / 'ifgram_list.txt'
            pairs = self._read_pairs(pair_file)

        from mintpy.stdproc.engine.config import get_opt
        chain_has = {s.name for s in chain}
        #: unwrap coherence type: auto | complex | phsig | none
        unwrap_coh_cfg = get_opt(self.config.raw, 'slc2ifg.unwrap.coh_type',
                                 fallback='auto')
        #: External coherence input for SNAPHU weighting: reuse coherence
        #: rasters produced elsewhere (e.g. ISCE2 or a different filter /
        #: multilook) by pointing slc2ifg.unwrap.coh_dir (+ coh_pattern) at
        #: them.  Lookup order per pair {d1}_{d2}:
        #:   coh_dir/{d1}_{d2}/{coh_pattern}   (engine product-tree layout)
        #:   coh_dir/{coh_pattern}             (flat fallback)
        coh_dir_cfg = get_opt(self.config.raw, 'slc2ifg.unwrap.coh_dir',
                              fallback=None)
        coh_pat_cfg = get_opt(self.config.raw, 'slc2ifg.unwrap.coh_pattern',
                              fallback='*.tif')

        for d1, d2 in pairs:
            dp = f"{d1}_{d2}"
            ext = naming.int_ext(self.processor)

            # --- current ifg source (variant, stage base dir, node key) ---
            cur_variant = entry_variant if entry_variant is not None else 'fullres'
            cur_base = base_dir
            cur_path = naming.ifg_path(base_dir, d1, d2, cur_variant,
                                       self.processor)
            if entry_variant is not None:
                # mid-chain entry: no upstream node — the file already exists
                cur_node = None
            else:
                # source node: stitch node (multi-burst) or generate_ifgram node
                st_key = f"stitch#{dp}#{ext.lstrip('.').replace('.', '_')}"
                if st_key in g.nodes:
                    cur_node = st_key
                elif f"generate_ifgram#single#{dp}" in g.nodes:
                    cur_node = f"generate_ifgram#single#{dp}"
                else:
                    raise ValueError(f"No source node found for date pair {dp}")

            ph_out: Optional[Tuple[Path, str]] = None   # (path, node) of phsig
            unw_out: Optional[Tuple[Path, str]] = None  # (path, node) of unwrap

            # --- complex coherence source (fullres-only) ---
            # per-burst products are stitched in multi-burst mode; single
            # burst uses the per-burst product directly.  Mid-chain entry
            # (entry_variant set) creates the fullres complex-coherence node
            # here, before the stage walk, so unwrap's coherence resolution
            # below already sees it.
            cpx_path: Optional[Path] = None
            cpx_node: Optional[str] = None
            if 'complex_coh' in chain_has:
                cpx_ext = naming.coh_ext(self.processor, 'cpx')
                st_key = f"stitch#{dp}#{cpx_ext.lstrip('.').replace('.', '_')}"
                if st_key in g.nodes:
                    cpx_node = st_key
                    cpx_path = g.nodes[st_key].ctx.outputs['stitched']
                elif f"complex_coh#single#{dp}" in g.nodes:
                    cpx_node = f"complex_coh#single#{dp}"
                    cpx_path = g.nodes[cpx_node].ctx.outputs['coh']
                elif f"complex_coh#{dp}" in g.nodes:
                    cpx_node = f"complex_coh#{dp}"
                    cpx_path = g.nodes[cpx_node].ctx.outputs['coh']
                elif entry_variant is not None:
                    ckey = f"complex_coh#{dp}"
                    cout = naming.coh_path(self.ifgram_dir, d1, d2, 'fullres',
                                           'cpx', self.processor)
                    ctx = ToolContext(
                        tool_name='complex_coh',
                        inputs={'pairs_file': None,
                                'slc_dir': self.slc_input,
                                'date1': d1, 'date2': d2,
                                'burst': None},
                        outputs={'coh': cout},
                        params=self._tool_params('complex_coh'),
                        work_dir=self.engine_dir,
                    )
                    g.add_node(TaskNode(key=ckey, tool=get_tool('complex_coh'),
                                        label=f"complex_coh {dp}", ctx=ctx))
                    cpx_node, cpx_path = ckey, cout

            for stage in chain:
                sname = stage.name
                if sname == 'multilook':
                    out_variant = naming.next_variant(cur_variant, 'multilook')
                    mkey = f"multilook#{dp}"
                    mout = naming.ifg_path(self.ml_dir, d1, d2, out_variant,
                                           self.processor)
                    ctx = ToolContext(
                        tool_name='multilook',
                        inputs={'ifg': cur_path, 'date1': d1, 'date2': d2},
                        outputs={'ifg': mout},
                        params=self._tool_params('multilook'),
                        work_dir=self.engine_dir,
                    )
                    g.add_node(TaskNode(key=mkey, tool=get_tool('multilook'),
                                        label=f"multilook {dp}", ctx=ctx))
                    if cur_node is not None:
                        g.add_edge(cur_node, mkey)
                    cur_variant, cur_node, cur_path, cur_base = \
                        out_variant, mkey, mout, self.ml_dir

                elif sname == 'filter':
                    out_variant = naming.next_variant(cur_variant, 'filter')
                    fkey = f"filter#{dp}"
                    fout = naming.ifg_path(self.filter_dir, d1, d2, out_variant,
                                           self.processor)
                    ctx = ToolContext(
                        tool_name='filter',
                        inputs={'ifg': cur_path, 'date1': d1, 'date2': d2},
                        outputs={'ifg': fout},
                        params=self._tool_params('filter'),
                        work_dir=self.engine_dir,
                    )
                    g.add_node(TaskNode(key=fkey, tool=get_tool('filter'),
                                        label=f"filter {dp}", ctx=ctx))
                    if cur_node is not None:
                        g.add_edge(cur_node, fkey)
                    cur_variant, cur_node, cur_path, cur_base = \
                        out_variant, fkey, fout, self.filter_dir

                elif sname == 'phsig_coh':
                    # phase-sigma coherence lives in the same stage dir as
                    # the ifg it derives from
                    phkey = f"phsig_coh#{dp}"
                    ph_out_path = naming.coh_path(cur_base, d1, d2, cur_variant,
                                                  'phsig', self.processor)
                    ctx = ToolContext(
                        tool_name='phsig_coh',
                        inputs={'ifg': cur_path, 'date1': d1, 'date2': d2},
                        outputs={'coh': ph_out_path},
                        params=self._tool_params('phsig_coh'),
                        work_dir=self.engine_dir,
                    )
                    g.add_node(TaskNode(key=phkey, tool=get_tool('phsig_coh'),
                                        label=f"phsig_coh {dp}", ctx=ctx))
                    if cur_node is not None:
                        g.add_edge(cur_node, phkey)
                    ph_out = (ph_out_path, phkey)

                elif sname == 'unwrap':
                    ukey = f"unwrap#{dp}"
                    u_out = naming.unw_path(self.unwrap_dir, d1, d2,
                                            cur_variant, self.processor)
                    c_out = naming.conncomp_path(self.unwrap_dir, d1, d2,
                                                 cur_variant, self.processor)

                    # Resolve the coherence input from slc2ifg.unwrap.coh_type
                    # (auto|complex|phsig|none).  An explicit type validates
                    # the chain, but when the producing stage is *not* in the
                    # chain (mid-chain entry / "run only unwrap") a matching
                    # coherence raster already on disk is reused instead of
                    # forcing regeneration.  'none' (or 'auto' with no
                    # coherence available) runs SNAPHU with weight 1 (uniform).
                    coh_cfg = str(unwrap_coh_cfg or 'auto').lower()
                    # expected on-disk rasters (used as the fallback below)
                    phsig_existing = naming.coh_path(
                        cur_base, d1, d2, cur_variant, 'phsig', self.processor)
                    cpx_existing = naming.coh_path(
                        self.ifgram_dir, d1, d2, 'fullres', 'cpx', self.processor)
                    if coh_cfg in ('phsig', 'complex', 'none'):
                        coh_type = coh_cfg
                    elif coh_cfg == 'auto':
                        if ph_out is not None:
                            coh_type = 'phsig'
                        elif cpx_path is not None and cur_variant == 'fullres':
                            coh_type = 'complex'
                        elif cpx_existing.is_file() and cur_variant == 'fullres':
                            coh_type = 'complex'   # reuse an existing complex coh
                        elif phsig_existing.is_file():
                            coh_type = 'phsig'     # reuse an existing phsig coh
                        else:
                            coh_type = 'none'
                    else:
                        raise ValueError(
                            f"slc2ifg.unwrap.coh_type '{coh_cfg}' invalid, "
                            f"expected auto|complex|phsig|none (dp={dp})")

                    coh_input: Optional[Path] = None
                    coh_node: Optional[str] = None
                    # External coherence override: slc2ifg.unwrap.coh_dir +
                    # coh_pattern wins over the engine-generated coherence.
                    if coh_dir_cfg:
                        ext_dir = Path(coh_dir_cfg)
                        for cand_dir in (ext_dir / dp, ext_dir):
                            hits = sorted(cand_dir.glob(coh_pat_cfg))
                            if hits:
                                coh_input = Path(hits[0])
                                coh_type = 'external'
                                logger.info(
                                    "unwrap: using external coherence file "
                                    "%s (slc2ifg.unwrap.coh_dir / "
                                    "coh_pattern, dp=%s)", coh_input, dp)
                                break
                        if coh_input is None:
                            logger.warning(
                                "unwrap: no external coherence file matching "
                                "'%s' under %s or %s (dp=%s); falling back to "
                                "coh_type='%s'", coh_pat_cfg, ext_dir / dp,
                                ext_dir, dp, coh_type)
                    if coh_input is None and coh_type == 'phsig':
                        if ph_out is not None:
                            coh_input, coh_node = ph_out[0], ph_out[1]
                        elif phsig_existing.is_file():
                            # mid-chain entry: reuse a pre-generated phsig raster
                            coh_input, coh_node = phsig_existing, None
                        else:
                            raise ValueError(
                                f"unwrap: coh_type='phsig' but the "
                                f"'phsig_coh' stage is missing from "
                                f"engine.stages/tools (dp={dp}) and no "
                                f"existing raster at {phsig_existing}")
                    elif coh_input is None and coh_type == 'complex':
                        if cur_variant != 'fullres':
                            raise ValueError(
                                f"unwrap: complex coherence is fullres-only "
                                f"but the unwrapped variant is "
                                f"'{cur_variant}' (dp={dp}) — use "
                                f"coh_type='phsig' or run fullres")
                        if cpx_path is not None:
                            coh_input, coh_node = cpx_path, cpx_node
                        elif cpx_existing.is_file():
                            # mid-chain entry: reuse a pre-generated complex coh
                            coh_input, coh_node = cpx_existing, None
                        else:
                            raise ValueError(
                                f"unwrap: coh_type='complex' but the "
                                f"'complex_coh' stage is missing from "
                                f"engine.stages/tools (dp={dp}) and no "
                                f"existing raster at {cpx_existing}")
                    # 'none' -> coh_input stays None (SNAPHU weight = 1)

                    # Inject the *resolved* coherence type so the tool log
                    # shows what actually feeds SNAPHU (not the 'auto' config).
                    unw_params = dict(self._tool_params('unwrap'),
                                      coh_type=coh_type)
                    ctx = ToolContext(
                        tool_name='unwrap',
                        inputs={'ifg': cur_path, 'coh': coh_input,
                                'date1': d1, 'date2': d2},
                        outputs={'unw': u_out, 'conncomp': c_out},
                        params=unw_params,
                        work_dir=self.engine_dir,
                    )
                    g.add_node(TaskNode(key=ukey, tool=get_tool('unwrap'),
                                        label=f"unwrap {dp}", ctx=ctx))
                    if cur_node is not None:
                        g.add_edge(cur_node, ukey)
                    if coh_node is not None:
                        g.add_edge(coh_node, ukey)
                    unw_out = (u_out, ukey)

                elif sname == 'atmosphere':
                    if unw_out is None:
                        raise ValueError(
                            f"atmosphere requires the 'unwrap' stage before it "
                            f"in the chain (dp={dp})")
                    akey = f"atmosphere#{dp}"
                    a_out = naming.atm_path(self.unwrap_dir, d1, d2,
                                            cur_variant, self.processor)
                    ctx = ToolContext(
                        tool_name='atmosphere',
                        inputs={'unw': unw_out[0], 'date1': d1, 'date2': d2},
                        outputs={'unw': a_out},
                        params=self._tool_params('atmosphere'),
                        work_dir=self.engine_dir,
                    )
                    g.add_node(TaskNode(key=akey, tool=get_tool('atmosphere'),
                                        label=f"atmosphere {dp}", ctx=ctx))
                    g.add_edge(unw_out[1], akey)

                elif sname == 'complex_coh':
                    # Nodes already exist: per-burst nodes from Phase 1
                    # (full pipeline) or the per-pair pre-pass above
                    # (mid-chain entry) — nothing to wire here.
                    continue

                else:
                    # Stages outside the uniform phase (plan/per_burst/stitch/
                    # source) are already instantiated in Phases 0-2, so skip
                    # them here; unknown stages raise an error
                    if sname not in ('ifgram_list', 'generate_ifgram',
                                     'complex_coh', 'stitch', 'crop_slc'):
                        raise ValueError(
                            f"stage '{sname}' cannot be instantiated in this "
                            f"engine build (generic wiring lands in a future "
                            f"release)")

    # ------------------------------------------------------------------
    # Tool parameters
    # ------------------------------------------------------------------
    def _tool_params(self, tool: str) -> Dict:
        """Extract per-tool parameters from the raw MintPy-style config.

        Data-driven: reads the tool's declared ``params_spec``; only a small
        set of computed/cross-stage parameters remains as explicit special
        cases (see docs/engine_stages_design.md §4.2).
        """
        cfg = self.config.raw
        from mintpy.stdproc.engine.config import (
            get_bool_opt,
            get_float_opt,
            get_int_opt,
            get_opt,
            has_real_value,
        )

        base = {'processor': self.processor,
                'max_workers': self.config.max_workers or (os.cpu_count() or 1)}
        if self.config.tile_size:
            base['tile_size'] = self.config.tile_size

        # --- data-driven: tool-declared param specs ---
        t = get_tool(tool)
        for ps in t.params_spec:
            if ps.cfg is None:
                base[ps.key] = ps.default
                continue
            # Backend-scoped keys (slc2ifg.<stage>.<backend>.<param>) fall back
            # to the deprecated flat key when the new key is absent — with a
            # warning so the migration stays visible.
            read_key = ps.cfg
            if ps.legacy_cfg and not has_real_value(cfg, ps.cfg) \
                    and has_real_value(cfg, ps.legacy_cfg):
                logger.warning(
                    "config key '%s' is deprecated — use '%s' instead",
                    ps.legacy_cfg, ps.cfg)
                read_key = ps.legacy_cfg
            if ps.kind == 'bool':
                val = get_bool_opt(cfg, read_key, fallback=bool(ps.default))
            elif ps.kind == 'int':
                val = get_int_opt(cfg, read_key, fallback=ps.default)
            elif ps.kind == 'float':
                val = get_float_opt(cfg, read_key, fallback=ps.default)
            elif ps.kind == 'tuple':
                raw = get_opt(cfg, read_key)
                val = tuple(int(x) for x in raw.split(',')) if raw else ps.default
            else:
                val = get_opt(cfg, read_key, fallback=ps.default)
            # Backward compatibility: 0 / empty values fall back to the
            # default (consistent with the old _tool_params `or default`).
            # Bools are excluded so an explicit `false` is honored.
            if val is None or val == '' or \
                    (ps.kind in ('int', 'float') and val == 0):
                val = ps.default
            if val is not None:
                base[ps.key] = val

        # --- engine-injected / computed special cases ---
        if tool in gpu_tools():
            # flipped to True by _apply_gpu when the GPU backend is usable
            base['use_gpu'] = False
        if tool in ('phsig_coh', 'complex_coh'):
            # phase-sigma window size (looks product of the preceding stage)
            base['ps_nlks'] = self._ps_nlks()
        if tool == 'unwrap':
            # waterMaskFile from MintPy's load section doubles as the SNAPHU
            # mask when slc2ifg.unwrap.snaphu.mask_file is not set (the .wbd
            # water mask is auto-converted to the ifg grid by _unwrap_single).
            if 'mask_file' not in base or not base['mask_file']:
                wm = get_opt(cfg, 'mintpy.load.waterMaskFile')
                if wm:
                    base['mask_file'] = wm
            base['nlooks'] = (
                _legacy_opt(cfg, 'slc2ifg.unwrap.snaphu.nlooks',
                            'slc2ifg.unwrap.nlooks', 'float')
                or self._ps_nlks())
            base['ntiles'] = (
                _legacy_opt(cfg, 'slc2ifg.unwrap.snaphu.ntiles_row',
                            'slc2ifg.unwrap.ntiles_row', 'int',
                            default=1) or 1,
                _legacy_opt(cfg, 'slc2ifg.unwrap.snaphu.ntiles_col',
                            'slc2ifg.unwrap.ntiles_col', 'int',
                            default=1) or 1,
            )
        return base

    def _ps_nlks(self) -> float:
        cfg = self.config.raw
        from mintpy.stdproc.engine.config import get_float_opt, get_int_opt
        if self.run_multilook:
            lks_y = get_int_opt(cfg, 'slc2ifg.multilook.lks_y', fallback=1) or 1
            lks_x = get_int_opt(cfg, 'slc2ifg.multilook.lks_x', fallback=1) or 1
            default = float(lks_y * lks_x)
        else:
            default = 1.0
        return get_float_opt(cfg, 'slc2ifg.generate_coh.ps_nlks', fallback=default) or default

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------
    def plan(self, dry_run: bool = False) -> TaskGraph:
        """Build + validate the graph (does not execute tools)."""
        graph = self.build_graph()
        # Full node-by-node listing only when reviewing the plan or debugging;
        # normal runs get a compact per-tool count (keeps logs readable).
        verbose_summary = dry_run or logger.isEnabledFor(logging.DEBUG)
        logger.info("\n%s", graph.summary(verbose=verbose_summary))
        if dry_run:
            dot = graph.to_dot()
            (self.engine_dir / 'graph.dot').write_text(dot)
            try:
                import graphviz
                graphviz.Source(dot).render(self.engine_dir / 'graph', format='svg')
            except Exception:
                logger.info("graphviz not available — wrote graph.dot instead")
        return graph

    def run(self, dry_run: bool = False, skip_cleanup: bool = False) -> bool:
        """Build the DAG and execute it with Dask.

        Parameters
        ----------
        dry_run : bool
            Only plan (print the DAG, write graph.dot), do not execute.
        skip_cleanup : bool
            Do not apply the keep-intermediates policy afterwards
            (used by ``--restore`` so rebuilt products are kept).
        """
        from mintpy.stdproc.engine.resources import build_resource_plan

        graph = self.plan(dry_run=dry_run)
        if dry_run:
            logger.info("dry-run: DAG planned, nothing executed")
            return True

        plan = build_resource_plan(
            max_workers=self.config.max_workers,
            mem_limit_gb=self.config.mem_limit_gb,
            gpu=self.config.gpu,
            gpu_mem_limit_gb=self.config.gpu_mem_limit_gb,
        )

        results = execute_graph(graph, scheduler=self.config.scheduler, plan=plan)

        # --- manifest & cleanup ---
        for key, res in results.items():
            if isinstance(res, dict):
                self._manifest.record(key, list(res.values()))
        self._manifest.save()

        if not skip_cleanup:
            self._cleanup()
        return True

    def restore(self, dry_run: bool = False) -> bool:
        """Rebuild products that were deleted by a previous cleanup.

        Runs the full DAG — tools are idempotent, so only missing outputs are
        recomputed.  Cleanup is disabled, so nothing is deleted afterwards.
        """
        logger.info("Restore mode: rebuilding missing products "
                    "(keep_intermediates=all, cleanup disabled)")
        return self.run(dry_run=dry_run, skip_cleanup=True)

    def run_tool(self, tool_name: str, dry_run: bool = False) -> bool:
        """Standalone invocation: execute only the nodes of one tool.

        The full DAG is built (pair lists are generated), then only the
        requested tool's nodes are executed.  Their file inputs must already
        exist on disk (e.g. from a previous engine run).
        """
        graph = self.build_graph()
        keys = [k for k in graph.topo_order()
                if graph.nodes[k].tool.name == tool_name]
        if not keys:
            raise ValueError(f"Tool '{tool_name}' has no nodes in this run "
                             f"(is it enabled? available: {available_tools()})")

        # Validate file/dir inputs exist (upstream products from earlier runs)
        missing: List[str] = []
        for k in keys:
            node = graph.nodes[k]
            for port in node.tool.inputs:
                if port.kind not in ('file', 'dir', 'pairs_file'):
                    continue
                val = node.ctx.inputs.get(port.name)
                paths = val if isinstance(val, (list, tuple)) else [val]
                for p in paths:
                    if p is not None and not Path(p).exists():
                        missing.append(f"[{k}] {port.name}: {p}")
        if missing:
            raise FileNotFoundError(
                "Standalone tool inputs missing (run the full engine first):\n  "
                + "\n  ".join(missing))

        sub = graph.subgraph(keys)
        sub.validate()
        logger.info("Standalone tool '%s': %d node(s)", tool_name, len(keys))
        if dry_run:
            logger.info("\n%s", sub.summary())
            return True

        from mintpy.stdproc.engine.resources import build_resource_plan
        plan = build_resource_plan(
            max_workers=self.config.max_workers,
            mem_limit_gb=self.config.mem_limit_gb,
            gpu=self.config.gpu,
            gpu_mem_limit_gb=self.config.gpu_mem_limit_gb,
        )
        results = execute_graph(sub, scheduler=self.config.scheduler, plan=plan)
        for key, res in results.items():
            if isinstance(res, dict):
                self._manifest.record(key, list(res.values()))
        self._manifest.save()
        return True

    def _cleanup(self) -> None:
        """Apply the keep-intermediates policy (default: keep unw/phsig only)."""
        policy = self.config.keep_intermediates
        if policy == 'all':
            logger.info("keep_intermediates=all — nothing deleted")
            return
        to_delete = plan_cleanup(
            self._manifest,
            keep_policy=policy,
            keep_variants=self.config.keep_variants,
        )
        if not to_delete:
            logger.info("No intermediate files to delete")
            return
        logger.info("Cleaning %d intermediate file(s) (keep_policy='%s')",
                    len(to_delete), policy)
        for p in to_delete:
            logger.debug("  delete %s", p)
        execute_cleanup(to_delete)
        self._manifest.mark_deleted(to_delete)
        self._manifest.save()
