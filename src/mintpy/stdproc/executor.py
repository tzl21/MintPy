#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Execution backends for the MintPy slc2ifg pipeline.

Two interchangeable executors run the SAME slc2ifg processing functions
(guaranteeing identical products):

- :class:`BasicExecutor` — always available, pure MintPy. Stages run in
  chain order; date pairs within a stage are parallelised with MintPy's own
  joblib primitive (:func:`mintpy.utils.utils0.check_parallel`), honoring
  ``mintpy.compute.numWorker``.
- :class:`EngineExecutor` — the default backend, living inside
  ``mintpy.stdproc.engine`` (Dask scheduling, GPU kernels, tiling,
  artifact management).  Its heavy dependencies (dask/cupy) are only
  imported at run time.

:func:`get_executor` selects the backend from the config key
``mintpy.slc2ifg.engine`` (``none | insarflow | auto``); ``auto`` uses the
engine when importable, otherwise falls back to the basic executor — MintPy
can always run the full slc2ifg workflow.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from mintpy.utils import utils0 as ut

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Executor interface
# ------------------------------------------------------------------------
class Slc2ifgExecutor(ABC):
    """Run the slc2ifg chain from a parsed MintPy-style config dict."""

    @abstractmethod
    def run(self, cfg: dict) -> None:
        """Execute the pipeline (products written to disk)."""


def get_executor(cfg: dict) -> Slc2ifgExecutor:
    """Pick the execution backend from ``mintpy.slc2ifg.engine``.

    Values: ``none`` (basic only), ``insarflow`` (engine required),
    ``auto`` (engine when importable, else basic).
    """
    engine_cfg = str(cfg.get('mintpy.slc2ifg.engine', 'auto')).lower()
    if engine_cfg == 'none':
        return BasicExecutor()

    if engine_cfg == 'insarflow':
        return EngineExecutor()
    # auto: engine by default when dask is installed, otherwise fall back to
    # the basic executor (documented contract: MintPy can always complete the
    # slc2ifg workflow).
    import importlib.util
    if importlib.util.find_spec('dask') is None:
        logger.warning(
            "dask not found — falling back to the basic executor "
            "(set mintpy.slc2ifg.engine = none to silence this warning)")
        return BasicExecutor()
    logger.info("using the engine executor (set mintpy.slc2ifg.engine = none "
                "for the basic executor)")
    return EngineExecutor()


# ------------------------------------------------------------------------
# BasicExecutor — pure-MintPy sequential + joblib-parallel execution
# ------------------------------------------------------------------------
#: Default chain (identical stage set/order to the insarflow engine).
BASIC_CHAIN = [
    'crop_slc',      # optional (requires slc2ifg.bbox)
    'ifgram_list',   # eager planning stage (runs first, no pair parallelism)
    'generate_ifgram',
    'stitch',        # multi-burst only
    'multilook',
    'filter',
    'phsig_coh',
    'unwrap',
]

#: Stages that may run per date pair (skipped entirely when not configured).
PER_PAIR_STAGES = [
    'generate_ifgram', 'stitch', 'multilook', 'filter', 'phsig_coh', 'unwrap',
]


def _cfg_get(cfg: dict, key: str, fallback=None):
    """Read a config key, honoring 'auto'/empty/'none' as 'use fallback'.

    Semantics match the engine's ``get_opt`` (engine/config.py) so the two
    backends resolve identical values from the same cfg.
    """
    val = cfg.get(key)
    if val is None:
        return fallback
    val = str(val).strip()
    if val.lower() in ('auto', 'none', 'null', 'off', ''):
        return fallback
    return val


def _cfg_get_int(cfg: dict, key: str, fallback=None):
    val = _cfg_get(cfg, key)
    if val is None:
        return fallback
    try:
        return int(float(val))
    except ValueError:
        return fallback


def _cfg_get_float(cfg: dict, key: str, fallback=None):
    val = _cfg_get(cfg, key)
    if val is None:
        return fallback
    try:
        return float(val)
    except ValueError:
        return fallback


def _read_pairs(pair_file: Path) -> List[Tuple[str, str]]:
    """Parse a slc2ifg pair list (``date1-date2`` lines, '#' comments)."""
    pairs = []
    if not pair_file.is_file():
        return pairs
    for line in pair_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.replace('-', '_').split('_')
        if len(parts) == 2:
            pairs.append((parts[0], parts[1]))
    return pairs


class BasicExecutor(Slc2ifgExecutor):
    """Sequential chain + MintPy joblib parallelism over date pairs.

    Calls the identical slc2ifg module functions used by the insarflow
    engine tools, so products match the engine byte-for-byte for the same
    parameters.  All stages are idempotent (skip when output exists).
    """

    def __init__(self):
        self.cfg: dict = {}

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _opt(self, key, fallback=None):
        return _cfg_get(self.cfg, key, fallback)

    def _opt_int(self, key, fallback=None):
        return _cfg_get_int(self.cfg, key, fallback)

    def _opt_float(self, key, fallback=None):
        return _cfg_get_float(self.cfg, key, fallback)

    def _opt_bool(self, key, fallback=False):
        """Boolean read with the engine's semantics (''/'auto'/'none' -> fallback)."""
        val = self._opt(key)
        if val is None:
            return fallback
        return str(val).strip().lower() in ('true', 'yes', '1', 'on')

    def _ps_nlks(self, fallback=1.0) -> float:
        """Effective number of looks for phase-sigma / SNAPHU weighting.

        Mirrors the engine's ``_ps_nlks``: explicit
        ``slc2ifg.generate_coh.ps_nlks`` wins; otherwise the multilook
        product's ``lks_y * lks_x`` when multilook runs; else 1.0.
        """
        default = fallback
        if 'multilook' in self._tools_enabled():
            lks_y = self._opt_int('slc2ifg.multilook.lks_y', 1) or 1
            lks_x = self._opt_int('slc2ifg.multilook.lks_x', 1) or 1
            default = float(lks_y * lks_x)
        explicit = self._opt_float('slc2ifg.generate_coh.ps_nlks')
        return explicit if explicit else default

    def _parallel_ctx(self, n_items: int, max_parallel_num: int = 8):
        """Return (num_cores, parallel, Parallel, delayed) via MintPy helper."""
        return ut.check_parallel(n_items, print_msg=False,
                                 maxParallelNum=max_parallel_num)

    def _tools_enabled(self) -> List[str]:
        """Effective tool list: ``engine.stages`` (authoritative chain spec)
        or the default chain when unset/auto."""
        if 'engine.tools' in self.cfg:
            raise ValueError(
                "config key 'engine.tools' is deprecated and was removed — "
                "use 'engine.stages' instead (it is the single authoritative "
                "chain spec)")
        stages_cfg = self._opt('engine.stages', 'auto')
        if stages_cfg not in (None, 'auto'):
            return [t.strip() for t in str(stages_cfg).split(',') if t.strip()]
        return ['ifgram_list', 'generate_ifgram', 'stitch',
                'multilook', 'filter', 'unwrap']

    def _run_ifgram_list(self, slc_dir: Path, out_dir: Path,
                         bbox_cfg: Optional[Tuple[str, float]] = None) -> Path:
        """Eagerly generate the pair list (chain topology depends on it).

        ``bbox_cfg`` is the read-time-crop AOI ``(wsen_str, buffer)``: in
        select mode it restricts the quick-coherence screening to the AOI
        window instead of reading the whole SLC (the AOI is already cropped).
        """
        from mintpy.stdproc.ifgram_list import (
            filter_date_list,
            generate_pairs,
            get_date_list,
            write_pair_list,
        )
        from mintpy.stdproc.select_ifgrams import select_pairs

        mode = self._opt('slc2ifg.ifgram_list.mode', 'sequential')
        start_date = self._opt('slc2ifg.ifgram_list.start_date')
        end_date = self._opt('slc2ifg.ifgram_list.end_date')
        exclude_date = self._opt('slc2ifg.ifgram_list.exclude_date')
        nconn = self._opt_int('slc2ifg.ifgram_list.num_connections')

        dates = get_date_list(str(slc_dir))
        dates = filter_date_list(dates, start_date=start_date, end_date=end_date,
                                 exclude_date=exclude_date)
        pair_file = out_dir / 'ifgram_list.txt'

        if mode == 'select':
            # k-NN skeleton default for select mode is 3 (vs 5 sequential) —
            # identical to the engine's _select_params
            params = {'num_connections': nconn if nconn is not None else 3,
                      'slc_dir': str(slc_dir),
                      'processor': self._opt('slc2ifg.processor', 'isce3')}
            # typed reads with the same kinds/defaults as the engine (config
            # keys slc2ifg.ifgram_list.select.*), so both backends produce
            # identical selections
            readers = {
                'str': self._opt,
                'int': self._opt_int,
                'float': self._opt_float,
                'bool': self._opt_bool,
            }
            for key, kind in [
                    ('annual_windows', 'str'),
                    ('temp_baseline_max', 'int'),
                    ('perp_baseline_max', 'float'),
                    ('perp_baseline_file', 'str'),
                    ('weight_source', 'str'),
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
                if key == 'slc_pattern':
                    continue   # handled below (unified key)
                val = readers[kind](f'slc2ifg.ifgram_list.select.{key}')
                if val is not None:
                    params[key] = val
            # unified SLC pattern (slc2ifg.slc_pattern; legacy
            # select.slc_pattern as fallback — engine parity)
            pat = (self._opt('slc2ifg.slc_pattern')
                   or self._opt('slc2ifg.ifgram_list.select.slc_pattern'))
            if pat:
                params['slc_pattern'] = pat
            # AOI: restrict the quick coherence to the bbox+buffer window
            # (only when the SLCs are NOT already cropped on disk, i.e. when
            # the read-time crop is active and bbox_cfg is set).
            if bbox_cfg:
                from mintpy.stdproc.crop_slc_geo import parse_wsen
                wsen, buffer = bbox_cfg
                params['bbox'] = parse_wsen(wsen)
                params['bbox_buffer'] = float(buffer)
            # report / dot: resolve relative paths against the work dir
            # (mirrors the engine's _select_params)
            for out_key, cfgk in (
                    ('report_file', 'slc2ifg.ifgram_list.select.report'),
                    ('dot_file', 'slc2ifg.ifgram_list.select.dot')):
                rp = self._opt(cfgk)
                if rp:
                    p = Path(rp)
                    if not p.is_absolute() and getattr(self, 'work_dir', None):
                        p = (self.work_dir / p).resolve()
                    params[out_key] = str(p)
            pairs, _ = select_pairs(dates, params=params)
        else:
            nconn = nconn if nconn is not None else 5
            # annual_windows applies to ALL modes (canonical knob; the legacy
            # slc2ifg.ifgram_list.oneyear_interferograms key was removed)
            aw = self._opt('slc2ifg.ifgram_list.select.annual_windows')
            pairs = generate_pairs(
                dates, mode, nconn, oneyear_range=None,
                select_params={'annual_windows': aw} if aw is not None else None)

        write_pair_list(pairs, pair_file)
        logger.info("ifgram_list (%s): %d date(s) -> %d pair(s)",
                    mode, len(dates), len(pairs))
        return pair_file

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self, cfg: dict) -> None:
        self.cfg = cfg
        from mintpy.stdproc.utils import naming

        processor = self._opt('slc2ifg.processor', 'isce3')
        work_dir = Path(self._opt('slc2ifg.work_dir', './'))
        if not work_dir.is_absolute():
            work_dir = (Path.cwd() / work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir = work_dir

        slc_input = self._opt('slc2ifg.slc_input')
        if not slc_input:
            raise ValueError("slc2ifg.slc_input is required in the config")
        slc_dir = Path(slc_input)

        # unified ifgram output tree (same layout as the engine)
        ifg_out = Path(self._opt('slc2ifg.generate_ifgram.output_dir',
                                  str(work_dir / 'ifgrams')))
        if not ifg_out.is_absolute():
            ifg_out = (work_dir / ifg_out).resolve()
        ifg_out.mkdir(parents=True, exist_ok=True)

        tools = self._tools_enabled()
        n_workers = self._opt_int('mintpy.compute.numWorker', None) or \
            self._opt_int('engine.max_workers', None)

        if 'complex_coh' in tools:
            logger.warning(
                "complex_coh is only supported by the engine backend "
                "(mintpy.slc2ifg.engine = insarflow) — skipping it in the "
                "basic executor")

        # ---------- crop (optional, transforms the SLC input dir) ----------
        slc_base = slc_dir
        if 'crop_slc' in tools:
            slc_base = self._run_crop(slc_dir, work_dir)

        # Read-time crop (default): with crop_slc absent, generate_ifgram
        # reads only the slc2ifg.bbox region of the SLCs — no cropped SLC
        # files are stored.  isce2 (radar) SLCs are not georeferenced, so
        # this mode requires the crop_slc stage.
        bbox_cfg = None
        if 'crop_slc' not in tools:
            wsen = self._opt('slc2ifg.bbox') or self._opt('slc2ifg.crop_slc.wsen')
            if wsen:
                if processor == 'isce2':
                    raise ValueError(
                        "slc2ifg.bbox read-time crop requires geocoded SLCs "
                        "(isce3 GeoTIFF/HDF5) — for isce2 enable the "
                        "'crop_slc' stage instead")
                buffer = self._opt_float(
                    'slc2ifg.bbox_buffer',
                    self._opt_float('slc2ifg.crop_slc.buffer', 0.0))
                bbox_cfg = (str(wsen), float(buffer))

        # ---------- phase 1: per-burst ifgram_list + generate_ifgram -------
        bursts = self._discover_bursts(slc_base)
        multi = len(bursts) > 1
        logger.info("Mode: %s", "multi-burst" if multi else "single-burst")

        pair_files = {}
        ifg_sources: Dict[str, Path] = {}   # (burst, dp) -> ifg path
        for b in bursts:
            btag = b or 'single'
            b_slc = slc_base / b if b else slc_base
            b_out = ifg_out / b if b else ifg_out
            b_out.mkdir(parents=True, exist_ok=True)
            pair_file = self._run_ifgram_list(b_slc, b_out, bbox_cfg)
            pair_files[btag] = pair_file
            pairs = _read_pairs(pair_file)
            if not pairs:
                logger.warning("no pairs for burst %s — skipping", btag)
                continue

            if 'generate_ifgram' not in tools:
                # mid-chain entry: consume existing fullres ifgrams
                for d1, d2 in pairs:
                    ifg_sources[(btag, f'{d1}_{d2}')] = naming.ifg_path(
                        b_out, d1, d2, 'fullres', processor)
                continue

            # parallel generate_ifgram over date pairs
            tasks = [(b_slc, b_out, d1, d2, pair_file, processor, bbox_cfg)
                     for d1, d2 in pairs]
            results = self._map_pairs(
                tasks, self._generate_one, n_workers,
                desc=f'generate_ifgram ({btag})')
            for (d1, d2), ok, msg in results:
                if ok:
                    ifg_sources[(btag, f'{d1}_{d2}')] = naming.ifg_path(
                        b_out, d1, d2, 'fullres', processor)
                else:
                    logger.error("generate_ifgram %s_%s: %s", d1, d2, msg)

        # ---------- phase 2: stitch multi-burst (or direct single) ---------
        phase3_base = ifg_out
        if multi and 'stitch' in tools:
            stitched = work_dir / 'stitched'
            stitched.mkdir(parents=True, exist_ok=True)
            phase3_base = stitched
            for (btag, dp), src in list(ifg_sources.items()):
                d1, d2 = dp.split('_')
                # collect per-burst sources for this pair
                srcs = [ifg_sources[(bb, dp)]
                        for bb in bursts if (bb or 'single', dp) in ifg_sources]
                if len(srcs) < 1:
                    continue
                out = naming.ifg_path(stitched, d1, d2, 'fullres', processor)
                self._stitch_one(srcs, out, processor)

        # ---------- phase 3: uniform global chain per pair ----------------
        # Phase-3 pairs = the UNION of all per-burst pair lists (per-burst
        # date sets can differ; using only the first burst would silently
        # skip pairs that exist only in other bursts).
        all_pairs: set = set()
        for pf in pair_files.values():
            if pf is not None:
                all_pairs.update(_read_pairs(pf))
        pairs = sorted(all_pairs)
        if not pairs:
            logger.warning("empty pair list — nothing to process")
            return

        # uniform global stages only (generate_ifgram / stitch already ran in
        # phases 1-2; complex_coh is not supported in the basic executor)
        chain = [s for s in ('multilook', 'filter', 'phsig_coh', 'unwrap')
                 if s in tools]
        # phase-3 INPUTS come from phase3_base (the stitched tree in
        # multi-burst mode); OUTPUTS go to the unified ifgram tree — the
        # same layout as the engine (ml/filter/phsig/unwrap all live under
        # <ifgram_out_dir>/{date_pair}/).
        cur_variant = 'fullres'
        cur_in_base = phase3_base
        out_base = ifg_out
        for stage in chain:
            tasks = [(cur_in_base, out_base, cur_variant, d1, d2, processor)
                     for d1, d2 in pairs]
            handler = getattr(self, f'_stage_{stage}')
            results = self._map_pairs(tasks, handler, n_workers,
                                      desc=f'{stage}')
            for (d1, d2), ok, msg in results:
                if not ok:
                    logger.error("%s %s_%s: %s", stage, d1, d2, msg)
            # the next stage consumes this stage's outputs (unified tree)
            cur_in_base = out_base
            if stage == 'multilook':
                cur_variant = 'mli'
            elif stage == 'filter':
                cur_variant = 'filt_mli' if cur_variant == 'mli' else 'filt'
        logger.info("slc2ifg basic pipeline finished")

    # ------------------------------------------------------------------
    # stage workers (identical function calls as the insarflow engine)
    # ------------------------------------------------------------------
    def _map_pairs(self, tasks, worker, n_workers, desc=''):
        """Parallel map over date pairs via MintPy's joblib helper."""
        if not tasks:
            return []
        n_cores, parallel, Parallel, delayed = self._parallel_ctx(
            len(tasks), max_parallel_num=n_workers or 8)
        if parallel and n_cores > 1 and Parallel is not None:
            logger.info("%s: %d task(s) on %d worker(s)", desc,
                        len(tasks), n_cores)
            return Parallel(n_jobs=n_cores)(delayed(worker)(t) for t in tasks)
        return [worker(t) for t in tasks]

    def _generate_one(self, task):
        from mintpy.stdproc.generate_ifgram import (
            find_slc_file_by_date,
            process_single_pair,
        )
        from mintpy.stdproc.utils import naming
        b_slc, b_out, d1, d2, pair_file, processor, bbox_cfg = task
        try:
            slc_pattern = (self._opt('slc2ifg.slc_pattern')
                           or self._opt('slc2ifg.generate_ifgram.slc_pattern')
                           or naming.slc_pattern(processor))
            subdataset = self._opt('slc2ifg.generate_ifgram.subdataset',
                                   '/data/VV')
            slc1 = find_slc_file_by_date([b_slc], d1, slc_pattern, processor)
            slc2 = find_slc_file_by_date([b_slc], d2, slc_pattern, processor)
            if slc1 is None or slc2 is None:
                return (d1, d2), False, f"SLC missing for {d1}/{d2}"
            pair_dir = b_out / f'{d1}_{d2}'
            pair_dir.mkdir(parents=True, exist_ok=True)
            vrt_path = pair_dir / 'fullres.int.vrt'
            ifg_path = naming.ifg_path(b_out, d1, d2, 'fullres', processor)

            # Read-time crop: map the bbox to this pair's SLC pixel window
            window = None
            if bbox_cfg:
                from mintpy.stdproc.crop_slc_geo import bbox_to_window, parse_wsen
                wsen, buffer = bbox_cfg
                window = bbox_to_window(slc1, parse_wsen(wsen), subdataset,
                                        buffer)
                if window is None:
                    return (d1, d2), False, \
                        f"bbox {wsen} does not intersect SLC {slc1}"

            pair_info = (f'{d1}-{d2}', d1, d2, slc1, slc2, vrt_path,
                         ifg_path, True, subdataset, processor,
                         self._opt('slc2ifg.generate_ifgram.only_vrt', False),
                         window)
            date12, ok, msg = process_single_pair(pair_info)
            return (d1, d2), ok, msg
        except Exception as e:
            return (d1, d2), False, str(e)

    def _stitch_one(self, srcs, out, processor):
        from mintpy.stdproc.stitch import get_file_epsg, stitch_date_pair
        try:
            epsg_utm = 32605
            for f in srcs:
                epsg = get_file_epsg(f)
                if epsg:
                    epsg_utm = epsg
                    break
            ok = stitch_date_pair(list(srcs), out,
                                  self._opt_tuple('slc2ifg.stitch.out_bounds'),
                                  overwrite=True, epsg_utm=epsg_utm)
            return ok, '' if ok else 'stitch failed'
        except Exception as e:
            return False, str(e)

    def _stage_multilook(self, task):
        from mintpy.stdproc.multilook import multilook_tif
        from mintpy.stdproc.utils import naming
        base, out_base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            out_path = naming.ifg_path(out_base, d1, d2,
                                       naming.next_variant(variant, 'multilook'),
                                       processor)
            multilook_tif(str(in_path), str(out_path),
                          lks_y=self._opt_int('slc2ifg.multilook.lks_y', 1),
                          lks_x=self._opt_int('slc2ifg.multilook.lks_x', 1),
                          method=self._opt('slc2ifg.multilook.method', 'mean'),
                          processor=processor)
            return (d1, d2), True, ''
        except Exception as e:
            return (d1, d2), False, str(e)

    def _stage_filter(self, task):
        from mintpy.stdproc.filter import process_single_goldstein
        from mintpy.stdproc.utils import naming
        base, out_base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            out_path = naming.ifg_path(out_base, d1, d2,
                                       naming.next_variant(variant, 'filter'),
                                       processor)
            process_single_goldstein(
                str(in_path), base,
                self._opt_float('slc2ifg.filter.goldstein.alpha', 0.8),
                self._opt_int('slc2ifg.filter.goldstein.psize', 32),
                processor, output_file=out_path, gpu=False)
            return (d1, d2), True, ''
        except Exception as e:
            return (d1, d2), False, str(e)

    def _stage_phsig_coh(self, task):
        from mintpy.stdproc.generate_coh_phsig import (
            _write_band,
            read_complex_image,
        )
        from mintpy.stdproc.utils import naming
        base, out_base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            out_path = naming.coh_path(out_base, d1, d2, variant, 'phsig',
                                       processor)
            if out_path.exists():
                return (d1, d2), True, 'exists'
            ifg, meta = read_complex_image(str(in_path), processor)
            from mintpy.stdproc.generate_coh_phsig import (
                estimate_phsig_correlation,
            )
            coh = estimate_phsig_correlation(
                ifg,
                ps_win=self._opt_int('slc2ifg.generate_coh.ps_window_size', 5),
                grad_win=self._opt_int(
                    'slc2ifg.generate_coh.ps_gradient_window', 5),
                nlks=self._ps_nlks(),
            )
            _write_band(str(out_path), coh, meta, processor,
                        'phase-sigma correlation')
            return (d1, d2), True, ''
        except Exception as e:
            return (d1, d2), False, str(e)

    def _stage_unwrap(self, task):
        from mintpy.stdproc.unwrap_ifgram import _unwrap_single
        from mintpy.stdproc.utils import naming
        base, out_base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            unw_root = out_base
            unw_root.mkdir(parents=True, exist_ok=True)

            # ---- resolve the coherence input from slc2ifg.unwrap.coh_type
            # (auto|complex|phsig|none), mirroring the engine ----
            coh_cfg = str(self._opt('slc2ifg.unwrap.coh_type', 'auto')
                          or 'auto').lower()
            tools = self._tools_enabled()
            phsig_path = naming.coh_path(out_base, d1, d2, variant, 'phsig',
                                         processor)
            cpx_path = naming.coh_path(out_base, d1, d2, 'fullres', 'cpx',
                                       processor)
            if coh_cfg in ('phsig', 'complex', 'none'):
                coh_type = coh_cfg
            elif coh_cfg == 'auto':
                if 'phsig_coh' in tools:
                    coh_type = 'phsig'
                elif 'complex_coh' in tools and variant == 'fullres':
                    coh_type = 'complex'
                elif cpx_path.exists() and variant == 'fullres':
                    coh_type = 'complex'     # reuse an existing cpx raster
                elif phsig_path.exists():
                    coh_type = 'phsig'       # reuse an existing phsig raster
                else:
                    coh_type = 'none'        # SNAPHU weight 1 (uniform)
            else:
                raise ValueError(
                    f"slc2ifg.unwrap.coh_type '{coh_cfg}' invalid, expected "
                    f"auto|complex|phsig|none (dp={d1}_{d2})")

            cor = None
            if coh_type == 'phsig':
                if not phsig_path.exists():
                    raise ValueError(
                        f"unwrap: coh_type='phsig' but no phsig raster at "
                        f"{phsig_path} (run the phsig_coh stage first)")
                cor = phsig_path
            elif coh_type == 'complex':
                if not cpx_path.exists():
                    raise ValueError(
                        f"unwrap: coh_type='complex' but no complex coherence "
                        f"raster at {cpx_path}")
                cor = cpx_path
            # 'none' -> cor stays None (uniform weights)

            mask = self._opt('slc2ifg.unwrap.snaphu.mask_file')
            if not mask:
                mask = self._opt('mintpy.load.waterMaskFile')
            _unwrap_single(
                ifg_path=Path(in_path),
                cor_path=Path(cor) if cor else None,
                nlooks=(self._opt_float('slc2ifg.unwrap.snaphu.nlooks')
                        or self._opt_float('slc2ifg.unwrap.nlooks')
                        or self._ps_nlks()),
                output_dir=unw_root,
                processor=processor,
                snaphu_bin=self._opt('slc2ifg.unwrap.snaphu.binary'),
                cost_mode=self._opt('slc2ifg.unwrap.snaphu.cost_mode',
                                    'smooth'),
                init_method=self._opt('slc2ifg.unwrap.snaphu.init_method',
                                      'mcf'),
                mask_path=Path(mask) if mask else None,
                ntiles=(self._opt_int('slc2ifg.unwrap.snaphu.ntiles_row', 1),
                        self._opt_int('slc2ifg.unwrap.snaphu.ntiles_col', 1)),
                nproc=self._opt_int('slc2ifg.unwrap.snaphu.nproc', 1),
                keep_scratch=self._opt('slc2ifg.unwrap.snaphu.keep_scratch',
                                       False) in ('True', 'true', 'yes', '1'),
            )
            return (d1, d2), True, ''
        except Exception as e:
            return (d1, d2), False, str(e)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _opt_tuple(self, key, fallback=None):
        val = self._opt(key)
        if val is None:
            return fallback
        import re as _re
        parts = [x.strip() for x in _re.split(r'[\s,]+', str(val)) if x.strip()]
        try:
            if len(parts) != 4:
                raise ValueError(f"expected 4 numbers, got {len(parts)}")
            return tuple(float(x) for x in parts)
        except ValueError:
            logger.warning("Invalid %s value %r (expected 4 numbers, "
                           "space/comma separated) — ignoring", key, val)
            return fallback

    def _discover_bursts(self, slc_dir: Path) -> List[Optional[str]]:
        import re
        _BURST_RE = re.compile(r'^t\d+_\d+_iw\d+$')
        if not slc_dir.is_dir():
            return [None]
        bursts = sorted(e.name for e in slc_dir.iterdir()
                        if e.is_dir() and _BURST_RE.match(e.name))
        if bursts:
            logger.info("Discovered %d burst(s): %s", len(bursts), bursts)
            return bursts
        return [None]

    def _run_crop(self, slc_dir: Path, work_dir: Path) -> Path:
        """Crop SLCs to bbox, mirroring the engine CropSlcTool: dispatch on
        processor (isce2 -> radar / isce3 -> geo), honour the crop pattern
        and the ifgram_list date filter (start/end/exclude), and abort on a
        non-zero exit code."""
        import re as _re
        from mintpy.stdproc.utils import naming

        processor = self._opt('slc2ifg.processor', 'isce3')
        # AOI: unified slc2ifg.bbox; legacy crop_slc.wsen as fallback
        wsen = self._opt('slc2ifg.bbox') or self._opt('slc2ifg.crop_slc.wsen')
        if not wsen:
            raise ValueError(
                "crop_slc requires slc2ifg.bbox "
                "(legacy alias: slc2ifg.crop_slc.wsen)")
        out_dir = work_dir / 'cropped_slc'
        out_dir.mkdir(parents=True, exist_ok=True)

        # date filter (start/end/exclude) -> --file-list, mirroring the
        # engine's _add_crop_node (non-date files are always kept)
        from mintpy.stdproc.ifgram_list import parse_exclude_dates
        start_date = self._opt('slc2ifg.ifgram_list.start_date')
        end_date = self._opt('slc2ifg.ifgram_list.end_date')
        ex_dates = parse_exclude_dates(
            self._opt('slc2ifg.ifgram_list.exclude_date'))
        pattern = (self._opt('slc2ifg.slc_pattern')
                   or self._opt('slc2ifg.crop_slc.pattern')
                   or naming.slc_pattern(processor))
        candidates = sorted(Path(slc_dir).glob(f"**/{pattern}"))
        keep = []
        for p in candidates:
            m = _re.search(r'(20\d{6})', p.name)
            if not m:
                keep.append(str(p))      # non-date assets always kept
                continue
            d = m.group(1)
            if start_date and d < str(start_date):
                continue
            if end_date and d > str(end_date):
                continue
            if d in ex_dates:
                continue
            keep.append(str(p))
        if not keep:
            raise ValueError(
                f"No SLC files match the crop date range "
                f"[{start_date or '-inf'}, {end_date or '+inf'}] "
                f"excl {ex_dates} under {slc_dir}")

        list_file = work_dir / 'crop_file_list.txt'
        list_file.write_text('\n'.join(keep) + '\n')

        wsen_tokens = [x for x in _re.split(r'[\s,]+', str(wsen)) if x.strip()]
        if len(wsen_tokens) != 4:
            raise ValueError(
                f"slc2ifg.bbox must be 4 numbers (W S E N), got {wsen!r}")
        args_list = [
            '--input-dir', str(slc_dir),
            '--file-list', str(list_file),
            '--output-dir', str(out_dir),
            '--wsen'] + wsen_tokens + [
            '--buffer', str(self._opt_float(
                'slc2ifg.bbox_buffer',
                self._opt_float('slc2ifg.crop_slc.buffer', 0.0))),
            '--prefix', str(self._opt('slc2ifg.crop_slc.prefix', '')),
            '--max-workers', str(self._opt_int('engine.max_workers', 1) or 1),
        ]
        if self._opt('slc2ifg.crop_slc.geom_dir'):
            args_list += ['--geom-dir', str(self._opt('slc2ifg.crop_slc.geom_dir'))]
        if self._opt('slc2ifg.crop_slc.by_burst', 'false').lower() in ('true', 'yes', '1'):
            args_list.append('--by-burst')
        args_list.append('--no-burst-dirs')

        if processor == 'isce2':
            from mintpy.stdproc.crop_slc_rdr import main as rdr_main
            from mintpy.stdproc.crop_slc_rdr import parse_arguments as rdr_parse
            ret = rdr_main(rdr_parse(args_list))
        else:
            from mintpy.stdproc.crop_slc_geo import main as geo_main
            from mintpy.stdproc.crop_slc_geo import parse_arguments as geo_parse
            ret = geo_main(geo_parse(args_list))
        if ret not in (0, None):
            raise RuntimeError(f"crop_slc failed with exit code {ret}")
        return out_dir


# ------------------------------------------------------------------------
# EngineExecutor — optional insarflow backend (lazy import)
# ------------------------------------------------------------------------
class EngineExecutor(Slc2ifgExecutor):
    """Delegate the run to the insarflow engine (Dask, GPU, tiling)."""

    def run(self, cfg: dict) -> None:
        # The engine reads the same MintPy-style flat config from a FILE.
        # The caller (mintpy/slc2ifg.py) passes the config file path in cfg.
        from mintpy.stdproc.engine.config import load_engine_config
        from mintpy.stdproc.engine.engine import Engine

        config_file = cfg.get('_cfg_file')
        if not config_file:
            raise ValueError(
                "EngineExecutor requires the config file path "
                "(run via `mintpy slc2ifg <cfg>` — basic executor is used "
                "for in-memory dict configs)")
        engine_cfg = load_engine_config(str(config_file))
        Engine(engine_cfg).run()
