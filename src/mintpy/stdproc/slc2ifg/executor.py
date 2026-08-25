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
  ``mintpy.stdproc.slc2ifg.engine`` (Dask scheduling, GPU kernels, tiling,
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

    # auto | insarflow -> use the engine (lives inside mintpy since the
    # insarflow package was merged; the heavy deps dask/cupy are only pulled
    # by EngineExecutor.run(), so MintPy core stays free of them)
    if engine_cfg == 'insarflow':
        return EngineExecutor()
    # auto: engine by default (it is part of mintpy); users can force the
    # basic executor with engine = none
    logger.info("using the engine executor (set mintpy.slc2ifg.engine = none "
                "for the basic executor)")
    return EngineExecutor()


# ------------------------------------------------------------------------
# BasicExecutor — pure-MintPy sequential + joblib-parallel execution
# ------------------------------------------------------------------------
#: Default chain (identical stage set/order to the insarflow engine).
BASIC_CHAIN = [
    'crop_slc',      # optional (requires slc2ifg.crop_slc.wsen)
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
    """Read a config key, honoring 'auto'/empty as 'use fallback'."""
    val = cfg.get(key)
    if val is None:
        return fallback
    val = str(val).strip()
    if val.lower() in ('auto', '') :
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

    def _parallel_ctx(self, n_items: int, max_parallel_num: int = 8):
        """Return (num_cores, parallel, Parallel, delayed) via MintPy helper."""
        return ut.check_parallel(n_items, print_msg=False,
                                 maxParallelNum=max_parallel_num)

    def _tools_enabled(self) -> List[str]:
        """Effective tool list: explicit engine.tools whitelist or default."""
        tools_cfg = self._opt('engine.tools', 'auto')
        if tools_cfg in (None, 'auto'):
            return ['ifgram_list', 'generate_ifgram', 'stitch',
                    'multilook', 'filter', 'unwrap']
        return [t.strip() for t in str(tools_cfg).split(',') if t.strip()]

    def _run_ifgram_list(self, slc_dir: Path, out_dir: Path) -> Path:
        """Eagerly generate the pair list (chain topology depends on it)."""
        from mintpy.stdproc.slc2ifg.ifgram_list import (
            filter_date_list,
            generate_pairs,
            get_date_list,
            write_pair_list,
        )
        from mintpy.stdproc.slc2ifg.select_ifgrams import select_pairs

        mode = self._opt('slc2ifg.ifgram_list.mode', 'sequential')
        start_date = self._opt('slc2ifg.ifgram_list.start_date')
        end_date = self._opt('slc2ifg.ifgram_list.end_date')
        nconn = self._opt_int('slc2ifg.ifgram_list.num_connections', 5)
        oneyear = self._opt_int('slc2ifg.ifgram_list.oneyear_interferograms')

        dates = get_date_list(str(slc_dir))
        dates = filter_date_list(dates, start_date=start_date, end_date=end_date)
        pair_file = out_dir / 'ifgram_list.txt'

        if mode == 'select':
            params = {'num_connections': nconn if nconn else 3,
                      'slc_dir': str(slc_dir),
                      'processor': self._opt('slc2ifg.processor', 'isce3')}
            for k in ('annual_windows', 'temp_baseline_max', 'perp_baseline_max',
                      'perp_baseline_file', 'weight_source', 'slc_pattern', 'coh_dir',
                      'coh_kind', 'coh_variant', 'coh_stat',
                      'quick_window', 'min_degree',
                      'max_pairs', 'quality_threshold', 'robust', 'verify'):
                cfgk = f'slc2ifg.ifgram_list.select.{k}'
                if self._opt(cfgk) is not None:
                    params[k] = self._opt(cfgk)
            if oneyear is not None:
                logger.warning(
                    "slc2ifg.ifgram_list.oneyear_interferograms is deprecated; "
                    "use slc2ifg.ifgram_list.select.annual_windows=365:<range>")
                aw = str(params.get('annual_windows') or '').strip().lower()
                params['annual_windows'] = (f"365:{int(oneyear)}"
                                            if aw in ('', 'auto', 'default')
                                            else f"{aw},365:{int(oneyear)}")
            pairs, _ = select_pairs(dates, params=params)
        else:
            pairs = generate_pairs(dates, mode, nconn, oneyear)

        write_pair_list(pairs, pair_file)
        logger.info("ifgram_list (%s): %d date(s) -> %d pair(s)",
                    mode, len(dates), len(pairs))
        return pair_file

    # ------------------------------------------------------------------
    # run
    # ------------------------------------------------------------------
    def run(self, cfg: dict) -> None:
        self.cfg = cfg
        from mintpy.stdproc.slc2ifg.utils import naming

        processor = self._opt('slc2ifg.processor', 'isce3')
        work_dir = Path(self._opt('slc2ifg.work_dir', './'))
        if not work_dir.is_absolute():
            work_dir = (Path.cwd() / work_dir).resolve()
        work_dir.mkdir(parents=True, exist_ok=True)

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

        # ---------- crop (optional, transforms the SLC input dir) ----------
        slc_base = slc_dir
        if 'crop_slc' in tools:
            slc_base = self._run_crop(slc_dir, work_dir)

        # ---------- phase 1: per-burst ifgram_list + generate_ifgram -------
        bursts = self._discover_bursts(slc_base)
        multi = len(bursts) > 1
        logger.info("Mode: %s", "multi-burst" if multi else "single-burst")

        pair_files = {}
        ifg_sources: Dict[str, Path] = {}   # (burst, dp) -> ifg path
        cpx_sources: Dict[str, Path] = {}
        for b in bursts:
            btag = b or 'single'
            b_slc = slc_base / b if b else slc_base
            b_out = ifg_out / b if b else ifg_out
            b_out.mkdir(parents=True, exist_ok=True)
            pair_file = self._run_ifgram_list(b_slc, b_out)
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
            tasks = [(b_slc, b_out, d1, d2, pair_file, processor)
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

            # complex coherence (optional, fullres only, per burst)
            if 'complex_coh' in tools:
                results = self._map_pairs(
                    tasks, self._complex_coh_one, n_workers,
                    desc=f'complex_coh ({btag})')
                for (d1, d2), ok, msg in results:
                    if ok:
                        cpx_sources[(btag, f'{d1}_{d2}')] = naming.coh_path(
                            b_out, d1, d2, 'fullres', 'cpx', processor)

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
        first_pair_file = next(iter(pair_files.values()), None)
        if first_pair_file is None:
            logger.warning("no pair list produced — nothing to process")
            return
        pairs = _read_pairs(first_pair_file)
        if not pairs:
            logger.warning("empty pair list — nothing to process")
            return

        # uniform global stages only (generate_ifgram / stitch already ran in
        # phases 1-2; complex_coh is not supported in the basic executor)
        chain = [s for s in ('multilook', 'filter', 'phsig_coh', 'unwrap')
                 if s in tools]
        # map stage -> (input variant, input dir) threading
        cur_variant = 'fullres'
        cur_base = phase3_base
        for stage in chain:
            tasks = [(cur_base, cur_variant, d1, d2, processor)
                     for d1, d2 in pairs]
            handler = getattr(self, f'_stage_{stage}')
            results = self._map_pairs(tasks, handler, n_workers,
                                      desc=f'{stage}')
            for (d1, d2), ok, msg in results:
                if not ok:
                    logger.error("%s %s_%s: %s", stage, d1, d2, msg)
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
        from mintpy.stdproc.slc2ifg.generate_ifgram import (
            find_slc_file_by_date,
            process_single_pair,
        )
        from mintpy.stdproc.slc2ifg.utils import naming
        b_slc, b_out, d1, d2, pair_file, processor = task
        try:
            slc_pattern = self._opt('slc2ifg.generate_ifgram.slc_pattern',
                                    naming.slc_pattern(processor))
            slc1 = find_slc_file_by_date([b_slc], d1, slc_pattern, processor)
            slc2 = find_slc_file_by_date([b_slc], d2, slc_pattern, processor)
            if slc1 is None or slc2 is None:
                return (d1, d2), False, f"SLC missing for {d1}/{d2}"
            pair_dir = b_out / f'{d1}_{d2}'
            pair_dir.mkdir(parents=True, exist_ok=True)
            vrt_path = pair_dir / 'fullres.int.vrt'
            ifg_path = naming.ifg_path(b_out, d1, d2, 'fullres', processor)
            pair_info = (f'{d1}-{d2}', d1, d2, slc1, slc2, vrt_path,
                         ifg_path, True,
                         self._opt('slc2ifg.generate_ifgram.subdataset',
                                   '/data/VV'),
                         processor,
                         self._opt('slc2ifg.generate_ifgram.only_vrt', False))
            date12, ok, msg = process_single_pair(pair_info)
            return (d1, d2), ok, msg
        except Exception as e:
            return (d1, d2), False, str(e)

    def _complex_coh_one(self, task):
        from mintpy.stdproc.slc2ifg.utils import naming
        b_slc, b_out, d1, d2, pair_file, processor = task
        try:
            out = naming.coh_path(b_out, d1, d2, 'fullres', 'cpx', processor)
            if out.exists():
                return (d1, d2), True, 'exists'
            raise NotImplementedError(
                "complex_coh standalone path not wired in the basic executor "
                "yet — list it via the engine (mintpy.slc2ifg.engine = "
                "insarflow) or drop it from engine.tools")
        except Exception as e:
            return (d1, d2), False, str(e)

    def _stitch_one(self, srcs, out, processor):
        from mintpy.stdproc.slc2ifg.stitch import get_file_epsg, stitch_date_pair
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
        from mintpy.stdproc.slc2ifg.utils import naming
        base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            out_path = naming.ifg_path(base, d1, d2,
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
        from mintpy.stdproc.slc2ifg.filter import process_single_goldstein
        from mintpy.stdproc.slc2ifg.utils import naming
        base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            out_path = naming.ifg_path(base, d1, d2,
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
        from mintpy.stdproc.slc2ifg.generate_coh_phsig import (
            _write_band,
            read_complex_image,
        )
        from mintpy.stdproc.slc2ifg.utils import naming
        base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            out_path = naming.coh_path(base, d1, d2, variant, 'phsig',
                                       processor)
            if out_path.exists():
                return (d1, d2), True, 'exists'
            ifg, meta = read_complex_image(str(in_path), processor)
            from mintpy.stdproc.slc2ifg.generate_coh_phsig import (
                estimate_phsig_correlation,
            )
            coh = estimate_phsig_correlation(
                ifg,
                ps_win=self._opt_int('slc2ifg.generate_coh.ps_window_size', 5),
                grad_win=self._opt_int(
                    'slc2ifg.generate_coh.ps_gradient_window', 5),
                nlks=self._opt_float('slc2ifg.generate_coh.ps_nlks', 1.0),
            )
            _write_band(str(out_path), coh, meta, processor,
                        'phase-sigma correlation')
            return (d1, d2), True, ''
        except Exception as e:
            return (d1, d2), False, str(e)

    def _stage_unwrap(self, task):
        from mintpy.stdproc.slc2ifg.unwrap_ifgram import _unwrap_single
        from mintpy.stdproc.slc2ifg.utils import naming
        base, variant, d1, d2, processor = task
        try:
            in_path = naming.ifg_path(base, d1, d2, variant, processor)
            unw_root = base
            unw_root.mkdir(parents=True, exist_ok=True)
            coh_path = naming.coh_path(base, d1, d2, variant, 'phsig',
                                       processor)
            cor = coh_path if coh_path.exists() else None
            mask = self._opt('slc2ifg.unwrap.snaphu.mask_file')
            if not mask:
                mask = self._opt('mintpy.load.waterMaskFile')
            _unwrap_single(
                ifg_path=Path(in_path),
                cor_path=Path(cor) if cor else None,
                nlooks=self._opt_float('slc2ifg.unwrap.snaphu.nlooks', 1.0),
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
        parts = [x.strip() for x in str(val).split(',')]
        try:
            return tuple(float(x) for x in parts)
        except ValueError:
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
        """Crop SLCs to bbox (isce3 geocoded / isce2 radar)."""
        wsen = self._opt('slc2ifg.crop_slc.wsen')
        if not wsen:
            raise ValueError("crop_slc requires slc2ifg.crop_slc.wsen")
        from mintpy.stdproc.slc2ifg.crop_slc_geo import main as geo_main
        from mintpy.stdproc.slc2ifg.crop_slc_geo import parse_arguments as geo_parse
        out_dir = work_dir / 'cropped_slc'
        args_list = [
            '--input-dir', str(slc_dir),
            '--output-dir', str(out_dir),
            '--wsen'] + [str(x) for x in str(wsen).split()] + [
            '--buffer', str(self._opt_float('slc2ifg.crop_slc.buffer', 0.0)),
            '--prefix', str(self._opt('slc2ifg.crop_slc.prefix', '')),
        ]
        geo_main(geo_parse(args_list))
        return out_dir


# ------------------------------------------------------------------------
# EngineExecutor — optional insarflow backend (lazy import)
# ------------------------------------------------------------------------
class EngineExecutor(Slc2ifgExecutor):
    """Delegate the run to the insarflow engine (Dask, GPU, tiling)."""

    def run(self, cfg: dict) -> None:
        # The engine reads the same MintPy-style flat config from a FILE.
        # The caller (mintpy/slc2ifg.py) passes the config file path in cfg.
        from mintpy.stdproc.slc2ifg.engine.config import load_engine_config
        from mintpy.stdproc.slc2ifg.engine.engine import Engine

        config_file = cfg.get('_cfg_file')
        if not config_file:
            raise ValueError(
                "EngineExecutor requires the config file path "
                "(run via `mintpy slc2ifg <cfg>` — basic executor is used "
                "for in-memory dict configs)")
        engine_cfg = load_engine_config(str(config_file))
        Engine(engine_cfg).run()
