#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Configuration for the MintPy slc2ifg engine.

Configurations are MintPy-style (no section header; a virtual
``[slc2ifg]`` section is added internally by :func:`read_config`).
The engine section controls scheduling/resources/product management:

    engine.stages = auto             # auto (default chain) or explicit stage list (exactly the listed stages run; the single authoritative chain spec)
    engine.scheduler = auto          # threaded | distributed (auto -> threaded)
    engine.max_workers = auto        # task concurrency (default: n_cpu; single source of truth)
    engine.mem_limit_gb = auto       # host memory budget (default: 80%)
    engine.gpu = auto                # auto | true | false
    engine.gpu_mem_limit_gb = auto   # default: 70% of GPU memory
    engine.tile_size = auto          # intra-ifg tiling (M4)
    engine.keep_intermediates = none # none | all | stage list
    engine.work_dir = auto           # default: <work_dir>/engine

This module is self-contained: it owns the config I/O helpers (formerly part
of the removed ``run_slc2ifg.py``) plus the engine configuration model.
"""

from __future__ import annotations

import configparser
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _template_file() -> Path:
    """Path of the default configuration template shipped with the package.

    Resolved from the mintpy package via ``importlib.resources`` WITHOUT
    importing the stdproc subpackage itself — the engine's config layer must
    stay importable without osgeo (unit tests / plugin contexts).
    """
    import importlib.resources
    try:
        ref = (importlib.resources.files('mintpy')
               / 'stdproc' / 'template' / 'slc2ifg.cfg')
        return Path(str(ref))
    except (ImportError, ModuleNotFoundError):
        # fallback: engine source tree (uninstalled checkout / tests)
        return Path(__file__).resolve().parents[1] / 'template' / 'slc2ifg.cfg'



# ------------------------------------------------------------------------
# Config I/O helpers (MintPy-style, virtual [slc2ifg] section)
# ------------------------------------------------------------------------
def _clean_config_value(raw: str) -> str:
    """Remove BOM and inline comments (``# ...``), then strip whitespace.

    Commas are legitimate value separators (e.g. ``engine.tools = a,b,c``),
    so they are NOT treated as comment delimiters — only ``#`` is.
    """
    if raw.startswith('\ufeff'):
        raw = raw[1:]
    # Treat '#' as an inline comment only when preceded by whitespace, so
    # legitimate values containing '#' (e.g. URL fragments) survive.
    if ' #' in raw:
        raw = raw.split(' #', 1)[0]
    return raw.strip()


def get_opt(config: configparser.ConfigParser, key: str,
            fallback: Optional[str] = None) -> Optional[str]:
    """Get string option from the 'slc2ifg' section."""
    if config.has_option('slc2ifg', key):
        cleaned = _clean_config_value(config.get('slc2ifg', key))
        if cleaned.lower() in ('auto', 'none', 'null', 'off', '') :
            return fallback
        return cleaned
    return fallback


def get_bool_opt(config: configparser.ConfigParser, key: str,
                 fallback: bool = False) -> bool:
    """Get boolean option from the 'slc2ifg' section."""
    if config.has_option('slc2ifg', key):
        cleaned = _clean_config_value(config.get('slc2ifg', key))
        if cleaned.lower() == 'auto' or cleaned == '':
            return fallback
        return cleaned.lower() in ('true', 'yes', '1', 'on')
    return fallback


def get_int_opt(config: configparser.ConfigParser, key: str,
                fallback: Optional[int] = None) -> Optional[int]:
    """Get integer option from the 'slc2ifg' section."""
    if config.has_option('slc2ifg', key):
        cleaned = _clean_config_value(config.get('slc2ifg', key))
        if cleaned.lower() in ('auto', 'none', 'null', 'off', ''):
            return fallback
        try:
            return int(cleaned)
        except ValueError:
            logger.warning(f"Invalid integer for {key}: {cleaned}")
            return fallback
    return fallback


def get_float_opt(config: configparser.ConfigParser, key: str,
                  fallback: Optional[float] = None) -> Optional[float]:
    """Get float option from the 'slc2ifg' section."""
    if config.has_option('slc2ifg', key):
        cleaned = _clean_config_value(config.get('slc2ifg', key))
        if cleaned.lower() in ('auto', 'none', 'null', 'off', ''):
            return fallback
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Invalid float for {key}: {cleaned}")
            return fallback
    return fallback


def has_real_value(config: configparser.ConfigParser, key: str) -> bool:
    """Return True when ``key`` is set to a real (non-auto, non-empty) value.

    Used to decide whether a deprecated legacy key should be consulted: only
    when the new key is *absent* (or explicitly ``auto``) does the engine fall
    back to the legacy key.
    """
    if not config.has_option('slc2ifg', key):
        return False
    cleaned = _clean_config_value(config.get('slc2ifg', key))
    return cleaned != '' and cleaned.lower() != 'auto'


def read_config(user_config_file: Optional[str]) -> configparser.ConfigParser:
    """Read and merge template and user configuration.

    Returns a ``ConfigParser`` whose options live in the virtual
    ``[slc2ifg]`` section (config files have no section header).
    ``RawConfigParser`` is used so ``%`` in comments/values is never treated
    as interpolation syntax.
    """
    config = configparser.RawConfigParser(strict=False)

    template_file = _template_file()
    if template_file.exists():
        with open(template_file, 'r', encoding='utf-8-sig') as f:
            template_content = f.read()
        if not template_content.lstrip().startswith('['):
            template_content = '[slc2ifg]\n' + template_content
        config.read_string(template_content)
    else:
        logger.warning(f"Template file not found: {template_file}")

    if not user_config_file:
        return config

    user_path = Path(user_config_file)
    if not user_path.exists():
        logger.warning(f"User configuration file not found: {user_config_file}")
        return config

    with open(user_path, 'r', encoding='utf-8-sig') as f:
        content = f.read()

    if not content.lstrip().startswith('['):
        logger.info("User config has no section header, adding [slc2ifg] automatically.")
        content = '[slc2ifg]\n' + content
    else:
        logger.info(f"Loaded configuration from {user_config_file}")

    config.read_string(content)
    if user_config_file:
        config.config_dir = str(Path(user_config_file).parent.resolve())
    else:
        config.config_dir = str(Path.cwd())
    return config


# ------------------------------------------------------------------------
# Engine configuration model
# ------------------------------------------------------------------------
#: Tools enabled by 'engine.tools = auto' (explicit list — nothing is
#: force-enabled; phsig_coh is NOT included: it runs only when listed).
AUTO_TOOLS = [
    'ifgram_list',
    'generate_ifgram',
    'multilook',
    'filter',
    'stitch',
    'unwrap',
]


@dataclass
class EngineConfig:
    """Parsed engine configuration."""

    # --- paths ---
    slc_input: str
    work_dir: Path
    engine_work_dir: Path
    processor: str

    # --- tools ---
    tools: List[str] = field(default_factory=list)
    #: engine.stages — explicit processing chain (None = built-in default)
    stages: Optional[List[str]] = None

    # --- scheduling / resources ---
    scheduler: str = 'threaded'
    max_workers: Optional[int] = None
    mem_limit_gb: Optional[float] = None
    gpu: str = 'auto'
    gpu_mem_limit_gb: Optional[float] = None
    tile_size: Optional[int] = None

    # --- product management ---
    keep_intermediates: str = 'none'
    manifest_hash: bool = False

    # --- raw config (for tool params) ---
    raw: object = None

    @property
    def ifgram_out_dir(self) -> Path:
        """Root directory of ALL interferogram products: ``<work_dir>/ifgrams``.

        Interferogram filenames are unique (date pair + variant), so every
        processing result — full-res, multilooked, filtered, coherence,
        unwrapped, connected components — lives in this one directory tree:
        ``<ifgram_out_dir>/{date1}_{date2}/xxx.ext``.
        """
        return self.work_dir / 'ifgrams'

    @property
    def stitched_dir(self) -> Path:
        return self.work_dir / 'stitched'

    @property
    def ml_out_dir(self) -> Path:
        # unified: same tree as the interferograms
        return self.ifgram_out_dir

    @property
    def filter_out_dir(self) -> Path:
        # unified: same tree as the interferograms
        return self.ifgram_out_dir

    @property
    def unwrap_out_dir(self) -> Path:
        # unified: same tree as the interferograms
        return self.ifgram_out_dir


def load_engine_config(config_file: Optional[str]) -> EngineConfig:
    """Read a MintPy-style config file and extract the engine section."""
    config = read_config(config_file)
    from mintpy.stdproc.config_map import normalize_config
    normalize_config(config)
    work_dir = get_opt(config, 'slc2ifg.work_dir', fallback='./')
    work_path = Path(work_dir)
    if not work_path.is_absolute():
        cfg_dir = getattr(config, 'config_dir', str(Path.cwd()))
        work_path = (Path(cfg_dir) / work_path).resolve()
    else:
        work_path = work_path.resolve()
    work_path.mkdir(parents=True, exist_ok=True)

    processor = get_opt(config, 'slc2ifg.processor', fallback='isce3')
    if processor not in ('isce2', 'isce3'):
        logger.warning("Invalid processor '%s', defaulting to isce3", processor)
        processor = 'isce3'

    slc_input = get_opt(config, 'slc2ifg.slc_input')
    if not slc_input:
        raise ValueError("Missing required configuration: slc2ifg.slc_input")
    # slc_input may be a glob (e.g. 'xxx/t*/*/'); keep the pattern intact and
    # only make it absolute relative to the config file directory.
    slc_path = Path(slc_input)
    if not slc_path.is_absolute():
        cfg_dir = getattr(config, 'config_dir', str(Path.cwd()))
        slc_path = Path(cfg_dir) / slc_path

    # --- engine section: processing chain ---
    # engine.stages is the single authoritative chain spec.  The legacy
    # engine.tools key is a deprecated alias (warned and mapped to stages).
    stages_cfg = get_opt(config, 'engine.stages', fallback=None)
    stages: Optional[List[str]] = None
    if stages_cfg:
        stages = [s.strip() for s in stages_cfg.split(',') if s.strip()]
    else:
        tools_cfg = get_opt(config, 'engine.tools', fallback=None)
        if tools_cfg and str(tools_cfg).strip().lower() not in ('auto', ''):
            logger.warning(
                "engine.tools is deprecated; use engine.stages instead")
            stages = [t.strip() for t in str(tools_cfg).split(',') if t.strip()]

    # tools whitelist (drives engine.py run_* / entry-mode flags): derived
    # from the explicit chain, else the lean AUTO_TOOLS default.
    if stages is not None:
        tools = list(stages)
    else:
        tools = list(AUTO_TOOLS)

    engine_work = get_opt(config, 'engine.work_dir', fallback=str(work_path / 'engine'))
    engine_work_path = Path(engine_work)
    if not engine_work_path.is_absolute():
        engine_work_path = (work_path / engine_work_path).resolve()
    engine_work_path.mkdir(parents=True, exist_ok=True)

    return EngineConfig(
        slc_input=str(slc_path),
        work_dir=work_path,
        engine_work_dir=engine_work_path,
        processor=processor,
        tools=tools,
        stages=stages,
        scheduler=get_opt(config, 'engine.scheduler', fallback='threaded') or 'threaded',
        # engine.max_workers is the single concurrency knob (legacy
        # slc2ifg.max_workers is intentionally NOT consulted anymore).
        max_workers=get_int_opt(config, 'engine.max_workers'),
        mem_limit_gb=get_float_opt(config, 'engine.mem_limit_gb'),
        gpu=get_opt(config, 'engine.gpu', fallback='auto') or 'auto',
        gpu_mem_limit_gb=get_float_opt(config, 'engine.gpu_mem_limit_gb'),
        tile_size=get_int_opt(config, 'engine.tile_size'),
        keep_intermediates=get_opt(config, 'engine.keep_intermediates', fallback='none') or 'none',
        manifest_hash=get_bool_opt(config, 'engine.manifest_hash', fallback=False),
        raw=config,
    )