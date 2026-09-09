#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Processing chain — the data source for engine graph construction.

The chain describes "which stages the pipeline consists of, which tool each
stage uses, and where its products are written":
it replaces the hard-coded stage order inside ``build_graph()``. Three-level
merge rules (see ``docs/engine_stages_design.md`` §3/§5):

1. **Built-in default chain** (``DEFAULT_CHAIN``): the official pipeline
   maintained with each MintPy release;
2. **Tool-declared chain fragments**: third-party tools declare their own
   stages in ``Tool.stage``, and once registered they are merged into the
   known stage table by :func:`collect_fragments`;
3. **User configuration override**: ``engine.stages`` lists stages
   explicitly (default = built-in default chain).

Resource allocation (CPU/GPU/memory) is not handled here — it is declared by
``Tool.resource`` and consumed by the scheduler
(see ``docs/engine_design.md`` §6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------
# Stage
# ------------------------------------------------------------------------
@dataclass
class Stage:
    """A processing stage (one link in the chain).

    Parameters
    ----------
    name : str
        Stage name (also the name used in the ``engine.stages`` config).
    tool : str
        Tool name in ``TOOL_REGISTRY`` (defaults to the same as ``name``).
    scope : str
        ``'source'`` (input directory transform, e.g. crop) / ``'plan'``
        (eager planning, e.g. ifgram_list) / ``'per_burst'`` (one per burst) /
        ``'stitch'`` (per-burst → global fan-in) / ``'global'`` (one per
        date pair).
    core : bool
        ``True`` = infrastructure stage of the default chain (informational:
        since the whitelist gating below is purely explicit, core stages only
        run when they are in ``engine.stages``/``AUTO_TOOLS`` or listed in
        ``engine.stages`` — nothing is force-enabled).
    optional : bool
        ``True`` means disabled in the default chain; enabled only when
        listed in the ``engine.stages`` whitelist or explicitly in
        ``engine.stages`` (e.g. complex_coh, crop_slc).
    requires : tuple[str, ...]
        Names of prerequisite stages (must appear before this stage,
        otherwise an error is raised).
    out_dir : str
        Engine directory key where products are written:
        ``'ifgram'|'ml'|'filter'|'unwrap'|'stitched'``
        (mapped to the corresponding ``EngineConfig`` attribute; most are
        equal under the current unified output directory).
    desc : str
        Human-readable description (shown by ``--list-stages``).
    """

    name: str
    tool: str = ''
    scope: str = 'global'
    core: bool = False
    optional: bool = False
    requires: Tuple[str, ...] = ()
    out_dir: str = 'ifgram'
    desc: str = ''


# ------------------------------------------------------------------------
# Built-in default chain (updated with each MintPy release)
# ------------------------------------------------------------------------
DEFAULT_CHAIN: List[Stage] = [
    Stage('crop_slc', scope='source', optional=True,
          desc='Crop SLC to a geographic bbox (requires slc2ifg.bbox)'),
    Stage('ifgram_list', scope='plan', core=True,
          desc='Generate date pair list (eager; runs before graph building)'),
    Stage('generate_ifgram', scope='per_burst', core=True,
          desc='Generate fullres interferograms per burst (dolphin)'),
    Stage('complex_coh', scope='per_burst', optional=True,
          desc='Complex coherence per burst (fullres only; not consumed downstream)'),
    Stage('stitch', scope='stitch',
          desc='Stitch per-burst products across bursts → global (required for multi-burst)'),
    Stage('multilook', out_dir='ml',
          desc='Multilook: fullres→mli / filt→filt_mli'),
    Stage('filter', out_dir='filter',
          desc='Filter (Goldstein): mli→filt_mli / fullres→filt'),
    Stage('phsig_coh',
          desc='Phase-sigma coherence (follows the variant of its input ifg); '
               'NOT enabled by default — list it explicitly to compute'),
    Stage('unwrap', out_dir='unwrap',
          desc='Unwrap (SNAPHU; algorithm switched by slc2ifg.unwrap.algorithm; '
               'coherence input via slc2ifg.unwrap.coh_type: phsig|complex|none)'),
    Stage('atmosphere', requires=('unwrap',), optional=True, out_dir='unwrap',
          desc='Atmospheric correction (skeleton: method=none is pass-through copy)'),
]


# ------------------------------------------------------------------------
# Three-level merge resolution
# ------------------------------------------------------------------------
def collect_fragments() -> Dict[str, Stage]:
    """Collect self-declared chain fragments from registered tools (stages
    outside the built-in default chain).

    Third-party tools declare a ``StageSpec`` in ``Tool.stage`` and it is
    merged in once registered; built-in tools are covered by
    ``DEFAULT_CHAIN`` and are not collected again.
    """
    from mintpy.stdproc.engine.tool import TOOL_REGISTRY

    builtin = {s.name for s in DEFAULT_CHAIN}
    out: Dict[str, Stage] = {}
    for name, cls in TOOL_REGISTRY.items():
        spec = getattr(cls, 'stage', None)
        if spec is None or spec.name in builtin:
            continue
        out[spec.name] = spec.to_stage()
        logger.info("Registered stage fragment '%s' from tool '%s'",
                    spec.name, name)
    return out


def resolve_chain(
    stages_cfg: Optional[List[str]],
    tools: List[str],
) -> List[Stage]:
    """Three-level merge: user-explicit stages (or built-in default chain) +
    tool fragments + whitelist gating.

    Parameters
    ----------
    stages_cfg : list of str or None
        Parsed result of ``engine.stages``; ``None`` = built-in default
        chain.
    tools : list of str
        ``engine.stages`` whitelist (already includes the special toggling
        for complex_coh/crop_slc).

    Returns
    -------
    list of Stage
        The effective chain after enabling (in execution order).

    Raises
    ------
    ValueError
        Unknown stage, or incorrect ``requires`` ordering.
    """
    known: Dict[str, Stage] = {s.name: s for s in DEFAULT_CHAIN}
    known.update(collect_fragments())

    if stages_cfg:
        explicit = [s.strip() for s in stages_cfg if s.strip()]
        base: List[Stage] = []
        for name in explicit:
            if name not in known:
                raise ValueError(
                    f"Unknown stage '{name}' in engine.stages. "
                    f"Available: {sorted(known)}")
            base.append(known[name])
    else:
        explicit = None
        base = list(DEFAULT_CHAIN)

    # requires ordering check: prerequisite stages must already have appeared
    seen: List[str] = []
    for s in base:
        for r in s.requires:
            if r not in seen:
                raise ValueError(
                    f"Stage '{s.name}' requires '{r}', which must appear "
                    f"before it in the chain (got: {[x.name for x in base]})")
        seen.append(s.name)

    # Gating: with explicit stages, run exactly those listed (authoritative);
    # the default chain runs only the stages explicitly whitelisted in
    # ``engine.stages``.  Nothing is force-enabled: e.g. phsig_coh is computed
    # only when explicitly requested, and AUTO_TOOLS defines what
    # ``engine.stages = auto`` runs.
    if explicit is not None:
        return base

    out: List[Stage] = []
    for s in base:
        if (s.tool or s.name) in tools:
            out.append(s)
    return out


# ------------------------------------------------------------------------
# Display
# ------------------------------------------------------------------------
def format_chain(chain: List[Stage]) -> str:
    """Human-readable chain description (``--list-stages`` output)."""
    lines = [f"Processing chain: {len(chain)} stage(s)"]
    for i, s in enumerate(chain, 1):
        tag = s.scope
        if s.optional:
            tag += ", optional"
        if s.core:
            tag += ", core"
        lines.append(
            f"  {i:2d}. {s.name:<16} tool={s.tool or s.name:<14} "
            f"scope={s.scope:<10} "
            f"out_dir={s.out_dir:<8}{('requires=' + ','.join(s.requires)) if s.requires else ''}")
        if s.desc:
            lines.append(f"       {s.desc}")
    return '\n'.join(lines)
