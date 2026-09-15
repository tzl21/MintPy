#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Auto-discovery of coherence rasters under the ifgram product tree.

The pipeline writes coherence as
``<root>/{date1}_{date2}/{variant}.{kind}.coh.tif`` (see
:mod:`mintpy.stdproc.utils.naming`); ``{kind}`` is ``phsig`` (phase-sigma) or
``cpx`` (complex) and ``{variant}`` one of ``fullres/mli/filt/filt_mli``.
Both are recovered from the filename, so there is no ``slc2ifg.coh_kind`` /
``slc2ifg.coh_variant`` configuration key: the default lookup is
``<work_dir>/ifgrams/*/*.coh.tif``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

from .naming import COH_KINDS, IFG_VARIANTS

logger = logging.getLogger(__name__)

#: kind preference (phase-sigma is the standard engine-chain product)
KIND_PREFERENCE: Tuple[str, ...] = ('phsig', 'cpx')

#: variant preference (most-processed first)
VARIANT_PREFERENCE: Tuple[str, ...] = ('filt_mli', 'filt', 'mli', 'fullres')


def parse_coh_name(name: Union[str, Path]) -> Optional[Tuple[str, str]]:
    """Parse a coherence filename into ``(variant, kind)``.

    Returns ``None`` when the name is not a ``{variant}.{kind}.coh[.tif]``
    coherence raster.
    """
    stem = Path(name).name
    if stem.endswith('.tif'):
        stem = stem[:-4]
    if not stem.endswith('.coh'):
        return None
    stem = stem[:-4]
    kind = None
    for k in COH_KINDS:
        if stem.endswith('.' + k):
            kind = k
            stem = stem[:-(len(k) + 1)]
            break
    if kind is None:
        return None
    variant = stem if stem in IFG_VARIANTS else 'fullres'
    return variant, kind


def list_coh_rasters(ifgram_root: Union[str, Path], date1: str,
                     date2: str) -> List[Tuple[Path, str, str]]:
    """List ``(path, variant, kind)`` coherence rasters for one date pair."""
    pair_dir = Path(ifgram_root) / f'{date1}_{date2}'
    if not pair_dir.is_dir():
        return []
    out: List[Tuple[Path, str, str]] = []
    for pattern in ('*.coh.tif', '*.coh'):
        for p in sorted(pair_dir.glob(pattern)):
            parsed = parse_coh_name(p.name)
            if parsed is not None:
                out.append((p, parsed[0], parsed[1]))
    return out


def find_coh_raster(ifgram_root: Union[str, Path], date1: str, date2: str,
                    prefer_variant: Optional[str] = None,
                    prefer_kind: Optional[str] = None,
                    require_kind: bool = False) -> Optional[Path]:
    """Find the best coherence raster for ``{date1}_{date2}``.

    ``None`` when the pair has no coherence raster — i.e. "no coherence file
    exists".  Otherwise the filename encodes kind and variant; the preference
    is ``prefer_kind`` (default phsig) then ``prefer_variant`` (default
    filt_mli), then the canonical order.  With ``require_kind=True`` only
    rasters of exactly ``prefer_kind`` are considered.
    """
    cands = list_coh_rasters(ifgram_root, date1, date2)
    if require_kind and prefer_kind:
        cands = [c for c in cands if c[2] == prefer_kind]
    if not cands:
        return None

    kind_order = ([prefer_kind] if prefer_kind else []) + [
        k for k in KIND_PREFERENCE if k != prefer_kind]
    variant_order = ([prefer_variant] if prefer_variant else []) + [
        v for v in VARIANT_PREFERENCE if v != prefer_variant]

    def rank(item):
        _p, variant, kind = item
        k = kind_order.index(kind) if kind in kind_order else len(kind_order)
        v = variant_order.index(variant) if variant in variant_order else len(variant_order)
        return (k, v, item[0].name)

    return min(cands, key=rank)[0]
