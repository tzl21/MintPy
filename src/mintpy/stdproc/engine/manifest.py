#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg engine (moved from insarflow)
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Lightweight product manifest for the MintPy slc2ifg engine.

Records every file the engine produces (path, size, mtime, producing node)
so that the cleanup policy (default: keep only unw/conncomp/phsig) can
safely delete intermediates *after* the DAG completes, and so deleted
products remain re-buildable (idempotent tool re-runs).

Design decision (docs/engine_design.md §13): light by default — no content
hash unless ``manifest_hash`` is enabled.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MANIFEST_NAME = 'manifest.json'


class Manifest:
    """JSON-backed manifest of engine-produced files."""

    def __init__(self, path: Path, use_hash: bool = False):
        self.path = Path(path)
        self.use_hash = use_hash
        self._entries: Dict[str, Dict[str, Any]] = {}  # abs path -> info
        self._meta: Dict[str, Any] = {}  # run-level metadata (e.g. chain)

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path, use_hash: bool = False) -> 'Manifest':
        m = cls(path, use_hash=use_hash)
        if m.path.exists():
            try:
                data = json.loads(m.path.read_text())
                m._entries = data.get('files', {})
                m._meta = data.get('meta', {})
            except Exception as e:
                logger.warning("Failed to load manifest %s: %s", m.path, e)
        return m

    # ------------------------------------------------------------------
    def set_meta(self, **meta: Any) -> None:
        """Record run-level metadata (e.g. the effective processing chain)."""
        self._meta.update(meta)

    # ------------------------------------------------------------------
    def record(self, node_key: str, paths: List[Path]) -> None:
        """Record files produced by a node."""
        for p in paths:
            p = Path(p)
            if not p.exists():
                continue
            info = {
                'size': p.stat().st_size,
                'mtime': p.stat().st_mtime,
                'node': node_key,
                'deleted': False,
            }
            if self.use_hash:
                info['sha256'] = self._sha256(p)
            self._entries[str(p.resolve())] = info

    def mark_deleted(self, paths: List[Path]) -> None:
        for p in paths:
            key = str(Path(p).resolve())
            if key in self._entries:
                self._entries[key]['deleted'] = True
            else:
                self._entries[key] = {'deleted': True, 'node': 'cleanup'}

    # ------------------------------------------------------------------
    def produced_files(self, deleted: bool = False) -> List[Path]:
        """Paths of recorded files (optionally only non-deleted)."""
        return [
            Path(k) for k, v in self._entries.items()
            if v.get('deleted', False) == deleted
        ]

    def nodes(self) -> List[str]:
        return sorted({v.get('node', '') for v in self._entries.values() if v.get('node')})

    # ------------------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'meta': self._meta, 'files': self._entries},
                                  indent=1))
        tmp.replace(self.path)

    @staticmethod
    def _sha256(path: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()


# ------------------------------------------------------------------------
# Cleanup policy
# ------------------------------------------------------------------------
def plan_cleanup(
    manifest: Manifest,
    keep_policy: str,            # 'none' | 'all' | comma-separated stages
    keep_variants: Optional[List[str]] = None,
    keep_suffixes: Optional[List[str]] = None,
) -> List[Path]:
    """Compute the list of engine-produced files to delete.

    Parameters
    ----------
    manifest : Manifest
        Recorded engine products.
    keep_policy : str
        ``'none'`` (default): keep only final products + SLC inputs + pair
        lists; ``'all'``: keep everything; otherwise a comma-separated list
        of tool/stage names to keep (``'generate_ifgram,filter'``).
    keep_variants : list of str, optional
        Interferogram variants always kept (e.g. ``['fullres']``).
    keep_suffixes : list of str, optional
        Extra filename suffixes always kept.

    Returns
    -------
    list of Path
        Files to delete (only engine-produced, non-deleted entries).
    """
    if keep_policy == 'all':
        return []

    keep_variants = keep_variants or []
    keep_suffixes = keep_suffixes or []
    # Final products + inputs are always kept
    keep_suffixes = list(keep_suffixes) + [
        '.unw', '.unw.tif', '.unw.conncomp', '.unw.conncomp.tif', '.conncomp', '.conncomp.tif',
        '.phsig.coh', '.phsig.coh.tif',
        '.slc', '.slc.tif', '.slc.h5',
        'ifgram_list.txt', MANIFEST_NAME,
    ]
    if keep_policy != 'none':
        keep_nodes = {s.strip() for s in keep_policy.split(',') if s.strip()}
    else:
        keep_nodes = set()

    to_delete = []
    for p in manifest.produced_files(deleted=False):
        name = p.name
        if _kept_by_name(name, keep_suffixes, keep_variants):
            continue
        if _kept_by_node(p, keep_nodes, manifest):
            continue
        if p.is_dir() and _dir_contains_kept(
                p, keep_suffixes, keep_variants, keep_nodes, manifest):
            # A directory physically containing any kept final product
            # (e.g. unw/conncomp/phsig inside ``{date1}_{date2}/``) must
            # NOT be removed recursively — rmtree would destroy the kept
            # product along with the intermediates.
            continue
        to_delete.append(p)
    return to_delete


def _kept_by_name(name: str, keep_suffixes: List[str],
                  keep_variants: List[str]) -> bool:
    """Whether a filename is protected by the suffix / variant keep rules."""
    if any(name.endswith(s) for s in keep_suffixes):
        return True
    if any(name.startswith(v) or f"_{v}." in name for v in keep_variants):
        return True
    return False


def _kept_by_node(p: Path, keep_nodes, manifest) -> bool:
    """Whether a recorded product is protected by the stage-name keep rule."""
    if not keep_nodes:
        return False
    node = manifest._entries.get(str(p.resolve()), {}).get('node', '')
    return node.split('#', 1)[0] in keep_nodes


def _dir_contains_kept(dirpath: Path, keep_suffixes: List[str],
                       keep_variants: List[str], keep_nodes,
                       manifest) -> bool:
    """True if any file under ``dirpath`` is protected by the keep policy."""
    for root, _dirs, files in os.walk(dirpath):
        rootp = Path(root)
        for fn in files:
            fp = rootp / fn
            if _kept_by_name(fn, keep_suffixes, keep_variants):
                return True
            if _kept_by_node(fp, keep_nodes, manifest):
                return True
    return False


#: Companion files that must be removed together with a deleted product.
#: ENVI writes ``{stem}.hdr`` (extension replaced); ISCE2 XML / GDAL aux.xml
#: / VRT are appended to the full filename.
_COMPANION_APPEND_SUFFIXES = ('.xml', '.aux.xml', '.vrt')


def _companions(p: Path) -> List[Path]:
    out = []
    stem_hdr = p.parent / f"{p.stem}.hdr"
    if stem_hdr != p:
        out.append(stem_hdr)
    out.extend(Path(str(p) + s) for s in _COMPANION_APPEND_SUFFIXES)
    return out


def execute_cleanup(paths: List[Path], dry_run: bool = False) -> int:
    """Delete files and their companion files (or preview).

    Directories (e.g. the ``crop_slc`` output dir) are removed recursively;
    symlinks are unlinked, never followed.

    Returns number of primary files handled.
    """
    import shutil

    n = 0
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        targets = [p] + _companions(p)
        for t in targets:
            if not t.exists():
                continue
            if dry_run:
                logger.info("[dry-run] would delete %s", t)
            else:
                try:
                    if t.is_dir() and not t.is_symlink():
                        shutil.rmtree(t)
                    else:
                        t.unlink()
                    logger.info("deleted %s", t)
                except OSError as e:
                    logger.warning("failed to delete %s: %s", t, e)
        n += 1
    return n
