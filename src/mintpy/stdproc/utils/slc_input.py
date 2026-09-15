#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""Resolve ``slc2ifg.slc_input`` (a path or glob) into SLC files.

``slc2ifg.slc_input`` may be a plain directory, a directory glob, or a
multi-burst glob.  The SLC filename pattern is **inferred** from the files that
are actually present, so no ``slc2ifg.slc_pattern`` key is needed:

    flat        -> ``slc_input = xxx/y*/``
                   ``xxx/y1/yyyymmdd.slc.tif``
    multi-burst -> ``slc_input = xxx/t*/*/``
                   ``xxx/t001_.../yyyymmdd/xxx.h5``

In the multi-burst layout the burst id is the parent folder of the date folder
and is extracted automatically (any ancestor matching ``t\\d+_\\d+_iw\\d+``).
"""

from __future__ import annotations

import fnmatch
import glob
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

#: burst id pattern used by the per-burst layout, e.g. t124_264305_iw2
BURST_RE = re.compile(r'^t\d+_\d+_iw\d+$')

#: 8-digit acquisition date
DATE_RE = re.compile(r'(20\d{6})')

#: SLC filename candidates, in preference order.  The first candidate that
#: matches any file in a directory determines the inferred pattern.
SLC_PATTERNS: Tuple[str, ...] = (
    '*.slc.tif',
    '*.slc.h5',
    '*.slc.full',      # ISCE2 topsApp merged SLC (ENVI + .hdr/.xml)
    '*.slc',
    '*.h5',
    '*.hdf5',
    '*.tif',
    '*.tiff',
    '*.rdr',
)

#: how deep below each glob match to look for SLC files
_MAX_DEPTH = 5


def is_burst_id(name: str) -> bool:
    """Return True if ``name`` looks like an ISCE3 burst id."""
    return bool(BURST_RE.match(str(name)))


def extract_date(name: str) -> Optional[str]:
    """Extract a ``YYYYMMDD`` date from a file name or directory name."""
    m = DATE_RE.search(str(name))
    return m.group(1) if m else None


def infer_slc_pattern(filenames, processor: str = 'isce3') -> str:
    """Infer the SLC glob pattern from a list of file names.

    Falls back to the processor default (``*.slc`` for isce2, ``*.slc.*``
    otherwise) when nothing matches a known candidate.
    """
    names = list(filenames)
    for pat in SLC_PATTERNS:
        if any(fnmatch.fnmatch(n, pat) for n in names):
            return pat
    return '*.slc' if processor == 'isce2' else '*.slc.*'


def _is_slc_name(name: str) -> bool:
    if 'static' in name.lower():
        return False
    return any(fnmatch.fnmatch(name, pat) for pat in SLC_PATTERNS)


@dataclass
class SlcInput:
    """Resolved ``slc2ifg.slc_input``.

    Attributes
    ----------
    root : str
        The original (config-level) input string.
    pattern : str
        Inferred SLC filename glob, e.g. ``*.slc.tif`` or ``*.h5``.
    processor : str
        ``isce3`` or ``isce2``.
    scan_dirs : dict
        ``{burst_id or None: [directories]}`` — directories to scan for dates
        and SLC files.  ``None`` is the single-burst (flat) case.
    files : dict
        ``{(burst_id or None, date): Path}`` — resolved per-date SLC file.
    """

    root: str
    pattern: str
    processor: str
    scan_dirs: Dict[Optional[str], List[Path]] = field(default_factory=dict)
    files: Dict[Tuple[Optional[str], str], Path] = field(default_factory=dict)

    @property
    def bursts(self) -> List[Optional[str]]:
        """Burst ids in order; ``[None]`` for single-burst input."""
        names = sorted(b for b in self.scan_dirs if b is not None)
        return names if names else [None]

    @property
    def is_multi_burst(self) -> bool:
        return any(b is not None for b in self.scan_dirs)

    def dirs(self, burst: Optional[str] = None) -> List[Path]:
        """Directories to scan for ``burst`` (or the flat dirs)."""
        if burst in self.scan_dirs:
            return self.scan_dirs[burst]
        if burst is None:
            return self.scan_dirs.get(None, [])
        return []

    def date_list(self, burst: Optional[str] = None) -> List[str]:
        """Sorted dates that have an SLC for ``burst`` (all dates if None)."""
        if burst is None and None not in self.scan_dirs:
            return sorted({d for (_b, d) in self.files})
        return sorted({d for (b, d) in self.files if b == burst})

    def file_for(self, burst: Optional[str], date: str) -> Optional[Path]:
        p = self.files.get((burst, date))
        if p is None and burst is None:
            # tolerate a burst-qualified lookup falling back to flat
            p = self.files.get((None, date))
        return p

    def input_dirs(self) -> List[Path]:
        """All directories that hold SLC files (deduplicated, ordered)."""
        out: List[Path] = []
        for dirs in self.scan_dirs.values():
            for d in dirs:
                if d not in out:
                    out.append(d)
        return out


def _expand(pattern: str) -> List[Path]:
    """Glob-expand ``pattern``; a plain existing directory matches itself."""
    matches = sorted(glob.glob(pattern, recursive=True))
    if not matches and os.path.isdir(pattern):
        matches = [pattern]
    return [Path(m) for m in matches]


def _walk_slc_files(root: Path) -> List[Path]:
    """Collect SLC-like files at most :data:`_MAX_DEPTH` levels below ``root``."""
    out: List[Path] = []
    root = Path(root)
    if root.is_file():
        return [root] if _is_slc_name(root.name) else []
    if not root.is_dir():
        return out
    base_depth = len(root.parts)
    for cur, dirs, files in os.walk(root):
        depth = len(Path(cur).parts) - base_depth
        if depth >= _MAX_DEPTH:
            dirs[:] = []
        for name in sorted(files):
            if _is_slc_name(name):
                out.append(Path(cur) / name)
    return sorted(out)


def _burst_of(path: Path, max_up: int = 4) -> Optional[str]:
    """Nearest ancestor of ``path`` matching the burst-id pattern."""
    for i, parent in enumerate(path.resolve().parents):
        if i >= max_up:
            break
        if is_burst_id(parent.name):
            return parent.name
    return None


def _date_of(path: Path, max_up: int = 4) -> Optional[str]:
    """Date for ``path``: from its name first, then from ancestor dir names."""
    d = extract_date(path.stem) or extract_date(path.name)
    if d:
        return d
    for i, parent in enumerate(path.resolve().parents):
        if i >= max_up:
            break
        if len(parent.name) == 8 and extract_date(parent.name):
            return parent.name
    return None


#: standard ISCE2 geometry directory name (topsApp merged tree)
GEOM_DIR_NAME = 'geom_reference'


def standard_geom_dir(slc_input) -> Optional[Path]:
    """Standard ISCE2 geometry dir derived from ``slc2ifg.slc_input``.

    The ISCE2 topsApp tree keeps the SLCs in ``<merged>/SLC`` and the geometry
    in ``<merged>/geom_reference`` (``lat.rdr.full`` / ``lon.rdr.full`` / ...).
    ``slc_input`` is typically ``<merged>/SLC/*``; the non-glob prefix and its
    ancestors are searched for a ``geom_reference`` directory, so the location
    is fixed by the standard layout and needs no configuration key.

    Returns ``None`` when no ``geom_reference`` directory is found.
    """
    if isinstance(slc_input, (list, tuple)):
        slc_input = slc_input[0] if slc_input else ''
    raw = str(slc_input)
    cut = len(raw)
    for i, ch in enumerate(raw):
        if ch in '*?[':
            cut = i
            break
    prefix = raw[:cut].rstrip('/') or '.'
    p = Path(prefix).resolve()
    for base in [p] + list(p.parents):
        cand = base / GEOM_DIR_NAME
        if cand.is_dir():
            return cand
    return None


def resolve_slc_input(slc_input, config_dir: str = '.',
                      processor: str = 'isce3') -> SlcInput:
    """Expand ``slc2ifg.slc_input`` into a :class:`SlcInput`.

    Parameters
    ----------
    slc_input : str or Path
        A directory, a directory glob, or a multi-burst glob.
    config_dir : str
        Base directory for a relative ``slc_input`` (the config file's dir).
    processor : str
        ``isce3`` or ``isce2`` (only used for the fallback pattern).

    Returns
    -------
    SlcInput

    Raises
    ------
    ValueError
        When the pattern matches nothing or no SLC file can be found.
    """
    raw = str(slc_input)
    if not Path(raw).is_absolute():
        raw = str(Path(config_dir) / raw)

    matches = _expand(raw)
    if not matches:
        raise ValueError(
            f"slc2ifg.slc_input={slc_input!r} matches no file or directory")

    roots: List[Path] = []
    for m in matches:
        p = m if m.is_dir() else m.parent
        if p not in roots:
            roots.append(p)
    roots = sorted(roots)

    all_files = sorted({f for r in roots for f in _walk_slc_files(r)})
    if not all_files:
        raise ValueError(
            f"slc2ifg.slc_input={slc_input!r} contains no SLC file "
            f"(looked for {', '.join(SLC_PATTERNS)})")

    pattern = infer_slc_pattern([f.name for f in all_files], processor)

    file_map: Dict[Tuple[Optional[str], str], Path] = {}
    dir_map: Dict[Optional[str], List[Path]] = {}
    for f in all_files:
        burst = _burst_of(f)
        date = _date_of(f)
        if date is None:
            logger.warning("cannot extract a date from %s — skipped", f)
            continue
        key = (burst, date)
        if key in file_map and file_map[key] != f:
            logger.warning(
                "multiple SLC candidates for burst=%s date=%s: %s — using %s",
                burst, date, [file_map[key].name, f.name], file_map[key].name)
            continue
        file_map[key] = f
        dir_map.setdefault(burst, [])
        d = f.parent
        if d not in dir_map[burst]:
            dir_map[burst].append(d)

    for burst in dir_map:
        dir_map[burst] = sorted(dir_map[burst])

    if dir_map:
        n_bursts = len([b for b in dir_map if b is not None])
        logger.info(
            "slc_input: %d file(s), pattern=%s, %s",
            len(file_map), pattern,
            f"{n_bursts} burst(s)" if n_bursts else "single burst")

    return SlcInput(root=str(slc_input), pattern=pattern, processor=processor,
                    scan_dirs=dir_map, files=file_map)
