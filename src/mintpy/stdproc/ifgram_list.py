#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Script for generating interferometric pairs and processing configuration.
Supports multiple pair generation modes.
"""

import argparse
import glob
import os
import re
import logging
from datetime import datetime, timedelta

# Configure logging ONLY when the host app has no handlers yet (importing
# this module must not clobber the caller's logging setup).
if not logging.root.handlers:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DEFAULT_NUM_CONNECTIONS = 5
DEFAULT_TIMEOUT = 3600
DEFAULT_MODE = 'sequential'

# Date pattern to match YYYYMMDD format
DATE_PATTERN = re.compile(r'(\d{4})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])')

# Date pair line: YYYYMMDD<sep>YYYYMMDD (sep: - _ , ; : or whitespace); any
# trailing columns / '#' comment are ignored (e.g. '20240107-20240119  100.5')
PAIR_LINE_PATTERN = re.compile(r'^\s*(\d{8})\s*[-_,;:\s]\s*(\d{8})\b')

# Pair generation modes
PAIR_GENERATORS = {}

#########################################################################
def register_pair_generator(mode_name):
    """
    Decorator to register pair generation functions.
    This allows easy addition of new pair generation modes.
    """
    def decorator(func):
        PAIR_GENERATORS[mode_name] = func
        return func
    return decorator


def _window_pairs(date_list, windows):
    """Pairs ``(d1, d2)`` (``d1 < d2``) whose temporal baseline falls within
    any ``(center, tol)`` window."""
    if not windows:
        return set()
    dts = {d: datetime.strptime(d, '%Y%m%d') for d in date_list}
    dl = sorted(date_list)
    pairs = set()
    for i in range(len(dl)):
        for j in range(i + 1, len(dl)):
            dt = (dts[dl[j]] - dts[dl[i]]).days
            if any(abs(dt - c) <= t for c, t in windows):
                pairs.add((dl[i], dl[j]))
    return pairs


def _resolve_window_pairs(date_list, oneyear_range=None, select_params=None):
    """Annual-window pairs from the unified ``annual_windows`` spec.

    ``select_params['annual_windows']`` is the canonical spec (``auto`` |
    ``none`` | ``center:tol,...``).  The legacy ``oneyear_range``
    (``slc2ifg.ifgram_list.oneyear_interferograms``) is mapped to an extra
    ``(365, range)`` window with a deprecation warning.
    """
    from .select_ifgrams import auto_annual_windows, parse_annual_windows
    windows = []
    aw = (select_params or {}).get('annual_windows')
    if aw is not None:
        v = str(aw).strip().lower()
        if v in ('auto', 'default', ''):
            windows.extend(auto_annual_windows(date_list))
        elif v not in ('none', 'off', '0', 'disable'):
            parsed = parse_annual_windows(aw)
            windows.extend(parsed if parsed is not None
                           else auto_annual_windows(date_list))
    if oneyear_range is not None:
        logger.warning(
            "slc2ifg.ifgram_list.oneyear_interferograms is deprecated; use "
            "slc2ifg.ifgram_list.select.annual_windows=365:<range> instead")
        windows.append((365, int(oneyear_range)))
    return _window_pairs(date_list, windows)


@register_pair_generator('sequential')
def generate_sequential_pairs(date_list, num_connections, oneyear_range=None,
                              select_params=None):
    """
    Generate interferometric pairs using sequential nearest neighbors.
    
    Parameters:
        date_list: List of dates in YYYYMMDD format
        num_connections: Number of nearest neighbor connections
        oneyear_range: Days range for one-year interferograms
        select_params: Not used in this mode, kept for API consistency
    
    Returns:
        Sorted list of (date1, date2) pairs
    """
    pairs = set()
    num_dates = len(date_list)
    
    # Generate nearest neighbor pairs
    max_connections = min(num_connections + 1, num_dates)
    for i in range(num_dates - 1):
        for j in range(i + 1, min(i + max_connections, num_dates)):
            pairs.add((date_list[i], date_list[j]))
    
    logger.info(f"Generated {len(pairs)} pairs from {num_connections} nearest neighbor connections")
    
    # Annual-window interferograms (unified annual_windows spec)
    win_pairs = _resolve_window_pairs(date_list, oneyear_range, select_params)
    if win_pairs:
        pairs.update(win_pairs)
        logger.info(f"Added {len(win_pairs)} annual-window pairs, total: {len(pairs)} pairs")
    
    return sorted(pairs)


@register_pair_generator('reference')
def generate_reference_pairs(date_list, num_connections=None, oneyear_range=None,
                             select_params=None):
    """
    Generate interferometric pairs with respect to a reference date.
    All SLCs are paired with the earliest SLC in the time series.
    
    Parameters:
        date_list: List of dates in YYYYMMDD format
        num_connections: Not used in this mode, kept for API consistency
        oneyear_range: Days range for one-year interferograms
        select_params: Not used in this mode, kept for API consistency
    
    Returns:
        Sorted list of (date1, date2) pairs
    """
    if len(date_list) < 2:
        return []
    
    # Find the earliest date (reference date)
    reference_date = min(date_list)
    logger.info(f"Using reference date: {reference_date}")
    
    # Generate pairs with all other dates
    pairs = set()
    for date in date_list:
        if date != reference_date:
            # Ensure consistent ordering (earlier date first)
            pair = (reference_date, date) if reference_date < date else (date, reference_date)
            pairs.add(pair)
    
    logger.info(f"Generated {len(pairs)} reference pairs")
    
    # Annual-window interferograms (unified annual_windows spec)
    win_pairs = _resolve_window_pairs(date_list, oneyear_range, select_params)
    if win_pairs:
        pairs.update(win_pairs)
        logger.info(f"Added {len(win_pairs)} annual-window pairs, total: {len(pairs)} pairs")
    
    return sorted(pairs)


def generate_oneyear_pairs(date_list, day_range):
    """
    Generate pairs that are approximately one year apart within specified day range.
    
    Parameters:
        date_list: List of dates in YYYYMMDD format
        day_range: Days range for one-year interferograms
    
    Returns:
        Set of (date1, date2) pairs
    """
    pairs = set()
    
    # Convert dates to datetime objects
    date_objs = [datetime.strptime(date, '%Y%m%d') for date in date_list]
    
    for i, date1 in enumerate(date_objs):
        target_date = date1 + timedelta(days=365)
        date_range_start = target_date - timedelta(days=day_range)
        date_range_end = target_date + timedelta(days=day_range)
        
        # Find dates within the range
        for j, date2 in enumerate(date_objs):
            if i != j and date_range_start <= date2 <= date_range_end:
                # Ensure consistent ordering (earlier date first)
                date_str1 = date_list[i]
                date_str2 = date_list[j]
                pair = (date_str1, date_str2) if date_str1 < date_str2 else (date_str2, date_str1)
                pairs.add(pair)
    
    return pairs


@register_pair_generator('select')
def generate_select_pairs(date_list, num_connections=None, oneyear_range=None,
                          select_params=None):
    """
    Coherence-aware connected interferogram selection.

    Generates a candidate network (k temporal nearest neighbours + pairs in
    user-specified temporal windows such as half-year / one-year + optional
    all pairs within a temporal-baseline cap), weights every candidate by
    coherence (measured on downsampled SLCs, read from existing coherence
    rasters, or predicted by a temporal-decorrelation model), then selects
    a *connected* subset that maximises coherence: a maximum-weight spanning
    tree plus greedy augmentation (per-date minimum degree, optional edge
    budget / quality floor / bridge repair).

    See ``mintpy.stdproc.select_ifgrams`` and ``docs/ifgram_selection.md``.

    Parameters:
        date_list: List of dates in YYYYMMDD format
        num_connections: k temporal nearest neighbours for the skeleton
        oneyear_range: Ignored (use select_params['annual_windows'])
        select_params: dict of selection parameters (see
            ``select_ifgrams.DEFAULT_PARAMS``); may carry 'slc_dir' and
            'processor' for on-the-fly quick coherence.

    Returns:
        Sorted list of (date1, date2) pairs
    """
    from .select_ifgrams import select_pairs

    params = dict(select_params or {})
    if num_connections is not None and 'num_connections' not in params:
        params['num_connections'] = num_connections
    # legacy oneyear_interferograms -> annual_windows (365, range)
    if oneyear_range is not None:
        logger.warning(
            "slc2ifg.ifgram_list.oneyear_interferograms is deprecated; use "
            "slc2ifg.ifgram_list.select.annual_windows=365:<range> instead")
        aw = str(params.get('annual_windows') or '').strip().lower()
        if aw in ('', 'auto', 'default'):
            params['annual_windows'] = f"365:{int(oneyear_range)}"
        else:
            params['annual_windows'] = f"{aw},365:{int(oneyear_range)}"
    slc_dir = params.pop('slc_dir', None)
    processor = params.pop('processor', 'isce3')
    pairs, report = select_pairs(date_list, slc_dir=slc_dir,
                                 processor=processor, params=params)
    logger.info(
        "select mode: %d dates -> %d candidate(s) -> %d selected pair(s) "
        "(connected=%s, rank=%s, min_degree=%s, weight_source=%s)",
        report['n_dates'], report['n_candidates'], report['n_selected'],
        report.get('connected'), report.get('rank'),
        report.get('min_degree_actual'), report.get('weight_source'))
    return pairs


@register_pair_generator('file')
def generate_file_pairs(date_list, num_connections=None, oneyear_range=None,
                        select_params=None):
    """
    Explicit interferogram pair list read from a file (mode='file').

    The file is used **verbatim** — ``num_connections``, ``annual_windows``
    and the date filters do not apply.  Every referenced date must exist in
    ``date_list`` (i.e. have an SLC): a pair without input data cannot be
    produced, so a missing date raises instead of being dropped silently.

    Parameters:
        date_list: List of dates in YYYYMMDD format (the SLC dates)
        num_connections: Ignored
        oneyear_range: Ignored
        select_params: dict carrying 'pair_file' (required)

    Returns:
        Sorted list of (date1, date2) pairs
    """
    params = dict(select_params or {})
    pair_file = params.get('pair_file')
    if not pair_file:
        raise ValueError(
            "ifgram_list mode='file' requires slc2ifg.ifgram_list.pair_file "
            "(a text file with one 'YYYYMMDD-YYYYMMDD' pair per line)")
    pairs = read_pair_list(pair_file)
    check_pair_dates(pairs, date_list, source=str(pair_file))
    logger.info("file mode: %d pair(s) used verbatim from %s",
                len(pairs), pair_file)
    return pairs


#########################################################################
def _str2bool(value):
    """argparse type: parse 'true'/'false'/'1'/'0'/'yes'/'no' into a bool."""
    if isinstance(value, bool):
        return value
    v = str(value).strip().lower()
    if v in ('true', 'yes', '1', 'on'):
        return True
    if v in ('false', 'no', '0', 'off'):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: '{value}'")


def extract_date_from_string(text):
    """
    Extract date in YYYYMMDD format from a string.
    
    Parameters:
        text (str): Input string that may contain a date
        
    Returns:
        str or None: Extracted date in YYYYMMDD format, or None if no valid date found
    """
    match = DATE_PATTERN.search(text)
    if match:
        date_str = match.group(0)
        # Validate that this is a real date
        try:
            datetime.strptime(date_str, '%Y%m%d')
            return date_str
        except ValueError:
            return None
    return None


def get_date_list(slc_dir):
    """
    Get sorted list of dates from SLC directory, handling various path formats.
    
    Supports multiple formats:
    - Directories named with dates: /path/20180917/
    - Files containing dates: /path/20180917.slc, /path/S1_20180917.tif
    - Nested structures: /path/subdir/20180917/, /path/subdir/20180917.slc
    """
    date_set = set()
    
    # Check if the SLC directory exists
    if not os.path.exists(slc_dir):
        raise ValueError(f"SLC directory does not exist: {slc_dir}")
    
    # Search for all files and directories recursively
    _SLC_EXTS = ('.slc', '.slc.tif', '.tif', '.tiff', '.h5', '.hdf5',
                 '.slc.full', '.rdr', '.int', '.int.tif')
    for root, dirs, files in os.walk(slc_dir):
        # Directory names must be EXACTLY an 8-digit date — a pair-named
        # directory like 20200101_20200113 must not inject spurious dates.
        for dir_name in dirs:
            if DATE_PATTERN.fullmatch(dir_name):
                date = extract_date_from_string(dir_name)
                if date:
                    date_set.add(date)

        # File names: only accept SLC-like extensions (the primary path used
        # to pick dates out of ANY file, e.g. stray pair-named products).
        for file_name in files:
            if not file_name.lower().endswith(_SLC_EXTS):
                continue
            date = extract_date_from_string(file_name)
            if date:
                date_set.add(date)
    
    # If no dates found with pattern matching, try alternative approaches
    if not date_set:
        date_set = fallback_date_extraction(slc_dir)
    
    if not date_set:
        raise ValueError(f"No valid dates found in SLC directory: {slc_dir}")
    
    date_list = sorted(list(date_set))
    # The full date list can be hundreds of entries long (and is repeated per
    # burst in multi-burst mode); keep it out of the default log.
    logger.debug("Found %d unique dates: %s", len(date_list), date_list)
    return date_list


def parse_exclude_dates(exclude_date):
    """Normalize ``exclude_date`` into a sorted list of ``YYYYMMDD`` strings.

    Accepts ``None`` / ``'auto'`` / ``'none'`` / ``''`` (no exclusion),
    a comma- and/or whitespace-separated string of dates (e.g.
    ``'20200101, 20200615'``), or a list of date strings (programmatic
    API). Invalid entries are dropped with a warning.
    """
    if exclude_date is None:
        return []
    if isinstance(exclude_date, str):
        v = exclude_date.strip()
        if v.lower() in ('', 'auto', 'none', 'off', 'no', '0', 'false'):
            return []
        raw = re.split(r'[,\s]+', v)
    elif isinstance(exclude_date, (int, float)):
        raw = [str(exclude_date)]
    else:
        raw = list(exclude_date)
    out = []
    for d in raw:
        d = str(d).strip()
        if not d:
            continue
        try:
            dt = datetime.strptime(d, '%Y%m%d')
        except ValueError:
            logger.warning("Invalid exclude date, ignored: %r", d)
            continue
        out.append(dt.strftime('%Y%m%d'))
    return sorted(set(out))


def filter_date_list(date_list, start_date=None, end_date=None, exclude_date=None):
    """Keep dates within ``[start_date, end_date]`` (YYYYMMDD, inclusive)
    and drop the dates listed in ``exclude_date``.

    Dates are compared as strings (zero-padded YYYYMMDD sorts lexically).
    ``exclude_date`` accepts ``None`` / ``'auto'`` / comma-or-space
    separated ``YYYYMMDD`` dates / a list of date strings; e.g.
    ``'20200101,20200615'`` removes bad acquisitions from the network
    *before* pairing so no interferogram involves them.
    Returns a new sorted list.
    """
    ex_dates = parse_exclude_dates(exclude_date)
    if start_date is None and end_date is None and not ex_dates:
        return list(date_list)
    out = []
    for d in sorted(date_list):
        if start_date and d < str(start_date):
            continue
        if end_date and d > str(end_date):
            continue
        if d in ex_dates:
            continue
        out.append(d)
    logger.info("Date filter [%s, %s] excl [%s]: %d date(s) kept",
                start_date or '-inf', end_date or '+inf',
                ','.join(ex_dates) or 'none', len(out))
    missing = sorted(set(ex_dates) - set(date_list))
    if missing:
        logger.warning("Exclude date(s) not found in the SLC date list "
                       "(ignored): %s", ','.join(missing))
    return out


def fallback_date_extraction(slc_dir):
    """
    Fallback method to extract dates when pattern matching fails.
    
    This function tries alternative strategies to find dates:
    1. Look for common SLC file extensions
    2. Check if directory names look like dates
    3. Use glob patterns as last resort
    """
    date_set = set()
    
    # Common SLC file extensions
    slc_extensions = ['.slc', '.SLC', '.tif', '.TIF', '.tiff', '.TIFF']
    
    # Walk through directory structure
    for root, dirs, files in os.walk(slc_dir):
        # Check files with SLC extensions
        for file_name in files:
            # If file has SLC-like extension, try to extract date from filename
            if any(file_name.endswith(ext) for ext in slc_extensions):
                # Remove extension and try to extract date
                base_name = os.path.splitext(file_name)[0]
                date = extract_date_from_string(base_name)
                if date:
                    date_set.add(date)
        
        # For directories, check if they look like dates (8 digits)
        for dir_name in dirs:
            if dir_name.isdigit() and len(dir_name) == 8:
                try:
                    datetime.strptime(dir_name, '%Y%m%d')
                    date_set.add(dir_name)
                except ValueError:
                    continue
    
    # Last resort: use glob pattern
    if not date_set:
        try:
            # Look for any 8-digit sequences in the path
            all_items = glob.glob(os.path.join(slc_dir, "**"), recursive=True)
            for item in all_items:
                date = extract_date_from_string(item)
                if date:
                    date_set.add(date)
        except Exception:
            pass
    
    return date_set


def generate_pairs(date_list, mode, num_connections, oneyear_range=None,
                   select_params=None):
    """
    Generate interferometric pairs using specified mode.
    
    Parameters:
        date_list: List of dates in YYYYMMDD format
        mode: Pair generation mode
        num_connections: Number of nearest neighbor connections
        oneyear_range: Days range for one-year interferograms
        select_params: Optional dict of parameters for the 'select' mode
    
    Returns:
        Sorted list of (date1, date2) pairs
    """
    if mode not in PAIR_GENERATORS:
        raise ValueError(f"Unknown mode: {mode}. Available modes: {list(PAIR_GENERATORS.keys())}")
    
    logger.info(f"Generating pairs using '{mode}' mode")
    return PAIR_GENERATORS[mode](date_list, num_connections, oneyear_range,
                                 select_params)


def write_pair_list(pairs, output_file):
    """Write pairs to output file (atomically: temp file + rename)."""
    out_path = os.path.abspath(output_file)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    tmp = f"{out_path}.tmp"
    with open(tmp, 'w') as f:
        f.write("# Interferometric pairs generated by ifgram_list.py\n")
        f.write("# Date12\n")
        for date1, date2 in pairs:
            f.write(f"    {date1}-{date2}\n")
    os.replace(tmp, out_path)
    logger.info(f"Wrote {len(pairs)} pairs to {output_file}")
    return output_file


def read_pair_list(pair_file):
    """Read an explicit interferogram pair list (one pair per line).

    Accepted per line (a trailing ``# comment`` and any extra columns are
    ignored): ``YYYYMMDD-YYYYMMDD``, ``YYYYMMDD_YYYYMMDD`` or
    ``YYYYMMDD YYYYMMDD``.  Blank lines and ``#`` comments are skipped;
    duplicates are removed, the dates of every pair are ordered and the
    result is sorted.  Raises ValueError on an unparsable line or an empty
    file.
    """
    if not os.path.isfile(pair_file):
        raise ValueError(f"pair file not found: {pair_file}")
    pairs, seen = [], set()
    with open(pair_file, 'r', encoding='utf-8-sig') as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.split('#', 1)[0].strip()
            if not line:
                continue
            m = PAIR_LINE_PATTERN.match(line)
            if m is None:
                raise ValueError(
                    f"{pair_file}:{lineno}: cannot parse {raw.strip()!r} — "
                    f"expected 'YYYYMMDD-YYYYMMDD' (one pair per line)")
            d1, d2 = m.group(1), m.group(2)
            if d1 == d2:
                raise ValueError(
                    f"{pair_file}:{lineno}: identical dates {d1}")
            key = (d1, d2) if d1 < d2 else (d2, d1)
            if key not in seen:
                seen.add(key)
                pairs.append(key)
    if not pairs:
        raise ValueError(f"no interferogram pairs found in {pair_file}")
    return sorted(pairs)


def check_pair_dates(pairs, date_list, source='pair list'):
    """Raise ValueError when a pair references a date absent from date_list."""
    known = set(date_list or [])
    missing = sorted({d for a, b in pairs for d in (a, b)} - known)
    if missing:
        shown = ', '.join(missing[:8]) + (' ...' if len(missing) > 8 else '')
        raise ValueError(
            f"{source}: {len(missing)} date(s) have no SLC in the input "
            f"directory ({shown}) — trim the pair file or add the missing "
            f"SLCs")


def _select_params_from_args(args):
    """Build the select-mode parameter dict from CLI arguments."""
    params = {}
    mapping = [
        ('select_annual_windows', 'annual_windows'),
        ('select_temp_baseline_max', 'temp_baseline_max'),
        ('select_perp_baseline_max', 'perp_baseline_max'),
        ('select_perp_baseline_file', 'perp_baseline_file'),
        ('select_quick_nlks', 'quick_nlks'),
        ('select_quick_max_workers', 'quick_max_workers'),
        ('select_min_degree', 'min_degree'),
        ('select_max_pairs', 'max_pairs'),
        ('select_quick_debias', 'quick_debias'),
        ('select_robust', 'robust'),
    ]
    for cli_name, param_name in mapping:
        val = getattr(args, cli_name, None)
        if val is not None:
            params[param_name] = val
    # AOI: restrict the quick coherence to the bbox+buffer window
    if getattr(args, 'bbox', None) is not None:
        params['bbox'] = tuple(float(v) for v in args.bbox)
        params['bbox_buffer'] = float(getattr(args, 'bbox_buffer', None) or 0.0)
    if args.select_report:
        params['report_file'] = os.path.join(args.work_dir, args.select_report) \
            if not os.path.isabs(args.select_report) else args.select_report
    if args.select_dot:
        params['dot_file'] = os.path.join(args.work_dir, args.select_dot) \
            if not os.path.isabs(args.select_dot) else args.select_dot
    return params


