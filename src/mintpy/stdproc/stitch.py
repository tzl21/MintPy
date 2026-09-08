#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Stitch per-burst interferograms and coherence maps into unified products.

Uses ``stitching_utils.stitch_arrays`` (pure-numpy pixel-offset copy,
gdal_merge-style) for correct coordinate handling and alignment.

Fixed output structure (see ``utils.naming``):
    ``output_dir/{date1}_{date2}/fullres.int[.tif]``
    ``output_dir/{date1}_{date2}/fullres.cpx.coh[.tif]``

When ``--out-bounds`` is NOT given, the output covers the *complete* union
extent of all bursts (full-track stitching).  When given, results are
clipped to the requested ``W S E N`` bbox (EPSG:4326).
"""

import argparse
import concurrent.futures
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from osgeo import gdal

from .utils.naming import extract_date_pair
from .utils.stitching_utils import stitch_arrays, _write_geotiff

gdal.UseExceptions()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stitch per-burst interferograms and coherence maps."
    )
    parser.add_argument('--processor', type=str, choices=['isce2', 'isce3'],
                        required=True, help="Processor type")
    parser.add_argument('--burst-dir', type=str, required=True, nargs='+',
                        help="Root directories containing per-burst results")
    parser.add_argument('--output-dir', type=str, required=True,
                        help="Output directory for stitched products")
    parser.add_argument('--file-types', type=str, nargs='+',
                        default=['.int.tif', '.cpx.coh.tif'],
                        help="File extensions to stitch")
    parser.add_argument('--out-bounds', type=float, nargs=4,
                        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'), default=None,
                        help="Final crop bounds in EPSG:4326. "
                             "If not set, the full union extent of all bursts is stitched.")
    parser.add_argument('--output-prefix', type=str, default='',
                        help="Prefix for output subdirectories")
    parser.add_argument('--max-workers', type=int, default=1,
                        help="Number of parallel stitch workers")
    parser.add_argument('--overwrite', action='store_true',
                        help="Overwrite existing output files")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose output")
    if args_list is None:
        return parser.parse_args()
    return parser.parse_args(args_list)


def discover_burst_dirs(root_dir: Path) -> List[Path]:
    burst_pattern = re.compile(r'^t\d+_\d+_iw\d+$')
    if not root_dir.exists():
        return []
    burst_dirs = []
    for entry in sorted(root_dir.iterdir()):
        if entry.is_dir() and burst_pattern.match(entry.name):
            burst_dirs.append(entry)
    if burst_dirs:
        logger.info(f"Found {len(burst_dirs)} burst directories in {root_dir}")
    return burst_dirs


def group_files_by_date_pair(
    burst_dirs: List[Path],
    file_types: List[str]
) -> Dict[str, Dict[str, List[Path]]]:
    """Group per-burst files by ``{date1}_{date2}`` (from the date-pair
    subdirectory name) and file type."""
    grouped: Dict[str, Dict[str, List[Path]]] = defaultdict(lambda: defaultdict(list))
    for burst_dir in burst_dirs:
        for ftype in file_types:
            for file_path in sorted(burst_dir.glob(f"**/*{ftype}")):
                dp = extract_date_pair(file_path.parent.name) or extract_date_pair(file_path.name)
                key = dp if dp else file_path.parent.name
                grouped[key][ftype].append(file_path)
    logger.info(f"Grouped files into {len(grouped)} date pairs across {len(burst_dirs)} bursts")
    return dict(grouped)


def get_file_epsg(file_path: Path) -> Optional[int]:
    ds = gdal.Open(str(file_path))
    if ds is None:
        return None
    wkt = ds.GetProjection()
    if not wkt:
        ds = None
        return None
    from osgeo import osr
    srs = osr.SpatialReference()
    try:
        srs.ImportFromWkt(wkt)
        code = srs.GetAuthorityCode(None)
    except Exception:
        code = None
    ds = None
    return int(code) if code else None


def stitch_date_pair(file_list: List[Path], output_path: Path,
                     out_bounds: Optional[Tuple[float, float, float, float]],
                     overwrite: bool, epsg_utm: int = 32605) -> bool:
    """Stitch one date pair from multiple burst files.

    ``out_bounds=None`` stitches the complete union extent of all bursts.
    """
    try:
        stitched, out_gt, proj = stitch_arrays(
            file_list, bbox_wsen=out_bounds, epsg_utm=epsg_utm
        )
        _write_geotiff(output_path, stitched, out_gt, proj, overwrite)
        logger.info(f"Stitched {len(file_list)} files -> {output_path}")
        return True
    except Exception as e:
        logger.error(f"Stitch failed for {output_path}: {e}")
        return False


def stitch_all(args: argparse.Namespace) -> int:
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    out_bounds = tuple(args.out_bounds) if args.out_bounds else None
    if out_bounds:
        logger.info(f"Output bounds (EPSG:4326): W={out_bounds[0]:.6f} S={out_bounds[1]:.6f} "
                     f"E={out_bounds[2]:.6f} N={out_bounds[3]:.6f}")
    else:
        logger.info("No --out-bounds given: stitching the COMPLETE union extent of all bursts.")

    total_success = 0
    total_failed = 0

    for root_dir_str in args.burst_dir:
        root_dir = Path(root_dir_str)
        if not root_dir.exists():
            logger.warning(f"Directory does not exist: {root_dir}")
            continue

        burst_dirs = discover_burst_dirs(root_dir)
        if not burst_dirs:
            logger.info(f"No burst directories in {root_dir}, treating as flat")
            burst_dirs = [root_dir]

        epsg_utm = 32605
        detected = False
        for bd in burst_dirs:
            for ft in args.file_types:
                candidates = list(bd.glob(f"**/*{ft}"))
                if candidates:
                    epsg = get_file_epsg(candidates[0])
                    if epsg:
                        epsg_utm = epsg
                        detected = True
                        break
            if detected:
                break

        grouped = group_files_by_date_pair(burst_dirs, args.file_types)
        if not grouped:
            logger.warning(f"No files found in {root_dir}")
            continue

        tasks = []
        for date_pair, type_files in sorted(grouped.items()):
            for ftype, flist in type_files.items():
                if not flist:
                    continue
                # Preserve fixed structure:
                # output_dir/{prefix}{date_pair}/{name} (prefix may be empty)
                out_dir = Path(args.output_dir)
                if args.output_prefix:
                    out_dir = out_dir / args.output_prefix
                out_path = out_dir / date_pair / Path(flist[0]).name
                tasks.append({
                    'file_list': flist,
                    'output_path': out_path,
                    'out_bounds': out_bounds,
                    'overwrite': args.overwrite,
                    'epsg_utm': epsg_utm,
                })

        max_workers = args.max_workers if args.max_workers else min(4, len(tasks))
        logger.info(f"Processing {len(tasks)} stitch tasks with {max_workers} workers")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max(max_workers, 1)) as executor:
            futures = {
                executor.submit(stitch_date_pair, **task): task
                for task in tasks
            }
            for future in concurrent.futures.as_completed(futures):
                ok = future.result()
                if ok:
                    total_success += 1
                else:
                    total_failed += 1

    logger.info(f"Stitching complete: {total_success} succeeded, {total_failed} failed")
    return 0 if total_failed == 0 else 1


def main(args: Optional[argparse.Namespace] = None) -> int:
    if args is None:
        args = parse_arguments()
    return stitch_all(args)


if __name__ == '__main__':
    sys.exit(main())
