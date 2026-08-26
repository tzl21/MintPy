#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Script for generating interferograms from a list of date pairs.
It creates VRT interferograms and, by default, materialises them into
actual raster files (GeoTIFF for ISCE3, ENVI for ISCE2).

Fixed output structure (see ``utils.naming``):
    ``output_dir/{date1}_{date2}/fullres.int``      (isce2)
    ``output_dir/{date1}_{date2}/fullres.int.tif``  (isce3)

SLC inputs follow the fixed naming ``yyyymmdd.slc`` (isce2) /
``yyyymmdd.slc.tif`` / ``yyyymmdd.slc.h5`` (isce3).

Examples:
    # ISCE3: create VRTs and GeoTIFFs
    generate_ifgram.py --processor isce3 --pairs-file ifgram_list.txt \
        --slc-dir /data/slc --output-dir /data/ifgs

    # ISCE2: create VRTs and ENVI int files, skip existing
    generate_ifgram.py --processor isce2 --pairs-file pairs.txt \
        --slc-dir /data/slc --output-dir /data/ifgs \
        --slc-pattern *.slc --ifg-pattern *.int
"""

import argparse
import glob
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from dolphin.interferogram import VRTInterferogram
from osgeo import gdal

from .utils.naming import (
    ifg_path,
    int_ext,
    slc_pattern,
)
from .utils.slc2ifg_utils import create_xml_for_binary, is_hdf5_file

gdal.UseExceptions()

# ------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

#: Rows processed at a time when materialising a VRT (bounds peak memory)
_MATERIALISE_BLOCK_ROWS = 2048


def expand_directories(directory_patterns):
    """Expand directory patterns into a list of Path objects."""
    expanded = []
    for pattern in directory_patterns:
        matches = glob.glob(pattern, recursive=False)
        if not matches:
            logger.warning(f"No directories found matching: {pattern}")
        expanded.extend(Path(m) for m in matches)
    return expanded


def find_slc_file_by_date(slc_dirs, target_date, slc_pattern, processor):
    """
    Find SLC file that contains the target date in its filename.

    Also warns if the file extension doesn't fit the processor.
    """
    expected_exts = ['.slc', '.rdr', '.full'] if processor == 'isce2' else ['.tif', '.tiff', '.h5', '.hdf5']
    for slc_dir in slc_dirs:
        pattern = f"*{target_date}{slc_pattern}"
        matching_files = sorted(
            list(slc_dir.glob(pattern)) + list(slc_dir.glob(f"*/{pattern}")))
        if matching_files:
            if len(matching_files) > 1:
                logger.warning(
                    "Multiple SLC candidates for date %s: %s — using %s",
                    target_date,
                    [f.name for f in matching_files],
                    matching_files[0].name)
            found = matching_files[0]
            ext = found.suffix.lower()
            if ext not in expected_exts:
                logger.warning(
                    "Processor '%s' expects extensions %s, but found '%s' (%s).",
                    processor, expected_exts, ext, found.name
                )
            return found
    return None


def create_interferogram_from_vrt(vrt_path: Path, output_path: Path, processor: str) -> bool:
    """Materialise the VRT to a CFloat32 raster, atomically.

    The product is written to ``<output>.tmp`` and renamed into place only
    after a successful close, so an interrupted run never leaves a partial
    file at the final path (the ENVI ``.hdr`` sidecar is renamed along).
    """
    tmp_path = Path(str(output_path) + '.tmp')
    try:
        ref_ds = gdal.Open(str(vrt_path), gdal.GA_ReadOnly)
        if ref_ds is None:
            logger.error("Cannot open VRT: %s", vrt_path)
            return False
        geotransform = ref_ds.GetGeoTransform()
        projection = ref_ds.GetProjection()
        rows, cols = ref_ds.RasterYSize, ref_ds.RasterXSize
        ref_band = ref_ds.GetRasterBand(1)

        driver_name = 'GTiff' if processor == 'isce3' else 'ENVI'
        driver = gdal.GetDriverByName(driver_name)
        if driver is None:
            logger.error("GDAL driver %s not available.", driver_name)
            return False

        options = ['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER'] \
            if processor == 'isce3' else []

        out_ds = driver.Create(
            str(tmp_path), cols, rows, 1, gdal.GDT_CFloat32, options=options
        )
        if out_ds is None:
            logger.error("Cannot create output: %s", output_path)
            return False
        out_ds.SetGeoTransform(geotransform)
        out_ds.SetProjection(projection)
        out_band = out_ds.GetRasterBand(1)

        # Block-wise copy: read a chunk of the VRT and write it out
        for r0 in range(0, rows, _MATERIALISE_BLOCK_ROWS):
            r1 = min(r0 + _MATERIALISE_BLOCK_ROWS, rows)
            block = ref_band.ReadAsArray(0, r0, cols, r1 - r0)
            if block is None:
                raise RuntimeError("VRT block read failed")
            out_band.WriteArray(block, 0, r0)
        out_band = None
        out_ds = None
        ref_band = None
        ref_ds = None

        # atomic rename: data file + ENVI .hdr sidecar
        tmp_path.replace(output_path)
        hdr_tmp = Path(str(tmp_path) + '.hdr')
        if hdr_tmp.exists():
            hdr_tmp.replace(Path(str(output_path) + '.hdr'))

        if processor == 'isce2':
            create_xml_for_binary(output_path, family='intimage',
                                  description='Complex interferogram')
        return True

    except Exception as e:
        logger.error("Error materialising VRT %s: %s", vrt_path.name, e)
        for stray in (tmp_path, Path(str(tmp_path) + '.hdr')):
            try:
                if stray.exists():
                    stray.unlink()
            except OSError:
                pass
        return False


def _vrt_is_valid(vrt_path: Path) -> bool:
    """Cheap well-formedness check for an existing VRT (XML)."""
    if not vrt_path.exists() or vrt_path.stat().st_size == 0:
        return False
    try:
        import xml.etree.ElementTree as ET
        ET.parse(vrt_path)
        return True
    except Exception:
        return False


def process_single_pair(pair_info):
    """
    Worker function that creates a VRT (if needed) and, if not only_vrt,
    materialises it into a true interferogram file.

    pair_info is a tuple containing:
        (date_pair_str, date1, date2, slc1_path, slc2_path, vrt_path, ifg_path,
         verify_slc, subdataset, processor, only_vrt)
    """
    (date12, date1, date2, slc1_path, slc2_path, vrt_path, ifg_path,
     verify_slc, subdataset, processor, only_vrt) = pair_info

    # --- 1. Check SLCs ---
    if slc1_path is None or not slc1_path.exists():
        return (date12, False, "First SLC file not found")
    if slc2_path is None or not slc2_path.exists():
        return (date12, False, "Second SLC file not found")

    # --- 2. VRT creation ---
    if vrt_path.exists() and _vrt_is_valid(vrt_path):
        logger.debug("VRT already exists: %s", vrt_path.name)
    else:
        if vrt_path.exists():
            logger.warning("Existing VRT appears corrupt, rebuilding: %s",
                           vrt_path.name)
            try:
                vrt_path.unlink()
            except OSError:
                pass
        try:
            vrt_args = {
                "ref_slc": str(slc1_path),
                "sec_slc": str(slc2_path),
                "path": str(vrt_path),
                "outdir": str(vrt_path.parent),
                "verify_slcs": verify_slc,
                "write": True,
            }
            if subdataset and (is_hdf5_file(slc1_path) or is_hdf5_file(slc2_path)):
                vrt_args["subdataset"] = subdataset

            VRTInterferogram(**vrt_args)
            logger.debug("Created VRT: %s", vrt_path.name)
        except Exception as e:
            return (date12, False, f"VRT creation failed: {e}")

    # --- 3. Materialisation (if not only_vrt) ---
    if not only_vrt:
        if ifg_path.exists() and ifg_path.stat().st_size > 0:
            logger.debug("Interferogram already exists: %s", ifg_path.name)
            return (date12, True, f"Interferogram already exists: {ifg_path.name}")
        if ifg_path.exists():
            logger.warning("Existing interferogram is empty/partial, "
                           "re-materialising: %s", ifg_path.name)
            try:
                ifg_path.unlink()
            except OSError:
                pass
        success = create_interferogram_from_vrt(vrt_path, ifg_path, processor)
        if success:
            return (date12, True, f"Created interferogram: {ifg_path.name}")
        else:
            return (date12, False, "Failed to materialise interferogram from VRT")
    else:
        return (date12, True, f"VRT created: {vrt_path.name}")


# ------------------------------------------------------------------------
def _detect_burst_dirs(slc_dirs):
    """Detect burst subdirectories under the given SLC directory list.

    If slc_dirs contains a single parent directory whose subdirectories
    match the burst ID pattern (e.g. t124_264305_iw2), return a dict of
    {burst_id: burst_dir_path}. Otherwise return {None: slc_dirs} (flat mode).
    """
    import re
    burst_pattern = re.compile(r'^t\d+_\d+_iw\d+$')
    burst_map = {}
    for slc_dir in slc_dirs:
        if slc_dir.is_dir():
            for entry in sorted(slc_dir.iterdir()):
                if entry.is_dir() and burst_pattern.match(entry.name):
                    burst_map[entry.name] = entry
    if burst_map:
        logger.info("Detected %d burst(s): %s", len(burst_map), list(burst_map.keys()))
        return burst_map
    # If all slc_dirs are already burst directories, use flat output mode
    # (the engine handles burst separation via per-burst output dirs)
    burst_dirs = [d for d in slc_dirs if burst_pattern.match(d.name)]
    if burst_dirs and len(burst_dirs) == len(slc_dirs):
        logger.info("Input directories are burst-specific — flat output mode")
        return {None: slc_dirs}
    return {None: slc_dirs}


# Main generation routine
# ------------------------------------------------------------------------
def generate_ifgram(pairs_file, slc_dir_patterns, output_dir, processor,
                    slc_pattern, subdataset, no_verify, only_vrt,
                    vrt_pattern, ifg_pattern, max_workers, verbose):
    """
    Full VRT + (optional) materialisation pipeline.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    slc_dirs = expand_directories(slc_dir_patterns)
    if not slc_dirs:
        logger.error("No SLC directories found.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read pairs
    try:
        pairs_df = pd.read_csv(pairs_file, comment='#', sep=r'\s+', names=['date12'])
        logger.info("Found %d interferometric pairs.", len(pairs_df))
    except Exception as e:
        logger.error("Error reading pairs file %s: %s", pairs_file, e)
        return

    # Detect burst subdirectories for per-burst processing
    burst_map = _detect_burst_dirs(slc_dirs)

    # Prepare list of tasks — per-burst if multiple bursts detected
    tasks = []
    for burst_id, burst_dirs in burst_map.items():
        # Resolve SLC dirs: for named burst use a single dir, for flat use all
        burst_slc_dirs = [burst_dirs] if burst_id else burst_dirs
        bust_out = output_dir / burst_id if burst_id else output_dir

        seen: dict = {}
        for _, row in pairs_df.iterrows():
            parts = str(row['date12']).split('-')
            if len(parts) != 2 or not (len(parts[0]) == 8 and len(parts[1]) == 8):
                raise ValueError(
                    f"Malformed date12 in pairs file: {row['date12']!r} "
                    "(expected YYYYMMDD-YYYYMMDD)")
            date1, date2 = parts

            slc1 = find_slc_file_by_date(burst_slc_dirs, date1, slc_pattern, processor)
            slc2 = find_slc_file_by_date(burst_slc_dirs, date2, slc_pattern, processor)

            # Fixed structure: output_dir/{date1}_{date2}/fullres.int[.tif]
            pair_out = bust_out / f"{date1}_{date2}"
            pair_out.mkdir(parents=True, exist_ok=True)
            vrt_path = pair_out / "fullres.int.vrt"
            ifg_file = ifg_path(bust_out, date1, date2, variant='fullres', processor=processor)

            # dedupe duplicate pair rows (avoids concurrent writes to the
            # same VRT/ifg path from two workers)
            dedup_key = (str(vrt_path), str(ifg_file))
            if dedup_key in seen:
                logger.warning("Duplicate pair row %s ignored", row['date12'])
                continue
            seen[dedup_key] = True

            tasks.append((
                row['date12'], date1, date2, slc1, slc2, vrt_path, ifg_file,
                not no_verify, subdataset, processor, only_vrt
            ))

    if not tasks:
        logger.error("No tasks to process.")
        return False

    if max_workers is None:
        max_workers = min((os.cpu_count() or 1), len(tasks))
    logger.info("Processing %d pairs with %d parallel workers (threads).", len(tasks), max_workers)

    successful = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(process_single_pair, t): t[0] for t in tasks}

        for future in future_to_pair:
            date12 = future_to_pair[future]
            try:
                date12, success, msg = future.result()
                if success:
                    logger.info("SUCCESS %s: %s", date12, msg)
                    successful += 1
                else:
                    logger.error("FAILED %s: %s", date12, msg)
                    errors += 1
            except Exception as e:
                logger.error("FAILED %s: unexpected error: %s", date12, e)
                errors += 1

    logger.info("Processing complete: %d successful, %d failed.", successful, errors)
    logger.info("Output directory: %s", output_dir)
    return errors == 0 and successful > 0


# ------------------------------------------------------------------------
# Command‑line interface
# ------------------------------------------------------------------------
def parse_arguments(args_list=None):
    """Parse command line arguments (compatible with engine invocation)."""
    parser = argparse.ArgumentParser(
        description="Generate VRT interferograms and (optionally) materialise them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ISCE3: VRT + GeoTIFF output
  %(prog)s --processor isce3 --pairs-file ifgram_list.txt \\
        --slc-dir /data/slc --output-dir /data/ifgs

  # ISCE2: VRT + ENVI int output, skip materialisation
  %(prog)s --processor isce2 --pairs-file pairs.txt \\
        --slc-dir /data/slc --output-dir /data/ifgs \\
        --slc-pattern *.slc --ifg-pattern *.int --only-vrt
        """
    )
    parser.add_argument('--processor', required=True, choices=['isce2', 'isce3'],
                        help="Processor type: 'isce2' or 'isce3'")
    parser.add_argument('--pairs-file', required=True,
                        help="Path to interferometric pairs file (e.g., ifgram_list.txt)")
    parser.add_argument('--slc-dir', nargs='+', required=True,
                        help="One or more directories containing SLC files (wildcards allowed)")
    parser.add_argument('--output-dir', required=True,
                        help="Output directory for interferogram files")
    parser.add_argument('--slc-pattern', default=None,
                        help="Pattern for SLC files (e.g., '*.slc.tif', '*.slc'). "
                             "Default: '*.slc.*' for isce3, '*.slc' for isce2.")
    parser.add_argument('--subdataset', default="/data/VV",
                        help="Subdataset to use for HDF5/NetCDF files (default: /data/VV)")
    parser.add_argument('--no-verify', action='store_true',
                        help="Skip SLC verification")
    parser.add_argument('--only-vrt', action='store_true',
                        help="Only create VRT interferograms, do not materialise them")
    parser.add_argument('--vrt-pattern', default='*.int.vrt',
                        help="Pattern for VRT files (default: '*.int.vrt')")
    parser.add_argument('--ifg-pattern', default=None,
                        help="Pattern for output interferograms. "
                             "Default: '*.int.tif' for isce3, '*.int' for isce2.")
    parser.add_argument('--max-workers', type=int, default=None,
                        help="Maximum number of parallel workers (default: auto)")
    parser.add_argument('--verbose', '-v', action='store_true',
                        help="Verbose logging")
    return parser.parse_args(args_list) if args_list else parser.parse_args()


def main(args=None):
    if args is None:
        args = parse_arguments()

    # Set processor‑dependent defaults
    if args.slc_pattern is None:
        args.slc_pattern = slc_pattern(args.processor)
    if args.ifg_pattern is None:
        args.ifg_pattern = int_ext(args.processor)

    logger.info("Processor: %s", args.processor)
    logger.info("SLC pattern: %s", args.slc_pattern)
    logger.info("Output interferogram pattern: %s", args.ifg_pattern)

    ok = generate_ifgram(
        pairs_file=args.pairs_file,
        slc_dir_patterns=args.slc_dir,
        output_dir=args.output_dir,
        processor=args.processor,
        slc_pattern=args.slc_pattern,
        subdataset=args.subdataset,
        no_verify=args.no_verify,
        only_vrt=args.only_vrt,
        vrt_pattern=args.vrt_pattern,
        ifg_pattern=args.ifg_pattern,
        max_workers=args.max_workers,
        verbose=args.verbose,
    )
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
