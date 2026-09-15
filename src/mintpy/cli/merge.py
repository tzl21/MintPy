#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################

import logging
import sys

from mintpy.stdproc.utils.log_utils import setup_logging
from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

from pathlib import Path

from mintpy.stdproc.utils.stitching_utils import DEFAULT_TIFF_OPTIONS
from mintpy.stdproc.merge import (
    discover_burst_ids_from_dirs,
    merge_baselines,
    merge_geometry_files,
    merge_interferograms,
    merge_slc_files,
)

EXAMPLE = """example:
  merge.py --processor isce3 --mode intf --intf-dir ./ifgrams --output-dir ./merged
  merge.py --processor isce3 --mode geometry --geom-dir ./geom --output-dir ./merged
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Merge various data types across multiple burst IDs'
    """Parse command line arguments with enhanced .h5 support."""
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE, subparsers=subparsers)

    parser.add_argument(
        "--processor",
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF/HDF5)"
    )

    parser.add_argument(
        "--burst-ids",
        nargs="+",
        default=None,
        help="List of burst IDs to merge (e.g., t124_264305_iw2). If not provided, auto-detected from directory structure."
    )

    parser.add_argument(
        "--mode",
        nargs="+",
        choices=["baseline", "intf", "slc", "geometry"],
        required=True,
        help="Processing mode(s): baseline, intf, slc, geometry"
    )

    # Path arguments for different data types
    parser.add_argument(
        "--baseline-dir",
        default=".",
        help="Base directory for baseline files (default: current directory)"
    )

    parser.add_argument(
        "--intf-dir",
        default=".",
        help="Base directory for interferogram files (default: current directory)"
    )

    parser.add_argument(
        "--slc-dir",
        default=".",
        help="Base directory for SLC files (default: current directory)"
    )

    parser.add_argument(
        "--geom-dir",
        default=".",
        help="Base directory for geometry files (default: current directory)"
    )

    parser.add_argument(
        "--output-dir",
        default="./merged",
        help="Output directory for merged files (default: ./merged)"
    )

    parser.add_argument(
        "--intf-types",
        nargs="+",
        default=[".int.tif", ".phsig.coh.tif", ".cpx.coh.tif", ".unw.tif", ".unw.conncomp.tif"],
        help="Interferogram file types to process (default: .int.tif .phsig.coh.tif .cpx.coh.tif .unw.tif .unw.conncomp.tif)"
    )

    parser.add_argument(
        "--geom-types",
        nargs="+",
        default=["height.tif", "layover_shadow_mask.tif", "local_incidence_angle.tif",
                "los_east.tif", "los_north.tif"],
        help="Geometry file types to process"
    )

    # Enhanced .h5 specific parameters
    parser.add_argument(
        "--h5-dataset-mapping",
        type=str,
        help="JSON string or path to JSON file mapping output filenames to .h5 dataset paths. "
             "Example: '{\"dem.tif\": \"/data/z\", \"los_east.tif\": \"/data/los_east\"}'"
    )

    parser.add_argument(
        "--h5-group",
        default="/data",
        help="Base HDF5 group path for geometry datasets (default: /data)"
    )

    parser.add_argument(
        "--h5-x-coords",
        default="x_coordinates",
        help="Dataset name for x coordinates in .h5 file (default: x_coordinates)"
    )

    parser.add_argument(
        "--h5-y-coords",
        default="y_coordinates",
        help="Dataset name for y coordinates in .h5 file (default: y_coordinates)"
    )

    parser.add_argument(
        "--h5-projection",
        default="projection",
        help="Dataset name for projection information in .h5 file (default: projection)"
    )

    # Stitching parameters
    parser.add_argument(
        "--out-bounds",
        nargs=4,
        type=float,
        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
        help="Output bounds for merged images in EPSG:4326 (WGS84) coordinates: left bottom right top"
    )

    parser.add_argument(
        "--out-bounds-epsg",
        type=int,
        default=4326,
        help="EPSG code for the output bounds (default: 4326 for WGS84)"
    )

    parser.add_argument(
        "--dest-epsg",
        type=int,
        help="EPSG code for the output projection. If not specified, uses the EPSG of input data."
    )

    parser.add_argument(
        "--out-nodata",
        type=float,
        help="Nodata value for output files"
    )

    parser.add_argument(
        "--in-nodata",
        type=float,
        help="Override input files' nodata value during merging"
    )

    parser.add_argument(
        "--resample-alg",
        default="lanczos",
        choices=["near", "bilinear", "cubic", "cubicspline", "lanczos", "average", "mode"],
        help="Resampling algorithm (default: lanczos)"
    )

    parser.add_argument(
        "--driver",
        default="GTiff",
        help="GDAL driver for output files (default: GTiff)"
    )

    parser.add_argument(
        "--output-suffix",
        default=".tif",
        help="Suffix for output files (default: .tif)"
    )

    parser.add_argument(
        "--output-prefix",
        default="",
        help="Prefix for output files before date"
    )

    parser.add_argument(
        "--options",
        nargs="+",
        default=DEFAULT_TIFF_OPTIONS,
        help="GDAL creation options for output files"
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files"
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of parallel workers for stitching (default: 1)"
    )

    parser.add_argument(
        "--target-aligned-pixels",
        action="store_true",
        default=True,
        help="Adjust output image bounds so pixel coordinates are integer multiples of pixel size"
    )

    parser.add_argument(
        "--strides",
        nargs=2,
        type=int,
        metavar=("X_STRIDE", "Y_STRIDE"),
        default=[1, 1],
        help="Subsampling factors: x_stride y_stride (default: 1 1)"
    )

    parser.add_argument(
        "--out-dtype",
        help="Output data type (e.g., float32, uint8)"
    )

    parser.add_argument(
        "--create-only",
        action="store_true",
        help="Create empty output file, do not write data"
    )

    parser.add_argument(
        "--keep-temp-files",
        action="store_true",
        help="Keep temporary extracted .h5 files for debugging"
    )

    parser.add_argument(
        "--force-epsg-conversion",
        action="store_true",
        default=False,
        help="Force EPSG conversion even if data EPSG cannot be detected (default: False)"
    )
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    setup_logging(verbose=getattr(inps, "verbose", False))
    """Main function to coordinate merging of different data types."""

    # Convert paths to Path objects
    baseline_dir = Path(inps.baseline_dir)
    intf_dir = Path(inps.intf_dir)
    slc_dir = Path(inps.slc_dir)
    geom_dir = Path(inps.geom_dir)
    output_dir = Path(inps.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    burst_ids = inps.burst_ids
    if burst_ids is None:
        logger.info("No burst IDs specified, auto-detecting from directory structure...")
        burst_ids = discover_burst_ids_from_dirs(
            baseline_dir, intf_dir, slc_dir, geom_dir
        )
        if not burst_ids:
            logger.error("No burst IDs found. Ensure at least one of --baseline-dir, --intf-dir, --slc-dir, or --geom-dir contains burst ID subdirectories.")
            return 1
        logger.info(f"Auto-detected {len(burst_ids)} burst IDs: {burst_ids}")

    logger.info(f"Processor: {inps.processor}")
    logger.info(f"Processing burst IDs: {burst_ids}")
    logger.info(f"Modes: {inps.mode}")
    logger.info(f"Output directory: {output_dir}")

    if inps.out_bounds:
        logger.info(f"Input bounds (EPSG:{inps.out_bounds_epsg}): {inps.out_bounds}")

    results = {}

    if "baseline" in inps.mode:
        logger.info("Merging baseline files...")
        baseline_results = merge_baselines(
            burst_ids, baseline_dir, output_dir, inps.overwrite
        )
        results["baselines"] = baseline_results
        logger.info(f"Merged {len(baseline_results)} baseline date pairs")

    if "intf" in inps.mode:
        logger.info("Merging interferogram files...")
        intf_results = merge_interferograms(
            burst_ids, intf_dir, output_dir,
            inps.intf_types, inps, inps.processor
        )
        results["interferograms"] = intf_results
        logger.info(f"Merged {len(intf_results)} interferogram files")

    if "slc" in inps.mode:
        logger.info("Merging SLC files...")
        slc_results = merge_slc_files(
            burst_ids, slc_dir, output_dir,
            inps, inps.processor
        )
        results["slc"] = slc_results
        logger.info(f"Merged {len(slc_results)} SLC dates")

    if "geometry" in inps.mode:
        logger.info("Merging geometry files...")
        geom_results = merge_geometry_files(
            burst_ids, geom_dir, output_dir,
            inps.geom_types, inps
        )
        results["geometry"] = geom_results
        logger.info(f"Merged {len(geom_results)} geometry files")

    logger.info(f"All processing complete. Results saved to {output_dir}")

    # Print summary via logger
    summary_lines = ["=" * 50, "MERGE PROCESSING SUMMARY", "=" * 50]
    for line in summary_lines:
        logger.info(line)

    for mode, mode_results in results.items():
        logger.info(f"\n{mode.upper()}:")
        if isinstance(mode_results, dict):
            for key, value in mode_results.items():
                if isinstance(value, Path):
                    logger.info(f"  {key}: {value}")
                else:
                    logger.info(f"  {key}: {len(value) if isinstance(value, list) else value}")
        else:
            logger.info(f"  Results: {mode_results}")

    return 0


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
