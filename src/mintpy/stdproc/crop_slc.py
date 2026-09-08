#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Main script for cropping SLC files with processor-specific workflow selection.

This script serves as a unified entry point for cropping SLC data produced by
different InSAR processors (ISCE2 and ISCE3). It dispatches to the appropriate
sub-script based on the `--processor` argument.
"""

import sys
import argparse
import logging

from .crop_slc_geo import parse_arguments as geo_parse
from .crop_slc_geo import main as geo_main
from .crop_slc_rdr import parse_arguments as rdr_parse
from .crop_slc_rdr import main as rdr_main


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments(args_list=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Crop SLC files with processor-specific workflow selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ISCE3 (geocoded) processing
  crop_slc.py --processor isce3 --input-dir ./input --output-dir ./output --wsen 102.8 27.3 103.3 27.6 --max-workers 4

  # ISCE2 (radar) processing
  crop_slc.py --processor isce2 --input-dir ./input --output-dir ./output --wsen 102.8 27.3 103.3 27.6 --geom-dir /path/to/geom
        """
    )

    parser.add_argument(
        "--processor",
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF/HDF5)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        nargs='+',
        required=True,
        help="Directory/pattern containing input SLC files. Can be multiple paths or glob patterns"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where cropped files will be saved"
    )
    parser.add_argument(
        "--wsen",
        type=float,
        nargs=4,
        required=True,
        metavar=('WEST', 'SOUTH', 'EAST', 'NORTH'),
        help="Crop bounds in WSEN format (West, South, East, North)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.slc.tif",
        help="File pattern for input files"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=0.0,
        help="Buffer to add around the crop area in degrees. "
             "Set large enough so coherence windows cover bbox after multilooking."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Maximum number of parallel workers"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Prefix to add to output filenames"
    )
    parser.add_argument(
        "--geom-dir",
        type=str,
        help="Directory containing coordinate files for ISCE2 (radar) processing"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        dest="no_skip_existing",
        default=False,
        help="Do not skip existing files (reprocess everything)"
    )
    parser.add_argument(
        "--by-burst",
        action="store_true",
        default=False,
        help="Output per-burst directory structure (burst ID detected from input path)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    if args_list is None:
        return parser.parse_args()
    else:
        return parser.parse_args(args_list)


def build_processor_args(args, processor):
    """
    Build argument list for the selected processor sub-script.
    """
    common = [
        "--input-dir"] + args.input_dir + [
        "--output-dir", args.output_dir,
        "--wsen", str(args.wsen[0]), str(args.wsen[1]), str(args.wsen[2]), str(args.wsen[3]),
        "--pattern", args.pattern,
        "--buffer", str(args.buffer),
        "--prefix", args.prefix,
    ]
    if args.max_workers:
        common.extend(["--max-workers", str(args.max_workers)])
    if args.verbose:
        common.append("--verbose")
    if args.no_skip_existing:
        common.append("--no-skip-existing")
    if args.by_burst:
        common.append("--by-burst")

    if processor == 'isce2':
        if not args.geom_dir:
            raise ValueError("--geom-dir is required for ISCE2 processing")
        common.extend(["--geom-dir", args.geom_dir])
        return common
    elif processor == 'isce3':
        return common
    else:
        raise ValueError(f"Unsupported processor: {processor}")


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("SLC Cropping Script")
    logger.info("=" * 50)
    logger.info(f"Processor: {args.processor}")
    logger.info(f"Crop bounds: {args.wsen}")

    sub_args_list = build_processor_args(args, args.processor)

    if args.processor == 'isce2':
        logger.info("Dispatching to ISCE2 (radar coordinates) workflow...")
        sub_args = rdr_parse(sub_args_list)
        return rdr_main(sub_args)
    elif args.processor == 'isce3':
        logger.info("Dispatching to ISCE3 (geocoded) workflow...")
        sub_args = geo_parse(sub_args_list)
        return geo_main(sub_args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logging.error("\nProcessing interrupted by user")
        sys.exit(130)
    except Exception as e:
        logging.error(f"\nUnexpected error: {e}")
        raise
