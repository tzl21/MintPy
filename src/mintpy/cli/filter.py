#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################

import logging
import sys
from pathlib import Path

from mintpy.stdproc.utils.log_utils import setup_logging
from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

from mintpy.stdproc.filter import (
    find_files_by_pattern,
    process_goldstein_files,
    process_long_wavelength_files,
)

EXAMPLE = """example:
  filter.py --processor isce3 --filter-type goldstein --input-dir ./ifgrams --pattern *.int.tif --output-dir ./filtered
  filter.py --processor isce3 --filter-type long_wavelength --input-dir ./unwrapped --wavelength-cutoff 25000
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Apply Goldstein or long-wavelength filters to interferometric data'
    """Parse command line arguments.

    Args:
        args_list: Optional list of arguments (for testing). If None, uses sys.argv.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE, subparsers=subparsers)

    parser.add_argument(
        '--processor',
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF/HDF5)"
    )

    parser.add_argument(
        '--filter-type',
        choices=['goldstein', 'long_wavelength'],
        default='goldstein',
        help='Type of filter to apply: goldstein or long_wavelength'
    )

    parser.add_argument(
        'file', nargs='*',
        help='Input file(s) to filter (single-file mode); ignored when --input-dir is given'
    )

    parser.add_argument(
        '--input-dir',
        help='Directory containing input files to be filtered (batch mode)'
    )

    parser.add_argument(
        '--pattern',
        help='File pattern for input files. If not set, uses processor default: '
             '".int.tif" for isce3, ".int" for isce2.'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for filtered files'
    )

    # Common filter parameters
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4)'
    )

    # Goldstein filter specific parameters
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.8,
        help='Goldstein filter parameter (0=no filtering, 1=max filtering) (default: 0.8)'
    )

    parser.add_argument(
        '--psize',
        type=int,
        default=32,
        help='Patch size for Goldstein filter (default: 32)'
    )

    # Long-wavelength filter specific parameters
    parser.add_argument(
        '--wavelength-cutoff',
        type=float,
        default=25000,
        help='Spatial wavelength cutoff in meters (default: 25000)'
    )

    parser.add_argument(
        '--pixel-spacing',
        type=float,
        default=30,
        help='Pixel spacing in meters (default: 30)'
    )

    parser.add_argument(
        '--correlation-cutoff',
        type=float,
        default=0.5,
        help='Correlation threshold for masking (default: 0.5)'
    )

    # Masking files for long-wavelength filter (directory + pattern)
    parser.add_argument(
        '--cor-dir',
        default=None,
        help='Directory containing correlation files for masking'
    )

    parser.add_argument(
        '--cor-pattern',
        help='Pattern for correlation files. If not set, uses processor default: '
             '"*.phsig.coh.tif" for isce3, "*.phsig.coh" for isce2.'
    )

    parser.add_argument(
        '--conncomp-dir',
        default=None,
        help='Directory containing connected component files for masking'
    )

    parser.add_argument(
        '--conncomp-pattern',
        help='Pattern for connected component files. If not set, uses processor default: '
             '".unw.conncomp.tif" for isce3, ".unw.conncomp" for isce2.'
    )

    parser.add_argument(
        '--temporal-coherence-file',
        default=None,
        help='Temporal coherence file for masking'
    )

    parser.add_argument(
        '--fill-value',
        type=float,
        default=None,
        help='Value to place in masked output pixels (default: interpolate)'
    )

    parser.add_argument(
        '--scratch-dir',
        default=None,
        help='Directory for temporary files (default: system temp)'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.pattern is None:
        inps.pattern = "**/*.int.tif"
    if getattr(inps, "cor_pattern", None) is None:
        inps.cor_pattern = "**/*.phsig.coh.tif"
    if getattr(inps, "conncomp_pattern", None) is None:
        inps.conncomp_pattern = "**/*.unw.conncomp.tif"
    if not inps.file and not inps.input_dir:
        raise SystemExit('ERROR: either input file(s) or --input-dir is required')
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    setup_logging(verbose=getattr(inps, "verbose", False))
    """Main function to handle command line arguments and execute the script."""

    output_dir = Path(inps.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find input files: an explicit file list (single-file mode) or a directory scan
    if inps.file:
        input_files = [str(f) for f in inps.file]
        logger.info("Found %d input file(s)", len(input_files))
    else:
        input_dir = Path(inps.input_dir)
        if not input_dir.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return 1
        input_files = find_files_by_pattern(input_dir, inps.pattern, inps.processor)
        if not input_files:
            logger.error(f"No files found matching pattern '{inps.pattern}' in {input_dir}")
            return 1
        logger.info(f"Found {len(input_files)} files matching pattern '{inps.pattern}' in {input_dir}")

    # Validate filter-specific parameters
    if inps.filter_type == 'goldstein':
        if inps.alpha < 0 or inps.alpha > 1:
            logger.error("Alpha must be between 0 and 1")
            return 1
        if inps.psize <= 0:
            logger.error("Patch size must be positive")
            return 1

    elif inps.filter_type == 'long_wavelength':
        if inps.wavelength_cutoff <= 0:
            logger.error("Wavelength cutoff must be positive")
            return 1
        if inps.pixel_spacing <= 0:
            logger.error("Pixel spacing must be positive")
            return 1

        # Find correlation files if directory provided
        cor_files = None
        if inps.cor_dir:
            cor_dir = Path(inps.cor_dir)
            if not cor_dir.exists():
                logger.error(f"Correlation directory not found: {cor_dir}")
                return 1
            cor_files = find_files_by_pattern(cor_dir, inps.cor_pattern, inps.processor)
            if not cor_files:
                logger.warning(f"No correlation files found matching pattern '{inps.cor_pattern}' in {cor_dir}")
            else:
                logger.info(f"Found {len(cor_files)} correlation files")

        # Find connected component files if directory provided
        conncomp_files = None
        if inps.conncomp_dir:
            conncomp_dir = Path(inps.conncomp_dir)
            if not conncomp_dir.exists():
                logger.error(f"Connected component directory not found: {conncomp_dir}")
                return 1
            conncomp_files = find_files_by_pattern(conncomp_dir, inps.conncomp_pattern, inps.processor)
            if not conncomp_files:
                logger.warning(f"No connected component files found matching pattern '{inps.conncomp_pattern}' in {conncomp_dir}")
            else:
                logger.info(f"Found {len(conncomp_files)} connected component files")

    # Process files based on filter type
    try:
        if inps.filter_type == 'goldstein':
            process_goldstein_files(input_files, output_dir, inps)
        else:
            process_long_wavelength_files(input_files, output_dir, inps, cor_files, conncomp_files)
    except Exception as e:
        logger.error(f"Filtering failed: {e}")
        return 1

    logger.info("All filtering tasks completed successfully.")
    return 0


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
