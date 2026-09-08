#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Unified coherence generation script.

This script serves as a single entry point to generate both phase-sigma correlation
(from interferograms) and traditional complex coherence (from SLC pairs).
It dispatches to the appropriate sub-modules based on provided arguments.
"""

import argparse
import logging
import sys
from pathlib import Path

from .generate_coh_phsig import main as phsig_main
from .generate_coh_phsig import parse_arguments as phsig_parse
from .generate_coh_complex import main as complex_main
from .generate_coh_complex import parse_arguments as complex_parse


def setup_logging(verbose: bool = False):
    """Setup basic logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def parse_arguments(args_list=None):
    """Parse command line arguments for unified coherence generation."""
    parser = argparse.ArgumentParser(
        description="Generate both phase-sigma correlation and traditional coherence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # For ISCE3 (geocoded) data:
  generate_coh.py --processor isce3 --input *.int.tif --pairs-file ifgram_list.txt \\
      --slc-dir ../t124*/*/ --output-dir ./coh

  # For ISCE2 (radar) data:
  generate_coh.py --processor isce2 --input *.int --pairs-file ifgram_list.txt \\
      --slc-dir ../merged/SLC/ --slc-pattern *.slc --output-dir ./coh
        """
    )

    parser.add_argument(
        '--processor',
        type=str,
        choices=['isce2', 'isce3'],
        required=True,
        help="Processor type: 'isce2' (radar coordinates, ENVI format) or 'isce3' (geocoded, GeoTIFF/HDF5)"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        nargs='+',
        help='Interferogram files or glob pattern for phase-sigma (e.g., ./filtered/*.int.tif)'
    )
    parser.add_argument(
        '--pairs-file',
        type=str,
        default=None,
        help='Path to interferometric pairs list file (e.g., ifgram_list.txt) for complex coherence'
    )
    parser.add_argument(
        '--slc-dir',
        type=str,
        default=None,
        nargs='+',
        help='Directory patterns containing SLC files (can use wildcards like ../t124*iw2/*/)'
    )
    parser.add_argument(
        '--slc-pattern',
        type=str,
        default='*.slc.*',
        help='Pattern for SLC files (e.g., "*.slc.tif", "*.slc", "*.slc.h5") (default: "*.slc.*")'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='.',
        help='Output directory for all coherence products'
    )

    # Phase-sigma specific options
    ps_group = parser.add_argument_group('Phase-sigma correlation options')
    ps_group.add_argument(
        '--ps-window-size',
        type=int,
        default=5,
        help='Phase sigma window size (default: 5)'
    )
    ps_group.add_argument(
        '--ps-gradient-window',
        type=int,
        default=5,
        help='Gradient estimation window size (default: 5)'
    )
    ps_group.add_argument(
        '--ps-nlks',
        type=float,
        default=1.0,
        help='Number of looks for phase sigma estimator (default: 1.0)'
    )
    ps_group.add_argument(
        '--keep-sigma',
        action='store_true',
        help='Also write phase standard deviation raster'
    )

    # Complex coherence specific options
    cc_group = parser.add_argument_group('Complex coherence options')
    cc_group.add_argument(
        '--cc-window-size',
        type=int,
        default=5,
        help='Sliding window size for coherence estimation (default: 5)'
    )

    # Processing options
    proc_group = parser.add_argument_group('Processing options')
    proc_group.add_argument(
        '--max-workers',
        type=int,
        default=1,
        help='Number of parallel workers for each sub-module (default: 1)'
    )
    proc_group.add_argument(
        '--skip-phase-sigma',
        action='store_true',
        help='Skip phase-sigma correlation generation'
    )
    proc_group.add_argument(
        '--skip-complex-coherence',
        action='store_true',
        help='Skip complex coherence generation'
    )

    log_group = parser.add_argument_group('Logging')
    log_group.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args(args_list) if args_list else parser.parse_args()


def build_phsig_args(args):
    """Construct argument list for phase-sigma module."""
    sub_args = [
        '--processor', args.processor,
        '--input'] + list(args.input) + [
        '--output-dir', args.output_dir,
        '--window-size', str(args.ps_window_size),
        '--gradient-window', str(args.ps_gradient_window),
        '--nlks', str(args.ps_nlks),
        '--max-workers', str(args.max_workers),
    ]
    if args.keep_sigma:
        sub_args.append('--keep-sigma')
    if args.verbose:
        sub_args.append('--verbose')
    return sub_args


def build_complex_args(args):
    """Construct argument list for complex coherence module."""
    sub_args = [
        '--processor', args.processor,
        '--pairs-file', args.pairs_file,
        '--slc-dir'] + args.slc_dir + [
        '--slc-pattern', args.slc_pattern,
        '--output', args.output_dir,
        '--window-size', str(args.cc_window_size),
        '--max-workers', str(args.max_workers),
    ]
    if args.verbose:
        sub_args.append('--verbose')
    return sub_args


def main():
    """Main entry point."""
    args = parse_arguments()
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    logger.info("Unified Coherence Generation")
    logger.info(f"Processor: {args.processor}")
    logger.info(f"Output directory: {args.output_dir}")

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    exit_code = 0

    # Run phase-sigma correlation if not skipped
    if not args.skip_phase_sigma:
        logger.info("--- Running Phase-Sigma Correlation ---")
        ps_args = build_phsig_args(args)
        ps_parsed = phsig_parse(ps_args)
        try:
            ret = phsig_main(ps_parsed)
            if ret != 0:
                logger.error("Phase-sigma correlation failed.")
                exit_code = 1
        except Exception as e:
            logger.error(f"Exception in phase-sigma module: {e}")
            exit_code = 1
    else:
        logger.info("Skipping phase-sigma correlation.")

    # Validate required args for complex coherence
    if not args.skip_complex_coherence:
        if not args.pairs_file:
            logger.error("--pairs-file is required for complex coherence (or use --skip-complex-coherence)")
            return 1
        if not args.slc_dir:
            logger.error("--slc-dir is required for complex coherence (or use --skip-complex-coherence)")
            return 1

    # Run complex coherence if not skipped
    if not args.skip_complex_coherence:
        logger.info("--- Running Complex Coherence ---")
        cc_args = build_complex_args(args)
        cc_parsed = complex_parse(cc_args)
        try:
            ret = complex_main(cc_parsed)
            if ret != 0:
                logger.error("Complex coherence failed.")
                exit_code = 1
        except Exception as e:
            logger.error(f"Exception in complex coherence module: {e}")
            exit_code = 1
    else:
        logger.info("Skipping complex coherence.")

    logger.info("All coherence generation tasks completed.")
    return exit_code


if __name__ == '__main__':
    sys.exit(main())