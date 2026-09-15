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

import os

from mintpy.stdproc.ifgram_list import (
    DEFAULT_MODE,
    _str2bool,
    DEFAULT_NUM_CONNECTIONS,
    PAIR_GENERATORS,
    _select_params_from_args,
    filter_date_list,
    generate_pairs,
    get_date_list,
    write_pair_list,
)

EXAMPLE = """example:
  ifgram_list.py --slc ./slc --mode sequential -n 5
  ifgram_list.py --slc ./slc --mode select --select-weight-source coherence
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Generate the interferometric pair list (ifgram_list.txt)'
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE, subparsers=subparsers)

    # Input directories
    input_group = parser.add_argument_group('Input directories')
    input_group.add_argument('--slc', dest='slc_dir', default='merged/SLC',
                           help='Directory containing merged SLCs (default: merged/SLC)')
    input_group.add_argument('--bbox', type=float, nargs=4,
                           metavar=('W', 'S', 'E', 'N'), default=None,
                           help='WSEN AOI bbox (EPSG:4326, degrees); in select mode '
                                'the quick coherence reads only this window of the '
                                'SLCs instead of the whole scene (default: None)')
    input_group.add_argument('--bbox-buffer', dest='bbox_buffer', type=float,
                           default=None,
                           help='Buffer in degrees around --bbox (default: 0.0)')
    input_group.add_argument('-w', '--work-dir', dest='work_dir', default='./',
                           help='Working directory (default: current directory)')

    # Output directories
    output_group = parser.add_argument_group('Output directories')
    output_group.add_argument('-o', '--outdir', dest='out_dir', default='interferograms',
                            help='Output directory for ifgram_list.txt (default: interferograms)')

    # Network generation parameters
    network_group = parser.add_argument_group('Network generation parameters')
    network_group.add_argument('-n', '--num-connections', type=int, default=None,
                             help='Number of nearest neighbor connections (default: 5 '
                                  'for sequential/reference; 3 for select k-NN skeleton)')
    network_group.add_argument('--mode', choices=list(PAIR_GENERATORS.keys()), default=DEFAULT_MODE,
                             help=f'Pair generation mode (default: {DEFAULT_MODE}):\n'
                                  f'  sequential  k-NN (set with -n)\n'
                                  f'  reference   star network around the earliest date\n'
                                  f'  select      coherence-aware selection (--select-*)\n'
                                  f'  file        explicit pair list (--pair-file, used verbatim)\n'
                                  f'Details: docs/ifgram_list_modes.md')
    network_group.add_argument('--pair-file', type=str, dest='pair_file', default=None,
                             help="Explicit pair list for --mode file: one "
                                  "'YYYYMMDD-YYYYMMDD' pair per line (blank "
                                  "lines and '#' comments allowed); used "
                                  "verbatim, replaces -n / --select-*")
    network_group.add_argument('--oneyear-interferograms', type=int, dest='oneyear_interferograms',
                             help='Days range for one-year interferograms')
    network_group.add_argument('--start-date', type=str, dest='start_date', default=None,
                             help='Only keep dates >= YYYYMMDD (inclusive)')
    network_group.add_argument('--end-date', type=str, dest='end_date', default=None,
                             help='Only keep dates <= YYYYMMDD (inclusive)')
    network_group.add_argument('--exclude-date', type=str, dest='exclude_date',
                             action='append', default=None,
                             help='Exclude specific SLC date(s) YYYYMMDD (e.g. bad '
                                  'acquisitions); repeatable and/or comma/space '
                                  'separated')
    network_group.add_argument('--processor', type=str, choices=['isce2', 'isce3'], default='isce3',
                             help='Processor type for quick coherence in select mode (default: isce3)')

    # Selection parameters (mode=select)
    select_group = parser.add_argument_group(
        'Selection parameters (mode=select) — coherence-aware connected '
        'selection (see docs/ifgram_list_modes.md)')
    select_group.add_argument('--select-weight-source', dest='select_weight_source',
                             choices=['model', 'coherence', 'mixed'], default=None,
                             help='Quality source: model (temporal decorrelation), '
                                  'coherence (measured: existing rasters or quick '
                                  'coherence on downsampled SLCs), mixed (measured '
                                  'with model fallback) (default: model)')
    select_group.add_argument('--select-annual-windows', dest='select_annual_windows',
                             default=None,
                             help='Temporal windows as "center:tol" day pairs, comma '
                                  'separated, e.g. "182:10,365:15" (default: 182:10,365:15)')
    select_group.add_argument('--select-temp-baseline-max', dest='select_temp_baseline_max',
                             type=int, default=None,
                             help='Also include ALL pairs within this many days (None = off)')
    select_group.add_argument('--select-perp-baseline-max', dest='select_perp_baseline_max',
                             type=float, default=None,
                             help='Maximum perpendicular baseline in metres (None = off)')
    select_group.add_argument('--select-perp-baseline-file', dest='select_perp_baseline_file',
                             default=None,
                             help='Two-column "date bperp" text file (None = off)')
    select_group.add_argument('--select-coh-dir', dest='select_coh_dir', default=None,
                             help='Existing coherence rasters dir ({date1}_{date2}/xxx_coh) '
                                  '(None = use quick coherence from SLCs)')
    select_group.add_argument('--select-coh-kind', dest='select_coh_kind',
                             choices=['phsig', 'cpx'], default=None,
                             help='Coherence kind for existing rasters (default: phsig)')
    select_group.add_argument('--select-coh-variant', dest='select_coh_variant',
                             choices=['fullres', 'mli', 'filt', 'filt_mli'], default=None,
                             help='Coherence variant for existing rasters (default: filt_mli)')
    select_group.add_argument('--select-coh-stat', dest='select_coh_stat',
                             choices=['mean', 'median', 'usable_frac', 'fisher'], default=None,
                             help='Statistic aggregating a coherence map (default: mean)')
    select_group.add_argument('--select-coh-usable-threshold', dest='select_coh_usable_threshold',
                             type=float, default=None,
                             help='Threshold for stat=usable_frac (default: 0.3)')
    select_group.add_argument('--select-quick-window', dest='select_quick_window',
                             type=int, default=None,
                             help='Coherence window for quick coherence (default: 5)')
    select_group.add_argument('--select-quick-nlks', dest='select_quick_nlks',
                             type=int, default=None,
                             help='Block-mean downsampling factor of the --bbox window '
                                  'for quick coherence (default: 1 = full resolution)')
    select_group.add_argument('--select-quick-debias', dest='select_quick_debias',
                             type=_str2bool, metavar='{true,false}', default=None,
                             help='Touzi bias correction for quick coherence (default: true)')
    select_group.add_argument('--select-quick-max-workers', dest='select_quick_max_workers',
                             type=int, default=None,
                             help='Threads for on-the-fly coherence (default: 1 = serial)')
    select_group.add_argument('--select-model-tau-days', dest='select_model_tau_days',
                             type=float, default=None,
                             help='Temporal decorrelation time constant in days (default: 90)')
    select_group.add_argument('--select-model-gamma0', dest='select_model_gamma0',
                             type=float, default=None,
                             help='Coherence at zero temporal baseline (default: 1.0)')
    select_group.add_argument('--select-min-degree', dest='select_min_degree',
                             type=int, default=None,
                             help='Minimum interferograms per date (default: 2; 0/1 = pure spanning tree)')
    select_group.add_argument('--select-max-pairs', dest='select_max_pairs',
                             type=int, default=None,
                             help='Global edge budget (None = unbounded)')
    select_group.add_argument('--select-quality-threshold', dest='select_quality_threshold',
                             type=float, default=None,
                             help='Augmentation floor on the quality weight (default: 0.0)')
    select_group.add_argument('--select-robust', dest='select_robust',
                             type=_str2bool, metavar='{true,false}', default=None,
                             help='Repair bridges with highest-weight crossings, 2-edge '
                                  'robustness where possible (default: false)')
    select_group.add_argument('--select-report', dest='select_report', default=None,
                             help='Write the selection report JSON to this path (None = off)')
    select_group.add_argument('--select-dot', dest='select_dot', default=None,
                             help='Write the selected network as GraphViz DOT to this path (None = off)')
    return parser
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.mode not in PAIR_GENERATORS:
        raise ValueError(f"Invalid mode '{inps.mode}'. Available modes: {list(PAIR_GENERATORS.keys())}")
    inps.work_dir = os.path.abspath(inps.work_dir)
    inps.slc_dir = os.path.join(inps.work_dir, inps.slc_dir)
    inps.out_dir = os.path.join(inps.work_dir, inps.out_dir)
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)
    setup_logging(verbose=getattr(inps, "verbose", False))
    """Main function."""
    try:
        
        # Get date list
        date_list = get_date_list(inps.slc_dir)

        # Optional date-range / exclusion filter
        date_list = filter_date_list(
            date_list,
            start_date=getattr(inps, 'start_date', None),
            end_date=getattr(inps, 'end_date', None),
            exclude_date=getattr(inps, 'exclude_date', None),
        )

        output_file = os.path.join(inps.out_dir, 'ifgram_list.txt')

        if inps.mode == 'select':
            # Coherence-aware connected selection (see select_ifgrams.py)
            from mintpy.stdproc.select_ifgrams import select_pairs
            params = _select_params_from_args(inps)
            if inps.num_connections is not None:
                params['num_connections'] = inps.num_connections
            pairs, report = select_pairs(
                date_list, slc_dir=inps.slc_dir, processor=inps.processor,
                params=params)
        else:
            # Generate pairs
            select_params = {}
            if getattr(inps, 'select_annual_windows', None):
                select_params['annual_windows'] = inps.select_annual_windows
            if getattr(inps, 'pair_file', None):
                select_params['pair_file'] = inps.pair_file
            pairs = generate_pairs(
                date_list=date_list,
                mode=inps.mode,
                num_connections=inps.num_connections
                if inps.num_connections is not None else DEFAULT_NUM_CONNECTIONS,
                oneyear_range=inps.oneyear_interferograms,
                select_params=select_params or None,
            )
        
        # Write pair list
        write_pair_list(pairs, output_file)
        
        logger.info("Processing completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
