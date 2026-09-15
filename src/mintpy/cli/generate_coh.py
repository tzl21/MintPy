#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################

import logging
import sys

from mintpy.utils.arg_utils import create_argument_parser

logger = logging.getLogger(__name__)

EXAMPLE = """example:
  # phase-sigma coherence from interferograms (isce3)
  generate_coh.py --processor isce3 --input './filtered/*.int.tif' --output-dir ./coh

  # complex coherence from SLC pairs (isce3)
  generate_coh.py --processor isce3 --pairs-file ifgram_list.txt --slc-dir ./slc \\
      --output-dir ./coh --skip-phase-sigma

  # both, plus the phase std-dev raster
  generate_coh.py --processor isce3 --input './ifgrams/*/*.int.tif' \\
      --pairs-file ifgram_list.txt --slc-dir ./slc --keep-sigma
"""


def create_parser(subparsers=None):
    name = __name__.split('.')[-1]
    synopsis = 'Generate coherence products (complex and phase-sigma)'
    parser = create_argument_parser(
        name, synopsis=synopsis, description=synopsis, epilog=EXAMPLE,
        subparsers=subparsers)

    parser.add_argument('--processor', choices=['isce2', 'isce3'], required=True,
                        help="Processor type: 'isce2' (radar coordinates) or 'isce3' (geocoded)")
    parser.add_argument('--input', nargs='+',
                        help='Interferogram file(s) or glob for the phase-sigma estimator')
    parser.add_argument('--output-dir', default='.',
                        help='Output directory for all coherence products (default: %(default)s)')
    parser.add_argument('--pairs-file',
                        help='Interferometric pairs list for the complex-coherence estimator')
    parser.add_argument('--slc-dir', nargs='+',
                        help='Directory pattern(s) containing SLC files (complex coherence)')
    parser.add_argument('--slc-pattern', default=None,
                        help='SLC glob (default: pipeline product pattern then processor raw pattern)')
    parser.add_argument('--subdataset', default='/data/VV',
                        help='HDF5 subdataset (default: %(default)s)')

    ps = parser.add_argument_group('Phase-sigma coherence options')
    ps.add_argument('--ps-window-size', type=int, default=5,
                    help='Phase-sigma window size (default: %(default)s)')
    ps.add_argument('--ps-gradient-window', type=int, default=5,
                    help='Gradient estimation window size (default: %(default)s)')
    ps.add_argument('--ps-nlks', type=float, default=1.0,
                    help='Number of looks (default: %(default)s)')
    ps.add_argument('--keep-sigma', action='store_true',
                    help='Also write the phase std-dev raster (xxx.phsig.sigma.tif)')

    cc = parser.add_argument_group('Complex coherence options')
    cc.add_argument('--cc-window-size', type=int, default=5,
                    help='Sliding window size (default: %(default)s)')
    cc.add_argument('--cc-window-type', choices=['triangular', 'uniform'],
                    default='triangular',
                    help='Window weighting: triangular (ISCE2 Bartlett) or uniform (default: %(default)s)')

    proc = parser.add_argument_group('Processing options')
    proc.add_argument('--max-workers', type=int, default=1,
                      help='Number of parallel workers (default: %(default)s)')
    proc.add_argument('--skip-phase-sigma', action='store_true',
                      help='Skip the phase-sigma estimator')
    proc.add_argument('--skip-complex-coherence', action='store_true',
                      help='Skip the complex-coherence estimator')
    proc.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    return parser


def cmd_line_parse(iargs=None):
    parser = create_parser()
    inps = parser.parse_args(args=iargs)
    if inps.skip_phase_sigma and inps.skip_complex_coherence:
        parser.error('both estimators are skipped -- nothing to do')
    if not inps.skip_phase_sigma and not inps.input:
        parser.error('--input is required for phase-sigma coherence '
                     '(or use --skip-phase-sigma)')
    if not inps.skip_complex_coherence and not (inps.pairs_file and inps.slc_dir):
        parser.error('--pairs-file and --slc-dir are required for complex coherence '
                     '(or use --skip-complex-coherence)')
    return inps


##################################################################################################
def main(iargs=None):
    inps = cmd_line_parse(iargs)

    from mintpy.stdproc.generate_coh import generate_coh
    from mintpy.stdproc.utils.log_utils import setup_logging

    setup_logging(verbose=inps.verbose)

    slc_pattern = inps.slc_pattern
    if slc_pattern is None:
        from mintpy.stdproc.utils import naming
        slc_pattern = list(naming.slc_patterns(inps.processor))

    return generate_coh(
        processor=inps.processor,
        input_files=inps.input,
        output_dir=inps.output_dir,
        pairs_file=inps.pairs_file,
        slc_dir=inps.slc_dir,
        slc_pattern=slc_pattern,
        skip_phase_sigma=inps.skip_phase_sigma,
        skip_complex_coherence=inps.skip_complex_coherence,
        ps_window_size=inps.ps_window_size,
        ps_gradient_window=inps.ps_gradient_window,
        ps_nlks=inps.ps_nlks,
        keep_sigma=inps.keep_sigma,
        cc_window_size=inps.cc_window_size,
        cc_window_type=inps.cc_window_type,
        max_workers=inps.max_workers,
        subdataset=inps.subdataset,
    )


###################################################################################################
if __name__ == '__main__':
    main(sys.argv[1:])
