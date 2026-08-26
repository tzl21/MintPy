#!/usr/bin/env python3
############################################################
# Program is part of MintPy                                #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
SLC-to-interferogram pre-processing (slc2ifg) runner for MintPy.

Generates unwrapped interferograms (+ coherence, connected components) from
a directory of SLC images, as a pre-processing step before MintPy's
``load_data``. Two interchangeable execution backends (see
``mintpy.stdproc.slc2ifg.executor``):

- **basic** (default, no extra install): sequential chain with MintPy's own
  joblib parallelism over date pairs (``mintpy.compute.numWorker``);
- **insarflow engine** (``pip install mintpy[engine]``): Dask-scheduled
  CPU/GPU execution with tiling and artifact management.

The backend is selected by ``mintpy.slc2ifg.engine = none | insarflow |
auto`` in the config.  With ``auto`` (default) the engine is used when the
``insarflow`` package is importable, otherwise the basic executor runs the
full pipeline — MintPy can always complete the slc2ifg workflow.

Configuration is MintPy-style (flat ``key = value``, no section header),
see ``mintpy/defaults/smallbaselineApp.cfg`` for the ``mintpy.slc2ifg.*`` /
``slc2ifg.*`` / ``engine.*`` keys and ``docs/slc2ifg.md``.
"""

import logging
import os
import sys

from mintpy.utils import readfile


def read_slc2ifg_template(cfg_file):
    """Read the slc2ifg default template (merges with the user config file)."""
    import mintpy
    tfile = os.path.join(os.path.dirname(mintpy.__file__),
                         'stdproc/slc2ifg/template/slc2ifg.cfg')
    tdict = readfile.read_template(tfile) if os.path.isfile(tfile) else {}
    cdict = readfile.read_template(cfg_file) if cfg_file else {}
    tdict.update(cdict)
    return tdict


def run_slc2ifg(cfg_file=None, template=None):
    """Run the slc2ifg pipeline from a config file or a template dict.

    Parameters: cfg_file  - str, path to the MintPy-style config file
                template  - dict, pre-parsed config (takes precedence)
    Returns:    None
    """
    from mintpy.stdproc.slc2ifg.executor import get_executor

    if isinstance(template, str):
        # --template as a raw config string: parse it into a dict
        # (read_template accepts a raw string directly)
        template = readfile.read_template(template)
    cfg = template if template else read_slc2ifg_template(cfg_file)
    if cfg_file:
        cfg['_cfg_file'] = os.path.abspath(cfg_file)

    # validate the mandatory input
    slc_input = cfg.get('slc2ifg.slc_input')
    if not slc_input:
        raise ValueError(
            'slc2ifg.slc_input is required in the config '
            '(path to the SLC directory)')

    # ensure logging is configured (the engine's logger needs a root config)
    if not logging.root.handlers:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    executor = get_executor(cfg)
    print(f'run slc2ifg with the {type(executor).__name__} ...')
    executor.run(cfg)
    return


def main(iargs=None):
    """Command line entry (mintpy slc2ifg <cfg>)."""
    from mintpy.cli import slc2ifg as cli

    inps = cli.parse_arguments(iargs)
    run_slc2ifg(cfg_file=inps.cfg_file,
                template=None if inps.cfg_file else inps.template)
    return 0


if __name__ == '__main__':
    sys.exit(main())
