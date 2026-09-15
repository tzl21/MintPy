#!/usr/bin/env python3
"""Logging setup shared by the slc2ifg command line interfaces."""

import logging
import sys
from typing import Optional

LOGFMT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


def setup_logging(verbose: bool = False, log_file: Optional[str] = None,
                  logger_name: Optional[str] = None) -> logging.Logger:
    """Configure console (and optional file) logging.

    Parameters
    ----------
    verbose : bool
    log_file : str, optional
    logger_name : str, optional
        Configure this named logger instead of the root logger.
    """
    logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter(LOGFMT, datefmt='%Y-%m-%d %H:%M:%S')
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logging.getLogger('osgeo').setLevel(logging.WARNING)
    return logger
