#!/usr/bin/env python3
############################################################
# Program is part of MintPy / slc2ifg (moved from insarflow)  #
# Copyright (c) 2026, Zhenli Tang, Zhang Yunjun            #
# Author: Zhenli Tang, 2026                                #
############################################################
"""
Canonical file/directory naming for the slc2ifg pipeline.

Fixed structures (this module is the single source of truth):

SLC products (``slc_dir``):
    isce2:  ``slc_dir/yyyymmdd.slc``
    isce3:  ``slc_dir/yyyymmdd.slc.tif``  or  ``slc_dir/yyyymmdd.slc.h5``

Interferogram products (``intf_dir``):
    isce2:  ``intf_dir/date_pair/xxx.int``
            ``intf_dir/date_pair/xxx.unw``
            ``intf_dir/date_pair/xxx.conncomp``
            ``intf_dir/date_pair/xxx_cpx.coh``        (complex coherence)
            ``intf_dir/date_pair/xxx_phsig.coh``      (phase-sigma coherence)
    isce3:  same with ``.tif`` appended: ``xxx.int.tif``, ``xxx.unw.tif``,
            ``xxx.conncomp.tif``, ``xxx_cpx.coh.tif``, ``xxx_phsig.coh.tif``

where ``date_pair`` = ``{date1}_{date2}`` and the interferogram variant
``xxx`` is one of:

    fullres   — no processing applied
    mli       — multilooked only
    filt      — filtered only
    filt_mli  — multilooked + filtered
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

# ------------------------------------------------------------------------
# Variants & constants
# ------------------------------------------------------------------------
#: Interferogram variants in canonical order
IFG_VARIANTS: Tuple[str, ...] = ("fullres", "mli", "filt_mli", "filt")

#: Coherence product kinds
COH_KIND_CPX = "cpx"
COH_KIND_PHSIG = "phsig"
COH_KINDS: Tuple[str, ...] = (COH_KIND_CPX, COH_KIND_PHSIG)

#: Raster product kinds living in a date-pair directory
PRODUCT_KINDS: Tuple[str, ...] = ("int", "unw", "conncomp", "coh")

#: Date-pair pattern: 20240107_20240119 (may appear as a prefix of filenames)
DATE_PAIR_RE = re.compile(r"(\d{8})[-_](\d{8})")


def is_valid_variant(name: str) -> bool:
    """Return True if ``name`` is one of the canonical variants."""
    return name in IFG_VARIANTS


def is_date_pair_dir(name: str) -> bool:
    """Return True if ``name`` looks like a ``{date1}_{date2}`` directory."""
    return bool(DATE_PAIR_RE.match(name) and len(DATE_PAIR_RE.match(name).group(0)) == len(name))


def extract_date_pair(name: str) -> Optional[str]:
    """Extract ``{date1}_{date2}`` from a string (filename, dir name or path)."""
    m = DATE_PAIR_RE.search(name)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    return None


# ------------------------------------------------------------------------
# SLC naming
# ------------------------------------------------------------------------
def slc_pattern(processor: str) -> str:
    """Default SLC glob pattern for a processor.

    isce2 -> ``*.slc``; isce3 -> ``*.slc.*`` (matches both ``.slc.tif`` and
    ``.slc.h5``).
    """
    if processor == "isce2":
        return "*.slc"
    return "*.slc.*"


def slc_file(date: str, processor: str) -> str:
    """Canonical SLC filename for a date.

    isce2 -> ``yyyymmdd.slc``; isce3 -> ``yyyymmdd.slc.tif``
    (the ``.h5`` form is treated as an *input* variant only; the pipeline's
    own SLC outputs use GeoTIFF).
    """
    if processor == "isce2":
        return f"{date}.slc"
    return f"{date}.slc.tif"


# ------------------------------------------------------------------------
# Product extension helpers
# ------------------------------------------------------------------------
def _ext(processor: str, suffix: str) -> str:
    """Append ``.tif`` for isce3 (``suffix`` already includes the leading dot)."""
    return f"{suffix}.tif" if processor == "isce3" else suffix


def int_ext(processor: str) -> str:
    """Extension of wrapped interferograms: ``.int`` / ``.int.tif``."""
    return _ext(processor, ".int")


def unw_ext(processor: str) -> str:
    """Extension of unwrapped interferograms: ``.unw`` / ``.unw.tif``."""
    return _ext(processor, ".unw")


def conncomp_ext(processor: str) -> str:
    """Extension of connected-component files: ``.conncomp`` / ``.conncomp.tif``."""
    return _ext(processor, ".conncomp")


def coh_ext(processor: str, kind: str = COH_KIND_PHSIG) -> str:
    """Extension of coherence files.

    Complex coherence uses the ``_cpx`` infix, phase-sigma the ``_phsig``
    infix: ``xxx_cpx.coh[.tif]`` / ``xxx_phsig.coh[.tif]``.
    """
    if kind not in COH_KINDS:
        raise ValueError(f"Unknown coherence kind '{kind}', expected {COH_KINDS}")
    return _ext(processor, f"_{kind}.coh")


# ------------------------------------------------------------------------
# Date-pair directory helpers
# ------------------------------------------------------------------------
def date_pair_dir(output_dir: Union[str, Path], date1: str, date2: str) -> Path:
    """Path of the ``{date1}_{date2}`` subdirectory under ``output_dir``."""
    return Path(output_dir) / f"{date1}_{date2}"


def glob_date_pair_dirs(output_dir: Union[str, Path]) -> List[Path]:
    """List ``{date1}_{date2}`` subdirectories directly under ``output_dir``."""
    out = Path(output_dir)
    if not out.is_dir():
        return []
    return sorted(
        p for p in out.iterdir()
        if p.is_dir() and is_date_pair_dir(p.name)
    )


# ------------------------------------------------------------------------
# Full product paths
# ------------------------------------------------------------------------
def ifg_path(
    output_dir: Union[str, Path],
    date1: str,
    date2: str,
    variant: str = "fullres",
    processor: str = "isce3",
) -> Path:
    """Path of the wrapped interferogram ``{date_pair}/{variant}.int[.tif]``."""
    if not is_valid_variant(variant):
        raise ValueError(f"Invalid variant '{variant}', expected one of {IFG_VARIANTS}")
    return date_pair_dir(output_dir, date1, date2) / f"{variant}{int_ext(processor)}"


def unw_path(
    output_dir: Union[str, Path],
    date1: str,
    date2: str,
    variant: str = "fullres",
    processor: str = "isce3",
) -> Path:
    """Path of the unwrapped phase ``{date_pair}/{variant}.unw[.tif]``."""
    if not is_valid_variant(variant):
        raise ValueError(f"Invalid variant '{variant}', expected one of {IFG_VARIANTS}")
    return date_pair_dir(output_dir, date1, date2) / f"{variant}{unw_ext(processor)}"


def conncomp_path(
    output_dir: Union[str, Path],
    date1: str,
    date2: str,
    variant: str = "fullres",
    processor: str = "isce3",
) -> Path:
    """Path of connected components ``{date_pair}/{variant}.conncomp[.tif]``."""
    if not is_valid_variant(variant):
        raise ValueError(f"Invalid variant '{variant}', expected one of {IFG_VARIANTS}")
    return date_pair_dir(output_dir, date1, date2) / f"{variant}{conncomp_ext(processor)}"


def coh_path(
    output_dir: Union[str, Path],
    date1: str,
    date2: str,
    variant: str = "fullres",
    kind: str = COH_KIND_PHSIG,
    processor: str = "isce3",
) -> Path:
    """Path of a coherence raster ``{date_pair}/{variant}_{kind}.coh[.tif]``."""
    if not is_valid_variant(variant):
        raise ValueError(f"Invalid variant '{variant}', expected one of {IFG_VARIANTS}")
    if kind not in COH_KINDS:
        raise ValueError(f"Unknown coherence kind '{kind}', expected {COH_KINDS}")
    return date_pair_dir(output_dir, date1, date2) / f"{variant}{coh_ext(processor, kind)}"


#: Infixes of atmosphere-corrected products (appended to the base product name)
ATM_INFIX = ".atm"


def atm_path(
    output_dir: Union[str, Path],
    date1: str,
    date2: str,
    variant: str = "fullres",
    processor: str = "isce3",
) -> Path:
    """Path of an atmosphere-corrected unwrapped phase
    ``{date_pair}/{variant}.atm.unw[.tif]`` (e.g. ``filt_mli.atm.unw.tif``).

    The variant is inherited from the input unwrapped phase it derives from.
    """
    if not is_valid_variant(variant):
        raise ValueError(f"Invalid variant '{variant}', expected one of {IFG_VARIANTS}")
    return date_pair_dir(output_dir, date1, date2) / (
        f"{variant}{ATM_INFIX}{unw_ext(processor)}")


# ------------------------------------------------------------------------
# Parsing helpers (filename -> variant)
# ------------------------------------------------------------------------
def strip_product_extensions(name: str, processor: str) -> str:
    """Remove known product extensions from a filename, returning the variant.

    ``fullres.int.tif`` -> ``fullres``; ``mli_phsig.coh`` -> ``mli``
    (the ``_cpx``/``_phsig`` coherence infix is also removed).
    """
    stem = name
    # Remove trailing processor extension (.tif for isce3)
    if processor == "isce3" and stem.endswith(".tif"):
        stem = stem[:-4]
    # Remove product extension
    for prod_ext in (".conncomp", ".int", ".unw", ".coh"):
        if stem.endswith(prod_ext):
            stem = stem[: -len(prod_ext)]
            break
    # Remove coherence infix (and other product infixes such as .atm)
    for kind in COH_KINDS + (ATM_INFIX,):
        if stem.endswith(kind):
            stem = stem[: -len(kind)]
            break
    return stem


def variant_of(filename: Union[str, Path], processor: str = "isce3") -> str:
    """Return the canonical variant of a product file, defaulting to ``fullres``."""
    variant = strip_product_extensions(Path(filename).name, processor)
    return variant if is_valid_variant(variant) else "fullres"


def next_variant(variant: str, stage: str) -> str:
    """Map an input variant to the output variant of a processing stage.

    Parameters
    ----------
    variant : str
        Input interferogram variant.
    stage : {'multilook', 'filter'}
        Processing stage applied.

    Returns
    -------
    str
        ``multilook``: fullres->mli, filt->filt_mli, (mli/filt_mli unchanged)
        ``filter``:   fullres->filt, mli->filt_mli, (filt/filt_mli unchanged)
    """
    if stage == "multilook":
        return "mli" if variant == "fullres" else (
            "filt_mli" if variant == "filt" else variant)
    elif stage == "filter":
        return "filt" if variant == "fullres" else (
            "filt_mli" if variant == "mli" else variant)
    raise ValueError(f"Unknown stage '{stage}', expected 'multilook' or 'filter'")
