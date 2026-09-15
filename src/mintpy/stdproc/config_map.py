#!/usr/bin/env python3
"""Canonical slc2ifg configuration keys and their legacy aliases.

Functional duplicates used to live under several per-tool namespaces
(``crop_slc.wsen`` vs ``bbox``, ``generate_coh.ps_nlks`` vs ``nlooks``, ...).
The canonical key is the SINGLE top-level ``slc2ifg.<name>`` (e.g.
``slc2ifg.bbox``, ``slc2ifg.unwrap.mask_file``); the old keys keep working for
one deprecation cycle through :func:`normalize_config`, which is applied once at
every entry point that reads a raw config dict.

Keys that were removed outright (their feature is now automatic) are listed in
:data:`REMOVED_KEYS`: they are warned about and dropped, so old configs keep
loading instead of failing.
"""

import logging

logger = logging.getLogger(__name__)

#: legacy key -> canonical key.  Applied with the CANONICAL key taking priority
#: when both are present in the same config.
LEGACY_ALIASES = {
    # AOI
    'slc2ifg.crop_slc.wsen': 'slc2ifg.bbox',
    'slc2ifg.crop_slc.buffer': 'slc2ifg.bbox_buffer',
    # looks
    'slc2ifg.unwrap.nlooks': 'slc2ifg.nlooks',
    'slc2ifg.unwrap.snaphu.nlooks': 'slc2ifg.nlooks',
    'slc2ifg.generate_coh.ps_nlks': 'slc2ifg.nlooks',
    # unwrap mask (nonzero = valid/unwrappable, 0 = excluded)
    'slc2ifg.mask': 'slc2ifg.unwrap.mask_file',
    'slc2ifg.unwrap.snaphu.mask_file': 'slc2ifg.unwrap.mask_file',
    # no-skip-existing is an engine-wide knob
    'slc2ifg.crop_slc.no_skip_existing': 'engine.no_skip_existing',
    # executor selector
    'mintpy.slc2ifg.engine': 'slc2ifg.engine',
}

#: keys that are accepted but no longer have any effect.  Every key whose
#: feature became automatic (slc pattern inference, HDF5 subdataset detection,
#: coherence auto-discovery, fixed product paths) is listed here so existing
#: configurations keep loading with a deprecation warning.
REMOVED_KEYS = {
    'mintpy.slc2ifg.skip',
    'slc2ifg.ifgram_list.oneyear_interferograms',
    'slc2ifg.generate_coh.skip_phase_sigma',
    'slc2ifg.generate_coh.skip_complex_coherence',
    # SLC discovery: the pattern is inferred from slc2ifg.slc_input
    'slc2ifg.slc_pattern',
    'slc2ifg.crop_slc.pattern',
    'slc2ifg.generate_ifgram.slc_pattern',
    'slc2ifg.generate_coh.slc_pattern',
    'slc2ifg.ifgram_list.select.slc_pattern',
    # HDF5 subdataset is auto-detected (/data/[VV,VH,HH], preferring VV)
    'slc2ifg.subdataset',
    'slc2ifg.generate_ifgram.subdataset',
    'slc2ifg.generate_coh.subdataset',
    # product patterns: all slc2ifg products are .tif
    'slc2ifg.ifg_pattern',
    'slc2ifg.cor_pattern',
    'slc2ifg.unw_pattern',
    # coherence rasters are auto-discovered under <work_dir>/ifgrams/*
    'slc2ifg.coh_dir',
    'slc2ifg.coh_pattern',
    'slc2ifg.coh_kind',
    'slc2ifg.coh_variant',
    'slc2ifg.coh_stat',
    'slc2ifg.coh_usable_threshold',
    'slc2ifg.unwrap.coh_dir',
    'slc2ifg.unwrap.coh_pattern',
    'slc2ifg.ifgram_list.select.coh_dir',
    'slc2ifg.ifgram_list.select.coh_kind',
    'slc2ifg.ifgram_list.select.coh_variant',
    'slc2ifg.ifgram_list.select.coh_stat',
    'slc2ifg.ifgram_list.select.coh_usable_threshold',
    # geometry is no longer cropped/multilooked by the pipeline; load_data
    # downsamples a full-resolution geometry file to the interferogram size
    'slc2ifg.geom_dir',
    'slc2ifg.crop_slc.geom_dir',
    'slc2ifg.multilook.geom_dir',
    'slc2ifg.crop_slc.prefix',
    'slc2ifg.crop_slc.by_burst',
    # product paths / verification are fixed
    'slc2ifg.generate_ifgram.output_dir',
    'slc2ifg.generate_ifgram.no_verify',
    # selection: fixed window, measured-coherence-only weights, no floor
    'slc2ifg.ifgram_list.select.quick_window',
    'slc2ifg.ifgram_list.select.weight_source',
    'slc2ifg.ifgram_list.select.model_tau_days',
    'slc2ifg.ifgram_list.select.model_gamma0',
    'slc2ifg.ifgram_list.select.quality_threshold',
    # engine mid-chain entry reads <work_dir>/ifgrams automatically
    'engine.input_dir',
    'engine.input_variant',
    'engine.keep_variants',
    # stitch file types are detected under <work_dir>/ifgrams; bounds follow bbox
    'slc2ifg.stitch.file_types',
    'slc2ifg.stitch.out_bounds',
}


def normalize_config(config):
    """Map legacy keys onto the canonical top-level keys, in place.

    Accepts either a plain ``dict`` or the ``configparser.ConfigParser``
    returned by ``engine.config.read_config``.  The canonical key always wins
    when both are present; a deprecation warning is logged per legacy key.
    """
    section = 'slc2ifg'
    is_parser = hasattr(config, 'has_option')

    def has(key):
        return config.has_option(section, key) if is_parser else key in config

    def get(key):
        if is_parser:
            return config.get(section, key, raw=True)
        return config.get(key)

    def set_(key, value):
        if is_parser:
            config.set(section, key, str(value))
        else:
            config[key] = value

    def remove(key):
        if is_parser:
            if config.has_option(section, key):
                config.remove_option(section, key)
        else:
            config.pop(key, None)

    def is_unset(value):
        # MintPy 'auto'/'none' are placeholders, not real values.  Template
        # values keep their inline comment as part of the string, so only the
        # first token counts.
        import re
        head = re.split(r'[\s#]', str(value).strip().lower(), 1)[0]
        return head in ('', 'auto', 'none', 'null')

    for old, new in LEGACY_ALIASES.items():
        if not has(old):
            continue
        if is_unset(get(old)):
            remove(old)
            continue
        if is_unset(get(new)):
            set_(new, get(old))
        logger.warning("config key '%s' is deprecated, use '%s'", old, new)
        remove(old)

    for key in REMOVED_KEYS:
        if has(key):
            logger.warning("config key '%s' is no longer used and is ignored", key)
            remove(key)

    return config
