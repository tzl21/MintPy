#!/usr/bin/env python3
"""Canonical slc2ifg configuration keys and their legacy aliases.

Functional duplicates used to live under several per-tool namespaces
(``crop_slc.wsen`` vs ``bbox``, ``generate_coh.ps_nlks`` vs ``nlooks``, ...).
The canonical key is the SINGLE top-level ``slc2ifg.<name>`` (e.g.
``slc2ifg.bbox``, ``slc2ifg.unwrap.mask_file``); the old keys keep working for
one deprecation cycle through :func:`normalize_config`, which is applied once at
every entry point that reads a raw config dict.

Keys that no longer exist are simply not read by the code: they may stay in a
user config without any effect (and without an error).
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

    return config
