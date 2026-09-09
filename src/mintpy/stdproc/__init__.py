"""slc2ifg processing modules under stdproc (lazy star-export).

The heavy GDAL-dependent modules (crop/filter/coh/multilook/stitch/unwrap)
are imported **lazily** on attribute access: importing ``mintpy.stdproc``
itself must not require osgeo, because the engine's config/chain layers and
core MintPy modules (``mintpy/multilook.py`` etc.) are imported in
environments without GDAL (unit tests, plugin contexts).

Named imports keep working unchanged::

    from mintpy.stdproc import multilook_tif   # -> lazy load
    import mintpy.stdproc.ifgram_list          # -> submodule import
"""

import importlib

_LAZY_SUBMODULES = (
    'crop_slc_geo',
    'crop_slc_rdr',
    'filter',
    'generate_coh',
    'generate_ifgram',
    'multilook',
    'stitch',
    'unwrap_ifgram',
)
_loaded: dict = {}


def __getattr__(name: str):
    for mod_name in _LAZY_SUBMODULES:
        mod = _loaded.get(mod_name)
        if mod is None:
            try:
                mod = importlib.import_module(f'.{mod_name}', __name__)
            except ImportError:
                mod = False
            _loaded[mod_name] = mod
        if mod and hasattr(mod, name):
            return getattr(mod, name)
    raise AttributeError(
        f"module 'mintpy.stdproc' has no attribute {name!r}")
