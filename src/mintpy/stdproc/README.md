# insarflow.slc2ifg — Core InSAR Processing Workflow

`insarflow.slc2ifg` provides the interferogram generation pipeline steps:
cropping, pair selection, interferogram generation (VRT + materialisation),
multilooking, filtering (Goldstein / long-wavelength), coherence
(complex + phase-sigma), stitching and phase unwrapping (SNAPHU).

Each step is an independent, idempotent script that can be used standalone,
but the **recommended entry point is the engine** (see the package
`README.md` and `docs/engine_design.md`), which schedules these steps as
Dask tasks with CPU/GPU kernels and tiling.

## Fixed product structure

SLCs and interferogram products follow a fixed naming scheme (see
`insarflow.slc2ifg.utils.naming`):

```
slc_dir/yyyymmdd.slc                       (isce2)
slc_dir/yyyymmdd.slc.tif | .slc.h5         (isce3)

intf_dir/{date1}_{date2}/xxx.int[.tif]        wrapped interferogram
intf_dir/{date1}_{date2}/xxx.unw[.tif]        unwrapped phase
intf_dir/{date1}_{date2}/xxx.unw.conncomp[.tif] connected components
intf_dir/{date1}_{date2}/xxx.cpx.coh[.tif]    complex coherence (optional)
intf_dir/{date1}_{date2}/xxx.phsig.coh[.tif]  phase-sigma coherence
```

with variant `xxx ∈ {fullres, mli, filt_mli, filt}`.

## Standalone scripts

```bash
# Coherence-aware connected pair selection (3-NN + half-year/one-year pairs,
# ranked by measured coherence; see docs/ifgram_selection.md)
ifgram_list.py --slc ./slc --mode select \
    --select-min-degree 2 --select-report selection_report.json

# Generate interferograms from a pairs file
generate_ifgram.py --processor isce3 --pairs-file ifgram_list.txt \
    --slc-dir ./slc --output-dir ./ifgrams

# Multilook / filter / phase-sigma coherence / unwrap
multilook.py --processor isce3 --input-dir ./ifgrams --pattern '**/*.int.tif' \
    --output-dir ./ml --lks-y 4 --lks-x 4
filter.py --processor isce3 --filter-type goldstein --input-dir ./ml \
    --pattern '**/*.int.tif' --output-dir ./filtered
generate_coh.py --processor isce3 --input './filtered/**/*.int.tif' \
    --output-dir ./filtered
unwrap_ifgram.py --processor isce3 --ifg-dir ./filtered --cor-dir ./filtered \
    --output-dir ./unwrapped --ifg-pattern '**/*.int.tif' \
    --cor-pattern '**/*.phsig.coh.tif'

# Stitch multi-burst products (file types auto-detected; full union extent
# by default, or clipped to --bbox)
stitch.py --processor isce3 --burst-dir ./ifgrams --output-dir ./stitched
```

All scripts skip already-generated outputs, making interrupted runs safe to
resume.

## Configuration

MintPy-style config files (no section header; a virtual `[slc2ifg]` section
is added internally). The default template lives at
`src/mintpy/stdproc/template/slc2ifg.cfg`; the engine reads the same format
(see `mintpy.stdproc.engine.config`).

`slc2ifg.slc_input` accepts a path **or a glob** (e.g. `xxx/y*/` for a flat
tree, `xxx/t*/*/` for a multi-burst tree — the burst id is extracted
automatically), and the SLC filename pattern is inferred from the files
found. The HDF5 subdataset is auto-detected (`/data/[VV,VH,HH]`, preferring
VV), and coherence rasters are auto-discovered under `<work_dir>/ifgrams`
(kind/variant inferred from the filename).
