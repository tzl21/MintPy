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
intf_dir/{date1}_{date2}/xxx.conncomp[.tif]   connected components
intf_dir/{date1}_{date2}/xxx_cpx.coh[.tif]    complex coherence (optional)
intf_dir/{date1}_{date2}/xxx_phsig.coh[.tif]  phase-sigma coherence
```

with variant `xxx ∈ {fullres, mli, filt_mli, filt}`.

## Standalone scripts

```bash
# Coherence-aware connected pair selection (3-NN + half-year/one-year pairs,
# ranked by quick coherence on downsampled SLCs; see docs/ifgram_selection.md)
ifgram_list.py --slc ./slc --mode select --select-weight-source coherence \
    --select-min-degree 2 --select-report selection_report.json

# Generate interferograms from a pairs file
generate_ifgram.py --processor isce3 --pairs-file ifgram_list.txt \
    --slc-dir ./slc --output-dir ./ifgrams

# Multilook / filter / phase-sigma coherence / unwrap
multilook.py --processor isce3 --input-dir ./ifgrams --pattern '**/*.int.tif' \
    --output-dir ./ml --lks-y 4 --lks-x 4
filter.py --processor isce3 --filter-type goldstein --input-dir ./ml \
    --pattern '**/*.int.tif' --output-dir ./filtered
generate_coh_phsig.py --processor isce3 --input './filtered/**/*.int.tif' \
    --output-dir ./filtered
unwrap_ifgram.py --processor isce3 --ifg-dir ./filtered --cor-dir ./filtered \
    --output-dir ./unwrapped --ifg-pattern '**/*.int.tif' \
    --cor-pattern '**/*_phsig.coh.tif'

# Stitch multi-burst products (full union extent by default)
stitch.py --processor isce3 --burst-dir ./ifgrams --output-dir ./stitched
```

All scripts skip already-generated outputs, making interrupted runs safe to
resume.

## Configuration

MintPy-style config files (no section header; a virtual `[slc2ifg]` section
is added internally). The default template lives at
`src/insarflow/slc2ifg/template/slc2ifg.cfg`. The engine reads the same
config format (see `insarflow.engine.config`).
