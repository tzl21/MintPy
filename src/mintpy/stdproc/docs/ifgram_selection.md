# Coherence-aware interferogram selection (ifgram_list `select` mode)

This document develops the theory behind selecting a *small, high-quality,
connected* interferogram network for SBAS time-series inversion, and
describes the algorithm implemented in
`mintpy.stdproc.select_ifgrams` (engine mode
`slc2ifg.ifgram_list.mode = select`).

> **Looking for the mechanics?** A mode-by-mode deep dive of
> `sequential` / `reference` / `select` (step-by-step algorithms, worked
> examples, every parameter in detail, edge cases, recipes and a decision
> guide) lives in [`ifgram_list_modes.md`](ifgram_list_modes.md). This
> document focuses on the theory and the selection algorithm itself.

## 1. Motivation

A classic k-nearest-neighbour (k-NN) network with large `k` (e.g. `k = 8`)
generates many interferograms. The usual justification is redundancy for
the SBAS least-squares inversion. But when the temporal phase **closure**
is guaranteed (e.g. by tinsaru), redundancy buys almost nothing in terms of
consistency — and costs a lot:

* compute (each interferogram is a full complex multiply + filter +
  unwrap), and
* risk (every additional interferogram is one more chance of an
  unwrapping error).

The right question is therefore not *"how many interferograms?"* but
*"which interferograms?"*: keep the network **connected** (the only hard
constraint for a solvable SBAS system), and fill it with the
**highest-coherence** pairs, which is what this selection does.

## 2. Theory

### 2.1 The SBAS system and the design matrix

Let `T = {t_1 < ... < t_N}` be the sorted acquisition dates and
`G' = (T, E')` the graph of selected interferograms. Each interferogram
gives an observation

    psi_ij = phi_j - phi_i + eps_ij ,

where `phi` is the per-date phase (relative to a reference date). Stacking
the `M = |E'|` equations gives

    psi = H phi + eps,

with `H in R^{M x N}` the **oriented incidence matrix** of `G'` (row
`(i,j)` has `-1` on the master column and `+1` on the slave column).
Dropping the reference-date column (`t_ref = t_1` in this pipeline) gives
`H_{-ref} in R^{M x (N-1)}` and the normal equations `(H^T H) x = H^T psi`.

### 2.2 Rank <=> connectivity (the only hard constraint)

By the incidence-matrix rank theorem,

    rank(H) = N - c(G'),

where `c(G')` is the number of connected components. Removing the
reference column leaves the rank unchanged when `t_ref` is the only node
in its component of the *full* graph — in particular, for a connected
`G'`, `rank(H_{-ref}) = N - 1`, i.e. **`H^T H` is invertible iff the
interferogram network is connected**. Connectivity is therefore
*necessary and sufficient* for a unique SBAS solution; nothing else about
the network shape matters for solvability.

### 2.3 Closure => zero residual, whatever the edge count

tinsaru guarantees temporal phase closure: every triangle is
cycle-consistent, so the wrapped phases satisfy
`psi_ij + psi_jk + psi_ki = 0 (mod 2 pi)`. After unwrapping, the
consistent phases satisfy `psi = H phi` *exactly* for some `phi`.
Consequently, for **any connected** `G'`:

* the least-squares residual `|| H x - psi ||` is **zero** — redundant
  interferograms never "fix" anything (verified in
  `tests/test_select_ifgrams.py::test_zero_residual_closure`);
* the estimate is `x = (H^T W H)^{-1} H^T W psi` with covariance
  `Cov[x] = (H^T W H)^{-1}`, `W = diag(w_ij)` the information weights.

Redundancy beyond a spanning tree only provides:

1. **noise reduction** — each extra edge contributes information
   `w_ij ~ gamma_ij^2 / (1 - gamma_ij^2)` (per look; Touzi 1999 noise
   model `sigma^2 ~ (1 - gamma^2)/(2 L gamma^2)`), so a *high-coherence*
   edge is worth many low-coherence ones. By the Sherman–Morrison
   formula, adding an edge with weight `w` strictly reduces the trace of
   the estimate covariance by
   `w |(H^T W H)^{-1} h|^2 / (1 + w h^T (H^T W H)^{-1} h) > 0`
   (A-optimality: tested in
   `test_added_edges_reduce_estimate_covariance`);
2. **robustness** — if one interferogram fails to unwrap, a pure spanning
   tree splits into two components and the whole inversion becomes
   rank-deficient; a denser network (minimum degree >= 2, or bridge
   repair) survives single-edge failures.

### 2.4 Consequence: the near-optimal selection

Since (a) solvability needs only connectivity and (b) the value of every
extra edge is proportional to its coherence, the near-optimal network is:

> **span with the highest-quality edges, then augment greedily with the
> highest-quality remaining edges** — subject to a per-date minimum
> degree, an edge budget, and a quality floor.

The maximum-weight spanning tree maximises the total tree weight among
all spanning trees (Kruskal's greedy algorithm is exact for this), and
the greedy augmentation is the standard, near-optimal heuristic for the
underlying optimal-experimental-design problem (maximising information
per unit cost).

## 3. The algorithm

Implemented in `select_ifgrams.select_ifgrams`; orchestrated by
`select_ifgrams.select_pairs` (also exposed as
`ifgram_list` mode `select`).

```
1. generate_candidates(dates, ...)
       k temporal nearest neighbours          (e.g. k=3)
     + window pairs                          (e.g. (182,10), (365,15): half-year + one-year)
     + all pairs within temp_baseline_max    (optional; "fully connected within a window")
     - perpendicular-baseline cap            (optional)
     => candidate edge set E

2. weight every candidate  w_ij in [0,1] (or >= 0 for 'fisher')
       model       w = gamma0 * exp(-dt/tau)          [temporal decorrelation model]
       coherence   measured: existing coherence rasters, or
                   quick complex coherence on downsampled SLCs
       mixed       measured where available, model fills the gaps

3. select_ifgrams(E, w)
       a. maximum spanning tree (Kruskal, descending w)   => connected, N-1 edges
       b. greedy min-degree augmentation                  => every date in >= min_degree ifgs
       c. budget fill with the highest-weight edges       (max_pairs)
       d. optional bridge repair                          (2-edge robustness where possible)

4. verify_selection
       connectivity check (union-find)  +  rank(H_{-ref}) == N-1
       -> guarantee reported in the log and the JSON report
```

### 3.1 Candidate generation — both of your ideas are instances

* **"3-NN + half-year/one-year pairs"** — `num_connections = 3`,
  `annual_windows = auto` (the default: windows derived from the
  dominant repeat cycle; `182:10,365:15` is the equivalent explicit
  spec on a 12-day grid). The k-NN skeleton keeps short
  temporal baselines (low decorrelation, easy unwrapping); the window
  pairs carry the long-term deformation signal that a k-NN network
  misses. This is the recommended default.
* **"fully connected on a small window / downsampled SLCs"** —
  `num_connections = 0`, `annual_windows = none`, `temp_baseline_max = N`
  (or any window), then let `weight_source = coherence` with the
  on-the-fly quick coherence: every candidate is screened on a grid
  sample of the SLCs (`quick_grid`×`quick_grid` of `quick_block`×
  `quick_block` windows), which costs milliseconds per pair, and the
  selection keeps only the high-coherence ones. For large `N`, prefer
  capping `temp_baseline_max` so the `N(N-1)/2` candidate set stays
  manageable.

### 3.2 Quality weights

| source | formula / statistic | when to use |
|---|---|---|
| `model` | `gamma0 * exp(-dt/tau)` | no SLCs/coherence handy; cold start; stable scenes |
| `coherence` (rasters) | mean / median / usable-fraction / Fisher info of an existing `coh` map | reuse a pilot run's coherence products |
| `coherence` (quick) | same statistics of complex coherence computed on **downsampled** SLCs (Touzi debias) | the standard choice; SLCs are already available |
| `mixed` | measured where present, model fills gaps | robustness against missing rasters |

`usable_frac` (fraction of pixels with coherence >= a threshold) is a
good proxy for the *unwrappable area*; `fisher`
(`gamma^2/(1-gamma^2)`) is the information-theoretic weight matching
section 2.3 — both are reasonable drop-in alternatives to the plain mean.

### 3.3 Selection guarantees

* **Connectivity / full rank** — hard guarantee. If the *candidate* graph
  is disconnected, `select_ifgrams` raises `ValueError` instead of
  silently producing a singular SBAS system.
* **Quality priority** — tree edges form the maximum-weight spanning tree
  (verified by brute force in the tests); augmentation proceeds strictly
  by descending weight.
* **`min_degree >= 2`** — every date participates in at least 2
  interferograms (no single-ifg leaf; resilience to one failed edge).
  `min_degree = 1` gives the pure maximum-weight spanning tree
  (`N-1` pairs — the absolute minimum).
* **`robust = true`** — replaces every bridge by the highest-weight
  candidate crossing its cut, where such a candidate exists (2-edge
  robustness).
* **`quality_threshold`** — augmentation floor; the spanning tree may
  still use below-threshold edges, because connectivity is a hard
  guarantee (a warning is logged).
* **Determinism** — ties in weight are broken lexicographically, so the
  output is reproducible.

### 3.4 Verification

`verify_selection` re-checks both conditions on the *final* selection and
reports `connected`, `rank` and `full_rank`. With
`slc2ifg.ifgram_list.select.report` set, the full report (counts, weight
sums, degree stats, verification results, parameter snapshot) is written
as JSON. The engine logs a one-line summary during graph building.

## 4. Usage

### 4.1 Engine (recommended)

```ini
slc2ifg.ifgram_list.mode = select

# 3-NN skeleton + half-year / one-year pairs, ranked by measured
# coherence computed on downsampled SLCs (no extra products needed)
slc2ifg.ifgram_list.select.weight_source = coherence
slc2ifg.ifgram_list.select.min_degree = 2
slc2ifg.ifgram_list.select.robust = true
slc2ifg.ifgram_list.select.report = ifgram_selection.json

# optional tuning
# slc2ifg.ifgram_list.num_connections = 3
# slc2ifg.ifgram_list.select.annual_windows = 182:10,365:15   # or auto (default)
# slc2ifg.ifgram_list.select.temp_baseline_max = 60
# slc2ifg.ifgram_list.select.quick_window = 5
# slc2ifg.ifgram_list.select.max_pairs = 300
# slc2ifg.ifgram_list.select.quality_threshold = 0.3
```

The selection runs eagerly during graph construction (the graph topology
depends on it), the same way `sequential` mode does today.

### 4.2 Standalone CLI

```bash
python mintpy.stdproc.ifgram_list --slc ./slc --mode select \
    --select-weight-source model --select-min-degree 2 \
    --select-report report.json
```

### 4.3 Weight sources in detail

* `model` — needs nothing but the date list.
* `coherence` with `select.coh_dir` — reads existing
  `{coh_dir}/{date1}_{date2}/{variant}_{kind}.coh[.tif]` rasters
  (e.g. `filt_mli.phsig.coh.tif` produced by a pilot run); missing
  rasters fall back to weight 0 (or to the model in `mixed` mode).
* `coherence` without `coh_dir` — on-the-fly quick coherence from the
  SLC directory (block-average downsample, complex correlation, Touzi
  debias, robust statistic).
* `mixed` — measured where available, `model` fills gaps.

## 5. Practical recommendations

* **Default recipe** (C-band, Sentinel-1-like, 12-day cycle):
  `num_connections = 3`, `annual_windows = auto` (windows derived from
  the repeat cycle; `182:10,365:15` on a 12-day grid),
  `weight_source = coherence` (quick), `min_degree = 2`,
  `robust = true`. This typically replaces a k=5-8 NN network with
  roughly half to two-thirds the interferograms at higher average
  coherence.
* **Very long series / low-coherence scenes**: raise `min_degree`
  to 3-4 or lower `model_tau_days` so the model prefers short
  baselines, and let `quality_threshold` cut the tail.
* **Robustness to unwrap failures**: `min_degree >= 2` + `robust = true`
  is the cheap insurance; full 2-edge-connectivity is guaranteed only
  where the candidate set provides crossing edges.
* **The SBAS solve itself is unchanged**: feed the resulting
  `ifgram_list.txt` to the standard inversion. Thanks to closure the
  residual stays zero; the benefit shows up as lower variance and fewer
  failed interferograms.

## 6. References

* Touzi, R., Lopes, A., Bruniquel, J., Vachon, P.W. (1999). *Coherence
  estimation for SAR imagery*. IEEE TGRS 37(1).
  (magnitude-bias correction used by the quick-coherence path)
* Berardino, P., Fornaro, G., Lanari, R., Sansosti, E. (2002). *A new
  algorithm for surface deformation monitoring based on small baseline
  differential SAR interferograms*. IEEE TGRS 40(11).
* Perissin, D., Wang, T. (2012). *Repeat-pass SAR interferometry with
  partially coherent targets*. IEEE TGRS 50(1). (coherence-weighted
  network selection ideas)
