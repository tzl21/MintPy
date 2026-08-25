# ifgram_list pair-generation modes in detail

This document explains, mode by mode, how `ifgram_list` turns the
acquisition-date list into the interferogram pair list that drives the
whole pipeline (interferogram generation → unwrapping → SBAS inversion).

Three modes exist, selected by `slc2ifg.ifgram_list.mode`
(or `--mode` in the standalone CLI):

| mode | name | edges | quality-aware | long baselines | when to use |
|---|---|---|---|---|---|
| `sequential` | k temporal nearest neighbours | ≈ k·N | no | only via `annual_windows` (one-year = `365:<range>`) | simple, dense short-baseline network |
| `reference` | star network (single master) | N−1 | no | long for late dates | tiny stacks / stable targets / pilot runs |
| `select` | coherence-aware connected selection | tunable (default ≈ 1.5·N) | **yes** | yes (annual windows) | the recommended default for production SBAS |

The theory behind `select` (why connectivity is the only hard constraint,
why coherence is the right objective) is in
`docs/ifgram_selection.md`; this document is about the mechanics of each
mode.

---

## 1. `sequential` — k temporal nearest neighbours

### 1.1 Algorithm

For each date `t_i`, pair it with the next `k` dates:

    for i in 0 .. N-2:
        for j in i+1 .. min(i+k, N-1):
            emit (t_i, t_j)

The network is the **k-th power of the temporal path graph**.

### 1.2 Worked example

`N = 12` dates on a 12-day repeat cycle, `k = 5`:

* pairs = `Σ_i min(k, N−1−i)` = `5·7 + 4 + 3 + 2 + 1` = **45 pairs**;
* every pair has a temporal baseline of `1..5` repeat cycles = **12–60 days**;
* the graph is connected for any `k ≥ 1` (the `(t_i, t_{i+1})` chain is
  always present), so the SBAS design matrix always has full rank.

### 1.3 Properties

* **Short baselines only** — high SNR, easy unwrapping, but the
  long-term (seasonal / multi-year) deformation signal is only weakly
  constrained: it must be reconstructed by chaining many short
  interferograms.
* **Quality-blind** — a low-coherence pair (e.g. a summer pair over a
  vegetated area) is generated with the same probability as a
  high-coherence one; with large `k` the network is dominated by
  mediocre pairs that mostly add unwrap risk.
* **Redundancy ≈ k** — every date participates in up to `2k` pairs.
* **Deterministic** and trivially reproducible.

### 1.4 Parameters

| key | default | effect |
|---|---|---|
| `num_connections` | `5` | `k`. `k ≥ N−1` degenerates to the complete graph. |
| `annual_windows` | `auto` | additionally emit every pair whose baseline is within any `center:tol` window (e.g. one-year = `365:<range>`). Canonical knob for annual/one-year pairs in every mode (the legacy `oneyear_interferograms` is a deprecated alias). |
| `start_date` / `end_date` | — | dates outside `[start, end]` are dropped *before* pairing (also filters the crop file list). |

### 1.5 When to use

Quick pilots, small stacks, or as the baseline you want to replace:
`sequential` is what the old k=5/k=8 recipes generate (285 / 444 pairs
for a 60-date stack).

---

## 2. `reference` — star network

### 2.1 Algorithm

Pick the earliest date as the reference (master) and pair it with every
other date:

    ref = min(dates)
    emit (ref, t_i)  for every t_i != ref

### 2.2 Worked example

`N = 12` dates → **11 pairs**, all sharing the earliest master.

### 2.3 Properties

* **Minimal network** — exactly `N−1` edges (a spanning tree), the
  theoretical minimum for solvability.
* **Single-ifg phases** — every slave date appears in exactly one
  interferogram, so its phase is the noisiest possible estimate, and a
  single failed unwrap splits the star (every edge is a bridge).
* **Long baselines for late dates** — the last date is `(N−1)·12` days
  from the master (e.g. 132 days for the example), where temporal
  decorrelation is usually severe — unless the targets are very stable
  (PS-like).

### 2.4 Parameters

Same `start_date` / `end_date` / `annual_windows` as
`sequential` (`num_connections` is ignored).

### 2.5 When to use

Tiny stacks, extremely stable scatterers, or a deliberately minimal
network where you accept the noise for the compute saving. Not
recommended for general SBAS — this is the motivation for `select`.

---

## 3. `select` — coherence-aware connected selection

The full pipeline has five steps; each is controllable through
`slc2ifg.ifgram_list.select.*` keys (standalone CLI: `--select-*`).

### 3.1 Pipeline overview

```
dates ──► ① candidate generation ──► ② weighting ──► ③ selection ──► ④ verification ──► pair list
                  │                        │              │                │
                  │                        │              │                └─ JSON report (optional)
                  ▼                        ▼              ▼
          k-NN + windows +          model /          max spanning tree
          temp-baseline cap        coherence /       + min-degree
          (+ perp filter)          mixed             + budget / threshold
                                                     + bridge repair
```

### 3.2 Step ① — candidate generation (`generate_candidates`)

The candidate set is the **union** of up to three rule families, then
deduplicated and sorted (`d1 < d2`):

1. **k-NN skeleton** — `num_connections` (default **3**): the same rule
   as `sequential` with a smaller `k`. Keeps short-baseline, high-SNR
   pairs and guarantees the candidate graph is connected (`k ≥ 1`).
2. **Window pairs** — `annual_windows` (default **`auto`**): with `auto`,
   the half-year / one-year windows are **derived from the dominant
   repeat cycle** of the date list (centers snapped to the multiples of
   the repeat interval nearest 182.5 / 365.25 days, tolerance ± one
   repeat interval), so they stay valid for irregular sampling, missing
   acquisitions and leap years.  An explicit spec overrides the
   derivation: every pair whose baseline `Δt` satisfies
   `|Δt − center| ≤ tol` for any `(center, tol)` entry — e.g.
   `annual_windows = 182:10,365:15` adds **half-year pairs**
   (`Δt ∈ [172, 192] d`) and **one-year pairs** (`Δt ∈ [350, 380] d`).
   These are the "long-baseline" interferograms that constrain
   seasonal / long-term deformation and are missing from a k-NN
   network. On a 12-day cycle a one-year pair is 360 or 372 days, both
   inside the 365±15 window. `none` disables the rule.
3. **Temporal-baseline-limited full connection** — `temp_baseline_max`
   (default off): *all* pairs with `Δt ≤ limit`. This is the
   "fully connected within a window" flavour: with a small AOI / small
   `N` it is affordable and gives the selector maximum freedom.

Optional **perpendicular-baseline filter** — `perp_baseline_max` with a
`perp_baseline_file` (two-column `date bperp` text file): drops
candidates with `|B⊥_i − B⊥_j| > limit`. Applied to every rule above.

**Why these three?** k-NN guarantees connectivity and supplies
short-baseline SNR; window pairs supply the long-term signal; the
baseline cap supplies freedom to pick the best subset. Tuning rule 1/2
up and rule 3 down gives the "sparse skeleton" recipe; rule 3 alone
(others off) gives the "full screening" recipe (see §3.6).

Example (60 dates, 12-day cycle, defaults): 174 k-NN pairs + ~148
window pairs = **322 candidates** (a 60-date stack has 1770 possible
pairs, so this already cuts the search space ~5.5×).

### 3.3 Step ② — weighting (`weight_source`)

Every candidate gets a scalar quality `w` (larger = better):

| source | formula / statistic | prerequisites |
|---|---|---|
| `model` (default) | `w = γ0 · exp(−Δt/τ)` (+ optional perp factor) | date list only |
| `coherence` + `coh_dir` | statistic of an existing coherence raster | a pilot run's products |
| `coherence` (no `coh_dir`) | complex coherence on **downsampled** SLCs | SLC directory |
| `mixed` | measured where available, `model` fills gaps | SLCs or rasters + model |

**`model` details** — `model_tau_days` (default **90**) is the
temporal-decorrelation time constant, `model_gamma0` (default **1.0**)
the coherence at zero baseline. Example weights with τ = 90 d:
`Δt = 12 d → 0.875`, `Δt = 24 d → 0.766`, `Δt = 360 d → 0.018`. The
model therefore strongly prefers short baselines — good for a cold
start, but it cannot *know* that a particular annual pair is actually
coherent; that requires the measured sources.

**`coherence` raster details** — looks up
`{coh_dir}/{date1}_{date2}/{variant}_{kind}.coh[.tif]`, e.g.
`filt_mli_phsig.coh.tif` (`coh_variant` default `filt_mli`,
`coh_kind` default `phsig` — the products of the standard engine chain).
A missing raster yields weight 0 in `coherence` mode, or a model
fallback in `mixed` mode (warned in the log).

**`coherence` quick (on-the-fly) details** — the "screening" source:
each SLC is read as a regular `quick_grid`×`quick_grid` grid of
`quick_block`×`quick_block` sample windows (0.04% of the pixels, no
whole-image read), the complex correlation magnitude is computed
with a `quick_window`×`quick_window` boxcar (default 5), corrected for
the magnitude bias (`quick_debias`, default on; Touzi 1999 with
`L = (factor·window)²` effective looks:
`γ ≈ (γ̂ − 1/(2L)) / (1 − 1/(2L))`), and aggregated to one scalar. Cost:
milliseconds per pair — cheap enough to screen *all* candidates;
`quick_max_workers > 1` parallelises the per-pair screening with a
thread pool (the downsampled SLCs are loaded once up front).

**Aggregation statistic** (`coh_stat` / `quick_stat`, default `mean`):

| stat | definition | meaning |
|---|---|---|
| `mean` | mean over valid (γ > 0) pixels | average coherence; sensitive to large incoherent areas |
| `median` | median over valid pixels | robust to water/shadow outliers |
| `usable_frac` | fraction of pixels with γ ≥ `coh_usable_threshold` (default 0.3) | proxy for the **unwrappable area**; good for sparse AOIs |
| `fisher` | mean of `γ²/(1−γ²)` | the information weight from the theory; unbounded (≥ 0), so `quality_threshold` is in fisher units |

### 3.4 Step ③ — selection (`select_ifgrams`)

Candidates are ordered by **descending `w`** (ties broken
lexicographically — deterministic), then processed in four phases:

**Phase A — maximum spanning tree** (always): Kruskal's algorithm adds
edges in the above order whenever they join two components. Result:
exactly `N−1` edges, guaranteed connectivity, and — among *all* spanning
trees — the maximum total weight (this is exact, not a heuristic). If
the candidate graph is itself disconnected, a `ValueError` is raised
with hints (relax `temp_baseline_max`, raise `num_connections`).

**Phase B — max-mean augmentation** (`min_degree`, default **2**):
continuing down the order, an edge is added whenever it either lifts one
of its endpoints to the `min_degree` target, or its weight is **at or
above the current running mean** (adding it can never lower the average
coherence).  The running mean is refreshed after each sweep and the
sweep repeats to a fixed point.  The objective is the *mean* (average)
coherence, not the sum: the final network keeps every edge at or above
the converged mean plus only the few below-mean edges forced by the
connectivity / `min_degree` lower bound.  Why `min_degree`: tree leaves
(degree 1) estimate their phase from a single interferogram — the
noisiest possible — and a single unwrap failure disconnects them.
`min_degree = 2` gives every date ≥ 2 interferograms. `min_degree = 1`
disables this phase (pure spanning tree, `N−1` pairs — the absolute
minimum). If the target cannot be met (candidates too sparse or below
the quality floor), the unmet dates are warned and listed in the report.

**Phase C — budget fill** (`max_pairs`): if the current count is below
`max_pairs`, keep adding the highest-weight remaining edges (still
above the quality floor) until the budget is reached. This is the knob
for hard compute limits (e.g. total unwrap time).

**Phase D — bridge repair** (`robust`, default off): find all bridges
(cut edges — single points of failure) with Tarjan's algorithm; for
each, add the highest-weight *unselected* candidate crossing its cut,
iterating until no bridge remains or no replacement exists. Result:
2-edge robustness wherever the candidate set allows. Note that even
with `min_degree = 2` a degree-2 node can still sit on a bridge; this
phase removes that.

**`quality_threshold`** (default 0): a floor applied to *augmentation*
(phases B, C, D). The spanning tree (phase A) may still use
below-threshold edges, because connectivity is a hard guarantee — a
warning is logged when that happens.

### 3.5 Step ④ — verification, report and diagram

* Connectivity is re-checked on the final selection (union-find).
* If `verify` (default true) and `N ≤ 2000`, the SBAS design matrix
  `H_{-ref}` is built and `rank == N−1` is asserted (full rank ⇔
  connected).
* With `select.report` set, a JSON report is written: counts
  (`n_candidates`, `n_selected`, `n_tree_edges`, `n_bridge_repairs`),
  weight statistics (sum, mean, min/median), degree statistics
  (`min_degree_actual`, `max_degree_actual`, `degree_unmet`),
  verification (`connected`, `rank`, `full_rank`), the tree edge list
  (`tree_edges`, for the diagram), `weight_source` and a full parameter
  snapshot. The engine also logs a one-line summary during graph
  building.
* With `select.dot` set, the selected network is written as GraphViz
  DOT: dates laid out left-to-right in time order, spanning-tree edges
  bold, augmentation edges solid, quality weights as edge labels
  (`dot -Tpng ifgram_network.dot -o ifgram_network.png` to render).

### 3.6 Two recommended recipes

**Recipe 1 — "3-NN + half-year/one-year pairs"** (replaces a k=5/8 NN):

```ini
slc2ifg.ifgram_list.mode = select
slc2ifg.ifgram_list.select.weight_source = coherence   # quick coherence on downsampled SLCs
slc2ifg.ifgram_list.select.min_degree = 2
slc2ifg.ifgram_list.select.robust = true
# defaults already provide: num_connections=3, annual_windows=auto
```

On a 60-date stack this yields ≈ 89 pairs (vs 285 for k=5) at higher
average coherence, full rank guaranteed.

**Recipe 2 — "full screening on a small window"** (your
fully-connected-on-downsampled-SLCs idea):

```ini
slc2ifg.ifgram_list.mode = select
slc2ifg.ifgram_list.select.weight_source = coherence
slc2ifg.ifgram_list.select.num_connections = 0        # no k-NN skeleton
slc2ifg.ifgram_list.select.annual_windows = none      # no window pairs
slc2ifg.ifgram_list.select.temp_baseline_max = 48     # all pairs <= 48 days
slc2ifg.ifgram_list.select.min_degree = 2
```

Every candidate inside the 48-day window is screened by quick coherence
and only the coherent, connectivity-required subset survives.
(`annual_windows = none`, an empty value, `off` or `0` all disable the
window-pair rule.)

### 3.7 Full parameter reference

| key | default | effect |
|---|---|---|
| `num_connections` | `3` | k-NN skeleton size (0 = off). |
| `annual_windows` | `auto` | `auto` derives half-year / one-year windows from the repeat cycle; explicit `center:tol` day-pair list (comma-separated) overrides; `none`/`off`/`0`/empty disables the rule. |
| `temp_baseline_max` | — | include all pairs with `Δt ≤` this many days. |
| `perp_baseline_max` / `perp_baseline_file` | — | perpendicular-baseline cap (m) + `date bperp` file. |
| `weight_source` | `model` | `model` / `coherence` / `mixed`. |
| `model_tau_days` / `model_gamma0` | `90` / `1.0` | temporal-decorrelation model. |
| `coh_dir` / `coh_kind` / `coh_variant` | — / `phsig` / `filt_mli` | existing-coherence source. |
| `coh_stat` / `coh_usable_threshold` | `mean` / `0.3` | raster aggregation. |
| `quick_window` / `quick_debias` | `5` / `true` | on-the-fly coherence boxcar / Touzi bias correction. |
| `quick_grid` / `quick_block` | `12` / `16` | on-the-fly grid sampling density. |
| `quick_max_workers` | `1` | threads for the per-pair quick-coherence screening. |
| `quick_stat` / `quick_usable_threshold` | `mean` / `0.3` | quick aggregation. |
| `min_degree` | `2` | minimum interferograms per date (0/1 = spanning tree only). |
| `max_pairs` | — | global edge budget. |
| `quality_threshold` | `0` | augmentation floor on `w`. |
| `robust` | `false` | bridge repair (2-edge robustness). |
| `verify` | `true` | rank/connectivity verification. |
| `report` | — | JSON report path. |
| `dot` | — | GraphViz DOT path (network diagram). |

### 3.8 Edge cases and failure modes

* `N < 2` → `ValueError` ("at least 2 acquisition dates").
* `N = 2` → the single pair, if present in the candidates.
* Candidate graph disconnected → `ValueError` (no connected subset
  exists) with relaxation hints.
* Candidate count `< N−1` → `ValueError` (not enough edges to span).
* Missing SLC file / coherence raster → weight `None` → treated as 0
  (or model fallback in `mixed`), warned in the log.
* Unknown date in a pair → `ValueError`.
* High `quality_threshold` → tree still guarantees connectivity
  (below-threshold edges allowed with a warning); `min_degree` may be
  left unmet (warned + listed in the report).
* Reproducibility: the sort is `(−w, d1, d2)`, so equal weights give a
  deterministic outcome.

---

## 4. Choosing a mode — decision guide

| situation | recommendation |
|---|---|
| production SBAS, SLCs available | `select`, `weight_source = coherence` (quick), `min_degree = 2`, `robust = true` |
| no SLCs yet / cold start / config test | `select`, `weight_source = model` |
| tiny stack (N ≤ ~15), quick pilot | `sequential` k=3–5, or `reference` if targets are very stable |
| reuse a previous run's coherence | `select`, `weight_source = coherence`, `coh_dir = <previous ifgrams>` |
| strict compute budget (unwrap time) | `select` + `max_pairs` |
| exactly reproduce an old k-NN network | `sequential` |
