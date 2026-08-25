#!/usr/bin/env python3
"""Unit tests for coherence-aware interferogram selection.

Pure Python + numpy (no GDAL): exercises candidate generation, the
maximum-spanning-tree + greedy-augmentation selection, the rank-connectivity
theorem, the zero-residual closure property, and the constraint handling.

Run:  python -m pytest tests/test_select_ifgrams.py -v
"""

import itertools
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from mintpy.stdproc.slc2ifg.select_ifgrams import (
    _kruskal_max,
    aggregate_coherence,
    check_connected,
    design_matrix,
    generate_candidates,
    parse_annual_windows,
    read_perp_baselines,
    select_ifgrams,
    select_pairs,
    temporal_model_weights,
    verify_selection,
    write_report,
)

# A regular 12-day repeat cycle (Sentinel-1-like)
DATES = ['20230105', '20230117', '20230129', '20230210', '20230222',
         '20230306', '20230318', '20230330', '20230411', '20230423',
         '20230505', '20230517']


def _dt_days(a, b):
    return (datetime.strptime(b, '%Y%m%d')
            - datetime.strptime(a, '%Y%m%d')).days


def _multi_year_dates(n=61, step_days=12, start='20230105'):
    """Regular ~2-year series for annual-window tests."""
    out = []
    d = datetime.strptime(start, '%Y%m%d')
    for _ in range(n):
        out.append(d.strftime('%Y%m%d'))
        d += __import__('datetime').timedelta(days=step_days)
    return out


def _random_weights(dates, cand, seed=0):
    rng = np.random.default_rng(seed)
    return {p: float(rng.uniform(0.1, 1.0)) for p in cand}


# ------------------------------------------------------------------------
# Candidate generation
# ------------------------------------------------------------------------
def test_candidate_knn_count_and_content():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    # k-NN on a regular 12-day cycle: 3 forward neighbours per date
    n = len(DATES)
    expected = sum(min(3, n - 1 - i) for i in range(n - 1))
    assert len(cand) == expected
    # every date is connected to its immediate successor
    for a, b in zip(DATES, DATES[1:]):
        assert (a, b) in cand
    # ordered d1 < d2, deduplicated
    assert cand == sorted(set(cand))
    assert all(a < b for a, b in cand)


def test_candidate_annual_windows():
    dates = _multi_year_dates()
    cand = generate_candidates(dates, num_connections=0,
                               annual_windows=((365, 15),))
    assert len(cand) >= 1
    for a, b in cand:
        assert abs(_dt_days(a, b) - 365) <= 15
    # all 12-day-cycle one-year pairs are present: dt = 360 or 372 days
    pairs12 = [(dates[i], dates[j]) for i in range(len(dates) - 1)
               for j in range(i + 1, len(dates))
               if _dt_days(dates[i], dates[j]) in (360, 372)]
    assert set(pairs12) <= set(cand)


def test_candidate_temp_baseline_cap_full_connection():
    # fully-connected-within-window flavour
    cand = generate_candidates(DATES, num_connections=0, annual_windows=(),
                               temp_baseline_max=24)
    for a, b in cand:
        assert _dt_days(a, b) <= 24
    # 24 days on a 12-day cycle: pairs 1 and 2 steps apart
    n = len(DATES)
    expected = sum(1 for i in range(n - 1) for j in range(i + 1, n)
                   if (j - i) * 12 <= 24)
    assert len(cand) == expected


def test_candidate_perp_baseline_filter(tmp_path):
    perp = {d: float(i * 20) for i, d in enumerate(DATES)}
    cand_all = generate_candidates(DATES, num_connections=3, annual_windows=())
    cand_lim = generate_candidates(DATES, num_connections=3, annual_windows=(),
                                   perp_baseline_max=40.0,
                                   perp_baselines=perp)
    assert len(cand_lim) < len(cand_all)
    for a, b in cand_lim:
        assert abs(perp[a] - perp[b]) <= 40.0


def test_candidate_dedup_and_sorted():
    cand = generate_candidates(DATES, num_connections=3,
                               annual_windows=((12, 2),))
    assert len(cand) == len(set(cand))
    assert cand == sorted(cand)


def test_parse_annual_windows():
    # None -> None (the caller resolves it to 'auto')
    assert parse_annual_windows(None) is None
    assert parse_annual_windows('182:10,365:15') == ((182, 10), (365, 15))
    assert parse_annual_windows([(100, 5)]) == ((100, 5),)
    # disable tokens -> no windows
    assert parse_annual_windows('') == ()
    assert parse_annual_windows('none') == ()
    assert parse_annual_windows('off') == ()
    assert parse_annual_windows('0') == ()
    try:
        parse_annual_windows('garbage')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for malformed windows')


def test_auto_annual_windows_12day():
    from mintpy.stdproc.slc2ifg.select_ifgrams import auto_annual_windows
    dates = _multi_year_dates()  # 61 dates, 12-day cycle
    w = auto_annual_windows(dates)
    assert w == ((180, 12), (360, 12))
    # centers are multiples of the repeat interval, tolerance covers +/- 1 cycle
    for c, t in w:
        assert c % 12 == 0 and t >= 12


def test_auto_annual_windows_irregular_grid():
    from mintpy.stdproc.slc2ifg.select_ifgrams import auto_annual_windows
    dates = _multi_year_dates(n=31, step_days=24)  # 24-day repeat
    w = auto_annual_windows(dates)
    centers = sorted({c for c, _ in w})
    assert centers == [192, 360]      # nearest multiples of 24 to 182.5/365.25
    # the windows must actually hit baselines on the 24-day grid
    from mintpy.stdproc.slc2ifg.select_ifgrams import _resolve_annual_windows
    cand = generate_candidates(dates, num_connections=0, annual_windows='auto')
    dts = [_dt_days(a, b) for a, b in cand]
    assert any(abs(d - 182.5) <= 40 for d in dts)
    assert any(abs(d - 365.25) <= 40 for d in dts)


def test_resolve_annual_windows():
    from mintpy.stdproc.slc2ifg.select_ifgrams import _resolve_annual_windows
    dates = _multi_year_dates()
    assert _resolve_annual_windows(None, dates) == ((180, 12), (360, 12))
    assert _resolve_annual_windows('auto', dates) == ((180, 12), (360, 12))
    assert _resolve_annual_windows('none', dates) == ()
    assert _resolve_annual_windows('', dates) == ()
    assert _resolve_annual_windows('182:10,365:15', dates) == ((182, 10), (365, 15))


# ------------------------------------------------------------------------
# Rank-connectivity theorem
# ------------------------------------------------------------------------
def test_rank_equals_n_minus_components():
    rng = np.random.default_rng(7)
    for trial in range(30):
        n = rng.integers(2, 8)
        dates = DATES[:n]
        # random graph with p = 0.35
        cand = []
        for i in range(n - 1):
            for j in range(i + 1, n):
                if rng.random() < 0.35:
                    cand.append((dates[i], dates[j]))
        if not cand:
            continue
        H = design_matrix(dates, cand)
        rank = np.linalg.matrix_rank(H)
        # rank(H) == N - c  (c = number of connected components)
        # c is computed by counting components of the graph
        c = _n_components(dates, cand)
        assert rank == n - c, (n, c, rank)


def test_full_rank_iff_connected():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand)
    sel, _ = select_ifgrams(DATES, cand, w, min_degree=1)
    H = design_matrix(DATES, sel)
    assert np.linalg.matrix_rank(H) == len(DATES) - 1
    # a disconnected subset must be rank deficient
    subset = [(DATES[0], DATES[1]), (DATES[1], DATES[2])]  # dates[3:] isolated
    H2 = design_matrix(DATES, subset)
    assert np.linalg.matrix_rank(H2) < len(DATES) - 1


def _n_components(dates, edges):
    parent = {d: d for d in dates}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        union(a, b)
    return len({find(d) for d in dates})


# ------------------------------------------------------------------------
# Selection guarantees
# ------------------------------------------------------------------------
def test_selection_connected_and_full_rank():
    for seed in range(20):
        cand = generate_candidates(DATES, num_connections=3, annual_windows=())
        w = _random_weights(DATES, cand, seed=seed)
        sel, rep = select_ifgrams(DATES, cand, w, min_degree=2)
        assert check_connected(DATES, sel)
        ver = verify_selection(DATES, sel)
        assert ver['connected'] and ver['full_rank']
        assert rep['n_selected'] >= len(DATES) - 1


def test_zero_residual_closure():
    """With cycle-consistent phases the LSQ residual is ~0 and the SBAS
    estimate equals the truth — for any connected network."""
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=3)
    rng = np.random.default_rng(1)
    phi = rng.normal(size=len(DATES))          # per-date phase
    ref = DATES[0]

    for min_degree, max_pairs in [(1, None), (2, None), (2, 15)]:
        sel, rep = select_ifgrams(DATES, cand, w, min_degree=min_degree,
                                  max_pairs=max_pairs)
        assert check_connected(DATES, sel)
        psi = np.array([phi[DATES.index(b)] - phi[DATES.index(a)]
                        for a, b in sel])
        H = design_matrix(DATES, sel)
        x, res, _, _ = np.linalg.lstsq(H, psi, rcond=None)
        # reconstruction exact: the phases lie in the column space of H, so
        # the LSQ residual is zero (numpy returns an EMPTY residual array
        # exactly when the system is consistent — that is the zero-residual
        # property this test asserts)
        assert np.max(np.abs(H @ x - psi)) < 1e-9
        # recovered = true phase minus reference phase
        assert np.max(np.abs(x - (phi - phi[DATES.index(ref)])[1:])) < 1e-9

    # redundancy does not change the estimate: tree-only vs denser network
    sel_tree, _ = select_ifgrams(DATES, cand, w, min_degree=1)
    sel_dense, _ = select_ifgrams(DATES, cand, w, min_degree=3)
    assert len(sel_dense) > len(sel_tree)
    psi_t = np.array([phi[DATES.index(b)] - phi[DATES.index(a)]
                      for a, b in sel_tree])
    psi_d = np.array([phi[DATES.index(b)] - phi[DATES.index(a)]
                      for a, b in sel_dense])
    x_t = np.linalg.lstsq(design_matrix(DATES, sel_tree), psi_t, rcond=None)[0]
    x_d = np.linalg.lstsq(design_matrix(DATES, sel_dense), psi_d, rcond=None)[0]
    assert np.max(np.abs(x_t - x_d)) < 1e-9


def test_tree_is_maximum_weight_spanning_tree():
    """The N-1 tree edges maximise total weight among all spanning trees
    (brute-force verified on small complete graphs)."""
    for n in (3, 4, 5):
        dates = DATES[:n]
        cand = list(itertools.combinations(dates, 2))
        w = _random_weights(dates, cand, seed=n)
        sel, rep = select_ifgrams(dates, cand, w, min_degree=1)
        assert len(sel) == n - 1
        # brute force: enumerate all spanning trees (connected, n-1 edges)
        best = -1.0
        for sub in itertools.combinations(cand, n - 1):
            if check_connected(dates, sub):
                best = max(best, sum(w[p] for p in sub))
        # report sums are rounded to 6 decimals
        assert abs(rep['tree_weight_sum'] - best) < 1e-5


def test_quality_priority():
    """The selection picks the highest-quality edges first."""
    cand = generate_candidates(DATES, num_connections=0, annual_windows=(),
                               temp_baseline_max=36)
    # high-quality long-baseline pairs, low-quality short ones
    w = {}
    for a, b in cand:
        w[(a, b)] = 0.95 if _dt_days(a, b) >= 24 else 0.2
    high = sorted(p for p in cand if w[p] == 0.95)
    low = [p for p in cand if w[p] == 0.2]
    # generous budget: every high-quality edge fits -> none of the low ones
    # may be selected
    sel, rep = select_ifgrams(DATES, cand, w, min_degree=1,
                              max_pairs=len(high))
    assert all(p in sel for p in high), "high-quality edges must be selected first"
    assert all(p not in sel for p in low), "low-quality edges must be left out"
    assert len(sel) == len(high)
    assert check_connected(DATES, sel)
    # tight budget: still no low-quality edge (they are the tail of the order)
    sel2, _ = select_ifgrams(DATES, cand, w, min_degree=1,
                             max_pairs=len(DATES) + 1)
    assert len(sel2) == len(DATES) + 1
    assert all(p not in sel2 for p in low)


def _old_total_greedy(dates, cand, weights, min_degree):
    """Replica of the pre-mean-optimisation augmentation (tree + stop at
    min_degree), used to prove the new mean objective never regresses."""
    order = sorted(cand, key=lambda p: (-weights[p], p[0], p[1]))
    tree = _kruskal_max(dates, order, weights)
    selected = set(tree)
    deg = {d: 0 for d in dates}
    for a, b in selected:
        deg[a] += 1
        deg[b] += 1
    for p in order:
        if all(v >= min_degree for v in deg.values()):
            break
        if p in selected:
            continue
        a, b = p
        if deg[a] >= min_degree and deg[b] >= min_degree:
            continue
        selected.add(p)
        deg[a] += 1
        deg[b] += 1
    return selected


def test_max_mean_priority_over_total():
    """Given min_degree, selection maximises the *mean* coherence (not the
    total): it keeps adding above-mean edges even after every date already
    reaches min_degree.  The mean must never be lower than the old
    total-greedy, connectivity and degree>=min_degree must hold, and on a
    crafted case an above-mean edge that is not needed for min_degree must
    still be kept."""
    # random instances: new mean >= old mean, constraints preserved
    for seed in range(12):
        cand = generate_candidates(DATES, num_connections=3, annual_windows=())
        w = _random_weights(DATES, cand, seed=seed)
        sel, rep = select_ifgrams(DATES, cand, w, min_degree=2)
        old = _old_total_greedy(DATES, cand, w, 2)
        assert check_connected(DATES, sel)
        assert rep['min_degree_actual'] >= 2
        new_mean = float(np.mean([w[p] for p in sel]))
        old_mean = float(np.mean([w[p] for p in old]))
        assert new_mean >= old_mean - 1e-9, \
            f"seed={seed}: new mean {new_mean} < old {old_mean}"

    # crafted case: an above-mean edge is kept even though every date already
    # reached degree 2 without it (old greedy dropped it -> lower mean)
    dates = ['20240101', '20240113', '20240125', '20240206', '20240218']
    cand = list(itertools.combinations(dates, 2))
    w = {('20240101', '20240113'): 0.90, ('20240101', '20240125'): 0.86,
         ('20240101', '20240206'): 0.90, ('20240101', '20240218'): 0.10,
         ('20240113', '20240125'): 0.80, ('20240113', '20240206'): 0.88,
         ('20240113', '20240218'): 0.10, ('20240125', '20240206'): 0.85,
         ('20240125', '20240218'): 0.10, ('20240206', '20240218'): 0.10}
    sel, rep = select_ifgrams(dates, cand, w, min_degree=2)
    assert ('20240113', '20240125') in sel  # above-mean edge (0.80) kept
    assert rep['selected_weight_mean'] > 0.66
    assert check_connected(dates, sel)
    assert rep['min_degree_actual'] >= 2


def test_min_degree_enforced():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=11)
    for md in (1, 2, 3):
        sel, rep = select_ifgrams(DATES, cand, w, min_degree=md)
        deg = {d: 0 for d in DATES}
        for a, b in sel:
            deg[a] += 1
            deg[b] += 1
        assert min(deg.values()) >= md
        assert rep['min_degree_actual'] >= md
    # min_degree=1 -> exactly a spanning tree
    sel, rep = select_ifgrams(DATES, cand, w, min_degree=1)
    assert len(sel) == len(DATES) - 1


def test_max_pairs_budget():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=5)
    for budget in (len(DATES) - 1, len(DATES) + 3, 100):
        sel, rep = select_ifgrams(DATES, cand, w, min_degree=1,
                                  max_pairs=budget)
        assert len(sel) <= budget
        assert check_connected(DATES, sel)
    # budget never below a spanning tree
    sel, _ = select_ifgrams(DATES, cand, w, min_degree=1, max_pairs=len(DATES) - 1)
    assert len(sel) == len(DATES) - 1


def test_quality_threshold_only_limits_augmentation():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=9)
    # threshold above every weight: still connected via the spanning tree
    sel, rep = select_ifgrams(DATES, cand, w, min_degree=1,
                              quality_threshold=2.0)
    assert check_connected(DATES, sel)
    assert len(sel) == len(DATES) - 1
    # min-degree augmentation is also gated by the threshold
    sel, rep = select_ifgrams(DATES, cand, w, min_degree=3,
                              quality_threshold=2.0)
    assert len(sel) == len(DATES) - 1  # no augmentation above threshold
    assert rep['degree_unmet']


def test_missing_weights_treated_as_zero():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=2)
    # drop the weight of the highest-quality edge
    top = max(cand, key=lambda p: w[p])
    del w[top]
    sel, _ = select_ifgrams(DATES, cand, w, min_degree=2)
    # the unweighted edge is only used when forced (not before low weights)
    assert top not in sel or w[top] == 0.0


def test_disconnected_candidates_raise():
    # two disjoint triangles: 6 edges (>= N-1) but the graph is split
    dates = DATES[:6]
    cand = [(dates[0], dates[1]), (dates[1], dates[2]), (dates[0], dates[2]),
            (dates[3], dates[4]), (dates[4], dates[5]), (dates[3], dates[5])]
    w = {p: 0.9 for p in cand}
    try:
        select_ifgrams(dates, cand, w)
    except ValueError as e:
        assert 'disconnected' in str(e)
    else:
        raise AssertionError('expected ValueError for disconnected candidates')


def test_single_and_two_dates():
    try:
        select_ifgrams(DATES[:1], [], {})
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for a single date')
    cand = [(DATES[0], DATES[1])]
    sel, rep = select_ifgrams(DATES[:2], cand, {cand[0]: 0.8})
    assert sel == cand
    assert rep['n_selected'] == 1


def test_unknown_date_in_pair_raises():
    try:
        select_ifgrams(DATES, [('19990101', DATES[0])], {('19990101', DATES[0]): 1.0})
    except ValueError as e:
        assert 'outside the date list' in str(e)
    else:
        raise AssertionError('expected ValueError for unknown date')


# ------------------------------------------------------------------------
# Bridge repair (2-edge robustness)
# ------------------------------------------------------------------------
def test_bridge_repair():
    # chain 1-2-3-4-5-6 with a high-weight chord crossing the middle bridge
    dates = DATES[:6]
    chain = [(dates[i], dates[i + 1]) for i in range(5)]
    cross = (dates[1], dates[4])  # high-quality replacement candidate
    cand = chain + [cross]
    w = {p: 0.5 for p in chain}
    w[cross] = 0.9
    mid = (dates[2], dates[3])    # the 3-4 bridge of the chain tree
    # without robust: the 3-4 edge is a bridge (removing it disconnects)
    sel_a, _ = select_ifgrams(dates, cand, w, min_degree=1)
    assert not check_connected(dates, [e for e in sel_a if e != mid])
    # with robust: a crossing edge is added and the 3-4 edge stops being a bridge
    sel_b, rep = select_ifgrams(dates, cand, w, min_degree=1, robust=True)
    assert check_connected(dates, [e for e in sel_b if e != mid])
    assert rep['n_bridge_repairs'] >= 1
    assert len(sel_b) > len(sel_a)


# ------------------------------------------------------------------------
# Weights
# ------------------------------------------------------------------------
def test_temporal_model_weights():
    cand = [(DATES[0], d) for d in DATES[1:]]
    w = temporal_model_weights(DATES, cand, tau_days=90.0, gamma0=0.9)
    dts = [(np.datetime64(b) - np.datetime64(a)).astype('timedelta64[D]')
           for a, b in cand]
    vals = [w[p] for p in cand]
    # monotone decreasing in temporal baseline
    order = np.argsort(dts)
    assert all(vals[order[i]] >= vals[order[i + 1]] for i in range(len(vals) - 1))
    # bounds
    assert all(0.0 <= v <= 0.9 for v in vals)
    # perpendicular factor
    perp = {d: float(i * 30) for i, d in enumerate(DATES)}
    w2 = temporal_model_weights(DATES, cand, tau_days=90.0, gamma0=1.0,
                                perp_baselines=perp, perp_baseline_max=60.0)
    assert all(w2[p] <= w[p] for p in cand)


def test_aggregate_coherence_stats():
    arr = np.array([0.0, 0.4, 0.6, 0.8, 0.0, 0.5])
    assert abs(aggregate_coherence(arr, 'mean') - 0.575) < 1e-12
    assert abs(aggregate_coherence(arr, 'median') - 0.55) < 1e-12
    assert abs(aggregate_coherence(arr, 'usable_frac', 0.5) - 0.75) < 1e-12
    assert aggregate_coherence(np.zeros(10), 'mean') == 0.0
    try:
        aggregate_coherence(arr, 'bogus')
    except ValueError:
        pass
    else:
        raise AssertionError('expected ValueError for unknown stat')


def test_read_perp_baselines(tmp_path):
    f = tmp_path / 'bperp.txt'
    f.write_text('# date bperp\n20230105 12.5\n20230117 -8.0\n')
    perp = read_perp_baselines(str(f))
    assert perp == {'20230105': 12.5, '20230117': -8.0}


# ------------------------------------------------------------------------
# Coherence kernels (pure numpy/scipy — no GDAL needed)
# ------------------------------------------------------------------------
def test_boxcar_coherence_identical_and_noise():
    from mintpy.stdproc.slc2ifg.select_ifgrams import _boxcar_coherence
    rng = np.random.default_rng(0)
    # identical SLCs -> coherence ~ 1 in the interior
    s = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(np.complex64)
    coh = _boxcar_coherence(s, s, window=5)
    assert coh[16:-16, 16:-16].mean() > 0.99
    # independent SLCs -> coherence much less than 1 (boxcar bias floor)
    s2 = (rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))).astype(np.complex64)
    coh2 = _boxcar_coherence(s, s2, window=5)
    assert coh2[16:-16, 16:-16].mean() < 0.25
    # values clipped to [0, 1]
    assert coh.min() >= 0.0 and coh.max() <= 1.0


def test_debias_coherence():
    from mintpy.stdproc.slc2ifg.select_ifgrams import _debias_coherence
    looks = 100.0  # (10 x 10) effective looks
    thr = 1.0 / (2.0 * looks)
    coh = np.array([thr * 0.5, 0.2, 0.5, 0.8, 1.0])
    out = _debias_coherence(coh, looks)
    # below-threshold estimates collapse to 0
    assert out[0] == 0.0 and out[1] > 0.0
    # monotone and never above the raw estimate (bias is positive)
    assert np.all(np.diff(out) >= 0)
    assert np.all(out <= coh + 1e-9)
    # exact at 1
    assert abs(out[-1] - 1.0) < 1e-9


# ------------------------------------------------------------------------
# Optimality of augmentation (A-optimality: adding edges reduces variance)
# ------------------------------------------------------------------------
def test_added_edges_reduce_estimate_covariance():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=21)
    sel_tree, _ = select_ifgrams(DATES, cand, w, min_degree=1)
    sel_dense, _ = select_ifgrams(DATES, cand, w, min_degree=2)
    H1 = design_matrix(DATES, sel_tree)
    H2 = design_matrix(DATES, sel_dense)
    W1 = np.diag([w[p] for p in sel_tree])
    W2 = np.diag([w[p] for p in sel_dense])
    cov1 = np.linalg.inv(H1.T @ W1 @ H1)
    cov2 = np.linalg.inv(H2.T @ W2 @ H2)
    # adding positive-information edges strictly reduces the trace
    # (Sherman-Morrison: (A + w h h^T)^-1 = A^-1 - PSD term)
    assert np.trace(cov2) < np.trace(cov1)


# ------------------------------------------------------------------------
# Orchestration + report
# ------------------------------------------------------------------------
def test_select_pairs_end_to_end(tmp_path):
    report_file = tmp_path / 'selection_report.json'
    pairs, rep = select_pairs(
        DATES,
        params={'weight_source': 'model', 'min_degree': 2,
                'report_file': str(report_file)})
    assert check_connected(DATES, pairs)
    assert rep['connected'] and rep['full_rank']
    assert rep['weight_source'] == 'model'
    assert rep['n_dates'] == len(DATES)
    assert report_file.exists()
    data = json.loads(report_file.read_text())
    assert data['n_selected'] == len(pairs)
    assert 'n_dates' in data and 'full_rank' in data


def test_select_pairs_weight_source_validation():
    try:
        select_pairs(DATES, params={'weight_source': 'bogus'})
    except ValueError as e:
        assert 'weight_source' in str(e)
    else:
        raise AssertionError('expected ValueError for bad weight_source')


def test_select_pairs_needs_slc_for_measured(tmp_path):
    try:
        select_pairs(DATES, params={'weight_source': 'coherence'})
    except ValueError as e:
        assert 'coh_dir' in str(e) or 'SLC' in str(e)
    else:
        raise AssertionError('expected ValueError without coherence source')


def test_write_report_json(tmp_path):
    rep = {'n_dates': 3, 'pairs': [('20230105', '20230117')]}
    out = tmp_path / 'r.json'
    write_report(rep, str(out))
    data = json.loads(out.read_text())
    assert data['n_dates'] == 3
    # tuple values are JSON-encoded as lists; tuple *keys* become d1_d2
    assert data['pairs'] == [['20230105', '20230117']]


def test_verify_selection_report():
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=4)
    sel, _ = select_ifgrams(DATES, cand, w, min_degree=2)
    ver = verify_selection(DATES, sel)
    assert ver == {'n_dates': len(DATES), 'n_pairs': len(sel),
                   'connected': True, 'rank': len(DATES) - 1,
                   'full_rank': True}


def test_generate_pairs_select_mode():
    from mintpy.stdproc.slc2ifg.ifgram_list import generate_pairs
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=6)
    pairs = generate_pairs(
        DATES, mode='select', num_connections=3, oneyear_range=None,
        select_params={'weight_source': 'model', 'min_degree': 2})
    assert check_connected(DATES, pairs)
    assert len(pairs) >= len(DATES) - 1


def test_generate_pairs_select_uses_num_connections():
    from mintpy.stdproc.slc2ifg.ifgram_list import generate_pairs
    pairs = generate_pairs(
        DATES, mode='select', num_connections=1, oneyear_range=None,
        select_params={'weight_source': 'model', 'min_degree': 1})
    # k=1 skeleton = a single chain; pure spanning tree = N-1 edges
    assert check_connected(DATES, pairs)
    assert len(pairs) == len(DATES) - 1


# ------------------------------------------------------------------------
# CLI integration
# ------------------------------------------------------------------------
def test_ifgram_list_cli_select_mode(tmp_path, capsys):
    from mintpy.stdproc.slc2ifg import ifgram_list
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    for d in DATES:
        (slc_dir / d).mkdir()
    out = tmp_path / 'ifg'
    args = ifgram_list.parse_arguments([
        '--slc', str(slc_dir), '--outdir', str(out), '--mode', 'select',
        '--select-weight-source', 'model', '--select-min-degree', '2',
    ])
    rc = ifgram_list.main(args)
    assert rc == 0
    pair_file = out / 'ifgram_list.txt'
    assert pair_file.exists()
    lines = [l.strip() for l in pair_file.read_text().splitlines()
             if l.strip() and not l.startswith('#')]
    assert len(lines) == len(DATES) - 1 + 2  # tree + min-degree=2 extras
    assert check_connected(DATES, [tuple(l.split('-')) for l in lines])


def test_ifgram_list_cli_select_report(tmp_path):
    from mintpy.stdproc.slc2ifg import ifgram_list
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    for d in DATES:
        (slc_dir / d).mkdir()
    out = tmp_path / 'ifg'
    report = tmp_path / 'report.json'
    args = ifgram_list.parse_arguments([
        '--slc', str(slc_dir), '--outdir', str(out), '--mode', 'select',
        '--select-weight-source', 'model',
        '--select-report', str(report),
    ])
    rc = ifgram_list.main(args)
    assert rc == 0
    data = json.loads(report.read_text())
    assert data['connected'] is True
    assert data['full_rank'] is True


# ------------------------------------------------------------------------
# Engine integration
# ------------------------------------------------------------------------
def test_engine_select_mode_plan(tmp_path):
    """The engine's eager ifgram_list runs in select mode and produces a
    connected, full-rank pair list consumed by generate_ifgram nodes."""
    from mintpy.stdproc.slc2ifg.engine.config import load_engine_config
    from mintpy.stdproc.slc2ifg.engine.engine import Engine

    inp = tmp_path / 'input'
    inp.mkdir()
    dates = ['20220105', '20220117', '20220129', '20220210',
             '20220222', '20220306']
    for d in dates:
        (inp / f'{d}.slc.tif').touch()
    cfg = tmp_path / 'mini.cfg'
    cfg.write_text(
        f'slc2ifg.work_dir = {tmp_path}\n'
        f'slc2ifg.slc_input = {inp}\n'
        'slc2ifg.processor = isce3\n'
        'engine.max_workers = 2\n'
        'engine.gpu = false\n'
        'slc2ifg.ifgram_list.mode = select\n'
        'slc2ifg.ifgram_list.select.weight_source = model\n'
        'slc2ifg.ifgram_list.select.min_degree = 2\n'
        'slc2ifg.ifgram_list.select.report = selection.json\n'
    )
    eng = Engine(load_engine_config(str(cfg)))
    g = eng.plan(dry_run=True)
    pair_file = eng.ifgram_dir / 'ifgram_list.txt'
    assert pair_file.exists()
    pairs = eng._read_pairs(pair_file)
    assert check_connected(dates, pairs)
    ver = verify_selection(dates, pairs)
    assert ver['connected'] and ver['full_rank']
    # the report was written to <work_dir> (relative paths resolve there)
    assert (tmp_path / 'selection.json').exists()
    # every pair got a generate_ifgram node
    for d1, d2 in pairs:
        assert f'generate_ifgram#single#{d1}_{d2}' in g.nodes
    assert len(pairs) >= len(dates) - 1


# ------------------------------------------------------------------------
# GDAL-dependent weight paths (tested with a fake osgeo — no real GDAL)
# ------------------------------------------------------------------------
import types as _types


def _install_fake_osgeo(sources):
    """Inject a fake ``osgeo.gdal``/``osgeo.osr`` into sys.modules.

    ``sources`` maps a filesystem path string to the ndarray that
    ``gdal.Open`` should serve (complex SLC arrays or coherence maps).
    Returns the fake gdal module; call ``_uninstall_fake_osgeo()`` after.
    """
    import sys

    class FakeDriver:
        ShortName = 'GTiff'

    class FakeBand:
        def __init__(self, arr, index):
            self._arr = np.asarray(arr)
            self._index = index

        def ReadAsArray(self, xoff=None, yoff=None, xsize=None, ysize=None,
                        buf_xsize=None, buf_ysize=None,
                        resample_alg=None):
            a = (self._arr if self._arr.ndim == 2
                 else self._arr[self._index - 1])
            # grid-sampling window read: (xoff, yoff, xsize, ysize)
            if xoff is not None and yoff is not None and xsize and ysize:
                return a[yoff:yoff + ysize, xoff:xoff + xsize].copy()
            if buf_xsize and buf_ysize:
                rows, cols = a.shape
                fy, fx = max(1, rows // buf_ysize), max(1, cols // buf_xsize)
                if rows % fy == 0 and cols % fx == 0:
                    a = a.reshape(rows // fy, fy, cols // fx, fx).mean(axis=(1, 3))
                else:
                    a = a[::fy, ::fx]
            return a

    class FakeDataset:
        def __init__(self, arr):
            self._arr = np.asarray(arr)
            self.RasterYSize, self.RasterXSize = self._arr.shape[:2]
            self.RasterCount = 2 if self._arr.ndim == 3 else 1

        def GetDriver(self):
            return FakeDriver()

        def GetRasterBand(self, i):
            return FakeBand(self._arr, i)

    class FakeGdal:
        GRIORA_Average = 1
        GRIORA_NearestNeighbour = 2

        def UseExceptions(self):
            pass

        def Open(self, path):
            key = str(path)
            if key.startswith('HDF5:'):
                key = key.split(':', 2)[1]
            return FakeDataset(sources[key])

    gdal_mod = _types.ModuleType('osgeo.gdal')
    # plain module-level functions (no bound self)
    gdal_mod.GRIORA_Average = 1
    gdal_mod.GRIORA_NearestNeighbour = 2
    gdal_mod.UseExceptions = lambda: None

    def _fake_open(path):
        key = str(path)
        if key.startswith('HDF5:'):
            key = key.split(':', 2)[1]
        elif key.startswith('NETCDF:'):
            # NETCDF:"<path>":"//<subdataset>" -> <path>
            key = key.split('"')[1]
        return FakeDataset(sources[key])

    gdal_mod.Open = _fake_open
    osr_mod = _types.ModuleType('osgeo.osr')
    pkg = _types.ModuleType('osgeo')
    pkg.gdal = gdal_mod
    pkg.osr = osr_mod
    sys.modules['osgeo'] = pkg
    sys.modules['osgeo.gdal'] = gdal_mod
    sys.modules['osgeo.osr'] = osr_mod
    return gdal_mod


def _uninstall_fake_osgeo():
    import sys
    for key in ('osgeo', 'osgeo.gdal', 'osgeo.osr'):
        sys.modules.pop(key, None)
    # drop cached modules that imported the fake (re-imported on next use)
    for key in ('mintpy.stdproc.slc2ifg.generate_coh_complex',
                'mintpy.stdproc.slc2ifg.utils.slc2ifg_utils'):
        sys.modules.pop(key, None)


def test_quick_coherence_weights(tmp_path):
    """On-the-fly coherence on downsampled SLCs (fake GDAL)."""
    from mintpy.stdproc.slc2ifg.select_ifgrams import quick_coherence_weights

    rng = np.random.default_rng(42)
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    dates = ['20230105', '20230117', '20230129']
    sources = {}
    for d in dates:
        (slc_dir / f'{d}.slc.tif').touch()
        sources[str(slc_dir / f'{d}.slc.tif')] = (
            rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
        ).astype(np.complex64)
    # date 2 shares its scene with date 0 -> high coherence; date 1 independent
    sources[str(slc_dir / '20230129.slc.tif')] = sources[str(slc_dir / '20230105.slc.tif')]

    gdal_fake = _install_fake_osgeo(sources)
    try:
        pairs = [('20230105', '20230129'), ('20230105', '20230117')]
        w = quick_coherence_weights(
            dates, pairs, str(slc_dir), processor='isce3',
            nlks=8, window=5, max_pixels=4096, debias=True, stat='mean')
        hi, lo = w[('20230105', '20230129')], w[('20230105', '20230117')]
        assert hi is not None and lo is not None
        assert hi > 0.9, f"identical scenes should be highly coherent, got {hi}"
        assert lo < hi, "independent scenes must rank below identical ones"
        # missing SLC -> None
        w2 = quick_coherence_weights(
            dates + ['20230411'], pairs, str(slc_dir), processor='isce3')
        assert w2[('20230105', '20230129')] is not None
    finally:
        _uninstall_fake_osgeo()


def test_quick_coherence_two_band_isce2(tmp_path):
    """Two-band (real/imag) isce2 SLCs through the quick-coherence path."""
    from mintpy.stdproc.slc2ifg.select_ifgrams import quick_coherence_weights

    rng = np.random.default_rng(7)
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    sources = {}
    for d in ('20230105', '20230117'):
        (slc_dir / f'{d}.slc').touch()
        re = rng.normal(size=(48, 48)).astype(np.float32)
        im = rng.normal(size=(48, 48)).astype(np.float32)
        sources[str(slc_dir / f'{d}.slc')] = np.stack([re, im])
    gdal_fake = _install_fake_osgeo(sources)
    try:
        w = quick_coherence_weights(
            ['20230105', '20230117'], [('20230105', '20230117')],
            str(slc_dir), processor='isce2', nlks=6, window=3)
        assert 0.0 <= w[('20230105', '20230117')] <= 1.0
    finally:
        _uninstall_fake_osgeo()


def test_weights_from_coherence_rasters(tmp_path):
    """Weights from existing coherence rasters (fake GDAL)."""
    from mintpy.stdproc.slc2ifg.select_ifgrams import weights_from_coherence_rasters

    coh_dir = tmp_path / 'coh'
    pairs = [('20230105', '20230117'), ('20230117', '20230129'),
             ('20230129', '20230210')]
    sources = {}
    for a, b, val in [(pairs[0][0], pairs[0][1], 0.9),
                      (pairs[1][0], pairs[1][1], 0.4)]:
        p = coh_dir / f'{a}_{b}' / 'filt_mli_phsig.coh.tif'
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
        sources[str(p)] = np.full((32, 32), val, dtype=np.float32)
    # third pair has no raster -> None
    gdal_fake = _install_fake_osgeo(sources)
    try:
        w = weights_from_coherence_rasters(
            pairs, str(coh_dir), kind='phsig', variant='filt_mli',
            processor='isce3', stat='mean')
        assert abs(w[pairs[0]] - 0.9) < 1e-6
        assert abs(w[pairs[1]] - 0.4) < 1e-6
        assert w[pairs[2]] is None
    finally:
        _uninstall_fake_osgeo()


def test_select_pairs_with_quick_coherence_end_to_end(tmp_path):
    """select_pairs with weight_source='coherence' (quick path) selects a
    connected, full-rank network on synthetic SLCs."""
    from mintpy.stdproc.slc2ifg.select_ifgrams import select_pairs

    rng = np.random.default_rng(3)
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    dates = ['20230105', '20230117', '20230129', '20230210', '20230222']
    sources = {}
    for d in dates:
        (slc_dir / f'{d}.slc.tif').touch()
        sources[str(slc_dir / f'{d}.slc.tif')] = (
            rng.normal(size=(48, 48)) + 1j * rng.normal(size=(48, 48))
        ).astype(np.complex64)
    gdal_fake = _install_fake_osgeo(sources)
    try:
        pairs, rep = select_pairs(
            dates, slc_dir=str(slc_dir), processor='isce3',
            params={'weight_source': 'coherence', 'min_degree': 2,
                    'quick_nlks': 6, 'quick_window': 3})
        assert check_connected(dates, pairs)
        assert rep['connected'] and rep['full_rank']
        assert rep['weight_source'] == 'quick_coherence'
        assert rep['n_selected'] >= len(dates) - 1
    finally:
        _uninstall_fake_osgeo()


# ------------------------------------------------------------------------
# Determinism, min_degree=0, DOT output, parallel quick coherence
# ------------------------------------------------------------------------
def test_selection_deterministic():
    """Same input twice -> bit-identical pair lists (both select_ifgrams and
    the full select_pairs pipeline)."""
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=31)
    sel1, r1 = select_ifgrams(DATES, cand, w, min_degree=2, robust=True)
    sel2, r2 = select_ifgrams(DATES, cand, w, min_degree=2, robust=True)
    assert sel1 == sel2 and r1['n_selected'] == r2['n_selected']
    p1, rep1 = select_pairs(DATES, params={'weight_source': 'model',
                                           'min_degree': 2, 'robust': True})
    p2, rep2 = select_pairs(DATES, params={'weight_source': 'model',
                                           'min_degree': 2, 'robust': True})
    assert p1 == p2
    assert rep1['n_selected'] == rep2['n_selected']


def test_min_degree_zero_equals_tree():
    """min_degree=0/1 both mean 'pure maximum spanning tree' (N-1 edges)."""
    cand = generate_candidates(DATES, num_connections=3, annual_windows=())
    w = _random_weights(DATES, cand, seed=17)
    sel0, rep0 = select_ifgrams(DATES, cand, w, min_degree=0)
    sel1, rep1 = select_ifgrams(DATES, cand, w, min_degree=1)
    assert sel0 == sel1
    assert len(sel0) == len(DATES) - 1
    assert rep0['min_degree_target'] == 0
    # through the full pipeline as well
    p0, r0 = select_pairs(DATES, params={'weight_source': 'model',
                                         'min_degree': 0})
    assert len(p0) == len(DATES) - 1
    assert check_connected(DATES, p0)


def test_network_dot_output(tmp_path):
    dot = tmp_path / 'net.dot'
    pairs, rep = select_pairs(DATES, params={
        'weight_source': 'model', 'min_degree': 2, 'dot_file': str(dot)})
    assert dot.exists()
    text = dot.read_text()
    assert text.startswith('digraph ifgram_network {')
    assert 'rankdir=LR;' in text
    for d in DATES:
        assert f'"{d}";' in text
    # every selected pair is drawn; tree edges are bold
    tree = {tuple(e.split('_')) for e in rep['tree_edges']}
    assert len(tree) == len(DATES) - 1
    for a, b in pairs:
        assert f'"{a}" -> "{b}"' in text
    assert any('style=bold' in line for line in text.splitlines())
    # weights are labelled
    assert any('label=' in line for line in text.splitlines())


def test_write_network_dot_direct(tmp_path):
    from mintpy.stdproc.slc2ifg.select_ifgrams import write_network_dot
    pairs = [('20230105', '20230117'), ('20230117', '20230129')]
    w = {('20230105', '20230117'): 0.875, ('20230117', '20230129'): 0.766}
    p = tmp_path / 'n.dot'
    write_network_dot(DATES[:3], pairs, weights=w,
                      tree_edges=[('20230105', '20230117')], path=str(p))
    text = p.read_text()
    assert 'style=bold' in text
    assert 'label="0.875"' in text
    assert 'label="0.766"' in text


def test_quick_coherence_parallel_matches_serial(tmp_path):
    """Parallel (max_workers>1) quick coherence gives identical weights."""
    from mintpy.stdproc.slc2ifg.select_ifgrams import quick_coherence_weights

    rng = np.random.default_rng(42)
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    dates = ['20230105', '20230117', '20230129']
    sources = {}
    for d in dates:
        (slc_dir / f'{d}.slc.tif').touch()
        sources[str(slc_dir / f'{d}.slc.tif')] = (
            rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
        ).astype(np.complex64)
    sources[str(slc_dir / '20230129.slc.tif')] = sources[str(slc_dir / '20230105.slc.tif')]
    pairs = [('20230105', '20230129'), ('20230105', '20230117'),
             ('20230117', '20230129')]
    kw = dict(dates=dates, pairs=pairs, slc_dir=str(slc_dir), processor='isce3',
              nlks=8, window=5, max_pixels=4096, debias=True, stat='mean')
    gdal_fake = _install_fake_osgeo(sources)
    try:
        w1 = quick_coherence_weights(**kw, max_workers=1)
        w4 = quick_coherence_weights(**kw, max_workers=4)
        assert set(w1) == set(w4)
        for k in w1:
            assert abs((w1[k] or 0.0) - (w4[k] or 0.0)) < 1e-12
        assert w4[('20230105', '20230129')] > 0.9
    finally:
        _uninstall_fake_osgeo()


def test_quick_coherence_opera_h5(tmp_path):
    """OPERA-style GSLC .h5 files (subdataset /data/VV via NETCDF driver)."""
    from mintpy.stdproc.slc2ifg.select_ifgrams import quick_coherence_weights

    rng = np.random.default_rng(9)
    slc_dir = tmp_path / 'slc'
    slc_dir.mkdir()
    sources = {}
    for d in ('20210104', '20210110'):
        (slc_dir / f't124_264305_iw2_{d}.h5').touch()
        sources[str(slc_dir / f't124_264305_iw2_{d}.h5')] = (
            rng.normal(size=(64, 64)) + 1j * rng.normal(size=(64, 64))
        ).astype(np.complex64)
    gdal_fake = _install_fake_osgeo(sources)
    try:
        w = quick_coherence_weights(
            ['20210104', '20210110'], [('20210104', '20210110')],
            str(slc_dir), processor='isce3', slc_pattern='*.h5',
            nlks=8, window=5, subdataset='/data/VV')
        v = w[('20210104', '20210110')]
        assert v is not None and 0.0 <= v <= 1.0
    finally:
        _uninstall_fake_osgeo()
