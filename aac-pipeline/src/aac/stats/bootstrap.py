"""Percentile bootstrap over items (ARCH section 5.2).

The resampling RNG is a pure function of the table cell: `ci(values, cell_key=...)`
seeds scipy.stats.bootstrap with derive_seed(cell_key), so adding, removing or
reordering other cells never changes a cell's interval. n_resamples and level default to
configs/stats.yaml (1000, 0.95). Three primitives:

 ci / ci_seeded single-sample percentile CI of the mean (scipy.stats.bootstrap)
 paired_ci / paired_ci_seeded paired index resample: the SAME index array is applied to both
 systems and `statistic(x[idx], y[idx])` is collected (pre-registered specification)
 ci_index generic index-resample CI for a statistic over item indices
 (Table 4.6 selective macro-F1 uses it)
 unpaired_ci_index two independent index resamples per draw (two different item sets), for an
 unpaired difference such as Table 4.2's H1 secondary 
 cluster_ci item-cluster bootstrap: items are resampled and every row of a drawn item travels
 with it (Table 4.4: all recipes of an item together), optionally within
 strata (the attack subsets of a pooled cell)
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import stats as _scipy_stats

from aac.run.hashing import derive_seed

N_RESAMPLES = 1000
LEVEL = 0.95


def _as_1d(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name}: empty input; a CI over no items is not a number")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name}: non-finite values")
    return arr


def ci_seeded(values, *, seed: int, n_resamples: int = N_RESAMPLES, level: float = LEVEL) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean with an explicit integer seed
 (scipy.stats.bootstrap, method="percentile", random_state=seed)."""
    x = _as_1d(values, "ci")
    res = _scipy_stats.bootstrap((x,), np.mean, n_resamples=int(n_resamples), confidence_level=float(level),
                                 method="percentile", random_state=int(seed))
    return float(res.confidence_interval.low), float(res.confidence_interval.high)


def ci(values, *, cell_key: str, n_resamples: int = N_RESAMPLES, level: float = LEVEL) -> tuple[float, float]:
    """Percentile bootstrap CI keyed by the table cell: rng = derive_seed(cell_key)."""
    if not cell_key:
        raise ValueError("cell_key is required: every CI is keyed by its table cell")
    return ci_seeded(values, seed=derive_seed(cell_key), n_resamples=n_resamples, level=level)


def paired_ci_seeded(x, y, statistic: Callable[[np.ndarray, np.ndarray], float], *, seed: int,
                     n_resamples: int = N_RESAMPLES, level: float = LEVEL) -> tuple[float, float]:
    """Paired item bootstrap: one index resample per draw applied to BOTH systems;
 percentiles (numpy linear interpolation) of statistic(x[idx], y[idx])."""
    xa = _as_1d(x, "paired_ci.x")
    ya = _as_1d(y, "paired_ci.y")
    if xa.shape != ya.shape:
        raise ValueError("paired_ci: x and y must have the same length (same items)")
    rng = np.random.default_rng(int(seed))
    n = xa.size
    draws = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n, n)
        draws[i] = float(statistic(xa[idx], ya[idx]))
    alpha = (1.0 - float(level)) / 2.0
    lo, hi = np.percentile(draws, [100.0 * alpha, 100.0 * (1.0 - alpha)])
    return float(lo), float(hi)


def paired_ci(x, y, statistic: Callable[[np.ndarray, np.ndarray], float], *, cell_key: str,
              n_resamples: int = N_RESAMPLES, level: float = LEVEL) -> tuple[float, float]:
    if not cell_key:
        raise ValueError("cell_key is required: every CI is keyed by its table cell")
    return paired_ci_seeded(x, y, statistic, seed=derive_seed(cell_key), n_resamples=n_resamples, level=level)


def ci_index(stat: Callable[[np.ndarray], float], n_items: int, *, n_resamples: int, seed: int,
             level: float = LEVEL,
             draw: Callable[[np.random.Generator], np.ndarray] | None = None) -> tuple[float, float]:
    """stat(idx) receives an integer index array of length n_items (a resample); the
 caller derives `seed` from its cell key. `draw(rng)` replaces the default uniform
 resample (rng.integers(0, n_items, n_items)) for designs such as a class-stratified
 bootstrap; it must use only the generator it is given ."""
    if int(n_items) <= 0:
        raise ValueError("ci_index: no items")
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n_items, size=n_items) if draw is None else draw(rng)
        vals[i] = stat(idx)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def unpaired_ci_index(stat: Callable[[np.ndarray, np.ndarray], float], n_a: int, n_b: int, *, n_resamples: int,
                      seed: int, level: float = LEVEL) -> tuple[float, float]:
    """stat(idx_a, idx_b) receives two independent uniform resamples (lengths n_a and n_b) drawn in that order
 from one generator seeded with `seed` (the caller derives it from its cell key); percentile interval
 (numpy quantile, as ci_index). Used when the two systems are scored on different item sets, so no
 paired resample exists."""
    if int(n_a) <= 0 or int(n_b) <= 0:
        raise ValueError("unpaired_ci_index: no items")
    rng = np.random.default_rng(int(seed))
    vals = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        ia = rng.integers(0, n_a, size=n_a)
        ib = rng.integers(0, n_b, size=n_b)
        vals[i] = stat(ia, ib)
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _cluster_draw(rng: np.random.Generator, n: int, strata: np.ndarray | None) -> np.ndarray:
    if strata is None:
        return rng.integers(0, n, size=n)
    parts = []
    for s in sorted(set(strata.tolist())):
        members = np.flatnonzero(strata == s)
        parts.append(members[rng.integers(0, len(members), len(members))])
    return np.concatenate(parts)


def cluster_ci(values_by_item, statistic: Callable[[np.ndarray], float], *, cell_key: str,
               n_resamples: int = N_RESAMPLES, level: float = LEVEL, strata=None) -> tuple[float, float]:
    """Item-cluster percentile bootstrap (Field and Welsh 2007): `values_by_item[i]` holds the rows of item i (a
 scalar, a 1-D array of row values or a 2-D array rows x columns). Each draw resamples item indices with
 replacement (rng = derive_seed(cell_key), the keyed convention of ci / ci_index), concatenates the rows of the
 drawn items in draw order and collects `statistic(rows)`; numpy quantile percentiles as ci_index. With one row
 per item and no strata this equals ci_index(lambda idx: statistic(x[idx]), n, seed=derive_seed(cell_key)).
 `strata` (one label per item) resamples within each stratum (labels in sorted order), so a pooled cell keeps
 its per-subset item counts."""
    if not cell_key:
        raise ValueError("cell_key is required: every CI is keyed by its table cell")
    groups = [np.atleast_1d(np.asarray(g, dtype=float)) for g in values_by_item]
    n = len(groups)
    if n == 0:
        raise ValueError("cluster_ci: no items")
    if any(g.shape[0] == 0 for g in groups):
        raise ValueError("cluster_ci: an item without rows")
    st = None
    if strata is not None:
        st = np.asarray(list(strata), dtype=object)
        if st.shape[0] != n:
            raise ValueError("cluster_ci: one stratum label per item")
    rng = np.random.default_rng(derive_seed(cell_key))
    vals = np.empty(int(n_resamples), dtype=float)
    for i in range(int(n_resamples)):
        idx = _cluster_draw(rng, n, st)
        vals[i] = float(statistic(np.concatenate([groups[j] for j in idx], axis=0)))
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return float(lo), float(hi)
