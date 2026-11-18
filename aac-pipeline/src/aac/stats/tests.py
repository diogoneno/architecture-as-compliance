"""Paired and item-level tests (pre-registered specification.

 paired_t_dz scipy.stats.ttest_rel on per-seed metrics + Cohen's d_z = mean(d)/std(d, ddof=1)
 wilcoxon_descriptive scipy.stats.wilcoxon(method="exact"), reported as DESCRIPTIVE with its exact
 two-sided floor p = 2 / 2**n_eff (0.0625 at n_eff = 5: cannot reach 0.05), where
 n_eff counts the non-zero differences scipy keeps (zero_method="wilcox")
 mcnemar_exact statsmodels mcnemar(exact=True) on the 2x2 "decided and correct vs otherwise" table
 welch_t scipy.stats.ttest_ind(equal_var=False) on two independent samples of per-seed metrics
 (an unpaired, seed-level comparison; H1 secondary on different test sets)

Functions return numbers and flags only; formatting (e.g. `n/a (zero variance)`) lives in
aac.reports.fmt. Every function raises ValueError on empty input; paired tests on n < 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as _scipy_stats
from statsmodels.stats.contingency_tables import mcnemar


@dataclass(frozen=True)
class TestResult:
    __test__ = False  # not a pytest collectable
    statistic: float | None
    p: float | None
    n: int
    d_z: float | None = None
    degenerate: bool = False
    floor_p: float | None = None
    n_eff: int | None = None

    @property
    def t(self) -> float | None:
        return self.statistic


def _paired(a, b, name: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError(f"{name}: empty input")
    if x.shape != y.shape:
        raise ValueError(f"{name}: paired inputs must have the same length")
    if x.size < 2:
        raise ValueError(f"{name}: a paired test needs at least two seeds")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError(f"{name}: non-finite values")
    return x, y



# equal in exact arithmetic often differ in the last ulp, which would give t ~ 1e15 and p ~ 1e-32.
DEGENERATE_REL_SD = 1e-12


def paired_t_dz(a, b) -> TestResult:
    """Paired t over per-seed differences d = a - b with Cohen's d_z. Zero variance of d
 (all differences equal, up to float rounding: sd <= 1e-12 * max(1, max|d|)) is reported
 as degenerate=True with no statistic, never NaN and never an absurd p."""
    x, y = _paired(a, b, "paired_t_dz")
    d = x - y
    sd = float(np.std(d, ddof=1))
    scale = max(1.0, float(np.max(np.abs(d))))
    if sd <= DEGENERATE_REL_SD * scale:
        return TestResult(statistic=None, p=None, n=int(d.size), d_z=None, degenerate=True)
    res = _scipy_stats.ttest_rel(x, y)
    return TestResult(statistic=float(res.statistic), p=float(res.pvalue), n=int(d.size),
                      d_z=float(np.mean(d) / sd), degenerate=False)


def wilcoxon_floor_p(n: int) -> float:
    """Exact minimum two-sided p of the signed-rank test at n pairs (all signs equal): 2 / 2**n."""
    if int(n) < 1:
        raise ValueError("wilcoxon_floor_p: n must be positive")
    return 2.0 / float(2 ** int(n))


def wilcoxon_descriptive(a, b) -> TestResult:
    """Exact Wilcoxon signed-rank on per-seed pairs, with the floor p stated (descriptive)."""
    x, y = _paired(a, b, "wilcoxon_descriptive")
    d = x - y
    n = int(d.size)
    n_eff = int(np.count_nonzero(d))  
    if n_eff == 0:
        return TestResult(statistic=None, p=None, n=n, degenerate=True, floor_p=None, n_eff=0)
    res = _scipy_stats.wilcoxon(x, y, method="exact")
    return TestResult(statistic=float(res.statistic), p=float(res.pvalue), n=n, floor_p=wilcoxon_floor_p(n_eff),
                      n_eff=n_eff)


def mcnemar_exact(table_2x2) -> TestResult:
    """Exact McNemar on a 2x2 table [[a, b], [c, d]] (b, c discordant)."""
    arr = np.asarray(table_2x2)
    if arr.size == 0:
        raise ValueError("mcnemar_exact: empty table")
    if arr.shape != (2, 2):
        raise ValueError(f"mcnemar_exact: expected a 2x2 table, got shape {arr.shape}")
    if np.any(arr < 0):
        raise ValueError("mcnemar_exact: negative counts")
    res = mcnemar(arr, exact=True)
    return TestResult(statistic=float(res.statistic), p=float(res.pvalue), n=int(arr.sum()))


def welch_t(a, b) -> TestResult:
    """Welch's unequal-variance t-test of mean(a) - mean(b) on two independent samples (e.g. per-seed
 macro-F1 of two systems evaluated on different test sets). d_z carries no meaning here and stays None;
 zero variance in both samples is degenerate=True (no statistic, never NaN)."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        raise ValueError("welch_t: empty input")
    if x.size < 2 or y.size < 2:
        raise ValueError("welch_t: each sample needs at least two values")
    if not (np.all(np.isfinite(x)) and np.all(np.isfinite(y))):
        raise ValueError("welch_t: non-finite values")
    scale = max(1.0, float(np.max(np.abs(np.concatenate([x, y])))))
    if float(np.std(x, ddof=1)) <= DEGENERATE_REL_SD * scale and float(np.std(y, ddof=1)) <= DEGENERATE_REL_SD * scale:
        return TestResult(statistic=None, p=None, n=int(x.size + y.size), degenerate=True)
    res = _scipy_stats.ttest_ind(x, y, equal_var=False)
    return TestResult(statistic=float(res.statistic), p=float(res.pvalue), n=int(x.size + y.size))


def welch_df(a, b) -> float:
    """Welch-Satterthwaite degrees of freedom of welch_t(a, b) (reported beside the statistic)."""
    x = np.asarray(a, dtype=float).ravel()
    y = np.asarray(b, dtype=float).ravel()
    vx, vy = float(np.var(x, ddof=1)) / x.size, float(np.var(y, ddof=1)) / y.size
    denom = vx ** 2 / (x.size - 1) + vy ** 2 / (y.size - 1)
    return float((vx + vy) ** 2 / denom) if denom > 0 else float("nan")
