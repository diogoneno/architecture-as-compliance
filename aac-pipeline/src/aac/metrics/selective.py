"""Selective-prediction metrics (binary, Refuted positive per ) and ECE.

The deferral rule lives here (`threshold_for_coverage`, `defer_mask`) so that the C5 gate and
Table 4.6 apply one rule and can never disagree . An item's key is the pair
(calibrated confidence, raw-score confidence), each max(p, 1 - p) rounded to 6 d.p.; the second
value breaks ties on isotonic plateaus deterministically."""
from __future__ import annotations

import math

import numpy as np

KEY_DECIMALS = 6


def binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    f1s = []
    for pos in (True, False):
        tp = np.sum((y_true == pos) & (y_pred == pos))
        fp = np.sum((y_true != pos) & (y_pred == pos))
        fn = np.sum((y_true == pos) & (y_pred != pos))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


def selective_metrics(y_true: np.ndarray, y_pred: np.ndarray, auto_mask: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    m = np.asarray(auto_mask, dtype=bool)
    n = len(y_true)
    n_auto = int(m.sum())
    if n_auto == 0:
        return {"coverage": 0.0, "macro_f1_binary": float("nan"), "risk": float("nan"), "n_auto": 0}
    return {
        "coverage": n_auto / n if n else 0.0,
        "macro_f1_binary": binary_macro_f1(y_true[m], y_pred[m]),
        "risk": float(np.mean(y_true[m] != y_pred[m])),
        "n_auto": n_auto,
    }


def residual_error_shares(y_true: np.ndarray, y_pred: np.ndarray, auto_mask: np.ndarray) -> dict:
    """FN = Refuted item decided non-Refuted; shares over the errors in the auto-decided subset."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    m = np.asarray(auto_mask, dtype=bool)
    fn = int(np.sum(m & y_true & ~y_pred))
    fp = int(np.sum(m & ~y_true & y_pred))
    total = fn + fp
    return {"fn": fn, "fp": fp, "fn_share": fn / total if total else None, "fp_share": fp / total if total else None}


def ece(probs: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(probs, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(probs[sel].mean() - y[sel].mean())
    return float(total)


def confidence(p) -> np.ndarray:
    """Vectorised max(p, 1 - p)."""
    p = np.asarray(p, dtype=float)
    return np.maximum(p, 1.0 - p)


def _keys(cal_probs, raw_scores) -> tuple[np.ndarray, np.ndarray]:
    c = np.round(confidence(cal_probs), KEY_DECIMALS)
    r = np.round(confidence(raw_scores), KEY_DECIMALS)
    if c.shape != r.shape or c.ndim != 1:
        raise ValueError("cal_probs and raw_scores must be 1-D arrays of equal length")
    if not (np.all(np.isfinite(c)) and np.all(np.isfinite(r))):
        raise ValueError("cal_probs and raw_scores must be finite")
    return c, r


def threshold_for_coverage(cal_probs, raw_scores, coverage: float) -> tuple[float, float] | None:
    """Threshold pair (c, r) that auto-decides `coverage` of the items; None defers nothing.

 Keys sort ascending lexicographically; n_defer = floor((1 - coverage) * n + 0.5). The pair is
 the key of the first auto-decided item (index n_defer). A defer-everything target (n_defer >= n)
 gives the next float above the largest key ( rule), not rounded."""
    coverage = float(coverage)
    if not 0.0 <= coverage <= 1.0:
        raise ValueError(f"coverage target must be within [0, 1], got {coverage}")
    c, r = _keys(cal_probs, raw_scores)
    n = len(c)
    if n == 0:
        raise ValueError("threshold_for_coverage: no calibration items")
    if coverage >= 1.0:
        return None
    n_defer = int(math.floor((1.0 - coverage) * n + 0.5))
    if n_defer <= 0:
        return None
    order = np.lexsort((r, c))  # primary key c, secondary key r, both ascending; stable
    if n_defer >= n:
        last = order[-1]
        return (math.nextafter(float(c[last]), math.inf), math.nextafter(float(r[last]), math.inf))
    i = order[n_defer]
    return (round(float(c[i]), KEY_DECIMALS), round(float(r[i]), KEY_DECIMALS))


def defer_mask(cal_probs, raw_scores, pair) -> np.ndarray:
    """True where the item's key is lexicographically below `pair`; None defers nothing."""
    c, r = _keys(cal_probs, raw_scores)
    if pair is None:
        return np.zeros(len(c), dtype=bool)
    pc, pr = float(pair[0]), float(pair[1])
    return (c < pc) | ((c == pc) & (r < pr))


def risk_coverage_curve(y_true, y_pred, cal_probs, raw_scores) -> tuple[np.ndarray, np.ndarray]:
    """Rank items by key descending (index order breaks full ties); risk at coverage k/n is the
 error rate among the k most confident items, k = 1..n."""
    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    c, r = _keys(cal_probs, raw_scores)
    n = len(c)
    if not (len(y_true) == len(y_pred) == n):
        raise ValueError("y_true, y_pred and the scores must have equal length")
    if n == 0:
        raise ValueError("risk_coverage_curve: no items")
    order = np.lexsort((np.arange(n), -r, -c))
    errors = (y_true != y_pred)[order].astype(float)
    k = np.arange(1, n + 1, dtype=float)
    return k / n, np.cumsum(errors) / k


def aurc(y_true, y_pred, cal_probs, raw_scores) -> float:
    """Area under the risk-coverage curve: mean risk over k = 1..n."""
    _, risk = risk_coverage_curve(y_true, y_pred, cal_probs, raw_scores)
    return float(np.mean(risk))
