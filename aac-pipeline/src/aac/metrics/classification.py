"""Classification metrics via scikit-learn (: binary headline, Refuted positive;
four-way only for Table 4.1 row 6; AUROC on P(false))."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import f1_score, roc_auc_score


def _pair(y_true, y_other, name: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true).ravel()
    b = np.asarray(y_other).ravel()
    if a.size == 0 or b.size == 0:
        raise ValueError(f"{name}: empty input")
    if a.shape != b.shape:
        raise ValueError(f"{name}: inputs must have the same length")
    return a, b


def macro_f1_binary(y_true, decisions) -> float:
    """Macro-F1 over {Refuted (1), rest (0)}; `decisions` are the binary auto-decisions."""
    y, d = _pair(y_true, decisions, "macro_f1_binary")
    return float(f1_score(y.astype(int), d.astype(int), average="macro", labels=[0, 1], zero_division=0.0))


def macro_f1_4way(y_true, argmax) -> float:
    """Macro-F1 over the four AVeriTeC labels (indices 0..3) from the encoder argmax."""
    y, p = _pair(y_true, argmax, "macro_f1_4way")
    return float(f1_score(y.astype(int), p.astype(int), average="macro", labels=[0, 1, 2, 3], zero_division=0.0))


def auroc(y_true_binary, p_false) -> float:
    y, p = _pair(y_true_binary, p_false, "auroc")
    if len(np.unique(y.astype(int))) < 2:
        raise ValueError("auroc: both classes must be present")
    return float(roc_auc_score(y.astype(int), p.astype(float)))


def auroc_fast(y_true_binary, p_false) -> float:
    """AUROC by the Mann-Whitney rank statistic (average ranks on ties): equal to sklearn's
 roc_auc_score (asserted in tests/unit/test_metrics.py) and cheap enough for 1,000 bootstrap
 resamples per cell. Raises when one class is absent, like auroc."""
    y, p = _pair(y_true_binary, p_false, "auroc_fast")
    y = y.astype(bool)
    n_pos = int(y.sum())
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError("auroc_fast: both classes must be present")
    ranks = rankdata(p.astype(float), method="average")
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))
