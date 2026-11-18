"""Residual-error composition of the auto-decided subset (Table 4.6 FN/FP shares)."""
from __future__ import annotations

import numpy as np

from aac.metrics.selective import residual_error_shares


def fn_fp_shares(y_true, y_pred, auto_mask) -> dict:
    """{fn, fp, fn_share, fp_share}: FN = Refuted item auto-decided non-Refuted; shares over
 the residual errors of the auto-decided subset (None when there are no errors)."""
    y = np.asarray(y_true).ravel()
    if y.size == 0:
        raise ValueError("fn_fp_shares: empty input")
    p = np.asarray(y_pred).ravel()
    m = np.asarray(auto_mask).ravel()
    if not (y.shape == p.shape == m.shape):
        raise ValueError("fn_fp_shares: inputs must have the same length")
    return residual_error_shares(y.astype(bool), p.astype(bool), m.astype(bool))


def delta_fn_share(fn_share: float | None, fn_share_at_full_coverage: float | None) -> float | None:
    """FN share at a coverage target minus the FN share at 100 percent coverage (n/a if either is None)."""
    if fn_share is None or fn_share_at_full_coverage is None:
        return None
    return float(fn_share) - float(fn_share_at_full_coverage)
