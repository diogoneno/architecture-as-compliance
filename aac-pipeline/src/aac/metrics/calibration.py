"""Calibration metrics: ECE with 15 equal-width bins on 1-D P(false) via netcal
(pre-registered specification
reliability-bin table for Figure 4.1.

netcal ECE(15) and the right-closed 15-bin reference agree except when calibrated probabilities
lie exactly on a k/15 bin edge, where the two assign the item to neighbouring bins. Example
(03-VALUE-AUDIT-0314 observation 1): R1-114138606e-s3 has 18 calibrated probabilities on an edge;
the reference gives 0.0369 and netcal 0.0268. The tables use netcal (pre-registered specification
10a ("1-D netcal == manual 15-bin") holds only when no value sits on a k/15 edge.

netcal is imported here and nowhere else in aac."""
from __future__ import annotations

import numpy as np
from netcal.metrics import ECE

BINS = 15


def _pair(p, y, name: str) -> tuple[np.ndarray, np.ndarray]:
    pa = np.asarray(p, dtype=float).ravel()
    ya = np.asarray(y, dtype=float).ravel()
    if pa.size == 0 or ya.size == 0:
        raise ValueError(f"{name}: empty input")
    if pa.shape != ya.shape:
        raise ValueError(f"{name}: inputs must have the same length")
    if np.any(pa < 0.0) or np.any(pa > 1.0):
        raise ValueError(f"{name}: probabilities outside [0, 1]")
    return pa, ya


def ece15(p_false, y_binary) -> float:
    """netcal.metrics.ECE(bins=15).measure on 1-D P(false) and binary labels."""
    p, y = _pair(p_false, y_binary, "ece15")
    return float(ECE(bins=15).measure(p, y.astype(int)))


def ece_reference(p, y, bins: int = BINS) -> float:
    """Own 15-bin equal-width ECE, bins closed on the right (test oracle for ece15; differs from netcal only
 when a probability lies exactly on a k/15 edge, see the module docstring)."""
    pa, ya = _pair(p, y, "ece_reference")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(pa, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            total += sel.mean() * abs(pa[sel].mean() - ya[sel].mean())
    return float(total)


def reliability_bins(p, y, bins: int = BINS) -> list[dict]:
    """One row per bin: {bin, lo, hi, n, confidence, accuracy}; empty bins carry None."""
    pa, ya = _pair(p, y, "reliability_bins")
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(pa, edges[1:-1], right=True), 0, bins - 1)
    rows = []
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        rows.append({
            "bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": n,
            "confidence": float(pa[sel].mean()) if n else None,
            "accuracy": float(ya[sel].mean()) if n else None,
        })
    return rows
