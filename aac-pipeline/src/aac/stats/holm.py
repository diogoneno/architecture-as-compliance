"""Holm step-down correction within a specification family (pre-registered specification)."""
from __future__ import annotations

import numpy as np
from statsmodels.stats.multitest import multipletests


def holm(p_values) -> list[float]:
    """Adjusted p-values in the CALLER'S input order (statsmodels multipletests, method="holm")."""
    p = np.asarray(list(p_values), dtype=float).ravel()
    if p.size == 0:
        raise ValueError("holm: empty family")
    if np.any(~np.isfinite(p)) or np.any(p < 0) or np.any(p > 1):
        raise ValueError("holm: p-values must be finite and within [0, 1]")
    _, adjusted, _, _ = multipletests(p, method="holm")
    return [float(v) for v in adjusted]


def holm_family(tests: dict, ordered: list[str]) -> dict[str, float]:
    """Holm-adjusted p for every test id in `ordered` (the family's computed tests, in specification order).
 A degenerate test (no p, e.g. zero variance) enters the family as p = 1.0 instead of being
 dropped dropping it would shrink m and make the correction less conservative."""
    if not ordered:
        return {}
    ps = [1.0 if (tests[t].degenerate or tests[t].p is None) else float(tests[t].p) for t in ordered]
    return dict(zip(ordered, holm(ps)))
