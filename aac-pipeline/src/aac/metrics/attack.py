"""Attack metrics over attack-ledger rows (contracts/attack-ledger.schema.json).

Conventions (pre-registered specification):
 attacked row queries_used > 0 (a row with 0 queries records an item whose clean decision was
 already wrong or deferred: in the subset, never attacked, excluded from denominators)
 ASR #end_state == success / #attacked. `success` is the strict wrong AUTO-decision:
 for victim `full` a flipped example that the post-hoc gate deferred ends as
 `deferred` / posthoc_gate `deflected`, not as success
 deflection rate #posthoc_gate == deflected / #attacked rows carrying a post-hoc gate (victim full);
 None when no such rows (content_only has no gate: Table 4.4 renders n/a)
 queries/success median queries_used over success rows; None without successes
 BODEGA score mean of the rows' `bodega_score` when present (Phase 4 adds it to the contract);
 None on stub ledgers, rendered n/a
"""
from __future__ import annotations

import statistics


def _rows(rows, name: str) -> list[dict]:
    rows = list(rows)
    if not rows:
        raise ValueError(f"{name}: empty ledger")
    return rows


def attacked(rows) -> list[dict]:
    return [r for r in rows if int(r["queries_used"]) > 0]


def asr(rows) -> float | None:
    att = attacked(_rows(rows, "asr"))
    if not att:
        return None
    return sum(1 for r in att if r["end_state"] == "success") / len(att)


def deflection_rate(rows) -> float | None:
    gated = [r for r in attacked(_rows(rows, "deflection_rate")) if r.get("posthoc_gate") is not None]
    if not gated:
        return None
    return sum(1 for r in gated if r["posthoc_gate"] == "deflected") / len(gated)


def queries_per_success_median(rows) -> float | None:
    q = [int(r["queries_used"]) for r in _rows(rows, "queries_per_success_median") if r["end_state"] == "success"]
    if not q:
        return None
    return float(statistics.median(q))


def bodega_score_mean(rows) -> float | None:
    vals = [float(r["bodega_score"]) for r in _rows(rows, "bodega_score_mean") if r.get("bodega_score") is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)
