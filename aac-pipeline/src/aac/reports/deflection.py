"""Deflection references for Table 4.4 (7, gap-review item 1). Zero GPU: every value
is read from sealed records.

The registered H3-deflection cell (deflected / attacked gated victim-B rows, CI excludes 0) stays as registered.
Victim B is attacked UN-gated and its seed-0 calibrator can put a first flip on a plateau inside the deferral band,
so a deflection rate alone cannot say whether the gate stopped an attack or the attack merely crossed into the
band. The five references printed beside it (long CSV, family `R3-supplementary`):

 (a) deflection minus clean deferral on the same rows: the deflection rate minus the share of the attacked rows
 whose item was already deferred when clean (the victim-source R1 run's test record of that item); paired
 item-cluster bootstrap (all rows of an item travel together)
 (b) deflection on clean-auto-decided items: the deflection rate over the rows whose item the gate auto-decided
 when clean (clean deferrals cannot inflate it)
 (c) abstention reasons of the deflected rows, from each row's post-hoc record (no_evidence,
 conflicting_evidence, low_confidence): only the evidence reasons can be attack-specific beyond the band
 (d) conditional deflection = deflected / (deflected + strict success): of the attacks that flipped the
 un-gated decision, the share the gate caught
 (e) matched-confidence control: each post-hoc final is matched to the clean test records with the SAME
 calibrated value (an isotonic map has discrete plateaus; the 15 ECE bins would mix deferred and auto-decided
 plateaus) and the reference is the observed deflection minus the clean deferral rate at that value; for a
 temperature map (continuous values) the match falls back to the 15 equal-width ECE bins of calibration.json

Post-hoc records are the LAST record id of a ledger row whose posthoc_gate is flip or deflected (the victim
server lists an item's record ids in order: clean, attack queries, post-hoc). This module reads only records
(claim_hash, never text) and calibration/gate files; the caller supplies them."""
from __future__ import annotations

import numpy as np

from aac.stats.bootstrap import cluster_ci

REASONS = ("no_evidence", "conflicting_evidence", "low_confidence")
GATED = ("flip", "deflected")


class DeflectionInputError(ValueError):
    """Inputs that cannot support a reference (an attacked item without a clean record, an inconsistent record)."""


def posthoc_record_id(row: dict) -> str | None:
    """The post-hoc record id of a ledger row (its last record id) when the server ran the gated query."""
    return row["record_ids"][-1] if row.get("posthoc_gate") in GATED else None


def attacked_b(rows) -> list[dict]:
    return [r for r in rows if r["victim"] == "full" and int(r["queries_used"]) > 0 and r.get("posthoc_gate") is not None]


def _item(r: dict) -> tuple[str, str]:
    return (r["subset"], r["item_id"])


def _groups(rows: list[dict], columns) -> tuple[list[np.ndarray], list]:
    """Rows grouped by item (first-appearance order) as arrays of the given per-row columns."""
    order, by = [], {}
    for r in rows:
        k = _item(r)
        if k not in by:
            by[k] = []
            order.append(k)
        by[k].append([float(c(r)) for c in columns])
    return [np.asarray(by[k], dtype=float) for k in order], order


def _match_key(value: float, method: str, bins: int):
    v = round(float(value), 6)
    if method == "isotonic":
        return ("plateau", v)
    return ("bin", min(int(v * bins), bins - 1))


def deflection_references(victim_b_rows, posthoc_records: dict, clean_records, calibration: dict, gate: dict, *,
                          cell_key: str, n_resamples: int, level: float) -> dict:
    """References (a)-(e) over the attacked victim-B rows (victim full, queries_used > 0, posthoc_gate set).
 posthoc_records: {record_id: record} holding every post-hoc record of those rows. clean_records: the test-split
 records of the victim-source R1 run (all of them; (a), (b) use the attacked items', (e) all). Returns
 {"a".."e": {...}} with value(s), CI, counts and a `rule` text; a reference without a defined value has
 value None and a `why`."""
    rows = attacked_b(victim_b_rows)
    if not rows:
        raise DeflectionInputError("no attacked victim-B rows with a post-hoc gate")
    clean = {}
    for rec in clean_records:
        if rec["claim_id"] in clean:
            raise DeflectionInputError(f"two clean test records for {rec['claim_id']}")
        clean[rec["claim_id"]] = rec
    missing = sorted({r["item_id"] for r in rows if r["item_id"] not in clean})
    if missing:
        raise DeflectionInputError(f"{len(missing)} attacked victim-B items have no clean test record (e.g. {missing[0]})")
    deflected = lambda r: 1.0 if r["posthoc_gate"] == "deflected" else 0.0  # noqa: E731
    success = lambda r: 1.0 if r["end_state"] == "success" else 0.0  # noqa: E731
    clean_def = lambda r: 1.0 if clean[r["item_id"]]["abstention_flag"] else 0.0  # noqa: E731
    kw = dict(n_resamples=n_resamples, level=level)
    n_items = len({_item(r) for r in rows})
    out: dict[str, dict] = {}

    # (a) deflection minus clean deferral on the same rows
    groups, _ = _groups(rows, (deflected, clean_def))
    stat_a = lambda m: float(m[:, 0].mean() - m[:, 1].mean())  # noqa: E731
    d_rate = float(np.mean([deflected(r) for r in rows]))
    c_rate = float(np.mean([clean_def(r) for r in rows]))
    out["a"] = {"value": d_rate - c_rate, "deflection": d_rate, "clean_deferral": c_rate,
                "ci": cluster_ci(groups, stat_a, cell_key=f"{cell_key}.ref-a", **kw), "n_rows": len(rows), "n_items": n_items,
                "rule": "deflection rate minus the clean deferral rate of the same attacked victim-B rows (clean = the "
                        "victim-source R1 test record of the item); paired item-cluster bootstrap"}

    # (b) deflection on items auto-decided when clean
    auto = [r for r in rows if not clean[r["item_id"]]["abstention_flag"]]
    if auto:
        g_b, _ = _groups(auto, (deflected,))
        out["b"] = {"value": float(np.mean([deflected(r) for r in auto])),
                    "ci": cluster_ci(g_b, lambda m: float(m.mean()), cell_key=f"{cell_key}.ref-b", **kw),
                    "n_rows": len(auto), "n_items": len({_item(r) for r in auto}),
                    "rule": "deflection rate over the attacked victim-B rows whose item was auto-decided when clean"}
    else:
        out["b"] = {"value": None, "ci": None, "n_rows": 0, "n_items": 0, "why": "n/a (no clean-auto-decided item)",
                    "rule": "deflection rate over the attacked victim-B rows whose item was auto-decided when clean"}

    # (c) abstention reasons of the deflected rows (post-hoc records)
    counts = {k: 0 for k in REASONS}
    n_def = 0
    for r in rows:
        if r["posthoc_gate"] != "deflected":
            continue
        rec = posthoc_records.get(posthoc_record_id(r))
        if rec is None:
            raise DeflectionInputError(f"no post-hoc record for {r['item_id']} / {r['attack']}")
        if not rec["abstention_flag"] or rec["abstention_reason"] not in counts:
            raise DeflectionInputError(f"post-hoc record of a deflected row ({r['item_id']} / {r['attack']}) carries "
                                       f"abstention_flag {rec['abstention_flag']} / reason {rec['abstention_reason']!r}")
        counts[rec["abstention_reason"]] += 1
        n_def += 1
    out["c"] = {"counts": counts, "n_deflected": n_def,
                "rule": "abstention_reason of the post-hoc record of each deflected victim-B row"}

    # (d) conditional deflection
    caught = [r for r in rows if r["posthoc_gate"] == "deflected" or r["end_state"] == "success"]
    if caught:
        g_d, _ = _groups(caught, (deflected,))
        out["d"] = {"value": float(np.mean([deflected(r) for r in caught])),
                    "ci": cluster_ci(g_d, lambda m: float(m.mean()), cell_key=f"{cell_key}.ref-d", **kw),
                    "n_rows": len(caught), "n_deflected": int(sum(deflected(r) for r in caught)),
                    "n_success": int(sum(success(r) for r in caught)),
                    "rule": "deflected / (deflected + strict success) over the attacked victim-B rows"}
    else:
        out["d"] = {"value": None, "ci": None, "n_rows": 0, "n_deflected": 0, "n_success": 0,
                    "why": "n/a (no deflected or successful item)",
                    "rule": "deflected / (deflected + strict success) over the attacked victim-B rows"}

    # (e) matched-confidence control
    method = str(calibration.get("selected"))
    bins = int(calibration.get("bins") or 15)
    pool: dict = {}
    for rec in clean.values():
        pool.setdefault(_match_key(rec["calibrated_probability"], method, bins), []).append(1.0 if rec["abstention_flag"] else 0.0)
    finals, unmatched = [], 0
    for r in rows:
        rid = posthoc_record_id(r)
        if rid is None:
            continue
        rec = posthoc_records.get(rid)
        if rec is None:
            raise DeflectionInputError(f"no post-hoc record for {r['item_id']} / {r['attack']}")
        if bool(rec["abstention_flag"]) != (r["posthoc_gate"] == "deflected"):
            raise DeflectionInputError(f"post-hoc record of {r['item_id']} / {r['attack']} disagrees with posthoc_gate")
        key = _match_key(rec["calibrated_probability"], method, bins)
        if key not in pool:
            unmatched += 1
            continue
        finals.append({**r, "_matched": float(np.mean(pool[key])), "_key": key})
    match_rule = ("exact calibrated value (isotonic plateau) among the clean test records" if method == "isotonic"
                  else f"{bins} equal-width ECE bins of the calibrated value among the clean test records "
                       f"(calibration {method}: no plateaus)")
    if finals:
        g_e, _ = _groups(finals, (deflected, lambda r: r["_matched"]))
        obs = float(np.mean([deflected(r) for r in finals]))
        exp = float(np.mean([r["_matched"] for r in finals]))
        out["e"] = {"value": obs - exp, "observed": obs, "matched_clean_deferral": exp,
                    "ci": cluster_ci(g_e, stat_a, cell_key=f"{cell_key}.ref-e", **kw), "n_rows": len(finals),
                    "n_unmatched": unmatched, "method": method, "match": match_rule,
                    "rule": "deflected share of the post-hoc finals minus the clean deferral rate at the same "
                            "calibrated value; match on the " + match_rule}
    else:
        out["e"] = {"value": None, "ci": None, "n_rows": 0, "n_unmatched": unmatched, "method": method,
                    "match": match_rule, "why": "n/a (no post-hoc final with a matched clean record)",
                    "rule": "deflected share of the post-hoc finals minus the clean deferral rate at the same "
                            "calibrated value; match on the " + match_rule}
    out["gate_threshold"] = gate.get("threshold")
    return out
