"""Table 4.6 (coverage versus risk) and deferral_reasons.csv from the sealed
r1-chrono-full seed-0 run.

Deferral rule : a report threshold that is a two-element list [c, r] (the fitted gate,
FittedGate gate.json) goes through aac.metrics.selective.defer_mask on the records' calibrated_probability
and raw_score, the same rule C5 applied; a float threshold (the stub gate) keeps the earlier rule
max(p, 1 - p) >= threshold; null defers nothing. The Threshold cell shows c (3 d.p.); the long CSV adds the
pair. Deferral reasons follow the gate's reason_precedence (no_evidence, conflicting_evidence,
low_confidence) with the run's relevance floor (evidence_coverage) and conflict floor; at the gate's own
coverage target they equal the logged abstention_reason counts. AURC (methodological note
item-bootstrap CI (seed key boot.table-4.6.aurc) is one non-shell long row."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aac.data.fixtures import load_labels
from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.metrics.selective import aurc, binary_macro_f1, defer_mask, residual_error_shares, selective_metrics
from aac.reports import fmt
from aac.run.hashing import derive_seed
from aac.run.index import load_index
from aac.run.seal import verify_sealed
from aac.stats.bootstrap import ci_index as boot_ci
from aac.run.paths import reports_dir, runs_dir

TABLE = "4.6"
REASONS = "deferral_reasons"
ROWS = (("100", 1.0), ("90", 0.9), ("80 (pre-specified)", 0.8), ("70", 0.7))
AURC_ROW = "AURC (all test items)"
AURC_KEY = "boot.table-4.6.aurc"
REASON_PRECEDENCE = ("no_evidence", "conflicting_evidence", "low_confidence")  # c5_abstention.REASON_PRECEDENCE (no import: architectural isolation constraint)


def auto_mask(thr, prob: np.ndarray, raw: np.ndarray) -> np.ndarray:
    """Auto-decided items under one report threshold: None -> all; [c, r] -> not defer_mask (the C5 rule);
 a float -> max(p, 1 - p) >= thr (the stub gate's earlier rule)."""
    prob = np.asarray(prob, dtype=float)
    if thr is None:
        return np.ones(len(prob), dtype=bool)
    if isinstance(thr, (list, tuple)):
        if len(thr) != 2:
            raise ValueError(f"report threshold pair must have two elements, got {thr!r}")
        return ~defer_mask(prob, raw, thr)
    return np.maximum(prob, 1.0 - prob) >= float(thr)


def deferral_reason(rec: dict, conflict_floor: float, precedence=REASON_PRECEDENCE) -> str:
    """The reason of a deferred record: the logged abstention_reason when C5 deferred it, else the first reason of
 `precedence` that applies (no_evidence: evidence_coverage false; conflicting_evidence: support_agg and
 refute_agg both at or above the conflict floor; low_confidence otherwise)."""
    if rec["abstention_flag"] and rec["abstention_reason"]:
        return rec["abstention_reason"]
    fi = rec["fusion_inputs"]
    applies = {"no_evidence": not rec["evidence_coverage"],
               "conflicting_evidence": fi["support_agg"] >= conflict_floor and fi["refute_agg"] >= conflict_floor,
               "low_confidence": True}
    for reason in precedence:
        if applies[reason]:
            return reason
    return "low_confidence"


def select_run(root: Path, *, regime: str = "R1", system: str = "full", seed: int = 0, name: str = "r1-chrono-full") -> dict:
    runs = [r for r in load_index(runs_dir(root) / "INDEX.json")["runs"]
            if r["regime"] == regime and r["system"] == system and r["seed"] == seed and r.get("name") == name]
    if len(runs) != 1:
        raise RuntimeError(f"expected exactly one sealed {regime}/{system}/{name} seed {seed} run in runs/INDEX.json, found {len(runs)}")
    return runs[0]


def load_test_records(run_dir: Path) -> tuple[list[dict], str]:
    shards = list_shards(run_dir / SHARD_DIR)
    recs = [r for p in shards for r in read_shard(p) if r["split"] == "test"]
    # shard order (part-0000, part-0001, ...), never hash-value order: a hash-sorted list would reorder whenever
    # the shard bytes move (e.g. a different log clock), although only the hash values themselves should move
    sidecars = [p.with_name(p.name + ".sha256").read_text(encoding="ascii").split()[0] for p in shards]
    return sorted(recs, key=lambda r: r["seq"]), ";".join(sidecars)


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    entry = select_run(root)
    run_dir = runs_dir(root) / entry["run_id"]
    sealed = verify_sealed(run_dir)
    if sealed != entry["sealed_sha256"]:
        raise RuntimeError(f"{entry['run_id']}: SEALED {sealed[:12]} disagrees with runs/INDEX.json")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    gate = json.loads((run_dir / "gate.json").read_text(encoding="utf-8"))
    recs, shard_hashes = load_test_records(run_dir)
    labels = load_labels(root / resolved["data"]["labels"])
    ch, dv = entry["config_hash"], resolved["data_version"]
    n_boot = int(resolved["stats"]["bootstrap"])
    level = float(resolved["stats"]["level"])  
    y_true = np.array([labels[r["claim_id"]] == "Refuted" for r in recs])
    prob = np.array([r["calibrated_probability"] for r in recs])
    raw = np.array([r["raw_score"] for r in recs])
    y_pred = prob >= 0.5
    precedence = tuple(gate.get("reason_precedence") or REASON_PRECEDENCE)
    reasons = {}
    shell_rows, long_rows, reason_rows, reason_long = [], [], [], []
    fn_share_100 = None
    common = dict(config_hash=ch, data_version=dv, run_ids=entry["run_id"], source_shards_sha256=shard_hashes)
    for label, cov in ROWS:
        thr = gate["report_thresholds"][f"{cov:.2f}"]
        mask = auto_mask(thr, prob, raw)
        if isinstance(thr, (list, tuple)) and abs(cov - float(gate.get("coverage_target", -1))) < 1e-9:
            logged = np.array([not r["abstention_flag"] for r in recs])
            if not np.array_equal(mask, logged):
                raise RuntimeError(f"{entry['run_id']}: Table 4.6 deferral at the gate's own target {cov:.2f} disagrees with the "
                                   f"logged abstention_flag on {int(np.sum(mask != logged))} items (one rule for C5 and Table 4.6)")
        sm = selective_metrics(y_true, y_pred, mask)
        shares = residual_error_shares(y_true, y_pred, mask)
        idx_auto = np.flatnonzero(mask)
        lo, hi = boot_ci(lambda idx: binary_macro_f1(y_true[idx_auto][idx], y_pred[idx_auto][idx]), len(idx_auto),
                         n_resamples=n_boot, seed=derive_seed(f"boot.table-4.6.{label}.selective_macro_f1"), level=level)
        deferred = [r for r, m in zip(recs, mask) if not m]
        # deferral reasons at this coverage: re-derive the reason from the logged record where the
        # logged threshold applies (80), else classify by the logged no_evidence/conflict signals
        counts = {"low_confidence": 0, "no_evidence": 0, "conflicting_evidence": 0}
        for r in deferred:
            counts[deferral_reason(r, gate["conflict_floor"], precedence)] += 1
        n_def = len(deferred)
        reasons[label] = counts
        fn_share = shares["fn_share"]
        if cov == 1.0:
            fn_share_100 = fn_share
            thr_cell, delta_cell = "none", "reference"
            cov_cell = "100"
        else:
            thr_cell = "none" if thr is None else f"{float(thr[0] if isinstance(thr, (list, tuple)) else thr):.3f}"
            delta_cell = "n/a" if fn_share is None or fn_share_100 is None else fmt.fmt_pp(fn_share - fn_share_100)
            cov_cell = fmt.fmt_rate(sm["coverage"])
        row = {
            "coverage_target": label,
            "threshold": thr_cell,
            "coverage_achieved": cov_cell,
            "selective_macro_f1": fmt.fmt_ci(sm["macro_f1_binary"], lo, hi),
            "selective_risk": fmt.fmt_rate(sm["risk"]),
            "fn_share_residual": "n/a" if fn_share is None else fmt.fmt_rate(fn_share),
            "fp_share_residual": "n/a" if shares["fp_share"] is None else fmt.fmt_rate(shares["fp_share"]),
            "delta_fn_share_vs_100": delta_cell,
        }
        shell_rows.append(row)
        n_auto = sm["n_auto"]
        L = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, label, col, metric, value, n_items=n_auto, n_seeds=1, seeds="0", **common, **kw))
        L("Threshold", "threshold_confidence", thr_cell)
        if isinstance(thr, (list, tuple)):
            L("Threshold pair (calibrated confidence; raw-score confidence)", "threshold_pair_c_r", f"{float(thr[0]):.6f}; {float(thr[1]):.6f}",
              test="gate.json report_thresholds pair; an item is deferred when its (calibrated, raw-score) confidence key is "
                   "lexicographically below the pair (aac.metrics.selective.defer_mask, the C5 rule)")
        L("Coverage achieved (percent)", "coverage_achieved_pct", cov_cell)
        L("Selective macro-F1 [95% CI]", "selective_macro_f1_binary", fmt.fmt_f1(sm["macro_f1_binary"]), ci_lo=fmt.fmt_f1(lo), ci_hi=fmt.fmt_f1(hi), test="percentile bootstrap", family="R1")
        L("Selective risk", "selective_risk_pct", row["selective_risk"])
        L("False-negative share of residual errors (percent)", "fn_share_residual_pct", row["fn_share_residual"])
        L("False-positive share (percent)", "fp_share_residual_pct", row["fp_share_residual"])
        L("Change in false-negative share vs 100 percent", "delta_fn_share_vs_100_pp", delta_cell)
        rr = {"coverage_target": label}
        for key, reason in (("reason_low_conf", "low_confidence"), ("reason_no_evidence", "no_evidence"), ("reason_conflict", "conflicting_evidence")):
            rr[key] = "n/a" if n_def == 0 else fmt.fmt_rate(counts[reason] / n_def)
        reason_rows.append(rr)
        for col, key in (("Low confidence (percent of deferrals)", "reason_low_conf"),
                         ("No evidence above the relevance floor (percent of deferrals)", "reason_no_evidence"),
                         ("Conflicting evidence (percent of deferrals)", "reason_conflict")):
            reason_long.append(fmt.long_row(REASONS, label, col, key + "_pct", rr[key], n_items=n_def, n_seeds=1, seeds="0", **common))
    # AURC over all test items (methodological note)
    n_all = len(recs)
    aurc_val = aurc(y_true, y_pred, prob, raw)
    a_lo, a_hi = boot_ci(lambda idx: aurc(y_true[idx], y_pred[idx], prob[idx], raw[idx]), n_all,
                         n_resamples=n_boot, seed=derive_seed(AURC_KEY), level=level)
    long_rows.append(fmt.long_row(TABLE, AURC_ROW, "AURC [95% CI]", "aurc", fmt.fmt_metric(aurc_val), ci_lo=fmt.fmt_metric(a_lo),
                                  ci_hi=fmt.fmt_metric(a_hi), n_items=n_all, n_seeds=1, seeds="0",
                                  test=(f"area under the selective-risk curve, mean risk over k = 1..n; "
                                        f"percentile bootstrap over items, {n_boot} resamples, seed key {AURC_KEY}"), **common))
    p1, l1 = fmt.emit(TABLE, shell_rows, long_rows, config_hash=ch, data_version=dv, out_dir=out_dir)
    p2, l2 = fmt.emit(REASONS, reason_rows, reason_long, config_hash=ch, data_version=dv, out_dir=out_dir)
    return [p1, l1, p2, l2]

