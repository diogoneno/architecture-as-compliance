"""C5: confidence threshold fitted on val_cal for the coverage target; sets abstention_flag /
reason (low_confidence | no_evidence | conflicting_evidence). StubGate thresholds calibrated
confidence alone; FittedGate uses the shared deferral rule of
aac.metrics.selective with the raw-confidence tie-break, the pause setting and written reasons."""
from __future__ import annotations

import math

import numpy as np

from aac.metrics.selective import defer_mask, threshold_for_coverage


def confidence(p: float) -> float:
    return max(p, 1.0 - p)


def fit_threshold(probs: list[float], coverage: float) -> float | None:
    """Threshold on confidence that keeps `coverage` of the items (None for 1.0).

 when the target defers every item (n_defer >= n, e.g. coverage 0.0 or a tiny
 calibration set) the threshold is the next float above the largest confidence, so every
 item falls below it; it is not rounded (rounding could land back on that confidence)."""
    if not 0.0 <= coverage:
        raise ValueError(f"coverage target must be within [0, 1], got {coverage}")
    if coverage >= 1.0:
        return None
    conf = np.sort(np.asarray([confidence(p) for p in probs], dtype=float))
    if conf.size == 0:
        raise ValueError("fit_threshold: no calibration items")
    n_defer = int(np.floor((1.0 - coverage) * len(conf) + 0.5))
    if n_defer <= 0:
        return None
    if n_defer >= len(conf):
        return float(np.nextafter(conf[-1], np.inf))
    return round(float(conf[n_defer]), 6)


class StubGate:
    name = "stub-gate-v1"

    def __init__(self, cfg: dict) -> None:
        self.coverage_target = float(cfg["coverage_target"])
        self.report_coverages = [float(c) for c in cfg["report_coverages"]]
        self.conflict_floor = float(cfg["conflict_floor"])
        self.threshold: float | None = None
        self.report: dict = {}

    def fit(self, probs: list[float], raw: list[float] | None = None) -> dict:
        """`raw` is accepted and ignored, so a regime calls StubGate and FittedGate the same way."""
        self.threshold = fit_threshold(probs, self.coverage_target)
        self.report = {
            "coverage_target": self.coverage_target,
            "threshold": self.threshold,
            "n_val_cal": len(probs),
            "report_thresholds": {f"{c:.2f}": fit_threshold(probs, c) for c in self.report_coverages},
            "conflict_floor": self.conflict_floor,
        }
        return self.report

    def load(self, report: dict) -> dict:
        """Restore a fitted threshold from a sealed gate.json."""
        self.threshold = None if report["threshold"] is None else float(report["threshold"])
        self.coverage_target = float(report["coverage_target"])
        self.report = dict(report)
        return self.report

    def apply(self, prob: float, c2_out: dict | None, fuse_out: dict) -> dict:
        if self.threshold is None and not self.report:
            raise RuntimeError("StubGate.apply before fit")
        flag = self.threshold is not None and confidence(prob) < self.threshold
        reason = None
        if flag:
            fi = fuse_out["fusion_inputs"]
            if c2_out is not None and not c2_out["evidence_coverage"]:
                reason = "no_evidence"
            elif fi["support_agg"] >= self.conflict_floor and fi["refute_agg"] >= self.conflict_floor:
                reason = "conflicting_evidence"
            else:
                reason = "low_confidence"
        return {"abstention_flag": flag, "abstention_reason": reason,
                "coverage_target": self.coverage_target, "threshold": self.threshold}


# --- : the fitted abstention gate ----------------------------------------

REASON_PRECEDENCE = ("no_evidence", "conflicting_evidence", "low_confidence")
REASON_DEFINITIONS = {
    "no_evidence": ("No retained passage reaches the relevance floor chosen on val_cal, so "
                    "evidence_coverage is false (retrieval on)."),
    "conflicting_evidence": "support_agg and refute_agg are both at or above the conflict floor.",
    "low_confidence": ("The calibrated confidence is below the threshold and neither evidence "
                       "reason applies."),
}
PAUSE_THRESHOLD = math.nextafter(1.0, math.inf)


class FittedGate:
    """C5 for `c5.gate: fitted`. The key of an item is (calibrated confidence, raw-score confidence);
 the threshold pair comes from threshold_for_coverage on val_cal and the deferral from defer_mask,
 the same functions Table 4.6 uses. The pause setting is the flag `pause`, never a number: it
 defers every item, and the record threshold is math.nextafter(1.0, inf)."""

    name = "fitted-gate-v1"

    def __init__(self, cfg: dict) -> None:
        if cfg["tie_break"] != "raw-confidence":
            raise ValueError("c5.tie_break must be 'raw-confidence'")
        if not isinstance(cfg["pause"], bool):
            raise ValueError("c5.pause must be true or false")
        self.coverage_target = float(cfg["coverage_target"])
        if not 0.0 <= self.coverage_target <= 1.0:
            raise ValueError("c5.coverage_target must be within [0, 1]")
        self.report_coverages = [float(c) for c in cfg["report_coverages"]]
        self.fit_on = str(cfg["fit_on"])
        self.conflict_floor = float(cfg["conflict_floor"])
        self.tie_break = "raw-confidence"
        self.pause = bool(cfg["pause"])
        self.pair: tuple[float, float] | None = None
        self._fitted = False
        self.report: dict = {}

    def fit(self, probs: list[float], raw: list[float]) -> dict:
        cal = np.asarray(probs, dtype=float)
        rs = np.asarray(raw, dtype=float)
        if cal.shape != rs.shape:
            raise ValueError("FittedGate.fit: probs and raw must have equal length")
        pair = threshold_for_coverage(cal, rs, self.coverage_target)
        thresholds: dict = {}
        achieved: dict = {}
        for cov in sorted({*self.report_coverages, self.coverage_target}, reverse=True):
            p = threshold_for_coverage(cal, rs, cov)
            key = f"{cov:.2f}"
            if cov in self.report_coverages:
                thresholds[key] = None if p is None else [p[0], p[1]]
            achieved[key] = round(float(1.0 - defer_mask(cal, rs, p).mean()), 6)
        report = {
            "gate": self.name, "coverage_target": self.coverage_target,
            "threshold": None if pair is None else pair[0],
            "threshold_pair": None if pair is None else [pair[0], pair[1]],
            "report_thresholds": thresholds,
            "coverage_achieved": achieved, "n_val_cal": int(len(cal)), "fit_on": self.fit_on,
            "conflict_floor": self.conflict_floor, "tie_break": self.tie_break, "pause": self.pause,
            "pause_rule": "pause true defers every item; the record threshold is math.nextafter(1.0, inf)",
            "reason_precedence": list(REASON_PRECEDENCE), "reason_definitions": dict(REASON_DEFINITIONS),
        }
        return self.load(report)

    def load(self, report: dict) -> dict:
        """Restore a fitted gate from a sealed gate.json."""
        if report["tie_break"] != "raw-confidence" or tuple(report["reason_precedence"]) != REASON_PRECEDENCE:
            raise ValueError("gate report does not match the raw-confidence tie-break and reason precedence")
        tp = report["threshold_pair"]
        self.pair = None if tp is None else (float(tp[0]), float(tp[1]))
        self.coverage_target = float(report["coverage_target"])
        self.conflict_floor = float(report["conflict_floor"])
        self.pause = bool(report["pause"])
        self._fitted = True
        self.report = dict(report)
        return self.report

    def reason(self, c2_out: dict | None, fuse_out: dict) -> str:
        fi = fuse_out["fusion_inputs"]
        if c2_out is not None and not c2_out["evidence_coverage"]:
            return "no_evidence"
        if fi["support_agg"] >= self.conflict_floor and fi["refute_agg"] >= self.conflict_floor:
            return "conflicting_evidence"
        return "low_confidence"

    def apply(self, prob: float, c2_out: dict | None, fuse_out: dict) -> dict:
        if not self._fitted:
            raise RuntimeError("FittedGate.apply before fit or load")
        flag: bool
        threshold: float | None
        if self.pause:
            flag = True
            threshold = PAUSE_THRESHOLD
        else:
            flag = bool(defer_mask([prob], [fuse_out["raw_score"]], self.pair)[0])
            threshold = None if self.pair is None else self.pair[0]
        return {"abstention_flag": flag, "abstention_reason": self.reason(c2_out, fuse_out) if flag else None,
                "coverage_target": self.coverage_target, "threshold": threshold}
