"""The single choke point (invariant L1): RecordBuilder.commit is the only producer
of a Decision, and it appends the schema-validated Appendix-B record first."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from aac.run.hashing import claim_hash, record_id as _record_id

_TOKEN = object()

NEUTRAL_RETRIEVAL = {"evidence_ids": [], "evidence_scores": [], "evidence_coverage": False}
NEUTRAL_GATE = {"abstention_flag": False, "abstention_reason": None, "coverage_target": 1.0, "threshold": None}


@dataclass(frozen=True)
class Decision:
    record_id: str
    decision: str
    calibrated_probability: float
    abstention_flag: bool
    _token: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self._token is not _TOKEN:
            raise TypeError("Decision has no public constructor; RecordBuilder.commit() is its only producer")


@dataclass
class RunContext:
    cfg_resolved: dict
    config_hash: str
    data_version: str
    model_version: str
    run_id: str
    sink: Any
    clock: Any
    toggles: Any
    seed: int


class RecordBuilder:
    def __init__(self, ctx: RunContext, req, c1_out: dict, c2_out: dict | None, fuse_out: dict) -> None:
        self.ctx = ctx
        self.req = req
        self.c1_out = c1_out
        self.c2_out = c2_out
        self.fuse_out = fuse_out
        self.cal_out: dict | None = None
        self.gate_out: dict | None = None
        if c2_out is not None and len(c2_out["evidence_ids"]) != len(c2_out["evidence_scores"]):
            raise ValueError("evidence_ids and evidence_scores must have equal length")

    def with_calibration(self, cal_out: dict | None) -> "RecordBuilder":
        self.cal_out = cal_out
        return self

    def with_gate(self, gate_out: dict | None) -> "RecordBuilder":
        self.gate_out = gate_out
        return self

    def _to_appendix_b(self, seq: int, rid: str, timestamp: str) -> dict:
        raw = float(self.fuse_out["raw_score"])
        ev = self.c2_out if self.c2_out is not None else NEUTRAL_RETRIEVAL
        cal = self.cal_out if self.cal_out is not None else {"calibrated_probability": raw, "calibration_method": "none"}
        gate = self.gate_out if self.gate_out is not None else NEUTRAL_GATE
        deferred = bool(gate["abstention_flag"])
        return {
            "record_id": rid,
            "timestamp": timestamp,
            "seq": seq,
            "model_version": self.ctx.model_version,
            "data_version": self.ctx.data_version,
            "config_hash": self.ctx.config_hash,
            "claim_id": self.req.claim_id,
            "claim_hash": claim_hash(self.req.text),
            "encoder_distribution": [float(x) for x in self.c1_out["encoder_distribution"]],
            "evidence_ids": list(ev["evidence_ids"]),
            "evidence_scores": [float(x) for x in ev["evidence_scores"]],
            "evidence_coverage": bool(ev["evidence_coverage"]),
            "fusion_inputs": dict(self.fuse_out["fusion_inputs"]),
            "raw_score": raw,
            "calibrated_probability": float(cal["calibrated_probability"]),
            "calibration_method": cal["calibration_method"],
            "abstention_flag": deferred,
            "abstention_reason": gate["abstention_reason"],
            "coverage_target": float(gate["coverage_target"]),
            "threshold": None if gate["threshold"] is None else float(gate["threshold"]),
            "decision": "abstain-to-human" if deferred else "auto-decide",
            "analyst_queue_id": f"q-{rid[:8]}" if deferred else None,
            "analyst_action": None,
            "analyst_action_timestamp": None,
            "seed": int(self.ctx.seed),
            "regime": self.req.regime,
            "split": self.req.split,
        }

    def commit(self) -> Decision:
        seq = self.ctx.clock.next_seq()
        shard = seq // self.ctx.sink.shard_size
        rid = _record_id(self.ctx.run_id, shard, seq)
        rec = self._to_appendix_b(seq, rid, self.ctx.clock.now_iso_ms())
        self.ctx.sink.append(rec)  # validates; raises SchemaViolation before any Decision exists
        return Decision(rec["record_id"], rec["decision"], rec["calibrated_probability"], rec["abstention_flag"], _token=_TOKEN)
