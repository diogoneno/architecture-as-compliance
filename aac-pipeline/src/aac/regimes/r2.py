"""R2 (cross-dataset, zero-shot): C4/C5 fitted on the AVeriTeC val_cal partition, then the
InCrediblAE-shaped targets. FC: the item's supplied evidence replaces retrieval (C1 sees the
claim only). RD: retrieval over the pooled fixture store with evidence_coverage reported
(NullEvidence is not an RD strategy).

Scorer truncation ((a), 2026-10-17): the real C2 may cut an RD text that
alone does not fit the scorer window, for the scorer pair only. The Appendix B record schema is closed and frozen at
v1.0, so the per-record flag is the sealed sidecar scorer-truncation.json (one row per record with a cut pair,
keyed by claim_id and split, as the record is) and the counts go into run-record items:
scorer_truncated_<split> (records) and scorer_truncated_pairs_<split> (pairs). A run whose C2 has no such counter
(stub, content-only) writes neither."""
from __future__ import annotations

from pathlib import Path

from aac.pipeline import Pipeline, RunContext
from aac.regimes.common import TRUNCATION_FILE, TRUNCATION_RULE, evaluate_targets, targets_of  # noqa: F401 (re-export)
from aac.regimes.common import write_scorer_truncations

__all__ = ["TRUNCATION_FILE", "TRUNCATION_RULE", "run", "write_truncations"]


def write_truncations(pipeline: Pipeline, run_dir: Path, summary: dict, splits: list[str]) -> None:
    """The scorer-truncation sidecar and its run-record counts (module docstring; aac.regimes.common)."""
    write_scorer_truncations(pipeline, run_dir, summary, splits)


def run(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path) -> dict:
    targets = targets_of(resolved, ["fc", "rd"])
    bad = [t for t in targets if t not in ("fc", "rd")]
    if bad:
        raise ValueError(f"R2 targets are fc and rd only; got {bad}")
    summary = evaluate_targets(pipeline, ctx, resolved, root, run_dir, {t: t for t in targets})
    write_truncations(pipeline, run_dir, summary, list(targets))
    return summary
