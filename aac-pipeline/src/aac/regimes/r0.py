"""BASE (Table 4.1 stand-ins on the skeleton): the stub content-only system over the clean
fc/rd fixtures (rows 1-2) and the full stub system with the per-claim store as gold evidence
over the AVeriTeC fixture (row 5). The real BODEGA reproduction and the AVeriTeC baseline
replace these runs in Phase 2."""
from __future__ import annotations

from pathlib import Path

from aac.pipeline import Pipeline, RunContext
from aac.regimes.common import evaluate_targets, targets_of

SPLIT = {"averitec_chrono": "test", "fc": "fc", "rd": "rd"}


def run(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path) -> dict:
    targets = targets_of(resolved, ["averitec_chrono"])
    return evaluate_targets(pipeline, ctx, resolved, root, run_dir, {t: SPLIT[t] for t in targets})
