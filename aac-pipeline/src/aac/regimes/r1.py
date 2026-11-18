"""R1 (time-split) driver: chronological or random split per data.split_manifest.

Fit steps (regimes.common.evaluate_targets): the relevance floor and pass F (val_fuse, C3 fit) when the system has
a real C2 and a fitted C3; pass A (val_cal, C4 then C5) when the system calibrates or abstains. Test pass: the test
partition of data.split_manifest under the configured system's toggles (records split=test; Table 4.2 / 4.6 read
them). The target is `averitec_test` or its legacy label `averitec_chrono`: both name the test
partition of the run's split manifest, chronological or random."""
from __future__ import annotations

from pathlib import Path

from aac.pipeline import Pipeline, RunContext
from aac.regimes.common import TEST_TARGETS, evaluate_targets, targets_of


def run(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path) -> dict:
    targets = targets_of(resolved, ["averitec_chrono"])
    if len(targets) != 1 or targets[0] not in TEST_TARGETS:
        raise ValueError(f"R1 evaluates the test partition of data.split_manifest only (target averitec_test or "
                         f"averitec_chrono); data.targets={targets}")
    return evaluate_targets(pipeline, ctx, resolved, root, run_dir, {targets[0]: "test"})
