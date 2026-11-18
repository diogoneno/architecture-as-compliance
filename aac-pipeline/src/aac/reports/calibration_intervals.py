"""ECE before and after C4 with 95 percent intervals, including ECE under transfer (shell
`calibration_intervals`).

Rows (contract order): the content-only encoder and the full pipeline on R1 random (test), R1 chronological (test),
R2 FC transfer (attack set) and R2 RD transfer (attack set). The runs are exactly the ones Table 4.2 (R1 rows) and
Table 4.3 (transfer rows) read: the run names come from their ROWS / TARGETS constants, the records from
sources.require_runs / load_runs (seal verified against runs/INDEX.json) / aligned / positives_for.

Per cell:
 point mean over seeds of netcal ECE(bins=15) on 1-D P(false) (pre-registered specification):
 before C4 on raw_score, after C4 on calibrated_probability. The content-only encoder has no C4, so its
 after-C4 cell is `n/a (no C4)`.
 interval percentile item bootstrap with the run's own stats settings (resamples, level) and items as the unit:
 one index resample is applied to every seed and the seed-mean ECE is collected (seeds as replicates,
 sources.seeds_as_replicates_ci), rng = derive_seed(f"boot.calibration_intervals.{row}.{col}")
 with row = "<System> | <Regime and split>" and col the shell column key, so reordering or adding rows
 moves no interval.

ECE sums absolute per-bin gaps, so it is biased upward on a resample: the percentile interval can lie partly or wholly
above the point estimate (measured on the real runs). The interval is kept as pre-registered , not
bias-corrected, and a header comment line of both CSVs says so (BIAS_NOTE).

A probability exactly on a k/15 edge is binned the netcal way; the own right-closed reference (ece_reference) differs
on such edges (03-VALUE-AUDIT-0314 observation 1, R1-114138606e-s3); the table uses netcal.

A row whose sealed runs are absent raises MissingRun naming the run unless a fired configs/cuts.yaml step covers the
cell (then every value cell of the row reads `not run (cut: <step> on <date>)`). A namespace whose runs read the
fixture corpus (the stub smoke chain) gets no table: build returns (registry.REAL_ONLY)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from aac.metrics.calibration import ece15
from aac.reports import fmt, sources, table_4_2, table_4_3
from aac.run.paths import reports_dir, runs_dir

TABLE = "calibration_intervals"
CONTENT, FULL = "Content-only encoder", "Full pipeline"
NO_C4 = "n/a (no C4)"
TEST = ("netcal ECE(bins=15) on 1-D P(false), mean over seeds; item bootstrap, seeds as replicates")
BIAS_NOTE = ("note: ECE sums absolute per-bin gaps, so it is biased upward on a bootstrap resample; the percentile interval "
             "(as pre-registered, not bias-corrected) can therefore lie partly or wholly above the point estimate")


def _r1_name(system: str, split: str) -> str:
    """The run name Table 4.2 reads for (system, split) on its non-selective rows."""
    for sys_, split_, name, _calibrated, selective, _ in table_4_2.ROWS:
        if sys_ == system and split_ == split and not selective:
            return name
    raise KeyError(f"Table 4.2 has no non-selective row ({system!r}, {split!r})")


def _r2_name(system: str, target: str) -> str:
    """The run name Table 4.3 reads for (system, target) on its non-selective rows."""
    for row, _key, _r1, prefix, selective in table_4_3.ROWS:
        if row == system and not selective:
            return f"r2-{target}-{prefix}"
    raise KeyError(f"Table 4.3 has no non-selective row {system!r}")


_TARGET_LABEL = dict(table_4_3.TARGETS)
# (system, regime label) -> (regime, run name, split of the records, calibrated?)
ROWS: dict[tuple[str, str], tuple[str, str, str, bool]] = {
    (CONTENT, "R1 random (test)"): ("R1", _r1_name(CONTENT, "Random"), "test", False),
    (CONTENT, "R1 chronological (test)"): ("R1", _r1_name(CONTENT, "Chronological"), "test", False),
    (FULL, "R1 random (test)"): ("R1", _r1_name(FULL, "Random"), "test", True),
    (FULL, "R1 chronological (test)"): ("R1", _r1_name(FULL, "Chronological"), "test", True),
    (CONTENT, "R2 FC transfer (attack set)"): ("R2", _r2_name(CONTENT, "fc"), "fc", False),
    (CONTENT, "R2 RD transfer (attack set)"): ("R2", _r2_name(CONTENT, "rd"), "rd", False),
    (FULL, "R2 FC transfer (attack set)"): ("R2", _r2_name(FULL, "fc"), "fc", True),
    (FULL, "R2 RD transfer (attack set)"): ("R2", _r2_name(FULL, "rd"), "rd", True),
}


def cell_key(row_ident: str, col: str) -> str:
    return f"boot.calibration_intervals.{row_ident}.{col}"


def ece_cell(probs: list[np.ndarray], y: np.ndarray, *, key: str, n_resamples: int, level: float) -> tuple[float, tuple[float, float]]:
    """(mean over seeds of ece15(p_seed, y), percentile item-bootstrap interval with seeds as replicates)."""
    point = float(np.mean([ece15(p, y) for p in probs]))
    stats = [(lambda idx, p=p: ece15(p[idx], y[idx])) for p in probs]
    lo, hi = sources.seeds_as_replicates_ci(stats, len(y), cell_key=key, n_resamples=n_resamples, level=level)
    return point, (lo, hi)


def _is_fixture(root: Path) -> bool:
    """True when the namespace's runs read the fixture corpus (the stub chain): judged on the first indexed run of
 any row (all rows of a namespace share one data manifest)."""
    for (regime, name, _split, _cal) in ROWS.values():
        entries = sources.select_runs(root, regime=regime, name=name)
        if entries:
            resolved = json.loads((runs_dir(root) / entries[0]["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))
            return resolved["data"].get("dataset") != "averitec"
    return False


def build(root: Path, out_dir: Path | None = None, shells: dict | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    shells = shells or fmt.load_shells()
    if _is_fixture(root):
        return []  # fixture corpus (stub namespace): the table is real-only (registry.REAL_ONLY)
    shell = shells[TABLE]
    labels = {c["key"]: c["label"] for c in shell["columns"]}
    shell_rows, long_rows, all_runs = [], [], []
    data_version = ""
    for system, regime_label in shell["rows"]:
        if (system, regime_label) not in ROWS:
            raise KeyError(f"{TABLE}: shell row ({system!r}, {regime_label!r}) has no run mapping")
        regime, name, split, calibrated = ROWS[(system, regime_label)]
        ident = f"{system} | {regime_label}"
        state, entries = sources.require_runs(root, regime=regime, name=name, table=TABLE, row_ident=ident, col_key="ece_before")
        if state == "cut":
            cell = fmt.NOT_RUN(entries)
            cells = {"system": system, "regime": regime_label}
            for key in ("n_items", "seeds", "ece_before", "ece_after"):
                cells[key] = cell
                long_rows.append(fmt.long_row(TABLE, ident, labels[key], key, cell, config_hash="", data_version=data_version or "n/a",
                                              run_ids="n/a", source_shards_sha256="n/a", test=f"cut {entries} (configs/cuts.yaml)"))
            shell_rows.append(cells)
            continue
        runs = sources.load_runs(root, entries)
        all_runs += runs
        ids, per_seed = sources.aligned(runs, split)
        y = sources.positives_for(root, runs[0], per_seed[0])
        n = len(ids)
        n_boot, lvl = sources.resamples(root, runs[0]), sources.level(runs[0])
        prov = sources.provenance(runs)
        data_version = data_version or prov["data_version"]
        common = dict(n_items=n, n_seeds=len(runs), seeds=sources.seeds_text(runs), **prov)
        L = lambda key, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, ident, labels[key], metric, value, **{**common, **kw}))
        L("n_items", "n_items", str(n), test=f"{split} records aligned on claim_id across seeds ({regime}/{name})")
        L("seeds", "seeds", sources.seeds_text(runs), test=f"sealed {regime}/{name} runs in runs/INDEX.json")
        cells = {"system": system, "regime": regime_label, "n_items": str(n), "seeds": sources.seeds_text(runs)}
        for key, field in (("ece_before", "raw_score"), ("ece_after", "calibrated_probability")):
            if key == "ece_after" and not calibrated:
                cells[key] = NO_C4
                L(key, "ece15_calibrated_probability", NO_C4, test="content-only system has no C4")
                continue
            probs = [sources.arr(recs, field) for recs in per_seed]
            k = cell_key(ident, key)
            point, (lo, hi) = ece_cell(probs, y, key=k, n_resamples=n_boot, level=lvl)
            cells[key] = fmt.fmt_ci(point, lo, hi)
            L(key, f"ece15_{field}", fmt.fmt_metric(point), ci_lo=fmt.fmt_metric(lo), ci_hi=fmt.fmt_metric(hi),
              test=f"{TEST}; {n_boot} resamples, level {lvl}, seed key {k}; on {field}")
        shell_rows.append({c["key"]: cells[c["key"]] for c in shell["columns"]})
    if not all_runs:
        raise sources.MissingRun(f"{TABLE}: no row has sealed runs")
    prov_all = sources.provenance(all_runs)
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=prov_all["config_hash"], data_version=prov_all["data_version"],
                         out_dir=out_dir, shells=shells, comment_lines=[BIAS_NOTE]))
