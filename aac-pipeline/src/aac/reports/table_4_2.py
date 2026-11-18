"""Table 4.2 (R1 time-split) from the sealed R1 runs: binary macro-F1 (decision P(Refuted) >= 0.5 on
calibrated_probability) and AUROC with item-bootstrap CIs (seeds as replicates), ECE before/after C4
(netcal, 15 bins) on raw_score / calibrated_probability, `reference` on random rows, row 5 selective, cutoff
date in the header comment.

Change column (H1, pre-registered specification
random and chronological test sets are different item sets, so the pre-registered paired item-level tests run
on the claims the two test sets share:
 primary paired percentile item bootstrap of the difference in seed-mean binary macro-F1 (chrono minus
 random) on the shared claims, the same index resample applied to both systems and every seed; and
 the exact McNemar test on the per-item outcome "correct" of the two systems on the shared claims,
 the per-item outcome taken over seeds by majority (correct in more than half of the seeds). The
 McNemar p enters Holm within the R1 family; the shell cell shows the Holm p, the mean paired
 difference and its bootstrap interval. Per-seed McNemar tables are long rows.
 secondary over ALL items of both test sets (unpaired): Welch t on the per-seed macro-F1 values and an
 unpaired item bootstrap of the difference in seed-mean macro-F1 (two independent resamples).
 long only the seed-matched paired t (seed s of chrono against seed s of random) with d_z, labeled as such;
 it is not a decision criterion because the seeds are not paired items.

Sensitivity variant : when sealed r1-chrono-full-sens runs exist with the r1-chrono-full seeds,
non-shell long rows repeat the row 4 and row 5 metrics for that variant and add a paired item bootstrap of its
difference against r1-chrono-full (not in a Holm family). The shell never moves for it.

allow_pending (CLI `aac reports build --table 4.2 --allow-pending`): a row whose sealed runs do not
exist yet is written with every value cell `pending `; while any test of the R1 Holm family is missing,
every Change cell of the family is `pending ` too, and its long row keeps the raw primary p and the
paired difference with p_holm pending. Without allow_pending a missing row raises MissingRun."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aac.metrics.calibration import ece15
from aac.metrics.classification import auroc, auroc_fast, macro_f1_binary
from aac.metrics.selective import binary_macro_f1 as _f1_fast
from aac.reports import fmt, sources
from aac.run.hashing import derive_seed
from aac.stats.bootstrap import ci_index, unpaired_ci_index
from aac.stats.families import family as holm_family
from aac.stats.holm import holm_family as holm_family_adjusted
from aac.stats.tests import mcnemar_exact, paired_t_dz, welch_df, welch_t
from aac.run.paths import reports_dir

TABLE = "4.2"
REGIME = "R1"
ROWS = (
    # (system label, split label, config name, calibrated?, selective?, family test id or None)
    ("Content-only encoder", "Random", "r1-random-content-only", False, False, None),
    ("Content-only encoder", "Chronological", "r1-chrono-content-only", False, False, "H1-chrono-vs-random-content"),
    ("Full pipeline", "Random", "r1-random-full", True, False, None),
    ("Full pipeline", "Chronological", "r1-chrono-full", True, False, "H1-chrono-vs-random-full"),
    ("Full pipeline, abstention at 80 percent coverage", "Chronological", "r1-chrono-full", True, True, None),
)
NOT_IN_FAMILY = "n/a (not in the R1 family)"
DECISION_RULE = "decision P(Refuted) >= 0.5 on calibrated_probability"
CHANGE_COL = "Change vs random (p; effect size)"
SENS_NAME = "r1-chrono-full-sens"
SENS_SYSTEM = "Full pipeline, sensitivity (answer-URL and after-claim documents removed)"
SENS_SELECTIVE = SENS_SYSTEM + ", abstention at 80 percent coverage"
SENS_TEST = "paired percentile bootstrap (sensitivity, not in a Holm family)"


def _row_metrics(root: Path, entries: list[dict], ident: str, selective: bool, calibrated: bool) -> dict:
    runs = sources.load_runs(root, entries)
    ids, per_seed = sources.aligned(runs, "test")
    y = sources.positives_for(root, runs[0], per_seed[0])
    n = len(ids)
    n_boot, lvl = sources.resamples(root, runs[0]), sources.level(runs[0])
    probs = [sources.arr(recs, "calibrated_probability") for recs in per_seed]
    raws = [sources.arr(recs, "raw_score") for recs in per_seed]
    masks = [sources.auto_mask(recs) if selective else np.ones(n, dtype=bool) for recs in per_seed]
    f1_seed = [macro_f1_binary(y[m], p[m] >= 0.5) for p, m in zip(probs, masks)]
    f1_stats = [(lambda idx, p=p, m=m: _f1_fast(y[idx][m[idx]], (p[idx] >= 0.5)[m[idx]])) for p, m in zip(probs, masks)]
    f1_ci = sources.seeds_as_replicates_ci(f1_stats, n, cell_key=f"boot.table-4.2.{ident}.macro_f1_binary", n_resamples=n_boot, level=lvl)
    auroc_seed, auroc_ci, auroc_note = [], None, ""
    try:
        auroc_seed = [auroc(y[m], p[m]) for p, m in zip(probs, masks)]
        def _au(s, p, m):  # s is already a class-stratified resample (sources.stratified)
            return auroc_fast(y[s][m[s]], p[s][m[s]])

        au_stats = [(lambda idx, p=p, m=m: _au(idx, p, m)) for p, m in zip(probs, masks)]
        auroc_ci = sources.seeds_as_replicates_ci(au_stats, n, cell_key=f"boot.table-4.2.{ident}.auroc", n_resamples=n_boot,
                                                  level=lvl, draw=lambda rng: sources.stratified(rng, y))
    except ValueError as exc:  # one class only among the auto-decided items
        auroc_seed, auroc_ci, auroc_note = [], None, str(exc)
    ece_before = float(np.mean([ece15(r[m], y[m]) for r, m in zip(raws, masks)]))
    ece_after = float(np.mean([ece15(p[m], y[m]) for p, m in zip(probs, masks)])) if calibrated else None
    coverage = float(np.mean([m.mean() for m in masks]))
    return {"runs": runs, "n_items": n, "ids": ids, "y": y, "probs": probs, "masks": masks, "n_boot": n_boot, "level": lvl,
            "f1_seed": f1_seed, "f1": float(np.mean(f1_seed)), "f1_ci": f1_ci,
            "auroc_seed": auroc_seed, "auroc": float(np.mean(auroc_seed)) if auroc_seed else None, "auroc_ci": auroc_ci,
            "auroc_note": auroc_note, "ece_before": ece_before, "ece_after": ece_after, "coverage": coverage}


def _seed_mean_f1(y: np.ndarray, preds: list[np.ndarray], idx: np.ndarray) -> float:
    return float(np.mean([_f1_fast(y[idx], p[idx]) for p in preds]))


def h1_tests(chrono: dict, random: dict, ident: str) -> dict:
    """Primary, secondary and seed-matched H1 statistics for one system (chrono minus random).

 Primary on the shared claims: paired item bootstrap of the seed-mean macro-F1 difference and exact McNemar on
 the seed-majority outcome "correct"; per-seed McNemar tables beside it. Secondary on all items: Welch t on
 per-seed macro-F1 and an unpaired item bootstrap. Seeds must match (same seed numbers, same order)."""
    if [r.seed for r in chrono["runs"]] != [r.seed for r in random["runs"]]:
        raise RuntimeError(f"Table 4.2 {ident}: chronological and random seeds differ")
    pos_c = {cid: i for i, cid in enumerate(chrono["ids"])}
    pos_r = {cid: i for i, cid in enumerate(random["ids"])}
    shared = sorted(set(pos_c) & set(pos_r))
    if not shared:
        raise RuntimeError(f"Table 4.2 {ident}: the random and chronological test sets share no claim; "
                           f"the pre-registered paired test cannot run")
    ic = np.asarray([pos_c[c] for c in shared], dtype=int)
    ir = np.asarray([pos_r[c] for c in shared], dtype=int)
    y = chrono["y"][ic]
    if not np.array_equal(y, random["y"][ir]):
        raise RuntimeError(f"Table 4.2 {ident}: the shared claims carry different labels in the two runs")
    pred_c = [p[ic] >= 0.5 for p in chrono["probs"]]
    pred_r = [p[ir] >= 0.5 for p in random["probs"]]
    n = len(shared)
    f1_c = [_f1_fast(y, p) for p in pred_c]
    f1_r = [_f1_fast(y, p) for p in pred_r]
    delta = float(np.mean(f1_c) - np.mean(f1_r))
    boot_lo, boot_hi = ci_index(lambda idx: _seed_mean_f1(y, pred_c, idx) - _seed_mean_f1(y, pred_r, idx), n,
                                n_resamples=chrono["n_boot"], seed=derive_seed(f"boot.table-4.2.{ident}.h1_shared_delta"),
                                level=chrono["level"])

    def _table(ok_r: np.ndarray, ok_c: np.ndarray) -> list[list[int]]:
        # rows: random correct / not; columns: chrono correct / not; b = random right only, c = chrono right only
        return [[int(np.sum(ok_r & ok_c)), int(np.sum(ok_r & ~ok_c))], [int(np.sum(~ok_r & ok_c)), int(np.sum(~ok_r & ~ok_c))]]

    ok_c = [p == y for p in pred_c]
    ok_r = [p == y for p in pred_r]
    per_seed = []
    for run, a, b in zip(chrono["runs"], ok_r, ok_c):
        tab = _table(a, b)
        per_seed.append({"seed": run.seed, "table": tab, "result": mcnemar_exact(tab)})
    maj_c = np.mean(ok_c, axis=0) > 0.5
    maj_r = np.mean(ok_r, axis=0) > 0.5
    maj_table = _table(maj_r, maj_c)
    mcn = mcnemar_exact(maj_table)
    # secondary: all items of both test sets, unpaired
    yc, yr = chrono["y"], random["y"]
    full_c = [p >= 0.5 for p in chrono["probs"]]
    full_r = [p >= 0.5 for p in random["probs"]]
    delta_all = float(np.mean(chrono["f1_seed"]) - np.mean(random["f1_seed"]))
    welch = welch_t(chrono["f1_seed"], random["f1_seed"])
    df = welch_df(chrono["f1_seed"], random["f1_seed"]) if not welch.degenerate else None
    un_lo, un_hi = unpaired_ci_index(lambda a, b: _seed_mean_f1(yc, full_c, a) - _seed_mean_f1(yr, full_r, b),
                                     len(yc), len(yr), n_resamples=chrono["n_boot"],
                                     seed=derive_seed(f"boot.table-4.2.{ident}.h1_unpaired_delta"), level=chrono["level"])
    seed_t = paired_t_dz(chrono["f1_seed"], random["f1_seed"])
    return {"n_shared": n, "n_shared_pos": int(np.sum(y)), "shared_f1_chrono": f1_c, "shared_f1_random": f1_r,
            "delta": delta, "boot_ci": (boot_lo, boot_hi), "mcnemar": mcn, "mcnemar_table": maj_table, "per_seed": per_seed,
            "delta_all": delta_all, "welch": welch, "welch_df": df, "unpaired_ci": (un_lo, un_hi),
            "n_chrono": len(yc), "n_random": len(yr), "seed_t": seed_t}


def _sens_rows(root: Path, base: dict) -> tuple[list[dict], list]:
    """(non-shell long rows, runs) of r1-chrono-full-sens: rows 4-5 metrics and the paired difference against
 r1-chrono-full; (, ) when no sealed sensitivity run exists. Seeds must equal the r1-chrono-full seeds."""
    entries = sources.select_runs(root, regime=REGIME, name=SENS_NAME)
    if not entries:
        return [], []
    base_seeds = [r.seed for r in base["runs"]]
    if [int(e["seed"]) for e in entries] != base_seeds:
        raise RuntimeError(f"Table 4.2 sensitivity: {SENS_NAME} seeds {[int(e['seed']) for e in entries]} differ from "
                           f"r1-chrono-full seeds {base_seeds}")
    out, runs = [], []
    for label, selective in ((SENS_SYSTEM, False), (SENS_SELECTIVE, True)):
        ident = f"{label} | Chronological"
        d = _row_metrics(root, entries, ident, selective, True)
        runs = d["runs"]
        prov = sources.provenance(d["runs"])
        common = dict(n_items=d["n_items"], n_seeds=len(d["runs"]), seeds=sources.seeds_text(d["runs"]), **prov)
        L = lambda col, metric, value, **kw: out.append(fmt.long_row(TABLE, ident, col, metric, value, **{**common, **kw}))
        if selective:
            L("Macro-F1 [95% CI]", "selective_macro_f1_binary", fmt.fmt_f1(d["f1"]), ci_lo=fmt.fmt_f1(d["f1_ci"][0]),
              ci_hi=fmt.fmt_f1(d["f1_ci"][1]), test=f"percentile bootstrap (auto-decided items; seeds as replicates); {DECISION_RULE}; sensitivity variant",
              family="none")
            L("Macro-F1 [95% CI] (coverage achieved, percent)", "coverage_achieved_pct", fmt.fmt_rate(d["coverage"]), family="none")
        else:
            L("Macro-F1 [95% CI]", "macro_f1_binary", fmt.fmt_f1(d["f1"]), ci_lo=fmt.fmt_f1(d["f1_ci"][0]), ci_hi=fmt.fmt_f1(d["f1_ci"][1]),
              test=f"percentile bootstrap (items; seeds as replicates); {DECISION_RULE}; sensitivity variant", family="none")
        if d["auroc"] is None:
            L("AUROC [95% CI]", "auroc", "n/a (one class among the auto-decided items)", test=d["auroc_note"], family="none")
        else:
            L("AUROC [95% CI]", "auroc", fmt.fmt_metric(d["auroc"]), ci_lo=fmt.fmt_metric(d["auroc_ci"][0]), ci_hi=fmt.fmt_metric(d["auroc_ci"][1]),
              test="percentile bootstrap (class-stratified items; seeds as replicates); sensitivity variant", family="none")
        if selective:
            L("ECE before C4", "ece15_raw_score", fmt.NA, test="n/a by the shell: selective row", family="none")
        else:
            L("ECE before C4", "ece15_raw_score", fmt.fmt_metric(d["ece_before"]), test="netcal ECE(bins=15), mean over seeds; sensitivity variant", family="none")
        L("ECE after C4", "ece15_calibrated_probability", fmt.fmt_metric(d["ece_after"]), test="netcal ECE(bins=15), mean over seeds; sensitivity variant", family="none")
        if selective:
            continue
        # paired item bootstrap against r1-chrono-full: the same claims (same test set), one index resample for both
        if d["ids"] != base["ids"]:
            raise RuntimeError("Table 4.2 sensitivity: r1-chrono-full-sens and r1-chrono-full scored different test items")
        y = base["y"]
        if not np.array_equal(y, d["y"]):
            raise RuntimeError("Table 4.2 sensitivity: labels differ between r1-chrono-full-sens and r1-chrono-full")
        pred_s = [p >= 0.5 for p in d["probs"]]
        pred_b = [p >= 0.5 for p in base["probs"]]
        delta = float(np.mean(d["f1_seed"]) - np.mean(base["f1_seed"]))
        lo, hi = ci_index(lambda idx, _y=y, _ps=pred_s, _pb=pred_b: _seed_mean_f1(_y, _ps, idx) - _seed_mean_f1(_y, _pb, idx), len(y),
                          n_resamples=d["n_boot"], seed=derive_seed(f"boot.table-4.2.{ident}.delta_vs_r1_chrono_full"), level=d["level"])
        prov_both = sources.provenance(list(d["runs"]) + list(base["runs"]))
        out.append(fmt.long_row(TABLE, ident, "Change vs r1-chrono-full (sensitivity minus main)", "delta_macro_f1_vs_r1_chrono_full",
                                fmt.fmt_signed(delta), ci_lo=fmt.fmt_signed(lo), ci_hi=fmt.fmt_signed(hi), n_items=len(y),
                                n_seeds=len(d["runs"]), seeds=sources.seeds_text(d["runs"]), effect_size=f"{delta:.4f}",
                                effect_size_kind="paired_delta_macro_f1_sens_minus_main",
                                test=SENS_TEST, family="none", **prov_both))
    return out, list(runs)


PENDING_PHASE = "Phase 3"


def _h1_long_rows(L, h: dict, test_id: str) -> None:
    """Non-shell long rows of one H1 test: per-seed McNemar, the secondary tests and the seed-matched paired t."""
    fam = f"{REGIME}:{test_id}"
    L("Change vs random, primary: shared claims (n; Refuted)", "n_shared_items", str(h["n_shared"]), n_items=h["n_shared"],
      test=f"claims in both the random and the chronological test set; {h['n_shared_pos']} of them Refuted", family=fam)
    for ps in h["per_seed"]:
        (n11, b), (c, n00) = ps["table"]
        L(f"Change vs random, primary: McNemar seed {ps['seed']} (shared claims)", "mcnemar_exact_per_seed",
          f"b={b}; c={c}", p_raw=f"{ps['result'].p:.6f}", n_items=h["n_shared"], n_seeds=1, seeds=str(ps["seed"]),
          test=(f"exact McNemar, seed {ps['seed']} of both systems; table [[{n11}, {b}], [{c}, {n00}]] "
                f"(b = random right only, c = chrono right only); reported, Holm uses the seed-majority test"), family=fam)
    welch = h["welch"]
    if welch.degenerate:
        L("Change vs random, secondary: Welch t (all items)", "welch_t_seed_level_unpaired", fmt.fmt_signed(h["delta_all"], 4),
          effect_size_kind="zero variance", test="Welch t on per-seed macro-F1, all items of both test sets: degenerate", family="secondary")
    else:
        L("Change vs random, secondary: Welch t (all items)", "welch_t_seed_level_unpaired", fmt.fmt_signed(h["delta_all"], 4),
          p_raw=f"{welch.p:.6f}", effect_size=f"{welch.statistic:.4f}", effect_size_kind="welch_t",
          n_items=f"{h['n_chrono']};{h['n_random']}",
          test=(f"Welch t on per-seed binary macro-F1 (chrono minus random), all items of both test sets, unpaired; "
                f"t={welch.statistic:.3f}, df={h['welch_df']:.2f}; secondary, not in the Holm family"), family="secondary")
    lo, hi = h["unpaired_ci"]
    L("Change vs random, secondary: unpaired item bootstrap (all items)", "unpaired_item_bootstrap_delta", fmt.fmt_signed(h["delta_all"]),
      ci_lo=fmt.fmt_signed(lo), ci_hi=fmt.fmt_signed(hi), n_items=f"{h['n_chrono']};{h['n_random']}",
      effect_size=f"{h['delta_all']:.4f}", effect_size_kind="delta_macro_f1_unpaired",
      test="percentile bootstrap, two independent item resamples per draw (all items of both test sets; seeds as replicates); secondary",
      family="secondary")
    st = h["seed_t"]
    if st.degenerate:
        L("Change vs random, seed-matched paired t (not primary)", "seed_matched_paired_t_dz", fmt.NA + " (zero variance)",
          effect_size_kind="zero variance", test="seed-matched paired t over per-seed differences: degenerate", family="none")
    else:
        L("Change vs random, seed-matched paired t (not primary)", "seed_matched_paired_t_dz", fmt.fmt_signed(h["delta_all"], 4),
          p_raw=f"{st.p:.6f}", effect_size=f"{st.d_z:.4f}", effect_size_kind="cohen_dz",
          n_items=f"{h['n_chrono']};{h['n_random']}",
          test=("seed-matched paired t: seed s of chrono against seed s of random on their own (different) test sets; "
                "the seeds are not paired items, so this is neither the primary nor the secondary test (protocol notes)"),
          family="none")


def build(root: Path, out_dir: Path | None = None, allow_pending: bool = False) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    data: dict[tuple[str, str], dict] = {}
    pending_rows: set[tuple[str, str]] = set()
    for system, split, name, calibrated, selective, _ in ROWS:
        ident = f"{system} | {split}"
        try:
            state, entries = sources.require_runs(root, regime=REGIME, name=name, table=TABLE, row_ident=ident, col_key="macro_f1_binary")
        except sources.MissingRun:
            if not allow_pending:
                raise
            pending_rows.add((system, split))
            continue
        if state == "cut":
            raise sources.MissingRun(f"Table 4.2 row {ident!r} is never blanked by a cut")
        data[(system, split)] = _row_metrics(root, entries, ident, selective, calibrated)
    if not data:
        raise sources.MissingRun("Table 4.2: no row has sealed runs")
    
    fam = holm_family(REGIME, root)
    h1: dict[str, dict] = {}
    for system, split, name, calibrated, selective, test_id in ROWS:
        if test_id is None or (system, "Chronological") not in data or (system, "Random") not in data:
            continue
        h1[test_id] = h1_tests(data[(system, "Chronological")], data[(system, "Random")], f"{system} | Chronological")
    tests = {t: h["mcnemar"] for t, h in h1.items()}
    ordered = [t for t in fam if t in tests]
    if sorted(ordered) != sorted(tests):
        raise RuntimeError(f"Table 4.2 tests {sorted(tests)} are not all in the R1 family {fam}")
    family_complete = sorted(tests) == sorted(fam)
    if not family_complete and not allow_pending:
        raise sources.MissingRun(f"Table 4.2: R1 family {fam} incomplete (have {sorted(tests)})")
    adj = holm_family_adjusted(tests, ordered) if family_complete else {}  
    cut_src = data.get(("Full pipeline", "Chronological")) or data.get(("Content-only encoder", "Chronological"))
    if cut_src is None:
        raise sources.MissingRun("Table 4.2: no chronological run to read the cutoff date from")
    cutoff = sources.split_manifest(root, cut_src["runs"][0])["cutoff_date"]
    data_version = sources.provenance([r for d in data.values() for r in d["runs"]])["data_version"]
    shell_rows, long_rows = [], []
    pend = fmt.PENDING(PENDING_PHASE)
    for system, split, name, calibrated, selective, test_id in ROWS:
        ident = f"{system} | {split}"
        if (system, split) in pending_rows:
            for col_key, lab in (("macro_f1_binary", "Macro-F1 [95% CI]"), ("auroc", "AUROC [95% CI]"), ("ece_before", "ECE before C4"),
                                 ("ece_after", "ECE after C4"), ("delta_vs_random", CHANGE_COL)):
                long_rows.append(fmt.long_row(TABLE, ident, lab, col_key, pend, config_hash="", data_version=data_version,
                                              run_ids="n/a", source_shards_sha256="n/a",
                                              test=f"no sealed {REGIME}/{name} run yet; filled in {PENDING_PHASE}"))
            shell_rows.append({"system": system, "split": split, "macro_f1_binary": pend, "auroc": pend,
                               "ece_before": pend, "ece_after": pend, "delta_vs_random": pend})
            continue
        d = data[(system, split)]
        prov = sources.provenance(d["runs"])
        common = dict(n_items=d["n_items"], n_seeds=len(d["runs"]), seeds=sources.seeds_text(d["runs"]), **prov)
        L = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, ident, col, metric, value, **{**common, **kw}))
        f1_txt = fmt.fmt_ci(d["f1"], *d["f1_ci"])
        if selective:
            f1_cell = f"selective: {f1_txt} (coverage achieved {fmt.fmt_rate(d['coverage'])})"
            L("Macro-F1 [95% CI]", "selective_macro_f1_binary; coverage_achieved_pct", f1_cell,
              ci_lo=fmt.fmt_f1(d["f1_ci"][0]), ci_hi=fmt.fmt_f1(d["f1_ci"][1]),
              test=f"percentile bootstrap (auto-decided items; seeds as replicates); {DECISION_RULE}", family=REGIME)
            L("Macro-F1 [95% CI] (coverage achieved, percent)", "coverage_achieved_pct", fmt.fmt_rate(d["coverage"]))
        else:
            f1_cell = f1_txt
            L("Macro-F1 [95% CI]", "macro_f1_binary", fmt.fmt_f1(d["f1"]), ci_lo=fmt.fmt_f1(d["f1_ci"][0]), ci_hi=fmt.fmt_f1(d["f1_ci"][1]),
              test=f"percentile bootstrap (items; seeds as replicates); {DECISION_RULE}", family=REGIME)
        if d["auroc"] is None:
            auroc_cell = "n/a (one class among the auto-decided items)"
            L("AUROC [95% CI]", "auroc", auroc_cell, test=d["auroc_note"])
        else:
            auroc_cell = fmt.fmt_ci(d["auroc"], *d["auroc_ci"])
            L("AUROC [95% CI]", "auroc", fmt.fmt_metric(d["auroc"]), ci_lo=fmt.fmt_metric(d["auroc_ci"][0]), ci_hi=fmt.fmt_metric(d["auroc_ci"][1]),
              test="percentile bootstrap (class-stratified items; seeds as replicates)", family=REGIME)
        if selective:
            ece_before_cell = fmt.NA
            L("ECE before C4", "ece15_raw_score", ece_before_cell, test="n/a by the shell: selective row")
        else:
            ece_before_cell = fmt.fmt_metric(d["ece_before"])
            L("ECE before C4", "ece15_raw_score", ece_before_cell, test="netcal ECE(bins=15), mean over seeds")
        if d["ece_after"] is None:
            ece_after_cell = "n/a (no C4)"
            L("ECE after C4", "ece15_calibrated_probability", ece_after_cell, test="content-only system has no C4")
        else:
            ece_after_cell = fmt.fmt_metric(d["ece_after"])
            L("ECE after C4", "ece15_calibrated_probability", ece_after_cell, test="netcal ECE(bins=15), mean over seeds")
        if split == "Random":
            change = "reference"
            L(CHANGE_COL, "delta_macro_f1_vs_random", change)
        elif test_id is None:
            change = NOT_IN_FAMILY
            L(CHANGE_COL, "delta_macro_f1_vs_random", change, family=REGIME)
        else:
            h = h1[test_id]
            mcn = h["mcnemar"]
            lo, hi = h["boot_ci"]
            prov_pair = sources.provenance(list(d["runs"]) + list(data[(system, "Random")]["runs"]))
            (n11, b), (c, n00) = h["mcnemar_table"]
            primary_txt = (f"primary: exact McNemar on the {h['n_shared']} claims shared by the random and "
                           f"chronological test sets, per-item outcome correct in more than half of the seeds "
                           f"(table [[{n11}, {b}], [{c}, {n00}]], b = random right only, c = chrono right only), with the paired "
                           f"percentile item bootstrap (same index resample for both systems and every seed) of the difference in "
                           f"seed-mean binary macro-F1 on those claims (chrono minus random); {DECISION_RULE}")
            Lp = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(
                TABLE, ident, col, metric, value, **{**common, **prov_pair, "n_items": h["n_shared"], **kw}))
            if not family_complete:
                missing = [x for x in fam if x not in tests]
                change = pend
                Lp(CHANGE_COL, "delta_macro_f1_vs_random", pend, p_raw=f"{mcn.p:.6f}", p_holm=pend,
                   ci_lo=fmt.fmt_signed(lo), ci_hi=fmt.fmt_signed(hi), effect_size=f"{h['delta']:.4f}",
                   effect_size_kind="paired_delta_macro_f1_shared_claims",
                   test=f"{primary_txt}; Holm within R1 waits for {', '.join(missing)} ({PENDING_PHASE})", family=f"{REGIME}:{test_id}")
            else:
                change = f"{fmt.fmt_p(adj[test_id])} (Holm); diff={fmt.fmt_signed(h['delta'])} [{fmt.fmt_signed(lo)}, {fmt.fmt_signed(hi)}]"
                Lp(CHANGE_COL, "delta_macro_f1_vs_random", fmt.fmt_signed(h["delta"]), p_raw=f"{mcn.p:.6f}",
                   p_holm=f"{adj[test_id]:.6f}", ci_lo=fmt.fmt_signed(lo), ci_hi=fmt.fmt_signed(hi), effect_size=f"{h['delta']:.4f}",
                   effect_size_kind="paired_delta_macro_f1_shared_claims",
                   test=f"{primary_txt}; Holm within R1 on the McNemar p", family=f"{REGIME}:{test_id}")
            Lc = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, ident, col, metric, value, **{**common, **prov_pair, **kw}))
            _h1_long_rows(Lc, h, test_id)
        shell_rows.append({"system": system, "split": split, "macro_f1_binary": f1_cell, "auroc": auroc_cell,
                           "ece_before": ece_before_cell, "ece_after": ece_after_cell, "delta_vs_random": change})
    all_runs = [r for d in data.values() for r in d["runs"]]
    base = data.get(("Full pipeline", "Chronological"))
    if base is not None:
        sens_long, sens_runs = _sens_rows(root, base)
        long_rows += sens_long
        all_runs += sens_runs
    prov_all = sources.provenance(all_runs)
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=prov_all["config_hash"], data_version=prov_all["data_version"],
                         out_dir=out_dir, header_extra={"cutoff_date": cutoff}, allow_pending=allow_pending))
