"""Table 4.3 (R2 cross-dataset): in-dataset cell `macro-F1 x; AUROC y; ECE z` from the R1 random
held-out runs, FC/RD cells from the R2 runs (binary labels, label 1 = non-credible = Refuted-positive),
mean transfer loss = in-dataset minus transfer macro-F1 (positive = loss), abstention rates
`x / y / z` on the selective row, and row 4 linked to the Table 4.1 long rows. The H2 family
(H2-fc, H2-rd) is a paired t over per-seed (in-dataset minus transfer) differences of the full system,
Holm within R2. The RD column honours cut-4-drop-rd once fired (`not run (cut: ...)`).

Evidence coverage (pre-registered specification
long row gives the share of target records with evidence_coverage true (percent, 1 d.p.; same items and seeds
as the cell: all items, or the auto-decided items on the selective row; mean over seeds)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aac.metrics.calibration import ece15
from aac.metrics.classification import auroc, macro_f1_binary
from aac.metrics.selective import binary_macro_f1 as _f1_fast
from aac.reports import fmt, sources, table_4_1
from aac.stats.families import family as holm_family
from aac.stats.holm import holm_family as holm_family_adjusted
from aac.stats.tests import paired_t_dz
from aac.run.paths import reports_dir

TABLE = "4.3"
REGIME = "R2"
ROWS = (
    # (row label, system key, in-dataset R1 config, R2 config prefix, selective?)
    ("Content-only encoder", "content_only", "r1-random-content-only", "content-only", False),
    ("Full pipeline", "full", "r1-random-full", "full", False),
    ("Full pipeline, abstention at 80 percent coverage (selective metrics)", "full", "r1-random-full", "full", True),
)
PUBLISHED_ROW = "Published BERT victim (from Table 4.1)"
TARGETS = (("fc", "FC (transfer)"), ("rd", "RD (transfer)"))
TARGET_TEST = {"fc": "H2-fc", "rd": "H2-rd"}
DECISION_RULE = "decision P(Refuted) >= 0.5 on calibrated_probability"  # as Table 4.2 


def _metrics(root: Path, entries: list[dict], split: str, ident: str, col: str, selective: bool) -> dict:
    runs = sources.load_runs(root, entries)
    ids, per_seed = sources.aligned(runs, split)
    y = sources.positives_for(root, runs[0], per_seed[0])
    n = len(ids)
    n_boot, lvl = sources.resamples(root, runs[0]), sources.level(runs[0])
    probs = [sources.arr(recs, "calibrated_probability") for recs in per_seed]
    masks = [sources.auto_mask(recs) if selective else np.ones(n, dtype=bool) for recs in per_seed]
    f1_seed = [macro_f1_binary(y[m], p[m] >= 0.5) for p, m in zip(probs, masks)]
    stats = [(lambda idx, p=p, m=m: _f1_fast(y[idx][m[idx]], (p[idx] >= 0.5)[m[idx]])) for p, m in zip(probs, masks)]
    ci = sources.seeds_as_replicates_ci(stats, n, cell_key=f"boot.table-4.3.{ident}.{col}", n_resamples=n_boot, level=lvl)
    try:
        au = float(np.mean([auroc(y[m], p[m]) for p, m in zip(probs, masks)]))
    except ValueError:
        au = None
    ece = float(np.mean([ece15(p[m], y[m]) for p, m in zip(probs, masks)]))
    abst = float(np.mean([1.0 - m.mean() for m in masks]))
    cov = [np.asarray([bool(r["evidence_coverage"]) for r in recs], dtype=bool) for recs in per_seed]
    ev_cov = float(np.mean([c[m].mean() if m.any() else 0.0 for c, m in zip(cov, masks)]))
    ev_n = float(np.mean([m.sum() for m in masks]))
    return {"runs": runs, "n_items": n, "f1_seed": f1_seed, "f1": float(np.mean(f1_seed)), "ci": ci, "auroc": au,
            "ece": ece, "abstention": abst, "evidence_coverage": ev_cov, "evidence_coverage_items": ev_n}


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    shell_rows, long_rows, all_runs = [], [], []
    data: dict[str, dict] = {}
    for row, system, r1_name, r2_prefix, selective in ROWS:
        state, entries = sources.require_runs(root, regime="R1", name=r1_name, table=TABLE, row_ident=row, col_key="averitec_heldout")
        if state == "cut":
            raise sources.MissingRun(f"Table 4.3 in-dataset cell of {row!r} is never blanked by a cut")
        d = {"in": _metrics(root, entries, "test", row, "averitec_heldout", selective)}
        for target, _ in TARGETS:
            state, entries = sources.require_runs(root, regime=REGIME, name=f"r2-{target}-{r2_prefix}", table=TABLE, row_ident=row, col_key=target)
            d[target] = {"cut": entries} if state == "cut" else _metrics(root, entries, target, row, target, selective)
        data[row] = d
    # H2 family on the full (non-selective) system: per-seed in-dataset minus transfer, paired t, Holm within R2
    fam = holm_family(REGIME, root)
    full = data["Full pipeline"]
    tests = {}
    for target, _ in TARGETS:
        if "cut" in full[target]:
            continue
        if [r.seed for r in full["in"]["runs"]] != [r.seed for r in full[target]["runs"]]:
            raise RuntimeError(f"Table 4.3 {target}: in-dataset and transfer seeds differ")
        tests[TARGET_TEST[target]] = paired_t_dz(full["in"]["f1_seed"], full[target]["f1_seed"])
    ordered = [t for t in fam if t in tests]
    adj = holm_family_adjusted(tests, ordered)  
    for row, system, r1_name, r2_prefix, selective in ROWS:
        d = data[row]
        runs_here = list(d["in"]["runs"]) + [r for t, _ in TARGETS if "cut" not in d[t] for r in d[t]["runs"]]
        all_runs += runs_here
        prov_in = sources.provenance(d["in"]["runs"])
        n_seeds = len(d["in"]["runs"])
        common_in = dict(n_items=d["in"]["n_items"], n_seeds=n_seeds, seeds=sources.seeds_text(d["in"]["runs"]), **prov_in)
        L = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, row, col, metric, value, **{**common_in, **kw}))
        au_txt = fmt.fmt_metric(d["in"]["auroc"]) if d["in"]["auroc"] is not None else "n/a (one class)"
        in_cell = f"macro-F1 {fmt.fmt_f1(d['in']['f1'])}; AUROC {au_txt}; ECE {fmt.fmt_metric(d['in']['ece'])}"
        in_col = "AVeriTeC held-out (in-dataset)"
        L(in_col, "macro_f1_binary; auroc; ece15", in_cell, test="R1 random held-out" + (" (auto-decided items)" if selective else "") + f"; {DECISION_RULE}")
        L(in_col + " (macro-F1)", "macro_f1_binary", fmt.fmt_f1(d["in"]["f1"]), ci_lo=fmt.fmt_f1(d["in"]["ci"][0]), ci_hi=fmt.fmt_f1(d["in"]["ci"][1]),
          test=f"percentile bootstrap (items; seeds as replicates); {DECISION_RULE}")
        L(in_col + " (AUROC)", "auroc", au_txt)
        L(in_col + " (ECE)", "ece15_calibrated_probability", fmt.fmt_metric(d["in"]["ece"]), test="netcal ECE(bins=15), mean over seeds")
        cells = {"system": row, "averitec_heldout": in_cell}
        losses, abst = [], [fmt.fmt_rate(d["in"]["abstention"])]
        for target, label in TARGETS:
            t = d[target]
            if "cut" in t:
                cell = fmt.NOT_RUN(t["cut"])
                cells[target] = cell
                long_rows.append(fmt.long_row(TABLE, row, label, "macro_f1_binary", cell, config_hash="", data_version=prov_in["data_version"],
                                              run_ids="n/a", source_shards_sha256="n/a", test=f"cut {t['cut']} (configs/cuts.yaml)"))
                abst.append(fmt.NA)
                continue
            prov_t = sources.provenance(t["runs"])
            kw = dict(n_items=t["n_items"], n_seeds=len(t["runs"]), seeds=sources.seeds_text(t["runs"]), **prov_t,
                      ci_lo=fmt.fmt_f1(t["ci"][0]), ci_hi=fmt.fmt_f1(t["ci"][1]),
                      test=f"percentile bootstrap (items; seeds as replicates); {DECISION_RULE}")
            test_id = TARGET_TEST[target]
            if row == "Full pipeline" and test_id in tests:
                tr = tests[test_id]
                if tr.degenerate:
                    kw.update(effect_size_kind="zero variance", test=kw["test"] + "; paired t over per-seed (in-dataset minus transfer): degenerate", family=f"{REGIME}:{test_id}")
                else:
                    kw.update(p_raw=f"{tr.p:.6f}", p_holm=f"{adj[test_id]:.6f}", effect_size=f"{tr.d_z:.4f}", effect_size_kind="cohen_dz",
                              test=kw["test"] + "; paired t over per-seed (in-dataset minus transfer); Holm within R2", family=f"{REGIME}:{test_id}")
            cells[target] = fmt.fmt_ci(t["f1"], *t["ci"])
            long_rows.append(fmt.long_row(TABLE, row, label, "macro_f1_binary", fmt.fmt_f1(t["f1"]), **kw))
            if system == "full":
                long_rows.append(fmt.long_row(
                    TABLE, row, f"{label}, evidence coverage (percent)", "evidence_coverage_pct", fmt.fmt_rate(t["evidence_coverage"]),
                    n_items=fmt.fmt_count(t["evidence_coverage_items"]), n_seeds=len(t["runs"]), seeds=sources.seeds_text(t["runs"]), **prov_t,
                    test=("share of target records with evidence_coverage true (a retained passage at or above the relevance floor), "
                          + ("auto-decided items, " if selective else "all items, ") + "mean over seeds (specification item 16)")))
            losses.append(d["in"]["f1"] - t["f1"])
            abst.append(fmt.fmt_rate(t["abstention"]))
        if losses:
            loss = float(np.mean(losses))
            cells["transfer_loss"] = fmt.fmt_signed(loss, 3)
            L("Mean transfer loss", "transfer_loss_macro_f1", cells["transfer_loss"], n_items=len(losses),
              test="in-dataset minus transfer macro-F1, mean over the targets present (positive = loss)")
        else:
            cells["transfer_loss"] = "n/a (no transfer target run)"
            L("Mean transfer loss", "transfer_loss_macro_f1", cells["transfer_loss"])
        if selective:
            cells["abstention_rate"] = " / ".join(abst)
            L("Abstention rate in-dataset / FC / RD", "abstention_rate_pct", cells["abstention_rate"], test="deferred share of items, mean over seeds")
        else:
            cells["abstention_rate"] = fmt.NA
            L("Abstention rate in-dataset / FC / RD", "abstention_rate_pct", fmt.NA, test="n/a by the shell: non-selective row")
        shell_rows.append(cells)
    # row 4: the published BERT victim, linked to Table 4.1 rows 1-2 (analyst-transcribed values or n/a)
    t41 = {c["row"]: c for c in table_4_1.rows_data(root)}
    cells = {"system": PUBLISHED_ROW, "averitec_heldout": fmt.NA}
    published_metrics = []
    for target, label in TARGETS:
        src = t41[f"BERT victim, {target.upper()}, clean"]
        prov = sources.provenance(src["runs"])
        cells[target] = table_4_1.published_cell(src["spec"])
        # the metric is the one transcribed for the Table 4.1 row (configs/published_baselines.yaml metric_key:
        # positive-class F1 for the BODEGA Table 3 values), never relabelled as binary macro-F1
        metric_key = src["spec"].get("metric_key") or "macro_f1_binary"
        published_metrics.append(f"{target.upper()}: {src['spec'].get('metric') or metric_key}")
        long_rows.append(fmt.long_row(TABLE, PUBLISHED_ROW, label, f"published_{metric_key}", cells[target], n_items=src["n_items"],
                                      n_seeds=len(src["runs"]), seeds=sources.seeds_text(src["runs"]),
                                      test=f"table-4.1 row 'BERT victim, {target.upper()}, clean' (configs/published_baselines.yaml)",
                                      config_hash=prov["config_hash"], data_version=prov["data_version"],
                                      run_ids=f"table-4.1:BERT victim, {target.upper()}, clean <- {prov['run_ids']}",
                                      source_shards_sha256=prov["source_shards_sha256"]))
        all_runs += src["runs"]
    cells["transfer_loss"] = fmt.NA
    cells["abstention_rate"] = fmt.NA
    prov41 = sources.provenance(t41["BERT victim, FC, clean"]["runs"])
    for col in ("AVeriTeC held-out (in-dataset)", "Mean transfer loss", "Abstention rate in-dataset / FC / RD"):
        long_rows.append(fmt.long_row(TABLE, PUBLISHED_ROW, col, "n/a", fmt.NA, test="n/a by the shell: published row",
                                      config_hash=prov41["config_hash"], data_version=prov41["data_version"],
                                      run_ids="table-4.1", source_shards_sha256="n/a"))
    shell_rows.append(cells)
    prov_all = sources.provenance(all_runs)
    note = (f"note: row '{PUBLISHED_ROW}' prints the published values as transcribed for Table 4.1 rows 1-2 in their own metric "
            f"({'; '.join(published_metrics)}; configs/published_baselines.yaml), not the binary macro-F1 of rows 1-3, "
            "so that row is a reference point, not a like-for-like comparison")
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=prov_all["config_hash"], data_version=prov_all["data_version"], out_dir=out_dir,
                         comment_lines=[note]))
