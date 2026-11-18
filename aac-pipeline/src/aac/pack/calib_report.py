"""Article 15 calibration report: evidence-pack/calibration-report.md and the copied
evidence-pack/fig-4.1-reliability.png (Table D.3 row 5; brief §1 item 6).

ECE before and after C4 per Table 4.2 row and the in-dataset ECE of Table 4.3 come from the long
CSVs (netcal ECE, 15 bins, as computed by the report generators); the chosen map per run comes from
each sealed R1/R2 run's calibration.json; Figure 4.1 is copied byte for byte from reports/ and
embedded by relative path, so the pack hashes the same bytes reports/INDEX.json lists.

the template is chosen by the namespace kind (configs/pack/datasheet-profiles.yaml, keyed by the
data manifest the sealed R1/R2 runs name). `fixture` renders calibration-report.template.md exactly as before (stub
goldens unchanged); `real` renders calibration-report.v2.template.md, which adds ECE before and after C4 with intervals
per regime and ECE under transfer (copied from reports/calibration-intervals.csv) and a declared-accuracy table
(Table 4.2 macro-F1 and AUROC cells, Table 4.3 FC / RD transfer cells, ECE after C4 with its interval), every cell
copied from a report shell with shell_value; nothing is computed here."""
from __future__ import annotations

import shutil
from pathlib import Path

from aac.pack.datasheet import load_profile
from aac.pack.inputs import (PackContext, PackInputError, index_entries, md_table, read_json, read_long,
                             read_shell, render, require_headings, shell_value, write_text)
from aac.run.hashing import sha256_file
from aac.run.seal import verify_sealed
from aac.run.paths import reports_dir, runs_dir

GENERATED_BY = "aac.pack.calib_report"
FILE = "calibration-report.md"
FIG = "fig-4.1-reliability.png"
REQUIRED = ["Method selection", "ECE before and after", "Reliability diagram"]
ECE_COLS = ("ECE before C4", "ECE after C4")
TEMPLATES = {"fixture": "calibration-report.template.md", "real": "calibration-report.v2.template.md"}
REAL_HEADINGS = REQUIRED + ["ECE under transfer", "Declared accuracy levels"]
INTERVALS = "calibration-intervals.csv"
T42, T43 = "table-4.2-r1.csv", "table-4.3-r2.csv"
SYSTEMS = ("Content-only encoder", "Full pipeline")
# declared-accuracy rows: (regime label of the intervals shell, Table 4.2 split or Table 4.3 column)
DECLARED = (("R1 random (test)", "4.2", "Random"), ("R1 chronological (test)", "4.2", "Chronological"),
            ("R2 FC transfer (attack set)", "4.3", "FC (transfer)"), ("R2 RD transfer (attack set)", "4.3", "RD (transfer)"))
NOT_IN_43 = "n/a (Table 4.3 reports macro-F1 only)"


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    sel_rows, manifests = [], set()
    for e in index_entries(root):
        if e["regime"] not in ("R1", "R2"):
            continue
        run_dir = runs_dir(root) / e["run_id"]
        if not (run_dir / "calibration.json").is_file():
            continue
        if verify_sealed(run_dir) != e["sealed_sha256"]:
            raise PackInputError(f"{e['run_id']}: SEALED disagrees with runs/INDEX.json")
        manifests.add(read_json(run_dir / "config.resolved.json")["data"]["manifest"])
        cal = read_json(run_dir / "calibration.json")
        oof = cal["val_ece_out_of_fold"]
        sel_rows.append([e["run_id"], f"{e['regime']}/{e['name']}", str(e["seed"]), cal["selected"],
                         f"{oof['temperature']:.3f}", f"{oof['isotonic']:.3f}", str(cal["n_val_cal"])])
    if not sel_rows:
        raise PackInputError("calibration report: no sealed R1/R2 run carries calibration.json")
    sel_rows.sort(key=lambda r: (r[1], int(r[2])))
    rule = read_json(runs_dir(root) / sel_rows[0][0] / "calibration.json")["selection_rule"]
    long42 = read_long(reports_dir(root) / "long" / "table-4.2-r1.long.csv")
    ece_rows = []
    for r in long42:
        if r["col_label"] in ECE_COLS:
            ece_rows.append([r["row_label"], r["col_label"], r["value"], r["n_seeds"], r["test"]])
    ece_rows.sort(key=lambda x: (x[0], x[1]))
    long43 = read_long(reports_dir(root) / "long" / "table-4.3-r2.long.csv")
    r2_rows = [[r["row_label"], r["value"], r["n_seeds"], r["test"]] for r in long43
               if r["col_label"] == "AVeriTeC held-out (in-dataset) (ECE)"]
    src = reports_dir(root) / FIG
    if not src.is_file():
        raise PackInputError(f"missing reports/{FIG}; run `make reports` first")
    if len(manifests) != 1:
        raise PackInputError(f"calibration report: sealed R1/R2 runs name {len(manifests)} data manifests: {sorted(manifests)}")
    kind = load_profile(root, next(iter(manifests)))["kind"]
    if kind not in TEMPLATES:
        raise PackInputError(f"calibration report: no template for namespace kind {kind!r}")
    common = dict(selection_rule=rule,
                  selection_table=md_table(["Run", "Configuration", "Seed", "Selected map", "Out-of-fold ECE, temperature",
                                            "Out-of-fold ECE, isotonic", "val_cal items"], sel_rows),
                  ece_table=md_table(["Table 4.2 row", "Column", "Value", "Seeds", "Computation"], ece_rows),
                  r2_table=md_table(["Table 4.3 row", "In-dataset ECE after C4", "Seeds", "Computation"], r2_rows), fig=FIG)
    extra, inputs = {}, "reports/long/table-4.2-r1.long.csv; reports/long/table-4.3-r2.long.csv; sealed R1/R2 calibration.json"
    if kind == "real":
        extra = _real_fields(root)
        inputs += f"; reports/{INTERVALS}; reports/{T42}; reports/{T43}"
    shutil.copyfile(src, pack / FIG)
    text = render(root, TEMPLATES[kind], **common, **extra)
    require_headings(text, REAL_HEADINGS if kind == "real" else REQUIRED, FILE)
    write_text(pack / FILE, text)
    ctx.record(FILE, GENERATED_BY, inputs)
    ctx.record(FIG, "aac.reports.fig_4_1 (copied byte for byte from reports/)", f"reports/{FIG}")
    return [FILE, FIG]


def _real_fields(root: Path) -> dict:
    """Template fields of the real profile, every cell copied from a report shell (no metric is computed here)."""
    rep = reports_dir(root)
    meta_ci, head_ci, rows_ci = read_shell(rep / INTERVALS)
    _, head42, rows42 = read_shell(rep / T42)
    _, head43, rows43 = read_shell(rep / T43)
    r1 = [r for r in rows_ci if r[1].startswith("R1 ")]
    r2 = [r for r in rows_ci if r[1].startswith("R2 ")]
    if len(r1) != 4 or len(r2) != 4:
        raise PackInputError(f"reports/{INTERVALS}: expected 4 R1 and 4 R2 rows, found {len(r1)} and {len(r2)}")
    declared = []
    for system in SYSTEMS:
        for regime, table, key in DECLARED:
            ece_after = shell_value(head_ci, rows_ci, f"{system} | {regime}", "ECE after C4 [95% CI]", label_columns=2)
            if table == "4.2":
                f1 = shell_value(head42, rows42, f"{system} | {key}", "Macro-F1 [95% CI]", label_columns=2)
                au = shell_value(head42, rows42, f"{system} | {key}", "AUROC [95% CI]", label_columns=2)
                src = f"Table 4.2 ({key})"
            else:
                f1 = shell_value(head43, rows43, system, key, label_columns=1)
                au = NOT_IN_43
                src = f"Table 4.3 ({key})"
            declared.append([system, regime, f1, au, ece_after, f"{src}; reports/{INTERVALS}"])
    notes = [ln[len("# note:"):].strip() for ln in (rep / INTERVALS).read_text(encoding="utf-8").splitlines()[1:]
             if ln.startswith("# note:")]
    if not notes:
        raise PackInputError(f"reports/{INTERVALS}: the bootstrap note line is missing from the header")
    prov = []
    for f in (INTERVALS, T42, T43):
        meta, _, _ = read_shell(rep / f)
        prov.append([f"reports/{f}", sha256_file(rep / f), meta["data_version"],
                     str(len(meta["config_hash"].split(";")))])
    return {"ece_interval_table": md_table(head_ci, r1), "transfer_table": md_table(head_ci, r2),
            "declared_table": md_table(["System", "Regime and split", "Macro-F1 [95% CI]", "AUROC [95% CI]",
                                        "ECE after C4 [95% CI]", "Copied from"], declared),
            "intervals_config_hash": meta_ci["config_hash"], "intervals_note": " ".join(notes),
            "provenance_table": md_table(["Shell", "sha256 of the file", "data_version", "Config hashes in its header (n)"], prov),
            "edge_source": "src/aac/metrics/calibration.py"}
