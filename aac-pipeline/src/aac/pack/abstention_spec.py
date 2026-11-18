"""Article 14 abstention specification: evidence-pack/abstention-spec.md (Table D.3 row 4;
draft-v0.5.md:279, brief §1 item 5, text-contradictions item 8).

Reads Table 4.6 and deferral_reasons (shell + long CSV), the C5 configuration and the sealed
gate.json of the R1 chronological full-pipeline seed-0 run (the run behind Table 4.6), and the
relevance floor (the run's sealed evidence-floor.json when present, else the C2 configuration field).
Records the coverage target, the threshold and how it was selected, the deferral reason distribution,
the error-pattern shift from the FN/FP shares, the Art. 14(4)(a)-(e) table and the pause setting
(threshold = infinity routes every item to the queue).

the template is chosen by the namespace kind (configs/pack/datasheet-profiles.yaml, keyed by the
data manifest the sealed run names). `fixture` renders abstention-spec.template.md exactly as before (stub goldens
unchanged); `real` renders abstention-spec.v2.template.md, which prints the threshold pair, the coverage achieved on
val_cal, the relevance floor and the pause setting from the sealed gate.json / evidence-floor.json at their stored
precision (number tokens as written in the file, never re-derived from the 3-decimal Table 4.6 cells, which appear
beside them labeled as the Table 4.6 rendering), the reason definitions and precedence verbatim, the deferral rule as
coded in aac.metrics.selective.defer_mask (DEFERRAL_RULE, pinned by a unit test), the Art. 14(4)(a)-(e) table and the
built-in versus deployer measures from configs/pack/art14-measures.yaml, and the Art. 14(5) note."""
from __future__ import annotations

from pathlib import Path

import json

import yaml

from aac.pack.datasheet import load_profile
from aac.pack.inputs import (PackContext, PackInputError, long_cell, md_table, read_json, read_long, read_shell, render,
                             require_headings, sealed_run, shell_md, shell_value, write_text)
from aac.run.paths import reports_dir

GENERATED_BY = "aac.pack.abstention_spec"
FILE = "abstention-spec.md"
REQUIRED = ["Threshold", "Deferral reasons", "Coverage versus risk"]
TARGET_ROW = "80 (pre-specified)"
REFERENCE_ROW = "100"
FN_COL = "False-negative share of residual errors (percent)"
FP_COL = "False-positive share (percent)"
DELTA_COL = "Change in false-negative share vs 100 percent"
FLOOR_FILE = "evidence-floor.json"
TEMPLATES = {"fixture": "abstention-spec.template.md", "real": "abstention-spec.v2.template.md"}
REAL_HEADINGS = REQUIRED + ["Pause setting", "Article 14(4) mapping", "Built-in and deployer measures", "Article 14(5)"]
MEASURES = "configs/pack/art14-measures.yaml"
# The deferral rule exactly as aac.metrics.selective.defer_mask codes it: (c < pc) | ((c == pc) & (r < pr)), with
# c, r = max(p, 1 - p) rounded to KEY_DECIMALS (6). tests/unit/test_abstention_spec_real.py checks this sentence
# against defer_mask at the sealed pair and one step either side.
DEFERRAL_RULE = ("An item is deferred when its key (calibrated confidence, raw confidence), each max(p, 1 - p) rounded "
                 "to 6 d.p., is lexicographically below the threshold pair: its calibrated confidence is below the first "
                 "element, or equal to it with its raw confidence below the second element; an item whose key equals the "
                 "pair is auto-decided (aac.metrics.selective.defer_mask).")


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    entry, run_dir, resolved = sealed_run(root, regime="R1", name="r1-chrono-full", seed=0)
    gate = read_json(run_dir / "gate.json")
    c5 = resolved["components"]["c5"]
    c2 = resolved["components"]["c2"]
    meta46, t46 = shell_md(root, "table-4.6-coverage-risk.csv")
    _, tdr = shell_md(root, "deferral_reasons.csv")
    long46 = read_long(reports_dir(root) / "long" / "table-4.6-coverage-risk.long.csv")
    longdr = read_long(reports_dir(root) / "long" / "deferral_reasons.long.csv")
    _, h46, r46 = read_shell(reports_dir(root) / "table-4.6-coverage-risk.csv")
    thr_cell = long_cell(long46, TARGET_ROW, "Threshold", "threshold_confidence")
    n_deferrals = long_cell(longdr, TARGET_ROW, "Low confidence (percent of deferrals)")["n_items"]
    # the relevance floor: a real run seals it in evidence-floor.json (value + rule); the fixture run has none
    # and keeps reading the C2 configuration field, so the stub bytes do not move 
    floor_file = run_dir / FLOOR_FILE
    if floor_file.is_file():
        floor = read_json(floor_file)
        relevance_floor = f"{floor['value']} (sealed {FLOOR_FILE}, rule {floor['rule']})"
    else:
        relevance_floor = c2["relevance_floor"]
    report_thresholds = ", ".join(f"{k}: {'none' if v is None else v}" for k, v in sorted(gate["report_thresholds"].items(), reverse=True))
    fields = dict(
        run_id=entry["run_id"], config_hash=entry["config_hash"], data_version=resolved["data_version"],
        coverage_target=f"{round(float(c5['coverage_target']) * 100)}",
        report_coverages=", ".join(f"{round(float(c) * 100)}" for c in c5["report_coverages"]),
        threshold_cell=thr_cell["value"], threshold_sealed=gate["threshold"], n_val_cal=gate["n_val_cal"],
        report_thresholds=report_thresholds, fit_on=c5["fit_on"], conflict_floor=c5["conflict_floor"],
        relevance_floor=relevance_floor, n_test_items=thr_cell["n_items"], n_deferrals=n_deferrals,
        fn_ref=long_cell(long46, REFERENCE_ROW, FN_COL)["value"], fp_ref=long_cell(long46, REFERENCE_ROW, FP_COL)["value"],
        fn_target=long_cell(long46, TARGET_ROW, FN_COL)["value"], fp_target=long_cell(long46, TARGET_ROW, FP_COL)["value"],
        delta_fn=long_cell(long46, TARGET_ROW, DELTA_COL)["value"],
        t46=t46, tdr=tdr, table_config_hash=meta46["config_hash"],
    )
    kind = load_profile(root, resolved["data"]["manifest"])["kind"]
    if kind not in TEMPLATES:
        raise PackInputError(f"abstention spec: no template for namespace kind {kind!r}")
    if kind == "real":
        fields.update(_real_fields(root, run_dir, h46, r46))
    text = render(root, TEMPLATES[kind], **fields)
    require_headings(text, REAL_HEADINGS if kind == "real" else REQUIRED, FILE)
    write_text(pack / FILE, text)
    inputs = ("reports/table-4.6-coverage-risk.csv; reports/deferral_reasons.csv (+ long CSVs); "
              f"runs/{entry['run_id']}/gate.json (sealed); C5 configuration")
    if floor_file.is_file():
        inputs += f"; runs/{entry['run_id']}/{FLOOR_FILE} (sealed)"
    if kind == "real":
        inputs += f"; {MEASURES}"
    ctx.record(FILE, GENERATED_BY, inputs)
    return [FILE]


def _raw_json(path: Path) -> dict:
    """A sealed JSON file with every number kept as the token written in the file (stored precision, no float round
 trip); strings, booleans and null as usual."""
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_float=str, parse_int=str)


def _pair(value) -> str:
    return "none" if value is None else f"({value[0]}, {value[1]})"


def _real_fields(root: Path, run_dir: Path, h46: list[str], r46: list[list[str]]) -> dict:
    """Template fields of the real profile: sealed values verbatim, the Table 4.6 cells beside them, the versioned
 Art. 14 input rendered as tables. Nothing is computed."""
    floor_file = run_dir / FLOOR_FILE
    if not floor_file.is_file():
        raise PackInputError(f"{run_dir.name}: a real run must seal {FLOOR_FILE} (the relevance floor)")
    gate, floor = _raw_json(run_dir / "gate.json"), _raw_json(floor_file)
    with open(Path(root) / MEASURES, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    for e in spec["article_14_4"] + spec["measures"]:
        if not e.get("source"):
            raise PackInputError(f"{MEASURES}: entry without a source: {e}")
    target = float(gate["coverage_target"])
    thr_rows = []
    for key in sorted(gate["coverage_achieved"], key=float, reverse=True):
        label = f"{round(float(key) * 100)}" + (" (pre-specified)" if float(key) == target else "")
        thr = gate["report_thresholds"][key] if key in gate["report_thresholds"] else "n/a"
        thr_rows.append([key, "n/a (not a reported coverage)" if thr == "n/a" else _pair(thr),
                         gate["coverage_achieved"][key], label, shell_value(h46, r46, label, "Threshold"),
                         shell_value(h46, r46, label, "Coverage achieved (percent)")])
    precedence = list(gate["reason_precedence"])
    defs = gate["reason_definitions"]
    if set(precedence) != set(defs):
        raise PackInputError("gate.json: reason_precedence and reason_definitions name different reasons")
    return dict(
        coverage_target_raw=gate["coverage_target"], threshold_pair=_pair(gate["threshold_pair"]),
        gate_name=gate["gate"], tie_break=gate["tie_break"], conflict_floor_raw=gate["conflict_floor"],
        n_val_cal_raw=gate["n_val_cal"], fit_on_raw=gate["fit_on"],
        threshold_table=md_table(["Coverage key", "Threshold pair (sealed)", "Coverage on val_cal (sealed)",
                                  "Table 4.6 row", "Threshold (Table 4.6)", "Test coverage, % (Table 4.6)"], thr_rows),
        deferral_rule=DEFERRAL_RULE,
        floor_value=floor["value"], floor_rule=floor["rule"], floor_rank_index=floor["rank_index"],
        floor_n_val_cal=floor["n_val_cal"], floor_evidence_run=floor["evidence_run_id"],
        floor_rule_text=spec["relevance_floor_rule_text"]["text"], floor_rule_source=spec["relevance_floor_rule_text"]["source"],
        reason_precedence=" > ".join(f"`{r}`" for r in precedence),
        reason_table=md_table(["Reason", "Definition (gate.json)"], [[r, defs[r]] for r in precedence]),
        pause=json.dumps(gate["pause"]), pause_rule=gate["pause_rule"],
        art14_table=md_table(["Art. 14(4)", "Capability (the Act)", "What this pack shows", "Where",
                              "Not shown without participants", "Source"],
                             [[e["item"], e["capability"], e["shown"], e["where"], e["not_shown"], e["source"]]
                              for e in spec["article_14_4"]]),
        measures_table=md_table(["Measure", "Built in or deployer", "Artifact", "Source"],
                                [[m["measure"], m["kind"], m["artifact"], m["source"]] for m in spec["measures"]]),
    )
