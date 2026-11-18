"""Article 13 model card: evidence-pack/model-card.md (Table D.3 row 3; draft-v0.5.md:279,
brief §1 item 4).

Rendered with huggingface_hub.ModelCard.from_template(ModelCardData(...), template_str=<own Jinja
file configs/pack/modelcard.template.md>, **fields): the template holds the Art. 13(3)(a)-(f)
headings and the pack-files.yaml heading set; the fields are shell tables copied from reports/,
EvalResult entries read from the long CSVs (macro_f1_binary / auroc / ece cells), the known gaps
carried over from the datasheet input, and the provenance (config_hash, data_version, the
v1.0 reference). No push, no network call, no timestamp: `card.save` writes the rendered
text only.

the template is chosen by the namespace kind (configs/pack/datasheet-profiles.yaml, keyed by the
data manifest the sealed runs name; TEMPLATES). `fixture` renders modelcard.template.md exactly as before (stub
goldens unchanged). `real` renders modelcard.v2.template.md: the known-gaps file named by the profile's `known_gaps`
key, the release input configs/pack/modelcard-release.yaml (licence, data terms, provider line; analyst item O3), the
intended purpose from configs/pack/intended-purpose.md, and EvalResult entries labeled with the real dataset names.
Every value is a copied shell cell, a long-row value, a sealed config value or a count; a missing input raises
PackInputError naming it, and two EvalResult entries with the same model-index key raise (`_check_unique`)."""
from __future__ import annotations

from pathlib import Path

import yaml

from aac.pack.datasheet import KNOWN_GAPS, _known_gaps, load_profile, sealed_data_version
from aac.pack.inputs import (PackContext, PackInputError, as_float, file_sha, md_table, read_long, read_shell,
                             require_headings, sealed_run, shell_md, shell_value, template_path, write_text)
from aac.run.index import load_index
from aac.run.paths import probes_dir, reports_dir, runs_dir

GENERATED_BY = "aac.pack.modelcard"
FILE = "model-card.md"
REQUIRED = ["Model details", "Intended use", "Evaluation", "Limitations", "Ethical considerations"]
ART13 = ["Art. 13(3)(a)", "Art. 13(3)(b)", "Art. 13(3)(c)", "Art. 13(3)(d)", "Art. 13(3)(e)", "Art. 13(3)(f)"]
BASE_MODEL = "microsoft/mdeberta-v3-base"
TEMPLATES = {"fixture": "modelcard.template.md", "real": "modelcard.v2.template.md"}
RELEASE = "configs/pack/modelcard-release.yaml"
INTENDED = "configs/pack/intended-purpose.md"
LICENCES = "configs/licences.yaml"
REAL_MODEL_NAME = "aac-pipeline full pipeline"
REAL_TAGS = ["misinformation-detection", "selective-prediction", "eu-ai-act"]
CONTACT_NULL = "given in the submitted dissertation; not part of this pack"
# Real-kind dataset labels: (dataset_type, dataset_name, dataset_split) per source.
REAL_AVERITEC = ("averitec", "AVeriTeC {split} split (test)", "test")
REAL_TRANSFER = {"FC": ("incrediblae-fc", "InCrediblAE FC (zero-shot transfer)", "attack"),
                 "RD": ("incrediblae-rd", "InCrediblAE RD (zero-shot transfer)", "attack")}
# Art. 13(3)(b)(ii) declared-accuracy table: shell file -> (label columns, declared columns, row filter).
DECLARED = (
    ("table-4.2-r1.csv", 2, ("Macro-F1 [95% CI]", "AUROC [95% CI]"), None),
    ("table-4.3-r2.csv", 1, ("FC (transfer)", "RD (transfer)"), "Published"),  # the published row is not this system
    ("calibration-intervals.csv", 2, ("ECE before C4 [95% CI]", "ECE after C4 [95% CI]"), None),
    ("table-4.4-r3.csv", 2, ("Attack-success rate (percent)",), None),  # pooled shell cells
)
# Art. 13(3)(b)(vi) input specification: (label, sealed config key path).
INPUT_SPEC = (
    ("Dataset of the claims", "data.dataset"),
    ("Claims file directory", "data.claims"),
    ("C1 encoder", "components.c1.hf_id"),
    ("C1 encoder revision", "components.c1.revision"),
    ("C1 maximum input length (tokens)", "components.c1.max_length"),
    ("Evidence source", "components.c2.provider"),
    ("Evidence run (frozen BM25 shortlist and dense scores)", "components.c2.evidence.run_id"),
    ("Dense candidates per claim", "components.c2.k_dense"),
    ("Re-ranked evidence passages kept (top-k)", "components.c2.k"),
    ("Dense encoder maximum length (tokens)", "components.c2.dense.max_seq_length"),
    ("Re-ranker maximum length (tokens)", "components.c2.reranker.max_length"),
    ("Abstention coverage target", "components.c5.coverage_target"),
)
# Art. 13(3)(c) frozen checkpoints: (label, key path of the sha256, key path of the producing run).
CHECKPOINTS = (
    ("C1 encoder checkpoint", "components.c1.checkpoint.sha256", "components.c1.checkpoint.run_id"),
    ("C2 entailment-head checkpoint", "components.c2.scorer.sha256", "components.c2.scorer.run_id"),
    ("C2 evidence file", "components.c2.evidence.sha256", "components.c2.evidence.run_id"),
)
# Art. 13(3)(b)(iv) output fields of a decision-log record (each checked against the schema's properties).
OUTPUT_FIELDS = ("decision", "calibrated_probability", "calibration_method", "abstention_flag", "abstention_reason",
                 "threshold", "evidence_ids", "evidence_scores", "evidence_coverage")
# Shells whose `# note:` header lines the real card prints under the copied table (template field n_<key>).
NOTED_SHELLS = (("t42", "table-4.2-r1.csv"), ("t43", "table-4.3-r2.csv"), ("tci", "calibration-intervals.csv"),
                ("t44", "table-4.4-r3.csv"), ("t45", "table-4.5-ablations.csv"), ("t46", "table-4.6-coverage-risk.csv"))
SCHEMA = "contracts/decision-log.schema.json"
SHELLS_CONTRACT = "contracts/table-shells.json"
UNIQUE_KEY = ("task_name", "dataset_type", "dataset_config", "dataset_split", "metric_type", "metric_name")
CI_COLS = {  # calibration-intervals column label -> (metric key, EvalResult metric_name), real kind only
    "ECE before C4 [95% CI]": ("ece15_raw_score", "ECE (15 bins) before C4"),
    "ECE after C4 [95% CI]": ("ece15_calibrated_probability", "ECE (15 bins) after C4"),
}
EVAL_COLS = {  # Table 4.2 column label -> (metric key in the long CSV, EvalResult metric_type, metric_name)
    "Macro-F1 [95% CI]": ("macro_f1_binary", "f1", "Binary macro-F1 (Refuted vs rest)"),
    "AUROC [95% CI]": ("auroc", "roc_auc", "AUROC"),
    "ECE before C4": ("ece15_raw_score", "ece", "ECE (15 bins) before C4"),
    "ECE after C4": ("ece15_calibrated_probability", "ece", "ECE (15 bins) after C4"),
}


def _slug(text: str) -> str:
    return "-".join("".join(ch.lower() if ch.isalnum() else " " for ch in text).split())


def _check_unique(results: list) -> list:
    """ adjacency: model-index groups entries by (task, dataset, config, split) and lists metrics by
 (type, name); two entries with the same key would render as one ambiguous metric. Equal values under different
 keys are separate entries and pass."""
    seen = {}
    for i, r in enumerate(results):
        key = tuple(getattr(r, k) for k in UNIQUE_KEY)
        if key in seen:
            raise PackInputError(f"model card: EvalResult entries {seen[key]} and {i} share the model-index key "
                                 f"{dict(zip(UNIQUE_KEY, key))}")
        seen[key] = i
    return results


def eval_results(root: Path, kind: str = "fixture") -> list:
    """EvalResult entries in long-CSV row order: Table 4.2 (R1), Table 4.3 (R2 transfer macro-F1), and for the real
 kind the R2 transfer ECE rows of calibration-intervals (the R1 ECE points are Table 4.2's)."""
    from huggingface_hub import EvalResult

    real = kind == "real"
    out = []
    shell42 = set()
    if real:
        _, _, rows42 = read_shell(reports_dir(root) / "table-4.2-r1.csv")
        shell42 = {" | ".join(r[:2]) for r in rows42}
    for r in read_long(reports_dir(root) / "long" / "table-4.2-r1.long.csv"):
        spec = EVAL_COLS.get(r["col_label"])
        if not spec or r["metric"] != spec[0] or as_float(r["value"]) is None:
            continue
        if real and r["row_label"] not in shell42:
            continue  # long-only rows (the sensitivity run) stay in the long CSV; the card declares the shell rows
        system, split = [s.strip() for s in r["row_label"].split(" | ")]
        if real:
            dtype, dname, dsplit = REAL_AVERITEC[0], REAL_AVERITEC[1].format(split=split.lower()), REAL_AVERITEC[2]
        else:
            dtype, dname, dsplit = "averitec-fixture", f"AVeriTeC fixture stand-in, {split.lower()} split", "test"
        out.append(EvalResult(task_type="text-classification", task_name=f"{system} (R1)",
                              dataset_type=dtype, dataset_name=dname,
                              dataset_config=_slug(f"{system} {split}"),  # model-index groups by (task, dataset, config, split)
                              dataset_split=dsplit, metric_type=spec[1], metric_name=spec[2], metric_value=float(r["value"])))
    for r in read_long(reports_dir(root) / "long" / "table-4.3-r2.long.csv"):
        if r["col_label"] in ("FC (transfer)", "RD (transfer)") and r["metric"] == "macro_f1_binary" and as_float(r["value"]) is not None:
            target = r["col_label"].split()[0]
            if real:
                dtype, dname, dsplit = REAL_TRANSFER[target]
            else:
                dtype, dname, dsplit = (f"incrediblae-{target.lower()}-fixture",
                                        f"InCrediblAE {target} fixture stand-in (zero-shot transfer)", "test")
            out.append(EvalResult(task_type="text-classification", task_name=f"{r['row_label']} (R2)",
                                  dataset_type=dtype, dataset_name=dname,
                                  dataset_config=_slug(r["row_label"]),
                                  dataset_split=dsplit, metric_type="f1", metric_name="Binary macro-F1 (label 1 = non-credible)",
                                  metric_value=float(r["value"])))
    if real:
        for r in read_long(reports_dir(root) / "long" / "calibration-intervals.long.csv"):
            regime = r["row_label"].split(" | ")[1] if " | " in r["row_label"] else ""
            if not regime.startswith("R2 ") or as_float(r["value"]) is None:
                continue
            spec = CI_COLS.get(r["col_label"])
            if not spec or r["metric"] != spec[0]:
                continue
            system = r["row_label"].split(" | ")[0]
            target = regime.split()[1]
            dtype, dname, dsplit = REAL_TRANSFER[target]
            out.append(EvalResult(task_type="text-classification", task_name=f"{system} (R2)",
                                  dataset_type=dtype, dataset_name=dname, dataset_config=_slug(system),
                                  dataset_split=dsplit, metric_type="ece", metric_name=spec[1],
                                  metric_value=float(r["value"])))
    if not out:
        raise PackInputError("model card: no numeric macro_f1_binary / auroc / ece long rows in tables 4.2 and 4.3")
    return _check_unique(out)


def _check_fields(template: str, supplied: set[str], file: str) -> None:
    """from_template renders with jinja2's default Undefined (a missing field prints as an empty string); refuse a
 template variable the generator did not supply instead."""
    import jinja2
    import jinja2.meta

    names = jinja2.meta.find_undeclared_variables(jinja2.Environment(autoescape=True).parse(template))
    missing = sorted(names - supplied)
    if missing:
        raise PackInputError(f"{file}: template asks for fields the generator does not supply: {missing}")


def _yaml(root: Path, rel: str) -> dict:
    p = Path(root) / rel
    if not p.is_file():
        raise PackInputError(f"missing versioned input {rel}")
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PackInputError(f"{rel}: expected a mapping")
    return data


def _text(root: Path, rel: str) -> str:
    p = Path(root) / rel
    if not p.is_file():
        raise PackInputError(f"missing versioned input {rel}")
    return p.read_text(encoding="utf-8").rstrip("\n")


def _release(root: Path) -> dict:
    """configs/pack/modelcard-release.yaml, cross-checked against the licence register (configs/licences.yaml)."""
    rel = _yaml(root, RELEASE)
    for k in ("license", "license_source", "data_terms_sentence", "data_terms_source", "incrediblae_terms",
              "incrediblae_terms_source", "provider"):
        if not rel.get(k):
            raise PackInputError(f"{RELEASE}: missing key {k!r}")
    if "contact" not in rel:
        raise PackInputError(f"{RELEASE}: missing key 'contact' (null renders as {CONTACT_NULL!r})")
    lic = _yaml(root, LICENCES)
    averitec = lic["data"]["averitec"]["licence"]
    packaging = lic["data"]["incrediblae"]["packaging_terms"]
    if averitec not in rel["data_terms_sentence"]:
        raise PackInputError(f"{RELEASE}: data_terms_sentence does not name the AVeriTeC licence {averitec!r} of {LICENCES}")
    if rel["incrediblae_terms"] != packaging:
        raise PackInputError(f"{RELEASE}: incrediblae_terms {rel['incrediblae_terms']!r} != {LICENCES} packaging_terms {packaging!r}")
    for k in ("groups_note", "changes", "lifetime", "retention"):
        if not isinstance(rel.get(k), dict) or not rel[k].get("text") or not rel[k].get("source"):
            raise PackInputError(f"{RELEASE}: {k} needs `text` and `source`")
    if not rel.get("misuse") or not all(m.get("text") and m.get("source") for m in rel["misuse"]):
        raise PackInputError(f"{RELEASE}: misuse needs at least one entry, each with `text` and `source`")
    prov = rel["provider"]
    for k in ("name", "role", "title", "university", "supervisor", "nature", "source"):
        if not prov.get(k):
            raise PackInputError(f"{RELEASE}: provider.{k} missing")
    rel["_licences"] = lic
    return rel


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    from huggingface_hub import ModelCard, ModelCardData

    root, pack = Path(root), Path(pack)
    manifest, dv = sealed_data_version(root, ctx.runs)
    profile = load_profile(root, manifest)
    kind = profile.get("kind")
    if kind not in TEMPLATES:
        raise PackInputError(f"model card: no template for namespace kind {kind!r}")
    gaps_rel = profile.get("known_gaps")
    if not gaps_rel:
        raise PackInputError(f"configs/pack/datasheet-profiles.yaml: profile {manifest} has no known_gaps key")
    known_gaps = _known_gaps(root, gaps_rel)  # PackInputError naming the file when it is missing
    entry, _, resolved = sealed_run(root, regime="R1", name="r1-chrono-full", seed=0)
    _, h46, r46 = read_shell(reports_dir(root) / "table-4.6-coverage-risk.csv")
    _, hdr, rdr = read_shell(reports_dir(root) / "deferral_reasons.csv")
    target_row = "80 (pre-specified)"
    deferral = {lab: shell_value(hdr, rdr, target_row, lab) for lab in hdr[1:]}
    index = load_index(runs_dir(root) / "INDEX.json")
    spec_ref = index.get("day1_docs", {}).get("spec_v1_commit", "N/A")
    if not spec_ref:
        spec_ref = "N/A"
    tables = {}
    consumed = set()
    for key, file in (("t42", "table-4.2-r1.csv"), ("t43", "table-4.3-r2.csv"), ("t44", "table-4.4-r3.csv"),
                      ("t45", "table-4.5-ablations.csv"), ("t46", "table-4.6-coverage-risk.csv"), ("td2", "table-D.2-run-record.csv")):
        meta, md = shell_md(root, file)
        tables[key] = md
        consumed.update(meta.get("config_hash", "").split(";"))
    _, d1h, d1r = read_shell(reports_dir(root) / "table-D.1-stack.csv")
    runtime = shell_value(d1h, d1r, "Runtime", "Version recorded at build")
    counts = ctx.counts
    fields = dict(
        system_config_hash=entry["config_hash"], system_run_id=entry["run_id"], data_version=dv,
        consumed_list="\n".join(f"  - `{h}`" for h in sorted(h for h in consumed if h)), spec_v1_commit=spec_ref,
        coverage_target=f"{round(float(resolved['components']['c5']['coverage_target']) * 100)}",
        threshold=shell_value(h46, r46, target_row, "Threshold"),
        coverage_achieved=shell_value(h46, r46, target_row, "Coverage achieved (percent)"),
        deferral_list="\n".join(f"  - {k}: {v}" for k, v in deferral.items()),
        known_gaps=known_gaps, known_gaps_file=gaps_rel,
        runtime=runtime, schema_sha256=file_sha(root, "contracts/decision-log.schema.json"),
        n_records=counts.records, violations=counts.violations,
        records_with_claim_text=counts.records_with_claim_text, records_with_user_id=counts.records_with_user_id,
        **tables,
    )
    if kind == "real":
        release = _release(root)
        fields.update(_real_fields(root, release, entry, resolved))
        card_data = ModelCardData(language="en", license=release["license"], base_model=BASE_MODEL,
                                  model_name=REAL_MODEL_NAME, tags=list(REAL_TAGS),
                                  eval_results=eval_results(root, kind))
        inputs = ("reports/table-4.*.csv and long CSVs; reports/calibration-intervals.csv and long CSV; "
                  "reports/subgroup-deferral.csv; table-D.1/D.2; configuration hash; "
                  f"{gaps_rel} (versioned input); {RELEASE}; {INTENDED}; configs/pack/{TEMPLATES[kind]}")
    else:
        card_data = ModelCardData(language="en", license="mit", base_model=BASE_MODEL,
                                  model_name="aac-pipeline full pipeline (Phase 1 stub skeleton)",
                                  tags=["misinformation-detection", "selective-prediction", "eu-ai-act", "stub-skeleton"],
                                  eval_results=eval_results(root, kind))
        inputs = ("reports/table-4.*.csv and long CSVs; table-D.1/D.2; configuration hash; "
                  f"{KNOWN_GAPS} (versioned input); configs/pack/modelcard.template.md")
    template = template_path(root, TEMPLATES[kind]).read_text(encoding="utf-8")
    _check_fields(template, set(fields) | set(card_data.to_dict()) | {"card_data"}, FILE)
    card = ModelCard.from_template(card_data, template_str=template, **fields)
    text = str(card.content)
    require_headings(text, REQUIRED, FILE)
    missing = [h for h in ART13 if h not in text]
    if missing:
        raise PackInputError(f"{FILE}: Art. 13(3) headings missing {missing}")
    target = pack / FILE
    card.save(target)
    write_text(target, target.read_text(encoding="utf-8"))  # normalize to LF + one trailing newline
    ctx.record(FILE, GENERATED_BY, inputs)
    return [FILE]


def _real_fields(root: Path, release: dict, entry: dict, resolved: dict) -> dict:
    """Fields only the real template asks for: provider and licence lines, intended purpose, calibration intervals."""
    prov = release["provider"]
    lic = release["_licences"]
    provider_line = (f"{prov['name']}, {prov['role']} \"{prov['title']}\", {prov['university']}; supervisor "
                     f"{prov['supervisor']}; {prov['nature']} (source: {prov['source']}).")
    contact = release["contact"] if release["contact"] else CONTACT_NULL
    inc = lic["data"]["incrediblae"]
    licence_rows = [
        ["Base encoder `microsoft/mdeberta-v3-base` (the YAML `license` above)", release["license"], release["license_source"]],
        ["Fine-tuned system (release terms)", lic["data"]["averitec"]["licence"] + " (inherited from the training data)",
         release["data_terms_source"]],
        ["AVeriTeC train and dev with the knowledge store", lic["data"]["averitec"]["licence"], f"{LICENCES} data.averitec"],
        ["InCrediblAE packaging (CLEF 2024 CheckThat! Task 6)", release["incrediblae_terms"], release["incrediblae_terms_source"]],
        [f"InCrediblAE FC source corpus ({inc['FC']['source_corpus']})", inc["FC"]["licence"], f"{LICENCES} data.incrediblae.FC"],
        [f"InCrediblAE RD source corpus ({inc['RD']['source_corpus']})", inc["RD"]["licence"], f"{LICENCES} data.incrediblae.RD"],
    ]
    _, cih, cir = read_shell(reports_dir(root) / "calibration-intervals.csv")
    return dict(
        provider_line=provider_line, contact_line=contact, license_string=release["license"],
        data_terms_sentence=release["data_terms_sentence"].strip(),
        licence_table=md_table(["Component", "Licence", "Source"], licence_rows),
        intended_purpose=_text(root, INTENDED), intended_purpose_file=INTENDED,
        tci=md_table(cih, cir), template_file=f"configs/pack/{TEMPLATES['real']}",
        **{f"n_{key}": _note_block(root, file) for key, file in NOTED_SHELLS},
        declared_table=declared_table(root),
        misuse_list="\n".join(f"- {m['text'].strip()} (source: {m['source']})" for m in release["misuse"]),
        output_fields=_output_fields(root),
        groups_note=_sourced(release["groups_note"]), group_table=shell_md(root, "subgroup-deferral.csv")[1],
        group_notes=_shell_notes(root, "subgroup_deferral"),
        input_spec=input_spec(resolved),
        changes_text=_sourced(release["changes"]), checkpoint_table=_checkpoints(resolved),
        compute_text=_compute(root), lifetime_text=_sourced(release["lifetime"]),
        schema_fields=", ".join(f"`{k}`" for k in _schema(root)["properties"]),
        retention_text=_sourced(release["retention"]),
    )


def _note_block(root: Path, file: str) -> str:
    """The shell's `# note:` header lines, verbatim, as a block appended under its table ('' when it has none)."""
    path = reports_dir(root) / file
    if not path.is_file():
        raise PackInputError(f"missing report {file}; run `make reports` first")
    notes = [ln[len("# note: "):] for ln in path.read_text(encoding="utf-8").splitlines() if ln.startswith("# note: ")]
    if not notes:
        return ""
    return "\n\nNotes copied verbatim from the shell header:\n\n" + "\n".join(f"- {n}" for n in notes)


def _sourced(entry: dict) -> str:
    return f"{entry['text'].strip()} (source: {entry['source']})"


def _lookup(resolved: dict, path: str):
    obj = resolved
    for k in path.split("."):
        if not isinstance(obj, dict) or k not in obj:
            raise PackInputError(f"model card: sealed config.resolved.json has no key {path}")
        obj = obj[k]
    if obj is None or isinstance(obj, (dict, list)):
        raise PackInputError(f"model card: sealed config key {path} is not a scalar ({obj!r})")
    return obj


def _cell(value) -> str:
    """A sealed config scalar as stored (json token for numbers and booleans, the string itself otherwise)."""
    import json

    return value if isinstance(value, str) else json.dumps(value)


def declared_table(root: Path) -> str:
    """Art. 13(3)(b)(ii): every declared cell copied verbatim from its shell, with the shell, row and column."""
    rows = []
    for file, labels, cols, skip in DECLARED:
        _, hdr, body = read_shell(reports_dir(root) / file)
        missing = [c for c in cols if c not in hdr]
        if missing:
            raise PackInputError(f"model card: reports/{file} lacks the declared columns {missing}")
        for r in body:
            label = " | ".join(r[:labels])
            if skip and label.startswith(skip):
                continue
            for c in cols:
                rows.append([f"`reports/{file}`", label, c, r[hdr.index(c)]])
    if not rows:
        raise PackInputError("model card: the declared-accuracy shells hold no rows")
    return md_table(["Source shell", "Row", "Column", "Value"], rows)


def input_spec(resolved: dict) -> str:
    rows = [[label, _cell(_lookup(resolved, key)), f"`{key}`"] for label, key in INPUT_SPEC]
    return md_table(["Input", "Value", "Sealed config key"], rows)


def _checkpoints(resolved: dict) -> str:
    rows = [[label, f"`{_lookup(resolved, sha)}`", _cell(_lookup(resolved, run)), f"`{sha}`"]
            for label, sha, run in CHECKPOINTS]
    return md_table(["Frozen artifact", "SHA-256", "Producing run", "Sealed config key"], rows)


def _schema(root: Path) -> dict:
    import json

    p = Path(root) / SCHEMA
    if not p.is_file():
        raise PackInputError(f"missing contract {SCHEMA}")
    return json.loads(p.read_text(encoding="utf-8"))


def _output_fields(root: Path) -> str:
    props = _schema(root)["properties"]
    missing = [f for f in OUTPUT_FIELDS if f not in props]
    if missing:
        raise PackInputError(f"model card: {SCHEMA} lacks the output fields {missing}")
    return ", ".join(f"`{f}`" for f in OUTPUT_FIELDS)


def _shell_notes(root: Path, key: str) -> str:
    import json

    p = Path(root) / SHELLS_CONTRACT
    notes = json.loads(p.read_text(encoding="utf-8"))["tables"].get(key, {}).get("notes")
    if not notes:
        raise PackInputError(f"{SHELLS_CONTRACT}: table {key} has no notes")
    return notes


def _compute(root: Path) -> str:
    """Art. 13(3)(e): Table D.2 cells, the Table D.1 Runtime cell (rendered by the template) and the GPU host probe."""
    import json

    _, h2, r2 = read_shell(reports_dir(root) / "table-D.2-run-record.csv")
    lines = [f"- Compute used, all sealed runs (`reports/table-D.2-run-record.csv`, row Total): "
             f"{shell_value(h2, r2, 'Total', 'Configurations run')}; GPU-hours {shell_value(h2, r2, 'Total', 'GPU-hours')}.",
             "- GPU-hours per regime (same table, column GPU-hours):"]
    lines += [f"  - {r[0]}: {r[h2.index('GPU-hours')]}" for r in r2 if r[0] != "Total"]
    probes = []
    pdir = probes_dir(root)
    for p in sorted(pdir.glob("*.json")) if pdir.is_dir() else []:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("gpu"):
            probes.append(d)
    if probes:
        for d in probes:
            lines.append(f"- GPU: {d['gpu']}, {d.get('memory_total_mib', 'unknown')} MiB, NVIDIA driver "
                         f"{d.get('driver') or 'unknown'} (GPU host probe under `runs/_probes/`, fields `gpu`, "
                         "`memory_total_mib`, `driver`).")
    else:
        lines.append("- GPU: unknown (no host probe under `runs/_probes/` records a GPU).")
    return "\n".join(lines)
