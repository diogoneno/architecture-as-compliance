"""Appendix A crosswalk and the Table 5.1 skeleton: evidence-pack/crosswalk.md .

Inputs are versioned and committed: configs/crosswalk.csv (one row per Article or Article limb and pack artifact,
seeded from draft Table A.1 with the draft line as `source`) and configs/pack/table-5-1-skeleton.yaml (per draft
Table 5.1 row: the Table 3.1 claim, the Appendix A articles, and references into committed report shells). The page
carries the banner "Readiness pointers, not conformity claims" (Appendix A wording, draft-v0.5.md:583): every
ISO/IEC 42001 control is shown with its `iso_verified` flag and every JTC 21 pointer with its status.

`What Chapter 4 showed` holds `<label>: <value> (<file> row <row>, column <column>)`, the value copied with
shell_value (never computed); C3 uses the pack's own schema-validation counts (PackContext.counts). The Verdict is
the analyst's the generator prints a fixed placeholder. Table 4.7 and Table 4.8 are never read (they are
outside the pack). In a fixture-kind namespace (the stub chain) a reference to a real-only report renders
`not in this namespace (real-only table)`; any other reference that resolves to no single shell cell raises
PackInputError naming the row."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import yaml

from aac.pack.inputs import (PackContext, PackInputError, md_table, read_shell, render, require_headings,
                             shell_value, write_text)
from aac.run.paths import reports_dir, runs_dir

GENERATED_BY = "aac.pack.crosswalk"
FILE = "crosswalk.md"
TEMPLATE = "crosswalk.template.md"
REQUIRED = ["Appendix A", "Table 5.1 skeleton"]
BANNER = "Readiness pointers, not conformity claims"
BANNER_SENTENCE = ("Draft-standard references are readiness pointers, not conformity claims: no harmonised standard had "
                   "been cited in the Official Journal by October 2026, and each draft's scope will be fixed by its "
                   "Annex ZA when published.")
BANNER_SOURCE = "draft-v0.5.md:583"
CROSSWALK = ("configs", "crosswalk.csv")
SKELETON = ("configs", "pack", "table-5-1-skeleton.yaml")
COLUMNS = ["article", "paragraph", "obligation", "component", "artifact_file", "nist_subcategory", "iso_control",
           "iso_verified", "jtc21_pointer", "jtc21_status", "other_pointer", "source"]
ISO_FLAGS = ("verified", "unverified")
JTC21_FLAGS = ("draft",)

PHASE5_FILES = ("crosswalk.md", "index.md", "analyst-events.jsonl", "analyst-join.json")
TABLE_51_HEADER = ["Component and Article (claim as in Table 3.1)", "What Chapter 4 showed", "Verdict",
                   "Crosswalk status (Appendix A)"]
VERDICT = "for the analyst (Phase 7)"
NOT_HERE = "not in this namespace (real-only table)"
# the report shells the skeleton may copy (Table 4.7 and Table 4.8 are outside the pack and never read)
ALLOWED_SHELLS = ("table-4.4-r3.csv", "table-4.5-ablations.csv", "table-4.6-coverage-risk.csv",
                  "calibration-intervals.csv", "deferral_reasons.csv")
COUNT_KEYS = {"records": "Decision-log records (schema-validation.txt records)",
              "violations": "Schema violations (schema-validation.txt violations)",
              "records_with_claim_text": "Records with claim text (schema-validation.txt records_with_claim_text)",
              "records_with_user_id": "Records with a user id (schema-validation.txt records_with_user_id)"}
APPENDIX_HEADER = ["AI Act Article", "Obligation", "Component", "Pack artifact", "NIST AI RMF", "ISO/IEC 42001",
                   "JTC 21", "Other pointer", "Source"]


def pack_file_names(root: Path) -> set[str]:
    with open(Path(root) / "contracts" / "pack-files.yaml", encoding="utf-8") as fh:
        files = {s["file"] for s in yaml.safe_load(fh)["files"]}
    return files | set(PHASE5_FILES)


def article_key(row: dict) -> str:
    return f"{row['article']} ({row['paragraph']})" if row["paragraph"] else row["article"]


def load_crosswalk(root: Path) -> list[dict]:
    """The rows of configs/crosswalk.csv in file order, validated: exact header, the two flag vocabularies, a known pack
 file per row and a draft-line source. Any violation raises PackInputError naming the row."""
    root = Path(root)
    p = root.joinpath(*CROSSWALK)
    if not p.is_file():
        raise PackInputError(f"missing versioned input {'/'.join(CROSSWALK)}")
    reader = csv.reader(io.StringIO(p.read_text(encoding="utf-8")))
    rows = [r for r in reader if r]
    if not rows or rows[0] != COLUMNS:
        raise PackInputError(f"{'/'.join(CROSSWALK)}: header must be {','.join(COLUMNS)}")
    known = pack_file_names(root)
    out = []
    for n, r in enumerate(rows[1:], start=2):
        if len(r) != len(COLUMNS):
            raise PackInputError(f"{'/'.join(CROSSWALK)} row {n}: {len(r)} fields, expected {len(COLUMNS)}")
        row = dict(zip(COLUMNS, r))
        where = f"{'/'.join(CROSSWALK)} row {n} ({article_key(row)}, {row['artifact_file']})"
        if row["iso_verified"] not in ISO_FLAGS:
            raise PackInputError(f"{where}: iso_verified {row['iso_verified']!r} is not one of {list(ISO_FLAGS)}")
        if row["jtc21_status"] not in JTC21_FLAGS:
            raise PackInputError(f"{where}: jtc21_status {row['jtc21_status']!r} is not one of {list(JTC21_FLAGS)}")
        if row["artifact_file"] not in known:
            raise PackInputError(f"{where}: artifact_file {row['artifact_file']!r} is not a pack file "
                                 f"(contracts/pack-files.yaml or {', '.join(PHASE5_FILES)})")
        if not row["article"] or not row["source"].startswith("draft-v0.5.md:"):
            raise PackInputError(f"{where}: needs an article and a `draft-v0.5.md:<line>` source")
        out.append(row)
    if not out:
        raise PackInputError(f"{'/'.join(CROSSWALK)}: no rows")
    return out


def group_articles(rows: list[dict]) -> list[dict]:
    """One entry per Article (limb) in source order; the pointer fields must agree across its artifact rows."""
    groups: dict[str, dict] = {}
    for row in rows:
        key = article_key(row)
        if key not in groups:
            groups[key] = {**row, "key": key, "files": [row["artifact_file"]]}
            continue
        g = groups[key]
        for col in COLUMNS:
            if col != "artifact_file" and g[col] != row[col]:
                raise PackInputError(f"{'/'.join(CROSSWALK)}: rows of {key} disagree on {col}")
        g["files"].append(row["artifact_file"])
    return list(groups.values())


def iso_cell(g: dict) -> str:
    return f"{g['iso_control'] or 'none in Table A.1'} ({g['iso_verified']})"


def jtc_cell(g: dict) -> str:
    return f"{g['jtc21_pointer']} (status: {g['jtc21_status']})" if g["jtc21_pointer"] else "none in Table A.1"


def status_cell(g: dict) -> str:
    parts = [g["other_pointer"], g["nist_subcategory"], f"{g['iso_control']} ({g['iso_verified']})" if g["iso_control"] else "",
             f"{g['jtc21_pointer']} (status: {g['jtc21_status']})" if g["jtc21_pointer"] else ""]
    return "; ".join(p for p in parts if p)


def load_skeleton(root: Path) -> list[dict]:
    p = Path(root).joinpath(*SKELETON)
    if not p.is_file():
        raise PackInputError(f"missing versioned input {'/'.join(SKELETON)}")
    comps = (yaml.safe_load(p.read_text(encoding="utf-8")) or {}).get("components")
    if not isinstance(comps, list) or not comps:
        raise PackInputError(f"{'/'.join(SKELETON)}: no components")
    return comps


def real_only_files(root: Path) -> set[str]:
    """Report files written only in a real namespace (aac.reports.registry.REAL_ONLY ids that have a shell)."""
    from aac.reports import fmt
    from aac.reports.registry import REAL_ONLY

    shells = fmt.load_shells(Path(root) / "contracts" / "table-shells.json")
    return {shells[t]["file"] for t in REAL_ONLY if t in shells}


def namespace_kind(root: Path, runs: list[dict]) -> str:
    """`fixture` or `real`: the datasheet profile kind of the one data manifest the namespace's runs name."""
    from aac.pack.datasheet import load_profile

    manifests = set()
    for e in runs:
        resolved = json.loads((runs_dir(root) / e["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))
        manifests.add(resolved.get("data", {}).get("manifest"))
    if len(manifests) != 1:
        raise PackInputError(f"crosswalk: the namespace's runs name {len(manifests)} data manifests: {sorted(map(str, manifests))}")
    return load_profile(root, next(iter(manifests)))["kind"]


def _cell(root: Path, ref: dict, kind: str, real_only: set[str], cache: dict, claim: str) -> str:
    missing = [k for k in ("file", "row", "column", "label") if not ref.get(k)]
    if missing:
        raise PackInputError(f"Table 5.1 skeleton row {claim!r}: cell reference lacks {missing}")
    f, row, col, label = ref["file"], ref["row"], ref["column"], ref["label"]
    where = f"({f} row {row}, column {col})"
    if f not in ALLOWED_SHELLS:
        raise PackInputError(f"Table 5.1 skeleton row {claim!r}: {f} is not a shell the skeleton may copy {list(ALLOWED_SHELLS)}")
    if kind != "real" and f in real_only:
        return f"{label}: {NOT_HERE} {where}"
    if f not in cache:
        cache[f] = read_shell(reports_dir(root) / f)
    _, header, rows = cache[f]
    if col not in header:
        raise PackInputError(f"Table 5.1 skeleton row {claim!r}: {f} has no column {col!r}")
    try:
        value = shell_value(header, rows, row, col, label_columns=row.count(" | ") + 1)
    except PackInputError as exc:
        raise PackInputError(f"Table 5.1 skeleton row {claim!r}: {f}: {exc}") from exc
    return f"{label}: {value} {where}"


def skeleton_rows(root: Path, ctx: PackContext, groups: list[dict], kind: str) -> list[list[str]]:
    by_key = {g["key"]: g for g in groups}
    known = pack_file_names(root)
    real_only = real_only_files(root)
    cache: dict = {}
    out = []
    for comp in load_skeleton(root):
        claim = comp.get("claim")
        if not claim or not str(comp.get("source", "")).startswith("draft-v0.5.md:"):
            raise PackInputError(f"{'/'.join(SKELETON)}: every component needs a claim and a draft-v0.5.md source")
        shown = []
        files = comp.get("pack_files") or []
        if files:
            unknown = [f for f in files if f not in known]
            if unknown:
                raise PackInputError(f"Table 5.1 skeleton row {claim!r}: {unknown} are not pack files")
            shown.append("Pack files: " + "; ".join(files))
        for key in comp.get("pack_counts") or []:
            if key not in COUNT_KEYS:
                raise PackInputError(f"Table 5.1 skeleton row {claim!r}: unknown pack count {key!r}")
            if ctx.counts is None:
                raise PackInputError(f"Table 5.1 skeleton row {claim!r}: the pack's schema-validation counts are absent")
            shown.append(f"{COUNT_KEYS[key]}: {getattr(ctx.counts, key)}")
        for ref in comp.get("cells") or []:
            shown.append(_cell(root, ref, kind, real_only, cache, claim))
        if not shown:
            raise PackInputError(f"Table 5.1 skeleton row {claim!r}: nothing to show")
        status = []
        for key in comp.get("crosswalk_rows") or []:
            if key not in by_key:
                raise PackInputError(f"Table 5.1 skeleton row {claim!r}: crosswalk row {key!r} is not in configs/crosswalk.csv")
            status.append(status_cell(by_key[key]))
        if not status:
            raise PackInputError(f"Table 5.1 skeleton row {claim!r}: no crosswalk row")
        out.append([claim, "; ".join(shown), VERDICT, "; ".join(status)])
    return out


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    groups = group_articles(load_crosswalk(root))
    kind = namespace_kind(root, ctx.runs)
    appendix = [[g["key"], g["obligation"], g["component"], "; ".join(g["files"]), g["nist_subcategory"] or "none in Table A.1",
                 iso_cell(g), jtc_cell(g), g["other_pointer"] or "none in Table A.1", g["source"]] for g in groups]
    skeleton = skeleton_rows(root, ctx, groups, kind)
    text = render(root, TEMPLATE, banner=BANNER, banner_sentence=BANNER_SENTENCE, banner_source=BANNER_SOURCE,
                  generated_by=GENERATED_BY, appendix_table=md_table(APPENDIX_HEADER, appendix),
                  skeleton_table=md_table(TABLE_51_HEADER, skeleton))
    require_headings(text, REQUIRED, FILE)
    write_text(pack / FILE, text)
    used = sorted({c["file"] for comp in load_skeleton(root) for c in (comp.get("cells") or [])})
    ctx.record(FILE, GENERATED_BY, "configs/crosswalk.csv; configs/pack/table-5-1-skeleton.yaml; "
               + "; ".join(f"reports/{f}" for f in used) + "; schema-validation counts")
    return [FILE]
