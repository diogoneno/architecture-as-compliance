"""The pack index: evidence-pack/index.md .

Runs after every other pack generator and before manifest.json (places it last in GENERATORS). Per pack
file written so far: the Article (configs/crosswalk.csv; `n/a (not in the crosswalk)` when the crosswalk names no
row for it), the generator, the inputs and the SHA-256 of the file as written, in the order build_pack wrote them
(decision-log.manifest.json and schema-validation.txt first, then PackContext.written); one line each counts the
decision-log shard files and the copied report files. Then the intended-purpose / out-of-scope statement (versioned
input configs/pack/intended-purpose.md, copied verbatim) and the change control: the v1.0 commit from
runs/INDEX.json day1_docs and, per run name, the distinct config_hash and release git_sha values of the sealed runs
(each run checked against runs/INDEX.json and its seal recomputed, as calib_report does). No date or timestamp is
written; the page is a pure function of committed inputs and of the files already in the pack."""
from __future__ import annotations

from pathlib import Path

from aac.pack.crosswalk import article_key, load_crosswalk
from aac.pack.inputs import (PackContext, PackInputError, index_entries, md_table, read_json, render,
                             require_headings, write_text)
from aac.run.hashing import sha256_file
from aac.run.paths import runs_dir
from aac.run.seal import UnsealedRun, verify_sealed

GENERATED_BY = "aac.pack.pack_index"
FILE = "index.md"
TEMPLATE = "index.template.md"
REQUIRED = ["Pack index", "Intended purpose and out of scope", "Change control"]
INTENDED_PURPOSE = ("configs", "pack", "intended-purpose.md")
NOT_IN_CROSSWALK = "n/a (not in the crosswalk)"
# written by build_pack before the generators run (generator and inputs as in contracts/pack-files.yaml)
BEFORE_GENERATORS = ("decision-log.manifest.json", "schema-validation.txt")
# written after the index (or the index itself): never listed
NOT_LISTED = (FILE, "manifest.json", "table-D.3-manifest.csv", "MANIFEST.sha256")
EXCLUSION = ("It does not list itself (`index.md`), `manifest.json`, `table-D.3-manifest.csv` or `MANIFEST.sha256`, which are written "
             "after it (`MANIFEST.sha256` hashes every pack file, this index included).")
FILES_HEADER = ["File", "Article (Appendix A)", "Generator", "Inputs", "SHA-256"]
CHANGE_HEADER = ["Run name", "Sealed runs (n)", "config_hash (distinct)", "Release git_sha (distinct)"]


def _count_files(d: Path) -> int:
    return sum(1 for p in d.rglob("*") if p.is_file()) if d.is_dir() else 0


def _articles(root: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for row in load_crosswalk(root):
        keys = out.setdefault(row["artifact_file"], [])
        if article_key(row) not in keys:
            keys.append(article_key(row))
    return out


def _change_control(root: Path, runs: list[dict], verified: dict[str, str] | None = None) -> list[list[str]]:
    """Per run name: sealed runs, distinct config_hash and release git_sha. Each run must equal its INDEX entry and
 carry the INDEX seal: `verified` holds the digests build_pack recomputed in this process (decision:
 passed through PackContext instead of hashing the 182 real runs a second time, about 66 s per build); a run
 without one is recomputed here."""
    index = {e["run_id"]: e for e in index_entries(root)}
    verified = verified or {}
    by_name: dict[str, tuple[list, set, set]] = {}
    for run in runs:
        rid = run["run_id"]
        if index.get(rid) != run:
            raise PackInputError(f"{rid}: the pack's run entry disagrees with runs/INDEX.json")
        run_dir = runs_dir(root) / rid
        try:
            sealed = verified[rid] if rid in verified else verify_sealed(run_dir)
        except UnsealedRun as exc:
            raise PackInputError(f"{rid}: {exc}") from exc
        if sealed != run["sealed_sha256"]:
            raise PackInputError(f"{rid}: SEALED disagrees with runs/INDEX.json")
        rr = read_json(run_dir / "run-record.json")
        if rr.get("config_hash") != run["config_hash"]:
            raise PackInputError(f"{rid}: run-record.json config_hash disagrees with runs/INDEX.json")
        if not rr.get("name") or not rr.get("git_sha"):
            raise PackInputError(f"{rid}: run-record.json lacks name or git_sha")
        ids, hashes, shas = by_name.setdefault(rr["name"], ([], set(), set()))
        ids.append(rid)
        hashes.add(rr["config_hash"])
        shas.add(rr["git_sha"])
    return [[name, str(len(ids)), "; ".join(sorted(h)), "; ".join(sorted(s))]
            for name, (ids, h, s) in sorted(by_name.items())]


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    ip = root.joinpath(*INTENDED_PURPOSE)
    if not ip.is_file():
        raise PackInputError(f"missing versioned input {'/'.join(INTENDED_PURPOSE)}")
    intended = ip.read_text(encoding="utf-8").strip()
    if not intended:
        raise PackInputError(f"{'/'.join(INTENDED_PURPOSE)} is empty")
    articles = _articles(root)
    import yaml

    with open(root / "contracts" / "pack-files.yaml", encoding="utf-8") as fh:
        specs = {s["file"]: s for s in yaml.safe_load(fh)["files"]}
    listed = [{"file": f, "generated_by": specs[f]["generated_by"], "inputs": specs[f]["inputs"]} for f in BEFORE_GENERATORS]
    listed += [w for w in ctx.written if w["file"] not in BEFORE_GENERATORS and w["file"] not in NOT_LISTED]
    seen, rows = set(), []
    for w in listed:
        f = w["file"]
        if f in seen:
            raise PackInputError(f"pack index: {f} was recorded twice")
        seen.add(f)
        if not (pack / f).is_file():
            raise PackInputError(f"pack index: {f} is recorded but absent from the pack")
        rows.append([f, "; ".join(articles.get(f, [])) or NOT_IN_CROSSWALK, w["generated_by"], w["inputs"],
                     sha256_file(pack / f)])
    shards = _count_files(pack / "decision-log")
    reports = _count_files(pack / "reports")
    index = read_json(runs_dir(root) / "INDEX.json")
    spec_ref = (index.get("day1_docs") or {}).get("spec_v1_commit", "N/A")
    if not spec_ref:
        spec_ref = "N/A"
    text = render(root, TEMPLATE, generated_by=GENERATED_BY, exclusion=EXCLUSION,
                  files_table=md_table(FILES_HEADER, rows),
                  shards_line=(f"Decision log: {shards} files under decision-log/ (sealed shards and their .sha256 "
                               "sidecars, copied by aac.pack.decision_log.copy_sealed; each hashed in manifest.json "
                               "and MANIFEST.sha256)."),
                  reports_line=(f"Reports: {reports} files under reports/ (shell CSVs, long CSVs and figures copied "
                                "verbatim from the namespace's reports/; each hashed in manifest.json and "
                                "MANIFEST.sha256)."),
                  intended_purpose=intended, spec_commit=spec_ref, n_runs=len(ctx.runs),
                  change_table=md_table(CHANGE_HEADER, _change_control(root, ctx.runs, ctx.verified_seals)))
    require_headings(text, REQUIRED, FILE)
    write_text(pack / FILE, text)
    ctx.record(FILE, GENERATED_BY, "the pack files written before it; configs/crosswalk.csv; "
               f"{'/'.join(INTENDED_PURPOSE)}; runs/INDEX.json; sealed run-record.json files")
    return [FILE]
