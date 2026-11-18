"""`aac pack build`: the evidence-pack core. Reads sealed runs and reports/long/ only;
never recomputes metrics; writes no timestamp, host name or absolute path."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml

from aac.logging.validate import LeakScanUnavailable, fixture_texts_for, write_counts
from aac.logging.writer import SHARD_DIR
from aac.pack.decision_log import copy_sealed, data_block_key
from aac.pack.inputs import PackContext
from aac.pack.runs_manifest import check_runs_manifest
from aac.pack.verify import check_input_run
from aac.reports import fmt
from aac.run.hashing import dumps_sorted, sha256_file
from aac.run.index import load_index
from aac.run.paths import listdir
from aac.run import paths as aac_paths

D3 = "D.3"
PACK_OWN = ("manifest.json", "MANIFEST.sha256", "table-D.3-manifest.csv")
# Generators in contracts/pack-files.yaml order. decision-log.manifest.json and schema-validation.txt are
# written by copy_sealed + write_counts before these run; fig-4.1-reliability.png is copied by calib_report.
# : crosswalk (Appendix A, Table 5.1 skeleton), the analyst join and the pack index, which lists
# every file written before it (PackContext.written) and therefore runs last.
GENERATORS = ("aac.pack.datasheet", "aac.pack.modelcard", "aac.pack.abstention_spec", "aac.pack.analyst_mock",
              "aac.pack.calib_report", "aac.pack.asr_report", "aac.pack.crosswalk", "aac.pack.analyst_join",
              "aac.pack.pack_index")
BUILT_EARLIER = {"aac.pack.decision_log.copy_sealed", "aac.logging.validate.validate_shards",
                 "aac.reports.fig_4_1", "aac.pack.manifest.write_manifest", "aac.pack.manifest.build_pack"}


class NoSealedRuns(RuntimeError):
    """runs/INDEX.json lists no sealed run: the pack refuses rather than writing an empty manifest."""


# Table 4.7 is written by scripts/regen_check.sh ABOUT the pack; copying it into the pack would make the
# pack depend on the previous verification (a cycle), so it stays in reports/ only. Table 4.8 reads
# Table 4.7 (RQ4) and the analyst's verdicts, so it is built after the verification: the order is linear,
# pack -> verification (Table 4.7) -> Table 4.8, and neither table is ever copied into the pack (every copy site
# below tests NOT_COPIED_PREFIXES), so a later verdict never moves the pack.
NOT_COPIED_PREFIXES = ("table-4.7-verifiability", "table-4.8-hypotheses")


class MissingPackFile(RuntimeError):
    """A contracts/pack-files.yaml file is absent after every generator ran."""


class UnsafePackTarget(RuntimeError):
    """`--out` names a directory build_pack must not delete ."""


def check_pack_target(root: Path, pack: Path) -> None:
    """build_pack replaces `pack` wholesale, so refuse any target that is not provably a pack
 never the project root, the repository or one of their ancestors; an existing
 non-empty directory only when it is the default pack_dir (AAC_OUT-aware) or carries both pack markers
 (manifest.json and MANIFEST.sha256)."""
    target = Path(pack).resolve()
    root_r = Path(root).resolve()
    if root_r == target or root_r.is_relative_to(target):
        raise UnsafePackTarget(f"refusing to delete {pack}: it is the project root or one of its ancestors")
    if not target.exists():
        return
    if not target.is_dir():
        raise UnsafePackTarget(f"refusing to replace {pack}: not a directory")
    if not any(target.iterdir()) or target == aac_paths.pack_dir(root_r).resolve():
        return
    if not ((target / "manifest.json").is_file() and (target / "MANIFEST.sha256").is_file()):
        raise UnsafePackTarget(f"refusing to delete {pack}: not empty and not an evidence pack "
                               "(no manifest.json + MANIFEST.sha256)")


def _walk(base: Path) -> list[str]:
    out = []

    def rec(d: Path, rel: str) -> None:
        for name in listdir(d):
            p = d / name
            r = f"{rel}{name}"
            if p.is_dir():
                rec(p, r + "/")
            else:
                out.append(r)

    rec(base, "")
    return sorted(out)


def write_manifest(pack_dir: Path) -> Path:
    """sha256sum format, one line per pack file, sorted by path (LC_ALL=C order; _walk sorts)."""
    lines = [f"{sha256_file(pack_dir / rel)}  {rel}" for rel in _walk(pack_dir) if rel != "MANIFEST.sha256"]
    target = pack_dir / "MANIFEST.sha256"
    # tmp + os.replace: a kill mid-write never leaves a truncated MANIFEST.sha256. `lines` is computed before the
    # temp file exists, so it is never listed; _walk does not skip dot-files, hence the finally that removes it.
    tmp = target.with_name(f".{target.name}.tmp")
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as fh:
            fh.write("".join(line + "\n" for line in lines))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target


def _copied_report_csvs(rep_dir: Path) -> list[Path]:
    """The reports CSVs build_pack copies (top level and long/, Tables 4.7 and 4.8 excluded), in copy order."""
    out: list[Path] = []
    if rep_dir.is_dir():
        out += [rep_dir / n for n in listdir(rep_dir)
                if (rep_dir / n).is_file() and n.endswith(".csv") and not n.startswith(NOT_COPIED_PREFIXES)]
        long_dir = rep_dir / "long"
        if long_dir.is_dir():
            out += [long_dir / n for n in listdir(long_dir) if n.endswith(".csv") and not n.startswith(NOT_COPIED_PREFIXES)]
    return out


def load_pack_files(root: Path) -> list[dict]:
    with open(root / "contracts" / "pack-files.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["files"]


def partition_runs(root: Path, runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """(logged, logless), both in the given order: logged runs have a runs/<id>/decision-log/ directory.
 Every run is seal-checked by build_pack; only logged runs contribute shards, texts and counts
 (: 50 of the 182 real runs, e.g. baselines and C1/C2 training runs, write no decision log)."""
    logged, logless = [], []
    for run in runs:
        has_log = (aac_paths.runs_dir(root) / run["run_id"] / SHARD_DIR).is_dir()
        (logged if has_log else logless).append(run)
    return logged, logless


def build_pack(root: Path, out_dir: Path | None = None) -> Path:
    root = Path(root)
    pack = Path(out_dir) if out_dir else aac_paths.pack_dir(root)
    index = load_index(aac_paths.runs_dir(root) / "INDEX.json")
    runs = sorted(index["runs"], key=lambda r: r["run_id"])
    if not runs:  # refuse before touching an existing pack
        raise NoSealedRuns("runs/INDEX.json lists no sealed runs; refusing to write an empty evidence pack")
    # : the run set must equal the committed runs/MANIFEST.sha256, before the target is touched
    check_runs_manifest(root, runs)
    # : an interim `pending (...)` cell never reaches the pack; refuse before touching the target
    fmt.assert_no_pending(_copied_report_csvs(aac_paths.reports_dir(root)))
    check_pack_target(root, pack)
    if pack.exists():
        shutil.rmtree(pack)
    pack.mkdir(parents=True)
    verified_seals: dict[str, str] = {}
    for run in runs:  # every indexed run is seal-checked, logged or not
        verified_seals[run["run_id"]] = check_input_run(aac_paths.runs_dir(root) / run["run_id"], run)
    logged, _logless = partition_runs(root, runs)
    # claim texts only for runs that carry records to scan, loaded once per distinct data block
    texts_by_block: dict[str, set[str]] = {}
    block_of_run: dict[str, str] = {}
    for run in logged:
        resolved = json.loads((aac_paths.runs_dir(root) / run["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))
        key = data_block_key(resolved)
        block_of_run[run["run_id"]] = key
        if key in texts_by_block:
            continue
        try:
            texts_by_block[key] = fixture_texts_for(resolved, root)
        except LeakScanUnavailable as exc:
            raise LeakScanUnavailable(f"{run['run_id']}: {exc}") from exc
    shard_entries, counts = copy_sealed(root, logged, pack, texts_by_block, block_of_run)
    write_counts(pack / "schema-validation.txt", counts)
    config_hashes = sorted({r["config_hash"] for r in runs})
    data_versions = sorted({json.loads((aac_paths.runs_dir(root) / r["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))["data_version"] for r in runs})
    # long CSVs are inputs of later pack generators; copy the report tables so the pack carries them
    reports_dir = aac_paths.reports_dir(root)
    copied_reports = []
    if reports_dir.is_dir():
        (pack / "reports").mkdir()
        for name in listdir(reports_dir):
            p = reports_dir / name
            if p.is_file() and (name.endswith(".csv") or name.endswith(".png")) and not name.startswith(NOT_COPIED_PREFIXES):
                shutil.copyfile(p, pack / "reports" / name)
                copied_reports.append(f"reports/{name}")
        long_dir = reports_dir / "long"
        if long_dir.is_dir():
            (pack / "reports" / "long").mkdir()
            for name in listdir(long_dir):
                if name.endswith(".csv") and not name.startswith(NOT_COPIED_PREFIXES):
                    shutil.copyfile(long_dir / name, pack / "reports" / "long" / name)
                    copied_reports.append(f"reports/long/{name}")
    # manifest.json: every pack file except the pack's own closing files
    entries = []
    for e in shard_entries:
        entries.append({"file": e["file"], "generated_by": "aac.pack.decision_log.copy_sealed",
                        "inputs": f"runs/{e['run_id']}/decision-log (sealed)", "sha256": sha256_file(pack / e["file"])})
        entries.append({"file": e["file"] + ".sha256", "generated_by": "aac.logging.writer.ShardWriter",
                        "inputs": e["file"], "sha256": sha256_file(pack / (e["file"] + ".sha256"))})
    entries.append({"file": "decision-log.manifest.json", "generated_by": "aac.pack.decision_log.copy_sealed",
                    "inputs": "sealed runs listed in runs/INDEX.json", "sha256": sha256_file(pack / "decision-log.manifest.json")})
    entries.append({"file": "schema-validation.txt", "generated_by": "aac.logging.validate.validate_shards",
                    "inputs": "evidence-pack/decision-log/*.jsonl.gz", "sha256": sha256_file(pack / "schema-validation.txt")})
    for rel in sorted(copied_reports):
        entries.append({"file": rel, "generated_by": "aac.reports (copied verbatim)", "inputs": rel,
                        "sha256": sha256_file(pack / rel)})
    # the Table D.3 generators, in contracts/pack-files.yaml order (inputs only: reports/, sealed runs, configs)
    import importlib

    specs = load_pack_files(root)
    declared = []
    for s in specs:
        if s["generated_by"] not in declared:
            declared.append(s["generated_by"])
    unknown = [g for g in declared if g not in GENERATORS and g not in BUILT_EARLIER]
    if unknown:
        raise MissingPackFile(f"contracts/pack-files.yaml names generators with no implementation: {unknown}")
    ctx = PackContext(root=root, pack=pack, runs=runs, config_hashes=config_hashes, data_versions=data_versions, counts=counts,
                      verified_seals=verified_seals)
    for name in GENERATORS:
        importlib.import_module(name).build(root, pack, ctx)
    for w in ctx.written:
        entries.append({**w, "sha256": sha256_file(pack / w["file"])})
    entries.sort(key=lambda e: e["file"])
    absent = [s for s in specs if s["file"] not in PACK_OWN and not (pack / s["file"]).is_file()]
    if absent:
        raise MissingPackFile("pack files absent after generation: " + ", ".join(f"{a['file']} ({a['generated_by']})" for a in absent))
    manifest = {"files": entries, "config_hashes": config_hashes, "data_versions": data_versions}
    with open(pack / "manifest.json", "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(manifest))
    from aac.pack.crosswalk import namespace_kind

    _write_d3(root, pack, entries, absent, config_hashes, data_versions, kind=namespace_kind(root, runs))
    write_manifest(pack)
    return pack


def _inputs_of(spec: dict, kind: str) -> str:
    """The D.3 Inputs text of one pack-files.yaml entry: `inputs_real` in a real namespace when the entry has one
 (the real generator reads other files than the fixture one), else `inputs` (fixture bytes unchanged)."""
    return spec["inputs_real"] if kind == "real" and spec.get("inputs_real") else spec["inputs"]


def _row_cells(row_label: str, specs_row: list[dict], by_file: dict, kind: str) -> tuple[list[str], str, str]:
    """(files present, Generated by, Inputs) of one D.3 row; an absent file names its generator and plan."""
    present = [s for s in specs_row if s["file"] in by_file]
    missing = [s for s in specs_row if s["file"] not in by_file]
    if row_label == "evidence-pack/decision-log.jsonl; schema-validation.txt":
        files = sorted(f for f in by_file if f.startswith("decision-log/") or f in ("decision-log.manifest.json", "schema-validation.txt"))
        gen = "aac.pack.decision_log.copy_sealed; aac.logging.validate.validate_shards"
        inputs = "sealed runs listed in runs/INDEX.json (sharded, manifest-hashed log)"
    else:
        files = sorted(s["file"] for s in present)
        gen = "; ".join(dict.fromkeys(f"{s['generated_by']} (absent; Plan {s['plan']})" if s in missing else s["generated_by"]
                                      for s in specs_row))  # one name per generator, in contract order
        inputs = "; ".join(sorted({_inputs_of(s, kind) for s in specs_row}))
    return files, gen, inputs


def _digest(files: list[str], by_file: dict) -> str:
    """One file: its sha256; several: sha256 of the sorted concatenation of their sha256 values; none: n/a."""
    if not files:
        return "n/a"
    concat = "".join(sorted(by_file[f]["sha256"] for f in files)).encode("ascii")
    return hashlib.sha256(concat).hexdigest() if len(files) > 1 else by_file[files[0]]["sha256"]


def _write_d3(root: Path, pack: Path, entries: list[dict], absent: list[dict], config_hashes: list[str],
              data_versions: list[str], kind: str = "fixture") -> None:
    """The six draft D.3 shell rows in draft order, then one long-only row group per
 pack-files.yaml row label that is neither a shell row nor null (the pack's own closing files), sorted by row
 label: files outside the draft's six rows are listed beside them, never merged into a draft row."""
    shells = fmt.load_shells(root / "contracts" / "table-shells.json")
    by_file = {e["file"]: e for e in entries}
    specs = load_pack_files(root)
    ch, dv = ";".join(config_hashes), ";".join(data_versions)
    shell_rows, long_rows = [], []
    shell_labels = list(shells[D3]["rows"])
    long_only = sorted({s["row"] for s in specs if s.get("row") and s["row"] not in shell_labels})
    for row_label in shell_labels + long_only:
        specs_row = [s for s in specs if s.get("row") == row_label]
        files, gen, inputs = _row_cells(row_label, specs_row, by_file, kind)
        digest = _digest(files, by_file)
        if row_label in shell_labels:
            shell_rows.append({"file": row_label, "generated_by": gen, "inputs": inputs, "sha256": digest})
        common = dict(config_hash=ch, data_version=dv, run_ids="all sealed runs", source_shards_sha256="n/a")
        long_rows.append(fmt.long_row(D3, row_label, "Generated by", "generator", gen, **common))
        long_rows.append(fmt.long_row(D3, row_label, "Inputs", "inputs", inputs, **common))
        long_rows.append(fmt.long_row(D3, row_label, "SHA-256 at submission", "sha256_sorted_concat" if len(files) > 1 else "sha256", digest,
                                      n_items=len(files), **common))
    fmt.emit(D3, shell_rows, long_rows, config_hash=ch, data_version=dv, out_dir=pack, shells=shells)
