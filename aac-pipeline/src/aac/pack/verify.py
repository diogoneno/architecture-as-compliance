"""Pack-side refusals and verification."""
from __future__ import annotations

from pathlib import Path

from aac.run.hashing import sha256_file
from aac.run.seal import UnsealedRun, verify_sealed
from aac.run.paths import reports_dir


def check_input_run(run_dir: Path, index_entry: dict) -> str:
    sealed = verify_sealed(run_dir)
    if sealed != index_entry.get("sealed_sha256"):
        raise UnsealedRun(f"{run_dir.name}: SEALED {sealed[:12]} disagrees with runs/INDEX.json {str(index_entry.get('sealed_sha256'))[:12]}")
    return sealed


def verify_pack(pack_dir: Path) -> list[str]:
    """Return the list of files whose sha256 does not match MANIFEST.sha256 (empty = ok)."""
    bad = []
    for line in (Path(pack_dir) / "MANIFEST.sha256").read_text(encoding="ascii").splitlines():
        digest, _, rel = line.partition("  ")
        if sha256_file(Path(pack_dir) / rel) != digest:
            bad.append(rel)
    return bad


# --------------------------------------------------------------------------- Table 4.7 (RQ4)

T47 = "4.7"
T47_SUMMARY = "table-4.7-verifiability.summary.json"
COMMAND = "scripts/regen_check.sh --against-committed"
# The long-row "Regenerated on clean checkout" test text, by who regenerated the pack :
# "regen" = scripts/regen_check.sh in the working tree (the stub chain; byte-identical to the pre-05-05 text),
# "clean-checkout" = scripts/verify_clean_checkout.sh <commit> in a fresh git worktree at the commit.
HOW_TEXT = {
    "regen": ("regenerated twice from the working tree's sealed runs, committed configs and dataset hashes (T-D1, two "
              "environments; sealed runs are not in git, so no separate clean-checkout build is made) and compared "
              "with the committed evidence-pack/table-D.3-manifest.csv"),
    "clean-checkout": ("regenerated twice (T-D1, two environments) in a fresh git worktree at the commit with its own "
                       "frozen environment (uv sync --frozen from the committed uv.lock), the sealed runs its "
                       "runs/INDEX.json lists linked read-only and three raw files (AVeriTeC train.json and dev.json, "
                       "InCrediblAE FC attack.tsv) copied and hash-checked against data/raw/MANIFEST.expected.sha256, "
                       "then compared with the commit's evidence-pack/ file by file and by Table D.3 digest; RD lane B "
                       "(pseudonymized tweets) is absent there, so RD texts are not scanned for claim text in this build"),
}
# (Table 4.7 row, obligation, Table D.3 row whose files and sorted-concatenation digest the row checks)
T47_ROWS = (
    ("Dataset datasheet", "Art. 10", "evidence-pack/datasheet-averitec.md; datasheet-incrediblae.md"),
    ("Decision log and schema validation", "Art. 12", "evidence-pack/decision-log.jsonl; schema-validation.txt"),
    ("Model card", "Art. 13", "evidence-pack/model-card.md"),
    ("Abstention specification and analyst-interface mock-up", "Art. 14", "evidence-pack/abstention-spec.md; analyst-ui-mockup.html"),
    ("Calibration report", "Art. 15", "evidence-pack/calibration-report.md; fig-4.1-reliability.png"),
    ("Attack-success-rate report", "Art. 15", "evidence-pack/asr-report.md"),
)


def _d3(pack: Path) -> dict[str, dict]:
    import csv
    import io

    path = Path(pack) / "table-D.3-manifest.csv"
    if not path.is_file():
        return {}
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if not ln.startswith("#")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    return {r[0]: {"generated_by": r[1], "inputs": r[2], "sha256": r[3]} for r in rows[1:]}


def _commit(root: Path) -> str:
    import os
    import subprocess

    from aac.run.clock import SENTINEL_SHA, DirtyTree, git_sha

    if os.environ.get("AAC_GIT_SHA"):
        
        
        # labeled so Table 4.7 never presents it as a commit. Otherwise fall through to HEAD (+ -dirty).
        try:
            sha = git_sha(root)
        except (ValueError, DirtyTree, OSError, subprocess.CalledProcessError):
            sha = None
        if sha == SENTINEL_SHA:
            return f"{sha} (smoke sentinel; stub runs, not a commit)"
        if sha:
            return sha
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "."], capture_output=True, text=True,
                               check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"
    return f"{sha}-dirty" if dirty else sha


def _submitted(committed: Path, row_files: list[str], d3_hex: str) -> tuple[str, bool]:
    """(submitted digest, every comparable committed file present). The digest is recomputed from the committed
 files themselves by the Table D.3 rule (one file: its sha256; several: sha256 of the sorted concatenation of
 their sha256 values), so a committed file edited without re-hashing D.3 cannot pass on the strength of the
 D.3 line alone. Shards (`*.jsonl.gz`) are not committed (gitignored); a row that includes shards keeps the
 committed D.3 value and the caller byte-compares its other files instead."""
    if not row_files or any(f.endswith(".jsonl.gz") for f in row_files):
        return d3_hex, True
    paths = [Path(committed) / f for f in row_files]
    if not all(p.is_file() for p in paths):
        return "absent", False
    hexes = sorted(sha256_file(p) for p in paths)
    if len(hexes) == 1:
        return hexes[0], True
    import hashlib

    return hashlib.sha256("".join(hexes).encode("ascii")).hexdigest(), True


def _same_bytes(committed: Path, regenerated: Path, rel: str) -> bool:
    """The committed copy of `rel` is byte-identical to the regenerated one. A decision-log/ sidecar may be absent
 only when the committed pack has no decision-log/ directory at all (the stub golden pack, which `make
 golden-diff` compares with --exclude=decision-log); its digests are still checked through the D.3 row."""
    c = Path(committed) / rel
    if not c.is_file():
        return rel.startswith("decision-log/") and not (Path(committed) / "decision-log").is_dir()
    return c.read_bytes() == (Path(regenerated) / rel).read_bytes()


def write_table_4_7(root: Path, regenerated: Path, second: Path, committed: Path, *, t_d1_identical: bool,
                    out_dir: Path | None = None, command: str = COMMAND, how: str = "regen") -> tuple[Path, Path, bool]:
    """Write reports/table-4.7-verifiability.csv (+ long sidecar) and its summary JSON.

 Per row: regenerated = every file of the D.3 row exists in the regenerated pack; Hash match = first 12
 hex of the regenerated D.3 digest + identical/DIFFERS against the committed pack's D.3 digest; Result =
 pass only when regenerated, identical to the committed pack and identical across the two builds.
 `how` picks the long-row "Regenerated on clean checkout" text (HOW_TEXT key); `command` is the summary's
 command string. The submitted digest is recomputed from the committed files where they are committed
 (`_submitted`), and every non-shard file of the row must be byte-identical to the regenerated one.
 Returns (shell path, summary path, all rows pass)."""
    import json

    from aac.logging.validate import read_counts
    from aac.reports import fmt
    from aac.run.hashing import dumps_sorted

    if how not in HOW_TEXT:
        raise ValueError(f"how must be one of {sorted(HOW_TEXT)}, not {how!r}")
    root, regenerated, second, committed = Path(root), Path(regenerated), Path(second), Path(committed)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    man = json.loads((regenerated / "manifest.json").read_text(encoding="ascii"))
    ch, dv = ";".join(man["config_hashes"]), ";".join(man["data_versions"])
    files = {e["file"] for e in man["files"]}
    d_regen, d_second, d_comm = _d3(regenerated), _d3(second), _d3(committed)
    shell_rows, long_rows, all_pass = [], [], True
    common = dict(config_hash=ch, data_version=dv, run_ids="all sealed runs", source_shards_sha256="n/a")
    for label, obligation, d3_row in T47_ROWS:
        reg = d_regen.get(d3_row)
        if reg is None:
            raise RuntimeError(f"regenerated pack has no Table D.3 row {d3_row!r}")
        names = [n.strip() for n in d3_row.replace("evidence-pack/", "").split(";")]
        if d3_row.startswith("evidence-pack/decision-log.jsonl"):
            present = "decision-log.manifest.json" in files and "schema-validation.txt" in files
            row_files = sorted(f for f in files if f.startswith("decision-log/") or f in ("decision-log.manifest.json", "schema-validation.txt"))
        else:
            present = all(n in files for n in names)
            row_files = sorted(n for n in names if n in files)
        regen_hex = reg["sha256"]
        sub_hex, sub_complete = _submitted(committed, row_files, d_comm.get(d3_row, {}).get("sha256", "absent"))
        same_bytes = sub_complete and all(_same_bytes(committed, regenerated, f) for f in row_files if not f.endswith(".jsonl.gz"))
        identical = present and regen_hex == sub_hex and regen_hex != "n/a" and same_bytes
        stable = d_second.get(d3_row, {}).get("sha256") == regen_hex and t_d1_identical
        ok = identical and stable
        all_pass = all_pass and ok
        row = {"artifact": label, "obligation": obligation, "generated_by": reg["generated_by"],
               "regenerated_clean_checkout": "yes" if present else "no",
               "hash_match": f"{regen_hex[:12]} {'identical' if identical else 'DIFFERS'}",
               "result": "pass" if ok else "fail"}
        shell_rows.append(row)
        how_text = HOW_TEXT[how]
        long_rows += [
            fmt.long_row(T47, label, "Obligation", "obligation", obligation, **common),
            fmt.long_row(T47, label, "Generated by", "generated_by", reg["generated_by"], test=f"Table D.3 row {d3_row}", **common),
            fmt.long_row(T47, label, "Regenerated on clean checkout", "regenerated", row["regenerated_clean_checkout"],
                         n_items=len(names), test=how_text, **common),
            fmt.long_row(T47, label, "Hash match", "hash_match", row["hash_match"], test="sha256 (sorted concatenation for multi-file rows)", **common),
            fmt.long_row(T47, label, "Hash match (sha256 submitted)", "sha256_submitted", sub_hex, **common),
            fmt.long_row(T47, label, "Hash match (sha256 regenerated)", "sha256_regenerated", regen_hex, **common),
            fmt.long_row(T47, label, "Result", "result", row["result"],
                         test="pass = regenerated, identical to the committed pack and identical across both T-D1 builds", **common),
        ]
    shell, _ = fmt.emit(T47, shell_rows, long_rows, config_hash=ch, data_version=dv, out_dir=out_dir)
    counts = read_counts(regenerated / "schema-validation.txt")
    summary = {"n_records": counts.records, "violations": counts.violations,
               "records_with_claim_text": counts.records_with_claim_text, "records_with_user_id": counts.records_with_user_id,
               "commit": _commit(root), "command": command}
    target = out_dir / T47_SUMMARY
    with open(target, "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(summary))
    return shell, target, all_pass
