"""`aac prisma workbook init|verify` (PRISMA-01, 06-CONTEXT D1).

init copies the canonical v05 prisma_records.csv field for field into appendix-c/screening/workbook.csv
 (11 original columns, then schema.ADDED_COLUMNS), after asserting the run is the canonical one:
 unique_across_sources == 3491, every failed_queries_per_source value 0, row count 3491 and the record
 keys equal to the union of prisma_ids.json. It writes INIT.md (which run is canonical and why, the column
 table) and, only when absent, RUN-FOLDER.sha256: the sha256sum manifest of every file of every
 appendix-c/run-* folder (paths with a __pycache__ component excluded). An existing manifest is verified,
 never rewritten, so re-running init cannot bless a modified run folder. A workbook that holds any
 decision cell is never overwritten, with or without --force.
verify re-checks the copy (header, row order, keys and the 11 original fields as strings: the source is CRLF,
 the workbook LF, so never bytes) and re-hashes the run folders against the committed manifest. Six of the
 11 source columns are decision columns (screen_decision ... included_in_chapter2) that the source holds as
 empty templates: a field there counts as equal when it equals the source, or when the source field is empty
 and the analyst's merged decision fills it . Identification fields must always be equal.

Exit codes: 0 when everything holds, 1 when a check fails or init refuses, 2 for usage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from aac.prisma import schema
from aac.prisma._io import (
    OutsideScreening,
    csv_text,
    default_appendix_dir,
    read_csv,
    screening_dir,
    workbook_lock,
    write_text_atomic,
)
from aac.run.hashing import sha256_file

WORKBOOK = "workbook.csv"
INIT_MD = "INIT.md"
MANIFEST = "RUN-FOLDER.sha256"
MANIFEST_RE = re.compile(r"^([0-9a-f]{64})  (run-[^/\s][^/]*/[^\n]+)$")
ARXIV_PREFIX = "10.48550/arxiv."


class CanonicalRunError(RuntimeError):
    """The source run folder fails a canonical assertion (D1): the workbook is not built from it."""


class RunFolderChanged(RuntimeError):
    """A run-* folder differs from the committed RUN-FOLDER.sha256 (changed digest, removed or added file)."""


class WorkbookRefused(RuntimeError):
    """init will not overwrite the existing workbook."""


# ------------------------------------------------------------------ derived columns


def title_norm(t: str | None) -> str:
    """The v05 title key without its cut-off.

 prisma_search_v05.py line 44: re.sub(r"\\W", "", unicodedata.normalize("NFKC", t or "").casefold), used
 as a key only when >= 25 characters. Here short titles are grouped too: two rows titled "Deep Fakes" are as
 much a candidate duplicate for the screener as two long ones (operational constraint)."""
    return re.sub(r"\W", "", unicodedata.normalize("NFKC", t or "").casefold())


def title_dup_groups(rows: list[dict[str, str]]) -> dict[str, str]:
    """{key: group id} for rows whose non-empty normalized title is shared by two or more rows.

 Ids g0001, g0002, ... follow the order of each group's smallest key. A hint for the screener, never a
 decision: same-title records are not always the same work."""
    by_title: dict[str, list[str]] = {}
    for r in rows:
        n = title_norm(r.get("title"))
        if n:
            by_title.setdefault(n, []).append(r["key"])
    groups = sorted((sorted(keys) for keys in by_title.values() if len(keys) > 1), key=lambda ks: ks[0])
    out: dict[str, str] = {}
    for i, keys in enumerate(groups, start=1):
        for k in keys:
            out[k] = f"g{i:04d}"
    return out


def doi_url(key: str) -> str:
    """Resolver URL from the record key, without network; empty for a normalized-title key."""
    if key.startswith(ARXIV_PREFIX):
        return f"https://arxiv.org/abs/{key[len(ARXIV_PREFIX):]}"
    if key.startswith("10."):
        return f"https://doi.org/{key}"
    if key.startswith("https://"):
        return key
    return ""


# ------------------------------------------------------------------ canonical run


def _rel(path: Path, appendix: Path) -> str:
    """A path for messages: relative to the appendix's parent (the dissertation root for the real one)."""
    p, base = Path(path).resolve(), Path(appendix).resolve().parent
    return p.relative_to(base).as_posix() if p.is_relative_to(base) else str(path)


def check_canonical(run_dir: Path, expected_unique: int) -> dict:
    """Assert the four D1 facts on run_dir; return the values read. Raises CanonicalRunError naming the failure."""
    run_dir = Path(run_dir)
    for name in ("prisma_summary.json", "prisma_ids.json", "prisma_records.csv"):
        if not (run_dir / name).is_file():
            raise CanonicalRunError(f"{run_dir.name}: {name} is absent")
    summary = json.loads((run_dir / "prisma_summary.json").read_text(encoding="utf-8"))
    unique = summary.get("unique_across_sources")
    if unique != expected_unique:
        raise CanonicalRunError(
            f"{run_dir.name}: unique_across_sources is {unique!r}, expected {expected_unique}")
    failed = summary.get("failed_queries_per_source")
    if not isinstance(failed, dict) or not failed or any(v != 0 for v in failed.values()):
        raise CanonicalRunError(f"{run_dir.name}: failed_queries_per_source is {failed!r}, expected every value 0")
    header, rows = read_csv(run_dir / "prisma_records.csv")
    if tuple(header) != schema.ORIGINAL_COLUMNS:
        raise CanonicalRunError(f"{run_dir.name}: prisma_records.csv header is {header!r}")
    if len(rows) != expected_unique:
        raise CanonicalRunError(f"{run_dir.name}: prisma_records.csv has {len(rows)} rows, expected {expected_unique}")
    keys = [r["key"] for r in rows]
    if len(set(keys)) != len(keys):
        raise CanonicalRunError(f"{run_dir.name}: prisma_records.csv repeats {len(keys) - len(set(keys))} key(s)")
    ids = json.loads((run_dir / "prisma_ids.json").read_text(encoding="utf-8"))
    union: set[str] = set()
    for v in ids.values():
        union |= set(v)
    if set(keys) != union:
        raise CanonicalRunError(
            f"{run_dir.name}: record keys differ from the prisma_ids.json union "
            f"({len(set(keys) - union)} only in records, {len(union - set(keys))} only in ids)")
    return {
        "run": run_dir.name,
        "run_date": summary.get("run_date"),
        "failed_queries_per_source": failed,
        "unique_across_sources": unique,
        "rows": len(rows),
        "union": len(union),
        "rows_equal_union": True,
        "header": header,
        "records": rows,
    }


# ------------------------------------------------------------------ run-folder manifest


def manifest_lines(appendix: Path) -> list[str]:
    """`<sha256> <path relative to appendix>` for every regular file of every run-* folder, byte-sorted.

 Paths with a __pycache__ component are skipped (one already exists in run-20261013-0915Z-v03/;
 operational constraint)."""
    appendix = Path(appendix)
    out: list[tuple[bytes, str]] = []
    for d in sorted(p for p in appendix.glob("run-*") if p.is_dir()):
        for p in d.rglob("*"):
            rel_parts = p.relative_to(appendix).parts
            if "__pycache__" in rel_parts or not p.is_file():
                continue
            rel = p.relative_to(appendix).as_posix()
            if not rel.isascii() or "\\" in rel or "\n" in rel:
                raise RunFolderChanged(f"{rel!r}: a path sha256sum cannot list plainly")
            out.append((rel.encode("ascii"), f"{sha256_file(p)}  {rel}"))
    return [line for _, line in sorted(out)]


def _parse_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="ascii").split("\n"), start=1):
        if not line:
            continue
        m = MANIFEST_RE.match(line)
        if not m or m.group(2) in entries:
            raise RunFolderChanged(f"{MANIFEST} line {n} is malformed or repeated: {line!r}")
        entries[m.group(2)] = m.group(1)
    return entries


def manifest_diff(appendix: Path, lines: list[str] | None = None) -> list[str]:
    """Differences between the committed manifest and the run folders now, each naming its path."""
    target = screening_dir(appendix) / MANIFEST
    if not target.is_file():
        return [f"{_rel(target, appendix)} (absent)"]
    recorded = _parse_manifest(target)
    now = {line[66:]: line[:64] for line in (manifest_lines(appendix) if lines is None else lines)}
    diffs = []
    for path in sorted(set(recorded) | set(now), key=lambda s: s.encode("ascii")):
        if path not in now:
            diffs.append(f"{path} (removed)")
        elif path not in recorded:
            diffs.append(f"{path} (added)")
        elif now[path] != recorded[path]:
            diffs.append(f"{path} (digest differs)")
    return diffs


def _folders(lines: list[str]) -> int:
    return len({line[66:].split("/", 1)[0] for line in lines})


# ------------------------------------------------------------------ init


@dataclass
class InitResult:
    rows: int
    columns: int
    run: str
    title_dup_groups: int
    title_dup_rows: int
    manifest_files: int
    manifest_folders: int
    manifest_action: str  # "written" or "verified"
    workbook_action: str  # "written" or "unchanged"
    paths: dict[str, Path] = field(default_factory=dict)


def _superseded_line(appendix: Path) -> str | None:
    readme = Path(appendix) / schema.SUPERSEDED_RUN / "README.md"
    if not readme.is_file():
        return None
    for line in readme.read_text(encoding="utf-8").splitlines():
        if "SUPERSEDED" in line:
            return line.strip()
    return None


def init_md(obs: dict, superseded: str | None, groups: int, dup_rows: int, n_files: int, n_folders: int) -> str:
    """INIT.md: deterministic (no clock); the only time in it is the quoted prisma_summary.json run_date."""
    failed = ", ".join(f"{k} {v}" for k, v in obs["failed_queries_per_source"].items())
    lines = [
        "# Screening workbook: canonical run record",
        "",
        "Generated by `aac prisma workbook init`; regenerate, never edit. The workbook of record is",
        "`appendix-c/screening/workbook.csv`.",
        "",
        "## Canonical run",
        "",
        f"Canonical run folder: `appendix-c/{obs['run']}/` (source file `prisma_records.csv`).",
        "",
        "Assertions checked by init before the workbook was written (values as read):",
        "",
        f"- prisma_summary.json run_date, quoted as recorded: `{obs['run_date']}`",
        f"- prisma_summary.json failed_queries_per_source: {failed} (every value 0: true)",
        f"- prisma_summary.json unique_across_sources: {obs['unique_across_sources']}",
        f"- prisma_records.csv rows: {obs['rows']}; prisma_ids.json union: {obs['union']}; "
        f"record keys == union: {obs['rows_equal_union']}",
        "",
        "Why this run and not an earlier one: completeness alone does not decide it (an earlier run also",
        "recorded zero failed queries); supersession does. The README of the superseded run says:",
        "",
    ]
    if superseded:
        lines.append(f"> {superseded}")
        lines.append(f"> (`appendix-c/{schema.SUPERSEDED_RUN}/README.md`)")
    else:
        lines.append(f"> (no SUPERSEDED line found in `appendix-c/{schema.SUPERSEDED_RUN}/README.md`)")
    lines += [
        "",
        f"Canonical record note: earlier run `{schema.SUPERSEDED_RUN}/` is superseded.",
        "It is flagged to the analyst (PHASE6-ONE-SHOT F1) and not edited (hands-off rule, D3).",
        "",
        "## Title duplicate hints",
        "",
        "`title_dup_group` marks rows whose normalized titles are equal (the v05 rule of prisma_search_v05.py",
        "line 44 without its 25-character cut-off). Hints for the screener only; a duplicate is recorded by",
        "the analyst with the reason code \"duplicate at title level\", never by this script.",
        "",
        f"- title_dup_groups: {groups}",
        f"- title_dup_rows: {dup_rows}",
        "",
        "## Run-folder manifest",
        "",
        f"`appendix-c/screening/{MANIFEST}` lists every file of the {n_folders} `appendix-c/run-*` folders",
        f"({n_files} files) in sha256sum format, relative to `appendix-c/`, byte-sorted. Paths with a",
        "`__pycache__` component are excluded (a Python cache is not part of the record and one already",
        "exists in a run folder; 06-RESEARCH Pitfall 3). `appendix-c/_superseded/` is not a run-* folder and",
        "is not listed. appendix-c/ is not tracked by git, so this committed manifest is the proof that the",
        "run folders stay unchanged. Check it with no project code:",
        "",
        f"    cd appendix-c && sha256sum -c --quiet screening/{MANIFEST}",
        "",
        "init writes the manifest only when it is absent; an existing manifest is verified, never rewritten.",
        "",
        "## Columns",
        "",
        "| # | Column | Meaning | Filled by | Stage |",
        "|---|---|---|---|---|",
    ]
    for i, col in enumerate(schema.WORKBOOK_COLUMNS, start=1):
        meaning, who, stage = schema.COLUMN_DOCS[col]
        lines.append(f"| {i} | `{col}` | {meaning} | {who} | {stage} |")
    lines += [
        "",
        f"Columns 1-{len(schema.ORIGINAL_COLUMNS)} are the source columns in source order, copied field for "
        "field; the copy check compares them as strings (the source is CRLF, the workbook LF).",
        "Reason codes (controlled vocabulary): " + "; ".join(schema.REASON_CODES) + ".",
        "",
    ]
    return "\n".join(lines)


def _build_rows(records: list[dict[str, str]]) -> tuple[list[list[str]], dict[str, str]]:
    groups = title_dup_groups(records)
    rows = []
    for r in records:
        added = {c: "" for c in schema.ADDED_COLUMNS}
        added["doi_url"] = doi_url(r["key"])
        added["title_dup_group"] = groups.get(r["key"], "")
        rows.append([r[c] for c in schema.ORIGINAL_COLUMNS] + [added[c] for c in schema.ADDED_COLUMNS])
    return rows, groups


def _write_if_changed(path: Path, text: str, appendix: Path, encoding: str = "utf-8") -> bool:
    if path.is_file() and path.read_bytes() == text.encode(encoding):
        return False
    write_text_atomic(path, text, appendix, encoding=encoding)
    return True


def init_workbook(appendix: Path, run_name: str = schema.CANONICAL_RUN,
                  expected_unique: int = schema.CANONICAL_UNIQUE, force: bool = False) -> InitResult:
    appendix = Path(appendix)
    obs = check_canonical(appendix / run_name, expected_unique)
    scr = screening_dir(appendix)
    wb, md, mf = scr / WORKBOOK, scr / INIT_MD, scr / MANIFEST
    with workbook_lock(appendix, shared=False):
        rows, groups = _build_rows(obs["records"])
        wb_text = csv_text(schema.WORKBOOK_COLUMNS, rows)
        lines = manifest_lines(appendix)
        # Refusals first, so a refused init writes nothing at all.
        wb_unchanged = wb.is_file() and wb.read_bytes() == wb_text.encode("utf-8")
        if wb.is_file() and not wb_unchanged:
            _, existing = read_csv(wb)
            held = [(r.get("key", "?"), c) for r in existing for c in schema.DECISION_COLUMNS if (r.get(c) or "").strip()]
            if held:
                k, c = held[0]
                raise WorkbookRefused(
                    f"workbook holds recorded decisions; init never overwrites ({len(held)} non-empty decision "
                    f"cell(s), first: key {k!r} column {c})")
            if not force:
                raise WorkbookRefused(f"{_rel(wb, appendix)} exists and differs; pass --force to rebuild it "
                                      "(refused anyway once any decision cell is filled)")
        if mf.is_file():
            diffs = manifest_diff(appendix, lines)
            if diffs:
                raise RunFolderChanged("run folders differ from the committed manifest; init never rewrites it: "
                                       + ", ".join(diffs))
            manifest_action = "verified"
        else:
            manifest_action = "written"
        dup_rows = len(groups)
        md_text = init_md(obs, _superseded_line(appendix), len(set(groups.values())), dup_rows,
                          len(lines), _folders(lines))
        if not wb_unchanged:
            write_text_atomic(wb, wb_text, appendix)
        _write_if_changed(md, md_text, appendix)
        if manifest_action == "written":
            write_text_atomic(mf, "".join(line + "\n" for line in lines), appendix, encoding="ascii")
    return InitResult(
        rows=len(rows), columns=len(schema.WORKBOOK_COLUMNS), run=run_name,
        title_dup_groups=len(set(groups.values())), title_dup_rows=dup_rows,
        manifest_files=len(lines), manifest_folders=_folders(lines), manifest_action=manifest_action,
        workbook_action="unchanged" if wb_unchanged else "written",
        paths={"workbook": wb, "init": md, "manifest": mf},
    )


# ------------------------------------------------------------------ verify


# Source columns the analyst fills (empty templates in prisma_records.csv); see check_copy.
SOURCE_DECISION_COLUMNS: tuple[str, ...] = tuple(c for c in schema.ORIGINAL_COLUMNS if c in schema.DECISION_COLUMNS)


def _field_equal(col: str, src: str, wb: str | None) -> bool:
    return wb == src or (col in SOURCE_DECISION_COLUMNS and src == "" and wb is not None)


@dataclass
class CopyResult:
    rows: int
    columns: int
    run: str
    differs: list[str]
    filled: int = 0  # source-empty decision cells the analyst has filled since the copy

    @property
    def ok(self) -> bool:
        return not self.differs


def check_copy(appendix: Path, run_name: str = schema.CANONICAL_RUN) -> CopyResult:
    """Header, row order, keys and the 11 original fields of every row, compared as strings (_field_equal)."""
    appendix = Path(appendix)
    _, src = read_csv(appendix / run_name / "prisma_records.csv")
    header, wb = read_csv(screening_dir(appendix) / WORKBOOK)
    differs: list[str] = []
    if tuple(header) != schema.WORKBOOK_COLUMNS:
        differs.append("<header>")
    src_keys = [r["key"] for r in src]
    wb_keys = [r.get("key", "") for r in wb]
    src_set, wb_set = set(src_keys), set(wb_keys)
    seen: set[str] = set()

    def flag(k: str) -> None:
        if k not in seen:
            seen.add(k)
            differs.append(k)

    for k in src_keys:
        if k not in wb_set:
            flag(k)  # removed from the workbook
    counts = Counter(wb_keys)
    for k in wb_keys:
        if k not in src_set or counts[k] > 1:
            flag(k)  # added to, or repeated in, the workbook
    filled = 0
    for s, w in zip(src, wb):
        if not all(_field_equal(c, s[c], w.get(c)) for c in schema.ORIGINAL_COLUMNS):
            flag(w.get("key") or s["key"])  # reordered, or a field changed
        filled += sum(1 for c in SOURCE_DECISION_COLUMNS if s[c] == "" and (w.get(c) or ""))
    return CopyResult(rows=len(wb), columns=len(header), run=run_name, differs=differs, filled=filled)


def verify(appendix: Path, run_name: str = schema.CANONICAL_RUN) -> tuple[CopyResult, list[str], list[str]]:
    """(copy result, manifest differences, the run-folder lines hashed now)."""
    appendix = Path(appendix)
    with workbook_lock(appendix, shared=True):
        copy = check_copy(appendix, run_name)
        lines = manifest_lines(appendix)
        diffs = manifest_diff(appendix, lines)
    return copy, diffs, lines


# ------------------------------------------------------------------ CLI


def _short(items: list[str], limit: int = 20) -> str:
    head = ", ".join(items[:limit])
    return head + (f", ... (+{len(items) - limit} more)" if len(items) > limit else "")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma workbook")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("init", help="build appendix-c/screening/workbook.csv from the canonical v05 run")
    p.add_argument("--force", action="store_true", help="rebuild a differing workbook that holds no decision")
    p.add_argument("--appendix-dir", default=None, help="the appendix-c directory (default: beside aac-pipeline/)")
    p = sub.add_parser("verify", help="re-check the copy and the run-folder manifest")
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    scr = screening_dir(appendix)
    try:
        if args.sub == "init":
            # read at call time (not bound as defaults) so the constants stay the single source
            res = init_workbook(appendix, run_name=schema.CANONICAL_RUN,
                                expected_unique=schema.CANONICAL_UNIQUE, force=args.force)
            tail = " (unchanged)" if res.workbook_action == "unchanged" else ""
            print(f"workbook: {_rel(scr / WORKBOOK, appendix)} rows={res.rows} columns={res.columns} "
                  f"source={res.run} title_dup_groups={res.title_dup_groups} title_dup_rows={res.title_dup_rows}{tail}")
            print(f"run-manifest: {_rel(scr / MANIFEST, appendix)} files={res.manifest_files} "
                  f"folders={res.manifest_folders} ({res.manifest_action})")
            return 0
        copy, diffs, lines = verify(appendix, run_name=schema.CANONICAL_RUN)
    except (CanonicalRunError, RunFolderChanged, WorkbookRefused, OutsideScreening, FileNotFoundError) as exc:
        print(f"aac prisma workbook: {exc}", file=sys.stderr)
        return 1
    print(f"workbook: {_rel(scr / WORKBOOK, appendix)} rows={copy.rows} columns={copy.columns} source={copy.run}")
    if copy.ok:
        print(f"copy-check: {copy.rows} rows, {len(schema.ORIGINAL_COLUMNS)} original columns equal")
        if copy.filled:
            print(f"decisions: {copy.filled} source-empty decision cells filled by the analyst "
                  f"({len(SOURCE_DECISION_COLUMNS)} decision columns of the source)")
    else:
        print(f"copy-check: DIFFERS {_short(copy.differs)}")
    if diffs:
        print(f"run-folders: CHANGED {_short(diffs)}")
    else:
        print(f"run-folders: unchanged ({len(lines)} files, {_folders(lines)} folders)")
    return 0 if copy.ok and not diffs else 1


if __name__ == "__main__":
    sys.exit(main())
