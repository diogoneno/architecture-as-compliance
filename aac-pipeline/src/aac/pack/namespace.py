"""Scratch evidence-pack namespaces : `python -m aac.pack.namespace link|clean`.

A namespace is a directory under <root>/build/ that `AAC_OUT` can point at. `link` gives it
 runs/<id> read-only symlinks to the real sealed run directories (<root>/runs/<id>), never copies
 runs/INDEX.json the real index with `runs` filtered to the selection (every other key kept)
 runs/MANIFEST.sha256 one sha256sum line per linked run (the gate `aac pack build` checks)
 reports/ the real reports/*.csv, reports/*.png and reports/long/*.csv as committed at HEAD
so a pack can be built from real sealed runs without writing anything into the real namespace.
`clean` removes every symlink first (os.unlink on the link itself, never on a path through it),
asserts that none remains, and only then removes the directory, so a cleanup can never delete or
modify a real run through a link (operational constraint).

Selections: `all` = every indexed run; `thin` = the R1 chronological full-pipeline seed-0 run plus the
first BASE run in INDEX order that has no decision-log/ and whose data block names rd_items (a log-less
RD baseline), both chosen by rule. `link` refuses when the real runs/MANIFEST.sha256 exists and disagrees
with the real INDEX, and writes the namespace's own runs/MANIFEST.sha256 (aac.pack.runs_manifest). Imports:
stdlib (and the git binary), aac.run and aac.pack.runs_manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from aac.pack.runs_manifest import RunsManifestMismatch, check_runs_manifest, write_runs_manifest
from aac.run.hashing import dumps_sorted
from aac.run.index import load_index
from aac.run.paths import root as project_root
from aac.run.paths import runs_dir
from aac.run.seal import verify_sealed

SELECTIONS = ("thin", "all")
THIN_SYSTEM = {"regime": "R1", "name": "r1-chrono-full", "seed": 0}
LOG_DIR = "decision-log"


class NamespaceRefused(RuntimeError):
    """A namespace operation outside <root>/build/, onto an existing directory, or with an unknown selection."""


def _real_runs(root: Path) -> Path:
    """The source namespace's runs/ (aac.run.paths.runs_dir, AAC_OUT-aware). `main` clears AAC_OUT, so the CLI
 always links from the real namespace."""
    return runs_dir(root)


def _resolved(root: Path, run_id: str) -> dict:
    return json.loads((_real_runs(root) / run_id / "config.resolved.json").read_text(encoding="utf-8"))


def select_runs(root: Path, entries: list[dict], selection: str) -> list[dict]:
    """The INDEX entries of a selection, in INDEX order."""
    if selection == "all":
        return list(entries)
    if selection != "thin":
        raise NamespaceRefused(f"unknown run selection {selection!r} (known: {', '.join(SELECTIONS)})")
    hits = [e for e in entries if e["regime"] == THIN_SYSTEM["regime"] and e.get("name") == THIN_SYSTEM["name"]
            and int(e["seed"]) == THIN_SYSTEM["seed"]]
    if len(hits) != 1:
        raise NamespaceRefused(f"runs/INDEX.json lists {len(hits)} R1/r1-chrono-full seed 0 runs (need exactly 1)")
    system = hits[0]
    if verify_sealed(_real_runs(root) / system["run_id"]) != system["sealed_sha256"]:
        raise NamespaceRefused(f"{system['run_id']}: SEALED disagrees with runs/INDEX.json")
    baseline = None
    for e in entries:
        if e["regime"] != "BASE" or (_real_runs(root) / e["run_id"] / LOG_DIR).is_dir():
            continue
        if _resolved(root, e["run_id"]).get("data", {}).get("rd_items"):
            baseline = e
            break
    if baseline is None:
        raise NamespaceRefused("runs/INDEX.json lists no BASE run without decision-log/ whose data block names rd_items")
    chosen = {system["run_id"], baseline["run_id"]}
    return [e for e in entries if e["run_id"] in chosen]


def _under_build(root: Path, dest: Path) -> Path:
    """dest made absolute (relative paths are taken from root) and refused unless strictly under <root>/build/.
 The parent is resolved; the last component is kept as given so a symlinked dest is never followed."""
    root = Path(root).resolve()
    dest = Path(dest) if Path(dest).is_absolute() else root / dest
    dest = dest.parent.resolve() / dest.name
    build = root / "build"
    if dest.name in ("", ".", "..") or not dest.is_relative_to(build) or dest == build:
        raise NamespaceRefused(f"refusing namespace {dest}: it must be a directory under {build}")
    return dest


def link_namespace(root: Path, dest: Path, selection: str) -> Path:
    root = Path(root).resolve()
    dest = _under_build(root, dest)
    if dest.exists() or dest.is_symlink():
        raise NamespaceRefused(f"refusing namespace {dest}: it exists (run `clean` first)")
    index = load_index(_real_runs(root) / "INDEX.json")
    chosen = select_runs(root, index["runs"], selection)
    if not chosen:
        raise NamespaceRefused("runs/INDEX.json lists no sealed runs")
    if (_real_runs(root) / "MANIFEST.sha256").is_file():  # the committed run list must hold before anything is linked
        try:
            check_runs_manifest(root, index["runs"], _real_runs(root))
        except RunsManifestMismatch as exc:
            raise NamespaceRefused(f"real runs/MANIFEST.sha256: {exc}") from exc
    (dest / "runs").mkdir(parents=True)
    for e in chosen:
        target = (_real_runs(root) / e["run_id"]).resolve()
        if not (target / "SEALED").is_file():
            raise NamespaceRefused(f"{e['run_id']}: indexed but no sealed run directory at {target}")
        os.symlink(target, dest / "runs" / e["run_id"], target_is_directory=True)
    filtered = {**index, "runs": chosen}
    with open(dest / "runs" / "INDEX.json", "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(filtered))
    write_runs_manifest(root, dest / "runs")  # the namespace's own runs/MANIFEST.sha256 (Task 2)
    (dest / "reports" / "long").mkdir(parents=True)
    for rel in committed_reports(root):
        (dest / rel).write_bytes(_git(root, "show", f"HEAD:./{rel}"))
    return dest


def _git(root: Path, *args: str) -> bytes:
    proc = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    if proc.returncode != 0:
        raise NamespaceRefused(f"git {' '.join(args)} failed: {proc.stderr.decode('utf-8', 'replace').strip()[:200]}")
    return proc.stdout


def committed_reports(root: Path) -> list[str]:
    """reports/*.csv, reports/*.png and reports/long/*.csv as committed at HEAD (never the working tree, which a
 concurrent plan may be rewriting); no reports/INDEX.json, no sub-folders such as ch4-feed-*."""
    names = _git(root, "ls-tree", "-r", "-z", "--name-only", "HEAD", "--", "reports").decode("utf-8").split("\0")
    out = []
    for name in names:
        rel = name.split("/", 1)[1] if name.startswith("aac-pipeline/") else name  # ls-tree paths are repo-relative
        parts = rel.split("/")
        if len(parts) == 2 and parts[0] == "reports" and rel.endswith((".csv", ".png")):
            out.append(rel)
        elif len(parts) == 3 and parts[:2] == ["reports", "long"] and rel.endswith(".csv"):
            out.append(rel)
    return sorted(out)


def _symlinks(dest: Path) -> list[Path]:
    out = []
    for dirpath, dirnames, filenames in os.walk(dest, followlinks=False):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                out.append(p)
    return sorted(out)


def clean_namespace(root: Path, dest: Path) -> int:
    """Unlink every symlink under dest (the link, never its target), assert none remains, then rmtree dest.
 Returns the number of links removed."""
    dest = _under_build(root, dest)
    if dest.is_symlink():
        raise NamespaceRefused(f"refusing to clean {dest}: the namespace itself is a symlink")
    if not dest.exists():
        return 0
    links = _symlinks(dest)
    for link in links:
        os.unlink(str(link).rstrip("/"))
    left = _symlinks(dest)
    if left:
        raise NamespaceRefused(f"refusing to remove {dest}: symlinks remain after unlinking ({left[:3]})")
    shutil.rmtree(dest)
    return len(links)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.pack.namespace")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("link")
    p.add_argument("--out", required=True)
    p.add_argument("--runs", choices=SELECTIONS, required=True)
    p = sub.add_parser("clean")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    os.environ.pop("AAC_OUT", None)  # the source is always the real namespace; a namespace is never linked from another
    root = project_root()
    try:
        if args.cmd == "link":
            dest = link_namespace(root, Path(args.out), args.runs)
            n = sum(1 for p in (dest / "runs").iterdir() if p.is_symlink())
            print(f"namespace: {dest.relative_to(root)} runs={args.runs} linked={n}")
        else:
            n = clean_namespace(root, Path(args.out))
            print(f"namespace cleaned: {args.out} (links removed={n})")
    except NamespaceRefused as exc:
        print(f"aac.pack.namespace: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
