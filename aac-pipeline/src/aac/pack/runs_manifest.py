"""runs/MANIFEST.sha256 : the frozen list of sealed runs the evidence pack consumes.

One line per run listed in runs/INDEX.json, in plain sha256sum format,
 <sha256 of runs/<id>/SEALED> runs/<id>/SEALED
sorted by path (LC_ALL=C), ASCII, LF, so an examiner can run `sha256sum -c runs/MANIFEST.sha256` from the
namespace root with no project code. SEALED itself holds the digest over every file of the run
(aac.run.seal), so the chain manifest line -> SEALED -> run contents is checkable end to end.

`aac pack build` calls check_runs_manifest before it touches the target: a digest that differs from the SEALED
file, an INDEX run the manifest lacks, a manifest run INDEX lacks, two lines for one run or a malformed line
raise RunsManifestMismatch naming the run. An absent manifest is accepted only in a namespace whose runs all
read the fixture corpus (datasheet profile kind `fixture`: the stub chain and its tests), and says so on stderr.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from aac.run.hashing import sha256_file
from aac.run.index import load_index
from aac.run.paths import runs_dir

RUNS_MANIFEST_REL = "runs/MANIFEST.sha256"
LINE_RE = re.compile(r"^([0-9a-f]{64})  runs/([A-Za-z0-9][A-Za-z0-9._-]*)/SEALED$")
RUN_IN_LINE_RE = re.compile(r"runs/([A-Za-z0-9][A-Za-z0-9._-]*)/")


class RunsManifestMismatch(RuntimeError):
    """runs/MANIFEST.sha256 disagrees with runs/INDEX.json or a SEALED file: the pack refuses."""


def _runs(root: Path, runs_path: Path | None) -> Path:
    return Path(runs_path) if runs_path is not None else runs_dir(root)


def _ids(entries: list[dict]) -> list[str]:
    ids = [e["run_id"] for e in entries]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise RunsManifestMismatch(f"runs/INDEX.json lists run(s) more than once: {', '.join(dup)}")
    return ids


def manifest_lines(root: Path, entries: list[dict], runs_path: Path | None = None) -> list[str]:
    """The manifest lines for `entries`, sorted by path."""
    runs = _runs(root, runs_path)
    out = []
    for rid in sorted(_ids(entries)):
        sealed = runs / rid / "SEALED"
        if not sealed.is_file():
            raise RunsManifestMismatch(f"{rid}: indexed but {sealed.name} is absent; the run is not sealed")
        out.append((f"runs/{rid}/SEALED", f"{sha256_file(sealed)}  runs/{rid}/SEALED"))
    return [line for _, line in sorted(out)]


def write_runs_manifest(root: Path, runs_path: Path | None = None) -> Path:
    """Write <runs>/MANIFEST.sha256 for every run in <runs>/INDEX.json (tmp file + os.replace)."""
    runs = _runs(root, runs_path)
    lines = manifest_lines(root, load_index(runs / "INDEX.json")["runs"], runs)
    target = runs / "MANIFEST.sha256"
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write("".join(line + "\n" for line in lines))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)
    return target


def is_fixture_namespace(root: Path, entries: list[dict], runs_path: Path | None = None) -> bool:
    """True when every run's data manifest has the datasheet profile kind `fixture`."""
    from aac.pack.datasheet import load_profile
    from aac.pack.inputs import PackInputError

    runs = _runs(root, runs_path)
    if not entries:
        return False
    for e in entries:
        try:
            resolved = json.loads((runs / e["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))
            if load_profile(root, resolved["data"]["manifest"])["kind"] != "fixture":
                return False
        except (OSError, ValueError, KeyError, TypeError, PackInputError):
            return False
    return True


def check_runs_manifest(root: Path, entries: list[dict], runs_path: Path | None = None) -> str:
    """Return "ok" or "absent (fixture namespace)"; raise RunsManifestMismatch on any disagreement."""
    runs = _runs(root, runs_path)
    target = runs / "MANIFEST.sha256"
    if not target.is_file():
        if is_fixture_namespace(root, entries, runs):
            print("runs manifest: absent (fixture namespace)", file=sys.stderr)
            return "absent (fixture namespace)"
        raise RunsManifestMismatch(f"{RUNS_MANIFEST_REL} is absent in a namespace with real runs; write it with "
                                   "`aac index manifest` and commit it")
    index_ids = set(_ids(entries))
    problems: list[str] = []
    seen: dict[str, str] = {}
    raw = target.read_bytes()
    if not raw.isascii() or b"\r" in raw or (raw and not raw.endswith(b"\n")):
        problems.append(f"{RUNS_MANIFEST_REL}: not ASCII with LF line ends")
    lines = raw.decode("ascii", "replace").split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    for n, line in enumerate(lines, 1):
        m = LINE_RE.match(line)
        if not m:
            named = RUN_IN_LINE_RE.search(line)
            problems.append(f"line {n} malformed{' (run ' + named.group(1) + ')' if named else ''}: expected "
                            "'<sha256>  runs/<id>/SEALED'")
            continue
        digest, rid = m.groups()
        if rid in seen:
            problems.append(f"{rid}: listed more than once")
            continue
        seen[rid] = digest
    for rid in sorted(index_ids - set(seen)):
        problems.append(f"{rid}: in runs/INDEX.json but not in {RUNS_MANIFEST_REL}")
    for rid in sorted(set(seen) - index_ids):
        problems.append(f"{rid}: in {RUNS_MANIFEST_REL} but not in runs/INDEX.json")
    for rid in sorted(set(seen) & index_ids):
        sealed = runs / rid / "SEALED"
        if not sealed.is_file():
            problems.append(f"{rid}: SEALED absent")
        elif sha256_file(sealed) != seen[rid]:
            problems.append(f"{rid}: SEALED digest differs from {RUNS_MANIFEST_REL}")
    if problems:
        raise RunsManifestMismatch("; ".join(problems))
    return "ok"
