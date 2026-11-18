"""runs/INDEX.json: the committed index of sealed runs.

Every write is a MERGE: load the existing file, change only the top-level key the
caller owns (`runs` for the sealed-run writers, `day1_docs` for `aac index docs`), keep
every other top-level key byte-for-byte, and write through tmp + os.replace so a
concurrent reader never sees a torn file. Every read-modify-write holds an exclusive
fcntl.flock on the sibling `<INDEX>.lock` for its whole load-modify-write , so two concurrent writers (parallel plans, a gpu-pull) never lose an entry.

`day1_docs` is a REFERENCE, not the record: the record of truth is
specification's signed `Approved:` line plus the annotated git tag `v1.0`. The block names
the tag's commit and tagger date and, as a convenience, the sha256 of the three day-1
documents as stored at the tag (each re-derivable from `git show v1.0:<path>`)."""
from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
from contextlib import contextmanager
from pathlib import Path

from aac.run.hashing import dumps_sorted, sha256_bytes, sha256_file
from aac.run.paths import runs_dir

RUNS_KEY = "runs"
DAY1_DOCS_KEY = "day1_docs"
PREREG_TAG = "v1.0"
DAY1_DOC_FILES = ("README.md",)
APPEND_ONLY_DOC_FILES = frozenset({"text-contradictions.md"})  # appended by later phases; the tag blob must stay a prefix
_HEX40 = re.compile(r"^[0-9a-f]{40}$")


def load_index(path: Path) -> dict:
    path = Path(path)
    if not path.is_file():
        return {RUNS_KEY: []}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    data.setdefault(RUNS_KEY, [])
    return data


def _write_atomic(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(data))
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


@contextmanager
def _locked(path: Path):
    """Exclusive flock on `<INDEX>.lock` (opened for append, never truncated), held for the whole
 load-modify-write and released in finally. Not re-entrant: callers inside the lock use the
 `_unlocked` helpers."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_name(path.name + ".lock"), "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _replace_runs_unlocked(path: Path, runs: list[dict]) -> None:
    data = load_index(path)
    data[RUNS_KEY] = sorted(runs, key=lambda r: r["run_id"])
    _write_atomic(path, data)


def replace_runs(path: Path, runs: list[dict]) -> None:
    with _locked(path):
        _replace_runs_unlocked(path, runs)


def filter_runs(path: Path, keep) -> list[dict]:
    """Drop every run entry for which keep(entry) is false; read, filter and write under ONE flock ( of the
 Phase 2 security audit: a list read outside the lock would drop an entry a concurrent writer appended in
 between). Returns the dropped entries."""
    with _locked(path):
        runs = load_index(path)[RUNS_KEY]
        kept = [r for r in runs if keep(r)]
        dropped = [r for r in runs if not keep(r)]
        if dropped:
            _replace_runs_unlocked(path, kept)
    return dropped


def append_index(path: Path, entry: dict) -> None:
    if "run_id" not in entry:
        raise ValueError("index entry needs a run_id")
    with _locked(path):
        data = load_index(path)
        runs = [r for r in data[RUNS_KEY] if r.get("run_id") != entry["run_id"]]
        runs.append(entry)
        _replace_runs_unlocked(path, runs)


def entry_from_run_dir(run_dir: Path) -> dict:
    """The INDEX entry of a sealed run, read from its run-record.json and SEALED (seal verified)."""
    from aac.run.seal import verify_sealed

    run_dir = Path(run_dir)
    sealed = verify_sealed(run_dir)
    rr = json.loads((run_dir / "run-record.json").read_text(encoding="utf-8"))
    if rr["run_id"] != run_dir.name:
        raise RuntimeError(f"{run_dir}: run-record run_id {rr['run_id']} differs from the directory name")
    return {"run_id": rr["run_id"], "regime": rr["regime"], "system": rr["system"], "name": rr["name"],
            "seed": int(rr["seed"]), "config_hash": rr["config_hash"], "sealed_sha256": sealed,
            "record_count": int(rr["record_count"])}


def sealed_run_dirs(runs_dir: Path) -> list[Path]:
    """Every runs/<run_id>/ carrying a SEALED manifest (service directories `_*` excluded), sorted."""
    runs_dir = Path(runs_dir)
    if not runs_dir.is_dir():
        return []
    return sorted(d for d in runs_dir.iterdir() if d.is_dir() and not d.name.startswith("_") and (d / "SEALED").is_file())


def rebuild_runs(runs_dir: Path) -> list[dict]:
    """`aac index rebuild`: replace ONLY the `runs` list of runs/INDEX.json with the sealed runs on
 disk; every other top-level key ('s day1_docs) is preserved.

 : the scan of runs/ is the "load" of this load-modify-write, so it runs under the same flock as the
 write. A scan outside the lock lost the entry of a run that a concurrent seal or gpu-pull indexed between the
 scan and the write (its directory not yet, or only just, on disk when scanned)."""
    path = Path(runs_dir) / "INDEX.json"
    with _locked(path):
        entries = [entry_from_run_dir(d) for d in sealed_run_dirs(runs_dir)]
        _replace_runs_unlocked(path, entries)
    return entries


def find_run(path: Path, run_id: str) -> dict | None:
    for r in load_index(path)[RUNS_KEY]:
        if r.get("run_id") == run_id:
            return r
    return None


class StaleRun(RuntimeError):
    """ an indexed sealed run must not be reused (an upstream run it depends on was superseded, or
 the fitted files it copied differ from its source run's current files)."""


FITTED_COPIES = ("calibration.json", "gate.json", "fusion.json", "evidence-floor.json")


def current_sealed_run(runs_dir: Path, regime: str, name: str, seed: int) -> str | None:
    """The reuse check of mk/r2.mk (r2-runs, ev-rd): the one indexed sealed run of (regime, name, seed), or None when
 there is none. Raises StaleRun when more than one is indexed, when a depends_on run id is no longer indexed (its
 configuration changed and it was superseded), or when a fitted copy (`source_run_id` set) differs from the current
 file of that source run (the source was re-sealed under the same run id). Read-only."""
    runs_dir = Path(runs_dir)
    runs = load_index(runs_dir / "INDEX.json")[RUNS_KEY]
    hits = [r for r in runs if r["regime"] == regime and r.get("name") == name and int(r["seed"]) == int(seed)]
    if len(hits) > 1:
        raise StaleRun(f"more than one indexed run for {regime} {name} {seed}")
    if not hits or not (runs_dir / hits[0]["run_id"] / "SEALED").is_file():
        return None
    rid = hits[0]["run_id"]
    run_dir = runs_dir / rid
    indexed = {r["run_id"] for r in runs}
    rec = json.loads((run_dir / "run-record.json").read_text(encoding="utf-8"))
    gone = [x for x in rec.get("depends_on") or [] if x not in indexed]
    if gone:
        raise StaleRun(f"{rid}: depends_on {', '.join(gone)} is no longer the current indexed run (superseded); "
                       f"re-run {name} seed {seed} with --force")
    for fname in FITTED_COPIES:
        if not (run_dir / fname).is_file():
            continue
        copied = json.loads((run_dir / fname).read_text(encoding="utf-8"))
        src = copied.pop("source_run_id", None)
        if src is None:
            continue
        current = runs_dir / src / fname
        if not current.is_file() or json.loads(current.read_text(encoding="utf-8")) != copied:
            raise StaleRun(f"{rid}: {fname} differs from the current {src}/{fname} (source re-sealed); "
                           f"re-run {name} seed {seed} with --force")
    return rid


# ------------------------------------------------------------------------------- day1_docs 


def _git(repo: Path, *args: str, binary: bool = False):
    """`git -C <repo> <args>`; stdout (str, stripped) or bytes when binary. Raises CalledProcessError."""
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=True)
    return proc.stdout if binary else proc.stdout.decode("utf-8").strip()


def _read_bytes(path: Path) -> bytes:
    return Path(path).read_bytes()


def day1_docs_from_tag(root: Path, tag: str = PREREG_TAG) -> dict:
    """The `day1_docs` block computed from git at `tag` (annotated tag required):
 {prereg_v1_commit, prereg_v1_tag_date, files: {<name>: sha256 of `git show <tag>:<rel>/<name>`}}.

 Refuses (RuntimeError) when the tag is missing or lightweight, or when a working-tree copy of
 specification or execution schedule differs from the blob at the tag: those documents are frozen by the tag
 and the index must never describe a drifted tree as the pre-registration.
 text-contradictions.md is append-only by its own ownership rule (later phases append items, never
 edit, renumber or delete one), so for it the check is that the tag blob is a byte prefix of the
 working-tree file ; the recorded hash is still that of the blob at the tag."""
    root = Path(root).resolve()
    repo = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    rel = root.relative_to(repo).as_posix()
    prefix = "" if rel == "." else rel + "/"
    try:
        kind = _git(repo, "cat-file", "-t", tag)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"tag {tag} does not exist: {exc.stderr.decode('utf-8', 'replace').strip()}") from exc
    if kind != "tag":
        raise RuntimeError(f"{tag} is a {kind}, not an annotated tag (D-02 requires `git tag -a {tag}`)")
    commit = _git(repo, "rev-parse", f"{tag}^{{commit}}")
    if not _HEX40.match(commit):
        raise RuntimeError(f"{tag}^{{commit}} did not resolve to a 40-hex sha: {commit!r}")
    tag_date = _git(repo, "for-each-ref", "--format=%(taggerdate:iso-strict)", f"refs/tags/{tag}")
    if not tag_date:
        raise RuntimeError(f"{tag} carries no tagger date")
    files = {}
    for name in DAY1_DOC_FILES:
        blob = _git(repo, "show", f"{tag}:{prefix}{name}", binary=True)
        at_tag = sha256_bytes(blob)
        if name in APPEND_ONLY_DOC_FILES:
            if not _read_bytes(root / name).startswith(blob):
                raise RuntimeError(
                    f"{prefix}{name} in the working tree is not an append-only extension of `git show {tag}:{prefix}{name}` "
                    f"({at_tag[:12]}): an existing byte was edited, renumbered or deleted; refusing to record")
        else:
            in_tree = sha256_file(root / name)
            if at_tag != in_tree:
                raise RuntimeError(
                    f"{prefix}{name} in the working tree ({in_tree[:12]}) differs from `git show {tag}:{prefix}{name}` "
                    f"({at_tag[:12]}); the day-1 documents are frozen by the tag (D-03), refusing to record")
        files[name] = at_tag  # always the blob AT THE TAG: appending items never changes the recorded block
    return {"prereg_v1_commit": commit, "prereg_v1_tag_date": tag_date, "files": files}


def replace_day1_docs(path: Path, block: dict) -> bool:
    """Merge-write ONLY the `day1_docs` key (the `runs` list is left byte-identical).
 Returns True when the file changed, False when the block was already recorded verbatim."""
    with _locked(path):
        data = load_index(path)
        if data.get(DAY1_DOCS_KEY) == block:
            return False
        data[DAY1_DOCS_KEY] = block
        _write_atomic(path, data)
        return True


def record_day1_docs(root: Path, tag: str = PREREG_TAG) -> tuple[dict, bool]:
    """`aac index docs`: compute the block from git at `tag` and merge it into runs/INDEX.json.
 Idempotent: a second call finds the identical block and writes nothing."""
    block = day1_docs_from_tag(root, tag)
    changed = replace_day1_docs(runs_dir(root) / "INDEX.json", block)
    return block, changed


def copy_day1_docs(root: Path, source_index: Path, tag: str = PREREG_TAG) -> tuple[dict, bool]:
    """`aac index docs --from-index`: copy the day1_docs block recorded in `source_index` into
 runs_dir(root)/INDEX.json (merge-write; `runs` untouched). Refuses (RuntimeError) when the source
 has no block, when the block is malformed, or when its prereg_v1_commit differs from
 `git rev-parse <tag>^{commit}`. It never recomputes from git: the block is carried verbatim."""
    source_index = Path(source_index)
    if not source_index.is_file():
        raise RuntimeError(f"{source_index}: no such index")
    block = load_index(source_index).get(DAY1_DOCS_KEY)
    if not isinstance(block, dict) or not _HEX40.match(str(block.get("prereg_v1_commit", ""))):
        raise RuntimeError(f"{source_index}: no well-formed {DAY1_DOCS_KEY} block to copy")
    root = Path(root).resolve()
    repo = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    try:
        want = _git(repo, "rev-parse", f"{tag}^{{commit}}")
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"tag {tag} does not resolve: {exc.stderr.decode('utf-8', 'replace').strip()}") from exc
    if block["prereg_v1_commit"] != want:
        raise RuntimeError(f"{source_index}: {DAY1_DOCS_KEY}.prereg_v1_commit {block['prereg_v1_commit'][:12]} != "
                           f"{tag}^{{commit}} {want[:12]}; refusing to copy a stale reference")
    target = runs_dir(root) / "INDEX.json"
    if target.resolve() == source_index.resolve():
        return block, False
    return block, replace_day1_docs(target, block)
