"""Copy sealed shards and .sha256 sidecars into evidence-pack/decision-log/ sorted by
(regime, config_hash, seed, shard); decision-log.manifest.json lists them by hash over
the decompressed bytes.

Validation is shard-parallel and serial-equivalent : shards are copied serially in
(regime, config_hash, seed, shard) order; shard_facts (sha256 of the decompressed bytes, record count, last
integer seq) runs in a process pool; serial_starts applies validate_shards' seq recurrence to give every
shard the `expect_seq_from` the one-core pass would have reached; then every shard is validated with that
offset in a second pool whose initializer receives each data block's text set and precomputed shingles
once and loads one validator; Counts are summed in task order. Entries, decision-log.manifest.json and
Counts are byte-for-byte those of the serial algorithm. AAC_PACK_WORKERS sets the pool size (default
min(16, cpu count)); it enters no hash and no output. Any worker exception propagates and aborts the build
before MANIFEST.sha256 is written."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from aac.logging.leakscan import make_shingles
from aac.logging.schema import load_validator
from aac.logging.validate import Counts, validate_shards
from aac.logging.writer import SHARD_DIR, shard_bytes
from aac.run.hashing import dumps_sorted
from aac.run.paths import listdir
from aac.run.paths import runs_dir

PACK_DIR = "decision-log"
WORKERS_ENV = "AAC_PACK_WORKERS"
DEFAULT_MAX_WORKERS = 16


def pack_workers() -> int:
    """Pool size: AAC_PACK_WORKERS (a positive integer) or min(16, cpu count). Never part of any hash or output."""
    env = os.environ.get(WORKERS_ENV)
    if env is None or env == "":
        return max(1, min(DEFAULT_MAX_WORKERS, os.cpu_count() or 1))
    try:
        n = int(env)
    except ValueError:
        raise ValueError(f"{WORKERS_ENV}={env!r} is not an integer") from None
    if n < 1:
        raise ValueError(f"{WORKERS_ENV}={env!r} must be at least 1")
    return n


def data_block_key(resolved: dict) -> str:
    """Canonical JSON of the resolved `data` block: runs with equal keys read the same texts."""
    return json.dumps(resolved.get("data", {}), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def shard_facts(path: Path) -> tuple[str, int, tuple[int, object] | None]:
    """(sha256 of the decompressed bytes, record count, (position, value) of the last record whose seq is an
 integer, or None). The record count and the integer test are exactly those of validate_shards."""
    raw = shard_bytes(path)
    n, last = 0, None
    for line in raw.split(b"\n"):
        if not line:
            continue
        seq = json.loads(line).get("seq")
        if isinstance(seq, int):
            last = (n, seq)
        n += 1
    return hashlib.sha256(raw).hexdigest(), n, last


def serial_starts(facts: list[tuple[str, int, tuple[int, object] | None]], start: int = 0) -> list:
    """The `expect_seq_from` of each shard of ONE run under validate_shards' recurrence
 `expected = (seq if isinstance(seq, int) else expected) + 1`: after a shard of n records whose last integer
 seq v sits at position i, expected = v + 1 + (n - 1 - i); with no integer seq, expected grows by n."""
    out, expected = [], start
    for _, n, last in facts:
        out.append(expected)
        if last is None:
            expected = expected + n
        else:
            i, v = last
            expected = v + 1 + (n - 1 - i)
    return out


# -- process pool ---------------------------------------------------------------------------------

_BLOCKS: dict[str, tuple[set[str], frozenset]] = {}
_VALIDATOR = None


def _init(blocks: dict[str, tuple[set[str], frozenset]]) -> None:
    global _BLOCKS, _VALIDATOR
    _BLOCKS = blocks
    _VALIDATOR = load_validator()


def _validate_task(task: tuple[str, object, str]) -> Counts:
    path, start, block = task
    texts, shingles = _BLOCKS[block]
    return validate_shards([Path(path)], texts, validator=_VALIDATOR, expect_seq_from=start, shingles=shingles)


def _map(fn, items: list, workers: int, initializer=None, initargs=()) -> list:
    """Order-preserving map; in-process for one worker (same functions, no pool)."""
    if not items:
        return []
    if workers <= 1:
        if initializer:
            initializer(*initargs)
        return [fn(x) for x in items]
    chunk = max(1, len(items) // (workers * 8))
    with ProcessPoolExecutor(max_workers=min(workers, len(items)), initializer=initializer, initargs=initargs) as pool:
        return list(pool.map(fn, items, chunksize=chunk))


def copy_sealed(root: Path, runs: list[dict], out_dir: Path, texts_by_block: dict[str, set[str]],
                block_of_run: dict[str, str], workers: int | None = None) -> tuple[list[dict], Counts]:
    workers = pack_workers() if workers is None else workers
    target = Path(out_dir) / PACK_DIR
    target.mkdir(parents=True, exist_ok=True)
    copied: list[tuple[dict, Path]] = []
    for run in sorted(runs, key=lambda r: (r["regime"], r["config_hash"], r["seed"])):
        src = runs_dir(root) / run["run_id"] / SHARD_DIR
        names = sorted(n for n in listdir(src) if n.startswith("part-") and n.endswith(".jsonl.gz"))
        for name in names:
            dst = target / f"{run['run_id']}-{name}"
            shutil.copyfile(src / name, dst)
            shutil.copyfile(src / (name + ".sha256"), dst.with_name(dst.name + ".sha256"))
            copied.append((run, dst))
    facts = _map(shard_facts, [str(dst) for _, dst in copied], workers)
    entries, tasks = [], []
    by_run: dict[str, list[int]] = {}
    for i, ((run, dst), (digest, n, _)) in enumerate(zip(copied, facts)):
        entries.append({"file": f"{PACK_DIR}/{dst.name}", "run_id": run["run_id"], "regime": run["regime"],
                        "seed": run["seed"], "config_hash": run["config_hash"], "sha256": digest, "n": n})
        by_run.setdefault(run["run_id"], []).append(i)
    starts: dict[int, object] = {}
    for idx in by_run.values():  # seq is contiguous per run: offsets restart at 0 for every run
        for i, s in zip(idx, serial_starts([facts[i] for i in idx])):
            starts[i] = s
    for i, (run, dst) in enumerate(copied):
        tasks.append((str(dst), starts[i], block_of_run[run["run_id"]]))
    used = sorted({t[2] for t in tasks})
    blocks = {b: (set(texts_by_block[b]), make_shingles(texts_by_block[b])) for b in used}
    totals = Counts()
    for c in _map(_validate_task, tasks, workers, initializer=_init, initargs=(blocks,)):
        totals = totals.add(c)
    manifest = {"shards": entries, "record_count": sum(e["n"] for e in entries),
                "hash_rule": "sha256 over the decompressed bytes of each shard"}
    with open(Path(out_dir) / "decision-log.manifest.json", "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(manifest))
    return entries, totals
