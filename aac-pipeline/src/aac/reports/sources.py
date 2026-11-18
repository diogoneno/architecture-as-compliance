"""Shared readers for the report generators ( provenance).

Every generator reads sealed runs through runs/INDEX.json, verifies the seal, and carries the
run's config_hash, data_version, run_id and shard hashes into each long row. A generator that
needs a run which is not in the index raises MissingRun unless a FIRED configs/cuts.yaml step
blanks that cell, in which case the caller renders fmt.NOT_RUN(step). Nothing here recomputes
a metric: the metric and statistics modules do that on the records these helpers return."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aac.data.evidence_providers import load_items
from aac.data.fixtures import LABELS, load_labels, load_split
from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.reports import fmt
from aac.run.hashing import derive_seed, sha256_file
from aac.run.index import load_index
from aac.run.seal import verify_sealed
from aac.stats.bootstrap import ci_index
from aac.run.paths import runs_dir

POSITIVE_LABEL = "Refuted"  # 


class MissingRun(RuntimeError):
    """A required sealed run is absent from runs/INDEX.json and no fired cut covers the cell."""


@dataclass(frozen=True)
class Run:
    entry: dict
    run_dir: Path
    resolved: dict
    records: tuple
    shard_hashes: str

    @property
    def run_id(self) -> str:
        return self.entry["run_id"]

    @property
    def seed(self) -> int:
        return int(self.entry["seed"])

    @property
    def config_hash(self) -> str:
        return self.entry["config_hash"]

    @property
    def data_version(self) -> str:
        return self.resolved["data_version"]

    def split(self, name: str) -> list[dict]:
        recs = [r for r in self.records if r["split"] == name]
        if not recs:
            raise MissingRun(f"{self.run_id}: no records with split {name!r}")
        return recs


def index_runs(root: Path) -> list[dict]:
    return sorted(load_index(runs_dir(root) / "INDEX.json")["runs"], key=lambda r: (r["run_id"]))


def select_runs(root: Path, *, regime: str, name: str, system: str | None = None) -> list[dict]:
    runs = [r for r in index_runs(root)
            if r["regime"] == regime and r.get("name") == name and (system is None or r["system"] == system)]
    runs.sort(key=lambda r: int(r["seed"]))
    seeds = [int(r["seed"]) for r in runs]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"runs/INDEX.json lists {regime}/{name} more than once for a seed: {seeds}")
    return runs


def require_runs(root: Path, *, regime: str, name: str, table: str, row_ident: str, col_key: str,
                 system: str | None = None) -> tuple[str, object]:
    """('runs', [index entries]) or ('cut', step_id) when the cell is blanked by a fired cut.
 Raises MissingRun otherwise: a missing run is never a silent blank."""
    runs = select_runs(root, regime=regime, name=name, system=system)
    if runs:
        return "runs", runs
    step = fmt.active_cut_for(table, row_ident, col_key)
    if step:
        return "cut", step
    raise MissingRun(f"table {table} cell ({row_ident!r}, {col_key!r}) needs sealed {regime}/{name} run(s) "
                     f"in runs/INDEX.json and no fired configs/cuts.yaml step blanks it")


def load_run(root: Path, entry: dict) -> Run:
    run_dir = runs_dir(root) / entry["run_id"]
    sealed = verify_sealed(run_dir)
    if sealed != entry["sealed_sha256"]:
        raise RuntimeError(f"{entry['run_id']}: SEALED {sealed[:12]} disagrees with runs/INDEX.json")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    shards = list_shards(run_dir / SHARD_DIR)
    recs = sorted((r for p in shards for r in read_shard(p)), key=lambda r: r["seq"])
    # shard order (part-0000, part-0001, ...), never hash-value order: a hash-sorted list would reorder whenever
    # the shard bytes move (e.g. a different log clock), although only the hash values themselves should move
    sidecars = [p.with_name(p.name + ".sha256").read_text(encoding="ascii").split()[0] for p in shards]
    return Run(entry=entry, run_dir=run_dir, resolved=resolved, records=tuple(recs), shard_hashes=";".join(sidecars))


@dataclass(frozen=True)
class ExternalRun(Run):
    """A sealed external BASE run : no decision-log shards; result.json (+ items.jsonl when the tool
 has per-item outcomes). shard_hashes = `result:<sha256>[items:<sha256>]` so provenance names the files."""
    result: dict = None  # type: ignore[assignment]
    items: tuple = ()


def load_external(root: Path, entry: dict) -> ExternalRun:
    """Verify the seal of an external run and read its result.json and items.jsonl ."""
    run_dir = runs_dir(root) / entry["run_id"]
    sealed = verify_sealed(run_dir)
    if sealed != entry["sealed_sha256"]:
        raise RuntimeError(f"{entry['run_id']}: SEALED {sealed[:12]} disagrees with runs/INDEX.json")
    if entry.get("system") != "external":
        raise ValueError(f"{entry['run_id']}: system {entry.get('system')!r} is not an external run")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    res_path, items_path = run_dir / "result.json", run_dir / "items.jsonl"
    result = json.loads(res_path.read_text(encoding="utf-8"))
    parts = [f"result:{sha256_file(res_path)}"]
    items: tuple = ()
    if items_path.is_file():
        items = tuple(json.loads(ln) for ln in items_path.read_text(encoding="utf-8").splitlines() if ln.strip())
        parts.append(f"items:{sha256_file(items_path)}")
    return ExternalRun(entry=entry, run_dir=run_dir, resolved=resolved, records=(), shard_hashes=";".join(parts),
                       result=result, items=items)


def load_runs(root: Path, entries: list[dict]) -> list[Run]:
    return [load_run(root, e) for e in entries]


def provenance(runs: list[Run]) -> dict:
    """config_hash / data_version / run_ids / source_shards_sha256 for long rows over these runs."""
    if not runs:
        raise ValueError("provenance over zero runs")
    return {
        "config_hash": ";".join(sorted({r.config_hash for r in runs})),
        "data_version": ";".join(sorted({r.data_version for r in runs})),
        "run_ids": ";".join(r.run_id for r in runs),
        "source_shards_sha256": ";".join(r.shard_hashes for r in runs),
    }


def seeds_text(runs: list[Run]) -> str:
    return ", ".join(str(r.seed) for r in runs)


# --------------------------------------------------------------------------- labels

def target_of(claim_id: str) -> str:
    """The InCrediblAE task of an item id: `incrediblae/<task>/attack/<row>` (the real committed item files) or `<task>/<n>` (the fixture item files). : the real ids carry the `incrediblae/` prefix,
 so the first segment alone named no task."""
    parts = str(claim_id).split("/")
    return parts[1] if parts[0] == "incrediblae" and len(parts) > 1 else parts[0]


def positives_for(root: Path, run: Run, records: list[dict]) -> np.ndarray:
    """Binary truth (Refuted positive): AVeriTeC fixture labels for fx/ ids, InCrediblAE label 1 for fc/rd."""
    data = run.resolved["data"]
    labels = load_labels(Path(root) / data["labels"])
    cache: dict[str, dict] = {}
    out = []
    for r in records:
        cid = r["claim_id"]
        if cid in labels:
            out.append(labels[cid] == POSITIVE_LABEL)
            continue
        target = target_of(cid)
        path = data.get(f"{target}_items")
        if not path:
            raise MissingRun(f"{run.run_id}: no label source for item {cid!r}")
        if target not in cache:
            cache[target] = load_items(Path(root) / path)
        out.append(int(cache[target][cid]["label"]) == 1)
    return np.asarray(out, dtype=bool)


def label_index_for(root: Path, run: Run, records: list[dict]) -> np.ndarray:
    labels = load_labels(Path(root) / run.resolved["data"]["labels"])
    return np.asarray([LABELS.index(labels[r["claim_id"]]) for r in records], dtype=int)


def aligned(runs: list[Run], split: str) -> tuple[list[str], list[list[dict]]]:
    """Records of `split` per run, aligned on claim_id (same items for every seed), sorted by claim_id."""
    per_run = []
    for run in runs:
        recs = {r["claim_id"]: r for r in run.split(split)}
        per_run.append(recs)
    ids = sorted(set.intersection(*(set(d) for d in per_run)))
    if not ids:
        raise MissingRun(f"no common items across seeds for split {split!r}")
    if any(len(d) != len(ids) for d in per_run):
        raise RuntimeError(f"split {split!r}: seeds evaluated different item sets")
    return ids, [[d[i] for i in ids] for d in per_run]


def arr(records: list[dict], key: str) -> np.ndarray:
    return np.asarray([r[key] for r in records], dtype=float)


def auto_mask(records: list[dict]) -> np.ndarray:
    return np.asarray([not r["abstention_flag"] for r in records], dtype=bool)


# --------------------------------------------------------------------------- bootstrap glue

def seeds_as_replicates_ci(stat_per_seed, n_items: int, *, cell_key: str, n_resamples: int, level: float = 0.95,
                           draw=None):
    """Item bootstrap with seeds as replicates: one index resample is applied to every seed and the
 seed-mean of stat is collected (aac.stats.bootstrap.ci_index, RNG = derive_seed(cell_key)).
 `draw(rng)` overrides the uniform resample (e.g. `lambda rng: stratified(rng, y)`)."""
    return ci_index(lambda idx: float(np.mean([f(idx) for f in stat_per_seed])), n_items,
                    n_resamples=n_resamples, seed=derive_seed(cell_key), level=level, draw=draw)


def stratified(rng: np.random.Generator, y: np.ndarray) -> np.ndarray:
    """A class-stratified resample with replacement (same class counts as y), so AUROC is defined
 on every resample: each stratum gets its own uniform integers from the cell-keyed generator,
 so every member of a stratum is equally likely (no modulo mapping)."""
    y = np.asarray(y, dtype=bool)
    pos = np.flatnonzero(y)
    neg = np.flatnonzero(~y)
    return np.concatenate([pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]])


def split_manifest(root: Path, run: Run) -> dict:
    return load_split(Path(root) / run.resolved["data"]["split_manifest"])


def resamples(root: Path, run: Run) -> int:
    return int(run.resolved["stats"]["bootstrap"])


def level(run: Run) -> float:
    return float(run.resolved["stats"].get("level", 0.95))
