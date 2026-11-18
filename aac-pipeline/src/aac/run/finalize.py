"""The shared head and tail of every sealing run kind (`aac run`, `aac train`, later kinds).

claim_run_dir: where a run is written, with the supersede/--force semantics of `aac run` (a sealed
run is replaced only with --force, through runs/_staging/, and only after the replacement sealed).
finalize_run: the run-record (validated against contracts/run-record.schema.json), SEALED, the
supersede of the previous run(s) for the same (regime, system, name, seed) key, and the
runs_dir/INDEX.json entry. torch-free (aac.run carries no torch import)."""
from __future__ import annotations

import fcntl
import json
import platform
import shutil
from pathlib import Path

from aac.run import hashing
from aac.run import paths as aac_paths


class RunRefused(Exception):
    """The run cannot start (exit 2): a sealed target without --force, or a run in progress."""


class FinalizeRefused(Exception):
    """The run finished but is not sealed (exit 1)."""


def rel(p: Path, root: Path) -> Path:
    """p relative to the project root when it lies under it (AAC_OUT may point elsewhere), else p."""
    p = Path(p)
    return p.relative_to(root) if p.is_relative_to(root) else p


def move_aside(runs_dir: Path, run_dir: Path, tag: str) -> Path:
    """Move run_dir to runs/_superseded/<name>-<tag>[-N] (first free name). Never deletes."""
    base = runs_dir / "_superseded" / f"{run_dir.name}-{tag}"
    target, n = base, 1
    while target.exists():
        n += 1
        target = base.with_name(f"{base.name}-{n}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(run_dir), str(target))
    return target


def supersede_dir(runs_dir: Path, run_dir: Path, sealed: str) -> None:
    from aac.run.seal import UnsealedRun, read_seal

    target = runs_dir / "_superseded" / f"{run_dir.name}-{sealed[:8]}"
    try:
        identical = target.is_dir() and read_seal(target) == sealed
    except UnsealedRun:
        identical = False
    if identical:
        shutil.rmtree(run_dir)  # a copy with the same full SEALED digest is already preserved
    else:
        move_aside(runs_dir, run_dir, sealed[:8])


def lock_held(run_dir: Path) -> bool:
    """True when another process holds the run's ShardWriter lock (a run in progress)."""
    lock = run_dir / ".lock"
    if not lock.is_file():
        return False
    with open(lock, "a") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fh, fcntl.LOCK_UN)
    return False


def claim_run_dir(root: Path, rid: str, key: tuple, force: bool, *,
                  resumable: bool = False, resume: bool = False) -> tuple[Path, Path, set[str]]:
    """(run_dir, target_dir, stale_ids). A changed configuration (new config_hash, hence new run_id)
 for the same (regime, system, name, seed) key supersedes the previous sealed run: the old run
 directory is preserved under runs/_superseded/ and dropped from runs/INDEX.json, so the index
 lists exactly one current run per configuration key. this happens only AFTER the new run
 has sealed (finalize_run); a failed run leaves the previous one indexed.

 An existing UNSEALED run directory (a crashed or killed attempt; of the Phase 2 review) is refused
 while its writer lock is held. Otherwise it is moved to runs/_superseded/<rid>-unsealed[-N] (never deleted),
 so no leftover (a 1.1 GB checkpoint.safetensors.part, a partial bench array) enters the new run, with or
 without --force. Exception: `resumable=True` (`aac run`, whose ShardWriter resumes from verified shards) reuses it without --force, and seal refuses any stray *.tmp / *.part / *.tmp.npy file.

 `resume=True` (the victim server's `--resume`, passed by attack_session.sh on its bounded retry; of the
 Phase 3 review): an existing unlocked directory is never moved aside but resumed, with or without --force: the
 staging directory runs/_staging/<rid> of a --force start on a SEALED run, or the unsealed target itself."""
    from aac.run.index import load_index

    runs_dir = aac_paths.runs_dir(root)
    target_dir = runs_dir / rid
    run_dir = target_dir  # where this run is written; a staging directory when it replaces a sealed run
    stale_ids = {r["run_id"] for r in load_index(runs_dir / "INDEX.json")["runs"]
                 if (r["regime"], r["system"], r.get("name"), r["seed"]) == key and r["run_id"] != rid}
    if (target_dir / "SEALED").is_file():
        if not force:
            raise RunRefused(f"refusing: {rel(target_dir, root)} is SEALED (use --force to supersede)")
        run_dir = runs_dir / "_staging" / rid
        if run_dir.exists():
            if lock_held(run_dir):
                raise RunRefused(f"refusing: {rel(run_dir, root)} is locked by a run in progress")
            if not resume:
                move_aside(runs_dir, run_dir, "staging")
    elif run_dir.exists():
        if lock_held(run_dir):
            raise RunRefused(f"refusing: {rel(run_dir, root)} is locked by a run in progress")
        if (force and not resume) or not (resumable or resume):
            moved_to = move_aside(runs_dir, run_dir, "unsealed")
            print(f"superseded: unsealed {rid} moved to runs/_superseded/{moved_to.name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, target_dir, stale_ids


def gpu_hours(stages: list[dict]) -> float:
    """Sum of the sealed stage seconds whose device starts with `cuda`, in hours (a CPU run, or any run
 under the FixedClock whose sealed seconds are 0.0, records 0.0). Table D.2 applies the same rule."""
    return round(sum(float(s["seconds"]) for s in stages if str(s["device"]).startswith("cuda")) / 3600.0, 6)


def finalize_run(root: Path, run_dir: Path, ident: dict, *, summary: dict, record_count: int, shards: list,
                 clock, started: str, finished: str, system: str, determinism_level: str,
                 target_dir: Path | None = None, stale_ids: set[str] | frozenset = frozenset(),
                 violations: int = 0) -> str:
    """Write run-record.json, seal, supersede and index; return the SEALED digest.

 Raises FinalizeRefused (nothing sealed, the previous run stays indexed) when the record violates
 contracts/run-record.schema.json or `violations` > 0 (the record is written first, as `aac run`
 always did, so the refusal can be inspected)."""
    from jsonschema import Draft202012Validator

    from aac.run.index import append_index, filter_runs, load_index
    from aac.run.seal import read_seal, seal

    resolved, rid = ident["resolved"], ident["run_id"]
    runs_dir = aac_paths.runs_dir(root)
    target_dir = Path(target_dir) if target_dir is not None else runs_dir / rid
    seed = int(resolved["seed"])
    record = {
        "run_id": rid, "regime": resolved["regime"], "system": system, "name": resolved["name"],
        "config_hash": ident["config_hash"], "data_version": ident["data_version"], "model_version": ident["model_version"],
        "git_sha": ident["git_sha"], "seed": seed, "configurations": summary["configurations"],
        "items": summary["items"], "stages": summary["stages"], "gpu_hours": gpu_hours(summary["stages"]),
        "depends_on": list(summary.get("depends_on") or []),
        "determinism_level": determinism_level,
        "host": platform.node(), "started_utc": started, "finished_utc": finished,
        "record_count": int(record_count),
        "shards": [{"file": s.file, "sha256": s.sha256, "n": s.n} for s in shards],
        "run_epoch": int(getattr(clock, "epoch_s")), "clock_source": clock.source,
    }
    rr_schema = json.loads((root / "contracts" / "run-record.schema.json").read_text(encoding="utf-8"))
    errs = sorted(e.message for e in Draft202012Validator(rr_schema).iter_errors(record))
    if errs:
        raise FinalizeRefused(f"run-record.json violates contracts/run-record.schema.json: {errs[0]}")
    (run_dir / "run-record.json").write_text(hashing.dumps_sorted(record), encoding="ascii")
    if violations:
        raise FinalizeRefused(f"refusing to seal {rid}: violations={violations}")
    digest = seal(run_dir)
    
    if Path(run_dir) != target_dir:
        old = read_seal(target_dir)
        supersede_dir(runs_dir, target_dir, old)
        shutil.move(str(run_dir), str(target_dir))
        print(f"superseded: previous run moved to runs/_superseded/{rid}-{old[:8]}")
    if stale_ids:
        index_runs = load_index(runs_dir / "INDEX.json")["runs"]
        stale = [r for r in index_runs if r["run_id"] in stale_ids]
        for r in stale:
            old_dir = runs_dir / r["run_id"]
            if old_dir.is_dir():
                supersede_dir(runs_dir, old_dir, r["sealed_sha256"])
            print(f"superseded: {r['run_id']} (configuration changed) moved to runs/_superseded/")
        
        # read above (a concurrent gpu-pull or parallel run may have appended in between)
        filter_runs(runs_dir / "INDEX.json", lambda r: r["run_id"] not in stale_ids)
    append_index(runs_dir / "INDEX.json", {
        "run_id": rid, "regime": resolved["regime"], "system": system, "name": resolved["name"],
        "seed": seed, "config_hash": ident["config_hash"], "sealed_sha256": digest, "record_count": int(record_count),
    })
    return digest
