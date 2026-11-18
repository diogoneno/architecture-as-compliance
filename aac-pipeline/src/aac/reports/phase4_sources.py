"""Phase 4 run sources for the report generators (2).

The Phase 4 tables read a DECLARED list of sealed runs (configs/reports/phase4-runs.yaml), never one hard-coded
name. This module selects the profile of the namespace, loads the run lists, and returns the union of the R3
attack ledgers with three checks per run: the seal is recomputed and must equal runs/INDEX.json, invariant L2 holds
strictly (aac.attacks.ledger.l2_check), and no (item_id, attack, victim, subset) key appears in two runs
(DuplicateLedgerKey). Runs are loaded "light": the decision-log shards are hashed for provenance but their records
are read only on request (read_records), because a real victim-B run holds hundreds of thousands of records.

 select_profile(root) tinyworld | stub | real (see the profile file's header)
 load_profile(root, name) the profile's sections (r3, bodega, r4, supplementary); stub -> r3: [r3-stub]
 union_ledgers(...) (rows tagged with `_run_id`, runs) over the indexed runs of the listed names
 subset_half(run) first200 | last200 | full from the run's resolved data.attack_subset
 cut_fired(step_id) the fired_on date of a configs/cuts.yaml step, else None (cut-2 blanks no cell, so
 fmt.active_cut_for cannot see it)
 read_records(run, ids) the decision records with these record_ids (streamed shard by shard)
 read_split(run, split) every record of one split (the victim-source R1 run's clean test records)
 read_json(run, name) a sealed JSON sidecar (calibration.json, gate.json)
 read_bodega_scores(...) the text-free per-row BODEGA scores of the sealed BS runs (interface)

Nothing here computes a metric. No adversarial text is read: ledgers, records (claim_hash only) and score files."""
from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

from aac.attacks.ledger import l2_check, read_ledger
from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.reports import fmt, sources
from aac.run.paths import runs_dir
from aac.run.seal import verify_sealed

PROFILE_FILE = Path("configs") / "reports" / "phase4-runs.yaml"
STUB_PROFILE = "stub"
STUB_R3 = "r3-stub"
FIXTURE_DATASET = "fixture"
CUT_BEAM = "cut-1-beamattack"
CUT_SUBSET = "cut-2-subset-200"
HALVES = ("first200", "last200", "full")
BODEGA_FILE = "bodega-scores.jsonl"
BODEGA_REGIME = "BS"
BODEGA_FIELDS = ("item_id", "attack", "victim", "subset", "source_run_id", "success_strict", "semantic", "character",
                 "bodega")
BODEGA_TOL = 1e-6


class DuplicateLedgerKey(RuntimeError):
    """Two runs of one union hold the same (item_id, attack, victim, subset) ledger key."""


class L2Violation(RuntimeError):
    """A run of the union fails invariant L2 (strict)."""


class BodegaScoreError(RuntimeError):
    """A BODEGA score row does not match the R3 ledger it claims to score."""


def ledger_key(row: dict) -> tuple[str, str, str, str]:
    return (row["item_id"], row["attack"], row["victim"], row["subset"])


# --------------------------------------------------------------------------- profiles

def load_profiles(root: Path) -> dict:
    with open(Path(root) / PROFILE_FILE, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    profiles = (data or {}).get("profiles")
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError(f"{PROFILE_FILE}: expected a top-level `profiles:` mapping")
    return profiles


def _resolved(root: Path, entry: dict) -> dict:
    path = runs_dir(root) / entry["run_id"] / "config.resolved.json"
    return json.loads(path.read_text(encoding="utf-8"))


def select_profile(root: Path) -> str:
    """`tinyworld` when any tinyworld r3 name is indexed; else `stub` when an indexed R3 run reads the fixture corpus
 (the stub smoke chain, registry.REAL_ONLY's predicate); else `real`. The tiny world also reads a fixture-format
 corpus, so its run names are checked first."""
    root = Path(root)
    entries = [e for e in sources.index_runs(root) if e.get("regime") == "R3"]
    names = {e.get("name") for e in entries}
    tiny = load_profiles(root).get("tinyworld", {}).get("r3") or []
    if names & set(tiny):
        return "tinyworld"
    for e in entries:
        if _resolved(root, e).get("data", {}).get("dataset") == FIXTURE_DATASET:
            return STUB_PROFILE
    return "real"


def load_profile(root: Path, name: str | None = None) -> dict:
    """{"name": <profile>, "r3": [...], "bodega": [...], "r4": {...}, "supplementary": {...}} for `name` (default:
 select_profile). The stub profile is not in the file: it reads r3-stub only."""
    root = Path(root)
    name = name or select_profile(root)
    if name == STUB_PROFILE:
        return {"name": STUB_PROFILE, "r3": [STUB_R3], "bodega": [], "r4": {}, "supplementary": {}}
    profiles = load_profiles(root)
    if name not in profiles:
        raise KeyError(f"{PROFILE_FILE} has no profile {name!r}")
    prof = profiles[name]
    return {"name": name, "r3": list(prof.get("r3") or []), "bodega": list(prof.get("bodega") or []),
            "r4": dict(prof.get("r4") or {}), "supplementary": dict(prof.get("supplementary") or {})}


# --------------------------------------------------------------------------- runs

def indexed(root: Path, names, *, regime: str | None = None) -> list[dict]:
    """Index entries of the listed names, in list order then seed; a name that is not indexed contributes nothing."""
    order = {n: i for i, n in enumerate(names)}
    if len(order) != len(list(names)):
        raise ValueError(f"run list names a run twice: {list(names)}")
    out = [e for e in sources.index_runs(root) if e.get("name") in order and (regime is None or e["regime"] == regime)]
    out.sort(key=lambda e: (order[e["name"]], int(e["seed"])))
    return out


def load_run_light(root: Path, entry: dict) -> sources.Run:
    """Verify the seal against runs/INDEX.json and read the resolved config and the shard sidecar hashes; the
 decision records are NOT loaded (records=)."""
    run_dir = runs_dir(root) / entry["run_id"]
    sealed = verify_sealed(run_dir)
    if sealed != entry["sealed_sha256"]:
        raise RuntimeError(f"{entry['run_id']}: SEALED {sealed[:12]} disagrees with runs/INDEX.json")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    shards = list_shards(run_dir / SHARD_DIR)
    sidecars = [p.with_name(p.name + ".sha256").read_text(encoding="ascii").split()[0] for p in shards]
    return sources.Run(entry=entry, run_dir=run_dir, resolved=resolved, records=(), shard_hashes=";".join(sidecars))


def subset_half(run: sources.Run) -> str:
    """first200 | last200 | full: the half of averitec-attack-subset-v1 a run attacked (split files
 `...-first200.json` / `...-last200.json`); any other subset file (or none) is `full`."""
    path = str((run.resolved.get("data") or {}).get("attack_subset") or "")
    stem = Path(path).name
    if stem.endswith("-first200.json"):
        return "first200"
    if stem.endswith("-last200.json"):
        return "last200"
    return "full"


def union_ledgers(root: Path, names, *, table: str, allow_pending: bool = False,
                  regime: str = "R3") -> tuple[list[dict], list[sources.Run]]:
    """Rows of every indexed run of `names` (each row a copy tagged `_run_id`), and the runs. Per run: seal verified,
 strict L2. Across runs: one (item_id, attack, victim, subset) key, else DuplicateLedgerKey. No run indexed:
 MissingRun, unless allow_pending=True, which returns (, ) for the caller to render pending cells."""
    root = Path(root)
    entries = indexed(root, names, regime=regime)
    if not entries:
        if allow_pending:
            return [], []
        raise sources.MissingRun(f"table {table}: none of the {regime} runs {list(names)} is in runs/INDEX.json")
    runs, rows, owner = [], [], {}
    for entry in entries:
        run = load_run_light(root, entry)
        l2 = l2_check(run.run_dir, strict=True)
        if not l2.ok:
            raise L2Violation(f"{run.run_id}: {l2.as_line()}")
        for r in read_ledger(run.run_dir):
            key = ledger_key(r)
            if key in owner:
                raise DuplicateLedgerKey(f"table {table}: ledger key {key} is held by {owner[key]} and {run.run_id}")
            owner[key] = run.run_id
            rows.append({**r, "_run_id": run.run_id})
        runs.append(run)
    return rows, runs


def cut_fired(step_id: str) -> str | None:
    """fired_on of an active, fired configs/cuts.yaml step, else None."""
    step = fmt.cut_step(step_id)
    if step.get("active") and step.get("fired_on"):
        return str(step["fired_on"])
    return None


def read_records(run: sources.Run, record_ids) -> dict[str, dict]:
    """The decision records of `run` whose record_id is in `record_ids` (KeyError when one is absent)."""
    wanted = set(record_ids)
    out: dict[str, dict] = {}
    if not wanted:
        return out
    for p in list_shards(run.run_dir / SHARD_DIR):
        for rec in read_shard(p):
            if rec["record_id"] in wanted:
                out[rec["record_id"]] = rec
        if len(out) == len(wanted):
            break
    missing = wanted - set(out)
    if missing:
        raise KeyError(f"{run.run_id}: {len(missing)} record ids absent from the decision log")
    return out


def read_split(run: sources.Run, split: str) -> list[dict]:
    """Every decision record of `run` with this split (e.g. the victim-source R1 run's `test` records), seq order."""
    recs = [rec for p in list_shards(run.run_dir / SHARD_DIR) for rec in read_shard(p) if rec["split"] == split]
    if not recs:
        raise sources.MissingRun(f"{run.run_id}: no records with split {split!r}")
    return sorted(recs, key=lambda r: r["seq"])


def read_json(run: sources.Run, name: str) -> dict:
    """A sealed JSON sidecar of `run` (calibration.json, gate.json); the seal was verified by load_run_light."""
    path = run.run_dir / name
    if not path.is_file():
        raise sources.MissingRun(f"{run.run_id}: no {name}")
    return json.loads(path.read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- BODEGA scores (interface)

def _unit(x, what: str) -> float:
    v = float(x)
    if math.isnan(v) or v < -BODEGA_TOL or v > 1.0 + BODEGA_TOL:
        raise BodegaScoreError(f"{what}: {x!r} is not a score in [0, 1]")
    return v


def read_bodega_scores(root: Path, names, *, ledger_rows: list[dict],
                       extra_rows: list[dict] | None = None) -> tuple[dict, list[sources.Run]]:
    """{ledger key: score row} and the BS runs, from every indexed sealed BODEGA scoring run of `names`. Each row
 must score an ATTACKED row of `ledger_rows` (same key, source_run_id = that row's run, success_strict = its
 end_state is success); a key scored twice, an unknown key, or bodega != success x semantic x character is
 refused. No run indexed: ({}, ).
 extra_rows : the ledger rows of supplementary runs scored in the same BS run (bs-gpu-host scores the
 gate-aware run, addition A, whose ledger keys repeat victim B's main keys). A score row whose source_run_id is a run
 of extra_rows is checked against extra_rows instead and stored under (source_run_id, *ledger key), so it never
 shadows or collides with a main-table key; without extra_rows such a row is refused as before."""
    root = Path(root)
    ledger = {ledger_key(r): r for r in ledger_rows}
    extra = {(r["_run_id"], *ledger_key(r)): r for r in (extra_rows or [])}
    extra_runs = {r["_run_id"] for r in (extra_rows or [])}
    scores: dict[tuple, dict] = {}
    runs = []
    for entry in indexed(root, names, regime=BODEGA_REGIME):
        run = load_run_light(root, entry)
        path = run.run_dir / BODEGA_FILE
        if not path.is_file():
            raise BodegaScoreError(f"{run.run_id}: sealed BODEGA run without {BODEGA_FILE}")
        for n, line in enumerate(path.read_bytes().split(b"\n"), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = [f for f in BODEGA_FIELDS if f not in row]
            if missing:
                raise BodegaScoreError(f"{run.run_id} line {n}: missing {missing}")
            key = ledger_key(row)
            if row["source_run_id"] in extra_runs:
                key = (row["source_run_id"], *key)
            if key in scores:
                raise BodegaScoreError(f"{run.run_id}: ledger key {key} scored twice")
            src = extra.get(key) if len(key) == 5 else ledger.get(key)
            if src is None or int(src["queries_used"]) <= 0:
                raise BodegaScoreError(f"{run.run_id}: score row {key} matches no attacked ledger row")
            if row["source_run_id"] != src["_run_id"]:
                raise BodegaScoreError(f"{run.run_id}: {key} names {row['source_run_id']}, the ledger row is in {src['_run_id']}")
            success = bool(row["success_strict"])
            if success != (src["end_state"] == "success"):
                raise BodegaScoreError(f"{run.run_id}: {key} success_strict {success} disagrees with end_state {src['end_state']!r}")
            bodega = _unit(row["bodega"], f"{key} bodega")
            if success:
                if row["semantic"] is None or row["character"] is None:
                    raise BodegaScoreError(f"{run.run_id}: {key} is a strict success without semantic/character scores")
                product = _unit(row["semantic"], f"{key} semantic") * _unit(row["character"], f"{key} character")
            else:
                product = 0.0
            if abs(product - bodega) > BODEGA_TOL:
                raise BodegaScoreError(f"{run.run_id}: {key} bodega {bodega} != success x semantic x character {product}")
            scores[key] = {**row, "_bs_run_id": run.run_id}
        runs.append(run)
    return scores, runs
