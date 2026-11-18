"""Attack ledger (contracts/attack-ledger.schema.json) and invariant L2.

One row per attacked item, validated on append; `attack-ledger.jsonl` is written
atomically on close in canonical JSON. L2 : the number of decision-log
records of the run equals the sum of len(record_ids) over the ledger rows, and every
record_id is listed exactly once. The CLI refuses to seal an R3 run when L2 fails."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from jsonschema import Draft202012Validator

from aac.logging.schema import format_checker
from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.run.hashing import canonical_rounded as canonical
from aac.run.paths import root as project_root

LEDGER_FILE = "attack-ledger.jsonl"
ADVERSARIAL_FILE = "attack-adversarial.jsonl"  # (item, attack, victim, subset, round, claim_hash): hashes only


class LedgerViolation(Exception):
    pass


def schema_path(root: Path | None = None) -> Path:
    return (Path(root) if root is not None else project_root()) / "contracts" / "attack-ledger.schema.json"


def load_ledger_validator(root: Path | None = None) -> Draft202012Validator:
    schema = json.loads(schema_path(root).read_text(encoding="utf-8"))
    return Draft202012Validator(schema, format_checker=format_checker())


class LedgerWriter:
    def __init__(self, run_dir: Path, root: Path | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.validator = load_ledger_validator(root)
        self.rows: list[dict] = []
        self._seen: set[tuple[str, str, str, str]] = set()

    def append(self, row: dict) -> dict:
        errs = sorted(e.message for e in self.validator.iter_errors(row))
        if errs:
            raise LedgerViolation(f"attack-ledger row violates the contract: {errs[0]}", )
        key = (row["item_id"], row["attack"], row["victim"], row["subset"])
        if key in self._seen:
            raise LedgerViolation(f"duplicate ledger row for {key}")
        if int(row["queries_used"]) > int(row["budget"]):
            raise LedgerViolation("queries_used exceeds budget")
        if row["victim"] == "content_only" and row["posthoc_gate"] is not None:
            raise LedgerViolation("posthoc_gate must be null for victim content_only")
        self._seen.add(key)
        self.rows.append(row)
        return row

    def close(self) -> Path:
        target = self.run_dir / LEDGER_FILE
        tmp = target.with_name(target.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(b"".join(canonical(r) + b"\n" for r in self.rows))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
        return target


def write_adversarial(run_dir: Path, rows: list[dict]) -> Path:
    target = Path(run_dir) / ADVERSARIAL_FILE
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(b"".join(canonical(r) + b"\n" for r in rows))
    os.replace(tmp, target)
    return target


def read_ledger(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / LEDGER_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{run_dir} has no {LEDGER_FILE}")
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line]


def read_adversarial(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / ADVERSARIAL_FILE
    return [json.loads(line) for line in path.read_bytes().split(b"\n") if line] if path.is_file() else []


@dataclass(frozen=True)
class L2Result:
    n_records: int
    n_queries: int
    ok: bool
    detail: str

    strict: bool = True

    def as_line(self) -> str:
        if self.ok and self.strict:
            return f"L2: ok ({self.n_records} records == {self.n_queries} ledger queries)"
        if self.ok:
            return f"L2: ok (lenient: {self.n_queries} ledger queries all present once among {self.n_records} records)"
        return f"L2: FAIL ({self.n_records} records, {self.n_queries} ledger queries): {self.detail}"


def l2_check(run_dir: Path, *, strict: bool = True) -> L2Result:
    """strict: every run record is a ledger query (R3). Non-strict (R4 replay ledgers inside a
 run that also holds fit/test records): every ledger record_id exists exactly once."""
    run_dir = Path(run_dir)
    rows = read_ledger(run_dir)
    listed: list[str] = [rid for r in rows for rid in r["record_ids"]]
    on_disk = [rec["record_id"] for p in list_shards(run_dir / SHARD_DIR) for rec in read_shard(p)]
    n_records, n_queries = len(on_disk), len(listed)
    problems = []
    if len(set(listed)) != n_queries:
        problems.append("duplicate record_ids across ledger rows")
    missing = set(listed) - set(on_disk)
    if missing:
        problems.append(f"{len(missing)} ledger record_ids absent from the decision log")
    if strict and n_records != n_queries:
        problems.append("record count differs from the ledger's query count")
    return L2Result(n_records, n_queries, not problems, "; ".join(problems), strict=bool(strict))
