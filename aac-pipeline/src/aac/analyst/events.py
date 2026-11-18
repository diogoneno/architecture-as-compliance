"""Append-only analyst events (6).

An analyst action on a deferred decision (`confirm`, `override` or `return`) is one line of
`<out_root>/analyst/analyst-events.jsonl`: {record_id, analyst_action, analyst_action_timestamp}, validated
against contracts/analyst-events.schema.json (Draft 2020-12, additionalProperties false). The sealed
decision-log record is never touched: its analyst_action / analyst_action_timestamp stay null, and
aac.pack.analyst_join joins the events onto the records at pack time.

Writer contract (append_event / write_event):
 - the record must exist in exactly one sealed run of the namespace (every logged run listed in runs/INDEX.json
 is seal-checked against its INDEX digest before its shards are read) and its decision must be
 `abstain-to-human`; otherwise AnalystEventRefused and the file is not opened for writing;
 - the file is opened O_WRONLY|O_APPEND|O_CREAT (never truncated, never rewritten), an exclusive fcntl.flock is
 held while the existing bytes are checked (every line whole and schema-valid) and the new line is written with
 one os.write, then fsync; so the previous bytes are always an exact prefix of the new file, two concurrent
 writers land as two whole lines, and an interrupted append leaves no new line or one whole line;
 - one canonical JSON line per event (sorted keys, compact separators, ASCII) plus LF; the timestamp comes from
 aac.run.clock (UTC, `Z`; AAC_RUN_EPOCH gives the fixed clock for tests).
The record id, action and line number are the only things printed; no claim or evidence text is read into an event.
Imports: stdlib, jsonschema (through aac.logging.schema's format checker), aac.logging and aac.run."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path

from aac.logging.schema import format_checker
from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, shard_bytes
from aac.run.clock import make_clock
from aac.run.index import load_index
from aac.run.paths import out_root, root as project_root, runs_dir
from aac.run.seal import UnsealedRun, verify_sealed

EVENTS_DIR = "analyst"
EVENTS_FILE = "analyst-events.jsonl"
SCHEMA_FILE = ("contracts", "analyst-events.schema.json")
ACTIONS = ("confirm", "override", "return")
DEFERRED = "abstain-to-human"
RECORD_KEYS = ("record_id", "run_id", "seq", "decision", "abstention_reason", "analyst_queue_id")


class AnalystEventRefused(ValueError):
    """An event that cannot be appended (unknown or non-deferred record, invalid event) or an events file that
 does not validate (the message names the line)."""


def events_path(root: Path | None = None) -> Path:
    """<out_root>/analyst/analyst-events.jsonl of the selected namespace (AAC_OUT-aware)."""
    return out_root(root) / EVENTS_DIR / EVENTS_FILE


def _validator(root: Path | None = None):
    from jsonschema import Draft202012Validator

    base = Path(root) if root is not None else project_root()
    schema = json.loads(base.joinpath(*SCHEMA_FILE).read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=format_checker())


def canonical_line(event: dict) -> bytes:
    """The one byte form of an event line (without the LF)."""
    return json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def validate_event(event: dict, root: Path | None = None, validator=None) -> None:
    """Raise AnalystEventRefused unless the event validates against contracts/analyst-events.schema.json."""
    validator = validator or _validator(root)
    if not isinstance(event, dict):
        raise AnalystEventRefused("an analyst event must be a JSON object")
    errs = sorted(f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in validator.iter_errors(event))
    if errs:
        raise AnalystEventRefused(f"analyst event violates contracts/analyst-events.schema.json: {errs[0]}")


# -- the sealed record ------------------------------------------------------------------------------

def find_records(root: Path, record_ids) -> dict[str, dict]:
    """{record_id: {record_id, run_id, seq, decision, abstention_reason, analyst_queue_id}} for every requested id.

 Walks the runs/INDEX.json runs (sorted by run id) that have a decision-log/, checks each seal against its INDEX
 digest before reading it, and scans the shards (a line is parsed only when it contains the id). Refuses an id
 held by no sealed run, by more than one record, or whose decision is not abstain-to-human."""
    wanted = {str(r) for r in record_ids}
    hits: dict[str, list[dict]] = {r: [] for r in wanted}
    needles = {r: r.encode("ascii", "replace") for r in wanted}
    if not wanted:
        return {}
    entries = sorted(load_index(runs_dir(root) / "INDEX.json")["runs"], key=lambda e: e["run_id"])
    for entry in entries:
        run_dir = runs_dir(root) / entry["run_id"]
        if not (run_dir / SHARD_DIR).is_dir():
            continue
        try:
            digest = verify_sealed(run_dir)
        except UnsealedRun as exc:
            raise AnalystEventRefused(f"{entry['run_id']}: {exc}") from exc
        if digest != entry["sealed_sha256"]:
            raise AnalystEventRefused(f"{entry['run_id']}: SEALED disagrees with runs/INDEX.json")
        for shard in list_shards(run_dir / SHARD_DIR):
            raw = shard_bytes(shard)
            present = [r for r, b in needles.items() if b in raw]
            if not present:
                continue
            for line in raw.split(b"\n"):
                if not line:
                    continue
                for r in present:
                    if needles[r] in line:
                        rec = json.loads(line)
                        if rec.get("record_id") == r:
                            hits[r].append({"record_id": r, "run_id": entry["run_id"], "seq": rec.get("seq"),
                                            "decision": rec.get("decision"),
                                            "abstention_reason": rec.get("abstention_reason"),
                                            "analyst_queue_id": rec.get("analyst_queue_id")})
    out = {}
    for r in sorted(wanted):
        found = hits[r]
        if not found:
            raise AnalystEventRefused(f"record {r}: no sealed run of this namespace holds it")
        if len(found) > 1:
            raise AnalystEventRefused(f"record {r}: held by {len(found)} records ({', '.join(sorted({f['run_id'] for f in found}))})")
        if found[0]["decision"] != DEFERRED:
            raise AnalystEventRefused(f"record {r}: decision is {found[0]['decision']!r}, not {DEFERRED!r} "
                                      f"(only a deferred case goes to an analyst)")
        out[r] = found[0]
    return out


def find_record(root: Path, record_id: str) -> dict:
    """The one sealed, deferred record with this id (see find_records)."""
    return find_records(root, [record_id])[str(record_id)]


# -- the file ---------------------------------------------------------------------------------------

def _parse(data: bytes, validator, *, name: str) -> list[tuple[bytes, dict]]:
    if data and not data.endswith(b"\n"):
        n = data.count(b"\n") + 1
        raise AnalystEventRefused(f"{name} line {n}: not LF-terminated (a torn append)")
    out = []
    for i, line in enumerate(data.split(b"\n")[:-1] if data else [], start=1):
        try:
            event = json.loads(line)
        except ValueError as exc:
            raise AnalystEventRefused(f"{name} line {i}: not JSON ({type(exc).__name__})") from None
        try:
            validate_event(event, validator=validator)
        except AnalystEventRefused as exc:
            raise AnalystEventRefused(f"{name} line {i}: {exc}") from None
        if canonical_line(event) != line:
            raise AnalystEventRefused(f"{name} line {i}: not the canonical line form (sorted keys, compact, ASCII)")
        out.append((line, event))
    return out


def load_events(path: Path, root: Path | None = None) -> list[tuple[bytes, dict]]:
    """[(line bytes without LF, event)] in file order; an absent file is no events. Every line must be whole,
 schema-valid and canonical; the first bad line raises AnalystEventRefused naming its number."""
    path = Path(path)
    if not path.is_file():
        return []
    return _parse(path.read_bytes(), _validator(root), name=path.name)


def write_event(path: Path, event: dict, root: Path | None = None) -> int:
    """Append one validated event line under an exclusive flock with fsync; return its 1-based line number.
 The existing bytes are re-read and checked under the lock and never rewritten."""
    validator = _validator(root)
    validate_event(event, validator=validator)
    data = canonical_line(event) + b"\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            existing = _parse(path.read_bytes(), validator, name=path.name)
            view = memoryview(data)
            while view:
                n = os.write(fd, view)
                view = view[n:]
            os.fsync(fd)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    return len(existing) + 1


def append_event(root: Path, record_id: str, action: str, *, events: Path | None = None, clock=None) -> dict:
    """Check the record (sealed, unique, abstain-to-human), then append {record_id, analyst_action,
 analyst_action_timestamp} to `events` (default events_path(root)). Returns {event, line, record, path}."""
    root = Path(root)
    if action not in ACTIONS:
        raise AnalystEventRefused(f"analyst action {action!r} is not one of {', '.join(ACTIONS)}")
    event = {"record_id": str(record_id), "analyst_action": action,
             "analyst_action_timestamp": (clock or make_clock()).now_iso_ms()}
    validator = _validator(root)
    validate_event(event, validator=validator)  # malformed id / timestamp refused before any scan
    record = find_record(root, str(record_id))
    path = Path(events) if events is not None else events_path(root)
    line = write_event(path, event, root)
    return {"event": event, "line": line, "record": record, "path": path}
