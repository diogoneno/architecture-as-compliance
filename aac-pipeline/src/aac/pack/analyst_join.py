"""pack-time join: evidence-pack/analyst-events.jsonl and evidence-pack/analyst-join.json.

The namespace's append-only analyst/analyst-events.jsonl (aac.analyst.events) is copied byte for byte into the
pack (an empty file when the namespace has none; the stub namespace has none, and the real one is committed
empty until a person reviews a case). Every event is joined onto its sealed decision-log record, which is read,
never written: analyst-join.json lists per event, in file order, record_id, run_id, seq, decision,
abstention_reason, analyst_queue_id, analyst_action and event_line_sha256 (sha256 of the event's line bytes
without the LF), plus events_file_sha256, n_events and seals_reverified: the sorted run ids whose seal was
recomputed after the join and found equal to runs/INDEX.json (proof that no record was mutated). The join file
carries no timestamp (the Art. 12 action times live only in the verbatim events copy).

Registered in the pack build by (GENERATORS, contracts/pack-files.yaml and the lint exclusion of the
verbatim copy); until then `python -m aac.pack.analyst_join render --out DIR [--events PATH]` renders both files
into DIR from the selected namespace (AAC_OUT)."""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

from aac.analyst.events import AnalystEventRefused, events_path, find_records, load_events
from aac.pack.inputs import PackContext, PackInputError, index_entries
from aac.run.hashing import dumps_sorted
from aac.run.paths import runs_dir
from aac.run.seal import UnsealedRun, verify_sealed

GENERATED_BY = "aac.pack.analyst_join"
EVENTS_COPY = "analyst-events.jsonl"
JOIN_FILE = "analyst-join.json"
FILES = (EVENTS_COPY, JOIN_FILE)
RULE = ("each line of analyst/analyst-events.jsonl (copied verbatim to analyst-events.jsonl) joined on record_id to "
        "the one sealed decision-log record that holds it (decision abstain-to-human); the record is read, never "
        "written; event_line_sha256 = sha256 of the line bytes without the LF; seals_reverified = the runs whose "
        "seal was recomputed after the join and equals runs/INDEX.json")
INPUTS = "analyst/analyst-events.jsonl; sealed runs listed in runs/INDEX.json"


def build(root: Path, pack: Path, ctx: PackContext, events: Path | None = None) -> list[str]:
    root, pack = Path(root), Path(pack)
    src = Path(events) if events is not None else events_path(root)
    raw = src.read_bytes() if src.is_file() else b""
    try:
        lines = load_events(src, root) if src.is_file() else []
        records = find_records(root, [e["record_id"] for _, e in lines])
    except AnalystEventRefused as exc:
        raise PackInputError(f"analyst join: {exc}") from None
    joined = []
    for line, event in lines:
        rec = records[event["record_id"]]
        joined.append({**rec, "analyst_action": event["analyst_action"],
                       "event_line_sha256": hashlib.sha256(line).hexdigest()})
    touched = sorted({j["run_id"] for j in joined})
    by_id = {e["run_id"]: e for e in index_entries(root)}
    for run_id in touched:  # after the join: every source run still has the seal INDEX recorded
        try:
            digest = verify_sealed(runs_dir(root) / run_id)
        except UnsealedRun as exc:
            raise PackInputError(f"analyst join: {run_id}: {exc}") from None
        if digest != by_id[run_id]["sealed_sha256"]:
            raise PackInputError(f"analyst join: {run_id}: SEALED disagrees with runs/INDEX.json after the join")
    pack.mkdir(parents=True, exist_ok=True)
    (pack / EVENTS_COPY).write_bytes(raw)
    doc = {"rule": RULE, "events_file_sha256": hashlib.sha256(raw).hexdigest(), "n_events": len(joined),
           "joined": joined, "seals_reverified": touched}
    with open(pack / JOIN_FILE, "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(doc))
    for name in FILES:
        ctx.record(name, GENERATED_BY, INPUTS)
    return list(FILES)


def main(argv: list[str] | None = None) -> int:
    import json

    from aac.run.paths import root as project_root

    ap = argparse.ArgumentParser(prog="python -m aac.pack.analyst_join",
                                 description="Render analyst-events.jsonl and analyst-join.json from the selected namespace.")
    ap.add_argument("command", choices=("render",))
    ap.add_argument("--out", required=True, help="target directory (created; the two files are overwritten)")
    ap.add_argument("--events", default=None, help="events file (default: <namespace>/analyst/analyst-events.jsonl)")
    args = ap.parse_args(argv)
    root = project_root()
    out = Path(args.out)
    ctx = PackContext(root=root, pack=out, runs=index_entries(root), config_hashes=[], data_versions=[])
    try:
        build(root, out, ctx, events=Path(args.events) if args.events else None)
    except PackInputError as exc:
        print(f"analyst-join: {exc}", file=sys.stderr)
        return 1
    doc = json.loads((out / JOIN_FILE).read_text(encoding="ascii"))
    print(f"analyst-join: {out / JOIN_FILE} n_events={doc['n_events']} seals_reverified={len(doc['seals_reverified'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
