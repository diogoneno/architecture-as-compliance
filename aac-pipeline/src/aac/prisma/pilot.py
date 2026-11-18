"""`aac prisma pilot start|stop|record|gate` (PRISMA-02, 06-CONTEXT D1-D2).

The timing record is appendix-c/screening/pilot-timing.csv (segment, start_utc, end_utc, start_as_given,
end_as_given), one row per screening segment, so breaks are extra segments.
- start / stop stamp the UTC clock at the analyst's own invocation: an analyst record (06-CONTEXT open
 question 2). `stop --discard` removes an open segment without recording it.
- record takes ISO 8601 times that carry `Z` or an explicit `+HH:MM` / `-HH:MM` offset (the analyst's clock is
 Europe/Sofia), keeps the given strings verbatim in the *_as_given columns and stores YYYY-MM-DDTHH:MM:SSZ.
A time without an offset, an end not after its start, a segment over 8 hours and overlapping segments are refused.

gate counts the pilot rows screened from the workbook (pilot keys with a non-empty screen_decision; never typed),
sums the recorded segments in hours and compares the unrounded rate with the threshold (D2: at or above 200
titles per hour = PROCEED, below = FALLBACK). It writes appendix-c/screening/pilot-gate.md (deterministic: no
clock) and prints `pilot-gate: <VERDICT> rate=<r> titles/h rows=<n> hours=<h> threshold=200`. It refuses, in this
order, while pilot rows are undecided, while no timing is recorded, while a segment is open, and when the pilot
rows carry more than one rule_version. `--declined "<analyst words>"` records a declined pilot (FALLBACK, the
words verbatim) in pilot-declined.txt; while that file exists and pilot rows are undecided, a plain gate writes the
same declined record again, byte for byte.

Exit codes: 0 when the gate or the timing command ran, 1 when it is refused, 2 for usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Sequence

from aac.prisma import schema
from aac.prisma._io import (
    OutsideScreening,
    default_appendix_dir,
    read_csv,
    screening_dir,
    workbook_lock,
    write_csv_atomic,
    write_text_atomic,
)
from aac.prisma.batches import NO_RULE_ID, RULES_FILE, RulesError, load_rules, rules_status
from aac.prisma.sample import BATCHES, PILOT_NAME, keys_sha256

TIMING_FILE = "pilot-timing.csv"
TIMING_HEADER = ("segment", "start_utc", "end_utc", "start_as_given", "end_as_given")
GATE_FILE = "pilot-gate.md"
DECLINED_FILE = "pilot-declined.txt"
AUTOMATION_FILE = "automation.json"
WORKBOOK = "workbook.csv"
THRESHOLD = 200.0
MAX_SEGMENT_HOURS = 8.0
CLOCK = "clock (aac prisma pilot {verb})"
UTC_FMT = "%Y-%m-%dT%H:%M:%SZ"
ISO_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(:\d{2})?(Z|[+-]\d{2}:\d{2})$")
PRECISION_NONE = "precision: not applicable (no proposal column used; no automation tool)"
D2_SENTENCE = ("D2: at or above 200 titles per hour the full title screen proceeds; below it the pre-registered "
               "fallback applies (Appendix C reports the recorded counts as they stand and states the pending "
               "number, dated 1 Nov 2026).")
HOURS_SENTENCE = ("The analyst's hours end 2026-10-28, so the realistic expectation is a partial record on "
                  "1 Nov 2026 even when the verdict is PROCEED (06-RESEARCH Open Question 7).")


class PilotNotReady(RuntimeError):
    """The gate cannot be computed yet (undecided rows, no timing, an open segment, mixed rule versions)."""


class TimingRefused(RuntimeError):
    """A timing record is refused (no offset, end not after start, over 8 h, overlap, open segment)."""


def _now() -> _dt.datetime:
    """The UTC clock, read only by `pilot start|stop` at the analyst's invocation (tests monkeypatch this)."""
    return _dt.datetime.now(_dt.timezone.utc)


def _utc(ts: _dt.datetime) -> str:
    return ts.astimezone(_dt.timezone.utc).strftime(UTC_FMT)


def parse_given(s: str) -> _dt.datetime:
    """An ISO 8601 time with Z or an explicit offset; anything else is refused."""
    s = (s or "").strip()
    if not ISO_RE.match(s):
        raise TimingRefused(f"time {s!r} needs the form YYYY-MM-DDTHH:MM[:SS] with Z or +HH:MM / -HH:MM "
                            "(a time without an offset is ambiguous: your clock is Europe/Sofia)")
    ts = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    if ts.tzinfo is None:  # unreachable after ISO_RE; kept as a guard
        raise TimingRefused(f"time {s!r} has no UTC offset")
    return ts


def _parse_utc(s: str) -> _dt.datetime:
    return _dt.datetime.strptime(s, UTC_FMT).replace(tzinfo=_dt.timezone.utc)


def _hours(start: str, end: str) -> float:
    s, e = _dt.datetime.fromisoformat(start.replace("Z", "+00:00")), _dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
    if e <= s:
        raise ValueError(f"segment end {end} is not after start {start}")
    return (e - s).total_seconds() / 3600


def pilot_rate(segments: Sequence[tuple[str, str]], rows: int) -> float:
    """rows / (sum of the segment durations in hours), unrounded ("Rate gate")."""
    if not segments:
        raise ValueError("no segment")
    return rows / sum(_hours(s, e) for s, e in segments)


def verdict(rate: float, threshold: float = THRESHOLD) -> str:
    """D2: PROCEED iff the unrounded rate is at or above the threshold."""
    return "PROCEED" if rate >= threshold else "FALLBACK"


# ------------------------------------------------------------------ timing record


def _timing_path(appendix: Path) -> Path:
    return screening_dir(appendix) / TIMING_FILE


def read_timing(appendix: Path) -> list[dict[str, str]]:
    path = _timing_path(appendix)
    if not path.is_file():
        return []
    header, rows = read_csv(path)
    if tuple(header) != TIMING_HEADER:
        raise TimingRefused(f"{TIMING_FILE} header is {header!r}, expected {list(TIMING_HEADER)}")
    return rows


def _write_timing(appendix: Path, rows: list[dict[str, str]]) -> None:
    write_csv_atomic(_timing_path(appendix), TIMING_HEADER, ([r[c] for c in TIMING_HEADER] for r in rows), appendix)


def check_segments(rows: list[dict[str, str]]) -> list[str]:
    """Problems with closed segments: end not after start, over 8 h, overlaps (touching segments are allowed)."""
    probs, spans = [], []
    for r in rows:
        if not r["end_utc"]:
            continue
        s, e = _parse_utc(r["start_utc"]), _parse_utc(r["end_utc"])
        if e <= s:
            probs.append(f"segment {r['segment']}: end {r['end_utc']} is not after start {r['start_utc']}")
        elif (e - s).total_seconds() > MAX_SEGMENT_HOURS * 3600:
            probs.append(f"segment {r['segment']}: {(e - s).total_seconds() / 3600:.2f} h is over {MAX_SEGMENT_HOURS:g} h")
        spans.append((s, e, r["segment"]))
    spans.sort()
    for (s1, e1, n1), (s2, e2, n2) in zip(spans, spans[1:]):
        if s2 < e1:
            probs.append(f"segments {n1} and {n2} overlap")
    return probs


def _open(rows: list[dict[str, str]]) -> dict[str, str] | None:
    return next((r for r in rows if not r["end_utc"]), None)


def _next_segment(rows: list[dict[str, str]]) -> str:
    return str(max([int(r["segment"]) for r in rows] or [0]) + 1)


def start(appendix: Path) -> dict[str, str]:
    appendix = Path(appendix)
    with workbook_lock(appendix, shared=False):
        rows = read_timing(appendix)
        if _open(rows):
            raise TimingRefused(f"segment {_open(rows)['segment']} is still open; run `aac prisma pilot stop` first")
        now = _utc(_now())
        if any(r["end_utc"] and _parse_utc(r["end_utc"]) > _parse_utc(now) for r in rows):
            raise TimingRefused(f"a recorded segment ends after now ({now})")
        row = {"segment": _next_segment(rows), "start_utc": now, "end_utc": "",
               "start_as_given": CLOCK.format(verb="start"), "end_as_given": ""}
        _write_timing(appendix, rows + [row])
    return row


def stop(appendix: Path, discard: bool = False) -> tuple[dict[str, str], float | None]:
    appendix = Path(appendix)
    with workbook_lock(appendix, shared=False):
        rows = read_timing(appendix)
        row = _open(rows)
        if row is None:
            raise TimingRefused("no segment is open; run `aac prisma pilot start` first")
        if discard:
            _write_timing(appendix, [r for r in rows if r is not row])
            return row, None
        row["end_utc"] = _utc(_now())
        row["end_as_given"] = CLOCK.format(verb="stop")
        probs = check_segments(rows)
        if probs:
            raise TimingRefused("; ".join(probs) + " (nothing written; `aac prisma pilot stop --discard` drops the "
                                "open segment, `aac prisma pilot record` records given times)")
        _write_timing(appendix, rows)
    return row, _hours(row["start_utc"], row["end_utc"])


def record(appendix: Path, start_given: str, end_given: str) -> tuple[dict[str, str], float]:
    appendix = Path(appendix)
    s, e = parse_given(start_given), parse_given(end_given)
    with workbook_lock(appendix, shared=False):
        rows = read_timing(appendix)
        if _open(rows):
            raise TimingRefused(f"segment {_open(rows)['segment']} is open; stop or discard it first")
        row = {"segment": _next_segment(rows), "start_utc": _utc(s), "end_utc": _utc(e),
               "start_as_given": start_given.strip(), "end_as_given": end_given.strip()}
        probs = check_segments(rows + [row])
        if probs:
            raise TimingRefused("; ".join(probs))
        _write_timing(appendix, rows + [row])
    return row, _hours(row["start_utc"], row["end_utc"])


# ------------------------------------------------------------------ gate


def _rel(path: Path, appendix: Path) -> str:
    from aac.prisma.workbook import _rel as rel
    return rel(path, appendix)


def _pilot_keys(appendix: Path) -> list[str]:
    path = screening_dir(appendix) / BATCHES / f"{PILOT_NAME}.csv"
    if not path.is_file():
        raise PilotNotReady(f"no pilot batch ({_rel(path, appendix)}); run `aac prisma sample pilot` first")
    return [r["key"] for r in read_csv(path)[1]]


def _quote(text: str) -> list[str]:
    return ["> " + line if line else ">" for line in text.split("\n")]


def _declined_md(appendix: Path, words: str, n: int, decided: int) -> str:
    status = "pilot not run" if decided == 0 else f"pilot not completed ({decided} of {n} pilot rows decided)"
    return "\n".join([
        "# Pilot rate gate",
        "",
        "Generated by `aac prisma pilot gate`; regenerate, never edit.",
        "",
        f"- Verdict: FALLBACK ({status})",
        "- Rate: not measured; hours: not recorded",
        f"- Threshold: 200 titles per hour. {D2_SENTENCE}",
        f"- Pilot sample: `{_rel(screening_dir(appendix) / BATCHES / (PILOT_NAME + '.csv'), appendix)}` ({n} keys)",
        f"- Pilot rows decided (counted from `{WORKBOOK}`): {decided} of {n}",
        "",
        f"The analyst declined the pilot, in these words (verbatim, `appendix-c/screening/{DECLINED_FILE}`):",
        "",
        *_quote(words),
        "",
        HOURS_SENTENCE,
        "",
    ])


def _read_declined(appendix: Path) -> str | None:
    path = screening_dir(appendix) / DECLINED_FILE
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def _precision_lines(pilot_rows: list[dict[str, str]], appendix: Path) -> list[str]:
    proposed = [r for r in pilot_rows if (r.get("proposal") or "").strip()]
    if not proposed:
        return [PRECISION_NONE]
    if not (screening_dir(appendix) / AUTOMATION_FILE).is_file():
        raise PilotNotReady(f"{len(proposed)} pilot rows carry a proposal but appendix-c/screening/{AUTOMATION_FILE} "
                            "(tool, version, prompt hash, date) is absent; a proposal must be disclosed (PRISMA Item 8)")
    prop_ex = [r for r in proposed if r["proposal"].strip().lower() == "exclude"]
    agree = sum(1 for r in prop_ex if r["screen_decision"] == "exclude")
    missed = sum(1 for r in prop_ex if r["screen_decision"] == "include")
    if not prop_ex:
        return ["precision: not defined (no proposed exclusion on the pilot rows)"]
    return [f"precision of proposed exclusions: agree-exclude / proposed-exclude = {agree} / {len(prop_ex)} = "
            f"{agree / len(prop_ex):.3f}",
            f"analyst includes the proposal would have excluded: {missed}"]


def gate(appendix: Path, threshold: float = THRESHOLD, declined: str | None = None) -> dict:
    """Compute and write pilot-gate.md; return {verdict, rate, rows, hours, threshold, declined}."""
    appendix = Path(appendix)
    scr = screening_dir(appendix)
    keys = _pilot_keys(appendix)
    n = len(keys)
    with workbook_lock(appendix, shared=True):
        _, wb = read_csv(scr / WORKBOOK)
    by_key = {r["key"]: r for r in wb}
    missing = [k for k in keys if k not in by_key]
    if missing:
        raise PilotNotReady(f"{len(missing)} pilot keys are not in the workbook (first: {missing[0]})")
    pilot_rows = [by_key[k] for k in keys]
    decided = [r for r in pilot_rows if (r.get("screen_decision") or "").strip()]
    undecided = n - len(decided)
    recorded = _read_declined(appendix)
    if declined is not None:
        if not declined.strip():
            raise PilotNotReady("--declined needs the analyst's words")
        if undecided == 0:
            raise PilotNotReady(f"all {n} pilot rows are decided; run the gate without --declined")
        if recorded is not None and recorded != declined:
            raise PilotNotReady(f"appendix-c/screening/{DECLINED_FILE} already holds different words")
        if recorded is None:
            write_text_atomic(scr / DECLINED_FILE, declined + "\n", appendix)
        recorded = declined
    if recorded is not None and undecided > 0:
        write_text_atomic(scr / GATE_FILE, _declined_md(appendix, recorded, n, len(decided)), appendix)
        return {"verdict": "FALLBACK", "rate": None, "rows": len(decided), "hours": None,
                "threshold": threshold, "declined": True}
    if undecided:
        raise PilotNotReady(f"{undecided} of {n} pilot rows undecided")
    timing = read_timing(appendix)
    if not timing:
        raise PilotNotReady("no pilot timing recorded")
    if _open(timing):
        raise PilotNotReady("pilot timing has an open segment")
    probs = check_segments(timing)
    if probs:
        raise PilotNotReady("pilot timing is invalid: " + "; ".join(probs))
    versions = sorted({r.get("rule_version") or "" for r in decided})
    if len(versions) != 1 or not versions[0]:
        raise PilotNotReady(f"pilot rows carry {len(versions)} rule versions ({', '.join(v or '(none)' for v in versions)})")
    try:
        _, version_now = load_rules(appendix)
        status = rules_status(appendix)
    except RulesError as exc:
        raise PilotNotReady(str(exc)) from exc
    precision = _precision_lines(pilot_rows, appendix)
    rows = len(decided)
    segs = [(r["start_utc"], r["end_utc"]) for r in timing]
    hours = sum(_hours(s, e) for s, e in segs)
    rate = rows / hours
    v = verdict(rate, threshold)
    include = sum(1 for r in decided if r["screen_decision"] == "include")
    excl = Counter(r["exclude_reason"] for r in decided if r["screen_decision"] == "exclude")
    rule_ids = Counter(r.get("title_rule_id") or "" for r in decided)
    r0 = rule_ids.get(NO_RULE_ID, 0)
    sample_txt = scr / BATCHES / f"{PILOT_NAME}.sample.txt"
    lines = [
        "# Pilot rate gate",
        "",
        "Generated by `aac prisma pilot gate`; regenerate, never edit. Every value below is read from a file:",
        "rows from the workbook, times from the timing record.",
        "",
        "## Inputs",
        "",
        f"- Pilot sample: `{_rel(scr / BATCHES / (PILOT_NAME + '.csv'), appendix)}`, {n} keys, keys_sha256 "
        f"{keys_sha256(keys)} (record: `{_rel(sample_txt, appendix)}`)",
        f"- Rows screened: pilot keys with a non-empty screen_decision in `{_rel(scr / WORKBOOK, appendix)}`: {rows}",
        f"- Timing record: `{_rel(scr / TIMING_FILE, appendix)}`, {len(timing)} segment(s)",
        "",
        "| segment | start (UTC) | end (UTC) | start as given | end as given | hours |",
        "|---|---|---|---|---|---|",
    ]
    for r in timing:
        lines.append(f"| {r['segment']} | {r['start_utc']} | {r['end_utc']} | {r['start_as_given']} | "
                     f"{r['end_as_given']} | {_hours(r['start_utc'], r['end_utc']):.4f} |")
    lines += [
        "",
        f"- Total hours: {hours:.4f}",
        "",
        "## Rate and verdict",
        "",
        "- Formula: rate = rows screened / total hours",
        f"- Rate: {rows} / {hours:.4f} h = {rate:.1f} titles per hour (the unrounded rate is compared)",
        f"- Threshold: {threshold:g} titles per hour. {D2_SENTENCE}",
        f"- Verdict: {v}",
        "",
        "## Rule and decisions",
        "",
        f"- rule_version on the pilot rows: {versions[0]}",
        f"- rule_version of `{RULES_FILE}` now: {version_now} (same: {'yes' if version_now == versions[0] else 'no'})",
        f"- {status}",
        f"- Rule coverage: (decided - {NO_RULE_ID}) / decided = ({rows} - {r0}) / {rows} = {100 * (rows - r0) / rows:.1f} %",
        "- Rule ids used: " + ", ".join(f"{k} {c}" for k, c in sorted(rule_ids.items(), key=lambda kv: (len(kv[0]), kv[0]))),
        f"- Include: {include}",
        f"- Exclude: {rows - include}",
    ]
    for code in schema.REASON_CODES:
        lines.append(f"  - {code}: {excl.get(code, 0)}")
    lines += ["- " + p for p in precision]
    lines += ["", HOURS_SENTENCE, ""]
    write_text_atomic(scr / GATE_FILE, "\n".join(lines), appendix)
    return {"verdict": v, "rate": rate, "rows": rows, "hours": hours, "threshold": threshold, "declined": False}


def gate_line(res: dict) -> str:
    if res["declined"]:
        return (f"pilot-gate: FALLBACK rate=not-measured rows={res['rows']} hours=not-recorded "
                f"threshold={res['threshold']:g}")
    return (f"pilot-gate: {res['verdict']} rate={res['rate']:.1f} titles/h rows={res['rows']} "
            f"hours={res['hours']:.4f} threshold={res['threshold']:g}")


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma pilot")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("start", help="open a timing segment, stamped with the UTC clock now")
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("stop", help="close the open segment, stamped with the UTC clock now")
    p.add_argument("--discard", action="store_true", help="drop the open segment without recording it")
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("record", help="record a segment from given times that carry Z or +HH:MM")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("gate", help="compute the pilot rate and the D2 verdict; write pilot-gate.md")
    p.add_argument("--declined", default=None, help="the analyst's words declining the pilot (verbatim)")
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        if args.sub == "start":
            row = start(appendix)
            print(f"pilot: segment {row['segment']} started {row['start_utc']}")
        elif args.sub == "stop":
            row, h = stop(appendix, discard=args.discard)
            print(f"pilot: segment {row['segment']} discarded" if h is None
                  else f"pilot: segment {row['segment']} stopped {row['end_utc']} ({h:.4f} h)")
        elif args.sub == "record":
            row, h = record(appendix, args.start, args.end)
            print(f"pilot: segment {row['segment']} recorded {row['start_utc']}..{row['end_utc']} ({h:.4f} h)")
        else:
            print(gate_line(gate(appendix, declined=args.declined)))
    except (PilotNotReady, TimingRefused, FileNotFoundError, OutsideScreening, ValueError) as exc:
        print(f"aac prisma pilot {args.sub}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
