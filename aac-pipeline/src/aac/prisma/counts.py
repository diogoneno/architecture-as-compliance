"""`aac prisma count --as-of YYYY-MM-DD` (PRISMA-04, 06-CONTEXT D2/D4/D5; 4).

One state-agnostic counter: it reads the workbook as it stands and writes appendix-c/screening/table_c2_screening.csv
for a complete, a partial and an empty screen alike. There is no separate fallback path: under the D2 fallback the
same command, run with --as-of 2026-11-01, is the record. Every stage row carries its count and an explicit pending
number (the rows that have reached the stage and are not yet decided there), so an unfinished stage is never a blank.

Identification rows are read from the v05 table_c2_identification.csv and recomputed from prisma_ids.json (set sizes,
their sum, sum minus union); any disagreement raises IdentificationMismatch. The run-folder manifest and the workbook
copy check (aac.prisma.workbook.verify) run first; a changed run folder or a differing copy refuses the count.
Rows the screener marked `duplicate at title level` form a second de-duplication step (duplicates 1,572 + n, records
to screen 3,491 - n); they are neither screened nor excluded.

Invariants asserted on every run (InvariantBroken names the invariant and the observed values):
 screened = title excluded + title included
 title included = abstract excluded + reports sought + abstract pending
 reports sought = not retrieved + assessed + retrieval pending
 assessed = full-text excluded + studies included + full-text pending
and every exclusion carries a reason from schema.REASON_CODES; a decision at a stage the row never reached is refused.

The --as-of date is an argument echoed into the CSV header; nothing reads the clock.
Exit codes: 0 counted, 1 refused, 2 usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
import sys
from collections import Counter
from pathlib import Path

from aac.prisma import schema
from aac.prisma._io import (
    OutsideScreening,
    csv_text,
    default_appendix_dir,
    read_csv,
    screening_dir,
    workbook_lock,
    write_text_atomic,
)
from aac.prisma.sample import SECOND_SHARE, in_second_screen
from aac.run.hashing import sha256_file

TABLE_FILE = "table_c2_screening.csv"
TABLE_COLUMNS = ("stage", "label", "count", "pending", "source")
WORKBOOK = "workbook.csv"
IDENT_FILE = "table_c2_identification.csv"
IDS_FILE = "prisma_ids.json"
SUMMARY_FILE = "prisma_summary.json"
ROUTE_FILE = "ref-route-v05.csv"
TITLE_DUP = "duplicate at title level"
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HEADER_RE = re.compile(r"^# as_of=(\d{4}-\d{2}-\d{2}) workbook_sha256=([0-9a-f]{64}) run=(\S+) "
                       r"identification_run_date=(\d{4}-\d{2}-\d{2})$")
SOURCES = (("openalex", "OpenAlex"), ("arxiv", "arXiv"), ("s2", "Semantic Scholar"))
WB = f"screening/{WORKBOOK}"


class IdentificationMismatch(RuntimeError):
    """table_c2_identification.csv disagrees with prisma_ids.json (or the workbook row count): no count is written."""


class InvariantBroken(RuntimeError):
    """A PRISMA invariant or the reason vocabulary does not hold on the workbook as it stands."""


def as_of_date(s: str) -> str:
    if not DATE_RE.match(s or ""):
        raise ValueError(f"--as-of must be YYYY-MM-DD, got {s!r}")
    _dt.date.fromisoformat(s)
    return s


def _v(row: dict[str, str], col: str) -> str:
    return (row.get(col) or "").strip()


# ------------------------------------------------------------------ identification


def identification(appendix: Path, run_name: str = schema.CANONICAL_RUN) -> dict[str, int]:
    """The v05 identification values, each asserted equal to its recomputation from prisma_ids.json."""
    run = Path(appendix) / run_name
    _, rows = read_csv(run / IDENT_FILE)
    table = {r["row"]: r["n"] for r in rows}
    ids = json.loads((run / IDS_FILE).read_text(encoding="utf-8"))
    sets = {src: set(ids.get(src) or ()) for src, _ in SOURCES}
    union: set[str] = set()
    for s in sets.values():
        union |= s
    total = sum(len(s) for s in sets.values())
    want = {
        "records_identified_openalex": len(sets["openalex"]),
        "records_identified_arxiv": len(sets["arxiv"]),
        "records_identified_s2": len(sets["s2"]),
        "records_identified_api_databases": total,
        "duplicates_removed_api": total - len(union),
        "records_to_screen_api": len(union),
    }
    out: dict[str, int] = {}
    for key, value in want.items():
        got = table.get(key)
        if got is None or not got.strip().isdigit() or int(got) != value:
            raise IdentificationMismatch(
                f"{run_name}/{IDENT_FILE} {key}={got!r} but {IDS_FILE} gives {value}")
        out[key] = value
    return out


def identification_run_date(appendix: Path, run_name: str = schema.CANONICAL_RUN) -> str:
    """The date part of prisma_summary.json run_date (a clock time under reports/ would fail lint-determinism)."""
    summary = json.loads((Path(appendix) / run_name / SUMMARY_FILE).read_text(encoding="utf-8"))
    d = str(summary.get("run_date") or "")[:10]
    as_of_date(d)
    return d


def other_methods(appendix: Path) -> int:
    _, rows = read_csv(Path(appendix) / ROUTE_FILE)
    return sum(1 for r in rows if _v(r, "route") == "other")


# ------------------------------------------------------------------ kappa


def kappa_text(pairs: list[tuple[str, str]]) -> str:
    """Cohen's kappa (scikit-learn) over include/exclude pairs, to 3 decimals, or `not computable (<reason>)`."""
    if not pairs:
        return "not computable (no second-screen decisions)"
    if len(pairs) < 2:
        return "not computable (one second-screen decision)"
    first = {a for a, _ in pairs}
    second = {b for _, b in pairs}
    if len(first) < 2 or len(second) < 2:
        return "not computable (single class)"
    from sklearn.metrics import cohen_kappa_score

    k = float(cohen_kappa_score([a for a, _ in pairs], [b for _, b in pairs], labels=list(schema.DECISIONS)))
    return f"{k:.3f}"


# ------------------------------------------------------------------ the counter


def _row(stage: str, label: str, count, pending, source: str) -> dict[str, str]:
    return {"stage": stage, "label": label, "count": str(count), "pending": "" if pending is None else str(pending),
            "source": source}


def _check_reason(reason: str, where: str, problems: list[str]) -> None:
    if reason not in schema.REASON_CODES:
        problems.append(f"{where}: reason {reason!r} is not a reason code")


def _tally(wb: list[dict[str, str]]) -> dict:
    """Per-row stage walk; problems collects every vocabulary or stage violation."""
    problems: list[str] = []
    t = Counter()
    reasons = Counter()
    pairs: list[tuple[str, str]] = []
    screeners: set[str] = set()
    for r in wb:
        key = r.get("key", "")
        dec = _v(r, "screen_decision")
        reason = _v(r, "exclude_reason")
        if reason == TITLE_DUP:
            if dec != "exclude":
                problems.append(f"key {key}: exclude_reason {TITLE_DUP!r} with screen_decision {dec!r}")
            t["title_dup"] += 1
            for col in ("abstract_decision", "retrieved", "fulltext_decision"):
                if _v(r, col):
                    problems.append(f"key {key}: {col} set on a duplicate at title level")
        else:
            t["to_screen"] += 1
            if dec == "":
                t["title_pending"] += 1
                if reason:
                    problems.append(f"key {key}: exclude_reason {reason!r} without a screen_decision")
            elif dec == "exclude":
                t["title_excl"] += 1
                if not reason:
                    problems.append(f"key {key}: title exclusion without an exclude_reason")
                else:
                    _check_reason(reason, f"key {key} exclude_reason", problems)
            elif dec == "include":
                t["title_incl"] += 1
                if reason:
                    problems.append(f"key {key}: title include carries exclude_reason {reason!r}")
            else:
                problems.append(f"key {key}: screen_decision {dec!r} is not include or exclude")
        if dec in schema.DECISIONS and _v(r, "screener"):
            screeners.add(_v(r, "screener"))
        title_incl = dec == "include" and reason != TITLE_DUP
        # abstract stage: title includes only
        adec, areason = _v(r, "abstract_decision"), _v(r, "abstract_reason")
        if adec and not title_incl:
            problems.append(f"key {key}: abstract_decision {adec!r} on a row not included on title")
        sought = False
        if title_incl:
            src = _v(r, "abstract_source")
            if not src:
                t["abs_unfetched"] += 1
            elif src == "none":
                t["abs_without"] += 1
            else:
                t["abs_with"] += 1
            if adec == "":
                t["abs_pending"] += 1
            elif adec == "exclude":
                t["abs_excl"] += 1
                if not areason:
                    problems.append(f"key {key}: abstract exclusion without an abstract_reason")
                else:
                    _check_reason(areason, f"key {key} abstract_reason", problems)
            elif adec == "include":
                t["sought"] += 1
                sought = True
            else:
                problems.append(f"key {key}: abstract_decision {adec!r} is not include or exclude")
        # retrieval: reports sought only
        ret = _v(r, "retrieved")
        if ret and not sought:
            problems.append(f"key {key}: retrieved {ret!r} on a report that was not sought")
        assessed = False
        if sought:
            if ret == "":
                t["ret_pending"] += 1
            elif ret == "no":
                t["not_retrieved"] += 1
            elif ret == "yes":
                t["assessed"] += 1
                assessed = True
            else:
                problems.append(f"key {key}: retrieved {ret!r} is not yes or no")
        # full text: assessed reports only
        fdec, freason = _v(r, "fulltext_decision"), _v(r, "fulltext_reason")
        if fdec and not assessed:
            problems.append(f"key {key}: fulltext_decision {fdec!r} on a report that was not assessed")
        if assessed:
            if fdec == "":
                t["ft_pending"] += 1
            elif fdec == "exclude":
                t["ft_excl"] += 1
                if not freason:
                    problems.append(f"key {key}: full-text exclusion without a fulltext_reason")
                else:
                    _check_reason(freason, f"key {key} fulltext_reason", problems)
                    reasons[freason] += 1
            elif fdec == "include":
                t["included"] += 1
            else:
                problems.append(f"key {key}: fulltext_decision {fdec!r} is not include or exclude")
        if _v(r, "included_in_chapter2") == "yes":
            t["ch2_yes"] += 1
        if _v(r, "title_dup_group"):
            t["dup_hint_rows"] += 1
        if _v(r, "proposal"):
            t["proposals"] += 1
        # second screen: members of the hash sample that carry a title decision
        sdec = _v(r, "second_screen_decision")
        if sdec and sdec not in schema.DECISIONS:
            problems.append(f"key {key}: second_screen_decision {sdec!r} is not include or exclude")
        if sdec and not dec:
            problems.append(f"key {key}: second_screen_decision without a title decision")
        if dec and in_second_screen(key):
            t["second_sampled"] += 1
            if sdec in schema.DECISIONS:
                t["second_decided"] += 1
                t["second_agree"] += int(sdec == dec)
                pairs.append((dec, sdec))
    t["dup_hint_groups"] = len({_v(r, "title_dup_group") for r in wb if _v(r, "title_dup_group")})
    return {"t": t, "reasons": reasons, "pairs": pairs, "problems": problems, "screeners": sorted(screeners)}


def _invariants(t: Counter) -> None:
    checks = [
        ("screened = title excluded + title included", t["screened"], t["title_excl"] + t["title_incl"],
         f"{t['screened']} != {t['title_excl']} + {t['title_incl']}"),
        ("title included = abstract excluded + sought + abstract pending", t["title_incl"],
         t["abs_excl"] + t["sought"] + t["abs_pending"],
         f"{t['title_incl']} != {t['abs_excl']} + {t['sought']} + {t['abs_pending']}"),
        ("sought = not retrieved + assessed + retrieval pending", t["sought"],
         t["not_retrieved"] + t["assessed"] + t["ret_pending"],
         f"{t['sought']} != {t['not_retrieved']} + {t['assessed']} + {t['ret_pending']}"),
        ("assessed = full-text excluded + included + full-text pending", t["assessed"],
         t["ft_excl"] + t["included"] + t["ft_pending"],
         f"{t['assessed']} != {t['ft_excl']} + {t['included']} + {t['ft_pending']}"),
        ("records to screen = screened + title pending", t["to_screen"], t["screened"] + t["title_pending"],
         f"{t['to_screen']} != {t['screened']} + {t['title_pending']}"),
    ]
    for name, a, b, observed in checks:
        if a != b:
            raise InvariantBroken(f"invariant {name} broken: {observed}")


def compute(appendix: Path, as_of: str, run_name: str = schema.CANONICAL_RUN) -> list[dict[str, str]]:
    """The ordered Table C.2 rows (TABLE_COLUMNS) for the workbook as it stands on --as-of."""
    from aac.prisma import workbook

    appendix = Path(appendix)
    as_of_date(as_of)
    copy, diffs, _ = workbook.verify(appendix, run_name=run_name)
    if diffs:
        raise workbook.RunFolderChanged("run folders differ from RUN-FOLDER.sha256: " + ", ".join(diffs[:20]))
    if not copy.ok:
        raise workbook.RunFolderChanged("workbook copy check differs: " + ", ".join(copy.differs[:20]))
    ident = identification(appendix, run_name)
    other = other_methods(appendix)
    with workbook_lock(appendix, shared=True):
        _, wb = read_csv(screening_dir(appendix) / WORKBOOK)
    if len(wb) != ident["records_to_screen_api"]:
        raise IdentificationMismatch(f"{WB} has {len(wb)} rows but records_to_screen_api is "
                                     f"{ident['records_to_screen_api']}")
    res = _tally(wb)
    if res["problems"]:
        shown = "; ".join(res["problems"][:20])
        more = f" (+{len(res['problems']) - 20} more)" if len(res["problems"]) > 20 else ""
        raise InvariantBroken(f"{len(res['problems'])} workbook problem(s): {shown}{more}")
    t = res["t"]
    t["screened"] = t["title_excl"] + t["title_incl"]
    n_dup = t["title_dup"]
    to_screen = ident["records_to_screen_api"] - n_dup
    if to_screen != t["to_screen"]:
        raise InvariantBroken(f"invariant records to screen = union - title duplicates broken: "
                              f"{t['to_screen']} != {ident['records_to_screen_api']} - {n_dup}")
    _invariants(t)
    title_p, abs_p, ret_p, ft_p = t["title_pending"], t["abs_pending"], t["ret_pending"], t["ft_pending"]
    run_ident = f"{run_name}/{IDENT_FILE}"
    rows = [
        _row("identified_openalex", "Records identified from databases: OpenAlex",
             ident["records_identified_openalex"], None, f"{run_ident} = {IDS_FILE} openalex set size"),
        _row("identified_arxiv", "Records identified from databases: arXiv",
             ident["records_identified_arxiv"], None, f"{run_ident} = {IDS_FILE} arxiv set size"),
        _row("identified_s2", "Records identified from databases: Semantic Scholar",
             ident["records_identified_s2"], None, f"{run_ident} = {IDS_FILE} s2 set size"),
        _row("identified_total", "Records identified from databases: total",
             ident["records_identified_api_databases"], None, f"{run_ident} = sum of the {IDS_FILE} set sizes"),
        _row("identified_other_methods",
             "Records identified via other methods (citation searching, the September 2026 expert review, "
             "direct retrieval of legal and standards texts)",
             other, None, f"{ROUTE_FILE} rows with route=other"),
        _row("duplicates_removed_api", "Duplicate records removed across sources",
             ident["duplicates_removed_api"], None, f"{run_ident} = {IDS_FILE} sum - union"),
        _row("duplicates_removed_title", "Duplicate records removed at title level (screener-confirmed)",
             n_dup, None, f"{WB} rows with exclude_reason '{TITLE_DUP}'"),
        _row("duplicates_removed_total", "Duplicate records removed",
             ident["duplicates_removed_api"] + n_dup, None, "duplicates_removed_api + duplicates_removed_title"),
        _row("records_to_screen", "Records to screen", to_screen, None,
             f"{run_ident} records_to_screen_api - duplicates_removed_title"),
        _row("records_screened", "Records screened on title and abstract", t["screened"], title_p,
             f"{WB} rows with a screen_decision; pending = records_to_screen - screened"),
        _row("records_excluded_title", "Records excluded at screening: on title", t["title_excl"], title_p,
             f"{WB} screen_decision exclude; pending = title not yet screened"),
        _row("records_excluded_abstract", "Records excluded at screening: on abstract", t["abs_excl"], abs_p,
             f"{WB} abstract_decision exclude; pending = title includes without an abstract_decision"),
        _row("records_excluded", "Records excluded at screening", t["title_excl"] + t["abs_excl"], title_p + abs_p,
             "records_excluded_title + records_excluded_abstract; pending = title pending + abstract pending"),
        _row("reports_sought", "Reports sought for retrieval", t["sought"], abs_p,
             f"{WB} abstract_decision include; pending = abstract pending"),
        _row("reports_not_retrieved", "Reports not retrieved", t["not_retrieved"], ret_p,
             f"{WB} retrieved no among reports sought; pending = sought without retrieved"),
        _row("reports_assessed", "Reports assessed for eligibility (full text)", t["assessed"], ret_p,
             f"{WB} retrieved yes among reports sought; pending = sought - decided retrieval"),
    ]
    for code in schema.REASON_CODES:
        rows.append(_row(f"reports_excluded:{code}", f"Reports excluded, with reasons: {code}",
                         res["reasons"].get(code, 0), ft_p,
                         f"{WB} fulltext_decision exclude with fulltext_reason '{code}'; pending = full-text pending"))
    rows += [
        _row("reports_excluded_total", "Reports excluded, with reasons", t["ft_excl"], ft_p,
             f"{WB} fulltext_decision exclude; pending = assessed without a fulltext_decision"),
        _row("studies_included", "Studies included in the review", t["included"], ft_p,
             f"{WB} fulltext_decision include; pending = assessed - excluded - included"),
        _row("included_in_chapter2_yes", "Workbook rows marked included_in_chapter2 = yes", t["ch2_yes"], None,
             f"{WB} included_in_chapter2 yes"),
        _row("title_dup_candidate_groups", "Title-duplicate candidate groups (hints, never decisions)",
             t["dup_hint_groups"], None, f"{WB} distinct non-empty title_dup_group"),
        _row("title_dup_candidate_rows", "Rows in title-duplicate candidate groups", t["dup_hint_rows"], None,
             f"{WB} rows with a title_dup_group"),
        _row("abstracts_with_text", "Title includes with an abstract from an API", t["abs_with"], None,
             f"{WB} title includes with abstract_source set and not 'none'"),
        _row("abstracts_without_text", "Title includes with no abstract available", t["abs_without"],
             t["abs_unfetched"], f"{WB} title includes with abstract_source 'none'; pending = abstract_source empty"),
        _row("second_screen_share_percent", "Second-screen sample share (percent of title-decided rows)",
             f"{SECOND_SHARE * 100:g}", None, "aac.prisma.sample.SECOND_SHARE x 100"),
        _row("second_screen_sampled", "Second-screen sample: title-decided rows in the hash sample",
             t["second_sampled"], None, f"{WB} rows with a screen_decision and in_second_screen(key)"),
        _row("second_screen_decided", "Second-screen sample: rows with a second decision", t["second_decided"],
             t["second_sampled"] - t["second_decided"], f"{WB} second_screen_decision set on sampled rows"),
        _row("second_screen_agreements", "Second-screen sample: second decision equals the first",
             t["second_agree"], None, f"{WB} second_screen_decision = screen_decision"),
        _row("second_screen_kappa", "Second-screen agreement: Cohen's kappa (include/exclude)",
             kappa_text(res["pairs"]), None, "sklearn.metrics.cohen_kappa_score over the decided sample pairs"),
        _row("screeners_recorded", "Screener identities recorded on title-decided rows",
             len(res["screeners"]), None,
             f"{WB} screener" + (": " + "; ".join(res["screeners"]) if res["screeners"] else ": none recorded")),
        _row("automation_proposals", "Rows carrying an automation proposal", t["proposals"], None,
             f"{WB} rows with a non-empty proposal"),
    ]
    return rows


def header_line(appendix: Path, as_of: str, run_name: str = schema.CANONICAL_RUN) -> str:
    digest = sha256_file(screening_dir(appendix) / WORKBOOK)
    return (f"# as_of={as_of} workbook_sha256={digest} run={run_name} "
            f"identification_run_date={identification_run_date(appendix, run_name)}")


def table_text(appendix: Path, as_of: str, run_name: str = schema.CANONICAL_RUN) -> str:
    appendix = Path(appendix)
    with workbook_lock(appendix, shared=True):  # the digest and the rows come from the same workbook bytes
        rows = compute(appendix, as_of, run_name)
        head = header_line(appendix, as_of, run_name)
    return head + "\n" + csv_text(TABLE_COLUMNS, ([r[c] for c in TABLE_COLUMNS] for r in rows))


def write_table(appendix: Path, as_of: str, run_name: str = schema.CANONICAL_RUN) -> Path:
    appendix = Path(appendix)
    text = table_text(appendix, as_of, run_name)
    return write_text_atomic(screening_dir(appendix) / TABLE_FILE, text, appendix)


def read_table(path: Path) -> tuple[dict[str, str], list[dict[str, str]]]:
    """(header fields as_of / workbook_sha256 / run / identification_run_date, rows). Raises ValueError on a bad file."""
    text = Path(path).read_text(encoding="utf-8")
    first, _, rest = text.partition("\n")
    m = HEADER_RE.match(first)
    if not m:
        raise ValueError(f"{Path(path).name}: first line is not a table_c2_screening header: {first[:80]!r}")
    import csv
    import io

    reader = csv.DictReader(io.StringIO(rest, newline=""))
    rows = list(reader)
    if tuple(reader.fieldnames or ()) != TABLE_COLUMNS:
        raise ValueError(f"{Path(path).name}: columns {reader.fieldnames} are not {TABLE_COLUMNS}")
    head = dict(zip(("as_of", "workbook_sha256", "run", "identification_run_date"), m.groups()))
    return head, rows


def as_int(value: str) -> int:
    return int(value) if (value or "").strip() else 0


def summary_line(as_of: str, rows: list[dict[str, str]]) -> str:
    by = {r["stage"]: r for r in rows}
    return (f"count: as_of={as_of} to_screen={by['records_to_screen']['count']} "
            f"screened={by['records_screened']['count']} pending_title={by['records_screened']['pending']} "
            f"sought={by['reports_sought']['count']} assessed={by['reports_assessed']['count']} "
            f"included={by['studies_included']['count']} invariants=ok")


def main(argv: list[str] | None = None) -> int:
    from aac.prisma.workbook import RunFolderChanged

    ap = argparse.ArgumentParser(prog="aac prisma count")
    ap.add_argument("--as-of", required=True, help="the record date YYYY-MM-DD (an argument, never the clock)")
    ap.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        as_of = as_of_date(args.as_of)
        path = write_table(appendix, as_of)
        _, rows = read_table(path)
    except (IdentificationMismatch, InvariantBroken, RunFolderChanged, OutsideScreening, FileNotFoundError,
            ValueError) as exc:
        print(f"aac prisma count: {exc}", file=sys.stderr)
        return 1
    print(summary_line(as_of, rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
