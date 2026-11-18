"""`aac prisma consistency build-set|check` (PRISMA-05; 7, operational constraint, Open Q1).

build-set writes appendix-c/screening/chapter2-comparison-set.csv: one row per reference of draft v0.5 with
- route: from appendix-c/ref-route-v05.csv (read-only), matched by key, else by the citation's named surnames and
 year; `not-in-ref-route` for references the route file does not list (the v0.5 additions);
- sheet_key: the workbook key equal to the reference's identifier (arXiv ids as 10.48550/arxiv.<id>), else the
 workbook row whose normalized title equals the reference title (sheet_match `title`), else empty. So Cui et al.
 (2025), whose ACM DOI is not in the sheet, maps to its preprint row 10.48550/arxiv.2508.03092;
- cited_in_chapter2_heuristic: surname and year within 60 characters in Chapter 2 (a heuristic, labeled so).

check --set NAME compares the workbook's included_in_chapter2 marks with one comparison set:
- database-route: the references whose route is `database` (Q1 default; PHASE6-ONE-SHOT Q1);
- cited-in-chapter2: the references the heuristic finds in Chapter 2;
- as-written: all references (the "46 + 7" reading), with the reasons it cannot hold.
It lists (a) workbook rows marked yes outside the set, (b) set members in the sheet not marked yes, with their
furthest stage and decision, and (c) set members outside the sheet, separately and not counted. It writes
appendix-c/screening/consistency-differences.md (deterministic, no clock), prints
`consistency: set=<name> members=<m> differences=<n> (a=<x> b=<y>) outside_sheet=<z>` and exits 1 while any
(a) or (b) difference exists, 0 when none, 2 on error. It reads the workbook under the shared lock and never
writes it: PRISMA-05 is never marked satisfied by this script; the analyst resolves every difference.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Sequence

from aac.prisma._io import (
    OutsideScreening,
    default_appendix_dir,
    read_csv,
    screening_dir,
    workbook_lock,
    write_csv_atomic,
    write_text_atomic,
)
from aac.prisma.references import Reference, chapter2_text, cited_in, parse_references
from aac.prisma.workbook import title_norm

SETS: tuple[str, ...] = ("database-route", "cited-in-chapter2", "as-written")
SET_COLUMNS: tuple[str, ...] = ("reference", "first_author", "year", "identifier", "route", "sheet_key",
                                "sheet_match", "cited_in_chapter2_heuristic")
SET_FILE = "chapter2-comparison-set.csv"
DIFF_FILE = "consistency-differences.md"
ROUTE_FILE = "ref-route-v05.csv"
NOT_IN_ROUTE = "not-in-ref-route"
SET_DESCRIPTIONS = {
    "database-route": "references whose route in ref-route-v05.csv is `database` (06-RESEARCH Open Question 1 "
                      "default; PHASE6-ONE-SHOT Q1)",
    "cited-in-chapter2": "references the surname-and-year heuristic finds in Chapter 2 (heuristic; analyst to confirm)",
    "as-written": "all references of draft v0.5 (the \"46 + 7\" reading of PRISMA-05)",
}


class ConsistencyError(RuntimeError):
    """A missing or malformed input; exit 2."""


def _sheet_id(ref: Reference) -> str:
    if ref.identifier_type == "doi":
        return ref.identifier.lower()
    if ref.identifier_type == "arxiv":
        return f"10.48550/arxiv.{ref.identifier.lower()}"
    return ""


def _route_names(citation: str) -> tuple[list[str], str]:
    m = re.match(r"^(.*?)\s*\((\d{4}[a-z]?)\)\s*$", citation.strip())
    if not m:
        raise ConsistencyError(f"{ROUTE_FILE}: cannot read the citation {citation!r}")
    names = re.sub(r"\s+et al\.?$", "", m.group(1).strip())
    return [n.strip() for n in re.split(r",\s*|\s+and\s+", names) if n.strip()], m.group(2)


def _match_routes(refs: list[Reference], route_rows: list[dict[str, str]]) -> dict[int, dict[str, str]]:
    """{reference index: route row}; by key first, then by first surname + year + every named surname."""
    out: dict[int, dict[str, str]] = {}
    used: set[int] = set()
    by_id = {(_sheet_id(r) or ""): r for r in refs if _sheet_id(r)}
    pending: list[tuple[int, dict[str, str]]] = []
    for i, row in enumerate(route_rows):
        key = (row.get("key") or "").strip().lower()
        if key and key in by_id and by_id[key].index not in out:
            out[by_id[key].index] = row
            used.add(i)
        else:
            pending.append((i, row))
    for i, row in pending:
        names, year = _route_names(row["reference"])
        cands = [r for r in refs if r.index not in out and r.year == year and r.first_author == names[0]
                 and all(n in r.label for n in names)]
        if len(cands) == 1:
            out[cands[0].index] = row
            used.add(i)
        elif len(cands) > 1:
            raise ConsistencyError(f"{ROUTE_FILE}: {row['reference']!r} matches {len(cands)} references")
    missing = [route_rows[i]["reference"] for i in range(len(route_rows)) if i not in used]
    if missing:
        raise ConsistencyError(f"{ROUTE_FILE}: rows matching no reference of the draft: {missing}")
    return out


def _workbook(appendix: Path) -> list[dict[str, str]]:
    wb = screening_dir(appendix) / "workbook.csv"
    if not wb.is_file():
        raise ConsistencyError(f"{wb} not found (run `aac prisma workbook init` first)")
    with workbook_lock(appendix, shared=True):
        _, rows = read_csv(wb)
    return rows


def build_set(appendix: Path, draft: Path | None = None, route_file: Path | None = None) -> dict[str, int]:
    refs = parse_references(draft)
    rpath = Path(route_file) if route_file else Path(appendix) / ROUTE_FILE
    if not rpath.is_file():
        raise ConsistencyError(f"{rpath} not found")
    header, route_rows = read_csv(rpath)
    if header[:3] != ["reference", "key", "route"]:
        raise ConsistencyError(f"{rpath}: header must start reference,key,route")
    routes = _match_routes(refs, route_rows)
    rows = _workbook(appendix)
    keys = {r["key"] for r in rows}
    by_title: dict[str, list[str]] = {}
    for r in rows:
        t = title_norm(r["title"])
        if t:
            by_title.setdefault(t, []).append(r["key"])
    ch2 = chapter2_text(draft)
    out_rows: list[list[str]] = []
    counts = dict(refs=len(refs), database_route=0, in_sheet=0, cited_in_chapter2=0, not_in_ref_route=0)
    for ref in refs:
        route = routes[ref.index]["route"] if ref.index in routes else NOT_IN_ROUTE
        sid = _sheet_id(ref)
        if sid and sid in keys:
            sheet_key, how = sid, "identifier"
        else:
            hits = sorted(by_title.get(title_norm(ref.title), []))
            sheet_key, how = (hits[0], "title") if len(hits) == 1 else (
                (hits[0], f"title ({len(hits)} rows)") if hits else ("", ""))
        cited = cited_in(ch2, ref)
        counts["database_route"] += route == "database"
        counts["in_sheet"] += bool(sheet_key)
        counts["cited_in_chapter2"] += cited
        counts["not_in_ref_route"] += route == NOT_IN_ROUTE
        out_rows.append([ref.label, ref.first_author, ref.year, ref.identifier, route, sheet_key, how,
                         "yes" if cited else "no"])
    write_csv_atomic(screening_dir(appendix) / SET_FILE, SET_COLUMNS, out_rows, appendix)
    return counts


def build_line(c: dict[str, int]) -> str:
    return (f"comparison-set: refs={c['refs']} database_route={c['database_route']} in_sheet={c['in_sheet']} "
            f"cited_in_chapter2={c['cited_in_chapter2']} (heuristic)")


def _stage(row: dict[str, str]) -> str:
    """The member's furthest recorded stage and decision."""
    if row.get("fulltext_decision"):
        reason = f" ({row['fulltext_reason']})" if row.get("fulltext_reason") else ""
        return f"full text: {row['fulltext_decision']}{reason}"
    if row.get("retrieved"):
        return f"retrieval: retrieved={row['retrieved']}"
    if row.get("abstract_decision"):
        reason = f" ({row['abstract_reason']})" if row.get("abstract_reason") else ""
        return f"abstract: {row['abstract_decision']}{reason}"
    if row.get("screen_decision"):
        reason = f" ({row['exclude_reason']})" if row.get("exclude_reason") else ""
        return f"title: {row['screen_decision']}{reason}"
    return "title: pending"


def _load_set(appendix: Path) -> list[dict[str, str]]:
    p = screening_dir(appendix) / SET_FILE
    if not p.is_file():
        raise ConsistencyError(f"{p} not found (run `aac prisma consistency build-set` first)")
    header, rows = read_csv(p)
    if tuple(header) != SET_COLUMNS:
        raise ConsistencyError(f"{p}: header must be {','.join(SET_COLUMNS)}")
    return rows


def _md(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def check(appendix: Path, set_name: str) -> tuple[int, str]:
    if set_name not in SETS:
        raise ConsistencyError(f"--set must be one of {', '.join(SETS)}")
    cset = _load_set(appendix)
    if set_name == "database-route":
        members = [r for r in cset if r["route"] == "database"]
    elif set_name == "cited-in-chapter2":
        members = [r for r in cset if r["cited_in_chapter2_heuristic"] == "yes"]
    else:
        members = list(cset)
    rows = _workbook(appendix)
    by_key = {r["key"]: r for r in rows}
    member_keys = {m["sheet_key"]: m for m in members if m["sheet_key"]}
    for k in member_keys:
        if k not in by_key:
            raise ConsistencyError(f"{SET_FILE}: sheet_key {k!r} is not a workbook key (rebuild the set)")
    # same-title rows of members (preprint and published records of one work), a hint in (a) only
    title_of_member = {title_norm(by_key[k]["title"]): m for k, m in member_keys.items()}
    a = [r for r in rows if r.get("included_in_chapter2", "").strip().lower() == "yes" and r["key"] not in member_keys]
    b = [(m, by_key[m["sheet_key"]]) for m in members if m["sheet_key"]
         and by_key[m["sheet_key"]].get("included_in_chapter2", "").strip().lower() != "yes"]
    c = [m for m in members if not m["sheet_key"]]
    n = len(a) + len(b)
    line = (f"consistency: set={set_name} members={len(members)} differences={n} (a={len(a)} b={len(b)}) "
            f"outside_sheet={len(c)}")
    out = ["# PRISMA-05 consistency differences", "", line, "",
           "Generated by `aac prisma consistency check`; the workbook is read, never written. Every difference",
           "is listed with its reason; nothing is auto-corrected.", "",
           f"Set: {set_name}: {SET_DESCRIPTIONS[set_name]}.",
           f"Members: {len(members)} (from appendix-c/screening/{SET_FILE}).", "",
           "## (a) Workbook rows marked included_in_chapter2 = yes outside the set", ""]
    if not a:
        out.append("none")
    for r in sorted(a, key=lambda r: r["key"]):
        hint = ""
        m = title_of_member.get(title_norm(r["title"]))
        if m is not None:
            hint = f"; same normalized title as set member `{m['sheet_key']}` ({_md(m['reference'])})"
        out.append(f"- `{_md(r['key'])}` {_md(r['title'])} ({r['year']}): marked yes, not a member of {set_name}{hint}")
    out += ["", "## (b) Set members in the sheet not marked included_in_chapter2 = yes", ""]
    if not b:
        out.append("none")
    for m, r in b:
        mark = r.get("included_in_chapter2", "").strip() or "empty"
        out.append(f"- `{_md(m['sheet_key'])}` {_md(m['reference'])}: included_in_chapter2 {mark}; {_stage(r)}"
                   + (f"; sheet row found by {m['sheet_match']}" if m["sheet_match"] != "identifier" else ""))
    out += ["", "## (c) Set members outside the sheet (reported separately, not counted as differences)", ""]
    if not c:
        out.append("none")
    for m in c:
        out.append(f"- {_md(m['reference'])}: route {m['route']}; no workbook row (identifier `{_md(m['identifier'])}`)")
    if set_name == "as-written":
        total = len(cset)
        in_sheet = sum(1 for r in cset if r["sheet_key"])
        nir = sum(1 for r in cset if r["route"] == NOT_IN_ROUTE)
        cited = sum(1 for r in cset if r["cited_in_chapter2_heuristic"] == "yes")
        db = sum(1 for r in cset if r["route"] == "database")
        out += ["", "## Why the as-written set cannot hold", "",
                f"The \"46 + 7\" reading treats all {total} references as the studies Chapter 2 includes. Only "
                f"{in_sheet} of {total} have a workbook row, so {total - in_sheet} can never be marked "
                f"included_in_chapter2; {db} carry route `database` in ref-route-v05.csv; {nir} are not in "
                f"ref-route-v05.csv at all (the v0.5 additions); the surname-and-year heuristic finds {cited} of "
                f"{total} cited in Chapter 2. All counts are read from appendix-c/screening/{SET_FILE}."]
    out += ["", "PRISMA-05 is not marked satisfied by this file. The analyst resolves each difference in the workbook "
            "(this script never edits it), and PHASE6-ONE-SHOT Q1 chooses the comparison set.", ""]
    write_text_atomic(screening_dir(appendix) / DIFF_FILE, "\n".join(out), appendix)
    return (1 if n > 0 else 0), line


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma consistency")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("build-set", help=f"write appendix-c/screening/{SET_FILE}")
    p.add_argument("--draft", default=None)
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("check", help=f"list differences into appendix-c/screening/{DIFF_FILE}")
    p.add_argument("--set", required=True, choices=SETS)
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        if args.sub == "build-set":
            print(build_line(build_set(appendix, Path(args.draft) if args.draft else None)))
            return 0
        code, line = check(appendix, args.set)
        print(line)
        return code
    except (ConsistencyError, OutsideScreening, FileNotFoundError, ValueError) as exc:
        print(f"aac prisma consistency {args.sub}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
