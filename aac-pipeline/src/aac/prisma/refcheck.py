"""`aac prisma refcheck run|cell|record` (PRISMA-06; 8, operational constraint).

Per-reference verification of the 53 references of draft v0.5, routed by identifier type:
- DOI: Crossref `https://api.crossref.org/works/<doi>` (title[0], author[0].family, years of issued,
 published-online, published-print, and the journal issue's published-online and published-print recorded as
 issue-online and issue-print: the issue date, methodological note
 Crossref 404, DataCite `https://api.datacite.org/dois/<doi>`.
 A `10.48550/arxiv.*` DOI routes to the arXiv API (arXiv is the primary record of those DOIs).
- arXiv: `https://export.arxiv.org/api/query?id_list=<id>` (title, first author's name, years of v1 =
 published and latest = updated); a body with a DOCTYPE or ENTITY declaration is refused before parsing.
- URL: page fetch (first 256 KiB): final URL, status and the first <title>; no author or year can be compared.
 A 2xx with an empty body or a bot-check page (final URL path /challenge, or a "Verifying your browser" style
 title) is `not resolved`: the cited page was not seen, so it is never counted as a recorded page title.
 An `http://` reference URL is fetched as `https://` (HTTPS only) and the method says so.
- none (the 4 books): a row only from appendix-c/screening/refcheck-analyst.csv (`record`, verbatim words).

Match rules: titles NFKC + casefold + non-word characters removed (HTML tags dropped first), equal or one
contains the other; surnames NFKD with combining marks removed plus a small letter map (ł, ø, ß, ...); the year
matches ANY published date and the matching field is recorded. A mismatch stays a mismatch until a new dated
analyst row resolves it (the analyst row follows the machine row; a reference counts by its last row).

A row exists only for a check that executed (a stored response or an analyst record): a connection that never
answers gives an attempts row and no verification row, and the run line counts it as not checked. Every row's
date is the UTC date of the deciding response's HTTP Date header (analyst rows: the analyst's date), never
the local clock. Every response is in appendix-c/screening/refcheck-raw/ and `run --offline` rebuilds the CSV
byte for byte from it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import email.utils
import html
import json
import re
import sys
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from aac.prisma import httpget
from aac.prisma._io import (
    OutsideScreening,
    default_appendix_dir,
    read_csv,
    screening_dir,
    write_csv_atomic,
)
from aac.prisma.references import Reference, default_draft, parse_references

LOG_COLUMNS: tuple[str, ...] = (
    "reference", "url", "method", "check", "date",
    "identifier_type", "identifier", "resolver", "status",
    "title_match", "author_match", "year_match", "year_matched_field",
    "seen_title", "seen_first_author", "seen_years", "final_url", "response_sha256",
)
OPERATOR_COLUMNS: tuple[str, ...] = ("reference", "check", "method", "date", "words")
RAW_DIR = "refcheck-raw"
ATTEMPTS = "refcheck-attempts.csv"
ANALYST_RECORD = "refcheck-analyst.csv"
LOG_RE = re.compile(r"^reference-verification-(\d{4}-\d{2}-\d{2})\.csv$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SAMPLE_TYPES = ("doi", "arxiv", "url")
ATOM = "{http://www.w3.org/2005/Atom}"
CROSSREF_DATE_FIELDS = ("issued", "published-online", "published-print")
# The journal issue's own dates (Crossref `journal-issue`): an article online in December sits in the next
# year's volume and issue, and its citation year follows the volume (methodological note
# applied to every DOI reference; it can add a year match, never remove one.
CROSSREF_ISSUE_DATE_FIELDS = (("published-online", "issue-online"), ("published-print", "issue-print"))
# Letters NFKD does not decompose (they are not base + combining mark).
LETTER_MAP = str.maketrans({"ł": "l", "Ł": "L", "ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ß": "ss",
                            "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE", "ı": "i", "þ": "th", "ð": "d"})


class RefcheckRefused(RuntimeError):
    """A run that must not write (date argument inconsistent with the responses, bad arguments)."""


# ----------------------------------------------------------------------------------------------- matching
def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", " ", s or "")


def norm_title(t: str | None) -> str:
    return re.sub(r"\W", "", unicodedata.normalize("NFKC", html.unescape(_strip_tags(t or ""))).casefold())


def titles_match(a: str | None, b: str | None) -> bool:
    x, y = norm_title(a), norm_title(b)
    return bool(x) and bool(y) and (x == y or x in y or y in x)


def norm_name(n: str | None) -> str:
    s = unicodedata.normalize("NFKD", (n or "").translate(LETTER_MAP))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"\W", "", s.casefold())


def names_match(ref_surname: str, seen: str | None) -> bool:
    """The reference surname equals the seen family name, or ends the seen full name (arXiv gives full names)."""
    r, s = norm_name(ref_surname), norm_name(seen)
    return bool(r) and bool(s) and (r == s or s.endswith(r))


def _verdict(tm: bool, am: bool, ym: bool) -> str:
    bad = [n for n, ok in (("title", tm), ("author", am), ("year", ym)) if not ok]
    return "match" if not bad else "mismatch: " + ", ".join(bad)


def _yn(b: bool) -> str:
    return "yes" if b else "no"


def http_date_utc(value: str) -> str:
    """The UTC calendar date of an HTTP Date header ('' when absent or unparsable)."""
    if not value:
        return ""
    try:
        d = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return ""
    if d.tzinfo is None:
        d = d.replace(tzinfo=_dt.timezone.utc)
    return d.astimezone(_dt.timezone.utc).date().isoformat()


# ------------------------------------------------------------------------------------------------ context
@dataclass
class Ctx:
    appendix: Path
    offline: bool = False
    opener: object | None = None
    sleep: Callable[[float], None] | None = None
    clock: Callable[[], float] | None = None
    fetched_dates: list[str] = field(default_factory=list)

    @property
    def store(self) -> Path:
        return screening_dir(self.appendix) / RAW_DIR

    @property
    def attempts(self) -> Path:
        return screening_dir(self.appendix) / ATTEMPTS

    def get(self, url: str, max_bytes: int = httpget.API_MAX_BYTES) -> tuple[dict, bytes]:
        meta, body, now = httpget.fetch(url, store=self.store, attempts_log=self.attempts, appendix=self.appendix,
                                        offline=self.offline, max_bytes=max_bytes, opener=self.opener,
                                        sleep=self.sleep, clock=self.clock)
        if now:
            self.fetched_dates.append(meta.get("http_date", ""))
        return meta, body


def _row(ref: Reference, **kw: str) -> dict[str, str]:
    r = {c: "" for c in LOG_COLUMNS}
    r.update(reference=ref.label, url=ref.url, identifier_type=ref.identifier_type, identifier=ref.identifier)
    r.update({k: ("" if v is None else str(v)) for k, v in kw.items()})
    return r


def _base(meta: dict) -> dict[str, str]:
    return dict(status=str(meta["status"]), date=http_date_utc(meta.get("http_date", "")),
                final_url=meta.get("final_url", ""), response_sha256=meta["body_sha256"])


def _ok(meta: dict) -> bool:
    return 200 <= int(meta["status"]) < 300


# ---------------------------------------------------------------------------------------------- resolvers
def _years_row(ref: Reference, seen_title: str, seen_author: str, years: list[tuple[str, int]]) -> dict[str, str]:
    tm = titles_match(ref.title, seen_title)
    am = names_match(ref.first_author, seen_author)
    hit = next((f for f, y in years if y == ref.year_int), "")
    return dict(title_match=_yn(tm), author_match=_yn(am), year_match=_yn(bool(hit)), year_matched_field=hit,
                seen_title=re.sub(r"\s+", " ", html.unescape(_strip_tags(seen_title))).strip(),
                seen_first_author=seen_author, seen_years=";".join(f"{f}={y}" for f, y in years),
                check=_verdict(tm, am, bool(hit)))


def _crossref_fields(body: bytes) -> tuple[str, str, list[tuple[str, int]]]:
    msg = json.loads(body.decode("utf-8"))["message"]
    title = (msg.get("title") or [""])[0]
    authors = msg.get("author") or []
    first = authors[0] if authors else {}
    surname = first.get("family") or first.get("name") or ""
    years: list[tuple[str, int]] = []
    for f in CROSSREF_DATE_FIELDS:
        parts = ((msg.get(f) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0] is not None:
            years.append((f, int(parts[0])))
    issue = msg.get("journal-issue") or {}
    for f, label in CROSSREF_ISSUE_DATE_FIELDS:
        parts = ((issue.get(f) or {}).get("date-parts") or [[None]])[0]
        if parts and parts[0] is not None:
            years.append((label, int(parts[0])))
    return title, surname, years


def _datacite_fields(body: bytes) -> tuple[str, str, list[tuple[str, int]]]:
    attrs = json.loads(body.decode("utf-8"))["data"]["attributes"]
    title = ((attrs.get("titles") or [{}])[0]).get("title", "")
    creators = attrs.get("creators") or []
    c0 = creators[0] if creators else {}
    surname = c0.get("familyName") or (c0.get("name") or "").split(",")[0].strip()
    py = attrs.get("publicationYear")
    years = [("publicationYear", int(py))] if py not in (None, "") else []
    return title, surname, years


def check_doi(ref: Reference, ctx: Ctx, doi: str | None = None) -> dict[str, str] | None:
    doi = (doi or ref.identifier).lower()
    if doi.startswith("10.48550/arxiv."):
        return check_arxiv(ref, ctx, arxiv_id=doi[len("10.48550/arxiv."):])
    q = urllib.parse.quote(doi, safe="/:;()._-")
    cr_url = f"https://api.crossref.org/works/{q}"
    try:
        meta, body = ctx.get(cr_url)
    except (httpget.GaveUp, httpget.OfflineMiss):
        return None
    if int(meta["status"]) == 404:
        dc_url = f"https://api.datacite.org/dois/{q}"
        try:
            meta, body = ctx.get(dc_url)
        except (httpget.GaveUp, httpget.OfflineMiss):
            return None
        method, resolver, parse = "datacite api (crossref 404)", dc_url, _datacite_fields
    else:
        method, resolver, parse = "crossref api", cr_url, _crossref_fields
    if not _ok(meta):
        return _row(ref, method=method, resolver=resolver, check=f"not resolved (HTTP {meta['status']})", **_base(meta))
    try:
        title, surname, years = parse(body)
    except (ValueError, KeyError, TypeError, IndexError):
        return _row(ref, method=method, resolver=resolver, check=f"not resolved (HTTP {meta['status']}, unparsable)",
                    **_base(meta))
    return _row(ref, method=method, resolver=resolver, **_base(meta), **_years_row(ref, title, surname, years))


def parse_atom(body: bytes) -> tuple[str, str, list[tuple[str, int]]] | None:
    """(title, first author name, [(v1, year), (latest, year)]) or None when the feed holds no entry."""
    head = body.decode("utf-8", errors="replace")
    if "<!DOCTYPE" in head.upper() or "<!ENTITY" in head.upper():
        raise ValueError("arXiv body carries a DOCTYPE or ENTITY declaration: refused")
    root = ET.fromstring(body)
    for entry in root.findall(f"{ATOM}entry"):
        eid = (entry.findtext(f"{ATOM}id") or "")
        if "api/errors" in eid:
            continue
        title = entry.findtext(f"{ATOM}title") or ""
        a0 = entry.find(f"{ATOM}author")
        name = (a0.findtext(f"{ATOM}name") if a0 is not None else "") or ""
        years: list[tuple[str, int]] = []
        for tag, label in (("published", "v1"), ("updated", "latest")):
            v = entry.findtext(f"{ATOM}{tag}") or ""
            if re.match(r"^\d{4}", v):
                years.append((label, int(v[:4])))
        return re.sub(r"\s+", " ", title).strip(), re.sub(r"\s+", " ", name).strip(), years
    return None


def check_arxiv(ref: Reference, ctx: Ctx, arxiv_id: str | None = None) -> dict[str, str] | None:
    aid = arxiv_id or ref.identifier
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid, safe='./')}"
    try:
        meta, body = ctx.get(url)
    except (httpget.GaveUp, httpget.OfflineMiss):
        return None
    method = "arxiv api"
    if not _ok(meta):
        return _row(ref, method=method, resolver=url, check=f"not resolved (HTTP {meta['status']})", **_base(meta))
    try:
        parsed = parse_atom(body)
    except ValueError as exc:
        return _row(ref, method=method, resolver=url, check=f"not resolved (HTTP {meta['status']}, {exc})", **_base(meta))
    except ET.ParseError:
        return _row(ref, method=method, resolver=url, check=f"not resolved (HTTP {meta['status']}, unparsable)",
                    **_base(meta))
    if parsed is None:
        return _row(ref, method=method, resolver=url, check=f"not resolved (HTTP {meta['status']}, no entry)",
                    **_base(meta))
    title, name, years = parsed
    return _row(ref, method=method, resolver=url, **_base(meta), **_years_row(ref, title, name, years))


TITLE_RE = re.compile(rb"<title[^>]*>(.*?)</title\s*>", re.I | re.S)


def page_title(body: bytes, content_type: str = "") -> str:
    m = TITLE_RE.search(body)
    if not m:
        return ""
    cm = re.search(r"charset=([\w.-]+)", content_type or "", re.I)
    enc = cm.group(1) if cm else "utf-8"
    try:
        raw = m.group(1).decode(enc, errors="replace")
    except LookupError:
        raw = m.group(1).decode("utf-8", errors="replace")
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


INTERSTITIAL_TITLE_RE = re.compile(r"^(verifying your browser|just a moment|attention required|checking your browser)",
                                   re.I)


def is_interstitial(final_url: str, title: str) -> bool:
    """A bot-check page reached instead of the cited page (by its final URL path or its title)."""
    path = urllib.parse.urlsplit(final_url or "").path
    return bool(re.search(r"/challenge(?:/|$)", path)) or bool(INTERSTITIAL_TITLE_RE.match(title or ""))


def check_url(ref: Reference, ctx: Ctx) -> dict[str, str] | None:
    url = ref.url
    method = "page fetch"
    if url.startswith("http://"):
        url = "https://" + url[len("http://"):]
        method = "page fetch (http upgraded to https)"
    try:
        meta, body = ctx.get(url, max_bytes=httpget.PAGE_MAX_BYTES)
    except (httpget.GaveUp, httpget.OfflineMiss, httpget.NotHttps):
        return None
    if not _ok(meta):
        return _row(ref, method=method, resolver=url, check=f"not resolved (HTTP {meta['status']})", **_base(meta))
    if not body:
        # e.g. EUR-Lex answers 202 with an empty body (a scripted bot check): nothing of the page was seen.
        return _row(ref, method=method, resolver=url, check=f"not resolved (HTTP {meta['status']}, empty body)",
                    **_base(meta))
    t = page_title(body, meta.get("content_type", ""))
    if is_interstitial(meta.get("final_url", ""), t):
        # A bot-check page is not the cited page: its title is kept for the record but never counted.
        return _row(ref, method=method, resolver=url, check="not resolved (bot check page)", seen_title=t,
                    **_base(meta))
    if not t:
        return _row(ref, method=method, resolver=url, check="page fetched, no title", **_base(meta))
    return _row(ref, method=method, resolver=url, check="page title recorded", seen_title=t,
                title_match=_yn(titles_match(ref.title, t)), **_base(meta))


# ---------------------------------------------------------------------------------------------- analyst
def operator_rows(appendix: Path) -> list[dict[str, str]]:
    p = screening_dir(appendix) / ANALYST_RECORD
    if not p.is_file():
        return []
    header, rows = read_csv(p)
    if tuple(header) != OPERATOR_COLUMNS:
        raise RefcheckRefused(f"{p}: header must be {','.join(OPERATOR_COLUMNS)}")
    return rows


def record(appendix: Path, reference: str, check: str, method: str, date: str, words: str,
           draft: Path | None = None) -> dict[str, str]:
    """Append one analyst record verbatim. The reference must be a label of the draft's list."""
    labels = {r.label for r in parse_references(draft)}
    if reference not in labels:
        raise RefcheckRefused(f"unknown reference label {reference!r} (use the label exactly as in the log)")
    if not DATE_RE.match(date):
        raise RefcheckRefused(f"--date must be YYYY-MM-DD, got {date!r}")
    for name, v in (("check", check), ("method", method), ("words", words)):
        if not v.strip():
            raise RefcheckRefused(f"--{name} is empty")
    rows = operator_rows(appendix)
    row = dict(reference=reference, check=check, method=method, date=date, words=words)
    rows.append(row)
    write_csv_atomic(screening_dir(appendix) / ANALYST_RECORD, OPERATOR_COLUMNS,
                     ([r[c] for c in OPERATOR_COLUMNS] for r in rows), appendix)
    return row


# ---------------------------------------------------------------------------------------------- counting
def classify(check: str) -> str:
    c = check.strip()
    if c == "match" or c.startswith("match"):
        return "match"
    if c.startswith("mismatch"):
        return "mismatch"
    if c == "page title recorded":
        return "page_title"
    return "unresolved"


def tally(rows: list[dict[str, str]], total: int) -> dict[str, int]:
    """Count per reference by its LAST row (an analyst row after a machine row decides)."""
    last: dict[str, dict[str, str]] = {}
    for r in rows:
        last[r["reference"]] = r
    t = dict(rows=len(last), match=0, mismatch=0, page_title=0, unresolved=0)
    for r in last.values():
        t[classify(r["check"])] += 1
    t["not_checked"] = total - len(last)
    t["verified"] = t["match"] + t["page_title"]
    return t


def log_path(appendix: Path, date: str) -> Path:
    return Path(appendix) / f"reference-verification-{date}.csv"


def latest_log(appendix: Path) -> Path | None:
    logs = sorted(p for p in Path(appendix).glob("reference-verification-*.csv") if LOG_RE.match(p.name))
    return logs[-1] if logs else None


# ---------------------------------------------------------------------------------------------------- run
def _count_429(attempts: Path) -> int:
    if not attempts.is_file():
        return 0
    _, rows = read_csv(attempts)
    return sum(1 for r in rows if r.get("status") == "429")


def run(appendix: Path, draft: Path | None, date: str, *, offline: bool = False, sample_types: bool = False,
        only: str | None = None, opener=None, sleep=None, clock=None) -> dict[str, object]:
    if not DATE_RE.match(date or ""):
        raise RefcheckRefused(f"--date must be YYYY-MM-DD, got {date!r}")
    refs = parse_references(draft)
    if only is not None and only not in {r.label for r in refs}:
        raise RefcheckRefused(f"--only: unknown reference label {only!r}")
    selected: set[str]
    if only is not None:
        selected = {only}
    elif sample_types:
        selected = set()
        for t in SAMPLE_TYPES:
            first = next((r for r in refs if r.identifier_type == t), None)
            if first is not None:
                selected.add(first.label)
    else:
        selected = {r.label for r in refs}
    live = Ctx(appendix=Path(appendix), offline=offline, opener=opener, sleep=sleep, clock=clock)
    stored = Ctx(appendix=Path(appendix), offline=True)
    before_429 = _count_429(live.attempts)
    ops = operator_rows(appendix)
    rows: list[dict[str, str]] = []
    for ref in refs:
        ctx = live if ref.label in selected else stored
        machine: dict[str, str] | None = None
        if ref.identifier_type == "doi":
            machine = check_doi(ref, ctx)
        elif ref.identifier_type == "arxiv":
            machine = check_arxiv(ref, ctx)
        elif ref.identifier_type == "url":
            machine = check_url(ref, ctx)
        if machine is not None:
            rows.append(machine)
        for op in ops:
            if op["reference"] == ref.label:
                rows.append(_row(ref, method=f"analyst: {op['method']}", check=op["check"], date=op["date"],
                                 resolver="analyst"))
    # The --date argument names the file; it must agree with the responses fetched in this run.
    want = _dt.date.fromisoformat(date)
    for hd in live.fetched_dates:
        d = http_date_utc(hd)
        if not d:
            continue
        if abs((_dt.date.fromisoformat(d) - want).days) > 1:
            raise RefcheckRefused(f"--date {date} differs by more than one day from a response's Date header ({d}); "
                                  "nothing written (responses are kept in the store)")
    out = log_path(appendix, date)
    write_csv_atomic(out, LOG_COLUMNS, ([r[c] for c in LOG_COLUMNS] for r in rows), appendix)
    t = tally(rows, len(refs))
    t["retries_429"] = _count_429(live.attempts) - before_429
    t["total"] = len(refs)
    t["file"] = out.name
    t["csv_rows"] = len(rows)
    return t


def run_line(t: dict[str, object]) -> str:
    return (f"refcheck: rows={t['rows']} of {t['total']} match={t['match']} mismatch={t['mismatch']} "
            f"page_title={t['page_title']} unresolved={t['unresolved']} not_checked={t['not_checked']} "
            f"retries_429={t['retries_429']}")


def cell(log: Path, draft: Path | None = None) -> list[str]:
    """The two generated lines: `cell: <N> of <M>` and the Table C.1 text, both read from the log file."""
    header, rows = read_csv(Path(log))
    if tuple(header[: len(LOG_COLUMNS)]) != LOG_COLUMNS:
        raise RefcheckRefused(f"{log}: not a reference-verification log (header {header[:5]})")
    total = len(parse_references(draft))
    t = tally(rows, total)
    dates = sorted({r["date"] for r in rows if r["date"]})
    span = "no dated rows" if not dates else (dates[0] if len(dates) == 1 else f"{dates[0]} to {dates[-1]}")
    return [
        f"cell: {t['verified']} of {total}",
        (f"table-c1: {span}: {t['verified']} of {total} references verified (title, first author and year matched "
         f"for DOI and arXiv items, the year against the issued, online, print or issue date; page title "
         f"recorded for web items); {t['mismatch']} mismatches and "
         f"{t['unresolved']} not resolved listed in {Path(log).name}; {t['not_checked']} not checked"),
    ]


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma refcheck")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("run", help="verify the references; write appendix-c/reference-verification-<date>.csv")
    p.add_argument("--date", required=True, help="UTC date YYYY-MM-DD; must agree with the responses' Date headers")
    p.add_argument("--offline", action="store_true", help="rebuild from the raw store only; no network")
    p.add_argument("--sample-types", action="store_true", help="fetch only the first doi, arxiv and url reference")
    p.add_argument("--only", default=None, help="fetch only this reference label")
    p.add_argument("--draft", default=None)
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("cell", help="print the Table C.1 'N of 53' cell from a log file")
    p.add_argument("--file", default=None)
    p.add_argument("--draft", default=None)
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("record", help="append an analyst record (verbatim) to refcheck-analyst.csv")
    p.add_argument("--reference", required=True)
    p.add_argument("--check", required=True)
    p.add_argument("--method", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--words", required=True)
    p.add_argument("--draft", default=None)
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    draft = Path(args.draft) if args.draft else None
    try:
        if args.sub == "run":
            try:
                t = run(appendix, draft, args.date, offline=args.offline, sample_types=args.sample_types,
                        only=args.only)
            except httpget.BudgetExhausted as exc:
                print(f"aac prisma refcheck run: {exc}", file=sys.stderr)
                return 1
            print(f"refcheck: wrote {Path(appendix).name}/{t['file']} ({t['csv_rows']} rows)")
            print(run_line(t))
        elif args.sub == "cell":
            log = Path(args.file) if args.file else latest_log(appendix)
            if log is None:
                print("aac prisma refcheck cell: no reference-verification-YYYY-MM-DD.csv found", file=sys.stderr)
                return 1
            for line in cell(log, draft):
                print(line)
        else:
            row = record(appendix, args.reference, args.check, args.method, args.date, args.words, draft)
            print(f"refcheck: analyst record appended for {row['reference']} ({row['check']}, {row['date']})")
    except (RefcheckRefused, OutsideScreening, FileNotFoundError, ValueError) as exc:
        print(f"aac prisma refcheck {args.sub}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
