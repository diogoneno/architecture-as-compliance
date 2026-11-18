"""`aac prisma abstracts fetch [--offline]` (PRISMA-03; 6, Pitfalls 9-10).

Abstracts are fetched for the title survivors only: rows whose screen_decision is include (a row excluded as
`duplicate at title level` is never a survivor). Provider chain by key type (06-COVERAGE.md):
- `10.48550/arxiv.<id>` arXiv API `https://export.arxiv.org/api/query?id_list=<id>`, abstract = the entry's
 <summary>; a body with a DOCTYPE or ENTITY declaration is refused before parsing.
- other DOI OpenAlex `https://api.openalex.org/works/doi:<doi>` (abstract rebuilt from
 abstract_inverted_index), then Semantic Scholar
 `https://api.semanticscholar.org/graph/v1/paper/DOI:<doi>?fields=title,abstract`.
- `https://openalex.org/W<n>` OpenAlex `https://api.openalex.org/works/W<n>`.
- normalized-title key OpenAlex `https://api.openalex.org/works?search=<workbook title>&per-page=5`; only a
 result whose normalized title equals the key is accepted (provider
 `openalex-title-match`, matched_id its W id), for the analyst to confirm in the batch.

Every request goes through aac.prisma.httpget.fetch with the store appendix-c/screening/abstracts-raw/ and the
attempts log appendix-c/screening/abstracts-attempts.csv, so a stored response is never fetched again and
`--offline` rebuilds abstracts.csv byte for byte from the store. Outcomes: `abstract`,
`no abstract (<providers tried>)` (a recorded outcome, never an exclusion: the analyst decides such a row from
its title and landing page), `title match not exact`, and the pending outcomes `not fetched (budget)` (an OpenAlex
`Insufficient budget` refusal stops the chain for every remaining key), `not fetched (gave up)` and
`not fetched (offline)`. A re-run resumes the pending rows.

abstracts.csv is rewritten whole, sorted by key. Then, under the exclusive workbook lock, the only workbook cells
written are abstract_source (the provider when an abstract was found, else `none`; left empty while a row is
pending) and abstract_response_sha256 (sha256 of the deciding response body) of the survivors. No decision cell is
ever written here. No API key or mailto reaches a file: httpget builds headers in memory only.

The run line: `abstracts: survivors=<s> fetched=<f> with_text=<t> no_abstract=<n> pending=<p> retries_429=<q>`
(f = responses fetched over the network in this run, n = rows whose outcome is `no abstract (...)` or
`title match not exact`, q = 429 attempts logged in this run). Exit codes: 0 when nothing is pending, 1 when rows
are pending or a write is refused, 2 for usage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from aac.prisma import httpget, schema
from aac.prisma._io import (
    OutsideScreening,
    csv_text,
    default_appendix_dir,
    read_csv,
    screening_dir,
    workbook_lock,
    write_csv_atomic,
)
from aac.prisma.workbook import title_norm

ABSTRACTS_FILE = "abstracts.csv"
RAW_DIR = "abstracts-raw"
ATTEMPTS = "abstracts-attempts.csv"
WORKBOOK = "workbook.csv"
ABSTRACT_COLUMNS: tuple[str, ...] = (
    "key", "provider", "endpoint", "status", "response_sha256", "matched_id", "matched_title", "outcome", "abstract",
)
ARXIV_PREFIX = "10.48550/arxiv."
OPENALEX_W_RE = re.compile(r"^https://openalex\.org/(W\d+)$")
ATOM = "{http://www.w3.org/2005/Atom}"
TITLE_DUP_REASON = "duplicate at title level"
OUTCOME_ABSTRACT = "abstract"
OUTCOME_NOT_EXACT = "title match not exact"
PENDING_PREFIX = "not fetched ("
SEARCH_PER_PAGE = 5


class _Stop(Exception):
    """Internal: the chain for this key cannot continue (budget, gave up, offline)."""

    def __init__(self, outcome: str):
        self.outcome = outcome
        super().__init__(outcome)


@dataclass
class Ctx:
    appendix: Path
    offline: bool = False
    opener: object | None = None
    sleep: Callable[[float], None] | None = None
    clock: Callable[[], float] | None = None
    fetched: int = 0
    budget_hit: bool = False

    @property
    def store(self) -> Path:
        return screening_dir(self.appendix) / RAW_DIR

    @property
    def attempts(self) -> Path:
        return screening_dir(self.appendix) / ATTEMPTS

    def get(self, url: str) -> tuple[dict, bytes]:
        if self.budget_hit and httpget.load_stored(self.store, url) is None:
            raise _Stop("not fetched (budget)")  # the run stopped: only stored responses are still read
        try:
            meta, body, now = httpget.fetch(url, store=self.store, attempts_log=self.attempts,
                                            appendix=self.appendix, offline=self.offline, opener=self.opener,
                                            sleep=self.sleep, clock=self.clock)
        except httpget.BudgetExhausted:
            self.budget_hit = True
            raise _Stop("not fetched (budget)") from None
        except httpget.GaveUp:
            raise _Stop("not fetched (gave up)") from None
        except httpget.OfflineMiss:
            raise _Stop("not fetched (offline)") from None
        if now:
            self.fetched += 1
        return meta, body


@dataclass
class Result:
    rows: list[dict[str, str]] = field(default_factory=list)
    survivors: int = 0
    fetched: int = 0
    retries_429: int = 0
    abstracts_path: Path | None = None
    workbook_changed: int = 0

    @property
    def with_text(self) -> int:
        return sum(1 for r in self.rows if r["outcome"] == OUTCOME_ABSTRACT)

    @property
    def no_abstract(self) -> int:
        return sum(1 for r in self.rows if r["outcome"].startswith("no abstract (") or r["outcome"] == OUTCOME_NOT_EXACT)

    @property
    def pending(self) -> int:
        return sum(1 for r in self.rows if r["outcome"].startswith(PENDING_PREFIX))

    def line(self) -> str:
        return (f"abstracts: survivors={self.survivors} fetched={self.fetched} with_text={self.with_text} "
                f"no_abstract={self.no_abstract} pending={self.pending} retries_429={self.retries_429}")


# ------------------------------------------------------------------ parsing


def survivors(rows: Sequence[dict[str, str]]) -> list[dict[str, str]]:
    """Rows included on title, in key order; a row excluded as a title-level duplicate is never a survivor."""
    out = [r for r in rows
           if (r.get("screen_decision") or "").strip() == "include"
           and (r.get("exclude_reason") or "").strip() != TITLE_DUP_REASON]
    return sorted(out, key=lambda r: r["key"])


def rebuild_inverted_index(index: dict[str, Sequence[int]] | None) -> str:
    """OpenAlex abstract_inverted_index -> text: each word at each of its positions, joined by single spaces."""
    if not index:
        return ""
    placed: dict[int, str] = {}
    for word, positions in index.items():
        for p in positions or ():
            placed[int(p)] = word
    return " ".join(placed[p] for p in sorted(placed))


def _clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _json(body: bytes) -> object | None:
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def _wid(openalex_id: str | None) -> str:
    m = re.search(r"(W\d+)$", openalex_id or "")
    return m.group(1) if m else ""


def parse_arxiv(body: bytes) -> tuple[str, str, str] | None:
    """(arXiv id, title, summary) of the first entry, or None when the feed holds none. DOCTYPE/ENTITY refused."""
    head = body.decode("utf-8", errors="replace").upper()
    if "<!DOCTYPE" in head or "<!ENTITY" in head:
        raise ValueError("arXiv body carries a DOCTYPE or ENTITY declaration: refused")
    root = ET.fromstring(body)
    for entry in root.findall(f"{ATOM}entry"):
        eid = entry.findtext(f"{ATOM}id") or ""
        if "api/errors" in eid:
            continue
        aid = eid.rsplit("/abs/", 1)[-1] if "/abs/" in eid else eid
        return _clean(aid), _clean(entry.findtext(f"{ATOM}title")), _clean(entry.findtext(f"{ATOM}summary"))
    return None


def key_type(key: str) -> str:
    """arxiv | doi | openalex | title."""
    k = key.strip()
    if k.lower().startswith(ARXIV_PREFIX):
        return "arxiv"
    if k.startswith("10."):
        return "doi"
    if OPENALEX_W_RE.match(k):
        return "openalex"
    return "title"


def _quote_doi(doi: str) -> str:
    return urllib.parse.quote(doi, safe="/:;()")


# ------------------------------------------------------------------ provider chain


def _base(key: str, provider: str, url: str, meta: dict) -> dict[str, str]:
    r = {c: "" for c in ABSTRACT_COLUMNS}
    r.update(key=key, provider=provider, endpoint=url, status=str(meta.get("status", "")),
             response_sha256=meta.get("body_sha256", ""))
    return r


def _openalex_work(key: str, url: str, ctx: Ctx, provider: str = "openalex") -> dict[str, str]:
    meta, body = ctx.get(url)
    row = _base(key, provider, url, meta)
    data = _json(body) if meta.get("status") == 200 else None
    if isinstance(data, dict):
        row.update(matched_id=_wid(data.get("id")), matched_title=_clean(data.get("display_name") or data.get("title")))
        text = _clean(rebuild_inverted_index(data.get("abstract_inverted_index")))
        if text:
            row.update(outcome=OUTCOME_ABSTRACT, abstract=text)
    return row


def _s2_paper(key: str, doi: str, ctx: Ctx) -> dict[str, str]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{_quote_doi(doi)}?fields=title,abstract"
    meta, body = ctx.get(url)
    row = _base(key, "s2", url, meta)
    data = _json(body) if meta.get("status") == 200 else None
    if isinstance(data, dict):
        row.update(matched_id=str(data.get("paperId") or ""), matched_title=_clean(data.get("title")))
        text = _clean(data.get("abstract"))
        if text:
            row.update(outcome=OUTCOME_ABSTRACT, abstract=text)
    return row


def _arxiv(key: str, ctx: Ctx) -> dict[str, str]:
    aid = key.strip()[len(ARXIV_PREFIX):]
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(aid, safe='./')}"
    meta, body = ctx.get(url)
    row = _base(key, "arxiv", url, meta)
    if meta.get("status") == 200:
        try:
            parsed = parse_arxiv(body)
        except (ValueError, ET.ParseError):
            parsed = None
        if parsed is not None:
            mid, title, summary = parsed
            row.update(matched_id=mid, matched_title=title)
            if summary:
                row.update(outcome=OUTCOME_ABSTRACT, abstract=summary)
    return row


def _title_match(key: str, title: str, ctx: Ctx) -> dict[str, str]:
    url = (f"https://api.openalex.org/works?search={urllib.parse.quote(title, safe='')}"
           f"&per-page={SEARCH_PER_PAGE}")
    meta, body = ctx.get(url)
    row = _base(key, "openalex-title-match", url, meta)
    data = _json(body) if meta.get("status") == 200 else None
    results = data.get("results") if isinstance(data, dict) else None
    hit = None
    for res in results if isinstance(results, list) else []:
        if isinstance(res, dict) and title_norm(res.get("display_name") or res.get("title")) == key:
            hit = res
            break
    if hit is None:
        row["outcome"] = OUTCOME_NOT_EXACT
        return row
    row.update(matched_id=_wid(hit.get("id")), matched_title=_clean(hit.get("display_name") or hit.get("title")))
    text = _clean(rebuild_inverted_index(hit.get("abstract_inverted_index")))
    if text:
        row.update(outcome=OUTCOME_ABSTRACT, abstract=text)
    return row


def chain(wbrow: dict[str, str], ctx: Ctx) -> dict[str, str]:
    """The abstracts.csv row of one survivor."""
    key = wbrow["key"]
    kind = key_type(key)
    tried: list[str] = []
    row: dict[str, str] | None = None
    try:
        if kind == "arxiv":
            steps = [lambda: _arxiv(key, ctx)]
        elif kind == "doi":
            doi = key.strip()
            steps = [lambda: _openalex_work(key, f"https://api.openalex.org/works/doi:{_quote_doi(doi)}", ctx),
                     lambda: _s2_paper(key, doi, ctx)]
        elif kind == "openalex":
            wid = OPENALEX_W_RE.match(key.strip()).group(1)
            steps = [lambda: _openalex_work(key, f"https://api.openalex.org/works/{wid}", ctx)]
        else:
            steps = [lambda: _title_match(key, wbrow.get("title") or "", ctx)]
        for step in steps:
            row = step()
            tried.append(row["provider"])
            if row["outcome"] in (OUTCOME_ABSTRACT, OUTCOME_NOT_EXACT):
                return row
    except _Stop as stop:
        pending = {c: "" for c in ABSTRACT_COLUMNS}
        pending.update(key=key, outcome=stop.outcome)
        return pending
    assert row is not None
    row["outcome"] = f"no abstract ({', '.join(tried)})"
    return row


# ------------------------------------------------------------------ run


def _count_429(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(1 for r in read_csv(path)[1] if r.get("status") == "429")


def _read_workbook(appendix: Path) -> list[dict[str, str]]:
    header, rows = read_csv(screening_dir(appendix) / WORKBOOK)
    if tuple(header) != schema.WORKBOOK_COLUMNS:
        raise RuntimeError(f"{WORKBOOK} header is not the schema's 26 columns")
    return rows


def read_abstracts(appendix: Path) -> dict[str, dict[str, str]]:
    """{key: abstracts.csv row}; {} when the file is absent."""
    path = screening_dir(appendix) / ABSTRACTS_FILE
    if not path.is_file():
        return {}
    header, rows = read_csv(path)
    if tuple(header) != ABSTRACT_COLUMNS:
        raise RuntimeError(f"{ABSTRACTS_FILE}: unexpected header {header}")
    return {r["key"]: r for r in rows}


def fetch_abstracts(appendix: Path, *, offline: bool = False, opener=None, sleep=None, clock=None) -> Result:
    appendix = Path(appendix)
    scr = screening_dir(appendix)
    with workbook_lock(appendix, shared=True):
        surv = survivors(_read_workbook(appendix))
    ctx = Ctx(appendix=appendix, offline=offline, opener=opener, sleep=sleep, clock=clock)
    before_429 = _count_429(ctx.attempts)
    rows = [chain(r, ctx) for r in surv]
    res = Result(rows=rows, survivors=len(surv), fetched=ctx.fetched, retries_429=_count_429(ctx.attempts) - before_429)
    path = scr / ABSTRACTS_FILE
    text = csv_text(ABSTRACT_COLUMNS, ([r[c] for c in ABSTRACT_COLUMNS] for r in rows))
    if not path.is_file() or path.read_text(encoding="utf-8") != text:
        write_csv_atomic(path, ABSTRACT_COLUMNS, ([r[c] for c in ABSTRACT_COLUMNS] for r in rows), appendix)
    res.abstracts_path = path
    cells = {}
    for r in rows:
        if r["outcome"].startswith(PENDING_PREFIX):
            continue
        cells[r["key"]] = {"abstract_source": r["provider"] if r["outcome"] == OUTCOME_ABSTRACT else "none",
                           "abstract_response_sha256": r["response_sha256"]}
    if cells:
        with workbook_lock(appendix, shared=False):
            wb = _read_workbook(appendix)  # re-read: a merge may have run while the chain fetched
            changed = 0
            for w in wb:
                new = cells.get(w["key"])
                if new and any(w[c] != v for c, v in new.items()):
                    w.update(new)
                    changed += 1
            if changed:
                write_csv_atomic(scr / WORKBOOK, schema.WORKBOOK_COLUMNS,
                                 ([w[c] for c in schema.WORKBOOK_COLUMNS] for w in wb), appendix)
            res.workbook_changed = changed
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma abstracts")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("fetch", help="fetch (or --offline rebuild) the abstracts of the title survivors")
    p.add_argument("--offline", action="store_true", help="answer every request from abstracts-raw/ only")
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        res = fetch_abstracts(appendix, offline=args.offline)
    except (OutsideScreening, FileNotFoundError, RuntimeError, httpget.NotHttps) as exc:
        print(f"aac prisma abstracts {args.sub}: {exc}", file=sys.stderr)
        return 1
    print(res.line())
    return 1 if res.pending else 0


if __name__ == "__main__":
    sys.exit(main())
