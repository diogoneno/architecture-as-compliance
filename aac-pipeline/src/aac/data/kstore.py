"""AVeriTeC knowledge-store streaming reader and leakage audit (, specification item 18).

The four store zips (dev 500 members ``output_dev/<idx>.json``; ``train_0_999.zip`` and
``train_1000_1999.zip`` with bare ``<idx>.json``; ``train_2000_3067.zip`` with ``data_store/train/<idx>.json``)
are read member by member with ``ZipFile.open``; nothing is ever extracted to disk. Each line is one JSON
document with the keys ``claim_id, type, query, url, url2text`` (``url2text`` a list of sentence strings).

Two read paths:
 * ``iter_raw_lines`` yields the parsed line dict; only ``audit_store`` (the per-store counter) reads its
 ``type`` field, and nothing reads ``query``.
 * ``iter_docs`` yields ``StoreDoc(claim_id, doc_idx, url, host, sentences)``: the only document shape any
 index or shortlist may consume (Phase 3 BM25). It carries neither ``type`` nor ``query``.

``run_audit`` audits every labeled claim's store in a process pool (each worker opens its own ZipFile and
returns a result dict) and writes, once each through a temp file and ``os.replace``:
 ks-audit.csv one row per store, sorted by (split rank train=0 dev=1, idx), first line
 ``# config_hash=... data_version=... audit=ks-audit-v1 stores=N domains_sha256=...``
 ks-stats.json aggregates only (the datasheet regenerates from it without the 237 GB of stores)
 ks-audit-timing.json wall and CPU seconds (Table D.2 knowledge-store row); not deterministic by nature
Shares are integer counts divided once and written with ``.6f``; an empty store's shares are ``n/a``.
No store text and no query value is written anywhere: ids, the committed host list's hosts and counts only.
Type values reach ks-stats.json only as aggregate document counts per label (the six TYPES as columns, any
other label-like value under ``other_type_values``; a value that does not look like a label is bucketed).

stdlib + orjson (read path) + aac.run.hashing; never imports torch.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import resource
import sys
import time
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import orjson

from aac.run.hashing import canonical, data_version, dumps_sorted, sha256_bytes, sha256_file

AUDIT_ID = "ks-audit-v1"
SPLIT_RANK = {"train": 0, "dev": 1}
RAW_REL = Path("data/raw/averitec/data")
STORE_REL = Path("data/raw/averitec/data_store/knowledge_store")
EXPECTED_REL = Path("data/raw/MANIFEST.expected.sha256")
CLAIMS_INDEX_REL = Path("data/derived/averitec/claims-index.json")
DOMAINS_REL = Path("configs/factchecker-domains.txt")
STATS_REL = Path("data/derived/averitec/ks-stats.json")
TIMING_REL = Path("data/measured/ks-audit-timing.json")
LOCK_REL = Path("uv.lock")
DEFAULT_ZIPS = ("dev_knowledge_store.zip", "train/train_0_999.zip", "train/train_1000_1999.zip",
                "train/train_2000_3067.zip")
CSV_NAME, STATS_NAME, TIMING_NAME = "ks-audit.csv", "ks-stats.json", "ks-audit-timing.json"

# The six document types observed on the real stores (02-RESEARCH finding 6); anything else is counted as other.
TYPES = ("gold", "question", "question_duplicate", "answer", "gpt_question", "background")
COLUMNS = (
    "claim_id", "split", "idx", "n_docs",
    *(f"n_docs_{t}" for t in TYPES), "n_docs_other_type",
    "n_sentences", "n_unique_sentences", "bytes_uncompressed",
    "n_docs_factchecker_host", "share_factchecker_host",
    "n_docs_url_eq_answer_exact", "n_docs_url_eq_answer_norm", "n_docs_url_eq_answer_norm_gold",
    "n_docs_url_date_parsed", "n_docs_url_date_after_claim",
    "n_docs_filtered", "n_sentences_filtered", "n_lines_unparsable", "claim_date_null",
)
COUNT_COLUMNS = tuple(c for c in COLUMNS if c not in ("claim_id", "split", "idx", "share_factchecker_host"))

# URL-path dates (a proxy for publication date; the after-claim count is a lower bound). Day shapes win;
# a month shape is read only where no day shape follows it, so an invalid day never falls back to its month.
YEAR_MIN, YEAR_MAX = 1990, 2025
DAY_PATTERNS = (r"/(\d{4})/(\d{2})/(\d{2})(?![0-9])", r"/(\d{4})-(\d{2})-(\d{2})(?![0-9])")
MONTH_PATTERNS = (r"/(\d{4})/(\d{2})(?=/|$)(?!/\d{2}(?![0-9]))", r"/(\d{4})-(\d{2})(?=/|$)")
_DAY_RES = tuple(re.compile(p) for p in DAY_PATTERNS)
_MONTH_RES = tuple(re.compile(p) for p in MONTH_PATTERNS)

_ARCHIVE_RE = re.compile(r"^(?:[a-z][a-z0-9+.-]*:)?/*(?:www\.)?web\.archive\.org/web/[^/]*/+(.*)$", re.I | re.S)
_SCHEME_RE = re.compile(r"^(?:https?|ftp):/*", re.I)
_MEMBER_RES = (
    (re.compile(r"^output_dev/(\d+)\.json$"), "dev"),
    (re.compile(r"^data_store/train/(\d+)\.json$"), "train"),
    (re.compile(r"^(\d+)\.json$"), "train"),
)
_TRAIN_ZIP_RE = re.compile(r"^train_(\d+)_(\d+)\.zip$")
_DEV_ZIP_NAME = "dev_knowledge_store.zip"
HOST_LINE_RE = re.compile(r"^[a-z0-9.-]+$")
# A document type outside TYPES is aggregated under its own value only when that value looks like an
# annotation label; anything else is bucketed, so no free text can reach ks-stats.json.
TYPE_LABEL_RE = re.compile(r"^[A-Za-z0-9_+.-]{1,40}$")
UNRECOGNISED_TYPE, MISSING_TYPE = "<unrecognised>", "<missing>"


class StoreLayoutError(ValueError):
    """A store member, line or input file does not have the measured layout; nothing is written."""


# ---------------------------------------------------------------- member layout


def _zip_name(zip_path) -> str:
    if isinstance(zip_path, zipfile.ZipFile):
        return Path(zip_path.filename or "").name
    return Path(zip_path).name


def member_claim_id(zip_name: str, member: str) -> str:
    """(zip file name, member path) -> ``averitec/<split>/<idx>``; refuses unknown layouts and a member that
 contradicts its zip (dev layout in a train zip, train layout in the dev zip, idx outside train_A_B)."""
    for rx, split in _MEMBER_RES:
        m = rx.match(member)
        if m:
            idx = int(m.group(1))
            break
    else:
        raise StoreLayoutError(f"{zip_name}: member {member!r} matches none of the three store layouts")
    name = Path(zip_name).name
    tm = _TRAIN_ZIP_RE.match(name)
    if name == _DEV_ZIP_NAME and split != "dev":
        raise StoreLayoutError(f"{name}: train-layout member {member!r} inside the dev zip")
    if tm:
        lo, hi = int(tm.group(1)), int(tm.group(2))
        if split != "train" or not lo <= idx <= hi:
            raise StoreLayoutError(f"{name}: member {member!r} is not a train store in [{lo}, {hi}]")
    return f"averitec/{split}/{idx}"


def _order(cid: str) -> tuple[int, int]:
    _, split, idx = cid.split("/")
    return SPLIT_RANK[split], int(idx)


def list_members(zip_path) -> list[tuple[str, str]]:
    """[(claim_id, member)] of one zip, directory entries skipped, sorted by (split rank, idx)."""
    zf = zip_path if isinstance(zip_path, zipfile.ZipFile) else zipfile.ZipFile(zip_path)
    try:
        name = _zip_name(zf)
        out = [(member_claim_id(name, i.filename), i.filename) for i in zf.infolist() if not i.is_dir()]
    finally:
        if zf is not zip_path:
            zf.close()
    ids = [c for c, _ in out]
    if len(ids) != len(set(ids)):
        raise StoreLayoutError(f"{name}: two members map to the same claim id")
    return sorted(out, key=lambda cm: _order(cm[0]))


# ---------------------------------------------------------------- line reader


def _parse(line: bytes):
    """orjson first; the stdlib parser only for what orjson refuses (e.g. an escaped lone surrogate).
 Returns the parsed object or None when neither parser accepts the line."""
    try:
        return orjson.loads(line)
    except orjson.JSONDecodeError:
        pass
    try:
        return json.loads(line.decode("utf-8"))
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None


def _valid(rec) -> bool:
    if not isinstance(rec, dict):
        return False
    cid = rec.get("claim_id")
    if isinstance(cid, bool) or not isinstance(cid, (str, int)):
        return False
    sents = rec.get("url2text")
    return isinstance(rec.get("url"), str) and isinstance(sents, list) and all(isinstance(s, str) for s in sents)


def iter_raw_lines(zip_path, member: str, stats: dict | None = None) -> Iterator[tuple[int, dict]]:
    """Yield (physical line number, parsed line dict) of one store member, streamed from the zip.

 Blank lines are skipped. A line that does not parse, or lacks a string url / a list-of-strings url2text /
 a claim_id, is counted in ``stats["n_lines_unparsable"]`` and never evaluated. A well-formed line whose
 claim_id differs from the member's index raises StoreLayoutError (a wrong file, not a bad line)."""
    zf = zip_path if isinstance(zip_path, zipfile.ZipFile) else zipfile.ZipFile(zip_path)
    try:
        expected = member_claim_id(_zip_name(zf), member).rsplit("/", 1)[1]
        with zf.open(member) as fh:
            for n, line in enumerate(fh):
                if not line.strip():
                    continue
                rec = _parse(line)
                if not _valid(rec):
                    if stats is not None:
                        stats["n_lines_unparsable"] = stats.get("n_lines_unparsable", 0) + 1
                    continue
                if str(rec["claim_id"]) != expected:
                    raise StoreLayoutError(f"{member}: line {n} has claim_id {rec['claim_id']!r}, member index {expected}")
                yield n, rec
    finally:
        if zf is not zip_path:
            zf.close()


@dataclass(frozen=True)
class StoreDoc:
    """The only store-document shape an index or shortlist may consume (specification item 18): no type, no query."""
    claim_id: str
    doc_idx: int
    url: str
    host: str
    sentences: tuple[str, ...]


def iter_docs(zip_path, member: str, stats: dict | None = None) -> Iterator[StoreDoc]:
    """StoreDoc per valid line; doc_idx = the physical line number in the member."""
    cid = member_claim_id(_zip_name(zip_path), member)
    for n, rec in iter_raw_lines(zip_path, member, stats):
        yield StoreDoc(cid, n, rec["url"], url_host(rec["url"]), tuple(rec["url2text"]))


# ---------------------------------------------------------------- URLs


def unwrap_archive(url: str) -> str:
    """Remove every leading ``web.archive.org/web/<timestamp>/`` wrapper (scheme optional)."""
    u = url.strip()
    for _ in range(8):
        m = _ARCHIVE_RE.match(u)
        if not m:
            break
        u = m.group(1)
    return u


def _netloc_rest(url: str) -> tuple[str, str]:
    u = unwrap_archive(url).split("#", 1)[0]
    u = _SCHEME_RE.sub("", u).lstrip("/")
    cut = min([i for i in (u.find("/"), u.find("?")) if i >= 0], default=len(u))
    return u[:cut], u[cut:]


def _strip_www(host: str) -> str:
    return host[4:] if host.startswith("www.") else host


def url_host(url: str) -> str:
    """Host of the unwrapped URL: lowercased, user info, port and trailing dot removed, leading www. removed."""
    netloc, _ = _netloc_rest(url)
    host = netloc.rpartition("@")[2].split(":", 1)[0].lower().rstrip(".")
    return _strip_www(host)


def normalise_url(url: str) -> str:
    """Archive wrapper removed, scheme dropped, host lowercased, leading www. dropped, fragment and trailing
 slash stripped (path and query keep their case)."""
    netloc, rest = _netloc_rest(url)
    return (_strip_www(netloc.lower()) + rest).rstrip("/")


def _url_path(url: str) -> str:
    _, rest = _netloc_rest(url)
    return rest.split("?", 1)[0]


def _valid_date(y: str, m: str, d: str = "01") -> date | None:
    if not YEAR_MIN <= int(y) <= YEAR_MAX:
        return None
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return None


def url_path_date(url: str) -> tuple[date, str] | None:
    """(date, "day"|"month") from the URL path (patterns /YYYY/MM/DD/, /YYYY-MM-DD, /YYYY/MM/, /YYYY-MM/;
 years 1990-2025), a month-only date as its first day; None when no valid date is found."""
    path = _url_path(url)
    for rx in _DAY_RES:
        for m in rx.finditer(path):
            d = _valid_date(*m.groups())
            if d:
                return d, "day"
    for rx in _MONTH_RES:
        for m in rx.finditer(path):
            d = _valid_date(*m.groups())
            if d:
                return d, "month"
    return None


def url_date_after(parsed: tuple[date, str] | None, claim_date: date | None) -> bool:
    """Strictly later only; a month-only date counts when its first day is later; undated claims never count."""
    return parsed is not None and claim_date is not None and parsed[0] > claim_date


# ---------------------------------------------------------------- per-store audit


def _share(n: int, d: int) -> str:
    return f"{n / d:.6f}" if d else "n/a"


def _digest(s: str) -> bytes:
    return hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=16).digest()


def audit_store(zip_path, member: str, *, claim_date: date | None, answer_urls, domains: frozenset[str],
                per_host: dict | None = None, other_types: dict | None = None) -> dict:
    """One row (keys in COLUMNS order) for one store member. The only reader of a line's `type` field.
 ``per_host`` (optional) receives {listed host: filtered documents} and ``other_types`` (optional)
 {type value outside TYPES: documents} (label-like values only; else <unrecognised>/<missing>)."""
    cid = member_claim_id(_zip_name(zip_path), member)
    _, split, idx = cid.split("/")
    exact = {u for u in answer_urls if isinstance(u, str) and u}
    norm = {normalise_url(u) for u in exact} - {""}
    c = Counter()
    by_type = Counter()
    uniq: set[bytes] = set()
    stats: dict = {"n_lines_unparsable": 0}
    zf = zip_path if isinstance(zip_path, zipfile.ZipFile) else zipfile.ZipFile(zip_path)
    try:
        size = zf.getinfo(member).file_size
        for _, rec in iter_raw_lines(zf, member, stats):
            c["n_docs"] += 1
            typ = rec.get("type")
            if typ in TYPES:
                by_type[typ] += 1
            else:
                by_type[None] += 1
                if other_types is not None:
                    key = (MISSING_TYPE if typ is None else
                           typ if isinstance(typ, str) and TYPE_LABEL_RE.match(typ) else UNRECOGNISED_TYPE)
                    other_types[key] = other_types.get(key, 0) + 1
            url, sents = rec["url"], rec["url2text"]
            c["n_sentences"] += len(sents)
            uniq.update(_digest(s) for s in sents)
            host = url_host(url)
            if host in domains:
                c["n_docs_factchecker_host"] += 1
                c["n_docs_filtered"] += 1
                c["n_sentences_filtered"] += len(sents)
                if per_host is not None:
                    per_host[host] = per_host.get(host, 0) + 1
            if url in exact:
                c["n_docs_url_eq_answer_exact"] += 1
            if normalise_url(url) in norm:
                c["n_docs_url_eq_answer_norm"] += 1
                if typ == "gold":
                    c["n_docs_url_eq_answer_norm_gold"] += 1
            parsed = url_path_date(url)
            if parsed is not None:
                c["n_docs_url_date_parsed"] += 1
                if url_date_after(parsed, claim_date):
                    c["n_docs_url_date_after_claim"] += 1
    finally:
        if zf is not zip_path:
            zf.close()
    row = {"claim_id": cid, "split": split, "idx": int(idx), "n_docs": c["n_docs"]}
    for t in TYPES:
        row[f"n_docs_{t}"] = by_type[t]
    row["n_docs_other_type"] = by_type[None]
    row["n_sentences"] = c["n_sentences"]
    row["n_unique_sentences"] = len(uniq)
    row["bytes_uncompressed"] = size
    row["n_docs_factchecker_host"] = c["n_docs_factchecker_host"]
    row["share_factchecker_host"] = _share(c["n_docs_factchecker_host"], c["n_docs"])
    for k in ("n_docs_url_eq_answer_exact", "n_docs_url_eq_answer_norm", "n_docs_url_eq_answer_norm_gold",
              "n_docs_url_date_parsed", "n_docs_url_date_after_claim", "n_docs_filtered", "n_sentences_filtered"):
        row[k] = c[k]
    row["n_lines_unparsable"] = stats["n_lines_unparsable"]
    row["claim_date_null"] = int(claim_date is None)
    return row





def _read_list(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise StoreLayoutError(f"{Path(path).name}: top level is not a list")
    return data


def _fca_hosts(paths) -> tuple[list[str], int, int]:
    hosts, n_claims, n_unusable = [], 0, 0
    for p in paths:
        for c in _read_list(p):
            n_claims += 1
            f = c.get("fact_checking_article")
            h = url_host(f) if isinstance(f, str) else ""
            if h:
                hosts.append(h)
            else:
                n_unusable += 1
    return hosts, n_claims, n_unusable


def derive_domains(train_json, dev_json) -> list[str]:
    """Distinct hosts of the archive-unwrapped `fact_checking_article` URLs of train and dev, sorted."""
    return sorted(set(_fca_hosts((train_json, dev_json))[0]))


def write_domains(out, train_json, dev_json) -> list[str]:
    hosts, n_claims, n_unusable = _fca_hosts((train_json, dev_json))
    listed = sorted(set(hosts))
    bad = [h for h in listed if not HOST_LINE_RE.match(h)]
    if bad:
        raise StoreLayoutError(f"derived hosts outside [a-z0-9.-]: {bad}")
    header = [
        "# Pre-registered fact-checker host list (specification item 18).",
        "# Frozen before the knowledge-store audit: any later change is an Appendix F Table F.3 deviation row,",
        "#   never a silent edit.",
        "# Rule: the host of every claim's `fact_checking_article` URL in train.json and dev.json, after removing",
        "#   every leading web.archive.org/web/<timestamp>/ wrapper; host lowercased, user info, port and trailing",
        "#   dot removed, one leading `www.` removed; distinct hosts sorted (aac.data.kstore.derive_domains).",
        "# Match rule (audit and filter): a store document is filtered when the host of its URL, normalized the",
        "#   same way, equals a listed host exactly (no subdomain or suffix matching).",
        f"# Sources: averitec/data/{Path(train_json).name} sha256={sha256_file(Path(train_json))}; "
        f"averitec/data/{Path(dev_json).name} sha256={sha256_file(Path(dev_json))}",
        f"# Claims read: {n_claims}; without a usable fact_checking_article host: {n_unusable}",
        f"# Hosts: {len(listed)}",
    ]
    _write_atomic(Path(out), ("\n".join(header + listed) + "\n").encode("ascii"))
    return listed


def load_domains(path) -> frozenset[str]:
    hosts = []
    for n, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        if not HOST_LINE_RE.match(line):
            raise ValueError(f"{Path(path).name}:{n}: {line!r} is not one lowercase host")
        hosts.append(line)
    if len(hosts) != len(set(hosts)):
        raise ValueError(f"{Path(path).name}: duplicate host lines")
    if not hosts:
        raise ValueError(f"{Path(path).name}: no hosts")
    return frozenset(hosts)


# ---------------------------------------------------------------- whole-corpus audit

_WORKER_DOMAINS: frozenset[str] = frozenset()


def _init_worker(domains: frozenset[str]) -> None:
    global _WORKER_DOMAINS
    _WORKER_DOMAINS = domains


def _task(args) -> tuple[dict, dict, dict, float]:
    zip_path, member, claim_date, answer_urls = args
    t0 = time.process_time()
    per_host: dict = {}
    other: dict = {}
    row = audit_store(zip_path, member, claim_date=date.fromisoformat(claim_date) if claim_date else None,
                      answer_urls=answer_urls, domains=_WORKER_DOMAINS, per_host=per_host, other_types=other)
    return row, per_host, other, time.process_time() - t0


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _check_raw(raw_dir: Path, expected_path: Path) -> None:
    from aac.data.verify import parse_expected

    expected = parse_expected(expected_path)
    for split in SPLIT_RANK:
        rel = f"averitec/data/{split}.json"
        if rel not in expected:
            raise StoreLayoutError(f"{rel} has no expected sha256 line; refusing to audit")
        actual = sha256_file(raw_dir / f"{split}.json")
        if actual != expected[rel]:
            raise StoreLayoutError(f"{rel}: sha256 {actual} != expected {expected[rel]}; refusing to audit")


def _answer_urls(raw_dir: Path) -> dict[str, tuple[str, ...]]:
    out = {}
    for split in SPLIT_RANK:
        for idx, c in enumerate(_read_list(raw_dir / f"{split}.json")):
            urls = {a.get("source_url") for q in (c.get("questions") or []) for a in (q.get("answers") or [])}
            out[f"averitec/{split}/{idx}"] = tuple(sorted(u for u in urls if isinstance(u, str) and u))
    return out


def _rusage_cpu() -> float:
    s, ch = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return s.ru_utime + s.ru_stime + ch.ru_utime + ch.ru_stime


def _csv_bytes(first_line: str, rows: list[dict]) -> bytes:
    buf = io.StringIO()
    buf.write(first_line + "\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    for r in rows:
        w.writerow([r[c] for c in COLUMNS])
    return buf.getvalue().encode("ascii")


def _agg(rows: list[dict]) -> dict:
    tot = {k: sum(r[k] for r in rows) for k in COUNT_COLUMNS}
    return {
        "n_stores": len(rows),
        "n_stores_empty": sum(1 for r in rows if r["n_docs"] == 0),
        "n_docs": tot["n_docs"],
        "n_docs_by_type": {**{t: tot[f"n_docs_{t}"] for t in TYPES}, "other_type": tot["n_docs_other_type"]},
        "n_sentences": tot["n_sentences"],
        "n_unique_sentences_summed_over_stores": tot["n_unique_sentences"],
        "bytes_uncompressed": tot["bytes_uncompressed"],
        "n_lines_unparsable": tot["n_lines_unparsable"],
    }


def build_stats(rows: list[dict], per_host: dict[str, int], domains: frozenset[str], provenance: dict,
                other_types: dict[str, int] | None = None) -> dict:
    """Aggregates only (counts, shares as `.6f` strings of count ratios); no text, no type or query values."""
    by_split = {s: [r for r in rows if r["split"] == s] for s in SPLIT_RANK}
    tot = {k: sum(r[k] for r in rows) for k in COUNT_COLUMNS}
    n_docs, n_sent = tot["n_docs"], tot["n_sentences"]
    return {
        "provenance": provenance,
        "totals": {"train": _agg(by_split["train"]), "dev": _agg(by_split["dev"]), "all": _agg(rows)},
        "type_shares": {**{t: _share(tot[f"n_docs_{t}"], n_docs) for t in TYPES},
                        "other_type": _share(tot["n_docs_other_type"], n_docs)},
        "other_type_values": dict(sorted((other_types or {}).items())),
        "filter": {
            "rule": "specification item 18: drop a document when its URL host (archive-unwrapped, lowercased, www. removed) "
                    "equals a host of configs/factchecker-domains.txt",
            "n_hosts_listed": len(domains),
            "n_docs_filtered": tot["n_docs_filtered"],
            "n_docs_filtered_by_split": {s: sum(r["n_docs_filtered"] for r in v) for s, v in by_split.items()},
            "n_sentences_filtered": tot["n_sentences_filtered"],
            "share_docs_filtered": _share(tot["n_docs_filtered"], n_docs),
            "share_sentences_filtered": _share(tot["n_sentences_filtered"], n_sent),
            "n_stores_with_filtered_docs": sum(1 for r in rows if r["n_docs_filtered"]),
            "per_host": {h: per_host.get(h, 0) for h in sorted(domains)},
        },
        "leakage": {
            "n_docs_factchecker_host": tot["n_docs_factchecker_host"],
            "share_factchecker_host": _share(tot["n_docs_factchecker_host"], n_docs),
            "n_stores_with_factchecker_host": sum(1 for r in rows if r["n_docs_factchecker_host"]),
            "n_docs_url_eq_answer_exact": tot["n_docs_url_eq_answer_exact"],
            "share_url_eq_answer_exact": _share(tot["n_docs_url_eq_answer_exact"], n_docs),
            "n_stores_with_url_eq_answer_exact": sum(1 for r in rows if r["n_docs_url_eq_answer_exact"]),
            "n_docs_url_eq_answer_norm": tot["n_docs_url_eq_answer_norm"],
            "share_url_eq_answer_norm": _share(tot["n_docs_url_eq_answer_norm"], n_docs),
            "n_stores_with_url_eq_answer_norm": sum(1 for r in rows if r["n_docs_url_eq_answer_norm"]),
            "n_docs_url_eq_answer_norm_gold": tot["n_docs_url_eq_answer_norm_gold"],
            "share_gold_docs_url_eq_answer_norm": _share(tot["n_docs_url_eq_answer_norm_gold"], tot["n_docs_gold"]),
        },
        "url_date": {
            "patterns": list(DAY_PATTERNS + MONTH_PATTERNS),
            "year_range": [YEAR_MIN, YEAR_MAX],
            "n_docs_url_date_parsed": tot["n_docs_url_date_parsed"],
            "coverage": _share(tot["n_docs_url_date_parsed"], n_docs),
            "n_docs_url_date_after_claim": tot["n_docs_url_date_after_claim"],
            "share_of_parsed_after_claim": _share(tot["n_docs_url_date_after_claim"], tot["n_docs_url_date_parsed"]),
            "share_of_docs_after_claim": _share(tot["n_docs_url_date_after_claim"], n_docs),
            "after_claim_is_lower_bound": True,
            "n_claims_date_null": tot["claim_date_null"],
            "n_docs_in_undated_stores": sum(r["n_docs"] for r in rows if r["claim_date_null"]),
        },
        "unparsable": {
            "n_lines_unparsable": tot["n_lines_unparsable"],
            "n_stores_with_unparsable": sum(1 for r in rows if r["n_lines_unparsable"]),
        },
    }


def run_audit(root, out_dir, limit: int | None = None, workers: int = 8, *, zips=DEFAULT_ZIPS,
              git_sha: str | None = None, progress=None) -> dict:
    """Audit the stores of every labeled claim (or the first ``limit`` in (split rank, idx) order) and write
 ks-audit.csv, ks-stats.json and ks-audit-timing.json into out_dir. Returns a small summary dict."""
    root = Path(root)
    out_dir = Path(out_dir) if Path(out_dir).is_absolute() else root / out_dir
    raw_dir, store_dir = root / RAW_REL, root / STORE_REL
    expected_path, index_path, domains_path = root / EXPECTED_REL, root / CLAIMS_INDEX_REL, root / DOMAINS_REL
    _check_raw(raw_dir, expected_path)
    answers = _answer_urls(raw_dir)
    index = _read_list(index_path)
    dates = {r["claim_id"]: r["date"] for r in index}
    if set(dates) != set(answers) or len(dates) != len(index):
        raise StoreLayoutError("claims-index.json and the raw train/dev JSON name different claims")
    tasks = []
    for z in zips:
        zp = store_dir / z
        for cid, member in list_members(zp):
            tasks.append((_order(cid), cid, str(zp), member))
    tasks.sort()
    ids = [t[1] for t in tasks]
    if len(ids) != len(set(ids)):
        raise StoreLayoutError("a claim has more than one store across the zips")
    extra = sorted(set(ids) - set(dates), key=_order)
    if extra:
        raise StoreLayoutError(f"{len(extra)} stores name no labeled claim, first {extra[0]}")
    if limit is None:
        missing = sorted(set(dates) - set(ids), key=_order)
        if missing:
            raise StoreLayoutError(f"no store for {len(missing)} labeled claims, first {missing[0]}")
    else:
        tasks = tasks[:limit]
    domains = load_domains(domains_path)
    if git_sha is None:
        from aac.run.clock import git_sha as _git_sha

        git_sha = _git_sha(cwd=root)
    domains_sha = sha256_file(domains_path)
    dv = data_version(expected_path)
    cfg = {"audit": AUDIT_ID, "git_sha": git_sha, "lock_sha256": sha256_file(root / LOCK_REL),
           "factchecker_domains_sha256": domains_sha, "claims_index_sha256": sha256_file(index_path),
           "url_date_patterns": list(DAY_PATTERNS + MONTH_PATTERNS), "url_date_years": [YEAR_MIN, YEAR_MAX],
           "types": list(TYPES), "limit": limit}
    ch = sha256_bytes(canonical(cfg))
    args = [(zp, member, dates[cid], answers[cid]) for _, cid, zp, member in tasks]

    t0, c0 = time.perf_counter(), _rusage_cpu()
    results, per_host, other_types, cpu_workers = [], Counter(), Counter(), 0.0
    say = progress or (lambda msg: None)
    if workers <= 1:
        _init_worker(domains)
        stream = map(_task, args)
        pool = None
    else:
        import multiprocessing as mp

        pool = mp.get_context("spawn").Pool(workers, initializer=_init_worker, initargs=(domains,))
        stream = pool.imap_unordered(_task, args, chunksize=1)
    try:
        for row, ph, ot, cpu in stream:
            results.append(row)
            per_host.update(ph)
            other_types.update(ot)
            cpu_workers += cpu
            if len(results) % 100 == 0 or len(results) == len(args):
                say(f"ks-audit: {len(results)}/{len(args)} stores")
        if pool is not None:
            pool.close()
    except BaseException:
        if pool is not None:
            pool.terminate()
        raise
    finally:
        if pool is not None:
            pool.join()
    wall, cpu = time.perf_counter() - t0, _rusage_cpu() - c0

    rows = sorted(results, key=lambda r: (SPLIT_RANK[r["split"]], r["idx"]))
    first = (f"# config_hash={ch} data_version={dv} audit={AUDIT_ID} stores={len(rows)} "
             f"domains_sha256={domains_sha}")
    provenance = {"audit": AUDIT_ID, "config_hash": ch, "data_version": dv, "domains_sha256": domains_sha,
                  "claims_index_sha256": cfg["claims_index_sha256"], "git_sha": git_sha, "stores": len(rows),
                  "limit": limit}
    stats = build_stats(rows, dict(per_host), domains, provenance, dict(other_types))
    timing = {"audit": AUDIT_ID, "config_hash": ch, "release_sha": git_sha, "stores": len(rows), "workers": workers,
              "wall_seconds": round(wall, 3), "cpu_seconds": round(cpu, 3),
              "cpu_seconds_workers": round(cpu_workers, 3),
              "bytes_uncompressed": sum(r["bytes_uncompressed"] for r in rows),
              "cpu_seconds_source": "resource.getrusage(RUSAGE_SELF + RUSAGE_CHILDREN) around the pool; "
                                    "cpu_seconds_workers = summed per-store time.process_time()"}
    csv_bytes = _csv_bytes(first, rows)
    _write_atomic(out_dir / CSV_NAME, csv_bytes)
    _write_atomic(out_dir / STATS_NAME, dumps_sorted(stats).encode("ascii"))
    _write_atomic(out_dir / TIMING_NAME, dumps_sorted(timing).encode("ascii"))
    return {"stores": len(rows), "config_hash": ch, "csv_sha256": sha256_bytes(csv_bytes), "wall_seconds": wall,
            "cpu_seconds": cpu, "out_dir": out_dir}


# ---------------------------------------------------------------- placing a pulled audit (make ks-pull)

_HEADER_RE = re.compile(r"^# config_hash=([0-9a-f]{64}) data_version=([0-9a-f]{64}) audit=ks-audit-v1 "
                        r"stores=(\d+) domains_sha256=([0-9a-f]{64})$")


def place(from_dir, root, *, log_name: str, log_sha256: str) -> dict[str, Path]:
    """Install a pulled audit: ks-audit.csv -> reports/ (selected namespace), ks-stats.json ->
 data/derived/averitec/, ks-audit-timing.json (+ log name and sha256) -> data/measured/."""
    from aac.run.paths import reports_dir

    from_dir, root = Path(from_dir), Path(root)
    csv_bytes = (from_dir / CSV_NAME).read_bytes()
    first = csv_bytes.split(b"\n", 1)[0].decode("ascii")
    m = _HEADER_RE.match(first)
    if not m:
        raise StoreLayoutError(f"{CSV_NAME}: first line is not the ks-audit-v1 provenance header")
    stats = json.loads((from_dir / STATS_NAME).read_text(encoding="ascii"))
    timing = json.loads((from_dir / TIMING_NAME).read_text(encoding="ascii"))
    if stats["provenance"]["config_hash"] != m.group(1) or timing["config_hash"] != m.group(1):
        raise StoreLayoutError("ks-audit.csv, ks-stats.json and ks-audit-timing.json carry different config_hash")
    n_rows = len(csv_bytes.decode("ascii").splitlines()) - 2
    if n_rows != int(m.group(3)):
        raise StoreLayoutError(f"{CSV_NAME}: {n_rows} data rows, header says stores={m.group(3)}")
    
    # different claim index, would otherwise become reports/ks-audit.csv and the datasheet's ks-stats.json.
    prov = stats["provenance"]
    if prov.get("limit") is not None:
        raise StoreLayoutError(f"{STATS_NAME}: a --limit {prov['limit']} tracer audit; only the full audit is installed")
    index_path = root / CLAIMS_INDEX_REL
    n_claims = len(_read_list(index_path))
    if int(m.group(3)) != n_claims or int(prov.get("stores", -1)) != n_claims:
        raise StoreLayoutError(f"{CSV_NAME}: stores={m.group(3)} (provenance {prov.get('stores')}) != {n_claims} labeled "
                               f"claims in {CLAIMS_INDEX_REL}; only the full audit is installed")
    if prov.get("claims_index_sha256") != sha256_file(index_path):
        raise StoreLayoutError(f"{STATS_NAME}: claims_index_sha256 differs from the current {CLAIMS_INDEX_REL}; "
                               "re-run the audit on the current claim index")
    if not re.fullmatch(r"[0-9a-f]{64}", log_sha256):
        raise ValueError("log_sha256 must be 64 lowercase hex characters")
    timing["log"] = Path(log_name).name
    timing["log_sha256"] = log_sha256
    dest = {CSV_NAME: reports_dir(root) / CSV_NAME, STATS_NAME: root / STATS_REL, TIMING_NAME: root / TIMING_REL}
    _write_atomic(dest[CSV_NAME], csv_bytes)
    _write_atomic(dest[STATS_NAME], (from_dir / STATS_NAME).read_bytes())
    _write_atomic(dest[TIMING_NAME], dumps_sorted(timing).encode("ascii"))
    return dest


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import root as project_root

    ap = argparse.ArgumentParser(prog="python -m aac.data.kstore")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("domains", help="derive configs/factchecker-domains.txt from train/dev.json")
    p.add_argument("--out", default=str(DOMAINS_REL))
    p = sub.add_parser("audit", help="audit the knowledge stores (streamed, never extracted)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p = sub.add_parser("place", help="install a pulled audit into reports/, data/derived and data/measured")
    p.add_argument("--from", dest="from_dir", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--log-sha256", required=True)
    a = ap.parse_args(argv)
    root = project_root()
    try:
        if a.cmd == "domains":
            raw = root / RAW_REL
            _check_raw(raw, root / EXPECTED_REL)
            out = Path(a.out) if Path(a.out).is_absolute() else root / a.out
            hosts = write_domains(out, raw / "train.json", raw / "dev.json")
            print(f"ks-domains: {len(hosts)} hosts -> {out.relative_to(root) if out.is_relative_to(root) else out} "
                  f"sha256={sha256_file(out)}")
        elif a.cmd == "audit":
            if a.limit is not None and a.limit < 1:
                raise ValueError("--limit must be >= 1")
            res = run_audit(root, a.out_dir, limit=a.limit, workers=a.workers, progress=lambda m: print(m, flush=True))
            print(f"ks-audit: stores {res['stores']} config_hash {res['config_hash']} csv_sha256 {res['csv_sha256']} "
                  f"wall {res['wall_seconds']:.1f} s cpu {res['cpu_seconds']:.1f} s")
            for name in (CSV_NAME, STATS_NAME, TIMING_NAME):
                print(f"ks-audit: wrote {name} sha256={sha256_file(res['out_dir'] / name)}")
        else:
            dest = place(a.from_dir, root, log_name=a.log, log_sha256=a.log_sha256)
            for name, path in dest.items():
                print(f"ks-pull: placed {path.relative_to(root)} sha256={sha256_file(path)}")
    except (StoreLayoutError, ValueError, OSError) as e:
        print(f"kstore {a.cmd}: FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
