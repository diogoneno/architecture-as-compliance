"""Paced, logged, replayable HTTPS GET for aac.prisma (PRISMA-06; 06-COVERAGE.md).

Stdlib only (urllib.request, gzip, json): no new dependency, so uv.lock and every sealed config_hash stay put.

fetch(url, store=..., attempts_log=..., appendix=...) is the only network door of the reference checks:
- HTTPS only (NotHttps for any other scheme, also for a redirect target); default certificate verification.
- Resumable and idempotent: a URL whose `<store>/<request_id>.meta.json` exists is answered from the store
 without a request; with offline=True a missing record raises OfflineMiss and nothing touches the network.
- Per-host pacing (PACING seconds between two requests to one host); timeout 30 s; at most max_bytes are kept
 (2 MiB for APIs, the page fetch passes 256 KiB) and `truncated` records a longer body.
- Up to 5 attempts on 429, 500-504, timeouts and connection errors; the wait is a numeric Retry-After below
 120 s, else 2 x 2^attempt s. An OpenAlex 429 whose body says `Insufficient budget` raises BudgetExhausted
 at once. Every attempt appends one row to the attempts CSV (no wall-clock column).
- A final non-retryable answer (404, 403, 410, ...) is stored like a 200: it is an executed check.
- Stored: `<request_id>.meta.json` (canonical JSON: url, status, http_date, final_url, content_type,
 body_sha256, truncated) and `<body_sha256>.gz` (gzip, mtime 0, no file name). Request headers are built in
 memory only and never stored, so no API key or mailto reaches a file .
Secrets come only from the environment: AAC_PRISMA_MAILTO (User-Agent), OPENALEX_API_KEY (Bearer, OpenAlex),
S2_API_KEY (x-api-key, Semantic Scholar). URLs never carry a secret, so request_id = sha256(url) is safe.
"""
from __future__ import annotations

import gzip
import json
import os
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from aac.prisma._io import assert_confined, csv_text, read_csv, write_text_atomic
from aac.run.hashing import sha256_bytes

VERSION = "0.6"
TIMEOUT_S = 30
MAX_ATTEMPTS = 5
API_MAX_BYTES = 2 * 1024 * 1024
PAGE_MAX_BYTES = 256 * 1024
RETRY_STATUSES = frozenset({429, 500, 501, 502, 503, 504})
RETRY_AFTER_CAP_S = 120.0

PACING: dict[str, float] = {
    "api.openalex.org": 1.1,
    "export.arxiv.org": 3.5,
    "api.semanticscholar.org": 5.0,
    "api.crossref.org": 1.0,
    "api.datacite.org": 1.0,
}
DEFAULT_PACING_S = 2.0
S2_KEYED_PACING_S = 1.1

ATTEMPT_COLUMNS: tuple[str, ...] = ("request_id", "host", "attempt", "status", "retry_after", "wait_seconds", "outcome")

_LAST_REQUEST: dict[str, float] = {}  # host -> clock of the last request sent in this process


class NotHttps(ValueError):
    """A URL (or a redirect target) whose scheme is not https: refused before any byte is sent."""


class OfflineMiss(RuntimeError):
    """offline=True and no stored response for the URL."""


class BudgetExhausted(RuntimeError):
    """OpenAlex answered 429 `Insufficient budget`: stop the run, keep what was fetched (06-COVERAGE)."""


class GaveUp(RuntimeError):
    """Every attempt failed (connection errors or retryable statuses); nothing is stored, no row is written."""


def pacing_for(host: str) -> float:
    if host == "api.semanticscholar.org" and os.environ.get("S2_API_KEY"):
        return S2_KEYED_PACING_S
    return PACING.get(host, DEFAULT_PACING_S)


def headers_for(url: str) -> dict[str, str]:
    """Request headers, in memory only. Never logged or stored."""
    mailto = os.environ.get("AAC_PRISMA_MAILTO")
    ua = f"aac-prisma/{VERSION} (mailto:{mailto})" if mailto else f"aac-prisma/{VERSION}"
    h = {"User-Agent": ua}
    host = urllib.parse.urlsplit(url).hostname or ""
    if host == "api.openalex.org" and os.environ.get("OPENALEX_API_KEY"):
        h["Authorization"] = "Bearer " + os.environ["OPENALEX_API_KEY"]
    if host == "api.semanticscholar.org" and os.environ.get("S2_API_KEY"):
        h["x-api-key"] = os.environ["S2_API_KEY"]
    return h


def request_id(url: str) -> str:
    return sha256_bytes(url.encode("utf-8"))


def _require_https(url: str) -> str:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "https" or not parts.hostname:
        raise NotHttps(f"refusing non-https URL {url!r} (aac.prisma fetches HTTPS only)")
    return parts.hostname


class _HttpsOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D401 - urllib hook
        _require_https(urllib.parse.urljoin(req.full_url, newurl))
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _default_opener():
    return urllib.request.build_opener(_HttpsOnlyRedirect())


def _write_bytes_atomic(path: Path, data: bytes, appendix: Path) -> Path:
    target = assert_confined(path, appendix)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def gzip_bytes(data: bytes) -> bytes:
    """Deterministic gzip: mtime 0, no file name in the header."""
    return gzip.compress(data, compresslevel=9, mtime=0)


def meta_path(store: Path, url: str) -> Path:
    return Path(store) / f"{request_id(url)}.meta.json"


def load_stored(store: Path, url: str) -> tuple[dict, bytes] | None:
    mp = meta_path(store, url)
    if not mp.is_file():
        return None
    meta = json.loads(mp.read_text(encoding="ascii"))
    body = gzip.decompress((Path(store) / f"{meta['body_sha256']}.gz").read_bytes())
    if sha256_bytes(body) != meta["body_sha256"]:
        raise RuntimeError(f"stored body for {url} does not match its sha256 {meta['body_sha256']}")
    return meta, body


def _append_attempts(attempts_log: Path, rows: list[dict[str, str]], appendix: Path) -> None:
    if not rows:
        return
    existing: list[dict[str, str]] = []
    if Path(attempts_log).is_file():
        header, existing = read_csv(Path(attempts_log))
        if tuple(header) != ATTEMPT_COLUMNS:
            raise RuntimeError(f"{attempts_log}: unexpected header {header}")
    all_rows = existing + rows
    write_text_atomic(Path(attempts_log),
                      csv_text(ATTEMPT_COLUMNS, ([r[c] for c in ATTEMPT_COLUMNS] for r in all_rows)), appendix)


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    v = value.strip()
    try:
        f = float(v)
    except ValueError:
        return None
    return f if 0 <= f < RETRY_AFTER_CAP_S else None


def _fmt_s(x: float) -> str:
    return f"{x:g}"


def fetch(url: str, *, store: Path, attempts_log: Path, appendix: Path, offline: bool = False,
          max_bytes: int = API_MAX_BYTES, opener=None, sleep: Callable[[float], None] | None = None,
          clock: Callable[[], float] | None = None) -> tuple[dict, bytes, bool]:
    """(meta, body, fetched_now). See the module docstring for the contract."""
    host = _require_https(url)
    stored = load_stored(store, url)
    if stored is not None:
        return stored[0], stored[1], False
    if offline:
        raise OfflineMiss(f"offline: no stored response for {url} (request_id {request_id(url)})")
    opener = opener or _default_opener()
    sleep = sleep or time.sleep
    clock = clock or time.monotonic
    rid = request_id(url)
    rows: list[dict[str, str]] = []
    try:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            last = _LAST_REQUEST.get(host)
            if last is not None:
                gap = pacing_for(host) - (clock() - last)
                if gap > 0:
                    sleep(gap)
            _LAST_REQUEST[host] = clock()
            req = urllib.request.Request(url, headers=headers_for(url), method="GET")
            status: int | None = None
            hdrs = None
            final_url = url
            data = b""
            conn_error = None
            try:
                resp = opener.open(req, timeout=TIMEOUT_S)
                try:
                    status = int(getattr(resp, "status", None) or resp.getcode())
                    hdrs = resp.headers
                    final_url = resp.geturl() or url
                    data = resp.read(max_bytes + 1)
                finally:
                    resp.close()
            except NotHttps:
                raise
            except urllib.error.HTTPError as e:
                status = int(e.code)
                hdrs = e.headers
                final_url = e.geturl() or url
                try:
                    data = e.read(max_bytes + 1) or b""
                except Exception:  # a body that cannot be read is an empty body
                    data = b""
                finally:
                    e.close()
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                conn_error = type(e).__name__
            ra_raw = hdrs.get("Retry-After") if hdrs is not None else None
            if conn_error is None and status == 429 and host == "api.openalex.org" and b"Insufficient budget" in data:
                rows.append(dict(request_id=rid, host=host, attempt=str(attempt), status="429",
                                 retry_after=ra_raw or "", wait_seconds="0", outcome="budget"))
                raise BudgetExhausted(f"OpenAlex budget exhausted at {url}; fetched responses are kept")
            retryable = conn_error is not None or status in RETRY_STATUSES
            if not retryable:
                truncated = len(data) > max_bytes
                body = data[:max_bytes]
                bsha = sha256_bytes(body)
                http_date = (hdrs.get("Date") if hdrs is not None else None) or ""
                meta = {
                    "url": url, "status": status, "http_date": http_date, "final_url": final_url,
                    "content_type": (hdrs.get("Content-Type") if hdrs is not None else None) or "",
                    "body_sha256": bsha, "truncated": truncated,
                }
                _write_bytes_atomic(Path(store) / f"{bsha}.gz", gzip_bytes(body), appendix)
                _write_bytes_atomic(meta_path(store, url),
                                    json.dumps(meta, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n",
                                    appendix)
                rows.append(dict(request_id=rid, host=host, attempt=str(attempt), status=str(status),
                                 retry_after=ra_raw or "", wait_seconds="0", outcome="ok"))
                return meta, body, True
            if attempt == MAX_ATTEMPTS:
                rows.append(dict(request_id=rid, host=host, attempt=str(attempt),
                                 status=str(status) if status is not None else conn_error,
                                 retry_after=ra_raw or "", wait_seconds="0", outcome="gave-up"))
                raise GaveUp(f"{url}: {MAX_ATTEMPTS} attempts failed (last: {status if status is not None else conn_error})")
            ra = _retry_after(ra_raw)
            wait = ra if ra is not None else 2.0 * 2 ** attempt
            rows.append(dict(request_id=rid, host=host, attempt=str(attempt),
                             status=str(status) if status is not None else conn_error,
                             retry_after=ra_raw or "", wait_seconds=_fmt_s(wait), outcome="retry"))
            sleep(wait)
        raise AssertionError("unreachable")  # pragma: no cover
    finally:
        _append_attempts(attempts_log, rows, appendix)
