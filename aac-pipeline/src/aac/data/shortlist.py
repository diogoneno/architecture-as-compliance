"""S1: per-claim BM25 shortlists over the AVeriTeC knowledge stores, as a sealed run kind (retrieval stage).

`python -m aac.data.shortlist run --config <aac-shortlist/1 yaml> --seed N [--workers W] [--force]` seals
`SL-<hash10>-s<seed>` (record_count 0, system `shortlist`). One task per claim store, in a spawn process pool:

 1. stream the store with aac.data.kstore.iter_docs only (a StoreDoc has no `type` and no `query`);
 2. drop every document whose host is in the specification item 18 host list (`shortlist.domains`);
 3. flag each kept document: `url_eq_answer` (kstore.normalise_url of its URL equals a normalized answer
 source_url of the claim) and `url_date_after_claim` (kstore.url_date_after over kstore.url_path_date);
 4. skip blank sentences (strip() empty); collapse exact duplicate sentences to their first (doc_idx, sent_idx);
 5. tokenise corpus and claim with bm25s.tokenize(stopwords="en", stemmer=PyStemmer english), index with
 bm25s.BM25 (lucene, k1 1.5, b 0.75: the bm25s 0.3.13 defaults, pinned here), score every sentence;
 6. rank by (-round(score, 6), doc_idx, sent_idx); primary = the first top_n + reserve rows; variant = the
 first top_n rows among documents without a flag named in `variant_exclude` (same scores, no second pass).

Run directory (text-free):
 shortlist.jsonl.gz gzip (mtime 0, no file name); one canonical JSON line per claim in (split rank, idx) order:
 {"claim_id", "claim_hash", "n_docs", "n_docs_filtered", "n_sentences", "primary": [[doc_idx,
 sent_idx, score6, url_eq_answer, url_date_after_claim], ...], "variant": [...]}; flags are 0/1,
 n_sentences counts the indexed sentences (after the filter, blank and duplicate rules)
 shortlist.json the resolved shortlist block, counts, shortlist_sha256 (sha256 of the decompressed jsonl
 bytes), private_cache (path relative to out_root), private_cache_sha256 (decompressed), limit
 wallclock.jsonl per-store seconds, CPU seconds, raw sentence counts and worker peak RSS; the bm25 stage
Passage id: `<claim_id>#d<doc_idx>#s<sent_idx>`. The sentence text of every listed passage (primary union
variant, claim order then rank order) goes ONLY to <out_root>/data/derived/private/shortlist/<run_id>/
sentences.jsonl.gz (directory 0700, file 0600). No claim or sentence text is printed or written anywhere else.

Every configured store zip is re-hashed against its recorded sha256 (data/raw/MANIFEST.expected.sha256 for
`zip_base: data_raw`, the manifest named by data.manifest for `zip_base: project`) before it is read; a mismatch
stops the run before any run directory exists. The worker count is a CLI argument, never a config field.
Never imports torch.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import resource
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import numpy as np

from aac.data import kstore

SYSTEM = "shortlist"            # run-record `system` (index key: SL, shortlist, name, seed)
RESULT_SCHEMA = "aac-shortlist-result/1"
SHORTLIST_JSONL = "shortlist.jsonl.gz"
SHORTLIST_JSON = "shortlist.json"
CACHE_FILE = "sentences.jsonl.gz"
PRIVATE_REL = Path("data/derived/private/shortlist")
EXPECTED_RAW_REL = Path("data/raw/MANIFEST.expected.sha256")
FLAGS = ("url_eq_answer", "url_date_after_claim")
DECIMALS = 6
BM25_PARAMS = {"method": "lucene", "k1": 1.5, "b": 0.75}
STOPWORDS = "en"
STEMMER = "english"
GZIP_LEVEL = 6
# Order-key safety margin: every row whose rounded score can tie the k-th rounded score is a candidate.
_CAND_EPS = 2e-6
PASSAGE_ID_MAX = 128
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
CLAIM_ID_RE = re.compile(r"^averitec/(train|dev)/\d+$")
DETERMINISM = ("deterministic CPU: bm25s 0.3.13 BM25 (lucene, k1 1.5, b 0.75), PyStemmer english stemmer, bm25s "
               "English stop words; scores rounded to 6 d.p. before the order key (-score6, doc_idx, sent_idx)")


class ShortlistError(ValueError):
    """An input fails a hash, layout or selection check; nothing is sealed."""


def passage_id(claim_id: str, doc_idx: int, sent_idx: int) -> str:
    pid = f"{claim_id}#d{doc_idx}#s{sent_idx}"
    if len(pid) > PASSAGE_ID_MAX:
        raise ShortlistError(f"passage id longer than {PASSAGE_ID_MAX} characters")
    return pid


# ---------------------------------------------------------------- one store


def _rank(scores64: np.ndarray, doc_of: np.ndarray, sent_of: np.ndarray, idx: np.ndarray, k: int) -> list[tuple[int, float]]:
    """[(row, score6)] of the first k rows of `idx` by (-round(score, 6), doc_idx, sent_idx)."""
    if k <= 0 or idx.size == 0:
        return []
    s = scores64[idx]
    if idx.size > k:
        kth = np.partition(s, idx.size - k)[idx.size - k]   # the k-th largest raw score
        keep = s >= kth - _CAND_EPS                           # superset of every row that can rank in the first k
        idx, s = idx[keep], s[keep]
    r = np.fromiter((round(float(x), DECIMALS) for x in s), dtype=np.float64, count=s.size)
    order = np.lexsort((sent_of[idx], doc_of[idx], -r))[:k]
    return [(int(idx[o]), float(r[o])) for o in order]


def store_shortlist(zip_path, member: str, *, claim_text: str, claim_date: date | None, answer_urls,
                    domains: frozenset[str], top_n: int, reserve: int, variant_exclude) -> dict:
    """BM25 shortlist of one claim store. Returns {"primary", "variant", "texts", "counts"}: rows
 [doc_idx, sent_idx, score6, url_eq_answer, url_date_after_claim]; texts {(doc_idx, sent_idx): sentence} for
 every listed passage; counts n_docs, n_docs_filtered, n_sentences (indexed), n_sentences_raw (every sentence
 of every valid document), n_sentences_filtered, n_blank, n_duplicate, n_lines_unparsable."""
    import bm25s
    import Stemmer

    exclude = set(variant_exclude)
    unknown = exclude - set(FLAGS)
    if unknown:
        raise ShortlistError(f"variant_exclude names unknown flags {sorted(unknown)}")
    answers = {kstore.normalise_url(u) for u in answer_urls if isinstance(u, str) and u} - {""}
    c = {"n_docs": 0, "n_docs_filtered": 0, "n_sentences": 0, "n_sentences_raw": 0, "n_sentences_filtered": 0,
         "n_blank": 0, "n_duplicate": 0}
    stats = {"n_lines_unparsable": 0}
    texts: list[str] = []
    doc_of: list[int] = []
    sent_of: list[int] = []
    flags: dict[int, tuple[int, int]] = {}
    seen: set[str] = set()
    for doc in kstore.iter_docs(zip_path, member, stats):
        c["n_docs"] += 1
        c["n_sentences_raw"] += len(doc.sentences)
        if doc.host in domains:
            c["n_docs_filtered"] += 1
            c["n_sentences_filtered"] += len(doc.sentences)
            continue
        flags[doc.doc_idx] = (int(kstore.normalise_url(doc.url) in answers),
                              int(kstore.url_date_after(kstore.url_path_date(doc.url), claim_date)))
        for s_idx, s in enumerate(doc.sentences):
            if not s.strip():
                c["n_blank"] += 1
                continue
            if s in seen:
                c["n_duplicate"] += 1
                continue
            seen.add(s)
            texts.append(s)
            doc_of.append(doc.doc_idx)
            sent_of.append(s_idx)
    seen.clear()
    c["n_sentences"] = len(texts)
    c["n_lines_unparsable"] = stats["n_lines_unparsable"]
    if not texts:
        return {"primary": [], "variant": [], "texts": {}, "counts": c}

    stemmer = Stemmer.Stemmer(STEMMER)
    corpus = bm25s.tokenize(texts, stopwords=STOPWORDS, stemmer=stemmer, show_progress=False)
    bm = bm25s.BM25(**BM25_PARAMS)
    bm.index(corpus, show_progress=False)
    del corpus
    query = bm25s.tokenize([claim_text], stopwords=STOPWORDS, stemmer=stemmer, return_ids=False,
                           show_progress=False)[0]
    ids = bm.get_tokens_ids(list(query))
    scores = np.asarray(bm.get_scores_from_ids(ids), dtype=np.float64) if ids else np.zeros(len(texts))
    del bm
    if scores.shape != (len(texts),):
        raise ShortlistError(f"bm25s returned {scores.shape} scores for {len(texts)} sentences")
    doc_arr = np.asarray(doc_of, dtype=np.int64)
    sent_arr = np.asarray(sent_of, dtype=np.int64)
    all_idx = np.arange(len(texts))
    fl = np.array([flags[d] for d in doc_of], dtype=np.int8).reshape(-1, 2)
    excl = np.zeros(len(texts), dtype=bool)
    for j, name in enumerate(FLAGS):
        if name in exclude:
            excl |= fl[:, j] == 1
    primary = _rank(scores, doc_arr, sent_arr, all_idx, top_n + reserve)
    variant = _rank(scores, doc_arr, sent_arr, all_idx[~excl], top_n)

    def rows(ranked):
        return [[int(doc_arr[i]), int(sent_arr[i]), s6, int(fl[i, 0]), int(fl[i, 1])] for i, s6 in ranked]

    listed = {(int(doc_arr[i]), int(sent_arr[i])): texts[i] for i, _ in primary + variant}
    return {"primary": rows(primary), "variant": rows(variant), "texts": listed, "counts": c}


# ---------------------------------------------------------------- worker pool

_W: dict = {}


def _init_worker(params: dict) -> None:
    import bm25s  # noqa: F401 (imported once per worker, so no store's timing carries the import)
    import Stemmer  # noqa: F401

    _W.clear()
    _W.update(params)


def _reset_peak_rss() -> bool:
    """Reset the kernel's peak-RSS mark of this process (Linux /proc/self/clear_refs, value 5)."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
        return True
    except OSError:
        return False


def _peak_rss_bytes(reset_ok: bool) -> tuple[int, str]:
    if reset_ok:
        try:
            for line in Path("/proc/self/status").read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024, "VmHWM since this store (clear_refs 5)"
        except OSError:
            pass
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024, "ru_maxrss of the worker so far"


def _store_task(args) -> dict:
    from aac.run.hashing import claim_hash

    zip_path, member, claim_id, claim_text, claim_date, answer_urls = args
    reset_ok = _reset_peak_rss()
    t0, c0 = time.perf_counter(), time.process_time()
    with zipfile.ZipFile(zip_path) as zf:
        res = store_shortlist(zf, member, claim_text=claim_text,
                              claim_date=date.fromisoformat(claim_date) if claim_date else None,
                              answer_urls=answer_urls, domains=_W["domains"], top_n=_W["top_n"],
                              reserve=_W["reserve"], variant_exclude=_W["variant_exclude"])
    wall, cpu = time.perf_counter() - t0, time.process_time() - c0
    rss, rss_source = _peak_rss_bytes(reset_ok)
    cnt = res["counts"]
    line = {"claim_id": claim_id, "claim_hash": claim_hash(claim_text),
            "n_docs": cnt["n_docs"], "n_docs_filtered": cnt["n_docs_filtered"], "n_sentences": cnt["n_sentences"],
            "primary": res["primary"], "variant": res["variant"]}
    cache, done = [], set()
    for row in res["primary"] + res["variant"]:
        key = (row[0], row[1])
        if key not in done:
            done.add(key)
            cache.append((passage_id(claim_id, row[0], row[1]), res["texts"][key]))
    timing = {"stage": "bm25-store", "claim_id": claim_id, "seconds": round(wall, DECIMALS),
              "cpu_seconds": round(cpu, DECIMALS), "n_sentences_raw": cnt["n_sentences_raw"],
              "n_sentences": cnt["n_sentences"], "rss_bytes": rss, "rss_source": rss_source}
    return {"line": line, "cache": cache, "counts": cnt, "timing": timing}


def _hash_task(path: str) -> tuple[str, str]:
    from aac.run.hashing import sha256_file

    return path, sha256_file(Path(path))


# ---------------------------------------------------------------- inputs


def zip_paths(root: Path, sl: dict) -> list[Path]:
    base = Path(root) / "data" / "raw" if sl["zip_base"] == "data_raw" else Path(root)
    return [base / z for z in sl["store_zips"]]


def _expected_zip_hashes(root: Path, resolved: dict) -> dict[Path, str]:
    """{zip path: recorded sha256}. data_raw: data/raw/MANIFEST.expected.sha256 keyed by the path under data/raw;
 project: the manifest named by data.manifest keyed by the path relative to the manifest's directory."""
    from aac.data.verify import parse_expected

    sl = resolved["shortlist"]
    paths = zip_paths(root, sl)
    if sl["zip_base"] == "data_raw":
        expected = parse_expected(Path(root) / EXPECTED_RAW_REL)
        keys = list(sl["store_zips"])
        where = EXPECTED_RAW_REL.as_posix()
    else:
        manifest = Path(root) / resolved["data"]["manifest"]
        expected = parse_expected(manifest)
        keys = [os.path.relpath(p, manifest.parent) for p in paths]
        where = resolved["data"]["manifest"]
    out = {}
    for p, k in zip(paths, keys):
        if k not in expected:
            raise ShortlistError(f"{k}: no recorded sha256 in {where}; refusing to read the store")
        out[p] = expected[k]
    return out


def _check_domains(root: Path, resolved: dict) -> str:
    """Re-hash the host list against the run's data manifest; returns its sha256."""
    from aac.data.verify import parse_expected
    from aac.run.hashing import sha256_file

    rel = resolved["shortlist"]["domains"]
    path = Path(root) / rel
    if resolved["data"]["dataset"] != "fixture":
        from aac.data.manifest import verify_entries

        verify_entries(root, resolved["data"]["manifest"], [rel])
    else:
        manifest = Path(root) / resolved["data"]["manifest"]
        key = os.path.relpath(path, manifest.parent)
        listed = parse_expected(manifest)
        if key not in listed or listed[key] != sha256_file(path):
            raise ShortlistError(f"{rel}: not listed in {resolved['data']['manifest']} or sha256 differs")
    return sha256_file(path)


def _claims(root: Path, claims_dir: str) -> dict[str, dict]:
    """{claim_id: {text, date (ISO or None), answer_urls}} from the raw train/dev JSON (text never written)."""
    from aac.data.averitec import load_claims

    raw_dir = Path(root) / claims_dir
    claims = load_claims(raw_dir)
    answers = kstore._answer_urls(raw_dir)
    if set(claims) != set(answers):
        raise ShortlistError("claim text and answer URLs name different claims")
    return {cid: {"text": c["text"], "date": c["date"], "answer_urls": answers[cid]} for cid, c in claims.items()}


def select_stores(stores: list[tuple[str, str, str]], select: dict, claims: dict) -> tuple[list, bool]:
    """(selected (claim_id, zip path, member) in (split rank, idx) order, subset flag)."""
    by_id = {}
    for cid, zp, member in stores:
        if cid in by_id:
            raise ShortlistError(f"{cid} has more than one store across the configured zips")
        by_id[cid] = (cid, zp, member)
    ordered = sorted(by_id.values(), key=lambda t: kstore._order(t[0]))
    limit, include = select.get("limit"), list(select.get("include") or [])
    for cid in include:
        if not CLAIM_ID_RE.match(cid) or cid not in by_id:
            raise ShortlistError(f"stores.include names {cid!r}, which has no store in the configured zips")
    subset = limit is not None or bool(include)
    if subset:
        chosen = {t[0] for t in ordered[:limit]} if limit is not None else set()
        chosen |= set(include)
        ordered = [t for t in ordered if t[0] in chosen]
    else:
        missing = sorted(set(claims) - set(by_id), key=kstore._order)
        if missing:
            raise ShortlistError(f"no store for {len(missing)} claims, first {missing[0]}")
    extra = [t[0] for t in ordered if t[0] not in claims]
    if extra:
        raise ShortlistError(f"{len(extra)} selected stores name no claim in data.claims, first {extra[0]}")
    return ordered, subset


# ---------------------------------------------------------------- private writer


class _GzLines:
    """gzip (mtime 0, no file name) written to `<path>.tmp`, sha256 over the decompressed bytes; commit() renames."""

    def __init__(self, path: Path, mode: int | None = None):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        fd = os.open(self.tmp, flags, mode if mode is not None else 0o644)
        if mode is not None:
            os.chmod(self.tmp, mode)
        self.raw = os.fdopen(fd, "wb")
        self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=GZIP_LEVEL)
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, data: bytes) -> None:
        self.gz.write(data)
        self.h.update(data)
        self.n += 1

    def commit(self) -> str:
        self.gz.close()
        self.raw.close()
        os.replace(self.tmp, self.path)
        return self.h.hexdigest()

    def abort(self) -> None:
        try:
            self.gz.close()
            self.raw.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def private_cache_dir(root: Path, run_id: str) -> Path:
    from aac.run import paths as aac_paths

    return aac_paths.out_root(root) / PRIVATE_REL / run_id


def _make_private_dir(root: Path, run_id: str) -> Path:
    from aac.run import paths as aac_paths

    d = private_cache_dir(root, run_id)
    d.mkdir(parents=True, exist_ok=True)
    priv = aac_paths.out_root(root) / "data" / "derived" / "private"
    for p in (priv, priv / "shortlist", d):
        os.chmod(p, 0o700)
    return d


# ---------------------------------------------------------------- run


def run(config: Path, seed: int, *, workers: int = 8, force: bool = False, progress=None) -> dict:
    from aac.run import hashing
    from aac.run.clock import make_clock
    from aac.run.config import ShortlistConfig, load_any
    from aac.run.env import write_env
    from aac.run.finalize import claim_run_dir, finalize_run
    from aac.run.identity import identity, lock_path
    from aac.run.paths import out_root
    from aac.run.paths import root as project_root

    say = progress or (lambda msg: None)
    if workers < 1:
        raise ShortlistError("--workers must be >= 1")
    root = project_root()
    cfg = load_any(Path(config))
    if not isinstance(cfg, ShortlistConfig):
        raise ShortlistError(f"{config} is not an aac-shortlist/1 config")
    ident = identity(root, Path(config), seed)
    resolved, rid = ident["resolved"], ident["run_id"]
    sl = resolved["shortlist"]

    # inputs first: nothing is claimed or written when a hash or a selection check fails
    import multiprocessing as mp

    expected = _expected_zip_hashes(root, resolved)
    domains_sha = _check_domains(root, resolved)
    domains = kstore.load_domains(root / sl["domains"])
    claims = _claims(root, resolved["data"]["claims"])
    # one BLAS/OpenMP thread per worker: spawned workers import numpy afresh and inherit this, so the CPU load is
    
    for var in THREAD_VARS:
        os.environ.setdefault(var, "1")
    ctx = mp.get_context("spawn")
    t_hash = time.perf_counter()
    with ctx.Pool(min(workers, len(expected))) as pool:
        actual = dict(pool.map(_hash_task, [str(p) for p in expected], chunksize=1))
    hash_seconds = time.perf_counter() - t_hash
    for p, sha in expected.items():
        if actual[str(p)] != sha:
            raise ShortlistError(f"{p.name}: sha256 {actual[str(p)]} != recorded {sha}; refusing to read the store")
    say(f"shortlist: {len(expected)} store zips match their recorded sha256 ({hash_seconds:.1f} s)")
    stores = [(cid, str(zp), member) for zp in expected for cid, member in kstore.list_members(zp)]
    selected, subset = select_stores(stores, sl["stores"], claims)

    key = (resolved["regime"], SYSTEM, resolved["name"], seed)
    run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, force)
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, lock_path(root))
    started = clock.now_iso_ms()
    priv_dir = _make_private_dir(root, rid)
    cache_path = priv_dir / CACHE_FILE
    out = _GzLines(run_dir / SHORTLIST_JSONL)
    cache = _GzLines(cache_path, mode=0o600)
    params = {"domains": domains, "top_n": int(sl["top_n"]), "reserve": int(sl["reserve"]),
              "variant_exclude": tuple(sl["variant_exclude"])}
    tasks = [(zp, member, cid, claims[cid]["text"], claims[cid]["date"], claims[cid]["answer_urls"])
             for cid, zp, member in selected]
    totals = {k: 0 for k in ("n_docs", "n_docs_filtered", "n_sentences", "n_sentences_raw", "n_sentences_filtered",
                             "n_blank", "n_duplicate", "n_lines_unparsable", "n_primary_rows", "n_variant_rows",
                             "n_stores_empty", "n_stores_short_primary", "n_stores_short_variant")}
    timings: list[dict] = []
    t0 = time.perf_counter()
    try:
        if workers == 1:
            _init_worker(params)
            stream, pool = map(_store_task, tasks), None
        else:
            pool = ctx.Pool(workers, initializer=_init_worker, initargs=(params,))
            stream = pool.imap(_store_task, tasks, chunksize=1)    # ordered: lines leave in claim order
        try:
            for n, res in enumerate(stream, 1):
                line = res["line"]
                out.write(hashing.canonical(line) + b"\n")
                for pid, text in res["cache"]:
                    cache.write(json.dumps({"passage_id": pid, "text": text}, ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8", "surrogatepass") + b"\n")
                for k in ("n_docs", "n_docs_filtered", "n_sentences", "n_sentences_raw", "n_sentences_filtered",
                          "n_blank", "n_duplicate", "n_lines_unparsable"):
                    totals[k] += res["counts"][k]
                totals["n_primary_rows"] += len(line["primary"])
                totals["n_variant_rows"] += len(line["variant"])
                totals["n_stores_empty"] += int(res["counts"]["n_sentences"] == 0)
                totals["n_stores_short_primary"] += int(len(line["primary"]) < params["top_n"] + params["reserve"])
                totals["n_stores_short_variant"] += int(len(line["variant"]) < params["top_n"])
                timings.append(res["timing"])
                if n % 50 == 0 or n == len(tasks):
                    say(f"shortlist: {n}/{len(tasks)} stores")
            if pool is not None:
                pool.close()
        except BaseException:
            if pool is not None:
                pool.terminate()
            raise
        finally:
            if pool is not None:
                pool.join()
        wall = time.perf_counter() - t0
        shortlist_sha = out.commit()
        n_cache = cache.n
        cache_sha = cache.commit()
    except BaseException:
        out.abort()
        cache.abort()
        raise
    os.chmod(cache_path, 0o600)
    finished = clock.now_iso_ms()

    cpu_workers = round(sum(t["cpu_seconds"] for t in timings), DECIMALS)
    result = {
        "schema": RESULT_SCHEMA, "run_id": rid, "shortlist": sl, "limit": subset, "stores": len(tasks),
        "counts": {**totals, "n_passages_cached": n_cache},
        "bm25": {"library": "bm25s", **BM25_PARAMS, "stopwords": STOPWORDS, "stemmer": f"PyStemmer {STEMMER}",
                 "order_key": "(-round(score, 6), doc_idx, sent_idx)"},
        "row_fields": ["doc_idx", "sent_idx", "score6", *FLAGS],
        "passage_id_format": "<claim_id>#d<doc_idx>#s<sent_idx>",
        "shortlist_sha256": shortlist_sha,
        "private_cache": (PRIVATE_REL / rid / CACHE_FILE).as_posix(),
        "private_cache_sha256": cache_sha,
        "domains_sha256": domains_sha,
        "store_zips_sha256": {Path(z).as_posix(): expected[p] for z, p in zip(sl["store_zips"], expected)},
    }
    (run_dir / SHORTLIST_JSON).write_text(hashing.dumps_sorted(result), encoding="ascii")
    stage = {"stage": "bm25", "device": "cpu", "seconds": round(wall, DECIMALS), "items": len(tasks),
             "cpu_seconds_workers": cpu_workers, "workers": workers}
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        fh.write(json.dumps({"stage": "verify-zips", "seconds": round(hash_seconds, DECIMALS),
                             "items": len(expected)}, sort_keys=True) + "\n")
        for t in timings:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
        fh.write(json.dumps(stage, sort_keys=True) + "\n")
    summary = {"configurations": [f"shortlist/{resolved['name']}"],
               "items": {"stores": len(tasks), "passages_cached": n_cache},
               "stages": [{"name": "bm25", "device": "cpu",
                           "seconds": round(wall, DECIMALS) if clock.source == "wall" else 0.0, "items": len(tasks)}]}
    digest = finalize_run(root, run_dir, ident, summary=summary, record_count=0, shards=[], clock=clock,
                          started=started, finished=finished, system=SYSTEM, determinism_level=DETERMINISM,
                          target_dir=target_dir, stale_ids=stale_ids)
    return {"run_id": rid, "stores": len(tasks), "shortlist_sha256": shortlist_sha, "private_cache_sha256": cache_sha,
            "passages_cached": n_cache, "wall_seconds": wall, "cpu_seconds_workers": cpu_workers, "sealed": digest,
            "run_dir": target_dir, "out_root": out_root(root)}


# ---------------------------------------------------------------- report


def report(run_id: str) -> dict:
    """Counts, wall and CPU seconds, CPU seconds per million (raw) sentences and the largest worker RSS of a
 sealed SL run, read from its sealed files only."""
    from aac.run.hashing import RUN_ID_RE
    from aac.run.paths import runs_dir

    if not RUN_ID_RE.match(run_id) or not run_id.startswith("SL-"):
        raise ShortlistError(f"{run_id!r} is not an SL run id")
    d = runs_dir() / run_id
    if not (d / "SEALED").is_file():
        raise ShortlistError(f"{run_id}: no SEALED run directory under {runs_dir()}")
    res = json.loads((d / SHORTLIST_JSON).read_text(encoding="ascii"))
    rec = json.loads((d / "run-record.json").read_text(encoding="ascii"))
    lines = [json.loads(x) for x in (d / "wallclock.jsonl").read_text(encoding="ascii").splitlines() if x.strip()]
    per = [x for x in lines if x.get("stage") == "bm25-store"]
    stage = next(x for x in lines if x.get("stage") == "bm25")
    raw = sum(int(x["n_sentences_raw"]) for x in per)
    cpu = float(stage["cpu_seconds_workers"])
    largest = max(per, key=lambda x: (int(x["n_sentences_raw"]), x["claim_id"])) if per else None
    return {
        "run_id": run_id, "release_sha": rec["git_sha"], "stores": res["stores"], "limit": res["limit"],
        "sentences": raw, "sentences_indexed": res["counts"]["n_sentences"],
        "wall_s": float(stage["seconds"]), "cpu_s": cpu, "workers": int(stage["workers"]),
        "cpu_s_per_million_sentences": round(cpu / raw * 1e6, DECIMALS) if raw else None,
        "max_worker_rss_bytes": max((int(x["rss_bytes"]) for x in per), default=0),
        "rss_source": sorted({x["rss_source"] for x in per}),
        "largest_store": ({"claim_id": largest["claim_id"], "n_sentences": int(largest["n_sentences_raw"]),
                           "seconds": float(largest["seconds"]), "rss_bytes": int(largest["rss_bytes"])}
                          if largest else None),
        "shortlist_sha256": res["shortlist_sha256"], "private_cache_sha256": res["private_cache_sha256"],
        "counts": res["counts"],
    }


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.data.shortlist")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="seal an SL run from an aac-shortlist/1 config (retrieval stage)")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("report", help="counts, timing and memory of a sealed SL run")
    p.add_argument("--run-id", required=True)
    p.add_argument("--json", action="store_true", help="print the report as sorted JSON")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "run":
            res = run(Path(a.config), a.seed, workers=a.workers, force=a.force, progress=lambda m: print(m, flush=True))
            print(f"shortlist {res['run_id']}: {res['stores']} stores, {res['passages_cached']} passages cached, "
                  f"wall {res['wall_seconds']:.1f} s, SEALED {res['sealed'][:12]}")
            print(f"shortlist: run_id={res['run_id']} shortlist_sha256={res['shortlist_sha256']}")
        else:
            rep = report(a.run_id)
            if a.json:
                from aac.run.hashing import dumps_sorted

                sys.stdout.write(dumps_sorted(rep))
            else:
                print(f"shortlist report {rep['run_id']}: stores {rep['stores']} (subset {rep['limit']}), sentences "
                      f"{rep['sentences']} raw / {rep['sentences_indexed']} indexed, wall {rep['wall_s']:.1f} s, cpu "
                      f"{rep['cpu_s']:.1f} s on {rep['workers']} workers, cpu s per million sentences "
                      f"{rep['cpu_s_per_million_sentences']}, max worker RSS {rep['max_worker_rss_bytes']} B")
                if rep["largest_store"]:
                    ls = rep["largest_store"]
                    print(f"shortlist report: largest store {ls['claim_id']} {ls['n_sentences']} sentences "
                          f"{ls['seconds']:.1f} s RSS {ls['rss_bytes']} B")
    except Exception as e:  # noqa: BLE001 (verdict by exit code; messages carry ids and paths only)
        from aac.data.manifest import ManifestError
        from aac.run.finalize import FinalizeRefused, RunRefused

        if isinstance(e, (ShortlistError, kstore.StoreLayoutError, ManifestError, RunRefused, FinalizeRefused, OSError,
                          ValueError)):
            print(f"shortlist {a.cmd}: FAIL: {type(e).__name__}: {e}", file=sys.stderr)
            return 2 if isinstance(e, RunRefused) else 1
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
