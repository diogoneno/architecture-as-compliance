"""S2: frozen per-claim evidence as a sealed run kind (dense and re-rank stages).

`python -m aac.components.c2_evidence_run run --config <aac-evidence/1 yaml> --seed N [--force]` seals
`EV-<hash10>-s<seed>` (record_count 0, system `evidence`). Retrieval is seed-independent: one EV run serves every
R1/R2/R3 seed. Mode `per_claim` (mode `pooled_rd`, is described after the private files):

 1. the SL run named by evidence.shortlist_from is resolved by aac.run.identity (run id and sealed digest inside
 config_hash); here its shortlist.jsonl.gz and its private sentence cache are re-hashed against the sha256 values
 its shortlist.json records before any row is used; an SL subset run (limit true) is refused unless this config
 also selects a subset (stores.limit or stores.include); every claim hash must equal the SL line's claim_hash;
 2. per claim (SL order; stores.limit = the first N claims, stores.include adds claim ids) and per configured list
 (primary -> the SL "primary" list, sensitivity-b -> the SL "variant" list): the first n_passages rows;
 3. every distinct listed passage is embedded once by DenseEncoder (own CLS pooling, L2), in chunks of
 EMBED_CHUNK_CLAIMS claims with the fixed batch plan, and stored as fp16 in claim order;
 4. each claim is embedded alone (batch size 1) and kept float32; dense score = c2_retrieval.dense_scores(claim
 vector, stored fp16 vectors); the first k_dense by rank go to the Reranker as (claim, passage) pairs; the
 first k re-ranked by rank are the evidence. Order key everywhere: (-round(score, 6), passage_id).

Run directory (text-free):
 evidence.jsonl.gz gzip (mtime 0, no file name); one canonical line per claim and list, claim order then
 evidence.variants order: {"claim_id", "claim_hash", "variant", "dense": [[passage_id, score6],
 ...], "rerank": [[passage_id, score6], ...]}
 evidence.json the resolved evidence block, counts, evidence_sha256 (decompressed jsonl), the SL inputs and
 their hashes, private file paths (relative to out_root) with sha256 each, limit flag, device,
 configured and applied precision
 wallclock.jsonl stage seconds (read-inputs, load-models, embed, rerank)
Private files (<out_root>/data/derived/private/evidence/<run_id>/, directory 0700, files 0600, written as .tmp and
moved into place only after every stage succeeded):
 vectors.npy fp16, one row per embedded passage, in index order
 index.jsonl.gz per claim: {"claim_id", "start", "count", "passage_ids"} (rows start .. start+count-1)
 pooled/passages.jsonl.gz, pooled/vectors.npy when evidence.pooled.enabled: the pooled RD corpus (pre-registered specification
 clarification, methodological note
 claims of evidence.pooled.partitions (chronological pre-cutoff partitions of data.split_manifest),
 claim order then rank order, keeping the first passage of each distinct sha256(text);
 {"pool_idx", "passage_id", "text"} lines and their fp16 vectors (copied rows, pool order)
No claim or passage text is printed or written outside the private directory.

Mode `pooled_rd` (pre-registered specification.
 1. the per_claim EV run named by evidence.pool_from (resolved by identity into evidence.pool) is re-verified: sealed
 digest equal to the ref, pooled bundle enabled, pooled/passages.jsonl.gz (decompressed) and pooled/vectors.npy
 re-hashed against its evidence.json;
 2. the RD items of data.rd_items are read with aac.data.evidence_providers.load_incrediblae_items: the committed
 text-free item file reads lane B (cpu-host only; LaneBMissing stops the run elsewhere) and checks every text
 against its text_hash; a fixture item file carries its text;
 3. one BM25 index over the pooled passages (the S1 recipe: bm25s lucene k1 1.5 b 0.75, English stop words,
 PyStemmer english); each RD text (the whole text) is the query; candidates = the first bm25_top_n by
 (-round(score, 6), pool_idx);
 4. each RD text is embedded alone (batch size 1, truncated at dense.max_seq_length) and scored against the stored
 fp16 pooled vectors of its candidates; rank(..., k_dense); re-rank of (RD text, passage) pairs (truncated at
 reranker.max_length); rank(..., k).
 Output: evidence.jsonl.gz lines with claim_id = item_id, claim_hash = sha256 of the RD text, variant primary;
 evidence.json; wallclock.jsonl (stages read-inputs, bm25, load-models, embed, rerank). No private files and no text."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from aac.components.c2_retrieval import DECIMALS, DenseEncoder, Reranker, dense_scores, rank

SYSTEM = "evidence"             # run-record `system` (index key: EV, evidence, name, seed)
RESULT_SCHEMA = "aac-evidence-result/1"
EVIDENCE_JSONL = "evidence.jsonl.gz"
EVIDENCE_JSON = "evidence.json"
VECTORS_FILE = "vectors.npy"
INDEX_FILE = "index.jsonl.gz"
PRIVATE_REL = Path("data/derived/private/evidence")
POOL_DIR = "pooled"
POOL_PASSAGES = "passages.jsonl.gz"
POOL_COPY_ROWS = 65536
SL_LIST = {"primary": "primary", "sensitivity-b": "variant"}
EMBED_CHUNK_CLAIMS = 32         # claims per embedding chunk (fixed: batch composition, hence float noise, is fixed too)
PROGRESS_EVERY = 50


class EvidenceError(ValueError):
    """An input fails a hash, selection or mode check; nothing is sealed."""


# ---------------------------------------------------------------- small writers


class _GzLines:
    """gzip (mtime 0, no file name) to `<path>.tmp`, sha256 over the decompressed bytes; commit renames."""

    def __init__(self, path: Path, mode: int | None = None, *, defer: bool = False):
        self.path = Path(path)
        self.tmp = self.path.with_name(self.path.name + ".tmp")
        fd = os.open(self.tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode if mode is not None else 0o644)
        if mode is not None:
            os.chmod(self.tmp, mode)
        self.raw = os.fdopen(fd, "wb")
        self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0, compresslevel=6)
        self.h = hashlib.sha256()
        self.n = 0
        self.defer = defer

    def write(self, data: bytes) -> None:
        self.gz.write(data)
        self.h.update(data)
        self.n += 1

    def close(self) -> str:
        """Close the tmp file; returns the decompressed sha256. Without `defer`, also moves it into place."""
        self.gz.close()
        self.raw.close()
        if not self.defer:
            os.replace(self.tmp, self.path)
        return self.h.hexdigest()

    def abort(self) -> None:
        try:
            if not self.raw.closed:
                self.gz.close()
                self.raw.close()
        finally:
            self.tmp.unlink(missing_ok=True)


def _line(obj) -> bytes:
    from aac.run.hashing import canonical

    return canonical(obj) + b"\n"


def private_dir(root: Path, run_id: str) -> Path:
    from aac.run import paths as aac_paths

    return aac_paths.out_root(root) / PRIVATE_REL / run_id


def _make_private_dir(root: Path, run_id: str) -> Path:
    from aac.run import paths as aac_paths

    d = private_dir(root, run_id)
    d.mkdir(parents=True, exist_ok=True)
    priv = aac_paths.out_root(root) / "data" / "derived" / "private"
    for p in (priv, priv / "evidence", d):
        os.chmod(p, 0o700)
    return d


def _gz_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- inputs


def read_shortlist_run(root: Path, ref: dict) -> tuple[dict, Path]:
    """(shortlist.json, run dir) of the SL run `ref`; the shortlist file is re-hashed against shortlist_sha256."""
    from aac.run.paths import runs_dir

    d = runs_dir(root) / ref["run_id"]
    if not (d / "SEALED").is_file():
        raise EvidenceError(f"SL run {ref['run_id']} has no SEALED run directory under {runs_dir(root)}")
    meta = json.loads((d / "shortlist.json").read_text(encoding="ascii"))
    sha = _gz_sha256(d / "shortlist.jsonl.gz")
    if sha != meta["shortlist_sha256"]:
        raise EvidenceError(f"{ref['run_id']}/shortlist.jsonl.gz: sha256 {sha} != shortlist_sha256 "
                            f"{meta['shortlist_sha256']}; refusing to read it")
    return meta, d


def select_claims(lines, stores: dict) -> tuple[list[dict], bool]:
    """(selected SL lines in SL order, subset flag): stores.limit = the first N lines, stores.include adds ids.
 `lines` may be a stream (one parsed line at a time), so a 3,568-line SL file is never held whole."""
    limit, include = stores.get("limit"), set(stores.get("include") or [])
    subset = limit is not None or bool(include)
    out, seen = [], set()
    for n, x in enumerate(lines):
        seen.add(x["claim_id"])
        if not subset or (limit is not None and n < limit) or x["claim_id"] in include:
            out.append(x)
    for cid in sorted(include - seen):
        raise EvidenceError(f"stores.include names {cid!r}, which the SL run does not list")
    return out, subset


def read_cache_texts(cache: Path, sha256: str, wanted: set[str]) -> dict[str, str]:
    """{passage_id: text} for the wanted ids of the private SL cache; the whole cache is re-hashed (decompressed)
 against `sha256` before anything is returned."""
    h = hashlib.sha256()
    texts: dict[str, str] = {}
    with gzip.open(cache, "rb") as fh:
        for raw in fh:
            h.update(raw)
            e = json.loads(raw)
            if e["passage_id"] in wanted:
                texts[e["passage_id"]] = e["text"]
    if h.hexdigest() != sha256:
        raise EvidenceError(f"{cache.name}: sha256 {h.hexdigest()} != private_cache_sha256 {sha256}; refusing to use it")
    missing = wanted - set(texts)
    if missing:
        raise EvidenceError(f"{len(missing)} listed passages have no text in the private SL cache")
    return texts


def _pid(claim_id: str, row) -> str:
    return f"{claim_id}#d{int(row[0])}#s{int(row[1])}"


def plan_claims(selected: list[dict], variants: list[str], n_passages: int) -> list[dict]:
    """Per claim: {claim_id, claim_hash, lists {variant: [passage_id, ...]}, order [distinct ids, first seen]}."""
    out = []
    for x in selected:
        lists = {v: [_pid(x["claim_id"], r) for r in x[SL_LIST[v]][:n_passages]] for v in variants}
        order, seen = [], set()
        for v in variants:
            for p in lists[v]:
                if p not in seen:
                    seen.add(p)
                    order.append(p)
        out.append({"claim_id": x["claim_id"], "claim_hash": x["claim_hash"], "lists": lists, "order": order})
    return out


def build_pooled(plans: list[dict], texts: dict[str, str], vec_file: Path, pool_dir: Path, partitions: list[str],
                 split: dict, subset: bool) -> dict:
    """Write pooled/passages.jsonl.gz and pooled/vectors.npy as .tmp files (moved into place by the caller) and return
 their metadata. `plans` are in claim order and their rows in `vec_file` are consecutive in plan order."""
    from aac.data import kstore
    from aac.run.hashing import sha256_file

    members: set[str] = set()
    for part in partitions:
        if part not in split or not isinstance(split[part], list):
            raise EvidenceError(f"data.split_manifest has no partition {part!r}")
        members |= set(split[part])
    in_run = {pl["claim_id"] for pl in plans}
    missing = sorted(members - in_run, key=kstore._order)
    if missing and not subset:
        raise EvidenceError(f"{len(missing)} claims of pooled.partitions are not in the SL run (first {missing[0]})")
    pool: list[tuple[str, int]] = []
    seen: set[str] = set()
    start = n_claims = n_dup = 0
    for pl in plans:
        if pl["claim_id"] in members:
            n_claims += 1
            pos = {p: start + j for j, p in enumerate(pl["order"])}
            for p in pl["lists"]["primary"]:
                h = hashlib.sha256(texts[p].encode("utf-8", "surrogatepass")).hexdigest()
                if h in seen:
                    n_dup += 1
                    continue
                seen.add(h)
                pool.append((p, pos[p]))
        start += len(pl["order"])
    pool_dir.mkdir(exist_ok=True)
    os.chmod(pool_dir, 0o700)
    w = _GzLines(pool_dir / POOL_PASSAGES, mode=0o600, defer=True)
    vtmp = pool_dir / (VECTORS_FILE + ".tmp")
    try:
        for i, (pid, _) in enumerate(pool):
            w.write(json.dumps({"pool_idx": i, "passage_id": pid, "text": texts[pid]}, ensure_ascii=False,
                               sort_keys=True, separators=(",", ":")).encode("utf-8", "surrogatepass") + b"\n")
        passages_sha = w.close()
        src = np.load(vec_file, mmap_mode="r")
        dst = np.lib.format.open_memmap(vtmp, mode="w+", dtype=np.float16, shape=(len(pool), src.shape[1]))
        os.chmod(vtmp, 0o600)
        rows = np.asarray([r for _, r in pool], dtype=np.int64)
        for i in range(0, len(pool), POOL_COPY_ROWS):
            dst[i:i + POOL_COPY_ROWS] = src[rows[i:i + POOL_COPY_ROWS]]
        dst.flush()
        shape = [len(pool), int(src.shape[1])]
        del dst, src
    except BaseException:
        w.abort()
        vtmp.unlink(missing_ok=True)
        raise
    return {"tmp": [(w.tmp, w.path), (vtmp, pool_dir / VECTORS_FILE)],
            "meta": {"enabled": True, "partitions": list(partitions), "dedup": "sentence-sha256",
                     "source_list": "primary", "n_claims": n_claims, "n_claims_missing": len(missing),
                     "n_passages": len(pool), "n_duplicates_dropped": n_dup,
                     "passages_sha256": passages_sha, "passages_sha256_of": "decompressed bytes",
                     "vectors_sha256": sha256_file(vtmp), "vectors_dtype": "float16", "vectors_shape": shape}}


# ---------------------------------------------------------------- run


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def run(config: Path, seed: int, *, force: bool = False, progress=None) -> dict:
    cublas_at_start = os.environ.get("CUBLAS_WORKSPACE_CONFIG")  # before seed_everything's setdefault (operational constraint)
    from aac.data.averitec import load_claims
    from aac.run import hashing
    from aac.run.clock import make_clock
    from aac.run.config import EvidenceConfig, load_any
    from aac.run.env import write_env
    from aac.run.finalize import claim_run_dir, finalize_run
    from aac.run.identity import identity, lock_path
    from aac.run.paths import out_root
    from aac.run.paths import root as project_root

    say = progress or (lambda msg: None)
    root = project_root()
    cfg = load_any(Path(config))
    if not isinstance(cfg, EvidenceConfig):
        raise EvidenceError(f"{config} is not an aac-evidence/1 config")
    if cfg.evidence.mode == "pooled_rd":
        return run_pooled_rd(Path(config), seed, force=force, say=say, cublas_at_start=cublas_at_start)
    ident = identity(root, Path(config), seed)
    resolved, rid = ident["resolved"], ident["run_id"]
    ev = resolved["evidence"]
    ref = ev["shortlist"]

    # ---- inputs first: nothing is claimed or written when a hash or a selection check fails
    t_in = time.perf_counter()
    sl_meta, sl_dir = read_shortlist_run(root, ref)
    if sl_meta["limit"] and ev["stores"]["limit"] is None and not ev["stores"]["include"]:
        raise EvidenceError(f"SL run {ref['run_id']} is a subset run (limit true); an EV run over it must select a "
                            "subset too (stores.limit or stores.include)")
    top_n = int(sl_meta["shortlist"]["top_n"])
    if int(ev["n_passages"]) > top_n:
        raise EvidenceError(f"evidence.n_passages {ev['n_passages']} exceeds the SL run's top_n {top_n}")
    n_keep = int(ev["n_passages"])

    def trimmed(fh):  # keep only the rows this run reads: the first n_passages of each list
        for raw in fh:
            x = json.loads(raw)
            yield {"claim_id": x["claim_id"], "claim_hash": x["claim_hash"],
                   **{SL_LIST[v]: x[SL_LIST[v]][:n_keep] for v in ev["variants"]}}

    with gzip.open(sl_dir / "shortlist.jsonl.gz", "rb") as fh:
        selected, subset = select_claims(trimmed(fh), ev["stores"])
    claims = load_claims(root / resolved["data"]["claims"])
    for x in selected:
        c = claims.get(x["claim_id"])
        if c is None or hashing.claim_hash(c["text"]) != x["claim_hash"]:
            raise EvidenceError(f"{x['claim_id']}: claim text in data.claims does not match the SL run's claim_hash")
    plans = plan_claims(selected, list(ev["variants"]), n_keep)
    del selected
    wanted = {p for pl in plans for p in pl["order"]}
    texts = read_cache_texts(out_root(root) / sl_meta["private_cache"], sl_meta["private_cache_sha256"], wanted)
    read_seconds = time.perf_counter() - t_in
    say(f"evidence: SL run {ref['run_id']} verified; {len(plans)} claims, {len(wanted)} passages to embed")

    # ---- determinism, then the run directory
    from aac.components.c1_train import CUBLAS_OK
    from aac.run.seeds import seed_everything

    seed_everything(seed)
    import torch

    strict = ev["determinism"] == "strict"
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    device = _device()
    if device == "cuda" and strict and cublas_at_start not in CUBLAS_OK:
        raise EvidenceError(f"CUBLAS_WORKSPACE_CONFIG must be one of {CUBLAS_OK} in the process environment before "
                            f"Python starts (got {cublas_at_start!r}); scripts/gpu_run_remote.sh exports it")

    key = (resolved["regime"], SYSTEM, resolved["name"], seed)
    run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, force)
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, lock_path(root))
    started = clock.now_iso_ms()
    priv = _make_private_dir(root, rid)

    t_load = time.perf_counter()
    dense = DenseEncoder(ev["dense"], root, device)
    reranker = Reranker(ev["reranker"], root, device)
    load_seconds = time.perf_counter() - t_load
    dim = int(dense.model.config.hidden_size)

    n_rows = sum(len(pl["order"]) for pl in plans)
    vec_path, idx_path = priv / VECTORS_FILE, priv / INDEX_FILE
    vec_tmp = vec_path.with_name(VECTORS_FILE + ".tmp")
    out = _GzLines(run_dir / EVIDENCE_JSONL)
    index = _GzLines(idx_path, mode=0o600, defer=True)
    pending = [index]
    k_dense, k = int(ev["k_dense"]), int(ev["k"])
    counts = {"n_claims": len(plans), "n_lists": 0, "n_passages_embedded": n_rows, "n_claims_embedded": 0,
              "n_rerank_pairs": 0, "n_lists_short": 0, "n_lists_empty": 0}
    embed_s = rerank_s = 0.0
    try:
        vecs = np.lib.format.open_memmap(vec_tmp, mode="w+", dtype=np.float16, shape=(n_rows, dim))
        os.chmod(vec_tmp, 0o600)
        start = 0
        for c0 in range(0, len(plans), EMBED_CHUNK_CLAIMS):
            chunk = plans[c0:c0 + EMBED_CHUNK_CLAIMS]
            ids = [p for pl in chunk for p in pl["order"]]
            t0 = time.perf_counter()
            v16 = dense.encode([texts[p] for p in ids], batch_size=int(ev["dense"]["batch_size"])).astype(np.float16)
            vecs[start:start + len(ids)] = v16
            claim_vecs = [dense.encode([claims[pl["claim_id"]]["text"]], batch_size=1)[0] for pl in chunk]
            embed_s += time.perf_counter() - t0
            counts["n_claims_embedded"] += len(chunk)
            row_of = {p: start + j for j, p in enumerate(ids)}
            # dense stage per list, then one re-rank call for every distinct (claim, passage) pair of the chunk
            dense_top: list[dict] = []
            pairs: list[tuple[str, str]] = []
            pair_at: dict[tuple[str, str], int] = {}
            for pl, cv in zip(chunk, claim_vecs, strict=True):
                tops = {}
                for v in ev["variants"]:
                    lst = pl["lists"][v]
                    counts["n_lists"] += 1
                    counts["n_lists_short"] += int(len(lst) < int(ev["n_passages"]))
                    counts["n_lists_empty"] += int(not lst)
                    rows = np.asarray([row_of[p] - start for p in lst], dtype=np.int64)
                    s = dense_scores(cv, v16[rows]) if lst else np.zeros(0)
                    tops[v] = rank(lst, s, k_dense)
                    for p, _ in tops[v]:
                        if (pl["claim_id"], p) not in pair_at:
                            pair_at[(pl["claim_id"], p)] = len(pairs)
                            pairs.append((claims[pl["claim_id"]]["text"], texts[p]))
                dense_top.append(tops)
            t0 = time.perf_counter()
            rr = reranker.score(pairs, batch_size=int(ev["reranker"]["batch_size"]))
            rerank_s += time.perf_counter() - t0
            counts["n_rerank_pairs"] += len(pairs)
            claim_start = start
            for pl, tops in zip(chunk, dense_top, strict=True):
                for v in ev["variants"]:
                    top_ids = [p for p, _ in tops[v]]
                    rs = [rr[pair_at[(pl["claim_id"], p)]] for p in top_ids]
                    out.write(_line({"claim_id": pl["claim_id"], "claim_hash": pl["claim_hash"], "variant": v,
                                     "dense": [[p, s6] for p, s6 in tops[v]],
                                     "rerank": [[p, s6] for p, s6 in rank(top_ids, rs, k)]}))
                index.write(_line({"claim_id": pl["claim_id"], "start": claim_start, "count": len(pl["order"]),
                                   "passage_ids": pl["order"]}))
                claim_start += len(pl["order"])
            start += len(ids)
            done = min(c0 + EMBED_CHUNK_CLAIMS, len(plans))
            if done % PROGRESS_EVERY < EMBED_CHUNK_CLAIMS or done == len(plans):
                say(f"evidence: {done}/{len(plans)} claims")
        if start != n_rows:
            raise RuntimeError(f"embedded {start} rows, planned {n_rows}")
        vecs.flush()
        del vecs
        evidence_sha = out.close()
        index_sha = index.close()
    except BaseException:
        out.abort()
        for w in pending:
            w.abort()
        vec_tmp.unlink(missing_ok=True)
        raise
    finally:
        dense.close()
        reranker.close()
    from aac.run.hashing import sha256_file

    vec_sha = sha256_file(vec_tmp)
    moves = [(vec_tmp, vec_path), (index.tmp, idx_path)]
    pooled_meta: dict = {"enabled": False}
    if ev["pooled"]["enabled"]:
        t0 = time.perf_counter()
        try:
            split = json.loads((root / resolved["data"]["split_manifest"]).read_text(encoding="utf-8"))
            built = build_pooled(plans, texts, vec_tmp, priv / POOL_DIR, list(ev["pooled"]["partitions"]), split,
                                 subset)
        except BaseException:
            for tmp, _ in moves:
                tmp.unlink(missing_ok=True)
            raise
        moves += built["tmp"]
        rel = PRIVATE_REL / rid / POOL_DIR
        pooled_meta = {**built["meta"], "passages_path": (rel / POOL_PASSAGES).as_posix(),
                       "vectors_path": (rel / VECTORS_FILE).as_posix()}
        pool_seconds = time.perf_counter() - t0
        say(f"evidence: pooled bundle {pooled_meta['n_passages']} passages from {pooled_meta['n_claims']} claims")
    # every stage succeeded: move the private files into place (0600 in a 0700 directory)
    for tmp, final in moves:
        os.chmod(tmp, 0o600)
        os.replace(tmp, final)
    finished = clock.now_iso_ms()

    result = {
        "schema": RESULT_SCHEMA, "run_id": rid, "evidence": ev, "limit": subset,
        "shortlist_run": {"run_id": ref["run_id"], "sealed_sha256": ref["sha256"],
                          "shortlist_sha256": sl_meta["shortlist_sha256"],
                          "private_cache_sha256": sl_meta["private_cache_sha256"], "limit": sl_meta["limit"],
                          "top_n": top_n, "reserve": int(sl_meta["shortlist"]["reserve"])},
        "counts": counts,
        "evidence_sha256": evidence_sha,
        "line_fields": ["claim_id", "claim_hash", "variant", "dense", "rerank"],
        "variant_lists": {v: SL_LIST[v] for v in ev["variants"]},
        "order_key": "(-round(score, 6), passage_id)",
        "dense_score": "float64 dot product of the float32 claim vector (batch size 1) with the stored fp16 passage vectors",
        "embed_chunk_claims": EMBED_CHUNK_CLAIMS,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "precision": {"configured": ev["dense"]["precision"], "applied": dense.precision_applied,
                      "reranker": "float32"},
        "private": {
            "vectors": {"path": (PRIVATE_REL / rid / VECTORS_FILE).as_posix(), "sha256": vec_sha,
                        "dtype": "float16", "shape": [n_rows, dim]},
            "index": {"path": (PRIVATE_REL / rid / INDEX_FILE).as_posix(), "sha256": index_sha,
                      "sha256_of": "decompressed bytes"},
        },
        "pooled": pooled_meta,
    }
    (run_dir / EVIDENCE_JSON).write_text(hashing.dumps_sorted(result), encoding="ascii")
    timings = [{"stage": "read-inputs", "seconds": round(read_seconds, DECIMALS), "items": len(plans)},
               {"stage": "load-models", "seconds": round(load_seconds, DECIMALS), "device": device},
               {"stage": "embed", "seconds": round(embed_s, DECIMALS), "device": device,
                "items": n_rows + counts["n_claims_embedded"]},
               {"stage": "rerank", "seconds": round(rerank_s, DECIMALS), "device": device,
                "items": counts["n_rerank_pairs"]}]
    if pooled_meta["enabled"]:
        timings.append({"stage": "pooled", "seconds": round(pool_seconds, DECIMALS), "items": pooled_meta["n_passages"]})
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for t in timings:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
    wall = clock.source == "wall"
    summary = {"configurations": [f"evidence/{resolved['name']}"],
               "items": {"claims": len(plans), "passages_embedded": n_rows, "rerank_pairs": counts["n_rerank_pairs"]},
               "stages": [{"name": "embed", "device": device, "seconds": round(embed_s, DECIMALS) if wall else 0.0,
                           "items": n_rows + counts["n_claims_embedded"]},
                          {"name": "rerank", "device": device, "seconds": round(rerank_s, DECIMALS) if wall else 0.0,
                           "items": counts["n_rerank_pairs"]}],
               "depends_on": [ref["run_id"]]}
    determinism = (f"seeded; torch deterministic algorithms {'strict' if strict else 'warn-only'}; BGE-M3 own CLS "
                   f"pooling + L2 ({dense.precision_applied}), reranker float32 sigmoid; fixed batch plans; scores "
                   "rounded to 6 d.p. before the order key (-score6, passage_id)")
    digest = finalize_run(root, run_dir, ident, summary=summary, record_count=0, shards=[], clock=clock,
                          started=started, finished=finished, system=SYSTEM, determinism_level=determinism,
                          target_dir=target_dir, stale_ids=stale_ids)
    return {"run_id": rid, "claims": len(plans), "evidence_sha256": evidence_sha, "passages_embedded": n_rows,
            "sealed": digest, "run_dir": target_dir, "device": device}


# ---------------------------------------------------------------- pooled_rd 

POOL_RANK_EPS = 2e-6   # candidate margin of the BM25 top-n (the S1 rule: every row whose rounded score can tie the n-th)


def pool_rank(scores, k: int) -> list[tuple[int, float]]:
    """[(pool_idx, score6)] of the first k pooled passages by (-round(score, 6), pool_idx)."""
    s_all = np.asarray(scores, dtype=np.float64).reshape(-1)
    n = s_all.size
    if k <= 0 or n == 0:
        return []
    idx = np.arange(n)
    s = s_all
    if n > k:
        kth = np.partition(s, n - k)[n - k]
        keep = s >= kth - POOL_RANK_EPS
        idx, s = idx[keep], s[keep]
    r = np.fromiter((round(float(x), DECIMALS) for x in s), dtype=np.float64, count=s.size)
    order = np.lexsort((idx, -r))[:k]
    return [(int(idx[o]), float(r[o])) for o in order]


def read_pool_run(root: Path, ref: dict) -> tuple[dict, list[str], list[str], np.ndarray]:
    """(evidence.json, pooled passage ids, texts, fp16 vectors memory-mapped) of the per_claim EV run `ref`, after its
 sealed digest equals ref["sha256"] and the pooled files equal their recorded sha256."""
    from aac.run.hashing import sha256_file
    from aac.run.paths import out_root, runs_dir
    from aac.run.seal import verify_sealed

    d = runs_dir(root) / ref["run_id"]
    if not (d / "SEALED").is_file():
        raise EvidenceError(f"pool EV run {ref['run_id']} has no SEALED run directory under {runs_dir(root)}")
    digest = verify_sealed(d)
    if digest != ref["sha256"]:
        raise EvidenceError(f"pool EV run {ref['run_id']}: sealed digest {digest} != ref {ref['sha256']}; refusing it")
    meta = json.loads((d / EVIDENCE_JSON).read_text(encoding="ascii"))
    pooled = meta.get("pooled") or {}
    if meta["evidence"]["mode"] != "per_claim" or not pooled.get("enabled"):
        raise EvidenceError(f"pool EV run {ref['run_id']} has no pooled bundle (per_claim with pooled.enabled true)")
    h = hashlib.sha256()
    ids: list[str] = []
    texts: list[str] = []
    with gzip.open(out_root(root) / pooled["passages_path"], "rb") as fh:
        for raw in fh:
            h.update(raw)
            e = json.loads(raw)
            if int(e["pool_idx"]) != len(ids):
                raise EvidenceError(f"pool EV run {ref['run_id']}: pooled passages are not in pool_idx order")
            ids.append(e["passage_id"])
            texts.append(e["text"])
    if h.hexdigest() != pooled["passages_sha256"]:
        raise EvidenceError(f"pool EV run {ref['run_id']}: pooled passages sha256 {h.hexdigest()} != recorded "
                            f"{pooled['passages_sha256']}; refusing them")
    vpath = out_root(root) / pooled["vectors_path"]
    if sha256_file(vpath) != pooled["vectors_sha256"]:
        raise EvidenceError(f"pool EV run {ref['run_id']}: pooled vectors sha256 differs from evidence.json")
    vecs = np.load(vpath, mmap_mode="r")
    if list(vecs.shape) != list(pooled["vectors_shape"]) or vecs.shape[0] != len(ids):
        raise EvidenceError(f"pool EV run {ref['run_id']}: pooled vectors shape {list(vecs.shape)} does not match")
    return meta, ids, texts, vecs


def bm25_candidates(pool_texts: list[str], queries: list[str], top_n: int) -> list[list[tuple[int, float]]]:
    """Per query: [(pool_idx, bm25 score6)] of the first top_n pooled passages (the S1 BM25 recipe)."""
    import bm25s
    import Stemmer

    from aac.data.shortlist import BM25_PARAMS, STEMMER, STOPWORDS

    stemmer = Stemmer.Stemmer(STEMMER)
    corpus = bm25s.tokenize(pool_texts, stopwords=STOPWORDS, stemmer=stemmer, show_progress=False)
    bm = bm25s.BM25(**BM25_PARAMS)
    bm.index(corpus, show_progress=False)
    del corpus
    out = []
    for q in queries:
        toks = bm25s.tokenize([q], stopwords=STOPWORDS, stemmer=stemmer, return_ids=False, show_progress=False)[0]
        ids = bm.get_tokens_ids(list(toks))
        scores = np.asarray(bm.get_scores_from_ids(ids), dtype=np.float64) if ids else np.zeros(len(pool_texts))
        if scores.shape != (len(pool_texts),):
            raise EvidenceError(f"bm25s returned {scores.shape} scores for {len(pool_texts)} pooled passages")
        out.append(pool_rank(scores, top_n))
    return out


def run_pooled_rd(config: Path, seed: int, *, force: bool, say, cublas_at_start) -> dict:
    from aac.data.evidence_providers import load_incrediblae_items
    from aac.data.shortlist import BM25_PARAMS, STEMMER, STOPWORDS
    from aac.run import hashing
    from aac.run.clock import make_clock
    from aac.run.env import write_env
    from aac.run.finalize import claim_run_dir, finalize_run
    from aac.run.hashing import sha256_file
    from aac.run.identity import identity, lock_path
    from aac.run.paths import root as project_root

    root = project_root()
    ident = identity(root, Path(config), seed)
    resolved, rid = ident["resolved"], ident["run_id"]
    ev = resolved["evidence"]
    ref = ev["pool"]
    if list(ev["variants"]) != ["primary"]:
        raise EvidenceError(f"evidence.mode pooled_rd writes one list: variants must be [primary], not {ev['variants']}")
    if ev["stores"]["limit"] is not None or ev["stores"]["include"]:
        raise EvidenceError("evidence.mode pooled_rd reads every RD item of data.rd_items; stores must select nothing")
    rd_rel = resolved["data"].get("rd_items")
    if not rd_rel:
        raise EvidenceError("evidence.mode pooled_rd needs data.rd_items")

    # ---- inputs first: nothing is claimed or written when a hash or a text check fails
    t_in = time.perf_counter()
    pool_meta, pool_ids, pool_texts, pool_vecs = read_pool_run(root, ref)
    items = load_incrediblae_items(root, rd_rel)      # LaneBMissing off cpu-host; a changed text raises (item id only)
    item_ids = list(items)
    rd_texts = [items[i]["text"] for i in item_ids]
    rd_sha = sha256_file(root / rd_rel)
    read_seconds = time.perf_counter() - t_in
    say(f"evidence: pool EV run {ref['run_id']} verified; {len(pool_ids)} pooled passages, {len(item_ids)} RD items")
    t0 = time.perf_counter()
    cands = bm25_candidates(pool_texts, rd_texts, int(ev["bm25_top_n"]))
    bm25_seconds = time.perf_counter() - t0

    from aac.components.c1_train import CUBLAS_OK
    from aac.run.seeds import seed_everything

    seed_everything(seed)
    import torch

    strict = ev["determinism"] == "strict"
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    device = _device()
    if device == "cuda" and strict and cublas_at_start not in CUBLAS_OK:
        raise EvidenceError(f"CUBLAS_WORKSPACE_CONFIG must be one of {CUBLAS_OK} in the process environment before "
                            f"Python starts (got {cublas_at_start!r}); scripts/gpu_run_remote.sh exports it")
    key = (resolved["regime"], SYSTEM, resolved["name"], seed)
    run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, force)
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, lock_path(root))
    started = clock.now_iso_ms()

    t_load = time.perf_counter()
    dense = DenseEncoder(ev["dense"], root, device)
    reranker = Reranker(ev["reranker"], root, device)
    load_seconds = time.perf_counter() - t_load
    k_dense, k = int(ev["k_dense"]), int(ev["k"])
    counts = {"n_items": len(item_ids), "n_items_empty_text": sum(not t for t in rd_texts),
              "n_pool_passages": len(pool_ids), "n_bm25_candidates": sum(len(c) for c in cands),
              "n_items_no_candidates": sum(not c for c in cands), "n_rerank_pairs": 0}
    idx_of = {pid: n for n, pid in enumerate(pool_ids)}
    out = _GzLines(run_dir / EVIDENCE_JSONL)
    embed_s = rerank_s = 0.0
    try:
        for c0 in range(0, len(item_ids), EMBED_CHUNK_CLAIMS):
            chunk = list(range(c0, min(c0 + EMBED_CHUNK_CLAIMS, len(item_ids))))
            t0 = time.perf_counter()
            tops = []
            for i in chunk:
                rows = [pi for pi, _ in cands[i]]
                if rows:
                    cv = dense.encode([rd_texts[i]], batch_size=1)[0]
                    s = dense_scores(cv, np.asarray(pool_vecs[np.asarray(rows, dtype=np.int64)]))
                    tops.append(rank([pool_ids[r] for r in rows], s, k_dense))
                else:
                    tops.append([])
            embed_s += time.perf_counter() - t0
            pairs = [(rd_texts[i], pool_texts[idx_of[p]]) for i, top in zip(chunk, tops, strict=True) for p, _ in top]
            t0 = time.perf_counter()
            rr = reranker.score(pairs, batch_size=int(ev["reranker"]["batch_size"]))
            rerank_s += time.perf_counter() - t0
            counts["n_rerank_pairs"] += len(pairs)
            j = 0
            for i, top in zip(chunk, tops, strict=True):
                ids = [p for p, _ in top]
                rs = rr[j:j + len(ids)]
                j += len(ids)
                out.write(_line({"claim_id": item_ids[i], "claim_hash": hashing.claim_hash(rd_texts[i]),
                                 "variant": "primary", "dense": [[p, s6] for p, s6 in top],
                                 "rerank": [[p, s6] for p, s6 in rank(ids, rs, k)]}))
            say(f"evidence: {chunk[-1] + 1}/{len(item_ids)} RD items")
        evidence_sha = out.close()
    except BaseException:
        out.abort()
        raise
    finally:
        dense.close()
        reranker.close()
    finished = clock.now_iso_ms()
    result = {
        "schema": RESULT_SCHEMA, "run_id": rid, "evidence": ev, "limit": bool(pool_meta["limit"]),
        "pool_run": {"run_id": ref["run_id"], "sealed_sha256": ref["sha256"],
                     "passages_sha256": pool_meta["pooled"]["passages_sha256"],
                     "vectors_sha256": pool_meta["pooled"]["vectors_sha256"],
                     "n_passages": len(pool_ids), "partitions": pool_meta["pooled"]["partitions"],
                     "limit": bool(pool_meta["limit"])},
        "rd_items": {"path": Path(rd_rel).as_posix(), "sha256": rd_sha, "n_items": len(item_ids)},
        "counts": counts,
        "evidence_sha256": evidence_sha,
        "line_fields": ["claim_id", "claim_hash", "variant", "dense", "rerank"],
        "bm25": {"library": "bm25s", **BM25_PARAMS, "stopwords": STOPWORDS, "stemmer": f"PyStemmer {STEMMER}",
                 "query": "the whole RD text", "order_key": "(-round(score, 6), pool_idx)"},
        "order_key": "(-round(score, 6), passage_id)",
        "dense_score": "float64 dot product of the float32 RD-text vector (batch size 1, truncated at "
                       "dense.max_seq_length) with the stored fp16 pooled vectors",
        "embed_chunk_claims": EMBED_CHUNK_CLAIMS,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "precision": {"configured": ev["dense"]["precision"], "applied": dense.precision_applied,
                      "reranker": "float32"},
        "private": {},
        "pooled": {"enabled": False},
    }
    (run_dir / EVIDENCE_JSON).write_text(hashing.dumps_sorted(result), encoding="ascii")
    timings = [{"stage": "read-inputs", "seconds": round(read_seconds, DECIMALS), "items": len(item_ids)},
               {"stage": "bm25", "seconds": round(bm25_seconds, DECIMALS), "device": "cpu", "items": len(item_ids)},
               {"stage": "load-models", "seconds": round(load_seconds, DECIMALS), "device": device},
               {"stage": "embed", "seconds": round(embed_s, DECIMALS), "device": device, "items": len(item_ids)},
               {"stage": "rerank", "seconds": round(rerank_s, DECIMALS), "device": device,
                "items": counts["n_rerank_pairs"]}]
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for t in timings:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
    wall = clock.source == "wall"
    summary = {"configurations": [f"evidence/{resolved['name']}"],
               "items": {"rd_items": len(item_ids), "pool_passages": len(pool_ids),
                         "rerank_pairs": counts["n_rerank_pairs"]},
               "stages": [{"name": "bm25", "device": "cpu", "seconds": round(bm25_seconds, DECIMALS) if wall else 0.0,
                           "items": len(item_ids)},
                          {"name": "embed", "device": device, "seconds": round(embed_s, DECIMALS) if wall else 0.0,
                           "items": len(item_ids)},
                          {"name": "rerank", "device": device, "seconds": round(rerank_s, DECIMALS) if wall else 0.0,
                           "items": counts["n_rerank_pairs"]}],
               "depends_on": [ref["run_id"]]}
    determinism = (f"seeded; torch deterministic algorithms {'strict' if strict else 'warn-only'}; bm25s BM25 over the "
                   f"pooled bundle (order key (-score6, pool_idx)); BGE-M3 own CLS pooling + L2 "
                   f"({dense.precision_applied}), reranker float32 sigmoid; fixed batch plans; scores rounded to 6 d.p. "
                   "before the order key (-score6, passage_id)")
    digest = finalize_run(root, run_dir, ident, summary=summary, record_count=0, shards=[], clock=clock,
                          started=started, finished=finished, system=SYSTEM, determinism_level=determinism,
                          target_dir=target_dir, stale_ids=stale_ids)
    return {"run_id": rid, "mode": "pooled_rd", "claims": len(item_ids), "pool_passages": len(pool_ids),
            "evidence_sha256": evidence_sha, "passages_embedded": 0, "sealed": digest, "run_dir": target_dir,
            "device": device}


# ---------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    from aac.data.evidence_providers import ItemTextMismatch
    from aac.data.incrediblae import LaneBMissing
    from aac.run.finalize import FinalizeRefused, RunRefused

    ap = argparse.ArgumentParser(prog="python -m aac.components.c2_evidence_run")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="seal an EV run from an aac-evidence/1 config")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    try:
        res = run(Path(a.config), a.seed, force=a.force, progress=lambda m: print(m, flush=True))
    except RunRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    except (EvidenceError, FinalizeRefused, LookupError, LaneBMissing, ItemTextMismatch) as exc:
        print(f"evidence: FAIL: {exc}", file=sys.stderr)
        return 1
    if res.get("mode") == "pooled_rd":
        print(f"evidence {res['run_id']}: {res['claims']} RD items over {res['pool_passages']} pooled passages on "
              f"{res['device']}, SEALED {res['sealed'][:12]}")
    else:
        print(f"evidence {res['run_id']}: {res['claims']} claims, {res['passages_embedded']} passages embedded on "
              f"{res['device']}, SEALED {res['sealed'][:12]}")
    print(f"evidence: run_id={res['run_id']} evidence_sha256={res['evidence_sha256']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
