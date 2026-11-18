"""BENCH run kind : retrieval-model throughput, bit-identity and batch-1 latency.

`python -m aac.bench.retrieval run --config <aac-bench/1 yaml> --seed N [--force]` seals
`BENCH-<hash10>-s<seed>` (record_count 0) whose bench.json records, for every configured model, precision
and batch size, passages/s, tokens/s (tokenizer counts after truncation, special tokens included), peak GPU
memory and wall seconds over the same input; the bit-identity of repeated passes; and batch-1 latency.

Input ("Benchmarks and determinism double run"): the first `n_passages` distinct url2text
sentences of the stores of one zip, in claim order, streamed through aac.data.kstore.iter_docs (never
extracted; exact-string dedup, first occurrence kept, blank sentences skipped). The text is written ONLY to
<out_root>/data/derived/private/bench-input.jsonl (mode 0600, gitignored, excluded from backup and from
gpu-push); the run directory records its sha256 and counts, never a sentence .

Nothing here promises bit-identical GPU re-inference (pre-registered specification
they measure, and a mismatch is a recorded result."""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from aac.data import kstore
from aac.run import hashing
from aac.run import paths as aac_paths

PRIVATE_REL = Path("data/derived/private")
INPUT_FILE = "bench-input.jsonl"
BENCH_JSON = "bench.json"
BENCH_SYSTEM = "bench"           # run-record `system` of a benchmark run (index key: BENCH, bench, name, seed)
RESULT_SCHEMA = "aac-bench-result/1"
LATENCY_WARMUP = 10              # untimed calls before every latency stage 
DECIMALS = 6
# Identity measurements keep full float precision: rounding 1e-7 to 0.000000 would hide a mismatch.
UNROUNDED_KEYS = frozenset({"max_abs_diff", "min_cosine"})
DETERMINISM = ("seeded; torch-deterministic-warn-only; inference only, precisions as configured; "
               "bit-identity measured and recorded, not promised (specification item 17)")


# --------------------------------------------------------------------------------------------- input


def private_dir(root: Path) -> Path:
    """<out_root>/data/derived/private: the smoke (AAC_OUT=build/bench-smoke) never touches the real one."""
    return aac_paths.out_root(root) / PRIVATE_REL


def input_path(root: Path) -> Path:
    return private_dir(root) / INPUT_FILE


def zip_path(root: Path, inp: dict) -> Path:
    base = Path(root) / "data" / "raw" if inp["zip_base"] == "data_raw" else Path(root)
    return base / inp["zip"]


def _write_private(path: Path, lines: list[str]) -> str:
    """Write the private input (dir 0700, file 0600, tmp + os.replace) and return its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        fh.writelines(lines)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return hashing.sha256_file(path)


def build_input(root: Path, resolved: dict, stats: dict | None = None) -> tuple[list[str], list[str], str]:
    """(texts, claim_ids, sha256 of the private input file). Stores are read in claim order (list_members),
 at most `max_stores`; within a store, documents in line order and sentences in url2text order; the first
 occurrence of each exact sentence string is kept until `n_passages`. Fewer distinct sentences than
 `n_passages` raises ValueError (the benchmark never runs on a silently smaller input)."""
    inp = resolved["bench"]["input"]
    target = int(inp["n_passages"])
    st = stats if stats is not None else {}
    for k in ("n_stores_read", "n_docs_read", "n_sentences_seen", "n_duplicates_skipped", "n_blank_skipped",
              "n_lines_unparsable"):
        st.setdefault(k, 0)
    seen: set[str] = set()
    texts: list[str] = []
    cids: list[str] = []
    with zipfile.ZipFile(zip_path(root, inp)) as zf:
        for cid, member in kstore.list_members(zf)[: int(inp["max_stores"])]:
            if len(texts) >= target:
                break
            st["n_stores_read"] += 1
            docs = kstore.iter_docs(zf, member, st)
            try:
                for doc in docs:
                    st["n_docs_read"] += 1
                    for s in doc.sentences:
                        st["n_sentences_seen"] += 1
                        if not s.strip():
                            st["n_blank_skipped"] += 1
                            continue
                        if s in seen:
                            st["n_duplicates_skipped"] += 1
                            continue
                        seen.add(s)
                        texts.append(s)
                        cids.append(cid)
                        if len(texts) >= target:
                            break
                    if len(texts) >= target:
                        break
            finally:
                docs.close()
    if len(texts) < target:
        raise ValueError(f"bench input: {len(texts)} distinct sentences in the first {st['n_stores_read']} stores, "
                         f"fewer than n_passages {target}")
    lines = [json.dumps({"claim_id": c, "text": t}, ensure_ascii=False, sort_keys=True) + "\n"
             for c, t in zip(cids, texts)]
    sha = _write_private(input_path(root), lines)
    return texts, cids, sha


def read_input(path: Path, sha256: str) -> tuple[list[str], list[str]]:
    """(texts, claim_ids) of a private input file whose sha256 must equal `sha256`."""
    actual = hashing.sha256_file(path)
    if actual != sha256:
        raise ValueError(f"bench input sha256 {actual} != expected {sha256}")
    texts, cids = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            texts.append(rec["text"])
            cids.append(rec["claim_id"])
    return texts, cids


def load_claims(root: Path, data: dict) -> dict[str, dict]:
    if data["dataset"] == "averitec":
        from aac.data.averitec import load_claims as load_averitec_claims

        return load_averitec_claims(Path(root) / data["claims"])
    from aac.data.fixtures import load_claims as load_fixture_claims

    return load_fixture_claims(Path(root) / data["claims"])


def _claim_order(cid: str) -> tuple[int, int]:
    _, split, idx = cid.split("/")
    return kstore.SPLIT_RANK[split], int(idx)


def latency_claims(claims: dict[str, dict], first_cid: str) -> list[str]:
    """Claim ids of the input's split (e.g. averitec/dev/*) in claim order: the c1 latency queries."""
    prefix = first_cid.rsplit("/", 1)[0] + "/"
    return sorted((c for c in claims if c.startswith(prefix)), key=_claim_order)


def items_for(loader: str, texts: list[str], cids: list[str], claims: dict[str, dict]) -> list:
    """Encoder inputs: the passages (bi-encoder) or (own claim, passage) pairs (cross-encoder)."""
    if loader == "sentence_transformer":
        return list(texts)
    return [(claims[c]["text"], t) for c, t in zip(cids, texts)]


# --------------------------------------------------------------------------------------------- models


def device_of() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.synchronize()


def _offline(root: Path) -> str:
    os.environ.setdefault("HF_HOME", str(Path(root) / ".hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers.utils import logging as hf_logging

    from aac.components.c1_train import hf_cache_dir

    hf_logging.set_verbosity_error()
    return hf_cache_dir(root)


class BenchModel:
    """One pinned retrieval model at one precision: `encode(batch)` -> float32 numpy (embeddings, or sigmoid
 scores for the cross-encoder) and `token_counts(items)` (tokenizer counts after truncation)."""

    def __init__(self, mcfg: dict, root: Path, device: str, precision: str) -> None:
        import torch

        cache = _offline(root)
        self.key, self.loader = mcfg["key"], mcfg["loader"]
        self.max_seq_length = int(mcfg["max_seq_length"])
        self.precision, self.device = precision, device
        kw = {"revision": mcfg["revision"], "cache_folder": cache, "local_files_only": True, "device": device}
        if self.loader == "sentence_transformer":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(mcfg["hf_id"], **kw)
        else:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(mcfg["hf_id"], max_length=self.max_seq_length, **kw)
        self.model.max_seq_length = self.max_seq_length
        if precision == "bfloat16":
            self.model.to(torch.bfloat16)
        self.model.eval()

    def token_counts(self, items: list) -> list[int]:
        tok = self.model.tokenizer
        if self.loader == "sentence_transformer":
            enc = tok(list(items), truncation=True, max_length=self.max_seq_length)
        else:
            enc = tok([a for a, _ in items], [b for _, b in items], truncation=True, max_length=self.max_seq_length)
        return [len(x) for x in enc["input_ids"]]

    def encode(self, batch: list) -> np.ndarray:
        import torch

        with torch.inference_mode():
            if self.loader == "sentence_transformer":
                out = self.model.encode(list(batch), batch_size=len(batch), convert_to_tensor=True,
                                        show_progress_bar=False)
            else:
                out = self.model.predict(list(batch), batch_size=len(batch), convert_to_tensor=True,
                                         show_progress_bar=False)
        return out.float().cpu().numpy()

    def close(self) -> None:
        del self.model
        if self.device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


def batch_plan(counts: list[int], batch_size: int) -> list[list[int]]:
    """Indices sorted by token count (longest first, ties by index), cut into fixed batches: every pass
 over the same input sees identical batches."""
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def encode_pass(model, items: list, plan: list[list[int]]) -> np.ndarray:
    """One pass over `plan`; rows returned in input order."""
    out: np.ndarray | None = None
    for b in plan:
        arr = model.encode([items[i] for i in b])
        if out is None:
            out = np.empty((len(items),) + arr.shape[1:], dtype=arr.dtype)
        out[b] = arr
    assert out is not None
    return out


def throughput(model, items: list, precision: str, batch_size: int, device: str,
               counts: list[int] | None = None) -> dict:
    """One untimed warm-up batch (the longest), then one timed pass over the whole input."""
    counts = counts if counts is not None else model.token_counts(items)
    plan = batch_plan(counts, batch_size)
    model.encode([items[i] for i in plan[0]])
    _sync(device)
    peak = None
    if device.startswith("cuda"):
        import torch

        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    encode_pass(model, items, plan)
    _sync(device)
    wall = time.perf_counter() - t0
    if device.startswith("cuda"):
        import torch

        peak = int(torch.cuda.max_memory_allocated())
    n, n_tok = len(items), int(sum(counts))
    return {"model": model.key, "precision": precision, "batch_size": int(batch_size), "status": "measured",
            "n_passages": n, "n_tokens": n_tok, "n_batches": len(plan), "warmup_batches": 1,
            "wall_seconds": wall, "passages_per_s": n / wall, "tokens_per_s": n_tok / wall,
            "peak_memory_bytes": peak}


# --------------------------------------------------------------------------------------------- identity


def array_identity(a, b) -> dict:
    """{equal (np.array_equal), max_abs_diff, min_cosine (row-wise, 2-D only; None for 1-D scores)}."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return {"equal": False, "max_abs_diff": None, "min_cosine": None,
                "shape": [list(a.shape), list(b.shape)]}
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    max_abs = float(np.max(np.abs(a64 - b64))) if a.size else 0.0
    min_cos = None
    if a.ndim == 2 and a.shape[0]:
        na, nb = np.linalg.norm(a64, axis=1), np.linalg.norm(b64, axis=1)
        dots = np.sum(a64 * b64, axis=1)
        both_zero = (na == 0) & (nb == 0)
        denom = np.where(na * nb == 0, 1.0, na * nb)
        cos = np.where(both_zero, 1.0, np.where(na * nb == 0, 0.0, dots / denom))
        # identical rows are cosine 1.0 exactly (the float64 dot/norm ratio can land one ulp below)
        cos = np.where(np.all(a64 == b64, axis=1), 1.0, np.clip(cos, -1.0, 1.0))
        min_cos = float(np.min(cos))
    return {"equal": bool(np.array_equal(a, b)), "max_abs_diff": max_abs, "min_cosine": min_cos}


def npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return buf.getvalue()


def bit_identity(model, items: list, precision: str, batch_size: int, counts: list[int]) -> tuple[dict, np.ndarray]:
    """Two passes in this process over identical batches. Returns (result, first-pass array)."""
    plan = batch_plan(counts, batch_size)
    first = encode_pass(model, items, plan)
    second = encode_pass(model, items, plan)
    res = {"model": model.key, "precision": precision, "batch_size": int(batch_size), "status": "measured",
           "in_process": array_identity(first, second),
           "first_pass_npy_sha256": hashing.sha256_bytes(npy_bytes(first))}
    return res, first


def second_process_result(parent: np.ndarray, npy_path: Path) -> dict:
    """Compare the parent's first-pass array with the .npy a fresh process wrote: array identity plus the
 sha256 of both .npy encodings (equal sha256 <=> equal dtype, shape and bytes)."""
    child = np.load(npy_path, allow_pickle=False)
    p_sha = hashing.sha256_bytes(npy_bytes(parent))
    c_sha = hashing.sha256_file(npy_path)
    return {**array_identity(parent, child), "npy_sha256": c_sha, "parent_npy_sha256": p_sha,
            "npy_sha256_equal": p_sha == c_sha}


def run_second_pass(config: Path, key: str, precision: str, batch_size: int, *, seed: int, out_dir: Path,
                    input_sha256: str | None = None) -> tuple[np.ndarray, Path]:
    """Encode the same input once more in a FRESH process (`python -m aac.bench.retrieval second-pass`, same
 environment, same seed, same length-sorted batches) and return (its array, the .npy it wrote)."""
    out = Path(out_dir) / f"second-pass-{key}-{precision}-b{int(batch_size)}.npy"
    cmd = [sys.executable, "-m", "aac.bench.retrieval", "second-pass", "--config", str(config), "--model", key,
           "--precision", precision, "--batch-size", str(int(batch_size)), "--seed", str(int(seed)), "--out", str(out)]
    if input_sha256:
        cmd += ["--input-sha256", input_sha256]
    proc = subprocess.run(cmd, env=dict(os.environ), capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"second-pass {key}/{precision}/b{batch_size} exited {proc.returncode}: {proc.stderr[-2000:]}")
    return np.load(out, allow_pickle=False), out


# --------------------------------------------------------------------------------------------- latency


def latency_summary(timings: list[float]) -> dict:
    """median and p95 (numpy percentile, method `linear`) of per-call seconds, 6 d.p."""
    t = np.asarray(timings, dtype=np.float64)
    if not t.size:
        raise ValueError("latency_summary needs at least one timing")
    return {"n_queries": int(t.size), "median_s": round(float(np.percentile(t, 50, method="linear")), DECIMALS),
            "p95_s": round(float(np.percentile(t, 95, method="linear")), DECIMALS)}


def time_calls(fn, calls: list, device: str, warmup: int = LATENCY_WARMUP) -> list[float]:
    """`warmup` untimed calls (the first calls cycled), then one timed call per element of `calls`."""
    for i in range(warmup):
        fn(calls[i % len(calls)])
    _sync(device)
    out = []
    for c in calls:
        t0 = time.perf_counter()
        fn(c)
        _sync(device)
        out.append(time.perf_counter() - t0)
    return out


def _cycle(seq: list, n: int) -> list:
    return [seq[i % len(seq)] for i in range(n)]


def latency_bge_m3(model, texts: list[str], queries: int, device: str) -> dict:
    """BGE-M3 single-sentence encoding: one passage per call, input order (cycled)."""
    timings = time_calls(lambda t: model.encode([t]), _cycle(texts, queries), device)
    return {"stage": "bge-m3", "model": model.key, "precision": model.precision, "batch_size": 1, "status": "measured",
            "unit": "one passage per call", "warmup": LATENCY_WARMUP, **latency_summary(timings)}


def latency_c1(encoder, claims: dict[str, dict], order: list[str], queries: int, device: str) -> dict:
    """C1 batch 1: encoder.encode on single claims in claim order (cycled), as Pipeline.infer calls it."""
    calls = [SimpleNamespace(claim_id=c, text=claims[c]["text"]) for c in _cycle(order, queries)]
    timings = time_calls(encoder.encode, calls, device)
    return {"stage": "c1", "model": getattr(encoder, "name", "c1"), "precision": "float32", "batch_size": 1,
            "status": "measured", "unit": "one claim per call (MdebertaEncoder.encode)", "warmup": LATENCY_WARMUP,
            **latency_summary(timings)}


def latency_reranker(model, texts: list[str], cids: list[str], claims: dict[str, dict], queries: int, k: int,
                     device: str) -> dict:
    """Reranker call: one claim with k passages per call. Query q takes the k passages starting at (q*k) mod n
 (wrapping) and the claim of the window's first passage."""
    n = len(texts)
    calls = []
    for q in range(queries):
        start = (q * k) % n
        claim = claims[cids[start]]["text"]
        calls.append([(claim, texts[(start + j) % n]) for j in range(k)])
    timings = time_calls(model.encode, calls, device)
    return {"stage": "reranker", "model": model.key, "precision": model.precision, "batch_size": int(k),
            "status": "measured", "unit": f"one claim with {int(k)} passages per call", "warmup": LATENCY_WARMUP,
            **latency_summary(timings)}


# --------------------------------------------------------------------------------------------- run


def _round(obj, key: str | None = None):
    if isinstance(obj, float) and key not in UNROUNDED_KEYS:
        return round(obj, DECIMALS)
    if isinstance(obj, dict):
        return {k: _round(v, k) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v, key) for v in obj]
    return obj


def _c1_model_cfg(root: Path, bench: dict) -> dict:
    """The C1 encoder config of the checkpoint run (hf_id, revision, num_labels, max_length from its model.json)."""
    ref = bench["c1"]["checkpoint"]
    model = json.loads((aac_paths.runs_dir(root) / ref["run_id"] / "model.json").read_text(encoding="utf-8"))
    return {"enabled": True, "kind": "mdeberta-v3-base", "hf_id": model["hf_id"], "revision": model["revision"],
            "num_labels": int(model["num_labels"]), "max_length": int(model["max_length"]), "checkpoint": ref}


def run_bench(resolved: dict, root: Path, run_dir: Path, seed: int, clock) -> dict:
    """Measure everything, write run_dir/bench.json and return the run summary for finalize_run."""
    import torch

    from aac.run.seeds import seed_everything

    t_start = time.perf_counter()
    seed_everything(seed)
    bench = resolved["bench"]
    device = device_of()
    claims = load_claims(root, resolved["data"])
    in_stats: dict = {}
    texts, cids, sha = build_input(root, resolved, in_stats)
    missing = sorted({c for c in cids if c not in claims})
    if missing:
        raise ValueError(f"bench input: {len(missing)} store claim ids have no claim text in data.claims "
                         f"(first {missing[0]})")
    timings: list[dict] = [{"stage": "build_input", "seconds": round(time.perf_counter() - t_start, DECIMALS),
                            "items": len(texts)}]
    tokens: dict = {}
    thr: list[dict] = []
    ident: list[dict] = []
    lat: list[dict] = []
    latency_cfg = bench["latency"]
    for mcfg in bench["models"]:
        items = items_for(mcfg["loader"], texts, cids, claims)
        for precision in bench["precisions"]:
            if precision == "bfloat16" and device == "cpu":
                for bs in bench["batch_sizes"]:
                    thr.append({"model": mcfg["key"], "precision": precision, "batch_size": int(bs),
                                "status": "skipped: cpu"})
                ident.append({"model": mcfg["key"], "precision": precision, "batch_size": int(bench["batch_sizes"][0]),
                              "status": "skipped: cpu"})
                continue
            t0 = time.perf_counter()
            model = BenchModel(mcfg, root, device, precision)
            counts = model.token_counts(items)
            tokens.setdefault(mcfg["key"], {
                "n_tokens": int(sum(counts)), "mean_tokens_per_item": sum(counts) / len(counts),
                "max_tokens_per_item": int(max(counts)), "max_seq_length": int(mcfg["max_seq_length"]),
                "item": "passage" if mcfg["loader"] == "sentence_transformer" else "(own claim, passage) pair",
                "count_rule": "tokenizer input_ids after truncation, special tokens included"})
            for bs in bench["batch_sizes"]:
                thr.append(throughput(model, items, precision, int(bs), device, counts))
            bs0 = int(bench["batch_sizes"][0])
            res, first = bit_identity(model, items, precision, bs0, counts)
            if precision == "float32" and mcfg["loader"] == "sentence_transformer" and "bge-m3" in latency_cfg["stages"]:
                lat.append(latency_bge_m3(model, texts, int(latency_cfg["queries"]), device))
            if precision == "float32" and mcfg["loader"] == "cross_encoder" and "reranker" in latency_cfg["stages"]:
                lat.append(latency_reranker(model, texts, cids, claims, int(latency_cfg["queries"]),
                                            int(latency_cfg["reranker_pairs_per_query"]), device))
            model.close()  # free the device before the fresh process loads its own copy
            with tempfile.TemporaryDirectory(dir=private_dir(root)) as tmp:
                _child, npy = run_second_pass(run_dir / "config.resolved.json", mcfg["key"], precision, bs0, seed=seed,
                                              out_dir=Path(tmp), input_sha256=sha)
                res["second_process"] = second_process_result(first, npy)
            ident.append(res)
            timings.append({"stage": f"{mcfg['key']}/{precision}", "seconds": round(time.perf_counter() - t0, DECIMALS),
                            "items": len(items)})
    for stage in ("bge-m3", "reranker"):
        if stage in latency_cfg["stages"] and not any(x["stage"] == stage for x in lat):
            lat.append({"stage": stage, "status": "not measured: no float32 pass of that model in this config"})
    c1 = bench["c1"]
    ref = c1["checkpoint"]
    if "c1" in latency_cfg["stages"]:
        from aac.components.c1_encoder import MdebertaEncoder

        t0 = time.perf_counter()
        encoder = MdebertaEncoder(_c1_model_cfg(root, bench), root, seed)
        lat.insert(0, latency_c1(encoder, claims, latency_claims(claims, cids[0]), int(latency_cfg["queries"]),
                                 encoder.device))
        del encoder
        timings.append({"stage": "c1-latency", "seconds": round(time.perf_counter() - t0, DECIMALS),
                        "items": int(latency_cfg["queries"])})
    result = {
        "schema": RESULT_SCHEMA,
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        "torch_threads": int(torch.get_num_threads()),
        "input": {
            "kind": bench["input"]["kind"], "zip": bench["input"]["zip"], "zip_base": bench["input"]["zip_base"],
            "zip_bytes": int(zip_path(root, bench["input"]).stat().st_size),
            "file": (PRIVATE_REL / INPUT_FILE).as_posix(), "file_namespace": "out_root (AAC_OUT), never the run directory",
            "sha256": sha, "n_passages": len(texts), "first_claim_id": cids[0], "last_claim_id": cids[-1],
            "host_filter": "not applied (throughput input; specification item 18 applies to retrieval, Phase 3)",
            **in_stats, "tokens": tokens},
        "models": [{k: m[k] for k in ("key", "hf_id", "revision", "loader", "max_seq_length")} for m in bench["models"]],
        "throughput": thr,
        "bit_identity": ident,
        "latency": lat,
        "latency_warmup": LATENCY_WARMUP,
        "latency_cfg": latency_cfg,
        "c1": {"trained_by": c1["trained_by"], "run_id": ref["run_id"], "checkpoint_sha256": ref["sha256"]},
        "projection": bench["projection"],
    }
    (run_dir / BENCH_JSON).write_text(hashing.dumps_sorted(_round(result)), encoding="ascii")
    total = time.perf_counter() - t_start
    sealed_seconds = round(total, DECIMALS) if clock.source == "wall" else 0.0
    return {
        "configurations": [f"bench/{resolved['name']}"],
        "items": {"passages": len(texts)},
        "stages": [{"name": "bench", "device": device, "seconds": sealed_seconds, "items": len(texts)}],
        "timings": timings + [{"stage": "bench", "seconds": round(total, DECIMALS), "items": len(texts), "device": device}],
        "depends_on": [ref["run_id"]],
        "determinism_level": DETERMINISM,
    }


# --------------------------------------------------------------------------------------------- smoke store


_WORDS = ("harbour", "lantern", "council", "ferry", "orchard", "granite", "meadow", "signal", "copper", "willow",
          "bridge", "quarry", "beacon", "thistle", "canal", "pylon", "saffron", "gravel", "tunnel", "marble",
          "heron", "cobalt", "furnace", "ledger", "pepper", "rafter", "silo", "tundra", "velvet", "walnut")


def _smoke_sentence(k: int) -> str:
    n = 4 + (k * 5) % 23
    return f"Synthetic bench sentence {k} " + " ".join(_WORDS[(k * 7 + j * 3) % len(_WORDS)] for j in range(n))


def make_smoke_store(out_dir: Path, n_stores: int = 3, docs_per_store: int = 4, sents_per_doc: int = 20) -> dict:
    """Invented three-member dev store (output_dev/<i>.json) + claims.json + MANIFEST.sha256, byte-identical on
 every call: every 8th sentence repeats an earlier one and each document carries one blank sentence, so the
 dedup and blank rules are exercised (240 sentences, 210 distinct non-blank)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    k = 0
    members: dict[str, bytes] = {}
    for i in range(n_stores):
        lines = []
        for d in range(docs_per_store):
            sents = []
            for _ in range(sents_per_doc):
                sents.append(_smoke_sentence(k - 5) if k % 8 == 7 else _smoke_sentence(k))
                k += 1
            sents.insert(d % sents_per_doc, "   ")
            lines.append(json.dumps({"claim_id": str(i), "type": "background", "query": "synthetic",
                                     "url": f"https://example.org/bench/{i}/{d}", "url2text": sents},
                                    sort_keys=False) + "\n")
        members[f"output_dev/{i}.json"] = "".join(lines).encode("utf-8")
    zpath = out_dir / "dev_knowledge_store.zip"
    tmp = zpath.with_name(zpath.name + ".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name])
    os.replace(tmp, zpath)
    claims = [{"claim_id": f"averitec/dev/{i}", "date": "2020-09-0" + str(i + 1),
               "text": f"Synthetic smoke claim {i}: the harbour council reopened {i + 2} ferry routes"}
              for i in range(n_stores)]
    (out_dir / "claims.json").write_text(json.dumps(claims, indent=1, sort_keys=True, ensure_ascii=True) + "\n",
                                         encoding="ascii")
    manifest = "".join(f"{hashing.sha256_file(out_dir / n)}  {n}\n" for n in sorted(("claims.json", zpath.name)))
    (out_dir / "MANIFEST.sha256").write_text(manifest, encoding="ascii")
    return {"zip": zpath, "n_sentences": k, "manifest_data_version": hashing.data_version(out_dir / "MANIFEST.sha256")}


# --------------------------------------------------------------------------------------------- CLI


def cmd_run(args) -> int:
    from aac.run.clock import make_clock
    from aac.run.config import BenchConfig, load_any
    from aac.run.env import write_env
    from aac.run.finalize import FinalizeRefused, RunRefused, claim_run_dir, finalize_run
    from aac.run.identity import identity, lock_path
    from aac.run.paths import root as project_root

    root = project_root()
    cfg = load_any(Path(args.config))
    if not isinstance(cfg, BenchConfig):
        print(f"refusing: {args.config} is not an aac-bench/1 config", file=sys.stderr)
        return 2
    ident = identity(root, Path(args.config), args.seed)
    resolved, rid = ident["resolved"], ident["run_id"]
    key = (resolved["regime"], BENCH_SYSTEM, resolved["name"], args.seed)
    try:
        run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, args.force)
    except RunRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, lock_path(root))
    started = clock.now_iso_ms()
    summary = run_bench(resolved, root, run_dir, args.seed, clock)
    finished = clock.now_iso_ms()
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for t in summary["timings"]:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
    try:
        digest = finalize_run(root, run_dir, ident, summary=summary, record_count=0, shards=[], clock=clock,
                              started=started, finished=finished, system=BENCH_SYSTEM,
                              determinism_level=summary["determinism_level"], target_dir=target_dir,
                              stale_ids=stale_ids)
    except FinalizeRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    res = json.loads((target_dir / BENCH_JSON).read_text(encoding="utf-8"))
    n_meas = sum(1 for t in res["throughput"] if t["status"] == "measured")
    print(f"bench {rid}: {res['input']['n_passages']} passages on {res['device']}, {n_meas} throughput cells measured, "
          f"input sha256 {res['input']['sha256'][:12]}, SEALED {digest[:12]}")
    return 0


def _load_resolved(path: Path) -> dict:
    """config.resolved.json of the parent run (what run_bench passes), or an aac-bench/1 YAML."""
    if Path(path).suffix == ".json":
        return json.loads(Path(path).read_text(encoding="utf-8"))
    from aac.run.config import load_any

    return load_any(Path(path)).model_dump(by_alias=True)


def cmd_second_pass(args) -> int:
    """The fresh-process pass of the bit-identity check: same input (sha256-checked), same model, precision,
 seed and batch plan; writes only the .npy named by --out (a private temp dir of the parent)."""
    from aac.run.paths import root as project_root
    from aac.run.seeds import seed_everything

    root = project_root()
    resolved = _load_resolved(Path(args.config))
    seed_everything(args.seed)
    hits = [m for m in resolved["bench"]["models"] if m["key"] == args.model]
    if len(hits) != 1:
        print(f"second-pass: no bench model {args.model!r}", file=sys.stderr)
        return 2
    texts, cids = read_input(input_path(root), args.input_sha256)
    items = items_for(hits[0]["loader"], texts, cids, load_claims(root, resolved["data"]))
    model = BenchModel(hits[0], root, device_of(), args.precision)
    arr = encode_pass(model, items, batch_plan(model.token_counts(items), args.batch_size))
    out = Path(args.out)
    tmp = out.with_name(out.name + ".tmp.npy")
    np.save(tmp, np.ascontiguousarray(arr), allow_pickle=False)
    os.replace(tmp, out)
    return 0


def cmd_make_smoke_store(args) -> int:
    info = make_smoke_store(Path(args.out))
    print(f"smoke store: {info['zip']} ({info['n_sentences']} sentences) data_version={info['manifest_data_version']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.bench.retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="seal a BENCH run from an aac-bench/1 config")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("second-pass", help="fresh-process pass of the bit-identity check (called by run_bench)")
    p.add_argument("--config", required=True, help="the parent run's config.resolved.json (or an aac-bench/1 YAML)")
    p.add_argument("--model", required=True)
    p.add_argument("--precision", required=True, choices=["float32", "bfloat16"])
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--input-sha256", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_second_pass)
    p = sub.add_parser("make-smoke-store", help="write the invented three-member store used by make bench-smoke")
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_make_smoke_store)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    sys.exit(main())
