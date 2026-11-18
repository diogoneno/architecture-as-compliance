"""C2 dense and re-rank stages : BGE-M3 with own CLS pooling, bge-reranker-v2-m3, rank helpers.

 DenseEncoder AutoTokenizer + AutoModel at the pinned revision, offline, weights float32. Output vector = the CLS
 token of last_hidden_state, L2-normalized (the pooling of the sentence-transformers BGE-M3 modules:
 Transformer -> Pooling(cls) -> Normalize; tests/smoke/test_c2_pooling.py checks cosine >= 0.999
 against sentence-transformers 6.1.0 on 20 passages). Inputs truncated at max_seq_length. bf16
 autocast only when precision is bfloat16 AND the device is cuda; CPU computes in float32.
 Reranker sentence_transformers.CrossEncoder at the pinned revision, weights float32, sigmoid activation:
 scores in [0, 1] for (claim, passage) pairs.
 rank the first k of sorted (-round(score, 6), id): one order key for every ranking (operational constraint).
 dense_scores claim vector (float32) against stored fp16 passage vectors, both widened exactly to float64 before
 the dot product, so the C2 recompute path reproduces the EV run's scores.

Both encoders use a fixed batch plan (token count descending, ties by input index; aac.bench.retrieval.batch_plan
rule), so a repeated call over the same input sees the same batches. torch and transformers are imported inside
functions: importing this module stays cheap.

 EvidenceC2 the real C2 inference component (cache rule, providers). It asks the
 request's EvidenceProvider `lookup(claim_id, text)`: a CacheHit (clean claim, frozen EV evidence) is
 used as is; Candidates with vectors get the dense stage (the request text embedded ALONE, batch size
 1, as the EV run embedded claims) over the stored fp16 rows, rank(..., k_dense), a re-rank of
 (request text, passage) pairs and rank(..., k); Candidates with rerank_only skip the dense stage. The
 kept passages go to scorer.score_many([(passage text, request text)]) -> [(pS, pR, pN)]. Models load
 on first use only, so a clean pass answered from the cache never loads BGE-M3 or the re-ranker.
 retrieve_many batches re-rank pairs and scorer pairs across requests, in request order.
 RD requests (provider key `pooled`) may have their scorer pair cut when the RD text alone does not
 fit the scorer window ((a)): scorer.score_many_flagged
 flags each cut pair, an RD c2_out carries `_scorer_truncated` (pairs cut; stripped before the record) and
 EvidenceC2.scorer_truncations lists (claim_id, split, pairs cut) for the run summary. 
 an R3/R4 adversarial text (query_ctx phase attack, posthoc or replay) may be cut the same way; its row
 also carries phase and query_index (the victim server's scorer-truncation.json).
 python -m aac.components.c2_retrieval smoke ... the `make c2-smoke` checks (mk/c2.mk)."""
from __future__ import annotations

import argparse
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

from aac.data.evidence_providers import CacheHit, Candidates, Passage
from aac.run.hashing import claim_hash

DECIMALS = 6
# Item 15, option (a) (analyst, 2026-10-17): only RD texts may be cut for the scorer pair.
CUT_PROVIDER_KEYS = frozenset({"pooled"})

# A perturbed claim (e.g. character-level edits) may exceed the scorer window although no clean AVeriTeC claim does;
# refusing it faulted the whole victim server. Clean-phase texts and requests without query_ctx are never cut.
CUT_PHASES = frozenset({"attack", "posthoc", "replay"})


def _may_cut(req) -> bool:
    if getattr(req, "evidence_provider_key", None) in CUT_PROVIDER_KEYS:
        return True
    qc = getattr(req, "query_ctx", None)
    return isinstance(qc, dict) and qc.get("phase") in CUT_PHASES


def rank(ids, scores, k: int) -> list[tuple[str, float]]:
    """[(id, score6)] of the first k pairs by (-round(score, 6), id)."""
    pairs = [(str(i), round(float(s), DECIMALS)) for i, s in zip(ids, scores, strict=True)]
    pairs.sort(key=lambda t: (-t[1], t[0]))
    return pairs[:max(int(k), 0)]


def batch_plan(counts: list[int], batch_size: int) -> list[list[int]]:
    """Indices sorted by token count (longest first, ties by index), cut into fixed batches."""
    order = sorted(range(len(counts)), key=lambda i: (-counts[i], i))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)]


def dense_scores(claim_vec: np.ndarray, passage_vecs: np.ndarray) -> np.ndarray:
    """Dot products of one claim vector with stored passage vectors (fp16 or fp32 rows), in float64.
 Widening fp16 -> fp32 -> fp64 is exact, so the score depends only on the stored values."""
    c = np.asarray(claim_vec, dtype=np.float32).astype(np.float64).reshape(-1)
    v = np.asarray(passage_vecs).astype(np.float32).astype(np.float64)
    if v.ndim != 2 or v.shape[1] != c.shape[0]:
        raise ValueError(f"passage vectors {v.shape} do not match the claim vector {c.shape}")
    return v @ c


def _offline(root: Path) -> str:
    from aac.components.c1_train import _offline_env, hf_cache_dir

    _offline_env(root)
    return hf_cache_dir(root)


def _fp32_params(model, what: str) -> None:
    import torch

    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        raise TypeError(f"{what} parameters must all be float32, got {sorted(map(str, dtypes))}")


class DenseEncoder:
    """BGE-M3 dense vectors with own CLS pooling (cfg = a DenseCfg dict)."""

    def __init__(self, cfg: dict, root: Path, device: str) -> None:
        if cfg["pooling"] != "cls" or cfg["normalize"] is not True:
            raise ValueError("DenseEncoder implements pooling cls with normalize true only")
        cache = _offline(root)
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.cfg = dict(cfg)
        self.device = device
        self.max_seq_length = int(cfg["max_seq_length"])
        self.autocast = cfg["precision"] == "bfloat16" and str(device).startswith("cuda")
        self.precision_applied = "bfloat16-autocast" if self.autocast else "float32"
        self.tokenizer = AutoTokenizer.from_pretrained(cfg["hf_id"], revision=cfg["revision"], local_files_only=True,
                                                       cache_dir=cache)
        model = AutoModel.from_pretrained(cfg["hf_id"], revision=cfg["revision"], dtype=torch.float32,
                                          local_files_only=True, cache_dir=cache)
        _fp32_params(model, "BGE-M3")
        self.model = model.to(device).eval()

    def token_counts(self, texts: list[str]) -> list[int]:
        enc = self.tokenizer(list(texts), truncation=True, max_length=self.max_seq_length)
        return [len(x) for x in enc["input_ids"]]

    def _forward(self, texts: list[str]) -> np.ndarray:
        import torch

        enc = self.tokenizer(list(texts), padding=True, truncation=True, max_length=self.max_seq_length,
                             return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            if self.autocast:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    hidden = self.model(**enc).last_hidden_state
            else:
                hidden = self.model(**enc).last_hidden_state
            cls = hidden[:, 0].float()
            out = torch.nn.functional.normalize(cls, p=2, dim=1)
        return out.cpu().numpy().astype(np.float32)

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        """float32 (n, dim) unit vectors, rows in input order; fixed batch plan over token counts."""
        texts = list(texts)
        if not texts:
            return np.zeros((0, int(self.model.config.hidden_size)), dtype=np.float32)
        plan = batch_plan(self.token_counts(texts), int(batch_size))
        out: np.ndarray | None = None
        for b in plan:
            arr = self._forward([texts[i] for i in b])
            if out is None:
                out = np.empty((len(texts), arr.shape[1]), dtype=np.float32)
            out[b] = arr
        assert out is not None
        return out

    def close(self) -> None:
        del self.model
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


class Reranker:
    """bge-reranker-v2-m3 cross-encoder (cfg = a RerankCfg dict): sigmoid scores of (claim, passage) pairs."""

    def __init__(self, cfg: dict, root: Path, device: str) -> None:
        if cfg["activation"] != "sigmoid":
            raise ValueError("Reranker implements activation sigmoid only")
        cache = _offline(root)
        import torch
        from sentence_transformers import CrossEncoder

        self.cfg = dict(cfg)
        self.device = device
        self.max_length = int(cfg["max_length"])
        self.model = CrossEncoder(cfg["hf_id"], revision=cfg["revision"], cache_folder=cache, local_files_only=True,
                                  device=device, max_length=self.max_length, activation_fn=torch.nn.Sigmoid(),
                                  model_kwargs={"dtype": torch.float32})
        self.model.max_length = self.max_length
        _fp32_params(self.model, "bge-reranker-v2-m3")
        self.model.eval()

    def token_counts(self, pairs: list[tuple[str, str]]) -> list[int]:
        tok = self.model.tokenizer
        enc = tok([a for a, _ in pairs], [b for _, b in pairs], truncation=True, max_length=self.max_length)
        return [len(x) for x in enc["input_ids"]]

    def score(self, pairs: list[tuple[str, str]], *, batch_size: int) -> np.ndarray:
        """float64 sigmoid scores, input order; fixed batch plan over pair token counts."""
        import torch

        pairs = [(str(a), str(b)) for a, b in pairs]
        out = np.empty(len(pairs), dtype=np.float64)
        if not pairs:
            return out
        for b in batch_plan(self.token_counts(pairs), int(batch_size)):
            with torch.inference_mode():
                s = self.model.predict([pairs[i] for i in b], batch_size=len(b), convert_to_tensor=True,
                                       show_progress_bar=False)
            out[b] = s.float().cpu().numpy().reshape(-1).astype(np.float64)
        return out

    def close(self) -> None:
        del self.model
        if str(self.device).startswith("cuda"):
            import torch

            torch.cuda.empty_cache()


# ---------------------------------------------------------------- C2 inference 


class EvidenceC2:
    """Real C2 (cfg = a resolved RealC2Cfg dict; reads dense, reranker, k_dense, k). `scorer` is any object with
 score_many(pairs) -> [(pS, pR, pN)] over (passage_text, claim_text) pairs (builds it). `dense` and
 `reranker` may be passed in (tests); otherwise they are built from cfg on first use.

 Output (the c2_out contract of ): evidence_ids, evidence_scores (rerank, 6 d.p.), evidence_coverage
 (top-1 rerank score >= floor; False when there is no passage) and the private _passages, _support, _refute lists
 (underscore keys never reach a record: C3 strips them)."""

    name = "bge-m3-rerank-entail-v1"

    def __init__(self, cfg: dict, root: Path, *, scorer, device: str = "cpu", dense=None, reranker=None) -> None:
        missing = [k for k in ("dense", "reranker", "k_dense", "k") if k not in cfg]
        if missing:
            raise ValueError(f"EvidenceC2 config lacks {missing}")
        self.cfg = dict(cfg)
        self.root = Path(root)
        self.device = device
        self.scorer = scorer
        self.k_dense, self.k = int(cfg["k_dense"]), int(cfg["k"])
        if not 0 < self.k <= self.k_dense:
            raise ValueError(f"EvidenceC2 needs 0 < k ({self.k}) <= k_dense ({self.k_dense})")
        self._dense = dense
        self._reranker = reranker
        self.floor: float | None = None
        self.scorer_truncations: list[dict] = []   # {claim_id, split, pairs} per request with a cut scorer pair

    # models on first use
    @property
    def dense(self):
        if self._dense is None:
            self._dense = DenseEncoder(self.cfg["dense"], self.root, self.device)
        return self._dense

    @property
    def reranker(self):
        if self._reranker is None:
            self._reranker = Reranker(self.cfg["reranker"], self.root, self.device)
        return self._reranker

    def models_loaded(self) -> dict[str, bool]:
        return {"dense": self._dense is not None, "reranker": self._reranker is not None}

    def set_floor(self, value: float) -> None:
        """The relevance floor (rule p10-top1-rerank-val_cal, computed by the caller before any C3 fit; operational constraint)."""
        v = float(value)
        if not math.isfinite(v):
            raise ValueError(f"relevance floor must be finite, got {value!r}")
        self.floor = round(v, DECIMALS)

    def retrieve(self, req, provider) -> dict:
        return self.retrieve_many([req], [provider])[0]

    def retrieve_many(self, reqs: list, providers: list) -> list[dict]:
        if self.floor is None:
            raise RuntimeError("EvidenceC2.retrieve before set_floor(): the relevance floor must exist first (Pitfall 3)")
        reqs, providers = list(reqs), list(providers)
        if len(reqs) != len(providers):
            raise ValueError(f"{len(reqs)} requests but {len(providers)} providers")
        plans: list[tuple] = []
        pairs: list[tuple[str, str]] = []
        for req, prov in zip(reqs, providers, strict=True):
            lookup = getattr(prov, "lookup", None)
            if lookup is None:
                raise TypeError(f"provider {type(prov).__name__} has no lookup(claim_id, text); EvidenceC2 needs a real "
                                "EvidenceProvider (AveritecPerClaimStore, SuppliedEvidence, PooledRdStore)")
            got = lookup(req.claim_id, req.text)
            if isinstance(got, CacheHit):
                plans.append(("hit", list(got.passages)[:self.k]))
                continue
            if not isinstance(got, Candidates):
                raise TypeError(f"provider lookup returned {type(got).__name__}, not CacheHit or Candidates")
            ids, texts = list(got.passage_ids), list(got.texts)
            if len(ids) != len(texts) or len(set(ids)) != len(ids):
                raise ValueError(f"{req.claim_id}: candidate ids and texts must be aligned and unique")
            if got.rerank_only:
                sel = list(range(len(ids)))
            elif got.vectors is None:
                raise ValueError(f"{req.claim_id}: Candidates without vectors must be rerank_only")
            elif not ids:
                sel = []
            else:
                cv = self.dense.encode([req.text], batch_size=1)[0]   # alone, as the EV run embedded claims
                pos = {p: i for i, p in enumerate(ids)}
                sel = [pos[p] for p, _ in rank(ids, dense_scores(cv, got.vectors), self.k_dense)]
            plans.append(("rr", [ids[i] for i in sel], [texts[i] for i in sel], len(pairs)))
            pairs.extend((req.text, texts[i]) for i in sel)
        rr = (self.reranker.score(pairs, batch_size=int(self.cfg["reranker"]["batch_size"])) if pairs
              else np.zeros(0, dtype=np.float64))
        kept: list[list[Passage]] = []
        for plan in plans:
            if plan[0] == "hit":
                kept.append(plan[1])
                continue
            _, ids, texts, start = plan
            text_of = dict(zip(ids, texts, strict=True))
            top = rank(ids, rr[start:start + len(ids)], self.k)
            kept.append([Passage(p, claim_hash(text_of[p]), s6, text_of[p]) for p, s6 in top])
        spairs = [(p.text, req.text) for req, ps in zip(reqs, kept, strict=True) for p in ps]
        cut = [_may_cut(req) for req, ps in zip(reqs, kept, strict=True) for _ in ps]
        flagged = getattr(self.scorer, "score_many_flagged", None)
        if not spairs:
            probs, flags = [], []
        elif flagged is not None:
            probs, flags = flagged(spairs, cut)
            probs, flags = list(probs), [bool(f) for f in flags]
        else:   # a scorer without flags (tests, the smoke HashScorer) never cuts a pair
            probs, flags = list(self.scorer.score_many(spairs)), [False] * len(spairs)
        if len(probs) != len(spairs) or len(flags) != len(spairs):
            raise ValueError(f"scorer returned {len(probs)} outputs and {len(flags)} flags for {len(spairs)} pairs")
        if any(f and not c for f, c in zip(flags, cut, strict=True)):
            raise ValueError("scorer flagged a cut pair that EvidenceC2 did not mark as cuttable")
        out, j = [], 0
        for req, ps in zip(reqs, kept, strict=True):
            mine = probs[j:j + len(ps)]
            n_cut = sum(flags[j:j + len(ps)])
            j += len(ps)
            scores = [round(float(p.score), DECIMALS) for p in ps]
            if n_cut:
                row = {"claim_id": req.claim_id, "split": getattr(req, "split", None), "pairs": int(n_cut)}
                qc = getattr(req, "query_ctx", None)
                if isinstance(qc, dict):   # R3/R4: the victim server maps the query index to the record id
                    row.update(phase=qc.get("phase"), query_index=qc.get("query_index"))
                self.scorer_truncations.append(row)
            one = {
                "evidence_ids": [p.id for p in ps],
                "evidence_scores": scores,
                "evidence_coverage": bool(scores) and max(scores) >= self.floor,
                "_passages": ps,
                "_support": [round(float(t[0]), DECIMALS) for t in mine],
                "_refute": [round(float(t[1]), DECIMALS) for t in mine],
            }
            if getattr(req, "evidence_provider_key", None) in CUT_PROVIDER_KEYS:
                one["_scorer_truncated"] = int(n_cut)   # RD requests only; other c2_out dicts keep their key set
            out.append(one)
        return out


# ---------------------------------------------------------------- make c2-smoke


class HashScorer:
    """Smoke-only scorer: fixed (pS, pR, pN) per (passage, claim) pair from sha256; no model."""

    def score_many(self, pairs):
        out = []
        for a, b in pairs:
            h = hashlib.sha256(f"{a}\x1f{b}".encode("utf-8", "surrogatepass")).digest()
            w = [1 + h[0], 1 + h[1], 1 + h[2]]
            out.append(tuple(x / sum(w) for x in w))
        return out


def _same(a: dict, b: dict) -> bool:
    """Equal at 6 d.p.: same ids in order, rerank scores within 1e-6 (one unit of the 6th decimal; operational constraint: batch
 composition moves float values below that), same scorer outputs."""
    return (a["evidence_ids"] == b["evidence_ids"] and a["evidence_coverage"] == b["evidence_coverage"]
            and all(abs(x - y) <= 1.0000001e-6 for x, y in zip(a["evidence_scores"], b["evidence_scores"], strict=True))
            and a["_support"] == b["_support"] and a["_refute"] == b["_refute"])


def smoke_cache(ev_name: str, say=print) -> int:
    """Cache vs forced recompute on every claim of the EV run `ev_name` (seed 0), and a perturbed claim through the
 recompute path. Prints counts and run ids only, never a text."""
    from aac.data.averitec import load_claims
    from aac.data.evidence_providers import AveritecPerClaimStore
    from aac.pipeline.pipeline import Request
    from aac.run.identity import resolve_run_ref
    from aac.run.paths import root as project_root
    from aac.run.paths import runs_dir
    import json

    root = project_root()
    ref = resolve_run_ref(root, "EV", ev_name, 0)
    resolved = json.loads((runs_dir(root) / ref["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))
    ev = resolved["evidence"]
    store = AveritecPerClaimStore(root, ref, "primary", allow_limit=True)
    cfg = {"dense": ev["dense"], "reranker": ev["reranker"], "k_dense": ev["k_dense"], "k": ev["k"]}
    c2 = EvidenceC2(cfg, root, scorer=HashScorer(), device="cpu")
    tops = sorted(store.top1_score(c) for c in store.claim_ids())
    c2.set_floor(tops[max(0, math.ceil(0.10 * len(tops)) - 1)])   # smoke floor: nearest-rank p10 of the top-1 scores
    claims = load_claims(root / resolved["data"]["claims"])
    ids = store.claim_ids()
    reqs = [Request(claim_id=c, text=claims[c]["text"], evidence_provider_key=store.key, regime="C2/smoke",
                    split="smoke", seed=0) for c in ids]
    cached = c2.retrieve_many(reqs, [store] * len(reqs))
    if any(c2.models_loaded().values()):
        raise SystemExit("c2-smoke: FAIL: the clean pass loaded a model (the cache must answer it)")
    say(f"c2-smoke: clean pass of {len(reqs)} claims answered from {ref['run_id']} with no model loaded")
    store.force_recompute = True
    recomputed = c2.retrieve_many(reqs, [store] * len(reqs))
    single = [c2.retrieve(r, store) for r in reqs[:4]]
    store.force_recompute = False
    bad = [r.claim_id for r, a, b in zip(reqs, cached, recomputed, strict=True) if not _same(a, b)]
    if bad:
        raise SystemExit(f"c2-smoke: FAIL: recompute != cache on {len(bad)} claims (first {bad[0]})")
    exact = sum(a["evidence_scores"] == b["evidence_scores"] for a, b in zip(cached, recomputed, strict=True))
    if any(not _same(a, b) for a, b in zip(single, recomputed[:4], strict=True)):
        raise SystemExit("c2-smoke: FAIL: retrieve != retrieve_many")
    pert = Request(claim_id=ids[0], text=claims[ids[0]]["text"] + " (edited)", evidence_provider_key=store.key,
                   regime="C2/smoke", split="smoke", seed=0)
    if not isinstance(store.lookup(pert.claim_id, pert.text), Candidates):
        raise SystemExit("c2-smoke: FAIL: a perturbed claim was answered from the cache")
    po = c2.retrieve(pert, store)
    if len(po["evidence_ids"]) != min(c2.k, len(store._lists[ids[0]])):
        raise SystemExit("c2-smoke: FAIL: the perturbed claim got no recomputed evidence")
    say(f"c2-smoke: perturbed claim took the recompute path ({len(po['evidence_ids'])} passages)")
    say(f"c2-smoke: {exact} of {len(reqs)} claims bit-equal at 6 d.p., the rest within 1e-6")
    say(f"c2-smoke: cache==recompute on {len(reqs)} claims")
    return c2.floor


def _files_hold_text(run_dir: Path, texts: list[str]) -> list[str]:
    """Names of the files of run_dir (gzip files decompressed) that contain any of `texts` (a python read of every
 file; the c2-smoke check that a pooled_rd run directory is text-free)."""
    import gzip

    hits = []
    for f in sorted(Path(run_dir).rglob("*")):
        if not f.is_file():
            continue
        raw = f.read_bytes()
        if f.suffix == ".gz":
            raw = gzip.decompress(raw)
        body = raw.decode("utf-8", "replace")
        if any(t and t in body for t in texts):
            hits.append(f.relative_to(run_dir).as_posix())
    return hits


def smoke_rd_fc(rd_ev_name: str, fc_items: str, floor: float, say=print) -> int:
    """PooledRdStore over the pooled_rd EV run `rd_ev_name` (one fixture RD item from the cache, a changed RD text
 refused, the coverage rate at `floor`, no RD text in the run directory) and SuppliedEvidence on one fixture FC
 item (re-ranker score in [0, 1]). Prints run ids and counts only."""
    import json

    from aac.data.evidence_providers import (PooledRdStore, PooledRecomputeRefused, SuppliedEvidence,
                                             load_incrediblae_items)
    from aac.pipeline.pipeline import Request
    from aac.run.identity import resolve_run_ref
    from aac.run.paths import root as project_root
    from aac.run.paths import runs_dir

    root = project_root()
    ref = resolve_run_ref(root, "EV", rd_ev_name, 0)
    rd_dir = runs_dir(root) / ref["run_id"]
    resolved = json.loads((rd_dir / "config.resolved.json").read_text(encoding="utf-8"))
    ev = resolved["evidence"]
    items = load_incrediblae_items(root, resolved["data"]["rd_items"])
    store = PooledRdStore(root, ref)
    cfg = {"dense": ev["dense"], "reranker": ev["reranker"], "k_dense": ev["k_dense"], "k": ev["k"]}
    c2 = EvidenceC2(cfg, root, scorer=HashScorer(), device="cpu")
    c2.set_floor(floor)
    ids = store.item_ids()
    if sorted(ids) != sorted(items):
        raise SystemExit(f"c2-smoke: FAIL: {ref['run_id']} holds {len(ids)} RD items, the item file {len(items)}")
    reqs = [Request(claim_id=i, text=items[i]["text"], evidence_provider_key=store.key, regime="C2/smoke",
                    split="smoke", seed=0) for i in ids]
    outs = c2.retrieve_many(reqs, [store] * len(reqs))
    if any(c2.models_loaded().values()) or not all(0 < len(o["evidence_ids"]) <= c2.k for o in outs):
        raise SystemExit("c2-smoke: FAIL: pooled RD answers must come from the cache with 1..k passages")
    try:
        store.lookup(ids[0], items[ids[0]]["text"] + " ")
    except PooledRecomputeRefused:
        pass
    else:
        raise SystemExit("c2-smoke: FAIL: a changed RD text was answered")
    leaked = _files_hold_text(rd_dir, [it["text"] for it in items.values()])
    if leaked:
        raise SystemExit(f"c2-smoke: FAIL: RD text found in {ref['run_id']}: {leaked}")
    covered = sum(o["evidence_coverage"] for o in outs)
    say(f"c2-smoke: pooled RD {len(ids)} items from the cache, coverage {covered}/{len(ids)} at floor {c2.floor}; "
        f"changed text refused; run directory text-free")
    fc = load_incrediblae_items(root, fc_items)
    sup = SuppliedEvidence(fc)
    fid = sorted(fc)[0]
    o = c2.retrieve(Request(claim_id=fid, text=fc[fid]["text"], evidence_provider_key=sup.key, regime="C2/smoke",
                            split="smoke", seed=0), sup)
    if o["evidence_ids"] != [f"{fid}#supplied"] or not 0.0 <= o["evidence_scores"][0] <= 1.0:
        raise SystemExit("c2-smoke: FAIL: supplied FC evidence must be one passage with a re-ranker score in [0, 1]")
    if not c2.models_loaded()["reranker"] or c2.models_loaded()["dense"]:
        raise SystemExit("c2-smoke: FAIL: supplied evidence must use the re-ranker and no dense stage")
    say(f"c2-smoke: supplied FC item scored by the re-ranker ({o['evidence_scores'][0]:.6f})")
    say(f"c2-smoke: pooled_rd {ref['run_id']} supplied ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.components.c2_retrieval")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke", help="make c2-smoke checks on the tiny world (CPU)")
    p.add_argument("--ev-name", required=True)
    p.add_argument("--rd-ev-name", help="a pooled_rd EV run name: also check PooledRdStore and SuppliedEvidence")
    p.add_argument("--fc-items", default="tests/fixtures/corpus/fc-fx.json")
    a = ap.parse_args(argv)
    say = lambda m: print(m, flush=True)  # noqa: E731
    floor = smoke_cache(a.ev_name, say=say)
    return smoke_rd_fc(a.rd_ev_name, a.fc_items, floor, say=say) if a.rd_ev_name else 0


if __name__ == "__main__":
    sys.exit(main())
