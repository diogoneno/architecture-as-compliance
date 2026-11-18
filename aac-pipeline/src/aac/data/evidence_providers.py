"""EvidenceProvider strategies, chosen by dataset (ARCH section 4.2; D-05).

 FixtureEvidence per-claim store: tests/fixtures/corpus/evidence.json (AVeriTeC-shaped)
 SuppliedEvidence the evidence sentence carried in the item replaces retrieval (InCrediblAE FC)
 PooledFixtureEvidence retrieval over the POOLED fixture store, scored by a deterministic
 text/passage affinity (InCrediblAE RD per D-05: NullEvidence is not an RD
 strategy; the evidence-coverage rate is reported)

Real providers  answer `lookup(claim_id, text)` for aac.components.c2_retrieval.EvidenceC2:
 AveritecPerClaimStore the frozen per-claim evidence of a sealed EV run (key "averitec_store"): CacheHit when
 claim_hash(text) equals the clean claim hash the EV run recorded, otherwise Candidates = the
 claim's full shortlist (passage ids, texts, stored fp16 vectors) for the recompute path
 SuppliedEvidence (key "supplied", InCrediblAE FC) Candidates(rerank_only) = the item's one evidence sentence,
 so its score is the real re-ranker score of the (claim, evidence) pair; C1 sees the claim alone
 PooledRdStore (key "pooled", InCrediblAE RD) the cached top k of a sealed pooled_rd EV run; CacheHit when
 claim_hash(text) equals the recorded RD text hash, otherwise a refusal: RD is attacked only
 through victim A (specification item 13), so pooled recompute is not offered
 GoldQaEvidence (key "goldqa", addition C oracle) each AVeriTeC claim's gold question-answer
 evidence, read from the raw train/dev JSON at run time, one passage per non-empty answer in the
 scorer's gold-text format (protocol notes row 28; the formatter is injected by the factory, so one
 implementation of the format exists), as Candidates(rerank_only): EvidenceC2 scores them with the
 pinned re-ranker as it scores FC's supplied evidence; top1_score(claim_id) = the highest
 re-ranker score of the clean claim against its gold passages (0.0 with none), for the floor
load_incrediblae_items reads FC/RD texts at run time and checks each against the committed hash .
Passage.text holds the passage sentence in memory only: it is excluded from comparison and repr, and no record,
fusion input or run file carries it (03-RESEARCH Pattern 2). This module never imports aac.components (T-L2:
aac.regimes imports it)."""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from aac.run.hashing import claim_hash, derive_seed


@dataclass(frozen=True)
class Passage:
    id: str
    text_hash: str
    score: float
    text: str = field(default="", compare=False, repr=False)   # memory only; never written to a record


@dataclass(frozen=True)
class CacheHit:
    """The frozen evidence of a clean claim: its top-k passages with their cached rerank scores and texts."""

    passages: list[Passage]


@dataclass(frozen=True)
class Candidates:
    """Passages EvidenceC2 must score itself. `vectors` (fp16 rows aligned with passage_ids) present: dense stage, then
 re-rank of the top k_dense. `rerank_only`: re-rank every candidate (no dense stage)."""

    passage_ids: list[str]
    texts: list[str]
    vectors: object | None
    rerank_only: bool


class EvidenceStoreError(ValueError):
    """A sealed evidence input fails its digest, hash, mode or selection check; nothing is answered from it."""


class EvidenceProvider(Protocol):
    def retrieve(self, claim_id: str, text: str, k: int) -> list[Passage]: ...


def load_items(path: Path) -> dict[str, dict]:
    """InCrediblAE-shaped fixture items: [{item_id, text, label (1 = non-credible), evidence?}]."""
    with open(path, encoding="utf-8") as fh:
        items = json.load(fh)
    out = {}
    for it in items:
        if it["item_id"] in out:
            raise ValueError(f"{path}: duplicate item_id {it['item_id']}")
        if it["label"] not in (0, 1):
            raise ValueError(f"{path}: label must be 0 or 1")
        out[it["item_id"]] = it
    return out


class FixtureEvidence:
    """Reads tests/fixtures/corpus/evidence.json: {claim_id: [{id, text_hash, score}]}."""

    key = "fixture"

    def __init__(self, path: Path) -> None:
        with open(path, encoding="utf-8") as fh:
            self._data = json.load(fh)

    def retrieve(self, claim_id: str, text: str, k: int) -> list[Passage]:
        rows = self._data.get(claim_id, [])
        passages = [Passage(r["id"], r["text_hash"], float(r["score"])) for r in rows]
        passages.sort(key=lambda p: (-p.score, p.id))
        return passages[:k]


class SuppliedEvidence:
    """One passage per item: the supplied evidence string, so C1 sees the claim alone and the evidence arm sees
 exactly the item's evidence (BUILD-BRIEF R2-FC). retrieve() is the stub path (score 1.0); lookup() is the real
 path : Candidates(rerank_only=True) with that one passage, which EvidenceC2 scores with the re-ranker
 on the (claim, evidence) pair. Passage.text holds the evidence in memory only."""

    key = "supplied"

    def __init__(self, items: dict[str, dict]) -> None:
        self._items = items

    def retrieve(self, claim_id: str, text: str, k: int) -> list[Passage]:
        it = self._items.get(claim_id)
        if it is None or not it.get("evidence"):
            return []
        ev = it["evidence"]
        return [Passage(f"{claim_id}#supplied", hashlib.sha256(ev.encode("utf-8")).hexdigest(), 1.0, ev)][:k]

    def lookup(self, claim_id: str, text: str) -> Candidates:
        it = self._items.get(claim_id)
        if it is None or not it.get("evidence"):
            return Candidates([], [], None, rerank_only=True)
        return Candidates([f"{claim_id}#supplied"], [it["evidence"]], None, rerank_only=True)


class PooledFixtureEvidence:
    """Retrieval over every passage of the fixture store regardless of claim: the stub for a
 pooled AVeriTeC index. Affinity = (hash-derived text/passage similarity) x passage score,
 a pure function of (claim text, passage id)."""

    key = "pooled"

    def __init__(self, path: Path) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        self._pool = sorted(
            (Passage(r["id"], r["text_hash"], float(r["score"])) for rows in data.values() for r in rows),
            key=lambda p: p.id,
        )

    def retrieve(self, claim_id: str, text: str, k: int) -> list[Passage]:
        h = claim_hash(text)
        scored = []
        for p in self._pool:
            sim = (derive_seed(f"pooled|{h}|{p.id}") % 1_000_000) / 1_000_000.0
            scored.append(Passage(p.id, p.text_hash, round(sim * p.score, 6)))
        scored.sort(key=lambda p: (-p.score, p.id))
        return scored[:k]


# ---------------------------------------------------------------- real providers 

SL_LIST = {"primary": "primary", "sensitivity-b": "variant"}   # EV variant -> SL list (c2_evidence_run.SL_LIST)


def _gz_lines_checked(path: Path, sha256: str, what: str, keep=None) -> list[dict]:
    """Parsed JSON lines of a gzip file whose decompressed bytes must hash to `sha256` (only the lines for which
 keep(line) is true, when given). The whole file is read and checked before any line is returned."""
    h = hashlib.sha256()
    rows = []
    with gzip.open(path, "rb") as fh:
        for raw in fh:
            h.update(raw)
            x = json.loads(raw)
            if keep is None or keep(x):
                rows.append(x)
    if h.hexdigest() != sha256:
        raise EvidenceStoreError(f"{what}: sha256 {h.hexdigest()} != recorded {sha256}; refusing to use it")
    return rows


def _sealed_run_dir(root: Path, ref: dict, what: str) -> Path:
    """The run directory of `ref` ({run_id, sha256}); its SEALED digest is recomputed and must equal ref["sha256"]."""
    from aac.run.paths import runs_dir
    from aac.run.seal import verify_sealed

    d = runs_dir(root) / ref["run_id"]
    if not (d / "SEALED").is_file():
        raise EvidenceStoreError(f"{what} {ref['run_id']} has no SEALED run directory under {runs_dir(root)}")
    digest = verify_sealed(d)
    if digest != ref["sha256"]:
        raise EvidenceStoreError(f"{what} {ref['run_id']}: sealed digest {digest} != ref {ref['sha256']}; refusing it")
    return d


def _pid(claim_id: str, row) -> str:
    return f"{claim_id}#d{int(row[0])}#s{int(row[1])}"


class AveritecPerClaimStore:
    """Frozen per-claim evidence of a sealed per_claim EV run (key "averitec_store"; retrieval cache rule).

 Load-time checks (all before any answer): the EV run's sealed digest equals `ev_ref`; evidence.json limit true is
 refused unless allow_limit (tests, CPU smoke); evidence.jsonl.gz, the SL run (sealed digest as recorded by the EV
 run), its shortlist.jsonl.gz and private sentence cache, the private index and the private fp16 vectors are each
 re-hashed against the recorded sha256. Per claim it keeps the cached rerank list of `variant`, the clean claim
 hash and the full `variant` list (first n_passages SL rows), with texts and vector rows (memory-mapped).

 lookup(claim_id, text): CacheHit (cached top k with rerank scores and texts) when claim_hash(text) equals the clean
 claim hash and force_recompute is false; otherwise Candidates with the full list, texts and fp16 vectors."""

    key = "averitec_store"

    def __init__(self, root: Path, ev_ref: dict, variant: str, *, allow_limit: bool = False) -> None:
        import numpy as np

        from aac.run.hashing import sha256_file
        from aac.run.paths import out_root

        root = Path(root)
        self.force_recompute = False
        self.variant = str(variant)
        d = _sealed_run_dir(root, ev_ref, "EV run")
        meta = json.loads((d / "evidence.json").read_text(encoding="ascii"))
        ev = meta["evidence"]
        if ev["mode"] != "per_claim":
            raise EvidenceStoreError(f"EV run {ev_ref['run_id']} is mode {ev['mode']}, not per_claim")
        if meta["limit"] and not allow_limit:
            raise EvidenceStoreError(f"EV run {ev_ref['run_id']} is a subset run (limit true); pass allow_limit only "
                                     "in tests and the CPU smoke")
        if self.variant not in ev["variants"]:
            raise EvidenceStoreError(f"variant {self.variant!r} is not a list of EV run {ev_ref['run_id']} "
                                     f"{ev['variants']}")
        self.run_id = ev_ref["run_id"]
        self.k = int(ev["k"])
        self.k_dense = int(ev["k_dense"])
        n_keep = int(ev["n_passages"])
        self._cache: dict[str, tuple[str, list[tuple[str, float]]]] = {}
        for x in _gz_lines_checked(d / "evidence.jsonl.gz", meta["evidence_sha256"], f"{self.run_id}/evidence.jsonl.gz",
                                   keep=lambda x: x["variant"] == self.variant):
            self._cache[x["claim_id"]] = (x["claim_hash"], [(str(p), float(s)) for p, s in x["rerank"]])
        sl = meta["shortlist_run"]
        sl_dir = _sealed_run_dir(root, {"run_id": sl["run_id"], "sha256": sl["sealed_sha256"]}, "SL run")
        sl_meta = json.loads((sl_dir / "shortlist.json").read_text(encoding="ascii"))
        if sl_meta["shortlist_sha256"] != sl["shortlist_sha256"] or \
                sl_meta["private_cache_sha256"] != sl["private_cache_sha256"]:
            raise EvidenceStoreError(f"SL run {sl['run_id']}: shortlist.json differs from the EV run's record")
        self._lists: dict[str, list[str]] = {}
        for x in _gz_lines_checked(sl_dir / "shortlist.jsonl.gz", sl["shortlist_sha256"],
                                   f"{sl['run_id']}/shortlist.jsonl.gz", keep=lambda x: x["claim_id"] in self._cache):
            if x["claim_hash"] != self._cache[x["claim_id"]][0]:
                raise EvidenceStoreError(f"{x['claim_id']}: SL and EV claim hashes differ")
            self._lists[x["claim_id"]] = [_pid(x["claim_id"], r) for r in x[SL_LIST[self.variant]][:n_keep]]
        missing = sorted(set(self._cache) - set(self._lists))
        if missing:
            raise EvidenceStoreError(f"{len(missing)} EV claims have no SL line (first {missing[0]})")
        wanted = {p for lst in self._lists.values() for p in lst}
        self._texts: dict[str, str] = {}
        for e in _gz_lines_checked(out_root(root) / sl_meta["private_cache"], sl["private_cache_sha256"],
                                   "private SL sentence cache", keep=lambda e: e["passage_id"] in wanted):
            self._texts[e["passage_id"]] = e["text"]
        if wanted - set(self._texts):
            raise EvidenceStoreError(f"{len(wanted - set(self._texts))} listed passages have no text in the SL cache")
        priv = meta["private"]
        self._row: dict[str, int] = {}
        for x in _gz_lines_checked(out_root(root) / priv["index"]["path"], priv["index"]["sha256"], "private index",
                                   keep=lambda x: x["claim_id"] in self._lists):
            for j, p in enumerate(x["passage_ids"]):
                self._row[p] = int(x["start"]) + j
        vec_path = out_root(root) / priv["vectors"]["path"]
        if sha256_file(vec_path) != priv["vectors"]["sha256"]:
            raise EvidenceStoreError(f"{self.run_id}: private vectors.npy sha256 differs from evidence.json")
        self._vectors = np.load(vec_path, mmap_mode="r")
        if list(self._vectors.shape) != list(priv["vectors"]["shape"]) or self._vectors.dtype != np.float16:
            raise EvidenceStoreError(f"{self.run_id}: vectors.npy shape/dtype differ from evidence.json")
        if wanted - set(self._row):
            raise EvidenceStoreError(f"{len(wanted - set(self._row))} listed passages have no vector row")

    def claim_ids(self) -> list[str]:
        return list(self._cache)

    def _entry(self, claim_id: str):
        try:
            return self._cache[claim_id]
        except KeyError:
            raise KeyError(f"{claim_id!r} is not a claim of EV run {self.run_id}") from None

    def lookup(self, claim_id: str, text: str) -> CacheHit | Candidates:
        import numpy as np

        clean_hash, rerank = self._entry(claim_id)
        if not self.force_recompute and claim_hash(text) == clean_hash:
            return CacheHit([Passage(p, claim_hash(self._texts[p]), s, self._texts[p]) for p, s in rerank])
        ids = list(self._lists[claim_id])
        rows = np.asarray([self._row[p] for p in ids], dtype=np.int64)
        vecs = np.asarray(self._vectors[rows]) if ids else np.zeros((0, self._vectors.shape[1]), dtype=np.float16)
        return Candidates(ids, [self._texts[p] for p in ids], vecs, rerank_only=False)

    def top1_score(self, claim_id: str) -> float:
        """The cached top-1 rerank score of the clean claim (0.0 for an empty list): the relevance-floor input."""
        rerank = self._entry(claim_id)[1]
        return float(rerank[0][1]) if rerank else 0.0


class ItemTextMismatch(ValueError):
    """A run-time FC/RD text does not hash to its committed claim_hash / evidence_hash / text_hash (or is absent).
 The message names the item id only, never the text."""


def load_incrediblae_items(root: Path, path) -> dict[str, dict]:
    """{item_id: {"text", "label", "evidence" (FC only)}} for an InCrediblAE-shaped item file.

 The committed text-free item file of a task (data/derived/incrediblae/<fc|rd>-attack.items.json under `root`)
 reads its texts at run time through aac.data.incrediblae.load_texts: FC from the hash-checked raw attack.tsv, RD
 from lane B (LaneBMissing propagates on a host without it). FC: claim_hash(claim) must equal claim_hash and
 claim_hash(evidence) must equal evidence_hash; RD: claim_hash(lane-B text) must equal text_hash. A difference
 raises ItemTextMismatch naming the item id only. Any other path is a fixture item file that carries its text
 (load_items). Texts are returned to the caller only; nothing is written."""
    from aac.data import incrediblae

    root = Path(root)
    p = root / path
    for task in ("fc", "rd"):
        if p.resolve() == (root / incrediblae.ITEMS_REL / f"{task}-attack.items.json").resolve():
            break
    else:
        return {iid: {k: it[k] for k in ("text", "label", "evidence") if k in it} for iid, it in load_items(p).items()}
    with open(p, encoding="utf-8") as fh:
        committed = json.load(fh)
    texts = incrediblae.load_texts(root, task)
    out: dict[str, dict] = {}
    for it in committed:
        iid = it["item_id"]
        if iid in out:
            raise ValueError(f"{path}: duplicate item_id {iid}")
        if it["label"] not in (0, 1):
            raise ValueError(f"{path}: {iid}: label must be 0 or 1")
        t = texts.get(iid)
        if t is None:
            raise ItemTextMismatch(f"{iid}: no run-time text for this committed item")
        if task == "fc":
            if claim_hash(t["text"]) != it["claim_hash"]:
                raise ItemTextMismatch(f"{iid}: run-time claim does not match the committed claim_hash")
            if claim_hash(t["evidence"]) != it["evidence_hash"]:
                raise ItemTextMismatch(f"{iid}: run-time evidence does not match the committed evidence_hash")
            out[iid] = {"text": t["text"], "label": int(it["label"]), "evidence": t["evidence"]}
        else:
            if claim_hash(t["text"]) != it["text_hash"]:
                raise ItemTextMismatch(f"{iid}: run-time RD text does not match the committed text_hash")
            out[iid] = {"text": t["text"], "label": int(it["label"])}
    return out


class PooledRecomputeRefused(ValueError):
    """An RD text differs from the text a pooled_rd EV run recorded: the pooled store answers from its cache only."""


class PooledRdStore:
    """Cached RD evidence of a sealed pooled_rd EV run (key "pooled"; specification item 16).

 Load-time checks: the pooled_rd run's sealed digest equals `rd_ev_ref`; its evidence.jsonl.gz equals the recorded
 sha256; the pool EV run it names has the recorded sealed digest and pooled-passages sha256; the pooled passage
 texts (decompressed) equal that sha256. lookup(item_id, text) returns CacheHit (cached top k, rerank scores,
 texts) when claim_hash(text) equals the recorded RD text hash; any other text raises PooledRecomputeRefused,
 because RD is attacked only through victim A (specification item 13) and pooled recompute is not offered."""

    key = "pooled"

    def __init__(self, root: Path, rd_ev_ref: dict, *, allow_limit: bool = False) -> None:
        from aac.run.paths import out_root

        root = Path(root)
        d = _sealed_run_dir(root, rd_ev_ref, "pooled_rd EV run")
        meta = json.loads((d / "evidence.json").read_text(encoding="ascii"))
        if meta["evidence"]["mode"] != "pooled_rd":
            raise EvidenceStoreError(f"EV run {rd_ev_ref['run_id']} is mode {meta['evidence']['mode']}, not pooled_rd")
        if meta["limit"] and not allow_limit:
            raise EvidenceStoreError(f"EV run {rd_ev_ref['run_id']} reads a subset pool (limit true); pass allow_limit "
                                     "only in tests and the CPU smoke")
        self.run_id = rd_ev_ref["run_id"]
        self._cache: dict[str, tuple[str, list[tuple[str, float]]]] = {}
        for x in _gz_lines_checked(d / "evidence.jsonl.gz", meta["evidence_sha256"], f"{self.run_id}/evidence.jsonl.gz"):
            self._cache[x["claim_id"]] = (x["claim_hash"], [(str(p), float(s)) for p, s in x["rerank"]])
        pool = meta["pool_run"]
        pdir = _sealed_run_dir(root, {"run_id": pool["run_id"], "sha256": pool["sealed_sha256"]}, "pool EV run")
        pooled = json.loads((pdir / "evidence.json").read_text(encoding="ascii")).get("pooled") or {}
        if not pooled.get("enabled") or pooled.get("passages_sha256") != pool["passages_sha256"]:
            raise EvidenceStoreError(f"pool EV run {pool['run_id']}: pooled bundle differs from the pooled_rd record")
        wanted = {p for _, lst in self._cache.values() for p, _ in lst}
        self._texts = {e["passage_id"]: e["text"] for e in _gz_lines_checked(
            out_root(root) / pooled["passages_path"], pooled["passages_sha256"], "pooled passages",
            keep=lambda e: e["passage_id"] in wanted)}
        if wanted - set(self._texts):
            raise EvidenceStoreError(f"{len(wanted - set(self._texts))} cached RD passages are not in the pooled bundle")

    def item_ids(self) -> list[str]:
        return list(self._cache)

    def _entry(self, item_id: str):
        try:
            return self._cache[item_id]
        except KeyError:
            raise KeyError(f"{item_id!r} is not an RD item of pooled_rd EV run {self.run_id}") from None

    def lookup(self, claim_id: str, text: str) -> CacheHit:
        recorded, rerank = self._entry(claim_id)
        if claim_hash(text) != recorded:
            raise PooledRecomputeRefused(f"{claim_id}: the text differs from the RD text that pooled_rd EV run "
                                         f"{self.run_id} recorded; pooled recompute is not offered (RD is attacked only "
                                         "through victim A, specification item 13)")
        return CacheHit([Passage(p, claim_hash(self._texts[p]), s, self._texts[p]) for p, s in rerank])

    def top1_score(self, claim_id: str) -> float:
        rerank = self._entry(claim_id)[1]
        return float(rerank[0][1]) if rerank else 0.0


class GoldQaEvidence:
    """The gold question-answer evidence of each AVeriTeC claim (key "goldqa"; addition C, the oracle ceiling of the
 gap review). Never a system result.

 Load: the raw train.json / dev.json of `raw_dir` (aac.data.averitec.read_raw; the same files data.claims names),
 read at run time and held in memory; nothing is written or logged, records carry claim_hash only. Per claim, one
 passage per answer of every question, in question then answer order, id `<claim_id>#q<i>a<j>` (the scorer's pair
 ref), text = `gold_text(question, answer)` (the scorer's gold-text formatter of protocol notes row 28, injected by the
 composition root so the format has one implementation; an empty text gives no passage, as in the scorer data).
 Unanswerable answers are kept: the scorer was trained on them (label NEUTRAL), so the passage is in its training
 format.

 lookup(claim_id, text) -> Candidates(rerank_only=True): EvidenceC2 scores every gold passage with the pinned
 re-ranker against the request text, as for FC's SuppliedEvidence. The passages depend on the claim id only (a
 perturbed text reads the same gold evidence). top1_score(claim_id) -> the highest re-ranker score of the clean
 claim text against its gold passages, 6 d.p. (0.0 for a claim without a passage): the relevance-floor input
 (rule p10-top1-rerank-val_cal). It needs a re-ranker bound with bind_reranker (the factory binds EvidenceC2's).

 Which claims are read is decided by the regime, not here: the floor and passes F and A ask for val_cal and val_fuse
 claims only; only the test pass asks for test claims (the oracle). `accessed` records the claim ids asked for, in
 memory, so a test can assert the claim set of each pass (T-04-25)."""

    key = "goldqa"

    def __init__(self, raw_dir: Path, gold_text: Callable[[str, dict], str], *,
                 rerank: Callable[[list[tuple[str, str]]], object] | None = None) -> None:
        from aac.data.averitec import SPLITS, claim_id as averitec_claim_id, read_raw

        raw_dir = Path(raw_dir)
        if not raw_dir.is_dir():
            raise EvidenceStoreError(f"GoldQaEvidence reads the raw AVeriTeC train/dev JSON directory; {raw_dir} is not "
                                     "a directory")
        raw = read_raw(raw_dir)
        self._claims: dict[str, str] = {}
        self._gold: dict[str, list[tuple[str, str]]] = {}
        for split in SPLITS:
            for idx, c in enumerate(raw[split]):
                cid = averitec_claim_id(split, idx)
                self._claims[cid] = str(c.get("claim") or "")
                passages = []
                for qi, q in enumerate(c.get("questions") or []):
                    for ai, a in enumerate(q.get("answers") or []):
                        text = gold_text(str(q.get("question") or ""), a)
                        if text:
                            passages.append((f"{cid}#q{qi}a{ai}", text))
                self._gold[cid] = passages
        self._rerank = rerank
        self._top1: dict[str, float] = {}
        self.accessed: set[str] = set()

    def bind_reranker(self, rerank: Callable[[list[tuple[str, str]]], object]) -> None:
        """rerank(pairs) -> scores aligned with the (claim text, passage text) pairs (sigmoid, [0, 1])."""
        self._rerank = rerank

    def claim_ids(self) -> list[str]:
        return list(self._gold)

    def _entry(self, claim_id: str) -> list[tuple[str, str]]:
        try:
            passages = self._gold[claim_id]
        except KeyError:
            raise KeyError(f"{claim_id!r} is not an AVeriTeC train/dev claim") from None
        self.accessed.add(claim_id)
        return passages

    def n_passages(self, claim_id: str) -> int:
        return len(self._entry(claim_id))

    def lookup(self, claim_id: str, text: str) -> Candidates:
        passages = self._entry(claim_id)
        return Candidates([p for p, _ in passages], [t for _, t in passages], None, rerank_only=True)

    def top1_score(self, claim_id: str) -> float:
        passages = self._entry(claim_id)
        if claim_id in self._top1:
            return self._top1[claim_id]
        if not passages:
            value = 0.0
        else:
            if self._rerank is None:
                raise RuntimeError("GoldQaEvidence.top1_score needs a re-ranker: the factory binds EvidenceC2's")
            claim = self._claims[claim_id]
            scores = [float(x) for x in self._rerank([(claim, t) for _, t in passages])]
            if len(scores) != len(passages):
                raise ValueError(f"{claim_id}: the re-ranker returned {len(scores)} scores for {len(passages)} pairs")
            value = round(max(scores), 6)
        self._top1[claim_id] = value
        return value
