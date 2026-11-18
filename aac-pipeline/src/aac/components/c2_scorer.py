"""C2 entailment scorer, inference side .

EntailmentScorer loads the fine-tuned mDeBERTa-v3-base weights file of a sealed C2S training run
(aac.components.c2_scorer_train) and returns (P(SUPPORT), P(REFUTE), P(NEUTRAL)) per (passage, claim) pair:

 1. hash gate first: the sha256 of checkpoint.safetensors must equal both the ref's sha256 (resolved into config_hash by
 aac.run.identity.resolve_real_c2_scorer) and the value the sealed model.json records; nothing is loaded otherwise;
 2. the base model at the pinned revision, offline, float32 weights (asserted), the state dict loaded strictly;
 3. pairs are tokenised passage first with truncation "only_first" at the training max_length (the claim is never cut),
 scored in fixed batches under inference_mode in float32; softmax; 6 d.p. with the first value = 1 - sum(rest).
 One exception ((a), 2026-10-17): a pair the caller marks `cut`
 whose claim alone leaves no room is re-encoded with "longest_first" (CUT_TRUNCATION), so both texts keep their
 first tokens and the pair fills the max_length window. score_many_flagged returns one truncation flag per pair.
 EvidenceC2 marks only RD requests (provider key `pooled`); every other over-long pair is still refused.

Plans 03-07 and 03-09 call score_many; the fusion aggregates are theirs (methodological note).
CLI (CPU smoke only): `python -m aac.components.c2_scorer smoke --run-id <C2S run id>` scores one invented pair.
torch and transformers are imported inside functions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aac.components.c1_encoder import round_distribution
from aac.run.config import SCORER_LABELS

CHECKPOINT = "checkpoint.safetensors"
MODEL_JSON = "model.json"
TRUNCATION = "only_first"   # the passage (first sequence) is cut; the claim is kept whole
CUT_TRUNCATION = "longest_first"   # item 15 (a): a marked pair whose claim alone does not fit keeps the first tokens


class ScorerInputTooLong(ValueError):
    """A claim alone is longer than max_length, so "only_first" truncation cannot fit the pair."""


def _overflow_error(exc: Exception) -> bool:
    """The fast (Rust) tokenizer raises a plain Exception "Truncation error: Sequence to truncate too short ..." when
 "only_first" cannot fit a pair; older versions returned the pair untruncated with a log message instead."""
    return "Truncation error" in str(exc)


def tokenize_pairs(tok, passages: list[str], claims: list[str], max_length: int):
    """Passage-first pair encoding with "only_first" truncation and longest padding; refuses an over-long pair (the
 tokenizer either returns it untruncated with only a log message or raises a plain truncation Exception)."""
    try:
        enc = tok(list(passages), list(claims), padding="longest", truncation=TRUNCATION, max_length=int(max_length),
                  return_tensors="pt")
    except Exception as exc:  # noqa: BLE001 (the tokenizers library raises a bare Exception)
        if _overflow_error(exc):
            raise ScorerInputTooLong(f"a (passage, claim) pair does not fit max_length {max_length}: the claim alone "
                                     "does not fit") from exc
        raise
    if int(enc["input_ids"].shape[1]) > int(max_length):
        raise ScorerInputTooLong(f"a (passage, claim) pair encodes to {int(enc['input_ids'].shape[1])} tokens > "
                                 f"max_length {max_length}: the claim alone does not fit")
    return enc


def tokenize_pairs_cut(tok, passages: list[str], claims: list[str], max_length: int, cut: list[bool]):
    """(encoding, flags): tokenize_pairs, except that a pair with cut[i] True that "only_first" cannot fit (its claim
 alone is too long) is re-encoded with CUT_TRUNCATION ("longest_first": both texts keep their first tokens, the pair
 fills max_length) and flagged True. When no pair overflows, the encoding is exactly tokenize_pairs' output. An
 over-long pair with cut[i] False raises ScorerInputTooLong, as before (item 15, option (a))."""
    passages, claims, cut = list(passages), list(claims), [bool(c) for c in cut]
    if not len(passages) == len(claims) == len(cut):
        raise ValueError(f"{len(passages)} passages, {len(claims)} claims and {len(cut)} cut flags must be equal")
    try:
        return tokenize_pairs(tok, passages, claims, max_length), [False] * len(passages)
    except ScorerInputTooLong:
        pass
    feats, flags = [], []
    for passage, claim, may_cut in zip(passages, claims, cut, strict=True):
        try:
            one = tok(passage, claim, truncation=TRUNCATION, max_length=int(max_length))
            flag = len(one["input_ids"]) > int(max_length)
        except Exception as exc:  # noqa: BLE001 (the tokenizers library raises a bare Exception)
            if not _overflow_error(exc):
                raise
            flag = True
        if flag:
            if not may_cut:
                raise ScorerInputTooLong(f"a (passage, claim) pair does not fit max_length {max_length}: the claim "
                                         "alone does not fit")
            one = tok(passage, claim, truncation=CUT_TRUNCATION, max_length=int(max_length))
            if len(one["input_ids"]) > int(max_length):
                raise ScorerInputTooLong(f"{CUT_TRUNCATION} could not fit a pair into max_length {max_length}")
        feats.append(dict(one))
        flags.append(flag)
    return tok.pad(feats, padding="longest", return_tensors="pt"), flags


class EntailmentScorer:
    name = "mdeberta-v3-base-entailment"

    def __init__(self, checkpoint_ref: dict, root: Path, *, max_length: int, batch_size: int = 32) -> None:
        from aac.run.checkpoints import ChecksumMismatch, sha256_file
        from aac.run.paths import runs_dir

        if not checkpoint_ref or not checkpoint_ref.get("run_id") or not checkpoint_ref.get("sha256"):
            raise ValueError("the scorer ref is unset: aac.run.identity resolves components.c2.scorer from "
                             "components.c2.scorer_trained_by")
        if int(batch_size) <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        run_dir = runs_dir(root) / checkpoint_ref["run_id"]
        path = run_dir / CHECKPOINT
        actual = sha256_file(path)  
        meta = json.loads((run_dir / MODEL_JSON).read_text(encoding="utf-8"))
        if actual != checkpoint_ref["sha256"] or actual != meta["checkpoint_sha256"]:
            raise ChecksumMismatch(f"{path}: sha256 {actual} != ref {checkpoint_ref['sha256']} / model.json "
                                   f"{meta['checkpoint_sha256']}; not loaded")
        if tuple(meta["labels"]) != SCORER_LABELS or int(meta["num_labels"]) != len(SCORER_LABELS):
            raise ValueError(f"{run_dir.name}/model.json labels {meta['labels']} are not the scorer labels {list(SCORER_LABELS)}")
        if int(meta["max_length"]) != int(max_length):
            raise ValueError(f"max_length {max_length} differs from the training max_length {meta['max_length']} of "
                             f"{run_dir.name}")
        import torch
        from safetensors.torch import load_file

        from aac.components.c1_train import assert_fp32, load_base_model

        self.run_id = checkpoint_ref["run_id"]
        self.checkpoint_sha256 = actual
        self.max_length = int(max_length)
        self.batch_size = int(batch_size)
        self.tokenizer, model = load_base_model({"hf_id": meta["hf_id"], "revision": meta["revision"],
                                                 "num_labels": len(SCORER_LABELS)}, root)
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        assert_fp32(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def score_many(self, pairs: list[tuple[str, str]]) -> list[tuple[float, float, float]]:
        """(P(SUPPORT), P(REFUTE), P(NEUTRAL)) per (passage_text, claim_text) pair, in input order, 6 d.p."""
        return self.score_many_flagged(pairs, [False] * len(pairs))[0]

    def score_many_flagged(self, pairs: list[tuple[str, str]], cut: list[bool]
                           ) -> tuple[list[tuple[float, float, float]], list[bool]]:
        """score_many plus one truncation flag per pair: a pair with cut[i] True whose claim alone does not fit is
 scored on its "longest_first" encoding and flagged (tokenize_pairs_cut; item 15, option (a))."""
        if len(cut) != len(pairs):
            raise ValueError(f"{len(pairs)} pairs but {len(cut)} cut flags")
        if not pairs:
            return [], []
        import torch

        out: list[tuple[float, float, float]] = []
        flags: list[bool] = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc, chunk_flags = tokenize_pairs_cut(self.tokenizer, [p for p, _ in chunk], [c for _, c in chunk],
                                                  self.max_length, list(cut[i:i + self.batch_size]))
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.inference_mode():
                logits = self.model(**enc).logits.float()
            probs = torch.softmax(logits, dim=-1).cpu().double().numpy()
            out.extend(tuple(round_distribution(row)) for row in probs)
            flags.extend(chunk_flags)
        return out, flags


SMOKE_PAIR = ("Tinyworld smoke passage: the harbour council reported that the bridge toll rose by four points.",
              "The harbour council says the bridge toll rose.")   # invented text, never AVeriTeC


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.components.c2_scorer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("smoke", help="load a sealed C2S run through the hash gate and score one invented pair")
    p.add_argument("--run-id", required=True)
    a = ap.parse_args(argv)
    from aac.run.checkpoints import sha256_file
    from aac.run.paths import root as project_root
    from aac.run.paths import runs_dir
    from aac.run.seal import verify_sealed

    root = project_root()
    run_dir = runs_dir(root) / a.run_id
    verify_sealed(run_dir)
    meta = json.loads((run_dir / MODEL_JSON).read_text(encoding="utf-8"))
    ref = {"run_id": a.run_id, "sha256": sha256_file(run_dir / CHECKPOINT)}
    scorer = EntailmentScorer(ref, root, max_length=int(meta["max_length"]), batch_size=4)
    (probs,) = scorer.score_many([SMOKE_PAIR])
    print(f"scorer-smoke: {a.run_id} {scorer.checkpoint_sha256[:12]} sum={round(sum(probs), 6)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
