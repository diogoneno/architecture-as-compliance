"""C1 encoders. StubEncoder: a 4-way distribution from a hash-seeded Dirichlet biased toward the fixture
label . MdebertaEncoder: the real C1 , mDeBERTa-v3-base with the sealed fine-tuned
checkpoint of the C1 training run named by components.c1.checkpoint (hash-gated before any load).

Both return {"encoder_distribution": [4 floats, 6 d.p., probs[0] = 1 - sum(rest)]} in the label order of
aac.data.fixtures.LABELS (index 1 = Refuted). torch/transformers are imported inside functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from aac.data.fixtures import LABELS
from aac.run.hashing import claim_hash, derive_seed


class StubEncoder:
    name = "stub-dirichlet-v1"

    def __init__(self, cfg: dict, labels: dict[str, str], seed: int) -> None:
        self.cfg = cfg
        self.labels = labels
        self.base_seed = derive_seed(f"stub.c1.{seed}")

    def encode(self, req) -> dict:
        rng = np.random.default_rng([self.base_seed, int(claim_hash(req.text)[:8], 16)])
        alpha = np.ones(4)
        label = self.labels.get(req.claim_id)
        if label in LABELS:
            alpha[LABELS.index(label)] += float(self.cfg["concentration"])
        probs = rng.dirichlet(alpha)
        return {"encoder_distribution": round_distribution(probs)}


ROUNDING_SLACK = 2e-6  # three components rounded up by at most 5e-7 each (plus float noise)


def round_distribution(probs) -> list[float]:
    """6 d.p., probs[0] = 1 - sum(rest), so the four floats sum to 1 at 6 d.p.

 when P(Supported) is below about 1.5e-6 and the others round up, that residual is
 negative. Then probs[0] is 0.0 and the residual goes to the largest other component. Every distribution whose
 residual is >= 0 (every record sealed so far) is unchanged byte for byte. A component outside [0, 1] raises."""
    p = np.round(np.asarray(probs, dtype=np.float64), 6)
    p[0] = round(1.0 - float(p[1:].sum()), 6)
    if p[0] < -ROUNDING_SLACK:  # not rounding noise: the input was not a probability distribution
        raise ValueError(f"distribution {np.asarray(probs).tolist()} sums to more than 1 beyond rounding")
    if p[0] < 0.0:
        p[0] = 0.0
        i = 1 + int(np.argmax(p[1:]))
        p[i] = round(1.0 - float(p.sum() - p[i]), 6)
    if (p < 0.0).any() or (p > 1.0).any():
        raise ValueError(f"rounded distribution {p.tolist()} has a component outside [0, 1]")
    return [float(x) for x in p]


def _load_base(c1: dict, root: Path):
    from aac.components.c1_train import load_base_model

    return load_base_model({"hf_id": c1["hf_id"], "revision": c1["revision"], "num_labels": c1["num_labels"]}, root)


class MdebertaEncoder:
    name = "mdeberta-v3-base"

    def __init__(self, cfg: dict, root: Path, seed: int) -> None:
        from aac.run.checkpoints import ChecksumMismatch, sha256_file
        from aac.run.paths import runs_dir

        ref = cfg.get("checkpoint")
        if not ref:
            raise ValueError("components.c1.checkpoint is unset: identity() resolves it from components.c1.trained_by")
        path = runs_dir(root) / ref["run_id"] / "checkpoint.safetensors"
        actual = sha256_file(path)  
        if actual != ref["sha256"]:
            raise ChecksumMismatch(f"{path}: sha256 {actual} != components.c1.checkpoint.sha256 {ref['sha256']}; not loaded")
        import torch
        from safetensors.torch import load_file

        self.cfg = cfg
        self.seed = seed
        self.max_length = int(cfg["max_length"])
        self.tokenizer, model = _load_base(cfg, root)
        model.load_state_dict(load_file(str(path), device="cpu"), strict=True)
        from aac.components.c1_train import assert_fp32

        assert_fp32(model)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device).eval()

    def distribution(self, text: str) -> list[float]:
        import torch

        enc = self.tokenizer([text], truncation=True, max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits.float()
        return round_distribution(torch.softmax(logits[0], dim=-1).cpu().double().numpy())

    def encode(self, req) -> dict:
        return {"encoder_distribution": self.distribution(req.text)}

    def encode_many(self, reqs) -> list[dict]:
        """One padded, truncated batch, one inference_mode forward; round_distribution per row, request order.
 Padding can move a probability by float noise; tests bound it at 1e-5 per component (operational constraint)."""
        if not reqs:
            return []
        import torch

        enc = self.tokenizer([r.text for r in reqs], padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits.float()
        probs = torch.softmax(logits, dim=-1).cpu().double().numpy()
        return [{"encoder_distribution": round_distribution(row)} for row in probs]


def _nearest_rank(sorted_values: list[int], pct: float) -> int:
    """Nearest-rank percentile: the value at rank ceil(pct/100 * n) (1-based) of the sorted list (an observed length)."""
    import math

    n = len(sorted_values)
    return int(sorted_values[max(1, math.ceil(pct / 100.0 * n)) - 1])


def token_stats(texts, tokenizer, max_length: int) -> dict:
    """Length of every text in tokenizer tokens INCLUDING the special tokens, untruncated (the empty string
 counts 2 for mDeBERTa: [CLS] [SEP]); p50/p95/p99 by nearest rank, max, and how many exceed max_length."""
    lengths = sorted(len(tokenizer(t, add_special_tokens=True, truncation=False)["input_ids"]) for t in texts)
    if not lengths:
        raise ValueError("token_stats needs at least one text")
    return {"n": len(lengths), "max_length": int(max_length), "p50": _nearest_rank(lengths, 50),
            "p95": _nearest_rank(lengths, 95), "p99": _nearest_rank(lengths, 99), "max": lengths[-1],
            "n_over_max_length": sum(1 for n in lengths if n > max_length), "percentile_rule": "nearest-rank",
            "special_tokens_included": True}


def main(argv: list[str] | None = None) -> int:
    """`python -m aac.components.c1_encoder token-stats --out data/derived/averitec/token-stats.json`:
 every labeled AVeriTeC claim (train + dev, raw JSON read at run time; only counts are written)."""
    import argparse

    from aac.data.averitec import load_claims
    from aac.run.config import TrainCfg, load_raw
    from aac.run.hashing import dumps_sorted
    from aac.run.paths import root as project_root

    ap = argparse.ArgumentParser(prog="python -m aac.components.c1_encoder")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("token-stats")
    p.add_argument("--out", required=True)
    p.add_argument("--claims", default="data/raw/averitec/data")
    p.add_argument("--recipe", default="configs/train/c1.yaml", help="hf_id, revision and max_length come from here")
    args = ap.parse_args(argv)
    root = project_root()
    recipe = TrainCfg.model_validate(load_raw(root / args.recipe)["train"])  # c1.yaml is the recipe block only
    from aac.components.c1_train import load_tokenizer

    tok = load_tokenizer(recipe.model.hf_id, recipe.model.revision, root)
    claims = load_claims(root / args.claims)
    stats = token_stats([claims[c]["text"] for c in sorted(claims)], tok, recipe.max_length)
    stats["tokenizer"] = {"hf_id": recipe.model.hf_id, "revision": recipe.model.revision}
    stats["source"] = "aac.data.averitec.load_claims(data/raw/averitec/data): all labeled train+dev claims"
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    try:  
        tmp.write_text(dumps_sorted(stats), encoding="ascii")
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    print(f"token-stats: n={stats['n']} p50={stats['p50']} p95={stats['p95']} p99={stats['p99']} max={stats['max']} "
          f"n_over_max_length={stats['n_over_max_length']} (max_length {stats['max_length']}) -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
