"""Builds a Pipeline from a resolved config. Only the CLI imports this module:
aac.regimes and aac.attacks must never import aac.components (architectural isolation constraint).

The factory is the composition root: it loads the fixture labels (plus the fc/rd item
labels mapped onto the four-way label space for the stub encoder's bias) and registers
every EvidenceProvider the run's targets need under its Request.evidence_provider_key.

: the component classes follow the config kind. components.c2.kind bge-m3-rerank-entail builds the real C2
(EvidenceC2 with the EntailmentScorer of the sealed C2S run components.c2.scorer and the real providers: the
AVeriTeC per-claim store, supplied FC evidence, the pooled RD cache); any other kind keeps StubEvidence and the
fixture providers. : provider goldqa (addition C oracle) registers GoldQaEvidence over the raw AVeriTeC
JSON of data.claims with the scorer's gold-text formatter (aac.components.c2_scorer_train.gold_text) and binds EvidenceC2's
re-ranker to it for top1_score (the relevance floor). c3 gate logistic-fitted -> LogisticGate, content-only-passthrough -> ContentOnlyFusion, else
StubFusion; c4 kind fitted -> FittedCalibration, else StubCalibration; c5 kind fitted-gate -> FittedGate, else
StubGate."""
from __future__ import annotations

from pathlib import Path

import json

from aac.components.c1_encoder import MdebertaEncoder, StubEncoder
from aac.components.c2_evidence import StubEvidence
from aac.components.c3_fusion import ContentOnlyFusion, LogisticGate, StubFusion
from aac.components.c4_calibration import FittedCalibration, StubCalibration
from aac.components.c5_abstention import FittedGate, StubGate
from aac.data.evidence_providers import (AveritecPerClaimStore, FixtureEvidence, GoldQaEvidence, PooledFixtureEvidence,
                                         PooledRdStore, SuppliedEvidence, load_incrediblae_items, load_items)
from aac.data.fixtures import load_labels
from aac.pipeline.pipeline import Pipeline
from aac.run.config import FITTED_C4_KIND, FITTED_C5_KIND, LOGISTIC_GATE, REAL_C2_KIND

BINARY_TO_LABEL = {1: "Refuted", 0: "Supported"}  # InCrediblAE label 1 = non-credible = the positive class


def _real_providers(c2: dict, data: dict, root: Path) -> dict:
    """The real EvidenceProviders by key (classes): averitec_store and pooled read the EV run components.c2
 .evidence (resolved by aac.run.identity), supplied reads the run-time FC texts of data.fc_items."""
    providers = {}
    if c2["provider"] == "averitec_store":
        providers["averitec_store"] = AveritecPerClaimStore(root, c2["evidence"], c2["variant"])
    if c2["provider"] == "pooled":
        providers["pooled"] = PooledRdStore(root, c2["evidence"])
    if c2["provider"] == "goldqa":  # addition C: the oracle reads the gold QA evidence of data.claims (raw JSON)
        from aac.components.c2_scorer_train import gold_text

        providers["goldqa"] = GoldQaEvidence(root / data["claims"], gold_text)
    if data.get("fc_items"):
        providers["supplied"] = SuppliedEvidence(load_incrediblae_items(root, data["fc_items"]))
    return providers


def bind_goldqa(providers: dict, c2) -> None:
    """Addition C: GoldQaEvidence.top1_score scores the clean claim against its gold passages with EvidenceC2's pinned
 re-ranker (loaded on first use, at the batch size of components.c2.reranker), the one EvidenceC2 itself uses on
 the gold Candidates."""
    gq = providers.get("goldqa")
    if gq is None or c2 is None:
        return
    batch = int(c2.cfg["reranker"]["batch_size"])
    gq.bind_reranker(lambda pairs: c2.reranker.score(pairs, batch_size=batch))


def _real_c2(c2: dict, root: Path, device: str):
    """EvidenceC2 with the EntailmentScorer of the sealed C2S run components.c2.scorer; max_length is the one the C2S
 run's model.json records (the scorer refuses any other)."""
    from aac.components.c2_retrieval import EvidenceC2
    from aac.components.c2_scorer import EntailmentScorer
    from aac.run.paths import runs_dir

    ref = c2.get("scorer")
    if not ref:
        raise ValueError("components.c2.scorer is unset: aac.run.identity resolves it from scorer_trained_by")
    meta = json.loads((runs_dir(root) / ref["run_id"] / "model.json").read_text(encoding="utf-8"))
    scorer = EntailmentScorer(ref, root, max_length=int(meta["max_length"]))
    return EvidenceC2(c2, root, scorer=scorer, device=device)


def _c3(cfg: dict):
    if cfg["gate"] == LOGISTIC_GATE:
        return LogisticGate(cfg)
    if cfg["gate"] == "content-only-passthrough":
        return ContentOnlyFusion(cfg)
    return StubFusion(cfg)


def build_pipeline(resolved: dict, root: Path, seed: int) -> Pipeline:
    comps = resolved["components"]
    data = resolved["data"]
    real_c2 = comps["c2"]["kind"] == REAL_C2_KIND
    labels = load_labels(root / data["labels"])
    providers = {}
    if not real_c2 and data["dataset"] == "fixture" and data.get("evidence"):
        providers["fixture"] = FixtureEvidence(root / data["evidence"])
        providers["pooled"] = PooledFixtureEvidence(root / data["evidence"])
    if data.get("fc_items"):
        fc = load_items(root / data["fc_items"])
        if not real_c2:
            providers["supplied"] = SuppliedEvidence(fc)
        labels.update({iid: BINARY_TO_LABEL[int(it["label"])] for iid, it in fc.items()})
    if data.get("rd_items"):
        rd = load_items(root / data["rd_items"])
        labels.update({iid: BINARY_TO_LABEL[int(it["label"])] for iid, it in rd.items()})
    if real_c2 and comps["c2"]["enabled"]:
        providers.update(_real_providers(comps["c2"], data, root))
    key = comps["c2"]["provider"]
    if comps["c2"]["enabled"] and key not in providers:
        raise ValueError(f"components.c2.provider {key!r} is not available for this run's data (have {sorted(providers)})")
    if comps["c1"]["kind"] == "mdeberta-v3-base":
        c1 = MdebertaEncoder(comps["c1"], root, seed)
    else:
        c1 = StubEncoder(comps["c1"], labels, seed)
    device = getattr(c1, "device", "cpu")
    if not real_c2:
        c2 = StubEvidence(comps["c2"])
    else:  # a disabled real C2 is never called (retrieval off): no weights file is loaded for it
        c2 = _real_c2(comps["c2"], root, device) if comps["c2"]["enabled"] else None
    bind_goldqa(providers, c2)
    c4 = FittedCalibration(comps["c4"]) if comps["c4"].get("kind") == FITTED_C4_KIND else StubCalibration(comps["c4"])
    c5 = FittedGate(comps["c5"]) if comps["c5"].get("kind") == FITTED_C5_KIND else StubGate(comps["c5"])
    return Pipeline(
        c1=c1,
        c2=c2,
        c3=_c3(comps["c3"]),
        c4=c4,
        c5=c5,
        provider=providers.get(key),
        providers=providers,
        device=device,
    )
