"""C2 stub: passages from an EvidenceProvider; coverage = any score >= relevance_floor."""
from __future__ import annotations


class StubEvidence:
    name = "stub-fixture-v1"

    def __init__(self, cfg: dict) -> None:
        self.k = int(cfg["k_retrieve"])
        self.floor = float(cfg["relevance_floor"])

    def retrieve(self, req, provider) -> dict:
        passages = provider.retrieve(req.claim_id, req.text, self.k)
        return {
            "evidence_ids": [p.id for p in passages],
            "evidence_scores": [round(p.score, 6) for p in passages],
            "evidence_coverage": any(p.score >= self.floor for p in passages),
            "_stances": [int(p.text_hash[:2], 16) % 2 for p in passages],  # 0 support, 1 refute (stub)
        }
