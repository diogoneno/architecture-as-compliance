"""Pipeline.infer: the only caller of RecordBuilder.commit; one record per request."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from aac.pipeline.record import Decision, RunContext


@dataclass(frozen=True)
class Request:
    claim_id: str
    text: str
    evidence_provider_key: str
    regime: str
    split: str
    seed: int
    query_ctx: Any = None


class Pipeline:
    def __init__(self, c1, c2, c3, c4, c5, provider, providers: dict | None = None, device: str = "cpu") -> None:
        self.c1, self.c2, self.c3, self.c4, self.c5, self.provider = c1, c2, c3, c4, c5, provider
        self.device = device  # where C1 runs; the run-record stage entries carry it (gpu_hours counts cuda stages)
        # EvidenceProvider strategy per Request.evidence_provider_key (ARCH section 4.2):
        
        # silently using the default provider (records would carry evidence from the wrong strategy).
        # A Pipeline built without a providers map serves `provider` for every key.
        self.providers = dict(providers or {})

    def provider_for(self, req: Request):
        if not self.providers:
            return self.provider
        try:
            return self.providers[req.evidence_provider_key]
        except KeyError:
            raise KeyError(f"no EvidenceProvider for {req.evidence_provider_key!r} "
                           f"(registered: {sorted(self.providers)})") from None

    def infer(self, reqs: list[Request], ctx: RunContext) -> list[Decision]:
        """One record per request, committed in request order before its Decision exists (sequential ordering rule).

 Batched hooks : when C1 has encode_many(reqs) and, with retrieval on, C2 has
 retrieve_many(reqs, providers), their outputs are computed for the whole batch first (one forward pass
 instead of one per text); fuse, calibration, gate and commit stay per request, in order. A component
 without the hook keeps the per-request path unchanged."""
        decisions: list[Decision] = []
        t = ctx.toggles
        if not reqs:
            return decisions
        c1_outs = None
        if hasattr(self.c1, "encode_many"):
            c1_outs = list(self.c1.encode_many(list(reqs)))
            if len(c1_outs) != len(reqs):
                raise ValueError(f"c1.encode_many returned {len(c1_outs)} outputs for {len(reqs)} requests")
        c2_outs = None
        if t.retrieval and hasattr(self.c2, "retrieve_many"):
            c2_outs = list(self.c2.retrieve_many(list(reqs), [self.provider_for(r) for r in reqs]))
            if len(c2_outs) != len(reqs):
                raise ValueError(f"c2.retrieve_many returned {len(c2_outs)} outputs for {len(reqs)} requests")
        for i, req in enumerate(reqs):
            c1_out = c1_outs[i] if c1_outs is not None else self.c1.encode(req)
            if not t.retrieval:
                c2_out = None
            else:
                c2_out = c2_outs[i] if c2_outs is not None else self.c2.retrieve(req, self.provider_for(req))
            rb = self.c3.fuse(ctx, req, c1_out, c2_out)
            raw = rb.fuse_out["raw_score"]
            cal_out = self.c4.apply(raw) if t.calibration else None
            if t.abstention:
                prob = cal_out["calibrated_probability"] if cal_out is not None else raw
                rb.with_gate(self.c5.apply(prob, c2_out, rb.fuse_out))
            rb.with_calibration(cal_out)
            decisions.append(rb.commit())
        return decisions
