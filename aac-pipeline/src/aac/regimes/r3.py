"""R3 (adversarial) stub driver: one sealed run per (config, seed) whose attack ledger holds
the (attack, victim, subset) cells of as rows.

Protocol (pre-registered specification):
 victim A `content_only` C1 only (retrieval, calibration, abstention off), subsets averitec_chrono, fc, rd
 victim B `full` C1->C4 un-gated (abstention off), AVeriTeC subset only; C4/C5 are the sealed
 R1 run's fitted maps (attack.victim_source, same seed): this run holds NO fit pass,
 so every record of the run is an attack-ledger query (invariant L2, strict)
 attacked set positive items (targeted, y = 1) whose CLEAN decision is correct; items whose clean
 decision is already wrong get a row with queries_used 0 (never attacked)
 queries every proposal is a Request through Pipeline.infer with query_ctx; the clean query
 and the post-hoc gate query are logged but not charged; attack-phase queries are
 charged to QueryBudget before inference (BudgetExhausted -> end_state budget_exhausted)
 end states exactly one of unchanged | success (wrong auto-decision) | deferred | budget_exhausted
 post-hoc gate (B) the flipped example re-run through the gated pipeline: deferred -> end_state deferred,
 posthoc_gate deflected; auto-decided -> success, flip; not flipped -> no_effect
 adversarial text never stored: attack-adversarial.jsonl keeps (item, attack, round, claim_hash) and
 the R4 replay regenerates the text from the stub adversary 
BODEGA score is n/a on stubs (no field in the ledger contract; Phase 4 adds it)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from aac.attacks.budget import BudgetExhausted, QueryBudget
from aac.attacks.ledger import LedgerWriter, write_adversarial
from aac.attacks.stub_adversary import StubAdversary
from aac.data.fixtures import load_split
from aac.pipeline import Pipeline, Request, RunContext, Toggles
from aac.regimes.common import check_toggles, load_source_maps, regime_label, select_source_run, target_items
from aac.run.hashing import claim_hash

VICTIM_TOGGLES = {
    "content_only": Toggles(retrieval=False, calibration=False, abstention=False),
    "full": Toggles(retrieval=True, calibration=True, abstention=False),  # un-gated C1->C4 
}
GATED = Toggles(retrieval=True, calibration=True, abstention=True)
PROVIDER_KEY = {"fc": "supplied", "rd": "pooled"}


__all__ = ["run", "select_source_run", "GATED", "PROVIDER_KEY", "VICTIM_TOGGLES"]  # select_source_run lives in common.py 


def _is_false(decision) -> bool:
    return decision.calibrated_probability >= 0.5  # binary decision: Refuted positive 


def run(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path) -> dict:
    atk = resolved["attack"]
    if atk is None:
        raise ValueError("R3 needs an `attack` block")
    if atk["adversary"] != "stub":
        raise ValueError("this driver runs the stub adversary only (Phase 3 adds the localhost victim)")
    check_toggles(resolved)
    seed = int(resolved["seed"])
    src_id = load_source_maps(pipeline, resolved, root, run_dir, atk["victim_source"])  

    subset_ids = {}
    if resolved["data"].get("attack_subset"):
        subset_ids["averitec_chrono"] = sorted(load_split(root / resolved["data"]["attack_subset"])["ids"])
    regime = regime_label(resolved)
    budget_limit = int(atk["budget"])
    ledger = LedgerWriter(run_dir, root)
    adversarial: list[dict] = []
    stages: list[dict] = []
    timings: list[dict] = []
    items_count: dict[str, int] = {}
    n_queries = 0

    def infer_one(req: Request, toggles: Toggles):
        return pipeline.infer([req], dataclasses.replace(ctx, toggles=toggles))[0]

    for attack in atk["attacks"]:
        adversary = StubAdversary(attack, seed)
        for victim in ("content_only", "full"):
            toggles = VICTIM_TOGGLES[victim]
            for subset in atk["victims"][victim]:
                t0 = time.perf_counter()
                items = target_items(root, resolved, subset)
                ids = subset_ids.get(subset, sorted(items))
                key = PROVIDER_KEY.get(subset, resolved["components"]["c2"]["provider"])
                positives = [i for i in ids if items[i]["positive"]]  # targeted mode: y = 1 only
                items_count[f"{subset}_positive"] = len(positives)
                cell_queries = 0
                for item_id in positives:
                    text = items[item_id]["text"]
                    split = f"attack_{subset}"

                    def req_for(cand: str, qi: int, phase: str) -> Request:
                        return Request(claim_id=item_id, text=cand, evidence_provider_key=key, regime=regime, split=split,
                                       seed=seed, query_ctx={"item_id": item_id, "attack": attack, "victim": victim,
                                                             "subset": subset, "query_index": qi, "phase": phase})

                    record_ids = []
                    clean = infer_one(req_for(text, 0, "clean"), toggles)
                    record_ids.append(clean.record_id)
                    cell_queries += 1
                    if not _is_false(clean):  # clean decision already wrong: in the subset, never attacked
                        ledger.append({"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                                       "queries_used": 0, "budget": budget_limit, "end_state": "unchanged",
                                       "posthoc_gate": None, "record_ids": record_ids})
                        adversarial.append({"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                                            "round": None, "claim_hash": None, "attacked": False})
                        continue
                    budget = QueryBudget(budget_limit)
                    end_state = None
                    flipped = False
                    final_text, final_round = None, None
                    rnd = 0
                    while True:
                        cand = adversary.propose(text, rnd)
                        if cand is None:
                            end_state = "unchanged"
                            break
                        try:
                            budget.charge(1)
                        except BudgetExhausted:
                            end_state = "budget_exhausted"
                            break
                        d = infer_one(req_for(cand, budget.used, "attack"), toggles)
                        record_ids.append(d.record_id)
                        cell_queries += 1
                        final_text, final_round = cand, rnd
                        if not _is_false(d):
                            flipped = True
                            end_state = "success"
                            break
                        rnd += 1
                    posthoc = None
                    if victim == "full":
                        if flipped:
                            g = infer_one(req_for(final_text, budget.used, "posthoc"), GATED)
                            record_ids.append(g.record_id)
                            cell_queries += 1
                            if g.abstention_flag:
                                posthoc, end_state = "deflected", "deferred"
                            else:
                                posthoc = "flip"
                        else:
                            posthoc = "no_effect"
                    ledger.append({"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                                   "queries_used": budget.used, "budget": budget_limit, "end_state": end_state,
                                   "posthoc_gate": posthoc, "record_ids": record_ids})
                    adversarial.append({"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                                        "round": final_round, "claim_hash": claim_hash(final_text) if final_text else None,
                                        "attacked": True})
                seconds = round(time.perf_counter() - t0, 6)
                stage = f"attack_{attack}_{victim}_{subset}"
                timings.append({"stage": stage, "seconds": seconds, "items": cell_queries})
                stages.append({"name": stage, "device": "cpu", "seconds": seconds if ctx.clock.source == "wall" else 0.0,
                               "items": cell_queries})
                n_queries += cell_queries
    ledger.close()
    write_adversarial(run_dir, adversarial)
    cells = sorted({(r["attack"], r["victim"], r["subset"]) for r in ledger.rows})
    (run_dir / "attack-cells.json").write_text(
        json.dumps({"cells": [list(c) for c in cells], "rows": len(ledger.rows), "queries": n_queries}, indent=1) + "\n",
        encoding="ascii")
    items_count.update({"ledger_rows": len(ledger.rows), "cells": len(cells), "attack_subset": len(subset_ids.get("averitec_chrono", []))})
    return {
        "configurations": [f"{victim}/{atk['victim_source'] if victim == 'full' else 'c1-only'}" for victim in ("content_only", "full")],
        "items": items_count, "stages": stages, "timings": timings,
        "n_fit": 0, "n_test": n_queries, "depends_on": [src_id], "l2_strict": True,
    }
