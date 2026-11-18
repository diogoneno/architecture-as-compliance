"""R4 (ablations): the ablated system (a `system` toggle on the one Pipeline.infer path, with
C3/C4/C5 refitted on the val partitions) evaluated over the R1 chronological test split
(records split=averitec_chrono), FC and RD (split=fc / rd), so one run per (config, seed)
covers Table 4.5's three evaluation columns.

Stub replay rule (`replay: {source, seed}`, ReplayCfg): for minus-calibration and minus-abstention the run also
re-runs victim B's final adversarial examples from the sealed stub R3 run through the ablated config (records
split=averitec_chrono_replay), regenerating them from the stub adversary, and writes a replay attack ledger (one query
per item). Kept byte for byte so the stub goldens do not move.

Store replay (`replay: {kind: private-store, sources, victim, subset, seed}`, StoreReplayCfg; ,
pre-registered specification ):
 maps components.c3.fitted_from (required; `R4/<name>` = the seed-0 R4 metric run of this system, a bare name =
 the R1 full run); loaded by evaluate_targets with an EMPTY target map, so the run holds no fit pass and
 no test-split record and every record is a ledger query (invariant L2, strict)
 sources each named sealed R3 run at replay.seed (resolved by name in runs/INDEX.json; its seal recomputed against
 the INDEX row); its ledger rows of replay.victim on replay.subset with queries_used > 0 (attacked items)
 texts the clean text from the AVeriTeC test partition (target_items); the final text from the R3 run's private
 store on this host (aac.attacks.private_store), its claim_hash asserted against the store line and the
 R3 run's attack-adversarial.jsonl (replay_plan); nothing is regenerated
 queries per source row one clean query (split averitec_chrono_replay_clean, query_ctx phase clean) and, when the
 ablated clean decision is correct (and auto-decided under a gated system) and a stored final exists, one
 final query (split averitec_chrono_replay, phase replay; Request.query_ctx makes a too-long final text
 cuttable for the scorer pair, methodological note)
 ledger one row per source row (replay_outcome): queries_used 0 = excluded (ablated clean wrong or deferred),
 1 = in the denominator; end_state success (auto-decided wrong), deferred (gated deferral, posthoc_gate
 deflected) or unchanged; record_ids list every logged record. Replay ASR = successes / rows with
 queries_used 1 (replay_asr). Seed 0 only (pre-registered specification)."""
from __future__ import annotations

import dataclasses
import time
from collections import OrderedDict
from pathlib import Path

from aac.attacks.ledger import LedgerWriter, read_adversarial, read_ledger
from aac.attacks.private_store import read_private_store
from aac.attacks.stub_adversary import StubAdversary
from aac.pipeline import Pipeline, Request, RunContext
from aac.regimes.common import (SourceMismatch, evaluate_targets, regime_label, target_items, targets_of,
                                 write_replay_truncations, write_scorer_truncations)
from aac.regimes.r3 import select_source_run
from aac.run.hashing import claim_hash
from aac.run.index import load_index
from aac.run.seal import verify_sealed
from aac.run.paths import runs_dir

SPLIT = {"averitec_chrono": "averitec_chrono", "fc": "fc", "rd": "rd"}
REPLAY_SPLIT = "averitec_chrono_replay"
REPLAY_SYSTEMS = ("ablation-no-calibration", "ablation-no-abstention")
STORE_REPLAY_KIND = "private-store"
REPLAY_CLEAN_SPLIT = "averitec_chrono_replay_clean"
STORE_REPLAY_SYSTEMS = ("full", "ablation-no-retrieval", "ablation-no-calibration", "ablation-no-abstention")
DECIMALS = 6


class ReplayMismatch(RuntimeError):
    """A store replay input does not agree with the sealed R3 run (a hash, a success without a stored final, an item
 outside the target). The run fails; no message carries a text."""


def run(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path) -> dict:
    replay = resolved.get("replay")
    if replay is not None and replay.get("kind") == STORE_REPLAY_KIND:
        return store_replay(pipeline, ctx, resolved, root, run_dir)
    targets = targets_of(resolved, ["averitec_chrono", "fc", "rd"])
    summary = evaluate_targets(pipeline, ctx, resolved, root, run_dir, {t: SPLIT[t] for t in targets})
    if "rd" in targets:  # methodological note)
        write_scorer_truncations(pipeline, run_dir, summary, ["rd"])
    if replay is None:
        return summary
    if resolved["system"] not in REPLAY_SYSTEMS:
        raise ValueError(f"replay applies to {REPLAY_SYSTEMS}, not to system {resolved['system']!r}")
    src = select_source_run(root, regime="R3", name=replay["source"], seed=int(replay["seed"]), system="full")
    src_dir = runs_dir(root) / src["run_id"]
    verify_sealed(src_dir)
    rows = [r for r in read_ledger(src_dir) if r["victim"] == "full" and int(r["queries_used"]) > 0]
    adv = {(a["item_id"], a["attack"], a["victim"], a["subset"]): a for a in read_adversarial(src_dir)}
    items = target_items(root, resolved, "averitec_chrono")
    toggles = ctx.toggles
    gated = toggles.abstention
    seed = int(resolved["seed"])
    ledger = LedgerWriter(run_dir, root)
    t0 = time.perf_counter()
    n = 0
    for r in rows:
        a = adv[(r["item_id"], r["attack"], r["victim"], r["subset"])]
        text = items[r["item_id"]]["text"]
        cand = StubAdversary(r["attack"], int(replay["seed"])).propose(text, int(a["round"]))
        if cand is None or claim_hash(cand) != a["claim_hash"]:
            raise RuntimeError(f"replay: regenerated adversarial text for {r['item_id']}/{r['attack']} does not match the R3 claim_hash")
        req = Request(claim_id=r["item_id"], text=cand, evidence_provider_key=resolved["components"]["c2"]["provider"],
                      regime=regime_label(resolved), split=REPLAY_SPLIT, seed=seed,
                      query_ctx={"item_id": r["item_id"], "attack": r["attack"], "victim": "full", "subset": r["subset"],
                                 "query_index": 1, "phase": "replay"})
        d = pipeline.infer([req], dataclasses.replace(ctx, toggles=toggles))[0]
        n += 1
        if d.abstention_flag:
            end_state, posthoc = "deferred", "deflected"
        elif d.calibrated_probability < 0.5:
            end_state, posthoc = "success", ("flip" if gated else None)
        else:
            end_state, posthoc = "unchanged", ("no_effect" if gated else None)
        ledger.append({"item_id": r["item_id"], "attack": r["attack"], "victim": "full", "subset": r["subset"],
                       "queries_used": 1, "budget": int(r["budget"]), "end_state": end_state, "posthoc_gate": posthoc,
                       "record_ids": [d.record_id]})
    ledger.close()
    seconds = round(time.perf_counter() - t0, 6)
    summary["timings"].append({"stage": "replay", "seconds": seconds, "items": n})
    summary["stages"].append({"name": "replay", "device": "cpu", "seconds": seconds if ctx.clock.source == "wall" else 0.0, "items": n})
    summary["items"][REPLAY_SPLIT] = n
    summary["depends_on"] = [src["run_id"]]
    summary["n_test"] += n
    return summary


# ------------------------------------------------------------------ store replay 
def _kept(p: float) -> bool:
    """The label (Refuted) is kept when P(Refuted) rounded to 6 d.p. exceeds its complement: the victim server's rule
 (aac.serve.victim_server._kept), so a tie at 0.5 is not kept, exactly as in the R3 run being replayed."""
    p = min(1.0, max(0.0, float(p)))
    return round(p, DECIMALS) > round(1.0 - p, DECIMALS)


def replay_plan(rows: list[dict], adv: dict, store: dict, items: dict) -> list[dict]:
    """[{"row", "clean", "final"}] per source ledger row, after the asserts (ReplayMismatch, never a text in it):
 the item is a positive item of the target; a recorded success (end_state success or deferred) has an
 attack-adversarial claim_hash and every other end state has none; that hash has a stored final whose claim_hash
 equals it. `final` is None for a row with no success."""
    plan = []
    for r in rows:
        key = (r["item_id"], r["attack"], r["victim"], r["subset"])
        if r["item_id"] not in items:
            raise ReplayMismatch(f"{key}: item not in the replay target (the AVeriTeC test partition of this run)")
        if not items[r["item_id"]]["positive"]:
            raise ReplayMismatch(f"{key}: the attacked item is not a positive (Refuted) item of the target")
        a = adv.get(key)
        if a is None:
            raise ReplayMismatch(f"{key}: no attack-adversarial.jsonl row in the sealed R3 run")
        want = a.get("claim_hash")
        succeeded = r["end_state"] in ("success", "deferred")
        if succeeded != (want is not None):
            raise ReplayMismatch(f"{key}: ledger end_state {r['end_state']} disagrees with the adversarial claim_hash "
                                 f"({'set' if want else 'null'})")
        final = None
        if want is not None:
            held = store.get(key)
            if held is None:
                raise ReplayMismatch(f"{key}: a success with no stored final text; the replay runs only on the host that "
                                     "made the R3 run")
            if held["claim_hash"] != want or claim_hash(held["text"]) != want:
                raise ReplayMismatch(f"{key}: the stored final's claim_hash {held['claim_hash'][:12]} differs from the "
                                     f"sealed R3 run's {want[:12]}")
            final = held["text"]
        plan.append({"row": r, "clean": items[r["item_id"]]["text"], "final": final})
    return plan


def replay_outcome(clean, final, *, gated: bool, victim: str) -> tuple[int, str, str | None]:
    """(queries_used, end_state, posthoc_gate) of one replay ledger row from the ablated clean decision and the final
 decision (None when no final was queried). Excluded (0, unchanged, null): the clean decision does not keep the
 label, or a gated system defers it. Otherwise in the denominator (1): success = auto-decided wrong, deferred = a
 gated deferral (deflected), unchanged = no final or the label kept. posthoc_gate is set for victim-B rows of a
 gated system only (the ledger contract: null for victim content_only)."""
    if not _kept(clean.calibrated_probability) or (gated and clean.abstention_flag):
        return 0, "unchanged", None
    tag = gated and victim == "full"
    if final is None:
        return 1, "unchanged", ("no_effect" if tag else None)
    if gated and final.abstention_flag:
        return 1, "deferred", ("deflected" if tag else None)
    if not _kept(final.calibrated_probability):
        return 1, "success", ("flip" if tag else None)
    return 1, "unchanged", ("no_effect" if tag else None)


def replay_asr(rows: list[dict]) -> tuple[int, int]:
    """(successes, denominator) of a replay ledger: rows with queries_used 1 form the denominator."""
    den = [r for r in rows if int(r["queries_used"]) == 1]
    return sum(r["end_state"] == "success" for r in den), len(den)


def _select_r3(root: Path, name: str, seed: int) -> dict:
    """The one indexed R3 run `name` at `seed` (any system: victim-A-only runs may carry another system label); its
 seal is recomputed and must equal the INDEX row. A source absent from runs/INDEX.json is a refusal: a cut that
 removes a source edits replay.sources explicitly."""
    runs = [r for r in load_index(runs_dir(root) / "INDEX.json")["runs"]
            if r["regime"] == "R3" and r.get("name") == name and r["seed"] == seed]
    if len(runs) != 1:
        raise RuntimeError(f"replay source: expected exactly one sealed R3 run {name!r} seed {seed} in runs/INDEX.json, "
                           f"found {len(runs)} (run it first, or edit replay.sources if a cut removed it)")
    src = runs[0]
    digest = verify_sealed(runs_dir(root) / src["run_id"])
    if digest != src.get("sealed_sha256"):
        raise SourceMismatch(f"{src['run_id']}: runs/INDEX.json sealed_sha256 differs from the recomputed seal")
    return src


def _chunks(seq: list, size: int):
    for i in range(0, len(seq), max(int(size), 1)):
        yield seq[i:i + size]


def store_replay(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path) -> dict:
    """The store replay run (module docstring). Requests are batched per (source, attack) group, where item ids are
 unique, clean texts first, then the finals of the eligible rows."""
    replay = resolved["replay"]
    system = resolved["system"]
    if system not in STORE_REPLAY_SYSTEMS:
        raise ValueError(f"a store replay applies to {STORE_REPLAY_SYSTEMS}, not to system {system!r}")
    if not resolved["components"]["c3"].get("fitted_from"):
        raise ValueError("a store replay loads its maps from components.c3.fitted_from (R4/<metric run> or an R1 full "
                         "run); it never fits, so every record is a ledger query")
    summary = evaluate_targets(pipeline, ctx, resolved, root, run_dir, {})
    seed, rseed = int(resolved["seed"]), int(replay["seed"])
    victim, subset = replay["victim"], replay["subset"]
    batch = int(resolved["batch_size"])
    items = target_items(root, resolved, subset)
    toggles = ctx.toggles
    gated = bool(toggles.abstention)
    ctx_t = dataclasses.replace(ctx, toggles=toggles)
    key = resolved["components"]["c2"]["provider"]
    regime = regime_label(resolved)
    ledger = LedgerWriter(run_dir, root)
    cut_rows = getattr(getattr(pipeline, "c2", None), "scorer_truncations", None)   # EvidenceC2 only (retrieval on)
    cuts: list[dict] = []
    sources: list[str] = []
    n_clean = n_final = 0
    t0 = time.perf_counter()

    def request(r: dict, text: str, split: str, phase: str, index: int) -> Request:
        return Request(claim_id=r["item_id"], text=text, evidence_provider_key=key, regime=regime, split=split,
                       seed=seed, query_ctx={"item_id": r["item_id"], "attack": r["attack"], "victim": r["victim"],
                                             "subset": r["subset"], "query_index": index, "phase": phase})

    for name in replay["sources"]:
        src = _select_r3(root, name, rseed)
        src_dir = runs_dir(root) / src["run_id"]
        rows = [r for r in read_ledger(src_dir)
                if r["victim"] == victim and r["subset"] == subset and int(r["queries_used"]) > 0]
        adv = {(a["item_id"], a["attack"], a["victim"], a["subset"]): a for a in read_adversarial(src_dir)}
        # a source with no success has no store file (the server writes one on the first success only)
        needed = any((adv.get((r["item_id"], r["attack"], r["victim"], r["subset"])) or {}).get("claim_hash")
                     for r in rows)
        store = read_private_store(root, src["run_id"]) if needed else {}
        plan = replay_plan(rows, adv, store, items)
        groups: OrderedDict[str, list[dict]] = OrderedDict()
        for p in plan:
            groups.setdefault(p["row"]["attack"], []).append(p)
        for group in groups.values():
            for chunk in _chunks(group, batch):
                n_cut0 = len(cut_rows) if cut_rows is not None else 0
                cleans = pipeline.infer([request(p["row"], p["clean"], REPLAY_CLEAN_SPLIT, "clean", 0) for p in chunk], ctx_t)
                n_clean += len(cleans)
                if cut_rows is not None and len(cut_rows) != n_cut0:
                    raise RuntimeError("replay: a clean text had a scorer pair cut (only phase replay may be cut, row 46)")
                todo = [i for i, (p, dc) in enumerate(zip(chunk, cleans)) if p["final"] is not None
                        and replay_outcome(dc, None, gated=gated, victim=victim)[0] == 1]
                finals = pipeline.infer([request(chunk[i]["row"], chunk[i]["final"], REPLAY_SPLIT, "replay", 1)
                                         for i in todo], ctx_t) if todo else []
                n_final += len(finals)
                by_index = dict(zip(todo, finals))
                if cut_rows is not None and len(cut_rows) > n_cut0:  # row 46: map each cut to its record (ids unique here)
                    by_item = {chunk[i]["row"]["item_id"]: (chunk[i]["row"], d) for i, d in by_index.items()}
                    for c in cut_rows[n_cut0:]:
                        r, d = by_item[c["claim_id"]]
                        cuts.append({"item_id": r["item_id"], "attack": r["attack"], "victim": r["victim"],
                                     "subset": r["subset"], "phase": c.get("phase"), "query_index": int(c["query_index"]),
                                     "record_id": d.record_id, "pairs": int(c["pairs"])})
                for i, (p, dc) in enumerate(zip(chunk, cleans)):
                    df = by_index.get(i)
                    used, end_state, posthoc = replay_outcome(dc, df, gated=gated, victim=victim)
                    r = p["row"]
                    ledger.append({"item_id": r["item_id"], "attack": r["attack"], "victim": r["victim"],
                                   "subset": r["subset"], "queries_used": used, "budget": int(r["budget"]),
                                   "end_state": end_state, "posthoc_gate": posthoc,
                                   "record_ids": [dc.record_id] + ([df.record_id] if df is not None else [])})
        sources.append(src["run_id"])
    ledger.close()
    write_replay_truncations(pipeline, run_dir, summary, cuts, REPLAY_SPLIT)
    seconds = round(time.perf_counter() - t0, 6)
    n = n_clean + n_final
    summary["timings"].append({"stage": "replay", "seconds": seconds, "items": n})
    summary["stages"].append({"name": "replay", "device": getattr(pipeline, "device", "cpu"),
                              "seconds": seconds if ctx.clock.source == "wall" else 0.0, "items": n})
    summary["items"][REPLAY_CLEAN_SPLIT] = n_clean
    summary["items"][REPLAY_SPLIT] = n_final
    summary["depends_on"] = sorted({*summary["depends_on"], *sources})
    summary["n_test"] += n
    summary["l2_strict"] = True
    return summary
