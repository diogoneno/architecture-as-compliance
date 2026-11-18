"""Shared regime mechanics: request building per target, the generic run loop, the fit passes and the
evaluation driver every non-attack regime uses.

Order inside a run whose system calibrates or abstains (3, operational constraint):
 1. relevance floor nearest-rank 10th percentile of the cached top-1 re-ranker score over the val_cal claims,
 set on C2 before any fit (evidence-floor.json)
 2. pass F val_fuse with calibration and abstention off, C3 in fit mode (records split val_fuse), then
 the C3 fit (fusion.json)
 3. pass A val_cal with calibration and abstention off (records split val_cal), then C4, then C5
 (calibration.json, gate.json)
 4. test pass the configured system's toggles
A run whose components.c3.fitted_from names a sealed R1 full run (or, `R4/<name>`: the R4 run of the same
system) skips 1-3 and loads the four files from that run at the same seed (load_fitted; the run lists it in depends_on).

Drivers import Pipeline and the data providers only, never the components package (architectural isolation constraint): every component call
here is an attribute call on the pipeline (pipeline.c2.set_floor, pipeline.c3.start_fit, pipeline.c3.fit, ...)."""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

from aac.data.evidence_providers import load_incrediblae_items
from aac.data.fixtures import load_claims, load_labels, load_split
from aac.pipeline import Pipeline, Request, RunContext, Toggles
from aac.run.hashing import dumps_sorted
from aac.run.index import load_index
from aac.run.paths import runs_dir
from aac.run.seal import verify_sealed

PROVIDER_KEY_FOR_TARGET = {"fc": "supplied", "rd": "pooled"}
POSITIVE_LABEL = "Refuted"  # : binary decision, Refuted positive; InCrediblAE label 1 maps onto it
# The test partition of data.split_manifest: `averitec_test` (new configs) and its legacy label

TEST_TARGETS = ("averitec_test", "averitec_chrono")
FLOOR_RULE = "p10-top1-rerank-val_cal"
FLOOR_FILE = "evidence-floor.json"
FUSION_FILE = "fusion.json"


def select_source_run(root: Path, *, regime: str, name: str, seed: int, system: str) -> dict:
    """The one indexed run of (regime, system, name, seed). Lives here (moved from regimes/r3.py in ) because
 the R1/R2 fitted_from path needs it and common.py importing r3.py would be circular; r3.py re-exports it."""
    runs = [r for r in load_index(runs_dir(root) / "INDEX.json")["runs"]
            if r["regime"] == regime and r.get("name") == name and r["seed"] == seed and r["system"] == system]
    if len(runs) != 1:
        raise RuntimeError(f"expected exactly one sealed {regime}/{system}/{name} seed {seed} run in runs/INDEX.json "
                           f"(run it first), found {len(runs)}")
    return runs[0]


def check_toggles(resolved: dict) -> Toggles:
    """components.*.enabled must agree with the system's toggles (ablations are toggles on one path)."""
    t = Toggles.from_system(resolved["system"])
    comps = resolved["components"]
    expected = {"c2": t.retrieval, "c4": t.calibration, "c5": t.abstention}
    for key, want in expected.items():
        if bool(comps[key]["enabled"]) != want:
            raise ValueError(f"components.{key}.enabled={comps[key]['enabled']} contradicts system {resolved['system']!r}")
    return t


def regime_label(resolved: dict) -> str:
    return f"{resolved['regime']}/{resolved['name']}"


def _claims(root: Path, resolved: dict) -> dict[str, dict]:
    """{claim_id: {"claim_id", "text", "date"}} by data.dataset: the fixture JSON, or the raw AVeriTeC
 train/dev JSON read at run time (aac.data.averitec.load_claims; text is never written out)."""
    data = resolved["data"]
    if data["dataset"] == "averitec":
        from aac.data.averitec import load_claims as load_averitec_claims

        return load_averitec_claims(root / data["claims"])
    return load_claims(root / data["claims"])


def build_requests(root: Path, resolved: dict, split: str, seed: int) -> list[Request]:
    """AVeriTeC partition `split` of data.split_manifest as Requests (evidence: c2.provider)."""
    claims = _claims(root, resolved)
    manifest = load_split(root / resolved["data"]["split_manifest"])
    regime = regime_label(resolved)
    return [
        Request(claim_id=cid, text=claims[cid]["text"], evidence_provider_key=resolved["components"]["c2"]["provider"],
                regime=regime, split=split, seed=seed)
        for cid in sorted(manifest[split])
    ]


def target_items(root: Path, resolved: dict, target: str) -> dict[str, dict]:
    """{item_id: {"text", "positive": bool}} for an evaluation target."""
    data = resolved["data"]
    if target in TEST_TARGETS:
        claims = _claims(root, resolved)
        labels = load_labels(root / data["labels"])
        ids = sorted(load_split(root / data["split_manifest"])["test"])
        return {cid: {"text": claims[cid]["text"], "positive": labels[cid] == POSITIVE_LABEL} for cid in ids}
    if target in ("fc", "rd"):
        # : the committed text-free item file reads its texts at run time (FC from the hash-checked raw
        # attack.tsv, RD from lane B on cpu-host) and checks every text against its committed hash; a fixture item file
        # carries its text. Labels come from the item file; label 1 (non-credible) is the positive class.
        path = data.get(f"{target}_items")
        if not path:
            raise ValueError(f"target {target!r} needs data.{target}_items")
        items = load_incrediblae_items(root, path)
        return {iid: {"text": it["text"], "positive": int(it["label"]) == 1} for iid, it in sorted(items.items())}
    raise ValueError(f"unknown target {target!r}")


def target_requests(root: Path, resolved: dict, target: str, split: str, seed: int,
                    ids: list[str] | None = None) -> list[Request]:
    items = target_items(root, resolved, target)
    key = PROVIDER_KEY_FOR_TARGET.get(target, resolved["components"]["c2"]["provider"])
    chosen = sorted(items) if ids is None else sorted(ids)
    missing = [i for i in chosen if i not in items]
    if missing:
        raise ValueError(f"ids absent from target {target}: {missing[:3]}")
    return [Request(claim_id=i, text=items[i]["text"], evidence_provider_key=key, regime=regime_label(resolved),
                    split=split, seed=seed) for i in chosen]


def run_regime(pipeline: Pipeline, ctx: RunContext, reqs: list[Request], toggles: Toggles, batch_size: int,
               stage: str, stages: list[dict], timings: list[dict]):
    """Run reqs through the pipeline in batches; append a stage entry (wall-clock seconds go to
 the unhashed timings list; the sealed stage entry carries seconds only under a wall clock)."""
    ctx_stage = dataclasses.replace(ctx, toggles=toggles)
    t0 = time.perf_counter()
    decisions = []
    for i in range(0, len(reqs), batch_size):
        decisions.extend(pipeline.infer(reqs[i:i + batch_size], ctx_stage))
    seconds = round(time.perf_counter() - t0, 6)
    timings.append({"stage": stage, "seconds": seconds, "items": len(reqs)})
    sealed_seconds = seconds if ctx.clock.source == "wall" else 0.0
    stages.append({"name": stage, "device": getattr(pipeline, "device", "cpu"), "seconds": sealed_seconds, "items": len(reqs)})
    return decisions


def nearest_rank_p10(values: list[float]) -> tuple[float, int]:
    """(value, index) of the nearest-rank 10th percentile: index ceil(0.10 x n) - 1 of the ascending values, computed
 in integers as (n + 9) // 10 - 1 (0.10 x n in floats can land above an integer, e.g. 0.1 x 30 > 3); 6 d.p."""
    v = sorted(float(x) for x in values)
    if not v:
        raise ValueError("relevance floor: no val_cal claims")
    idx = (len(v) + 9) // 10 - 1
    return round(v[idx], 6), idx


def floor_pass(pipeline: Pipeline, resolved: dict, root: Path, run_dir: Path) -> dict:
    """Step 1: the relevance floor (rule p10-top1-rerank-val_cal; methodological note
 score of each val_cal claim (provider.top1_score; 0.0 for a claim with no passage), set on C2 BEFORE any fit
 (operational constraint: evidence_coverage is a C3 input). No record is written; evidence-floor.json is sealed in the run."""
    c2 = resolved["components"]["c2"]
    if c2.get("floor_rule") != FLOOR_RULE:
        raise ValueError(f"components.c2.floor_rule must be {FLOOR_RULE!r}")
    reqs = build_requests(root, resolved, "val_cal", int(resolved["seed"]))
    values = []
    for r in reqs:
        top1 = getattr(pipeline.provider_for(r), "top1_score", None)
        if top1 is None:
            raise TypeError(f"provider {r.evidence_provider_key!r} has no top1_score: the floor needs the EV cache")
        values.append(float(top1(r.claim_id)))
    value, idx = nearest_rank_p10(values)
    pipeline.c2.set_floor(value)
    report = {"rule": FLOOR_RULE, "value": value, "n_val_cal": len(values), "rank_index": idx,
              "evidence_run_id": (c2.get("evidence") or {}).get("run_id")}
    (run_dir / FLOOR_FILE).write_text(dumps_sorted(report), encoding="ascii")
    return report


def fuse_pass(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path,
              system_toggles: Toggles, stages: list[dict], timings: list[dict]) -> int:
    """Step 2, pass F: val_fuse through the pipeline with calibration and abstention OFF and C3 in fit mode (raw_score
 = C1 P(Refuted) passthrough; records logged with split=val_fuse, so the gate's training rows are in the log, Art.
 12); then C3 fits on the rows it collected and fusion.json is sealed. y = 1 for Refuted ."""
    seed = int(resolved["seed"])
    labels = load_labels(root / resolved["data"]["labels"])
    reqs = build_requests(root, resolved, "val_fuse", seed)
    toggles = Toggles(retrieval=system_toggles.retrieval, calibration=False, abstention=False)
    pipeline.c3.start_fit()
    decisions = run_regime(pipeline, ctx, reqs, toggles, int(resolved["batch_size"]), "fit_val_fuse", stages, timings)
    report = pipeline.c3.fit({r.claim_id: int(labels[r.claim_id] == POSITIVE_LABEL) for r in reqs})
    (run_dir / FUSION_FILE).write_text(dumps_sorted(report), encoding="ascii")
    return len(decisions)


def fit_pass(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path,
             system_toggles: Toggles, stages: list[dict], timings: list[dict]) -> int:
    """Pass A: val_cal through the pipeline with calibration and abstention OFF (records logged with
 split=val_cal) to obtain raw scores; C4 then C5 are fitted on them (ARCH A6; C5 also gets the raw scores for its
 raw-confidence tie-break) and the fitted reports are sealed inside the run as calibration.json / gate.json."""
    seed = int(resolved["seed"])
    batch = int(resolved["batch_size"])
    labels = load_labels(root / resolved["data"]["labels"])
    fit_toggles = Toggles(retrieval=system_toggles.retrieval, calibration=False, abstention=False)
    fit_reqs = build_requests(root, resolved, "val_cal", seed)
    fit_decisions = run_regime(pipeline, ctx, fit_reqs, fit_toggles, batch, "fit_val_cal", stages, timings)
    raw = [d.calibrated_probability for d in fit_decisions]  # == raw_score under calibration=False
    y_false = [int(labels[r.claim_id] == POSITIVE_LABEL) for r in fit_reqs]
    cal_report = pipeline.c4.fit(raw, y_false)
    cal_probs = [pipeline.c4.apply(s)["calibrated_probability"] for s in raw] if system_toggles.calibration else raw
    gate_report = pipeline.c5.fit(cal_probs, raw=raw)
    (run_dir / "calibration.json").write_text(dumps_sorted(cal_report), encoding="ascii")
    (run_dir / "gate.json").write_text(dumps_sorted(gate_report), encoding="ascii")
    return len(fit_decisions)


class SourceMismatch(RuntimeError):
    """ the sealed source run of fitted_from / victim_source does not fit this run (another C1/C2,
 an INDEX digest that is not the directory's seal, or fitted copies from another source in a resumed run)."""


# Keys an R2 run changes on purpose when it reads another evidence provider than its R1 source : the EV run
# and its embedding settings follow the provider. Every other C1/C2 key must equal the source's.
SOURCE_PROVIDER_KEYS = ("provider", "evidence_from", "evidence", "dense", "reranker")


def check_source_identity(resolved: dict, source_dir: Path, source_run_id: str) -> None:
    """the source run's components.c1 and (when this run retrieves) components.c2 must equal this run's, key by
 key (`enabled` excluded; SOURCE_PROVIDER_KEYS excluded only when the providers differ). Checked at load time,
 outside config_hash, so no sealed run id moves."""
    src = json.loads((Path(source_dir) / "config.resolved.json").read_text(encoding="utf-8"))["components"]
    mine = resolved["components"]
    diffs = [f"c1.{k}" for k in sorted(set(mine["c1"]) | set(src["c1"]))
             if k != "enabled" and mine["c1"].get(k) != src["c1"].get(k)]
    if mine["c2"].get("enabled"):
        same_provider = mine["c2"].get("provider") == src["c2"].get("provider")
        diffs += [f"c2.{k}" for k in sorted(set(mine["c2"]) | set(src["c2"]))
                  if k != "enabled" and (same_provider or k not in SOURCE_PROVIDER_KEYS)
                  and mine["c2"].get(k) != src["c2"].get(k)]
    if diffs:
        raise SourceMismatch(f"{source_run_id} was fitted on another C1/C2 than this run ({', '.join(diffs)} differ); "
                             "its C3/C4/C5 maps and relevance floor do not apply here")


R4_SOURCE_PREFIX = "R4/"


def source_ref(resolved: dict, name: str) -> tuple[str, str, str]:
    """(regime, run name, system) of a maps source name (components.c3.fitted_from, attack.victim_source). :
 `R4/<name>` names the R4 run `name` of THIS run's system (an R4 RD ablation reads its seed-matched R4 FC run; a
 replay reads its seed-0 R4 metric run); a bare name keeps its meaning, the R1 full run `name`."""
    if name.startswith(R4_SOURCE_PREFIX):
        short = name[len(R4_SOURCE_PREFIX):]
        if not short or "/" in short:
            raise ValueError(f"maps source {name!r}: expected R4/<run name>")
        return "R4", short, resolved["system"]
    return "R1", name, "full"


def load_source_maps(pipeline: Pipeline, resolved: dict, root: Path, run_dir: Path, name: str) -> str:
    """The seed-matched sealed source run `name` (components.c3.fitted_from, attack.victim_source; source_ref: the R1
 full run, or with `R4/` the R4 run of this run's system): its seal is recomputed and must equal its INDEX row, its
 C1/C2 must equal this run's (check_source_identity), then its fitted files are loaded and copied (load_fitted).
 Returns its run id."""
    regime, short, system = source_ref(resolved, name)
    src = select_source_run(root, regime=regime, name=short, seed=int(resolved["seed"]), system=system)
    src_dir = runs_dir(root) / src["run_id"]
    digest = verify_sealed(src_dir)
    if digest != src.get("sealed_sha256"):
        raise SourceMismatch(f"{src['run_id']}: runs/INDEX.json sealed_sha256 differs from the recomputed seal")
    check_source_identity(resolved, src_dir, src["run_id"])
    load_fitted(pipeline, src_dir, run_dir, src["run_id"])
    return src["run_id"]


def load_fitted(pipeline: Pipeline, source_dir: Path, run_dir: Path, source_run_id: str) -> None:
    """Restore C4/C5 from a sealed run's calibration.json / gate.json and copy them into this run. : also
 fusion.json into C3 and evidence-floor.json into C2 when the source run has them (a fitted C3 or a real C2 without
 its file is refused), each copied with source_run_id.

 a run directory that already holds a copy (a resumed `aac run` or victim-server session of the same run_id)
 must hold exactly the bytes about to be written; otherwise the source changed between sessions (re-sealed,
 superseded or re-pointed) and nothing is loaded or rewritten."""
    fusion_src, floor_src = source_dir / FUSION_FILE, source_dir / FLOOR_FILE
    if not fusion_src.is_file() and hasattr(pipeline.c3, "start_fit"):
        raise ValueError(f"{source_run_id} has no {FUSION_FILE}: the fitted C3 of this run cannot be restored")
    if not floor_src.is_file() and hasattr(pipeline.c2, "set_floor"):
        raise ValueError(f"{source_run_id} has no {FLOOR_FILE}: the relevance floor of this run cannot be restored")
    loaded = {name: json.loads((source_dir / name).read_text(encoding="utf-8"))
              for name in ("calibration.json", "gate.json", FUSION_FILE, FLOOR_FILE) if (source_dir / name).is_file()}
    copies = {name: dumps_sorted({**obj, "source_run_id": source_run_id}) for name, obj in loaded.items()}
    for name, text in copies.items():
        held = run_dir / name
        if held.is_file() and held.read_text(encoding="ascii") != text:
            raise SourceMismatch(f"{run_dir.name}: {name} in the run directory came from another source than "
                                 f"{source_run_id}; refusing to resume (re-run with --force)")
    pipeline.c4.load(loaded["calibration.json"])
    pipeline.c5.load(loaded["gate.json"])
    if FUSION_FILE in loaded:
        pipeline.c3.load(loaded[FUSION_FILE])
    if FLOOR_FILE in loaded and hasattr(pipeline.c2, "set_floor"):
        pipeline.c2.set_floor(loaded[FLOOR_FILE]["value"])
    for name, text in copies.items():
        (run_dir / name).write_text(text, encoding="ascii")


def _upstream_runs(resolved: dict) -> list[str]:
    """Run ids of the sealed EV and C2S runs a real C2 reads (resolved by aac.run.identity), for depends_on."""
    c2 = resolved["components"]["c2"]
    return sorted({ref["run_id"] for ref in (c2.get("evidence"), c2.get("scorer")) if isinstance(ref, dict)})


def evaluate_targets(pipeline: Pipeline, ctx: RunContext, resolved: dict, root: Path, run_dir: Path,
                     split_labels: dict[str, str]) -> dict:
    """Fit steps (module docstring) then one test pass per target, records tagged with the target's split label.
 Returns the run-record summary skeleton."""
    seed = int(resolved["seed"])
    batch = int(resolved["batch_size"])
    stages: list[dict] = []
    timings: list[dict] = []
    system_toggles = check_toggles(resolved)
    manifest = load_split(root / resolved["data"]["split_manifest"])
    items = {k: len(v) for k, v in manifest.items() if isinstance(v, list)}
    depends_on = _upstream_runs(resolved)
    n_fit = 0
    fitted_from = resolved["components"]["c3"].get("fitted_from")
    if fitted_from:
        src_id = load_source_maps(pipeline, resolved, root, run_dir, fitted_from)
        depends_on = sorted({*depends_on, src_id})
    else:
        if system_toggles.retrieval and hasattr(pipeline.c2, "set_floor"):
            floor_pass(pipeline, resolved, root, run_dir)
        
        # ablation-all-removed run with a logistic gate would otherwise fuse before start_fit); pass A only when the
        # system calibrates or abstains. Order unchanged for every existing config: floor, F, A, test.
        if hasattr(pipeline.c3, "start_fit"):
            n_fit += fuse_pass(pipeline, ctx, resolved, root, run_dir, system_toggles, stages, timings)
        if system_toggles.calibration or system_toggles.abstention:
            n_fit += fit_pass(pipeline, ctx, resolved, root, run_dir, system_toggles, stages, timings)
    n_test = 0
    for target, split in split_labels.items():
        reqs = target_requests(root, resolved, target, split, seed)
        decisions = run_regime(pipeline, ctx, reqs, system_toggles, batch, f"infer_{split}", stages, timings)
        items[split] = len(reqs)
        n_test += len(decisions)
    return {
        "configurations": [f"{resolved['system']}/{resolved['name']}"],
        "items": items, "stages": stages, "timings": timings,
        "n_fit": n_fit, "n_test": n_test, "depends_on": depends_on, "l2_strict": False,
    }


def targets_of(resolved: dict, default: list[str]) -> list[str]:
    t = list(resolved["data"].get("targets") or [])
    return t or list(default)


# ------------------------------------------------------------------ scorer truncation record (methodological note)
TRUNCATION_FILE = "scorer-truncation.json"
TRUNCATION_RULE = "item-15a: RD text cut for the scorer pair only (longest_first at the scorer max_length)"
TRUNCATION_RULE_REPLAY = ("item-15a extended to R4 store replays (protocol notes): a replayed adversarial final text "
                          "(phase replay) cut for the scorer pair only (longest_first at the scorer max_length)")


def _truncation_counter(pipeline: Pipeline):
    return getattr(getattr(pipeline, "c2", None), "scorer_truncations", None)


def write_scorer_truncations(pipeline: Pipeline, run_dir: Path, summary: dict, splits: list[str]) -> None:
    """(a) (moved here from regimes/r2.py by so R2 and R4 share
 it, R2's bytes unchanged): the sealed sidecar scorer-truncation.json (one row per record with a cut pair, keyed by
 claim_id and split, as the record is) and the run-record items scorer_truncated_<split> (records) and
 scorer_truncated_pairs_<split> (pairs). A run whose C2 has no such counter (stub, content-only, retrieval off)
 writes neither."""
    rows = _truncation_counter(pipeline)
    if rows is None:
        return
    rows = sorted(({"claim_id": r["claim_id"], "split": r["split"], "pairs": int(r["pairs"])} for r in rows),
                  key=lambda r: (str(r["split"]), r["claim_id"]))
    keys = [(r["split"], r["claim_id"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("scorer truncation: a (split, claim_id) appears twice")
    scorer = getattr(pipeline.c2, "scorer", None)
    for split in splits:
        mine = [r for r in rows if r["split"] == split]
        summary["items"][f"scorer_truncated_{split}"] = len(mine)
        summary["items"][f"scorer_truncated_pairs_{split}"] = sum(r["pairs"] for r in mine)
    report = {"rule": TRUNCATION_RULE, "max_length": getattr(scorer, "max_length", None),
              "by_split": {s: summary["items"][f"scorer_truncated_{s}"] for s in splits}, "records": rows}
    (run_dir / TRUNCATION_FILE).write_text(dumps_sorted(report), encoding="ascii")


def write_replay_truncations(pipeline: Pipeline, run_dir: Path, summary: dict, rows: list[dict], split: str) -> None:
    """methodological note
 victim server lists attack-phase cuts (record_id, item, attack, victim, subset, phase, query_index, pairs), and
 the run-record items scorer_truncated_<split> / scorer_truncated_pairs_<split>, zeros included. A run whose C2 has
 no counter (retrieval off) writes neither."""
    if _truncation_counter(pipeline) is None:
        return
    rows = sorted(rows, key=lambda r: (r["subset"], r["attack"], r["victim"], r["item_id"], r["query_index"], r["phase"]))
    if len({r["record_id"] for r in rows}) != len(rows):
        raise RuntimeError("scorer truncation: a record_id appears twice")
    summary["items"][f"scorer_truncated_{split}"] = len(rows)
    summary["items"][f"scorer_truncated_pairs_{split}"] = sum(int(r["pairs"]) for r in rows)
    scorer = getattr(pipeline.c2, "scorer", None)
    report = {"rule": TRUNCATION_RULE_REPLAY, "max_length": getattr(scorer, "max_length", None),
              "by_split": {split: len(rows)}, "records": rows}
    (run_dir / TRUNCATION_FILE).write_text(dumps_sorted(report), encoding="ascii")
