"""Table 4.5 (R4 ablations) from the run profile of configs/reports/phase4-runs.yaml .

Rows and sources (aac.reports.phase4_sources.load_profile(root)["r4"]; Table 4.5 shell note):
 Full pipeline R1 r1-chrono-full (split test), R2 r2-fc-full / r2-rd-full, R3 victim B's adaptive attack: the
 union of the profile's sealed R3 ledgers, victim full, subset averitec_chrono
 each ablation R1 r4-<key> (split averitec_chrono, refitted on the chronological val partitions), R2 r4-<key>-fc
 and r4-<key>-rd (the random-split system, methodological note
 of victim B's final adversarial examples (rows 53, 54); Minus retrieval also carries victim A's
 adaptive ASR as a labeled long row (the pre-registered specification)
 Content-only R1/R2 as an ablation (r4-content-only*), R3 victim A's adaptive attack (union, victim
 content_only, subset averitec_chrono)
The tiny world reads its own names (r1-tw-full, ...; profile tinyworld). The stub namespace keeps its single-run reading
(r3-stub for the victim cells, the stub R4 runs' own replay ledgers over their seeds, Minus retrieval = victim A) and
the same Delta rule and long rows as the other profiles.

R3 cell = mean over attacks of the strict ASR (attacked rows: queries_used > 0; aac.metrics.attack). Replay runs are
read at seed 0 only (pre-registered specification
BeamAttack rows (r3_asr_mean is then the mean over the four remaining attacks); cut-2-subset-200 (fired) keeps victim-B
rows of the first-200 runs; cut-4-drop-rd (fired) blanks the RD half of the composite R2 cell when its run is absent.

Delta vs full (pre-registered specification:
 primary the exact McNemar on the per-item outcome "decided and correct" (a deferred item is "otherwise"), taken over
 the seeds by majority, the ablation against the full pipeline on the same chronological test items; Holm
 within the R4 family; beside it the paired percentile item bootstrap (one index resample for both systems
 and every seed) of the difference in seed-mean binary macro-F1 (ablated minus full). The cell reads like
 the Table 4.2 Change cell: `p=<Holm p> (Holm); diff=<signed> [lo, hi]`.
 secondary the paired t over per-seed macro-F1 differences with Cohen's d_z: a long row labeled secondary.
The pairing key is (claim_id, seed): the two systems must hold the same seeds and the same claim ids, else the build
stops (no silent drop); a seed subset of the full pipeline's seeds pairs only after cut-3-r4-seeds-3 fired.

Long-CSV additions (methodological note
gate.json report threshold "0.80" through the C5 rule of aac.reports.table_4_6.auto_mask; n/a with the reason when a
run has no gate.json), 95 percent intervals on the ECE after C4 (item bootstrap, seeds as replicates) and on the R3
mean ASR (item-cluster bootstrap), and, per seed, the largest absolute calibrated-probability difference between Minus
abstention and Full (row 65: equal by construction; a fourth-decimal difference is numeric noise between sessions).

allow_pending (CLI `aac reports build --table 4.5 --allow-pending`, never in a committed report): a source that is not
indexed renders `pending ` in its own cell only; while any R4 family test is missing every Delta cell is pending
and its long row keeps the raw McNemar p. Without allow_pending a missing source raises MissingRun."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aac.attacks.ledger import read_ledger
from aac.metrics.calibration import ece15
from aac.metrics.classification import macro_f1_binary
from aac.metrics.selective import binary_macro_f1 as _f1_fast
from aac.reports import fmt, phase4_sources, sources
from aac.reports.table_4_6 import auto_mask as gate_auto_mask
from aac.run.hashing import derive_seed
from aac.stats.bootstrap import ci_index, cluster_ci
from aac.stats.families import family as holm_family
from aac.stats.holm import holm_family as holm_family_adjusted
from aac.stats.tests import mcnemar_exact, paired_t_dz
from aac.run.paths import reports_dir

TABLE = "4.5"
REGIME = "R4"
ATTACK_ORDER = ("bert_attack", "textfooler", "deepwordbug", "pwws", "beamattack")
FULL_ROW = "Full pipeline"
R1_COL = "R1 chronological macro-F1"
R2_COL = "R2 transfer macro-F1 (FC / RD)"
R3_COL = "R3 attack-success rate (mean over attacks)"
R3_VA_COL = R3_COL + ", victim A adaptive (protocol notes)"
ECE_COL = "ECE after C4"
DELTA_COL = "Delta vs full pipeline (p; effect size)"
DELTA_SEC_COL = "Delta vs full pipeline, secondary: paired t over seeds"
RISK_COL = "Selective risk at 80 percent coverage (percent)"
IDENTITY_COL = "Largest calibrated-probability difference vs Full"
CUT_SEEDS = "cut-3-r4-seeds-3"
CUT_RD = "cut-4-drop-rd"
PENDING_PHASE = "Phase 4"
DECISION_RULE = "decision P(Refuted) >= 0.5 on calibrated_probability"
NO_ATTACKED = "n/a (no attacked item)"
ABSTENTION_ROW = "Minus abstention (coverage 100 percent)"
RISK_COVERAGE = "0.80"
ABLATIONS = (
    # (row label, profile r4 key, calibrated?, R4 family test id)
    ("Minus retrieval", "no-retrieval", True, "H4-no-retrieval"),
    ("Minus calibration", "no-calibration", False, "H4-no-calibration"),
    (ABSTENTION_ROW, "no-abstention", True, "H4-no-abstention"),
    ("Content-only encoder (all three removed)", "content-only", False, "H4-content-only"),
)
FULL_NAMES = {  # (R1 chrono, R2 fc, R2 rd) of the Full pipeline row per profile (the shell note's sources)
    "real": ("r1-chrono-full", "r2-fc-full", "r2-rd-full"),
    "tinyworld": ("r1-tw-full", "r2-tw-fc-full", "r2-tw-rd-full"),
    phase4_sources.STUB_PROFILE: ("r1-chrono-full", "r2-fc-full", "r2-rd-full"),
}
IDENTITY_NOTE = ("Minus abstention shares C1 to C4 and the refit partitions with Full, so its R1, R2 and ECE values equal "
                 "Full's by construction (protocol notes); a difference in the fourth decimal between the two sessions "
                 "is numeric noise, not an ablation effect")


@dataclass(frozen=True)
class Src:
    regime: str
    name: str
    split: str


@dataclass(frozen=True)
class RowSpec:
    label: str
    r1: Src | None
    fc: Src | None
    rd: Src | None
    r3_mode: str            # victim_b | victim_a | replay (profile, seed 0) | replay_stub (the stub R4 runs' ledgers)
    replay: str | None
    calibrated: bool
    test_id: str | None
    extra_victim_a: bool = False


def row_specs(profile: dict) -> list[RowSpec]:
    """The five rows' sources for a loaded profile (phase4_sources.load_profile). A key the profile does not name gives
 a None source (pending under allow_pending, else MissingRun)."""
    name = profile["name"]
    if name not in FULL_NAMES:
        raise ValueError(f"Table 4.5: no Full-pipeline sources for profile {name!r}")
    r1, fc, rd = FULL_NAMES[name]
    specs = [RowSpec(FULL_ROW, Src("R1", r1, "test"), Src("R2", fc, "fc"), Src("R2", rd, "rd"), "victim_b", None, True, None)]
    for label, key, calibrated, test_id in ABLATIONS:
        if name == phase4_sources.STUB_PROFILE:
            run = f"r4-{key}"
            mode = "victim_a" if key in ("no-retrieval", "content-only") else "replay_stub"
            specs.append(RowSpec(label, Src("R4", run, "averitec_chrono"), Src("R4", run, "fc"), Src("R4", run, "rd"), mode,
                                 run if mode == "replay_stub" else None, calibrated, test_id))
            continue
        m = (profile.get("r4") or {}).get(key) or {}

        def src(k: str, split: str, m=m) -> Src | None:
            return Src("R4", m[k], split) if m.get(k) else None

        if key == "content-only":
            mode, replay, extra = "victim_a", None, False
        else:
            mode, replay, extra = "replay", m.get("replay"), key == "no-retrieval"
        specs.append(RowSpec(label, src("chrono", "averitec_chrono"), src("fc", "fc"), src("rd", "rd"), mode, replay,
                             calibrated, test_id, extra))
    return specs


# --------------------------------------------------------------------------- per-item metrics

def decided_mask(resolved: dict, records: list[dict]) -> np.ndarray:
    """True where the system decided the item itself: not abstention_flag. A system without C5 decides every item; a
 deferral logged by such a system is refused (it cannot exist)."""
    flags = np.asarray([bool(r["abstention_flag"]) for r in records], dtype=bool)
    c5 = ((resolved.get("components") or {}).get("c5") or {}).get("enabled", True)
    if c5 is False and flags.any():
        raise RuntimeError(f"Table 4.5: a run with no C5 logged {int(flags.sum())} deferred records")
    return ~flags


def _f1(root: Path, entries: list[dict], split: str, cell_key: str, *, decided: bool = False) -> dict:
    runs = sources.load_runs(root, entries)
    ids, per_seed = sources.aligned(runs, split)
    y = sources.positives_for(root, runs[0], per_seed[0])
    probs = [sources.arr(recs, "calibrated_probability") for recs in per_seed]
    f1_seed = [macro_f1_binary(y, p >= 0.5) for p in probs]
    stats = [(lambda idx, p=p: _f1_fast(y[idx], (p >= 0.5)[idx])) for p in probs]
    n_boot, lvl = sources.resamples(root, runs[0]), sources.level(runs[0])
    ci = sources.seeds_as_replicates_ci(stats, len(ids), cell_key=cell_key, n_resamples=n_boot, level=lvl)
    ece = float(np.mean([ece15(p, y) for p in probs]))
    out = {"runs": runs, "seeds": [r.seed for r in runs], "ids": ids, "y": y, "probs": probs, "per_seed": per_seed,
           "n_items": len(ids), "f1_seed": f1_seed, "f1": float(np.mean(f1_seed)), "ci": ci, "ece": ece,
           "n_boot": n_boot, "level": lvl}
    if decided:
        out["decided"] = [decided_mask(run.resolved, recs) for run, recs in zip(runs, per_seed)]
    return out


def _ece_ci(d: dict, cell_key: str) -> tuple[float, float]:
    """Item bootstrap (seeds as replicates) of the seed-mean netcal ECE(15)."""
    y = d["y"]
    stats = [(lambda idx, p=p: ece15(p[idx], y[idx])) for p in d["probs"]]
    return sources.seeds_as_replicates_ci(stats, d["n_items"], cell_key=cell_key, n_resamples=d["n_boot"], level=d["level"])


def selective_risk_80(d: dict) -> dict:
    """Selective risk (error rate among the auto-decided items) at the 80 percent report threshold of each R1 run's
 gate.json, through the C5 rule (table_4_6.auto_mask: [c, r] pair or the stub float); mean over seeds. value None
 with a reason when a run has no gate.json, no "0.80" threshold, or no auto-decided item."""
    risks, covs, thrs = [], [], []
    for run, recs, p in zip(d["runs"], d["per_seed"], d["probs"]):
        path = run.run_dir / "gate.json"
        if not path.is_file():
            return {"value": None, "reason": "n/a (no gate.json: no C5 fitted in this configuration)"}
        gate = phase4_sources.read_json(run, "gate.json")
        thresholds = gate.get("report_thresholds") or {}
        if RISK_COVERAGE not in thresholds:
            return {"value": None, "reason": f"n/a (gate.json of seed {run.seed} has no {RISK_COVERAGE} report threshold)"}
        thr = thresholds[RISK_COVERAGE]
        mask = gate_auto_mask(thr, p, sources.arr(recs, "raw_score"))
        if not mask.any():
            return {"value": None, "reason": f"n/a (no auto-decided item at seed {run.seed})"}
        y = d["y"]
        risks.append(float(np.mean(y[mask] != (p[mask] >= 0.5))))
        covs.append(float(mask.mean()))
        thrs.append(thr)
    return {"value": float(np.mean(risks)), "coverage": float(np.mean(covs)), "per_seed": list(zip(d["seeds"], risks, covs)),
            "thresholds": thrs}


def _seed_mean_f1(y: np.ndarray, preds: list[np.ndarray], idx: np.ndarray) -> float:
    return float(np.mean([_f1_fast(y[idx], p[idx]) for p in preds]))


def delta_tests(full: dict, abl: dict, *, ident: str, cut3: str | None = None) -> dict:
    """The Delta statistics of one ablation against the full pipeline (pairing key: claim_id + seed).

 `full` / `abl` carry seeds, ids (sorted claim ids), y, probs and decided per seed, f1_seed, n_boot and level (as
 _f1(..., decided=True) returns them). Primary: exact McNemar on the seed-majority outcome "decided and correct"
 (rows: full right / not; columns: ablation right / not) and the paired item bootstrap of the seed-mean macro-F1
 difference (ablated minus full). Secondary: paired t with d_z over the per-seed macro-F1 (None below two seeds).
 Also the largest absolute calibrated-probability difference per paired seed. Different seeds, or different claim
 ids, raise RuntimeError; a seed subset pairs only when cut3 (the fired_on of cut-3-r4-seeds-3) is given."""
    fs, as_ = list(full["seeds"]), list(abl["seeds"])
    if as_ == fs:
        pick = list(range(len(fs)))
    elif len(set(as_)) == len(as_) and set(as_) < set(fs):
        if not cut3:
            raise RuntimeError(f"Table 4.5 {ident}: the ablation has seeds {as_} and the full pipeline {fs}; a seed subset "
                               f"pairs only after {CUT_SEEDS} fired (pairing key: claim_id + seed)")
        pick = [fs.index(s) for s in as_]
    else:
        raise RuntimeError(f"Table 4.5 {ident}: ablation seeds {as_} do not pair with the full pipeline's seeds {fs} "
                           f"(pairing key: claim_id + seed)")
    if list(abl["ids"]) != list(full["ids"]):
        raise RuntimeError(f"Table 4.5 {ident}: the ablation's chronological items differ from the full pipeline's "
                           f"({len(abl['ids'])} vs {len(full['ids'])} claim_ids; pairing key: claim_id + seed)")
    y = np.asarray(full["y"], dtype=bool)
    if not np.array_equal(y, np.asarray(abl["y"], dtype=bool)):
        raise RuntimeError(f"Table 4.5 {ident}: the paired items carry different labels in the two systems")
    pf = [np.asarray(full["probs"][i]) >= 0.5 for i in pick]
    pa = [np.asarray(p) >= 0.5 for p in abl["probs"]]
    ok_f = [np.asarray(full["decided"][i], dtype=bool) & (p == y) for i, p in zip(pick, pf)]
    ok_a = [np.asarray(d, dtype=bool) & (p == y) for d, p in zip(abl["decided"], pa)]
    maj_f = np.mean(ok_f, axis=0) > 0.5
    maj_a = np.mean(ok_a, axis=0) > 0.5
    table = [[int(np.sum(maj_f & maj_a)), int(np.sum(maj_f & ~maj_a))],
             [int(np.sum(~maj_f & maj_a)), int(np.sum(~maj_f & ~maj_a))]]
    mcn = mcnemar_exact(table)
    f1_f = [float(full["f1_seed"][i]) for i in pick]
    f1_a = [float(v) for v in abl["f1_seed"]]
    delta = float(np.mean(f1_a) - np.mean(f1_f))
    n = len(y)
    lo, hi = ci_index(lambda idx: _seed_mean_f1(y, pa, idx) - _seed_mean_f1(y, pf, idx), n,
                      n_resamples=int(full["n_boot"]), seed=derive_seed(f"boot.table-4.5.{ident}.delta_vs_full"),
                      level=float(full["level"]))
    secondary = paired_t_dz(f1_a, f1_f) if len(pick) >= 2 else None
    prob_diff = [float(np.max(np.abs(np.asarray(abl["probs"][j], dtype=float) - np.asarray(full["probs"][i], dtype=float))))
                 for j, i in enumerate(pick)]
    return {"seeds": [fs[i] for i in pick], "n_items": n, "n_pos": int(np.sum(y)), "table": table, "mcnemar": mcn,
            "delta": delta, "boot_ci": (lo, hi), "secondary": secondary, "subset": len(pick) != len(fs),
            "max_abs_prob_diff": prob_diff}


def holm_r4(tests: dict, fam: list[str]) -> dict[str, float]:
    """Holm over the R4 family ids present, in the family's order (configs/stats.yaml)."""
    ordered = [t for t in fam if t in tests]
    if sorted(ordered) != sorted(tests):
        raise RuntimeError(f"Table 4.5 tests {sorted(tests)} are not all in the R4 family {fam}")
    return holm_family_adjusted(tests, ordered)  


def check_replay_seeds(entries: list[dict], name: str) -> None:
    """Replay ASR is read at seed 0 only (pre-registered specification."""
    bad = [e["run_id"] for e in entries if int(e["seed"]) != 0]
    if bad:
        raise RuntimeError(f"Table 4.5: {name} has runs of seeds other than 0 ({', '.join(bad)}); the replay ASR uses "
                           f"seed 0 only (specification item 15)")


def _attack_order(attacks) -> list[str]:
    present = set(attacks)
    return [a for a in ATTACK_ORDER if a in present] + sorted(present - set(ATTACK_ORDER))


def _r3_stat(arr: np.ndarray) -> float:
    """arr columns: seed index, attack index, success (0/1). Mean over seeds of the mean over attacks of the ASR."""
    seed_means = []
    for s in np.unique(arr[:, 0]):
        sub = arr[arr[:, 0] == s]
        seed_means.append(float(np.mean([float(np.mean(sub[sub[:, 1] == a, 2])) for a in np.unique(sub[:, 1])])))
    return float(np.mean(seed_means))


def r3_cell(seed_rows: list[tuple[int, list[dict]]], *, cell_key: str, n_resamples: int, level: float,
            with_ci: bool = True) -> dict:
    """R3 cell over ledger rows grouped by seed: value = mean over seeds of the mean over attacks of the strict ASR over
 attacked rows (queries_used > 0); None when no row was attacked. ci = item-cluster bootstrap (all rows of an item,
 every attack and seed, stay together; aac.stats.bootstrap.cluster_ci keyed by cell_key)."""
    att = [(s, r) for s, rows in seed_rows for r in rows if int(r["queries_used"]) > 0]
    if not att:
        return {"value": None, "n_items": 0, "ci": None, "parts": []}
    seeds = sorted({s for s, _ in att})
    attacks = _attack_order(r["attack"] for _, r in att)
    parts = []
    for s in seeds:
        rates = []
        for a in attacks:
            sub = [r for t, r in att if t == s and r["attack"] == a]
            if sub:
                rates.append(f"{a}={fmt.fmt_rate(sum(r['end_state'] == 'success' for r in sub) / len(sub))}")
        if len(seeds) == 1:
            parts.extend(rates)
        else:
            parts.append(f"seed {s}: " + "; ".join(rates))
    by_item: dict[str, list] = {}
    for s, r in att:
        by_item.setdefault(r["item_id"], []).append([seeds.index(s), attacks.index(r["attack"]), float(r["end_state"] == "success")])
    full_arr = np.asarray([row for k in sorted(by_item) for row in by_item[k]], dtype=float)
    value = _r3_stat(full_arr)
    ci = None
    if with_ci:
        ci = cluster_ci([np.asarray(by_item[k], dtype=float) for k in sorted(by_item)], _r3_stat, cell_key=cell_key,
                        n_resamples=n_resamples, level=level)
    return {"value": value, "n_items": len(att), "ci": ci, "parts": parts}


# --------------------------------------------------------------------------- sources

class _Pending:
    def __init__(self, reason: str) -> None:
        self.reason = reason


def _select(root: Path, src: Src | None, *, ident: str, col_key: str, allow_pending: bool, cut_step: str | None = None):
    """('runs', entries) | ('cut', step) | ('pending', reason) for one source; MissingRun when nothing covers it."""
    if src is None:
        reason = f"the profile of {phase4_sources.PROFILE_FILE} names no run for this cell"
        if allow_pending:
            return "pending", reason
        raise sources.MissingRun(f"Table 4.5 cell ({ident!r}, {col_key!r}): {reason}")
    entries = sources.select_runs(root, regime=src.regime, name=src.name)
    if entries:
        return "runs", entries
    step = fmt.active_cut_for(TABLE, ident, col_key)
    if step is None and cut_step and phase4_sources.cut_fired(cut_step):
        step = cut_step
    if step:
        return "cut", step
    if allow_pending:
        return "pending", f"no sealed {src.regime}/{src.name} run in runs/INDEX.json yet"
    raise sources.MissingRun(f"Table 4.5 cell ({ident!r}, {col_key!r}) needs sealed {src.regime}/{src.name} run(s) in "
                             f"runs/INDEX.json and no fired configs/cuts.yaml step blanks it")


def _union(root: Path, profile: dict, allow_pending: bool) -> dict:
    rows, runs = phase4_sources.union_ledgers(root, profile["r3"], table=TABLE, allow_pending=allow_pending)
    cut1 = phase4_sources.cut_fired(phase4_sources.CUT_BEAM)
    cut2 = phase4_sources.cut_fired(phase4_sources.CUT_SUBSET)
    if cut1:
        rows = [r for r in rows if r["attack"] != "beamattack"]
    if cut2:
        half = {r.run_id: phase4_sources.subset_half(r) for r in runs}
        rows = [r for r in rows if r["victim"] != "full" or half[r["_run_id"]] == "first200"]
    return {"rows": rows, "runs": {r.run_id: r for r in runs}, "cut1": cut1, "cut2": cut2}


def _meta(runs) -> tuple[int, float]:
    vals = {(sources.resamples(None, r), sources.level(r)) for r in runs}
    if len(vals) != 1:
        raise ValueError(f"Table 4.5: the R3 source runs disagree on stats.bootstrap / stats.level: {sorted(vals)}")
    return vals.pop()


def _victim_r3(union: dict, victim: str, ident: str, allow_pending: bool, cell_key: str):
    rows = [r for r in union["rows"] if r["victim"] == victim and r["subset"] == "averitec_chrono"]
    if not rows:
        reason = f"no {victim} ledger rows on averitec_chrono in the indexed R3 runs of the profile"
        if allow_pending:
            return _Pending(reason)
        raise sources.MissingRun(f"Table 4.5 {ident!r} R3 cell: {reason}")
    order = list(union["runs"])
    runs = [union["runs"][rid] for rid in sorted({r["_run_id"] for r in rows}, key=order.index)]
    n_boot, lvl = _meta(runs)
    c = r3_cell([(0, rows)], cell_key=cell_key, n_resamples=n_boot, level=lvl)
    return {**c, "runs": runs, "n_seeds": 1}


def _replay_r3(root: Path, name: str | None, ident: str, allow_pending: bool, cut1: str | None, cell_key: str):
    entries = phase4_sources.indexed(root, [name], regime=REGIME) if name else []
    if name:
        check_replay_seeds(entries, name)
    if not entries:
        reason = (f"no sealed {REGIME}/{name} replay run in runs/INDEX.json yet" if name
                  else f"the profile of {phase4_sources.PROFILE_FILE} names no replay run")
        if allow_pending:
            return _Pending(reason)
        raise sources.MissingRun(f"Table 4.5 {ident!r} R3 cell: {reason}")
    rows, runs = phase4_sources.union_ledgers(root, [name], table=TABLE, regime=REGIME)
    rows = [r for r in rows if r["subset"] == "averitec_chrono" and not (cut1 and r["attack"] == "beamattack")]
    n_boot, lvl = _meta(runs)
    c = r3_cell([(0, rows)], cell_key=cell_key, n_resamples=n_boot, level=lvl)
    return {**c, "runs": runs, "n_seeds": 1}


def _stub_victim_r3(root: Path, victim: str, ident: str, cell_key: str) -> dict:
    """Stub namespace: the r3-stub run's ledger (only the run whose ledger is read is cited)."""
    state, entries = sources.require_runs(root, regime="R3", name=phase4_sources.STUB_R3, table=TABLE, row_ident=ident,
                                          col_key="r3_asr_mean")
    if state == "cut":
        raise sources.MissingRun("Table 4.5: the R3 run is never blanked by a cut")
    run = phase4_sources.load_run_light(root, entries[0])
    rows = [r for r in read_ledger(run.run_dir) if r["victim"] == victim and r["subset"] == "averitec_chrono"]
    n_boot, lvl = _meta([run])
    return {**r3_cell([(0, rows)], cell_key=cell_key, n_resamples=n_boot, level=lvl), "runs": [run], "n_seeds": 1}


def _stub_replay_r3(r4_runs: list, cell_key: str) -> dict:
    """Stub namespace: the replay ledgers inside the stub R4 runs, one per seed (mean over seeds)."""
    seed_rows = [(run.seed, [r for r in read_ledger(run.run_dir) if r["subset"] == "averitec_chrono"]) for run in r4_runs]
    n_boot, lvl = _meta(r4_runs)
    return {**r3_cell(seed_rows, cell_key=cell_key, n_resamples=n_boot, level=lvl), "runs": list(r4_runs),
            "n_seeds": len(r4_runs)}


R3_NOTE = {
    "victim_b": ("adaptive attack on victim B (full pipeline C1->C4 un-gated, gate post hoc), AVeriTeC subset, union of the "
                 "sealed R3 ledgers of the profile (D-06)"),
    "victim_a": ("adaptive attack on victim A (content-only encoder), AVeriTeC subset, union of the sealed R3 ledgers of the "
                 "profile (D-06)"),
    "replay": ("REPLAY of victim B's final adversarial examples (seed 0) through the ablated configuration, sealed replay run "
               "(queries_used 1 = denominator, 0 = excluded), not an adaptive attack (protocol notes)"),
}
R3_NOTE_STUB = {
    "victim_b": "adaptive attack on victim B (full pipeline C1->C4 un-gated, gate post hoc), AVeriTeC subset, sealed R3 ledger (D-06)",
    "victim_a": "adaptive attack on victim A (content-only encoder), AVeriTeC subset, sealed R3 ledger: minus-retrieval under attack IS victim A (D-06)",
    "replay_stub": "REPLAY of victim B's final adversarial examples through the ablated configuration (R4 run's replay ledger), not an adaptive attack (D-06)",
}
R3_KIND = {"victim_b": "victim_b", "victim_a": "victim_a", "replay": "replay", "replay_stub": "replay"}


# --------------------------------------------------------------------------- build

def build(root: Path, out_dir: Path | None = None, allow_pending: bool = False) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    return _build(root, out_dir, phase4_sources.load_profile(root), allow_pending)


def _build(root: Path, out_dir: Path, profile: dict, allow_pending: bool) -> list[Path]:
    specs = row_specs(profile)
    stub = profile["name"] == phase4_sources.STUB_PROFILE
    pend = fmt.PENDING(PENDING_PHASE)
    union = None
    if not stub and any(s.r3_mode in ("victim_a", "victim_b") or s.extra_victim_a for s in specs):
        union = _union(root, profile, allow_pending)
    cut1 = None if stub else phase4_sources.cut_fired(phase4_sources.CUT_BEAM)
    cut3 = None if stub else phase4_sources.cut_fired(CUT_SEEDS)
    data: dict[str, dict] = {}
    for s in specs:
        d: dict = {"spec": s}
        state, val = _select(root, s.r1, ident=s.label, col_key="r1_chrono_macro_f1", allow_pending=allow_pending)
        if state == "cut":
            raise sources.MissingRun(f"Table 4.5 row {s.label!r} R1 cell is never blanked by a cut")
        d["r1"] = _Pending(val) if state == "pending" else _f1(root, val, s.r1.split, f"boot.table-4.5.{s.label}.r1_chrono_macro_f1",
                                                                  decided=True)
        for target, src in (("fc", s.fc), ("rd", s.rd)):
            state, val = _select(root, src, ident=s.label, col_key="r2_transfer_macro_f1", allow_pending=allow_pending,
                                 cut_step=None if stub or target != "rd" else CUT_RD)
            d[target] = ({"cut": val} if state == "cut" else _Pending(val) if state == "pending"
                         else _f1(root, val, src.split, f"boot.table-4.5.{s.label}.r2_{target}"))
        key = f"boot.table-4.5.{s.label}.r3_asr_mean"
        if s.r3_mode == "replay":
            d["r3"] = _replay_r3(root, s.replay, s.label, allow_pending, cut1, key)
        elif s.r3_mode == "replay_stub":
            d["r3"] = _stub_replay_r3(d["r1"]["runs"], key)
        elif stub:
            d["r3"] = _stub_victim_r3(root, "full" if s.r3_mode == "victim_b" else "content_only", s.label, key)
        else:
            d["r3"] = _victim_r3(union, "full" if s.r3_mode == "victim_b" else "content_only", s.label, allow_pending, key)
        if s.extra_victim_a:
            d["r3_va"] = _victim_r3(union, "content_only", s.label, allow_pending, f"boot.table-4.5.{s.label}.r3_victim_a")
        data[s.label] = d
    
    full = data[FULL_ROW]["r1"]
    tests, deltas = {}, {}
    for s in specs:
        if s.test_id is None:
            continue
        r1 = data[s.label]["r1"]
        if isinstance(full, _Pending) or isinstance(r1, _Pending):
            continue
        deltas[s.label] = delta_tests(full, r1, ident=s.label, cut3=cut3)
        tests[s.test_id] = deltas[s.label]["mcnemar"]
    fam = holm_family(REGIME, root)
    family_complete = sorted(tests) == sorted(fam)
    if not family_complete and not allow_pending:
        raise sources.MissingRun(f"Table 4.5: R4 family {fam} incomplete (have {sorted(tests)})")
    adj = holm_r4(tests, fam) if family_complete else {}
    loaded = [r for d in data.values() for k in ("r1", "fc", "rd", "r3", "r3_va")
              if isinstance(d.get(k), dict) and "runs" in d[k] for r in d[k]["runs"]]
    if not loaded:
        raise sources.MissingRun("Table 4.5: no source of any row is indexed")
    dv_any = sources.provenance(loaded)["data_version"]

    shell_rows, long_rows = [], []

    def pending_row(row: str, col: str, metric: str, reason: str) -> None:
        long_rows.append(fmt.long_row(TABLE, row, col, metric, pend, config_hash="", data_version=dv_any, run_ids="n/a",
                                      source_shards_sha256="n/a", test=f"{reason}; filled in {PENDING_PHASE}"))

    for s in specs:
        row, d = s.label, data[s.label]
        identity = f"; {IDENTITY_NOTE}" if row == ABSTENTION_ROW else ""
        cells = {"configuration": row}
        r1 = d["r1"]
        common: dict = {}
        if isinstance(r1, _Pending):
            cells["r1_chrono_macro_f1"] = pend
            pending_row(row, R1_COL, "macro_f1_binary", r1.reason)
        else:
            prov = sources.provenance(r1["runs"])
            common = dict(n_items=r1["n_items"], n_seeds=len(r1["runs"]), seeds=sources.seeds_text(r1["runs"]), **prov)
            cells["r1_chrono_macro_f1"] = fmt.fmt_ci(r1["f1"], *r1["ci"])
            long_rows.append(fmt.long_row(TABLE, row, R1_COL, "macro_f1_binary", fmt.fmt_f1(r1["f1"]),
                                          ci_lo=fmt.fmt_f1(r1["ci"][0]), ci_hi=fmt.fmt_f1(r1["ci"][1]),
                                          test="percentile bootstrap (items; seeds as replicates); chronological test split" + identity,
                                          **common))
        cells["r2_transfer_macro_f1"] = _r2_rows(long_rows, row, d, dv_any, pend, pending_row)
        cells["r3_asr_mean"] = _r3_rows(long_rows, row, s, d, union, cut1, stub, pending_row)
        # ECE after C4 (95 percent item-bootstrap interval in the long row, methodological note)
        if isinstance(r1, _Pending):
            cells["ece_after"] = pend
            pending_row(row, ECE_COL, "ece15_calibrated_probability", r1.reason)
        else:
            cells["ece_after"] = fmt.fmt_metric(r1["ece"])
            lo, hi = _ece_ci(r1, f"boot.table-4.5.{row}.ece_after")
            ci_txt = "; 95% item-bootstrap CI (seeds as replicates)"
            if s.calibrated:
                long_rows.append(fmt.long_row(TABLE, row, ECE_COL, "ece15_calibrated_probability", cells["ece_after"],
                                              ci_lo=fmt.fmt_metric(lo), ci_hi=fmt.fmt_metric(hi),
                                              test="netcal ECE(bins=15) on the chronological test split, mean over seeds" + ci_txt + identity,
                                              **common))
            else:
                long_rows.append(fmt.long_row(TABLE, row, ECE_COL, "ece15_decision_probability_no_c4", cells["ece_after"],
                                              ci_lo=fmt.fmt_metric(lo), ci_hi=fmt.fmt_metric(hi),
                                              test=("no C4 in this configuration: ECE of the uncalibrated decision probability "
                                                    "(what removing C4 leaves), mean over seeds" + ci_txt), **common))
            _risk_row(long_rows, row, r1, common)
        # Delta vs full
        if s.test_id is None:
            if isinstance(r1, _Pending):
                cells["delta_vs_full"] = pend
                pending_row(row, DELTA_COL, "delta_macro_f1_vs_full", r1.reason)
            else:
                cells["delta_vs_full"] = "reference"
                long_rows.append(fmt.long_row(TABLE, row, DELTA_COL, "delta_macro_f1_vs_full", "reference", **common))
        elif row not in deltas:
            cells["delta_vs_full"] = pend
            pending_row(row, DELTA_COL, "delta_macro_f1_vs_full", "the R1 source of this row or of the Full pipeline has no sealed run yet")
        else:
            cells["delta_vs_full"] = _delta_rows(long_rows, row, s, deltas[row], full, r1, adj, family_complete, fam, tests, pend)
        shell_rows.append(cells)
    prov_all = sources.provenance(loaded)
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=prov_all["config_hash"], data_version=prov_all["data_version"],
                         out_dir=out_dir, allow_pending=allow_pending,
                         comment_lines=["note: " + str(fmt.load_shells()[TABLE]["notes"])]))


def _r2_rows(long_rows: list, row: str, d: dict, dv_any: str, pend: str, pending_row) -> str:
    """The (fc) and (rd) long rows and the composite cell `fc / rd` (pending when either half is pending)."""
    parts, any_pending = [], False
    for target in ("fc", "rd"):
        t = d[target]
        col = f"{R2_COL} ({target})"
        if isinstance(t, _Pending):
            any_pending = True
            parts.append(pend)
            pending_row(row, col, "macro_f1_binary", t.reason)
        elif "cut" in t:
            parts.append(fmt.NOT_RUN(t["cut"]))
            long_rows.append(fmt.long_row(TABLE, row, col, "macro_f1_binary", parts[-1], config_hash="", data_version=dv_any,
                                          run_ids="n/a", source_shards_sha256="n/a", test=f"cut {t['cut']} (configs/cuts.yaml)"))
        else:
            parts.append(fmt.fmt_f1(t["f1"]))
            long_rows.append(fmt.long_row(TABLE, row, col, "macro_f1_binary", parts[-1], ci_lo=fmt.fmt_f1(t["ci"][0]),
                                          ci_hi=fmt.fmt_f1(t["ci"][1]), n_items=t["n_items"], n_seeds=len(t["runs"]),
                                          seeds=sources.seeds_text(t["runs"]),
                                          test="percentile bootstrap (items; seeds as replicates)" +
                                               (f"; {IDENTITY_NOTE}" if row == ABSTENTION_ROW else ""),
                                          **sources.provenance(t["runs"])))
    if any_pending:
        pending_row(row, R2_COL, "macro_f1_binary (fc / rd)", "a half of the composite cell has no sealed run yet: " + " / ".join(parts))
        return pend
    cell = " / ".join(parts)
    r2_runs = [r for tg in ("fc", "rd") if "cut" not in d[tg] for r in d[tg]["runs"]]
    
    per_target = {tg: len(d[tg]["runs"]) for tg in ("fc", "rd") if "cut" not in d[tg]}
    r2_seeds = ("n/a" if not per_target else next(iter(per_target.values())) if len(set(per_target.values())) == 1
                else " / ".join(f"{tg}={n}" for tg, n in per_target.items()))
    pr2 = sources.provenance(r2_runs) if r2_runs else {"config_hash": "", "data_version": dv_any, "run_ids": "n/a",
                                                       "source_shards_sha256": "n/a"}
    long_rows.append(fmt.long_row(TABLE, row, R2_COL, "macro_f1_binary (fc / rd)", cell, n_seeds=r2_seeds,
                                  test="composite cell: the (fc) and (rd) long rows carry each value", **pr2))
    return cell


def _rate_ci(ci) -> dict:
    return {} if ci is None else {"ci_lo": fmt.fmt_rate(ci[0]), "ci_hi": fmt.fmt_rate(ci[1])}


def _r3_rows(long_rows: list, row: str, s: RowSpec, d: dict, union: dict | None, cut1: str | None, stub: bool,
             pending_row) -> str:
    """The R3 long row (mean over attacks with its item-cluster interval) and, for Minus retrieval, victim A's row."""
    r3 = d["r3"]
    ci_txt = f"; 95% item-cluster bootstrap CI (all rows of an item together; cell key boot.table-4.5.{row}.r3_asr_mean)"
    if isinstance(r3, _Pending):
        cell = fmt.PENDING(PENDING_PHASE)
        pending_row(row, R3_COL, "asr_mean_pct", r3.reason)
    else:
        cell = NO_ATTACKED if r3["value"] is None else fmt.fmt_rate(r3["value"])
        if stub:
            note = R3_NOTE_STUB[s.r3_mode]
        else:
            note = R3_NOTE[s.r3_mode] + (f" ({s.replay})" if s.r3_mode == "replay" else "")
        if cut1:
            note += f"; {phase4_sources.CUT_BEAM} fired on {cut1}: mean over the four remaining attacks"
        if s.r3_mode == "victim_b" and union and union["cut2"]:
            note += f"; {phase4_sources.CUT_SUBSET} fired on {union['cut2']}: victim-B rows from the first-200 runs only"
        long_rows.append(fmt.long_row(TABLE, row, R3_COL, "asr_mean_pct", cell, n_items=r3["n_items"], n_seeds=r3["n_seeds"],
                                      seeds=sources.seeds_text(r3["runs"]), effect_size_kind=R3_KIND[s.r3_mode],
                                      test=note + "; per attack: " + (" | ".join(r3["parts"]) or "none attacked")
                                           + ("" if r3["ci"] is None else ci_txt),
                                      **_rate_ci(r3["ci"]), **sources.provenance(r3["runs"])))
    if s.extra_victim_a:
        va = d["r3_va"]
        if isinstance(va, _Pending):
            pending_row(row, R3_VA_COL, "asr_mean_pct", va.reason)
        else:
            long_rows.append(fmt.long_row(
                TABLE, row, R3_VA_COL, "asr_mean_pct", NO_ATTACKED if va["value"] is None else fmt.fmt_rate(va["value"]),
                n_items=va["n_items"], n_seeds=1, seeds=sources.seeds_text(va["runs"]), effect_size_kind="victim_a",
                test=("victim A (content-only encoder) adaptive attack on the AVeriTeC subset: the specification item 13 reading of "
                      "minus-retrieval, kept as a labeled row; the shell cell is the replay through the ablated "
                      "configuration (protocol notes); per attack: " + (" | ".join(va["parts"]) or "none attacked")
                      + ("" if va["ci"] is None else f"; 95% item-cluster bootstrap CI (cell key boot.table-4.5.{row}.r3_victim_a)")),
                family="none", **_rate_ci(va["ci"]), **sources.provenance(va["runs"])))
    return cell


def _risk_row(long_rows: list, row: str, r1: dict, common: dict) -> None:
    """Selective risk at 80 percent coverage from each R1 run's gate.json (methodological note)."""
    risk = selective_risk_80(r1)
    if risk["value"] is None:
        long_rows.append(fmt.long_row(TABLE, row, RISK_COL, "selective_risk_at_80_pct", risk["reason"], family="none",
                                      test="selective risk = error rate among the auto-decided items at the gate.json report "
                                           f"threshold {RISK_COVERAGE}", **common))
        return
    per = "; ".join(f"seed {sd}: risk {fmt.fmt_rate(rk)}, coverage {fmt.fmt_rate(cv)}" for sd, rk, cv in risk["per_seed"])
    long_rows.append(fmt.long_row(
        TABLE, row, RISK_COL, "selective_risk_at_80_pct", fmt.fmt_rate(risk["value"]), family="none",
        test=(f"selective risk = error rate among the auto-decided items ({DECISION_RULE}) at each run's gate.json report "
              f"threshold {RISK_COVERAGE} through the C5 rule (aac.reports.table_4_6.auto_mask), on the chronological test "
              f"split, mean over seeds; coverage achieved {fmt.fmt_rate(risk['coverage'])} (mean); {per}"), **common))


def _delta_rows(long_rows: list, row: str, s: RowSpec, dt: dict, full: dict, abl: dict, adj: dict, family_complete: bool,
                fam: list[str], tests: dict, pend: str) -> str:
    """Append the primary and secondary Delta long rows of one ablation (and the Minus abstention identity rows);
 return the shell cell."""
    paired = [r for r in full["runs"] if r.seed in dt["seeds"]] + list(abl["runs"])
    prov = sources.provenance(paired)
    common = dict(n_items=dt["n_items"], n_seeds=len(dt["seeds"]), seeds=", ".join(str(x) for x in dt["seeds"]), **prov)
    (n11, b), (c, n00) = dt["table"]
    lo, hi = dt["boot_ci"]
    txt = (f"primary (specification item 9): exact McNemar on the per-item outcome decided and correct "
           f"(a deferred item is otherwise), taken over seeds {', '.join(str(x) for x in dt['seeds'])} by majority, on the "
           f"{dt['n_items']} chronological test items ({dt['n_pos']} Refuted); table [[{n11}, {b}], [{c}, {n00}]] "
           f"(b = full right only, c = ablation right only); pairing key (claim_id, seed); with the paired percentile item "
           f"bootstrap (same index resample for both systems and every seed) of the difference in seed-mean binary macro-F1 "
           f"(ablated minus full); {DECISION_RULE}")
    if row == ABSTENTION_ROW:
        txt += ("; for Minus abstention the McNemar measures deferral (the full pipeline defers, the ablation decides every "
                "item) while the macro-F1 difference is zero by construction")
    if dt["subset"]:
        txt += f"; {CUT_SEEDS} fired: the full pipeline is paired on the ablation's seeds only"
    fam_id = f"{REGIME}:{s.test_id}"
    if family_complete:
        cell = f"{fmt.fmt_p(adj[s.test_id])} (Holm); diff={fmt.fmt_signed(dt['delta'])} [{fmt.fmt_signed(lo)}, {fmt.fmt_signed(hi)}]"
        long_rows.append(fmt.long_row(TABLE, row, DELTA_COL, "delta_macro_f1_vs_full", fmt.fmt_signed(dt["delta"]),
                                      p_raw=f"{dt['mcnemar'].p:.6f}", p_holm=f"{adj[s.test_id]:.6f}", ci_lo=fmt.fmt_signed(lo),
                                      ci_hi=fmt.fmt_signed(hi), effect_size=f"{dt['delta']:.4f}",
                                      effect_size_kind="paired_delta_macro_f1_vs_full",
                                      test=txt + "; Holm within R4 on the McNemar p", family=fam_id, **common))
    else:
        missing = [x for x in fam if x not in tests]
        cell = pend
        long_rows.append(fmt.long_row(TABLE, row, DELTA_COL, "delta_macro_f1_vs_full", pend, p_raw=f"{dt['mcnemar'].p:.6f}",
                                      p_holm=pend, ci_lo=fmt.fmt_signed(lo), ci_hi=fmt.fmt_signed(hi),
                                      effect_size=f"{dt['delta']:.4f}", effect_size_kind="paired_delta_macro_f1_vs_full",
                                      test=f"{txt}; Holm within R4 waits for {', '.join(missing)} ({PENDING_PHASE})",
                                      family=fam_id, **common))
    sec = dt["secondary"]
    sec_txt = ("secondary (specification item 9; not a decision criterion): paired t over the per-seed macro-F1 differences "
               "(ablated minus full) with Cohen's d_z")
    if sec is None:
        long_rows.append(fmt.long_row(TABLE, row, DELTA_SEC_COL, "paired_t_dz_seed_level", "n/a (fewer than two paired seeds)",
                                      test=sec_txt, family="secondary", **common))
    elif sec.degenerate:
        long_rows.append(fmt.long_row(TABLE, row, DELTA_SEC_COL, "paired_t_dz_seed_level", "n/a (zero variance)",
                                      effect_size_kind="zero variance", test=sec_txt + ": degenerate", family="secondary", **common))
    else:
        long_rows.append(fmt.long_row(TABLE, row, DELTA_SEC_COL, "paired_t_dz_seed_level", fmt.fmt_signed(dt["delta"], 4),
                                      p_raw=f"{sec.p:.6f}", effect_size=f"{sec.d_z:.4f}", effect_size_kind="cohen_dz",
                                      test=sec_txt, family="secondary", **common))
    if row == ABSTENTION_ROW:
        for sd, v in zip(dt["seeds"], dt["max_abs_prob_diff"]):
            long_rows.append(fmt.long_row(
                TABLE, row, f"{IDENTITY_COL}, seed {sd}", "max_abs_delta_calibrated_vs_full", f"{v:.6f}", n_items=dt["n_items"],
                n_seeds=1, seeds=str(sd), family="none",
                test=(f"largest absolute difference in calibrated_probability between Minus abstention and Full on the same "
                      f"chronological test items at seed {sd} (pairing key claim_id, seed), 6 d.p.; {IDENTITY_NOTE} "
                      f"(evaluation specification)"),
                **sources.provenance([r for r in full["runs"] if r.seed == sd] + [r for r in abl["runs"] if r.seed == sd])))
    return cell
