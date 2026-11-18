"""Table 4.4 (R3 adversarial) from the union of the sealed R3 attack ledgers named in configs/reports/phase4-runs.yaml
(the stub namespace keeps reading r3-stub), via aac.metrics.attack. Ten rows (5 attacks x 2 victims); the shell cell pools the attack subsets (AVeriTeC-chrono, FC, RD) and the long CSV carries one row per subset;
attacked n is victim-specific (long-CSV n_items). Every rate and score cell carries a 95 percent item-bootstrap
interval (the pooled cell resamples items within subset strata); the median queries per success keeps its
plain value in the shell and gets a bootstrap interval in a long row. Content-only Deflection = n/a.

H3 (pre-registered specification):
 H3-asr-full-vs-content ONE exact McNemar over the AVeriTeC items attacked by both victims under every recipe that
 ran for both; an item is evaded when any of those recipes succeeded strictly (worst case
 over the battery, as AutoAttack counts robustness). Effect size = pooled ASR difference
 (full minus content-only, over the items' (item, recipe) rows) with an item-cluster bootstrap
 interval (all recipes of a drawn item stay together)
 H3-deflection item-cluster percentile bootstrap of the deflection rate over attacked gated victim-B rows;
 no p, so p_holm = p_raw by the empty-case rule
 R3-per-recipe (long CSV only, not a stats.yaml family) exact McNemar per recipe, Holm across the recipes
Outside the stub profile the long CSV also carries the zero-GPU deflection references (a)-(e) (aac.reports.deflection),
the un-gated flip rate per recipe on victim B (the addition-A recipe rule input), the budget-exhausted share per
subset, the BeamAttack LIME queries per item (configs/attacks/beamattack.yaml) and the BODEGA score from sealed
BODEGA scoring runs (mean over attacked rows of strict success x semantic x character; `n/a` with the reason when no
scoring run is indexed). Cuts: cut-1-beamattack blanks the BeamAttack rows; cut-2-subset-200 restricts the victim-B
rows to the first-200 runs (real profiles).

Supplementary rows of the gap review's additions (family R3-supplementary, outside every Holm family,
never a shell cell), from the runs the profile's `supplementary` section names (stub profile: none):
 addition A gate_aware (r3-gate-aware; victim B through the gated-masked view, OpenAttackGatedCfg): per recipe the
 attacked n, strict ASR [CI], deflection [CI], median queries per success, budget-exhausted share, and
 the exact McNemar of strict success gate-aware vs gate-unaware (the union's victim-B rows of the same
 recipe) on the items both attacked
 addition B transfer (r4-transfer-a-to-b-replay): victim A's AVeriTeC finals through gated victim B, strict
 success / replay denominator beside victim A's own ASR on those rows; gate_only (r4-gate-only-a-replay)
 and the R4 minus-retrieval replay (r4-no-retrieval-replay, victim B's finals): gate-only deflection and
 ASR through the gated content-only view; per recipe and pooled, item-cluster bootstrap intervals
 references (a), (d) and (e) also per recipe (the deflection-controls figure, aac.reports.fig_deflection)
A named supplementary run that is not indexed renders `not run (cut: addition-A on 2026-10-30)` once its own cut rule
has passed (methodological note
otherwise raises MissingRun naming the run. A recipe that ran in the main table but has no row in a sealed addition-B
replay (its source was removed from replay.sources because its private store was incomplete) reads
`not run (cut: addition-B store incomplete on <the replay run's finished_utc date>)`."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from aac.metrics import attack as atk
from aac.reports import deflection, fmt, phase4_sources, sources
from aac.run.paths import reports_dir
from aac.stats.bootstrap import cluster_ci
from aac.stats.holm import holm
from aac.stats.tests import mcnemar_exact

TABLE = "4.4"
REGIME = "R3"
RUN_NAME = phase4_sources.STUB_R3  # the stub namespace source (keeps it)
ATTACKS = (("BERT-Attack", "bert_attack"), ("TextFooler", "textfooler"), ("DeepWordBug", "deepwordbug"),
           ("PWWS", "pwws"), ("BeamAttack", "beamattack"))
LABEL = {a: lab for lab, a in ATTACKS}
BODEGA_RECIPES = ("bert_attack", "textfooler", "deepwordbug", "pwws")
VICTIMS = (("Content-only encoder", "content_only"), ("Full pipeline", "full"))
SUBSETS = ("averitec_chrono", "fc", "rd")
COLS = {"asr": "Attack-success rate (percent)", "bodega_score": "BODEGA score",
        "queries_per_success_median": "Queries per success (median)", "deflection_rate": "Deflection rate (percent)"}
BODEGA_NOTE = "bodega_score: not computed on stubs (no field in contracts/attack-ledger.schema.json; Phase 4 adds it under D-18)"
NO_BS = "bodega_score: no sealed BODEGA scoring run"
POOLED_ROW = "all attacks (pooled, AVeriTeC subset)"
BEAM_CFG = Path("configs") / "attacks" / "beamattack.yaml"
FAM_ASR = "R3:H3-asr-full-vs-content"
FAM_DEFL = "R3:H3-deflection"
FAM_RECIPE = "R3-per-recipe"
FAM_SUPP = "R3-supplementary"
COL_DIFF = "Attack-success rate difference (pp), full minus content-only"
COL_RECIPE = "Attack-success rate (percent), per-recipe McNemar vs content-only (AVeriTeC)"
COL_QCI = "Queries per success (median), bootstrap CI"
COL_FLIP = "Un-gated flip rate (percent)"
COL_BUDGET = "Budget-exhausted share (percent)"
COL_LIME = "BeamAttack LIME queries per item"
NO_ATTACKED = "n/a (no attacked item)"
# : denominator categories and R46 counts (long CSV only; real profiles)
OPERATOR_R47 = ("Item 3: (a). Query-budget exhaustion is a legitimate attack failure and BODEGA's confusion score counts "
                "it as 0, so the 110 stay in the denominator; show them as their own row. R47 was about wall-clock "
                "deadline timeouts, a different thing \u2014 keep that rule for those only.")
OPERATOR_R47_SOURCE = "verified experimental session outcome"
COUNT_COLS = (  # (key, column label, metric)
    ("ledger", "Ledger rows (planned items with a ledger row)", "ledger_rows_n"),
    ("skipped", "Skipped items (clean decision already wrong, 0 queries; outside the ASR denominator)", "skipped_n"),
    ("attacked", "Attacked items (ASR denominator)", "attacked_n"),
    ("success", "Strict successes", "strict_success_n"),
    ("budget", "Budget-exhausted items (own category; in the ASR denominator as non-successes)", "budget_exhausted_n"),
    ("deferred", "Deferred items (un-gated flips the gate deferred)", "deferred_n"),
    ("in_store", "Successes in store (strict success + deferred: every stored un-gated flip)", "successes_in_store_n"),
    ("timed_out", "Timed-out items (wall-clock deadline, not finished; outside the ASR denominator, R47)", "timed_out_n"),
)
COL_ALT = ("Attack-success rate, alternative reading: strict successes / (attacked - budget-exhausted) "
           "(percent; not the shell value)")
R46_SOURCES = (Path("data") / "measured" / "phase4-gpu-p4-2.json", Path("data") / "measured" / "phase4-session.json",
               Path("data") / "measured" / "phase4-scoring-session.json")
R46_KEYS = ("successes_in_store", "entirely_beyond_256", "partly_beyond_256", "entirely_beyond_128", "partly_beyond_128")
R46_COL = "R46 perturbation position over stored successes (count)"
# : the gap review's additions (supplementary rows; family R3-supplementary)
CUT_TZ = "Europe/London"
# fires_at: the first instant at which the addition's own cut rule is true; date: the `on <date>` of the cell text
ADDITION_CUTS = {
    "addition-A": {"fires_at": "2026-10-30T12:00", "date": "2026-10-30",
                   "rule": "protocol notes row 50 (draft R-A): the gate-aware run had not started by 30 Oct 2026 12:00 Europe/London"},
    "addition-C": {"fires_at": "2026-11-01T00:00", "date": "2026-11-01",
                   "rule": "protocol notes row 51 (draft R-C): the oracle runs were not sealed by 31 Oct 2026 23:59 Europe/London"},
}
STORE_CUT = "addition-B store incomplete"
GATED_MASKED = "gated-masked"
SUPP_KEYS = ("gate_aware", "transfer", "gate_only", "gate_only_b")
COL_N = "Attacked n"
COL_MCN = "Strict success, gate-aware vs gate-unaware (exact McNemar, paired items)"
COL_TRANSFER = "Transfer strict-success rate (percent)"
COL_A_SAME = "Content-only encoder attack-success rate on the same rows (percent)"
COL_GO_DEFL = "Gate-only deflection rate (percent)"
COL_GO_ASR = "Gate-only attack-success rate (percent)"
REPLAY_VIEWS = {  # supplementary key -> (victim of the replayed finals, row label suffix, what)
    "transfer": ("content_only", "Content-only encoder finals through gated full pipeline (addition B transfer)",
                 "victim A's AVeriTeC finals replayed through gated victim B (r4-transfer-a-to-b-replay)"),
    "gate_only": ("content_only", "Content-only encoder finals through gated content-only view (addition B gate isolation)",
                  "victim A's AVeriTeC finals replayed through the gated content-only view (r4-gate-only-a-replay)"),
    "gate_only_b": ("full", "Full pipeline finals through gated content-only view (addition B gate isolation)",
                    "victim B's AVeriTeC finals replayed through the gated content-only view (r4-no-retrieval-replay)"),
}


# --------------------------------------------------------------------------- cells

def _stratum(r: dict) -> str:
    """Bootstrap stratum of a ledger row : the subset, and for a victim-B row of a halved run the half
 (`averitec_chrono|first200`, `averitec_chrono|last200`); cluster_ci draws strata in sorted label order, so a pooled
 victim-B cell draws first200 then last200 under one rng per cell."""
    h = r.get("_half")
    return r["subset"] if not h else f"{r['subset']}|{h}"


def _ci(values, strata, cell_key: str, meta: dict, statistic=None):
    groups = [[float(v)] for v in values]
    stat = statistic or (lambda m: float(np.mean(m)))
    return cluster_ci(groups, stat, cell_key=cell_key, strata=strata, n_resamples=meta["n_resamples"], level=meta["level"])


def _bodega(att: list[dict], scores: dict | None, bs_ids: str) -> tuple[float | None, list | None, str]:
    """(mean, per-row values, metric note) of the BODEGA cell; scores None = the stub profile (no scoring step)."""
    if scores is None:
        return atk.bodega_score_mean(att) if att else None, None, BODEGA_NOTE
    if not bs_ids:
        return None, None, NO_BS
    vals = [scores.get(phase4_sources.ledger_key(r)) for r in att]
    n_missing = sum(v is None for v in vals)
    if not att or n_missing == len(vals):
        return None, None, f"bodega_score: no score rows for these ledger rows in {bs_ids}"
    if n_missing:
        return None, None, f"bodega_score: {n_missing} of {len(vals)} attacked rows unscored in {bs_ids}; no partial mean"
    per = [float(v["bodega"]) for v in vals]
    return float(np.mean(per)), per, "bodega_score_mean"


def _cells(rows: list[dict], victim: str, *, cell_key: str, meta: dict, scores: dict | None = None, bs_ids: str = "") -> dict:
    att = atk.attacked(rows)
    strata = [_stratum(r) for r in att]
    succ = [r for r in att if r["end_state"] == "success"]
    out = {"n_attacked": len(att), "asr": atk.asr(rows), "asr_ci": None, "queries": atk.queries_per_success_median(rows),
           "queries_ci": None, "n_success": len(succ), "deflection": None, "deflection_ci": None, "n_gated": 0}
    if att:
        out["asr_ci"] = _ci([r["end_state"] == "success" for r in att], strata, f"{cell_key}.asr", meta)
    if succ:
        out["queries_ci"] = _ci([int(r["queries_used"]) for r in succ], [_stratum(r) for r in succ], f"{cell_key}.queries",
                                meta, statistic=lambda m: float(np.median(m)))
    if victim == "full":
        gated = [r for r in att if r.get("posthoc_gate") is not None]
        out["n_gated"] = len(gated)
        out["deflection"] = atk.deflection_rate(rows)
        if gated:
            out["deflection_ci"] = _ci([r["posthoc_gate"] == "deflected" for r in gated], [_stratum(r) for r in gated],
                                       f"{cell_key}.deflection", meta)
    b, per, note = _bodega(att, scores, bs_ids)
    out["bodega"], out["bodega_note"], out["bodega_ci"] = b, note, None
    if per is not None:
        out["bodega_ci"] = _ci(per, strata, f"{cell_key}.bodega", meta)
    return out


def _render(c: dict, victim: str) -> dict:
    if c["asr"] is None or not c["n_attacked"]:
        asr = NO_ATTACKED
    else:
        asr = fmt.fmt_rate_ci(c["asr"], *c["asr_ci"])
    bodega = fmt.NA if c["bodega"] is None else fmt.fmt_ci(c["bodega"], *c["bodega_ci"])
    queries = "n/a (no success)" if c["queries"] is None else fmt.fmt_count(c["queries"])
    if victim == "content_only":
        deflection_ = fmt.NA
    else:
        deflection_ = "n/a (no gated rows)" if c["deflection"] is None else fmt.fmt_rate_ci(c["deflection"], *c["deflection_ci"])
    return {"asr": asr, "bodega_score": bodega, "queries_per_success_median": queries, "deflection_rate": deflection_}


def _rate_ci_cols(ci) -> dict:
    return {} if ci is None else {"ci_lo": fmt.fmt_rate(ci[0]), "ci_hi": fmt.fmt_rate(ci[1])}


def _metric_ci_cols(ci) -> dict:
    return {} if ci is None else {"ci_lo": fmt.fmt_metric(ci[0]), "ci_hi": fmt.fmt_metric(ci[1])}


def _count_ci_cols(ci) -> dict:
    return {} if ci is None else {"ci_lo": fmt.fmt_count(ci[0]), "ci_hi": fmt.fmt_count(ci[1])}


# --------------------------------------------------------------------------- sources

def _source(root: Path, allow_pending: bool) -> tuple[str, list[dict], list]:
    """(profile, ledger rows tagged with `_run_id`, runs) from the profile's R3 run list ."""
    profile = phase4_sources.select_profile(root)
    names = phase4_sources.load_profile(root, profile)["r3"]
    rows, runs = phase4_sources.union_ledgers(root, names, table=TABLE)
    return profile, rows, runs


def _apply_cut2(rows: list[dict], runs: list, profile: str) -> tuple[list[dict], str | None]:
    """cut-2-subset-200 (fired, real profiles only): victim-B rows come from the first-200 runs only."""
    fired = None if profile == phase4_sources.STUB_PROFILE else phase4_sources.cut_fired(phase4_sources.CUT_SUBSET)
    if not fired:
        return rows, None
    half = {r.run_id: phase4_sources.subset_half(r) for r in runs}
    kept = [r for r in rows if r["victim"] != "full" or half[r["_run_id"]] == "first200"]
    return kept, fired


def _prov_for(rows: list[dict], by_run: dict, extra: list | None = None) -> tuple[dict, dict]:
    order = list(by_run)
    used = [by_run[rid] for rid in sorted({r["_run_id"] for r in rows}, key=order.index)] + list(extra or [])
    prov = sources.provenance(used)
    seeds = sorted({u.seed for u in used})
    return prov, dict(n_seeds=len(seeds), seeds=", ".join(str(s) for s in seeds))


def _meta(runs: list) -> dict:
    vals = {(sources.resamples(None, r), sources.level(r)) for r in runs}
    if len(vals) != 1:
        raise ValueError(f"Table 4.4: the R3 runs disagree on stats.bootstrap / stats.level: {sorted(vals)}")
    n, lvl = vals.pop()
    return {"n_resamples": n, "level": lvl}


def _ungated(ledger: list[dict]) -> dict[str, float]:
    rates = []
    for k, (_, a) in enumerate(ATTACKS):
        if a not in BODEGA_RECIPES:
            continue
        att = [r for r in ledger if r["victim"] == "full" and r["attack"] == a and int(r["queries_used"]) > 0]
        if att:
            rates.append((a, sum(r["end_state"] in ("success", "deferred") for r in att) / len(att), k))
    rates.sort(key=lambda t: (-t[1], t[2]))
    return {a: v for a, v, _ in rates}


def ungated_flip_rates(root: Path) -> dict[str, float]:
    """{recipe: (strict success + deferred) / attacked victim-B rows} over the four BODEGA recipes that have victim-B
 rows, ordered by rate (highest first), ties in Table 4.4 row order: the first key is the recipe the addition-A
 rule picks. Reads the same runs, after the same cut-2 restriction, as build."""
    root = Path(root)
    profile, ledger, runs = _source(root, False)
    ledger, _ = _apply_cut2(ledger, runs, profile)
    return _ungated(ledger)


def _lime_value(root: Path) -> tuple[str, str]:
    path = BEAM_CFG if Path(BEAM_CFG).is_absolute() else Path(root) / BEAM_CFG
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "lime_queries_per_item" not in cfg:
        raise KeyError(f"{path} has no lime_queries_per_item")
    return fmt.fmt_count(cfg["lime_queries_per_item"]), str(BEAM_CFG)


# --------------------------------------------------------------------------- build

def build(root: Path, out_dir: Path | None = None, allow_pending: bool = False) -> list[Path]:
    """Table 4.4 from the union of the profile's sealed R3 ledgers (stub namespace: r3-stub). A row with no ledger
 rows is `not run (cut: ...)` when a fired cut blanks it, `pending ` only with allow_pending (never in a
 committed report), else MissingRun."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    profile, ledger, runs = _source(root, allow_pending)
    stub = profile == phase4_sources.STUB_PROFILE
    ledger, cut2 = _apply_cut2(ledger, runs, profile)
    by_run = {r.run_id: r for r in runs}
    if not stub:  # : victim-B rows of a halved run carry their half (bootstrap strata, per-half long rows)
        for r in ledger:
            h = phase4_sources.subset_half(by_run[r["_run_id"]])
            if h != "full":
                r["_half"] = h
    prov_all = sources.provenance(runs)
    meta = _meta(runs)
    cut_rows = {}
    for attack_label, attack in ATTACKS:
        for victim_label, victim in VICTIMS:
            step = fmt.active_cut_for(TABLE, f"{attack_label} | {victim_label}", "asr")
            if step and not stub:
                cut_rows[(attack, victim)] = step  # a fired cut blanks the row even when some of its runs exist
    ledger = [r for r in ledger if (r["attack"], r["victim"]) not in cut_rows]
    scores, bs_runs, bs_ids = None, [], ""
    if not stub:
        bs_names = phase4_sources.load_profile(root, profile)["bodega"]
        sup = supplementary_profile(root, profile)
        ga_rows = []
        if "gate_aware" in sup and phase4_sources.indexed(root, [sup["gate_aware"]], regime=REGIME):
            ga_rows, _ = phase4_sources.union_ledgers(root, [sup["gate_aware"]], table=TABLE)
        scores, bs_runs = phase4_sources.read_bodega_scores(root, bs_names, ledger_rows=ledger, extra_rows=ga_rows)
        bs_ids = ";".join(r.run_id for r in bs_runs)
    shell_rows, long_rows = [], []
    for attack_label, attack in ATTACKS:
        for victim_label, victim in VICTIMS:
            ident = f"{attack_label} | {victim_label}"
            rows = [r for r in ledger if r["attack"] == attack and r["victim"] == victim]
            step = cut_rows.get((attack, victim)) or (fmt.active_cut_for(TABLE, ident, "asr") if not rows else None)
            if not rows:
                if step:
                    cell = fmt.NOT_RUN(step)
                    test = f"cut {step} (configs/cuts.yaml)"
                elif allow_pending:
                    cell = fmt.PENDING("Phase 4")
                    test = f"no ledger rows for this row in the indexed runs of profile {profile} (configs/reports/phase4-runs.yaml)"
                else:
                    raise sources.MissingRun(f"Table 4.4 row {ident!r}: no ledger rows in {prov_all['run_ids']} "
                                             f"(profile {profile}) and no fired cut blanks it")
                shell_rows.append({"attack": attack_label, "victim": victim_label, **{k: cell for k in COLS}})
                for key, col in COLS.items():
                    long_rows.append(fmt.long_row(TABLE, ident, col, key, cell, config_hash="", data_version=prov_all["data_version"],
                                                  run_ids="n/a", source_shards_sha256="n/a", test=test))
                continue
            prov, seeds = _prov_for(rows, by_run)
            common = dict(**seeds, **prov)
            key0 = f"boot.table-4.4.{ident}"
            pooled = _cells(rows, victim, cell_key=key0, meta=meta, scores=scores, bs_ids=bs_ids)
            rendered = _render(pooled, victim)
            shell_rows.append({"attack": attack_label, "victim": victim_label, **rendered})
            L = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, ident, col, metric, value, **{**common, **kw}))
            subsets_here = sorted({r["subset"] for r in rows})
            pooled_note = "pooled over subsets " + ", ".join(subsets_here)
            if cut2 and victim == "full":
                pooled_note += f"; {phase4_sources.CUT_SUBSET} fired on {cut2}: victim-B rows from the first-200 runs only (configs/cuts.yaml)"
            boot = f"; 95% item-bootstrap CI ({meta['n_resamples']} resamples, items within subset strata, cell key {key0}.<column>)"
            L(COLS["asr"], "asr_pct", rendered["asr"], n_items=pooled["n_attacked"], **_rate_ci_cols(pooled["asr_ci"]),
              test=pooled_note + "; ASR = strict successes / attacked rows (queries_used > 0)" + boot)
            bkw = _metric_ci_cols(pooled["bodega_ci"])
            bs_test = pooled_note + ("" if pooled["bodega_ci"] is None else
                                     f"; mean over attacked rows of strict success x semantic x character from {bs_ids}" + boot)
            L(COLS["bodega_score"], pooled["bodega_note"], rendered["bodega_score"], n_items=pooled["n_attacked"], test=bs_test, **bkw)
            L(COLS["queries_per_success_median"], "queries_per_success_median", rendered["queries_per_success_median"],
              n_items=pooled["n_success"], test=pooled_note + "; median queries_used over success rows")
            L(COL_QCI, "queries_per_success_median_ci", rendered["queries_per_success_median"], n_items=pooled["n_success"],
              **_count_ci_cols(pooled["queries_ci"]),
              test=pooled_note + "; percentile bootstrap of the median over success rows (H3 second clause has no "
                                 "pre-registered test; the verdict rule is Phase 5's)")
            L(COLS["deflection_rate"], "deflection_rate_pct", rendered["deflection_rate"], n_items=pooled["n_attacked"],
              **_rate_ci_cols(pooled["deflection_ci"]),
              test=("content-only victim has no gate: n/a by the shell" if victim == "content_only" else
                    pooled_note + "; deflected / attacked gated rows" + boot))
            for subset in subsets_here:
                sub_rows = [r for r in rows if r["subset"] == subset]
                c = _cells(sub_rows, victim, cell_key=f"{key0}.{subset}", meta=meta, scores=scores, bs_ids=bs_ids)
                rd = _render(c, victim)
                budget = sorted({int(r["budget"]) for r in sub_rows})
                note = f"subset {subset}; budget {budget[0] if len(budget) == 1 else budget}; n_correct_before {c['n_attacked']} of {len(sub_rows)} positives"
                L(f"{COLS['asr']} ({subset})", "asr_pct", rd["asr"], n_items=c["n_attacked"], test=note, **_rate_ci_cols(c["asr_ci"]))
                L(f"{COLS['bodega_score']} ({subset})", c["bodega_note"], rd["bodega_score"], n_items=c["n_attacked"], test=note,
                  **_metric_ci_cols(c["bodega_ci"]))
                L(f"{COLS['queries_per_success_median']} ({subset})", "queries_per_success_median", rd["queries_per_success_median"],
                  n_items=c["n_success"], test=note)
                L(f"{COLS['deflection_rate']} ({subset})", "deflection_rate_pct", rd["deflection_rate"], n_items=c["n_attacked"],
                  test=note, **_rate_ci_cols(c["deflection_ci"]))
                if not stub:
                    att = atk.attacked(sub_rows)
                    share = fmt.fmt_rate(sum(r["end_state"] == "budget_exhausted" for r in att) / len(att)) if att else NO_ATTACKED
                    L(f"{COL_BUDGET} ({subset})", "budget_exhausted_share_pct", share, n_items=len(att),
                      test=note + "; budget_exhausted rows / attacked rows (04-RESEARCH Pitfall 2: a recipe that spends the "
                                  "budget before any substitution reads as robustness)")
            if not stub:
                long_rows.extend(_count_rows(ident, victim, rows, by_run, common, subsets_here))
                long_rows.extend(_half_rows(ident, victim, rows, common, key0, meta))
            if not stub and attack == "beamattack":
                value, src = _lime_value(root)
                L(COL_LIME, "lime_queries_per_item", value, n_items=pooled["n_attacked"],
                  test=f"lime_queries_per_item from {src} (imp=bert, leave-one-out logit importance: no LIME sampling; "
                       "the rows are BeamAttack adapted to a 1,000-query budget)")
    long_rows.extend(_h3_long_rows(ledger, by_run, meta))
    if not stub:
        long_rows.extend(_flip_rows(ledger, by_run, meta))
        long_rows.extend(_reference_rows(root, ledger, by_run, meta))
        long_rows.extend(_r46_rows(root, ledger, by_run))
        long_rows.extend(_supplementary_rows(root, profile, ledger, by_run, meta, allow_pending, prov_all["data_version"],
                                             scores=scores, bs_ids=bs_ids))
    notes = None if stub else _notes(ledger, by_run, bs_ids, meta)
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=prov_all["config_hash"], data_version=prov_all["data_version"],
                         out_dir=out_dir, allow_pending=allow_pending, comment_lines=notes))


# --------------------------------------------------------------------------- H3

def h3_tests(rows: list[dict], run_meta: dict) -> dict:
    """The H3 statistics on ledger rows (6). run_meta: n_resamples, level.
 Returns {"asr": {...} | None, "per_recipe": [...], "deflection": {...} | None}."""
    n_boot, lvl = int(run_meta["n_resamples"]), float(run_meta["level"])
    av = [r for r in rows if r["subset"] == "averitec_chrono" and int(r["queries_used"]) > 0]
    by: dict[tuple, dict] = {}
    for r in av:
        by.setdefault((r["item_id"], r["victim"]), {})[r["attack"]] = r["end_state"] == "success"
    rec_a = {a for (_, v), d in by.items() if v == "content_only" for a in d}
    rec_b = {a for (_, v), d in by.items() if v == "full" for a in d}
    shared = [a for _, a in ATTACKS if a in rec_a and a in rec_b]
    out: dict = {"asr": None, "per_recipe": [], "deflection": None}
    if shared:
        both = sorted({i for (i, v) in by if v == "content_only"} & {i for (i, v) in by if v == "full"})
        items = [i for i in both if all(a in by[(i, "content_only")] and a in by[(i, "full")] for a in shared)]
        if items:
            a = np.asarray([any(by[(i, "content_only")][x] for x in shared) for i in items])
            b = np.asarray([any(by[(i, "full")][x] for x in shared) for i in items])
            table = [[int(np.sum(a & b)), int(np.sum(a & ~b))], [int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
            res = mcnemar_exact(table)
            groups = [np.asarray([[0.0, by[(i, "content_only")][x]] for x in shared] +
                                 [[1.0, by[(i, "full")][x]] for x in shared], dtype=float) for i in items]
            diff_stat = lambda m: float(m[m[:, 0] == 1, 1].mean() - m[m[:, 0] == 0, 1].mean())  # noqa: E731
            asr_a = float(np.mean([by[(i, "content_only")][x] for i in items for x in shared]))
            asr_b = float(np.mean([by[(i, "full")][x] for i in items for x in shared]))
            kw = dict(n_resamples=n_boot, level=lvl)
            out["asr"] = {
                "table": table, "p": float(res.p), "n_items": len(items), "recipes": shared,
                "asr_a": asr_a, "asr_b": asr_b, "diff": asr_b - asr_a,
                "worst_a": float(a.mean()), "worst_b": float(b.mean()),
                "diff_ci": cluster_ci(groups, diff_stat, cell_key="boot.table-4.4.H3-asr-full-vs-content.diff", **kw),
                "a_ci": cluster_ci([g[g[:, 0] == 0, 1] for g in groups], lambda m: float(m.mean()),
                                   cell_key="boot.table-4.4.H3-asr-full-vs-content.content_only", **kw),
                "b_ci": cluster_ci([g[g[:, 0] == 1, 1] for g in groups], lambda m: float(m.mean()),
                                   cell_key="boot.table-4.4.H3-asr-full-vs-content.full", **kw),
            }
        per = []
        for x in shared:
            its = sorted(i for i in both if x in by[(i, "content_only")] and x in by[(i, "full")])
            if not its:
                continue
            a = np.asarray([by[(i, "content_only")][x] for i in its])
            b = np.asarray([by[(i, "full")][x] for i in its])
            table = [[int(np.sum(a & b)), int(np.sum(a & ~b))], [int(np.sum(~a & b)), int(np.sum(~a & ~b))]]
            per.append({"recipe": x, "table": table, "p": float(mcnemar_exact(table).p), "n_items": len(its),
                        "asr_a": float(a.mean()), "asr_b": float(b.mean())})
        if per:
            for p, adj in zip(per, holm([p["p"] for p in per])):
                p["p_holm"] = float(adj)
        out["per_recipe"] = per
    gated = [r for r in av if r["victim"] == "full" and r.get("posthoc_gate") is not None]
    if gated:
        order, by_item = [], {}
        for r in gated:
            if r["item_id"] not in by_item:
                by_item[r["item_id"]] = []
                order.append(r["item_id"])
            by_item[r["item_id"]].append(1.0 if r["posthoc_gate"] == "deflected" else 0.0)
        ind = [v for i in order for v in by_item[i]]
        out["deflection"] = {"rate": float(np.mean(ind)), "n_rows": len(ind), "n_items": len(order),
                             "ci": cluster_ci([by_item[i] for i in order], lambda m: float(m.mean()),
                                              cell_key="boot.table-4.4.H3-deflection", n_resamples=n_boot, level=lvl)}
    return out


def _h3_long_rows(ledger: list[dict], by_run: dict, meta: dict) -> list[dict]:
    res = h3_tests(ledger, meta)
    av = [r for r in ledger if r["subset"] == "averitec_chrono" and int(r["queries_used"]) > 0]
    prov, seeds = _prov_for(av or ledger, by_run)
    common = dict(**seeds, **prov)
    out = []
    asr = res["asr"]
    if asr:
        p_holm = holm([asr["p"]])[0]  # one p in the frozen R3 family (H3-deflection carries none): p_holm = p_raw
        labels = ", ".join(LABEL[x] for x in asr["recipes"])
        lo, hi = asr["diff_ci"]
        out.append(fmt.long_row(
            TABLE, f"{POOLED_ROW} | Full pipeline", COLS["asr"], "asr_pct", fmt.fmt_rate(asr["asr_b"]),
            **_rate_ci_cols(asr["b_ci"]), n_items=asr["n_items"], effect_size=fmt.fmt_pp(asr["diff"]),
            effect_size_kind="pooled ASR difference full minus content-only (pp)",
            p_raw=f"{asr['p']:.6f}", p_holm=f"{p_holm:.6f}",
            test=(f"exact McNemar over {asr['n_items']} AVeriTeC items attacked by both victims under every shared recipe "
                  f"({labels}); item evaded = strict success under at least one recipe (worst case over the battery); "
                  f"table [[both, content-only only], [full only, neither]] = {asr['table']}; worst-case evasion "
                  f"content-only {fmt.fmt_rate(asr['worst_a'])}, full {fmt.fmt_rate(asr['worst_b'])} percent; pooled ASR "
                  f"over the items' (item, recipe) rows; effect-size interval in the row '{COL_DIFF}'"),
            family=FAM_ASR, **common))
        out.append(fmt.long_row(
            TABLE, f"{POOLED_ROW} | Content-only encoder", COLS["asr"], "asr_pct", fmt.fmt_rate(asr["asr_a"]),
            **_rate_ci_cols(asr["a_ci"]), n_items=asr["n_items"],
            test="paired reference of H3-asr-full-vs-content (same items and recipes; item-cluster bootstrap)",
            family=FAM_ASR, **common))
        out.append(fmt.long_row(
            TABLE, f"{POOLED_ROW} | Full pipeline", COL_DIFF, "asr_diff_pp", fmt.fmt_pp(asr["diff"]),
            ci_lo=fmt.fmt_pp(lo), ci_hi=fmt.fmt_pp(hi), n_items=asr["n_items"],
            test=(f"effect size of H3-asr-full-vs-content: item-cluster percentile bootstrap ({meta['n_resamples']} "
                  "resamples; items drawn with replacement, every (item, recipe) row of a drawn item kept; cell key "
                  "boot.table-4.4.H3-asr-full-vs-content.diff)"),
            family=FAM_ASR, **common))
    for p in res["per_recipe"]:
        out.append(fmt.long_row(
            TABLE, f"{LABEL[p['recipe']]} | Full pipeline", COL_RECIPE, "asr_pct", fmt.fmt_rate(p["asr_b"]),
            n_items=p["n_items"], p_raw=f"{p['p']:.6f}", p_holm=f"{p['p_holm']:.6f}", effect_size=fmt.fmt_pp(p["asr_b"] - p["asr_a"]),
            effect_size_kind="ASR difference full minus content-only on the paired items (pp)",
            test=(f"exact McNemar over {p['n_items']} AVeriTeC items attacked by both victims under {LABEL[p['recipe']]}; "
                  f"table {p['table']}; Holm across the {len(res['per_recipe'])} recipes that ran (supplementary, not a "
                  "configs/stats.yaml family)"),
            family=FAM_RECIPE, **common))
    d = res["deflection"]
    if d:
        out.append(fmt.long_row(
            TABLE, f"{POOLED_ROW} | Full pipeline", COLS["deflection_rate"], "deflection_rate_pct",
            fmt.fmt_rate(d["rate"]), **_rate_ci_cols(d["ci"]), n_items=d["n_rows"],
            test=(f"item-cluster percentile bootstrap over attacked gated victim-B rows ({d['n_items']} items); "
                  "H3-deflection holds when the CI excludes 0 (no p: one-sided rate); read beside the deflection "
                  "references (a)-(e), family R3-supplementary"),
            family=FAM_DEFL, **common))
    return out


# --------------------------------------------------------------------------- denominator categories 

def _planned(run, subset: str, victim: str) -> int | None:
    """Planned positive items of one (subset, victim) cell of an R3 run: run-record.json items
 `<subset>_<victim>_items` (written by the R3 regime from the run's target items); None when not recorded."""
    rec = json.loads((run.run_dir / "run-record.json").read_text(encoding="utf-8"))
    v = (rec.get("items") or {}).get(f"{subset}_{victim}_items")
    return None if v is None else int(v)


def _categories(rows: list[dict], by_run: dict, victim: str) -> dict:
    """Counts of one scope: ledger rows, skipped (queries_used 0: the victim's clean decision was already wrong),
 attacked, strict successes, budget-exhausted, deferred, successes in store (strict + deferred) and timed-out =
 planned items without a ledger row (the session deadline ended before the item finished; R47). Planned sums the
 run-record count of every (run, subset, attack) cell the scope's rows come from, so it never needs a text."""
    att = atk.attacked(rows)
    out = {"ledger": len(rows), "attacked": len(att), "skipped": len(rows) - len(att),
           "success": sum(r["end_state"] == "success" for r in att),
           "budget": sum(r["end_state"] == "budget_exhausted" for r in att),
           "deferred": sum(r["end_state"] == "deferred" for r in att)}
    out["in_store"] = out["success"] + out["deferred"]
    cells = sorted({(r["_run_id"], r["subset"], r["attack"]) for r in rows})
    planned = [_planned(by_run[rid], subset, victim) for rid, subset, _ in cells]
    out["planned"] = None if any(p is None for p in planned) else sum(planned)
    out["timed_out"] = None if out["planned"] is None else out["planned"] - out["ledger"]
    return out


def _count_rows(ident: str, victim: str, rows: list[dict], by_run: dict, common: dict, subsets: list[str]) -> list[dict]:
    """Per scope (pooled, each subset, each victim-B half): the denominator categories and the alternative ASR
 reading (strict successes / (attacked - budget-exhausted)), so either reading is computable from the long CSV."""
    scopes = [("", rows, "pooled over subsets " + ", ".join(subsets))]
    for subset in subsets:
        sub = [r for r in rows if r["subset"] == subset]
        scopes.append((f" ({subset})", sub, f"subset {subset}"))
        for half in sorted({r["_half"] for r in sub if r.get("_half")}):
            scopes.append((f" ({subset}, {half})", [r for r in sub if r.get("_half") == half], f"subset {subset}, half {half}"))
    out = []
    for suffix, sr, what in scopes:
        c = _categories(sr, by_run, victim)
        src = ";".join(sorted({r["_run_id"] for r in sr}))
        for key, col, metric in COUNT_COLS:
            if key in ("deferred", "in_store") and victim != "full":
                continue
            if key == "timed_out":
                value = "n/a (planned items not recorded in run-record.json)" if c["timed_out"] is None else fmt.fmt_count(c["timed_out"])
                test = (f"{what}; planned items (run-record items <subset>_<victim>_items of {src}) "
                        f"{c['planned'] if c['planned'] is not None else 'n/a'} minus ledger rows {c['ledger']}; a "
                        "timed-out item has no ledger row and is outside the ASR denominator (R47, wall-clock deadline only)")
            else:
                value = fmt.fmt_count(c[key])
                test = f"{what}; ledger end states of {src}"
                if key == "budget":
                    test += ("; budget-exhausted items stay in the ASR denominator as non-successes (specification item 13; "
                             f"{OPERATOR_R47_SOURCE}: \"{OPERATOR_R47}\")")
                if key == "skipped":
                    test += "; skipped = queries_used 0 (the victim's clean decision was already wrong), same rule on every subset and victim"
            out.append(fmt.long_row(TABLE, ident, col + suffix, metric, value, n_items=c["ledger"], test=test, **common))
        den = c["attacked"] - c["budget"]
        alt = "n/a (no attacked item outside the budget-exhausted category)" if den <= 0 else fmt.fmt_rate(c["success"] / den)
        out.append(fmt.long_row(TABLE, ident, COL_ALT + suffix, "asr_excl_budget_pct", alt, n_items=max(den, 0),
                                test=(f"{what}; {c['success']} / ({c['attacked']} - {c['budget']}); the R47 reading applied "
                                      "to budget exhaustion, reported for comparison only: the shell ASR keeps "
                                      f"budget-exhausted items in the denominator ({OPERATOR_R47_SOURCE})"), **common))
    return out


def _half_rows(ident: str, victim: str, rows: list[dict], common: dict, key0: str, meta: dict) -> list[dict]:
    """Victim-B rows of halved runs: ASR and deflection per half (subset label), cell key <key0>.<subset>.<half>."""
    out = []
    for subset in sorted({r["subset"] for r in rows if r.get("_half")}):
        for half in sorted({r["_half"] for r in rows if r["subset"] == subset and r.get("_half")}):
            hr = [r for r in rows if r["subset"] == subset and r.get("_half") == half]
            c = _cells(hr, victim, cell_key=f"{key0}.{subset}.{half}", meta=meta)
            rd = _render(c, victim)
            note = (f"subset {subset}, half {half} of averitec-attack-subset-v1 ({';'.join(sorted({r['_run_id'] for r in hr}))}); "
                    f"n_correct_before {c['n_attacked']} of {len(hr)} positives; 95% item-bootstrap CI ({meta['n_resamples']} "
                    f"resamples, cell key {key0}.{subset}.{half}.<column>)")
            out.append(fmt.long_row(TABLE, ident, f"{COLS['asr']} ({subset}, {half})", "asr_pct", rd["asr"],
                                    n_items=c["n_attacked"], test=note, **_rate_ci_cols(c["asr_ci"]), **common))
            if victim == "full":
                out.append(fmt.long_row(TABLE, ident, f"{COLS['deflection_rate']} ({subset}, {half})", "deflection_rate_pct",
                                        rd["deflection_rate"], n_items=c["n_attacked"], test=note,
                                        **_rate_ci_cols(c["deflection_ci"]), **common))
    return out


def r46_counts(root: Path) -> dict[tuple, tuple[dict, str]]:
    """{(run_id, attack, victim, subset): (counts over R46_KEYS, source file)} from the post-hoc R46 counts recorded in
 data/measured (computed on the host holding each private store; counts only). Two layouts: post_hoc_counts.runs.
 <run id>.cells.<attack|victim|subset>.token_position (every key present), and the cpu-host session's
 post_hoc_counts.<config name> = {run_id, r46: {cells: {<attack>: ...}, absent_key_means_zero: true}} (victim A, rd)."""
    out = {}
    for rel in R46_SOURCES:
        path = Path(root) / rel
        if not path.is_file():
            continue
        phc = json.loads(path.read_text(encoding="utf-8")).get("post_hoc_counts") or {}
        for rid, run in (phc.get("runs") or {}).items():
            for cell, v in (run.get("cells") or {}).items():
                attack, victim, subset = cell.split("|")
                tp = v.get("token_position", v)
                out[(rid, attack, victim, subset)] = ({k: int(tp[k]) for k in R46_KEYS}, str(rel))
        for name, v in phc.items():
            if not isinstance(v, dict) or "r46" not in v or "run_id" not in v:
                continue
            r46 = v["r46"]
            if not r46.get("absent_key_means_zero"):
                raise ValueError(f"{rel}: post_hoc_counts.{name}.r46 does not declare absent_key_means_zero")
            for attack, cnt in (r46.get("cells") or {}).items():
                out[(v["run_id"], attack, "content_only", "rd")] = ({k: int(cnt.get(k, 0)) for k in R46_KEYS}, str(rel))
    return out


def _r46_rows(root: Path, ledger: list[dict], by_run: dict) -> list[dict]:
    """R46 (analyst 2026-10-18): stored successes whose perturbation lies entirely / partly beyond token 256 and 128
 (C1 max_length) per (attack, victim, subset), summed over the cell's runs; `not counted` names the run without a
 recorded count. Family R3-supplementary; counts come from data/measured (r46_counts), not from a ledger."""
    counts = r46_counts(root)
    out = []
    for attack_label, attack in ATTACKS:
        for victim_label, victim in VICTIMS:
            ident = f"{attack_label} | {victim_label}"
            for subset in SUBSETS:
                rows = [r for r in ledger if r["attack"] == attack and r["victim"] == victim and r["subset"] == subset]
                if not rows:
                    continue
                rids = sorted({r["_run_id"] for r in rows})
                prov, seeds = _prov_for(rows, by_run)
                got = [counts.get((rid, attack, victim, subset)) for rid in rids]
                missing = [rid for rid, g in zip(rids, got) if g is None]
                srcs = sorted({g[1] for g in got if g})
                for k in R46_KEYS:
                    if missing:
                        value = f"not counted (no post-hoc R46 count for {';'.join(missing)} in data/measured)"
                    else:
                        value = fmt.fmt_count(sum(g[0][k] for g in got))
                    out.append(fmt.long_row(
                        TABLE, ident, f"{R46_COL}, {k} ({subset})", f"r46_{k}_n", value,
                        n_items=fmt.NA if missing else sum(g[0]["successes_in_store"] for g in got),
                        test=(f"post-hoc count over the private-store successes (strict success + deferred) of {';'.join(rids)}, "
                              f"run's C1 tokenizer, special tokens, untruncated; entirely beyond T iff the common prefix "
                              f">= T, partly iff prefix < T < changed end; source {', '.join(srcs) or 'none'}"),
                        family=FAM_SUPP, **seeds, **prov))
    return out


def _notes(ledger: list[dict], by_run: dict, bs_ids: str, meta: dict) -> list[str]:
    """Shell and long-CSV note lines of the real build , each computed from the rows it describes."""
    att = atk.attacked(ledger)
    timed = [_categories([r for r in ledger if (r["attack"], r["victim"]) == av], by_run, av[1])["timed_out"]
             for av in sorted({(r["attack"], r["victim"]) for r in ledger})]
    t_total = "n/a" if any(t is None for t in timed) else str(sum(timed))
    halves = sorted({r["_half"] for r in ledger if r["victim"] == "full" and r.get("_half")})
    b_rows = [r for r in ledger if r["victim"] == "full"]
    fired = [st["id"] for st in (fmt.load_cuts().get("steps") or []) if st.get("active") and st.get("fired_on")]
    return [
        ("note: ASR = strict successes / attacked rows (specification item 13); attacked = ledger rows with queries_used > 0; "
         "skipped rows (the victim's clean decision already wrong, 0 queries) are outside the denominator on every subset "
         f"and victim; budget-exhausted items ({sum(r['end_state'] == 'budget_exhausted' for r in att)} attacked rows over "
         "the table) stay in the denominator as non-successes and have their own long rows (budget_exhausted_n); the "
         "alternative reading strict successes / (attacked - budget-exhausted) is the long metric asr_excl_budget_pct, "
         f"not the shell value. Source: {OPERATOR_R47_SOURCE}, verbatim: \"{OPERATOR_R47}\""),
        (f"note: timed-out items (wall-clock session deadline reached before the item finished) are a separate category "
         f"outside the ASR denominator (R47); total over all cells: {t_total} (long metric timed_out_n)"),
        (f"note: intervals are 95 percent item percentile bootstraps ({meta['n_resamples']} resamples, rng = "
         "derive_seed(cell key)); cell key boot.table-4.4.<Attack> | <Victim>.<column> for a shell cell, "
         ".<subset>.<column> per subset and .<subset>.<half>.<column> per victim-B half; items are resampled within strata "
         "(the subset, and for a victim-B row of a halved run subset|half), strata drawn in sorted label order "
         "(averitec_chrono|first200 then averitec_chrono|last200) under one rng per cell"),
        (f"note: Full pipeline rows cover the AVeriTeC attack subset halves {', '.join(halves) if halves else 'none'} "
         f"({len(b_rows)} victim-B ledger rows per table, {len(atk.attacked(b_rows))} attacked); Content-only rows add FC "
         "and RD (per-subset long rows)"),
        (f"note: BODEGA score from {bs_ids or 'no sealed BODEGA scoring run'}; cuts fired: "
         f"{', '.join(fired) if fired else 'none'}; the cut-4-drop-rd blank of the RD long rows is not wired in this "
         "generator (no cut has fired; wiring it is required before a rebuild after cut-4)"),
    ]


# --------------------------------------------------------------------------- supplementary (real profiles)

def _flip_rows(ledger: list[dict], by_run: dict, meta: dict) -> list[dict]:
    rates = _ungated(ledger)
    out = []
    for rank, (a, v) in enumerate(rates.items(), 1):
        att = [r for r in ledger if r["victim"] == "full" and r["attack"] == a and int(r["queries_used"]) > 0]
        prov, seeds = _prov_for(att, by_run)
        ident = f"{LABEL[a]} | Full pipeline"
        ci = _ci([r["end_state"] in ("success", "deferred") for r in att], [_stratum(r) for r in att],
                 f"boot.table-4.4.{ident}.ungated_flip", meta)
        out.append(fmt.long_row(
            TABLE, ident, COL_FLIP, "ungated_flip_rate_pct", fmt.fmt_rate_ci(v, *ci), **_rate_ci_cols(ci), n_items=len(att),
            test=(f"addition-A recipe rule input: (strict success + deferred) / attacked victim-B rows; rank {rank} of "
                  f"{len(rates)} (ties in Table 4.4 row order)"),
            family=FAM_SUPP, **seeds, **prov))
    return out


def _reference_rows(root: Path, ledger: list[dict], by_run: dict, meta: dict) -> list[dict]:
    """Deflection references (a)-(e) over the attacked gated victim-B rows (aac.reports.deflection)."""
    b_rows = [r for r in deflection.attacked_b(ledger) if r["subset"] == "averitec_chrono"]
    if not b_rows:
        return []
    b_runs = [by_run[rid] for rid in sorted({r["_run_id"] for r in b_rows}, key=list(by_run).index)]
    vs = {(r.resolved.get("attack") or {}).get("victim_source") for r in b_runs}
    seeds_ = {r.seed for r in b_runs}
    prov, seeds = _prov_for(b_rows, by_run)
    row = f"{POOLED_ROW} | Full pipeline"
    if vs == {None}:
        return [fmt.long_row(TABLE, row, "deflection references (a)-(e)", "deflection_refs", fmt.NA,
                             test="n/a (the victim-B runs name no attack.victim_source)", family=FAM_SUPP, **seeds, **prov)]
    if len(vs) != 1 or None in vs or len(seeds_) != 1:
        raise ValueError(f"Table 4.4: victim-B runs disagree on attack.victim_source / seed: {sorted(map(str, vs))}, {sorted(seeds_)}")
    source, seed = vs.pop(), seeds_.pop()
    entries = [e for e in sources.index_runs(root) if e.get("name") == source and int(e["seed"]) == seed and e["regime"] == "R1"]
    if len(entries) != 1:
        raise sources.MissingRun(f"Table 4.4 deflection references need the sealed R1 run {source!r} seed {seed} "
                                 f"(attack.victim_source); found {len(entries)}")
    r1 = phase4_sources.load_run_light(root, entries[0])
    clean = phase4_sources.read_split(r1, "test")
    calibration = phase4_sources.read_json(r1, "calibration.json")
    gate = phase4_sources.read_json(r1, "gate.json")
    posthoc = {}
    for run in b_runs:
        ids = [deflection.posthoc_record_id(r) for r in b_rows if r["_run_id"] == run.run_id]
        posthoc.update(phase4_sources.read_records(run, [i for i in ids if i]))
    out = _refs_long(b_rows, by_run, r1, posthoc, clean, calibration, gate, meta, row, "boot.table-4.4.deflection-refs",
                     b_runs, source, seed, full=True)
    # : (a), (d) and (e) per recipe, the inputs of the deflection-controls figure (aac.reports.fig_deflection)
    for _, a_ in ATTACKS:
        mine = [r for r in b_rows if r["attack"] == a_]
        if mine:
            out.extend(_refs_long(mine, by_run, r1, posthoc, clean, calibration, gate, meta, f"{LABEL[a_]} | Full pipeline",
                                  f"boot.table-4.4.deflection-refs.{a_}", b_runs, source, seed, full=False))
    return out


def _refs_long(b_rows, by_run, r1, posthoc, clean, calibration, gate, meta, row, cell_key, b_runs, source, seed, *,
               full: bool) -> list[dict]:
    """Long rows of the deflection references over `b_rows`: all five on the pooled row (full=True; the rows, bytes unchanged), (a), (d) and (e) on a per-recipe row (full=False)."""
    refs = deflection.deflection_references(b_rows, posthoc, clean, calibration, gate, cell_key=cell_key,
                                            n_resamples=meta["n_resamples"], level=meta["level"])
    prov, seeds = _prov_for(b_rows, by_run, extra=[r1])
    src = (f"post-hoc records of {';'.join(r.run_id for r in b_runs if any(x['_run_id'] == r.run_id for x in b_rows))}; "
           f"clean test records, calibration.json and gate.json "
           f"of {r1.run_id} ({source}, attack.victim_source at seed {seed}); gate threshold {refs['gate_threshold']}")
    common = dict(family=FAM_SUPP, **seeds, **prov)
    out = []
    a = refs["a"]
    out.append(fmt.long_row(TABLE, row, "deflection reference (a): deflection minus clean deferral, same items (pp)",
                            "deflection_ref_a_pp", fmt.fmt_pp(a["value"]), ci_lo=fmt.fmt_pp(a["ci"][0]), ci_hi=fmt.fmt_pp(a["ci"][1]),
                            n_items=a["n_rows"],
                            test=(f"{a['rule']}; deflection {fmt.fmt_rate(a['deflection'])} vs clean deferral "
                                  f"{fmt.fmt_rate(a['clean_deferral'])} percent over {a['n_rows']} rows ({a['n_items']} items); {src}"),
                            **common))
    for key, metric, col in (("b", "deflection_ref_b_pct", "deflection reference (b): deflection on clean-auto-decided items (percent)"),
                             ("d", "deflection_ref_d_pct", "deflection reference (d): conditional deflection, deflected / (deflected + strict success) (percent)")):
        if key == "b" and not full:
            continue
        x = refs[key]
        value = x["why"] if x["value"] is None else fmt.fmt_rate_ci(x["value"], *x["ci"])
        extra = f"; {x['n_deflected']} deflected, {x['n_success']} strict successes" if key == "d" else ""
        out.append(fmt.long_row(TABLE, row, col, metric, value, **_rate_ci_cols(x["ci"]), n_items=x["n_rows"],
                                test=f"{x['rule']}{extra}; {src}", **common))
    c = refs["c"]
    for reason in (deflection.REASONS if full else ()):
        n = c["counts"][reason]
        out.append(fmt.long_row(TABLE, row, f"deflection reference (c): deflected rows by abstention reason, {reason}",
                                f"deflection_ref_c_{reason}", fmt.fmt_count(n), n_items=c["n_deflected"],
                                effect_size=fmt.fmt_rate(n / c["n_deflected"]) if c["n_deflected"] else fmt.NA,
                                effect_size_kind="share of the deflected rows (percent)",
                                test=f"{c['rule']}; {src}", **common))
    e = refs["e"]
    value = e["why"] if e["value"] is None else fmt.fmt_pp(e["value"])
    ekw = {} if e["ci"] is None else {"ci_lo": fmt.fmt_pp(e["ci"][0]), "ci_hi": fmt.fmt_pp(e["ci"][1])}
    detail = "" if e["value"] is None else (f"; observed {fmt.fmt_rate(e['observed'])} vs matched clean deferral "
                                            f"{fmt.fmt_rate(e['matched_clean_deferral'])} percent")
    out.append(fmt.long_row(TABLE, row, "deflection reference (e): matched-confidence control, observed minus matched clean deferral (pp)",
                            "deflection_ref_e_pp", value, **ekw, n_items=e["n_rows"],
                            test=f"{e['rule']}{detail}; {e['n_unmatched']} finals without a matched clean record; {src}", **common))
    return out


# --------------------------------------------------------------------------- additions A and B 

def _now() -> datetime:
    """The wall clock in the execution schedule timezone (configs/cuts.yaml `timezone`); tests replace it."""
    return datetime.now(ZoneInfo(CUT_TZ))


def cut_text(step: str, date: str) -> str:
    """The cut-cell text of configs/cuts.yaml (`cell_text`) for an addition step that is not a cuts.yaml step."""
    return str(fmt.load_cuts().get("cell_text") or "not run (cut: {step} on {date})").format(step=step, date=date)


def addition_cut(step: str) -> str | None:
    """`not run (cut: <step> on <date>)` once the addition's own cut rule has passed (ADDITION_CUTS), else None."""
    spec = ADDITION_CUTS[step]
    fires = datetime.fromisoformat(spec["fires_at"]).replace(tzinfo=ZoneInfo(CUT_TZ))
    return cut_text(step, spec["date"]) if _now() >= fires else None


def absent_addition(name: str, step: str | None, *, table: str, allow_pending: bool, what: str) -> tuple[str, str]:
    """(cell text, test note) for a named addition run that is not indexed: the cut text when its rule has passed, a
 pending cell under allow_pending, else MissingRun naming the run (never a silent blank)."""
    text = addition_cut(step) if step else None
    if text:
        return text, f"{what}: run {name} not in runs/INDEX.json; {ADDITION_CUTS[step]['rule']}"
    if allow_pending:
        return fmt.PENDING("Phase 4"), f"{what}: run {name} not in runs/INDEX.json yet (configs/reports/phase4-runs.yaml)"
    rule = f"; its cut rule has not passed ({ADDITION_CUTS[step]['rule']})" if step else ""
    raise sources.MissingRun(f"table {table}: {what} needs the sealed run {name!r} in runs/INDEX.json{rule}")


def supplementary_profile(root: Path, profile: str) -> dict[str, str]:
    """{key: run name} of the additions the profile names (SUPP_KEYS): gate_aware, transfer, gate_only, and, when
 gate_only is named, gate_only_b = the profile's R4 minus-retrieval replay (victim B's finals). Stub: {}."""
    if profile == phase4_sources.STUB_PROFILE:
        return {}
    prof = phase4_sources.load_profile(root, profile)
    sup = {k: v for k, v in prof["supplementary"].items() if k in ("gate_aware", "transfer", "gate_only") and v}
    replay_b = (prof["r4"].get("no-retrieval") or {}).get("replay")
    if "gate_only" in sup and replay_b:
        sup["gate_only_b"] = replay_b
    return sup


def _by_item(rows: list[dict], value) -> list[list[float]]:
    order, by = [], {}
    for r in rows:
        if r["item_id"] not in by:
            by[r["item_id"]] = []
            order.append(r["item_id"])
        by[r["item_id"]].append(float(value(r)))
    return [by[i] for i in order]


def _rate(rows: list[dict], value, key: str, meta: dict) -> tuple[float, tuple[float, float]]:
    """(mean of value over rows, item-cluster percentile bootstrap interval; every row of an item kept together)."""
    groups = _by_item(rows, value)
    point = float(np.mean([v for g in groups for v in g]))
    ci = cluster_ci(groups, lambda m: float(np.mean(m)), cell_key=key, n_resamples=meta["n_resamples"], level=meta["level"])
    return point, ci


def _supplementary_rows(root: Path, profile: str, ledger: list[dict], by_run: dict, meta: dict, allow_pending: bool,
                        data_version: str, *, scores: dict | None = None, bs_ids: str = "") -> list[dict]:
    sup = supplementary_profile(root, profile)
    out = []
    if "gate_aware" in sup:
        out.extend(_gate_aware_rows(root, sup["gate_aware"], ledger, by_run, meta, allow_pending, data_version,
                                    scores=scores, bs_ids=bs_ids))
    for key in ("transfer", "gate_only", "gate_only_b"):
        if key in sup:
            out.extend(_replay_rows(root, key, sup[key], ledger, by_run, meta, allow_pending, data_version))
    return out


def _absent_rows(row: str, cols, name: str, step: str | None, allow_pending: bool, what: str,
                 data_version: str) -> list[dict]:
    cell, note = absent_addition(name, step, table=TABLE, allow_pending=allow_pending, what=what)
    return [fmt.long_row(TABLE, row, col, metric, cell, config_hash="", data_version=data_version, run_ids="n/a",
                         source_shards_sha256="n/a", test=note, family=FAM_SUPP) for col, metric in cols]


def _gate_aware_rows(root: Path, name: str, ledger: list[dict], by_run: dict, meta: dict, allow_pending: bool,
                     data_version: str, *, scores: dict | None = None, bs_ids: str = "") -> list[dict]:
    """Addition A: the gate-aware run's per-recipe rows, and the paired exact McNemar against the gate-unaware union."""
    cols = ((COL_N, "attacked_n"), (COLS["asr"], "asr_pct"), (COLS["deflection_rate"], "deflection_rate_pct"),
            (COLS["queries_per_success_median"], "queries_per_success_median"), (COL_BUDGET, "budget_exhausted_share_pct"),
            (COL_MCN, "mcnemar_gate_aware_vs_unaware"))
    what = "addition A (gate-aware attack)"
    if not phase4_sources.indexed(root, [name], regime=REGIME):
        return _absent_rows("Gate-aware attack (addition A) | Full pipeline", cols, name, "addition-A", allow_pending,
                            what, data_version)
    rows, runs = phase4_sources.union_ledgers(root, [name], table=TABLE)
    for run in runs:
        if (run.resolved.get("attack") or {}).get("victim_view") != GATED_MASKED:
            raise ValueError(f"Table 4.4: supplementary gate_aware run {run.run_id} ({name}) is not a gated-masked R3 run "
                             "(attack.victim_view)")
    by = {r.run_id: r for r in runs}
    out = []
    av = [r for r in rows if r["victim"] == "full" and r["subset"] == "averitec_chrono"]
    unaware = {(r["item_id"], r["attack"]): r for r in ledger
               if r["victim"] == "full" and r["subset"] == "averitec_chrono" and int(r["queries_used"]) > 0}
    for _, a in ATTACKS:
        mine = [r for r in av if r["attack"] == a]
        if not mine:
            continue
        ident = f"Gate-aware attack (addition A), {LABEL[a]} | Full pipeline"
        prov, seeds = _prov_for(mine, by)
        L = lambda col, metric, value, **kw: out.append(fmt.long_row(TABLE, ident, col, metric, value, family=FAM_SUPP,
                                                                     **{**seeds, **prov, **kw}))
        att = atk.attacked(mine)
        key = f"boot.table-4.4.{ident}"
        L(COL_N, "attacked_n", fmt.fmt_count(len(att)), n_items=len(att),
          test=f"positive items attacked through the gated-masked view of victim B ({len(mine)} ledger rows; specification item 12 output)")
        if not att:
            for col, metric in cols[1:]:
                L(col, metric, NO_ATTACKED, n_items=0, test="no attacked item")
            continue
        succ = [r for r in att if r["end_state"] == "success"]
        asr, asr_ci = _rate(att, lambda r: r["end_state"] == "success", f"{key}.asr", meta)
        L(COLS["asr"], "asr_pct", fmt.fmt_rate_ci(asr, *asr_ci), n_items=len(att), **_rate_ci_cols(asr_ci),
          test="strict success (an auto-decided wrong final; no post-hoc query) / attacked rows; item-cluster bootstrap "
               f"({meta['n_resamples']} resamples, cell key {key}.asr)")
        if scores is not None:  # : the gate-aware run is scored in the same BS run (bs-gpu-host)
            k5 = lambda r: (r["_run_id"], *phase4_sources.ledger_key(r))  # noqa: E731
            have = [k5(r) in scores for r in att]
            if bs_ids and all(have):
                b, b_ci = _rate(att, lambda r: float(scores[k5(r)]["bodega"]), f"{key}.bodega", meta)
                L(COLS["bodega_score"], "bodega_score_mean", fmt.fmt_ci(b, *b_ci), n_items=len(att), **_metric_ci_cols(b_ci),
                  test=(f"mean over attacked rows of strict success x semantic x character from {bs_ids} (score rows keyed "
                        f"by source run {';'.join(sorted({r['_run_id'] for r in att}))}); item-cluster bootstrap (cell key "
                        f"{key}.bodega)"))
            else:
                why = NO_BS if not bs_ids else f"bodega_score: {have.count(False)} of {len(att)} attacked rows unscored in {bs_ids}"
                L(COLS["bodega_score"], why, fmt.NA, n_items=len(att), test=why)
        gated = [r for r in att if r.get("posthoc_gate") is not None]
        if gated:
            d, d_ci = _rate(gated, lambda r: r["posthoc_gate"] == "deflected", f"{key}.deflection", meta)
            L(COLS["deflection_rate"], "deflection_rate_pct", fmt.fmt_rate_ci(d, *d_ci), n_items=len(gated), **_rate_ci_cols(d_ci),
              test="deflected (an attack query was a deferred un-gated flip, s < 0.5, and no strict success occurred) / "
                   f"attacked rows; item-cluster bootstrap (cell key {key}.deflection)")
        else:
            L(COLS["deflection_rate"], "deflection_rate_pct", "n/a (no gated rows)", n_items=0, test="no row carries posthoc_gate")
        q = atk.queries_per_success_median(mine)
        L(COLS["queries_per_success_median"], "queries_per_success_median",
          "n/a (no success)" if q is None else fmt.fmt_count(q), n_items=len(succ), test="median queries_used over success rows")
        share = sum(r["end_state"] == "budget_exhausted" for r in att) / len(att)
        L(COL_BUDGET, "budget_exhausted_share_pct", fmt.fmt_rate(share), n_items=len(att),
          test=f"budget_exhausted rows / attacked rows; budget {sorted({int(r['budget']) for r in att})}")
        paired = [(r, unaware[(r["item_id"], a)]) for r in att if (r["item_id"], a) in unaware]
        if not paired:
            L(COL_MCN, "mcnemar_gate_aware_vs_unaware", "n/a (no item attacked by both)", n_items=0,
              test=f"no {LABEL[a]} victim-B row of the gate-unaware runs shares an attacked item with {name}")
            continue
        g = np.asarray([x["end_state"] == "success" for x, _ in paired])
        u = np.asarray([y["end_state"] == "success" for _, y in paired])
        table = [[int(np.sum(u & g)), int(np.sum(u & ~g))], [int(np.sum(~u & g)), int(np.sum(~u & ~g))]]
        res = mcnemar_exact(table)
        src_ids = sorted({y["_run_id"] for _, y in paired})
        both = sources.provenance(runs + [by_run[i] for i in src_ids])  # the gate-aware run and the paired gate-unaware runs
        L(COL_MCN, "mcnemar_gate_aware_vs_unaware", fmt.fmt_rate(float(g.mean())), n_items=len(paired),
          p_raw=f"{res.p:.6f}", effect_size=fmt.fmt_pp(float(g.mean() - u.mean())),
          effect_size_kind="strict ASR difference gate-aware minus gate-unaware on the paired items (pp)",
          test=(f"exact McNemar of strict success over {len(paired)} AVeriTeC items attacked by both the gate-aware run and "
                f"the gate-unaware {LABEL[a]} victim-B rows ({';'.join(src_ids)}); table [[both, gate-unaware only], "
                f"[gate-aware only, neither]] = {table}; value = gate-aware strict ASR on the paired items, gate-unaware "
                f"{fmt.fmt_rate(float(u.mean()))} percent; supplementary, outside the H3 Holm family"), **both)
    return out


def _finished_date(run) -> str:
    rec = json.loads((run.run_dir / "run-record.json").read_text(encoding="utf-8"))
    when = str(rec.get("finished_utc") or "")
    if len(when) < 10:
        raise ValueError(f"{run.run_id}: run-record.json has no finished_utc for the addition-B store cut date")
    return when[:10]


def _replay_rows(root: Path, key: str, name: str, ledger: list[dict], by_run: dict, meta: dict, allow_pending: bool,
                 data_version: str) -> list[dict]:
    """Addition B: a seed-0 store replay (regime R4) per recipe that ran in the main table for the replayed victim, and
 pooled. Denominator = replay rows with queries_used 1 (methodological note)."""
    victim, suffix, what = REPLAY_VIEWS[key]
    cols = ((COL_TRANSFER, "transfer_strict_success_pct"), (COL_A_SAME, "victim_a_asr_same_rows_pct")) if key == "transfer" \
        else ((COL_GO_DEFL, "gate_only_deflection_pct"), (COL_GO_ASR, "gate_only_asr_pct"))
    if not phase4_sources.indexed(root, [name], regime="R4"):
        return _absent_rows(f"all attacks (pooled, AVeriTeC subset) | {suffix}", cols, name, None, allow_pending,
                            f"addition B ({what})", data_version)
    rows, runs = phase4_sources.union_ledgers(root, [name], table=TABLE, regime="R4")
    by = {r.run_id: r for r in runs}
    rows = [r for r in rows if r["victim"] == victim and r["subset"] == "averitec_chrono"]
    ran = [a for _, a in ATTACKS if any(r["victim"] == victim and r["attack"] == a and r["subset"] == "averitec_chrono"
                                        and int(r["queries_used"]) > 0 for r in ledger)]
    main = {(r["item_id"], r["attack"]): r for r in ledger if r["victim"] == victim and r["subset"] == "averitec_chrono"}
    out = []
    den_all = []
    for a in ran + ["pooled"]:
        if a == "pooled":
            den = den_all
            ident = f"all attacks (pooled, AVeriTeC subset) | {suffix}"
        else:
            mine = [r for r in rows if r["attack"] == a]
            ident = f"{LABEL[a]} | {suffix}"
            if not mine:
                date = _finished_date(runs[0])
                cell = cut_text(STORE_CUT, date)
                for col, metric in cols:
                    out.append(fmt.long_row(TABLE, ident, col, metric, cell, config_hash="", data_version=data_version,
                                            run_ids="n/a", source_shards_sha256="n/a", family=FAM_SUPP,
                                            test=(f"{LABEL[a]} ran for this victim in the main table but has no row in the "
                                                  f"sealed replay {';'.join(by)} (its source left replay.sources: private "
                                                  f"store incomplete); date = the replay's finished_utc")))
                continue
            den = [r for r in mine if int(r["queries_used"]) == 1]
            den_all.extend(den)
        if den:
            prov, seeds = _prov_for(den, by)
        else:
            prov = sources.provenance(runs)
            seeds = dict(n_seeds=len({r.seed for r in runs}), seeds=", ".join(str(x) for x in sorted({r.seed for r in runs})))
        L = lambda col, metric, value, **kw: out.append(fmt.long_row(TABLE, ident, col, metric, value, family=FAM_SUPP,
                                                                     **{**seeds, **prov, **kw}))
        ck = f"boot.table-4.4.{ident}"
        if not den:
            for col, metric in cols:
                L(col, metric, "n/a (empty replay denominator)", n_items=0,
                  test="no replayed row whose ablated clean decision keeps the label (and is auto-decided)")
            continue
        basis = (f"{what}; denominator = replay rows with queries_used 1 (the replayed system's clean decision keeps the "
                 f"label and is auto-decided); item-cluster bootstrap ({meta['n_resamples']} resamples")
        if key == "transfer":
            v, ci = _rate(den, lambda r: r["end_state"] == "success", f"{ck}.transfer", meta)
            L(COL_TRANSFER, "transfer_strict_success_pct", fmt.fmt_rate_ci(v, *ci), n_items=len(den), **_rate_ci_cols(ci),
              test=basis + f", cell key {ck}.transfer); strict success = an auto-decided wrong decision on the final")
            src = [main.get((r["item_id"], r["attack"])) for r in den]
            if any(s is None for s in src):
                raise ValueError(f"Table 4.4: {name} replays a victim-A row that the main table's union does not hold")
            pairs = list(zip(den, src))
            va, va_ci = _rate([s for _, s in pairs], lambda r: r["end_state"] == "success", f"{ck}.victim_a", meta)
            a_prov = sources.provenance([by_run[i] for i in sorted({s["_run_id"] for s in src}, key=list(by_run).index)])
            L(COL_A_SAME, "victim_a_asr_same_rows_pct", fmt.fmt_rate_ci(va, *va_ci), n_items=len(den), **_rate_ci_cols(va_ci),
              **a_prov,
              test=("victim A's own strict ASR over the same (item, recipe) rows of the main table's union "
                    f"({';'.join(sorted({s['_run_id'] for s in src}))}); item-cluster bootstrap (cell key {ck}.victim_a)"))
        else:
            v, ci = _rate(den, lambda r: r["end_state"] == "deferred", f"{ck}.deflection", meta)
            L(COL_GO_DEFL, "gate_only_deflection_pct", fmt.fmt_rate_ci(v, *ci), n_items=len(den), **_rate_ci_cols(ci),
              test=basis + f", cell key {ck}.deflection); deflection = the gated view deferred the final")
            s, s_ci = _rate(den, lambda r: r["end_state"] == "success", f"{ck}.asr", meta)
            L(COL_GO_ASR, "gate_only_asr_pct", fmt.fmt_rate_ci(s, *s_ci), n_items=len(den), **_rate_ci_cols(s_ci),
              test=basis + f", cell key {ck}.asr); strict success = an auto-decided wrong decision on the final")
    return out
