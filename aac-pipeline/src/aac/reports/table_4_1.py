"""Table 4.1 (baseline reproduction) from sealed runs; Published values only from an analyst-owned
file (absent -> `n/a (published value unverified)` and Deviation `n/a`).

Two row plans , picked from the data manifest shared by every run of the namespace
(configs/reports/table-4.1-profiles.yaml; mixed manifests are refused):
 stub Phase 1 walking skeleton: BASE r0-*, R3 r3-stub, R1 r1-random-content-only (fixture corpus) with the
 frozen configs/reports/published_baselines.fixture.yaml. Unchanged, so the stub goldens do not move.
 real rows 1-2 BASE bodega-fc-clean / bodega-rd-clean (metric per the row's metric_key), rows 3-4 BASE
 bodega-*-bertattack-untargeted (ASR = successes_k / subset_size, bootstrap over k ones and n - k zeros;
 BODEGA's untargeted BERT-Attack, every attack.tsv row, like-for-like with the CheckThat! 2024 Task 6
 overview Table 5 confusion score; methodological note
 averitec-gold-baseline (four-way macro-F1, decision D2), row 6 R1 r1-chrono-content-only on the test
 records of the surviving dev claims (four-way macro-F1, seeds as replicates, decision D1), row 7 the
 AVeriTeC no-evidence baseline as a "comparison, not reproduction" (its Reproduced cell is row 6's cell), with configs/published_baselines.yaml. The real build also writes reports/published-targets.csv.

Targeted BERT-Attack (methodological note:
the sealed targeted runs bodega-*-bertattack (label-1 items the victim gets right; the BODEGA paper's Table 5
T-cell) keep every quantity they carried as rows 3-4 (ASR with interval, same cell key, successes k of n, BODEGA,
semantic and character scores, queries per example), under rows 3-4's row labels with col_label suffix
TARGETED_SUFFIX (the col + suffix convention of table_4_4._count_rows). Each run's result.json `targeted` flag is
checked against its reading; an absent run, untargeted or targeted, is MissingRun naming the config (never a fallback).

Chunked untargeted RD (methodological note
with the items processed), real plan only: row 4's shell cell pools EVERY run named in RD_CHUNK_TABLE (from
aac.baselines.rd_chunks, the table the chunk configs are written from; an absent chunk is MissingRun, never a partial
pool). Each run's result.json and config.resolved.json must carry its table row_range; the ranges must tile
[0, 415) exactly once; each chunk attacked exactly its rows. Pooled: k = sum successes_k, n = sum subset_size (= 415),
ASR = k / n with the same bootstrap over k ones and n - k zeros and the same cell key; queries per example and the
BODEGA score = item-weighted means of the per-run means (BODEGA averages both over every attacked item); semantic and
character scores = success-weighted means (BODEGAScore.compute averages them over the successful examples only). One
set of long rows per chunk (col_label + CHUNK suffix ", chunk cNN (rows a-b)", rows 0-based inclusive) carries that
run's id, config hash, k of n, queries per example and scores. The FC row reads its single run unchanged.

Addition C (gap review gap 8, methodological note
R1-supplementary, never a shell cell or a system result): "Full pipeline, gold QA evidence (oracle)" from the sealed
runs the real Phase 4 profile names as `supplementary.oracle` (r1-chrono-full-goldqa): binary macro-F1 [CI] on the
chronological test set (seeds as replicates), the four-way cell `n/a` with its reason (the full pipeline decides the
binary task; its only four-way output is the C1 encoder distribution, which the evidence does not change), and,
against the seed-matched r1-chrono-full runs on the same items, the paired item bootstrap of the seed-mean binary
macro-F1 difference and the exact McNemar on the seed-majority outcome "correct" (the Table 4.2 H1 pattern). An
absent oracle run reads `not run (cut: addition-C on 2026-11-01)` once its cut rule has passed (31 Oct 2026 23:59
Europe/London), `pending ` under allow_pending, else MissingRun naming the run.

Published values `published` is always a fraction in [0, 1], also on a `unit: percent`
row, whose display is 100 x the value (write a paper's 62.1 % as 0.621). Any other value is refused."""
from __future__ import annotations

import csv
import io
import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import f1_score

from aac.attacks.ledger import read_ledger
from aac.baselines import rd_chunks
from aac.metrics.classification import macro_f1_4way, macro_f1_binary
from aac.metrics.selective import binary_macro_f1 as _f1_fast
from aac.reports import fmt, phase4_sources, sources
from aac.run.hashing import canonical, derive_seed, sha256_bytes
from aac.stats.bootstrap import ci as boot_ci, ci_index
from aac.stats.tests import mcnemar_exact
from aac.run.paths import reports_dir, runs_dir

TABLE = "4.1"
PUBLISHED_FILE = Path("configs") / "published_baselines.yaml"
PROFILES_FILE = Path("configs") / "reports" / "table-4.1-profiles.yaml"
UNVERIFIED = "n/a (published value unverified)"
NOT_TRANSCRIBED = "n/a (not transcribed)"
PUBLISHED_TARGETS_FILE = "published-targets.csv"
DEV_PREFIX = "averitec/dev/"
METRIC_KEYS = ("macro_f1_binary", "f1_positive_class", "asr", "macro_f1_4way")
# Row 7 : the AVeriTeC paper's no-evidence BERT-large figure set against this project's
# content-only encoder. Its Reproduced cell IS row 6's cell (same runs, same records, same interval); only the
# published figure differs. Marked "comparison, not reproduction" in its labels and in the long-sheet test text.
NO_EVIDENCE_ROW = "AVeriTeC baseline, dev, no evidence (comparison, not reproduction)"
# methodological note
# long rows whose col_label ends with TARGETED_SUFFIX (the exact label the analyst named).
TARGETED_LABEL = "targeted (BODEGA Table 5 T-cell)"
TARGETED_SUFFIX = ", " + TARGETED_LABEL
# methodological note.
RD_CHUNK_TABLE = tuple((rd_chunks.config_name(c), c, start, stop) for c, start, stop in rd_chunks.CHUNKS)
RD_N_ROWS = rd_chunks.N_ROWS
ATTACK_ROWS = (  # (row label, untargeted config names: the shell cell, targeted config name: long rows only)
    ("BERT victim, FC, BERT-Attack", ("bodega-fc-bertattack-untargeted",), "bodega-fc-bertattack"),
    ("BERT victim, RD, BERT-Attack", tuple(n for n, _c, _s, _e in RD_CHUNK_TABLE), "bodega-rd-bertattack"),
)
POOLED_ROWS = {"BERT victim, RD, BERT-Attack": (RD_CHUNK_TABLE, RD_N_ROWS)}   # row -> (chunk table, rows to tile)
NO_SUCCESS = "n/a (no successful example)"
UNTARGETED_TEST = ("untargeted BERT-Attack (attack.py targeted=false): every attack.tsv row attacked; like-for-like "
                   "with the CheckThat! 2024 Task 6 overview Table 5 confusion score (protocol notes)")
TARGETED_TEST = ("targeted BERT-Attack (attack.py targeted=true): label-1 items the victim gets right; BODEGA Table 5 "
                 "T-cell; supplementary, not the shell value (protocol notes)")
COMPARISON_NOTE = ("comparison, not reproduction: Reproduced is row 6's value (this project's content-only encoder); the "
                   "published figure is the AVeriTeC paper's BERT-large without evidence (different model, training "
                   "recipe and split)")

# reports. No Deviation is ever computed for them (n/a), whatever configs/published_baselines.yaml holds;
# the reason is written into the long sidecar. (Row labels are the draft's exact strings.) Stub plan only:
# the real plan reads row 6 from the surviving dev claims (decision D1, methodological note).
NOT_COMPARABLE = {
    "Content-only encoder, AVeriTeC dev (this project)":
        "source run r1-random-content-only, test partition = a random 20 percent of the dated claims, not the "
        "official AVeriTeC dev set; not comparable with a published dev-set figure",
    NO_EVIDENCE_ROW:
        "source run r1-random-content-only, test partition = a random 20 percent of the dated claims, not the "
        "official AVeriTeC dev set; not comparable with a published dev-set figure",
}


class MixedManifests(RuntimeError):
    """The runs of one namespace name more than one data manifest: no single Table 4.1 plan applies."""


def profile(root: Path) -> dict:
    """{plan, published, manifest} for the data manifest shared by every run in runs/INDEX.json."""
    root = Path(root)
    entries = sources.index_runs(root)
    if not entries:
        raise sources.MissingRun(f"{runs_dir(root) / 'INDEX.json'} lists no run: no Table 4.1 plan applies")
    manifests: dict[str, list[str]] = {}
    for e in entries:
        resolved = json.loads((runs_dir(root) / e["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))
        manifests.setdefault(str(resolved["data"]["manifest"]), []).append(e["run_id"])
    if len(manifests) != 1:
        detail = "; ".join(f"{m}: {len(ids)} runs (e.g. {ids[0]})" for m, ids in sorted(manifests.items()))
        raise MixedManifests(f"the runs of {runs_dir(root)} name {len(manifests)} data manifests ({detail}); "
                             f"Table 4.1 refuses to mix plans")
    manifest = next(iter(manifests))
    with open(root / PROFILES_FILE, encoding="utf-8") as fh:
        profiles = yaml.safe_load(fh)["profiles"]
    hits = [p for p in profiles if p["manifest"] == manifest]
    if len(hits) != 1:
        raise KeyError(f"{PROFILES_FILE}: {len(hits)} profiles for data manifest {manifest!r} (expected 1)")
    if hits[0]["plan"] not in ("stub", "real"):
        raise ValueError(f"{PROFILES_FILE}: unknown plan {hits[0]['plan']!r}")
    return {"plan": hits[0]["plan"], "published": Path(hits[0]["published"]), "manifest": manifest}


class PublishedValueError(ValueError):
    """A `published` value is not a fraction in [0, 1] ."""


def published_fraction(spec: dict, where: str = PUBLISHED_FILE.as_posix()) -> float | None:
    """The `published` value of one row as a fraction, or None when it is absent.

 One convention for every row `published` is ALWAYS a fraction in [0, 1], also on a row whose `unit` is
 percent. `unit` only selects the display (percent: 100 x the value, 1 d.p.; Deviation in pp). A paper's
 62.1 % is written 0.621. Anything else raises PublishedValueError: no cell is ever rendered from it."""
    v = spec.get("published")
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise PublishedValueError(f"{where}: row {spec.get('row')!r} published {v!r} is not a number")
    f = float(v)
    if not (0.0 <= f <= 1.0):
        raise PublishedValueError(
            f"{where}: row {spec.get('row')!r} published {v} must be a fraction in [0, 1] (write 62.1 % as 0.621); "
            f"unit {spec.get('unit')!r} only selects the display")
    return f


def published(root: Path, path: Path | None = None) -> dict[str, dict]:
    rel = Path(path) if path is not None else PUBLISHED_FILE
    with open(Path(root) / rel, encoding="utf-8") as fh:
        rows = yaml.safe_load(fh)["rows"]
    out = {}
    for r in rows:
        if r["row"] in out:
            raise ValueError(f"{rel}: duplicate row {r['row']!r}")
        if r.get("unit") not in ("fraction", "percent"):
            raise ValueError(f"{rel}: row {r['row']!r} unit must be fraction or percent")
        if "metric_key" in r and r["metric_key"] not in METRIC_KEYS:
            raise ValueError(f"{rel}: row {r['row']!r} metric_key must be one of {METRIC_KEYS}")
        published_fraction(r, rel.as_posix())
        out[r["row"]] = r
    return out


def _f1_cell(root: Path, entries: list[dict], split: str, cell_key: str, four_way: bool = False,
             id_prefix: str | None = None) -> dict:
    runs = sources.load_runs(root, entries)
    ids, per_seed = sources.aligned(runs, split)
    if id_prefix is not None:  # real row 6: only the test records of the surviving dev claims (decision D1)
        keep = [i for i, cid in enumerate(ids) if cid.startswith(id_prefix)]
        if not keep:
            raise sources.MissingRun(f"{runs[0].run_id}: no {split!r} records with claim_id prefix {id_prefix!r}")
        ids = [ids[i] for i in keep]
        per_seed = [[recs[i] for i in keep] for recs in per_seed]
    n = len(ids)
    n_boot = sources.resamples(root, runs[0])
    if four_way:
        y = sources.label_index_for(root, runs[0], per_seed[0])
        preds = [np.asarray([int(np.argmax(r["encoder_distribution"])) for r in recs]) for recs in per_seed]
        points = [macro_f1_4way(y, p) for p in preds]
        stats = [(lambda idx, p=p: macro_f1_4way(y[idx], p[idx])) for p in preds]
    else:
        y = sources.positives_for(root, runs[0], per_seed[0])
        preds = [sources.arr(recs, "calibrated_probability") >= 0.5 for recs in per_seed]
        points = [macro_f1_binary(y, p) for p in preds]
        stats = [(lambda idx, p=p: _f1_fast(y[idx], p[idx])) for p in preds]
    lo, hi = sources.seeds_as_replicates_ci(stats, n, cell_key=cell_key, n_resamples=n_boot, level=sources.level(runs[0]))
    return {"point": float(np.mean(points)), "lo": lo, "hi": hi, "n_items": n, "runs": runs,
            "test": "percentile bootstrap (items; seeds as replicates)" if len(runs) > 1 else "percentile bootstrap (items)"}


def _asr_cell(root: Path, entries: list[dict], subset: str, cell_key: str) -> dict:
    runs = sources.load_runs(root, entries)
    rows = [r for r in read_ledger(runs[0].run_dir)
            if r["attack"] == "bert_attack" and r["victim"] == "content_only" and r["subset"] == subset and int(r["queries_used"]) > 0]
    if not rows:
        raise sources.MissingRun(f"{runs[0].run_id}: no attacked bert_attack/content_only rows for subset {subset!r}")
    ind = np.asarray([1.0 if r["end_state"] == "success" else 0.0 for r in rows])
    lo, hi = boot_ci(ind, cell_key=cell_key, n_resamples=sources.resamples(root, runs[0]), level=sources.level(runs[0]))
    return {"point": float(ind.mean()), "lo": lo, "hi": hi, "n_items": len(rows), "runs": runs,
            "test": "percentile bootstrap (attacked items)"}


def _external_one(root: Path, row: str, name: str) -> sources.ExternalRun:
    state, entries = sources.require_runs(root, table=TABLE, row_ident=row, col_key="reproduced", regime="BASE", name=name,
                                          system="external")
    if state == "cut":
        raise sources.MissingRun(f"Table 4.1 row {row!r} is never blanked by a cut")
    if len(entries) != 1:
        raise RuntimeError(f"Table 4.1 row {row!r}: {len(entries)} sealed BASE/{name} runs, expected exactly 1")
    return sources.load_external(root, entries[0])


def _close(a: float, b: float, what: str) -> None:
    if abs(float(a) - float(b)) > 1e-12:
        raise RuntimeError(f"{what}: recomputed {a!r} != recorded {b!r}")


def _item_ci(root: Path, run: sources.Run, stat, n: int, cell_key: str) -> tuple[float, float]:
    return sources.seeds_as_replicates_ci([stat], n, cell_key=cell_key, n_resamples=sources.resamples(root, run),
                                          level=sources.level(run))


def _bodega_clean_cell(root: Path, row: str, name: str, metric_key: str, cell_key: str) -> dict:
    """Rows 1-2: BODEGA test_victims.py outcomes (items.jsonl rows {row, item_id, y, pred}), label 1 positive."""
    run = _external_one(root, row, name)
    y = np.asarray([int(it["y"]) == 1 for it in run.items], dtype=bool)
    p = np.asarray([int(it["pred"]) == 1 for it in run.items], dtype=bool)
    n = len(y)
    if n == 0 or n != int(run.result["n_items"]):
        raise RuntimeError(f"{run.run_id}: items.jsonl has {n} rows, result.json n_items {run.result['n_items']}")

    def pos_f1(yy, pp):
        return float(f1_score(yy.astype(int), pp.astype(int), pos_label=1, average="binary", zero_division=0.0))

    values = {"macro_f1_binary": macro_f1_binary(y, p), "f1_positive_class": pos_f1(y, p)}
    _close(values["macro_f1_binary"], run.result["macro_f1_binary"], f"{run.run_id} macro_f1_binary")
    _close(values["f1_positive_class"], run.result["f1_positive"], f"{run.run_id} f1_positive")
    if metric_key not in values:
        raise ValueError(f"Table 4.1 row {row!r}: metric_key {metric_key!r} is not a clean-victim metric")
    stat = (lambda idx: _f1_fast(y[idx], p[idx])) if metric_key == "macro_f1_binary" else (lambda idx: pos_f1(y[idx], p[idx]))
    lo, hi = _item_ci(root, run, stat, n, cell_key)
    other = "f1_positive_class" if metric_key == "macro_f1_binary" else "macro_f1_binary"
    extras = [
        ("Reproduced (other F1 variant)", other, fmt.fmt_f1(values[other]),
         {"test": "recomputed from items.jsonl; not the headline metric (configs/published_baselines.yaml metric_key)"}),
        ("Reproduced (F1 printed by BODEGA)", "bodega_printed_f1_positive_class", fmt.fmt_f1(run.result["printed"]["f1_positive"]),
         {"test": "BODEGA test_victims.py stdout (result.json printed.f1_positive); equals the recomputed positive-class F1"}),
    ]
    return {"point": values[metric_key], "lo": lo, "hi": hi, "n_items": n, "runs": [run], "metric_name": metric_key,
            "test": "percentile bootstrap (items)", "extras": extras}


def _bodega_attack_cell(root: Path, row: str, name: str, cell_key: str, targeted: bool = False) -> dict:
    """Rows 3-4: ASR = successes_k / subset_size of BODEGA's unbudgeted BERT-Attack (no per-item file); untargeted
 for the shell cell, targeted for the TARGETED_LABEL long rows (methodological note)."""
    run = _external_one(root, row, name)
    if bool(run.result.get("targeted")) is not targeted:
        raise RuntimeError(f"{run.run_id} ({name}): result.json targeted={run.result.get('targeted')!r}, "
                           f"Table 4.1 row {row!r} reads it as targeted={targeted}")
    reading = TARGETED_TEST if targeted else UNTARGETED_TEST
    k, n = int(run.result["successes_k"]), int(run.result["subset_size"])
    if not 0 <= k <= n or n == 0:
        raise RuntimeError(f"{run.run_id}: successes_k {k} of subset_size {n}")
    _close(k / n, run.result["success_score"], f"{run.run_id} success_score")
    ind = np.asarray([1.0] * k + [0.0] * (n - k))
    lo, hi = boot_ci(ind, cell_key=cell_key, n_resamples=sources.resamples(root, run), level=sources.level(run))
    r = run.result
    extras = [
        ("Reproduced (successes k of n)", "successes_k_of_subset_size", f"{k} of {n}", {"test": f"BODEGA attack.py summary; {reading}"}),
        ("Reproduced (BODEGA score)", "bodega_score", fmt.fmt_metric(r["bodega_score"]), {"test": f"BODEGA attack.py summary; {reading}"}),
        ("Reproduced (semantic score)", "semantic_score", fmt.fmt_metric(r["semantic_score"]), {"test": f"BODEGA attack.py summary (BLEURT); {reading}"}),
        ("Reproduced (character score)", "character_score", fmt.fmt_metric(r["character_score"]), {"test": f"BODEGA attack.py summary; {reading}"}),
        ("Reproduced (queries per example)", "queries_per_example", fmt.fmt_count(round(float(r["queries_per_example"]), 1)),
         {"test": f"BODEGA attack.py summary; no query budget; {reading}"}),
    ]
    return {"point": k / n, "lo": lo, "hi": hi, "n_items": n, "runs": [run], "metric_name": "asr_pct",
            "test": f"percentile bootstrap (attacked items: k ones and n - k zeros); {reading}", "extras": extras}


def chunk_label(chunk_id: str, start: int, stop: int) -> str:
    """col_label suffix of a chunk's long rows: rows 0-based, inclusive (start .. stop - 1)."""
    return f", chunk {chunk_id} (rows {start}-{stop - 1})"


def check_tiling(ranges: list[tuple[int, int]], n_total: int, what: str) -> None:
    """The [start, stop) ranges cover [0, n_total) exactly once (any gap, overlap or duplicate raises)."""
    pos = 0
    for start, stop in sorted(ranges):
        if start != pos or stop <= start:
            raise RuntimeError(f"{what}: row ranges {sorted(ranges)} do not tile [0, {n_total}) exactly once "
                               f"(expected a range starting at {pos}, got [{start}, {stop}))")
        pos = stop
    if pos != n_total:
        raise RuntimeError(f"{what}: row ranges {sorted(ranges)} end at {pos}, not {n_total}")


def _weighted(pairs: list[tuple[int, float | None]], what: str) -> float | None:
    """sum(w x v) / sum(w) over the pairs with w > 0; a None value with w > 0 raises; None when sum(w) = 0."""
    total = sum(w for w, _v in pairs)
    if total == 0:
        return None
    if any(v is None for w, v in pairs if w > 0):
        raise RuntimeError(f"{what}: a chunk with a positive weight has no value")
    return float(sum(w * float(v) for w, v in pairs if w > 0) / total)


def _score_text(v: float | None) -> str:
    return NO_SUCCESS if v is None else fmt.fmt_metric(v)


def _bodega_attack_pooled(root: Path, row: str, table, n_total: int, cell_key: str) -> dict:
    """Row 4 from the chunk runs (module docstring, methodological note)."""
    runs, parts = [], []
    for name, cid, start, stop in table:
        run = _external_one(root, row, name)
        r = run.result
        if bool(r.get("targeted")):
            raise RuntimeError(f"{run.run_id} ({name}): result.json targeted={r.get('targeted')!r}; row {row!r} pools "
                               "untargeted chunks only")
        cfg_range = (run.resolved.get("external") or {}).get("row_range")
        if r.get("row_range") != [start, stop] or cfg_range != [start, stop]:
            raise RuntimeError(f"{run.run_id} ({name}): row_range result.json {r.get('row_range')!r}, config "
                               f"{cfg_range!r}; the chunk table says [{start}, {stop}]")
        k, n = int(r["successes_k"]), int(r["subset_size"])
        if n != stop - start or int(r["n_items"]) != n or not 0 <= k <= n:
            raise RuntimeError(f"{run.run_id} ({name}): successes_k {k}, subset_size {n}, n_items {r['n_items']} for "
                               f"rows [{start}, {stop})")
        _close(k / n, r["success_score"], f"{run.run_id} success_score")
        runs.append(run)
        parts.append((cid, start, stop, k, n, r))
    check_tiling([(p[1], p[2]) for p in parts], n_total, f"Table 4.1 row {row!r}")
    k_all, n_all = sum(p[3] for p in parts), sum(p[4] for p in parts)
    if n_all != n_total:
        raise RuntimeError(f"Table 4.1 row {row!r}: pooled n {n_all} != {n_total}")
    ind = np.asarray([1.0] * k_all + [0.0] * (n_all - k_all))
    lo, hi = boot_ci(ind, cell_key=cell_key, n_resamples=sources.resamples(root, runs[0]), level=sources.level(runs[0]))
    q = _weighted([(p[4], p[5]["queries_per_example"]) for p in parts], f"{row} queries_per_example")
    bodega = _weighted([(p[4], p[5]["bodega_score"]) for p in parts], f"{row} bodega_score")
    sem = _weighted([(p[3], p[5]["semantic_score"]) for p in parts], f"{row} semantic_score")
    char = _weighted([(p[3], p[5]["character_score"]) for p in parts], f"{row} character_score")
    pooled = (f"pooled over {len(parts)} chunk runs {table[0][0]} .. {table[-1][0]} (attack.tsv rows [0, {n_total}) tiled once, "
              f"one attack.py process per chunk)")
    reading = f"{UNTARGETED_TEST}; {pooled}"
    item_w = ("item-weighted mean of the per-run means BODEGA prints (BODEGA averages it over every attacked item)")
    succ_w = ("success-weighted mean of the per-run means BODEGA prints (BODEGA averages it over the successful "
              "examples only)")
    extras = [
        ("Reproduced (successes k of n)", "successes_k_of_subset_size", f"{k_all} of {n_all}",
         {"test": f"sum of the chunks' BODEGA attack.py summaries; {reading}"}),
        ("Reproduced (BODEGA score)", "bodega_score", _score_text(bodega), {"test": f"{item_w}; {reading}"}),
        ("Reproduced (semantic score)", "semantic_score", _score_text(sem), {"test": f"BLEURT; {succ_w}; {reading}"}),
        ("Reproduced (character score)", "character_score", _score_text(char), {"test": f"{succ_w}; {reading}"}),
        ("Reproduced (queries per example)", "queries_per_example", fmt.fmt_count(round(q, 1)),
         {"test": f"no query budget; {item_w}; {reading}"}),
    ]
    chunk_rows = []
    for run, (cid, start, stop, k, n, r) in zip(runs, parts):
        sfx = chunk_label(cid, start, stop)
        common = dict(n_items=n, n_seeds=1, seeds=str(run.seed), **sources.provenance([run]))
        note = f"BODEGA attack.py summary of chunk {cid} alone (attack.tsv rows {start}-{stop - 1}); {UNTARGETED_TEST}"
        for col, metric, value in (
                ("Reproduced (successes k of n)", "successes_k_of_subset_size", f"{k} of {n}"),
                ("Reproduced (BODEGA score)", "bodega_score", _score_text(r["bodega_score"])),
                ("Reproduced (semantic score)", "semantic_score", _score_text(r["semantic_score"] if k else None)),
                ("Reproduced (character score)", "character_score", _score_text(r["character_score"] if k else None)),
                ("Reproduced (queries per example)", "queries_per_example",
                 fmt.fmt_count(round(float(r["queries_per_example"]), 1)))):
            chunk_rows.append(fmt.long_row(TABLE, row, col + sfx, metric, value, test=note, **common))
    seeds = sorted({r.seed for r in runs})
    return {"point": k_all / n_all, "lo": lo, "hi": hi, "n_items": n_all, "runs": runs, "metric_name": "asr_pct",
            "test": f"percentile bootstrap (attacked items: k ones and n - k zeros, k and n summed over the chunks); {reading}",
            "extras": extras, "chunk_rows": chunk_rows, "n_seeds": len(seeds), "seeds": ", ".join(map(str, seeds))}


def targeted_rows(root: Path) -> list[dict]:
    """Real plan only: the targeted BERT-Attack runs as long rows under rows 3-4's labels, col_label + TARGETED_SUFFIX
 (module docstring). Same cell key as the shell cell they used to fill, so their interval is unchanged."""
    out = []
    for row, _untargeted, name in ATTACK_ROWS:
        c = _bodega_attack_cell(root, row, name, f"boot.table-4.1.{row}.reproduced", targeted=True)
        prov = sources.provenance(c["runs"])
        common = dict(n_items=c["n_items"], n_seeds=len(c["runs"]), seeds=sources.seeds_text(c["runs"]), **prov)
        out.append(fmt.long_row(TABLE, row, "Reproduced [95% CI]" + TARGETED_SUFFIX, c["metric_name"], fmt.fmt_rate(c["point"]),
                                ci_lo=fmt.fmt_rate(c["lo"]), ci_hi=fmt.fmt_rate(c["hi"]), test=c["test"], **common))
        for col, metric, value, kw in c["extras"]:
            out.append(fmt.long_row(TABLE, row, col + TARGETED_SUFFIX, metric, value, **{**common, **kw}))
    return out


def _averitec_gold_cell(root: Path, row: str, name: str, cell_key: str) -> dict:
    """Row 5: the AVeriTeC baseline on the 500 dev claims' gold evidence, four-way macro-F1 (decision D2)."""
    run = _external_one(root, row, name)
    y = np.asarray([int(it["y_index"]) for it in run.items], dtype=int)
    p = np.asarray([int(it["pred_index"]) for it in run.items], dtype=int)
    n = len(y)
    if n == 0 or n != int(run.result["n_claims"]):
        raise RuntimeError(f"{run.run_id}: items.jsonl has {n} rows, result.json n_claims {run.result['n_claims']}")
    point = macro_f1_4way(y, p)
    _close(point, run.result["macro_f1_4way"], f"{run.run_id} macro_f1_4way")
    pos = list(run.result["label_order"]).index(run.result["binary_positive_label"])
    yb, pb = y == pos, p == pos
    binary = macro_f1_binary(yb, pb)
    _close(binary, run.result["macro_f1_binary"], f"{run.run_id} macro_f1_binary")
    lo, hi = _item_ci(root, run, lambda idx: macro_f1_4way(y[idx], p[idx]), n, cell_key)
    extras = [("Reproduced (binary macro-F1)", "macro_f1_binary", fmt.fmt_f1(binary),
               {"test": f"recomputed from items.jsonl, {run.result['binary_positive_label']} positive; D-04 headline metric for comparisons"})]
    return {"point": point, "lo": lo, "hi": hi, "n_items": n, "runs": [run], "metric_name": "macro_f1_4way",
            "test": "percentile bootstrap (claims)", "extras": extras}


def _real_rows(root: Path, pub: dict[str, dict]) -> list[dict]:
    plan = [
        ("BERT victim, FC, clean", "bodega-fc-clean", "clean"),
        ("BERT victim, RD, clean", "bodega-rd-clean", "clean"),
        *((row, untargeted, "attack") for row, untargeted, _targeted in ATTACK_ROWS),   # names: a tuple
        ("AVeriTeC baseline, dev, gold evidence", "averitec-gold-baseline", "gold"),
        ("Content-only encoder, AVeriTeC dev (this project)", "r1-chrono-content-only", "encoder"),
    ]
    out = []
    for row, name, kind in plan:
        key = f"boot.table-4.1.{row}.reproduced"
        metric_key = pub[row].get("metric_key")
        if kind == "clean":
            cell = _bodega_clean_cell(root, row, name, metric_key or "macro_f1_binary", key)
        elif kind == "attack" and row in POOLED_ROWS:
            cell = _bodega_attack_pooled(root, row, *POOLED_ROWS[row], key)
        elif kind == "attack":
            (single,) = name
            cell = _bodega_attack_cell(root, row, single, key)
        elif kind == "gold":
            cell = _averitec_gold_cell(root, row, name, key)
        else:
            state, entries = sources.require_runs(root, table=TABLE, row_ident=row, col_key="reproduced", regime="R1", name=name)
            if state == "cut":
                raise sources.MissingRun(f"Table 4.1 row {row!r} is never blanked by a cut")
            cell = _f1_cell(root, entries, "test", key, True, id_prefix=DEV_PREFIX)
            cell.update({"metric_name": "macro_f1_4way", "extras": [],
                         "test": cell["test"] + f"; test records of the surviving AVeriTeC dev claims ({DEV_PREFIX}*), "
                                                "chronological-split encoders (decision D1)"})
        cell.update({"row": row, "spec": pub[row], "unit": pub[row]["unit"], "kind": kind, "split": "test",
                     "not_comparable": None})
        out.append(cell)
    return out + [_row7_cell(out[-1], pub)]


def _row7_cell(row6: dict, pub: dict[str, dict]) -> dict:
    """Row 7 reuses the row-6 computation (no second bootstrap, no duplicated code): same point, interval, runs."""
    cell = dict(row6)
    cell.update({"row": NO_EVIDENCE_ROW, "spec": pub[NO_EVIDENCE_ROW], "unit": pub[NO_EVIDENCE_ROW]["unit"],
                 "comparison_note": COMPARISON_NOTE})
    cell["test"] = row6["test"] + "; " + COMPARISON_NOTE
    return cell


def _stub_rows(root: Path, pub: dict[str, dict]) -> list[dict]:
    plan = [
        ("BERT victim, FC, clean", dict(regime="BASE", name="r0-fc-rd-clean"), ("f1", "fc", False)),
        ("BERT victim, RD, clean", dict(regime="BASE", name="r0-fc-rd-clean"), ("f1", "rd", False)),
        ("BERT victim, FC, BERT-Attack", dict(regime="R3", name="r3-stub"), ("asr", "fc", False)),
        ("BERT victim, RD, BERT-Attack", dict(regime="R3", name="r3-stub"), ("asr", "rd", False)),
        ("AVeriTeC baseline, dev, gold evidence", dict(regime="BASE", name="r0-averitec-gold"), ("f1", "test", False)),
        ("Content-only encoder, AVeriTeC dev (this project)", dict(regime="R1", name="r1-random-content-only"), ("f1", "test", True)),
    ]
    out = []
    for row, sel, (kind, split, four_way) in plan:
        state, entries = sources.require_runs(root, table=TABLE, row_ident=row, col_key="reproduced", **sel)
        if state == "cut":
            raise sources.MissingRun(f"Table 4.1 row {row!r} is never blanked by a cut")
        key = f"boot.table-4.1.{row}.reproduced"
        cell = _f1_cell(root, entries, split, key, four_way) if kind == "f1" else _asr_cell(root, entries, split, key)
        metric_name = {"asr": "asr_pct", "f1": "macro_f1_4way" if row.startswith("Content-only") else "macro_f1_binary"}[kind]
        cell.update({"row": row, "spec": pub[row], "unit": pub[row]["unit"], "kind": kind, "split": split,
                     "metric_name": metric_name, "extras": [], "not_comparable": NOT_COMPARABLE.get(row)})
        out.append(cell)
    cell7 = _row7_cell(out[-1], pub)
    cell7["not_comparable"] = NOT_COMPARABLE.get(NO_EVIDENCE_ROW)
    return out + [cell7]


def rows_data(root: Path) -> list[dict]:
    """The seven rows with point, CI, unit, provenance and the published spec (shared with Table 4.3)."""
    root = Path(root)
    prof = profile(root)
    pub = published(root, prof["published"])
    shell_rows = fmt.load_shells(root / "contracts" / "table-shells.json")[TABLE]["rows"]
    if prof["plan"] == "stub" and NO_EVIDENCE_ROW not in pub:
        # the stub's published-values file is a frozen byte copy from before (it has six rows and is never
        # edited); row 7 (19 Oct 2026) takes its descriptive fields from the live file with every analyst-owned value
        # null, like the rest of that fixture
        pub[NO_EVIDENCE_ROW] = {**published(root)[NO_EVIDENCE_ROW], "published": None, "page_ref": None}
    if sorted(pub) != sorted(shell_rows):
        raise ValueError(f"{prof['published']} rows {sorted(pub)} differ from the Table 4.1 shell rows")
    cells = _real_rows(root, pub) if prof["plan"] == "real" else _stub_rows(root, pub)
    for c in cells:
        c["plan"] = prof["plan"]
    return cells


def _fmt_point(cell: dict) -> tuple[str, str]:
    if cell["unit"] == "percent":
        return fmt.fmt_rate_ci(cell["point"], cell["lo"], cell["hi"]), fmt.fmt_rate(cell["point"])
    return fmt.fmt_ci(cell["point"], cell["lo"], cell["hi"]), fmt.fmt_f1(cell["point"])


def published_cell(spec: dict) -> str:
    v = published_fraction(spec)
    if v is None:
        return fmt.NA if spec.get("not_applicable") else UNVERIFIED
    return fmt.fmt_rate(v) if spec["unit"] == "percent" else fmt.fmt_f1(v)


def deviation_cell(cell: dict) -> str:
    v = published_fraction(cell["spec"])
    # a cell without the key is read with the stub rule (NOT_COMPARABLE); the real plan sets it to None
    note = cell["not_comparable"] if "not_comparable" in cell else NOT_COMPARABLE.get(cell["row"])
    if v is None or note:
        return fmt.NA
    d = cell["point"] - v
    return fmt.fmt_pp(d) if cell["unit"] == "percent" else fmt.fmt_signed(d, 3)


PUBLISHED_TARGET_COLUMNS = ("row", "source", "metric", "metric_key", "unit", "published", "page_ref")


def write_published_targets(root: Path, data_version: str, out_dir: Path | None = None) -> Path:
    """reports/published-targets.csv: configs/published_baselines.yaml rendered as it stands, one line per Table 4.1
 row, no value added. A null `published` / `page_ref` reads `n/a (not transcribed)`; a row with
 not_applicable reads `n/a (not applicable)`. First line: `# config_hash=<sha256 of the canonical JSON of the
 YAML> data_version=<data_version>`; build passes the single data_version of the Table 4.1 runs, which the
 real-report contract test compares with evidence-pack/data_version.txt."""
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    if not data_version or ";" in data_version:
        raise ValueError(f"published targets need one data_version, got {data_version!r}")
    with open(root / PUBLISHED_FILE, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    rows = list(published(root, PUBLISHED_FILE).values())
    buf = io.StringIO()
    buf.write(f"# config_hash={sha256_bytes(canonical(doc))} data_version={data_version}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(PUBLISHED_TARGET_COLUMNS)
    for r in rows:
        absent = "n/a (not applicable)" if r.get("not_applicable") else NOT_TRANSCRIBED
        w.writerow([r["row"], r["source"], r["metric"], r.get("metric_key") or fmt.NA, r["unit"],
                    absent if r.get("published") is None else str(r["published"]),
                    absent if r.get("page_ref") is None else str(r["page_ref"])])
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / PUBLISHED_TARGETS_FILE
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="utf-8")
    tmp.replace(target)
    return target


ORACLE_ROW = "Full pipeline, gold QA evidence (oracle)"
ORACLE_BASE = "r1-chrono-full"
FAM_ORACLE = "R1-supplementary"
ORACLE_FOUR_WAY = ("n/a (the full pipeline decides the binary task; its only four-way output is the C1 encoder "
                   "distribution, which gold evidence does not change)")
ORACLE_COLS = (("Binary macro-F1 [95% CI]", "macro_f1_binary"), ("Four-way macro-F1 [95% CI]", "macro_f1_4way"),
               ("Paired difference vs full pipeline, binary macro-F1", "macro_f1_binary_diff"),
               ("Exact McNemar vs full pipeline (seed majority)", "mcnemar_seed_majority"))


def oracle_rows(root: Path, data_version: str, allow_pending: bool = False) -> list[dict]:
    """Addition C long rows (module docstring); when the real profile names no oracle run."""
    from aac.reports.table_4_4 import absent_addition

    root = Path(root)
    name = phase4_sources.load_profile(root, "real")["supplementary"].get("oracle")
    if not name:
        return []
    entries = sources.select_runs(root, regime="R1", name=name, system="full")
    if not entries:
        cell, note = absent_addition(name, "addition-C", table=TABLE, allow_pending=allow_pending,
                                     what="addition C (gold-evidence oracle)")
        return [fmt.long_row(TABLE, ORACLE_ROW, col, metric, cell, config_hash="", data_version=data_version,
                             run_ids="n/a", source_shards_sha256="n/a", test=note, family=FAM_ORACLE)
                for col, metric in ORACLE_COLS]
    base_entries = sources.select_runs(root, regime="R1", name=ORACLE_BASE, system="full")
    by_seed = {int(e["seed"]): e for e in base_entries}
    missing = [int(e["seed"]) for e in entries if int(e["seed"]) not in by_seed]
    if missing:
        raise sources.MissingRun(f"Table 4.1 oracle row: no sealed {ORACLE_BASE} run for seed(s) {missing} to pair with {name}")
    runs = sources.load_runs(root, entries)
    base = sources.load_runs(root, [by_seed[int(e["seed"])] for e in entries])
    ids, per_seed = sources.aligned(runs, "test")
    ids_b, per_seed_b = sources.aligned(base, "test")
    if ids != ids_b:
        raise RuntimeError(f"Table 4.1 oracle row: {name} and {ORACLE_BASE} evaluated different test items")
    y = sources.positives_for(root, runs[0], per_seed[0])
    if not np.array_equal(y, sources.positives_for(root, base[0], per_seed_b[0])):
        raise RuntimeError("Table 4.1 oracle row: the oracle and the full pipeline carry different labels")
    n = len(ids)
    n_boot, lvl = sources.resamples(root, runs[0]), sources.level(runs[0])
    pred_o = [sources.arr(recs, "calibrated_probability") >= 0.5 for recs in per_seed]
    pred_b = [sources.arr(recs, "calibrated_probability") >= 0.5 for recs in per_seed_b]
    f1_o = [macro_f1_binary(y, p) for p in pred_o]
    f1_b = [macro_f1_binary(y, p) for p in pred_b]
    key = f"boot.table-4.1.{ORACLE_ROW}"
    lo, hi = sources.seeds_as_replicates_ci([(lambda idx, p=p: _f1_fast(y[idx], p[idx])) for p in pred_o], n,
                                            cell_key=f"{key}.macro_f1_binary", n_resamples=n_boot, level=lvl)

    def seed_mean(preds, idx):
        return float(np.mean([_f1_fast(y[idx], p[idx]) for p in preds]))

    delta = float(np.mean(f1_o) - np.mean(f1_b))
    d_lo, d_hi = ci_index(lambda idx: seed_mean(pred_o, idx) - seed_mean(pred_b, idx), n, n_resamples=n_boot,
                          seed=derive_seed(f"{key}.paired_delta"), level=lvl)
    ok_o = np.mean([p == y for p in pred_o], axis=0) > 0.5
    ok_b = np.mean([p == y for p in pred_b], axis=0) > 0.5
    table = [[int(np.sum(ok_b & ok_o)), int(np.sum(ok_b & ~ok_o))], [int(np.sum(~ok_b & ok_o)), int(np.sum(~ok_b & ~ok_o))]]
    mcn = mcnemar_exact(table)
    prov = sources.provenance(runs + base)
    common = dict(n_items=n, n_seeds=len(runs), seeds=sources.seeds_text(runs), family=FAM_ORACLE, **prov)
    seeds_note = f"seeds {sources.seeds_text(runs)} of {name} paired with the same seeds of {ORACLE_BASE}"
    return [
        fmt.long_row(TABLE, ORACLE_ROW, ORACLE_COLS[0][0], "macro_f1_binary", fmt.fmt_ci(float(np.mean(f1_o)), lo, hi),
                     ci_lo=fmt.fmt_f1(lo), ci_hi=fmt.fmt_f1(hi),
                     test=(f"binary macro-F1 (calibrated P(Refuted) >= 0.5, D-04) on the chronological test set, percentile "
                           f"bootstrap (items; seeds as replicates; cell key {key}.macro_f1_binary); oracle ceiling, never a "
                           f"system result; {seeds_note}"), **common),
        fmt.long_row(TABLE, ORACLE_ROW, ORACLE_COLS[1][0], "macro_f1_4way", ORACLE_FOUR_WAY,
                     test="not computed: no four-way decision exists downstream of C2", **common),
        fmt.long_row(TABLE, ORACLE_ROW, ORACLE_COLS[2][0], "macro_f1_binary_diff", fmt.fmt_signed(delta, 3),
                     ci_lo=fmt.fmt_signed(d_lo, 3), ci_hi=fmt.fmt_signed(d_hi, 3), effect_size=fmt.fmt_signed(delta, 3),
                     effect_size_kind="seed-mean binary macro-F1 difference, oracle minus full pipeline",
                     test=(f"paired percentile item bootstrap ({n_boot} resamples, one index resample for both systems and "
                           f"every seed; cell key {key}.paired_delta); full pipeline seed-mean "
                           f"{fmt.fmt_f1(float(np.mean(f1_b)))}; {seeds_note}"), **common),
        fmt.long_row(TABLE, ORACLE_ROW, ORACLE_COLS[3][0], "mcnemar_seed_majority", f"{mcn.p:.6f}", p_raw=f"{mcn.p:.6f}",
                     test=(f"exact McNemar on the per-item outcome correct, taken over seeds by majority; table "
                           f"[[both, full pipeline only], [oracle only, neither]] = {table}; supplementary (family "
                           f"{FAM_ORACLE}, outside the R1 Holm family); {seeds_note}"), **common),
    ]


def build(root: Path, out_dir: Path | None = None, allow_pending: bool = False) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    cells = rows_data(root)
    shell_rows, long_rows = [], []
    all_runs = [r for c in cells for r in c["runs"]]
    for c in cells:
        prov = sources.provenance(c["runs"])
        repro, point = _fmt_point(c)
        pub = published_cell(c["spec"])
        dev = deviation_cell(c)
        shell_rows.append({"comparison_point": c["row"], "source": c["spec"]["source"], "metric": c["spec"]["metric"],
                           "published": pub, "reproduced": repro, "deviation": dev})
        common = dict(n_items=c["n_items"], n_seeds=c.get("n_seeds", len(c["runs"])),
                      seeds=c.get("seeds", sources.seeds_text(c["runs"])), **prov)
        L = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, c["row"], col, metric, value, **{**common, **kw}))
        L("Source", "source", c["spec"]["source"])
        L("Metric", "metric", c["spec"]["metric"])
        L("Published", "published_" + c["unit"], pub,
          test="analyst-transcribed (configs/published_baselines.yaml)" + (f"; page_ref {c['spec']['page_ref']}" if c["spec"].get("page_ref") else "; page_ref absent"))
        unit_lo, unit_hi = ((fmt.fmt_rate(c["lo"]), fmt.fmt_rate(c["hi"])) if c["unit"] == "percent"
                            else (fmt.fmt_f1(c["lo"]), fmt.fmt_f1(c["hi"])))
        note = c.get("not_comparable")
        L("Reproduced [95% CI]", c["metric_name"], point, ci_lo=unit_lo, ci_hi=unit_hi,
          test=c["test"] + (f"; {note}" if note else ""))
        L("Deviation", "reproduced_minus_published", dev, **({"test": f"not computed: {note}"} if note else {}))
        for col, metric, value, kw in c.get("extras", []):
            L(col, metric, value, **kw)
        long_rows.extend(c.get("chunk_rows", []))   # pooled row 4 only (methodological note)
    prov_all = sources.provenance(all_runs)
    if cells[0]["plan"] == "real":  # targeted runs (row 87) and addition C : long rows only; never stub
        long_rows.extend(targeted_rows(root))
        long_rows.extend(oracle_rows(root, prov_all["data_version"], allow_pending))
    written = list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=prov_all["config_hash"],
                            data_version=prov_all["data_version"], out_dir=out_dir, allow_pending=allow_pending))
    if cells[0]["plan"] == "real":  # the stub namespace gets no published-targets view (its goldens stay unchanged)
        written.append(write_published_targets(root, prov_all["data_version"], out_dir))
    return written
