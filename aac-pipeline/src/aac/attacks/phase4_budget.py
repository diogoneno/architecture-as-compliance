"""The binding Phase 4 stop rule (1, .

 python -m aac.attacks.phase4_budget --batch P4-1 [--batch P4-3 ...] [--root PATH]

For the named batches (ids P4-1, P4-3, P4-2a, P4-2b, P4-4, P4-5, P4-6; batch file configs/batches/phase4-<id>.txt in
lower case) it prints one line per batch line with its expected and high GPU-hours, the batch totals, the Phase 4
GPU-hours used so far, the project GPU-hours used so far, the totals against the Phase 4 ceiling and the 50 GPU-hour
project envelope, and as its LAST line exactly one of

 STOP-RULE: ok (exit 0)
 STOP-RULE: stop (<reason>) (exit 3: do not dispatch; request analyst review)

A batch line without an estimate, a missing batch file or an unreadable input exits 2 without a STOP-RULE line.

Inputs (read only; no clock, no typed value):
 estimates configs/batches/phase4-attacks.yaml (this plan), phase4-r4.yaml and
 phase4-additions.yaml when present, validated by aac.dispatch.plan.load_estimates; the
 Phase 4 ceiling is ceiling_gpu_hours of phase4-attacks.yaml (38.762137 = phase4_gpu_wall_hours [L] of
 reports/r3-budget-projection.csv); the project envelope is aac.attacks.projection.ENVELOPE[0] (50)
 projection once the measuring runs are sealed in runs/INDEX.json the estimate of a line is replaced by the
 rule projection rule of phase4-attacks.yaml: P4-2a lines after the sealed `r3-a` seed 0 (TextFooler
 n_B x mean_q_A x 1.9 x s/query; DeepWordBug n_B x (20.8 + 3) x s/query; PWWS and BERT-Attack n_B x the
 pilot victim-B runner s/item, all from reports/r3-budget-projection.csv); P4-2b lines after the sealed
 r3-b-first200-<recipe> seed 0 (n_last / n_first x its attack-timings.json session_seconds). n_B is
 label_counts.Refuted of the line's data.attack_subset file
 retry 0 when scripts/attack_session.sh carries the marker line (one shared deadline across attempts);
 allowance otherwise one attack.session_timeout_s window per attack line (the high estimate when it is null)
 Phase 4 used data/measured/phase4-gpu-*.json: per file, sum over batches.* of gpu_hours + dropped_gpu_hours (gpu_hours counts finished items only), or session_gpu_hours when that is larger (preflight and pull
 time on the card); a file with neither adds 0
 project used aac.attacks.projection._used_gpu_hours (finished sessions), with each Phase 4 file's share replaced by
 its Phase 4 used value, so an open Phase 4 session counts by its finished batches and nothing twice

Stop rule: Phase 4 used + batch high > ceiling, or project used + batch high > 50, where batch high = the sum of the
lines' high values + the retry allowance. Imports: stdlib, PyYAML, aac.dispatch.plan, aac.run (no aac.components)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from aac.attacks import projection as P

BATCH_IDS = ("P4-1", "P4-3", "P4-2a", "P4-2b", "P4-4", "P4-5", "P4-6")
ESTIMATE_FILES = ("configs/batches/phase4-attacks.yaml", "configs/batches/phase4-r4.yaml",
                  "configs/batches/phase4-additions.yaml")
CEILING_FILE = ESTIMATE_FILES[0]
SESSION_GLOB = "phase4-gpu-*.json"
MEASURED_DIR = "data/measured"
ATTACK_SESSION = "scripts/attack_session.sh"
RETRY_DEADLINE_MARKER = "one deadline for every attempt of this session"
PROJECTION_CSV = "reports/r3-budget-projection.csv"
POSITIVE = "Refuted"
# 1 constants of the projection rule (phase4-attacks.yaml projection_rule)
TRANSFER_FACTOR = 1.9          # pilot BERT-Attack B/A queries per item 112.6/59.2, the conservative transfer [L]
DWB_TOKENS = 20.8              # mean regex tokens of the subset's Refuted claims (computed over raw files)
DWB_EXTRA = 3                  # origin get_pred + one check + the final check (OpenAttack DeepWordBug mechanics [M])
EXIT_OK, EXIT_ERROR, EXIT_STOP = 0, 2, 3


class BudgetError(RuntimeError):
    """An input the verdict needs is missing or malformed (exit 2, no verdict)."""


def _f6(x: float) -> str:
    return P._f6(x)


def batch_file(batch_id: str) -> str:
    return f"configs/batches/phase4-{batch_id.lower()}.txt"


# ---------------------------------------------------------------------------------------------- inputs
def _estimates(root: Path) -> tuple[float, dict]:
    from aac.dispatch import plan

    ceiling = None
    index: dict[tuple, object] = {}
    for rel in ESTIMATE_FILES:
        path = root / rel
        if not path.is_file():
            if rel == CEILING_FILE:
                raise BudgetError(f"{rel} not found")
            continue
        est = plan.load_estimates(path)
        if rel == CEILING_FILE:
            ceiling = float(est.ceiling_gpu_hours)
        for e in est.runs:
            index[(e.target, e.config, e.seed, Path(e.batch).as_posix())] = e
    return float(ceiling), index


def _read_lines(root: Path, batch_id: str) -> list[tuple[str, str, int]]:
    from aac.dispatch import plan

    path = root / batch_file(batch_id)
    if not path.is_file():
        raise BudgetError(f"batch {batch_id}: {batch_file(batch_id)} not found")
    try:
        return [(t, c, s) for _, t, c, s in plan.read_batch(path)]
    except plan.PlanRefused as exc:
        raise BudgetError(str(exc)) from exc


def _raw(root: Path, config: str) -> dict:
    from aac.run.config import load_raw

    path = root / config
    if not path.is_file():
        raise BudgetError(f"config {config} not found")
    return load_raw(path)


def _refuted(root: Path, raw: dict, config: str) -> int:
    rel = (raw.get("data") or {}).get("attack_subset")
    if not rel or not (root / rel).is_file():
        raise BudgetError(f"{config}: data.attack_subset {rel!r} not found (n_B of the projection rule)")
    return int(json.loads((root / rel).read_text(encoding="utf-8"))["label_counts"][POSITIVE])


def _csv_value(root: Path, attack: str, victim: str, subset: str, metric: str) -> float:
    path = root / PROJECTION_CSV
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    for r in csv.DictReader(lines):
        if (r["section"], r["attack"], r["victim"], r["subset"], r["metric"]) == ("measured", attack, victim, subset,
                                                                                 metric):
            return float(r["value"])
    raise BudgetError(f"{PROJECTION_CSV}: no measured {attack}/{victim}/{subset} {metric} row")


def _sealed(root: Path, name: str, seed: int = 0) -> Path | None:
    from aac.run.index import current_sealed_run
    from aac.run.paths import runs_dir

    runs = runs_dir(root)
    rid = current_sealed_run(runs, "R3", name, seed)
    return (runs / rid) if rid else None


def _mean_queries_attacked(run_dir: Path, attack: str, victim: str, subset: str) -> tuple[float, int] | None:
    from aac.attacks.ledger import read_adversarial, read_ledger

    attacked = {(a["item_id"], a["attack"], a["victim"], a["subset"]) for a in read_adversarial(run_dir) if a["attacked"]}
    q = [int(r["queries_used"]) for r in read_ledger(run_dir)
         if (r["attack"], r["victim"], r["subset"]) == (attack, victim, subset)
         and (r["item_id"], r["attack"], r["victim"], r["subset"]) in attacked]
    return (sum(q) / len(q), len(q)) if q else None


# ---------------------------------------------------------------------------------------------- projection rule
def _projected(root: Path, batch_id: str, config: str, raw: dict) -> tuple[float, str] | None:
    """(GPU-hours, source) by the projection rule once the measuring runs are sealed, else None (keep the estimate)."""
    attacks = list((raw.get("attack") or {}).get("attacks") or [])
    if len(attacks) != 1:
        return None
    recipe = attacks[0]
    if batch_id == "P4-2a":
        run_a = _sealed(root, "r3-a")
        if run_a is None:
            return None
        n = _refuted(root, raw, config)
        spq = _csv_value(root, "bert_attack", "full", "averitec_chrono", "runner_seconds_per_query")
        if recipe == "textfooler":
            mq = _mean_queries_attacked(run_a, "textfooler", "content_only", "averitec_chrono")
            if mq is None:
                return None
            q, k = mq
            s = n * q * TRANSFER_FACTOR * spq
            return s / 3600, (f"projection rule: {n} x mean_q_A {q:.6f} ({k} attacked, {run_a.name}) x "
                              f"{TRANSFER_FACTOR} x {spq:.6f} s [L]")
        if recipe == "deepwordbug":
            s = n * (DWB_TOKENS + DWB_EXTRA) * spq
            return s / 3600, f"projection rule: {n} x ({DWB_TOKENS} + {DWB_EXTRA}) x {spq:.6f} s ({run_a.name} sealed) [L]"
        if recipe in ("pwws", "bert_attack"):
            spi = _csv_value(root, recipe, "full", "averitec_chrono", "runner_seconds_per_item")
            return n * spi / 3600, f"projection rule: {n} x {spi:.6f} s/item (pilot victim B) [L]"
        return None
    if batch_id == "P4-2b":
        name = str(raw.get("name") or Path(config).stem)
        if "last200" not in name:
            return None
        first = name.replace("last200", "first200")
        run_f = _sealed(root, first)
        if run_f is None:
            return None
        n_last = _refuted(root, raw, config)
        first_cfg = Path(config).parent.joinpath(first + ".yaml").as_posix()
        n_first = _refuted(root, _raw(root, first_cfg), first_cfg)
        secs = float(json.loads((run_f / "attack-timings.json").read_text(encoding="utf-8"))["session_seconds"])
        return (n_last / n_first) * secs / 3600, (f"projection rule: {n_last}/{n_first} x {secs:.6f} s session "
                                                  f"({run_f.name}) [L]")
    return None


# ---------------------------------------------------------------------------------------------- used hours
def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def phase4_used(root: Path) -> tuple[float, dict[str, float], list[str]]:
    """(total, {rel: used}, rels) over data/measured/phase4-gpu-*.json (sorted)."""
    per: dict[str, float] = {}
    for p in sorted((root / MEASURED_DIR).glob(SESSION_GLOB)):
        doc = json.loads(p.read_text(encoding="utf-8"))
        batches = doc.get("batches") if isinstance(doc.get("batches"), dict) else {}
        s = 0.0
        for b in batches.values():
            if isinstance(b, dict):
                s += sum(_num(b.get(k)) or 0.0 for k in ("gpu_hours", "dropped_gpu_hours"))
        session = _num(doc.get("session_gpu_hours"))
        per[f"{MEASURED_DIR}/{p.name}"] = max(s, session or 0.0)
    return sum(per.values()), per, sorted(per)


def project_used(root: Path, p4: dict[str, float]) -> tuple[float, str]:
    total, src = P._used_gpu_hours(root)
    counted = [rel for rel in _measured_rels(root) if rel in p4]
    for rel in counted:   
        total -= _num(json.loads((root / rel).read_text(encoding="utf-8")).get("session_gpu_hours")) or 0.0
    extra = sum(p4.values())
    if extra:
        src += f"; Phase 4 files by phase4_budget: {' + '.join(sorted(p4))}"
    return total + extra, src


def _measured_rels(root: Path) -> list[str]:
    fn = getattr(P, "measured_sessions", None)
    return list(fn(root)) if fn else list(P.MEASURED_SESSIONS)


def retry_marker(root: Path) -> bool:
    path = root / ATTACK_SESSION
    return path.is_file() and any(RETRY_DEADLINE_MARKER in line and line.lstrip().startswith("#")
                                  for line in path.read_text(encoding="utf-8").splitlines())


# ---------------------------------------------------------------------------------------------- verdict
def evaluate(root: Path, batch_ids: list[str]) -> tuple[list[str], bool]:
    ceiling, index = _estimates(root)
    envelope = float(P.ENVELOPE[0])
    marker = retry_marker(root)
    out: list[str] = []
    exp_total = high_total = retry_total = 0.0
    for bid in batch_ids:
        b_exp = b_high = b_retry = 0.0
        for target, config, seed in _read_lines(root, bid):
            est = index.get((target, config, seed, batch_file(bid)))
            if est is None:
                raise BudgetError(f"batch {bid}: no estimate for `{target} {config} {seed}` in "
                                  f"{', '.join(r for r in ESTIMATE_FILES if (root / r).is_file())}")
            raw = _raw(root, config)
            proj = _projected(root, bid, config, raw) if target == "attack" else None
            if proj is not None:
                lo = hi = proj[0]
                how = f"({proj[1]})"
            else:
                lo, hi = float(est.est_gpu_hours[0]), float(est.est_gpu_hours[1])
                how = f"(estimate) {est.source}"
            retry = 0.0
            if target == "attack" and not marker:
                t = (raw.get("attack") or {}).get("session_timeout_s")
                retry = float(t) / 3600 if _num(t) else hi
            b_exp, b_high, b_retry = b_exp + lo, b_high + hi, b_retry + retry
            out.append(f"  {bid} {target} {config} {seed}: expected {_f6(lo)} high {_f6(hi)} GPU-h {how}")
        mark = ("Retry deadline marker present in " + ATTACK_SESSION) if marker else \
               ("Retry deadline marker absent from " + ATTACK_SESSION + ": one session_timeout_s window per attack line")
        out.append(f"phase4-budget: {bid} expected={_f6(b_exp)} high={_f6(b_high)} retry_allowance={_f6(b_retry)} "
                   f"({mark})")
        exp_total, high_total, retry_total = exp_total + b_exp, high_total + b_high, retry_total + b_retry
    used4, per, rels = phase4_used(root)
    out.append(f"phase4-budget: phase4 used {_f6(used4)} GPU-h "
               + (f"({' + '.join(f'{r} {_f6(per[r])}' for r in rels)})" if rels
                  else f"(no {MEASURED_DIR}/{SESSION_GLOB})"))
    usedp, srcp = project_used(root, per)
    out.append(f"phase4-budget: project used {_f6(usedp)} GPU-h ({srcp})")
    batch_high = high_total + retry_total
    t4, tp = used4 + batch_high, usedp + batch_high
    out.append(f"phase4-budget: phase4_used={_f6(used4)} batch_high={_f6(batch_high)} total={_f6(t4)} of {_f6(ceiling)}; "
               f"project_used={_f6(usedp)} total={_f6(tp)} of {envelope:g}")
    reasons = []
    if t4 > ceiling:
        reasons.append(f"Phase 4 total {_f6(t4)} > ceiling {_f6(ceiling)}")
    if tp > envelope:
        reasons.append(f"project total {_f6(tp)} > envelope {envelope:g}")
    out.append("STOP-RULE: ok" if not reasons else f"STOP-RULE: stop ({'; '.join(reasons)}; ask the analyst for a fresh go)")
    return out, not reasons


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.attacks.phase4_budget", description=__doc__.split("\n", 1)[0])
    ap.add_argument("--batch", action="append", required=True, choices=BATCH_IDS,
                    help="batch id (repeatable; the projection sums every named batch)")
    ap.add_argument("--root", default=None, help="project root (default: the aac-pipeline checkout)")
    args = ap.parse_args(argv)
    from aac.run.paths import root as project_root

    root = Path(args.root) if args.root else project_root()
    try:
        lines, ok = evaluate(root, list(args.batch))
    except (RuntimeError, FileNotFoundError, ValueError, KeyError) as exc:   # BudgetError, StaleRun
        print(f"phase4-budget: refused: {exc}", file=sys.stderr)
        return EXIT_ERROR
    for line in lines:
        print(line)
    return EXIT_OK if ok else EXIT_STOP


if __name__ == "__main__":
    sys.exit(main())
