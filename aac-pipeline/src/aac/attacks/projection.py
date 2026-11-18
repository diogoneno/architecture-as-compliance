"""reports/r3-budget-projection.csv (measurement) from ONE sealed R3 run.

 python -m aac.attacks.projection --run-id <R3 run> [--out PATH] (default: reports_dir/r3-budget-projection.csv)
 python -m aac.attacks.projection --check PATH

The run is read from runs_dir (AAC_OUT selects the namespace) after verify_sealed and a strict L2 check; an
unsealed run or an L2 failure is refused. Only these files are read, never a clock or a typed value:

 sealed run run-record.json, config.resolved.json, attack-ledger.jsonl, attack-adversarial.jsonl,
 attack-counters.jsonl, attack-timings.json, decision-log/part-*.jsonl.gz (sizes and line bytes)
 committed data/measured/phase2-gpu-session.json and, when present, data/measured/phase3-gpu-session.json
 and data/measured/phase3-pilot-session.json, then data/measured/phase4-gpu-*.json in sorted order
 (session_gpu_hours; a session file without that key
 is still in progress: it is skipped and named in the source cell); reports/long/table-4.1-baselines.long.csv (BODEGA queries per example);
 data/derived/splits/averitec-attack-subset-v1.json (label_counts, ids);
 data/derived/averitec/labels.csv (labels of the cut-2 ids); data/derived/incrediblae/
 {fc,rd}-attack.items.json (label 1 counts); configs/cuts.yaml (cut-2-subset-200 action)

Output (LF, ASCII, deterministic: the same sealed run gives the same bytes):

 # config_hash=<R3 config_hash> data_version=<R3 data_version> run=<R3 run_id>
 section,attack,victim,subset,metric,value,unit,source

section `measured` (per pilot cell, and session rows under attack/victim/subset `all`) or `projection` (per
(attack, victim, subset) of the Phase 4 battery, then the totals under `all`). A unit ending in ` [L]` marks an
estimate. A value that cannot be computed reads `not measured: <reason>`; never blank, NaN or a placeholder.
Precision (pre-registered specification;
counts and queries as integers. Imports: stdlib, PyYAML and aac.attacks.ledger / aac.run (no aac.components)."""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import re
import sys
from pathlib import Path

REPORT_FILE = "r3-budget-projection.csv"
COLUMNS = ("section", "attack", "victim", "subset", "metric", "value", "unit", "source")
SECTIONS = ("measured", "projection")
ALL = "all"
HEADER_RE = re.compile(r"^# config_hash=[0-9a-f]{64} data_version=[0-9a-f]{64} run=R3-[0-9a-f]{10}-s\d+$")
NOT_MEASURED = "not measured"
ESTIMATE = " [L]"


PHASE4_ATTACKS = ("bert_attack", "deepwordbug", "pwws", "textfooler", "beamattack")
PHASE4_CELLS = (("content_only", "averitec_chrono"), ("content_only", "fc"), ("content_only", "rd"),
                ("full", "averitec_chrono"))
CUT2_SUBSET = "averitec_chrono_cut2"     # victim B on the first N ids of the attack subset (configs/cuts.yaml)
CUT2_STEP = "cut-2-subset-200"

PHASE4_BUDGET = 1000
PHASE4_BUDGET_SOURCE = "Methodology specification (1,000 queries per item)"
# Project hardware envelope: 50 to 75 GPU-hours in total (PROJECT.md constraints; Table D.2 'of the 50 to 75 planned').
ENVELOPE = (50, 75)
ENVELOPE_SOURCE = "PROJECT.md constraints: 50 to 75 GPU-hours total (Table D.2 'of the 50 to 75 planned')"
BODEGA_NAMES = {"bert_attack": "BERT-Attack", "textfooler": "TextFooler", "deepwordbug": "DeepWordBug", "pwws": "PWWS"}
BODEGA_SUBSETS = {"fc": "FC", "rd": "RD"}

MEASURED_SESSIONS = ("data/measured/phase2-gpu-session.json", "data/measured/phase3-gpu-session.json",
                     "data/measured/phase3-pilot-session.json")
PHASE4_SESSIONS_GLOB = "phase4-gpu-*.json"   
TABLE41 = "reports/long/table-4.1-baselines.long.csv"
ATTACK_SUBSET = "data/derived/splits/averitec-attack-subset-v1.json"
AVERITEC_LABELS = "data/derived/averitec/labels.csv"
ITEMS = {"fc": "data/derived/incrediblae/fc-attack.items.json", "rd": "data/derived/incrediblae/rd-attack.items.json"}
CUTS = "configs/cuts.yaml"
POSITIVE = "Refuted"

CELL_METRICS = ("items_attacked", "items_skipped_wrong_clean", "queries_median", "queries_p90", "queries_max",
                "share_at_budget", "server_seconds_per_query", "rtt_seconds_per_query", "socket_overhead_share",
                "runner_seconds_per_item", "runner_seconds_per_query")
SESSION_METRICS = ("pilot_budget", "runner_device", "victim_device", "session_seconds", "sessions",
                   "run_record_gpu_hours", "records", "records_per_second", "bytes_per_record_gzip",
                   "bytes_per_record_raw")
PROJ_CELL_METRICS = ("attacked_n_upper", "seconds_per_item", "gpu_wall_hours")
TOTAL_METRICS = ("phase4_gpu_wall_hours", "phase4_gpu_wall_hours_cut_2", "used_gpu_hours", "remaining_at_50",
                 "remaining_at_75", "fits_50", "fits_75", "fits_50_cut_2", "fits_75_cut_2")
METRICS = frozenset(CELL_METRICS + SESSION_METRICS + PROJ_CELL_METRICS + TOTAL_METRICS)
ESTIMATE_METRICS = frozenset({"gpu_wall_hours", "phase4_gpu_wall_hours", "phase4_gpu_wall_hours_cut_2",
                              "fits_50", "fits_75", "fits_50_cut_2", "fits_75_cut_2"})
ATTACK_VALUES = frozenset(PHASE4_ATTACKS + (ALL,))
VICTIM_VALUES = frozenset(("content_only", "full", ALL))
SUBSET_VALUES = frozenset(("averitec_chrono", "fc", "rd", CUT2_SUBSET, ALL))
VALUE_RE = re.compile(r"^(-?[0-9]+(\.[0-9]+)?|true|false|cpu|cuda|not measured: [a-z0-9 ,;:()_./=-]+)$")
FORBIDDEN_CELLS = frozenset({"nan", "-nan", "inf", "-inf", "none", "null", "?", "n/a", "tbd", "todo"})


class ProjectionRefused(RuntimeError):
    """The R3 run is not sealed, its L2 check fails, or it is not an R3 run."""


# ---------------------------------------------------------------------------------------------- formatting
def _f6(x: float) -> str:
    s = f"{float(x):.6f}"
    return "0.000000" if s == "-0.000000" else s


def _pct(x: float) -> str:
    s = f"{100.0 * float(x):.1f}"
    return "0.0" if s == "-0.0" else s


def _nearest_rank(values: list[int], num: int, den: int) -> int:
    """Nearest-rank percentile num/den of a non-empty list, index ceil(n * num / den) - 1 in integers."""
    v = sorted(values)
    n = len(v)
    return v[max(0, (n * num + den - 1) // den - 1)]


def _row(section, attack, victim, subset, metric, value, unit, source) -> tuple:
    return (section, attack, victim, subset, metric, value, unit, source)


# ---------------------------------------------------------------------------------------------- readers
def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _bodega_queries(root: Path) -> dict[tuple[str, str], tuple[float, str]]:
    """(attack, subset) -> (BODEGA queries per example, run_ids) from the Table 4.1 long rows that hold one."""
    path = root / TABLE41
    out: dict[tuple[str, str], tuple[float, str]] = {}
    if not path.is_file():
        return out
    lines = [x for x in path.read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
    for r in csv.DictReader(lines):
        if r.get("metric") != "queries_per_example":
            continue
        for atk, name in BODEGA_NAMES.items():
            for sub, label in BODEGA_SUBSETS.items():
                if r["row_label"] == f"BERT victim, {label}, {name}":
                    out[(atk, sub)] = (float(r["value"]), r["run_ids"])
    return out


def _cut2_size(root: Path) -> tuple[int, int]:
    """(from, to) of the cut-2 action text in configs/cuts.yaml ('... from 400 to 200 items')."""
    import yaml

    data = yaml.safe_load((root / CUTS).read_text(encoding="utf-8"))
    for step in data["cuts"]["steps"]:
        if step["id"] == CUT2_STEP:
            m = re.search(r"from (\d+) to (\d+) items", step["action"])
            if not m:
                raise ValueError(f"{CUTS}: step {CUT2_STEP} action names no 'from N to M items'")
            return int(m.group(1)), int(m.group(2))
    raise KeyError(f"{CUTS} has no step {CUT2_STEP}")


def _attacked_upper(root: Path) -> dict[str, tuple[int, str]]:
    """subset -> (positive count, source) for the Phase 4 attack sets (targeted: positives only)."""
    sub = _json(root / ATTACK_SUBSET)
    out = {"averitec_chrono": (int(sub["label_counts"][POSITIVE]),
                               f"{ATTACK_SUBSET} label_counts.{POSITIVE} (of {int(sub['count'])} ids)")}
    for name, rel in ITEMS.items():
        items = _json(root / rel)
        out[name] = (sum(int(it["label"]) == 1 for it in items), f"{rel} items with label 1 (of {len(items)})")
    n_from, n_to = _cut2_size(root)
    labels = {}
    with open(root / AVERITEC_LABELS, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            labels[r["claim_id"]] = r["label"]
    first = list(sub["ids"])[:n_to]
    missing = [i for i in first if i not in labels]
    if missing:
        raise ValueError(f"{AVERITEC_LABELS}: {len(missing)} cut-2 ids have no label")
    out[CUT2_SUBSET] = (
        sum(labels[i] == POSITIVE for i in first),
        (
            f"{ATTACK_SUBSET} first {n_to} ids ({CUTS} {CUT2_STEP}: from {n_from} to {n_to}) "
            f"labeled {POSITIVE} in {AVERITEC_LABELS}"
        ),
    )
    return out


def measured_sessions(root: Path) -> list[str]:
    """The session files read for used_gpu_hours: MEASURED_SESSIONS in order, then data/measured/phase4-gpu-*.json
 sorted by name (operational constraint: otherwise used_gpu_hours stays at the Phase 3 value)."""
    phase4 = sorted(f"data/measured/{p.name}" for p in (Path(root) / "data/measured").glob(PHASE4_SESSIONS_GLOB))
    return list(MEASURED_SESSIONS) + [rel for rel in phase4 if rel not in MEASURED_SESSIONS]


def _used_gpu_hours(root: Path) -> tuple[float, str]:
    """Sum of session_gpu_hours over the session files that exist. A file without a numeric session_gpu_hours is a
 session still in progress (the key is written when the session ends): it adds nothing and the source names it."""
    total, used, open_ = 0.0, [], []
    for rel in measured_sessions(root):
        p = root / rel
        if not p.is_file():
            continue
        hours = _json(p).get("session_gpu_hours")
        if isinstance(hours, bool) or not isinstance(hours, (int, float)):
            open_.append(rel)
            continue
        total += float(hours)
        used.append(rel)
    if not used:
        raise FileNotFoundError(f"no finished session in {', '.join(measured_sessions(root))}"
                                + (f" (in progress: {', '.join(open_)})" if open_ else ""))
    src = "sum of session_gpu_hours in " + " + ".join(used)
    if open_:
        src += "; in progress, not counted: " + " + ".join(open_)
    return total, src


# ---------------------------------------------------------------------------------------------- build
def _open_run(root: Path, run_id: str) -> Path:
    from aac.attacks.ledger import l2_check
    from aac.run.paths import runs_dir
    from aac.run.seal import UnsealedRun, verify_sealed

    run_dir = runs_dir(root) / run_id
    if not run_dir.is_dir():
        raise ProjectionRefused(f"{run_id}: no run directory in {runs_dir(root)}")
    try:
        verify_sealed(run_dir)
    except UnsealedRun as exc:
        raise ProjectionRefused(f"{run_id}: not sealed ({exc})") from None
    rec = _json(run_dir / "run-record.json")
    if rec.get("regime") != "R3" or rec.get("run_id") != run_id:
        raise ProjectionRefused(f"{run_id}: not an R3 run record")
    l2 = l2_check(run_dir, strict=True)
    if not l2.ok:
        raise ProjectionRefused(f"{run_id}: {l2.as_line()}")
    return run_dir


def _measured_rows(run_dir: Path, budget: int) -> tuple[list[tuple], dict]:
    from aac.attacks.ledger import read_adversarial, read_ledger

    ledger = read_ledger(run_dir)
    adv = {(a["item_id"], a["attack"], a["victim"], a["subset"]): a for a in read_adversarial(run_dir)}
    counters = {(c["item_id"], c["attack"], c["victim"], c["subset"]): c for c in _jsonl(run_dir / "attack-counters.jsonl")}
    cells: dict[tuple, list[tuple[dict, dict, dict]]] = {}
    for row in ledger:
        key = (row["item_id"], row["attack"], row["victim"], row["subset"])
        if key not in counters or key not in adv:
            raise ProjectionRefused(f"{run_dir.name}: ledger row without counters or adversarial row")
        cells.setdefault((row["attack"], row["victim"], row["subset"]), []).append((row, counters[key], adv[key]))
    rows: list[tuple] = []
    rates: dict[tuple, dict] = {}
    src_l = "attack-ledger.jsonl"
    src_c = "attack-counters.jsonl"
    for (attack, victim, subset), items in sorted(cells.items()):
        head = ("measured", attack, victim, subset)
        attacked = [(r, c) for r, c, a in items if a["attacked"]]
        q = [int(r["queries_used"]) for r, _ in attacked]
        n_records = sum(len(r["record_ids"]) for r, _, _ in items)
        server_s = sum(float(c["server_seconds"]) for _, c, _ in items)
        rtt_s = sum(float(c["rtt_seconds"]) for _, c in attacked)
        runner_q = sum(int(c["runner_queries"]) for _, c in attacked)
        runner_s = sum(float(c["runner_seconds"]) for _, c in attacked)
        none = f"{NOT_MEASURED}: 0 items attacked"
        rows += [head + ("items_attacked", str(len(attacked)), "items", "attack-adversarial.jsonl attacked=true"),
                 head + ("items_skipped_wrong_clean", str(len(items) - len(attacked)), "items",
                         "attack-adversarial.jsonl attacked=false (clean decision already wrong)")]
        if q:
            exhausted = sum(r["end_state"] == "budget_exhausted" for r, _ in attacked)
            rows += [head + ("queries_median", str(_nearest_rank(q, 1, 2)), "queries (nearest rank)",
                             f"{src_l} queries_used of attacked items"),
                     head + ("queries_p90", str(_nearest_rank(q, 9, 10)), "queries (nearest rank)",
                             f"{src_l} queries_used of attacked items"),
                     head + ("queries_max", str(max(q)), "queries", f"{src_l} queries_used of attacked items"),
                     head + ("share_at_budget", _pct(exhausted / len(q)), f"percent (budget {budget})",
                             f"{src_l} end_state budget_exhausted / attacked items")]
        else:
            rows += [head + (m, none, u, f"{src_l} queries_used of attacked items") for m, u in
                     (("queries_median", "queries (nearest rank)"), ("queries_p90", "queries (nearest rank)"),
                      ("queries_max", "queries"), ("share_at_budget", f"percent (budget {budget})"))]
        srv = server_s / n_records if n_records else None
        rtt = rtt_s / runner_q if runner_q else None
        rps = runner_s / runner_q if runner_q else None
        rows.append(head + ("server_seconds_per_query", _f6(srv) if srv is not None else f"{NOT_MEASURED}: 0 records",
                            "s/query", f"{src_c} server_seconds (clean, attack, post-hoc) / {src_l} record_ids"))
        rows.append(head + ("rtt_seconds_per_query", _f6(rtt) if rtt is not None else f"{NOT_MEASURED}: 0 attack queries",
                            "s/query", f"{src_c} rtt_seconds / runner_queries (attack phase, runner clock)"))
        if srv is not None and rtt is not None and rtt > 0:
            share = _pct((rtt - srv) / rtt)
        else:
            share = f"{NOT_MEASURED}: 0 attack queries"
        rows.append(head + ("socket_overhead_share", share, "percent",
                            "(rtt_seconds_per_query - server_seconds_per_query) / rtt_seconds_per_query"))
        rows.append(head + ("runner_seconds_per_item", _f6(runner_s / len(attacked)) if attacked else none, "s/item",
                            f"{src_c} runner_seconds of attacked items (clean query included)"))
        rows.append(head + ("runner_seconds_per_query", _f6(rps) if rps is not None else f"{NOT_MEASURED}: 0 attack queries",
                            "s/query", f"{src_c} runner_seconds / runner_queries of attacked items"))
        rates[(attack, victim, subset)] = {"runner_s": runner_s, "runner_q": runner_q, "n_attacked": len(attacked)}
    return rows, rates


def _session_rows(run_dir: Path, resolved: dict, rec: dict) -> list[tuple]:
    timings = _json(run_dir / "attack-timings.json")
    head = ("measured", ALL, ALL, ALL)
    session = float(timings["session_seconds"])
    records = int(rec["record_count"])
    shards = sorted((run_dir / "decision-log").glob("part-*.jsonl.gz"))
    gz_bytes = sum(p.stat().st_size for p in shards)
    raw_bytes = 0
    for p in shards:
        with gzip.open(p, "rb") as g:
            raw_bytes += len(g.read())
    devices = sorted({s.get("device", "cpu") for s in rec.get("stages") or []}) or ["cpu"]
    none_r = f"{NOT_MEASURED}: 0 records"
    return [
        head + ("pilot_budget", str(int(resolved["attack"]["budget"])), "queries per item", "config.resolved.json attack.budget"),
        head + ("runner_device", str(resolved["attack"].get("runner_device", "cpu")), "device",
                "config.resolved.json attack.runner_device"),
        head + ("victim_device", devices[0] if len(devices) == 1 else devices[-1], "device", "run-record.json stages[].device"),
        head + ("session_seconds", _f6(session), "s", "attack-timings.json session_seconds (first query to shutdown)"),
        head + ("sessions", str(int(timings.get("sessions", 1))), "count", "attack-timings.json sessions (1 + resumes)"),
        head + ("run_record_gpu_hours", _f6(float(rec["gpu_hours"])), "GPU-hours", "run-record.json gpu_hours"),
        head + ("records", str(records), "records", "run-record.json record_count"),
        head + ("records_per_second", _f6(records / session) if session > 0 else f"{NOT_MEASURED}: session 0 s",
                "records/s", "run-record.json record_count / attack-timings.json session_seconds"),
        head + ("bytes_per_record_gzip", _f6(gz_bytes / records) if records else none_r, "bytes/record",
                "decision-log/part-*.jsonl.gz file sizes / record_count"),
        head + ("bytes_per_record_raw", _f6(raw_bytes / records) if records else none_r, "bytes/record",
                "decision-log/part-*.jsonl.gz decompressed bytes / record_count"),
    ]


def _projection_rows(root: Path, rates: dict, pilot_budget: int) -> list[tuple]:
    upper = _attacked_upper(root)
    bodega = _bodega_queries(root)
    used, used_src = _used_gpu_hours(root)

    def pooled(pred) -> tuple[float, str] | None:
        s = sum(v["runner_s"] for k, v in rates.items() if pred(k))
        q = sum(v["runner_q"] for k, v in rates.items() if pred(k))
        return (s / q, f"{q} queries") if q else None

    rows: list[tuple] = []
    hours: dict[tuple[str, str, str], float | None] = {}
    cells = [(a, v, s) for a in PHASE4_ATTACKS for v, s in PHASE4_CELLS] + \
            [(a, "full", CUT2_SUBSET) for a in PHASE4_ATTACKS]
    for attack, victim, subset in cells:
        head = ("projection", attack, victim, subset)
        n, n_src = upper[subset]
        rows.append(head + ("attacked_n_upper", str(n), "items (upper bound: positives)", n_src))
        pilot_subset = "averitec_chrono" if subset == CUT2_SUBSET else subset
        own = rates.get((attack, victim, pilot_subset))
        sec: float | None
        if own and own["n_attacked"] and own["runner_q"] and pilot_budget == PHASE4_BUDGET:
            sec = own["runner_s"] / own["n_attacked"]
            unit, src = "s/item", f"measured: runner_seconds_per_item of pilot cell {attack}/{victim}/{pilot_subset}"
        else:
            if own and own["runner_q"]:
                spq, spq_src = own["runner_s"] / own["runner_q"], f"pilot cell {attack}/{victim}/{pilot_subset}"
            else:
                # pooled over the pilot cells of the SAME victim only: victim A and victim B differ in cost by orders of
                # magnitude (evidence recompute), so one victim's rate never stands in for the other's
                got = pooled(lambda k, v=victim: k[1] == v)
                spq, spq_src = (got[0], f"pilot runner_seconds per query of victim {victim} pooled ({got[1]})") \
                    if got else (None, "")
            ref = bodega.get((attack, subset))
            if ref is not None:
                qn = min(ref[0], float(PHASE4_BUDGET))
                q_src = f"BODEGA queries per example {ref[0]} ({TABLE41}, run {ref[1]}), capped at the budget"
            else:
                qn = float(PHASE4_BUDGET)
                q_src = f"budget {PHASE4_BUDGET} as the upper bound ({PHASE4_BUDGET_SOURCE})"
            unit = "s/item" + ESTIMATE
            if spq is None:
                sec, src = None, f"no attack query of victim {victim} in the pilot"
            else:
                sec, src = spq * qn, f"{spq_src} x {q_src}"
        sec_cell = _f6(sec) if sec is not None else f"{NOT_MEASURED}: no attack query of victim {victim} in the pilot"
        rows.append(head + ("seconds_per_item", sec_cell, unit, src))
        h = None if sec is None else n * sec / 3600.0
        hours[(attack, victim, subset)] = h
        rows.append(head + ("gpu_wall_hours", _f6(h) if h is not None else f"{NOT_MEASURED}: no seconds per item",
                            "hours" + ESTIMATE, "attacked_n_upper x seconds_per_item / 3600 (session holds the GPU)"))

    head = ("projection", ALL, ALL, ALL)
    main = [hours[(a, v, s)] for a in PHASE4_ATTACKS for v, s in PHASE4_CELLS]
    cut2 = [hours[(a, v, CUT2_SUBSET if (v, s) == ("full", "averitec_chrono") else s)]
            for a in PHASE4_ATTACKS for v, s in PHASE4_CELLS]
    total = None if any(h is None for h in main) else sum(main)
    total_c2 = None if any(h is None for h in cut2) else sum(cut2)
    nm = f"{NOT_MEASURED}: a cell has no seconds per item"
    rows += [head + ("phase4_gpu_wall_hours", _f6(total) if total is not None else nm, "hours" + ESTIMATE,
                     "sum of gpu_wall_hours over the 20 Phase 4 cells"),
             head + ("phase4_gpu_wall_hours_cut_2", _f6(total_c2) if total_c2 is not None else nm, "hours" + ESTIMATE,
                     f"as phase4_gpu_wall_hours with victim B on {CUT2_SUBSET} ({CUTS} {CUT2_STEP})"),
             head + ("used_gpu_hours", _f6(used), "GPU-hours", used_src)]
    for cap in ENVELOPE:
        rem = cap - used
        rows.append(head + (f"remaining_at_{cap}", _f6(rem), "GPU-hours", f"{cap} - used_gpu_hours ({ENVELOPE_SOURCE})"))
    for cap in ENVELOPE:
        rem = cap - used
        for metric, t in ((f"fits_{cap}", total), (f"fits_{cap}_cut_2", total_c2)):
            base = "phase4_gpu_wall_hours" + ("_cut_2" if metric.endswith("_cut_2") else "")
            rows.append(head + (metric, ("true" if t <= rem else "false") if t is not None else nm, "bool" + ESTIMATE,
                                f"{base} <= remaining_at_{cap}"))
    return rows


def build(root: Path, run_id: str, out_path: Path) -> Path:
    """Write the projection CSV of the sealed R3 run `run_id` (runs_dir(root)) to out_path; return out_path."""
    root = Path(root)
    run_dir = _open_run(root, run_id)
    rec = _json(run_dir / "run-record.json")
    resolved = _json(run_dir / "config.resolved.json")
    budget = int(resolved["attack"]["budget"])
    measured, rates = _measured_rows(run_dir, budget)
    rows = measured + _session_rows(run_dir, resolved, rec) + _projection_rows(root, rates, budget)
    rows.sort(key=lambda r: r[:5])
    buf = io.StringIO(newline="")
    buf.write(f"# config_hash={rec['config_hash']} data_version={rec['data_version']} run={run_id}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    w.writerows(rows)
    data = buf.getvalue().encode("ascii")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(out_path)
    problems = check_projection_csv(out_path)
    if problems:
        raise ValueError(f"{out_path}: projection CSV fails its own contract: {problems[0]}")
    return out_path


# ---------------------------------------------------------------------------------------------- check
def check_projection_csv(path: Path) -> list[str]:
    """One message per violation of the CSV shape rules; an empty list means valid."""
    path = Path(path)
    if not path.is_file():
        return [f"{path}: no such file"]
    raw = path.read_bytes()
    problems: list[str] = []
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError:
        return ["file is not ASCII"]
    if "\r" in text:
        problems.append("line ends are not LF only")
    lines = text.split("\n")
    if not text.endswith("\n"):
        problems.append("file does not end with a newline")
    if not lines or not HEADER_RE.match(lines[0]):
        problems.append("first line is not '# config_hash=<64 hex> data_version=<64 hex> run=<R3 run id>'")
    rows = list(csv.reader(lines[1:]))
    rows = [r for r in rows if r]
    if not rows or tuple(rows[0]) != COLUMNS:
        problems.append("second line is not the header " + ",".join(COLUMNS))
        return problems
    body = rows[1:]
    keys = []
    for i, r in enumerate(body, start=3):
        if len(r) != len(COLUMNS):
            problems.append(f"line {i}: {len(r)} cells, expected {len(COLUMNS)}")
            continue
        d = dict(zip(COLUMNS, r))
        for col, val in d.items():
            if not val.strip():
                problems.append(f"line {i}: blank {col}")
            elif val.strip().lower() in FORBIDDEN_CELLS or "____" in val:
                problems.append(f"line {i}: placeholder in {col}: {val!r}")
        if d["section"] not in SECTIONS:
            problems.append(f"line {i}: unknown section {d['section']!r}")
        if d["attack"] not in ATTACK_VALUES:
            problems.append(f"line {i}: unknown attack {d['attack']!r}")
        if d["victim"] not in VICTIM_VALUES:
            problems.append(f"line {i}: unknown victim {d['victim']!r}")
        if d["subset"] not in SUBSET_VALUES:
            problems.append(f"line {i}: unknown subset {d['subset']!r}")
        if d["metric"] not in METRICS:
            problems.append(f"line {i}: unknown metric {d['metric']!r}")
        if not VALUE_RE.match(d["value"]):
            problems.append(f"line {i}: value {d['value']!r} is not a number, a bool, a device or 'not measured: ...'")
        if d["section"] == "projection" and d["metric"] in ESTIMATE_METRICS and not d["unit"].endswith(ESTIMATE):
            problems.append(f"line {i}: estimate {d['metric']} without the ' [L]' unit tag")
        keys.append(tuple(r[:5]))
    if keys != sorted(keys):
        problems.append("rows are not sorted by (section, attack, victim, subset, metric)")
    if len(set(keys)) != len(keys):
        problems.append("duplicate (section, attack, victim, subset, metric) rows")
    metrics = {k[4] for k in keys}
    for need in ("socket_overhead_share", "phase4_gpu_wall_hours", "used_gpu_hours"):
        if need not in metrics:
            problems.append(f"no {need} row")
    if not any(k[0] == "measured" for k in keys) or not any(k[0] == "projection" for k in keys):
        problems.append("a section has no row")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.attacks.projection", description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run-id")
    g.add_argument("--check", metavar="PATH")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.check:
        problems = check_projection_csv(Path(args.check))
        for p in problems:
            print(f"projection-check: {p}", file=sys.stderr)
        if not problems:
            n = sum(1 for _ in open(args.check, encoding="ascii")) - 2
            print(f"projection-check: {n} rows ok")
        return 1 if problems else 0
    from aac.run.paths import reports_dir, root as project_root

    root = project_root()
    out = Path(args.out) if args.out else reports_dir(root) / REPORT_FILE
    try:
        build(root, args.run_id, out)
    except ProjectionRefused as exc:
        print(f"projection: refused: {exc}", file=sys.stderr)
        return 2
    problems = check_projection_csv(out)
    if problems:
        for p in problems:
            print(f"projection: {p}", file=sys.stderr)
        return 1
    n = sum(1 for _ in open(out, encoding="ascii")) - 2
    print(f"projection: {out.name} {n} rows ok ({args.run_id})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
