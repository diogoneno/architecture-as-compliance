"""reports/bench-bge-m3.csv from a sealed BENCH run, the C1 determinism double run and
the recomputed Table F.2 indexing projection.

`python -m aac.bench.report --bench <name> --seed N [--c1-pair A B] [--r1-pair A B] [--out FILE]` reads the
runs of the selected namespace (runs_dir, AAC_OUT) after verify_sealed and writes one long CSV:

 # config_hash=<config_hash of every run read, ;-joined> data_version=<their shared data_version>
 section,model,precision,batch_size,metric,value,unit

sections throughput, latency, bit_identity, c1_double_run, f2_projection. A measurement that was not taken
is written as such (`not measured`, `skipped: cpu`), never as a number. torch-free."""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from pathlib import Path

from aac.run import paths as aac_paths

REPORT_FILE = "bench-bge-m3.csv"
COLUMNS = ("section", "model", "precision", "batch_size", "metric", "value", "unit")
SECTIONS = ("throughput", "latency", "bit_identity", "c1_double_run", "f2_projection")
REQUIRED_SECTIONS = ("throughput", "latency", "bit_identity", "f2_projection")
HEADER_RE = re.compile(r"^# config_hash=[0-9a-f]{64}(;[0-9a-f]{64})* data_version=[0-9a-f]{64}$")
NOT_MEASURED = "not measured"
FULL_PRECISION = frozenset({"max_abs_diff", "min_cosine"})
DEFAULT_BENCH = "bench-phase2"
DEFAULT_C1_PAIR = ("c1-chrono", "c1-chrono-repeat")
DEFAULT_R1_PAIR = ("r1-chrono-content-only", "r1-chrono-content-only-repeat")


class AmbiguousRun(LookupError):
    """More than one indexed run matches (regime, name, seed)."""


def find_run(root: Path, regime: str, name: str, seed: int) -> Path | None:
    """The sealed run directory of (regime, name, seed) in runs_dir/INDEX.json, verified; None if absent."""
    from aac.run.index import load_index
    from aac.run.seal import verify_sealed

    runs = aac_paths.runs_dir(root)
    hits = sorted(r["run_id"] for r in load_index(runs / "INDEX.json")["runs"]
                  if r.get("regime") == regime and r.get("name") == name and int(r.get("seed", -1)) == int(seed))
    if not hits:
        return None
    if len(hits) > 1:
        raise AmbiguousRun(f"{regime}/{name} seed {seed}: {len(hits)} indexed runs ({', '.join(hits)})")
    run_dir = runs / hits[0]
    verify_sealed(run_dir)
    return run_dir


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "run-record.json").read_text(encoding="utf-8"))


def f2_projection(n_claims: int, shortlist: int, mean_tokens: float, tokens_per_s: float) -> float:
    """Table F.2 indexing line: n_claims x N x mean tokens per passage / measured tokens per second / 3600 (hours)."""
    return round(n_claims * shortlist * mean_tokens / tokens_per_s / 3600.0, 6)


def rerank_projection(n_claims: int, pairs_per_query: int, pairs_per_s: float) -> float:
    """Table F.2 re-ranking line: n_claims x reranker pairs per query / measured reranker pairs per second / 3600 (hours)."""
    return round(n_claims * pairs_per_query / pairs_per_s / 3600.0, 6)


def compare_c1_pair(run_a: Path, run_b: Path) -> dict:
    """Checkpoint sha256 equality of two sealed C1 training runs (their model.json)."""
    sha = [json.loads((Path(d) / "model.json").read_text(encoding="utf-8"))["checkpoint_sha256"] for d in (run_a, run_b)]
    return {"checkpoint_sha256_equal": sha[0] == sha[1]}


def _r1_distributions(run_dir: Path) -> dict[tuple, list[float]]:
    from aac.logging.validate import list_shards
    from aac.logging.writer import SHARD_DIR, read_shard

    out: dict[tuple, list[float]] = {}
    for shard in list_shards(Path(run_dir) / SHARD_DIR):
        for rec in read_shard(shard):
            key = (rec.get("split"), rec["claim_id"])
            if key in out:
                raise ValueError(f"{run_dir}: two records for {key}; the double-run comparison needs one per claim")
            out[key] = [float(x) for x in rec["encoder_distribution"]]
    return out


def compare_r1_pair(run_a: Path, run_b: Path) -> dict:
    """Per-claim equality of encoder_distribution between two R1 content-only runs, aligned by (split, claim_id):
 {n_equal, n_total, max_abs_diff}. Different claim sets raise ValueError (not comparable)."""
    da, db = _r1_distributions(run_a), _r1_distributions(run_b)
    if set(da) != set(db):
        raise ValueError(f"the two R1 runs cover different claim sets ({len(da)} vs {len(db)} claim keys)")
    n_equal = sum(1 for k in da if da[k] == db[k])
    diffs = [abs(x - y) for k in da for x, y in zip(da[k], db[k])]
    return {"n_equal": n_equal, "n_total": len(da), "max_abs_diff": float(max(diffs)) if diffs else 0.0}


def fmt(value, metric: str = "") -> str:
    if value is None:
        return NOT_MEASURED
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value) if any(metric.endswith(k) for k in FULL_PRECISION) else f"{value:.6f}"
    return str(value)


def _rows_throughput(res: dict) -> list[tuple]:
    rows = [("throughput", "", "", "", "device", res["device"], ""),
            ("throughput", "", "", "", "device_name", res["device_name"], "")]
    for t in res["throughput"]:
        head = ("throughput", t["model"], t["precision"], str(t["batch_size"]))
        if t["status"] != "measured":
            rows.append(head + ("status", t["status"], ""))
            continue
        peak = t["peak_memory_bytes"]
        rows += [head + ("passages_per_s", fmt(t["passages_per_s"]), "passages/s"),
                 head + ("tokens_per_s", fmt(t["tokens_per_s"]), "tokens/s"),
                 head + ("wall_seconds", fmt(t["wall_seconds"]), "s"),
                 head + ("peak_memory_bytes", fmt(peak) if peak is not None else f"{NOT_MEASURED}: {res['device']}", "bytes"),
                 head + ("n_passages", fmt(t["n_passages"]), "count"),
                 head + ("n_tokens", fmt(t["n_tokens"]), "tokens")]
    return rows


def _rows_latency(res: dict) -> list[tuple]:
    rows = []
    for s in res["latency"]:
        if s.get("status", "measured") != "measured":
            rows.append(("latency", s["stage"], "", "", "status", s["status"], ""))
            continue
        head = ("latency", s["stage"] if s["stage"] == "c1" else s["model"], s["precision"], str(s["batch_size"]))
        rows += [head + ("median_s", fmt(s["median_s"]), "s"), head + ("p95_s", fmt(s["p95_s"]), "s"),
                 head + ("n_queries", fmt(s["n_queries"]), "count"), head + ("warmup_calls", fmt(s["warmup"]), "count"),
                 head + ("call", s["unit"], "")]
    return rows


def _identity_rows(head: tuple, prefix: str, ident: dict) -> list[tuple]:
    cos = ident.get("min_cosine")
    return [head + (f"{prefix}_equal", fmt(ident["equal"]), "bool"),
            head + (f"{prefix}_max_abs_diff", fmt(ident["max_abs_diff"], "max_abs_diff"), "abs"),
            head + (f"{prefix}_min_cosine", fmt(cos, "min_cosine") if cos is not None else "not applicable: 1-D scores",
                    "cosine")]


def _rows_bit_identity(res: dict) -> list[tuple]:
    rows = []
    for b in res["bit_identity"]:
        head = ("bit_identity", b["model"], b["precision"], str(b["batch_size"]))
        if b["status"] != "measured":
            rows.append(head + ("status", b["status"], ""))
            continue
        rows += _identity_rows(head, "in_process", b["in_process"])
        if "second_process" in b:
            sp = b["second_process"]
            rows += _identity_rows(head, "second_process", sp)
            rows.append(head + ("second_process_npy_sha256_equal", fmt(sp["npy_sha256_equal"]), "bool"))
        else:
            rows.append(head + ("second_process_equal", NOT_MEASURED, "bool"))
    return rows


def _rows_c1(root: Path, c1_pair, r1_pair, seed: int) -> tuple[list[tuple], list[Path]]:
    """C1 determinism double run : checkpoint sha256 of the two C1 training runs and the per-claim
 encoder_distribution equality of their two R1 content-only runs; `not measured` while a run is absent."""
    head = ("c1_double_run", "mdeberta-v3-base", "float32", "")
    c1_dirs = [find_run(root, "C1", n, seed) for n in c1_pair]
    r1_dirs = [find_run(root, "R1", n, seed) for n in r1_pair]
    read: list[Path] = []
    ck = n_eq = n_tot = mad = None
    if all(c1_dirs):
        ck = compare_c1_pair(*c1_dirs)["checkpoint_sha256_equal"]
        read += c1_dirs
    if all(r1_dirs):
        r = compare_r1_pair(*r1_dirs)
        n_eq, n_tot, mad = r["n_equal"], r["n_total"], r["max_abs_diff"]
        read += r1_dirs
    rows = [head + ("checkpoint_sha256_equal", fmt(ck), "bool"),
            head + ("n_encoder_distribution_equal", fmt(n_eq), "count"),
            head + ("n_encoder_distribution_total", fmt(n_tot), "count"),
            head + ("max_abs_diff", fmt(mad, "max_abs_diff"), "abs")]
    return rows, read


def _rows_f2(res: dict) -> list[tuple]:
    proj = res["projection"]
    n_claims = int(proj["n_claims"])
    loaders = {m["key"]: m["loader"] for m in res["models"]}
    rows = [("f2_projection", "", "", "", "n_claims", str(n_claims), "claims")]
    for key, tok in sorted(res["input"]["tokens"].items()):
        if loaders.get(key) == "sentence_transformer":
            rows.append(("f2_projection", key, "", "", "mean_tokens_per_passage", fmt(float(tok["mean_tokens_per_item"])), "tokens"))
    unit = f"{res['device']}-hours (projected)"
    for t in res["throughput"]:
        if t["status"] != "measured" or loaders.get(t["model"]) != "sentence_transformer":
            continue
        mean = float(res["input"]["tokens"][t["model"]]["mean_tokens_per_item"])
        for n in proj["shortlist_sizes"]:
            rows.append(("f2_projection", t["model"], t["precision"], str(t["batch_size"]), f"indexing_hours_n{int(n)}",
                         fmt(f2_projection(n_claims, int(n), mean, float(t["tokens_per_s"]))), unit))
    k = int(res["latency_cfg"]["reranker_pairs_per_query"])
    rows.append(("f2_projection", "", "", "", "reranker_pairs_per_query", str(k), "pairs"))
    for t in res["throughput"]:
        if t["status"] == "measured" and loaders.get(t["model"]) == "cross_encoder":
            rows.append(("f2_projection", t["model"], t["precision"], str(t["batch_size"]), f"rerank_hours_k{k}",
                         fmt(rerank_projection(n_claims, k, float(t["passages_per_s"]))), unit))
    return rows


def build_rows(root: Path, bench_dir: Path, c1_pair, r1_pair, seed: int) -> tuple[list[tuple], list[Path]]:
    res = json.loads((bench_dir / "bench.json").read_text(encoding="utf-8"))
    c1_rows, read = _rows_c1(root, c1_pair, r1_pair, seed)
    rows = _rows_throughput(res) + _rows_latency(res) + _rows_bit_identity(res) + c1_rows + _rows_f2(res)
    return rows, [bench_dir] + read


def write_report(root: Path, out_path: Path | None, bench_run_name: str, c1_pair=DEFAULT_C1_PAIR,
                 r1_pair=DEFAULT_R1_PAIR, seed: int = 0) -> Path:
    root = Path(root)
    bench_dir = find_run(root, "BENCH", bench_run_name, seed)
    if bench_dir is None:
        raise LookupError(f"no sealed BENCH run named {bench_run_name!r} at seed {seed} in {aac_paths.runs_dir(root)}")
    rows, read = build_rows(root, bench_dir, c1_pair, r1_pair, seed)
    records = [_record(d) for d in read]
    dvs = sorted({r["data_version"] for r in records})
    if len(dvs) != 1:
        raise ValueError(f"the runs read do not share one data_version: {dvs}")
    buf = io.StringIO()
    buf.write(f"# config_hash={';'.join(r['config_hash'] for r in records)} data_version={dvs[0]}\n")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(COLUMNS)
    w.writerows(rows)
    out = Path(out_path) if out_path is not None else aac_paths.reports_dir(root) / REPORT_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(buf.getvalue(), encoding="ascii")
    tmp.replace(out)
    return out


def check_bench_csv(path: Path) -> list[str]:
    """Contract problems of a bench CSV (empty list = passes): the provenance header, the seven columns, a
 known section on every row and at least one row in each of throughput, latency, bit_identity, f2_projection."""
    lines = Path(path).read_text(encoding="ascii").splitlines()
    problems = []
    if not lines or not HEADER_RE.match(lines[0]):
        problems.append(f"first line is not `# config_hash=<hex64>[;<hex64>...] data_version=<hex64>`: {lines[:1]}")
    body = lines[1:] if lines and lines[0].startswith("#") else lines
    rows = list(csv.reader(body))
    if not rows or tuple(rows[0]) != COLUMNS:
        problems.append(f"column header is not {','.join(COLUMNS)}: {rows[:1]}")
        return problems
    seen = set()
    for i, r in enumerate(rows[1:], start=3):
        if len(r) != len(COLUMNS):
            problems.append(f"line {i}: {len(r)} fields, expected {len(COLUMNS)}")
            continue
        if r[0] not in SECTIONS:
            problems.append(f"line {i}: unknown section {r[0]!r}")
        if r[5] == "":
            problems.append(f"line {i}: empty value")
        seen.add(r[0])
    problems += [f"no row in section {s}" for s in REQUIRED_SECTIONS if s not in seen]
    return problems


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import root as project_root

    ap = argparse.ArgumentParser(prog="python -m aac.bench.report")
    ap.add_argument("--bench", default=DEFAULT_BENCH, help="name of the sealed BENCH run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c1-pair", nargs=2, default=list(DEFAULT_C1_PAIR), metavar=("C1_A", "C1_B"))
    ap.add_argument("--r1-pair", nargs=2, default=list(DEFAULT_R1_PAIR), metavar=("R1_A", "R1_B"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    root = project_root()
    out = write_report(root, Path(args.out) if args.out else None, args.bench, tuple(args.c1_pair),
                       tuple(args.r1_pair), args.seed)
    problems = check_bench_csv(out)
    for p in problems:
        print(f"bench report: {p}", file=sys.stderr)
    n = len(out.read_text(encoding="ascii").splitlines()) - 2
    shown = out.relative_to(root) if out.is_relative_to(root) else out
    print(f"bench report: {shown} ({n} rows)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
