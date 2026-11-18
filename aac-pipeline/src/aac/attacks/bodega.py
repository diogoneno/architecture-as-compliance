"""`aac bodega-score --config <aac-bodega/1> --seed N [--force]`: the BODEGA score as a sealed post-attack CPU step
(4 and the anti-pattern "BLEURT inside a GPU session"; methodological note).

One BS run per (config, seed), on the host that holds the adversarial texts (gpu-host for AVeriTeC and FC, cpu-host for
RD; the private store never leaves its host):
 sources every R3 run of `source_runs` (resolved by aac.run.identity: name -> run_id + sealed digest, hashed into
 config_hash), its seal recomputed against that digest and runs/INDEX.json, invariant L2 strict
 rows its ledger rows with queries_used > 0 (the attacked rows, the Table 4.4 denominator)
 texts original = the item text of the source run's own resolved config (aac.regimes.common.target_items: AVeriTeC
 raw JSON, FC hash-checked attack.tsv, RD lane B), checked against the claim_hash of the row's clean
 decision record (record_ids[0], the text the server was sent); adversarial = the stored final of the
 private store (aac.attacks.private_store), its claim_hash equal to attack-adversarial.jsonl's, or the
 original when the row has no final (the replay_plan rules of aac.regimes.r4)
 success strict: ledger end_state `success` (a victim-B `deferred` row scores 0, pre-registered specification)
 scoring the texts go to a fresh `${TMPDIR:-/tmp}/aac-bodega-XXXX` (0700, files 0600) and the env-attack scorer
 attacks/scoring/bodega_score.py (BODEGA's own BODEGAScore with the task shim, CPU, BLEURT-20, LAMBO for RD)
 runs through scripts/attack_env.sh (`env -i`); test seam AAC_BODEGA_SCORER = a command line replacing the
 launcher (the env-main tests use a fake scorer)
 output bodega-scores.jsonl (one row per attacked row, ONLY the interface keys of INTERFACE_KEYS, the read
 interface of aac.reports.phase4_sources.read_bodega_scores) and aggregate.json (per-group BODEGAScore
 aggregates and the verified pins); sealed with record_count 0 (no decision records: the BASE pattern)
 refusals an output key outside the interface, a string that is not an id (whitespace, > 128 characters), a row
 that does not match its staged key, bodega != success x semantic x character, or a group mean that differs
 from BODEGAScore's aggregate: nothing is sealed
 staging deleted on success; kept (0700) on any failure, its path printed (the texts are never printed, logged or
 written anywhere else)
Prints ids and counts only."""
from __future__ import annotations

import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from aac.attacks.ledger import l2_check, read_adversarial, read_ledger
from aac.attacks.private_store import read_private_store
from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.run import hashing
from aac.run import paths as aac_paths
from aac.run.config import BODEGA_SHIM

REGIME = "BS"
SYSTEM = "bodega"
SCORES_FILE = "bodega-scores.jsonl"
AGGREGATE_FILE = "aggregate.json"
AGGREGATE_SCHEMA = "aac-bodega-aggregate/1"
# = aac.reports.phase4_sources.BODEGA_FIELDS (asserted by tests/unit/test_bodega_run.py)
INTERFACE_KEYS = ("item_id", "attack", "victim", "subset", "source_run_id", "success_strict", "semantic", "character",
                  "bodega")
ID_KEYS = ("item_id", "attack", "victim", "subset", "source_run_id")
MAX_STRING = 128
SAFE_STRING = re.compile(r"[A-Za-z0-9_.:/+=\-]{0,128}")
TOL = 1e-9
SCORER_SEAM = "AAC_BODEGA_SCORER"
SCORER_MODULE = "attacks.scoring.bodega_score"
ATTACK_ENV = Path("scripts") / "attack_env.sh"
ATTACK_PY = Path(".venv-attack") / "bin" / "python"
STAGING_PREFIX = "aac-bodega-"
LOCK_FILE = "uv.lock"   # = aac.run.identity.lock_path (identity is called by aac.cli: it reaches aac.components via
                        # aac.data.manifest, which the architectural isolation constraint contract forbids for aac.attacks)
_ERROR_CLASS = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt|Violation))\b")


class BodegaRunError(RuntimeError):
    """A source, a text, the scorer or its output does not check out; nothing is sealed (no text in the message)."""


class TextLeak(ValueError):
    """The scorer output holds a key outside the interface or a string that is not an id."""


# --------------------------------------------------------------------------- sources and staging input

def source_run(root: Path, ref: dict) -> Path:
    """The run directory of a resolved source {name, run_id, sealed_sha256}: indexed as that R3 run, its seal
 recomputed equal to the resolved digest and to runs/INDEX.json, invariant L2 strict."""
    from aac.run.index import load_index
    from aac.run.seal import verify_sealed

    runs = aac_paths.runs_dir(root)
    rows = [r for r in load_index(runs / "INDEX.json")["runs"] if r["run_id"] == ref["run_id"]]
    if len(rows) != 1 or rows[0].get("regime") != "R3" or rows[0].get("name") != ref["name"]:
        raise BodegaRunError(f"source {ref['name']} {ref['run_id']}: not indexed as that R3 run in {runs / 'INDEX.json'}")
    run_dir = runs / ref["run_id"]
    digest = verify_sealed(run_dir)
    if digest != ref["sealed_sha256"] or digest != rows[0].get("sealed_sha256"):
        raise BodegaRunError(f"source {ref['run_id']}: SEALED {digest[:12]} differs from the resolved "
                             f"{ref['sealed_sha256'][:12]} or runs/INDEX.json (re-resolve the identity)")
    l2 = l2_check(run_dir, strict=True)
    if not l2.ok:
        raise BodegaRunError(f"source {ref['run_id']}: {l2.as_line()}")
    return run_dir


def clean_hashes(run_dir: Path, record_ids: set[str]) -> dict[str, str]:
    """{record_id: claim_hash} of these decision records (streamed shard by shard)."""
    out = {}
    for p in list_shards(Path(run_dir) / SHARD_DIR):
        for rec in read_shard(p):
            if rec["record_id"] in record_ids:
                out[rec["record_id"]] = rec["claim_hash"]
    missing = record_ids - set(out)
    if missing:
        raise BodegaRunError(f"{Path(run_dir).name}: {len(missing)} clean record ids absent from the decision log")
    return out


def stage_rows(root: Path, ref: dict, run_dir: Path) -> list[dict]:
    """The scorer input rows (WITH texts) of one source run: its attacked ledger rows, in ledger order."""
    from aac.regimes.common import target_items

    rid = ref["run_id"]
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    rows = [r for r in read_ledger(run_dir) if int(r["queries_used"]) > 0]
    adv = {(a["item_id"], a["attack"], a["victim"], a["subset"]): a for a in read_adversarial(run_dir)}
    keys = [(r["item_id"], r["attack"], r["victim"], r["subset"]) for r in rows]
    # a source with no success has no store file (the server writes one on the first success only)
    store = read_private_store(root, rid) if any((adv.get(k) or {}).get("claim_hash") for k in keys) else {}
    clean = clean_hashes(run_dir, {r["record_ids"][0] for r in rows})
    items: dict[str, dict] = {}
    out = []
    for r, key in zip(rows, keys):
        subset = r["subset"]
        if subset not in BODEGA_SHIM:
            raise BodegaRunError(f"{rid}: {key}: no BODEGA task for subset {subset!r} (shim {sorted(BODEGA_SHIM)})")
        if subset not in items:
            items[subset] = target_items(root, resolved, subset)
        it = items[subset].get(r["item_id"])
        if it is None or not it["positive"]:
            raise BodegaRunError(f"{rid}: {key}: not a positive item of target {subset} in the source's resolved config")
        original = it["text"]
        if hashing.claim_hash(original) != clean[r["record_ids"][0]]:
            raise BodegaRunError(f"{rid}: {key}: the item text is not the clean text of record {r['record_ids'][0]}")
        a = adv.get(key)
        if a is None:
            raise BodegaRunError(f"{rid}: {key}: no attack-adversarial.jsonl row")
        want = a.get("claim_hash")
        if (r["end_state"] in ("success", "deferred")) != (want is not None):
            raise BodegaRunError(f"{rid}: {key}: end_state {r['end_state']} disagrees with the adversarial claim_hash "
                                 f"({'set' if want else 'null'})")
        final = original
        if want is not None:
            held = store.get(key)
            if held is None:
                raise BodegaRunError(f"{rid}: {key}: a success with no stored final text (score on the host that made "
                                     "the R3 run)")
            if held["claim_hash"] != want or hashing.claim_hash(held["text"]) != want:
                raise BodegaRunError(f"{rid}: {key}: the stored final's claim_hash differs from the sealed run's")
            final = held["text"]
        out.append({"item_id": r["item_id"], "attack": r["attack"], "victim": r["victim"], "subset": subset,
                    "source_run_id": rid, "task": BODEGA_SHIM[subset], "original": original, "adversarial": final,
                    "success_strict": r["end_state"] == "success"})
    return out


# --------------------------------------------------------------------------- staging and the scorer

def make_staging(root: Path) -> Path:
    """A fresh 0700 `${TMPDIR:-/tmp}/aac-bodega-XXXX`, refused inside the repository or the output namespace."""
    parent = Path(os.environ.get("TMPDIR") or "/tmp").resolve()
    forbidden = [Path(root).resolve(), aac_paths.out_root(root).resolve()]
    if (Path(root).resolve().parent / ".git").exists():
        forbidden.append(Path(root).resolve().parent)
    for f in forbidden:
        if parent == f or parent.is_relative_to(f):
            raise BodegaRunError(f"refusing to stage texts under {f} (TMPDIR inside the repository or the namespace)")
    d = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=str(parent)))
    os.chmod(d, 0o700)
    return d


def write_private(path: Path, lines: list[str]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="ascii", newline="\n") as fh:
        fh.writelines(lines)


def scorer_argv(root: Path, inp: Path, out: Path, summary: Path) -> list[str]:
    io = ["--in", str(inp), "--out", str(out), "--summary", str(summary)]
    seam = os.environ.get(SCORER_SEAM)
    if seam:
        return [*shlex.split(seam), *io]
    return [str(Path(root) / ATTACK_ENV), str(Path(root) / ATTACK_PY), "-m", SCORER_MODULE, *io]


def run_scorer(root: Path, staging: Path, rows: list[dict], threads: int) -> tuple[list[dict], dict, str, float]:
    """(output rows, scorer summary, the scorer's count line, seconds); every file stays in the staging dir."""
    inp, out, summ = staging / "in.jsonl", staging / "out.jsonl", staging / "summary.json"
    write_private(inp, [json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows])
    env = dict(os.environ)
    env.update({"AAC_THREADS": str(int(threads)), "AAC_GPU": ""})
    t0 = time.perf_counter()
    with os.fdopen(os.open(staging / "stdout.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as fo, \
            os.fdopen(os.open(staging / "stderr.log", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as fe:
        rc = subprocess.run(scorer_argv(root, inp, out, summ), cwd=str(root), env=env, stdin=subprocess.DEVNULL,
                            stdout=fo, stderr=fe, check=False).returncode
    seconds = time.perf_counter() - t0
    if rc != 0:
        err = (staging / "stderr.log").read_text(encoding="utf-8", errors="replace")
        classes = sorted({m.group(1) for ln in err.splitlines() if (m := _ERROR_CLASS.match(ln.strip()))})
        raise BodegaRunError(f"the scorer exited {rc} (error classes {classes or ['none']})")
    line = [ln for ln in (staging / "stdout.log").read_text(encoding="ascii").splitlines()
            if re.fullmatch(r"bodega-score: groups=\d+ rows=\d+ successes=\d+ assertions=\d+ ok", ln)]
    if len(line) != 1:
        raise BodegaRunError("the scorer printed no `bodega-score: ... ok` count line")
    rows_out = [json.loads(x) for x in out.read_text(encoding="ascii").splitlines() if x.strip()]
    return rows_out, json.loads(summ.read_text(encoding="ascii")), line[0], seconds


# --------------------------------------------------------------------------- output checks

def _unit(v, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0.0 or v > 1.0:
        raise BodegaRunError(f"{what}: {v!r} is not a finite score in [0, 1]")
    return float(v)


def assert_text_free(obj, where: str) -> None:
    """Every key and string value is an id (SAFE_STRING: no whitespace, at most 128 characters); numbers, booleans
 and nulls pass. TextLeak otherwise (the message names the path and the length, never the string)."""
    def walk(o, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k, f"{path}.<key>")
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            if len(o) > MAX_STRING or not SAFE_STRING.fullmatch(o):
                raise TextLeak(f"{where}: {path} holds a string that is not an id ({len(o)} chars)")
        elif o is not None and not isinstance(o, (bool, int, float)):
            raise TextLeak(f"{where}: {path} holds a {type(o).__name__}")
    walk(obj, where)


def check_output(staged: list[dict], out: list[dict], summary: dict) -> list[dict]:
    """The scorer output, checked row by row against the staged keys and as a whole against BODEGAScore's per-group
 aggregates; returns the rows with exactly INTERFACE_KEYS (TextLeak / BodegaRunError otherwise)."""
    if len(out) != len(staged):
        raise BodegaRunError(f"the scorer wrote {len(out)} rows for {len(staged)} staged rows")
    rows = []
    for n, (s, o) in enumerate(zip(staged, out), 1):
        if not isinstance(o, dict) or set(o) != set(INTERFACE_KEYS):
            extra = sorted(set(o) - set(INTERFACE_KEYS)) if isinstance(o, dict) else []
            raise TextLeak(f"output row {n}: keys outside the interface {extra} or missing interface keys; refusing "
                           "to seal")
        assert_text_free(o, f"output row {n}")
        if any(o[k] != s[k] for k in ID_KEYS):
            raise BodegaRunError(f"output row {n}: ids differ from the staged row {tuple(s[k] for k in ID_KEYS)}")
        if o["success_strict"] is not s["success_strict"]:
            raise BodegaRunError(f"output row {n}: success_strict {o['success_strict']} is not the ledger's {s['success_strict']}")
        b = _unit(o["bodega"], f"output row {n} bodega")
        if s["success_strict"]:
            product = _unit(o["semantic"], f"output row {n} semantic") * _unit(o["character"], f"output row {n} character")
        else:
            if o["semantic"] is not None or o["character"] is not None:
                raise BodegaRunError(f"output row {n}: semantic/character set for a row that is not a strict success")
            product = 0.0
        if abs(product - b) > 1e-12:
            raise BodegaRunError(f"output row {n}: bodega {b!r} != success x semantic x character {product!r}")
        rows.append({k: o[k] for k in INTERFACE_KEYS})
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["source_run_id"], r["attack"], r["victim"], r["subset"]), []).append(r)
    by_key = {(g["source_run_id"], g["attack"], g["victim"], g["subset"]): g for g in summary["groups"]}
    if set(by_key) != set(groups):
        raise BodegaRunError("the scorer summary's groups differ from the output rows' groups")
    for key, rs in groups.items():
        g = by_key[key]
        agg = g["bodegascore_aggregate"]["bodega"]
        mean = sum(r["bodega"] for r in rs) / len(rs)
        if g["n_attacked"] != len(rs) or g["n_success"] != sum(r["success_strict"] for r in rs):
            raise BodegaRunError(f"group {key}: the summary counts differ from the output rows")
        if agg is None or abs(mean - float(agg)) > TOL:
            raise BodegaRunError(f"group {key}: mean bodega {mean!r} != BODEGAScore aggregate {agg!r}")
    return rows


# --------------------------------------------------------------------------- the run

def _lines(rows: list[dict]) -> str:
    return "".join(json.dumps(r, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False) + "\n"
                   for r in rows)


def score(root: Path, ident: dict, *, force: bool = False, out=None) -> int:
    """Run and seal one BS run from a resolved identity (aac.run.identity.identity, called by `aac bodega-score` in
 aac.cli). 0 sealed, 1 failed (not sealed), 2 refused."""
    from aac.run.clock import make_clock
    from aac.run.env import write_env
    from aac.run.finalize import FinalizeRefused, RunRefused, claim_run_dir, finalize_run

    out = out or sys.stdout
    resolved, rid = ident["resolved"], ident["run_id"]
    seed = int(resolved["seed"])
    key = (resolved["regime"], SYSTEM, resolved["name"], seed)
    try:
        run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, force)
    except RunRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, Path(root) / LOCK_FILE)
    started = clock.now_iso_ms()
    staging = None
    try:
        staged, per_source = [], []
        for ref in resolved["source_runs"]:
            rows = stage_rows(root, ref, source_run(root, ref))
            staged.extend(rows)
            per_source.append({**ref, "attacked_rows": len(rows), "strict_successes": sum(r["success_strict"] for r in rows)})
        if not staged:
            raise BodegaRunError("the source runs hold no attacked ledger row (queries_used > 0)")
        staging = make_staging(root)
        out_rows, summary, count_line, seconds = run_scorer(root, staging, staged, resolved["bodega"]["threads"])
        rows = check_output(staged, out_rows, summary)
        pins = summary["pins"]
        want = resolved["bodega"]
        for k_pin, k_cfg in (("bodega_commit", "commit"), ("bleurt_revision", "bleurt_revision"),
                             ("bleurt_pytorch_commit", "bleurt_pytorch_commit"), ("lambo_commit", "lambo_commit")):
            if pins.get(k_pin) != want[k_cfg]:
                raise BodegaRunError(f"scorer pin {k_pin} {pins.get(k_pin)} differs from the config's {want[k_cfg]}")
        if summary.get("task_shim") != resolved["shim"]:
            raise BodegaRunError("the scorer's task shim differs from the config's")
        groups = [{k: g[k] for k in ("source_run_id", "attack", "victim", "subset", "task", "align_sentences",
                                     "n_attacked", "n_success", "bodegascore_aggregate")} for g in summary["groups"]]
        aggregate = {"schema": AGGREGATE_SCHEMA, "name": resolved["name"], "task_shim": resolved["shim"], "pins": pins,
                     "device": summary["device"], "threads": int(want["threads"]), "sources": per_source,
                     "groups": groups, "rows": len(rows), "strict_successes": sum(r["success_strict"] for r in rows),
                     "aggregate_equalities": int(summary["assertions"]), "tolerance": TOL}
        assert_text_free(aggregate, AGGREGATE_FILE)
    except (BodegaRunError, TextLeak) as exc:
        print(f"bodega-score {rid}: FAILED, not sealed: {type(exc).__name__}: {exc}", file=sys.stderr)
        if staging is not None:
            print(f"bodega-score {rid}: staging kept (0700, holds texts; delete by hand): {staging}", file=sys.stderr)
        return 1
    except Exception:
        if staging is not None:
            print(f"bodega-score {rid}: staging kept (0700, holds texts; delete by hand): {staging}", file=sys.stderr)
        raise
    (run_dir / SCORES_FILE).write_text(_lines(rows), encoding="ascii")
    (run_dir / AGGREGATE_FILE).write_text(hashing.dumps_sorted(aggregate), encoding="ascii")
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        fh.write(json.dumps({"stage": "bodega-score", "seconds": round(seconds, 3), "items": len(rows), "device": "cpu"},
                            sort_keys=True) + "\n")
    finished = clock.now_iso_ms()
    summ = {"configurations": [f"bodega/{resolved['name']}"],
            "items": {"attacked_rows": len(rows), "strict_successes": aggregate["strict_successes"],
                      "sources": len(per_source)},
            "stages": [{"name": "bodega-score", "device": "cpu",
                        "seconds": round(seconds, 3) if clock.source == "wall" else 0.0, "items": len(rows)}],
            "depends_on": [ref["run_id"] for ref in resolved["source_runs"]]}
    level = (f"BODEGA metrics/BODEGAScore.py {want['commit'][:8]} with the task shim; BLEURT-20 {want['bleurt_revision'][:8]} "
             f"fp32 eval on cpu, {int(want['threads'])} threads; LAMBO {want['lambo_commit'][:8]} alignment for RD")
    try:
        digest = finalize_run(root, run_dir, ident, summary=summ, record_count=0, shards=[], clock=clock,
                              started=started, finished=finished, system=SYSTEM, determinism_level=level,
                              target_dir=target_dir, stale_ids=stale_ids)
    except FinalizeRefused as exc:
        print(str(exc), file=sys.stderr)
        print(f"bodega-score {rid}: staging kept (0700, holds texts; delete by hand): {staging}", file=sys.stderr)
        return 1
    shutil.rmtree(staging)
    if staging.exists():
        print(f"bodega-score {rid}: could not remove the staging directory {staging}; remove it by hand", file=sys.stderr)
        return 1
    print(count_line.replace("bodega-score:", f"bodega-score {rid}: scorer"), file=out)
    print(f"bodega-score {rid}: {len(rows)} rows ({aggregate['strict_successes']} strict successes) from "
          f"{len(per_source)} source runs, {aggregate['aggregate_equalities']} aggregate equalities, text-free, "
          f"SEALED {digest[:12]}", file=out)
    return 0


def text_free_check(root: Path, run_id: str) -> int:
    """The number of score rows of a sealed BS run after three checks (make bodega-smoke): the seal holds; every row
 has exactly INTERFACE_KEYS and only id strings; and no original or adversarial text of a scored row (re-staged
 in memory from the source runs, never printed) occurs in any file of the run directory."""
    from aac.run.seal import verify_sealed

    run_dir = aac_paths.runs_dir(root) / run_id
    verify_sealed(run_dir)
    rows = [json.loads(x) for x in (run_dir / SCORES_FILE).read_text(encoding="ascii").splitlines() if x.strip()]
    for n, r in enumerate(rows, 1):
        if set(r) != set(INTERFACE_KEYS):
            raise TextLeak(f"{SCORES_FILE} row {n}: keys differ from the interface")
        assert_text_free(r, f"{SCORES_FILE} row {n}")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    texts = set()
    for ref in resolved["source_runs"]:
        for s in stage_rows(root, ref, source_run(root, ref)):
            texts.update(t for t in (s["original"], s["adversarial"]) if len(t) >= 8)
    blobs = [p.read_bytes() for p in sorted(run_dir.rglob("*")) if p.is_file()]
    for t in texts:
        needles = {t.encode("utf-8"), json.dumps(t, ensure_ascii=True)[1:-1].encode("ascii")}
        if any(nd in b for nd in needles for b in blobs):
            raise TextLeak(f"{run_id}: a staged text occurs in the run directory")
    return len(rows)
