"""aac CLI. Every subcommand exits non-zero on failure (the exit code is the verdict)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from aac.run import hashing
from aac.run.clock import make_clock
from aac.run.finalize import FinalizeRefused, RunRefused, claim_run_dir, finalize_run
from aac.run.finalize import lock_held as _lock_held
from aac.run.finalize import move_aside as _move_aside
from aac.run.finalize import rel as _rel
from aac.run.identity import identity as _identity
from aac.run.identity import lock_path as _lock_path
from aac.run.paths import root as project_root
from aac.run import paths as aac_paths


def cmd_fixtures_build(args) -> int:
    from aac.data.fixtures import build

    manifest = build(project_root())
    print(f"fixtures: {manifest.relative_to(project_root())} data_version={hashing.data_version(manifest)}")
    return 0


def cmd_run_id(args) -> int:
    root = project_root()
    print(_identity(root, Path(args.config), args.seed)["run_id"])
    return 0


def _driver(regime: str):
    from aac.regimes import r0, r1, r2, r3, r4

    drivers = {"BASE": r0.run, "R1": r1.run, "R2": r2.run, "R3": r3.run, "R4": r4.run}
    if regime not in drivers:
        raise ValueError(f"regime {regime!r} has no driver (have {sorted(drivers)})")
    return drivers[regime]


SENTINEL_SHA = "0" * 40  # the smoke harness's fixed AAC_GIT_SHA (Makefile SMOKE_ENV, mk/smoke.mk)
STUB_DETERMINISM = "seeded; cpu numpy stubs; torch-deterministic-warn-only"
MDEBERTA_DETERMINISM = "seeded; mdeberta-v3-base fp32 inference (sealed C1 checkpoint); torch-deterministic-warn-only"


def _run_origin(run_dir: Path) -> tuple[str | None, str | None]:
    """(git_sha, clock_source) a run directory records: run-record.json when sealed, else the
 git_sha in config.resolved.json (clock unknown). (None, None) when neither is readable."""
    try:
        rr = run_dir / "run-record.json"
        if rr.is_file():
            rec = json.loads(rr.read_text(encoding="utf-8"))
            return rec.get("git_sha"), rec.get("clock_source")
        cr = run_dir / "config.resolved.json"
        if cr.is_file():
            return (json.loads(cr.read_text(encoding="utf-8")).get("code") or {}).get("git_sha"), None
    except (OSError, ValueError):
        pass
    return None, None


def cmd_run(args) -> int:
    from aac.attacks.ledger import LEDGER_FILE, l2_check
    from aac.logging import ShardWriter, load_validator, validate_run, write_counts
    from aac.logging.validate import fixture_texts_for
    from aac.pipeline import RunContext, Toggles
    from aac.pipeline.factory import build_pipeline
    from aac.regimes.common import check_toggles
    from aac.run.config import RunConfig
    from aac.run.env import write_env
    from aac.run.seeds import seed_everything

    root = project_root()
    ident = _identity(root, Path(args.config), args.seed)
    if not isinstance(ident["cfg"], RunConfig):
        print(f"refusing: {args.config} is not an aac-config/1 regime config (use `aac train` for aac-train/1)", file=sys.stderr)
        return 2
    resolved, rid, ch = ident["resolved"], ident["run_id"], ident["config_hash"]
    check_toggles(resolved)
    driver = _driver(resolved["regime"])
    key = (resolved["regime"], resolved["system"], resolved["name"], args.seed)
    try:  # resumable: the ShardWriter resumes a crashed attempt from its verified shards 
        run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, args.force, resumable=True)
    except RunRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, _lock_path(root))
    seed_everything(args.seed)
    pipeline = build_pipeline(resolved, root, args.seed)
    validator = load_validator()
    started = clock.now_iso_ms()
    writer = ShardWriter(run_dir, rid, validator, shard_size=args.shard_size)
    ctx = RunContext(cfg_resolved=resolved, config_hash=ch, data_version=ident["data_version"],
                     model_version=ident["model_version"], run_id=rid, sink=writer, clock=clock,
                     toggles=Toggles.from_system(resolved["system"]), seed=args.seed)
    try:
        summary = driver(pipeline, ctx, resolved, root, run_dir)
    finally:
        shards = writer.close()
    finished = clock.now_iso_ms()
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for t in summary["timings"]:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
    counts = validate_run(run_dir, fixture_texts_for(resolved, root), validator)
    write_counts(run_dir / "schema-validation.txt", counts)
    if counts.records_with_claim_text or counts.records_with_user_id:  
        print(f"refusing to seal {rid}: records_with_claim_text={counts.records_with_claim_text} "
              f"records_with_user_id={counts.records_with_user_id}", file=sys.stderr)
        return 1
    if (run_dir / LEDGER_FILE).is_file():
        l2 = l2_check(run_dir, strict=bool(summary.get("l2_strict")))
        print(l2.as_line())
        if not l2.ok:
            print(f"refusing to seal {rid}: invariant L2 failed", file=sys.stderr)
            return 1
    c1 = resolved["components"]["c1"]
    if c1.get("checkpoint"):  # the sealed C1 training run whose weights this run loaded 
        summary["depends_on"] = sorted({*(summary.get("depends_on") or []), c1["checkpoint"]["run_id"]})
    determinism = MDEBERTA_DETERMINISM if c1["kind"] == "mdeberta-v3-base" else STUB_DETERMINISM
    try:
        digest = finalize_run(root, run_dir, ident, summary=summary, record_count=writer.record_count, shards=shards,
                              clock=clock, started=started, finished=finished, system=resolved["system"],
                              determinism_level=determinism, target_dir=target_dir, stale_ids=stale_ids,
                              violations=counts.violations)
    except FinalizeRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"run {rid}: {writer.record_count} records, violations={counts.violations}, SEALED {digest[:12]}")
    return 0 if counts.violations == 0 else 1


TRAIN_SYSTEM = "train"  # run-record `system` of a C1 or C2S training run (index key: C1|C2S, train, name, seed)


def cmd_train(args) -> int:
    """`aac train --config <aac-train/1 | aac-train-scorer/1> --seed N [--force]`: seal a C1 training run or
 a C2 scorer training run (aac.components.c2_scorer_train).

 The datasheet gate and the split-manifest assertions run first, before git identity, torch or any
 model load: an Art. 10 refusal is reported even on a dirty tree. A scorer run then checks its EV run inside
 train_scorer (after identity has resolved scorer.evidence), still before any model load."""
    from aac.components import c1_train, c2_scorer_train
    from aac.run.config import ScorerTrainConfig, TrainConfig, load_any, resolve
    from aac.run.env import write_env

    root = project_root()
    cfg = load_any(Path(args.config))
    if not isinstance(cfg, (TrainConfig, ScorerTrainConfig)):
        print(f"refusing: {args.config} is not an aac-train/1 or aac-train-scorer/1 config (use `aac run` for regime "
              "configs)", file=sys.stderr)
        return 2
    scorer = isinstance(cfg, ScorerTrainConfig)
    refusals = (c1_train.DatasheetGateClosed, c1_train.CutoffViolation, c1_train.ManifestViolation,
                c2_scorer_train.ScorerDataError)
    pre = resolve(cfg, git_sha=SENTINEL_SHA, lock_sha256="0" * 64, seed=args.seed)  # identity-free view for the gate
    try:
        (c2_scorer_train.preflight if scorer else c1_train.preflight)(pre, root)
    except refusals as exc:
        print(f"refusing to train: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    ident = _identity(root, Path(args.config), args.seed)
    resolved, rid = ident["resolved"], ident["run_id"]
    key = (resolved["regime"], TRAIN_SYSTEM, resolved["name"], args.seed)
    try:
        run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, args.force)
    except RunRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, _lock_path(root))
    started = clock.now_iso_ms()
    try:
        summary = (c2_scorer_train.train_scorer if scorer else c1_train.train_c1)(resolved, root, run_dir, args.seed, clock)
    except c1_train.NonFiniteLoss as exc:
        print(f"refusing to seal {rid}: NonFiniteLoss: {exc}", file=sys.stderr)
        return 1
    except refusals as exc:  # an EV/SL input check of the scorer: nothing sealed (the staging dir is left for --force)
        print(f"refusing to train {rid}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finished = clock.now_iso_ms()
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for t in summary["timings"]:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
    try:
        digest = finalize_run(root, run_dir, ident, summary=summary, record_count=0, shards=[], clock=clock,
                              started=started, finished=finished, system=TRAIN_SYSTEM,
                              determinism_level=summary["determinism_level"], target_dir=target_dir,
                              stale_ids=stale_ids)
    except FinalizeRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    model = json.loads((target_dir / c1_train.MODEL_JSON).read_text(encoding="utf-8"))
    n, unit = (summary["items"]["train_pairs"], "pairs") if scorer else (summary["items"]["train"], "items")
    print(f"train {rid}: {n} {unit}, {summary['steps']} steps, "
          f"checkpoint sha256 {model['checkpoint_sha256']}, SEALED {digest[:12]}")
    return 0


def cmd_validate(args) -> int:
    from aac.logging import validate_run, write_counts
    from aac.logging.validate import fixture_texts_for, read_counts
    from aac.run.seal import verify_sealed

    root = project_root()
    run_dir = aac_paths.runs_dir(root) / args.run
    verify_sealed(run_dir)
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    counts = validate_run(run_dir, fixture_texts_for(resolved, root))
    out_dir = aac_paths.runs_dir(root) / "_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.run}.schema-validation.txt"
    write_counts(out, counts)
    sealed_counts = read_counts(run_dir / "schema-validation.txt")
    print(counts.as_lines(), end="")
    if counts != sealed_counts:
        print("independent validation disagrees with the sealed schema-validation.txt", file=sys.stderr)
        return 1
    return 0 if counts.violations == 0 and counts.records_with_claim_text == 0 and counts.records_with_user_id == 0 else 1


def cmd_verify_hash(args) -> int:
    run_dir = Path(args.run_dir)
    verdict = hashing.verify_hash(run_dir, project_root())
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    recomputed = hashing.config_hash(resolved, git_sha=resolved["code"]["git_sha"], lock_sha256=env["lock_sha256"],
                                     data_version=hashing.data_version(project_root() / resolved["data"]["manifest"]))
    print(f"config_hash: {verdict} {recomputed}")
    return 0 if verdict == "match" else 1


def cmd_data_version(args) -> int:
    print(hashing.data_version(Path(args.manifest)))
    return 0


def cmd_reports_build(args) -> int:
    from aac.reports.registry import AllowPendingRefused, build_all, build_table

    root = project_root()
    out = Path(args.out) if args.out else None
    if not args.all and not args.table:
        print("aac reports build: give --table <id> or --all", file=sys.stderr)
        return 2
    if args.allow_pending and (args.all or not args.table):
        print("aac reports build: --allow-pending needs --table <id> and is never allowed with --all", file=sys.stderr)
        return 2
    try:
        paths = build_all(root, out) if args.all else build_table(args.table, root, out, allow_pending=args.allow_pending)
    except AllowPendingRefused as exc:
        print(f"aac reports build: {exc}", file=sys.stderr)
        return 2
    for p in paths:
        print(f"wrote {Path(p).relative_to(root) if Path(p).is_relative_to(root) else p}")
    return 0


def cmd_reports_index(args) -> int:
    from aac.reports.index import write_index

    root = project_root()
    target = write_index(root, Path(args.out) if args.out else None)
    n = len(json.loads(target.read_text(encoding="utf-8")))
    print(f"wrote {target.relative_to(root) if target.is_relative_to(root) else target} ({n} report files)")
    return 0


def cmd_pack_build(args) -> int:
    from aac.logging.validate import LeakScanUnavailable
    from aac.pack.manifest import NoSealedRuns, UnsafePackTarget, build_pack, partition_runs
    from aac.pack.runs_manifest import RunsManifestMismatch
    from aac.reports.fmt import PendingCellFound
    from aac.run.index import load_index

    root = project_root()
    try:
        pack = build_pack(root, Path(args.out) if args.out else None)
    except UnsafePackTarget as exc:
        print(f"aac pack build: {exc}", file=sys.stderr)
        return 2
    except PendingCellFound as exc:
        print(f"refusing: pending cell in {_rel(exc.file, root)} ({exc.row}, {exc.column}): the evidence pack never "
              "carries an interim value", file=sys.stderr)
        return 1
    except (NoSealedRuns, RunsManifestMismatch, LeakScanUnavailable) as exc:
        print(f"aac pack build: refusing: {exc}", file=sys.stderr)
        return 1
    runs = sorted(load_index(aac_paths.runs_dir(root) / "INDEX.json")["runs"], key=lambda r: r["run_id"])
    logged, logless = partition_runs(root, runs)
    print(f"pack: {pack if args.out else _rel(pack, root)} runs={len(runs)} logged={len(logged)} "
          f"without_decision_log={len(logless)}")
    print("runs without decision-log: " + (" ".join(r["run_id"] for r in logless) if logless else "none"))
    if args.print_d3:  # : the D.3 bytes of the pack just written, read from the file (never rendered twice)
        sys.stdout.flush()
        sys.stdout.write((pack / "table-D.3-manifest.csv").read_text(encoding="utf-8"))
    return 0


def cmd_pack_demo_case(args) -> int:
    """Write configs/analyst/demo-case.json by rule from the sealed R1 chrono full seed-0 run (a versioned input)."""
    from aac.pack.analyst_mock import write_demo_case

    root = project_root()
    target = write_demo_case(root)
    case = json.loads(target.read_text(encoding="utf-8"))
    print(f"demo case: {target.relative_to(root)} record_id={case['record_id']} claim_id={case['claim_id']}")
    return 0


def cmd_pack_verify_table(args) -> int:
    """Table 4.7 + summary from two regenerated packs and the committed pack (called by regen_check.sh and, with
 --how clean-checkout --command --out, by scripts/verify_clean_checkout.sh)."""
    from aac.pack.verify import COMMAND, write_table_4_7

    root = project_root()
    shell, summary, ok = write_table_4_7(root, Path(args.regenerated), Path(args.second), Path(args.committed),
                                         t_d1_identical=args.t_d1 == "identical", how=args.how,
                                         command=args.command or COMMAND,
                                         out_dir=Path(args.out) if args.out else None)
    rows = [ln for ln in shell.read_text(encoding="utf-8").splitlines()[2:] if ln]
    n_pass = sum(1 for ln in rows if ln.endswith(",pass"))
    print(f"table-4.7: {shell.relative_to(root) if shell.is_relative_to(root) else shell} ({len(rows)} rows, {n_pass} pass); "
          f"summary {summary.name}")
    return 0 if ok else 1


def cmd_pack_provenance(args) -> int:
    """: write provenance/PACK-PROVENANCE.md for the pack commit . Never signs; refuses to
 overwrite a signed record."""
    from aac.pack.provenance import ProvenanceSigned, write_provenance

    root = project_root()
    try:
        target = write_provenance(root, args.commit, screenshot=Path(args.screenshot) if args.screenshot else None,
                                  verify_line=args.verify_line,
                                  clock=(lambda: args.verified_at) if args.verified_at else None)
    except ProvenanceSigned as exc:
        print(f"aac pack provenance: refusing: {exc}", file=sys.stderr)
        return 1
    except (ValueError, FileNotFoundError) as exc:
        print(f"aac pack provenance: {exc}", file=sys.stderr)
        return 1
    print(f"provenance: {target.relative_to(root)} (unsigned; sign-off is analyst item O7)")
    return 0


def cmd_analyst_record(args) -> int:
    """: append one analyst event (confirm / override / return) for a deferred record of the selected
 namespace's sealed runs to its append-only analyst/analyst-events.jsonl (or --events). The sealed record is
 never touched; aac.pack.analyst_join joins the events at pack time."""
    from aac.analyst.events import AnalystEventRefused, append_event

    root = project_root()
    try:
        out = append_event(root, args.record_id, args.action, events=Path(args.events) if args.events else None)
    except AnalystEventRefused as exc:
        print(f"aac analyst record: refusing: {exc}", file=sys.stderr)
        return 1
    print(f"analyst event: {args.action} {args.record_id} appended (line {out['line']})")
    return 0


def cmd_lint_determinism(args) -> int:
    """no absolute path, host, user, timestamp, CRLF, NaN, `____` or PNG Software chunk in the
 derived outputs (reports/, evidence-pack/ minus the verbatim sealed shards)."""
    from aac.pack.lint import lint_tree

    root = project_root()
    # base = the selected output namespace (AAC_OUT), so rel paths stay `reports/...` / `evidence-pack/...`;
    # repo_root (absolute-path rule, fixture texts) stays the project root
    hits = lint_tree(aac_paths.out_root(root), repo_root=root,
                     dirs=tuple(args.dirs) if args.dirs else ("reports", "evidence-pack"))
    for h in hits:
        print(h.as_line())
    print(f"lint-determinism: {len(hits)} hits")
    return 0 if not hits else 1


def cmd_matrix_list(args) -> int:
    from aac.run.matrix import expand_run_ids

    for cfg, seed, rid in expand_run_ids(project_root(), args.matrix):
        print(f"{cfg} {seed} {rid}")
    return 0


def cmd_matrix_count(args) -> int:
    from aac.run.matrix import count

    print(count(project_root(), args.matrix))
    return 0


def cmd_matrix_prune(args) -> int:
    """Move every stub run the matrix no longer plans (e.g. a tracer run whose data_version
 moved) to runs/_superseded/ and drop it from runs/INDEX.json.

 Safe by construction it runs only under the smoke sentinels, it only touches runs
 it generated itself (run-record git_sha == the zero sentinel and clock_source == fixed;
 an unsealed directory needs the sentinel git_sha in config.resolved.json and no held
 writer lock), it never deletes (unsealed directories are moved aside too), and when any
 unplanned run or index row is not provably a sentinel stub run it touches nothing and
 exits 2. Real runs therefore survive `make smoke` / `make mini-runs`; Phase 2 moves the stub matrix into its own output root."""
    import os

    from aac.run.index import filter_runs, load_index
    from aac.run.matrix import expand_run_ids
    from aac.run.seal import read_seal

    if os.environ.get("AAC_GIT_SHA") != SENTINEL_SHA or not os.environ.get("AAC_RUN_EPOCH"):
        print("refusing: `aac matrix prune` only prunes stub runs and runs only under the smoke sentinels "
              "(AAC_GIT_SHA=0*40 and AAC_RUN_EPOCH set)", file=sys.stderr)
        return 2
    root = project_root()
    runs_dir = aac_paths.runs_dir(root)
    planned = {rid for _, _, rid in expand_run_ids(root, args.matrix)}
    dirs = sorted(p for p in runs_dir.iterdir() if p.is_dir() and not p.name.startswith("_")) if runs_dir.is_dir() else []
    ours: list[Path] = []
    foreign: list[str] = []
    for d in dirs:
        if d.name in planned:
            continue
        sha, clk = _run_origin(d)
        sealed = (d / "SEALED").is_file()
        if sha != SENTINEL_SHA:
            foreign.append(f"{d.name}: git_sha {sha!r} is not the smoke sentinel")
        elif sealed and clk != "fixed":
            foreign.append(f"{d.name}: clock_source {clk!r} is not fixed")
        elif not sealed and _lock_held(d):
            foreign.append(f"{d.name}: unsealed and locked by a run in progress")
        else:
            ours.append(d)
    ours_names = {d.name for d in ours}
    index = load_index(runs_dir / "INDEX.json")
    for r in index["runs"]:
        if r["run_id"] not in planned and r["run_id"] not in ours_names:
            foreign.append(f"INDEX row {r['run_id']}: no sentinel stub run directory backs it")
    if foreign:
        for f in foreign:
            print(f"matrix prune: not a stub run of this matrix: {f}", file=sys.stderr)
        print(f"refusing: {len(foreign)} unplanned run(s)/row(s) are not provably sentinel stub runs; nothing was "
              "touched (keep real runs out of a smoke run store)", file=sys.stderr)
        return 2
    moved = 0
    for d in ours:
        if (d / "SEALED").is_file():
            _move_aside(runs_dir, d, read_seal(d)[:8])
            print(f"pruned: {d.name} not in the matrix, moved to runs/_superseded/")
        else:
            target = _move_aside(runs_dir, d, "unsealed")
            print(f"pruned: unsealed {d.name} moved to runs/_superseded/{target.name}")
        moved += 1
    
    dropped = filter_runs(runs_dir / "INDEX.json", lambda r: r["run_id"] in planned)
    print(f"matrix prune: {moved} run(s) moved, {len(dropped)} index row(s) dropped, 0 deleted")
    return 0


def cmd_index_rebuild(args) -> int:
    from aac.run.index import rebuild_runs

    root = project_root()
    entries = rebuild_runs(aac_paths.runs_dir(root))
    if args.check_matrix:
        from aac.run.matrix import expand_run_ids

        planned = [rid for _, _, rid in expand_run_ids(root, args.check_matrix)]
        print(f"INDEX rows: {len(entries)} / planned: {len(planned)}")
        have = {e["run_id"] for e in entries}
        missing = sorted(set(planned) - have)
        extra = sorted(have - set(planned))
        if missing or extra:
            print(f"INDEX differs from the matrix: missing {missing}, extra {extra}", file=sys.stderr)
            return 1
        return 0
    print(f"INDEX rows: {len(entries)}")
    return 0


def cmd_index_manifest(args) -> int:
    """Write (or with --check verify) runs/MANIFEST.sha256 of the selected namespace ."""
    from aac.pack.runs_manifest import RunsManifestMismatch, check_runs_manifest, write_runs_manifest
    from aac.run.index import load_index

    root = project_root()
    entries = load_index(aac_paths.runs_dir(root) / "INDEX.json")["runs"]
    try:
        if args.check:
            status = check_runs_manifest(root, entries)
            print(f"runs manifest: {status} ({len(entries)} runs)")
            return 0
        target = write_runs_manifest(root)
    except RunsManifestMismatch as exc:
        print(f"aac index manifest: {exc}", file=sys.stderr)
        return 1
    print(f"wrote {_rel(target, root)} ({len(entries)} runs)")
    return 0


def cmd_index_docs(args) -> int:
    """Record the v1.0 reference (commit, tagger date, day-1 document hashes at the tag) under
 runs/INDEX.json `day1_docs` by merge-write; `runs` is untouched .

 --from-index FILE copies the day1_docs block recorded in FILE (relative paths are read
 from the project root) into runs_dir/INDEX.json instead of recomputing it from git, after checking
 that its prereg_v1_commit equals `git rev-parse <tag>^{commit}`. The smoke chain uses it to give the
 stub index (AAC_OUT=build/smoke) the same block as the real index: text-contradictions.md is
 append-only, so a recomputation would legitimately differ from the tag once it grows."""
    from aac.run.index import DAY1_DOCS_KEY, copy_day1_docs, record_day1_docs

    root = project_root()
    try:
        if args.from_index:
            src = Path(args.from_index)
            block, changed = copy_day1_docs(root, src if src.is_absolute() else root / src, args.tag)
        else:
            block, changed = record_day1_docs(root, args.tag)
    except RuntimeError as exc:
        print(f"aac index docs: {exc}", file=sys.stderr)
        return 1
    state = "recorded" if changed else "unchanged"
    print(f"{DAY1_DOCS_KEY} {state}: {args.tag} -> {block['prereg_v1_commit']} ({block['prereg_v1_tag_date']})")
    for name, digest in block["files"].items():
        print(f"  {name} {digest}")
    return 0


def cmd_ledger_check(args) -> int:
    from aac.attacks.ledger import l2_check

    rc = 0
    for d in args.run_dir:
        run_dir = Path(d)
        res = l2_check(run_dir, strict=not args.lenient)
        print(f"{run_dir.name}: {res.as_line()}")
        rc = rc or (0 if res.ok else 1)
    return rc


def cmd_bodega_score(args) -> int:
    """`aac bodega-score --config <aac-bodega/1> --seed N [--force]`: the BODEGA score of the source R3 runs as a sealed,
 text-free BS run (aac.attacks.bodega; the env-attack scorer runs BODEGA's own BODEGAScore on the CPU)."""
    from aac.attacks.bodega import score
    from aac.run.config import BodegaScoreConfig, load_any
    from aac.run.identity import MissingRun
    from aac.run.seal import UnsealedRun

    root = project_root()
    cfg = load_any(Path(args.config))
    if not isinstance(cfg, BodegaScoreConfig):
        print(f"refusing: {args.config} is not an aac-bodega/1 config", file=sys.stderr)
        return 2
    try:
        ident = _identity(root, Path(args.config), args.seed)
    except (MissingRun, UnsealedRun) as exc:
        print(f"refusing: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return score(root, ident, force=args.force)



# so later plans add a module here and never touch the argparse tree below.
PRISMA_GROUPS: dict[str, str] = {
    "workbook": "aac.prisma.workbook",
    "sample": "aac.prisma.sample",
    "batch": "aac.prisma.batches",
    "pilot": "aac.prisma.pilot",
    "abstracts": "aac.prisma.abstracts",
    "refcheck": "aac.prisma.refcheck",
    "consistency": "aac.prisma.consistency",
    "count": "aac.prisma.counts",
    "figure": "aac.prisma.figure",
    "paste": "aac.prisma.paste",
}


def _prisma_usage() -> str:
    return "usage: aac prisma <group> ...\ngroups: " + ", ".join(PRISMA_GROUPS)


def cmd_prisma(argv: list[str]) -> int:
    """Dispatch `aac prisma <group> <args>` to PRISMA_GROUPS[group].main(<args>); 2 for usage or a group not built yet."""
    import importlib

    if argv and argv[0] in ("-h", "--help"):
        print(_prisma_usage())
        return 0
    if not argv or argv[0] not in PRISMA_GROUPS:
        if argv:
            print(f"aac prisma: unknown group {argv[0]!r}", file=sys.stderr)
        print(_prisma_usage(), file=sys.stderr)
        return 2
    group, rest = argv[0], list(argv[1:])
    name = PRISMA_GROUPS[group]
    try:
        mod = importlib.import_module(name)
    except ModuleNotFoundError as exc:
        if exc.name is not None and (exc.name == name or name.startswith(exc.name + ".")):
            print(f"aac prisma {group}: not built yet ({name})", file=sys.stderr)
            return 2
        raise  # a module that exists but fails to import is a bug, not "not built yet"
    return int(mod.main(rest))







def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "prisma":  # before argparse, so an option right after the group reaches the module
        return cmd_prisma(argv[1:])

    from aac.reports.registry import buildable

    ap = argparse.ArgumentParser(prog="aac")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prisma", help="Appendix C PRISMA close-out: aac prisma <group> ... (groups: "
                       + ", ".join(PRISMA_GROUPS) + ")", add_help=False)
    p.set_defaults(fn=lambda a: cmd_prisma([]))

    fx = sub.add_parser("fixtures").add_subparsers(dest="sub", required=True)
    fx.add_parser("build").set_defaults(fn=cmd_fixtures_build)
    p = sub.add_parser("run"); p.add_argument("--config", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true"); p.add_argument("--shard-size", type=int, default=1000); p.set_defaults(fn=cmd_run)
    p = sub.add_parser("train", help="seal a C1 or C2 scorer training run")
    p.add_argument("--config", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_train)
    p = sub.add_parser("run-id"); p.add_argument("--config", required=True); p.add_argument("--seed", type=int, required=True); p.set_defaults(fn=cmd_run_id)
    p = sub.add_parser("validate"); p.add_argument("--run", required=True); p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("verify-hash"); p.add_argument("run_dir"); p.set_defaults(fn=cmd_verify_hash)
    p = sub.add_parser("data-version"); p.add_argument("manifest"); p.set_defaults(fn=cmd_data_version)
    rp = sub.add_parser("reports").add_subparsers(dest="sub", required=True)
    p = rp.add_parser("build"); p.add_argument("--table", choices=sorted(buildable())); p.add_argument("--all", action="store_true"); p.add_argument("--out")
    p.add_argument("--allow-pending", action="store_true",
                   help="with --table only: let the generator write `pending (Phase N)` interim cells (refused by aac pack build)")
    p.set_defaults(fn=cmd_reports_build)
    p = rp.add_parser("index"); p.add_argument("--out"); p.set_defaults(fn=cmd_reports_index)
    pk = sub.add_parser("pack").add_subparsers(dest="sub", required=True)
    p = pk.add_parser("build"); p.add_argument("--out"); p.add_argument("--print-d3", action="store_true")
    p.set_defaults(fn=cmd_pack_build)
    pk.add_parser("demo-case").set_defaults(fn=cmd_pack_demo_case)
    p = pk.add_parser("verify-table"); p.add_argument("--regenerated", required=True); p.add_argument("--second", required=True)
    p.add_argument("--committed", required=True); p.add_argument("--t-d1", choices=["identical", "differs"], required=True)
    p.add_argument("--how", choices=["clean-checkout", "regen"], default="regen",
                   help="who regenerated the pack: the long-row test text (aac.pack.verify.HOW_TEXT)")
    p.add_argument("--command", default=None, help="command string for the summary JSON (default aac.pack.verify.COMMAND)")
    p.add_argument("--out", default=None, help="reports directory to write into (default: reports_dir())")
    p.set_defaults(fn=cmd_pack_verify_table)
    p = pk.add_parser("provenance", help="write the detached provenance record provenance/PACK-PROVENANCE.md ")
    p.add_argument("--commit", required=True, help="the pack commit")
    p.add_argument("--screenshot", default=None, help="Appendix D PNG (default: none, the record names analyst item O8)")
    p.add_argument("--verify-line", default=None, help="the final `verify-clean-checkout: <sha12> identical (...)` line")
    p.add_argument("--verified-at", default=None, help="UTC time of the verification, e.g. 2026-10-21T19:00:00Z (default: now)")
    p.set_defaults(fn=cmd_pack_provenance)
    an = sub.add_parser("analyst", help="analyst write-back").add_subparsers(dest="sub", required=True)
    p = an.add_parser("record", help="append one analyst event for a deferred record (append-only; the record is never mutated)")
    p.add_argument("--record-id", required=True); p.add_argument("--action", required=True, choices=("confirm", "override", "return"))
    p.add_argument("--events", default=None, help="events file (default: <namespace>/analyst/analyst-events.jsonl)")
    p.set_defaults(fn=cmd_analyst_record)
    p = sub.add_parser("lint-determinism"); p.add_argument("dirs", nargs="*"); p.set_defaults(fn=cmd_lint_determinism)
    mx = sub.add_parser("matrix").add_subparsers(dest="sub", required=True)
    for name, fn in (("list", cmd_matrix_list), ("count", cmd_matrix_count), ("prune", cmd_matrix_prune)):
        p = mx.add_parser(name); p.add_argument("--matrix", default=None); p.set_defaults(fn=fn)
    ix = sub.add_parser("index").add_subparsers(dest="sub", required=True)
    p = ix.add_parser("rebuild"); p.add_argument("--check-matrix", default=None); p.set_defaults(fn=cmd_index_rebuild)
    p = ix.add_parser("manifest", help="write or --check runs/MANIFEST.sha256 (sha256sum format, one line per run)")
    p.add_argument("--check", action="store_true"); p.set_defaults(fn=cmd_index_manifest)
    p = ix.add_parser("docs"); p.add_argument("--tag", default="v1.0"); p.add_argument("--from-index", default=None)
    p.set_defaults(fn=cmd_index_docs)
    p = sub.add_parser("ledger-check"); p.add_argument("run_dir", nargs="+"); p.add_argument("--lenient", action="store_true"); p.set_defaults(fn=cmd_ledger_check)
    p = sub.add_parser("bodega-score", help="seal a BODEGA scoring run (aac-bodega/1, regime BS)")
    p.add_argument("--config", required=True); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_bodega_score)
    from aac.dispatch.commands import register as register_dispatch

    register_dispatch(sub)  # aac dispatch check|sentinel|clean-tree|env|push|pull|status|probe 
    args = ap.parse_args(argv)
    from aac.data.manifest import DataDrift

    try:
        return int(args.fn(args))
    except DataDrift as exc:  
        print(f"refusing: data drift: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
