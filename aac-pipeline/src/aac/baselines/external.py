"""External baseline runs (Table 4.1 rows 1-5).

`python -m aac.baselines.external run --config FILE [--config FILE ...] --seed N [--force]` seals one BASE run per
`aac-external/1` config in the selected namespace (runs_dir): config.resolved.json, env.json, result.json (parsed
numbers, hashes and the checkpoint's incompatible-keys report), items.jsonl (row-indexed outcomes, when the tool has
per-item outcomes), wallclock.jsonl and run-record.json with system `external` and record_count 0. There are no
Appendix-B records: these are the organisers' and the AVeriTeC authors' models, not this system's decisions
(methodological note).

Several --config values are resolved to identities BEFORE the first run starts: a sealed run appends to the tracked
runs/INDEX.json, which would make the next identity refuse a dirty tree.

Third-party output that may hold text (BODEGA prints every original and adversarial text, writes raw_*.tsv and a
VictimCache pickle) lives only in `private_staging`: a 0700 directory under /tmp, never inside the repository,
logs/ or runs/, deleted on success and on any exception. Only parsed numbers, row indices, item ids, labels and
64-hex hashes survive (`assert_text_free` walks every string before anything is written)."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from aac.run import hashing
from aac.run import paths as aac_paths

STAGING_PREFIX = "aac-baseline-"
STAGING_PARENT = Path("/tmp")
RESULT_JSON = "result.json"
ITEMS_JSONL = "items.jsonl"
SYSTEM = "external"
RESULT_SCHEMA = "aac-external-result/1"
DETERMINISM = {
    "bodega_test_victims": "external; BODEGA runs/test_victims.py unchanged; cpu fp32 argmax inference",
    "bodega_attack": "external; BODEGA runs/attack.py unchanged (its own seeds 10/10/0); cpu; unbudgeted",
    "averitec_veracity": "external; bert_veracity.ckpt cpu fp32 inference_mode; argmax per QA pair",
}

# Strings allowed in result.json / items.jsonl: identifiers, ids, hashes, URLs and numbers-as-text, no whitespace
# (a sentence of text cannot pass); the four AVeriTeC label names are the only strings with spaces.
SAFE_STRING = re.compile(r"[A-Za-z0-9_.:/+=\-]{0,200}")


class BaselineFailed(RuntimeError):
    """The third-party tool failed or its output did not check out; nothing is sealed."""


class StagingRefused(RuntimeError):
    """The staging directory would lie inside the repository, logs/ or runs/."""


class TextLeak(ValueError):
    """A string in result.json or items.jsonl is not an identifier, id, hash or label."""


def _forbidden_roots() -> list[Path]:
    root = aac_paths.root().resolve()
    out = [root, (root / "logs").resolve(), aac_paths.runs_dir(root).resolve(), aac_paths.out_root(root).resolve()]
    if (root.parent / ".git").exists():  # the repository root (aac-pipeline/ is a subtree of it)
        out.append(root.parent.resolve())
    return out


def _inside(path: Path, roots: list[Path]) -> Path | None:
    p = Path(path).resolve()
    for r in roots:
        if p == r or p.is_relative_to(r):
            return r
    return None


@contextmanager
def private_staging(parent: Path | str = STAGING_PARENT) -> Iterator[Path]:
    """A fresh 0700 directory `<parent>/aac-baseline-XXXXXX`, removed in `finally` (success or exception).

 Refuses (StagingRefused, nothing created) when parent lies inside the repository, logs/ or runs/."""
    roots = _forbidden_roots()
    hit = _inside(Path(parent), roots)
    if hit is not None:
        raise StagingRefused(f"refusing to stage third-party output under {hit} (repository, logs/ or runs/)")
    d = Path(tempfile.mkdtemp(prefix=STAGING_PREFIX, dir=str(parent)))
    try:
        os.chmod(d, 0o700)
        if _inside(d, roots) is not None:  # e.g. a symlinked parent resolving into the repository
            raise StagingRefused(f"refusing: {d} resolves inside the repository, logs/ or runs/")
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
        if d.exists():
            raise BaselineFailed(f"could not remove the private staging directory {d}; remove it by hand")


def assert_text_free(obj, where: str = "result") -> None:
    """Every key and string value must match SAFE_STRING or be one of the four label names (TextLeak otherwise)."""
    from aac.data.fixtures import LABELS

    allowed = set(LABELS)

    def walk(o, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(k, f"{path}.<key>")
                walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")
        elif isinstance(o, str):
            if o not in allowed and not SAFE_STRING.fullmatch(o):
                raise TextLeak(f"{where}: {path} holds a string that is not an id, hash or label ({len(o)} chars)")
        elif o is not None and not isinstance(o, (bool, int, float)):
            raise TextLeak(f"{where}: {path} holds a {type(o).__name__}")

    walk(obj, where)


def items_lines(items: list[dict]) -> str:
    """Sorted JSON lines (sort_keys, compact): one per item, in the order given (callers pass row/claim order)."""
    return "".join(json.dumps(it, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n" for it in items)


def _tool(name: str):
    if name == "bodega_test_victims":
        from aac.baselines.bodega import run_test_victims
        return run_test_victims
    if name == "bodega_attack":
        from aac.baselines.bodega import run_attack
        return run_attack
    if name == "averitec_veracity":
        from aac.baselines.averitec_gold import run_averitec
        return run_averitec
    raise ValueError(f"unknown external tool {name!r}")


def run_external(config: Path, seed: int, force: bool, ident: dict | None = None) -> int:
    """Run and seal one external baseline. Exit codes: 0 sealed, 1 failed (not sealed), 2 refused."""
    from aac.run.clock import make_clock
    from aac.run.config import ExternalConfig, load_any
    from aac.run.env import write_env
    from aac.run.finalize import FinalizeRefused, RunRefused, claim_run_dir, finalize_run
    from aac.run.identity import identity, lock_path

    root = aac_paths.root()
    cfg = load_any(Path(config))
    if not isinstance(cfg, ExternalConfig):
        print(f"refusing: {config} is not an aac-external/1 config", file=sys.stderr)
        return 2
    ident = ident or identity(root, Path(config), seed)
    resolved, rid = ident["resolved"], ident["run_id"]
    tool = resolved["external"]["tool"]
    key = (resolved["regime"], SYSTEM, resolved["name"], seed)
    try:
        run_dir, target_dir, stale_ids = claim_run_dir(root, rid, key, force)
    except RunRefused as exc:
        print(str(exc), file=sys.stderr)
        return 2
    clock = make_clock()
    (run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
    write_env(run_dir, lock_path(root))
    started = clock.now_iso_ms()
    t0 = time.perf_counter()
    print(f"external {rid}: {resolved['name']} ({tool}) started")
    try:
        out = _tool(tool)(root, resolved)
    except (BaselineFailed, StagingRefused) as exc:
        print(f"external {rid}: FAILED, not sealed: {exc}", file=sys.stderr)
        return 1
    seconds = time.perf_counter() - t0
    result = {"schema": RESULT_SCHEMA, "tool": tool, "name": resolved["name"], **out["result"]}
    items = out["items"]
    assert_text_free(result, RESULT_JSON)
    assert_text_free(items, ITEMS_JSONL)
    (run_dir / RESULT_JSON).write_text(hashing.dumps_sorted(result), encoding="ascii")
    if items:
        (run_dir / ITEMS_JSONL).write_text(items_lines(items), encoding="ascii")
    n = int(out["n"])
    finished = clock.now_iso_ms()
    timings = [*out.get("timings", []), {"stage": tool, "seconds": round(seconds, 3), "items": n, "device": "cpu"}]
    with open(run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
        for t in timings:
            fh.write(json.dumps(t, sort_keys=True) + "\n")
    summary = {
        "configurations": [f"external/{resolved['name']}"],
        "items": {"items": n},
        "stages": [{"name": tool, "device": "cpu", "seconds": round(seconds, 3) if clock.source == "wall" else 0.0,
                    "items": n}],
        "depends_on": [],
    }
    try:
        digest = finalize_run(root, run_dir, ident, summary=summary, record_count=0, shards=[], clock=clock,
                              started=started, finished=finished, system=SYSTEM, determinism_level=DETERMINISM[tool],
                              target_dir=target_dir, stale_ids=stale_ids)
    except FinalizeRefused as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"external {rid}: {resolved['name']} sealed in {seconds:.0f} s, {out.get('headline', '')} SEALED {digest[:12]}")
    return 0


def indexed_run(root: Path, name: str, seed: int) -> str | None:
    """The run_id runs_dir/INDEX.json lists as sealed for (BASE, external, name, seed), if any."""
    from aac.run.index import load_index

    runs = aac_paths.runs_dir(root)
    hits = [r["run_id"] for r in load_index(runs / "INDEX.json")["runs"]
            if (r.get("regime"), r.get("system"), r.get("name"), int(r.get("seed", -1))) == ("BASE", SYSTEM, name, seed)
            and (runs / r["run_id"] / "SEALED").is_file()]
    return hits[0] if len(hits) == 1 else None


def cmd_run(args) -> int:
    from aac.run.config import load_any
    from aac.run.identity import identity

    root = aac_paths.root()
    todo = []
    for c in args.config:
        name = load_any(Path(c)).name
        rid = None if args.force else (indexed_run(root, name, args.seed) if args.reuse_indexed else None)
        if rid is not None:
            print(f"external: reusing sealed {rid} for {name} seed {args.seed} (--reuse-indexed; --force re-runs)")
            continue
        todo.append(Path(c))
    # all identities first: the first sealed run modifies the tracked runs/INDEX.json (a dirty tree for the next)
    idents = [(c, identity(root, c, args.seed)) for c in todo]
    status = 0
    for i, (cfg, ident) in enumerate(idents):
        rc = run_external(cfg, args.seed, args.force, ident=ident)
        status = status or rc
        if rc != 0 and getattr(args, "fail_fast", False):
            left = [str(c) for c, _ in idents[i + 1:]]
            print(f"external: --fail-fast: {cfg} exited {rc}; not starting {len(left)} remaining config(s)", file=sys.stderr)
            break
    return status


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.baselines.external")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="seal external baseline runs from aac-external/1 configs ")
    p.add_argument("--config", action="append", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reuse-indexed", action="store_true",
                   help="skip a config whose (BASE, external, name, seed) key runs/INDEX.json already lists as sealed")
    p.add_argument("--fail-fast", action="store_true",
                   help="stop at the first config that does not seal (the configs run one after another; the chunked "
                        "untargeted RD BERT-Attack, make baselines-attack-rd-chunks)")
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
