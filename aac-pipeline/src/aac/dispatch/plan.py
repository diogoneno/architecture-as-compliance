"""`aac dispatch plan`: the checkpoint printer (shows its output to the analyst).

Reads the GPU batch files in the order they will run (`TARGET CONFIG SEED` per line, the run.sh --batch format)
and an estimates YAML, then prints every run in execution order with its GPU-hour estimate, its source and a
running total, followed by the totals and the announced ceiling. It refuses (exit 1) when
 - a batch file names no run ("announce zero runs" can never happen),
 - a line has no estimate (no run is named without one),
 - a line comes before the line that produces what it reads (across all batches, in the given order):
 an inference line (run, bench, attack) before the C1 training line its `components.c1.trained_by`
 (`bench.c1.trained_by`) names at the same seed ("inference before its training run"),
 a scorer training line before the evidence line its `scorer.evidence_from` names,
 a run line before the scorer training line its `components.c2.scorer_trained_by` names at
 `components.c2.scorer_seed` (or the run seed when that is null), or before the evidence line its
 `components.c2.evidence_from` names, an evidence line before the evidence line its `evidence.pool_from` names,
 an attack line before the R1 run line its `attack.victim_source` names at the same seed, a run line before
 the R1 run line its `components.c3.fitted_from` names at the same seed or, for
 `fitted_from: R4/<name>`, before the R4 run line `<name>` at the same seed , or a store-replay run
 line before the R3 attack lines its `replay.sources` name at `replay.seed` ;
 a producer absent from the plan was sealed in an earlier session and never blocks,
 - the plan's own high estimates add up to more than `ceiling_gpu_hours`.
Configs are read as merged YAML (aac.run.config.load_raw: `extends` and `!replace` applied), never validated.
Pure file reading: imports no ML library (no CUDA can start here) and never contacts the host."""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from aac.run.paths import root as project_root

TARGETS = ("run", "train", "bench", "evidence", "attack")
R4_PREFIX = "R4/"                 # = aac.regimes.common.R4_SOURCE_PREFIX (this module imports no regime code)
STORE_REPLAY_KIND = "private-store"   # = aac.run.config.STORE_REPLAY_KIND


class RunEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target: str
    config: str
    seed: int
    batch: str
    est_gpu_hours: tuple[float, float]
    source: str

    @field_validator("est_gpu_hours")
    @classmethod
    def _ordered(cls, v: tuple[float, float]) -> tuple[float, float]:
        lo, hi = v
        if lo < 0 or hi < lo:
            raise ValueError(f"est_gpu_hours must be [low, high] with 0 <= low <= high, got {list(v)}")
        return v

    @field_validator("target")
    @classmethod
    def _target(cls, v: str) -> str:
        if v not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {v!r}")
        return v


class Estimates(BaseModel):
    model_config = ConfigDict(extra="forbid")
    ceiling_gpu_hours: float
    projection_rule: str
    runs: list[RunEstimate]


class PlanRefused(Exception):
    exit_code = 1


def load_estimates(path: Path) -> Estimates:
    """The estimates YAML, validated (extra keys anywhere are an error). Raises ValueError."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    try:
        return Estimates.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"{path}: {exc}") from exc


def _norm(path: str | Path, root: Path) -> str:
    """A batch path as a root-relative POSIX string when it lies under root, else as given."""
    p = Path(path)
    absolute = p if p.is_absolute() else (Path.cwd() / p)
    try:
        return absolute.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def read_batch(path: Path) -> list[tuple[int, str, str, int]]:
    """(line number, target, config, seed) for every non-blank, non-comment line; malformed lines refuse."""
    out = []
    for lineno, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 3 or parts[0] not in TARGETS or not parts[2].isdigit():
            raise PlanRefused(f"batch {path} line {lineno}: expected 'TARGET CONFIG SEED' with TARGET in {TARGETS}: {line!r}")
        out.append((lineno, parts[0], parts[1], int(parts[2])))
    if not out:
        raise PlanRefused(f"batch {path} names no run")
    return out


Item = tuple[str, str, int]   # (kind, name, seed): kind EV, SL, C1, C2S or a regime (R1, R2, R3, ...)


def _raw(config: str, root: Path) -> dict:
    from aac.run.config import load_raw

    path = root / config
    if not path.is_file():
        raise PlanRefused(f"config {config} not found under {root}")
    return load_raw(path)


def _block(raw: dict, *keys: str) -> dict:
    cur: object = raw
    for key in keys:
        cur = cur.get(key) if isinstance(cur, dict) else None
    return cur if isinstance(cur, dict) else {}


def _names(config: str, raw: dict) -> set[str]:
    """The names a producing line answers to: its file stem and, when present, its `name` field."""
    return {Path(config).stem} | ({str(raw["name"])} if raw.get("name") else set())


def _is_scorer(raw: dict) -> bool:
    return raw.get("schema") == "aac-train-scorer/1" or bool(_block(raw, "scorer"))


def line_deps(target: str, config: str, seed: int, root: Path) -> tuple[set[Item], list[Item]]:
    """(produces, needs) of one batch line, from the merged YAML of its config ."""
    raw = _raw(config, root)
    names = _names(config, raw)
    produces: set[Item] = set()
    needs: list[Item] = []
    if target == "evidence":
        produces |= {("EV", n, seed) for n in names}
        ev = _block(raw, "evidence")
        if ev.get("shortlist_from"):
            needs.append(("SL", str(ev["shortlist_from"]), 0))
        if ev.get("pool_from"):
            needs.append(("EV", str(ev["pool_from"]), 0))
    elif target == "train":
        if _is_scorer(raw):
            produces |= {("C2S", n, seed) for n in names}
            if _block(raw, "scorer").get("evidence_from"):
                needs.append(("EV", str(_block(raw, "scorer")["evidence_from"]), 0))
        else:
            produces |= {("C1", n, seed) for n in names}
    else:  # run, bench, attack: inference lines
        
        c1 = _block(raw, "components", "c1") or _block(raw, "bench", "c1")
        if c1.get("trained_by"):
            needs.append(("C1", str(c1["trained_by"]), seed))
        c2 = _block(raw, "components", "c2")
        if c2.get("scorer_trained_by"):
            s_seed = c2.get("scorer_seed")
            needs.append(("C2S", str(c2["scorer_trained_by"]), seed if s_seed is None else int(s_seed)))
        if c2.get("evidence_from"):
            needs.append(("EV", str(c2["evidence_from"]), 0))
        if target == "attack" and _block(raw, "attack").get("victim_source"):
            needs.append(("R1", str(_block(raw, "attack")["victim_source"]), seed))
        
        # : `R4/<name>` names the seed-matched R4 run instead (aac.regimes.common.source_ref)
        fitted = _block(raw, "components", "c3").get("fitted_from")
        if fitted:
            fitted = str(fitted)
            needs.append(("R4", fitted[len(R4_PREFIX):], seed) if fitted.startswith(R4_PREFIX) else ("R1", fitted, seed))
        # : a store replay reads the sealed R3 runs of replay.sources at replay.seed (their private store)
        replay = _block(raw, "replay")
        if replay.get("kind") == STORE_REPLAY_KIND:
            needs.extend(("R3", str(src), int(replay.get("seed", 0))) for src in replay.get("sources") or [])
        if target in ("run", "attack") and raw.get("regime"):
            produces |= {(str(raw["regime"]), n, seed) for n in names}
    return produces, needs


def _violation(config: str, seed: int, need: Item) -> str:
    kind, name, nseed = need
    if kind == "C1":
        return f"inference before its training run (`{config}` seed {seed} needs `{name}` seed {nseed}, which runs later)"
    return (f"consumer before its producer (`{config}` seed {seed} needs {kind} `{name}` seed {nseed}, "
            f"which runs later)")


def build_plan(batches: list[Path], estimates: Estimates, *, root: Path | None = None) -> list[dict]:
    """The runs in execution order with their estimate; raises PlanRefused on any violated rule.

 One generic ordering rule : each line produces and needs (kind, name, seed) items; a need is violated
 only when no earlier line produced it and a later line does (a producer absent from the plan was sealed before)."""
    root = Path(root) if root else project_root()
    index = {(e.target, e.config, e.seed, _norm(e.batch, root)): e for e in estimates.runs}
    lines = [(batch, lineno, target, config, seed)
             for batch in batches for lineno, target, config, seed in read_batch(batch)]
    deps = [line_deps(target, config, seed, root) for _, _, target, config, seed in lines]
    produced_later: dict[Item, int] = {}
    for i, (produces, _) in enumerate(deps):
        for item in produces:
            produced_later[item] = i          # last index that produces the item
    rows: list[dict] = []
    produced: set[Item] = set()
    for i, (batch, lineno, target, config, seed) in enumerate(lines):
        bkey = _norm(batch, root)
        est = index.get((target, config, seed, bkey))
        if est is None:
            raise PlanRefused(f"batch {batch} line {lineno}: no GPU-hour estimate for `{target} {config} {seed}` "
                              f"(batch {bkey}) in the estimates file")
        produces, needs = deps[i]
        for need in needs:
            if need not in produced and produced_later.get(need, -1) > i:
                raise PlanRefused(f"batch {batch} line {lineno}: {_violation(config, seed, need)}")
        produced |= produces
        rows.append({"batch": bkey, "line": lineno, "target": target, "config": config, "seed": seed,
                     "low": est.est_gpu_hours[0], "high": est.est_gpu_hours[1], "source": est.source})
    return rows


def print_plan(batches: list[Path], estimates: Estimates, *, root: Path | None = None) -> int:
    """Print the checkpoint table; 0 when the plan may be announced, 1 when it is refused (reason on stderr)."""
    try:
        rows = build_plan([Path(b) for b in batches], estimates, root=root)
    except PlanRefused as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return exc.exit_code
    print(f"{'#':>3}  {'batch':<28} {'target':<8} {'config':<52} {'seed':>4}  {'est GPU-h':>11}  {'running':>11}  source")
    lo_sum = hi_sum = 0.0
    for i, r in enumerate(rows, start=1):
        lo_sum += r["low"]
        hi_sum += r["high"]
        est, run = f"{r['low']:.2f}-{r['high']:.2f}", f"{lo_sum:.2f}-{hi_sum:.2f}"
        print(f"{i:>3}  {r['batch']:<28} {r['target']:<8} {r['config']:<52} {r['seed']:>4}  {est:>11}  {run:>11}  {r['source']}")
    print(f"total: {len(rows)} runs, {lo_sum:.2f}-{hi_sum:.2f} GPU-h (ceiling {estimates.ceiling_gpu_hours} GPU-h)")
    print(f"projection rule: {estimates.projection_rule}")
    if hi_sum > estimates.ceiling_gpu_hours:
        print(f"refusing: announced ceiling exceeded by the plan itself ({hi_sum:.2f} > {estimates.ceiling_gpu_hours} GPU-h)",
              file=sys.stderr)
        return 1
    return 0


def cmd_plan(args) -> int:
    try:
        est = load_estimates(Path(args.estimates))
    except (OSError, ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    return print_plan([Path(b) for b in args.batch], est)


def register(dp) -> None:
    p = dp.add_parser("plan", help="execution plan: runs in execution order with GPU-hour estimates and totals")
    p.add_argument("--batch", action="append", required=True, help="batch file in execution order (repeatable)")
    p.add_argument("--estimates", required=True, help="estimates YAML (ceiling_gpu_hours, projection_rule, runs)")
    p.set_defaults(fn=cmd_plan)
