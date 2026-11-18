"""configs/run-matrix.yaml: the single source of the planned (config, seeds) run set.

Read by mk/runs.mk (`aac matrix list`), `aac matrix count`, `aac index rebuild --check-matrix`
and tests/smoke/test_mini_runs.py. No run count is typed anywhere else: every assertion
compares against len(expand)."""
from __future__ import annotations

from pathlib import Path

import yaml

MATRIX_REL = Path("configs") / "run-matrix.yaml"
REGIMES_REL = Path("configs") / "regimes"


def matrix_path(root: Path, path: Path | str | None = None) -> Path:
    if path is None:
        return Path(root) / MATRIX_REL
    p = Path(path)
    return p if p.is_absolute() else Path(root) / p


def load_matrix(root: Path, path: Path | str | None = None) -> list[dict]:
    mp = matrix_path(root, path)
    with open(mp, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    entries = data.get("matrix") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{mp}: expected a non-empty `matrix:` list")
    seen: set[str] = set()
    for e in entries:
        cfg = e.get("config")
        seeds = e.get("seeds")
        if not isinstance(cfg, str) or not isinstance(seeds, list) or not seeds:
            raise ValueError(f"{mp}: every entry needs `config` and a non-empty `seeds` list: {e}")
        if cfg in seen:
            raise ValueError(f"{mp}: {cfg} is listed twice")
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"{mp}: duplicate seeds for {cfg}")
        if not (Path(root) / cfg).is_file():
            raise ValueError(f"{mp}: {cfg} does not exist")
        seen.add(cfg)
    regimes = {p.relative_to(root).as_posix() for p in (Path(root) / REGIMES_REL).glob("*.yaml")}
    unlisted = sorted(regimes - seen)
    if unlisted:
        raise ValueError(f"{mp}: configs/regimes files not in the matrix: {unlisted}")
    return entries


def expand(root: Path, path: Path | str | None = None) -> list[tuple[str, int]]:
    """[(config_rel, seed)] in matrix order."""
    return [(e["config"], int(s)) for e in load_matrix(root, path) for s in e["seeds"]]


def count(root: Path, path: Path | str | None = None) -> int:
    return len(expand(root, path))


def expand_run_ids(root: Path, path: Path | str | None = None) -> list[tuple[str, int, str]]:
    """[(config_rel, seed, run_id)] under the current identity inputs (AAC_GIT_SHA, lockfile, data_version)."""
    from aac.run.identity import identity

    out = []
    for cfg, seed in expand(root, path):
        out.append((cfg, seed, identity(Path(root), Path(root) / cfg, seed)["run_id"]))
    if len({r for _, _, r in out}) != len(out):
        raise ValueError("the matrix expands to duplicate run_ids")
    return out
