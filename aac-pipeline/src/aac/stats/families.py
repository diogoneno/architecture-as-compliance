"""Holm families and the frozen statistics settings, read from configs/stats.yaml (the
file specification freezes). No test is chosen in code: regime -> ordered list of H1..H4 ids."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from aac.run.paths import root as project_root


def stats_path(root: Path | None = None) -> Path:
    return (Path(root) if root is not None else project_root()) / "configs" / "stats.yaml"


@lru_cache(maxsize=4)
def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "stats" not in data:
        raise ValueError(f"{path}: expected a top-level `stats:` mapping")
    return data["stats"]


def settings(root: Path | None = None) -> dict:
    """The whole frozen `stats:` block (bootstrap, level, primary/secondary/descriptive tests, correction, families)."""
    return dict(_load(str(stats_path(root))))


def family(regime: str, root: Path | None = None) -> list[str]:
    fams = settings(root).get("families") or {}
    if regime not in fams:
        raise ValueError(f"no Holm family for regime {regime!r}; configs/stats.yaml defines {sorted(fams)}")
    ids = list(fams[regime])
    if not ids:
        raise ValueError(f"Holm family for {regime!r} is empty")
    return ids
