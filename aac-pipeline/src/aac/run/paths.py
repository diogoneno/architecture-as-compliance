"""Project root resolution independent of the current working directory."""
from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    """Return the aac-pipeline root (the directory holding pyproject.toml).

 AAC_ROOT overrides; otherwise the editable install location is used
 (src/aac/run/paths.py -> parents[3] == aac-pipeline).
 """
    env = os.environ.get("AAC_ROOT")
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    candidate = here.parents[3]
    if not (candidate / "pyproject.toml").is_file():
        raise RuntimeError(f"cannot locate aac-pipeline root from {here}; set AAC_ROOT")
    return candidate


def out_root(root_: Path | str | None = None) -> Path:
    """The output namespace: where runs/, reports/ and evidence-pack/ live .

 AAC_OUT selects it: an absolute value is used as given, a relative one is resolved against
 the project root (`root_` when the caller passes its root, else root), and unset means the
 project root itself (the real namespace). The stub smoke chain exports AAC_OUT=build/smoke, so
 stub runs, stub tables and the stub pack never share a directory with real artifacts.
 runs/_probes/ is NOT under out_root: host probes are real-namespace facts (root/runs/_probes)."""
    base = Path(root_) if root_ is not None else root()
    env = os.environ.get("AAC_OUT")
    if not env:
        return base
    out = Path(env)
    return out if out.is_absolute() else base / out


def runs_dir(root_: Path | str | None = None) -> Path:
    """out_root/runs: sealed runs and runs/INDEX.json of the selected namespace."""
    return out_root(root_) / "runs"


def reports_dir(root_: Path | str | None = None) -> Path:
    """out_root/reports: the shell CSVs, long sidecars, figures and reports/INDEX.json."""
    return out_root(root_) / "reports"


def pack_dir(root_: Path | str | None = None) -> Path:
    """out_root/evidence-pack: the default `aac pack build` target."""
    return out_root(root_) / "evidence-pack"


def probes_dir(root_: Path | str | None = None) -> Path:
    """root/runs/_probes: host probes are real-namespace facts, never under AAC_OUT (see out_root)."""
    base = Path(root_) if root_ is not None else root()
    return base / "runs" / "_probes"


def listdir(path: Path) -> list[str]:
    """Directory listing helper.

 Returns names sorted unless AAC_LISTING=shuffled, in which case the order is
 shuffled so that any order dependence in a caller surfaces in test runs. Callers
 MUST sort what they consume; this helper exists to prove that they do.
 """
    names = os.listdir(path)
    if os.environ.get("AAC_LISTING") == "shuffled":
        import random

        random.Random().shuffle(names)
        return names
    return sorted(names)
