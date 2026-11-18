"""Clock sources : a real wall clock for real runs, a fixed clock under
AAC_RUN_EPOCH for smoke goldens, and git_sha for code identity."""
from __future__ import annotations

import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

_HEX40 = re.compile(r"^[0-9a-f]{40}$")


class DirtyTree(Exception):
    """git_sha refused: the working tree has uncommitted changes and AAC_GIT_SHA is unset."""


class Clock(Protocol):
    def now_iso_ms(self) -> str: ...
    def next_seq(self) -> int: ...


def _iso_ms(ms: int) -> str:
    dt = datetime.fromtimestamp(ms // 1000, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{ms % 1000:03d}Z"


class WallClock:
    source = "wall"

    def __init__(self) -> None:
        self._seq = 0

    def now_iso_ms(self) -> str:
        now = datetime.now(timezone.utc)
        return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"

    def next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        return s

    @property
    def seq(self) -> int:
        return self._seq

    def start_at(self, n: int) -> None:
        """Make the next issued seq equal n (the victim server resumes after its kept records)."""
        if int(n) < 0:
            raise ValueError("start_at needs n >= 0")
        self._seq = int(n)

    @property
    def epoch_s(self) -> int:
        return int(datetime.now(timezone.utc).timestamp())


class FixedClock:
    """timestamp = epoch_s*1000 + (last issued seq)*step_ms; records are a pure
 function of (run_epoch, seq)."""

    source = "fixed"

    def __init__(self, epoch_s: int, step_ms: int = 1) -> None:
        self.epoch_s = int(epoch_s)
        self.step_ms = int(step_ms)
        self._seq = 0
        self._last = -1

    def now_iso_ms(self) -> str:
        return _iso_ms(self.epoch_s * 1000 + max(self._last, 0) * self.step_ms)

    def next_seq(self) -> int:
        s = self._seq
        self._seq += 1
        self._last = s
        return s

    @property
    def seq(self) -> int:
        return self._seq

    def start_at(self, n: int) -> None:
        """Make the next issued seq equal n; the timestamp stays a function of the last issued seq."""
        if int(n) < 0:
            raise ValueError("start_at needs n >= 0")
        self._seq = int(n)
        self._last = int(n) - 1


def make_clock() -> WallClock | FixedClock:
    epoch = os.environ.get("AAC_RUN_EPOCH")
    if epoch is not None and epoch != "":
        return FixedClock(int(epoch))
    return WallClock()


SENTINEL_SHA = "0" * 40  # the smoke harness's fixed code identity (Makefile SMOKE_ENV, mk/smoke.mk)
RELEASE_SHA_FILE = "AAC_GIT_SHA"  # written into every git-archive release by aac.dispatch.sync


def _clean_head(cwd: Path | None) -> str:
    """`git rev-parse HEAD` when the archived subtree (cwd, i.e. aac-pipeline/) is clean. The check
 is scoped to `-- .` the analyst's untracked files at the repository root are outside
 the archived code and must not force an override."""
    target_cwd = str(cwd) if cwd else None
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", "."],
        capture_output=True,
        text=True,
        check=True,
        cwd=target_cwd,
    )
    if status.stdout.strip():
        raise DirtyTree("working tree is dirty; commit first or set AAC_GIT_SHA explicitly")
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=target_cwd,
    ).stdout.strip()
    if not _HEX40.match(sha):
        raise RuntimeError(f"unexpected git rev-parse output: {sha!r}")
    return sha


def git_sha(cwd: Path | None = None) -> str:
    """The code identity a run records 

 - AAC_GIT_SHA=0*40 (smoke sentinel): accepted only with AAC_RUN_EPOCH set (the smoke harness);
 - any other AAC_GIT_SHA: inside a release (an `AAC_GIT_SHA` file in cwd) it must equal that
 file; inside a git checkout it must equal HEAD with the subtree clean;
 - unset: `git rev-parse HEAD` on a clean subtree, else DirtyTree."""
    env = os.environ.get("AAC_GIT_SHA")
    if env:
        if not _HEX40.match(env):
            raise ValueError("AAC_GIT_SHA must be 40 lowercase hex characters")
        if env == SENTINEL_SHA:
            if not os.environ.get("AAC_RUN_EPOCH"):
                raise ValueError("AAC_GIT_SHA=0*40 is the smoke sentinel; it is accepted only under the smoke "
                                 "harness (AAC_RUN_EPOCH set), never for a real run")
            return env
        release = Path(cwd or ".") / RELEASE_SHA_FILE
        if release.is_file():
            recorded = release.read_text(encoding="ascii").strip()
            if recorded != env:
                raise ValueError(f"AAC_GIT_SHA {env[:12]} differs from the release's {RELEASE_SHA_FILE} file {recorded[:12]}")
            return env
        head = _clean_head(cwd)
        if env != head:
            raise ValueError(f"AAC_GIT_SHA {env[:12]} is not HEAD {head[:12]} of this checkout; unset it or check out that commit")
        return env
    return _clean_head(cwd)
