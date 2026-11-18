"""The GPU rule as code : gpu-run may start only when the analyst-created sentinel
exists AND nvidia-smi lists zero compute apps; gpu-push only from a clean tree .

Exit-code contract (used by the scripts and the smoke tests):
 0 free / clean 1 refused (GpuBusy, GpuNotCleared, DirtyTree) 2 Unreachable (ssh)
All remote output is parsed as UTF-8 CSV from `--format=csv,noheader`; the GPU name is kept as a
string and never used in a path."""
from __future__ import annotations

import csv
import io
import subprocess
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from aac.dispatch import remote as _remote
from aac.dispatch.remote import Remote, Unreachable  # noqa: F401 (re-exported for callers)

Runner = Callable[[str], subprocess.CompletedProcess]


class DispatchRefused(Exception):
    exit_code = 1


class GpuBusy(DispatchRefused):
    """nvidia-smi lists at least one compute app on the card."""


class GpuNotCleared(DispatchRefused):
    """the analyst's GPU_CLEARED sentinel is absent on the host."""


class DirtyTree(DispatchRefused):
    """`git status --porcelain` of the archived subtree is non-empty."""



# --query-compute-apps, so memory in use above this floor also counts as busy (an idle 3090 shows ~1 MiB).
MEMORY_BUSY_MIB = 512


@dataclass(frozen=True)
class GpuState:
    name: str | None
    driver: str | None
    memory_total_mib: int | None
    memory_used_mib: int | None
    compute_apps: list[dict] = field(default_factory=list)

    @property
    def busy(self) -> bool:
        mem_busy = self.memory_used_mib is not None and self.memory_used_mib > MEMORY_BUSY_MIB
        return len(self.compute_apps) > 0 or mem_busy

    def line(self) -> str:
        """The nvidia-smi line every dispatch prints (: name/driver/memory)."""
        return (f"nvidia-smi: {self.name}, driver {self.driver}, memory {self.memory_used_mib}/{self.memory_total_mib} MiB"
                f"; compute_apps={len(self.compute_apps)}")


def _rows(text: str) -> list[list[str]]:
    reader = csv.reader(io.StringIO(text), skipinitialspace=True)
    return [[c.strip() for c in row] for row in reader if any(c.strip() for c in row)]


def _mib(cell: str) -> int | None:
    head = cell.split()[0] if cell.split() else ""
    return int(head) if head.isdigit() else None


def parse_gpu_csv(text: str) -> tuple[str | None, str | None, int | None, int | None]:
    """`name, driver_version, memory.total, memory.used` of the first GPU (csv,noheader)."""
    rows = _rows(text)
    if not rows or len(rows[0]) < 4:
        return None, None, None, None
    name, driver, total, used = rows[0][:4]
    return name, driver, _mib(total), _mib(used)


def parse_compute_apps(text: str) -> list[dict]:
    """`pid, process_name` rows (csv,noheader). An empty listing is an empty list, never an error."""
    apps = []
    for row in _rows(text):
        pid = row[0] if row else ""
        name = row[1] if len(row) > 1 else ""
        apps.append({"pid": int(pid) if pid.isdigit() else pid, "process_name": name})
    return apps


def _default_runner(remote: Remote | None) -> Runner:
    rem = remote or _remote.load_remote()
    return lambda cmd: _remote.ssh(cmd, remote=rem)


def query_gpu(run: Runner | None = None, remote: Remote | None = None) -> GpuState:
    """Read name/driver/memory and the compute-app list from the host (no judgement)."""
    rem = remote or _remote.load_remote()
    run = run or _default_runner(rem)
    gpu = run(rem.gpu_query)
    apps = run(rem.apps_query)
    if gpu.returncode != 0:
        raise Unreachable(f"nvidia-smi failed on {rem.host} (exit {gpu.returncode}): {gpu.stderr.strip()}")
    name, driver, total, used = parse_gpu_csv(gpu.stdout)
    # An empty compute-apps listing exits 0 with no rows; a non-zero exit here is a real failure.
    if apps.returncode != 0:
        raise Unreachable(f"nvidia-smi --query-compute-apps failed on {rem.host} (exit {apps.returncode})")
    return GpuState(name, driver, total, used, parse_compute_apps(apps.stdout))


def check_gpu(run: Runner | None = None, remote: Remote | None = None) -> GpuState:
    """gpu-check: the state, or GpuBusy while any compute app holds the card."""
    state = query_gpu(run, remote)
    if state.busy:
        raise GpuBusy(state.line())
    return state


def sentinel_present(run: Runner | None = None, remote: Remote | None = None) -> bool:
    rem = remote or _remote.load_remote()
    run = run or _default_runner(rem)
    proc = run(f"test -f {rem.sentinel}")
    if proc.returncode not in (0, 1):
        raise Unreachable(f"sentinel test on {rem.host} exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.returncode == 0


def preflight_run(run: Runner | None = None, remote: Remote | None = None) -> GpuState:
    """gpu-run order: sentinel first, then gpu-check; stops at the first failure. Both are required:
 compute_apps == 0 without the sentinel refuses; the sentinel with compute_apps >= 1 refuses."""
    rem = remote or _remote.load_remote()
    run = run or _default_runner(rem)
    if not sentinel_present(run, rem):
        raise GpuNotCleared(f"GPU_CLEARED absent: {rem.sentinel} on {rem.host} (the analyst creates it)")
    return check_gpu(run, rem)


def git_status_porcelain(repo_root: Path, subtree: str = "aac-pipeline") -> str:
    proc = subprocess.run(["git", "-C", str(repo_root), "status", "--porcelain", "--", subtree],
                          capture_output=True, text=True, check=True)
    return proc.stdout


def assert_clean_tree(repo_root: Path | None = None, subtree: str = "aac-pipeline") -> None:
    """gpu-push guard: the archived subtree must have no modified, staged or untracked files, so the
 release named by the SHA is exactly the code that ran . Scoped to the subtree that
 `git archive HEAD aac-pipeline` ships; the analyst's untracked folders at the repo root
 (chapters/, appendix-c/, tools/) are not part of a release and do not block a dispatch."""
    from aac.run.paths import root as project_root

    root = Path(repo_root) if repo_root else project_root().parent
    dirty = git_status_porcelain(root, subtree)
    if dirty.strip():
        raise DirtyTree("dirty tree: commit first\n" + dirty.rstrip())


def driver_major(driver: str | None) -> int | None:
    if not driver:
        return None
    head = driver.split(".")[0]
    return int(head) if head.isdigit() else None


def torch_extra_for(driver: str | None, table: dict[str, str]) -> str | None:
    """The torch extra selected by the probed driver : the highest `>=N` rule the driver satisfies."""
    major = driver_major(driver)
    if major is None:
        return None
    best: tuple[int, str] | None = None
    for rule, extra in table.items():
        if not rule.startswith(">="):
            continue
        floor = int(rule[2:])
        if major >= floor and (best is None or floor > best[0]):
            best = (floor, extra)
    return best[1] if best else None
