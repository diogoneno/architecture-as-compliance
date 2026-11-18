"""ssh and scp to the GPU host, configured by configs/remote.yaml.

Every remote command runs as `ssh <opts> <host> <cmd>` in BatchMode with a connect timeout, so no
prompt can hang the dispatcher. A transport failure (ssh exit 255, a timeout or a missing ssh
binary) raises Unreachable; a non-zero exit of the REMOTE command is returned, not raised, because
for the guards that exit code is the answer (`test -f sentinel` -> 1 means absent)."""
from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml

from aac.run.paths import root as project_root

SSH_TRANSPORT_EXIT = 255


class Unreachable(Exception):
    """ssh could not reach the host (exit 255, timeout, no binary). Callers exit 2."""

    exit_code = 2


@dataclass(frozen=True)
class Remote:
    host: str
    aac_home: str
    releases: str
    raw: str
    runs: str
    derived: str
    hf_home: str
    tools: str
    sentinel: str
    tmux_session_prefix: str
    ssh_opts: tuple[str, ...]
    torch_extra_by_driver: dict[str, str]
    gpu_query: str
    apps_query: str
    probe_network: tuple[str, ...]
    # : the user-level toolchain of a release on the host (no sudo; configs/host-changes.yaml)
    uv_bin: str = ""
    python_install_dir: str = ""
    uv_cache_dir: str = ""
    python_versions: tuple[str, ...] = ()

    def env_lines(self) -> list[str]:
        """`aac dispatch env`: shell exports for the scripts (single source of truth = remote.yaml)."""
        pairs = {
            "AAC_REMOTE_HOST": self.host, "AAC_HOME": self.aac_home, "AAC_RELEASES": self.releases,
            "AAC_RAW": self.raw, "AAC_RUNS": self.runs, "AAC_DERIVED": self.derived, "AAC_HF_HOME": self.hf_home,
            "AAC_TOOLS": self.tools, "AAC_SENTINEL": self.sentinel, "AAC_TMUX_PREFIX": self.tmux_session_prefix,
            "AAC_SSH_OPTS": " ".join(self.ssh_opts),
            "AAC_UV_BIN": self.uv_bin, "AAC_PYTHON_DIR": self.python_install_dir, "AAC_UV_CACHE": self.uv_cache_dir,
            "AAC_PYTHON_VERSIONS": " ".join(self.python_versions),
        }
        return [f"{k}={shlex.quote(v)}" for k, v in pairs.items()]


def load_remote(path: Path | None = None) -> Remote:
    path = Path(path) if path else project_root() / "configs" / "remote.yaml"
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return Remote(
        host=str(cfg["host"]), aac_home=str(cfg["aac_home"]), releases=str(cfg["releases"]), raw=str(cfg["raw"]),
        runs=str(cfg["runs"]), derived=str(cfg["derived"]), hf_home=str(cfg["hf_home"]), tools=str(cfg["tools"]),
        sentinel=str(cfg["sentinel"]), tmux_session_prefix=str(cfg["tmux_session_prefix"]),
        ssh_opts=tuple(str(o) for o in cfg.get("ssh_opts", [])),
        torch_extra_by_driver={str(k): str(v) for k, v in (cfg.get("torch_extra_by_driver") or {}).items()},
        gpu_query=str(cfg["nvidia_smi"]["gpu"]), apps_query=str(cfg["nvidia_smi"]["apps"]),
        probe_network=tuple(str(h) for h in cfg.get("probe_network", [])),
        uv_bin=str(cfg.get("uv_bin", "")), python_install_dir=str(cfg.get("python_install_dir", "")),
        uv_cache_dir=str(cfg.get("uv_cache_dir", "")),
        python_versions=tuple(str(v) for v in cfg.get("python_versions", [])),
    )


def ssh_argv(remote: Remote, cmd: str, *, env: dict[str, str] | None = None) -> list[str]:
    prefix = "".join(f"{k}={shlex.quote(v)} " for k, v in sorted((env or {}).items()))
    return ["ssh", *remote.ssh_opts, remote.host, prefix + cmd]


def ssh(cmd: str, *, remote: Remote | None = None, timeout: float = 120, stdin: str | bytes | None = None,
        env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    """Run `cmd` on the host. Output is decoded as UTF-8 (errors replaced); never raises on a
 non-zero remote exit; raises Unreachable on a transport failure."""
    remote = remote or load_remote()
    argv = ssh_argv(remote, cmd, env=env)
    data = stdin.encode("utf-8") if isinstance(stdin, str) else stdin
    try:
        proc = subprocess.run(argv, input=data, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Unreachable(f"ssh {remote.host}: {exc}") from exc
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode == SSH_TRANSPORT_EXIT:
        raise Unreachable(f"ssh {remote.host} exit 255: {err.strip()}")
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)


def scp(src: str | Path, dst: str, *, remote: Remote | None = None, timeout: float = 600) -> subprocess.CompletedProcess:
    """scp a local file to `<host>:<dst>` (BatchMode; same transport semantics as ssh)."""
    remote = remote or load_remote()
    argv = ["scp", *remote.ssh_opts, "-q", str(src), f"{remote.host}:{dst}"]
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Unreachable(f"scp {remote.host}: {exc}") from exc
    out = proc.stdout.decode("utf-8", errors="replace")
    err = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode == SSH_TRANSPORT_EXIT:
        raise Unreachable(f"scp {remote.host} exit 255: {err.strip()}")
    return subprocess.CompletedProcess(argv, proc.returncode, out, err)
