"""gpu-status: the heartbeat STATUS.json and the tail of the newest run log of a release ."""
from __future__ import annotations

import re
import shlex

from aac.dispatch import remote as _remote
from aac.dispatch.remote import Remote
from aac.dispatch.sync import release_dir


PHASE_RE = re.compile(r"[A-Za-z0-9_-]+")  


def check_phase(phase: str | None) -> str | None:
    """ PHASE reaches a remote shell; refuse anything outside ^[A-Za-z0-9_-]+$."""
    if phase is not None and not PHASE_RE.fullmatch(phase):
        raise ValueError(f"PHASE {phase!r} must match ^[A-Za-z0-9_-]+$")
    return phase


def tail_status(sha: str, *, phase: str | None = None, lines: int = 20, remote: Remote | None = None,
                run=None) -> str:
    rem = remote or _remote.load_remote()
    run = run or (lambda cmd: _remote.ssh(cmd, remote=rem))
    rel = release_dir(rem, sha)
    check_phase(phase)
    session = f"{rem.tmux_session_prefix}{phase}" if phase else None
    q = shlex.quote
    parts = [
        f"cd {q(rel)} 2>/dev/null || {{ echo {q('release absent: ' + rel)}; exit 3; }}",
        "echo '== STATUS.json'; cat STATUS.json 2>/dev/null || echo '(no STATUS.json yet)'",
        f"echo '== newest log'; log=$(ls -1t logs/{shlex.quote(phase) + '-' if phase else ''}*.log 2>/dev/null | head -1); "
        f"if [ -n \"$log\" ]; then echo \"$log\"; tail -n {int(lines)} \"$log\"; else echo '(no log yet)'; fi",
        "echo '== tmux'; tmux ls 2>/dev/null || echo '(no tmux server)'",
    ]
    if session:
        parts.append(f"tmux has-session -t {q(session)} 2>/dev/null && echo {q(f'session {session}: running')} "
                     f"|| echo {q(f'session {session}: not running')}")
    proc = run("; ".join(parts))
    return proc.stdout + (proc.stderr if proc.returncode not in (0, 3) else "")
