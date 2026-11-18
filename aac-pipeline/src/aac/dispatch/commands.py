"""`aac dispatch <verb>`: check | sentinel | clean-tree | env | push | hf-push | pull | status | probe | plan.
Thin CLI over guard/sync/status/probe; the shell scripts in scripts/gpu_*.sh compose these.
Exit codes: 0 ok, 1 refused (GpuBusy / GpuNotCleared / DirtyTree / unsealed), 2 unreachable."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from aac.dispatch import guard, plan, probe, status, sync
from aac.dispatch import remote as _remote
from aac.dispatch.remote import Unreachable
from aac.run.paths import root as project_root


def _refused(exc: Exception, label: str) -> int:
    print(f"{label}: {exc}", file=sys.stderr)
    return getattr(exc, "exit_code", 1)


def cmd_check(args) -> int:
    try:
        state = guard.query_gpu()
    except Unreachable as exc:
        print(f"unreachable: {exc}", file=sys.stderr)
        print("compute_apps=unreachable")
        return 2
    print(state.line())
    for app in state.compute_apps:
        print(f"  pid {app['pid']}: {app['process_name']}")
    print(f"compute_apps={len(state.compute_apps)}")
    if state.busy:
        print("refusing: the card is held by another process (please free the device)", file=sys.stderr)
        return 1
    return 0


def cmd_sentinel(args) -> int:
    rem = _remote.load_remote()
    try:
        present = guard.sentinel_present(remote=rem)
    except Unreachable as exc:
        print(f"unreachable: {exc}", file=sys.stderr)
        return 2
    print(f"sentinel {rem.sentinel} on {rem.host}: {'present' if present else 'absent'}")
    return 0 if present else 1


def cmd_clean_tree(args) -> int:
    try:
        guard.assert_clean_tree()
    except guard.DirtyTree as exc:
        return _refused(exc, "refusing")
    print("clean tree")
    return 0


def cmd_env(args) -> int:
    for line in _remote.load_remote().env_lines():
        print(line)
    return 0


def _git(repo: Path, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *argv], capture_output=True, text=True)


def resolve_pinned_sha(sha: str, repo: Path) -> str:
    """--sha: a full 40-hex lowercase commit that exists and is reachable from HEAD . The
 release is `git archive SHA`, which cannot contain uncommitted bytes, so the tree is not inspected."""
    if not sync.is_full_sha(sha):
        raise guard.DispatchRefused(f"--sha must be a full 40-character lowercase hex commit, got {sha!r}")
    if _git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode != 0:
        raise guard.DispatchRefused(f"--sha {sha} is not a commit in {repo}")
    if _git(repo, "merge-base", "--is-ancestor", sha, "HEAD").returncode != 0:
        raise guard.DispatchRefused(f"--sha {sha} is not reachable from HEAD")
    return sha


def cmd_push(args) -> int:
    rem = _remote.load_remote()
    repo = project_root().parent
    try:
        if args.sha:
            sha = resolve_pinned_sha(args.sha, repo)
            print(f"pinned sha {sha}: the release is git archive {sha}; the working tree is not inspected")
        else:
            guard.assert_clean_tree(repo)
            sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    except guard.DispatchRefused as exc:
        return _refused(exc, "refusing")
    with tempfile.TemporaryDirectory(prefix="aac-derived-") as tmp:
        
        # preflight then re-hashes the persistent copy against the release's MANIFEST.sha256 (manifest-check item).
        try:
            derived = sync.archive_derived(sha, repo_root=repo, dest=Path(tmp))
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            return _refused(exc, "refusing")
        if args.dry_run:
            cmds = sync.release_commands(sha, rem)
            print(f"dry-run: release dir {rem.host}:{cmds['release']} (staged in {cmds['stage']})")
            for name, commit in sync.gitlinks(sha, repo_root=repo):
                print(f"dry-run: gitlink third_party/{name} @ {commit} -> {cmds['stage']}/third_party/{name}")
            if derived is None:
                print(f"dry-run: {sync.DERIVED_ARCHIVE_PATH} absent at {sha}: nothing to rsync")
            else:
                n = sum(1 for p in derived.rglob("*") if p.is_file())
                print(f"dry-run: data/derived from git archive {sha} ({n} files; the working tree is not read)")
                print("dry-run: rsync argv " + " ".join(sync.derived_rsync_argv(derived, rem)))
            print("dry-run: nothing contacted")
            return 0
        try:
            rel = sync.push_release(sha, repo_root=repo, remote=rem)
            print(f"release {sha} -> {rem.host}:{rel}")
            if derived is not None:
                argv = sync.derived_rsync_argv(derived, rem)
                proc = subprocess.run(argv, capture_output=True, text=True)
                if proc.returncode != 0:
                    print(f"rsync data/derived failed (exit {proc.returncode}): {proc.stderr.strip()}", file=sys.stderr)
                    return 1
                print(f"data/derived @ {sha} -> {rem.host}:{rem.derived} (git archive, rsync --checksum, private/ excluded)")
            else:
                print(f"{sync.DERIVED_ARCHIVE_PATH} absent at {sha}: nothing to rsync")
        except Unreachable as exc:
            return _refused(exc, "unreachable")
        except RuntimeError as exc:
            return _refused(exc, "refusing")
    return 0


def cmd_hf_push(args) -> int:
    """aac-pipeline/.hf/ -> hf_home on the host, rsync -rlt --checksum (links kept, never a delete flag):
 the hash-verified pinned snapshots of never re-fetched with `hf download` (refs/main)."""
    rem = _remote.load_remote()
    src = project_root() / ".hf"
    if not src.is_dir():
        print(f"refusing: {src} absent ", file=sys.stderr)
        return 1
    argv = sync.rsync_argv(str(src) + "/", f"{rem.host}:{rem.hf_home}/", extra=("--info=stats1",), remote=rem)
    print(" ".join(argv))
    if args.dry_run:
        return 0
    proc = subprocess.run(argv)
    if proc.returncode == 255:
        print(f"unreachable: rsync to {rem.host}", file=sys.stderr)
        return 2
    return 0 if proc.returncode == 0 else 1


def cmd_pull(args) -> int:
    from aac.run.index import append_index, entry_from_run_dir
    from aac.run.seal import UnsealedRun

    runs_dir = Path(args.runs_dir) if args.runs_dir else project_root() / "runs"
    try:
        if args.no_sync:
            run_dir = runs_dir / args.run_id
        else:
            run_dir = sync.pull_run(args.run_id, runs_dir=runs_dir)
    except Unreachable as exc:
        return _refused(exc, "unreachable")
    if not (run_dir / "SEALED").is_file():
        print(f"refusing: {run_dir} has no SEALED manifest; the run is not finished or not sealed", file=sys.stderr)
        return 1
    try:
        entry = entry_from_run_dir(run_dir)
    except (UnsealedRun, FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 1
    append_index(runs_dir / "INDEX.json", entry)
    print(f"pulled {args.run_id}: SEALED {entry['sealed_sha256'][:12]} verified; INDEX.json updated")
    return 0


def cmd_status(args) -> int:
    repo = project_root().parent
    sha = args.sha or subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    try:
        status.check_phase(args.phase)
        print(status.tail_status(sha, phase=args.phase, lines=args.lines), end="")
    except ValueError as exc:  
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    except Unreachable as exc:
        return _refused(exc, "unreachable")
    return 0


def cmd_probe(args) -> int:
    dest = probe.write_probe(args.host)
    print(f"probe: {dest.relative_to(project_root())}")
    return 0


def register(sub) -> None:
    dp = sub.add_parser("dispatch").add_subparsers(dest="sub", required=True)
    dp.add_parser("check").set_defaults(fn=cmd_check)
    dp.add_parser("sentinel").set_defaults(fn=cmd_sentinel)
    dp.add_parser("clean-tree").set_defaults(fn=cmd_clean_tree)
    dp.add_parser("env").set_defaults(fn=cmd_env)
    p = dp.add_parser("push"); p.add_argument("--sha", default=None, help="ship git archive SHA (an ancestor of HEAD) instead of a clean HEAD")
    p.add_argument("--dry-run", action="store_true", help="print the release dir, gitlinks and rsync argv; contact nothing")
    p.set_defaults(fn=cmd_push)
    p = dp.add_parser("hf-push"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_hf_push)
    p = dp.add_parser("pull"); p.add_argument("--run-id", required=True); p.add_argument("--runs-dir", default=None)
    p.add_argument("--no-sync", action="store_true", help="verify and index an already-local run directory"); p.set_defaults(fn=cmd_pull)
    p = dp.add_parser("status"); p.add_argument("--phase", default=None); p.add_argument("--sha", default=None)
    p.add_argument("--lines", type=int, default=20); p.set_defaults(fn=cmd_status)
    p = dp.add_parser("probe"); p.add_argument("--host", required=True); p.set_defaults(fn=cmd_probe)
    plan.register(dp)
