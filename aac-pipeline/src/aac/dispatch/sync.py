"""Transport between cpu-host and the GPU host: rsync --checksum (never --delete, /) and
`git archive <sha> aac-pipeline | ssh tar -x` for releases (no .git on the host)."""
from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

from aac.dispatch import remote as _remote
from aac.dispatch.remote import Remote, Unreachable

FORBIDDEN_FLAGS = ("--delete", "--delete-excluded", "--delete-after", "--delete-before", "--delete-during", "--delete-delay")
BASE_FLAGS = ("-rlt", "--checksum", "--no-perms", "--no-owner", "--no-group")


def rsync_argv(src: str, dst: str, *, extra: tuple[str, ...] = (), remote: Remote | None = None) -> list[str]:
    for flag in extra:
        if flag.split("=")[0] in FORBIDDEN_FLAGS:
            raise ValueError(f"refusing rsync flag {flag}: the manifests on /data/aac are authoritative, nothing is deleted")
    rem = remote or _remote.load_remote()
    ssh_cmd = " ".join(["ssh", *rem.ssh_opts])
    return ["rsync", *BASE_FLAGS, "-e", ssh_cmd, *extra, src, dst]


def rsync_checksum(src: str, dst: str, *, extra: tuple[str, ...] = (), remote: Remote | None = None,
                   timeout: float = 24 * 3600) -> subprocess.CompletedProcess:
    argv = rsync_argv(src, dst, extra=extra, remote=remote)
    try:
        proc = subprocess.run(argv, capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise Unreachable(f"rsync: {exc}") from exc
    return subprocess.CompletedProcess(argv, proc.returncode, proc.stdout.decode("utf-8", "replace"),
                                       proc.stderr.decode("utf-8", "replace"))


def release_dir(remote: Remote, sha: str) -> str:
    if not is_full_sha(sha):
        raise ValueError(f"not a full git sha: {sha!r}")
    return f"{remote.releases}/{sha}"


SUBMODULE_PREFIX = "aac-pipeline/third_party/"
DERIVED_EXCLUDES = ("--exclude", "private/")   
RAW_SUBDIRS = ("averitec", "incrediblae")       # replaced by links; the archived MANIFEST.expected.sha256 stays


def is_full_sha(sha: str) -> bool:
    return len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def gitlinks(sha: str, *, repo_root: Path, prefix: str = SUBMODULE_PREFIX) -> list[tuple[str, str]]:
    """(name, commit) for every 160000 (gitlink) entry under `prefix` in the tree of `sha`."""
    proc = subprocess.run(["git", "-C", str(repo_root), "ls-tree", sha, prefix], capture_output=True, text=True, check=True)
    out = []
    for line in proc.stdout.splitlines():
        meta, _, path = line.partition("\t")
        mode, kind, obj = meta.split()
        if mode == "160000" and kind == "commit":
            out.append((path[len(prefix):], obj))
    return out


DERIVED_ARCHIVE_PATH = "aac-pipeline/data/derived"


def archive_derived(sha: str, *, repo_root: Path, dest: Path) -> Path | None:
    """Extract data/derived of commit `sha` into `dest` (`git archive <sha> aac-pipeline/data/derived | tar -x`)
 and return the extracted data/derived directory, or None when the commit has no data/derived.

 the release's code and data_version.txt come from `sha`, so the derived files must too,
 never local working tree (dirty with parallel executors). data/derived/private/ is gitignored, so the archive
 cannot contain it; a private/ entry in the archive is refused all the same ."""
    if not is_full_sha(sha):
        raise ValueError(f"not a full git sha: {sha!r}")
    ls = subprocess.run(["git", "-C", str(repo_root), "ls-tree", "-d", "--name-only", sha, DERIVED_ARCHIVE_PATH],
                        capture_output=True, text=True, check=True)
    if not ls.stdout.strip():
        return None
    arc = subprocess.run(["git", "-C", str(repo_root), "archive", "--format=tar", sha, DERIVED_ARCHIVE_PATH],
                         capture_output=True, check=True)
    Path(dest).mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arc.stdout, capture_output=True, check=True)
    out = Path(dest) / DERIVED_ARCHIVE_PATH
    if (out / "private").exists():
        raise RuntimeError(f"git archive {sha} holds data/derived/private/: refusing to ship it ")
    return out


def derived_rsync_argv(local_derived: Path, remote: Remote) -> list[str]:
    """data/derived -> the host, without data/derived/private/ (the RD lane-B key material stays here)."""
    return rsync_argv(str(local_derived) + "/", f"{remote.host}:{remote.derived}/", extra=DERIVED_EXCLUDES, remote=remote)


def release_commands(sha: str, remote: Remote) -> dict[str, str]:
    """The remote shell commands of one release push, by step (also printed by `push --dry-run`).

 The release is assembled in releases/.staging-<sha>/ and renamed into place only after every step
 succeeded, so a release directory that carries AAC_GIT_SHA is complete; a finished release is never
 re-extracted (a SHA names exactly one tree). tar never writes through a link into /data/aac."""
    rel = release_dir(remote, sha)
    stage = f"{remote.releases}/.staging-{sha}"
    q = shlex.quote
    raw_links = [f"ln -s {q(remote.raw + '/' + d)} data/raw/{d}" for d in RAW_SUBDIRS]
    main = " && ".join([
        f"rm -rf {q(stage)}",
        f"mkdir -p {q(stage)} {q(remote.runs)} {q(remote.raw)} {q(remote.derived)} {q(remote.hf_home)}",
        f"tar -x --strip-components=1 -C {q(stage)}",
        f"cd {q(stage)}",
        # an EMPTY real index on first use, never a copy of the archived stub index (operational constraint / 11)
        f"{{ test -f {q(remote.runs)}/INDEX.json || printf '%s\\n' '{{\"runs\": []}}' > {q(remote.runs)}/INDEX.json; }}",
        "rm -rf runs data/derived " + " ".join(f"data/raw/{d}" for d in RAW_SUBDIRS),
        f"ln -s {q(remote.runs)} runs",
        "mkdir -p data/raw",
        *raw_links,
        f"ln -s {q(remote.derived)} data/derived",
        f"ln -s {q(remote.hf_home)} .hf",
    ])
    finish = " && ".join([
        f"cd {q(stage)}",
        f"printf '%s\\n' {q(sha)} > AAC_GIT_SHA",
        "cd /",
        f"mv -T {q(stage)} {q(rel)}",
    ])
    return {"release": rel, "stage": stage, "exists": f"test \"$(cat {q(rel)}/AAC_GIT_SHA 2>/dev/null)\" = {q(sha)}",
            "main": main, "finish": finish}


def submodule_command(stage: str, name: str) -> str:
    q = shlex.quote
    if "/" in name or name.startswith(".") or not name:
        raise ValueError(f"bad submodule name {name!r}")
    return f"mkdir -p {q(stage)}/third_party/{q(name)} && tar -x -C {q(stage)}/third_party/{q(name)}"


def push_release(sha: str, *, repo_root: Path, remote: Remote | None = None) -> str:
    """`git archive <sha> aac-pipeline` -> releases/<sha>/ on the host (+ AAC_GIT_SHA), with runs/,
 data/raw/{averitec,incrediblae} and data/derived replaced by links into the persistent /data/aac tree
 (the archived data/raw/MANIFEST.expected.sha256 is kept), `.hf -> hf_home`, and every gitlink under
 third_party/ extracted from the cpu-host submodule checkout at the recorded commit (git archive ships
 gitlinks as empty directories; operational constraint). Returns the remote release directory."""
    rem = remote or _remote.load_remote()
    if not is_full_sha(sha):
        raise ValueError(f"not a full git sha: {sha!r}")
    cmds = release_commands(sha, rem)
    links = gitlinks(sha, repo_root=repo_root)
    for name, commit in links:  # every submodule commit must be present locally before anything is sent
        sub = Path(repo_root) / SUBMODULE_PREFIX / name
        chk = subprocess.run(["git", "-C", str(sub), "cat-file", "-e", f"{commit}^{{commit}}"], capture_output=True)
        if chk.returncode != 0:
            raise RuntimeError(f"submodule {name}: commit {commit} is not in {sub} (git submodule update --init)")
    if _remote.ssh(cmds["exists"], remote=rem).returncode == 0:
        return cmds["release"]
    archive = subprocess.run(["git", "-C", str(repo_root), "archive", "--format=tar", sha, "aac-pipeline"],
                             capture_output=True, check=True)
    steps: list[tuple[str, str, bytes]] = [("main archive", cmds["main"], archive.stdout)]
    for name, commit in links:
        sub = Path(repo_root) / SUBMODULE_PREFIX / name
        tar = subprocess.run(["git", "-C", str(sub), "archive", "--format=tar", commit], capture_output=True, check=True)
        steps.append((f"third_party/{name} @ {commit[:12]}", submodule_command(cmds["stage"], name), tar.stdout))
    steps.append(("finish", cmds["finish"], b""))
    for label, cmd, data in steps:
        proc = _remote.ssh(cmd, remote=rem, stdin=data, timeout=1800)
        if proc.returncode != 0:
            raise RuntimeError(f"release step '{label}' failed on {rem.host} (exit {proc.returncode}): {proc.stderr.strip()}")
    return cmds["release"]


def pull_run(run_id: str, *, runs_dir: Path, remote: Remote | None = None) -> Path:
    """rsync --checksum <host>:runs/<run_id>/ -> runs_dir/<run_id>/ (no deletions either way)."""
    rem = remote or _remote.load_remote()
    if "/" in run_id or run_id.startswith(".") or not run_id:
        raise ValueError(f"bad run id {run_id!r}")
    dst = Path(runs_dir) / run_id
    dst.mkdir(parents=True, exist_ok=True)
    proc = rsync_checksum(f"{rem.host}:{rem.runs}/{run_id}/", str(dst) + "/", remote=rem)
    if proc.returncode != 0:
        raise RuntimeError(f"rsync of {run_id} from {rem.host} failed (exit {proc.returncode}): {proc.stderr.strip()}")
    return dst
