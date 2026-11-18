"""SEALED: sha256 over the sorted `<relpath> <sha256>` lines of every file in a run
directory (except SEALED itself, the writer lock and the unhashed wall-clock sidecar).
After sealing a run is immutable."""
from __future__ import annotations

from pathlib import Path

from aac.run.hashing import sha256_bytes, sha256_file

UNSEALED_FILES = frozenset({"SEALED", ".lock", "wallclock.jsonl"})
STRAY_PATTERNS = ("*.tmp", "*.part", "*.tmp.npy")


class AlreadySealed(Exception):
    pass


class UnsealedRun(Exception):
    pass


def _listing(run_dir: Path) -> list[str]:
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(run_dir).as_posix()
        if rel in UNSEALED_FILES or rel.endswith(".tmp"):
            continue
        lines.append(f"{rel}  {sha256_file(p)}")
    return sorted(lines)


def compute_seal(run_dir: Path) -> str:
    body = "".join(line + "\n" for line in _listing(run_dir)).encode("ascii")
    return sha256_bytes(body)


def seal(run_dir: Path) -> str:
    run_dir = Path(run_dir)
    target = run_dir / "SEALED"
    if target.exists():
        raise AlreadySealed(f"{run_dir} is already sealed")
    
    # checkpoint) or <name>.tmp.npy (bench arrays); none of them may be sealed into a run.
    stray = sorted(p for pat in STRAY_PATTERNS for p in run_dir.rglob(pat))
    if stray:
        raise RuntimeError(f"refusing to seal {run_dir}: stray temp files {stray}")
    digest = compute_seal(run_dir)
    target.write_text(digest + "\n", encoding="ascii")
    return digest


def read_seal(run_dir: Path) -> str:
    target = Path(run_dir) / "SEALED"
    if not target.is_file():
        raise UnsealedRun(f"{run_dir} has no SEALED manifest")
    return target.read_text(encoding="ascii").strip()


def verify_sealed(run_dir: Path) -> str:
    """Recompute the seal and compare with SEALED; return the digest or raise."""
    recorded = read_seal(run_dir)
    actual = compute_seal(Path(run_dir))
    if recorded != actual:
        raise UnsealedRun(f"{run_dir}: SEALED {recorded[:12]} != recomputed {actual[:12]}")
    return recorded
