"""Raw-file verification and the per-host data manifest (, ).

Every raw file is hashed on the host that holds it and compared with the recorded value in
``data/raw/MANIFEST.expected.sha256`` (values copied from data manifest) or
``configs/data/expected-extra.sha256`` (values re-read from a primary API, each with its URL and date).
A file with no recorded value is reported as ``measured-only`` and never asserted. Nothing here
writes a hash from memory; the manifest rows are measured.

``data/derived/data-manifest.json`` holds ``{"files": [rows]}`` sorted by (host, path), no
timestamps, written through a temp file and ``os.replace`` so an interrupted verify never leaves a
partial manifest. Re-running for one host replaces only that host's rows.

stdlib + aac.run.hashing only: this module never imports torch.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from aac.run.hashing import dumps_sorted, sha256_file

EXPECTED_RAW = Path("data/raw/MANIFEST.expected.sha256")
EXPECTED_EXTRA = Path("configs/data/expected-extra.sha256")
DATA_MANIFEST = Path("data/derived/data-manifest.json")
RAW_DIR = Path("data/raw")
LFS_SPEC_LINE = b"version https://git-lfs.github.com/spec/v1"
LFS_MAX_BYTES = 1024
MEASURED_ONLY = "measured-only"
SOURCES = {
    "averitec/": "hf:chenxwh/AVeriTeC@2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031",
    "incrediblae/": "gitlab:checkthat_lab/clef2024-checkthat-lab@6b9cf603de6eeb9c5faf6707455ecf2e1611a082",
}
SCAN_TOPS = ("averitec", "incrediblae")
SKIP_NAMES = frozenset({"download.log", "download.out"})
SKIP_SUFFIXES = (".part", ".sha256")
HEX64 = re.compile(r"[0-9a-f]{64}")  


class VerifyError(RuntimeError):
    """A recorded-hash file is malformed or a required input is missing."""


def parse_expected(path: str | Path) -> dict[str, str]:
    """`sha256sum -c` format (``<64 hex> <relpath>``); lines starting with # and blank lines skipped."""
    out: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest) or not rel:
            raise VerifyError(f"{path}:{n}: not a '<sha256>  <path>' line")
        if rel in out and out[rel] != digest:
            raise VerifyError(f"{path}:{n}: {rel} listed twice with different hashes")
        out[rel] = digest
    return out


def load_expected(root: str | Path) -> dict[str, str]:
    """Union of the raw manifest and the extra manifest (when present); a path in both must agree."""
    root = Path(root)
    exp = parse_expected(root / EXPECTED_RAW)
    extra_path = root / EXPECTED_EXTRA
    if extra_path.is_file():
        for rel, digest in parse_expected(extra_path).items():
            if rel in exp and exp[rel] != digest:
                raise VerifyError(f"{rel}: {EXPECTED_RAW} and {EXPECTED_EXTRA} disagree")
            exp[rel] = digest
    return exp


def is_lfs_pointer(path: str | Path) -> bool:
    """True for a Git LFS pointer file: under 1024 bytes and the first line is the LFS spec line."""
    p = Path(path)
    if p.stat().st_size >= LFS_MAX_BYTES:
        return False
    with open(p, "rb") as fh:
        first = fh.readline().rstrip(b"\r\n")
    return first == LFS_SPEC_LINE


def source_for(rel: str) -> str:
    for prefix, src in SOURCES.items():
        if rel.startswith(prefix):
            return src
    raise VerifyError(f"{rel}: no recorded source for this path")


def licence_key(rel: str) -> str:
    """Key into configs/licences.yaml `data:` (averitec, incrediblae.FC, incrediblae.RD, incrediblae.C19)."""
    parts = rel.split("/")
    if parts[0] == "averitec":
        return "averitec"
    if parts[0] == "incrediblae" and len(parts) > 2:
        return f"incrediblae.{parts[1]}"
    raise VerifyError(f"{rel}: no licence key for this path")


def make_row(host: str, rel: str, nbytes: int, sha256: str, lfs_pointer: bool, expected: dict[str, str]) -> dict:
    exp = expected.get(rel)
    return {
        "host": host,
        "path": rel,
        "bytes": int(nbytes),
        "sha256": sha256,
        "expected_sha256": exp if exp is not None else MEASURED_ONLY,
        "match": (sha256 == exp) if exp is not None else None,
        "lfs_pointer": bool(lfs_pointer),
        "source": source_for(rel),
        "licence_key": licence_key(rel),
    }


def _skipped(name: str) -> bool:
    return name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES)


def local_files(raw_dir: str | Path) -> list[str]:
    raw_dir = Path(raw_dir)
    rels = []
    for top in SCAN_TOPS:
        base = raw_dir / top
        if not base.is_dir():
            continue
        for dirpath, _dirs, files in os.walk(base):
            for name in files:
                p = Path(dirpath) / name
                if _skipped(name) or not p.is_file():
                    continue
                rels.append(p.relative_to(raw_dir).as_posix())
    return sorted(rels)


def local_rows(raw_dir: str | Path, host: str, expected: dict[str, str] | None = None) -> list[dict]:
    """Hash every regular file under raw_dir/{averitec,incrediblae} (same skip list as the remote script)."""
    raw_dir = Path(raw_dir)
    if expected is None:
        expected = load_expected(raw_dir.parent.parent)
    rows = []
    for rel in local_files(raw_dir):
        p = raw_dir / rel
        rows.append(make_row(host, rel, p.stat().st_size, sha256_file(p), is_lfs_pointer(p), expected))
    return rows


def parse_listing(text: str, host: str, expected: dict[str, str]) -> list[dict]:
    """Rows from scripts/raw_hashes_remote.sh output: ``sha256<TAB>bytes<TAB>pointer(0|1)<TAB>relpath``."""
    rows = []
    seen = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        f = line.split("\t")
        if len(f) != 4 or not HEX64.fullmatch(f[0]) or not f[1].isdigit() or f[2] not in ("0", "1") or not f[3]:
            raise VerifyError(f"listing line {n}: malformed")
        rel = f[3][2:] if f[3].startswith("./") else f[3]
        if rel in seen:
            raise VerifyError(f"listing line {n}: {rel} listed twice")
        seen.add(rel)
        rows.append(make_row(host, rel, int(f[1]), f[0], f[2] == "1", expected))
    return sorted(rows, key=lambda r: r["path"])


def _write_atomic(path: Path, text: str) -> None:
    """tmp + os.replace; a failed write (non-ASCII value, disk full, Ctrl-C) leaves no <name>.tmp behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def merge_manifest(out_path: str | Path, host: str, rows: list[dict]) -> dict:
    """Replace `host`'s rows in the manifest, keep every other host's rows, sort by (host, path), write atomically."""
    out_path = Path(out_path)
    if any(r["host"] != host for r in rows):
        raise VerifyError(f"merge_manifest: every row must belong to host {host}")
    kept = []
    if out_path.is_file():
        kept = [r for r in json.loads(out_path.read_text(encoding="utf-8"))["files"] if r["host"] != host]
    files = sorted(kept + list(rows), key=lambda r: (r["host"], r["path"]))
    doc = {"files": files}
    _write_atomic(out_path, dumps_sorted(doc))
    return doc


def promote(part: str | Path, expected_sha256: str) -> Path:
    """Rename ``<file>.part`` to ``<file>`` only when its sha256 equals the expected value .

 A matching final file already in place is kept (the part is discarded); a final file with a
 different hash is never overwritten. Raises VerifyError (both hashes named) on any mismatch.
 """
    part = Path(part)
    if part.suffix != ".part":
        raise VerifyError(f"{part}: promote takes a '.part' path")
    final = part.with_suffix("")
    actual = sha256_file(part)
    if actual != expected_sha256:
        raise VerifyError(f"{part}: sha256 {actual} != expected {expected_sha256}; not promoted")
    if final.exists():
        have = sha256_file(final)
        if have != expected_sha256:
            raise VerifyError(f"{final}: existing file sha256 {have} != expected {expected_sha256}; refusing to overwrite")
        part.unlink()
        return final
    os.replace(part, final)
    return final


def _print_rows(rows: list[dict]) -> None:
    for r in rows:
        if r["lfs_pointer"]:
            verdict = "LFS-POINTER"
        elif r["match"] is None:
            verdict = "measured-only"
        else:
            verdict = "OK" if r["match"] else "MISMATCH"
        print(f"{r['host']}\t{verdict}\t{r['bytes']}\t{r['sha256']}\t{r['path']}")


def verify(rows: list[dict], expected: dict[str, str], *, require: list[str], require_all_expected: bool) -> list[str]:
    """Return the list of failures (empty = pass)."""
    fails = []
    have = {r["path"] for r in rows}
    for r in rows:
        if r["lfs_pointer"]:
            fails.append(f"{r['path']}: Git LFS pointer, not the object")
        if r["match"] is False:
            fails.append(f"{r['path']}: sha256 {r['sha256']} != expected {r['expected_sha256']}")
    needed = set(require)
    if require_all_expected:
        needed |= set(expected)
    for rel in sorted(needed - have):
        fails.append(f"{rel}: required file missing")
    return fails


def promote_cli(root: Path, parts: list[str]) -> int:
    """Promote each ``data/raw/<rel>.part`` against its expected line (raw manifest + extra manifest)."""
    expected = load_expected(root)
    raw = (root / RAW_DIR).resolve()
    status = 0
    for part in parts:
        pp = Path(part)
        pp = (pp if pp.is_absolute() else root / pp).resolve()
        try:
            rel = pp.relative_to(raw).as_posix()
        except ValueError:
            print(f"promote: FAIL {part}: not under {RAW_DIR}", file=sys.stderr)
            status = 1
            continue
        rel = rel[: -len(".part")] if rel.endswith(".part") else rel
        if rel not in expected:
            print(f"promote: FAIL {rel}: no expected sha256 in {EXPECTED_RAW} or {EXPECTED_EXTRA}", file=sys.stderr)
            status = 1
            continue
        try:
            final = promote(pp, expected[rel])
        except (VerifyError, OSError) as e:
            print(f"promote: FAIL {e}", file=sys.stderr)
            status = 1
            continue
        print(f"promote: OK {final.relative_to(raw).as_posix()} sha256={expected[rel]}")
    return status


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import root as project_root

    ap = argparse.ArgumentParser(prog="python -m aac.data.verify", description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=None, help="project root (default: the aac-pipeline checkout)")
    ap.add_argument("--host", help="host whose rows are measured (remote host via --listing, or this host)")
    ap.add_argument("--listing", help="remote listing file (scripts/raw_hashes_remote.sh); omit to hash data/raw locally")
    ap.add_argument("--require-all-expected", action="store_true",
                    help="every line of data/raw/MANIFEST.expected.sha256 must be present")
    ap.add_argument("--require", nargs="*", default=[], help="paths (relative to data/raw) that must be present")
    ap.add_argument("--extra", action="store_true", help="also read configs/data/expected-extra.sha256")
    ap.add_argument("--promote", nargs="+", metavar="PART", help="rename <file>.part -> <file> only on a hash match")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else project_root()
    if a.promote:
        return promote_cli(root, a.promote)
    if not a.host:
        ap.error("--host is required unless --promote is given")
    raw_expected = parse_expected(root / EXPECTED_RAW)
    expected = load_expected(root) if a.extra else dict(raw_expected)
    if a.listing:
        rows = parse_listing(Path(a.listing).read_text(encoding="utf-8"), a.host, expected)
    else:
        rows = local_rows(root / RAW_DIR, a.host, expected)
    _print_rows(rows)
    fails = verify(rows, raw_expected if a.require_all_expected else expected,
                   require=a.require, require_all_expected=a.require_all_expected)
    if fails:
        for f in fails:
            print(f"verify: FAIL {a.host}: {f}", file=sys.stderr)
        print(f"verify: {a.host}: {len(fails)} failure(s); data-manifest.json NOT updated", file=sys.stderr)
        return 1
    out = Path(a.out) if a.out else root / DATA_MANIFEST
    merge_manifest(out, a.host, rows)
    n_ok = sum(1 for r in rows if r["match"] is True)
    n_mo = sum(1 for r in rows if r["match"] is None)
    print(f"verify: {a.host}: {len(rows)} files, {n_ok} match, {n_mo} measured-only, 0 failures -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
