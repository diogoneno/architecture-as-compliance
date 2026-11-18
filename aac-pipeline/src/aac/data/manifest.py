"""The data identity: data/derived/MANIFEST.sha256 ( success criterion 2, ).

`data_version = aac.run.hashing.data_version(data/derived/MANIFEST.sha256)` is folded into every real run's
config_hash and every decision-log record, and `evidence-pack/data_version.txt` (committed before the first
C1 fine-tune) must equal it: the C1 trainer's datasheet gate refuses otherwise (Art. 10 before Art. 15).

The manifest is sha256sum format (`<64 hex> <rel>`, LF, LC_ALL=C sorted) and is written LAST, after every
file it lists. Each `<rel>` resolves by its prefix:

 raw/<path> data/raw/<path>: every raw file verified (or measured only) on the GPU host, with the
 digest of that host's row in data/derived/data-manifest.json (the knowledge-store zips
 live only there; the CPU-only baseline checkpoint is a model artifact, not data)
 configs/<path> configs/factchecker-domains.txt (specification item 18: "hashed into the data version") and
 configs/data/splits.yaml (the split rules)
 <anything else> data/derived/<rel>: every file under data/derived/ except MANIFEST.sha256 itself,
 data-manifest.json (host bookkeeping; its raw facts enter as the raw/ rows) and the
 private lane, of which exactly one file is listed by hash: private/rd-attack.pseudo.jsonl
 (gitignored, never copied anywhere; its hash pins lane B without publishing it)

Derived halves : the victim-B subset
halves data/derived/splits/averitec-attack-subset-v1-{first200,last200,first100}.json are never listed. They are a pure
function of two LISTED files (the parent subset and averitec/labels.csv) and the committed generator
aac.data.subset_halves, so data_version covers their content through the parent: `verify_entries` and `check` accept
such a file only when it sits at one of those exact paths, its kind is attack-subset-half, its parent and the labels
file are listed and match, its source_sha256 equals the parent's listed digest, and regenerating it from the parent
gives byte-identical content (verify_half). Any other unlisted file under data/derived is still a difference, and
`write` never lists a half, so data_version does not move.

`check(root)` re-hashes every listed file present on this host, compares every raw row whose file is absent
here (the store zips, C19 files) with data-manifest.json, and returns non-zero on any difference, naming the
file. stdlib + aac.run.hashing only (no torch)."""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from aac.run.hashing import data_version, sha256_file
from aac.run.paths import root as project_root

MANIFEST_REL = Path("data/derived/MANIFEST.sha256")
DATA_MANIFEST_REL = Path("data/derived/data-manifest.json")
DERIVED_REL = Path("data/derived")
RAW_REL = Path("data/raw")
RAW_HOST = "turing"  # the host whose verified rows define the raw identity (all four store zips live there)
CONFIG_FILES = ("configs/data/splits.yaml", "configs/factchecker-domains.txt")
PRIVATE_DIR = "private"
PRIVATE_FILES = ("private/rd-attack.pseudo.jsonl",)
EXCLUDED_DERIVED = frozenset({"MANIFEST.sha256", "data-manifest.json"})
RESERVED_TOP = frozenset({"raw", "configs"})
STRAY_SUFFIXES = (".tmp", ".part", ".tmp.npy")
HEX64 = re.compile(r"[0-9a-f]{64}")
HALF_KIND = "attack-subset-half"


def half_rels() -> tuple[str, ...]:
    """Manifest-relative paths of the derived subset halves (the exact names aac.data.subset_halves writes)."""
    from aac.data import subset_halves as H  # noqa: PLC0415 (stdlib + aac.run.hashing; no cycle at import time)

    parent = Path(H.PARENT)
    base = parent.parent.relative_to(DERIVED_REL).as_posix()
    return tuple(f"{base}/{parent.stem}-{suffix}.json" for suffix, _, _ in H.HALVES)


def verify_half(root: Path, listed: dict[str, str], rel: str) -> str | None:
    """None when the unlisted derived half `rel` is accepted (see the module docstring), else the reason."""
    from aac.data import subset_halves as H  # noqa: PLC0415
    from aac.run.hashing import dumps_sorted  # noqa: PLC0415

    root = Path(root)
    if rel not in half_rels():
        return f"data/derived/{rel}: not a derived subset half"
    p = resolve(root, rel)
    if not p.is_file():
        return f"data/derived/{rel}: missing on this host"
    inputs = {}
    for src in (H.PARENT, H.LABELS):
        srel = manifest_rel(src)
        if srel not in listed:
            return f"data/derived/{rel}: its input {src} is not listed"
        sp = resolve(root, srel)
        if not sp.is_file():
            return f"data/derived/{rel}: its input {src} is missing on this host"
        got = sha256_file(sp)
        if got != listed[srel]:
            return f"data/derived/{rel}: its input {srel} on disk {got[:12]} != listed {listed[srel][:12]}"
        inputs[src] = sp
    data = p.read_bytes()
    try:
        doc = json.loads(data)
    except ValueError:
        return f"data/derived/{rel}: not JSON"
    if not isinstance(doc, dict) or doc.get("kind") != HALF_KIND or doc.get("source") != H.PARENT:
        return f"data/derived/{rel}: kind/source is not {HALF_KIND} of {H.PARENT}"
    if doc.get("source_sha256") != listed[manifest_rel(H.PARENT)]:
        return (f"data/derived/{rel}: source_sha256 {str(doc.get('source_sha256'))[:12]} != listed parent "
                f"{listed[manifest_rel(H.PARENT)][:12]}")
    try:
        want = H.build(inputs[H.PARENT], inputs[H.LABELS])[Path(rel).name]
    except (ValueError, KeyError) as exc:
        return f"data/derived/{rel}: cannot regenerate from {H.PARENT}: {exc}"
    if dumps_sorted(want).encode("ascii") != data:
        return f"data/derived/{rel}: differs from its regeneration from {H.PARENT} (aac.data.subset_halves)"
    return None


class ManifestError(RuntimeError):
    """The manifest cannot be written (a listed input is missing or ambiguous)."""


def _raw_rows(root: Path) -> dict[str, str]:
    """{raw/<path>: sha256} of every RAW_HOST row of data-manifest.json (verified or measured-only)."""
    p = Path(root) / DATA_MANIFEST_REL
    if not p.is_file():
        raise ManifestError(f"missing {DATA_MANIFEST_REL}: run `make data-verify` first")
    rows = json.loads(p.read_text(encoding="utf-8"))["files"]
    out: dict[str, str] = {}
    for r in rows:
        if r.get("host") != RAW_HOST:
            continue
        if r.get("match") is False or r.get("lfs_pointer"):
            raise ManifestError(f"{DATA_MANIFEST_REL}: {RAW_HOST} row {r['path']} is not verified (match false or LFS pointer)")
        out[f"raw/{r['path']}"] = r["sha256"]
    if not out:
        raise ManifestError(f"{DATA_MANIFEST_REL} holds no {RAW_HOST} rows")
    return out


def _derived_files(root: Path) -> list[str]:
    base = Path(root) / DERIVED_REL
    halves = set(half_rels())
    out = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(base).as_posix()
        top = rel.split("/", 1)[0]
        if top == PRIVATE_DIR or rel in EXCLUDED_DERIVED or p.name.startswith("."):
            continue
        if rel in halves:  # derived halves are verified against their listed parent, never listed (verify_half)
            continue
        if p.name.endswith(STRAY_SUFFIXES):  
            raise ManifestError(f"data/derived/{rel}: stray temp file from an interrupted write; delete it and re-run "
                                "the step that writes it")
        if top in RESERVED_TOP:
            raise ManifestError(f"data/derived/{rel}: the top-level names {sorted(RESERVED_TOP)} are reserved for raw/ and configs/ rows")
        out.append(rel)
    return out


def resolve(root: Path, rel: str) -> Path:
    """The on-disk path of a manifest entry (see the module docstring for the prefix rule)."""
    root = Path(root)
    if rel.startswith("raw/"):
        return root / RAW_REL / rel[len("raw/"):]
    if rel.startswith("configs/"):
        return root / rel
    return root / DERIVED_REL / rel


def manifest_lines(root: Path) -> list[str]:
    root = Path(root)
    entries: dict[str, str] = dict(_raw_rows(root))
    for rel in _derived_files(root):
        entries[rel] = sha256_file(resolve(root, rel))
    for rel in (*CONFIG_FILES, *PRIVATE_FILES):
        p = resolve(root, rel)
        if not p.is_file():
            hint = " (the private lane lives on CPU host only: run `make incrediblae-ingest` there)" if rel.startswith(PRIVATE_DIR) else ""
            raise ManifestError(f"missing {rel}{hint}")
        entries[rel] = sha256_file(p)
    lines = [f"{digest}  {rel}" for rel, digest in entries.items()]
    return sorted(lines, key=lambda s: s.encode("utf-8"))  # LC_ALL=C byte order


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def write_manifest(root: Path) -> Path:
    """Write data/derived/MANIFEST.sha256 (tmp + os.replace; byte-identical on a re-run)."""
    root = Path(root)
    lines = manifest_lines(root)
    target = root / MANIFEST_REL
    _write_atomic(target, "".join(line + "\n" for line in lines).encode("ascii"))
    return target


def parse(path: Path) -> list[tuple[str, str]]:
    out = []
    for n, line in enumerate(Path(path).read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or not HEX64.fullmatch(digest):  
            raise ManifestError(f"{path}:{n}: not a sha256sum line")
        out.append((rel, digest))
    return out


class DataDrift(ManifestError):
    """A file a run reads is not listed in the data manifest, or its bytes differ from the listed sha256
 ( of the Phase 2 review: the run would otherwise carry a data_version that does not cover its inputs)."""


# The path-valued keys of a `data:` block that a run, the trainer or a baseline reads (aac.run.config.DataCfg).
DATA_PATH_KEYS = ("split_manifest", "labels", "claims", "evidence", "fc_items", "rd_items", "attack_subset")


def manifest_rel(path: str) -> str:
    """The manifest entry of a project-relative path: data/derived/<x> -> <x>, data/raw/<x> -> raw/<x>,
 configs/<x> -> configs/<x>. Anything else (and the private lane) cannot be covered: DataDrift."""
    p = Path(path).as_posix().rstrip("/")
    if p.startswith(f"{DERIVED_REL.as_posix()}/"):
        rel = p[len(DERIVED_REL.as_posix()) + 1:]
        if rel.split("/", 1)[0] == PRIVATE_DIR:
            raise DataDrift(f"{path}: the private lane is never read by a run")
        return rel
    if p.startswith(f"{RAW_REL.as_posix()}/"):
        return "raw/" + p[len(RAW_REL.as_posix()) + 1:]
    if p.startswith("configs/"):
        return p
    raise DataDrift(f"{path}: outside data/derived, data/raw and configs; the data manifest cannot cover it")


def data_paths(data: dict) -> list[str]:
    """The non-empty path-valued keys of a resolved `data:` block, deduplicated, in DATA_PATH_KEYS order."""
    out: list[str] = []
    for key in DATA_PATH_KEYS:
        v = data.get(key)
        if v and v not in out:
            out.append(v)
    return out


def verify_entries(root: Path, manifest: str, paths: list[str]) -> int:
    """Re-hash every file in `paths` against the manifest `manifest` (project-relative) before a run reads it.

 A file path must be listed and match. A directory path (e.g. data.claims = data/raw/averitec/data) must have
 at least one listed entry under it; every listed entry under it that is present on this host must match.
 Raises DataDrift naming the first difference; returns the number of files re-hashed."""
    root = Path(root)
    target = root / manifest
    if not target.is_file():
        raise DataDrift(f"data manifest {manifest} does not exist")
    listed = dict(parse(target))
    n = 0
    for path in paths:
        rel = manifest_rel(path)
        on_disk = root / path
        if on_disk.is_dir():
            under = sorted(r for r in listed if r.startswith(rel + "/"))
            if not under:
                raise DataDrift(f"{path}: no entry under {rel}/ in {manifest}; run `make data-manifest`")
            for r in under:
                p = resolve(root, r)
                if p.is_file():
                    _check_one(p, r, listed[r], manifest)
                    n += 1
            continue
        if rel not in listed:
            if rel in half_rels():   # a derived subset half, checked against its listed parent
                why = verify_half(root, listed, rel)
                if why:
                    raise DataDrift(f"{path}: not listed in {manifest} and not an accepted derived half: {why}")
                n += 1
                continue
            raise DataDrift(f"{path}: not listed in {manifest}; run `make data-manifest`")
        if not on_disk.is_file():
            raise DataDrift(f"{path}: listed in {manifest} but missing on this host")
        _check_one(on_disk, rel, listed[rel], manifest)
        n += 1
    return n


def _check_one(p: Path, rel: str, digest: str, manifest: str) -> None:
    got = sha256_file(p)
    if got != digest:
        raise DataDrift(f"{rel}: on disk {got[:12]} != listed {digest[:12]} in {manifest}; the run would carry a "
                        f"stale data_version (run `make data-manifest` and re-render the datasheets)")


# Work directories the GPU host writes under its persistent data/derived (mk/ks.mk KS_OUT): their files are inputs
# to `make ks-pull`, never read by a run, and never listed. `check(..., ignore_unlisted=GPU_WORK_UNLISTED)` does not
# report them as unlisted; a LISTED file is always re-hashed.
GPU_WORK_UNLISTED = ("averitec/ks-audit/*", "averitec/ks-audit-limit-*/*")


def check(root: Path, *, excuse_private: bool = False, rehash_raw: bool = True,
          ignore_unlisted: tuple[str, ...] = ()) -> int:
    """0 when every listed file matches; 1 (each difference printed to stderr, naming the file) otherwise.

 excuse_private: a listed private/ file absent here is not a difference (the GPU host never holds the
 private lane). rehash_raw=False: raw/ rows are compared with data-manifest.json only, never re-hashed
 (the turing preflight: the 75 GB store zips were verified by `make data-verify`; a run re-hashes the raw
 files it reads through verify_entries)."""
    root = Path(root)
    target = root / MANIFEST_REL
    if not target.is_file():
        print(f"data manifest: missing {MANIFEST_REL}", file=sys.stderr)
        return 1
    bad: list[str] = []
    try:
        raw = _raw_rows(root)
        entries = parse(target)
    except (ManifestError, KeyError, ValueError) as exc:
        print(f"data manifest: {exc}", file=sys.stderr)
        return 1
    listed = {rel for rel, _ in entries}
    n_hashed = n_recorded = n_excused = 0
    for rel, digest in entries:
        p = resolve(root, rel)
        if rel.startswith("raw/"):
            if rel not in raw:
                bad.append(f"{rel}: listed but no {RAW_HOST} row in {DATA_MANIFEST_REL}")
                continue
            if raw[rel] != digest:
                bad.append(f"{rel}: listed {digest[:12]} != {DATA_MANIFEST_REL} {raw[rel][:12]}")
                continue
            if p.is_file() and rehash_raw:
                got = sha256_file(p)
                n_hashed += 1
                if got != digest:
                    bad.append(f"{rel}: on disk {got[:12]} != listed {digest[:12]}")
            else:
                n_recorded += 1  # held on the GPU host only (store zips, C19 files): checked against data-manifest.json
            continue
        if not p.is_file():
            if excuse_private and rel.split("/", 1)[0] == PRIVATE_DIR:
                n_excused += 1
                continue
            bad.append(f"{rel}: listed but missing on this host")
            continue
        got = sha256_file(p)
        n_hashed += 1
        if got != digest:
            bad.append(f"{rel}: on disk {got[:12]} != listed {digest[:12]}")
    for rel in sorted(set(raw) - listed):
        bad.append(f"{rel}: {RAW_HOST} row in {DATA_MANIFEST_REL} is not listed")
    try:
        unlisted = sorted(set(_derived_files(root)) - listed)
    except ManifestError as exc:
        unlisted, bad = [], bad + [str(exc)]
    from fnmatch import fnmatchcase

    for rel in unlisted:
        if any(fnmatchcase(rel, pat) for pat in ignore_unlisted):
            continue
        bad.append(f"data/derived/{rel}: present but not listed (rewrite the manifest: `make data-manifest`)")
    for rel in (*CONFIG_FILES, *PRIVATE_FILES):
        if rel not in listed:
            bad.append(f"{rel}: required entry not listed")
    n_halves = 0
    listed_digests = dict(entries)
    for rel in half_rels():   # present derived halves must regenerate from their listed parent
        if rel in listed or not resolve(root, rel).is_file():
            continue
        why = verify_half(root, listed_digests, rel)
        if why:
            bad.append(why)
        else:
            n_halves += 1
    for line in bad:
        print(f"data manifest: {line}", file=sys.stderr)
    if bad:
        print(f"data manifest: FAIL ({len(bad)} difference(s))", file=sys.stderr)
        return 1
    excused = f", {n_excused} private excused" if n_excused else ""
    excused += f", {n_halves} derived halves regenerated" if n_halves else ""
    print(f"data manifest: ok {len(entries)} entries ({n_hashed} re-hashed here, {n_recorded} checked against "
          f"{DATA_MANIFEST_REL}{excused}) data_version={data_version(target)}")
    return 0


def gate(root: Path, manifest: str = MANIFEST_REL.as_posix()) -> int:
    """`make datasheet-gate`: the C1 trainer's own gate function  over the real manifest."""
    from aac.components.c1_train import DatasheetGateClosed, datasheet_gate

    try:
        dv = datasheet_gate({"data": {"dataset": "averitec", "manifest": manifest}}, Path(root))
    except DatasheetGateClosed as exc:
        print(f"datasheet gate: closed: {exc}", file=sys.stderr)
        return 1
    print(f"datasheet gate: open data_version={dv}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.data.manifest", description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", choices=("write", "check", "gate"))
    ap.add_argument("--root", default=None, help="project root (default: the aac-pipeline checkout)")
    ap.add_argument("--excuse-private", action="store_true",
                    help="check: a listed private/ file absent on this host is not a difference (the GPU host)")
    ap.add_argument("--no-raw-rehash", action="store_true",
                    help="check: compare raw/ rows with data-manifest.json only, without re-hashing the raw files")
    ap.add_argument("--gpu-host", action="store_true",
                    help="check on the GPU host (remote GPU preflight): --excuse-private --no-raw-rehash, and the "
                         f"ks-audit work directories {list(GPU_WORK_UNLISTED)} are not reported as unlisted")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else project_root()
    if args.command == "write":
        try:
            target = write_manifest(root)
        except ManifestError as exc:
            print(f"data manifest: {exc}", file=sys.stderr)
            return 1
        print(f"data manifest: wrote {target.relative_to(root)} data_version={data_version(target)}")
        return 0
    if args.command == "check":
        if args.gpu_host:
            return check(root, excuse_private=True, rehash_raw=False, ignore_unlisted=GPU_WORK_UNLISTED)
        return check(root, excuse_private=args.excuse_private, rehash_raw=not args.no_raw_rehash)
    return gate(root)


if __name__ == "__main__":
    sys.exit(main())
