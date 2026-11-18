"""`aac prisma sample pilot|second-screen` (PRISMA-02, 06-CONTEXT D1).

Deterministic sampling without RNG state (2): rows are ordered by
sha256("<salt>|<key>") and the first n are taken. The order depends only on the salt string and the keys, never on
a Python or numpy RNG version or on the order the keys are read in.

pilot writes appendix-c/screening/batches/pilot-100.csv (title stage, every editable cell empty) and
 batches/pilot-100.sample.txt (salt, n, population, keys_sha256, source, method). It refuses to
 overwrite either file. `--check` re-derives the sample and prints the same line without writing;
 when the record or the batch exists it also compares them and exits 1 on any difference.
second-screen `--before YYYY-MM-DD` exports, as batches/second-screen-NN.csv (next free NN, stage second, blind:
 no first decision shown), the rows that are members of the 10 % second-screen sample, carry a title
 decision whose session_date is before the argument, have an empty second_screen_decision and are not
 listed in an earlier second-screen batch. Membership is a per-key threshold on the hash
 (in_second_screen), so a key's membership never changes while other rows are being decided.

Exit codes: 0 when the sample holds, 1 when a check fails or a write is refused, 2 for usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import re
import sys
from pathlib import Path
from typing import Iterable

from aac.prisma._io import OutsideScreening, default_appendix_dir, read_csv, screening_dir, workbook_lock
from aac.run.hashing import sha256_bytes

PILOT_SALT = "prisma-pilot-100-v1"
PILOT_N = 100
PILOT_NAME = "pilot-100"
SECOND_SALT = "prisma-second-screen-v1"
SECOND_SHARE = 0.10
TITLE_ORDER_SALT = "prisma-title-order-v1"
BATCHES = "batches"
WORKBOOK = "workbook.csv"
SECOND_RE = re.compile(r"^second-screen-(\d{2,})\.csv$")


class SampleMismatch(RuntimeError):
    """The re-derived sample differs from the recorded sample or from the batch keys."""


def _rank_key(salt: str, key: str) -> str:
    return sha256_bytes(f"{salt}|{key}".encode("utf-8"))


def hash_rank(keys: Iterable[str], salt: str) -> list[str]:
    """Keys ordered by sha256("<salt>|<key>") (ties, which need a sha256 collision, fall back to the key)."""
    return sorted(set(keys), key=lambda k: (_rank_key(salt, k), k))


def pilot_sample(keys: Iterable[str], salt: str = PILOT_SALT, n: int = PILOT_N) -> list[str]:
    """The first n keys by hash rank, returned sorted by key (a stable output order)."""
    ranked = hash_rank(keys, salt)
    if n < 1 or n > len(ranked):
        raise ValueError(f"cannot sample {n} of {len(ranked)} keys")
    return sorted(ranked[:n])


def in_second_screen(key: str, salt: str = SECOND_SALT, share: float = SECOND_SHARE) -> bool:
    """True when the first 8 hex digits of sha256("<salt>|<key>"), read as an integer, fall below share x 2^32.

 Each key's membership depends on that key alone, so the sample stays stable while rows are decided."""
    return int(_rank_key(salt, key)[:8], 16) < share * 2 ** 32


def keys_sha256(keys: Iterable[str]) -> str:
    """sha256 of the sorted keys joined by LF with a final LF."""
    return sha256_bytes("".join(k + "\n" for k in sorted(keys)).encode("utf-8"))


def sample_record(salt: str, n: int, population: int, digest: str) -> str:
    return (
        f"salt: {salt}\n"
        f"n: {n}\n"
        f"population: {population}\n"
        f"keys_sha256: {digest}\n"
        f"source: appendix-c/screening/{WORKBOOK}\n"
        f'method: rows ordered by sha256("<salt>|<key>"), first {n} taken (06-RESEARCH Pattern 2)\n'
    )


def _rel(path: Path, appendix: Path) -> str:
    from aac.prisma.workbook import _rel as rel
    return rel(path, appendix)


def _workbook_rows(appendix: Path) -> list[dict[str, str]]:
    with workbook_lock(appendix, shared=True):
        _, rows = read_csv(screening_dir(appendix) / WORKBOOK)
    return rows


def _recorded_digest(path: Path) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("keys_sha256: "):
            return line[len("keys_sha256: "):]
    return None


def pilot(appendix: Path, check: bool = False, salt: str = PILOT_SALT, n: int = PILOT_N,
          name: str = PILOT_NAME) -> tuple[str, int, list[str], Path]:
    """(keys_sha256, population, keys, batch path). check: compare with the record and the batch, write nothing."""
    from aac.prisma import batches

    appendix = Path(appendix)
    rows = _workbook_rows(appendix)
    keys = pilot_sample([r["key"] for r in rows], salt, n)
    digest = keys_sha256(keys)
    bdir = screening_dir(appendix) / BATCHES
    batch, record = bdir / f"{name}.csv", bdir / f"{name}.sample.txt"
    if check:
        problems = []
        if record.is_file():
            got = _recorded_digest(record)
            if got != digest:
                problems.append(f"{_rel(record, appendix)} records keys_sha256 {got}, re-derived {digest}")
        if batch.is_file():
            _, brows = read_csv(batch)
            if keys_sha256(r.get("key", "") for r in brows) != digest or len(brows) != len(keys):
                problems.append(f"{_rel(batch, appendix)} keys differ from the re-derived sample")
        if problems:
            raise SampleMismatch("; ".join(problems))
        return digest, len(rows), keys, batch
    for p in (batch, record):
        if p.exists():
            raise FileExistsError(f"{_rel(p, appendix)} exists; the pilot sample is written once and never overwritten")
    batches.export_batch(appendix, name, keys, "title")
    from aac.prisma._io import write_text_atomic
    write_text_atomic(record, sample_record(salt, n, len(rows), digest), appendix)
    return digest, len(rows), keys, batch


def _iso_date(s: str) -> str:
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", s or ""):
        raise ValueError(f"{s!r} is not a YYYY-MM-DD date")
    _dt.date.fromisoformat(s)
    return s


def second_screen(appendix: Path, before: str) -> tuple[int, Path]:
    """Export the eligible second-screen rows to the next free batches/second-screen-NN.csv."""
    from aac.prisma import batches

    appendix = Path(appendix)
    before = _iso_date(before)
    bdir = screening_dir(appendix) / BATCHES
    listed: set[str] = set()
    used = [0]
    if bdir.is_dir():
        for p in sorted(bdir.iterdir()):
            m = SECOND_RE.match(p.name)
            if m:
                used.append(int(m.group(1)))
                listed |= {r.get("key", "") for r in read_csv(p)[1]}
    rows = _workbook_rows(appendix)
    keys = [
        r["key"] for r in rows
        if in_second_screen(r["key"])
        and (r.get("screen_decision") or "").strip()
        and (r.get("session_date") or "") and r["session_date"] < before
        and not (r.get("second_screen_decision") or "").strip()
        and r["key"] not in listed
    ]
    if not keys:
        raise LookupError(f"no row is eligible for the second screen before {before}")
    name = f"second-screen-{max(used) + 1:02d}"
    path = batches.export_batch(appendix, name, hash_rank(keys, SECOND_SALT), "second")
    return len(keys), path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma sample")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("pilot", help="write (or --check) the 100-row pilot batch and its sample record")
    p.add_argument("--check", action="store_true", help="re-derive and compare; write nothing")
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("second-screen", help="export the 10 %% second-screen rows decided before a date")
    p.add_argument("--before", required=True, help="YYYY-MM-DD: title decisions with an earlier session_date")
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        if args.sub == "pilot":
            # constants read at call time (not bound as defaults), so they stay the single source
            digest, population, keys, batch = pilot(appendix, check=args.check, salt=PILOT_SALT, n=PILOT_N)
            print(f"sample: pilot n={len(keys)} population={population} salt={PILOT_SALT} keys_sha256={digest} "
                  f"-> {_rel(batch, appendix)}")
            return 0
        n, path = second_screen(appendix, args.before)
        print(f"sample: second-screen rows={n} before={args.before} -> {_rel(path, appendix)}")
        return 0
    except (SampleMismatch, FileExistsError, FileNotFoundError, LookupError, ValueError, OutsideScreening) as exc:
        print(f"aac prisma sample: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
