"""ShardWriter : atomic, resumable gzip(mtime=0) shards with .sha256 sidecars
over the decompressed bytes; MemorySink for tests. Both validate every record."""
from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import os
import re
from dataclasses import dataclass
from pathlib import Path

from aac.logging.schema import check
from aac.run.hashing import canonical_rounded as canonical

SHARD_DIR = "decision-log"
LOCK_FILE = ".lock"  
SHARD_FILE_RE = re.compile(r"part-(\d{4})\.jsonl\.gz(\.sha256)?")


class RunLocked(Exception):
    pass


@dataclass(frozen=True)
class ShardInfo:
    file: str
    sha256: str
    n: int


def shard_name(i: int) -> str:
    return f"part-{i:04d}.jsonl.gz"


def read_shard(path: Path) -> list[dict]:
    import json

    with gzip.open(path, "rb") as g:
        raw = g.read()
    return [json.loads(line) for line in raw.split(b"\n") if line]


def shard_bytes(path: Path) -> bytes:
    with gzip.open(path, "rb") as g:
        return g.read()


RESUME_IGNORED = ("timestamp",)  # the only field a faithful regeneration may change (the resumed session's clock)


def content_digest(rec: dict) -> bytes:
    """ sha256 of the canonical, 6 d.p.-rounded record without RESUME_IGNORED. A record read back from
 a shard and the same record regenerated in memory give the same digest. Computed on the fly, never written."""
    return hashlib.sha256(canonical({k: v for k, v in rec.items() if k not in RESUME_IGNORED})).digest()


def sidecar_ok(path: Path) -> bool:
    side = path.with_name(path.name + ".sha256")
    if not path.is_file() or not side.is_file():
        return False
    return hashlib.sha256(shard_bytes(path)).hexdigest() == side.read_text(encoding="ascii").split()[0]


class MemorySink:
    """In-memory sink with the same validation contract as ShardWriter."""

    def __init__(self, validator, shard_size: int = 1000) -> None:
        self.validator = validator
        self.shard_size = int(shard_size)
        self.records: list[dict] = []

    def append(self, rec: dict) -> None:
        check(self.validator, rec)
        self.records.append(rec)

    def close(self) -> list[ShardInfo]:
        return []


class ShardWriter:
    """Writes runs/<run_id>/decision-log/part-NNNN.jsonl.gz per shard_size records.

 open: takes an exclusive flock on runs/<run_id>/.lock (raises RunLocked if held),
 removes stale *.tmp files and verifies existing shards; verified shards are skipped
 on append (their record_ids AND their content, timestamp excluded, must match what is
 regenerated: of the Phase 3 review), unverified shards are rewritten. close: flushes the partial shard, releases the lock, returns ShardInfo
 for every shard in order and asserts n_written == n_committed.

 item_shards=True (opt-in, the R3 victim server's per-item resume): records are
 buffered until seal_item, which writes them as the next verified shard whatever its length (one
 shard per finished attack item; no automatic flush at shard_size). A reopened writer keeps every
 verified shard in index order (short shards included), deletes every shard file after the first
 unverified one and sets resume_seq to the kept record count; keep_shards(m) truncates to the first m.
 close drops an unsealed item buffer and reports it in dropped_records. Record ids keep the logical
 shard seq // shard_size (RecordBuilder is untouched); the physical shard index is the item index."""

    def __init__(self, run_dir: Path, run_id: str, validator, shard_size: int = 1000, *,
                 item_shards: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.run_id = run_id
        self.validator = validator
        self.shard_size = int(shard_size)
        self.item_shards = bool(item_shards)
        self.dropped_records = 0
        if self.shard_size <= 0:
            raise ValueError("shard_size must be positive")
        self.shard_dir = self.run_dir / SHARD_DIR
        self._lock_fd: int | None = None
        self._buffer: list[dict] = []
        self._shard_index = 0
        self._next_seq = 0
        self._committed = 0
        self._written = 0
        self._verified_ids: list[str] = []
        self._verified_digests: list[bytes] = []  # content_digest per verified record (memory only; shard format unchanged)
        self.resume_seq = 0
        self._shards: dict[int, ShardInfo] = {}
        self._open()

    # -- lifecycle -----------------------------------------------------------------
    def _open(self) -> None:
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.run_dir / LOCK_FILE, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            os.close(fd)
            raise RunLocked(f"{self.run_dir} already has an open ShardWriter") from exc
        self._lock_fd = fd
        for stale in self.shard_dir.glob("*.tmp"):
            stale.unlink()
        self._resume()

    def _resume(self) -> None:
        i = 0
        while True:
            path = self.shard_dir / shard_name(i)
            if not path.is_file() or not sidecar_ok(path):
                break
            records = read_shard(path)
            if any(r.get("seq") != self.resume_seq + k for k, r in enumerate(records)):
                break
            self._verified_ids.extend(r["record_id"] for r in records)
            self._verified_digests.extend(content_digest(r) for r in records)
            self.resume_seq += len(records)
            self._shards[i] = ShardInfo(shard_name(i), hashlib.sha256(shard_bytes(path)).hexdigest(), len(records))
            i += 1
            if not self.item_shards and len(records) < self.shard_size:
                break
        for stray in self.shard_dir.glob("part-*.jsonl.gz*"):
            m = SHARD_FILE_RE.fullmatch(stray.name)
            if m is None:  
                continue
            if int(m.group(1)) >= i:
                stray.unlink()
        self._shard_index = i
        self._next_seq = self.resume_seq
        self._committed = self.resume_seq
        self._written = self.resume_seq
        if not self.item_shards and self.resume_seq and self.resume_seq % self.shard_size != 0:
            # a verified partial last shard is final only if it was the last; rewrite it
            last = i - 1
            self._shard_index = last
            self.resume_seq -= self._shards[last].n
            del self._shards[last]
            self._verified_ids = self._verified_ids[: self.resume_seq]
            self._verified_digests = self._verified_digests[: self.resume_seq]
            self._next_seq = self._committed = self._written = self.resume_seq

    def append(self, rec: dict) -> None:
        if self._lock_fd is None:
            raise RuntimeError("ShardWriter is closed")
        check(self.validator, rec)
        seq = rec["seq"]
        if seq < self.resume_seq:
            if rec["record_id"] != self._verified_ids[seq]:
                raise RuntimeError(f"resume mismatch at seq {seq}: regenerated record_id differs from the verified shard")
            if content_digest(rec) != self._verified_digests[seq]:
                raise RuntimeError(f"resume mismatch at seq {seq}: regenerated record content differs from the verified "
                                   "shard (timestamp excluded); the fits of this run would not match its sealed records "
                                   "(re-run with --force)")
            return  # already on disk and verified; counted in _committed at resume
        if seq != self._next_seq:
            raise RuntimeError(f"records must arrive in seq order: expected {self._next_seq}, got {seq}")
        self._next_seq += 1
        self._committed += 1
        self._buffer.append(rec)
        if not self.item_shards and len(self._buffer) == self.shard_size:
            self._flush()

    @property
    def verified_shards(self) -> int:
        """Shards on disk and verified (kept on resume plus those written since)."""
        return len(self._shards)

    def seal_item(self) -> ShardInfo:
        """item_shards mode: write the buffered records as the next verified shard, whatever its length."""
        if not self.item_shards:
            raise RuntimeError("seal_item() needs ShardWriter(item_shards=True)")
        if self._lock_fd is None:
            raise RuntimeError("ShardWriter is closed")
        index = self._shard_index
        self._flush(force=True)
        return self._shards[index]

    def keep_shards(self, m: int) -> None:
        """item_shards mode, right after open: keep the first m verified shards, delete the rest."""
        if not self.item_shards:
            raise RuntimeError("keep_shards() needs ShardWriter(item_shards=True)")
        if self._buffer or self._next_seq != self.resume_seq:
            raise RuntimeError("keep_shards() runs before any append")
        if not 0 <= int(m) <= len(self._shards):
            raise ValueError(f"cannot keep {m} shards: {len(self._shards)} are verified")
        m = int(m)
        for stray in self.shard_dir.glob("part-*.jsonl.gz*"):
            hit = SHARD_FILE_RE.fullmatch(stray.name)
            if hit is not None and int(hit.group(1)) >= m:
                stray.unlink()
        for i in [k for k in self._shards if k >= m]:
            del self._shards[i]
        self.resume_seq = sum(self._shards[i].n for i in range(m))
        self._verified_ids = self._verified_ids[: self.resume_seq]
        self._verified_digests = self._verified_digests[: self.resume_seq]
        self._shard_index = m
        self._next_seq = self._committed = self._written = self.resume_seq

    def _flush(self, force: bool = False) -> None:
        if not self._buffer and not force:
            return
        raw = b"".join(canonical(r) + b"\n" for r in self._buffer)
        buf = io.BytesIO()
        with gzip.GzipFile(fileobj=buf, mode="wb", mtime=0, filename="") as g:
            g.write(raw)
        path = self.shard_dir / shard_name(self._shard_index)
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "wb") as fh:
            fh.write(buf.getvalue())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        digest = hashlib.sha256(raw).hexdigest()
        with open(path.with_name(path.name + ".sha256"), "w", encoding="ascii", newline="\n") as fh:
            fh.write(f"{digest}  {path.name}\n")
        self._shards[self._shard_index] = ShardInfo(path.name, digest, len(self._buffer))
        self._written += len(self._buffer)
        self._shard_index += 1
        self._buffer = []

    def close(self) -> list[ShardInfo]:
        if self._lock_fd is None:
            return [self._shards[i] for i in sorted(self._shards)]
        if self.item_shards and self._buffer:  # an unfinished item: never sealed, counted
            self.dropped_records += len(self._buffer)
            self._committed -= len(self._buffer)
            self._next_seq -= len(self._buffer)
            self._buffer = []
        try:
            self._flush()
            if self._written != self._committed:
                raise RuntimeError(f"invariant L1 broken: written {self._written} != committed {self._committed}")
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            self._lock_fd = None
        return [self._shards[i] for i in sorted(self._shards)]

    def abort(self) -> None:
        """Release the lock WITHOUT flushing the buffered partial shard (crash / error
 path). Shards already replaced into place stay; a stray .tmp is removed by the
 next writer's resume. Safe to call more than once."""
        if self._lock_fd is None:
            return
        self._buffer = []
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
        os.close(self._lock_fd)
        self._lock_fd = None

    @property
    def record_count(self) -> int:
        return self._committed

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self.abort()
        else:
            self.close()
