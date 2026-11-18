"""Identity definitions canonical JSON, config_hash, data_version,
model_version, record_id, derive_seed, run_id.

All of these are pure functions of bytes; nothing here reads a clock, a host
name or an absolute path.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from pathlib import Path
from typing import Any

# Fixed namespace for uuid5 record ids (derived once from the URL namespace; never changes).
AAC_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://aac-pipeline/decision-log")

# Keys excluded from config_hash at the top level of the resolved config.
EXCLUDED_TOP = frozenset({"seed", "run_id", "paths", "host", "time"})
# Path-valued keys under `data` excluded from config_hash (data_version covers their content).
EXCLUDED_DATA_PATH_KEYS = frozenset({"manifest", "split_manifest", "claims", "labels", "evidence", "fc_items", "rd_items", "attack_subset"})

# Decision-log records and ledger rows keep a stated 6-d.p. output rule (canonical_rounded);

FLOAT_DECIMALS = 6
RUN_ID_RE = re.compile(r"^[A-Z0-9]+-[0-9a-f]{10}-s\d+$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


def _normalise(obj: Any, decimals: int | None = None) -> Any:
    """Reject non-string mapping keys; round floats only when `decimals` is given.

 Without rounding a float is written by json's float repr, the shortest string that
 round-trips to the same IEEE double, so distinct values stay distinct and the
 bytes are still deterministic."""
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    if isinstance(obj, float):
        return obj if decimals is None else round(obj, decimals)
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                raise TypeError(f"canonical JSON requires string keys, got {type(k).__name__}: {k!r}")
            out[k] = _normalise(v, decimals)
        return out
    if isinstance(obj, (list, tuple)):
        return [_normalise(v, decimals) for v in obj]
    raise TypeError(f"canonical JSON cannot encode {type(obj).__name__}")


def canonical(obj: Any) -> bytes:
    """Canonical JSON bytes: sorted keys, no whitespace, ASCII only, no NaN/Inf,
 shortest round-trip floats (never rounded), string keys only. This is the
 config_hash / model_version input."""
    return json.dumps(
        _normalise(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def canonical_rounded(obj: Any) -> bytes:
    """canonical with floats rounded to FLOAT_DECIMALS: the stated output rule for
 decision-log records and attack-ledger rows only (scores at 6 d.p.)."""
    return json.dumps(
        _normalise(obj, FLOAT_DECIMALS), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def config_hash(resolved: dict, *, git_sha: str, lock_sha256: str, data_version: str) -> str:
    """SHA-256 over the canonical JSON of the resolved config minus the excluded keys,
 with the code SHA, the lockfile hash and data_version folded in."""
    body = {k: v for k, v in resolved.items() if k not in EXCLUDED_TOP}
    data = body.get("data")
    if isinstance(data, dict):
        body["data"] = {k: v for k, v in data.items() if k not in EXCLUDED_DATA_PATH_KEYS}
    env = dict(resolved.get("env") or {})
    env["lock_sha256"] = lock_sha256
    body["env"] = env
    body["code"] = {"git_sha": git_sha}
    body["data_version"] = data_version
    return sha256_bytes(canonical(body))


def data_version(manifest_path: Path) -> str:
    """sha256 over the LC_ALL=C-sorted lines of a sha256sum-format manifest ."""
    raw = Path(manifest_path).read_bytes()
    lines = sorted(line for line in raw.split(b"\n") if line.strip())
    return sha256_bytes(b"\n".join(lines) + b"\n")


def model_version(parts: dict[str, str]) -> str:
    return "m-" + sha256_bytes(canonical(parts))[:16]


def record_id(run_id: str, shard: int, seq: int) -> str:
    return str(uuid.uuid5(AAC_NAMESPACE, f"{run_id}|{shard}|{seq}"))


def derive_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:4], "big")


def run_id(regime: str, config_hash_: str, seed: int) -> str:
    rid = f"{regime}-{config_hash_[:10]}-s{seed}"
    if not RUN_ID_RE.match(rid):
        raise ValueError(f"run_id {rid!r} does not match {RUN_ID_RE.pattern}")
    return rid


def claim_hash(text: str) -> str:
    """sha256 of the exact UTF-8 bytes of the claim text; no normalization."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def dumps_sorted(obj: Any) -> str:
    """Deterministic, human-readable JSON (sorted keys, ASCII, 1-space indent, LF)."""
    return json.dumps(_normalise(obj), sort_keys=True, indent=1, ensure_ascii=True, allow_nan=False) + "\n"


def verify_hash(run_dir: Path, root: Path | None = None) -> str:
    """Recompute config_hash from runs/<id>/config.resolved.json + env.json (+ the data
 manifest named in the resolved config) and compare with every record's config_hash
 and run-record.json. Returns "match" or "mismatch" ( verifier)."""
    import json

    from aac.logging.validate import list_shards
    from aac.logging.writer import SHARD_DIR, read_shard
    from aac.run.paths import root as project_root

    root = Path(root) if root is not None else project_root()
    run_dir = Path(run_dir)
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    env = json.loads((run_dir / "env.json").read_text(encoding="utf-8"))
    dv = data_version(root / resolved["data"]["manifest"])
    recomputed = config_hash(resolved, git_sha=resolved["code"]["git_sha"], lock_sha256=env["lock_sha256"], data_version=dv)
    seen = {r["config_hash"] for p in list_shards(run_dir / SHARD_DIR) for r in read_shard(p)}
    rr = run_dir / "run-record.json"
    if rr.is_file():
        seen.add(json.loads(rr.read_text(encoding="utf-8"))["config_hash"])
    return "match" if seen == {recomputed} else "mismatch"
