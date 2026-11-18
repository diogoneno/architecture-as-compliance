"""Reader of the private adversarial store (pre-registered specification; .

The victim server (aac.serve.victim_server `_store_adversarial`) appends one JSON line per successful attack to
`<out>/data/derived/private/adversarial/<R3 run_id>.jsonl`: {claim_hash, item_id, attack, victim, subset, run_id,
text}, the file 0600 in a 0700 directory, on the host that ran the attack. The R4 replay (aac.regimes.r4) reads it
here, at run time, on that same host; the texts never leave it, are never logged and never appear in a message.

read_private_store refuses (PrivateStoreError):
 - a missing file (the replay runs only on the host that made the R3 run),
 - a file whose mode is not 0600 (or a symlink), or a directory open to group or others (not 0700),
 - a line that is not JSON, lacks a key, names another run, or whose claim_hash is not sha256 of its text,
 - a key (item_id, attack, victim, subset) stored twice with different texts. The store is append-only , so a resumed or forced session may append the same text again; identical lines collapse into one key.
Stdlib only, besides the output-namespace resolver."""
from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

from aac.run.paths import out_root

PRIVATE_ADVERSARIAL = Path("data") / "derived" / "private" / "adversarial"   # = aac.serve.victim_server's constant
KEYS = ("claim_hash", "item_id", "attack", "victim", "subset", "run_id", "text")


class PrivateStoreError(RuntimeError):
    """The private store of an R3 run is absent, exposed or inconsistent (no text is ever part of the message)."""


def store_path(root: Path | str, run_id: str) -> Path:
    """The store file of R3 run `run_id` in the output namespace of `root` (AAC_OUT applies, as for the server)."""
    return out_root(root) / PRIVATE_ADVERSARIAL / f"{run_id}.jsonl"


def _check_modes(path: Path) -> None:
    if path.is_symlink():
        raise PrivateStoreError(f"{path.name}: the private store is a symlink; refusing to read it")
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise PrivateStoreError(f"{path.name}: mode {mode:04o}, the private store must be 0600")
    dmode = stat.S_IMODE(path.parent.stat().st_mode)
    if dmode & 0o077:
        raise PrivateStoreError(f"{path.parent.name}/: mode {dmode:04o}, the private store directory must be 0700")


def read_private_store(root: Path | str, run_id: str) -> dict[tuple[str, str, str, str], dict]:
    """{(item_id, attack, victim, subset): {"claim_hash", "text"}} of R3 run `run_id`'s private store (module doc)."""
    path = store_path(root, run_id)
    if not path.is_file():
        raise PrivateStoreError(f"no private adversarial store for {run_id} in this output namespace; the replay runs "
                                "only on the host that made the R3 run (the store never leaves it)")
    _check_modes(path)
    out: dict[tuple[str, str, str, str], dict] = {}
    for lineno, line in enumerate(path.read_bytes().split(b"\n"), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise PrivateStoreError(f"{path.name} line {lineno}: not a JSON object") from None
        if not isinstance(row, dict) or any(k not in row for k in KEYS):
            raise PrivateStoreError(f"{path.name} line {lineno}: expected the keys {list(KEYS)}")
        if row["run_id"] != run_id:
            raise PrivateStoreError(f"{path.name} line {lineno}: run_id {row['run_id']} is not {run_id}")
        text = row["text"]
        if not isinstance(text, str) or hashlib.sha256(text.encode("utf-8")).hexdigest() != row["claim_hash"]:
            raise PrivateStoreError(f"{path.name} line {lineno}: claim_hash is not sha256 of the stored text")
        key = (str(row["item_id"]), str(row["attack"]), str(row["victim"]), str(row["subset"]))
        held = out.get(key)
        if held is not None and held["claim_hash"] != row["claim_hash"]:
            raise PrivateStoreError(f"{path.name} line {lineno}: {key} is stored twice with different texts "
                                    f"(claim_hash {held['claim_hash'][:12]} and {row['claim_hash'][:12]})")
        out[key] = {"claim_hash": row["claim_hash"], "text": text}
    return out
