"""Hash-gated loading of third-party checkpoints (.

A state dict is handed to torch only after the file's sha256 equals the recorded value (data/raw/MANIFEST.expected.sha256), and only through ``torch.load(weights_only=True,
map_location=...)``. BODEGA's ``*-cache.pickle`` victim caches are refused outright - they are never loaded.

torch is imported inside the function body on purpose: aac.run carries no module-level torch import
('s AST test asserts it), so report and pack code stay CPU-only and fast to import.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

_HEX64 = re.compile(r"[0-9a-f]{64}")


class ChecksumMismatch(ValueError):
    """The file's sha256 differs from the recorded value; nothing was loaded."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk_size), b""):
            h.update(block)
    return h.hexdigest()


def load_state_dict(path: str | Path, expected_sha256: str, *, map_location: str = "cpu") -> dict[str, Any]:
    """Return ``torch.load(path, weights_only=True, map_location=map_location)`` after the hash gate.

 Raises ValueError for a malformed expected hash or a ``.pickle`` path (BODEGA cache files are never
 loaded) and ChecksumMismatch when the file's sha256 differs; in both cases torch.load is never called.
 """
    p = Path(path)
    if p.suffix.lower() in {".pickle", ".pkl"}:
        raise ValueError(f"refusing to load {p.name}: pickle caches (e.g. BODEGA VictimCache) are never loaded")
    if not _HEX64.fullmatch(expected_sha256):
        raise ValueError("expected_sha256 must be a 64-hex sha256 copied from the recorded manifest")
    actual = sha256_file(p)
    if actual != expected_sha256:
        raise ChecksumMismatch(f"{p}: sha256 {actual} != recorded {expected_sha256}; not loaded")
    import torch  # noqa: PLC0415 (deliberately local; see module docstring)

    return torch.load(p, weights_only=True, map_location=map_location)
