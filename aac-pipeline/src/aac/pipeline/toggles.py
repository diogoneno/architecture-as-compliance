"""Ablation toggles: the same code path runs with explicit neutral values ."""
from __future__ import annotations

from dataclasses import dataclass

SYSTEMS = {
    "full": (True, True, True),
    "content_only": (False, False, False),
    "ablation-no-retrieval": (False, True, True),
    "ablation-no-calibration": (True, False, True),
    "ablation-no-abstention": (True, True, False),
    "ablation-all-removed": (False, False, False),
}


@dataclass(frozen=True)
class Toggles:
    retrieval: bool
    calibration: bool
    abstention: bool

    @classmethod
    def from_system(cls, system: str) -> "Toggles":
        return cls(*SYSTEMS[system])

    @classmethod
    def all_on(cls) -> "Toggles":
        return cls(True, True, True)

    @classmethod
    def all_off(cls) -> "Toggles":
        return cls(False, False, False)
