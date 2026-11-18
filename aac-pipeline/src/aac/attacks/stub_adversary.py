"""StubAdversary: deterministic word/character perturbations, no library (stub tier).

propose(text, round) returns the `round`-th candidate for the clean text, or None when
the stub has no further distinct perturbation (3 x the word count). The candidate is a
pure function of (attack name, seed, clean text, round): the R4 replay regenerates
victim B's final adversarial example from the R3 ledger's (item, attack, round) and checks
its claim_hash, so adversarial text never has to be stored ."""
from __future__ import annotations

import numpy as np

from aac.run.hashing import claim_hash, derive_seed

ATTACKS = ("bert_attack", "textfooler", "deepwordbug", "pwws", "beamattack")

_SUBS = {
    "announced": "declared", "denied": "disputed", "claimed": "asserted", "confirmed": "verified",
    "reported": "noted", "rejected": "refused", "approved": "endorsed", "measured": "gauged",
    "rise": "increase", "closure": "shutdown", "new": "fresh", "record": "unprecedented",
    "ban": "prohibition", "merger": "union", "drop": "fall", "recall": "withdrawal", "tax": "levy",
    "reopening": "relaunch", "cut": "reduction", "audit": "review", "council": "board",
    "minister": "secretary", "agency": "bureau", "company": "firm", "report": "study",
    "mayor": "governor", "committee": "panel", "regulator": "watchdog", "senator": "lawmaker",
    "university": "college", "clinic": "hospital", "union": "guild", "three": "four", "two": "three",
    "twelve": "eleven", "last": "past", "a": "one", "the": "that",
}


def _char_edit(word: str, rng: np.random.Generator) -> str:
    if len(word) < 2:
        return word + "x"
    op = int(rng.integers(3))
    i = int(rng.integers(len(word) - 1))
    if op == 0:  # swap adjacent
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == 1:  # delete
        return word[:i] + word[i + 1:]
    return word[:i] + chr(ord("a") + int(rng.integers(26))) + word[i:]  # insert


class StubAdversary:
    def __init__(self, name: str, seed: int) -> None:
        if name not in ATTACKS:
            raise ValueError(f"unknown attack {name!r}; expected one of {ATTACKS}")
        self.name = name
        self.seed = int(seed)

    def max_rounds(self, text: str) -> int:
        return 3 * max(1, len(text.split(" ")))

    def propose(self, text: str, round: int) -> str | None:
        words = text.split(" ")
        if round < 0 or round >= self.max_rounds(text):
            return None
        rng = np.random.default_rng(derive_seed(f"stub-adversary|{self.name}|{self.seed}|{claim_hash(text)}|{round}"))
        per_edit = 2 if self.name == "beamattack" else 1
        n_edits = min((round // 3 + 1) * per_edit, len(words))
        positions = sorted(int(i) for i in rng.choice(len(words), size=n_edits, replace=False))
        out = list(words)
        for pos in positions:
            w = out[pos]
            if self.name == "deepwordbug":
                out[pos] = _char_edit(w, rng)
            else:
                lower = w.lower()
                if lower in _SUBS:
                    sub = _SUBS[lower]
                    out[pos] = sub.capitalize() if w[:1].isupper() else sub
                else:
                    out[pos] = _char_edit(w, rng)
        cand = " ".join(out)
        return cand if cand != text else None
