"""Leak scan : does a record contain claim text or a user identifier?

Two claim-text checks run over every string of a record 
 - exact: the claim text as a substring of the decoded UTF-8 string (no normalization);
 - shingle: any SHINGLE_TOKENS-token window of a claim (lower-cased word tokens, so a
 truncated, re-cased, tokenised or whitespace-normalized copy is still caught); a claim
 shorter than SHINGLE_TOKENS but with at least MIN_SHINGLE_TOKENS tokens is one window."""
from __future__ import annotations

import re
from typing import Iterable, Iterator

USER_ID_RE = re.compile(r"@\w+|\b\d{15,20}\b")
TOKEN_RE = re.compile(r"\w+")
SHINGLE_TOKENS = 8
MIN_SHINGLE_TOKENS = 4

# Identity fields whose values the schema pins to a fixed hex/uuid/date-time pattern:
# no claim text or user identifier can hide there, and a hex digest that happens to be
# all digits (e.g. a model_version `m-<16 hex>`) would otherwise be a false positive.
PATTERN_BOUND_KEYS = frozenset({
    "record_id", "timestamp", "model_version", "data_version", "config_hash", "claim_hash",
    "analyst_action_timestamp",
})


def _tokens(s: str) -> list[str]:
    return TOKEN_RE.findall(s.lower())


def make_shingles(texts: Iterable[str]) -> frozenset[tuple[str, ...]]:
    """Token windows of every text (see module docstring); build once per run, pass to leakscan."""
    out: set[tuple[str, ...]] = set()
    for t in texts:
        toks = _tokens(t or "")
        if len(toks) >= SHINGLE_TOKENS:
            out.update(tuple(toks[i:i + SHINGLE_TOKENS]) for i in range(len(toks) - SHINGLE_TOKENS + 1))
        elif len(toks) >= MIN_SHINGLE_TOKENS:
            out.add(tuple(toks))
    return frozenset(out)


def _has_shingle(s: str, shingles: frozenset, lengths: frozenset[int]) -> bool:
    toks = _tokens(s)
    for k in lengths:
        for i in range(len(toks) - k + 1):
            if tuple(toks[i:i + k]) in shingles:
                return True
    return False


def _strings(obj: object, skip: frozenset = frozenset()) -> Iterator[str]:
    """Every string in obj; `skip` names keys excluded at the TOP LEVEL only (nested
 dicts are scanned in full, whatever their key names)."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in skip:
                continue
            yield from _strings(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _strings(v)


def leakscan(record: dict, fixture_texts: set[str], skip: frozenset = PATTERN_BOUND_KEYS,
             shingles: frozenset | None = None) -> tuple[bool, bool]:
    """(has_claim_text, has_user_id) over every string in the record except the
 pattern-bound identity keys in `skip` (top level only; nested keys are scanned).
 `shingles` defaults to make_shingles(fixture_texts); pass it precomputed for speed."""
    if shingles is None:
        shingles = make_shingles(fixture_texts or ())
    lengths = frozenset(len(s) for s in shingles)
    has_text = False
    has_uid = False
    for s in _strings(record, skip):
        if not has_text and fixture_texts and (any(t and t in s for t in fixture_texts)
                                               or (shingles and _has_shingle(s, shingles, lengths))):
            has_text = True
        if not has_uid and USER_ID_RE.search(s):
            has_uid = True
        if has_text and has_uid:
            break
    return has_text, has_uid
