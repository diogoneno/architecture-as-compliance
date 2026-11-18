"""Victim protocol v1, server side: limits, enums and request parsing .

Mirrors the client in attacks/runner/protocol.py and the test double attacks/runner/fake_victim_server.py.
`parse_request(path, body)` returns the checked fields or raises BadRequest. A BadRequest reason names
fields and limits only; it never echoes a text, because the server prints reasons and the runner prints
fault details."""
from __future__ import annotations

import json

from aac.run.paths import root as project_root

MAX_TEXTS = 64
MAX_BODY = 8 * 1024 * 1024
MAX_TEXT_CHARS = 65536
PHASES = ("clean", "attack", "posthoc")
OUTCOMES = ("success", "failed", "limit", "skipped")
ITEM_ID_MAX = 128  # attack-ledger.schema.json item_id maxLength

GET_PATHS = ("/v1/ping", "/v1/plan", "/v1/done")
POST_PATHS = ("/v1/items", "/v1/query", "/v1/end", "/v1/shutdown")


class BadRequest(ValueError):
    """A malformed request: answered with 400 {"error": "bad_request", "detail": <reason>} before inference."""


def _ledger_enums() -> dict[str, tuple[str, ...]]:
    schema = json.loads((project_root() / "contracts" / "attack-ledger.schema.json").read_text(encoding="utf-8"))
    props = schema["properties"]
    return {k: tuple(props[k]["enum"]) for k in ("attack", "victim", "subset")}


_ENUMS = _ledger_enums()
ATTACKS: tuple[str, ...] = _ENUMS["attack"]
VICTIMS: tuple[str, ...] = _ENUMS["victim"]
SUBSETS: tuple[str, ...] = _ENUMS["subset"]


def _str(body: dict, key: str, allowed: tuple[str, ...] | None = None, max_len: int = ITEM_ID_MAX) -> str:
    v = body.get(key)
    if not isinstance(v, str) or not v or len(v) > max_len:
        raise BadRequest(f"field {key!r} missing, empty, not a string or longer than {max_len}")
    if allowed is not None and v not in allowed:
        raise BadRequest(f"field {key!r} is not one of {list(allowed)}")
    return v


def _int(body: dict, key: str) -> int:
    v = body.get(key)
    if not isinstance(v, int) or isinstance(v, bool) or v < 0:
        raise BadRequest(f"field {key!r} missing or not a non-negative integer")
    return v


def _seconds(body: dict, key: str) -> float:
    v = body.get(key)
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0.0 <= float(v) < 1e9):
        raise BadRequest(f"field {key!r} missing or not a non-negative number of seconds")
    return float(v)


def _cell(body: dict) -> dict:
    return {"item_id": _str(body, "item_id"), "subset": _str(body, "subset", SUBSETS),
            "attack": _str(body, "attack", ATTACKS), "victim": _str(body, "victim", VICTIMS)}


def parse_request(path: str, body: dict) -> dict:
    """Checked fields of a POST body for `path`; raises BadRequest (reason names fields, never a text)."""
    if not isinstance(body, dict):
        raise BadRequest("body is not a JSON object")
    if path == "/v1/items":
        return {"subset": _str(body, "subset", SUBSETS), "victim": _str(body, "victim", VICTIMS)}
    if path == "/v1/query":
        out = _cell(body)
        out["phase"] = _str(body, "phase", PHASES)
        texts = body.get("texts")
        if not isinstance(texts, list) or not (1 <= len(texts) <= MAX_TEXTS):
            raise BadRequest(f"field 'texts' must be a list of 1 to {MAX_TEXTS} strings")
        if any(not isinstance(t, str) for t in texts):
            raise BadRequest("field 'texts' holds a non-string")
        if any(len(t) > MAX_TEXT_CHARS for t in texts):
            raise BadRequest(f"field 'texts' holds a text above {MAX_TEXT_CHARS} characters")
        if out["phase"] != "attack" and len(texts) != 1:
            raise BadRequest(f"phase {out['phase']!r} takes exactly one text")
        out["texts"] = list(texts)
        return out
    if path == "/v1/end":
        out = _cell(body)
        out["outcome"] = _str(body, "outcome", OUTCOMES)
        final = body.get("final_text")
        if out["outcome"] == "success":
            if not isinstance(final, str) or len(final) > MAX_TEXT_CHARS:
                raise BadRequest(f"outcome 'success' needs 'final_text' as a string of at most {MAX_TEXT_CHARS} characters")
        elif final is not None:
            raise BadRequest(f"outcome {out['outcome']!r} needs 'final_text' null")
        out["final_text"] = final
        out["runner_queries"] = _int(body, "runner_queries")
        out["openattack_invocations"] = _int(body, "openattack_invocations")
        out["runner_seconds"] = _seconds(body, "runner_seconds")
        out["rtt_seconds"] = _seconds(body, "rtt_seconds")
        return out
    if path == "/v1/shutdown":
        return {}
    raise BadRequest("unknown endpoint")
