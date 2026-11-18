"""Victim protocol v1 client: HTTP/1.1 over an AF_UNIX stream socket (server side ).

Runs in env-attack (CPython 3.10). Standard library only; it never imports `aac`.

Endpoints (JSON bodies, UTF-8):
 GET /v1/ping -> {"ok", "run_id", "budget"}
 GET /v1/plan -> {"cells": [[attack, victim, subset], ...], "budget"}
 POST /v1/items {"subset", "victim"} -> {"items": [{"item_id", "text", "label"}]}
 GET /v1/done -> {"done": [[item_id, attack, victim, subset], ...]}
 POST /v1/query {"item_id", "subset", "attack", "victim", "phase", "texts"}
 -> {"probs", "record_ids", "queries_used", "deferred"}; 409 budget_exhausted
 POST /v1/end {"item_id", "subset", "attack", "victim", "outcome", "final_text", "runner_queries",
 "openattack_invocations", "runner_seconds", "rtt_seconds"} -> {"end_state", "posthoc_gate", "ledger_rows"}
 POST /v1/shutdown -> {"rows", "records"}

Every failure raises the fault class the caller injects (the runner passes its PipelineFault, a BaseException).
A 409 is a fault too: the runner sends a batch only after OpenAttack's own limit check passed, so a server refusal
means the two counters disagree . Fault messages carry the status, the error key and the endpoint only;
they never carry a text, because the runner prints the message.
"""
import http.client
import json
import socket

PHASES = ("clean", "attack", "posthoc")
OUTCOMES = ("success", "failed", "limit", "skipped")
MAX_TEXTS_PER_QUERY = 64
MAX_TEXT_CHARS = 65536
MAX_BODY_BYTES = 8 * 1024 * 1024


class _UnixHTTPConnection(http.client.HTTPConnection):
    """http.client connection whose transport is an AF_UNIX stream socket."""

    def __init__(self, socket_path, timeout):
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


class VictimClient:
    """One request per connection; `fault` is the exception class raised on every protocol failure."""

    def __init__(self, socket_path, timeout=600.0, *, fault):
        if not (isinstance(fault, type) and issubclass(fault, BaseException)):
            raise TypeError("fault must be an exception class")
        self.socket_path = str(socket_path)
        self.timeout = float(timeout)
        self._fault = fault

    # ------------------------------------------------------------------ transport
    def _request(self, method, path, body=None):
        payload = None
        headers = {"Connection": "close"}
        if body is not None:
            payload = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            if len(payload) > MAX_BODY_BYTES:
                raise self._fault(f"{path}: request body above {MAX_BODY_BYTES} bytes")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(payload))
        conn = _UnixHTTPConnection(self.socket_path, self.timeout)
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            status = resp.status
            raw = resp.read()
        except (OSError, http.client.HTTPException) as exc:  # socket.timeout is an OSError
            raise self._fault(f"{path}: transport error {type(exc).__name__}") from None
        finally:
            conn.close()
        try:
            reply = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise self._fault(f"{path}: HTTP {status} with a malformed JSON body") from None
        if not isinstance(reply, dict):
            raise self._fault(f"{path}: HTTP {status} reply is not a JSON object")
        if status == 409:
            raise self._fault(f"{path}: HTTP 409 {reply.get('error')!s:.40} queries_used="
                              f"{reply.get('queries_used')!s:.20} (runner and server counters disagree)")
        if status != 200:
            raise self._fault(f"{path}: HTTP {status} {reply.get('error')!s:.40} "
                              f"detail={str(reply.get('detail'))[:80]}")
        return reply

    def _need(self, reply, path, key, kind):
        value = reply.get(key)
        if not isinstance(value, kind) or (isinstance(value, bool) and kind is not bool):
            raise self._fault(f"{path}: reply field {key!r} missing or not {kind.__name__}")
        return value

    # ------------------------------------------------------------------ endpoints
    def ping(self):
        reply = self._request("GET", "/v1/ping")
        self._need(reply, "/v1/ping", "budget", int)
        return reply

    def plan(self):
        reply = self._request("GET", "/v1/plan")
        cells = self._need(reply, "/v1/plan", "cells", list)
        self._need(reply, "/v1/plan", "budget", int)
        for cell in cells:
            if not (isinstance(cell, list) and len(cell) == 3 and all(isinstance(c, str) for c in cell)):
                raise self._fault("/v1/plan: a cell is not [attack, victim, subset]")
        return reply

    def items(self, subset, victim):
        reply = self._request("POST", "/v1/items", {"subset": subset, "victim": victim})
        items = self._need(reply, "/v1/items", "items", list)
        for it in items:
            if not (isinstance(it, dict) and isinstance(it.get("item_id"), str) and isinstance(it.get("text"), str)
                    and isinstance(it.get("label"), int)):
                raise self._fault("/v1/items: an item lacks item_id, text or label")
        return reply

    def done(self):
        reply = self._request("GET", "/v1/done")
        rows = self._need(reply, "/v1/done", "done", list)
        for row in rows:
            if not (isinstance(row, list) and len(row) == 4 and all(isinstance(c, str) for c in row)):
                raise self._fault("/v1/done: a row is not [item_id, attack, victim, subset]")
        return reply

    def query(self, item_id, attack, victim, subset, phase, texts):
        if phase not in PHASES:
            raise self._fault(f"/v1/query: unknown phase {phase!r}")
        if not (1 <= len(texts) <= MAX_TEXTS_PER_QUERY):
            raise self._fault(f"/v1/query: {len(texts)} texts (allowed 1 to {MAX_TEXTS_PER_QUERY})")
        if any(not isinstance(t, str) or len(t) > MAX_TEXT_CHARS for t in texts):
            raise self._fault(f"/v1/query: a text is not a string or exceeds {MAX_TEXT_CHARS} characters")
        reply = self._request("POST", "/v1/query", {"item_id": item_id, "subset": subset, "attack": attack,
                                                    "victim": victim, "phase": phase, "texts": list(texts)})
        probs = self._need(reply, "/v1/query", "probs", list)
        self._need(reply, "/v1/query", "queries_used", int)
        self._need(reply, "/v1/query", "record_ids", list)
        self._need(reply, "/v1/query", "deferred", list)
        if len(probs) != len(texts) or len(reply["record_ids"]) != len(texts) or len(reply["deferred"]) != len(texts):
            raise self._fault("/v1/query: reply length differs from the number of texts")
        for row in probs:
            if not (isinstance(row, list) and len(row) == 2 and all(isinstance(p, (int, float)) and not isinstance(p, bool)
                                                                    and 0.0 <= p <= 1.0 for p in row)):
                raise self._fault("/v1/query: a probs row is not two numbers in [0, 1]")
        return reply

    def end(self, *, item_id, subset, attack, victim, outcome, final_text, runner_queries, openattack_invocations,
            runner_seconds, rtt_seconds):
        if outcome not in OUTCOMES:
            raise self._fault(f"/v1/end: unknown outcome {outcome!r}")
        return self._request("POST", "/v1/end", {
            "item_id": item_id, "subset": subset, "attack": attack, "victim": victim, "outcome": outcome,
            "final_text": final_text, "runner_queries": int(runner_queries),
            "openattack_invocations": int(openattack_invocations), "runner_seconds": float(runner_seconds),
            "rtt_seconds": float(rtt_seconds)})

    def shutdown(self):
        return self._request("POST", "/v1/shutdown", {})
