"""Standard-library test double of the victim server (protocol v1; the real server is ).

It holds invented items in memory and never loads a model. Modes:
 never every text scores [0.1, 0.9] (label 1 is always kept, so an attack can only fail or hit the budget)
 flip the item's original text scores [0.1, 0.9]; any other text scores [0.6, 0.4] (first change flips it)
Rules shared with the real server:
 - only phase `attack` is charged, len(texts) per request, before inference, per (subset, item, attack, victim);
 - a request that would exceed the budget gets 409 {"error": "budget_exhausted", "queries_used"} and nothing is scored;
 - malformed requests get 400 {"error": "bad_request"}.
Test hooks: `fail_on_query=N` answers the Nth attack-phase request with 500 {"error": "pipeline_fault"};
`skew=k` adds k to every reported `queries_used` (a lying server, for the test); every /v1/end body is kept
and its key becomes a /v1/done row.
"""
import json
import os
import socketserver
import tempfile
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_TEXT_CHARS = 65536
PHASES = ("clean", "attack", "posthoc")
OUTCOMES = ("success", "failed", "limit", "skipped")
KEEP = [0.1, 0.9]
FLIP = [0.6, 0.4]


class _State:
    def __init__(self, items, mode, budget, fail_on_query, skew, cells):
        if mode not in ("never", "flip"):
            raise ValueError(f"unknown mode {mode!r}")
        self.items = [dict(it) for it in items]
        self.by_id = {it["item_id"]: it for it in self.items}
        self.mode = mode
        self.budget = int(budget)
        self.fail_on_query = fail_on_query
        self.skew = int(skew)
        self.cells = [list(c) for c in (cells or [["deepwordbug", "content_only", "rd"]])]
        self.counters = {}          # (subset, item_id, attack, victim) -> charged attack-phase texts
        self.attack_requests = 0    # attack-phase requests seen (fault injection counter)
        self.requests = {p: 0 for p in PHASES}
        self.records = 0
        self.ends = []
        self.done = []
        self.lock = threading.Lock()

    def score(self, item_id, text):
        if self.mode == "never":
            return list(KEEP)
        return list(KEEP) if text == self.by_id[item_id]["text"] else list(FLIP)


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - silence: no request line, no text, ever
        return

    def _reply(self, status, obj):
        body = json.dumps(obj, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BODY_BYTES:
            return None
        try:
            obj = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except (UnicodeDecodeError, ValueError):
            return None
        return obj if isinstance(obj, dict) else None

    def _bad(self, detail):
        self._reply(400, {"error": "bad_request", "detail": detail})

    def do_GET(self):
        st = self.server.state
        if self.path == "/v1/ping":
            self._reply(200, {"ok": True, "run_id": "fake-run", "budget": st.budget})
        elif self.path == "/v1/plan":
            self._reply(200, {"cells": st.cells, "budget": st.budget})
        elif self.path == "/v1/done":
            with st.lock:
                self._reply(200, {"done": [list(r) for r in st.done]})
        else:
            self._reply(404, {"error": "not_found"})

    def do_POST(self):
        st = self.server.state
        body = self._body()
        if body is None:
            return self._bad("body is not a JSON object within the size limit")
        if self.path == "/v1/items":
            items = [{"item_id": it["item_id"], "text": it["text"], "label": 1}
                     for it in st.items if int(it.get("label", 1)) == 1]
            return self._reply(200, {"items": items})
        if self.path == "/v1/query":
            return self._query(st, body)
        if self.path == "/v1/end":
            if body.get("outcome") not in OUTCOMES:
                return self._bad("outcome")
            with st.lock:
                st.ends.append(body)
                st.done.append([body.get("item_id"), body.get("attack"), body.get("victim"), body.get("subset")])
            return self._reply(200, {"end_state": body["outcome"], "posthoc_gate": None, "ledger_rows": 1})
        if self.path == "/v1/shutdown":
            with st.lock:
                reply = {"rows": len(st.ends), "records": st.records}
            self._reply(200, reply)
            threading.Thread(target=self.server.shutdown, daemon=True).start()
            return None
        return self._reply(404, {"error": "not_found"})

    def _query(self, st, body):
        texts = body.get("texts")
        phase = body.get("phase")
        item_id = body.get("item_id")
        if phase not in PHASES or item_id not in st.by_id:
            return self._bad("phase or item_id")
        if not (isinstance(texts, list) and 1 <= len(texts) <= 64
                and all(isinstance(t, str) and len(t) <= MAX_TEXT_CHARS for t in texts)):
            return self._bad("texts")
        key = (body.get("subset"), item_id, body.get("attack"), body.get("victim"))
        with st.lock:
            st.requests[phase] += 1
            used = st.counters.get(key, 0)
            if phase == "attack":
                st.attack_requests += 1
                if st.fail_on_query is not None and st.attack_requests == st.fail_on_query:
                    return self._reply(500, {"error": "pipeline_fault", "detail": "InjectedFault"})
                if used + len(texts) > st.budget:
                    return self._reply(409, {"error": "budget_exhausted", "queries_used": used})
                used += len(texts)
                st.counters[key] = used
            first = st.records
            st.records += len(texts)
        probs = [st.score(item_id, t) for t in texts]
        return self._reply(200, {"probs": probs, "record_ids": [f"fake-{first + i}" for i in range(len(texts))],
                                 "queries_used": used + st.skew, "deferred": [False] * len(texts)})


class _Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class FakeVictimHandle:
    def __init__(self, server, thread, socket_dir):
        self._server = server
        self._thread = thread
        self.socket_dir = socket_dir
        self.socket_path = str(Path(socket_dir) / "victim.sock")
        self.state = server.state

    @property
    def counters(self):
        with self.state.lock:
            return dict(self.state.counters)

    @property
    def ends(self):
        with self.state.lock:
            return [dict(e) for e in self.state.ends]

    def stop(self):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass


def serve_in_thread(socket_dir, items, mode, budget, fail_on_query=None, *, skew=0, cells=None):
    """Start the fake server on <socket_dir>/victim.sock (socket_dir created 0700) and return its handle."""
    socket_dir = Path(socket_dir) if socket_dir else Path(tempfile.mkdtemp(prefix="aac-fv-"))
    socket_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(socket_dir, 0o700)
    server = _Server(str(socket_dir / "victim.sock"), _Handler)
    server.state = _State(items, mode, budget, fail_on_query, skew, cells)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    return FakeVictimHandle(server, thread, socket_dir)


if __name__ == "__main__":  # manual use: python fake_victim_server.py <socket_dir> <items.json> <mode> <budget> [fail_on]
    import sys
    import time
    args = sys.argv[1:]
    handle = serve_in_thread(args[0], json.loads(Path(args[1]).read_text(encoding="utf-8")), args[2], int(args[3]),
                             int(args[4]) if len(args) > 4 else None)
    print(f"socket={handle.socket_path}", flush=True)
    try:
        while handle._thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        handle.stop()
