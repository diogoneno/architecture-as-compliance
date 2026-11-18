"""R3 victim server (server half of ).

 python -m aac.serve.victim_server serve --config <R3 config> --seed N --socket-dir <dir> [--force] [--resume]
 python -m aac.serve.victim_server ping --socket-dir <dir>
 python -m aac.serve.victim_server shutdown --socket-dir <dir>

One env-main process owns the R3 run: its identity and run directory, ONE ShardWriter (one seq), the per-item
QueryBudget, the attack ledger and the seal. The OpenAttack runner (env-attack) reaches it only over HTTP/1.1 on
`<socket-dir>/victim.sock` (victim protocol v1; client attacks/runner/protocol.py, test double
attacks/runner/fake_victim_server.py, whose rules this server mirrors).

Victims (pre-registered specification):
 content_only victim A: the content-only encoder (C1, raw_score = P(Refuted), calibration none); probs [1 - P, P]
 full victim B: C1 -> C4 un-gated, C4/C5 loaded from the seed-matched sealed R1 `attack.victim_source` run;
 probs [1 - s, s]. The server runs the gated post-hoc query itself on a successful example (logged,
 not charged): deferred -> end_state deferred / posthoc deflected, else success / flip.
 full, gated-masked view (addition A, `attack.victim_view: gated-masked`, OpenAttackGatedCfg): victim B's
 clean and attack queries run through the GATED decision function and the runner receives the pre-registered specification:
 the masked interface does not reveal a deferral), so a deferral reads as still Refuted and the search
 continues. A success needs an auto-decided wrong final; no post-hoc query runs; posthoc_gate is flip
 on success, deflected when the item ended without success after an attack query that was a deferred
 un-gated flip (s < 0.5), else no_effect.
Accounting: every text of every query is one Pipeline.infer request and one Appendix-B record. Phase `attack` is
charged to the item's QueryBudget BEFORE inference (a refused batch: 409, nothing inferred, no record); phases
`clean` and `posthoc` are logged and not charged. The ledger row of an item lists all its record ids in order, so
invariant L2 holds strictly.
Security (..15): socket in a 0700 directory owned by this uid; a peer whose SO_PEERCRED uid differs gets
403 before its body is read; at most 64 texts, 8 MiB per body and 65,536 characters per text (400 before
inference); one request at a time (no threads: the GPU is one resource and the log has one writer). Output: ids,
counts and exception class names only. Adversarial text is written only to the private 0600 store
`<out>/data/derived/private/adversarial/<run_id>.jsonl`, keyed by claim_hash, for the Phase 4 replay."""
from __future__ import annotations

import argparse
import http.client
import json
import os
import socket
import socketserver
import stat
import struct
import sys
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler
from pathlib import Path

from aac.serve.protocol import MAX_BODY, POST_PATHS, BadRequest, parse_request

SOCKET_NAME = "victim.sock"
COUNTERS_FILE = "attack-counters.jsonl"
PARTIAL_LEDGER = "attack-ledger.partial.jsonl"    # one line per finished item (ledger row + counters + stats), fsync
QUERY_JOURNAL = "attack-queries.partial.jsonl"    # one line per inference batch (key, phase, n): counts drops on resume
SESSIONS_JOURNAL = "attack-sessions.partial.jsonl"  
PARTIAL_FILES = (PARTIAL_LEDGER, QUERY_JOURNAL, SESSIONS_JOURNAL)  
RESUME_LOG = "resume-log.json"
RESUME_HOLD = "_resume"  
TIMINGS_FILE = "attack-timings.json"
CELLS_FILE = "attack-cells.json"
PRIVATE_ADVERSARIAL = Path("data") / "derived" / "private" / "adversarial"
TRUNCATION_RULE_R3 = ("item-15a extended to R3: a victim-B adversarial text (phase attack or posthoc) "
                      "cut for the scorer pair only (longest_first at the scorer max_length)")
SERVE_DETERMINISM = "victim server; adversary non-deterministic (OpenAttack runner); "
DECIMALS = 6
HANDLER_TIMEOUT_S = 120.0  # a stalled client cannot hold the single serving thread for longer


class ServeRefused(RuntimeError):
    """The server cannot start (exit 2): wrong config kind, unsafe socket directory, run refused."""


class _Reply(Exception):
    """Internal: an HTTP reply other than 200 (status, JSON object)."""

    def __init__(self, status: int, obj: dict) -> None:
        super().__init__(status)
        self.status = status
        self.obj = obj


def _bad(detail: str) -> _Reply:
    return _Reply(400, {"error": "bad_request", "detail": detail})


@dataclass
class ItemState:
    key: tuple  # (item_id, attack, victim, subset)
    budget: object  # aac.attacks.budget.QueryBudget
    record_ids: list = field(default_factory=list)
    attack_hashes: set = field(default_factory=set)
    server_seconds: float = 0.0
    truncations: list = field(default_factory=list)  
    clean_p: float | None = None  
    p_by_hash: dict = field(default_factory=dict)  
    refused_409: bool = False  
    deferred_flip: bool = False  # gated-masked view: an attack query was deferred with an un-gated flip (s < 0.5)


def _probs(p: float) -> list[float]:
    p = min(1.0, max(0.0, float(p)))
    return [round(1.0 - p, DECIMALS), round(p, DECIMALS)]


def _masked_probs(p: float, deferred: bool) -> list[float]:
    """pre-registered specification,
 [0.0, 1.0] on deferral, so a deferral reads as still Refuted."""
    return [0.0, 1.0] if deferred else _probs(p)


def _gate_aware_end(outcome: str, deferred_flip: bool) -> tuple[str, str | None]:
    """(end_state, posthoc_gate) of a gated-masked victim-B item (addition A): no post-hoc query runs, the attacker
 already saw the gate. success (an auto-decided wrong final, checked in end) -> (success, flip); limit ->
 budget_exhausted, failed -> unchanged, each deflected when an attack query was a deferred un-gated flip, else
 no_effect; skipped (never attacked) -> (unchanged, None)."""
    if outcome == "success":
        return "success", "flip"
    if outcome == "skipped":
        return "unchanged", None
    return ("budget_exhausted" if outcome == "limit" else "unchanged"), ("deflected" if deferred_flip else "no_effect")


def _kept(p: float) -> bool:
    """the label (1, Refuted) is kept when argmax of the probs the runner was sent is 1; ties go to index 0, as
 numpy's argmax in the runner (openattack_runner._attack_item) and OpenAttack's prediction do."""
    q = _probs(p)
    return q[1] > q[0]


class VictimService:
    """All server state and the endpoint logic, independent of the socket (unit-testable)."""

    def __init__(self, config: str | Path, seed: int, *, force: bool = False, resume: bool = False, out=None) -> None:
        import dataclasses

        from aac.attacks.ledger import LedgerWriter
        from aac.components.c3_fusion import FEATURES, ContentOnlyFusion
        from aac.data.fixtures import load_split
        from aac.logging import ShardWriter, load_validator
        from aac.pipeline import Pipeline, RunContext
        from aac.pipeline.factory import build_pipeline
        from aac.regimes.common import SourceMismatch, check_toggles, load_source_maps, regime_label, target_items
        from aac.regimes.r3 import GATED, PROVIDER_KEY, VICTIM_TOGGLES
        from aac.run import hashing
        from aac.run.clock import WallClock
        from aac.run.config import RunConfig
        from aac.run.env import write_env
        from aac.run.finalize import RunRefused, claim_run_dir
        from aac.run.identity import identity, lock_path
        from aac.run.paths import out_root, root as project_root, runs_dir
        from aac.run.seeds import seed_everything

        self.out = out or sys.stdout
        self._replace = dataclasses.replace
        root = project_root()
        self.root = root
        ident = identity(root, Path(config), int(seed))
        if not isinstance(ident["cfg"], RunConfig):
            raise ServeRefused(f"{config} is not an aac-config/1 regime config")
        resolved = ident["resolved"]
        if resolved["regime"] != "R3" or resolved.get("attack") is None:
            raise ServeRefused(f"{config}: the victim server serves R3 configs with an `attack` block only")
        check_toggles(resolved)
        self.ident, self.resolved, self.run_id = ident, resolved, ident["run_id"]
        self.seed = int(seed)
        atk = resolved["attack"]
        self.budget_limit = int(atk["budget"])
        key = (resolved["regime"], resolved["system"], resolved["name"], self.seed)
        try:
            self.run_dir, self.target_dir, self.stale_ids = claim_run_dir(root, self.run_id, key, force, resumable=True,
                                                                          resume=resume)
        except RunRefused as exc:
            raise ServeRefused(str(exc)) from None
        
        # directory that already holds a session's files is a genuine resume and may take a held state back
        self._fresh_dir = not any(self.run_dir.iterdir())
        
        # BEFORE this start writes into it, so a start that dies later (model load, source maps) cannot make a manual
        # `--force --resume` see a non-empty directory and restore the sealed run's hold
        self.hold_dir = runs_dir(root) / RESUME_HOLD / self.run_id
        if self._fresh_dir:
            self._quarantine_held()
        (self.run_dir / "config.resolved.json").write_text(hashing.dumps_sorted(resolved), encoding="ascii")
        write_env(self.run_dir, lock_path(root))
        seed_everything(self.seed)

        # victim B: the R3 config's own system (full), C4/C5 from the sealed seed-matched R1 source run
        pipe_b = build_pipeline(resolved, root, self.seed)
        
        # copies (written by the first session) that differ from the current source's (re-sealed or re-pointed R1)
        try:
            self.source_run_id = load_source_maps(pipe_b, resolved, root, self.run_dir, atk["victim_source"])
        except (SourceMismatch, RuntimeError, ValueError) as exc:
            raise ServeRefused(str(exc)) from None
        
        # calibration, no gate. It shares victim B's C1 object (same weights; one model in memory).
        c3_a = {"gate": "content-only-passthrough", "fit_on": "val_fuse", "features": list(FEATURES),
                "weights": [], "bias": 0.0}
        pipe_a = Pipeline(c1=pipe_b.c1, c2=pipe_b.c2, c3=ContentOnlyFusion(c3_a), c4=pipe_b.c4, c5=pipe_b.c5,
                          provider=pipe_b.provider, providers=pipe_b.providers, device=pipe_b.device)
        self.pipelines = {"content_only": pipe_a, "full": pipe_b}
        
        # scorer_truncated_* items (zeros included). A stub C2 has no counter and the run writes neither.
        self._has_truncation_counter = hasattr(pipe_b.c2, "scorer_truncations")
        self.toggles = dict(VICTIM_TOGGLES)
        self.gated = GATED
        # addition A : the gated-masked view serves victim B through the gate; the default path is unchanged
        self.gate_aware = atk.get("victim_view") == "gated-masked"
        if self.gate_aware:
            self.toggles["full"] = GATED
        self.provider_key = dict(PROVIDER_KEY)
        self.default_provider = resolved["components"]["c2"]["provider"]
        self.regime = regime_label(resolved)

        self.validator = load_validator()
        self.clock = WallClock()
        # item_shards: one verified shard per finished item, so a crash loses at most the open item 
        self.writer = ShardWriter(self.run_dir, self.run_id, self.validator, item_shards=True)
        self.ctx = RunContext(cfg_resolved=resolved, config_hash=ident["config_hash"],
                              data_version=ident["data_version"], model_version=ident["model_version"],
                              run_id=self.run_id, sink=self.writer, clock=self.clock,
                              toggles=self.toggles["full"], seed=self.seed)
        self.ledger = LedgerWriter(self.run_dir, root)
        self.started = self.clock.now_iso_ms()

        # plan cells and item lists (positives only; the AVeriTeC subset restricted to data.attack_subset)
        self.cells = [[a, v, s] for a in atk["attacks"] for v in ("content_only", "full") for s in atk["victims"][v]]
        subset_ids = {}
        if resolved["data"].get("attack_subset"):
            subset_ids["averitec_chrono"] = set(load_split(root / resolved["data"]["attack_subset"])["ids"])
        limit = atk.get("item_limit")
        self.items: dict[tuple[str, str], list[dict]] = {}
        texts_by_subset: dict[str, dict] = {}
        for _, victim, subset in self.cells:
            if (subset, victim) in self.items:
                continue
            if subset not in texts_by_subset:
                texts_by_subset[subset] = target_items(root, resolved, subset)
            items = texts_by_subset[subset]
            ids = sorted(i for i in items if items[i]["positive"] and (subset not in subset_ids or i in subset_ids[subset]))
            if limit is not None:
                ids = ids[: int(limit)]
            self.items[(subset, victim)] = [{"item_id": i, "text": items[i]["text"], "label": 1} for i in ids]
        self.item_ids = {k: {it["item_id"] for it in v} for k, v in self.items.items()}
        self.item_text = {(s, v, it["item_id"]): it["text"] for (s, v), its in self.items.items() for it in its}
        self.attack_subset_n = len(subset_ids.get("averitec_chrono", ()))

        self.done: list[list[str]] = []
        self.done_keys: set[tuple] = set()
        self.open: ItemState | None = None
        self.counters: list[dict] = []
        self.adversarial: list[dict] = []
        self.truncations: list[dict] = []
        self.cell_stats: dict[tuple, dict] = {}
        self.faulted = False
        self.stopping = False
        self.first_query_t: float | None = None
        self.private_store = out_root(root) / PRIVATE_ADVERSARIAL / f"{self.run_id}.jsonl"
        
        # directory skipped by every runs/ scan, deleted after the seal
        self.resume_log = {"resumed": False, "kept_items": 0, "dropped_records": 0, "resumes": []}
        try:
            self._resume()
            self._register_session()
        except BaseException:
            self.writer.abort()
            raise
        self.say(f"serving {self.run_id}: cells={len(self.cells)} budget={self.budget_limit} "
                 f"source={self.source_run_id} done={len(self.done)}")

    # ------------------------------------------------------------------ resume 
    @staticmethod
    def _read_jsonl(path: Path) -> list[dict]:
        """Complete JSON lines of an append-only file; a torn last line (a crash mid-write) is ignored."""
        if not path.is_file():
            return []
        rows = []
        for line in path.read_bytes().split(b"\n"):
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                break
        return rows

    @staticmethod
    def _rewrite_jsonl(path: Path, rows: list[dict]) -> None:
        tmp = path.with_name(path.name + ".tmp")
        with open(tmp, "w", encoding="ascii", newline="\n") as fh:
            fh.write("".join(json.dumps(r, sort_keys=True, ensure_ascii=True) + "\n" for r in rows))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)

    def _hold_partials(self) -> None:
        """move the working state out of the run directory just before finalize_run (seal hashes every
 file in it). A kill inside the seal window leaves it under runs/_resume/<rid>; the next start restores it."""
        self.hold_dir.mkdir(parents=True, exist_ok=True)
        for name in PARTIAL_FILES:
            if (self.run_dir / name).is_file():
                os.replace(self.run_dir / name, self.hold_dir / name)

    def _quarantine_held(self) -> None:
        """ follow-up: a hold that no genuine resume may take back (a --force start into a fresh directory, or
 a run that sealed) moves to runs/_resume/_stale/<rid>-<unix-seconds>/; it is never deleted silently."""
        if not self.hold_dir.is_dir():
            return
        stale = self.hold_dir.parent / "_stale" / f"{self.run_id}-{int(time.time())}"
        n = 0
        while stale.exists():
            n += 1
            stale = stale.with_name(f"{self.run_id}-{int(time.time())}-{n}")
        stale.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.hold_dir, stale)
        self.say(f"resume-hold: stale hold quarantined {stale}")

    def _restore_held(self, *, at_start: bool = False) -> None:
        """put held partial files back into the (unsealed) run directory; refuse when both copies exist."""
        if not self.hold_dir.is_dir():
            return
        if at_start and self._fresh_dir:
            self._quarantine_held()
            return
        for name in PARTIAL_FILES:
            held = self.hold_dir / name
            if not held.is_file():
                continue
            if (self.run_dir / name).exists():
                raise ServeRefused(f"{self.run_id}: {name} exists both in the run directory and under "
                                   f"runs/{RESUME_HOLD}/; inspect them (nothing was deleted)")
            os.replace(held, self.run_dir / name)
        self.hold_dir.rmdir()

    def _drop_held(self) -> None:
        for name in PARTIAL_FILES:
            (self.hold_dir / name).unlink(missing_ok=True)
        try:
            self.hold_dir.rmdir()
        except FileNotFoundError:
            pass

    def _register_session(self) -> None:
        """one durable line per server start, so `sessions` also counts a start whose resume kept nothing."""
        self.session_no = len(self._read_jsonl(self.run_dir / SESSIONS_JOURNAL)) + 1
        with open(self.run_dir / SESSIONS_JOURNAL, "a", encoding="ascii", newline="\n") as fh:
            fh.write(json.dumps({"session": self.session_no}, sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def _resume(self) -> None:
        """Keep the first m = min(partial rows, verified shards) finished items, drop everything after them,
 restart the seq after the kept records and record the drop in resume-log.json.

 verified shards are never truncated beyond the one item that end may have sealed without its
 partial row (seal_item, then the row); more is refused, and so is a final attack-ledger.jsonl (a finish
 that did not seal) listing more rows than the partial ledger."""
        from aac.attacks.ledger import LEDGER_FILE
        from aac.logging.writer import read_shard

        self._restore_held(at_start=True)
        partial = self._read_jsonl(self.run_dir / PARTIAL_LEDGER)
        journal = self._read_jsonl(self.run_dir / QUERY_JOURNAL)
        if not partial and not journal and self.writer.resume_seq == 0:
            self._reconcile_store({})  
            return
        final_rows = len(self._read_jsonl(self.run_dir / LEDGER_FILE))
        if self.writer.verified_shards > len(partial) + 1 or final_rows > len(partial):
            raise ServeRefused(f"{self.run_id}: {self.writer.verified_shards} verified shards and {final_rows} final "
                               f"ledger rows but {len(partial)} partial ledger rows; refusing to drop sealed items "
                               "(inspect, or re-run with --force)")
        m = min(len(partial), self.writer.verified_shards)
        
        # row and the store write): that item is dropped; a gap anywhere else is refused, nothing is deleted.
        stored = {(k, h) for k, h, _ in self._store_lines()}
        missing = [k for k, row in enumerate(partial[:m]) if self._success_key(row) not in (None, *stored)]
        if missing and missing[-1] == m - 1:
            m -= 1
            missing.pop()
        if missing:
            raise ServeRefused(f"{self.run_id}: the private adversarial store lacks the text of {len(missing)} kept "
                               "successful items; refusing to resume (inspect, or re-run with --force)")
        self.writer.keep_shards(m)
        kept = partial[:m]
        shard_dir = self.writer.shard_dir
        from aac.logging.writer import shard_name

        for k, row in enumerate(kept):
            on_disk = [r["record_id"] for r in read_shard(shard_dir / shard_name(k))]
            if on_disk != row["ledger"]["record_ids"]:
                raise ServeRefused(f"{self.run_id}: shard {k} and partial ledger row {k} disagree; "
                                   "the run cannot resume (re-run with --force)")
        self.clock.start_at(self.writer.resume_seq)
        kept_keys = set()
        for row in kept:
            led = row["ledger"]
            key = (led["item_id"], led["attack"], led["victim"], led["subset"])
            self.ledger.append(led)
            self.counters.append(row["counters"])
            self.adversarial.append(row["adversarial"])
            self.truncations.extend(row.get("truncations") or [])
            st = self.cell_stats.setdefault((led["attack"], led["victim"], led["subset"]),
                                            {"items": 0, "queries": 0, "seconds": 0.0})
            st["items"] += 1
            st["queries"] += len(led["record_ids"])
            st["seconds"] += float(row["counters"]["server_seconds"])
            self.done.append(list(key))
            self.done_keys.add(key)
            kept_keys.add(key)
        dropped = sum(int(j["n"]) for j in journal if tuple(j["key"]) not in kept_keys)
        
        # inference seconds of the dropped batches, read BEFORE the journal is cut to the kept items
        dropped_s = sum(float(j.get("seconds", 0.0)) for j in journal if tuple(j["key"]) not in kept_keys)
        previous = self._read_json(self.run_dir / RESUME_LOG) or {}
        spans = {str(k): float(v) for k, v in (previous.get("session_seconds") or {}).items()}
        for j in journal:
            if "session" in j:
                spans[str(j["session"])] = max(spans.get(str(j["session"]), 0.0), float(j.get("elapsed", 0.0)))
        self._rewrite_jsonl(self.run_dir / PARTIAL_LEDGER, kept)
        self._reconcile_store({sk[0]: sk[1] for sk in map(self._success_key, kept) if sk is not None})
        self._rewrite_jsonl(self.run_dir / QUERY_JOURNAL, [j for j in journal if tuple(j["key"]) in kept_keys])
        resumes = list(previous.get("resumes") or [])
        resumes.append({"kept_items": m, "kept_records": self.writer.resume_seq, "dropped_records": dropped,
                        "dropped_seconds": round(dropped_s, DECIMALS)})
        self.resume_log = {"resumed": True, "kept_items": m, "dropped_records": sum(r["dropped_records"] for r in resumes),
                           "dropped_seconds": round(sum(float(r.get("dropped_seconds", 0.0)) for r in resumes), DECIMALS),
                           "session_seconds": {k: round(v, DECIMALS) for k, v in sorted(spans.items())},
                           "resumes": resumes}
        self._write_resume_log()
        self.say(f"resumed {self.run_id}: kept_items={m} kept_records={self.writer.resume_seq} dropped_records={dropped}")

    @staticmethod
    def _read_json(path: Path):
        try:
            return json.loads(path.read_text(encoding="ascii")) if path.is_file() else None
        except ValueError:
            return None

    def _write_resume_log(self) -> None:
        from aac.run import hashing

        tmp = self.run_dir / (RESUME_LOG + ".tmp")
        tmp.write_text(hashing.dumps_sorted(self.resume_log), encoding="ascii")
        os.replace(tmp, self.run_dir / RESUME_LOG)

    def _journal(self, key: tuple, phase: str, n: int, seconds: float = 0.0) -> None:
        """One line per inference batch; also this session's number, the batch's inference seconds and the
 session's wall span so far (first query to the end of this batch), so a resume can account for both."""
        elapsed = 0.0 if self.first_query_t is None else time.perf_counter() - self.first_query_t
        with open(self.run_dir / QUERY_JOURNAL, "a", encoding="ascii", newline="\n") as fh:
            fh.write(json.dumps({"key": list(key), "phase": phase, "n": int(n), "session": self.session_no,
                                 "seconds": round(seconds, DECIMALS), "elapsed": round(elapsed, DECIMALS)},
                                sort_keys=True) + "\n")

    # ------------------------------------------------------------------ output
    def say(self, line: str) -> None:
        print(line, file=self.out, flush=True)

    # ------------------------------------------------------------------ endpoints
    def handle(self, method: str, path: str, body: dict | None) -> tuple[int, dict]:
        try:
            if method == "GET":
                if path == "/v1/ping":
                    return 200, {"ok": True, "run_id": self.run_id, "budget": self.budget_limit}
                if path == "/v1/plan":
                    return 200, {"cells": [list(c) for c in self.cells], "budget": self.budget_limit}
                if path == "/v1/done":
                    return 200, {"done": [list(r) for r in self.done]}
                return 404, {"error": "not_found"}
            if path not in POST_PATHS:
                return 404, {"error": "not_found"}
            try:
                req = parse_request(path, body if body is not None else {})
            except BadRequest as exc:
                raise _bad(str(exc)) from None
            if path == "/v1/items":
                return 200, self.items_for(req)
            if path == "/v1/query":
                return 200, self.query(req)
            if path == "/v1/end":
                return 200, self.end(req)
            return 200, self.shutdown()
        except _Reply as r:
            return r.status, r.obj
        except Exception as exc:  # noqa: BLE001 - never a traceback (a message may carry a text); class name only
            self.faulted = True
            self.say(f"server_fault {type(exc).__name__} path={path}")
            return 500, {"error": "server_fault", "detail": type(exc).__name__}

    def _check_cell(self, req: dict) -> tuple:
        cell = [req["attack"], req["victim"], req["subset"]]
        if cell not in self.cells:
            raise _bad("(attack, victim, subset) is not a cell of this run's plan")
        if req["item_id"] not in self.item_ids.get((req["subset"], req["victim"]), ()):
            raise _bad("item_id is not an item of this (subset, victim)")
        key = (req["item_id"], req["attack"], req["victim"], req["subset"])
        if key in self.done_keys:
            raise _bad("item already ended")
        return key

    def items_for(self, req: dict) -> dict:
        if (req["subset"], req["victim"]) not in self.items:
            raise _bad("(subset, victim) is not served by this run's plan")
        return {"items": [dict(it) for it in self.items[(req["subset"], req["victim"])]]}

    def _infer(self, item: ItemState, texts: list[str], phase: str, toggles, first_index: int):
        from aac.pipeline import Request

        item_id, attack, victim, subset = item.key
        key = self.provider_key.get(subset, self.default_provider)
        reqs = [Request(claim_id=item_id, text=t, evidence_provider_key=key, regime=self.regime,
                        split=f"attack_{subset}", seed=self.seed,
                        query_ctx={"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                                   "query_index": first_index + i, "phase": phase})
                for i, t in enumerate(texts)]
        if self.first_query_t is None:
            self.first_query_t = time.perf_counter()
        cut_rows = getattr(self.pipelines[victim].c2, "scorer_truncations", None)
        n_cut0 = len(cut_rows) if cut_rows is not None else 0
        t0 = time.perf_counter()
        try:
            decisions = self.pipelines[victim].infer(reqs, self._replace(self.ctx, toggles=toggles))
        except Exception as exc:  # noqa: BLE001 - the class name only: a message may carry a text
            self.faulted = True
            self.say(f"pipeline_fault {type(exc).__name__} item={item_id} phase={phase}")
            raise _Reply(500, {"error": "pipeline_fault", "detail": type(exc).__name__}) from None
        seconds = time.perf_counter() - t0
        self._journal(item.key, phase, len(decisions), seconds)
        item.server_seconds += seconds  
        item.record_ids.extend(d.record_id for d in decisions)
        if cut_rows is not None and len(cut_rows) > n_cut0:  
            by_index = {first_index + i: d.record_id for i, d in enumerate(decisions)}
            for r in cut_rows[n_cut0:]:
                item.truncations.append({"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                                         "phase": r["phase"], "query_index": int(r["query_index"]),
                                         "record_id": by_index[int(r["query_index"])], "pairs": int(r["pairs"])})
        return decisions

    def query(self, req: dict) -> dict:
        from aac.attacks.budget import BudgetExhausted, QueryBudget
        from aac.run.hashing import claim_hash

        if self.faulted:
            raise _Reply(500, {"error": "server_faulted", "detail": "a pipeline fault stopped this server"})
        key = self._check_cell(req)
        phase, texts = req["phase"], req["texts"]
        if phase == "posthoc":
            raise _bad("the server runs the post-hoc query itself (POST /v1/end)")
        if self.open is not None and self.open.key != key:
            raise _bad("another item is open; end it first")
        if phase == "clean":
            if self.open is not None:
                raise _bad("the clean query was already sent for this item")
            if texts != [self.item_text[(req["subset"], req["victim"], req["item_id"])]]:  
                raise _bad("the clean text differs from the served item text (send exactly one text: the item's)")
            item = ItemState(key=key, budget=QueryBudget(self.budget_limit))
            decisions = self._infer(item, texts, "clean", self.toggles[req["victim"]], 0)
            item.clean_p = self._sent_p(decisions[0])
            self.open = item
        else:
            item = self.open
            if item is None:
                raise _bad("send the clean query first")
            before = item.budget.used
            try:
                item.budget.charge(len(texts))  # before inference: a refused batch is never inferred
            except BudgetExhausted:
                item.refused_409 = True
                raise _Reply(409, {"error": "budget_exhausted", "queries_used": before}) from None
            decisions = self._infer(item, texts, "attack", self.toggles[req["victim"]], before + 1)
            item.attack_hashes.update(claim_hash(t) for t in texts)
            item.p_by_hash.update((claim_hash(t), self._sent_p(d)) for t, d in zip(texts, decisions))
            if self.gate_aware and any(d.abstention_flag and not _kept(d.calibrated_probability) for d in decisions):
                item.deferred_flip = True
        if self.gate_aware:
            return {"probs": [_masked_probs(d.calibrated_probability, bool(d.abstention_flag)) for d in decisions],
                    "record_ids": [d.record_id for d in decisions],
                    "queries_used": item.budget.used,
                    "deferred": [False] * len(decisions)}
        return {"probs": [_probs(d.calibrated_probability) for d in decisions],
                "record_ids": [d.record_id for d in decisions],
                "queries_used": item.budget.used,
                "deferred": [bool(d.abstention_flag) for d in decisions]}

    def _sent_p(self, d) -> float:
        """The P(Refuted) the runner was sent for decision d ( checks read it): the calibrated value, or under the
 gated-masked view 1.0 on deferral (pre-registered specification."""
        if self.gate_aware and d.abstention_flag:
            return 1.0
        return float(d.calibrated_probability)

    def end(self, req: dict) -> dict:
        from aac.run.hashing import claim_hash

        if self.faulted:
            raise _Reply(500, {"error": "server_faulted", "detail": "a pipeline fault stopped this server"})
        key = self._check_cell(req)
        item = self.open
        if item is None or item.key != key:
            raise _bad("this item is not open (send its clean query first)")
        used = item.budget.used
        outcome, victim = req["outcome"], req["victim"]
        if req["runner_queries"] != used:  # Counting disagreement is an error
            raise _bad(f"counter mismatch: runner_queries={req['runner_queries']} server queries_used={used}")
        if outcome == "skipped" and used:
            raise _bad("outcome 'skipped' with attack queries charged")
        
        if outcome == "skipped" and _kept(item.clean_p):
            raise _bad("outcome 'skipped' needs a clean decision that is already wrong; the logged one keeps the label")
        final = req["final_text"]
        if outcome == "success" and claim_hash(final) not in item.attack_hashes:
            raise _bad("final_text was not sent as an attack-phase query of this item")
        if outcome == "success" and _kept(item.p_by_hash[claim_hash(final)]):
            raise _bad("final_text did not flip the logged decision")
        if outcome == "limit" and not (item.refused_409 or used >= self.budget_limit
                                       or req["openattack_invocations"] == used + 1):
            raise _bad("outcome 'limit' without an exhausted budget (no 409, budget not used up, and no refused "
                       "OpenAttack call: openattack_invocations != queries + 1)")
        posthoc = None
        if self.gate_aware:  # addition A: the attacker saw the gate; no post-hoc query (end_state rule above)
            end_state, posthoc = _gate_aware_end(outcome, item.deferred_flip)
        elif outcome == "success":
            end_state = "success"
            if victim == "full":
                g = self._infer(item, [final], "posthoc", self.gated, used)[0]
                posthoc, end_state = ("deflected", "deferred") if g.abstention_flag else ("flip", "success")
        elif outcome == "limit":
            end_state = "budget_exhausted"
            posthoc = "no_effect" if victim == "full" else None
        else:  # failed, skipped
            end_state = "unchanged"
            posthoc = "no_effect" if (victim == "full" and outcome == "failed") else None
        item_id, attack, _, subset = key
        row = {"item_id": item_id, "attack": attack, "victim": victim, "subset": subset, "queries_used": used,
               "budget": self.budget_limit, "end_state": end_state, "posthoc_gate": posthoc,
               "record_ids": list(item.record_ids)}
        counters = {
            "item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
            "runner_queries": req["runner_queries"], "server_queries": used,
            "openattack_invocations": req["openattack_invocations"],
            "runner_seconds": round(req["runner_seconds"], DECIMALS), "rtt_seconds": round(req["rtt_seconds"], DECIMALS),
            "server_seconds": round(item.server_seconds, DECIMALS)}
        adversarial = {"item_id": item_id, "attack": attack, "victim": victim, "subset": subset,
                       "round": None, "claim_hash": claim_hash(final) if outcome == "success" else None,
                       "attacked": outcome != "skipped"}
        self.ledger.append(row)  # validates the row before anything is written
        # the item's records become one verified shard, and only then its ledger row is made durable
        self.writer.seal_item()
        with open(self.run_dir / PARTIAL_LEDGER, "a", encoding="ascii", newline="\n") as fh:
            fh.write(json.dumps({"ledger": row, "counters": counters, "adversarial": adversarial,
                                 "truncations": item.truncations}, sort_keys=True, ensure_ascii=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        
        # success without its text, which the next resume drops (it is the last row), never an orphan store entry
        if outcome == "success":
            self._store_adversarial(key, final)
        self.counters.append(counters)
        self.adversarial.append(adversarial)
        self.truncations.extend(item.truncations)
        st = self.cell_stats.setdefault((attack, victim, subset), {"items": 0, "queries": 0, "seconds": 0.0})
        st["items"] += 1
        st["queries"] += len(item.record_ids)
        st["seconds"] += item.server_seconds
        self.done.append([item_id, attack, victim, subset])
        self.done_keys.add(key)
        self.open = None
        return {"end_state": end_state, "posthoc_gate": posthoc, "ledger_rows": len(self.ledger.rows)}

    @staticmethod
    def _success_key(row: dict):
        """((item_id, attack, victim, subset), claim_hash) of a partial-ledger row that ended in success, else None."""
        h = (row.get("adversarial") or {}).get("claim_hash")
        if h is None:
            return None
        led = row["ledger"]
        return (led["item_id"], led["attack"], led["victim"], led["subset"]), h

    def _store_lines(self) -> list[tuple]:
        """[(key, claim_hash, raw line)] of the private store; an unparseable line gets key None (it is never kept)."""
        if not self.private_store.is_file():
            return []
        out = []
        for raw in self.private_store.read_bytes().split(b"\n"):
            if not raw.strip():
                continue
            try:
                r = json.loads(raw)
                key = (r["item_id"], r["attack"], r["victim"], r["subset"]) if r.get("run_id") == self.run_id else None
                out.append((key, r["claim_hash"], raw))
            except (ValueError, KeyError, TypeError, AttributeError):
                out.append((None, None, raw))
        return out

    def _reconcile_store(self, kept: dict) -> None:
        """ rewrite the private store (tmp + os.replace, 0600) to exactly one line per kept success
 {key: claim_hash}; every other line (a dropped item, a duplicate, another text for a kept key, a torn line, a
 previous session of this run_id) moves to adversarial/_superseded/<rid>-<unix>.jsonl (0600), never deleted
 silently. Only counts are printed."""
        lines = self._store_lines()
        keep, drop, seen = [], [], set()
        for key, h, raw in lines:
            if key is not None and key not in seen and kept.get(key) == h:
                keep.append(raw)
                seen.add(key)
            else:
                drop.append(raw)
        if not drop:
            return
        qdir = self.private_store.parent / "_superseded"
        qdir.mkdir(mode=0o700, exist_ok=True)
        os.chmod(qdir, 0o700)
        base = f"{self.run_id}-{int(time.time())}"
        q, n = qdir / f"{base}.jsonl", 0
        while q.exists():
            n += 1
            q = qdir / f"{base}-{n}.jsonl"
        self._write_private(q, drop)
        if keep:
            tmp = self.private_store.with_name(self.private_store.name + ".tmp")
            tmp.unlink(missing_ok=True)
            self._write_private(tmp, keep)
            os.replace(tmp, self.private_store)
        else:
            self.private_store.unlink()
        self.say(f"private-store: kept {len(keep)} entries, quarantined {len(drop)} to adversarial/_superseded/")

    @staticmethod
    def _write_private(path: Path, lines: list[bytes]) -> None:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, b"".join(ln + b"\n" for ln in lines))
            os.fsync(fd)
        finally:
            os.close(fd)

    def _store_adversarial(self, key: tuple, text: str) -> None:
        """Append the adversarial text to the private 0600 store (0700 directory), keyed by claim_hash."""
        from aac.run.hashing import claim_hash

        d = self.private_store.parent
        private = d.parent
        if not private.is_dir():  # data/derived/private: created 0700 here when absent (it already is on cpu-host)
            private.parent.mkdir(parents=True, exist_ok=True)
            private.mkdir(mode=0o700)
            os.chmod(private, 0o700)
        d.mkdir(mode=0o700, exist_ok=True)
        os.chmod(d, 0o700)
        line = json.dumps({"claim_hash": claim_hash(text), "item_id": key[0], "attack": key[1], "victim": key[2],
                           "subset": key[3], "run_id": self.run_id, "text": text},
                          sort_keys=True, ensure_ascii=True) + "\n"
        fd = os.open(self.private_store, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, line.encode("ascii"))
            os.fsync(fd)
        finally:
            os.close(fd)

    def shutdown(self) -> dict:
        self.stopping = True
        return {"rows": len(self.ledger.rows), "records": self.writer.record_count,
                "open_item": self.open is not None}

    def _write_truncations(self, items: dict) -> None:
        """the sealed scorer-truncation.json of an R3 run (item-15 rule extended to victim-B adversarial texts:
 one row per record whose scorer pair was cut, ids and counts only) and the run-record items
 scorer_truncated_attack_<subset> (records) / scorer_truncated_pairs_attack_<subset> (pairs), zeros included."""
        from aac.regimes.r2 import TRUNCATION_FILE
        from aac.run import hashing

        rows = sorted(self.truncations, key=lambda r: (r["subset"], r["attack"], r["victim"], r["item_id"],
                                                       r["query_index"], r["phase"]))
        splits = sorted({f"attack_{s}" for _, v, s in self.cells if v == "full"})
        for split in splits:
            mine = [r for r in rows if f"attack_{r['subset']}" == split]
            items[f"scorer_truncated_{split}"] = len(mine)
            items[f"scorer_truncated_pairs_{split}"] = sum(int(r["pairs"]) for r in mine)
        scorer = getattr(self.pipelines["full"].c2, "scorer", None)
        report = {"rule": TRUNCATION_RULE_R3, "max_length": getattr(scorer, "max_length", None),
                  "by_split": {s: items[f"scorer_truncated_{s}"] for s in splits}, "records": rows}
        (self.run_dir / TRUNCATION_FILE).write_text(hashing.dumps_sorted(report), encoding="ascii")

    # ------------------------------------------------------------------ seal
    def abort(self) -> None:
        self.writer.abort()

    def finish(self) -> int:
        """Write the attack files, validate, check L2 strictly and seal. 0 sealed, 1 refused."""
        from aac.attacks.ledger import l2_check, write_adversarial
        from aac.logging import validate_run, write_counts
        from aac.logging.validate import claim_texts_for
        from aac.run import hashing
        from aac.run.finalize import finalize_run

        if self.faulted:
            self.writer.abort()
            self.say(f"refusing to seal {self.run_id}: a pipeline fault occurred (run left unsealed)")
            return 1
        session = 0.0 if self.first_query_t is None else time.perf_counter() - self.first_query_t
        
        spans = {str(k): float(v) for k, v in (self.resume_log.get("session_seconds") or {}).items()}
        spans[str(self.session_no)] = max(spans.get(str(self.session_no), 0.0), session)
        self.resume_log["session_seconds"] = {k: round(v, DECIMALS) for k, v in sorted(spans.items())}
        open_seconds = self.open.server_seconds if self.open is not None else 0.0
        shards = self.writer.close()  # an unfinished item's buffered records are dropped here and counted
        self.resume_log["dropped_at_shutdown"] = self.writer.dropped_records
        self.resume_log["dropped_seconds_at_shutdown"] = round(open_seconds, DECIMALS)
        if self.writer.dropped_records:
            self.say(f"dropped unfinished item: {self.writer.dropped_records} records")
        self._write_resume_log()
        self.ledger.close()
        write_adversarial(self.run_dir, self.adversarial)
        with open(self.run_dir / COUNTERS_FILE, "wb") as fh:
            fh.write(b"".join(hashing.canonical_rounded(r) + b"\n" for r in self.counters))
        cells = []
        stages = []
        timings = []
        device = getattr(self.pipelines["full"], "device", "cpu")
        for attack, victim, subset in self.cells:
            st = self.cell_stats.get((attack, victim, subset), {"items": 0, "queries": 0, "seconds": 0.0})
            secs = round(st["seconds"], DECIMALS)
            cells.append({"attack": attack, "victim": victim, "subset": subset, "items": st["items"],
                          "queries": st["queries"], "inference_seconds": secs})
            name = f"attack_{attack}_{victim}_{subset}"
            stages.append({"name": name, "device": device, "seconds": secs, "items": st["queries"]})
            timings.append({"stage": name, "seconds": secs, "items": st["queries"]})
        (self.run_dir / TIMINGS_FILE).write_text(
            hashing.dumps_sorted({"session_seconds": round(sum(spans.values()), DECIMALS), "sessions": self.session_no,
                                  "dropped_queries": int(self.resume_log.get("dropped_records", 0))
                                  + int(self.writer.dropped_records),
                                  "dropped_seconds": round(float(self.resume_log.get("dropped_seconds", 0.0)) + open_seconds,
                                                           DECIMALS),
                                  "cells": cells}), encoding="ascii")
        n_records = self.writer.record_count
        (self.run_dir / CELLS_FILE).write_text(
            hashing.dumps_sorted({"cells": [list(c) for c in self.cells], "rows": len(self.ledger.rows),
                                  "queries": n_records}), encoding="ascii")
        finished = self.clock.now_iso_ms()
        with open(self.run_dir / "wallclock.jsonl", "w", encoding="ascii", newline="\n") as fh:
            for t in timings:
                fh.write(json.dumps(t, sort_keys=True) + "\n")
        counts = validate_run(self.run_dir, claim_texts_for(self.resolved, self.root), self.validator)
        write_counts(self.run_dir / "schema-validation.txt", counts)
        if counts.records_with_claim_text or counts.records_with_user_id:
            self.say(f"refusing to seal {self.run_id}: records_with_claim_text={counts.records_with_claim_text} "
                     f"records_with_user_id={counts.records_with_user_id}")
            return 1
        l2 = l2_check(self.run_dir, strict=True)
        self.say(l2.as_line())
        if not l2.ok:
            self.say(f"refusing to seal {self.run_id}: invariant L2 failed")
            return 1
        items = {f"{subset}_{victim}_items": len(v) for (subset, victim), v in sorted(self.items.items())}
        items.update({"ledger_rows": len(self.ledger.rows), "cells": len(self.cells),
                      "attack_subset": self.attack_subset_n})
        depends = {self.source_run_id}
        c1 = self.resolved["components"]["c1"]
        if c1.get("checkpoint"):
            depends.add(c1["checkpoint"]["run_id"])
        base = "mdeberta-v3-base fp32 inference (sealed C1 checkpoint)" if c1["kind"] == "mdeberta-v3-base" \
            else "cpu numpy stubs"
        if self._has_truncation_counter:
            self._write_truncations(items)
        full_cfg = ("full-gated-masked/" if self.gate_aware else "full/") + self.resolved["attack"]["victim_source"]
        summary = {"configurations": ["content_only/c1-only", full_cfg],
                   "items": items, "stages": stages, "timings": timings, "depends_on": sorted(depends),
                   "l2_strict": True}
        
        # a refused or failed seal puts it back, so a restart resumes every finished item instead of truncating
        self._hold_partials()
        try:
            digest = finalize_run(self.root, self.run_dir, self.ident, summary=summary, record_count=n_records,
                                  shards=shards, clock=self.clock, started=self.started, finished=finished,
                                  system=self.resolved["system"], determinism_level=SERVE_DETERMINISM + base,
                                  target_dir=self.target_dir, stale_ids=self.stale_ids, violations=counts.violations)
        except Exception as exc:  # noqa: BLE001 - FinalizeRefused, seal's stray-file RuntimeError, OSError
            if self.run_dir.is_dir() and not (self.run_dir / "SEALED").exists():
                self._restore_held()
            elif (self.run_dir / "SEALED").exists() or (not self.run_dir.is_dir() and (self.target_dir / "SEALED").exists()):
                self._quarantine_held()  # SEALED was written: the run is sealed, its hold must never be resumed
            self.say(f"refusing to seal {self.run_id}: {type(exc).__name__}")
            return 1
        self._drop_held()  # first thing after a successful seal: the hold's lifetime ends here
        self.say(f"served {self.run_id}: {n_records} records, {len(self.ledger.rows)} ledger rows, SEALED {digest[:12]}")
        return 0


# ---------------------------------------------------------------------- socket layer
class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    timeout = HANDLER_TIMEOUT_S

    def log_message(self, format, *args):  # noqa: A002 - silence: no request line, no text, ever
        return

    def _reply(self, status: int, obj: dict) -> None:
        body = json.dumps(obj, separators=(",", ":"), ensure_ascii=True).encode("ascii")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _peer_uid(self) -> int:
        creds = self.connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
        _pid, uid, _gid = struct.unpack("3i", creds)
        return uid

    def _peer_ok(self) -> bool:
        if self._peer_uid() != self.server.allowed_uid:  # SO_PEERCRED, checked before any body byte is read
            self._reply(403, {"error": "forbidden", "detail": "peer uid differs from the server uid"})
            return False
        return True

    def do_GET(self):  # noqa: N802
        if not self._peer_ok():
            return
        status, obj = self.server.service.handle("GET", self.path, None)
        self._reply(status, obj)

    def do_POST(self):  # noqa: N802
        if not self._peer_ok():
            return
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._reply(400, {"error": "bad_request", "detail": "Content-Length is not an integer"})
        if n < 0 or n > MAX_BODY:
            return self._reply(400, {"error": "bad_request", "detail": f"body above {MAX_BODY} bytes"})
        raw = self.rfile.read(n) if n else b"{}"
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return self._reply(400, {"error": "bad_request", "detail": "body is not UTF-8 JSON"})
        if not isinstance(body, dict):
            return self._reply(400, {"error": "bad_request", "detail": "body is not a JSON object"})
        status, obj = self.server.service.handle("POST", self.path, body)
        self._reply(status, obj)


class _Server(socketserver.UnixStreamServer):
    """Single-threaded on purpose (no ThreadingMixIn): one request at a time, one log writer."""

    allow_reuse_address = False
    timeout = 0.5  # handle_request returns after this, so the loop sees `stopping`


def prepare_socket_dir(socket_dir: Path) -> Path:
    """Create socket_dir 0700, or accept an existing one only when it is a real directory owned by this uid with
 no group/other bits. Return the socket path (a stale socket file of a dead server is removed)."""
    socket_dir = Path(socket_dir)
    if socket_dir.exists() or socket_dir.is_symlink():
        st = os.lstat(socket_dir)
        if not stat.S_ISDIR(st.st_mode):
            raise ServeRefused(f"{socket_dir} is not a directory")
        if st.st_uid != os.getuid():
            raise ServeRefused(f"{socket_dir} is not owned by this user")
        if st.st_mode & 0o077:
            raise ServeRefused(f"{socket_dir} has group/other permission bits (needs 0700)")
    else:
        os.mkdir(socket_dir, 0o700)
        os.chmod(socket_dir, 0o700)
    path = socket_dir / SOCKET_NAME
    if len(os.fsencode(str(path))) >= 108:
        raise ServeRefused("socket path too long for AF_UNIX (108 bytes)")
    if path.exists() or path.is_symlink():
        if not stat.S_ISSOCK(os.lstat(path).st_mode):
            raise ServeRefused(f"{path} exists and is not a socket")
        path.unlink()
    return path


def serve(config: str | Path, seed: int, socket_dir: str | Path, *, force: bool = False, resume: bool = False,
          ready=None, out=None) -> int:
    """Run the server until POST /v1/shutdown, then seal. Returns 0 sealed, 1 seal refused, 2 start refused."""
    out = out or sys.stdout
    try:
        path = prepare_socket_dir(Path(socket_dir))
        service = VictimService(config, seed, force=force, resume=resume, out=out)
    except ServeRefused as exc:
        print(f"refusing: {exc}", file=out, flush=True)
        return 2
    server = None
    try:
        server = _Server(str(path), _Handler)
        os.chmod(path, 0o600)
        server.service = service
        server.allowed_uid = os.getuid()
        if ready is not None:
            ready(service, server)
        while not service.stopping:
            server.handle_request()
    except BaseException:
        service.abort()
        raise
    finally:
        if server is not None:
            server.server_close()
        try:
            path.unlink()
        except FileNotFoundError:
            pass
    return service.finish()


# ---------------------------------------------------------------------- stdlib client (ping, shutdown)
class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float) -> None:
        super().__init__("localhost", timeout=timeout)
        self._socket_path = socket_path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self._socket_path)
        self.sock = sock


def request(socket_path: str | Path, method: str, path: str, body: dict | None = None,
            timeout: float = 600.0) -> tuple[int, dict]:
    """One HTTP request over the AF_UNIX socket; (status, JSON reply)."""
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Connection": "close"}
    if payload is not None:
        headers.update({"Content-Type": "application/json", "Content-Length": str(len(payload))})
    conn = _UnixHTTPConnection(str(socket_path), timeout)
    try:
        conn.request(method, path, body=payload, headers=headers)
        resp = conn.getresponse()
        return resp.status, json.loads(resp.read().decode("utf-8"))
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.serve.victim_server", description="R3 victim server")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--socket-dir", required=True)
    p.add_argument("--force", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="resume an existing unsealed or staging run directory, never move it aside (the retry start)")
    for name in ("ping", "shutdown"):
        q = sub.add_parser(name)
        q.add_argument("--socket-dir", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "serve":
        return serve(args.config, args.seed, args.socket_dir, force=args.force, resume=args.resume)
    sock = Path(args.socket_dir) / SOCKET_NAME
    try:
        if args.cmd == "ping":
            status, reply = request(sock, "GET", "/v1/ping", timeout=5.0)
            if status == 200 and reply.get("ok") is True:
                print(f"ping: {reply.get('run_id')}")
                return 0
            return 1
        status, reply = request(sock, "POST", "/v1/shutdown", {})
    except (OSError, http.client.HTTPException, ValueError) as exc:
        print(f"{args.cmd}: {type(exc).__name__}", file=sys.stderr)
        return 1
    if status != 200:
        print(f"shutdown: HTTP {status} {reply.get('error')}", file=sys.stderr)
        return 1
    print(f"shutdown: rows={reply.get('rows')} records={reply.get('records')} open_item={reply.get('open_item')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
