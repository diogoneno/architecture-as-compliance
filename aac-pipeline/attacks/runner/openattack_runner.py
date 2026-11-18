"""Own OpenAttack runner for R3 (client half of ).

Runs in env-attack (CPython 3.10, OpenAttack 2.1.1), launched only through scripts/attack_env.sh:

 scripts/attack_env.sh .venv-attack/bin/python attacks/runner/openattack_runner.py --socket <dir>/victim.sock \
 [--device cpu|cuda] [--attacks deepwordbug,pwws]

The victim is reached only through the protocol v1 client (protocol.py) on an AF_UNIX socket. Imports: the standard
library, numpy, torch (for torch.device only) and OpenAttack (never `aac`, never BODEGA's victim wrappers, so no pickle-keyed victim cache).

: AacVictim counts every attack-phase text before it sends it; each reply's `queries_used` must equal
that count; per item the count stays <= budget and OpenAttack's reported invocations equal the count (count + 1 when
OpenAttack's limit check refused the next call), so they stay <= budget + 1. Any disagreement raises PipelineFault.

Output: one line per attacked item (`item=... attack=... victim=... runner=... server=... openattack=... outcome=...`)
and one line per cell. No text is ever printed: OpenAttack's logger is silenced at import, because
attack_process logs the whole item (`data`) on any exception.
"""
if __name__ == "__main__":  # run as a script: re-import under the package name so there is one PipelineFault class
    import pathlib as _pathlib
    import sys as _sys
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
    from attacks.runner.openattack_runner import main as _main
    _sys.exit(_main())

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import OpenAttack
import torch
from OpenAttack.exceptions import InvokeLimitExceeded

from attacks.runner import recipes
from attacks.runner.protocol import MAX_TEXTS_PER_QUERY, VictimClient


def _silence_openattack_logging():
    """Drop every OpenAttack log record: attack_process logs the item data (the text) on exceptions."""
    level = logging.CRITICAL + 1
    names = ["OpenAttack", "OpenAttack.AttackEval", "OpenAttack.attack_eval.attack_eval"]
    names += [n for n in logging.root.manager.loggerDict if n.startswith("OpenAttack")]
    for name in names:
        lg = logging.getLogger(name)
        lg.setLevel(level)
        lg.propagate = False
        for h in list(lg.handlers):
            h.setLevel(level)
        if not any(isinstance(h, logging.NullHandler) for h in lg.handlers):
            lg.addHandler(logging.NullHandler())


_silence_openattack_logging()

AAC_ROOT = Path(__file__).resolve().parents[2]
OPENATTACK_HOME = AAC_ROOT / ".hf" / "openattack"
# DataManager fixes its asset paths from os.getcwd at import time, so a later chdir alone is not enough: point
# every asset at .hf/openattack/data explicitly (hash-checked by `python -m aac.attacks.assets` before any run).
OpenAttack.DataManager.set_path(str(OPENATTACK_HOME / "data"))


class PipelineFault(BaseException):
    """A victim-server or counting fault. It aborts the run; it is never an attack outcome.

 It subclasses BaseException on purpose: OpenAttack 2.1.1's attack_process catches every Exception and records
 the item as a failed attack (and logs the item data). A server fault recorded as "attack failed" would enter the
 ledger with a wrong outcome and a wrong count, so the fault must pass through attack_process untouched.
 """


class AacVictim(OpenAttack.Classifier):
    """OpenAttack classifier whose every call is one or more /v1/query requests with phase `attack`."""

    def __init__(self, client):
        self.client = client
        self.cell = None          # (attack, victim, subset) set by the runner per cell
        self.reset_item()

    def reset_item(self):
        self.runner_queries = 0   # attack-phase texts sent for the current item (our count)
        self.server_queries = 0   # last `queries_used` the server reported
        self.rtt_seconds = 0.0    # summed client-side round trips of the item's attack-phase queries
        self.limit_hit = False    # OpenAttack's invoke_limit refused a call for the current item

    def record_invoke(self, cnt, func, *args, **kwargs):
        # OpenAttack raises InvokeLimitExceeded here, before the call that would exceed the limit. Its reported
        # count is then invoke + 1, which can stay <= budget when the refused call is a batch; this flag records
        # the limit hit exactly instead of inferring it from "Query Exceeded".
        try:
            return super().record_invoke(cnt, func, *args, **kwargs)
        except InvokeLimitExceeded:
            self.limit_hit = True
            raise

    def _probs(self, input_):
        ctx = self.context
        if ctx is None or self.cell is None:
            raise PipelineFault("victim called outside an attack context (an uncounted query)")
        data = ctx.input
        attack, victim, subset = self.cell
        item_id = data.get("item_id")
        if data.get("subset") != subset or not isinstance(item_id, str):
            raise PipelineFault("attack context names no item of the current cell")
        rows = []
        for start in range(0, len(input_), MAX_TEXTS_PER_QUERY):
            chunk = list(input_[start:start + MAX_TEXTS_PER_QUERY])
            self.runner_queries += len(chunk)  # counted before sending 
            t0 = time.perf_counter()
            reply = self.client.query(item_id, attack, victim, subset, "attack", chunk)
            self.rtt_seconds += time.perf_counter() - t0
            self.server_queries = reply["queries_used"]
            if self.server_queries != self.runner_queries:
                raise PipelineFault(f"Query counter disagreement: server queries_used={self.server_queries} != runner count="
                                    f"{self.runner_queries} (item {item_id})")
            rows.extend(reply["probs"])
        return np.asarray(rows, dtype=np.float64)

    def get_prob(self, input_):
        return self._probs(input_)

    def get_pred(self, input_):
        return self._probs(input_).argmax(axis=1)


def _attack_item(client, victim, attacker, budget, item):
    """Attack one item; return the /v1/end fields. PipelineFault propagates (no /v1/end is sent)."""
    attack, victim_name, subset = victim.cell
    victim.reset_item()
    t_item = time.perf_counter()
    clean = client.query(item["item_id"], attack, victim_name, subset, "clean", [item["text"]])
    pred = int(np.argmax(clean["probs"][0]))
    if pred != int(item["label"]):
        return dict(outcome="skipped", final_text=None, runner_queries=0, openattack_invocations=0,
                    runner_seconds=time.perf_counter() - t_item, rtt_seconds=0.0, server_queries=0)
    attack_eval = OpenAttack.AttackEval(attacker, victim, language="english", invoke_limit=budget, metrics=[])
    data = {"x": item["text"], "y": int(item["label"]), "item_id": item["item_id"], "subset": subset}
    result = next(iter(attack_eval.ieval([data])))
    invocations = int(result["metrics"]["Victim Model Queries"])
    ours = victim.runner_queries
    if result["result"] is not None:
        outcome, final_text = "success", result["result"]
    elif victim.limit_hit or result["metrics"]["Query Exceeded"]:
        outcome, final_text = "limit", None
    else:
        outcome, final_text = "failed", None
    # per item: ours == server (checked per reply), ours <= budget, OpenAttack's count = ours (+1 on a limit hit).
    expected = ours + 1 if victim.limit_hit else ours
    if ours != victim.server_queries and ours > 0:
        raise PipelineFault(f"Query counter disagreement: runner={ours} server={victim.server_queries}")
    if ours > budget or invocations > budget + 1 or invocations != expected:
        raise PipelineFault(f"Query counter disagreement: runner={ours} openattack={invocations} budget={budget} "
                            f"limit_hit={victim.limit_hit}")
    return dict(outcome=outcome, final_text=final_text, runner_queries=ours, openattack_invocations=invocations,
                runner_seconds=time.perf_counter() - t_item, rtt_seconds=victim.rtt_seconds,
                server_queries=victim.server_queries)


def run(socket_path, *, device, attackers_for=recipes.build, attacks=None, supported=recipes.NAMES, timeout=600.0,
        out=None):
    """Attack every plan cell (optionally only `attacks`); return a list of per-item result dicts.

 Returns normally only when every selected cell finished. PipelineFault aborts the run.
 """
    out = out or sys.stdout
    device = device if isinstance(device, torch.device) else torch.device(device)
    client = VictimClient(socket_path, timeout, fault=PipelineFault)
    client.ping()
    plan = client.plan()
    budget = int(plan["budget"])
    cells = [tuple(c) for c in plan["cells"] if attacks is None or c[0] in attacks]
    unknown = sorted({c[0] for c in cells} - set(supported))
    if unknown:
        raise ValueError(f"attacks not supported by this runner: {', '.join(unknown)}")
    done = {tuple(r) for r in client.done()["done"]}
    victim = AacVictim(client)
    attackers = {}
    results = []
    for attack, victim_name, subset in cells:
        if attack not in attackers:
            attackers[attack] = attackers_for(attack, device)
        victim.cell = (attack, victim_name, subset)
        items = client.items(subset, victim_name)["items"]
        attacked = skipped_done = 0
        for item in items:
            if (item["item_id"], attack, victim_name, subset) in done:
                skipped_done += 1
                continue
            res = _attack_item(client, victim, attackers[attack], budget, item)
            client.end(item_id=item["item_id"], subset=subset, attack=attack, victim=victim_name,
                       outcome=res["outcome"], final_text=res["final_text"], runner_queries=res["runner_queries"],
                       openattack_invocations=res["openattack_invocations"],
                       runner_seconds=res["runner_seconds"], rtt_seconds=res["rtt_seconds"])
            attacked += 1
            print(f"item={item['item_id']} attack={attack} victim={victim_name} runner={res['runner_queries']} "
                  f"server={res['server_queries']} openattack={res['openattack_invocations']} "
                  f"outcome={res['outcome']}", file=out, flush=True)
            results.append(dict(res, item_id=item["item_id"], attack=attack, victim=victim_name, subset=subset))
        victim.cell = None
        print(f"cell attack={attack} victim={victim_name} subset={subset} items={len(items)} attacked={attacked} "
              f"skipped_done={skipped_done}", file=out, flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Own OpenAttack runner (protocol v1 client)")
    ap.add_argument("--socket", required=True)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--attacks", default=None, help="comma-separated ledger names; default: every plan cell")
    ap.add_argument("--timeout", type=float, default=600.0)
    args = ap.parse_args(argv)
    socket_path = str(Path(args.socket).resolve())  # resolve before the chdir below
    attacks = set(args.attacks.split(",")) if args.attacks else None
    os.chdir(OPENATTACK_HOME)  # OpenAttack's DataManager reads $CWD/data/<asset>
    try:
        run(socket_path, device=torch.device(args.device), attacks=attacks, timeout=args.timeout)
    except PipelineFault as exc:
        print(f"PipelineFault: {exc}", file=sys.stderr, flush=True)
        return 3
    except KeyboardInterrupt:
        print("KeyboardInterrupt", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:  # noqa: BLE001 - print the class only: a message may carry a text
        print(f"runner error: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 2
    return 0
