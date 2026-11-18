"""BODEGA score of attacked R3 rows with BODEGA's own code (4).

Runs in env-attack (CPython 3.10), launched only through scripts/attack_env.sh by `aac bodega-score`
(aac.attacks.bodega):

 scripts/attack_env.sh .venv-attack/bin/python -m attacks.scoring.bodega_score --in <staging>/in.jsonl \
 --out <staging>/out.jsonl --summary <staging>/summary.json

Imports: the standard library, numpy, editdistance, OpenAttack and BODEGA's `metrics` package (PYTHONPATH =
third_party/BODEGA, set by attack_env.sh); never `aac`.

Input (one JSON line per attacked ledger row, 0600 in a 0700 staging directory): item_id, attack, victim, subset,
source_run_id, task, original, adversarial, success_strict. Per (source_run_id, attack, victim, subset) group one
BODEGAScore pass (BODEGA's `compute`, so BODEGA's aggregate is the reference):
 task shim TASK_SHIM maps the subset to BODEGA's task id: averitec_chrono -> PR2 (whole string, one segment),
 fc -> FC (claim-only scope: one segment), rd -> RD (LAMBO sentence alignment). segment_sentences knows
 only PR2 / FC / RD / HN / C19; any other id returns and breaks the averaging, so an unknown subset
 raises
 success strict: a row whose ledger end_state is not `success` (a victim-B deferral included, pre-registered specification)
 enters BODEGA as s2 = None, exactly as attack.py's failed examples do, and scores 0
 per item from the same pass: BODEGA's normalise_for_lev + editdistance (character), the BLEURT-20 similarities
 BODEGA computed (captured, clipped to [0, 1] and averaged over the aligned segments as compute does)
 and BODEGA's own alignment (semantic); bodega = semantic x character
 assertion per group: the means of the per-item success, semantic, character (over successes) and bodega values
 equal BODEGAScore's four aggregates within 1e-9, so the per-item file reproduces the official score
Output: out.jsonl (one line per input line, in input order, ONLY the keys of OUT_KEYS: ids, a boolean, numbers) and
summary.json (per-group aggregates, assertion counts, the verified pins). Both 0600. Stdout carries counts only;
BODEGA's own progress prints are sent to stderr (the staging log). No text is ever printed (raw_path is never set).
"""
if __name__ == "__main__":  # run as a script: re-import under the package name (as openattack_runner does)
    import pathlib as _pathlib
    import sys as _sys
    _sys.path.insert(0, str(_pathlib.Path(__file__).resolve().parents[2]))
    from attacks.scoring.bodega_score import main as _main
    _sys.exit(_main())

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import stat
import subprocess
import sys
from pathlib import Path

AAC_ROOT = Path(__file__).resolve().parents[2]
BODEGA_DIR = AAC_ROOT / "third_party" / "BODEGA"
OPENATTACK_DATA = AAC_ROOT / ".hf" / "openattack" / "data"

TASK_SHIM = {"averitec_chrono": "PR2", "fc": "FC", "rd": "RD"}
ALIGN = {"PR2": False, "FC": False, "RD": True}   # RD: LAMBO alignment; PR2 and claim-only FC are one segment
IN_KEYS = ("item_id", "attack", "victim", "subset", "source_run_id", "task", "original", "adversarial", "success_strict")
OUT_KEYS = ("item_id", "attack", "victim", "subset", "source_run_id", "success_strict", "semantic", "character", "bodega")
GROUP_KEYS = ("source_run_id", "attack", "victim", "subset")
TOL = 1e-9


BODEGA_COMMIT = "2dda5a6e6d8a291327df79a44767efea8cab556f"
BODEGA_FILES_SHA256 = {
    "metrics/BODEGAScore.py": "d1494f6e2ff63416379b9659e9413f39af49afd7d475ffd6350981a3370e068a",
    "metrics/ScorePromise.py": "5e5fcf0505dab481cee302ef925161bfec6e78326d5fdb2fcc399cb62e3a7d86",
}
BLEURT_ID = "lucadiliello/BLEURT-20"
BLEURT_REVISION = "e22b9eb071dcb939da9fff08b20a7e746c72a830"
VCS_PINS = {"bleurt_pytorch": "279ca1bb4106bde5a89f0f82723197e23d8446cb",
            "lambo": "c699c6d07fc4b75d2baf758181c4e796220df238"}


class ScoringError(RuntimeError):
    """The input, a pin or the aggregate assertion does not check out (no text is ever part of the message)."""


def task_for(subset: str) -> str:
    """BODEGA's task id for an R3 subset (the runtime shim); an unknown subset raises."""
    try:
        return TASK_SHIM[subset]
    except KeyError:
        raise ScoringError(f"no BODEGA task for subset {subset!r} (TASK_SHIM covers {sorted(TASK_SHIM)})") from None


# --------------------------------------------------------------------------- pins

def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_pins(hf_home: Path | None = None) -> dict:
    """The verified pins: BODEGA's metric files by content, the HF cache's BLEURT-20 revision (refs/main, the one
 from_pretrained resolves offline) and the bleurt-pytorch / LAMBO commits from their installed dist-info."""
    for rel, want in BODEGA_FILES_SHA256.items():
        got = _sha256(BODEGA_DIR / rel)
        if got != want:
            raise ScoringError(f"third_party/BODEGA/{rel} sha256 {got[:12]} is not the {BODEGA_COMMIT[:8]} file {want[:12]}")
    hf_home = Path(hf_home or os.environ.get("HF_HOME") or (AAC_ROOT / ".hf"))
    ref = hf_home / "hub" / ("models--" + BLEURT_ID.replace("/", "--")) / "refs" / "main"
    rev = ref.read_text(encoding="ascii").strip() if ref.is_file() else None
    if rev != BLEURT_REVISION:
        raise ScoringError(f"{BLEURT_ID} in {hf_home}/hub resolves to {rev}, not the pinned {BLEURT_REVISION}")
    from importlib import metadata

    vcs = {}
    for dist, want in VCS_PINS.items():
        try:
            raw = metadata.distribution(dist.replace("_", "-")).read_text("direct_url.json")
        except metadata.PackageNotFoundError:
            raw = None
        got = (json.loads(raw).get("vcs_info") or {}).get("commit_id") if raw else None
        if got != want:
            raise ScoringError(f"{dist} is installed at {got}, not the pinned {want}")
        vcs[dist] = got
    checkout = None
    if (BODEGA_DIR / ".git").exists():
        out = subprocess.run(["git", "-C", str(BODEGA_DIR), "rev-parse", "HEAD"], capture_output=True, text=True)
        checkout = out.stdout.strip() if out.returncode == 0 else None
        if checkout is not None and checkout != BODEGA_COMMIT:
            raise ScoringError(f"third_party/BODEGA is checked out at {checkout}, not {BODEGA_COMMIT}")
    return {"bodega_commit": BODEGA_COMMIT, "bodega_checkout": checkout, "bodega_files_sha256": dict(BODEGA_FILES_SHA256),
            "bleurt_revision": rev, "bleurt_pytorch_commit": vcs["bleurt_pytorch"], "lambo_commit": vcs["lambo"]}


# --------------------------------------------------------------------------- input / output

def _private_writer(path: Path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w", encoding="ascii")


def read_input(path: Path) -> list[dict]:
    path = Path(path)
    if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ScoringError(f"{path.name}: the input must be a regular 0600 file")
    rows = []
    for n, line in enumerate(path.read_bytes().split(b"\n"), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            raise ScoringError(f"{path.name} line {n}: not JSON") from None
        if not isinstance(row, dict) or set(row) != set(IN_KEYS):
            raise ScoringError(f"{path.name} line {n}: expected exactly the keys {list(IN_KEYS)}")
        if row["task"] != task_for(row["subset"]):
            raise ScoringError(f"{path.name} line {n}: task {row['task']!r} is not the shim's for {row['subset']!r}")
        if not isinstance(row["original"], str) or not isinstance(row["adversarial"], str):
            raise ScoringError(f"{path.name} line {n}: original and adversarial must be strings")
        if not isinstance(row["success_strict"], bool):
            raise ScoringError(f"{path.name} line {n}: success_strict must be a boolean")
        rows.append(row)
    return rows


def groups_of(rows: list[dict]) -> list[tuple[tuple, list[int]]]:
    """[(group key, [row indices])] in first-seen order."""
    out: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        out.setdefault(tuple(r[k] for k in GROUP_KEYS), []).append(i)
    return list(out.items())


def _num(x):
    """A finite float, or None for NaN (BODEGA's mean over zero successes)."""
    if x is None:
        return None
    v = float(x)
    return None if math.isnan(v) else v


# --------------------------------------------------------------------------- scoring

def build_scorers(tasks, device: str = "cpu") -> dict:
    """{task: BODEGAScore} built as attack.py builds it (semantic_scorer BLEURT), BLEURT-20 loaded once and shared by
 shallow copies; align_sentences per ALIGN."""
    import OpenAttack

    OpenAttack.DataManager.set_path(str(OPENATTACK_DATA))
    from metrics.BODEGAScore import BODEGAScore

    tasks = sorted(set(tasks))
    base = BODEGAScore(device, tasks[0], align_sentences=ALIGN[tasks[0]], semantic_scorer="BLEURT")
    out = {}
    for t in tasks:
        s = copy.copy(base)
        s.task, s.align_sentences, s.promises = t, ALIGN[t], []
        out[t] = s
    return out


def clip_mean(sim, idxs) -> float:
    """BODEGA compute's semantic score of one item: each segment similarity clipped to [0, 1], averaged."""
    total = 0
    for idx in idxs:
        if sim[idx] <= 0:
            total += 0
        elif sim[idx] >= 1:
            total += 1
        else:
            total += sim[idx]
    return total / len(idxs)


def score_group(scorer, rows: list[dict]) -> tuple[list[dict], dict]:
    """(per-item numbers, aggregate) of one group through ONE BODEGAScore.compute pass."""
    import editdistance
    import numpy
    from metrics.ScorePromise import ScorePromise

    scorer.promises = [ScorePromise(r["original"], r["adversarial"] if r["success_strict"] else None, scorer) for r in rows]
    captured = {}
    original_ss = scorer.semantic_similarity

    def capture(ss_sentences):
        sim = original_ss(ss_sentences)
        captured["sim"] = numpy.array(sim)
        return sim

    scorer.semantic_similarity = capture
    try:
        with contextlib.redirect_stdout(sys.stderr):
            agg = scorer.compute()
            alignments = (scorer.align_sentences_greedy() if scorer.align_sentences
                          else [[(p.s1, p.s2)] for p in scorer.promises])
    finally:
        del scorer.semantic_similarity
    sim = captured["sim"]
    guide, n = [], 0
    for al in alignments:
        guide.append(list(range(n, n + len(al))))
        n += len(al)
    if n != len(sim):
        raise ScoringError(f"alignment rebuilt {n} segments, BODEGA scored {len(sim)}")
    items = []
    for p, idxs in zip(scorer.promises, guide):
        if p.s2 is None:
            items.append({"success_strict": False, "semantic": None, "character": None, "bodega": 0.0})
            continue
        n1, n2 = scorer.normalise_for_lev(p.s1), scorer.normalise_for_lev(p.s2)
        character = 1.0 - editdistance.eval(n1, n2) / max(len(n1), len(n2))
        semantic = float(clip_mean(sim, idxs))
        items.append({"success_strict": True, "semantic": semantic, "character": float(character),
                      "bodega": semantic * float(character)})
    scorer.promises = []
    return items, {"success": _num(agg[0]), "semantic": _num(agg[1]), "character": _num(agg[2]), "bodega": _num(agg[3])}


def check_aggregate(items: list[dict], agg: dict, where: str) -> int:
    """Assert the per-item means equal BODEGAScore's aggregates (TOL); returns the number of equalities checked."""
    succ = [it for it in items if it["success_strict"]]
    mine = {"success": len(succ) / len(items), "bodega": sum(it["bodega"] for it in items) / len(items),
            "semantic": (sum(it["semantic"] for it in succ) / len(succ)) if succ else None,
            "character": (sum(it["character"] for it in succ) / len(succ)) if succ else None}
    checked = 0
    for key in ("success", "semantic", "character", "bodega"):
        a, b = mine[key], agg[key]
        if a is None or b is None:
            if a is not b:
                raise ScoringError(f"{where}: {key} per-item mean {a} vs BODEGAScore aggregate {b}")
            continue
        if abs(a - b) > TOL:
            raise ScoringError(f"{where}: {key} per-item mean {a!r} differs from BODEGAScore's aggregate {b!r} by "
                               f"{abs(a - b):.3e} (> {TOL})")
        checked += 1
    return checked


def run(in_path: Path, out_path: Path, summary_path: Path, device: str = "cpu") -> dict:
    pins = verify_pins()
    rows = read_input(in_path)
    if not rows:
        raise ScoringError("no input rows")
    scorers = build_scorers([r["task"] for r in rows], device)
    out_rows: list[dict | None] = [None] * len(rows)
    groups, assertions = [], 0
    for key, idxs in groups_of(rows):
        task = rows[idxs[0]]["task"]
        items, agg = score_group(scorers[task], [rows[i] for i in idxs])
        where = "/".join(str(k) for k in key)
        assertions += check_aggregate(items, agg, where)
        for i, it in zip(idxs, items):
            r = rows[i]
            out_rows[i] = {"item_id": r["item_id"], "attack": r["attack"], "victim": r["victim"], "subset": r["subset"],
                           "source_run_id": r["source_run_id"], **it}
        groups.append({**dict(zip(GROUP_KEYS, key)), "task": task, "align_sentences": ALIGN[task], "n_attacked": len(idxs),
                       "n_success": sum(it["success_strict"] for it in items), "bodegascore_aggregate": agg})
    with _private_writer(out_path) as fh:
        for r in out_rows:
            fh.write(json.dumps({k: r[k] for k in OUT_KEYS}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    summary = {"pins": pins, "device": device, "groups": groups, "rows": len(rows), "assertions": assertions,
               "tolerance": TOL, "task_shim": dict(TASK_SHIM)}
    with _private_writer(summary_path) as fh:
        fh.write(json.dumps(summary, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m attacks.scoring.bodega_score")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu"])
    args = ap.parse_args(argv)
    try:
        s = run(Path(args.inp), Path(args.out), Path(args.summary), args.device)
    except ScoringError as exc:
        print(f"bodega-score: FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"bodega-score: groups={len(s['groups'])} rows={s['rows']} "
          f"successes={sum(g['n_success'] for g in s['groups'])} assertions={s['assertions']} ok")
    return 0

