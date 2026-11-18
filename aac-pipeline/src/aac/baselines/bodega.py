"""BODEGA baselines (Table 4.1 rows 1-4): the organisers' InCrediblAE BERT victims evaluated
with BODEGA's UNCHANGED scripts in env-attack (scripts/attack_env.sh, `env -i` allowlist) on the CPU host's CPU.

 run_test_victims runs/test_victims.py {task} BERT {staging}/preds.txt with HOME = {staging}/home (the script reads
 Path.home/data/BODEGA/{task}/{train,attack}.tsv and BERT-512.pth); prints Accuracy and the
 positive-class F1 2TP/(2TP+FP+FN) (victims/transformer.py:135-136); writes one argmax per
 attack.tsv row to preds.txt (transformer.py:137-140). Rows 1-2.
 run_attack runs/attack.py {task} {true|false} BERTattack BERT {data} {data}/BERT-512.pth {out}, argv[2] from
 external.targeted: true = targeted (label-1 items the victim already gets right,
 attack.py:110-111), false = untargeted (every attack.tsv row; like-for-like with the CheckThat! 2024 Task 6 overview Table 5). NO query budget either way
 (the published protocol; attack.py builds AttackEval without invoke_limit). Rows 3-4.

Inputs are sha256-checked against data/raw/MANIFEST.expected.sha256 before they are symlinked into the private
staging directory (aac.baselines.external.private_staging), so BODEGA's own torch.load (torch 2.14, default
weights_only=True) only ever sees a verified file. stdout and stderr go to two 0600 files in the staging directory,
are hashed into result.json and deleted with it; raw_*.tsv and the VictimCache pickle are written there too and
counted (never copied) before the deletion. Nothing printed by BODEGA is echoed except its parsed summary numbers and
exception class names."""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from aac.baselines.external import BaselineFailed, private_staging
from aac.run import hashing

RAW_DIR = Path("data/raw")
STAGED_FILES = ("train.tsv", "dev.tsv", "attack.tsv", "BERT-512.pth")
TEST_VICTIMS = Path("third_party/BODEGA/runs/test_victims.py")
ATTACK_PY = Path("third_party/BODEGA/runs/attack.py")
ATTACK_ENV = Path("scripts/attack_env.sh")
ATTACK_LOCK = Path("attacks/attack-cpu.lock.txt")   # env-attack's hash-pinned lock (its sha256 goes into result.json)
OPENATTACK_CWD = Path(".hf/openattack")   # OpenAttack's DataManager resolves its assets at $CWD/data/<name>
NUM_LABELS = 2
POSITIVE_LABEL = 1

TEST_VICTIMS_KEYS = (("Accuracy", "accuracy", float), ("F1", "f1_positive", float))
ATTACK_KEYS = (
    ("Subset size", "subset_size", int),
    ("Success score", "success_score", float),
    ("Semantic score", "semantic_score", float),
    ("Character score", "character_score", float),
    ("BODEGA score", "bodega_score", float),
    ("Queries per example", "queries_per_example", float),
)
_ERROR_CLASS = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception|Exit|Interrupt|Violation))\b")


# ------------------------------------------------------------------------------------------- parsers


def _number(text: str, cast):
    v = cast(text)
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _parse_keys(text: str, keys, what: str) -> dict:
    """The LAST `<Key>: <value>` line for each key (BODEGA prints `Subset size` twice; texts printed earlier by
 visualize=True cannot shadow the final summary). A missing line raises ValueError naming it."""
    found: dict[str, str] = {}
    for line in text.replace("\r", "\n").split("\n"):
        for label, _name, _cast in keys:
            if line.startswith(label + ": "):
                found[label] = line[len(label) + 2:].strip()
    out = {}
    for label, name, cast in keys:
        if label not in found:
            raise ValueError(f"{what}: no '{label}: ' line in the output")
        try:
            out[name] = _number(found[label], cast)
        except ValueError as exc:
            raise ValueError(f"{what}: '{label}' is not a {cast.__name__}") from exc
    return out


def parse_test_victims_stdout(text: str) -> dict:
    """{accuracy, f1_positive} from runs/test_victims.py (transformer.py:135-136 print format)."""
    return _parse_keys(text, TEST_VICTIMS_KEYS, "test_victims.py")


def parse_attack_stdout(text: str) -> dict:
    """The six summary values of runs/attack.py (lines 172-177; the report file results_*.txt has the same lines)."""
    if "Report found, exiting..." in text:
        raise ValueError("attack.py: 'Report found, exiting...' (the out dir already held a report)")
    return _parse_keys(text, ATTACK_KEYS, "attack.py")


def parse_preds(path: Path, n_items: int) -> list[int]:
    """One 0/1 argmax per attack.tsv row (transformer.py:139-140); a count other than n_items raises."""
    toks = Path(path).read_text(encoding="ascii").split()
    if len(toks) != n_items:
        raise ValueError(f"preds file has {len(toks)} predictions, the items file has {n_items} rows")
    bad = sorted({t for t in toks if t not in ("0", "1")})
    if bad:
        raise ValueError(f"preds file holds non-binary values {bad[:3]}")
    return [int(t) for t in toks]


def count_lines(path: Path) -> int:
    """Number of newline-terminated lines (one per successful example in BODEGA's raw_*.tsv: BODEGAScore.print_example
 writes `score \\t old \\t new \\n` after normalise_for_bert replaced every newline by a space). Nothing is kept."""
    n = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            n += chunk.count(b"\n")
    return n


def check_successes(success_score: float, subset_size: int, raw_success_lines: int) -> int:
    """successes_k = round(success_score x subset_size); raises when the raw-TSV line count differs."""
    k = 0 if subset_size == 0 else int(round(float(success_score) * int(subset_size)))
    if raw_success_lines != k:
        raise ValueError(f"raw TSV lists {raw_success_lines} successes, Success score x Subset size gives {k}")
    return k


# ------------------------------------------------------------------------------------------- metrics


def binary_metrics(y: list[int], pred: list[int]) -> dict:
    """Confusion counts, accuracy, positive-class F1 2TP/(2TP+FP+FN) (BODEGA's printed metric) and the binary
 macro-F1 (sklearn f1_score average='macro', labels [0, 1]); label 1 is the positive class."""
    from sklearn.metrics import f1_score

    if len(y) != len(pred):
        raise ValueError("y and pred differ in length")
    tp = sum(1 for a, b in zip(y, pred) if a == 1 and b == 1)
    fp = sum(1 for a, b in zip(y, pred) if a == 0 and b == 1)
    fn = sum(1 for a, b in zip(y, pred) if a == 1 and b == 0)
    tn = sum(1 for a, b in zip(y, pred) if a == 0 and b == 0)
    denom = 2 * tp + fp + fn
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "n": len(y),
        "accuracy": (tp + tn) / len(y) if y else 0.0,
        "f1_positive": (2 * tp / denom) if denom else 0.0,
        "macro_f1_binary": float(f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0)),
    }


# ------------------------------------------------------------------------------------------- checkpoint report


def incompatible_keys(state_dict: dict, model) -> dict:
    """{missing_keys, unexpected_keys}, each sorted, from model.load_state_dict(state_dict, strict=False)."""
    rep = model.load_state_dict(state_dict, strict=False)
    return {"missing_keys": sorted(rep.missing_keys), "unexpected_keys": sorted(rep.unexpected_keys)}



# lacks (a buffer older BERT state dicts saved); every sealed BASE run reports exactly this one.
ALLOWED_UNEXPECTED_KEYS = frozenset({"bert.embeddings.position_ids"})


def require_compatible(report: dict, what: str) -> None:
    """Raise BaselineFailed unless no key is missing and every unexpected key is allowlisted: a dropped
 classifier.weight or encoder layer would otherwise score a Table 4.1 row with a randomly initialised head."""
    missing = list(report["missing_keys"])
    extra = sorted(set(report["unexpected_keys"]) - ALLOWED_UNEXPECTED_KEYS)
    if missing or extra:
        raise BaselineFailed(f"{what}: checkpoint does not fit the model (missing_keys={missing}, unexpected keys "
                             f"outside {sorted(ALLOWED_UNEXPECTED_KEYS)}: {extra}); refusing to score with "
                             "randomly initialised weights")


def hf_cache_dir(root: Path) -> str:
    return str(Path(os.environ.get("HF_HOME") or Path(root) / ".hf") / "hub")


def bert_with_checkpoint(root: Path, ext: dict, num_labels: int, transform=None):
    """(model, report): BertForSequenceClassification(BertConfig.from_pretrained(base_model @ base_revision,
 num_labels)) loaded with the hash-gated state dict (aac.run.checkpoints.load_state_dict, weights_only=True)
 under strict=False in env-main; the incompatible-keys report is printed and returned , and any missing
 key or non-allowlisted unexpected key raises BaselineFailed ."""
    os.environ.setdefault("HF_HOME", str(Path(root) / ".hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import BertConfig, BertForSequenceClassification
    from transformers.utils import logging as hf_logging

    from aac.run.checkpoints import load_state_dict

    hf_logging.set_verbosity_error()
    sd = load_state_dict(Path(root) / RAW_DIR / ext["checkpoint_path"], ext["checkpoint_sha256"])
    if transform is not None:
        sd = transform(sd)
    config = BertConfig.from_pretrained(ext["base_model"], revision=ext["base_revision"], num_labels=num_labels,
                                        local_files_only=True, cache_dir=hf_cache_dir(root))
    model = BertForSequenceClassification(config)
    report = incompatible_keys(sd, model)
    print(f"incompatible keys ({ext['checkpoint_path']}): missing_keys={report['missing_keys']} "
          f"unexpected_keys={report['unexpected_keys']}")
    require_compatible(report, ext["checkpoint_path"])
    return model, report


def checkpoint_report(root: Path, ext: dict) -> dict:
    model, report = bert_with_checkpoint(root, ext, NUM_LABELS)
    del model
    return report


# ------------------------------------------------------------------------------------------- staging and launch


def load_items(root: Path, rel: str) -> list[dict]:
    """The committed text-free item file (aac.data.incrediblae): rows 0..n-1 in order, each {row, item_id, label}."""
    items = sorted(json.loads((Path(root) / rel).read_text(encoding="utf-8")), key=lambda r: int(r["row"]))
    if [int(r["row"]) for r in items] != list(range(len(items))):
        raise BaselineFailed(f"{rel}: rows are not 0..{len(items) - 1}")
    return items


def stage_inputs(root: Path, task: str, dest: Path, expected: dict[str, str] | None = None) -> dict[str, str]:
    """Hash-check data/raw/incrediblae/{task}/{train,dev,attack}.tsv and BERT-512.pth against the recorded manifest,
 then symlink each verified file into dest. Returns {file: sha256}."""
    from aac.data.verify import load_expected
    from aac.run.checkpoints import ChecksumMismatch, sha256_file

    expected = expected if expected is not None else load_expected(root)
    out = {}
    for name in STAGED_FILES:
        rel = f"incrediblae/{task}/{name}"
        src = (Path(root) / RAW_DIR / rel).resolve()
        if rel not in expected:
            raise BaselineFailed(f"{rel}: no recorded sha256 in data/raw/MANIFEST.expected.sha256")
        got = sha256_file(src)
        if got != expected[rel]:
            raise ChecksumMismatch(f"{rel}: sha256 {got} != recorded {expected[rel]}; not staged")
        (Path(dest) / name).symlink_to(src)
        out[name] = got
    return out


def restrict_rows(staged: Path, row_range, n_rows_expected: int) -> dict:
    """external.row_range : replace the staged attack.tsv symlink by a 0600 copy that
 holds only data rows [start, stop) of the verified file, byte for byte and in file order. attack.tsv has no header
 line (victims/transformer.py:44-46 parses every line's first field as int), so every line must start with a 0/1
 label field and end with a newline; the file must hold n_rows_expected rows (the committed item file). Only the
 label field is inspected, never a text. Returns {rows_total, rows_staged, row_range, staged_sha256}."""
    start, stop = int(row_range[0]), int(row_range[1])
    staged = Path(staged)
    src = staged.resolve()
    lines = src.read_bytes().split(b"\n")
    if lines[-1] != b"":
        raise BaselineFailed("attack.tsv does not end with a newline; refusing to cut rows")
    lines = lines[:-1]
    if len(lines) != n_rows_expected:
        raise BaselineFailed(f"attack.tsv has {len(lines)} rows, the item file {n_rows_expected}")
    for i, line in enumerate(lines):
        if line.split(b"\t", 1)[0] not in (b"0", b"1"):
            raise BaselineFailed(f"attack.tsv line {i + 1}: first field is not a 0/1 label (a header?); refusing")
    if not 0 <= start < stop <= len(lines):
        raise BaselineFailed(f"external.row_range {[start, stop]} is not inside [0, {len(lines)})")
    body = b"".join(line + b"\n" for line in lines[start:stop])
    tmp = staged.with_name(staged.name + ".part")
    with _private_file(tmp) as fh:
        fh.write(body)
    staged.unlink()
    tmp.rename(staged)
    return {"rows_total": len(lines), "rows_staged": stop - start, "row_range": [start, stop],
            "staged_sha256": hashing.sha256_file(staged)}


def _private_file(path: Path):
    return os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb")


def error_classes(stderr_text: str) -> list[str]:
    """Exception class names at line starts (never the message, which may quote a text)."""
    return sorted({m.group(1) for line in stderr_text.splitlines() if (m := _ERROR_CLASS.match(line.strip()))})


def launch(root: Path, argv: list[str], staging: Path, home: Path, threads: int) -> dict:
    """Run argv (scripts/attack_env.sh python ...) from .hf/openattack with AAC_ATTACK_HOME=home, AAC_THREADS and no
 GPU; stdout/stderr only to staging/{stdout,stderr}.log (0600). Returns {rc, seconds, stdout, stdout_sha256,
 stderr_sha256, error_classes}; the logs stay in staging (deleted with it)."""
    env = dict(os.environ)
    env.update({"AAC_ATTACK_HOME": str(home), "AAC_THREADS": str(int(threads)), "AAC_GPU": ""})
    out_p, err_p = Path(staging) / "stdout.log", Path(staging) / "stderr.log"
    t0 = time.perf_counter()
    with _private_file(out_p) as fo, _private_file(err_p) as fe:
        rc = subprocess.run(argv, cwd=str(Path(root) / OPENATTACK_CWD), env=env, stdin=subprocess.DEVNULL,
                            stdout=fo, stderr=fe, check=False).returncode
    seconds = time.perf_counter() - t0
    stderr_text = err_p.read_text(encoding="utf-8", errors="replace")
    return {"rc": rc, "seconds": seconds, "stdout": out_p.read_text(encoding="utf-8", errors="replace"),
            "stdout_sha256": hashing.sha256_file(out_p), "stderr_sha256": hashing.sha256_file(err_p),
            "error_classes": error_classes(stderr_text)}


def bodega_test_victims_argv(root: Path, task: str, victim: str, preds: Path) -> list[str]:
    return [str(Path(root) / ATTACK_ENV), "python", str(Path(root) / TEST_VICTIMS), task, victim, str(preds)]


def attack_argv(root: Path, task: str, attack: str, victim: str, data_dir: Path, out_dir: Path,
                targeted: bool = True) -> list[str]:
    """attack.py's own argument list (attack.py:39-47): task, targeted ("true"/"false"; attack.py:41 compares
 argv[2].lower with 'true'), attack, victim, data path, model path, out dir. Seven arguments, no budget: the
 published protocol is unbudgeted."""
    return [str(Path(root) / ATTACK_ENV), "python", str(Path(root) / ATTACK_PY), task,
            "true" if targeted else "false", attack, victim,
            str(data_dir), str(Path(data_dir) / "BERT-512.pth"), str(out_dir)]


def report_name(task: str, targeted: bool, attack: str, victim: str) -> str:
    return f"results_{task}_{targeted}_{attack}_{victim}.txt"     # attack.py:49


def raw_name(task: str, targeted: bool, attack: str, victim: str) -> str:
    return f"raw_{task}_{targeted}_{attack}_{victim}.tsv"         # attack.py:146


def assert_fresh_out_dir(out_dir: Path, task: str, targeted: bool, attack: str, victim: str) -> None:
    """attack.py exits early ('Report found, exiting...') when out_dir holds its report: every run gets a fresh,
 empty out dir, so a report or any other file there is refused before launch."""
    out_dir = Path(out_dir)
    if (out_dir / report_name(task, targeted, attack, victim)).exists():
        raise BaselineFailed(f"refusing: {out_dir} already holds an attack.py report")
    if any(out_dir.iterdir()):
        raise BaselineFailed(f"refusing: {out_dir} is not empty")


def _items_key(task: str) -> str:
    return {"FC": "fc_items", "RD": "rd_items"}[task]


# ------------------------------------------------------------------------------------------- runners


def run_test_victims(root: Path, resolved: dict) -> dict:
    """Table 4.1 rows 1-2: BODEGA's test_victims.py on the organisers' victim; metrics recomputed from preds.txt."""
    ext, data = resolved["external"], resolved["data"]
    task, victim = ext["task"], ext["victim"]
    items = load_items(root, data[_items_key(task)])
    report = checkpoint_report(root, ext)
    with private_staging() as st:
        home = st / "home"
        task_dir = home / "data" / "BODEGA" / task
        task_dir.mkdir(parents=True)
        shas = stage_inputs(root, task, task_dir)
        if shas["BERT-512.pth"] != ext["checkpoint_sha256"]:
            raise BaselineFailed("the staged BERT-512.pth is not the checkpoint named in external.checkpoint_path")
        preds_path = st / "preds.txt"
        res = launch(root, bodega_test_victims_argv(root, task, victim, preds_path), st, home, ext["threads"])
        if res["rc"] != 0:
            raise BaselineFailed(f"test_victims.py exited {res['rc']} (exception classes {res['error_classes']})")
        try:
            printed = parse_test_victims_stdout(res["stdout"])
            preds = parse_preds(preds_path, len(items))
        except ValueError as exc:
            raise BaselineFailed(str(exc)) from exc
        preds_sha = hashing.sha256_file(preds_path)
    y = [int(it["label"]) for it in items]
    m = binary_metrics(y, preds)
    if abs(m["f1_positive"] - printed["f1_positive"]) > 1e-9 or abs(m["accuracy"] - printed["accuracy"]) > 1e-9:
        raise BaselineFailed(f"printed F1/accuracy {printed} differ from the values recomputed from preds.txt "
                             f"({m['f1_positive']}, {m['accuracy']}): preds are not aligned with the items")
    result = {
        "task": task, "victim": victim, "positive_label": POSITIVE_LABEL,
        "n_items": len(items), "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
        "accuracy": m["accuracy"], "f1_positive": m["f1_positive"], "macro_f1_binary": m["macro_f1_binary"],
        "printed": {"accuracy": printed["accuracy"], "f1_positive": printed["f1_positive"]},
        "metric_printed_by_bodega": "f1_positive",
        "preds_sha256": preds_sha, "stdout_sha256": res["stdout_sha256"], "stderr_sha256": res["stderr_sha256"],
        "input_sha256": shas, "checkpoint_sha256": ext["checkpoint_sha256"], "bodega_commit": ext["bodega_commit"],
        "script": TEST_VICTIMS.as_posix(), "incompatible_keys": report,
        "env_attack_lock_sha256": hashing.sha256_file(Path(root) / ATTACK_LOCK),
    }
    rows = [{"row": int(it["row"]), "item_id": it["item_id"], "y": yy, "pred": pp}
            for it, yy, pp in zip(items, y, preds)]
    return {"result": result, "items": rows, "n": len(items),
            "timings": [{"stage": "test_victims.py", "seconds": round(res["seconds"], 3), "items": len(items)}],
            "headline": f"F1 printed {printed['f1_positive']:.4f} recomputed {m['f1_positive']:.4f} "
                        f"macro-F1 {m['macro_f1_binary']:.4f} n={len(items)},"}


def run_attack(root: Path, resolved: dict) -> dict:
    """Table 4.1 rows 3-4: BODEGA's attack.py, targeted or untargeted (external.targeted), unbudgeted; successes
 cross-checked against raw_*.tsv. external.row_range [start, stop) stages only those attack.tsv rows
 (restrict_rows); n_items and n_label1 then count the staged rows, result.json records row_range and the staged
 file's sha256, and an untargeted run must attack exactly the staged rows."""
    from aac.attacks.assets import verify_runtime_assets

    ext, data = resolved["external"], resolved["data"]
    task, attack, victim, targeted = ext["task"], ext["attack"], ext["victim"], bool(ext["targeted"])
    row_range = ext.get("row_range")
    items_all = load_items(root, data[_items_key(task)])
    items = items_all if row_range is None else items_all[int(row_range[0]):int(row_range[1])]
    report = checkpoint_report(root, ext)
    _ok, bad = verify_runtime_assets(root)
    if bad:
        raise BaselineFailed(f"runtime assets failed their pins: {bad}")
    with private_staging() as st:
        home, task_dir, out_dir = st / "home", st / "data" / task, st / "out"
        for d in (home, task_dir, out_dir):
            d.mkdir(parents=True)
        shas = stage_inputs(root, task, task_dir)
        if shas["BERT-512.pth"] != ext["checkpoint_sha256"]:
            raise BaselineFailed("the staged BERT-512.pth is not the checkpoint named in external.checkpoint_path")
        cut = None if row_range is None else restrict_rows(task_dir / "attack.tsv", row_range, len(items_all))
        assert_fresh_out_dir(out_dir, task, targeted, attack, victim)
        if list(task_dir.glob("*-cache.pickle")):
            raise BaselineFailed("a VictimCache pickle exists before launch")
        argv = attack_argv(root, task, attack, victim, task_dir, out_dir, targeted)
        res = launch(root, argv, st, home, ext["threads"])
        if res["rc"] != 0:
            raise BaselineFailed(f"attack.py exited {res['rc']} (exception classes {res['error_classes']})")
        rep_path = out_dir / report_name(task, targeted, attack, victim)
        raw_path = out_dir / raw_name(task, targeted, attack, victim)
        try:
            parsed = parse_attack_stdout(res["stdout"])
            if not rep_path.is_file():
                raise ValueError("attack.py wrote no report file")
            from_report = parse_attack_stdout(rep_path.read_text(encoding="utf-8", errors="replace"))
            if from_report != parsed:
                raise ValueError("the stdout summary and the report file disagree")
            raw_lines = count_lines(raw_path) if raw_path.is_file() else 0
            k = check_successes(parsed["success_score"], parsed["subset_size"], raw_lines)
            if cut is not None and not targeted and int(parsed["subset_size"]) != cut["rows_staged"]:
                raise ValueError(f"untargeted attack.py attacked {parsed['subset_size']} items, "
                                 f"{cut['rows_staged']} rows were staged")
        except ValueError as exc:
            raise BaselineFailed(str(exc)) from exc
        n_pickles = len(list(task_dir.glob("*-cache.pickle")))
    n_label1 = sum(1 for it in items if int(it["label"]) == POSITIVE_LABEL)
    result = {
        "task": task, "attack": attack, "victim": victim, "targeted": targeted, "budget": None,
        "n_items": len(items), "n_label1": n_label1, **parsed, "successes_k": k,
        "row_range": None if cut is None else cut["row_range"],
        "staged_attack_sha256": None if cut is None else cut["staged_sha256"],
        "raw_success_lines": raw_lines, "victim_cache_pickles_deleted": n_pickles,
        "stdout_sha256": res["stdout_sha256"], "stderr_sha256": res["stderr_sha256"],
        "input_sha256": shas, "checkpoint_sha256": ext["checkpoint_sha256"], "bodega_commit": ext["bodega_commit"],
        "script": ATTACK_PY.as_posix(), "incompatible_keys": report,
        "env_attack_lock_sha256": hashing.sha256_file(Path(root) / ATTACK_LOCK),
    }
    return {"result": result, "items": [], "n": int(parsed["subset_size"]),
            "timings": [{"stage": "attack.py", "seconds": round(res["seconds"], 3), "items": int(parsed["subset_size"])}],
            "headline": f"ASR {k}/{parsed['subset_size']} BODEGA {parsed['bodega_score']} "
                        f"queries/example {parsed['queries_per_example']},"}
