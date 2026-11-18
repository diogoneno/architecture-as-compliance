"""AVeriTeC baseline veracity model on the dev gold evidence (Table 4.1 row 5), env-main, CPU.

The checkpoint `pretrained_models/bert_veracity.ckpt` (PyTorch Lightning, keys `model.bert.*`, `model.classifier.*`)
is loaded only through aac.run.checkpoints.load_state_dict (sha256 = the HF LFS oid recorded in
configs/data/expected-extra.sha256, then torch.load(weights_only=True)); the `model.` prefix is stripped and the
state dict goes into BertForSequenceClassification(BertConfig.from_pretrained("bert-base-uncased" @ 86b5e093,
num_labels=4)) with strict=False; the incompatible-keys report is printed and recorded.

Inputs and aggregation are TRANSCRIBED from the baseline's own prediction script (SCRIPT_URL @ 2ca9dee2, sha256
SCRIPT_SHA256, fetched 2026-10-16), line numbers below refer to that file:

 LABEL order lines 10-15 (equal to aac.data.fixtures.LABELS)
 quadruple_to_string lines 46-59 "[CLAIM] " + claim + " [QUESTION] " + question + " " + answer [+ ", because "
 + bool_explanation.lower.strip when non-empty]
 prediction call lines 117-122 bool_explanation is always "" at prediction time
 tokenisation lines 25-41, 131: BertTokenizer, max_length=400, padding "longest", truncation
 model call lines 132-137 input_ids and attention_mask only; argmax over the logits per QA string
 empty evidence lines 124-128 "just output NEI" (the script then skips the claim in its output file;
 here it is scored as NEI and counted in n_zero_qa)
 aggregation lines 139-155 any stance 2/3 -> NEI; only 0s -> Supported; only 1s -> Refuted; else
 Conflicting

Gold evidence: one string per gold (question, answer) pair of dev.json; a question with several answers gives one
string per answer. Whether the paper's gold-evidence figure used exactly this per-QA protocol is unverified
(A7); the run records the protocol so the analyst can compare it with the paper."""
from __future__ import annotations

import json
from pathlib import Path

from aac.baselines.external import BaselineFailed

SCRIPT_URL = ("https://huggingface.co/chenxwh/AVeriTeC/raw/2ca9dee23a2a6fa64c5bd918e0cd28ed0aa09031/"
              "src/prediction/veracity_prediction.py")
SCRIPT_SHA256 = "9393d81776104edaf810415ff029ec1bf2263df123c29869ce837d605c2b63cb"
LABEL = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking")  # lines 10-15
NEI, CONFLICTING = 2, 3
MAX_LENGTH = 400              # tokenize_strings(max_length=400), line 29
PADDING = "longest"           # pad_to_max_length=False -> "longest", lines 30, 36
BOOL_EXPLANATION_AT_PREDICTION = ""   # line 121: quadruple_to_string(claim, question, answer, "")
CHECKPOINT_PREFIX = "model."  # SequenceClassificationModule keeps the HF model as `self.model`
NUM_LABELS = 4
DEV_JSON = "averitec/data/dev.json"   # relative to data/raw; its sha256 is in data/raw/MANIFEST.expected.sha256
SPLIT = "dev"


def quadruple_to_string(claim: str, question: str, answer: str, bool_explanation: str | None = "") -> str:
    """Transcribed from veracity_prediction.py lines 46-59."""
    if bool_explanation is not None and len(bool_explanation) > 0:
        bool_explanation = ", because " + bool_explanation.lower().strip()
    else:
        bool_explanation = ""
    return "[CLAIM] " + claim.strip() + " [QUESTION] " + question.strip() + " " + answer.strip() + bool_explanation


def aggregate(stances: list[int]) -> int:
    """Claim label index from the per-QA argmax stances: lines 139-155; an empty list -> NEI (lines 124-128)."""
    if len(stances) == 0:
        return NEI
    has_unanswerable = has_true = has_false = False
    for v in stances:
        if v == 0:
            has_true = True
        if v == 1:
            has_false = True
        if v in (2, 3):
            has_unanswerable = True
    if has_unanswerable:
        return NEI
    if has_true and not has_false:
        return 0
    if not has_true and has_false:
        return 1
    return CONFLICTING


def strip_prefix(state_dict: dict) -> dict:
    """Keep only `model.*` keys, with the prefix removed (the Lightning module's other entries are dropped)."""
    return {k[len(CHECKPOINT_PREFIX):]: v for k, v in state_dict.items() if k.startswith(CHECKPOINT_PREFIX)}


def lightning_state_dict(ckpt: dict) -> dict:
    if not isinstance(ckpt, dict) or "state_dict" not in ckpt:
        raise BaselineFailed("bert_veracity.ckpt holds no `state_dict` entry")
    sd = strip_prefix(ckpt["state_dict"])
    if not sd:
        raise BaselineFailed("bert_veracity.ckpt state_dict holds no `model.` keys")
    return sd


def qa_strings(claim: dict) -> list[str]:
    """One input string per gold (question, answer) pair, built as the script builds them at prediction time."""
    out = []
    for q in claim.get("questions") or []:
        for a in q.get("answers") or []:
            out.append(quadruple_to_string(claim["claim"], q["question"], a.get("answer") or "",
                                           BOOL_EXPLANATION_AT_PREDICTION))
    return out


def four_way_metrics(y: list[int], pred: list[int]) -> dict:
    """Four-way macro-F1 (the published metric), accuracy, per-class F1 and the confusion matrix (rows = gold), plus
 the binary macro-F1 with Refuted (index 1) as the positive class."""
    from sklearn.metrics import confusion_matrix, f1_score

    labels = list(range(NUM_LABELS))
    yb = [1 if v == 1 else 0 for v in y]
    pb = [1 if v == 1 else 0 for v in pred]
    return {
        "macro_f1_4way": float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        "accuracy_4way": sum(1 for a, b in zip(y, pred) if a == b) / len(y),
        "f1_per_class": [float(v) for v in f1_score(y, pred, average=None, labels=labels, zero_division=0)],
        "confusion_4way": confusion_matrix(y, pred, labels=labels).tolist(),
        "macro_f1_binary": float(f1_score(yb, pb, average="macro", labels=[0, 1], zero_division=0)),
    }


def _dev_claims(root: Path) -> tuple[list[dict], str]:
    from aac.data.verify import load_expected
    from aac.run.checkpoints import sha256_file

    path = Path(root) / "data" / "raw" / DEV_JSON
    expected = load_expected(root).get(DEV_JSON)
    got = sha256_file(path)
    if got != expected:
        raise BaselineFailed(f"{DEV_JSON}: sha256 {got} != recorded {expected}")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh), got


def run_averitec(root: Path, resolved: dict) -> dict:
    """Score the 500 dev claims with bert_veracity.ckpt on their gold QA pairs; text is held in memory only."""
    import torch
    from transformers import BertTokenizer

    from aac.baselines.bodega import bert_with_checkpoint, hf_cache_dir
    from aac.data.fixtures import LABELS

    if tuple(LABEL) != tuple(LABELS):
        raise BaselineFailed("the script's label order differs from aac.data.fixtures.LABELS")
    ext = resolved["external"]
    torch.manual_seed(int(resolved["seed"]))
    torch.set_num_threads(int(ext["threads"]))
    claims, dev_sha = _dev_claims(root)
    model, report = bert_with_checkpoint(root, ext, NUM_LABELS, transform=lightning_state_dict)
    model.eval()
    tok = BertTokenizer.from_pretrained(ext["base_model"], revision=ext["base_revision"], local_files_only=True,
                                        cache_dir=hf_cache_dir(root))
    rows, y, pred = [], [], []
    n_zero = n_strings = 0
    for idx, claim in enumerate(claims):
        strings = qa_strings(claim)
        if strings:
            enc = tok(strings, max_length=MAX_LENGTH, padding=PADDING, truncation=True, return_tensors="pt")
            with torch.inference_mode():
                logits = model(enc["input_ids"], attention_mask=enc["attention_mask"]).logits
            stances = torch.argmax(logits, axis=1).tolist()
        else:
            stances = []
            n_zero += 1
        n_strings += len(strings)
        p = aggregate(stances)
        yi = LABEL.index(claim["label"])
        y.append(yi)
        pred.append(p)
        rows.append({"claim_id": f"averitec/{SPLIT}/{idx}", "y_index": yi, "pred_index": p, "n_qa": len(strings)})
    m = four_way_metrics(y, pred)
    result = {
        "n_claims": len(claims), "n_zero_qa": n_zero, "n_qa_strings": n_strings, **m,
        "binary_positive_label": LABEL[1], "label_order": list(LABEL),
        "script_url": SCRIPT_URL, "script_sha256": SCRIPT_SHA256, "max_length": MAX_LENGTH, "padding": PADDING,
        "qa_unit": "one_string_per_gold_question_answer_pair", "bool_explanation_used": False,
        "zero_qa_rule": "NEI", "aggregation": "any_2_or_3:NEI/only_0:Supported/only_1:Refuted/else:Conflicting",
        "dev_json_sha256": dev_sha, "checkpoint_sha256": ext["checkpoint_sha256"],
        "base_model": ext["base_model"], "base_revision": ext["base_revision"], "incompatible_keys": report,
        "torch_threads": int(torch.get_num_threads()),
    }
    return {"result": result, "items": rows, "n": len(claims),
            "headline": f"macro-F1 4-way {m['macro_f1_4way']:.4f} binary {m['macro_f1_binary']:.4f} n={len(claims)},"}
