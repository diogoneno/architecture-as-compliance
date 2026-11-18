"""C2 scorer training run kind : weak-labeled (passage, claim) pairs -> mDeBERTa-v3-base with a
three-way SUPPORT / REFUTE / NEUTRAL head, sealed as `C2S-<hash10>-s<seed>`.

The recipe is the committed `scorer:` block of an aac-train-scorer/1 config (configs/train/c2s.yaml = the C1 recipe of
configs/train/c1.yaml with max_length 256); nothing here chooses a value. Order of operations in train_scorer:

 1. preflight, before torch or any model load: the C1 datasheet gate and split-manifest assertions
 (aac.components.c1_train.preflight) and the EV run checks (sealed digest equal to the resolved ref, not a subset
 run, evidence file hash);
 2. build_pairs from train-partition claims only (methodological note):
 gold every answer of the claim; text = the answer, or for a Boolean answer question + answer +
 boolean_explanation joined by single spaces; label = the claim label through pairs.label_map, an
 Unanswerable answer `pairs.unanswerable` (NEUTRAL); an empty text gives no pair (counted);
 retrieved the first pairs.retrieved_top_k passages of the claim's primary re-ranked EV list; text from the SL
 run's private sentence cache (sha256 checked); label = the claim label through label_map when the
 passage's SL row has url_eq_answer 1, else NEUTRAL;
 claims whose label is in pairs.exclude_claim_labels (Conflicting Evidence/Cherrypicking) give no pair; order =
 claim (split rank, idx) order, gold before retrieved, answer order or rank order; every pair's claim must be in
 the train partition and in no other partition (PartitionViolation otherwise);
 3. the C1 loop: seed_everything, strict deterministic algorithms (CUBLAS_WORKSPACE_CONFIG on CUDA), fp32 weights,
 bf16 autocast on CUDA only, AdamW, linear warm-up and decay, grad clip, isfinite on loss and grad norm
 (NonFiniteLoss), fixed epochs, the last weights file; pairs tokenised passage first, truncation "only_first";
 4. checkpoint.safetensors, model.json, train-log.jsonl (step, epoch, loss, grad_norm, lr) and pairs.json (counts and
 the sha256 of the sorted text-free pair list [claim_id, source, ref, label]). No text is written anywhere.

torch and transformers are imported inside functions: importing this module stays cheap."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from aac.components import c1_train
from aac.run.config import SCORER_LABELS
from aac.run.hashing import canonical, claim_hash, derive_seed, dumps_sorted, sha256_bytes

TRAIN_LOG = c1_train.TRAIN_LOG
CHECKPOINT = c1_train.CHECKPOINT
MODEL_JSON = c1_train.MODEL_JSON
PAIRS_JSON = "pairs.json"
PAIRS_SCHEMA = "aac-scorer-pairs/1"
PAIR_FIELDS = ("claim_id", "source", "ref", "label")
SOURCES = ("gold", "retrieved")
EV_VARIANT = "primary"           # the scorer trains on the primary re-ranked list (the inference list of R1 primary)
SL_FLAG_COL = 3                  # SL row [doc_idx, sent_idx, score6, url_eq_answer, url_date_after_claim]
UNANSWERABLE = "Unanswerable"
BOOLEAN = "Boolean"


class ScorerDataError(ValueError):
    """An EV/SL input fails a hash, subset or coverage check, or a label is inconsistent; nothing is trained."""


class PartitionViolation(c1_train.ManifestViolation):
    """A pair's claim is outside the train partition, or also listed in val_fuse, val_cal or test."""


# ---------------------------------------------------------------------------------------------- inputs

def train_view(resolved: dict) -> dict:
    """The resolved config seen through c1_train's `train` key (preflight reads train.train_partition)."""
    return {**resolved, "train": resolved["scorer"]}


def load_raw_claims(resolved: dict, root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    """({claim_id: {claim_id, text, date}}, {claim_id: raw AVeriTeC claim}) from the raw train.json/dev.json directory
 data.claims (the real data and the tiny world share the raw layout). Text stays in memory."""
    from aac.data.averitec import SPLITS, claim_id, parse_date, read_raw

    raw = read_raw(Path(root) / resolved["data"]["claims"])
    claims, by_id = {}, {}
    for split in SPLITS:
        for idx, c in enumerate(raw[split]):
            cid = claim_id(split, idx)
            claims[cid] = {"claim_id": cid, "text": c["claim"], "date": parse_date(c.get("claim_date"), cid)}
            by_id[cid] = c
    return claims, by_id


def _gz_lines_and_sha(path: Path):
    """(parsed lines, sha256 of the decompressed bytes)."""
    h = hashlib.sha256()
    out = []
    with gzip.open(path, "rb") as fh:
        for raw in fh:
            h.update(raw)
            out.append(json.loads(raw))
    return out, h.hexdigest()


def read_evidence_run(root: Path, ref: dict) -> tuple[dict, Path]:
    """(evidence.json, run dir) of the EV run `ref`, refused unless its SEALED digest equals ref.sha256 and it is not a
 subset run (limit true): a subset EV run would silently drop train claims from the scorer data."""
    from aac.run.paths import runs_dir
    from aac.run.seal import UnsealedRun, verify_sealed

    d = runs_dir(root) / ref["run_id"]
    try:
        digest = verify_sealed(d)
    except (UnsealedRun, FileNotFoundError) as exc:
        raise ScorerDataError(f"EV run {ref['run_id']}: {exc}") from exc
    if digest != ref["sha256"]:
        raise ScorerDataError(f"EV run {ref['run_id']}: sealed digest {digest} != resolved scorer.evidence {ref['sha256']}")
    meta = json.loads((d / "evidence.json").read_text(encoding="ascii"))
    if meta.get("limit") or (meta.get("shortlist_run") or {}).get("limit"):
        raise ScorerDataError(f"EV run {ref['run_id']} is marked limit (a subset run); the scorer trains only on a full "
                              "EV run")
    if EV_VARIANT not in meta["evidence"]["variants"]:
        raise ScorerDataError(f"EV run {ref['run_id']} has no {EV_VARIANT} list")
    return meta, d


def preflight(resolved: dict, root: Path, *, claims: dict | None = None, manifest: dict | None = None,
              read_committed: Callable[[Path], str | None] = c1_train.committed_data_version) -> str:
    """The C1 gate and manifest assertions (train.train_partition = scorer.train_partition), then, once identity has
 filled scorer.evidence, the EV run checks; all before any model load. Returns the gate verdict."""
    root = Path(root)
    claims = load_raw_claims(resolved, root)[0] if claims is None else claims
    gate = c1_train.preflight(train_view(resolved), root, claims=claims, manifest=manifest, read_committed=read_committed)
    ref = resolved["scorer"].get("evidence")
    if ref is not None:
        meta, _ = read_evidence_run(root, ref)
        k = int(resolved["scorer"]["pairs"]["retrieved_top_k"])
        if k > int(meta["evidence"]["k"]):
            raise ScorerDataError(f"scorer.pairs.retrieved_top_k {k} exceeds the EV run's k {meta['evidence']['k']}")
    return gate


def assert_train_only(rows, manifest: dict, partition: str = "train") -> None:
    """Every pair's claim is in `partition` and in no other partition of the split manifest ."""
    allowed = set(manifest[partition])
    others = {p: set(manifest.get(p) or []) for p in c1_train.PARTITIONS if p != partition}
    for row in rows:
        cid = row[0]
        if cid not in allowed:
            raise PartitionViolation(f"pair claim {cid} is not in the {partition} partition")
        for p, ids in others.items():
            if cid in ids:
                raise PartitionViolation(f"pair claim {cid} is also listed in {p}")


def gold_text(question: str, answer: dict) -> str:
    """Boolean: question, answer and boolean_explanation joined by single spaces; any other type: the answer text."""
    if answer.get("answer_type") == BOOLEAN:
        parts = [question, answer.get("answer"), answer.get("boolean_explanation")]
    else:
        parts = [answer.get("answer")]
    return " ".join(str(x).strip() for x in parts if x is not None and str(x).strip())


def gold_pairs(raw_claim: dict, claim_label: str, pairs_cfg: dict) -> tuple[list[tuple[str, str, str]], int]:
    """([(ref, text, label)], empty answers skipped) for one claim that is not excluded; ref = `q<i>a<j>`."""
    mapped = pairs_cfg["label_map"][claim_label]
    out, empty = [], 0
    for qi, q in enumerate(raw_claim.get("questions") or []):
        for ai, a in enumerate(q.get("answers") or []):
            text = gold_text(str(q.get("question") or ""), a)
            if not text:
                empty += 1
                continue
            label = pairs_cfg["unanswerable"] if a.get("answer_type") == UNANSWERABLE else mapped
            out.append((f"q{qi}a{ai}", text, label))
    return out, empty


def retrieved_label(url_eq_answer: int, claim_label: str, pairs_cfg: dict) -> str:
    """url-match-claim-label-else-neutral: a passage from an answer's source URL takes the claim label."""
    return pairs_cfg["label_map"][claim_label] if int(url_eq_answer) == 1 else "NEUTRAL"


def _pid(claim_id: str, row) -> str:
    return f"{claim_id}#d{int(row[0])}#s{int(row[1])}"


def _retrieved_inputs(root: Path, ev_ref: dict, claim_ids: list[str], claims: dict, k: int):
    """({claim_id: [(passage_id, url_eq_answer), ...] first k primary re-ranked}, {passage_id: text})."""
    from aac.components.c2_evidence_run import EvidenceError, read_cache_texts, read_shortlist_run
    from aac.run.paths import out_root
    from aac.run.seal import verify_sealed

    meta, ev_dir = read_evidence_run(root, ev_ref)
    lines, sha = _gz_lines_and_sha(ev_dir / "evidence.jsonl.gz")
    if sha != meta["evidence_sha256"]:
        raise ScorerDataError(f"{ev_ref['run_id']}/evidence.jsonl.gz: sha256 {sha} != evidence_sha256 {meta['evidence_sha256']}")
    wanted = set(claim_ids)
    top: dict[str, list[str]] = {}
    for x in lines:
        if x["variant"] != EV_VARIANT or x["claim_id"] not in wanted:
            continue
        if claim_hash(claims[x["claim_id"]]["text"]) != x["claim_hash"]:
            raise ScorerDataError(f"{x['claim_id']}: claim text in data.claims does not match the EV run's claim_hash")
        top[x["claim_id"]] = [p for p, _ in x["rerank"][:k]]
    del lines
    missing = [c for c in claim_ids if c not in top]
    if missing:
        raise ScorerDataError(f"EV run {ev_ref['run_id']} has no {EV_VARIANT} list for {len(missing)} train claims "
                              f"(first {missing[0]})")
    sl = meta["shortlist_run"]
    try:
        sl_meta, sl_dir = read_shortlist_run(root, {"run_id": sl["run_id"]})
    except EvidenceError as exc:
        raise ScorerDataError(str(exc)) from exc
    if verify_sealed(sl_dir) != sl["sealed_sha256"] or sl_meta["private_cache_sha256"] != sl["private_cache_sha256"]:
        raise ScorerDataError(f"SL run {sl['run_id']} differs from the one EV run {ev_ref['run_id']} recorded")
    flags: dict[str, int] = {}
    want_pids = {p for ps in top.values() for p in ps}
    with gzip.open(sl_dir / "shortlist.jsonl.gz", "rb") as fh:
        for raw in fh:
            x = json.loads(raw)
            if x["claim_id"] not in top:
                continue
            for row in x["primary"]:
                p = _pid(x["claim_id"], row)
                if p in want_pids:
                    flags[p] = int(row[SL_FLAG_COL])
    lost = want_pids - set(flags)
    if lost:
        raise ScorerDataError(f"{len(lost)} re-ranked passages have no primary row in SL run {sl['run_id']}")
    try:
        texts = read_cache_texts(out_root(root) / sl_meta["private_cache"], sl_meta["private_cache_sha256"], want_pids)
    except EvidenceError as exc:
        raise ScorerDataError(str(exc)) from exc
    return {c: [(p, flags[p]) for p in ps] for c, ps in top.items()}, texts


def build_pairs(resolved: dict, root: Path, *, claims: dict | None = None, raw_by_id: dict | None = None,
                manifest: dict | None = None) -> tuple[list[list[str]], list[tuple[str, str]], dict]:
    """(manifest rows [claim_id, source, ref, label], texts [(passage_text, claim_text)] aligned with the rows,
 counts). Rows carry no text; texts never leave memory."""
    from aac.data.averitec import order_key
    from aac.data.fixtures import load_labels

    root = Path(root)
    sc = resolved["scorer"]
    pc = sc["pairs"]
    if claims is None or raw_by_id is None:
        claims, raw_by_id = load_raw_claims(resolved, root)
    manifest = c1_train.load_manifest(resolved, root) if manifest is None else manifest
    labels = load_labels(root / resolved["data"]["labels"])
    partition = sc["train_partition"]
    ids = sorted(manifest[partition], key=order_key)
    excluded = set(pc["exclude_claim_labels"])
    for cid in ids:
        raw_label = raw_by_id[cid].get("label")
        if raw_label != labels.get(cid):
            raise ScorerDataError(f"{cid}: label {labels.get(cid)!r} in data.labels differs from the raw label {raw_label!r}")
    used = [c for c in ids if labels[c] not in excluded]
    k = int(pc["retrieved_top_k"])
    retrieved, ptexts = ({}, {})
    if k > 0:
        if sc.get("evidence") is None:
            raise ScorerDataError("scorer.evidence is unset: aac.run.identity resolves it from scorer.evidence_from")
        retrieved, ptexts = _retrieved_inputs(root, sc["evidence"], used, claims, k)
    rows: list[list[str]] = []
    texts: list[tuple[str, str]] = []
    empty = 0
    for cid in used:
        ctext = claims[cid]["text"]
        if pc["gold"]:
            g, e = gold_pairs(raw_by_id[cid], labels[cid], pc)
            empty += e
            for ref, text, label in g:
                rows.append([cid, "gold", ref, label])
                texts.append((text, ctext))
        for pid, flag in retrieved.get(cid, []):
            rows.append([cid, "retrieved", pid, retrieved_label(flag, labels[cid], pc)])
            texts.append((ptexts[pid], ctext))
    assert_train_only(rows, manifest, partition)
    counts = {"train_partition_claims": len(ids), "claims_used": len(used), "claims_excluded_by_label": len(ids) - len(used),
              "skipped_empty_gold": empty}
    return rows, texts, counts


def pairs_record(rows: list[list[str]], counts: dict, resolved: dict) -> dict:
    """pairs.json: counts by source and label and the sha256 of the sorted text-free pair list."""
    by_label = {lab: sum(1 for r in rows if r[3] == lab) for lab in SCORER_LABELS}
    by_source = {s: sum(1 for r in rows if r[1] == s) for s in SOURCES}
    by_source_label = {s: {lab: sum(1 for r in rows if r[1] == s and r[3] == lab) for lab in SCORER_LABELS}
                       for s in SOURCES}
    sc = resolved["scorer"]
    return {
        "schema": PAIRS_SCHEMA, "n_pairs": len(rows), "by_label": by_label, "by_source": by_source,
        "by_source_label": by_source_label, "counts": counts, "pair_fields": list(PAIR_FIELDS),
        "pairs_sha256": sha256_bytes(canonical(sorted(rows))), "pairs": sc["pairs"],
        "evidence_run": sc.get("evidence"), "ev_variant": EV_VARIANT, "labels": list(SCORER_LABELS),
    }


# ------------------------------------------------------------------------------------------- training

def train_scorer(resolved: dict, root: Path, run_dir: Path, seed: int, clock,
                 model_loader: Callable | None = None, loss_fn: Callable | None = None) -> dict:
    """Train the C2 scorer per resolved["scorer"]; write train-log.jsonl, checkpoint.safetensors, model.json and
 pairs.json into run_dir; return the run-record summary. Raises NonFiniteLoss (nothing sealed) on a NaN/Inf loss or
 grad norm. `model_loader(model_cfg, root) -> (tokenizer, model)` and `loss_fn` are test seams."""
    from aac.components.c2_scorer import tokenize_pairs

    root, run_dir = Path(root), Path(run_dir)
    t = resolved["scorer"]
    cublas_at_start = os.environ.get("CUBLAS_WORKSPACE_CONFIG")  # before seed_everything's setdefault (operational constraint)
    t_pairs = time.perf_counter()
    claims, raw_by_id = load_raw_claims(resolved, root)
    manifest = c1_train.load_manifest(resolved, root)
    preflight(resolved, root, claims=claims, manifest=manifest)
    rows, texts, counts = build_pairs(resolved, root, claims=claims, raw_by_id=raw_by_id, manifest=manifest)
    del claims, raw_by_id  # nothing outside the train partition is used from here on
    if not rows:
        raise ScorerDataError("no training pair: check pairs.gold, retrieved_top_k and the train partition")
    items = [(p, c, SCORER_LABELS.index(r[3])) for (p, c), r in zip(texts, rows)]
    del texts
    pairs_seconds = time.perf_counter() - t_pairs

    from aac.run.seeds import seed_everything

    seed_everything(seed)
    import torch
    from torch.utils.data import DataLoader

    strict = t["determinism"] == "strict"
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and strict and cublas_at_start not in c1_train.CUBLAS_OK:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be one of {c1_train.CUBLAS_OK} in the process environment before "
                           f"Python starts (got {cublas_at_start!r}); scripts/gpu_run_remote.sh exports it")

    t_load = time.perf_counter()
    tok, model = (model_loader or c1_train.load_base_model)(t["model"], root)
    c1_train.assert_fp32(model)
    model.to(device)
    model.train()
    load_seconds = time.perf_counter() - t_load

    max_length = int(t["max_length"])

    def collate(batch):
        enc = tokenize_pairs(tok, [b[0] for b in batch], [b[1] for b in batch], max_length)
        return dict(enc), torch.tensor([b[2] for b in batch], dtype=torch.long)

    partition = t["train_partition"]
    gen = torch.Generator().manual_seed(derive_seed(t["loader"]["seed_key_template"].format(split=partition, seed=seed)))
    loader = DataLoader(items, batch_size=int(t["batch_size"]), shuffle=bool(t["loader"]["shuffle"]), generator=gen,
                        num_workers=int(t["loader"]["num_workers"]), collate_fn=collate)
    epochs = int(t["epochs"])
    total = epochs * math.ceil(len(items) / int(t["batch_size"]))
    if t["max_steps"] is not None:
        total = min(total, int(t["max_steps"]))
    warmup = math.ceil(float(t["schedule"]["warmup_fraction"]) * total)
    opt_cfg = t["optimizer"]
    opt = torch.optim.AdamW(model.parameters(), lr=float(opt_cfg["lr"]), betas=tuple(float(b) for b in opt_cfg["betas"]),
                            eps=float(opt_cfg["eps"]), weight_decay=float(opt_cfg["weight_decay"]))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, c1_train.linear_warmup_decay(total, warmup))
    loss_of = loss_fn or c1_train.ce_loss
    clip = float(t["grad_clip_norm"])
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if device == "cuda" else nullcontext

    step = 0
    t_train = time.perf_counter()
    with open(run_dir / TRAIN_LOG, "w", encoding="ascii", newline="\n") as log:
        for epoch in range(epochs):
            if step >= total:
                break
            for enc, y in loader:
                if step >= total:
                    break
                enc = {k: v.to(device) for k, v in enc.items()}
                y = y.to(device)
                with autocast():
                    logits = model(**enc).logits  # never labels=: the model does not compute the loss
                loss = loss_of(logits, y)
                loss_value = float(loss.detach())
                if not math.isfinite(loss_value):
                    raise c1_train.NonFiniteLoss(f"step {step} (epoch {epoch}): loss {loss_value}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                gn_value = float(gn.detach())
                if not math.isfinite(gn_value):
                    raise c1_train.NonFiniteLoss(f"step {step} (epoch {epoch}): grad norm {gn_value}")
                lr_now = float(sched.get_last_lr()[0])
                opt.step()
                sched.step()
                log.write(c1_train.dumps_line({"step": step, "epoch": epoch, "loss": round(loss_value, 6),
                                               "grad_norm": round(gn_value, 6), "lr": round(lr_now, 12)}))
                step += 1
    train_seconds = time.perf_counter() - t_train
    if device == "cuda":
        torch.cuda.synchronize()

    model.eval()
    ck_sha = c1_train.save_checkpoint(model, run_dir / CHECKPOINT)
    n_params = int(sum(p.numel() for p in model.parameters()))
    (run_dir / MODEL_JSON).write_text(dumps_sorted({
        "hf_id": t["model"]["hf_id"], "revision": t["model"]["revision"], "num_labels": len(SCORER_LABELS),
        "labels": list(SCORER_LABELS), "max_length": max_length, "pair_order": t["pairs"]["order"],
        "truncation": t["pairs"]["truncation"], "checkpoint": CHECKPOINT, "checkpoint_format": "safetensors",
        "checkpoint_dtype": "float32", "checkpoint_sha256": ck_sha, "n_parameters": n_params,
    }), encoding="ascii")
    prec = pairs_record(rows, counts, resolved)
    (run_dir / PAIRS_JSON).write_text(dumps_sorted(prec), encoding="ascii")
    wall = clock.source == "wall"
    sealed_seconds = round(train_seconds + load_seconds, 6) if wall else 0.0
    level = (f"seeded; torch-deterministic-{'strict' if strict else 'warn-only'}; fp32 weights; "
             f"{'bf16 autocast (cuda)' if device == 'cuda' else 'fp32 compute (cpu)'}; num_workers {t['loader']['num_workers']}")
    n = len(items)
    return {
        "configurations": [f"train-scorer/{resolved['name']}"],
        "items": {"train_pairs": n, "train_claims": counts["claims_used"],
                  **{f"pairs_{lab}": prec["by_label"][lab] for lab in SCORER_LABELS}},
        "stages": [{"name": "pairs", "device": "cpu", "seconds": round(pairs_seconds, 6) if wall else 0.0, "items": n},
                   {"name": "train", "device": device, "seconds": sealed_seconds, "items": n}],
        "timings": [{"stage": "pairs", "seconds": round(pairs_seconds, 6), "items": n},
                    {"stage": "load_model", "seconds": round(load_seconds, 6), "items": 0},
                    {"stage": "train", "seconds": round(train_seconds, 6), "items": n, "steps": step,
                     "seconds_per_step": round(train_seconds / max(1, step), 6), "device": device,
                     "torch_threads": int(torch.get_num_threads())}],
        "depends_on": sorted({t["evidence"]["run_id"]}) if t.get("evidence") else [],
        "steps": step,
        "determinism_level": level,
    }
