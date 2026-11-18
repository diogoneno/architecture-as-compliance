"""C1 training run kind : a hand-written fine-tuning loop for mDeBERTa-v3-base.

The recipe is the committed `train:` block of an aac-train/1 config (configs/train/c1.yaml); nothing
here chooses a value. Order of operations in train_c1:

 1. preflight: the datasheet-before-training gate (Art. 10 before Art. 15; ordering) and the
 split-manifest assertions for the learned component , before torch or any model load;
 2. seed_everything, strict deterministic algorithms (and, on CUDA, CUBLAS_WORKSPACE_CONFIG set in the
 process environment before Python started);
 3. the base model at the pinned revision, offline, dtype=torch.float32 (asserted for every parameter);
 4. fixed epochs over the train partition only (no early stopping, no val/test access, pre-registered specification),
 loss computed here from the logits (no `labels=` to the model), isfinite asserted on the loss and
 the clipped grad norm every step;
 5. the last checkpoint as safetensors (fp32 state dict), its sha256 in model.json.

torch and transformers are imported inside functions: importing this module stays cheap."""
from __future__ import annotations

import json
import math
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable

from aac.data.fixtures import LABELS
from aac.run.hashing import data_version, derive_seed, dumps_sorted, sha256_file

TRAIN_LOG = "train-log.jsonl"
CHECKPOINT = "checkpoint.safetensors"
MODEL_JSON = "model.json"
DATA_VERSION_REL = Path("evidence-pack/data_version.txt")
RELEASE_SHA_FILE = "AAC_GIT_SHA"
CUBLAS_OK = (":4096:8", ":16:8")
PARTITIONS = ("train", "val_fuse", "val_cal", "test")


class NonFiniteLoss(RuntimeError):
    """A loss or clipped grad norm was NaN/Inf: the run aborts and is never sealed."""


class DatasheetGateClosed(RuntimeError):
    """The committed evidence-pack/data_version.txt is absent or differs from data_version(data.manifest)."""


class CutoffViolation(RuntimeError):
    """A chronological manifest breaks max(date over train, val_fuse, val_cal) <= cutoff_date < min(date over test)."""


class ManifestViolation(RuntimeError):
    """The train partition holds an undated, removed-duplicate, empty-text or unknown id."""


# ---------------------------------------------------------------------------------------------- data

def load_texts(resolved: dict, root: Path) -> dict[str, dict]:
    """{claim_id: {"text", "date"}} through the dataset's claim loader (text is never written out)."""
    data = resolved["data"]
    if data["dataset"] == "fixture":
        from aac.data.fixtures import load_claims
    elif data["dataset"] == "averitec":
        from aac.data.averitec import load_claims
    else:  # pragma: no cover - DataCfg.dataset is a Literal
        raise ValueError(f"unknown dataset {data['dataset']!r}")
    return load_claims(Path(root) / data["claims"])


def load_manifest(resolved: dict, root: Path) -> dict:
    with open(Path(root) / resolved["data"]["split_manifest"], encoding="utf-8") as fh:
        return json.load(fh)


def ordered(ids, dataset: str) -> list[str]:
    """AVeriTeC ids by (split rank, idx) (pre-registered specification."""
    if dataset == "averitec":
        from aac.data.averitec import order_key

        return sorted(ids, key=order_key)
    return sorted(ids)


def is_chronological(manifest: dict) -> bool:
    kind = manifest.get("kind")
    return kind == "chronological" or (kind is None and "cutoff_date" in manifest)


def _ids(entries) -> set[str]:
    """Excluded-id lists hold ids (`excluded_undated`) or {claim_id, group_id} rows (`excluded_duplicates`)."""
    return {e["claim_id"] if isinstance(e, dict) else e for e in (entries or [])}


def assert_manifest(manifest: dict, claims: dict[str, dict], partition: str = "train") -> None:
    """ for the learned component. Every R1 manifest: no undated, removed-duplicate, empty-text or
 unknown id in `partition`. A chronological manifest also: every train/val_fuse/val_cal date <= cutoff_date
 and every test date strictly later (operational constraint: `<=` for the cutoff day, never `max(train) < cutoff`)."""
    train = list(manifest[partition])
    excluded = {key: _ids(manifest.get(key)) for key in ("excluded_undated", "excluded_duplicates", "excluded_empty_text")}
    for cid in train:
        for key, ids in excluded.items():
            if cid in ids:
                raise ManifestViolation(f"{partition} id {cid} is listed in {key}")
        if cid not in claims:
            raise ManifestViolation(f"{partition} id {cid} is not in the claim loader's output")
        if not claims[cid].get("date"):
            raise ManifestViolation(f"{partition} id {cid} is undated")
        if not str(claims[cid].get("text") or "").strip():
            raise ManifestViolation(f"{partition} id {cid} has an empty claim text")
    if not is_chronological(manifest):
        return
    cutoff = manifest["cutoff_date"]
    for part in ("train", "val_fuse", "val_cal"):
        for cid in manifest.get(part) or []:
            d = (claims.get(cid) or {}).get("date")
            if not d or d > cutoff:
                raise CutoffViolation(f"{part} id {cid} dated {d} is not <= cutoff_date {cutoff}")
    for cid in manifest.get("test") or []:
        d = (claims.get(cid) or {}).get("date")
        if not d or d <= cutoff:
            raise CutoffViolation(f"test id {cid} dated {d} is not later than cutoff_date {cutoff}")


def committed_data_version(root: Path) -> str | None:
    """The data_version.txt committed at HEAD (`git show HEAD:./evidence-pack/data_version.txt`), or None.

 Inside a git-archive release (no .git, an AAC_GIT_SHA file at the root) the files on disk ARE the
 committed tree of that commit, so the file on disk is read."""
    root = Path(root)
    try:
        out = subprocess.run(["git", "show", f"HEAD:./{DATA_VERSION_REL.as_posix()}"], cwd=str(root),
                             capture_output=True, text=True, check=False)
    except FileNotFoundError:
        out = None
    if out is not None and out.returncode == 0:
        return out.stdout.strip() or None
    inside = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=str(root), capture_output=True,
                            text=True, check=False) if out is not None else None
    if (inside is None or inside.returncode != 0) and (root / RELEASE_SHA_FILE).is_file():
        p = root / DATA_VERSION_REL
        return p.read_text(encoding="ascii").strip() or None if p.is_file() else None
    return None


def datasheet_gate(resolved: dict, root: Path,
                   read_committed: Callable[[Path], str | None] = committed_data_version) -> str:
    """Refuse real-data training unless the committed data_version.txt equals data_version(data.manifest)
 AND every file the run reads still has the sha256 that manifest lists.
 Returns "skipped (fixture)" or the matching data_version."""
    data = resolved["data"]
    if data["dataset"] == "fixture":
        return "skipped (fixture)"
    committed = read_committed(Path(root))
    if committed is None:
        raise DatasheetGateClosed(
            f"no committed {DATA_VERSION_REL} at HEAD: render and commit the datasheets before training on "
            f"{data['dataset']} (Art. 10 before Art. 15)")
    manifest = Path(root) / data["manifest"]
    if not manifest.is_file():
        raise DatasheetGateClosed(f"data.manifest {data['manifest']} does not exist; committed {DATA_VERSION_REL} = {committed}")
    dv = data_version(manifest)
    if committed != dv:
        raise DatasheetGateClosed(f"committed {DATA_VERSION_REL} = {committed} != data_version({data['manifest']}) = {dv}")
    
    # file this run reads (split manifest, labels, the raw claim files, ...) against the manifest first.
    from aac.data.manifest import DataDrift, data_paths, verify_entries

    try:
        verify_entries(Path(root), data["manifest"], data_paths(data))
    except DataDrift as exc:
        raise DatasheetGateClosed(f"data drift under data_version {dv}: {exc}") from exc
    return dv


def preflight(resolved: dict, root: Path, *, claims: dict | None = None, manifest: dict | None = None,
              read_committed: Callable[[Path], str | None] = committed_data_version) -> str:
    """The gate, then the manifest assertions; both before any model load."""
    gate = datasheet_gate(resolved, root, read_committed)
    claims = load_texts(resolved, root) if claims is None else claims
    manifest = load_manifest(resolved, root) if manifest is None else manifest
    assert_manifest(manifest, claims, resolved["train"]["train_partition"])
    return gate


def select_train_items(resolved: dict, root: Path, claims: dict, manifest: dict) -> list[tuple[str, str, int]]:
    """(claim_id, text, label index) for the train partition only, in (split rank, idx) order."""
    from aac.data.fixtures import load_labels

    partition = resolved["train"]["train_partition"]
    labels = load_labels(Path(root) / resolved["data"]["labels"])
    ids = ordered(manifest[partition], resolved["data"]["dataset"])
    return [(cid, claims[cid]["text"], LABELS.index(labels[cid])) for cid in ids]


# --------------------------------------------------------------------------------------------- model

def hf_cache_dir(root: Path) -> str:
    """$HF_HOME/hub, else <root>/.hf/hub (the pinned cache). Passed explicitly as cache_dir=, because
 huggingface_hub reads HF_HOME once at import time."""
    return str(Path(os.environ.get("HF_HOME") or Path(root) / ".hf") / "hub")


def _offline_env(root: Path) -> None:
    os.environ.setdefault("HF_HOME", str(Path(root) / ".hf"))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    # The load report (MLM heads UNEXPECTED, pooler/classifier MISSING = newly initialised) is expected for a
    # sequence-classification head on a pretrained encoder; keep stderr to errors only.
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()


def load_tokenizer(hf_id: str, revision: str, root: Path):
    _offline_env(root)
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(hf_id, revision=revision, local_files_only=True, cache_dir=hf_cache_dir(root))


def load_base_model(model_cfg: dict, root: Path):
    """(tokenizer, model): the pinned revision, offline, num_labels, dtype=torch.float32 (transformers 5
 otherwise loads mDeBERTa as fp16, operational constraint); every parameter asserted float32."""
    _offline_env(root)
    import torch
    from transformers import AutoModelForSequenceClassification

    tok = load_tokenizer(model_cfg["hf_id"], model_cfg["revision"], root)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["hf_id"], revision=model_cfg["revision"], num_labels=int(model_cfg["num_labels"]),
        dtype=torch.float32, local_files_only=True, cache_dir=hf_cache_dir(root))
    assert_fp32(model)
    return tok, model


def assert_fp32(model) -> None:
    import torch

    dtypes = {p.dtype for p in model.parameters()}
    if dtypes != {torch.float32}:
        raise TypeError(f"C1 parameters must all be float32, got {sorted(map(str, dtypes))}")


def ce_loss(logits, y):
    """Mean negative log-softmax gathered at the label (own code; gather and its backward are on the
 deterministic list, unlike NLLLoss on CUDA, operational constraint). No class weights."""
    import torch

    lp = torch.log_softmax(logits.float(), dim=-1)
    return -(lp.gather(1, y.view(-1, 1))).mean()


def linear_warmup_decay(total_steps: int, warmup_steps: int) -> Callable[[int], float]:
    """LambdaLR factor for optimizer step k (0-based): k / W while k < W, then (T - k) / (T - W), floored at 0."""
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))
    return f


def save_checkpoint(model, path: Path) -> str:
    """safetensors of the contiguous fp32 state dict (cloned, so no tensor shares storage); returns its sha256."""
    import torch
    from safetensors.torch import save_file

    sd = {k: v.detach().to("cpu", torch.float32).contiguous().clone() for k, v in model.state_dict().items()}
    tmp = Path(path).with_name(Path(path).name + ".part")
    save_file(sd, str(tmp))
    os.replace(tmp, path)
    return sha256_file(path)


# ------------------------------------------------------------------------------------------- training

def train_c1(resolved: dict, root: Path, run_dir: Path, seed: int, clock,
             model_loader: Callable | None = None, loss_fn: Callable | None = None) -> dict:
    """Train C1 per resolved["train"]; write train-log.jsonl, checkpoint.safetensors and model.json into
 run_dir; return the run-record summary. Raises NonFiniteLoss (nothing sealed) on a NaN/Inf loss or
 grad norm. `model_loader(model_cfg, root) -> (tokenizer, model)` and `loss_fn` are test seams."""
    root, run_dir = Path(root), Path(run_dir)
    t = resolved["train"]
    cublas_at_start = os.environ.get("CUBLAS_WORKSPACE_CONFIG")  # before seed_everything's setdefault (operational constraint)
    claims = load_texts(resolved, root)
    manifest = load_manifest(resolved, root)
    preflight(resolved, root, claims=claims, manifest=manifest)
    items = select_train_items(resolved, root, claims, manifest)
    del claims  # nothing outside the train partition is used from here on

    from aac.run.seeds import seed_everything

    seed_everything(seed)
    import torch
    from torch.utils.data import DataLoader

    strict = t["determinism"] == "strict"
    torch.use_deterministic_algorithms(True, warn_only=not strict)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and strict and cublas_at_start not in CUBLAS_OK:
        raise RuntimeError(f"CUBLAS_WORKSPACE_CONFIG must be one of {CUBLAS_OK} in the process environment before "
                           f"Python starts (got {cublas_at_start!r}); scripts/gpu_run_remote.sh exports it")

    t_load = time.perf_counter()
    tok, model = (model_loader or load_base_model)(t["model"], root)
    assert_fp32(model)
    model.to(device)
    model.train()
    load_seconds = time.perf_counter() - t_load

    max_length = int(t["max_length"])

    def collate(batch):
        enc = tok([b[1] for b in batch], padding="longest", truncation=True, max_length=max_length,
                  return_tensors="pt")
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
    sched = torch.optim.lr_scheduler.LambdaLR(opt, linear_warmup_decay(total, warmup))
    loss_of = loss_fn or ce_loss
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
                    raise NonFiniteLoss(f"step {step} (epoch {epoch}): loss {loss_value}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                gn_value = float(gn.detach())
                if not math.isfinite(gn_value):
                    raise NonFiniteLoss(f"step {step} (epoch {epoch}): grad norm {gn_value}")
                lr_now = float(sched.get_last_lr()[0])
                opt.step()
                sched.step()
                log.write(dumps_line({"step": step, "epoch": epoch, "loss": round(loss_value, 6),
                                      "grad_norm": round(gn_value, 6), "lr": round(lr_now, 12)}))
                step += 1
    train_seconds = time.perf_counter() - t_train
    if device == "cuda":
        torch.cuda.synchronize()

    model.eval()
    ck_sha = save_checkpoint(model, run_dir / CHECKPOINT)
    n_params = int(sum(p.numel() for p in model.parameters()))
    (run_dir / MODEL_JSON).write_text(dumps_sorted({
        "hf_id": t["model"]["hf_id"], "revision": t["model"]["revision"], "num_labels": int(t["model"]["num_labels"]),
        "labels": list(LABELS), "max_length": max_length, "checkpoint": CHECKPOINT, "checkpoint_format": "safetensors",
        "checkpoint_dtype": "float32", "checkpoint_sha256": ck_sha, "n_parameters": n_params,
    }), encoding="ascii")
    sealed_seconds = round(train_seconds + load_seconds, 6) if clock.source == "wall" else 0.0
    level = (f"seeded; torch-deterministic-{'strict' if strict else 'warn-only'}; fp32 weights; "
             f"{'bf16 autocast (cuda)' if device == 'cuda' else 'fp32 compute (cpu)'}; num_workers {t['loader']['num_workers']}")
    return {
        "configurations": [f"train/{resolved['name']}"],
        "items": {partition: len(items)},
        "stages": [{"name": "train", "device": device, "seconds": sealed_seconds, "items": len(items)}],
        "timings": [{"stage": "load_model", "seconds": round(load_seconds, 6), "items": 0},
                    {"stage": "train", "seconds": round(train_seconds, 6), "items": len(items), "steps": step,
                     "seconds_per_step": round(train_seconds / max(1, step), 6), "device": device,
                     "torch_threads": int(torch.get_num_threads())}],
        "depends_on": [],
        "steps": step,
        "determinism_level": level,
    }


def dumps_line(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
