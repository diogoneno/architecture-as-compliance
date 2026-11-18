"""Deterministic synthetic fixture corpus (48 claims) under tests/fixtures/corpus/.

`build(root)` regenerates every file byte-identically; MANIFEST.sha256 (written last)
is the input of aac.run.hashing.data_version for fixture runs (D-09)."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from aac.run.hashing import derive_seed, dumps_sorted

LABELS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking")
N_CLAIMS = 48
CORPUS_REL = Path("tests/fixtures/corpus")
MANIFEST_FILES = (
    "claims.json", "labels.csv", "evidence.json", "splits/averitec-chrono-fx.json", "datasheet-stats.json",
    "fc-fx.json", "rd-fx.json", "splits/averitec-random-fx.json", "splits/averitec-attack-subset-fx.json",
)
N_FC = 24
N_RD = 24
N_ATTACK_SUBSET = 8

_SUBJECTS = ["The council", "A minister", "The agency", "The company", "A report", "The mayor", "The committee",
             "The regulator", "A senator", "The university", "The clinic", "The union"]
_VERBS = ["announced", "denied", "claimed", "confirmed", "reported", "rejected", "approved", "measured"]
_OBJECTS = ["a twelve percent rise in funding", "the closure of three plants", "a new vaccine trial",
            "record rainfall last March", "a ban on single-use cups", "the merger of two districts",
            "a drop in unemployment", "the recall of two models", "a tax on sugary drinks",
            "the reopening of the bridge", "a cut to bus services", "the audit of the stadium budget"]
_NON_ASCII = [
    "The café in São Paulo reported a façade repair costing twelve million réis",
    "Zürich's Bürgermeister announced a naïve plan for the Straße renovation",
    "The Ålesund council approved a new fjord ferry with Ørsted turbines",
]
_PASSAGE_WORDS = ["audit", "ledger", "minutes", "budget", "survey", "bulletin", "statement", "transcript"]


def _rng(key: str) -> np.random.Generator:
    return np.random.default_rng(derive_seed(key))


def _sentence(rng: np.random.Generator) -> str:
    return f"{_SUBJECTS[rng.integers(len(_SUBJECTS))]} {_VERBS[rng.integers(len(_VERBS))]} {_OBJECTS[rng.integers(len(_OBJECTS))]}"


def generate() -> dict:
    rng = _rng("fixture.corpus.v1")
    claims, labels, evidence = [], [], {}
    base = date(2020, 1, 1)
    for idx in range(N_CLAIMS):
        cid = f"fx/{idx}"
        if idx in (5, 17, 29):
            text = _NON_ASCII[(5, 17, 29).index(idx)]
        elif idx == 11:
            text = claims[10]["text"]  # exact duplicate of fx/10
        elif idx == 23:
            text = claims[22]["text"]  # exact duplicate of fx/22
        else:
            text = _sentence(rng)
        if idx in (3, 14, 26, 40):
            d = None
        else:
            d = (base + timedelta(days=int(rng.integers(0, 300)))).isoformat()
        claims.append({"claim_id": cid, "text": text, "date": d})
        labels.append({"claim_id": cid, "label": LABELS[int(rng.choice(4, p=[0.27, 0.57, 0.09, 0.07]))]})
        n_pass = int(rng.integers(0, 6))
        rows = []
        for k in range(n_pass):
            ptext = f"{_PASSAGE_WORDS[rng.integers(len(_PASSAGE_WORDS))]} {idx}-{k}: {_sentence(rng)}"
            rows.append({
                "id": f"{cid}#{k}",
                "text_hash": hashlib.sha256(ptext.encode("utf-8")).hexdigest(),
                "score": round(float(rng.uniform(0.0, 1.0)), 6),
            })
        evidence[cid] = rows
    return {"claims": claims, "labels": labels, "evidence": evidence}


def chrono_split(claims: list[dict]) -> dict:
    """Split rules: test = dated claims strictly after the cutoff, the cutoff being the
 latest date such that >= 20 % of dated claims are later; undated excluded; val
 carved from the pre-cutoff set by hash and split into val_fuse / val_cal by
 sha256(claim_id) mod 2."""
    dated = sorted((c for c in claims if c["date"]), key=lambda c: (c["date"], c["claim_id"]))
    undated = sorted(c["claim_id"] for c in claims if not c["date"])
    n = len(dated)
    dates = sorted({c["date"] for c in dated})
    cutoff = None
    for d in dates:
        later = sum(1 for c in dated if c["date"] > d)
        if later >= 0.2 * n:
            cutoff = d
    if cutoff is None:
        raise RuntimeError("no cutoff satisfies the 20 percent rule")
    test = sorted(c["claim_id"] for c in dated if c["date"] > cutoff)
    pre = [c["claim_id"] for c in dated if c["date"] <= cutoff]
    train, val_fuse, val_cal = [], [], []
    for cid in sorted(pre):
        h = int(hashlib.sha256(cid.encode("utf-8")).hexdigest(), 16)
        if (h >> 64) % 2 == 0:
            (val_fuse if h % 2 == 0 else val_cal).append(cid)
        else:
            train.append(cid)
    return {
        "name": "averitec-chrono-fx",
        "rule": "test = dated claims strictly after cutoff; cutoff = latest date with >= 20% of dated claims later; "
                "undated excluded; val = pre-cutoff claims with (sha256(claim_id) >> 64) % 2 == 0, "
                "split val_fuse/val_cal by sha256(claim_id) mod 2",
        "cutoff_date": cutoff,
        "train": train, "val_fuse": val_fuse, "val_cal": val_cal, "test": test,
        "excluded_undated": undated,
        "counts": {"train": len(train), "val_fuse": len(val_fuse), "val_cal": len(val_cal), "test": len(test),
                   "excluded_undated": len(undated)},
    }


def random_split(claims: list[dict], chrono: dict) -> dict:
    """R1 random split with the SAME partition sizes as the chronological split (same population:
 undated claims excluded from both, D-07); a derive_seed-keyed permutation of the dated ids."""
    dated = sorted(c["claim_id"] for c in claims if c["date"])
    undated = sorted(c["claim_id"] for c in claims if not c["date"])
    seed = derive_seed("fixture.split.random.v1")
    perm = np.random.default_rng(seed).permutation(len(dated))
    order = [dated[int(i)] for i in perm]
    counts = chrono["counts"]
    cut1 = counts["test"]
    cut2 = cut1 + counts["val_fuse"]
    cut3 = cut2 + counts["val_cal"]
    test, val_fuse, val_cal, train = order[:cut1], order[cut1:cut2], order[cut2:cut3], order[cut3:]
    return {
        "name": "averitec-random-fx",
        "rule": "dated claims permuted by default_rng(split_seed); partition sizes copied from averitec-chrono-fx "
                "(test, val_fuse, val_cal, train in that order); undated excluded (D-07)",
        "split_seed": seed,
        "train": sorted(train), "val_fuse": sorted(val_fuse), "val_cal": sorted(val_cal), "test": sorted(test),
        "excluded_undated": undated,
        "counts": {"train": len(train), "val_fuse": len(val_fuse), "val_cal": len(val_cal), "test": len(test),
                   "excluded_undated": len(undated)},
    }


def attack_subset(chrono: dict, n: int = N_ATTACK_SUBSET) -> dict:
    """R3 AVeriTeC attack subset: n ids drawn by a derive_seed-keyed permutation of the chrono test ids."""
    ids = sorted(chrono["test"])
    seed = derive_seed("fixture.attack-subset.v1")
    perm = np.random.default_rng(seed).permutation(len(ids))
    chosen = sorted(ids[int(i)] for i in perm[:n])
    return {"name": "averitec-attack-subset-fx", "source_split": chrono["name"], "source_partition": "test",
            "rule": "first n ids of a default_rng(subset_seed) permutation of the chronological test ids",
            "subset_seed": seed, "ids": chosen, "count": len(chosen)}


def generate_fc() -> list[dict]:
    """InCrediblAE-FC-shaped items: claim + one supplied evidence sentence, label 1 = non-credible."""
    rng = _rng("fixture.fc.v1")
    items = []
    for idx in range(N_FC):
        text = _sentence(rng)
        evidence = f"{_PASSAGE_WORDS[rng.integers(len(_PASSAGE_WORDS))]} extract: {_sentence(rng)}"
        items.append({"item_id": f"fc/{idx}", "text": text, "evidence": evidence, "label": int(rng.uniform() < 0.5)})
    return items


def generate_rd() -> list[dict]:
    """InCrediblAE-RD-shaped items: a thread-opening post, no supplied evidence, label 1 = non-credible."""
    rng = _rng("fixture.rd.v1")
    items = []
    for idx in range(N_RD):
        text = f"Thread: {_sentence(rng)}, sources say"
        items.append({"item_id": f"rd/{idx}", "text": text, "label": int(rng.uniform() < 0.5)})
    return items


def datasheet_stats(claims: list[dict], labels: list[dict]) -> dict:
    dated = [c["date"] for c in claims if c["date"]]
    counts = {lab: 0 for lab in LABELS}
    for row in labels:
        counts[row["label"]] += 1
    texts = [c["text"] for c in claims]
    return {
        "n_claims": len(claims),
        "n_dated": len(dated),
        "n_undated": len(claims) - len(dated),
        "date_min": min(dated),
        "date_max": max(dated),
        "label_counts": counts,
        "label_shares": {k: round(v / len(claims), 6) for k, v in counts.items()},
        "n_duplicate_texts": len(texts) - len(set(texts)),
        "n_non_ascii_texts": sum(1 for t in texts if not t.isascii()),
    }


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(dumps_sorted(obj))


def write_manifest(corpus_dir: Path) -> Path:
    lines = []
    for rel in MANIFEST_FILES:
        digest = hashlib.sha256((corpus_dir / rel).read_bytes()).hexdigest()
        lines.append(f"{digest}  {rel}")
    lines.sort()  # LC_ALL=C order (ASCII paths)
    target = corpus_dir / "MANIFEST.sha256"
    with open(target, "w", encoding="ascii", newline="\n") as fh:
        fh.write("".join(line + "\n" for line in lines))
    return target


def build(root: Path) -> Path:
    corpus = Path(root) / CORPUS_REL
    corpus.mkdir(parents=True, exist_ok=True)
    g = generate()
    _write_json(corpus / "claims.json", g["claims"])
    with open(corpus / "labels.csv", "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["claim_id", "label"])
        for row in g["labels"]:
            w.writerow([row["claim_id"], row["label"]])
    _write_json(corpus / "evidence.json", g["evidence"])
    chrono = chrono_split(g["claims"])
    _write_json(corpus / "splits" / "averitec-chrono-fx.json", chrono)
    _write_json(corpus / "datasheet-stats.json", datasheet_stats(g["claims"], g["labels"]))
    _write_json(corpus / "fc-fx.json", generate_fc())
    _write_json(corpus / "rd-fx.json", generate_rd())
    _write_json(corpus / "splits" / "averitec-random-fx.json", random_split(g["claims"], chrono))
    _write_json(corpus / "splits" / "averitec-attack-subset-fx.json", attack_subset(chrono))
    return write_manifest(corpus)  # last: data_version follows every fixture file (D-09)


def load_claims(path: Path) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {c["claim_id"]: c for c in json.load(fh)}


def load_labels(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return {row["claim_id"]: row["label"] for row in csv.DictReader(fh)}


def load_split(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
