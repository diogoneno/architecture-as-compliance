"""AVeriTeC ingest into committed, text-free derived tables ().

``ingest(root)`` refuses unless ``data/raw/averitec/data/{train,dev}.json`` hash to their lines in
``data/raw/MANIFEST.expected.sha256``. It parses ``claim_date`` with ``%d-%m-%Y`` only (specification item 3;
null -> undated, an unparsable string is an IngestError naming the claim), checks every label is in
``LABELS``, assigns ``claim_id = averitec/<train|dev>/<idx>`` by file position, groups exact
duplicates (specification item 7: byte-equal UTF-8, no case folding, no Unicode or whitespace normalization;
representative = lowest (split rank, idx); group id = claim_hash of the text) and writes, each through
a temp file and ``os.replace``:

 data/derived/averitec/claims-index.json one row per claim, ids/dates/labels/hashes/flags only
 data/derived/averitec/labels.csv claim_id,label
 data/derived/averitec/duplicates.json exact-duplicate groups (ids, labels, representative, flags)
 data/derived/averitec/datasheet-stats.json counts for the Art. 10 datasheet (keys of
 aac.data.fixtures.datasheet_stats plus additive ones)

No claim text, question, answer or store text is ever written (CC BY-NC 4.0). ``load_claims`` returns
the text at run time only, in the same ``{claim_id: {"text", "date"}}`` shape as
``aac.data.fixtures.load_claims``.

stdlib + aac.run.hashing + aac.data.verify only; never imports torch.
"""
from __future__ import annotations

import csv
import io
import json
import os
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from aac.data.fixtures import LABELS
from aac.data.verify import EXPECTED_RAW, parse_expected
from aac.run.hashing import claim_hash, dumps_sorted, sha256_file

SPLITS = ("train", "dev")
SPLIT_RANK = {"train": 0, "dev": 1}
RAW_REL = Path("data/raw/averitec/data")
OUT_REL = Path("data/derived/averitec")
DATE_FORMAT = "%d-%m-%Y"
META_FIELDS = ("speaker", "reporting_source", "location_ISO_code", "original_claim_url")
TOP_N = 10


class IngestError(ValueError):
    """The raw files fail a hash, date or label check; nothing is written."""


def claim_id(split: str, idx: int) -> str:
    return f"averitec/{split}/{idx}"


def parse_claim_id(cid: str) -> tuple[str, int]:
    ds, split, idx = cid.split("/")
    if ds != "averitec" or split not in SPLIT_RANK:
        raise ValueError(f"not an AVeriTeC claim id: {cid!r}")
    return split, int(idx)


def order_key(cid: str) -> tuple[int, int]:
    """(split rank, idx): train before dev, then file position."""
    split, idx = parse_claim_id(cid)
    return SPLIT_RANK[split], idx


def parse_date(value, cid: str) -> str | None:
    """specification item 3: `%d-%m-%Y` only; None -> undated; anything else that fails is an IngestError."""
    if value is None:
        return None
    if not isinstance(value, str):
        raise IngestError(f"{cid}: claim_date is {type(value).__name__}, not a string or null")
    try:
        return datetime.strptime(value, DATE_FORMAT).date().isoformat()
    except ValueError as e:
        raise IngestError(f"{cid}: claim_date does not parse with {DATE_FORMAT}") from e


def read_raw(raw_dir: str | Path) -> dict[str, list[dict]]:
    raw_dir = Path(raw_dir)
    out = {}
    for split in SPLITS:
        with open(raw_dir / f"{split}.json", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise IngestError(f"{split}.json: top level is not a list")
        out[split] = data
    return out


def check_raw_hashes(raw_dir: str | Path, expected: dict[str, str]) -> None:
    for split in SPLITS:
        rel = f"averitec/data/{split}.json"
        if rel not in expected:
            raise IngestError(f"{rel} has no line in {EXPECTED_RAW}; refusing to ingest")
        actual = sha256_file(Path(raw_dir) / f"{split}.json")
        if actual != expected[rel]:
            raise IngestError(f"{rel}: sha256 {actual} != expected {expected[rel]}; refusing to ingest")


def build_index(raw: dict[str, list[dict]]) -> tuple[list[dict], dict[str, list[str]]]:
    """Pure: raw claims -> (index rows sorted by (split rank, idx), {group_id: member ids} for groups of >= 2)."""
    rows = []
    by_text: dict[str, list[str]] = {}
    for split in SPLITS:
        for idx, c in enumerate(raw[split]):
            cid = claim_id(split, idx)
            text = c.get("claim")
            if not isinstance(text, str):
                raise IngestError(f"{cid}: claim is {type(text).__name__}, not a string")
            label = c.get("label")
            if label not in LABELS:
                raise IngestError(f"{cid}: label not in LABELS")
            rows.append({
                "claim_id": cid, "split": split, "idx": idx, "date": parse_date(c.get("claim_date"), cid),
                "label": label, "claim_hash": claim_hash(text), "empty_text": text == "",
            })
            by_text.setdefault(text, []).append(cid)
    groups = {}
    for text, members in by_text.items():
        if len(members) > 1:
            groups[claim_hash(text)] = sorted(members, key=order_key)
    member_group = {cid: gid for gid, ms in groups.items() for cid in ms}
    removed = {cid for ms in groups.values() for cid in ms[1:]}
    for r in rows:
        r["duplicate_group"] = member_group.get(r["claim_id"])
        r["removed_as_duplicate"] = r["claim_id"] in removed
    rows.sort(key=lambda r: (SPLIT_RANK[r["split"]], r["idx"]))
    return rows, groups


def duplicates_doc(rows: list[dict], groups: dict[str, list[str]]) -> dict:
    """{"groups": [...]} sorted by representative (split rank, idx); ids, labels and hashes only."""
    label = {r["claim_id"]: r["label"] for r in rows}
    out = []
    for gid, members in groups.items():
        labels = [label[c] for c in members]
        splits_ = {parse_claim_id(c)[0] for c in members}
        out.append({"group_id": gid, "members": members, "labels": labels, "representative": members[0],
                    "conflicting": len(set(labels)) > 1, "spans_train_dev": splits_ == set(SPLITS)})
    out.sort(key=lambda g: order_key(g["representative"]))
    return {"groups": out}


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and v.strip() == "")


def _top(values: list, n: int = TOP_N) -> list[dict]:
    """Top-n non-blank values by count (ties by value), then one `other` bucket with the remainder."""
    c = Counter(v for v in values if not _blank(v))
    ranked = sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))
    top = [{"value": k, "count": v} for k, v in ranked[:n]]
    top.append({"value": "other", "count": sum(v for _, v in ranked[n:])})
    return top


def _summary(xs: list[int]) -> dict:
    return {"n": len(xs), "min": min(xs), "max": max(xs), "mean": round(statistics.fmean(xs), 6),
            "median": float(statistics.median(xs))}


def datasheet_stats(raw: dict[str, list[dict]], rows: list[dict], dup: dict) -> dict:
    """Counts only (no text leaves this function). Keeps the keys pack/datasheet.py reads from the fixture
 stats (n_claims .. n_non_ascii_texts) and adds the composition and gap-audit keys."""
    texts = [c["claim"] for s in SPLITS for c in raw[s]]
    dated = [r["date"] for r in rows if r["date"] is not None]
    counts = {lab: sum(1 for r in rows if r["label"] == lab) for lab in LABELS}
    by_split = {s: {lab: sum(1 for r in rows if r["split"] == s and r["label"] == lab) for lab in LABELS} for s in SPLITS}
    groups = dup["groups"]
    date_of = {r["claim_id"]: r["date"] for r in rows}
    decades = Counter(f"{d[:3]}0s" for d in dated)
    claims_all = [c for s in SPLITS for c in raw[s]]
    q_per_claim = [len(c.get("questions") or []) for c in claims_all]
    a_per_q = [len(q.get("answers") or []) for c in claims_all for q in (c.get("questions") or [])]
    return {
        "n_claims": len(rows),
        "n_claims_by_split": {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS},
        "n_dated": len(dated),
        "n_undated": len(rows) - len(dated),
        "n_undated_by_split": {s: sum(1 for r in rows if r["split"] == s and r["date"] is None) for s in SPLITS},
        "date_min": min(dated),
        "date_max": max(dated),
        "label_counts": counts,
        "label_shares": {k: round(v / len(rows), 6) for k, v in counts.items()},
        "label_counts_by_split": by_split,
        "n_duplicate_texts": len(texts) - len(set(texts)),
        "n_non_ascii_texts": sum(1 for t in texts if not t.isascii()),
        "n_duplicate_groups": len(groups),
        "n_duplicate_members_removed": sum(len(g["members"]) - 1 for g in groups),
        "n_conflicting_groups": sum(1 for g in groups if g["conflicting"]),
        "n_groups_spanning_train_dev": sum(1 for g in groups if g["spans_train_dev"]),
        "n_groups_with_differing_dates": sum(1 for g in groups if len({date_of[c] for c in g["members"]}) > 1),
        "empty_text_ids": [r["claim_id"] for r in rows if r["empty_text"]],
        "n_blank_nonempty_texts": sum(1 for t in texts if t != "" and t.strip() == ""),
        "claim_date_decade_histogram": dict(sorted(decades.items())),
        "null_or_empty_counts": {s: {f: sum(1 for c in raw[s] if _blank(c.get(f))) for f in META_FIELDS} for s in SPLITS},
        "top_location_ISO_code": _top([c.get("location_ISO_code") for c in claims_all]),
        "top_reporting_source": _top([c.get("reporting_source") for c in claims_all]),
        "questions_per_claim": _summary(q_per_claim),
        "answers_per_question": _summary(a_per_q),
    }


def _write_atomic(path: Path, text: str) -> None:
    """tmp + os.replace; a failed write (non-ASCII value, disk full, Ctrl-C) leaves no <name>.tmp behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def labels_csv(rows: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["claim_id", "label"])
    for r in rows:
        w.writerow([r["claim_id"], r["label"]])
    return buf.getvalue()


def ingest(root: str | Path, *, raw_dir: str | Path | None = None, out_dir: str | Path | None = None,
           expected: dict[str, str] | None = None) -> dict[str, Path]:
    """Hash gate, then parse and write the derived tables. Returns {name: path} of the files written."""
    root = Path(root)
    raw_dir = Path(raw_dir) if raw_dir is not None else root / RAW_REL
    out_dir = Path(out_dir) if out_dir is not None else root / OUT_REL
    if expected is None:
        expected = parse_expected(root / EXPECTED_RAW)
    check_raw_hashes(raw_dir, expected)
    raw = read_raw(raw_dir)
    rows, groups = build_index(raw)
    dup = duplicates_doc(rows, groups)
    written = {}
    outputs = {
        "claims-index.json": dumps_sorted(rows),
        "labels.csv": labels_csv(rows),
        "duplicates.json": dumps_sorted(dup),
        "datasheet-stats.json": dumps_sorted(datasheet_stats(raw, rows, dup)),
    }
    for name, text in outputs.items():
        _write_atomic(out_dir / name, text)
        written[name] = out_dir / name
    return written


def load_index(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def load_claims(claims_dir: str | Path) -> dict[str, dict]:
    """Raw claim text keyed by claim_id, read at run time from data/raw/averitec/data/{train,dev}.json.

 Same shape as aac.data.fixtures.load_claims: {claim_id: {"claim_id", "text", "date"}}. The text is
 returned to the caller and never written by this module."""
    raw = read_raw(claims_dir)
    out = {}
    for split in SPLITS:
        for idx, c in enumerate(raw[split]):
            cid = claim_id(split, idx)
            out[cid] = {"claim_id": cid, "text": c["claim"], "date": parse_date(c.get("claim_date"), cid)}
    return out


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import root as project_root

    root = project_root()
    try:
        written = ingest(root)
    except IngestError as e:
        print(f"ingest: FAIL: {e}", file=sys.stderr)
        return 1
    rows = load_index(written["claims-index.json"])
    n = {s: sum(1 for r in rows if r["split"] == s) for s in SPLITS}
    n_undated = sum(1 for r in rows if r["date"] is None)
    n_groups = len({r["duplicate_group"] for r in rows if r["duplicate_group"]})
    n_removed = sum(1 for r in rows if r["removed_as_duplicate"])
    n_empty = sum(1 for r in rows if r["empty_text"])
    print(f"ingest: train {n['train']} dev {n['dev']} undated {n_undated} duplicate_groups {n_groups} "
          f"removed_as_duplicate {n_removed} empty_text {n_empty}")
    for name, p in sorted(written.items()):
        print(f"ingest: wrote {p.relative_to(root)} sha256={sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
