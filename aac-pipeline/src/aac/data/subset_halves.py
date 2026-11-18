"""Victim-B subset halves of the attack subset .

 python -m aac.data.subset_halves [--root PATH] [--out-dir DIR]

The victim server sorts item ids as strings after the data.attack_subset filter, so one 400-item victim-B run cannot be
cut gracefully to the first 200 ids of the file (04-RESEARCH Pattern 1). Victim B therefore runs on two committed
halves, and cut-2-subset-200 drops whole runs. From data/derived/splits/averitec-attack-subset-v1.json (the parent, read
in FILE order, which is (split rank, idx) order) it writes, next to the parent by default:

 averitec-attack-subset-v1-first200.json ids[0:200] (the cut-2 half; kept when cut-2 fires)
 averitec-attack-subset-v1-last200.json ids[200:400] (dropped whole by cut-2-subset-200)
 averitec-attack-subset-v1-first100.json ids[0:100] (the addition-A cut rule, protocol specification)

Each file: kind, name, rule, source, source_sha256 (sha256 of the parent file's bytes), count, label_counts (from
data/derived/averitec/labels.csv) and ids, written sorted-key with one-space indent (aac.run.hashing.dumps_sorted), LF,
ASCII, through tmp + os.replace. The same parent gives the same bytes. Refuses (nothing written) when the parent's
ids are not unique, its count differs from len(ids), or a slice id has no label."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from aac.run.hashing import dumps_sorted, sha256_file

PARENT = "data/derived/splits/averitec-attack-subset-v1.json"
LABELS = "data/derived/averitec/labels.csv"
# (suffix, start, stop): python slice of the parent's ids in file order
HALVES = (("first200", 0, 200), ("last200", 200, 400), ("first100", 0, 100))


class HalvesRefused(ValueError):
    """The parent subset or the labels do not support a well-defined slice; nothing is written."""


def _labels(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["claim_id"]: r["label"] for r in csv.DictReader(fh)}


def build(parent_path: Path, labels_path: Path, parent_rel: str = PARENT) -> dict[str, dict]:
    """{file name: document} for every half, from the parent subset file and the labels file."""
    parent = json.loads(Path(parent_path).read_text(encoding="utf-8"))
    ids = list(parent["ids"])
    if len(set(ids)) != len(ids) or int(parent["count"]) != len(ids):
        raise HalvesRefused(f"{parent_rel}: ids not unique or count {parent['count']} != {len(ids)}")
    labels = _labels(Path(labels_path))
    digest = sha256_file(Path(parent_path))
    stem = Path(parent_rel).stem
    out: dict[str, dict] = {}
    for suffix, start, stop in HALVES:
        if stop > len(ids):
            raise HalvesRefused(f"{parent_rel}: {len(ids)} ids, slice [{start}:{stop}] out of range")
        part = ids[start:stop]
        missing = [i for i in part if i not in labels]
        if missing:
            raise HalvesRefused(f"{LABELS}: {len(missing)} ids of {stem}-{suffix} have no label")
        counts: dict[str, int] = {}
        for i in part:
            counts[labels[i]] = counts.get(labels[i], 0) + 1
        out[f"{stem}-{suffix}.json"] = {
            "kind": "attack-subset-half",
            "name": f"{stem}-{suffix}",
            "rule": (f"ids[{start}:{stop}] of {parent_rel} in file order ((split rank, idx) order), python slice; "
                     f"labels from {LABELS} "),
            "source": parent_rel,
            "source_sha256": digest,
            "count": len(part),
            "label_counts": dict(sorted(counts.items())),
            "ids": part,
        }
    return out


def _write_atomic(path: Path, text: str) -> None:
    import os

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def write(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    docs = build(root / PARENT, root / LABELS)
    out_dir = Path(out_dir) if out_dir else (root / PARENT).parent
    paths = []
    for name, doc in docs.items():
        _write_atomic(out_dir / name, dumps_sorted(doc))
        paths.append(out_dir / name)
    return paths


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.data.subset_halves", description=__doc__.split("\n", 1)[0])
    ap.add_argument("--root", default=None, help="project root (default: the aac-pipeline checkout)")
    ap.add_argument("--out-dir", default=None, help="output directory (default: the parent's directory)")
    args = ap.parse_args(argv)
    from aac.run.paths import root as project_root

    root = Path(args.root) if args.root else project_root()
    try:
        paths = write(root, Path(args.out_dir) if args.out_dir else None)
    except (HalvesRefused, FileNotFoundError, KeyError) as exc:
        print(f"subset-halves: refused: {exc}", file=sys.stderr)
        return 1
    for p in paths:
        doc = json.loads(p.read_text(encoding="utf-8"))
        print(f"subset-halves: {p.name} count={doc['count']} Refuted={doc['label_counts'].get('Refuted', 0)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
