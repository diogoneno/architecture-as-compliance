"""AVeriTeC split manifests under the signed specification rules ().

Manifests are built from the committed, text-free ``data/derived/averitec/claims-index.json`` and the
rule values in ``configs/data/splits.yaml``; never from raw text. Every manifest records its rule, its
counts, the sha256 of the config and of the claim index, and an ``assertions`` block; the writer
refuses to write a manifest whose assertions are not all true.

Split criteria followed: 4 (cutoff = latest date with >= 20 percent of dated claims strictly later;
integer test ``5 * later >= n``; cutoff-day claims in train/val), 5 (every ordering by (parsed date,
split rank train=0 dev=1, idx)), 6 (undated excluded and listed), 7 (removed duplicates excluded and
listed with their group id), 8 (val_fuse/val_cal by ``int.from_bytes(sha256(claim_id), 'big') % 2``).
Design decisions 3 (empty claim excluded) and 4 (val = ``(h >> 64) % 5 == 0``) are config values,
pending confirmation at the GPU checkpoint.

stdlib + numpy + yaml + aac.run.hashing only; never imports torch.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

from aac.run.hashing import derive_seed, dumps_sorted, sha256_file

SPLIT_RANK = {"train": 0, "dev": 1}
CONFIG_REL = Path("configs/data/splits.yaml")
INDEX_REL = Path("data/derived/averitec/claims-index.json")
OUT_REL = Path("data/derived/splits")
PARTITIONS = ("train", "val_fuse", "val_cal", "test")


class SplitAssertionError(RuntimeError):
    """A manifest assertion is false; the manifest is not written."""


def load_cfg(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _h(cid: str) -> int:
    return int.from_bytes(hashlib.sha256(cid.encode("utf-8")).digest(), "big")


def in_val(cid: str, cfg: dict) -> bool:
    r = cfg["val_rule"]
    return (_h(cid) >> int(r["shift"])) % int(r["mod"]) == int(r["keep"])


def fuse_or_cal(cid: str, cfg: dict) -> str:
    r = cfg["fuse_cal_rule"]
    return "val_fuse" if _h(cid) % int(r["mod"]) == int(r["val_fuse_value"]) else "val_cal"


def _rank_idx(row: dict) -> tuple[int, int]:
    return SPLIT_RANK[row["split"]], int(row["idx"])


def prereg_key(row: dict) -> tuple[str, int, int]:
    """specification item 5: (parsed date, split rank, idx). Only defined for dated rows."""
    return row["date"], SPLIT_RANK[row["split"]], int(row["idx"])


def _ge_min(later: int, n: int, cfg: dict) -> bool:
    """later / n >= numerator / denominator, in integers (5 * later >= n for 1/5)."""
    m = cfg["test_share_min"]
    return int(m["denominator"]) * later >= int(m["numerator"]) * n


def chrono_cutoff(dates: list[str], cfg: dict | None = None) -> str:
    """specification item 4 over the dated population: the latest date d (among the claim dates) such that
 ``5 * #(dates > d) >= n``. Raises ValueError for an empty population or when no date qualifies."""
    cfg = cfg or {"test_share_min": {"numerator": 1, "denominator": 5}}
    ds = sorted(dates)
    n = len(ds)
    if n == 0:
        raise ValueError("chrono_cutoff: zero dated claims; no cutoff exists")
    cutoff = None
    for d in sorted(set(ds)):
        later = n - bisect.bisect_right(ds, d)
        if _ge_min(later, n, cfg):
            cutoff = d
    if cutoff is None:
        raise ValueError("chrono_cutoff: no date leaves the minimum test share strictly later")
    return cutoff


def population(index: list[dict], cfg: dict) -> dict:
    """R1 population and the listed exclusions. An id appears in every exclusion list it qualifies for."""
    excl_dup = [{"claim_id": r["claim_id"], "group_id": r["duplicate_group"]}
                for r in sorted(index, key=_rank_idx) if r["removed_as_duplicate"]]
    excl_empty = [r["claim_id"] for r in sorted(index, key=_rank_idx) if r["empty_text"]] if cfg["exclude_empty_text"] else []
    excl_undated = [r["claim_id"] for r in sorted(index, key=_rank_idx) if r["date"] is None]
    out = set(excl_empty) | set(excl_undated) | {e["claim_id"] for e in excl_dup}
    dated = sorted((r for r in index if r["claim_id"] not in out), key=prereg_key)
    return {"dated": dated, "excluded_undated": excl_undated, "excluded_duplicates": excl_dup,
            "excluded_empty_text": excl_empty}


def _sorted_ids(ids, by_id: dict) -> list[str]:
    return sorted(ids, key=lambda c: _rank_idx(by_id[c]))


def _common_assertions(parts: dict[str, list[str]], pop_ids: set[str], by_id: dict, cfg: dict, n_all: int,
                       excluded: set[str]) -> dict[str, bool]:
    all_ids = [c for p in PARTITIONS for c in parts[p]]
    flat = set(all_ids)
    return {
        "partitions_disjoint": len(all_ids) == len(flat),
        "partitions_cover_population": flat == pop_ids,
        "no_undated": all(by_id[c]["date"] is not None for c in flat),
        "no_removed_duplicate": not any(by_id[c]["removed_as_duplicate"] for c in flat),
        # with exclude_empty_text false the rule is "no exclusion", which is honored trivially
        "no_empty_text": (not cfg["exclude_empty_text"]) or not any(by_id[c]["empty_text"] for c in flat),
        "all_claims_accounted": len(flat | excluded) == n_all,
        "test_share_ge_min": _ge_min(len(parts["test"]), len(pop_ids), cfg),
    }


def _counts(parts: dict[str, list[str]], pop: dict) -> dict[str, int]:
    c = {p: len(parts[p]) for p in PARTITIONS}
    c["val"] = c["val_fuse"] + c["val_cal"]
    c["population_dated"] = len(pop["dated"])
    c["excluded_undated"] = len(pop["excluded_undated"])
    c["excluded_duplicates"] = len(pop["excluded_duplicates"])
    c["excluded_empty_text"] = len(pop["excluded_empty_text"])
    return c


def _excluded_ids(pop: dict) -> set[str]:
    return set(pop["excluded_undated"]) | set(pop["excluded_empty_text"]) | {e["claim_id"] for e in pop["excluded_duplicates"]}


def build_chrono(index: list[dict], cfg: dict, *, splits_config_sha256: str, claims_index_sha256: str,
                 name: str = "averitec-chrono-v1") -> dict:
    by_id = {r["claim_id"]: r for r in index}
    pop = population(index, cfg)
    dated = pop["dated"]
    if not dated:
        raise ValueError("build_chrono: zero dated claims in the population")
    n = len(dated)
    cutoff = chrono_cutoff([r["date"] for r in dated], cfg)
    test = [r["claim_id"] for r in dated if r["date"] > cutoff]
    pre = [r["claim_id"] for r in dated if r["date"] <= cutoff]
    parts = {p: [] for p in PARTITIONS}
    parts["test"] = test
    for cid in pre:
        parts[fuse_or_cal(cid, cfg) if in_val(cid, cfg) else "train"].append(cid)
    parts = {p: _sorted_ids(v, by_id) for p, v in parts.items()}
    trainval_dates = [by_id[c]["date"] for p in ("train", "val_fuse", "val_cal") for c in parts[p]]
    test_dates = [by_id[c]["date"] for c in parts["test"]]
    later_dates = sorted({r["date"] for r in dated if r["date"] > cutoff})
    first_test_date = later_dates[0] if later_dates else None
    later_than_next = sum(1 for r in dated if first_test_date is not None and r["date"] > first_test_date)
    assertions = _common_assertions(parts, {r["claim_id"] for r in dated}, by_id, cfg, len(index), _excluded_ids(pop))
    assertions.update({
        "max_trainval_date_le_cutoff": bool(trainval_dates) and max(trainval_dates) <= cutoff,
        "cutoff_lt_min_test_date": bool(test_dates) and cutoff < min(test_dates),
        "next_date_share_lt_min": first_test_date is not None and not _ge_min(later_than_next, n, cfg),
    })
    return {
        "name": name,
        "kind": "chronological",
        "rule": ("Population: dated AVeriTeC train+dev claims minus removed exact-byte duplicates (criterion 7) and, per "
                 "exclude_empty_text, empty claim strings (design decision 3); undated claims excluded and "
                 "listed (criterion 6). cutoff_date = the latest claim date d with denominator * #(dated > d) >= "
                 "numerator * n_dated (criterion 4, test_share_min 1/5, integer arithmetic). test = dated strictly after "
                 "the cutoff; claims on the cutoff day are train/val. val = pre-cutoff ids with "
                 "(int.from_bytes(sha256(claim_id), 'big') >> 64) % 5 == 0 (val_rule, design decision 4); "
                 "val_fuse if int.from_bytes(sha256(claim_id), 'big') % 2 == 0 else val_cal (criterion 8); train = the "
                 "rest. Partitions sorted by (split rank train=0 dev=1, idx)."),
        "splits_config_sha256": splits_config_sha256,
        "claims_index_sha256": claims_index_sha256,
        "cutoff_date": cutoff,
        "first_test_date": first_test_date,
        **parts,
        "excluded_undated": pop["excluded_undated"],
        "excluded_duplicates": pop["excluded_duplicates"],
        "excluded_empty_text": pop["excluded_empty_text"],
        "counts": _counts(parts, pop),
        "test_share": round(len(test) / n, 6),
        "assertions": assertions,
    }


def build_random(index: list[dict], chrono: dict, cfg: dict, *, splits_config_sha256: str, claims_index_sha256: str,
                 name: str = "averitec-random-v1") -> dict:
    """Same population as chrono; a derive_seed(random_seed_key) permutation of it in specification item 5 order;
 test and val counts copied from chrono; val partitioned by the item-8 hash rule (not by position)."""
    by_id = {r["claim_id"]: r for r in index}
    pop = population(index, cfg)
    dated = pop["dated"]
    if not dated:
        raise ValueError("build_random: zero dated claims in the population")
    seed = derive_seed(cfg["random_seed_key"])
    perm = np.random.default_rng(seed).permutation(len(dated))
    order = [dated[int(i)]["claim_id"] for i in perm]
    n_test, n_val = int(chrono["counts"]["test"]), int(chrono["counts"]["val"])
    parts = {p: [] for p in PARTITIONS}
    parts["test"] = order[:n_test]
    for cid in order[n_test:n_test + n_val]:
        parts[fuse_or_cal(cid, cfg)].append(cid)
    parts["train"] = order[n_test + n_val:]
    parts = {p: _sorted_ids(v, by_id) for p, v in parts.items()}
    pop_ids = {r["claim_id"] for r in dated}
    chrono_ids = {c for p in PARTITIONS for c in chrono[p]}
    assertions = _common_assertions(parts, pop_ids, by_id, cfg, len(index), _excluded_ids(pop))
    assertions["counts_equal_chrono"] = (len(parts["test"]) == n_test
                                         and len(parts["val_fuse"]) + len(parts["val_cal"]) == n_val)
    assertions["same_population_as_chrono"] = pop_ids == chrono_ids
    return {
        "name": name,
        "kind": "random",
        "rule": ("Same population and exclusions as averitec-chrono-v1. The population, ordered by (parsed date, split "
                 "rank, idx) (criterion 5), is permuted by numpy.random.default_rng(derive_seed(random_seed_key)); the first "
                 "n_test ids are test, the next n_val are val (n_test and n_val copied from averitec-chrono-v1), the rest "
                 "train; val is split val_fuse/val_cal by int.from_bytes(sha256(claim_id), 'big') % 2 (criterion 8). "
                 "Partitions sorted by (split rank, idx)."),
        "splits_config_sha256": splits_config_sha256,
        "claims_index_sha256": claims_index_sha256,
        "source_counts_from": chrono["name"],
        "random_seed_key": cfg["random_seed_key"],
        "random_seed": seed,
        **parts,
        "excluded_undated": pop["excluded_undated"],
        "excluded_duplicates": pop["excluded_duplicates"],
        "excluded_empty_text": pop["excluded_empty_text"],
        "counts": _counts(parts, pop),
        "test_share": round(len(parts["test"]) / len(dated), 6),
        "assertions": assertions,
    }


def build_attack_subset(chrono: dict, cfg: dict, index: list[dict], *, splits_config_sha256: str,
                        claims_index_sha256: str, name: str = "averitec-attack-subset-v1") -> dict:
    """The first min(size, n_test) ids of a derive_seed(seed_key) permutation of the chrono test ids taken
 in specification item 5 order (all labels); written sorted by (split rank, idx)."""
    by_id = {r["claim_id"]: r for r in index}
    test = sorted(chrono["test"], key=lambda c: prereg_key(by_id[c]))
    size = int(cfg["attack_subset"]["size"])
    seed = derive_seed(cfg["attack_subset"]["seed_key"])
    perm = np.random.default_rng(seed).permutation(len(test))
    ids = _sorted_ids([test[int(i)] for i in perm[:size]], by_id)
    return {
        "name": name,
        "kind": "attack-subset",
        "rule": ("The first min(size, n_test) ids of numpy.random.default_rng(derive_seed(seed_key)).permutation over the "
                 "averitec-chrono-v1 test ids ordered by (parsed date, split rank, idx) (criterion 5); all labels; sorted by "
                 "(split rank, idx)."),
        "splits_config_sha256": splits_config_sha256,
        "claims_index_sha256": claims_index_sha256,
        "source_split": chrono["name"],
        "source_partition": "test",
        "subset_seed_key": cfg["attack_subset"]["seed_key"],
        "subset_seed": seed,
        "size": size,
        "ids": ids,
        "count": len(ids),
        "label_counts": dict(sorted({lab: sum(1 for c in ids if by_id[c]["label"] == lab)
                                     for lab in {by_id[c]["label"] for c in ids}}.items())),
        "assertions": {
            "ids_in_chrono_test": set(ids) <= set(chrono["test"]),
            "ids_unique": len(ids) == len(set(ids)),
            "count_is_min_size_n_test": len(ids) == min(size, len(chrono["test"])),
        },
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


def write_manifest(path: str | Path, doc: dict) -> Path:
    """Refuse (SplitAssertionError, nothing written) unless every assertion is exactly True."""
    false = sorted(k for k, v in doc["assertions"].items() if v is not True)
    if not doc["assertions"] or false:
        raise SplitAssertionError(f"{doc.get('name')}: assertions not true: {false or ['<empty block>']}; not written")
    path = Path(path)
    _write_atomic(path, dumps_sorted(doc))
    return path


def build_all(root: str | Path) -> dict[str, Path]:
    root = Path(root)
    cfg = load_cfg(root / CONFIG_REL)
    index_path = root / INDEX_REL
    with open(index_path, encoding="utf-8") as fh:
        index = json.load(fh)
    shas = {"splits_config_sha256": sha256_file(root / CONFIG_REL), "claims_index_sha256": sha256_file(index_path)}
    chrono = build_chrono(index, cfg, **shas)
    rnd = build_random(index, chrono, cfg, **shas)
    sub = build_attack_subset(chrono, cfg, index, **shas)
    out = root / OUT_REL
    # nothing is written unless all three manifests carry all-true assertions
    for doc in (chrono, rnd, sub):
        false = sorted(k for k, v in doc["assertions"].items() if v is not True)
        if false:
            raise SplitAssertionError(f"{doc['name']}: assertions not true: {false}; nothing written")
    return {doc["name"]: write_manifest(out / f"{doc['name']}.json", doc) for doc in (chrono, rnd, sub)}


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import root as project_root

    root = project_root()
    try:
        written = build_all(root)
    except (SplitAssertionError, ValueError) as e:
        print(f"splits: FAIL: {e}", file=sys.stderr)
        return 1
    for name, p in sorted(written.items()):
        doc = json.loads(p.read_text(encoding="utf-8"))
        extra = f" cutoff_date {doc['cutoff_date']} test_share {doc['test_share']:.6f}" if "cutoff_date" in doc else ""
        counts = doc.get("counts", {"count": doc.get("count")})
        print(f"splits: {name}: counts {json.dumps(counts, sort_keys=True)}{extra}")
        print(f"splits: wrote {p.relative_to(root)} sha256={sha256_file(p)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
