"""The tiny synthetic AVeriTeC world : the shared CPU fixture of every Phase 3 plan.

THIS WORLD IS A SYNTHETIC TEST FIXTURE. Every claim, question, answer, URL and store sentence is invented
text produced by the code below; nothing here is read from, or derived from, the real AVeriTeC data. Its split
assignment is an explicit synthetic rule chosen so that each partition is non-empty at this size. It is NOT
the specification split rule (specification items 4-8) and no result on this world says anything about AVeriTeC.

`make_world(out_dir)` writes, byte-identical on every call (no clock, no host, no randomness):

 averitec/data/train.json, averitec/data/dev.json raw AVeriTeC field names (24 train + 8 dev claims)
 averitec/claims.json fixture format [{claim_id, date, text}] (ISO dates)
 averitec/labels.csv claim_id,label
 splits/tinyworld-chrono.json, splits/tinyworld-random.json
 keys of data/derived/splits/averitec-chrono-v1.json
 data_store/knowledge_store/dev_knowledge_store.zip members output_dev/<idx>.json
 data_store/knowledge_store/train/train_0_999.zip members <idx>.json
 domains.txt the format of configs/factchecker-domains.txt
 MANIFEST.sha256 sha256sum format over every other file, written last

Each claim's store holds 6 documents of 12 sentences. Document 0's URL equals (after kstore.normalise_url) the
source_url of the claim's Boolean answer; document 1's URL path carries a date after the claim date; document 2
sits on a host listed in domains.txt; document 3's URL path carries a date before the claim date; documents 4
and 5 are plain. Every document holds one blank sentence and one exact repeat of an earlier sentence (inside
document 0 for document 0, from the previous document otherwise), so the blank, duplicate, host-filter and
flag rules of aac.data.shortlist are all exercised.

CLI: `python -m aac.data.tinyworld make --out DIR`. stdlib + aac.run.hashing only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from datetime import date, timedelta
from pathlib import Path

from aac.run import hashing

LABELS = ("Supported", "Refuted", "Not Enough Evidence", "Conflicting Evidence/Cherrypicking")
N_TRAIN, N_DEV = 24, 8
N_DOCS, N_SENTS = 6, 12
SPLIT_SIZES = {"train": 14, "val_fuse": 4, "val_cal": 4, "test": 10}
STORE_REL = Path("data_store/knowledge_store")
DEV_ZIP = STORE_REL / "dev_knowledge_store.zip"
TRAIN_ZIP = STORE_REL / "train" / "train_0_999.zip"
FACTCHECK_HOSTS = ("factdesk.example", "truthwatch.example", "verify-now.example")
# Label of the k-th claim in (split rank, idx) order: Refuted is the largest class (14 of 32).
_LABEL_CYCLE = (1, 0, 1, 2, 1, 3, 1, 0)

_SUBJECTS = ("the harbour council", "the regional minister", "the water agency", "the ferry company",
             "the city mayor", "the health committee", "the grain regulator", "the coastal senator")
_TOPICS = ("bridge", "orchard", "granite", "lantern", "copper", "meadow", "beacon", "thistle", "canal", "pylon",
           "saffron", "gravel", "tunnel", "marble", "heron", "cobalt")
_PLACES = ("Northport", "Eastvale", "Southmere", "Westbrook", "Highfield", "Lowmoor", "Riverton", "Stonebay")
_FILLER = ("residents", "officials", "reported", "figures", "quarterly", "survey", "budget", "contract", "workers",
           "market", "transport", "school", "weather", "festival", "museum", "library", "railway", "factory",
           "harvest", "tourism", "housing", "energy", "election", "hospital")


def _claims_order() -> list[tuple[str, int]]:
    return [("train", i) for i in range(N_TRAIN)] + [("dev", i) for i in range(N_DEV)]


def _cid(split: str, idx: int) -> str:
    return f"averitec/{split}/{idx}"


def _claim_date(k: int) -> date:
    """Distinct, increasing dates across 2020 (k = position in (split rank, idx) order)."""
    return date(2020, 1, 5) + timedelta(days=11 * k)


def _claim_text(k: int) -> str:
    subj = _SUBJECTS[k % len(_SUBJECTS)]
    t1, t2 = _TOPICS[k % len(_TOPICS)], _TOPICS[(k * 5 + 3) % len(_TOPICS)]
    place = _PLACES[(k * 3) % len(_PLACES)]
    return f"Tinyworld claim {k}: {subj} said {t1} and {t2} prices in {place} doubled in 2020"


def _sentence(k: int, d: int, s: int) -> str:
    """Invented store sentence; about half of them reuse the claim's topic words so BM25 scores vary."""
    t1, t2 = _TOPICS[k % len(_TOPICS)], _TOPICS[(k * 5 + 3) % len(_TOPICS)]
    place = _PLACES[(k * 3) % len(_PLACES)]
    words = [_FILLER[(k * 7 + d * 5 + s * 3 + j) % len(_FILLER)] for j in range(3 + (k + d + s) % 5)]
    if (d + s) % 2 == 0:
        words.insert(1, t1)
    if (k + s) % 3 == 0:
        words.append(place)
    if (d * s) % 4 == 1:
        words.append(t2)
    return f"Store {k} document {d} line {s}: " + " ".join(words) + "."


def _answer_url(k: int) -> str:
    return f"https://news-{k % 4}.example/stories/claim-{k}/evidence"


def _doc_url(k: int, d: int, claim_date: date) -> str:
    if d == 0:
        # same page as the Boolean answer's source_url once normalized (www., scheme, trailing slash)
        return f"http://www.news-{k % 4}.example/stories/claim-{k}/evidence/"
    if d == 1:
        after = claim_date + timedelta(days=400)
        return f"https://archive-{k % 3}.example/{after.year:04d}/{after.month:02d}/{after.day:02d}/story-{k}"
    if d == 2:
        return f"https://www.{FACTCHECK_HOSTS[k % len(FACTCHECK_HOSTS)]}/checks/claim-{k}"
    if d == 3:
        before = claim_date - timedelta(days=200)
        return f"https://daily-{k % 5}.example/{before.year:04d}-{before.month:02d}-{before.day:02d}/item-{k}"
    return f"https://blog-{(k + d) % 6}.example/posts/{k}-{d}"


def _doc_sentences(k: int, d: int) -> list[str]:
    """12 sentences: 10 distinct, one blank, one exact repeat of an earlier sentence."""
    base = [_sentence(k, d, s) for s in range(N_SENTS - 2)]
    repeat = base[1] if d == 0 else _sentence(k, d - 1, 2)
    sents = base[:4] + ["   "] + base[4:7] + [repeat] + base[7:]
    assert len(sents) == N_SENTS
    return sents


_TYPES = ("gold", "background", "answer", "question", "gpt_question", "background")


def _store_member(k: int, idx: int, claim_date: date) -> bytes:
    lines = []
    for d in range(N_DOCS):
        lines.append(json.dumps({"claim_id": str(idx), "type": _TYPES[d], "query": f"synthetic query {d}",
                                 "url": _doc_url(k, d, claim_date), "url2text": _doc_sentences(k, d)},
                                ensure_ascii=True) + "\n")
    return "".join(lines).encode("ascii")


def _raw_claim(k: int, claim_date: date) -> dict:
    label = LABELS[_LABEL_CYCLE[k % len(_LABEL_CYCLE)]]
    yes = label == "Supported"
    q1_answers = [{"answer": "Yes" if yes else "No", "answer_type": "Boolean",
                   "boolean_explanation": f"Tinyworld explanation {k}: the source states the figure directly.",
                   "source_url": _answer_url(k), "source_medium": "Web text",
                   "cached_source_url": f"https://cache.example/{k}/q1"}]
    q2_answers = [{"answer": f"Tinyworld extract {k}: the price index rose by {k + 3} points.",
                   "answer_type": "Extractive", "source_url": f"https://stats-{k % 3}.example/tables/{k}",
                   "source_medium": "Web text", "cached_source_url": f"https://cache.example/{k}/q2"}]
    if k in (3, 27):   # at least two claims carry one Unanswerable answer
        q2_answers.append({"answer": "No answer could be found.", "answer_type": "Unanswerable",
                           "source_url": f"https://search-{k}.example/none", "source_medium": "Web text",
                           "cached_source_url": f"https://cache.example/{k}/q2u"})
    return {
        "claim": _claim_text(k),
        "required_reannotation": False,
        "label": label,
        "justification": f"Tinyworld justification {k}.",
        "claim_date": claim_date.strftime("%d-%m-%Y"),
        "speaker": _SUBJECTS[k % len(_SUBJECTS)],
        "original_claim_url": f"https://social-{k % 2}.example/post/{k}",
        "cached_original_claim_url": f"https://cache.example/{k}/claim",
        "fact_checking_article": f"https://{FACTCHECK_HOSTS[k % len(FACTCHECK_HOSTS)]}/articles/claim-{k}",
        "reporting_source": "Synthetic",
        "location_ISO_code": "GB",
        "claim_types": ["Numerical Claim"],
        "fact_checking_strategies": ["Numerical Comparison"],
        "questions": [{"question": f"Did the price of {_TOPICS[k % len(_TOPICS)]} double in 2020 (claim {k})?",
                       "answers": q1_answers},
                      {"question": f"By how much did the index rise (claim {k})?", "answers": q2_answers}],
    }


def _sorted_ids(ids: list[str]) -> list[str]:
    rank = {"train": 0, "dev": 1}
    return sorted(ids, key=lambda c: (rank[c.split("/")[1]], int(c.split("/")[2])))


def _split_manifests(order: list[str], dates: dict[str, date]) -> tuple[dict, dict]:
    """Synthetic assignment (not production): chrono test = the 10 latest dates; of the 22 earlier claims (date
 order) positions 3, 8, 13, 18 are val_fuse, 4, 9, 14, 19 val_cal, the rest train. Random: the claims ordered
 by sha256(claim_id) give test (10), val_fuse (4), val_cal (4), train (14)."""
    by_date = sorted(order, key=lambda c: (dates[c], order.index(c)))
    test = by_date[-SPLIT_SIZES["test"]:]
    pre = by_date[:-SPLIT_SIZES["test"]]
    fuse = [c for i, c in enumerate(pre) if i in (3, 8, 13, 18)]
    cal = [c for i, c in enumerate(pre) if i in (4, 9, 14, 19)]
    train = [c for c in pre if c not in fuse and c not in cal]
    counts = {"population_dated": len(order), "test": len(test), "train": len(train),
              "val": len(fuse) + len(cal), "val_fuse": len(fuse), "val_cal": len(cal)}
    chrono = {"name": "tinyworld-chrono", "kind": "chronological",
              "rule": "SYNTHETIC test fixture (aac.data.tinyworld), not the production rule: test = the 10 latest claim "
                      "dates; of the 22 earlier claims in date order, positions 3, 8, 13, 18 are val_fuse, positions "
                      "4, 9, 14, 19 are val_cal, the rest train. Partitions sorted by (split rank, idx).",
              "cutoff_date": dates[pre[-1]].isoformat(),
              "train": _sorted_ids(train), "val_fuse": _sorted_ids(fuse), "val_cal": _sorted_ids(cal),
              "test": _sorted_ids(test), "counts": counts}
    perm = sorted(order, key=lambda c: hashlib.sha256(c.encode("ascii")).hexdigest())
    n_t, n_f, n_c = SPLIT_SIZES["test"], SPLIT_SIZES["val_fuse"], SPLIT_SIZES["val_cal"]
    random_ = {"name": "tinyworld-random", "kind": "random",
               "rule": "SYNTHETIC test fixture (aac.data.tinyworld), not the production rule: claims ordered by "
                       "sha256(claim_id) hex; the first 10 are test, the next 4 val_fuse, the next 4 val_cal, the "
                       "rest train. Partitions sorted by (split rank, idx).",
               "cutoff_date": None,
               "test": _sorted_ids(perm[:n_t]), "val_fuse": _sorted_ids(perm[n_t:n_t + n_f]),
               "val_cal": _sorted_ids(perm[n_t + n_f:n_t + n_f + n_c]),
               "train": _sorted_ids(perm[n_t + n_f + n_c:]), "counts": dict(counts)}
    return chrono, random_


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w") as zf:
            for name in sorted(members):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                zf.writestr(info, members[name])
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _json(obj) -> bytes:
    return (json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True) + "\n").encode("ascii")


def make_world(out_dir) -> dict:
    """Write the world under out_dir (byte-identical on every call); returns a small summary dict."""
    out = Path(out_dir)
    order = _claims_order()
    raw = {"train": [], "dev": []}
    members = {"train": {}, "dev": {}}
    fixture, labels, dates, files = [], [], {}, []
    for k, (split, idx) in enumerate(order):
        cd = _claim_date(k)
        rc = _raw_claim(k, cd)
        raw[split].append(rc)
        cid = _cid(split, idx)
        dates[cid] = cd
        fixture.append({"claim_id": cid, "date": cd.isoformat(), "text": rc["claim"]})
        labels.append((cid, rc["label"]))
        name = f"output_dev/{idx}.json" if split == "dev" else f"{idx}.json"
        members[split][name] = _store_member(k, idx, cd)

    def put(rel: str, data: bytes) -> None:
        _write_bytes(out / rel, data)
        files.append(rel)

    put("averitec/data/train.json", _json(raw["train"]))
    put("averitec/data/dev.json", _json(raw["dev"]))
    put("averitec/claims.json", _json(fixture))
    put("averitec/labels.csv", ("claim_id,label\n" + "".join(f"{c},{lab}\n" for c, lab in labels)).encode("ascii"))
    ids = [_cid(s, i) for s, i in order]
    chrono, random_ = _split_manifests(ids, dates)
    put("splits/tinyworld-chrono.json", _json(chrono))
    put("splits/tinyworld-random.json", _json(random_))
    _write_zip(out / DEV_ZIP, members["dev"])
    files.append(DEV_ZIP.as_posix())
    _write_zip(out / TRAIN_ZIP, members["train"])
    files.append(TRAIN_ZIP.as_posix())
    header = ["# SYNTHETIC fact-checker host list of the tiny world (aac.data.tinyworld); not specification item 18's list.",
              "# Format of configs/factchecker-domains.txt: comment lines, then one lowercase host per line.",
              f"# Hosts: {len(FACTCHECK_HOSTS)}"]
    put("domains.txt", ("\n".join(header + sorted(FACTCHECK_HOSTS)) + "\n").encode("ascii"))
    manifest = "".join(f"{hashing.sha256_file(out / rel)}  {rel}\n" for rel in sorted(files))
    _write_bytes(out / "MANIFEST.sha256", manifest.encode("ascii"))
    return {"out": out, "claims": len(order), "stores": len(order), "files": sorted(files) + ["MANIFEST.sha256"],
            "data_version": hashing.data_version(out / "MANIFEST.sha256")}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m aac.data.tinyworld")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make", help="write the tiny synthetic AVeriTeC world (byte-identical on every call)")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    info = make_world(Path(a.out))
    print(f"tinyworld: {info['claims']} claims, {info['stores']} stores -> {a.out} data_version={info['data_version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
