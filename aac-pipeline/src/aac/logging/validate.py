"""validate_run : every record of every shard against the schema, seq
contiguous 0..n-1 per run, leak scan; writes schema-validation.txt as exactly four
LF-terminated key=value lines (no timestamp)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from aac.logging.leakscan import leakscan, make_shingles
from aac.logging.schema import errors_for, load_validator
from aac.logging.writer import SHARD_DIR, read_shard, shard_bytes

COUNT_KEYS = ("records", "violations", "records_with_claim_text", "records_with_user_id")


@dataclass
class Counts:
    records: int = 0
    violations: int = 0
    records_with_claim_text: int = 0
    records_with_user_id: int = 0

    def add(self, other: "Counts") -> "Counts":
        return Counts(
            self.records + other.records,
            self.violations + other.violations,
            self.records_with_claim_text + other.records_with_claim_text,
            self.records_with_user_id + other.records_with_user_id,
        )

    def as_lines(self) -> str:
        return "".join(f"{k}={getattr(self, k)}\n" for k in COUNT_KEYS)


def list_shards(shard_dir: Path) -> list[Path]:
    return sorted(p for p in Path(shard_dir).glob("part-*.jsonl.gz") if p.is_file())


def validate_shards(shard_paths: list[Path], fixture_texts: set[str] | None = None,
                    validator=None, expect_seq_from: int = 0, shingles: frozenset | None = None) -> Counts:
    """Validate an ordered list of shards that together form ONE run (seq contiguous).
 `validator` and `shingles` (make_shingles(fixture_texts)) may be passed precomputed by a caller that
 validates many shards with the same text set (aac.pack.decision_log); the defaults build them here."""
    validator = validator or load_validator()
    texts = fixture_texts or set()
    shingles = make_shingles(texts) if shingles is None else shingles
    counts = Counts()
    expected = expect_seq_from
    for path in shard_paths:
        raw = shard_bytes(path)
        side = path.with_name(path.name + ".sha256")
        if not side.is_file():  
            raise RuntimeError(f"{path.name}: missing .sha256 sidecar")
        if side.read_text(encoding="ascii").split()[0] != hashlib.sha256(raw).hexdigest():
            raise RuntimeError(f"{path.name}: .sha256 sidecar does not match the decompressed bytes")
        for line in raw.split(b"\n"):
            if not line:
                continue
            rec = json.loads(line)
            counts.records += 1
            bad = bool(errors_for(validator, rec))
            if rec.get("seq") != expected:
                bad = True
            expected = (rec.get("seq") if isinstance(rec.get("seq"), int) else expected) + 1
            if bad:
                counts.violations += 1
            has_text, has_uid = leakscan(rec, texts, shingles=shingles)
            counts.records_with_claim_text += int(has_text)
            counts.records_with_user_id += int(has_uid)
    return counts


def validate_run(run_dir: Path, fixture_texts: set[str] | None = None, validator=None) -> Counts:
    return validate_shards(list_shards(Path(run_dir) / SHARD_DIR), fixture_texts, validator)


def write_counts(path: Path, counts: Counts) -> None:
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        fh.write(counts.as_lines())


def read_counts(path: Path) -> Counts:
    vals = {}
    for line in Path(path).read_text(encoding="ascii").splitlines():
        k, _, v = line.partition("=")
        vals[k] = int(v)
    return Counts(**{k: vals[k] for k in COUNT_KEYS})


class LeakScanUnavailable(RuntimeError):
    """No claim texts could be loaded, so records_with_claim_text would be a number with no
 measurement behind it . The caller must not write or seal the count."""


TEXT_KEYS = ("text", "claim", "evidence")


def _texts_from(path: Path) -> set[str]:
    """String values under TEXT_KEYS of every item in a JSON list/dict-of-items file or a
 headed TSV/CSV file. Raises LeakScanUnavailable on a missing file or unknown layout."""
    import csv

    path = Path(path)
    if not path.is_file():
        raise LeakScanUnavailable(f"{path}: claim/item file not on this host; the leak scan cannot run")
    if path.suffix == ".json":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        items = data.values() if isinstance(data, dict) else data
        rows = [it for it in items if isinstance(it, dict)]
    elif path.suffix in (".tsv", ".csv"):
        with open(path, encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t" if path.suffix == ".tsv" else ",")
            if not set(reader.fieldnames or ()) & set(TEXT_KEYS):
                raise LeakScanUnavailable(f"{path}: no {'/'.join(TEXT_KEYS)} column; register a text loader")
            rows = list(reader)
    else:
        raise LeakScanUnavailable(f"{path}: unsupported file type for the leak scan")
    return {r[k] for r in rows for k in TEXT_KEYS if isinstance(r.get(k), str) and r[k]}


MAX_SCANNED_TEXT_CHARS = 128  # every record string is capped at maxLength 128 (decision-log schema); NOT a text filter


def _real_texts(data: dict, root: Path) -> set[str]:
    """Dataset averitec (operational constraint): the real texts the run could carry.

 data.claims a directory -> aac.data.averitec.load_claims (data/raw/averitec/data); a file -> _texts_from
 data.fc_items / data.rd_items the committed text-free item files -> aac.data.incrediblae.load_texts
 (FC claim + evidence from the hash-checked raw TSV; RD lane-B text, skipped when the private
 lane is absent on this host); any other file -> _texts_from
 Every non-empty text is returned, whatever its length a text longer than
 MAX_SCANNED_TEXT_CHARS cannot sit WHOLE inside a schema-valid record string, but a truncated copy of it can,
 and the 8-token shingle check catches that copy only if the long text is scanned (859 of the 3,568 AVeriTeC
 claims and most FC evidence texts are longer than 128 characters). Raises LeakScanUnavailable when nothing
 is loaded at all."""
    from aac.data import incrediblae
    from aac.data.averitec import load_claims

    loaded: set[str] = set()
    claims = data.get("claims")
    if claims:
        p = Path(root) / claims
        if p.is_dir():
            try:
                loaded |= {c["text"] for c in load_claims(p).values()}
            except (OSError, ValueError, KeyError) as exc:  # missing file, bad JSON or date (IngestError)
                raise LeakScanUnavailable(f"{p}: AVeriTeC claims not loadable on this host ({exc.__class__.__name__}); "
                                          "the leak scan cannot run") from exc
        else:
            loaded |= _texts_from(p)
    for key, task in (("fc_items", "fc"), ("rd_items", "rd")):
        if not data.get(key):
            continue
        p = Path(root) / data[key]
        committed = (Path(root) / incrediblae.ITEMS_REL / f"{task}-attack.items.json").resolve()
        if p.resolve() == committed:
            try:
                rows = incrediblae.load_texts(root, task)
            except incrediblae.LaneBMissing:
                continue  # RD text exists only on cpu-host; a host without lane B has none to leak
            except (OSError, ValueError) as exc:  # missing raw file, hash or layout failure (IngestError)
                raise LeakScanUnavailable(f"{p}: {task.upper()} texts not loadable on this host "
                                          f"({exc.__class__.__name__}); the leak scan cannot run") from exc
            loaded |= {v[k] for v in rows.values() for k in ("text", "evidence") if isinstance(v.get(k), str)}
        else:
            loaded |= _texts_from(p)
    if not loaded:
        raise LeakScanUnavailable(f"dataset {data.get('dataset')!r}: no claim texts loaded from data.claims/"
                                  "fc_items/rd_items; refusing to report records_with_claim_text")
    return {t for t in loaded if t}


def claim_texts_for(resolved: dict, root: Path) -> set[str]:
    """Every claim/item text (and FC evidence) the run read, for EVERY dataset , from
 data.claims, data.fc_items and data.rd_items. Raises LeakScanUnavailable when no text
 is loaded, so `records_with_claim_text=0` is never written without a check behind it.
 Dataset averitec goes through _real_texts (real loaders, every non-empty text); dataset incrediblae
 (external baselines) too, without data.claims; every other dataset (the fixture) takes the path below,
 unchanged."""
    data = resolved.get("data", {})
    if data.get("dataset") == "averitec":
        return _real_texts(data, Path(root))
    if data.get("dataset") == "incrediblae":
        # external BODEGA baselines : data.claims names the raw task directory, which no text loader
        # reads; the texts are those of the committed item files, loaded by the real loaders 
        return _real_texts({k: v for k, v in data.items() if k != "claims"}, Path(root))
    texts: set[str] = set()
    for key in ("claims", "fc_items", "rd_items"):
        if data.get(key):
            texts |= _texts_from(Path(root) / data[key])
    if not texts:
        raise LeakScanUnavailable(f"dataset {data.get('dataset')!r}: no claim texts loaded from data.claims/"
                                  "fc_items/rd_items; refusing to report records_with_claim_text")
    return texts


fixture_texts_for = claim_texts_for  # historical name kept for callers
