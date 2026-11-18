"""InCrediblAE FC and RD ingest: text-free item files, attack manifests and RD's pseudonymized lane B
(, ).

Two lanes for RD (02-RESEARCH "RD two lanes"):
 lane A the hash-checked raw ``data/raw/incrediblae/RD/*.tsv``, read only by BODEGA's unchanged scripts
  and never copied to any output;
 lane B ``data/derived/private/rd-attack.pseudo.jsonl`` (gitignored, mode 0600 in a 0700 directory):
 one ``{item_id, row, label, text}`` per line, sorted by row, where ``text`` is
 ``pseudonymise(raw_text, key)``. Deterministic given ``secrets/rd-hmac.key``.

``pseudonymise`` applies, in this order (hmac8 = first 8 hex of HMAC-SHA256(key, value)):
 1. URLs (scheme http/https, any case) -> ``https://<host lowercased>/<hmac8 of the full URL>``;
 trailing sentence punctuation is kept outside the URL; userinfo and port are dropped;
 2. ``@`` + word characters -> ``user_<hmac8 of the lowercased handle, '@' included>``;
 3. every remaining ``\\b\\d{15,20}\\b`` run -> ``id_<hmac8>``.
The output is then checked against ``aac.logging.leakscan.USER_ID_RE``; a match is an IngestError.

Committed files hold no text and no organiser id (FC ids repeat, RD ids embed status ids; Pitfall 8):
 data/derived/incrediblae/fc-attack.items.json [{item_id, row, label, claim_hash, evidence_hash}]
 data/derived/incrediblae/rd-attack.items.json [{item_id, row, label, text_hash}] (hash of the lane-B text)
 data/derived/splits/incrediblae-{fc,rd}-attack.json every row of the hash-verified attack.tsv
 data/derived/incrediblae-stats.json counts only, for the datasheet 
item_id = ``incrediblae/<fc|rd>/attack/<row>`` with row the 0-based data row in file order.

Rows are read the way BODEGA's ``victims/transformer.py`` reads them (``line.split('\\t')``, field
``strip()``, then its ``\\n`` / ``\\t`` / ``\\\\`` unescape), so lane B and the hashes describe exactly what
a BODEGA victim sees. A row whose column count is not 4 (FC: label, id, evidence, claim) or 3 (RD:
label, id, text), or whose label is not ``0``/``1``, stops ingest naming the 1-based line.

stdlib + aac.run.hashing + aac.data.verify + aac.logging.leakscan only; never imports torch.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import math
import os
import re
import secrets
import stat
import statistics
import sys
from pathlib import Path

from aac.data.verify import EXPECTED_RAW, parse_expected
from aac.logging.leakscan import USER_ID_RE
from aac.run.hashing import claim_hash, dumps_sorted, sha256_file

TASKS = ("FC", "RD")
COLUMNS = {"FC": ("label", "id", "evidence", "claim"), "RD": ("label", "id", "text")}
RAW_REL = Path("data/raw/incrediblae")
ITEMS_REL = Path("data/derived/incrediblae")
SPLITS_REL = Path("data/derived/splits")
STATS_REL = Path("data/derived/incrediblae-stats.json")
PRIVATE_REL = Path("data/derived/private")
LANE_B_NAME = "rd-attack.pseudo.jsonl"
KEY_REL = Path("secrets/rd-hmac.key")
KEY_BYTES = 32
RELEASE_SHA_FILE = "AAC_GIT_SHA"  # only turing releases carry it in the project root (aac.run.clock)

URL_RE = re.compile(r"(?i:https?)://[^\s<>\"]+")
URL_TRAIL = ".,;:!?)]}'\"…"
HANDLE_RE = re.compile(r"@\w+")
DIGIT_RUN_RE = re.compile(r"\b\d{15,20}\b")  # the user-id half of USER_ID_RE


class IngestError(ValueError):
    """A raw file fails its hash, layout or label check, or the key is unusable; nothing is written."""


class LaneBMissing(FileNotFoundError):
    """The private RD lane is not on this host (a clean checkout or remote host); run `make incrediblae-ingest` on the CPU host."""


def item_id(task: str, row: int) -> str:
    return f"incrediblae/{task.lower()}/attack/{row}"


def raw_rel(task: str) -> str:
    return f"incrediblae/{task}/attack.tsv"


# -- pseudonymization ---------------------------------------------------------------------------------

def _hmac8(key: bytes, value: str) -> str:
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:8]


def _url_sub(key: bytes):
    def sub(m: re.Match) -> str:
        url = m.group(0)
        core = url.rstrip(URL_TRAIL)
        tail = url[len(core):]
        if "://" not in core or core.endswith("://"):
            return url  # nothing but a scheme: leave it for the later passes
        authority = core.split("://", 1)[1].split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        host = authority.rsplit("@", 1)[-1].split(":", 1)[0].lower()
        return f"https://{host}/{_hmac8(key, core)}{tail}"
    return sub


def pseudonymise(text: str, key: bytes) -> str:
    """URLs, then handles, then 15-20 digit runs (see the module docstring). Raises IngestError if the
 result still matches USER_ID_RE (a host made of a handle-like or digit token, for example)."""
    out = URL_RE.sub(_url_sub(key), text)
    out = HANDLE_RE.sub(lambda m: f"user_{_hmac8(key, m.group(0).lower())}", out)
    out = DIGIT_RUN_RE.sub(lambda m: f"id_{_hmac8(key, m.group(0))}", out)
    if USER_ID_RE.search(out):
        raise IngestError("pseudonymised text still matches USER_ID_RE; refusing to write lane B")
    return out


# -- raw rows ---------------------------------------------------------------------------------------

def _bodega_unescape(field: str) -> str:
    """victims/transformer.py readfromfile_generator, verbatim order."""
    return field.strip().replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")


def read_attack_tsv(path: str | Path, task: str, *, expected: dict[str, str] | None = None,
                    root: str | Path | None = None) -> list[dict]:
    """Hash gate against MANIFEST.expected.sha256, then parse. Returns one dict per data row:
 FC {row, label, organiser_id, evidence, claim}; RD {row, label, organiser_id, text}. The texts are
 returned to the caller only; this function writes nothing."""
    if task not in COLUMNS:
        raise ValueError(f"task must be one of {TASKS}, not {task!r}")
    path = Path(path)
    rel = raw_rel(task)
    if expected is None:
        from aac.run.paths import root as project_root

        expected = parse_expected(Path(root) / EXPECTED_RAW if root is not None else project_root() / EXPECTED_RAW)
    if rel not in expected:
        raise IngestError(f"{rel} has no line in {EXPECTED_RAW}; refusing to ingest")
    actual = sha256_file(path)
    if actual != expected[rel]:
        raise IngestError(f"{rel}: sha256 {actual} != expected {expected[rel]}; refusing to ingest")
    want = len(COLUMNS[task])
    rows = []
    with open(path, encoding="utf-8") as fh:  # universal newlines, as BODEGA's open() reads it
        for lineno, line in enumerate(fh, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) != want:
                raise IngestError(f"{task} attack.tsv line {lineno}: {len(parts)} columns, expected {want} columns "
                                  f"({', '.join(COLUMNS[task])}); refusing to ingest")
            if parts[0] not in ("0", "1"):
                raise IngestError(f"{task} attack.tsv line {lineno}: label is not 0 or 1; refusing to ingest")
            rec = {"row": lineno - 1, "label": int(parts[0]), "organiser_id": parts[1]}
            for name, value in zip(COLUMNS[task][2:], parts[2:]):
                rec[name] = _bodega_unescape(value)
            rows.append(rec)
    return rows


# -- key --------------------------------------------------------------------------------------------

def ensure_key(path: str | Path) -> bytes:
    """Create the 32-byte HMAC key (secrets.token_bytes, mode 0600, directory 0700) only when absent;
 otherwise check and return it. Never overwrites and never prints the key."""
    path = Path(path)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, secrets.token_bytes(KEY_BYTES))
        finally:
            os.close(fd)
    mode = stat.S_IMODE(path.stat().st_mode)
    if mode != 0o600:
        raise IngestError(f"{path}: mode {mode:o}, expected 600; fix the permissions before ingest")
    key = path.read_bytes()
    if len(key) != KEY_BYTES:
        raise IngestError(f"{path}: key is not {KEY_BYTES} bytes; refusing to use it")
    return key


# -- outputs ----------------------------------------------------------------------------------------

def _write_atomic(path: Path, text: str, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if private:
        os.chmod(path.parent, 0o700)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600 if private else 0o644)
    try:
        if private:
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="ascii", newline="\n") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _p95(xs: list[int]) -> int:
    """Nearest-rank 95th percentile (an observed value; no interpolation)."""
    s = sorted(xs)
    return s[max(0, math.ceil(0.95 * len(s)) - 1)]


def _share(pos: int, n: int) -> str:
    from aac.reports.fmt import fmt_rate  # same formatting as pack/datasheet.py _items_stats

    return fmt_rate(pos / n) if n else "n/a"


def task_stats(task: str, rows: list[dict]) -> dict:
    """Counts only (no text leaves this function). Keys of pack/datasheet.py `_items_stats` plus additive ones;
 texts are the BODEGA-read raw fields (FC: the claim; RD: the tweet text before pseudonymization)."""
    texts = [r["claim"] if task == "FC" else r["text"] for r in rows]
    n = len(rows)
    pos = sum(1 for r in rows if r["label"] == 1)
    out = {
        "n_items": n,
        "n_label_1": pos,
        "label_1_share": _share(pos, n),
        "n_with_supplied_evidence": sum(1 for r in rows if r.get("evidence")),
        "n_duplicate_texts": n - len(set(texts)),
        "n_non_ascii_texts": sum(1 for t in texts if not t.isascii()),
        "n_empty_text": sum(1 for t in texts if t == ""),
        "n_unique_organiser_ids": len({r["organiser_id"] for r in rows}),
        "text_basis": "claim column as BODEGA reads it" if task == "FC" else "text column as BODEGA reads it, before pseudonymisation",
    }
    if task == "FC":
        ev = [r["evidence"] for r in rows]
        out["n_empty_evidence"] = sum(1 for e in ev if e == "")
        out["n_non_ascii_evidence"] = sum(1 for e in ev if not e.isascii())
    else:
        lengths = [len(t) for t in texts]
        out["n_texts_with_handles"] = sum(1 for t in texts if HANDLE_RE.search(t))
        out["n_texts_with_urls"] = sum(1 for t in texts if URL_RE.search(t))
        out["n_texts_with_digit_runs"] = sum(1 for t in texts if DIGIT_RUN_RE.search(t))
        out["text_length_chars"] = ({"median": float(statistics.median(lengths)), "p95_nearest_rank": _p95(lengths),
                                     "max": max(lengths)} if lengths else None)
    return out


def _manifest(task: str, rows: list[dict], source_sha256: str) -> dict:
    ids = [item_id(task, r["row"]) for r in rows]
    return {
        "name": f"incrediblae-{task.lower()}-attack",
        "kind": "attack-file",
        "source_file": raw_rel(task),
        "source_sha256": source_sha256,
        "rule": f"every data row of the hash-verified {raw_rel(task)} in file order; "
                f"item_id = incrediblae/{task.lower()}/attack/<0-based row> (never the organisers' id)",
        "ids": ids,
        "count": len(ids),
        "label_1_count": sum(1 for r in rows if r["label"] == 1),
        "assertions": {
            "ids_unique": len(set(ids)) == len(ids),
            "ids_row_indexed": ids == [item_id(task, i) for i in range(len(rows))],
            "count_equals_data_rows": len(ids) == len(rows),
        },
    }


def build(root: str | Path, *, key_path: str | Path | None = None, expected: dict[str, str] | None = None) -> dict[str, Path]:
    """Hash gate both attack files, then write the committed text-free files and the private lane B.
 Returns {relative path: path} of every file written."""
    root = Path(root)
    if expected is None:
        expected = parse_expected(root / EXPECTED_RAW)
    rows = {t: read_attack_tsv(root / RAW_REL / t / "attack.tsv", t, expected=expected) for t in TASKS}
    key = ensure_key(Path(key_path) if key_path is not None else root / KEY_REL)
    lane_b = [{"item_id": item_id("RD", r["row"]), "row": r["row"], "label": r["label"],
               "text": pseudonymise(r["text"], key)} for r in rows["RD"]]
    fc_items = [{"item_id": item_id("FC", r["row"]), "row": r["row"], "label": r["label"],
                 "claim_hash": claim_hash(r["claim"]), "evidence_hash": claim_hash(r["evidence"])} for r in rows["FC"]]
    rd_items = [{"item_id": b["item_id"], "row": b["row"], "label": b["label"], "text_hash": claim_hash(b["text"])}
                for b in lane_b]
    stats = {t: task_stats(t, rows[t]) for t in TASKS}
    stats["source_sha256"] = {raw_rel(t): expected[raw_rel(t)] for t in TASKS}
    manifests = {t: _manifest(t, rows[t], expected[raw_rel(t)]) for t in TASKS}
    for t, doc in manifests.items():
        false = sorted(k for k, v in doc["assertions"].items() if v is not True)
        if false:
            raise IngestError(f"{doc['name']}: assertions not true: {false}; nothing written")

    outputs = {
        ITEMS_REL / "fc-attack.items.json": dumps_sorted(fc_items),
        ITEMS_REL / "rd-attack.items.json": dumps_sorted(rd_items),
        SPLITS_REL / "incrediblae-fc-attack.json": dumps_sorted(manifests["FC"]),
        SPLITS_REL / "incrediblae-rd-attack.json": dumps_sorted(manifests["RD"]),
        STATS_REL: dumps_sorted(stats),
    }
    lane_text = "".join(json.dumps(b, sort_keys=True, ensure_ascii=True, separators=(",", ":")) + "\n"
                        for b in sorted(lane_b, key=lambda b: b["row"]))
    written = {}
    for rel, text in outputs.items():
        _write_atomic(root / rel, text)
        written[str(rel)] = root / rel
    lane_rel = PRIVATE_REL / LANE_B_NAME
    _write_atomic(root / lane_rel, lane_text, private=True)
    written[str(lane_rel)] = root / lane_rel
    return written




def load_texts(root: str | Path, task: str) -> dict[str, dict]:
    """Text keyed by item_id, read at run time; never written by this module.
 fc -> {item_id: {"text": claim, "evidence": evidence}} from the hash-checked raw attack.tsv;
 rd -> {item_id: {"text": lane-B text}} from the private lane (LaneBMissing when absent)."""
    root = Path(root)
    t = task.upper()
    if t == "FC":
        rows = read_attack_tsv(root / RAW_REL / "FC" / "attack.tsv", "FC", root=root)
        return {item_id("FC", r["row"]): {"text": r["claim"], "evidence": r["evidence"]} for r in rows}
    if t == "RD":
        p = root / PRIVATE_REL / LANE_B_NAME
        if not p.is_file():
            raise LaneBMissing(f"{PRIVATE_REL / LANE_B_NAME} is absent on this host (clean checkout or turing); "
                               "RD text is available only from lane B, built on CPU host by `make incrediblae-ingest`")
        out = {}
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                rec = json.loads(line)
                out[rec["item_id"]] = {"text": rec["text"]}
        return out
    raise ValueError(f"task must be fc or rd, not {task!r}")


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import root as project_root

    root = project_root()
    if (root / RELEASE_SHA_FILE).exists():
        print(f"incrediblae-ingest: FAIL: {root} is a turing release ({RELEASE_SHA_FILE} present); the HMAC key "
              "lives only on CPU host, run this there", file=sys.stderr)
        return 1
    try:
        written = build(root)
    except IngestError as e:
        print(f"incrediblae-ingest: FAIL: {e}", file=sys.stderr)
        return 1
    stats = json.loads((root / STATS_REL).read_text(encoding="utf-8"))
    for t in TASKS:
        s = stats[t]
        print(f"incrediblae-ingest: {t} rows {s['n_items']} label_1 {s['n_label_1']} empty_text {s['n_empty_text']} "
              f"unique_organiser_ids {s['n_unique_organiser_ids']}")
    lane = root / PRIVATE_REL / LANE_B_NAME
    with open(lane, encoding="utf-8") as fh:
        uid_lines = sum(1 for line in fh if USER_ID_RE.search(line))
    print(f"incrediblae-ingest: lane B lines matching USER_ID_RE: {uid_lines}")
    for rel, p in sorted(written.items()):
        print(f"incrediblae-ingest: wrote {rel} sha256={sha256_file(p)}")
    return 0 if uid_lines == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
