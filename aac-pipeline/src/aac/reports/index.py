"""reports/INDEX.json: every report file with its sha256 and the config_hashes / data_version it
consumed, as a JSON array sorted by file. CSVs declare their provenance on the first comment
line (`# config_hash=a;b data_version=...`); PNGs carry the same two keys as tEXt chunks
written by the figure generators (matplotlib `metadata=`). No timestamp, host or path."""
from __future__ import annotations

import json
import os
import struct
import zlib
from pathlib import Path

from aac.reports import fmt
from aac.run.hashing import sha256_file
from aac.run.paths import listdir
from aac.run.paths import reports_dir

INDEX_FILE = "INDEX.json"
INDEX_TMP = f".{INDEX_FILE}.tmp"  # write_index's temp file (tmp + os.replace); never listed by _walk
# Analyst exports (reports/ch4-feed-*/, paste-ready Markdown, not generator outputs): skipped as a FIRST path component
# only, the same rule as tests/contract/test_real_reports.py INDEX_EXEMPT_DIR_PREFIX (720065c exempted them in the
# contract but not here, so `aac reports index` refused the folder's .md files). reports/ch5-feed/ (the
# Chapter 3-5 write-in export, 2026-10-22, scripts/ch5-feed/build_all.py) follows the same rule, and so does


EXEMPT_DIR_PREFIX = ("ch4-feed-", "ch5-feed", "appendix-c", "write-in")


def png_text_chunks(path: Path) -> dict[str, str]:
    """tEXt chunks of a PNG as {keyword: text} (stdlib only)."""
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path}: not a PNG")
    out: dict[str, str] = {}
    pos = 8
    while pos + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"tEXt":
            key, _, text = body.partition(b"\x00")
            out[key.decode("latin-1")] = text.decode("latin-1")
        elif ctype == b"zTXt":
            key, _, rest = body.partition(b"\x00")
            out[key.decode("latin-1")] = zlib.decompress(rest[1:]).decode("latin-1")
        elif ctype == b"iTXt":
            key, _, rest = body.partition(b"\x00")
            comp_flag = rest[0]
            rest = rest[2:]
            _lang, _, rest = rest.partition(b"\x00")
            _trans, _, text = rest.partition(b"\x00")
            out[key.decode("latin-1")] = (zlib.decompress(text) if comp_flag else text).decode("utf-8")
        pos += 12 + length
        if ctype == b"IEND":
            break
    return out


def _walk(base: Path) -> list[str]:
    out: list[str] = []

    def rec(d: Path, rel: str) -> None:
        for name in listdir(d):
            p = d / name
            r = f"{rel}{name}"
            if p.is_dir():
                if not rel and name.startswith(EXEMPT_DIR_PREFIX):
                    continue
                rec(p, r + "/")
            elif r not in (INDEX_FILE, INDEX_TMP):
                out.append(r)

    rec(base, "")
    return sorted(out)


def provenance(path: Path) -> tuple[list[str], str]:
    if path.name.endswith(".summary.json"):
        # Table 4.7's summary (scripts/regen_check.sh) shares the provenance of its shell CSV
        return provenance(path.with_name(path.name[: -len(".summary.json")] + ".csv"))
    if path.suffix == ".csv":
        comments, _ = fmt.read_table(path)
        if not comments or not comments[0].startswith("# config_hash="):
            raise ValueError(f"{path}: missing the `# config_hash=` header comment")
        fields = fmt.header_fields(comments[0])
    elif path.suffix == ".png":
        fields = png_text_chunks(path)
        if "config_hash" not in fields or "data_version" not in fields:
            raise ValueError(f"{path}: PNG carries no config_hash/data_version tEXt chunk")
    else:
        raise ValueError(f"{path}: unknown report file type")
    hashes = sorted({h for h in str(fields["config_hash"]).split(";") if h})
    if not hashes:
        raise fmt.MissingConfigHash(f"{path}: empty config_hash in its provenance header")
    return hashes, str(fields["data_version"])


def write_index(root: Path, out_dir: Path | None = None) -> Path:
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    entries = []
    for rel in _walk(out_dir):
        p = out_dir / rel
        hashes, dv = provenance(p)
        entries.append({"file": rel, "sha256": sha256_file(p), "config_hashes": hashes, "data_version": dv})
    if not entries:
        raise RuntimeError(f"{out_dir}: no report files to index")
    entries.sort(key=lambda e: e["file"])
    target = out_dir / INDEX_FILE
    tmp = out_dir / INDEX_TMP
    try:
        with open(tmp, "w", encoding="ascii", newline="\n") as fh:
            fh.write(json.dumps(entries, indent=1, sort_keys=True, ensure_ascii=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            tmp.unlink()
    return target
