""": the detached provenance record of the submitted evidence pack.

`write_provenance(root, commit)` writes provenance/PACK-PROVENANCE.md: the pack commit, the sha256 of the
committed evidence-pack/MANIFEST.sha256 at that commit, the Table D.3 rows of the commit in shell order, the
Table 4.7 result (rows passed of six, command and commit from reports/table-4.7-verifiability.summary.json),
the sha256 at the commit of runs/MANIFEST.sha256, methodological notes and uv.lock (read with `git show`), the
Appendix D screenshot (path, sha256, width x height) or the analyst-item sentence, one Annex IV 2(g) sentence,
and the analyst's sign-off placeholder as the last line.

The record lives under provenance/, outside evidence-pack/ (so outside MANIFEST.sha256) and outside reports/
(so outside the determinism lint, `aac.pack.lint.SCAN_DIRS`): it carries the date and the signature, the pack
carries neither. Only the analyst signs: the writer emits the placeholder, and once the last line is anything
else (`ProvenanceSigned`) it refuses to overwrite the record. After signing, `check_frozen` fails whenever the
pack's MANIFEST.sha256 no longer matches the digest the signed record names (the freeze rule)."""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import io
import json
import os
import re
import struct
import subprocess
from pathlib import Path
from typing import Callable

from aac.run.paths import pack_dir, reports_dir

PLACEHOLDER = "Signed: ____ (analyst item O7)"
NO_SCREENSHOT = "screenshot: not taken (analyst item O8)"
RECORD_REL = Path("provenance") / "PACK-PROVENANCE.md"
T47_SHELL = "table-4.7-verifiability.csv"
T47_SUMMARY = "table-4.7-verifiability.summary.json"
VERIFIED_PREFIX = "Verified (UTC): "
MANIFEST_PREFIX = "evidence-pack/MANIFEST.sha256 sha256 at the pack commit: "
_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
_HEX64 = re.compile(r"[0-9a-f]{64}")
ANNEX_IV = ("Annex IV 2(g) of the EU AI Act asks for \"test logs and all test reports dated and signed by the "
            "responsible persons\"; the date and the signature live in this record, outside evidence-pack/ and "
            "reports/, because a date or a name inside the pack would break its byte-for-byte regeneration "
            "(MANIFEST.sha256, Table 4.7) and the determinism lint.")
RERUN = ("If an analyst reply (one-shot items O1, O2, O3, O5 or O7) changes a pack input, the pack is rebuilt "
         "and committed as a new pack commit C' , the clean-checkout verification and this "
         "record are re-run on C', and only the record for C' is signed.")


class ProvenanceSigned(RuntimeError):
    """The record already carries the analyst's sign-off line; it is never overwritten."""


class PackChangedAfterSigning(RuntimeError):
    """evidence-pack/MANIFEST.sha256 differs from the digest the signed provenance record names."""


def record_path(root: Path) -> Path:
    return Path(root) / RECORD_REL


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True).stdout


def _prefix(root: Path) -> str:
    """Path of `root` inside its repository ('' at the top level, 'aac-pipeline/' in this repository)."""
    return _git(root, "rev-parse", "--show-prefix").decode().strip()


def _show(root: Path, commit: str, rel: str) -> bytes:
    try:
        return _git(root, "show", f"{commit}:{_prefix(root)}{rel}")
    except subprocess.CalledProcessError as exc:
        raise FileNotFoundError(f"{rel} is not in commit {commit[:12]}") from exc


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _last_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _d3_rows(d3_csv: bytes) -> list[tuple[str, str]]:
    """(File, SHA-256 at submission) of each Table D.3 shell row, in file order."""
    text = d3_csv.decode("utf-8")
    body = "\n".join(ln for ln in text.splitlines() if not ln.startswith("#"))
    return [(r["File"], r["SHA-256 at submission"]) for r in csv.DictReader(io.StringIO(body))]


def _png_size(data: bytes) -> tuple[int, int]:
    if data[:8] != b"\x89PNG\r\n\x1a\n" or data[12:16] != b"IHDR":
        raise ValueError("screenshot is not a PNG file")
    return struct.unpack(">II", data[16:24])


def _table_4_7(root: Path, commit: str) -> list[str]:
    summary_p = reports_dir(root) / T47_SUMMARY
    shell_p = reports_dir(root) / T47_SHELL
    if not summary_p.is_file() or not shell_p.is_file():
        return ["Result: not run (reports/table-4.7-verifiability.csv or its summary JSON is absent)"]
    summary = json.loads(summary_p.read_text(encoding="ascii"))
    if summary.get("commit") != commit:
        raise ValueError(f"reports/{T47_SUMMARY} names commit {summary.get('commit')!r}, not the pack commit {commit}")
    rows = [ln for ln in shell_p.read_text(encoding="utf-8").splitlines()[2:] if ln]
    n_pass = sum(1 for ln in rows if ln.endswith(",pass"))
    return [f"Result: {n_pass} of {len(rows)} rows pass (reports/{T47_SHELL})",
            f"Command: {summary['command']}",
            f"Commit named by the summary: {summary['commit']}",
            f"Decision-log counts: n_records={summary['n_records']} violations={summary['violations']} "
            f"records_with_claim_text={summary['records_with_claim_text']} "
            f"records_with_user_id={summary['records_with_user_id']}"]


def render(root: Path, commit: str, *, screenshot: Path | None = None, verify_line: str | None = None,
           verified: str) -> str:
    root = Path(root)
    manifest = _show(root, commit, "evidence-pack/MANIFEST.sha256")
    d3 = _d3_rows(_show(root, commit, "evidence-pack/table-D.3-manifest.csv"))
    if verify_line is not None:
        want = f"verify-clean-checkout: {commit[:12]} identical ("
        if not verify_line.startswith(want):
            raise ValueError(f"the clean-checkout line must start with {want!r}")
    out = ["# Pack provenance record ", "",
           "Detached record of the submitted evidence pack. It is not listed in evidence-pack/MANIFEST.sha256 and "
           "not scanned by the determinism lint.", "",
           f"{VERIFIED_PREFIX}{verified}",
           f"Pack commit: {commit}",
           f"{MANIFEST_PREFIX}{_sha(manifest)}", "",
           "## Table D.3 at the pack commit (shell order)", "",
           "| File | SHA-256 at submission |", "|---|---|"]
    out += [f"| {f} | {h} |" for f, h in d3]
    out += ["", "## Table 4.7 (clean-checkout verification)", ""]
    out += _table_4_7(root, commit)
    out.append("Clean-checkout final line: " + (verify_line if verify_line is not None else "not recorded"))
    out += ["", "## Files at the pack commit outside the pack", ""]
    for rel in ("runs/MANIFEST.sha256", "uv.lock"):
        out.append(f"- {rel} sha256: {_sha(_show(root, commit, rel))}")
    out += ["", "## Appendix D screenshot (outside the hash-matched set)", ""]
    if screenshot is None:
        out.append(NO_SCREENSHOT)
    else:
        shot = Path(screenshot)
        shot_abs = shot if shot.is_absolute() else root / shot
        data = shot_abs.read_bytes()
        w, h = _png_size(data)
        rel_shot = shot_abs.resolve().relative_to(root.resolve())
        out.append(f"screenshot: {rel_shot.as_posix()} sha256 {_sha(data)} {w}x{h}")
    out += ["", "## Date and signature", "", ANNEX_IV, "", RERUN, "", PLACEHOLDER]
    return "\n".join(out) + "\n"


def write_provenance(root: Path, commit: str, *, screenshot: Path | None = None, verify_line: str | None = None,
                     clock: Callable[[], str] | None = None) -> Path:
    """Write provenance/PACK-PROVENANCE.md for `commit` (resolved to its full sha). Refuses (ProvenanceSigned)
 when the existing record's last line is anything but the placeholder; never writes a sign-off line."""
    root = Path(root)
    full = _git(root, "rev-parse", "--verify", f"{commit}^{{commit}}").decode().strip()
    target = record_path(root)
    if target.is_file() and _last_line(target.read_text(encoding="utf-8")) != PLACEHOLDER:
        raise ProvenanceSigned(f"{RECORD_REL} carries a sign-off line; it is never overwritten")
    verified = (clock or _utc_now)()
    if not _UTC.match(verified):
        raise ValueError(f"verification time must be UTC like 2026-10-21T18:00:00Z, not {verified!r}")
    text = render(root, full, screenshot=screenshot, verify_line=verify_line, verified=verified)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    os.replace(tmp, target)
    return target


def read_signed_digest(path: Path) -> str | None:
    """The MANIFEST digest a signed record names; None when the record is absent or still unsigned."""
    path = Path(path)
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if _last_line(text) == PLACEHOLDER:
        return None
    for ln in text.splitlines():
        if ln.startswith(MANIFEST_PREFIX):
            m = _HEX64.fullmatch(ln[len(MANIFEST_PREFIX):].strip())
            if m:
                return m.group(0)
    raise ValueError(f"{path} is signed but names no evidence-pack/MANIFEST.sha256 digest")


def check_frozen(root: Path) -> None:
    """Freeze rule: once the record is signed, the pack's MANIFEST.sha256 must still hash to the recorded digest.
 No-op while unsigned. Uses pack_dir(root): run it in the real namespace (AAC_OUT unset)."""
    want = read_signed_digest(record_path(root))
    if want is None:
        return
    have = _sha((pack_dir(root) / "MANIFEST.sha256").read_bytes())
    if have != want:
        raise PackChangedAfterSigning(f"evidence-pack/MANIFEST.sha256 is {have[:12]}, the signed record names {want[:12]}")
