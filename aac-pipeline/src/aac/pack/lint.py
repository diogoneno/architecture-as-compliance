"""Determinism lint (ARCHITECTURE §12 rule of record): derived outputs never carry
an absolute path, the host name, the user name, a wall-clock timestamp, CRLF line endings, NaN, the
draft's `____` placeholder, or a PNG `Software` text chunk.

Scope: reports/ and evidence-pack/ under the project root. Excluded by design:
evidence-pack/decision-log/*.jsonl.gz (verbatim sealed shards carry the Art. 12 timestamps),
evidence-pack/analyst-events.jsonl (: the verbatim copy of the append-only analyst events file, whose
analyst_action_timestamp values are Art. 12 action times; analyst-join.json carries no time and is scanned),
and runs/, logs/, _probes/ (inputs, not outputs; never under the scanned trees). The one sanctioned
host/user mention is Table D.2's `# host_change: ...` comment lines , rendered verbatim from the
versioned configs/host-changes.yaml; they are exempt from the host and user rules only.

One hit per (file, line); the kind is the first rule that matches in FORBIDDEN_PATTERNS order, so an
injected absolute path (which also contains the user name) is one `abs_path` hit.

adds two content rules (after the determinism rules): `user_id` (the leak scan's @handle /
15-20 digit pattern) and `claim_text` (the exact text or an 8-token shingle of any claim/item text of
the fixture corpus and of every indexed run, aac.logging.validate.claim_texts_for). Hash-bound values
(64-hex config_hash / data_version / sha256 cells, model_version m-<16 hex>, uuid record ids) are
removed from a line before the user_id rule, so a digest can never be read as a numeric user id.
evidence-pack/analyst-ui-mockup.html is exempt from `claim_text` only (accepted risk AR-01: the one
fixture claim text allowed into the pack). A run whose texts cannot be loaded is a `claim_text_unchecked`
hit on runs/INDEX.json, never a silent pass."""
from __future__ import annotations

import getpass
import json
import os
import re
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from aac.run.paths import runs_dir

SCAN_DIRS = ("reports", "evidence-pack")
EXCLUDE_GLOBS = ("evidence-pack/decision-log/*.jsonl.gz", "evidence-pack/analyst-events.jsonl")
EXCLUDE_PARTS = {"runs", "logs", "_probes"}
HOST_CHANGE_PREFIX = "# host_change: "
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
MIN_NAME_LEN = 3  # host/user names shorter than this would match ordinary words; recorded, not silently skipped

# Order matters: the first matching rule names the hit. abs_path/hostname/user are filled per run.
FORBIDDEN_PATTERNS: dict[str, str] = {
    "abs_path": "<absolute project path>",
    "hostname": "<host name, word-bounded>",
    "user": "<user name, word-bounded>",
    "timestamp": r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}",
    "crlf": r"\r$",
    "nan": r"\bNaN\b",
    "blank_slot": r"____",
    "png_software": "PNG tEXt/iTXt/zTXt chunk with keyword Software",
    "user_id": r"(?<![\w.+-])@\w+|\b\d{15,20}\b (after hash-bound values are removed; the @ of an e-mail address is not a handle)",
    "claim_text": "claim/item text of the fixture corpus or an indexed run (exact or 8-token shingle)",
}
HASH_TOKEN_RE = re.compile(r"(?<![0-9a-f])[0-9a-f]{64}(?![0-9a-f])|\bm-[0-9a-f]{16}\b"
                           r"|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
# The leak scan's USER_ID_RE, except that `@` directly after a word character is an e-mail address (e.g. the
# organisers' public list clef-factcheck@googlegroups.com in a datasheet), not a social-media handle.
LINT_USER_ID_RE = re.compile(r"(?<![\w.+-])@\w+|\b\d{15,20}\b")
CLAIM_TEXT_EXEMPT = frozenset({"evidence-pack/analyst-ui-mockup.html"})
FIXTURE_TEXT_FILES = ("tests/fixtures/corpus/claims.json", "tests/fixtures/corpus/fc-fx.json",
                      "tests/fixtures/corpus/rd-fx.json")


def claim_texts(repo_root: Path) -> tuple[set[str], list[Hit]]:
    """(texts, unchecked hits): the fixture corpus texts plus every indexed run's claim/item texts."""
    from aac.logging.validate import LeakScanUnavailable, _texts_from, claim_texts_for

    repo_root = Path(repo_root)
    texts: set[str] = set()
    for rel in FIXTURE_TEXT_FILES:
        if (repo_root / rel).is_file():
            texts |= _texts_from(repo_root / rel)
    unchecked: list[Hit] = []
    index = runs_dir(repo_root) / "INDEX.json"
    if index.is_file():
        for r in json.loads(index.read_text(encoding="utf-8")).get("runs", []):
            cr = runs_dir(repo_root) / r["run_id"] / "config.resolved.json"
            try:
                texts |= claim_texts_for(json.loads(cr.read_text(encoding="utf-8")), repo_root)
            except (OSError, LeakScanUnavailable) as exc:
                unchecked.append(Hit("runs/INDEX.json", 0, "claim_text_unchecked", f"{r['run_id']}: {exc}"[:160]))
    return texts, unchecked


class _ContentRules:
    def __init__(self, texts: set[str]) -> None:
        from aac.logging.leakscan import make_shingles

        self.uid = LINT_USER_ID_RE
        self.texts = {t for t in texts if t}
        self.shingles = make_shingles(self.texts)
        self.lengths = frozenset(len(s) for s in self.shingles)

    def check(self, line: str, rel: str) -> tuple[str, re.Match | None] | None:
        from aac.logging.leakscan import _has_shingle

        m = self.uid.search(HASH_TOKEN_RE.sub("", line))
        if m:
            return "user_id", None
        if rel not in CLAIM_TEXT_EXEMPT and self.texts and (
                any(t in line for t in self.texts) or (self.shingles and _has_shingle(line, self.shingles, self.lengths))):
            return "claim_text", None
        return None


@dataclass(frozen=True)
class Hit:
    file: str
    line: int
    kind: str
    excerpt: str

    def as_line(self) -> str:
        return f"{self.file}:{self.line}: {self.kind}: {self.excerpt}"


def _names() -> tuple[list[str], list[str]]:
    hosts = sorted({h for h in (socket.gethostname(), socket.gethostname().split(".")[0]) if h})
    try:
        user = os.environ.get("USER") or getpass.getuser()
    except Exception:  # noqa: BLE001 - no user database: nothing to look for
        user = ""
    return hosts, [user] if user else []


def _compile(repo_root: Path) -> list[tuple[str, re.Pattern | None]]:
    repo_root = Path(repo_root).resolve()
    paths = [str(repo_root)]
    if repo_root.parent != Path(repo_root.anchor) and len(repo_root.parent.parts) > 2:
        paths.append(str(repo_root.parent))
    hosts, users = _names()
    word = lambda names: re.compile("|".join(rf"(?<![\w.-]){re.escape(n)}(?![\w-])" for n in names)) if names else None  # noqa: E731
    rules: list[tuple[str, re.Pattern | None]] = [
        ("abs_path", re.compile("|".join(re.escape(p) for p in sorted(paths, key=len, reverse=True)))),
        ("hostname", word([h for h in hosts if len(h) >= MIN_NAME_LEN])),
        ("user", word([u for u in users if len(u) >= MIN_NAME_LEN])),
    ]
    for kind in ("timestamp", "crlf", "nan", "blank_slot"):
        rules.append((kind, re.compile(FORBIDDEN_PATTERNS[kind])))
    return rules


def _png_software(data: bytes) -> bool:
    i = len(PNG_SIGNATURE)
    while i + 8 <= len(data):
        length, ctype = struct.unpack(">I4s", data[i:i + 8])
        body = data[i + 8:i + 8 + length]
        if ctype in (b"tEXt", b"iTXt", b"zTXt") and body.split(b"\x00", 1)[0] == b"Software":
            return True
        if ctype == b"IEND":
            return False
        i += 12 + length
    return False


def _excluded(rel: str) -> bool:
    from fnmatch import fnmatch

    return any(fnmatch(rel, g) for g in EXCLUDE_GLOBS) or bool(EXCLUDE_PARTS & set(rel.split("/")))


def lint_file(path: Path, rel: str, rules, content: _ContentRules | None = None) -> list[Hit]:
    data = Path(path).read_bytes()
    if data.startswith(PNG_SIGNATURE):
        return [Hit(rel, 0, "png_software", "Software text chunk")] if _png_software(data) else []
    if b"\x00" in data:
        return []  # other binary files carry no text to lint
    hits = []
    for n, line in enumerate(data.decode("utf-8", errors="replace").split("\n"), start=1):
        sanctioned = line.startswith(HOST_CHANGE_PREFIX)
        for kind, rx in rules:
            if rx is None or (sanctioned and kind in ("hostname", "user")):
                continue
            m = rx.search(line)
            if m:
                hits.append(Hit(rel, n, kind, line[max(0, m.start() - 20):m.end() + 20].strip()))
                break
        else:
            found = content.check(line, rel) if content is not None else None
            if found:
                hits.append(Hit(rel, n, found[0], line.strip()[:60]))
    return hits


def lint_tree(base: Path, *, repo_root: Path | None = None, dirs: tuple[str, ...] = SCAN_DIRS,
              texts: set[str] | None = None) -> list[Hit]:
    """Every hit under base/<dir> for dir in `dirs`, sorted by (file, line). Claim texts come from
 `texts` when given, else from repo_root (fixture corpus + indexed runs)."""
    base = Path(base)
    rules = _compile(Path(repo_root) if repo_root else base)
    hits: list[Hit] = []
    if texts is None:
        texts, unchecked = claim_texts(Path(repo_root) if repo_root else base)
        hits.extend(unchecked)
    content = _ContentRules(texts)
    for d in dirs:
        top = base / d
        if not top.is_dir():
            continue
        for p in sorted(top.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(base).as_posix()
            if _excluded(rel):
                continue
            hits.extend(lint_file(p, rel, rules, content))
    return sorted(hits, key=lambda h: (h.file, h.line))
