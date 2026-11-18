"""The reference list of draft v0.5, parsed read-only (PRISMA-05/06; 8).

parse_references reads docs/dissertation/draft-v0.5.md between `# References` and the next `# ` heading. One
entry per paragraph; blank lines and the `Draft note:` paragraph are skipped. Each entry has the form
`<authors> (<year>[a-z]) '<title>', <venue>. Available at: <url>`; legal and standards entries may carry an
unquoted title (then the segment after the year up to the first full stop, title_quoted False).
The identifier type comes from the entry's FIRST URL: doi.org -> doi (DOI lower-cased), arxiv.org/abs -> arxiv,
any other URL -> url, no URL -> none.

cited_in(text, ref) is a heuristic (A7): the first author's surname followed by the year within 60
characters. It is labeled `heuristic` wherever it is printed; the analyst confirms any set built on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from aac.run.paths import root as project_root

YEAR_RE = re.compile(r"\((\d{4}[a-z]?)\)")
URL_RE = re.compile(r"https?://[^\s)]+")
QUOTED_RE = re.compile(r"'(.+?)',")
CITED_WINDOW = 60


@dataclass(frozen=True)
class Reference:
    index: int
    label: str
    first_author: str
    year: str
    title: str
    title_quoted: bool
    url: str
    identifier_type: str
    identifier: str

    @property
    def year_int(self) -> int:
        return int(self.year[:4])


def default_draft() -> Path:
    return project_root().parent / "docs" / "dissertation" / "draft-v0.5.md"


def _section(lines: list[str], heading_prefix: str) -> list[str]:
    start = None
    for i, ln in enumerate(lines):
        if start is None:
            if ln.startswith(heading_prefix):
                start = i + 1
        elif ln.startswith("# "):
            return lines[start:i]
    if start is None:
        raise ValueError(f"heading {heading_prefix!r} not found")
    return lines[start:]


def _classify(url: str) -> tuple[str, str]:
    if not url:
        return "none", ""
    m = re.match(r"https?://(?:dx\.)?doi\.org/(.+)$", url, flags=re.I)
    if m:
        return "doi", m.group(1).rstrip(".").lower()
    m = re.match(r"https?://arxiv\.org/abs/([^\s?#]+?)(?:v\d+)?$", url, flags=re.I)
    if m:
        return "arxiv", m.group(1)
    return "url", url


def _parse_entry(index: int, text: str) -> Reference:
    ym = YEAR_RE.search(text)
    if not ym:
        raise ValueError(f"reference {index}: no (year) group in {text[:80]!r}")
    label = text[: ym.end()]
    year = ym.group(1)
    authors = text[: ym.start()].strip()
    first_author = authors.split(",")[0].strip()
    rest = text[ym.end():].strip()
    qm = QUOTED_RE.match(rest)
    if qm:
        title, quoted = qm.group(1).strip(), True
    else:
        title, quoted = re.split(r"\.(?:\s|$)", rest, maxsplit=1)[0].strip(), False
    um = URL_RE.search(text)
    url = um.group(0).rstrip(".,;") if um else ""
    itype, ident = _classify(url)
    return Reference(index=index, label=label, first_author=first_author, year=year, title=title,
                     title_quoted=quoted, url=url, identifier_type=itype, identifier=ident)


def parse_references(draft: Path | None = None) -> list[Reference]:
    path = Path(draft) if draft else default_draft()
    lines = path.read_text(encoding="utf-8").splitlines()
    out: list[Reference] = []
    for ln in _section(lines, "# References"):
        s = ln.strip()
        if not s or s.startswith("Draft note:"):
            continue
        out.append(_parse_entry(len(out) + 1, s))
    labels = [r.label for r in out]
    if len(set(labels)) != len(labels):
        dup = sorted({x for x in labels if labels.count(x) > 1})
        raise ValueError(f"duplicate reference labels: {dup}")
    return out


def chapter2_text(draft: Path | None = None) -> str:
    path = Path(draft) if draft else default_draft()
    lines = path.read_text(encoding="utf-8").splitlines()
    return "\n".join(_section(lines, "# Chapter 2"))


def cited_in(text: str, ref: Reference) -> bool:
    """Heuristic: the surname, then the year (not followed by another letter suffix) within 60 characters."""
    year_re = re.compile(re.escape(ref.year) + (r"(?![a-z0-9])" if ref.year[-1].isdigit() else r"(?![a-z0-9])"))
    for m in re.finditer(re.escape(ref.first_author), text):
        window = text[m.end(): m.end() + CITED_WINDOW]
        if year_re.search(window):
            return True
    return False
