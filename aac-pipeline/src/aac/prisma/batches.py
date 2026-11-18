"""`aac prisma batch export|merge` (PRISMA-01 batch half, PRISMA-02).

Batch files are what the analyst screens: CSV, UTF-8, LF, no BOM, the read-only columns copied from the workbook
and the editable columns empty. A read-only cell that starts with `=`, `+`, `-` or `@` gets one leading apostrophe
in the batch copy only (spreadsheet formula guard, operational constraint); the workbook never holds it.

merge is all-or-nothing. It detects the stage from the exact header (a BOM is dropped, CRLF is read), validates
every row against the workbook and the numbered rule, and on any problem raises MergeRefused listing every problem
as `row <n> key <k>: <problem>` (row numbers as a spreadsheet shows them: the header is row 1) and writes nothing.
Otherwise it writes the workbook once, under the exclusive workbook lock. It only copies the analyst's own cells:
a blank screen_decision / exclude_reason is derived from the analyst's title_rule_id through the rule table, and
screener, session_date (an argument, never the clock) and rule_version are stamped on the rows whose decision the
batch sets. No code path here writes `proposal`; original columns are never taken from a batch.

Stages : `abstract` batches hold the title survivors with a fetched abstracts.csv row (matched_title
and abstract shown from abstracts.csv, display only: neither compared on merge nor written); `fulltext` batches
hold the abstract includes. merge enforces the stage order: no abstract decision without a title include, no
full-text cell without an abstract include, fulltext_decision only with retrieved yes, retrieved and
included_in_chapter2 only yes or no, every exclusion with a reason code; a rule id typed into abstract_decision or
fulltext_decision maps to its decision and reason. `export --stage abstract|fulltext` writes the next
<stage>-NN.csv with every eligible row not listed in an earlier batch of that stage (batches/ or batches/merged/).
`intake` copies returned files from the loki inbox (cmp-checked), merges every inbox file in name order, moves the
merged ones to batches/merged/, leaves a refused one with <name>.refused.txt, fetches the abstracts of new survivors
and issues the next abstract and full-text batches; its last line is
`prisma-intake: merged=<n> refused=<r> abstracts_pending=<p> new_batches=<b>` (exit 1 when a file was refused).

Exit codes: 0 on success, 1 when a merge or export is refused (intake: when any file was refused), 2 for usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import filecmp
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from aac.prisma import schema
from aac.prisma._io import (
    OutsideScreening,
    assert_confined,
    default_appendix_dir,
    read_csv,
    screening_dir,
    workbook_lock,
    write_csv_atomic,
    write_text_atomic,
)
from aac.prisma.sample import BATCHES, TITLE_ORDER_SALT, hash_rank
from aac.run.hashing import sha256_bytes

WORKBOOK = "workbook.csv"
RULES_FILE = "title-screen-rule.md"
REVISIONS = "revisions.csv"
REVISIONS_HEADER = ("key", "column", "old", "new", "session_date")
RULES_BEGIN = "<!-- rules:begin -->"
RULES_END = "<!-- rules:end -->"
NO_RULE_ID = "R0"
FORMULA_CHARS = ("=", "+", "-", "@")
# Reason codes whose rule needs a note naming the kept key.
NOTE_REASONS = ("superseded version", "duplicate at title level")

STAGES: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "title": (
        ("key", "title", "year", "title_dup_group", "title_rule_id", "screen_decision", "exclude_reason", "notes"),
        ("title_rule_id", "screen_decision", "exclude_reason", "notes"),
    ),
    # blind: no first decision, rule id or note is shown to the second screener
    "second": (
        ("key", "title", "year", "second_screen_decision"),
        ("second_screen_decision",),
    ),
    # matched_title and abstract are read from abstracts.csv : display only, never merged
    "abstract": (
        ("key", "title", "year", "doi_url", "abstract_source", "matched_title", "abstract", "abstract_decision",
         "abstract_reason", "notes"),
        ("abstract_decision", "abstract_reason", "notes"),
    ),
    "fulltext": (
        ("key", "title", "year", "doi_url", "retrieved", "fulltext_decision", "fulltext_reason", "included_in_chapter2",
         "notes"),
        ("retrieved", "fulltext_decision", "fulltext_reason", "included_in_chapter2", "notes"),
    ),
}
# The column whose value counts a row as decided in a merged batch of each stage.
DECIDED_COLUMN = {"title": "screen_decision", "second": "second_screen_decision", "abstract": "abstract_decision",
                  "fulltext": "fulltext_decision"}
# Batch columns that come from abstracts.csv, not the workbook: shown to the screener, not compared on merge
# (a spreadsheet may re-wrap a long abstract), never written to the workbook.
DISPLAY_ONLY: frozenset[str] = frozenset({"matched_title", "abstract"})
MERGED = "merged"
RULE_ID_RE = re.compile(r"^R\d+$")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class MergeRefused(RuntimeError):
    """The batch fails validation; nothing was written. `problems` lists each as `row <n> key <k>: <problem>`."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__(f"refused ({len(self.problems)} problems)")


class RulesError(RuntimeError):
    """The rule file is absent or its table between the markers is malformed."""


# ------------------------------------------------------------------ rules


def _rules_region(text: str) -> str:
    if text.count(RULES_BEGIN) != 1 or text.count(RULES_END) != 1:
        raise RulesError(f"{RULES_FILE} must hold exactly one {RULES_BEGIN} and one {RULES_END}")
    start = text.index(RULES_BEGIN) + len(RULES_BEGIN)
    end = text.index(RULES_END)
    if end < start:
        raise RulesError(f"{RULES_END} comes before {RULES_BEGIN} in {RULES_FILE}")
    return text[start:end]


def parse_rules(text: str) -> tuple[dict[str, tuple[str, str]], str]:
    """({id: (decision or "any", reason or "" or "any")}, rule_version = sha256 of the bytes between the markers)."""
    region = _rules_region(text)
    rules: dict[str, tuple[str, str]] = {}
    for line in region.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 3 or set(cells[0]) <= set("-: ") or cells[0].lower() == "id":
            continue  # separator or header row
        rid, decision, reason = cells[0].strip("`"), cells[1].lower(), cells[2]
        if not RULE_ID_RE.match(rid):
            raise RulesError(f"rule id {rid!r} is not R<number>")
        if rid in rules:
            raise RulesError(f"rule {rid} is listed twice")
        reason = "" if reason in ("-", "") else reason
        if decision == "any":
            if reason != "any":
                raise RulesError(f"rule {rid}: decision any needs reason code any")
        elif decision == "exclude":
            if reason not in schema.REASON_CODES:
                raise RulesError(f"rule {rid}: reason code {reason!r} is not in the controlled vocabulary")
        elif decision == "include":
            if reason:
                raise RulesError(f"rule {rid}: an include rule carries no reason code (write -)")
        else:
            raise RulesError(f"rule {rid}: decision {decision!r} is not include, exclude or any")
        rules[rid] = (decision, reason)
    if NO_RULE_ID not in rules or rules[NO_RULE_ID] != ("any", "any"):
        raise RulesError(f"{RULES_FILE} needs rule {NO_RULE_ID} with decision any and reason code any")
    return rules, sha256_bytes(region.encode("utf-8"))


def load_rules(appendix: Path) -> tuple[dict[str, tuple[str, str]], str]:
    path = screening_dir(appendix) / RULES_FILE
    if not path.is_file():
        raise RulesError(f"{path} is absent")
    return parse_rules(path.read_text(encoding="utf-8"))


def rules_status(appendix: Path) -> str:
    """The `Status:` line of the rule file (outside the markers, so approving it does not change rule_version)."""
    path = screening_dir(appendix) / RULES_FILE
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Status:"):
            return line.strip()
    return "Status: (no status line)"


# ------------------------------------------------------------------ export


def escape_cell(value: str) -> str:
    return "'" + value if value.startswith(FORMULA_CHARS) else value


def _matches(batch_value: str, wb_value: str) -> bool:
    """A read-only cell equals the workbook value, with or without the export's one apostrophe."""
    return batch_value == wb_value or batch_value == escape_cell(wb_value) != wb_value


def _rel(path: Path, appendix: Path) -> str:
    from aac.prisma.workbook import _rel as rel
    return rel(path, appendix)


def _read_workbook(appendix: Path) -> list[dict[str, str]]:
    header, rows = read_csv(screening_dir(appendix) / WORKBOOK)
    if tuple(header) != schema.WORKBOOK_COLUMNS:
        raise RuntimeError(f"{WORKBOOK} header is not the schema's 26 columns")
    return rows


def _stage_view(appendix: Path, wb: dict[str, dict[str, str]], stage: str) -> dict[str, dict[str, str]]:
    """Workbook rows, plus matched_title and abstract from abstracts.csv for the abstract stage."""
    if stage != "abstract":
        return wb
    from aac.prisma.abstracts import read_abstracts
    abstracts = read_abstracts(appendix)
    return {k: {**w, "matched_title": abstracts.get(k, {}).get("matched_title", ""),
                "abstract": abstracts.get(k, {}).get("abstract", "")} for k, w in wb.items()}


def _batch_rows(wb: dict[str, dict[str, str]], keys: Sequence[str], stage: str) -> list[list[str]]:
    header, editable = STAGES[stage]
    out = []
    for k in keys:
        if k not in wb:
            raise KeyError(f"key {k!r} is not in the workbook")
        out.append(["" if c in editable else escape_cell(wb[k][c]) for c in header])
    return out


def export_batch(appendix: Path, name: str, keys: Sequence[str], stage: str) -> Path:
    """Write batches/<name>.csv for the given keys in the given order; refuse to overwrite."""
    appendix = Path(appendix)
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name):
        raise ValueError(f"batch name {name!r}: lower-case letters, digits and hyphens only")
    target = screening_dir(appendix) / BATCHES / f"{name}.csv"
    if target.exists():
        raise FileExistsError(f"{_rel(target, appendix)} exists; a batch file is never overwritten")
    if len(set(keys)) != len(keys):
        raise ValueError("a batch lists each key once")
    with workbook_lock(appendix, shared=True):
        wb = {r["key"]: r for r in _read_workbook(appendix)}
    rows = _batch_rows(_stage_view(appendix, wb, stage), list(keys), stage)
    return write_csv_atomic(target, STAGES[stage][0], rows, appendix)


def _stage_listed(appendix: Path, stage: str) -> tuple[set[str], int]:
    """(keys listed in any <stage>-NN.csv under batches/ or batches/merged/, highest NN used)."""
    pat = re.compile(rf"^{re.escape(stage)}-(\d{{2,}})\.csv$")
    listed: set[str] = set()
    top = 0
    bdir = screening_dir(appendix) / BATCHES
    for d in (bdir, bdir / MERGED):
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            m = pat.match(p.name)
            if m and p.is_file():
                top = max(top, int(m.group(1)))
                listed |= {r.get("key", "") for r in read_csv(p)[1]}
    return listed, top


def _abstract_eligible(appendix: Path, rows: list[dict[str, str]]) -> list[str]:
    """Title includes with a fetched abstracts.csv row (pending rows wait for the fetch) and no abstract decision."""
    from aac.prisma.abstracts import PENDING_PREFIX, read_abstracts, survivors
    abstracts = read_abstracts(appendix)
    return [r["key"] for r in survivors(rows)
            if r["key"] in abstracts and not abstracts[r["key"]]["outcome"].startswith(PENDING_PREFIX)
            and not (r.get("abstract_decision") or "").strip()]


def _fulltext_eligible(appendix: Path, rows: list[dict[str, str]]) -> list[str]:
    """Abstract includes without a full-text decision."""
    return [r["key"] for r in rows if (r.get("abstract_decision") or "").strip() == "include"
            and not (r.get("fulltext_decision") or "").strip()]


ELIGIBLE = {"abstract": _abstract_eligible, "fulltext": _fulltext_eligible}


def export_stage_batch(appendix: Path, stage: str) -> tuple[Path | None, int]:
    """The next free batches/<stage>-NN.csv with every eligible row not listed in an earlier <stage> batch
 (batches/ or batches/merged/), in key order; (None, 0) when nothing is eligible (no file is written)."""
    appendix = Path(appendix)
    if stage not in ELIGIBLE:
        raise ValueError(f"stage {stage!r} has no stage batch (abstract, fulltext)")
    listed, top = _stage_listed(appendix, stage)
    with workbook_lock(appendix, shared=True):
        rows = _read_workbook(appendix)
    keys = sorted(k for k in ELIGIBLE[stage](appendix, rows) if k not in listed)
    if not keys:
        return None, 0
    return export_batch(appendix, f"{stage}-{top + 1:02d}", keys, stage), len(keys)


def export_title_batches(appendix: Path, size: int = 300, prefix: str = "title") -> list[tuple[Path, int]]:
    """Every row without a title decision, in TITLE_ORDER_SALT hash order, as <prefix>-01.csv, -02.csv, ...

 The undecided members of one title_dup_group stay adjacent in one batch (a batch may exceed size by that
 group). Refuses when any <prefix>-NN.csv already exists."""
    appendix = Path(appendix)
    if size < 1:
        raise ValueError("size must be at least 1")
    bdir = screening_dir(appendix) / BATCHES
    pat = re.compile(rf"^{re.escape(prefix)}-\d{{2,}}\.csv$")
    if bdir.is_dir():
        existing = sorted(p.name for p in bdir.iterdir() if pat.match(p.name))
        if existing:
            raise FileExistsError(f"title batches already exist ({', '.join(existing)}); export runs once")
    with workbook_lock(appendix, shared=True):
        rows = _read_workbook(appendix)
    undecided = {r["key"]: r for r in rows if not (r.get("screen_decision") or "").strip()}
    if not undecided:
        raise LookupError("every row carries a title decision; nothing to export")
    ranked = hash_rank(undecided, TITLE_ORDER_SALT)
    members: dict[str, list[str]] = {}
    for k in ranked:
        g = undecided[k].get("title_dup_group") or ""
        if g:
            members.setdefault(g, []).append(k)
    units: list[list[str]] = []
    placed: set[str] = set()
    for k in ranked:
        if k in placed:
            continue
        g = undecided[k].get("title_dup_group") or ""
        unit = members[g] if g else [k]
        placed.update(unit)
        units.append(unit)
    batches: list[list[str]] = [[]]
    for unit in units:
        if len(batches[-1]) >= size:
            batches.append([])
        batches[-1].extend(unit)
    width = max(2, len(str(len(batches))))
    out = []
    for i, keys in enumerate(batches, start=1):
        out.append((export_batch(appendix, f"{prefix}-{i:0{width}d}", keys, "title"), len(keys)))
    return out


# ------------------------------------------------------------------ merge


@dataclass
class MergeResult:
    path: Path
    stage: str
    rows: int
    decided: int
    changed: int
    revisions: int
    written: bool


def _iso_date(s: str) -> str:
    if not DATE_RE.match(s or ""):
        raise ValueError(f"session date {s!r} is not YYYY-MM-DD")
    _dt.date.fromisoformat(s)
    return s


def _detect_stage(header: list[str], stage: str | None) -> str | None:
    found = next((name for name, (h, _) in STAGES.items() if tuple(header) == h), None)
    if stage is not None and found is not None and found != stage:
        return None
    return found


def _norm_decision(value: str) -> str:
    v = value.strip().lower()
    return {"i": "include", "e": "exclude"}.get(v, v)


def _norm_reason(value: str) -> str:
    v = value.strip()
    for code in schema.REASON_CODES:
        if v.lower() == code.lower():
            return code
    return v


def _title_row(rec: dict[str, str], wbrow: dict[str, str], rules: dict[str, tuple[str, str]],
               problems: list[str], where: str) -> dict[str, str] | None:
    """The decision cells the batch row sets ({} when it sets none), or None after recording problems."""
    rid = rec["title_rule_id"].strip().upper()
    dec = _norm_decision(rec["screen_decision"])
    reason = _norm_reason(rec["exclude_reason"])
    note = rec["notes"].strip()
    bad = []
    if RULE_ID_RE.match(dec.upper()):  # a rule id typed into the decision cell
        if rid and rid != dec.upper():
            bad.append(f"title_rule_id {rid} and screen_decision {dec.upper()} name different rules")
        rid, dec = dec.upper(), ""
    if rid and rid not in rules:
        bad.append(f"title_rule_id {rid!r} is not a rule of {RULES_FILE}")
    if dec and dec not in schema.DECISIONS:
        bad.append(f"screen_decision {rec['screen_decision']!r} is not include or exclude")
    if reason and reason not in schema.REASON_CODES:
        bad.append(f"exclude_reason {rec['exclude_reason']!r} is not a reason code")
    if bad:
        problems.extend(f"{where}: {b}" for b in bad)
        return None
    if not (rid or dec or reason):
        return {"notes": note} if note else {}
    if not rid:
        problems.append(f"{where}: a decision needs a title_rule_id (write {NO_RULE_ID} when no rule fits)")
        return None
    r_dec, r_reason = rules[rid]
    if r_dec != "any":
        if dec and dec != r_dec:
            bad.append(f"rule {rid} means {r_dec}, screen_decision is {dec}")
        if r_dec == "exclude" and reason and reason != r_reason:
            bad.append(f"rule {rid} means reason {r_reason!r}, exclude_reason is {reason!r}")
        dec = dec or r_dec
        if r_dec == "exclude":
            reason = reason or r_reason
    elif not dec:
        bad.append(f"rule {rid} fits no rule: write screen_decision, exclude_reason and a note")
    if dec == "exclude" and not reason:
        bad.append("an exclusion needs an exclude_reason")
    if dec == "include" and reason:
        bad.append(f"an include carries no exclude_reason (got {reason!r})")
    effective_note = note or (wbrow.get("notes") or "").strip()
    if (r_dec == "any" or reason in NOTE_REASONS) and not effective_note:
        bad.append(f"rule {rid} needs a note" + (" naming the kept key" if reason in NOTE_REASONS else ""))
    if bad:
        problems.extend(f"{where}: {b}" for b in bad)
        return None
    out = {"title_rule_id": rid, "screen_decision": dec, "exclude_reason": reason}
    if note:
        out["notes"] = note
    return out


def _stage_decision(value: str, reason_in: str, rules: dict[str, tuple[str, str]], col: str, reason_col: str,
                    bad: list[str]) -> tuple[str, str]:
    """(decision, reason) of an abstract or full-text row: include/exclude/i/e, or a rule id of the rule file mapped
 to its decision and reason. Problems go to bad."""
    dec = _norm_decision(value)
    reason = _norm_reason(reason_in)
    if RULE_ID_RE.match(dec.upper()):
        rid = dec.upper()
        if rid not in rules:
            bad.append(f"{col} {value!r} is not a rule of {RULES_FILE}")
            return "", reason
        r_dec, r_reason = rules[rid]
        if r_dec == "any":
            bad.append(f"rule {rid} fits no rule: write {col} include or exclude, {reason_col} and a note")
            return "", reason
        if r_dec == "exclude" and reason and reason != r_reason:
            bad.append(f"rule {rid} means reason {r_reason!r}, {reason_col} is {reason!r}")
        dec = r_dec
        if r_dec == "exclude":
            reason = reason or r_reason
    elif dec and dec not in schema.DECISIONS:
        bad.append(f"{col} {value!r} is not include, exclude or a rule id")
        return "", reason
    if reason and reason not in schema.REASON_CODES:
        bad.append(f"{reason_col} {reason_in!r} is not a reason code")
    if dec == "exclude" and not reason:
        bad.append(f"an exclusion needs a {reason_col}")
    if dec == "include" and reason:
        bad.append(f"an include carries no {reason_col} (got {reason!r})")
    if reason and not dec:
        bad.append(f"{reason_col} {reason!r} without a {col}")
    return dec, reason


def _abstract_row(rec: dict[str, str], wbrow: dict[str, str], rules: dict[str, tuple[str, str]],
                  problems: list[str], where: str) -> dict[str, str] | None:
    bad: list[str] = []
    note = rec["notes"].strip()
    dec, reason = _stage_decision(rec["abstract_decision"], rec["abstract_reason"], rules, "abstract_decision",
                                  "abstract_reason", bad)
    if (dec or reason) and (wbrow.get("screen_decision") or "") != "include":
        bad.append("an abstract decision needs a title include (screen_decision is "
                   f"{(wbrow.get('screen_decision') or 'empty')!r})")
    if reason in NOTE_REASONS and not (note or (wbrow.get("notes") or "").strip()):
        bad.append(f"reason {reason!r} needs a note naming the kept key")
    if bad:
        problems.extend(f"{where}: {b}" for b in bad)
        return None
    out = {"abstract_decision": dec, "abstract_reason": reason} if dec else {}
    if note:
        out["notes"] = note
    return out


def _fulltext_row(rec: dict[str, str], wbrow: dict[str, str], rules: dict[str, tuple[str, str]],
                  problems: list[str], where: str) -> dict[str, str] | None:
    bad: list[str] = []
    note = rec["notes"].strip()
    retrieved = rec["retrieved"].strip().lower()
    ch2 = rec["included_in_chapter2"].strip().lower()
    if retrieved and retrieved not in schema.RETRIEVED:
        bad.append(f"retrieved {rec['retrieved']!r} is not yes or no")
    if ch2 and ch2 not in schema.CH2_VALUES:
        bad.append(f"included_in_chapter2 {rec['included_in_chapter2']!r} is not yes or no")
    dec, reason = _stage_decision(rec["fulltext_decision"], rec["fulltext_reason"], rules, "fulltext_decision",
                                  "fulltext_reason", bad)
    if (retrieved or dec or reason or ch2) and (wbrow.get("abstract_decision") or "") != "include":
        bad.append("a full-text decision needs an abstract include (abstract_decision is "
                   f"{(wbrow.get('abstract_decision') or 'empty')!r})")
    effective_retrieved = retrieved or (wbrow.get("retrieved") or "")
    if dec and effective_retrieved != "yes":
        bad.append(f"fulltext_decision {dec} needs retrieved yes (retrieved is {effective_retrieved or 'empty'!r})")
    if reason in NOTE_REASONS and not (note or (wbrow.get("notes") or "").strip()):
        bad.append(f"reason {reason!r} needs a note naming the kept key")
    if bad:
        problems.extend(f"{where}: {b}" for b in bad)
        return None
    out: dict[str, str] = {}
    if retrieved:
        out["retrieved"] = retrieved
    if dec:
        out.update(fulltext_decision=dec, fulltext_reason=reason)
    if ch2:
        out["included_in_chapter2"] = ch2
    if note:
        out["notes"] = note
    return out


def _second_row(rec: dict[str, str], problems: list[str], where: str) -> dict[str, str] | None:
    dec = _norm_decision(rec["second_screen_decision"])
    if not dec:
        return {}
    if dec not in schema.DECISIONS:
        problems.append(f"{where}: second_screen_decision {rec['second_screen_decision']!r} is not include or exclude")
        return None
    return {"second_screen_decision": dec}


def merge_batch(appendix: Path, path: Path, *, screener: str, session_date: str, allow_revise: bool = False,
                stage: str | None = None) -> MergeResult:
    appendix, path = Path(appendix), Path(path)
    session_date = _iso_date(session_date)
    screener = (screener or "").strip()
    if not screener or "\n" in screener or "\r" in screener:
        raise ValueError("--screener needs a one-line name")
    if stage is not None and stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    header, recs = read_csv(path)
    found = _detect_stage(header, stage)
    if found is None:
        want = f"the {stage} stage" if stage else "any stage"
        raise MergeRefused([f"row 1 key -: header {','.join(header)!r} matches {want} "
                            f"({'; '.join(','.join(h) for h, _ in STAGES.values())})"])
    stage = found
    cols, editable = STAGES[stage]
    rules, rule_version = load_rules(appendix) if stage != "second" else ({}, "")
    with workbook_lock(appendix, shared=False):
        rows = _read_workbook(appendix)
        wb = {r["key"]: r for r in rows}
        problems: list[str] = []
        seen: dict[str, int] = {}
        updates: dict[str, dict[str, str]] = {}
        revisions: list[tuple[str, str, str, str, str]] = []
        decided = 0
        for i, rec in enumerate(recs, start=2):
            key = rec.get("key") or ""
            where = f"row {i} key {key or '-'}"
            if None in rec or any(rec.get(c) is None for c in cols):
                problems.append(f"{where}: the row does not have the header's {len(cols)} fields")
                continue
            if key in seen:
                problems.append(f"{where}: key repeats row {seen[key]}")
                continue
            seen[key] = i
            if key not in wb:
                problems.append(f"{where}: key is not in the workbook")
                continue
            w = wb[key]
            ro_bad = [c for c in cols if c not in editable and c != "key" and c not in DISPLAY_ONLY
                      and not _matches(rec[c], w[c])]
            if ro_bad:
                problems.extend(f"{where}: {c} differs from the workbook ({rec[c]!r} vs {w[c]!r})" for c in ro_bad)
                continue
            if stage == "title":
                new = _title_row(rec, w, rules, problems, where)
            elif stage == "abstract":
                new = _abstract_row(rec, w, rules, problems, where)
            elif stage == "fulltext":
                new = _fulltext_row(rec, w, rules, problems, where)
            else:
                new = _second_row(rec, problems, where)
            if new is None:
                continue
            if stage == "title" and "screen_decision" in new:
                new.update(screener=screener, session_date=session_date, rule_version=rule_version)
            elif stage == "second" and new:
                new["second_screen_session_date"] = session_date
            changes = {c: v for c, v in new.items() if w[c] != v}
            decision_changes = {c: v for c, v in changes.items() if c not in ("screener", "session_date",
                                                                             "rule_version", "second_screen_session_date")}
            if not decision_changes:
                changes = {}
            held = [c for c in changes if w[c] and c in schema.DECISION_COLUMNS]
            if held and not allow_revise:
                problems.extend(f"{where}: {c} holds {w[c]!r}, the batch says {changes[c]!r} "
                                "(pass --allow-revise to change a recorded decision)" for c in held)
                continue
            if changes:
                updates[key] = changes
                revisions.extend((key, c, w[c], v, session_date) for c, v in changes.items() if w[c])
        if problems:
            raise MergeRefused(problems)
        for rec in recs:
            merged = {**wb[rec["key"]], **updates.get(rec["key"], {})}
            if merged[DECIDED_COLUMN[stage]]:
                decided += 1
        written = bool(updates)
        if written:
            scr = screening_dir(appendix)
            if revisions:
                rpath = scr / REVISIONS
                old = [[r[c] for c in REVISIONS_HEADER] for r in read_csv(rpath)[1]] if rpath.is_file() else []
                write_csv_atomic(rpath, REVISIONS_HEADER, old + [list(r) for r in revisions], appendix)
            for r in rows:
                r.update(updates.get(r["key"], {}))
            write_csv_atomic(scr / WORKBOOK, schema.WORKBOOK_COLUMNS,
                             ([r[c] for c in schema.WORKBOOK_COLUMNS] for r in rows), appendix)
    return MergeResult(path=path, stage=stage, rows=len(recs), decided=decided, changed=len(updates),
                       revisions=len(revisions), written=written)


# ------------------------------------------------------------------ intake

INBOX = "inbox"
REFUSED_SUFFIX = ".refused.txt"


@dataclass
class IntakeResult:
    copied: list[str] = field(default_factory=list)
    merged: list[MergeResult] = field(default_factory=list)
    refused: dict[str, list[str]] = field(default_factory=dict)
    abstracts: object | None = None  # aac.prisma.abstracts.Result
    new_batches: list[tuple[Path, int]] = field(default_factory=list)

    def line(self) -> str:
        pending = getattr(self.abstracts, "pending", 0)
        return (f"prisma-intake: merged={len(self.merged)} refused={len(self.refused)} abstracts_pending={pending} "
                f"new_batches={len(self.new_batches)}")


def _copy_checked(src: Path, dst: Path, appendix: Path | None) -> None:
    """Copy src to dst through a .tmp sibling and os.replace, then compare the bytes (RuntimeError on a difference).

 appendix given: dst must lie under appendix-c/screening/ (assert_confined); None: a loki target."""
    target = assert_confined(dst, appendix) if appendix is not None else Path(dst)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        tmp.write_bytes(Path(src).read_bytes())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    if not filecmp.cmp(src, target, shallow=False):
        raise RuntimeError(f"copy check failed: {target} differs from {src}")


def _refused_text(name: str, session_date: str, problems: Sequence[str]) -> str:
    return (f"{name}: refused by aac prisma batch intake (session date {session_date}); nothing of it was merged.\n"
            f"Fix every problem below, then return the file again under the same name.\n\n"
            + "".join(f"{p}\n" for p in problems))


def intake(appendix: Path, *, session_date: str, screener: str, loki_inbox: Path | None = None,
           loki_batches: Path | None = None, opener=None, sleep=None, clock=None) -> IntakeResult:
    """Copy new files from the loki inbox (cmp-checked), merge every inbox CSV in name order through merge_batch,
 move merged files to batches/merged/, leave refused files with <name>.refused.txt, fetch the abstracts of new
 survivors, issue the next abstract and full-text batches and copy them to loki_batches (cmp-checked).

 A file whose name is already in batches/merged/ is never copied again from loki (and refused when it is
 found in the inbox). No commit is made here; the caller commits the changed files with explicit paths."""
    from aac.prisma.abstracts import fetch_abstracts

    appendix = Path(appendix)
    session_date = _iso_date(session_date)
    screener = (screener or "").strip()
    if not screener or "\n" in screener or "\r" in screener:
        raise ValueError("--screener needs a one-line name")
    scr = screening_dir(appendix)
    inbox, merged_dir = scr / INBOX, scr / BATCHES / MERGED
    res = IntakeResult()
    if loki_inbox is not None:
        loki_inbox = Path(loki_inbox)
        if not loki_inbox.is_dir():
            raise FileNotFoundError(f"loki inbox {loki_inbox} is not a directory")
        for src in sorted(loki_inbox.iterdir()):
            if not (src.is_file() and src.suffix == ".csv") or (merged_dir / src.name).exists():
                continue
            dst = inbox / src.name
            if dst.exists():
                if not filecmp.cmp(src, dst, shallow=False):
                    res.refused[src.name] = [f"row 1 key -: the loki inbox copy differs from inbox/{src.name}; "
                                             "remove one of the two and return the file again"]
                continue
            _copy_checked(src, dst, appendix)
            res.copied.append(src.name)
    for path in sorted(inbox.glob("*.csv")) if inbox.is_dir() else []:
        name = path.name
        note = inbox / f"{name}{REFUSED_SUFFIX}"
        if name in res.refused:
            continue
        if (merged_dir / name).exists():
            res.refused[name] = [f"row 1 key -: batches/{MERGED}/{name} exists (a file of that name was merged "
                                 "before); return the file under a new name"]
            continue
        try:
            out = merge_batch(appendix, path, screener=screener, session_date=session_date)
        except MergeRefused as exc:
            res.refused[name] = exc.problems
            continue
        except (RulesError, ValueError, KeyError, RuntimeError) as exc:
            res.refused[name] = [f"row 1 key -: {exc}"]
            continue
        target = assert_confined(merged_dir / name, appendix)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, target)
        note.unlink(missing_ok=True)
        res.merged.append(out)
    for name, problems in res.refused.items():
        write_text_atomic(inbox / f"{name}{REFUSED_SUFFIX}", _refused_text(name, session_date, problems), appendix)
    res.abstracts = fetch_abstracts(appendix, opener=opener, sleep=sleep, clock=clock)
    for stage in ("abstract", "fulltext"):
        path, n = export_stage_batch(appendix, stage)
        if path is not None:
            res.new_batches.append((path, n))
    if loki_batches is not None:
        loki_batches = Path(loki_batches)
        if not loki_batches.is_dir():
            raise FileNotFoundError(f"loki batches folder {loki_batches} is not a directory")
        for path, _ in res.new_batches:
            dst = loki_batches / path.name
            if dst.exists() and not filecmp.cmp(path, dst, shallow=False):
                raise FileExistsError(f"{dst} exists and differs from {path.name}; a batch file is never overwritten")
            if not dst.exists():
                _copy_checked(path, dst, None)
    return res


# ------------------------------------------------------------------ CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma batch")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("export", help="title: every undecided row as title-NN.csv batches; abstract: the next "
                                      "abstract-NN.csv with every eligible row not issued before")
    p.add_argument("--stage", choices=["title", *sorted(ELIGIBLE)], required=True)
    p.add_argument("--size", type=int, default=300)
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("intake", help="copy, merge and file every returned batch; fetch abstracts; issue the next "
                                      "abstract and full-text batches")
    p.add_argument("--session-date", required=True, help="YYYY-MM-DD: the UTC date the files came back")
    p.add_argument("--screener", required=True)
    p.add_argument("--loki-inbox", default=None)
    p.add_argument("--loki-batches", default=None)
    p.add_argument("--appendix-dir", default=None)
    p = sub.add_parser("merge", help="validate a returned batch and merge it into the workbook (all or nothing)")
    p.add_argument("file")
    p.add_argument("--screener", required=True)
    p.add_argument("--session-date", required=True, help="YYYY-MM-DD: the UTC date the file came back")
    p.add_argument("--allow-revise", action="store_true")
    p.add_argument("--stage", choices=sorted(STAGES), default=None)
    p.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        if args.sub == "intake":
            ires = intake(appendix, session_date=args.session_date, screener=args.screener,
                          loki_inbox=Path(args.loki_inbox) if args.loki_inbox else None,
                          loki_batches=Path(args.loki_batches) if args.loki_batches else None)
            for name in ires.copied:
                print(f"copied: loki phase6-inbox/{name} -> {_rel(screening_dir(appendix) / INBOX / name, appendix)}")
            for m in ires.merged:
                print(f"merge: {MERGED}/{m.path.name} stage={m.stage} rows={m.rows} decided={m.decided} "
                      f"changed={m.changed}")
            for name, problems in ires.refused.items():
                print(f"refused: {name} ({len(problems)} problems; see inbox/{name}{REFUSED_SUFFIX})")
                for prob in problems:
                    print(f"  {prob}", file=sys.stderr)
            print(ires.abstracts.line())
            for path, n in ires.new_batches:
                print(f"export: {_rel(path, appendix)} stage={path.name.rsplit('-', 1)[0]} rows={n}")
            print(ires.line())
            return 1 if ires.refused else 0
        if args.sub == "export":
            if args.stage != "title":
                path, n = export_stage_batch(appendix, args.stage)
                print(f"export: {_rel(path, appendix)} stage={args.stage} rows={n}" if path is not None
                      else f"export: stage={args.stage} rows=0 (nothing to issue)")
                return 0
            for path, n in export_title_batches(appendix, size=args.size):
                print(f"export: {_rel(path, appendix)} stage=title rows={n}")
            return 0
        res = merge_batch(appendix, Path(args.file), screener=args.screener, session_date=args.session_date,
                          allow_revise=args.allow_revise, stage=args.stage)
    except MergeRefused as exc:
        print(f"aac prisma batch merge: refused ({len(exc.problems)} problems)", file=sys.stderr)
        for prob in exc.problems:
            print(prob, file=sys.stderr)
        return 1
    except (RulesError, FileExistsError, FileNotFoundError, KeyError, LookupError, ValueError, RuntimeError,
            OutsideScreening) as exc:
        print(f"aac prisma batch {args.sub}: {exc}", file=sys.stderr)
        return 1
    tail = "" if res.written else " (no write)"
    rev = f" revisions={res.revisions}" if res.revisions else ""
    print(f"merge: {_rel(res.path, appendix)} stage={res.stage} rows={res.rows} decided={res.decided} "
          f"changed={res.changed}{rev}{tail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
