"""`aac prisma paste --as-of YYYY-MM-DD` (PRISMA-07, PRISMA-02 Item 8 part; 9).

The paste-ready Appendix C text is generated last, from computed files only. build refuses (PasteRefused naming
the file) unless appendix-c/screening/table_c2_screening.csv exists for the same --as-of and for the current
workbook digest (its header's workbook_sha256), and pilot-gate.md, a reference-verification-YYYY-MM-DD.csv (the
newest, refcheck.latest_log) and consistency-differences.md exist. It reads those files plus
chapter2-comparison-set.csv (and, through refcheck.cell, the draft's reference list for the M of "N of M", which
must equal the comparison set's row count) and writes two files into aac-pipeline/reports/appendix-c/ only:

 APPENDIX-C-PASTE.md section 2.1 sentences, Table C.1 rows, Table C.2 rows and notes, the Figure C.1 caption, the
 PRISMA Item 8 statement and the flags for the analyst (stale pointers, never edited here);
 README.md every input path (relative to the dissertation root) with its sha256, the paste file's
 sha256 and the regeneration command.

Wording rules: a stage is written in the past tense only when its pending number is 0; otherwise the sentence starts
`As of <d Month yyyy>`. Every numeric token of the paste file must occur in an input file, be the row count of an
input CSV or a per-class row count of the reference log (the tally behind "N of M"); `d Month yyyy` and ISO dates
must be input dates; STRUCTURAL_TOKENS (PRISMA 2020, section 2.1, Item 8) and identifiers such as PRISMA-08, C.2 or
a file name in a code span are exempt. A token outside the inputs refuses the whole write. Dates come from --as-of; nothing reads the clock.
Exit codes: 0 written, 1 refused, 2 usage.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from aac.prisma import refcheck
from aac.prisma._io import default_appendix_dir, read_csv, screening_dir
from aac.prisma.counts import TABLE_FILE, as_int, as_of_date, read_table
from aac.prisma.figure import FIG_FILE, MONTHS, long_date
from aac.run.hashing import sha256_file
from aac.run.paths import reports_dir
from aac.run.paths import root as project_root

FEED_NAME = "appendix-c"
FEED_DIR = reports_dir() / FEED_NAME  # out_root/reports/appendix-c; the real namespace when AAC_OUT is unset
PASTE_FILE = "APPENDIX-C-PASTE.md"
README_FILE = "README.md"
GATE_FILE = "pilot-gate.md"
DIFF_FILE = "consistency-differences.md"
SET_FILE = "chapter2-comparison-set.csv"
WORKBOOK = "workbook.csv"
STRUCTURAL_TOKENS = ("2020", "2.1", "8")
NOT_SEARCHED = "not searched (removed)"
HAND_SEARCH_CLAUSE = ("thesis repositories (OATD, ProQuest, DART-Europe and the University of Essex research "
                      "repository) and IEEE Xplore are searched by hand with the same families;")
SEQUENCE_SENTENCE = ("The review was drafted first; the systematic identification search was run after it, and the "
                     "screen recorded in Appendix C was run afterwards as an audit of the review's coverage, not as "
                     "the step that selected the works it discusses.")
AUDIT_NOTE = ("The screen was run as an audit after the review was drafted: the identification run post-dates the "
              "drafting of Chapter 2, so the screen checks the review's coverage and did not select its works "
              "(analyst to accept or edit).")

ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4})-(\d{2})-(\d{2})(?!\d)")
LONG_DATE_RE = re.compile(r"\b(\d{1,2}) (" + "|".join(MONTHS) + r") (\d{4})\b")
IDENT_RE = re.compile(r"\b[A-Z][A-Za-z]*-\d+\b")
# A file name in a code span is an identifier (e.g. a dated file the flags name), not a recorded value.
FILE_SPAN_RE = re.compile(r"`[\w./-]+\.(?:md|csv|png|json|txt|py)`")
NUM_RE = re.compile(r"(?<![A-Za-z.\d,])(\d{1,3}(?:,\d{3})+(?!\d)|\d+)(\.\d+)?")
# Input files are CSV: a comma there separates fields, so inputs are also read as plain digit runs.
PLAIN_NUM_RE = re.compile(r"(?<![A-Za-z.\d])(\d+)(\.\d+)?")


class PasteRefused(RuntimeError):
    """A missing or stale input, a write outside the feed folder, or a number absent from the inputs."""


# ------------------------------------------------------------------ helpers


def _rel(path: Path, base: Path) -> str:
    p, b = Path(path).resolve(), Path(base).resolve()
    return p.relative_to(b).as_posix() if p.is_relative_to(b) else Path(path).name


def _num(n) -> str:
    return f"{int(n):,}"


def _count_cell(count: str, pending: int, when: str, upstream_pending: bool) -> str:
    c = _num(count)
    if pending > 0:
        return f"{c} ({_num(pending)} pending as of {when})"
    if upstream_pending:
        return f"{c} (as of {when}; an earlier stage is pending)"
    return c


def _gate_line(gate_text: str, prefix: str) -> str | None:
    for line in gate_text.splitlines():
        s = line.strip()
        if s.startswith("- " + prefix):
            return s[2:]
    return None


def numeric_tokens(text: str) -> list[str]:
    """Numeric tokens (maximal digit runs not preceded by a letter or a dot; thousands separators removed)."""
    return [m.group(1).replace(",", "") + (m.group(2) or "") for m in NUM_RE.finditer(text)]


def input_numbers(texts: list[str]) -> tuple[set[str], set[str]]:
    """(numeric tokens, ISO dates) found in the input texts."""
    nums: set[str] = set()
    dates: set[str] = set()
    for t in texts:
        dates |= {m.group(0) for m in ISO_DATE_RE.finditer(t)}
        plain = [m.group(1) + (m.group(2) or "") for m in PLAIN_NUM_RE.finditer(t)]
        for tok in numeric_tokens(t) + plain:
            nums.add(tok)
            if "." not in tok:
                nums.add(str(int(tok)))
    return nums, dates


def check_numbers(text: str, nums: set[str], dates: set[str]) -> int:
    """Count the numeric tokens and dates of text; raise PasteRefused on the first one absent from the inputs."""
    checked = 0
    text = FILE_SPAN_RE.sub(" ", text)
    rest = text
    for m in LONG_DATE_RE.finditer(text):
        iso = f"{int(m.group(3)):04d}-{MONTHS.index(m.group(2)) + 1:02d}-{int(m.group(1)):02d}"
        if iso not in dates:
            raise PasteRefused(f"date {m.group(0)!r} is not a date of the inputs")
        checked += 1
    rest = LONG_DATE_RE.sub(" ", rest)
    for m in ISO_DATE_RE.finditer(rest):
        if m.group(0) not in dates:
            raise PasteRefused(f"date {m.group(0)!r} is not a date of the inputs")
        checked += 1
    rest = ISO_DATE_RE.sub(" ", rest)
    rest = IDENT_RE.sub(" ", rest)
    for tok in numeric_tokens(rest):
        if tok in STRUCTURAL_TOKENS:
            continue
        plain = tok if "." in tok else str(int(tok))
        if tok not in nums and plain not in nums:
            raise PasteRefused(f"number {tok!r} in the paste text occurs in no input file")
        checked += 1
    return checked


def check_clean(text: str) -> None:
    """No clock fragment, no blank placeholder, no absolute path in a generated feed file."""
    for pat, what in ((r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}", "a clock time"), (r"____", "a blank placeholder"),
                      (r"(?<![\w.])/(?:home|tmp|mnt|usr|root|Users)/", "an absolute path")):
        m = re.search(pat, text)
        if m:
            raise PasteRefused(f"generated text holds {what}: {m.group(0)!r}")


# ------------------------------------------------------------------ inputs


def _inputs(appendix: Path, as_of: str, draft: Path | None) -> dict:
    scr = screening_dir(appendix)
    table = scr / TABLE_FILE
    if not table.is_file():
        raise PasteRefused(f"appendix-c/screening/{TABLE_FILE} is absent; run `aac prisma count --as-of {as_of}` first")
    try:
        head, rows = read_table(table)
    except ValueError as exc:
        raise PasteRefused(str(exc)) from exc
    if head["as_of"] != as_of:
        raise PasteRefused(f"appendix-c/screening/{TABLE_FILE} was counted as of {head['as_of']}, not {as_of}")
    wb = scr / WORKBOOK
    if not wb.is_file() or sha256_file(wb) != head["workbook_sha256"]:
        raise PasteRefused(f"appendix-c/screening/{TABLE_FILE} is stale: its workbook_sha256 is not the digest of "
                           f"appendix-c/screening/{WORKBOOK} now; re-run `aac prisma count --as-of {as_of}`")
    gate = scr / GATE_FILE
    if not gate.is_file():
        raise PasteRefused(f"appendix-c/screening/{GATE_FILE} is absent; run `aac prisma pilot gate` first")
    log = refcheck.latest_log(appendix)
    if log is None:
        raise PasteRefused("no appendix-c/reference-verification-YYYY-MM-DD.csv; run `aac prisma refcheck run` first")
    diff = scr / DIFF_FILE
    if not diff.is_file():
        raise PasteRefused(f"appendix-c/screening/{DIFF_FILE} is absent; run `aac prisma consistency check` first")
    cset = scr / SET_FILE
    if not cset.is_file():
        raise PasteRefused(f"appendix-c/screening/{SET_FILE} is absent; run `aac prisma consistency build-set` first")
    return {"table": table, "head": head, "rows": rows, "gate": gate, "log": log, "diff": diff, "set": cset}


# ------------------------------------------------------------------ the text


def render(inp: dict, as_of: str, cell_lines: list[str]) -> str:
    by = {r["stage"]: r for r in inp["rows"]}
    gate_text = inp["gate"].read_text(encoding="utf-8")
    diff_text = inp["diff"].read_text(encoding="utf-8")
    when = long_date(as_of)
    ident_when = long_date(inp["head"]["identification_run_date"])

    def c(stage: str) -> int:
        return int(by[stage]["count"])

    def p(stage: str) -> int:
        return as_int(by[stage]["pending"])

    title_p, abs_p, ret_p, ft_p = p("records_screened"), p("records_excluded_abstract"), p("reports_assessed"), \
        p("studies_included")
    n_dup = c("duplicates_removed_title")
    appendix_rel = lambda path: _rel(path, inp["appendix"].parent)  # noqa: E731
    consistency_line = next((ln.strip() for ln in diff_text.splitlines() if ln.startswith("consistency: set=")), None)
    if consistency_line is None:
        raise PasteRefused(f"appendix-c/screening/{DIFF_FILE} holds no `consistency: set=` line")
    table_c1 = next((ln for ln in cell_lines if ln.startswith("table-c1: ")), None)
    if table_c1 is None:
        raise PasteRefused("refcheck cell printed no `table-c1:` line")
    table_c1 = table_c1[len("table-c1: "):]

    out: list[str] = [
        "# Appendix C: paste-ready text",
        "",
        f"As of: {when}",
        "",
        "Generated by `aac prisma paste` from the files below; regenerate, never edit (README.md beside this file "
        "gives each input's sha256 and the command). Every number is read from these files. The analyst pastes "
        "from here into the dissertation; nothing in the dissertation is edited by this script.",
        "",
        "Inputs:",
        "",
        f"- `{appendix_rel(inp['table'])}` (counted as of {when}, workbook digest in its header)",
        f"- `{appendix_rel(inp['gate'])}`",
        f"- `{appendix_rel(inp['log'])}`",
        f"- `{appendix_rel(inp['diff'])}`",
        f"- `{appendix_rel(inp['set'])}` ({inp['set_rows']} references)",
        "",
        "## Section 2.1 (Review method)",
        "",
        "Identification sentence (replaces the sentence that begins `Identification searches were run`; the "
        "hand-search clause is dropped, see below):",
        "",
    ]
    ident = (f"> Identification searches were run on {ident_when} through the OpenAlex, arXiv and Semantic Scholar "
             f"APIs with the three Boolean families of Table C.1; they returned {_num(c('identified_total'))} records "
             f"(OpenAlex {_num(c('identified_openalex'))}, arXiv {_num(c('identified_arxiv'))}, Semantic Scholar "
             f"{_num(c('identified_s2'))}), of which {_num(c('duplicates_removed_api'))} were duplicates across sources")
    if n_dup:
        ident += (f" and a further {_num(n_dup)} were confirmed by the screener as duplicates at title level, leaving "
                  f"{_num(c('records_to_screen'))} records to screen;")
    else:
        ident += f", leaving {_num(c('records_to_screen'))} records to screen;"
    ident += (" regulatory and standards material was retrieved directly from EUR-Lex, the European Commission's "
              "digital-strategy pages and CEN-CENELEC.")
    out += [ident, ""]
    out += [
        f"Hand-search clause: delete `{HAND_SEARCH_CLAUSE}` (hand-searched sources: {NOT_SEARCHED}; no hand search "
        "was recorded).",
        "",
        "Screening sentences (add after `Records are screened on title and abstract and then on full text.`):",
        "",
    ]
    out += ["> " + " ".join(screening_sentences(by, when))]
    out += [
        "",
        "Flow-diagram sentence (replaces `the PRISMA flow diagram is added once screening is complete`):",
        "",
        f"> the PRISMA flow diagram (Figure C.1) gives the record as of {when}, with the number still pending at "
        "each stage.",
        "",
        "Sequence sentence (analyst to accept or edit):",
        "",
        f"> {SEQUENCE_SENTENCE}",
        "",
        "## Table C.1 rows",
        "",
        "| Element | Specification |",
        "|---|---|",
        (f"| Databases | Systematic identification by API on {ident_when}: OpenAlex (title and abstract, English; "
         "indexes Crossref and publisher metadata and so covers the ACL Anthology, IEEE, ACM, Springer and Elsevier "
         f"venues; {_num(c('identified_openalex'))} records), arXiv (all metadata fields; "
         f"{_num(c('identified_arxiv'))} records) and Semantic Scholar (bulk search, title and abstract; "
         f"{_num(c('identified_s2'))} records). Hand-searched sources (thesis repositories and IEEE Xplore): "
         f"{NOT_SEARCHED}. Regulatory and standards texts retrieved directly from EUR-Lex, the European "
         "Commission's digital-strategy pages and CEN-CENELEC. Google Scholar was used for discovery only and is "
         "not counted. |"),
        (f"| Update search | Systematic identification run, {ident_when}: Table C.2; as of {when}, "
         f"{_num(c('records_screened'))} of {_num(c('records_to_screen'))} records screened on title"
         + (f", {_num(title_p)} pending." if title_p else ".") + " |"),
        f"| Reference verification | {table_c1} |",
        f"| Hand-searched sources | {NOT_SEARCHED}; PRISMA-08 not run |",
        "",
        "Update search: only the systematic-run sentence above is generated; the exploratory update-search "
        "sentence of the current row is left to the analyst.",
        "",
        "## Table C.2 rows",
        "",
        "| PRISMA 2020 stage | Count |",
        "|---|---|",
    ]
    out += table_c2_rows(by, when)
    out += ["", "## Table C.2 notes", ""]
    out += table_c2_notes(by, when, consistency_line, appendix_rel(inp["diff"]))
    out += ["", "## Figure C.1 caption", ""]
    caption = (f"Figure C.1. PRISMA 2020 flow diagram of the database search as of {when} (identification run "
               f"{ident_when}). Counts are those recorded in the screening workbook on that date")
    if title_p or abs_p or ret_p or ft_p:
        caption += ("; a box with work still pending states its pending number"
                    + (f" ({_num(title_p)} records were pending title screening)" if title_p else "") + ".")
    else:
        caption += "; no stage had pending work."
    caption += (f" The other-methods arm ({_num(c('identified_other_methods'))} records, Table C.2) is not drawn. "
                f"File: `appendix-c/screening/{FIG_FILE}`.")
    out += [caption, ""]
    out += ["## PRISMA Item 8 (selection process)", ""]
    out += ["> " + " ".join(item8_sentences(by, when, gate_text, appendix_rel(inp["gate"])))]
    out += [
        "",
        "## Flags for the analyst",
        "",
        "- Earlier search run v04c is superseded by `appendix-c/run-20261013-1211Z-v05-all-records/`",
        f"run is `{inp['head']['run']}` (`appendix-c/screening/INIT.md`).",
        "- `chapters/REVISION-TODO-2026-10-13.md`: its header names the v04c run, not the v05 run.",
        "- `appendix-c/APPENDIX-C-DRAFT.md`: its headline identification total is the v04c figure; the v05 run "
        f"identified {_num(c('identified_total'))} records.",
        "- Table C.1 `Reference verification` cell: it cites `reference-verification-2026-10-13.csv`, a log that "
        f"does not exist; the generated row above cites `{inp['log'].name}`.",
        "",
    ]
    return "\n".join(out)


def screening_sentences(by: dict, when: str) -> list[str]:
    def c(stage: str) -> int:
        return int(by[stage]["count"])

    def p(stage: str) -> int:
        return as_int(by[stage]["pending"])

    s: list[str] = []
    n, scr, tp = c("records_to_screen"), c("records_screened"), p("records_screened")
    if tp > 0:
        s.append(f"As of {when}, {_num(scr)} of {_num(n)} records had been screened on title; {_num(tp)} were pending.")
    else:
        s.append(f"All {_num(n)} records were screened on title; {_num(c('records_excluded_title'))} were excluded at "
                 "title.")
    incl = c("records_screened") - c("records_excluded_title")
    if incl == 0:
        return s
    ap = p("records_excluded_abstract")
    if ap > 0:
        s.append(f"As of {when}, {_num(incl - ap)} of the {_num(incl)} title includes had been screened on abstract; "
                 f"{_num(ap)} were pending.")
    else:
        s.append(f"Of the {_num(incl)} title includes, {_num(c('records_excluded_abstract'))} were excluded on "
                 f"abstract and {_num(c('reports_sought'))} reports were sought for retrieval.")
    sought = c("reports_sought")
    if sought == 0:
        return s
    rp = p("reports_assessed")
    if rp > 0:
        s.append(f"As of {when}, retrieval had been decided for {_num(sought - rp)} of {_num(sought)} reports; "
                 f"{_num(rp)} were pending.")
    else:
        s.append(f"{_num(c('reports_not_retrieved'))} reports were not retrieved and {_num(c('reports_assessed'))} "
                 "were assessed in full text.")
    assessed = c("reports_assessed")
    if assessed == 0:
        return s
    fp = p("studies_included")
    if fp > 0:
        s.append(f"As of {when}, {_num(assessed - fp)} of {_num(assessed)} full texts had been decided; {_num(fp)} "
                 "were pending.")
    else:
        s.append(f"{_num(c('reports_excluded_total'))} reports were excluded with reasons and "
                 f"{_num(c('studies_included'))} studies were included.")
    return s


def table_c2_rows(by: dict, when: str) -> list[str]:
    def c(stage: str) -> int:
        return int(by[stage]["count"])

    def p(stage: str) -> int:
        return as_int(by[stage]["pending"])

    order = ["records_screened", "records_excluded", "reports_sought", "reports_assessed", "studies_included"]
    upstream = {st: any(p(prev) > 0 for prev in order[: order.index(st)]) for st in order}
    upstream["records_excluded_title"] = False
    upstream["records_excluded_abstract"] = p("records_screened") > 0
    upstream["reports_not_retrieved"] = upstream["reports_assessed"]
    upstream["reports_excluded_total"] = upstream["studies_included"]

    def row(stage: str, label: str | None = None) -> str:
        cell = _count_cell(by[stage]["count"], p(stage), when, upstream.get(stage, False))
        return f"| {label or by[stage]['label']} | {cell} |"

    dup = _num(c("duplicates_removed_total"))
    if c("duplicates_removed_title"):
        dup += (f" ({_num(c('duplicates_removed_api'))} across sources; {_num(c('duplicates_removed_title'))} at "
                "title level)")
    reasons = [(r["stage"].split(":", 1)[1], int(r["count"])) for r in by.values()
               if r["stage"].startswith("reports_excluded:") and int(r["count"]) > 0]
    excl = row("reports_excluded_total")
    if reasons:
        excl = excl[:-2] + "; " + "; ".join(f"{k} {_num(v)}" for k, v in reasons) + " |"
    return [
        row("identified_openalex"),
        row("identified_arxiv"),
        row("identified_s2"),
        row("identified_total"),
        f"| Records identified from registers and thesis repositories | {NOT_SEARCHED} |",
        row("identified_other_methods"),
        f"| {by['duplicates_removed_total']['label']} | {dup} |",
        row("records_screened"),
        row("records_excluded"),
        row("records_excluded_title"),
        row("records_excluded_abstract"),
        row("reports_sought"),
        row("reports_not_retrieved"),
        row("reports_assessed"),
        excl,
        row("studies_included"),
    ]


def table_c2_notes(by: dict, when: str, consistency_line: str, diff_rel: str) -> list[str]:
    def c(stage: str) -> int:
        return int(by[stage]["count"])

    def p(stage: str) -> int:
        return as_int(by[stage]["pending"])

    notes: list[str] = []
    n_dup = c("duplicates_removed_title")
    if n_dup:
        notes.append(f"- De-duplication in two steps: {_num(c('duplicates_removed_api'))} duplicates across sources "
                     f"(identification run) and {_num(n_dup)} at title level, each confirmed by the screener with the "
                     f"reason `duplicate at title level`; records to screen are therefore "
                     f"{_num(c('records_to_screen'))}.")
    else:
        notes.append(f"- De-duplication: as of {when} no record had been confirmed as a duplicate at title level; "
                     f"{_num(c('title_dup_candidate_groups'))} groups ({_num(c('title_dup_candidate_rows'))} rows) "
                     "share a normalized title and are flagged to the screener, never removed automatically, so "
                     f"duplicates removed stay at {_num(c('duplicates_removed_api'))} and records to screen at "
                     f"{_num(c('records_to_screen'))}.")
    survivors = c("records_screened") - c("records_excluded_title")
    if survivors:
        notes.append(f"- Abstract coverage: of {_num(survivors)} title includes, {_num(c('abstracts_with_text'))} "
                     f"have an abstract from an API and {_num(c('abstracts_without_text'))} have none available"
                     + (f"; {_num(p('abstracts_without_text'))} were not yet fetched" if p("abstracts_without_text")
                        else "")
                     + "; a record without an abstract is decided from its title and landing page.")
    else:
        notes.append(f"- Abstract coverage: as of {when} no record had been included on title, so no abstract had "
                     "been fetched or screened.")
    notes.append(f"- Other methods: the {_num(c('identified_other_methods'))} records identified via other methods "
                 "are counted in this table but did not pass through the database screen; Figure C.1 shows the "
                 "database arm only.")
    notes.append(f"- Consistency with Chapter 2 (PRISMA-05, not marked satisfied): `{consistency_line}`; every "
                 f"difference is listed in `{diff_rel}` for the analyst.")
    notes.append(f"- {AUDIT_NOTE}")
    return notes


def item8_sentences(by: dict, when: str, gate_text: str, gate_rel: str) -> list[str]:
    def c(stage: str) -> int:
        return int(by[stage]["count"])

    s: list[str] = []
    names = by["screeners_recorded"]["source"].partition(": ")[2]
    n_names = c("screeners_recorded")
    if n_names == 0:
        s.append("Selection process (PRISMA 2020 Item 8): one screener, the author, screens every record; no title "
                 f"decision had been merged as of {when}, so no screener identity is yet recorded in the workbook.")
    elif n_names == 1:
        s.append(f"Selection process (PRISMA 2020 Item 8): one screener ({names}) screened every record decided "
                 f"as of {when}.")
    else:
        s.append(f"Selection process (PRISMA 2020 Item 8): {_num(n_names)} screeners ({names}) made the title "
                 f"decisions recorded as of {when}.")
    verdict = _gate_line(gate_text, "Verdict:")
    rate = _gate_line(gate_text, "Rate:")
    if verdict is None or rate is None:
        raise PasteRefused(f"{gate_rel} holds no `- Verdict:` or `- Rate:` line")
    if "declined the pilot" in gate_text:
        s.append(f"The analyst declined the timed pilot, so no screening rate was measured (`{gate_rel}`: "
                 f"`{verdict}`; `{rate}`).")
    else:
        s.append(f"The timed pilot is recorded in `{gate_rel}`: `{verdict}`; `{rate}`.")
    sampled, decided, agree = c("second_screen_sampled"), c("second_screen_decided"), c("second_screen_agreements")
    kappa = by["second_screen_kappa"]["count"]
    share = by["second_screen_share_percent"]["count"]
    if decided == 0:
        s.append(f"A blind second screen of a {share} % hash sample of the title-decided records had not started as "
                 f"of {when} ({_num(decided)} of {_num(sampled)} sampled records had a second decision), so Cohen's "
                 f"kappa is {kappa}.")
    else:
        second = (f"A blind second screen of a {share} % hash sample of the title-decided records had, as of {when}, "
                  f"a second decision on {_num(decided)} of {_num(sampled)} sampled records ({_num(agree)} "
                  "agreements); Cohen's kappa ")
        second += f"{kappa}." if re.fullmatch(r"-?\d+\.\d{3}", kappa) else f"is {kappa}."
        s.append(second)
    if c("automation_proposals") == 0:
        s.append("No automation tool was used: no record carries a proposal, so model, version, prompt hash and date "
                 "are not applicable, and precision is not applicable (no proposal column used; no automation tool).")
    else:
        s.append(f"{_num(c('automation_proposals'))} records carry an automation proposal; the tool, version, prompt "
                 "hash and date are recorded in `appendix-c/screening/automation.json`.")
    return s


# ------------------------------------------------------------------ build


def build(appendix: Path, root: Path, as_of: str, draft: Path | None = None) -> tuple[Path, Path, int]:
    """Write APPENDIX-C-PASTE.md and README.md into <root>/reports/appendix-c/; return (paste, readme, numbers)."""
    appendix, root = Path(appendix), Path(root)
    as_of_date(as_of)
    inp = _inputs(appendix, as_of, draft)
    inp["appendix"] = appendix
    _, set_rows = read_csv(inp["set"])
    inp["set_rows"] = len(set_rows)
    try:
        cell_lines = refcheck.cell(inp["log"], draft)
    except (refcheck.RefcheckRefused, ValueError) as exc:
        raise PasteRefused(f"{inp['log'].name}: {exc}") from exc
    m = re.match(r"^cell: (\d+) of (\d+)$", cell_lines[0])
    if not m or int(m.group(2)) != len(set_rows):
        raise PasteRefused(f"refcheck `{cell_lines[0]}` disagrees with the {len(set_rows)} rows of {SET_FILE}")
    text = render(inp, as_of, cell_lines)
    check_clean(text)
    texts = [p.read_text(encoding="utf-8") for p in (inp["table"], inp["gate"], inp["log"], inp["diff"], inp["set"])]
    nums, dates = input_numbers(texts)
    _, log_rows = read_csv(inp["log"])
    tally = refcheck.tally(log_rows, int(m.group(2)))
    nums |= {str(v) for v in tally.values()}
    nums |= {str(len(read_csv(p)[1])) for p in (inp["log"], inp["set"])} | {str(len(inp["rows"]))}
    numbers = check_numbers(text, nums, dates)
    feed = (reports_dir(root) / FEED_NAME).resolve()
    paste_path = _write_feed(feed, PASTE_FILE, text)
    inputs = [inp["table"], inp["gate"], inp["log"], inp["diff"], inp["set"],
              Path(draft) if draft is not None else None]
    readme = readme_text(appendix, root, as_of, inputs, paste_path)
    check_clean(readme)
    readme_path = _write_feed(feed, README_FILE, readme)
    return paste_path, readme_path, numbers


def readme_text(appendix: Path, root: Path, as_of: str, inputs: list, paste_path: Path) -> str:
    from aac.prisma.references import default_draft

    base = Path(appendix).parent
    draft = inputs[-1] if inputs[-1] is not None else default_draft()
    lines = [
        f"# {_rel(paste_path.parent, Path(root).parent)}: Appendix C paste-ready export (Phase 6)",
        "",
        f"Generated by `aac prisma paste --as-of {as_of}`; regenerate with `make appendix-c AS_OF={as_of}` from "
        f"`{Path(root).name}/`, never edit. The folder is exempt from `aac reports index` and from the clean-checkout "
        "report comparison (like ch4-feed and ch5-feed) and never enters the evidence pack.",
        "",
        "| input | sha256 |",
        "|---|---|",
    ]
    for p in inputs[:-1]:
        lines.append(f"| `{_rel(p, base)}` | {sha256_file(p)} |")
    lines.append(f"| `{_rel(draft, base)}` (reference count, read by `aac prisma refcheck cell`) | "
                 f"{sha256_file(draft)} |")
    lines += [
        "",
        "| output | sha256 |",
        "|---|---|",
        f"| `{_rel(paste_path, Path(root).parent)}` | {sha256_file(paste_path)} |",
        "",
    ]
    return "\n".join(lines)


def _write_feed(feed: Path, name: str, text: str) -> Path:
    target = (feed / name).resolve()
    if target.parent != feed or name not in (PASTE_FILE, README_FILE):
        raise PasteRefused(f"refusing to write {target}: aac prisma paste writes only {PASTE_FILE} and {README_FILE} "
                           f"in reports/{FEED_NAME}/")
    feed.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return target


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma paste")
    ap.add_argument("--as-of", required=True, help="the record date YYYY-MM-DD; must equal the table's as_of")
    ap.add_argument("--appendix-dir", default=None)
    ap.add_argument("--draft", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    root = project_root()
    try:
        paste_path, _, numbers = build(appendix, root, args.as_of, Path(args.draft) if args.draft else None)
    except (PasteRefused, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"aac prisma paste: {exc}", file=sys.stderr)
        return 1
    print(f"paste: {_rel(paste_path, root.parent)} as_of={args.as_of} numbers={numbers} all-from-inputs=yes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
