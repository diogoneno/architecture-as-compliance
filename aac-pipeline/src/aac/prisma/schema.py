"""The screening workbook schema (PRISMA-01).

The workbook is appendix-c/screening/workbook.csv: the 11 columns of the canonical v05 prisma_records.csv in
their original order, copied field for field, followed by ADDED_COLUMNS. Later plans and the analyst's
returned batch files read these names, so a rename after decisions exist needs a migration.
"""
from __future__ import annotations

CANONICAL_RUN = "run-20261013-1211Z-v05-all-records"
CANONICAL_UNIQUE = 3491
# The run CANONICAL_RUN superseded (its README.md says so); INIT.md quotes that line as evidence for D1.
SUPERSEDED_RUN = "run-20261013-1011Z-v04c-all-records"

ORIGINAL_COLUMNS: tuple[str, ...] = (
    "key",
    "title",
    "year",
    "sources",
    "families",
    "screen_decision",
    "exclude_reason",
    "retrieved",
    "fulltext_decision",
    "fulltext_reason",
    "included_in_chapter2",
)

ADDED_COLUMNS: tuple[str, ...] = (
    "doi_url",
    "title_dup_group",
    "title_rule_id",
    "rule_version",
    "screener",
    "session_date",
    "proposal",
    "proposal_source",
    "abstract_source",
    "abstract_response_sha256",
    "abstract_decision",
    "abstract_reason",
    "second_screen_decision",
    "second_screen_session_date",
    "notes",
)

WORKBOOK_COLUMNS: tuple[str, ...] = ORIGINAL_COLUMNS + ADDED_COLUMNS

# Cells only the analyst fills; init refuses to overwrite a workbook in which any of them is non-empty.
DECISION_COLUMNS: tuple[str, ...] = (
    "screen_decision",
    "exclude_reason",
    "retrieved",
    "fulltext_decision",
    "fulltext_reason",
    "included_in_chapter2",
    "title_rule_id",
    "abstract_decision",
    "abstract_reason",
    "second_screen_decision",
    "notes",
)

# The controlled exclusion vocabulary, verbatim and in the order of REQUIREMENTS.md PRISMA-01.
REASON_CODES: tuple[str, ...] = (
    "out of scope",
    "no obligation or artifact",
    "superseded version",
    "not English",
    "deepfake-only",
    "policy commentary",
    "duplicate at title level",
)

DECISIONS: tuple[str, ...] = ("include", "exclude")
RETRIEVED: tuple[str, ...] = ("yes", "no")
CH2_VALUES: tuple[str, ...] = ("yes", "no")

# column -> (meaning, who fills it, stage). No `|` in any text (INIT.md renders this as a Markdown table).
COLUMN_DOCS: dict[str, tuple[str, str, str]] = {
    "key": ("record key from the v05 run: DOI, 10.48550/arxiv DOI, OpenAlex URL or normalized title", "copied from the run", "identification"),
    "title": ("title as recorded by the v05 run", "copied from the run", "identification"),
    "year": ("publication year as recorded by the v05 run", "copied from the run", "identification"),
    "sources": ("API sources that returned the record", "copied from the run", "identification"),
    "families": ("Boolean query families (Table C.1) that matched", "copied from the run", "identification"),
    "screen_decision": ("title-screen decision: include or exclude", "analyst", "title screen"),
    "exclude_reason": ("reason code for a title or full-text exclusion (controlled vocabulary)", "analyst", "title screen"),
    "retrieved": ("full text retrieved: yes or no", "analyst", "retrieval"),
    "fulltext_decision": ("full-text decision: include or exclude", "analyst", "eligibility"),
    "fulltext_reason": ("reason code for a full-text exclusion (controlled vocabulary)", "analyst", "eligibility"),
    "included_in_chapter2": ("record cited in Chapter 2: yes or no", "analyst", "inclusion"),
    "doi_url": ("resolver URL derived from key without network; empty for a title key", "aac prisma workbook init", "identification"),
    "title_dup_group": ("id shared by rows whose normalized titles are equal; a hint for the screener, never a decision", "aac prisma workbook init", "identification"),
    "title_rule_id": ("id of the approved screening rule that the title decision cites", "analyst", "title screen"),
    "rule_version": ("sha256 of the approved rule file in force for the decision", "aac prisma batch merge", "title screen"),
    "screener": ("who screened the row", "analyst (recorded on merge)", "title screen"),
    "session_date": ("UTC date on which the screener returned the batch that carried the title decision", "aac prisma batch merge", "title screen"),
    "proposal": ("automation proposal; stays empty (no automation tool in this phase)", "nobody", "title screen"),
    "proposal_source": ("tool, version and prompt hash of a proposal; stays empty", "nobody", "title screen"),
    "abstract_source": ("API that supplied the abstract; the text lives in appendix-c/screening/abstracts.csv", "aac prisma abstracts", "abstract screen"),
    "abstract_response_sha256": ("sha256 of the raw API response that supplied the abstract", "aac prisma abstracts", "abstract screen"),
    "abstract_decision": ("abstract-screen decision: include or exclude", "analyst", "abstract screen"),
    "abstract_reason": ("reason code for an abstract exclusion (controlled vocabulary)", "analyst", "abstract screen"),
    "second_screen_decision": ("independent second-screen decision on the sampled rows", "second screener", "second screen"),
    "second_screen_session_date": ("UTC date on which the second screener returned the sample", "aac prisma batch merge", "second screen"),
    "notes": ("free-text note by the screener", "analyst", "any stage"),
}

assert tuple(COLUMN_DOCS) == WORKBOOK_COLUMNS, "COLUMN_DOCS must document every workbook column in order"
assert len(WORKBOOK_COLUMNS) == 26
