"""`aac prisma figure --as-of YYYY-MM-DD` (PRISMA-04; 5).

Figure C.1 is drawn from appendix-c/screening/table_c2_screening.csv (the counter's output, never the workbook) with
the locked prisma-flowchart package (import name prisma_flow_diagram): database arm only (the other-methods arm is
not drawn, PHASE6-ONE-SHOT Q3 default; its 36 records stay in Table C.2 with a note), no previous-review arm, the
three API sources listed in the identification box. Every box keeps its recorded count; a box whose row has pending
work gets the line `(<n> pending as of <d Month yyyy>)` in its label, so an unfinished stage is never an empty box.

PNG only (PDF and SVG are not deterministic): the figure is built with the package's figure builder
(renderer._build_figure, because PrismaDiagram.save calls savefig without metadata control) on the Agg backend and
saved with metadata {"Software": None}; two processes give byte-identical files. The date comes from --as-of, which
must equal the table's as_of; nothing reads the clock.
Exit codes: 0 drawn, 1 refused, 2 usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import sys
from pathlib import Path

from aac.prisma._io import OutsideScreening, assert_confined, default_appendix_dir, screening_dir
from aac.prisma.counts import TABLE_FILE, as_int, as_of_date, read_table
from aac.run.hashing import sha256_file

FIG_FILE = "fig-C.1-prisma-flow.png"
MONTHS = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December")
# FlowData box key -> table_c2_screening.csv stage
BOX_STAGES = (
    ("records_screened", "records_screened"),
    ("records_excluded", "records_excluded"),
    ("dbr_sought_reports", "reports_sought"),
    ("dbr_notretrieved_reports", "reports_not_retrieved"),
    ("dbr_assessed", "reports_assessed"),
    ("dbr_excluded", "reports_excluded_total"),
    ("new_studies", "studies_included"),
)
SOURCE_ROWS = (("OpenAlex", "identified_openalex"), ("arXiv", "identified_arxiv"),
               ("Semantic Scholar", "identified_s2"))


class FigureRefused(RuntimeError):
    """The table is missing or was counted for another date."""


def long_date(iso: str) -> str:
    """2026-11-01 -> `1 November 2026` (fixed English month names; no locale)."""
    d = _dt.date.fromisoformat(iso)
    return f"{d.day} {MONTHS[d.month - 1]} {d.year}"


def flow_data(rows: list[dict[str, str]], as_of: str):
    """Pure: the table rows -> (FlowData, number of boxes labeled pending)."""
    from prisma_flow_diagram import FlowData
    from prisma_flow_diagram.model import DEFAULT_LABELS

    by = {r["stage"]: r for r in rows}

    def n(stage: str) -> int:
        return int(by[stage]["count"])

    when = long_date(as_of)
    labels: dict[str, str] = {}
    pending_boxes = 0
    for box, stage in BOX_STAGES:
        p = as_int(by[stage]["pending"])
        if p > 0:
            labels[box] = f"{DEFAULT_LABELS[box]}\n({p:,} pending as of {when})"
            pending_boxes += 1
    reasons = tuple((r["stage"].split(":", 1)[1], int(r["count"])) for r in rows
                    if r["stage"].startswith("reports_excluded:") and int(r["count"]) > 0)
    # The package draws no count for this box (only a reason list), so the total is put in the label; with no
    # reason above zero the box would otherwise read `Reports excluded` with no number at all.
    excl = labels.get("dbr_excluded", DEFAULT_LABELS["dbr_excluded"])
    head, sep, tail = excl.partition("\n")
    labels["dbr_excluded"] = f"{head} (n = {n('reports_excluded_total'):,})" + sep + tail
    if n("duplicates_removed_title") > 0:
        labels["duplicates"] = (f"{DEFAULT_LABELS['duplicates']} ({n('duplicates_removed_api'):,} across sources, "
                                f"{n('duplicates_removed_title'):,} at title level)")
    data = FlowData(
        database_results=n("identified_total"),
        database_specific_results=tuple((name, n(stage)) for name, stage in SOURCE_ROWS),
        duplicates=n("duplicates_removed_total"),
        records_screened=n("records_screened"),
        records_excluded=n("records_excluded"),
        dbr_sought_reports=n("reports_sought"),
        dbr_notretrieved_reports=n("reports_not_retrieved"),
        dbr_assessed=n("reports_assessed"),
        dbr_excluded=reasons,
        new_studies=n("studies_included"),
        labels=labels,
    )
    return data, pending_boxes


def build(appendix: Path, as_of: str) -> tuple[Path, int]:
    """Render the PNG for the table counted on as_of; returns (path, pending boxes)."""
    appendix = Path(appendix)
    as_of_date(as_of)
    table = screening_dir(appendix) / TABLE_FILE
    if not table.is_file():
        raise FigureRefused(f"appendix-c/screening/{TABLE_FILE} is absent; run `aac prisma count --as-of {as_of}`")
    head, rows = read_table(table)
    if head["as_of"] != as_of:
        raise FigureRefused(f"appendix-c/screening/{TABLE_FILE} was counted as of {head['as_of']}, not {as_of}")
    data, pending = flow_data(rows, as_of)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from prisma_flow_diagram import render_prisma
    from prisma_flow_diagram import renderer

    diag = render_prisma(data, previous=False, other=False, detail_databases=True)
    target = assert_confined(screening_dir(appendix) / FIG_FILE, appendix)
    tmp = target.with_name(target.name + ".tmp")
    fig = renderer._build_figure(diag.layout)
    try:
        fig.savefig(tmp, format="png", dpi=150, bbox_inches="tight", metadata={"Software": None})
        os.replace(tmp, target)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        plt.close(fig)
    return target, pending


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="aac prisma figure")
    ap.add_argument("--as-of", required=True, help="the record date YYYY-MM-DD; must equal the table's as_of")
    ap.add_argument("--appendix-dir", default=None)
    try:
        args = ap.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code or 0)
    appendix = Path(args.appendix_dir) if args.appendix_dir else default_appendix_dir()
    try:
        path, pending = build(appendix, args.as_of)
    except (FigureRefused, OutsideScreening, ValueError, KeyError) as exc:
        print(f"aac prisma figure: {exc}", file=sys.stderr)
        return 1
    print(f"figure: appendix-c/screening/{FIG_FILE} sha256={sha256_file(path)} pending_boxes={pending} "
          f"as_of={args.as_of}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
