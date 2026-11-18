"""Table 4.8 (hypotheses): Outcome = machine-generated clause list read from the long CSVs per
configs/hypothesis-rules.yaml (which sub-claims held, with Holm p and effect size); Verdict =
`suggested: <supported | partially supported | not supported>` unless configs/verdicts.yaml carries the
analyst's confirmation; the RQ4 row reads table-4.7 when present, else the pending text. the analyst's
confirmations are verdicts on the real data, so they apply only in the real namespace (AAC_OUT unset); a namespace
build (smoke, fixtures) keeps the rule suggestions. A `sensitivity` claim is report-only: it prints a long-CSV cell after
the verdict clauses as `sensitivity (not in the verdict): <label>: ...`, never enters the suggested verdict or a Holm
family, and prints `<label>: n/a (row absent)` when the namespace has no such row. A claim's optional `basis` text is
printed beside its id ."""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from aac.reports import fmt
from aac.run.paths import out_root, reports_dir

TABLE = "4.8"
VERDICTS = ("supported", "partially supported", "not supported")
T47_FILE = "table-4.7-verifiability.csv"


def _load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _long(root: Path, out_dir: Path, table_id: str) -> tuple[dict, dict[tuple[str, str], dict]]:
    shells = fmt.load_shells(root / "contracts" / "table-shells.json")
    path = out_dir / "long" / (Path(shells[table_id]["file"]).stem + ".long.csv")
    if not path.is_file():
        raise RuntimeError(f"Table 4.8 needs {path.relative_to(root) if path.is_relative_to(root) else path}: build table {table_id} first")
    comments, rows = fmt.read_table(path)
    cells = {(r["row_label"], r["col_label"]): r for r in rows}
    for key, text in _shell_cells(out_dir, shells[table_id]).items():
        if key in cells:
            cells[key] = {**cells[key], SHELL_KEY: text}
    return fmt.header_fields(comments[0]), cells


SHELL_KEY = "_shell_cell"  # the printed shell-CSV cell, attached to its long row (never written anywhere)
DIFF_RE = re.compile(r"diff=([+-]?[0-9]+(?:\.[0-9]+)?)")


def _shell_cells(out_dir: Path, shell: dict) -> dict[tuple[str, str], str]:
    """{(row_label, col_label): printed cell} from the written shell CSV next to the long CSV; empty when it is absent.
 row_label joins the label columns with ` | `, as the long CSVs do."""
    path = out_dir / Path(shell["file"]).name
    if not path.is_file():
        return {}
    _, rows = fmt.read_table(path)
    n_label = int(shell.get("label_columns", 1))
    out = {}
    for r in rows:
        cols = list(r.keys())
        ident = " | ".join(r[c] for c in cols[:n_label])
        for c in cols[n_label:]:
            out[(ident, c)] = r[c]
    return out


def effect_text(row: dict) -> str:
    """The effect size a holm_test clause prints. When the claim's shell cell prints `diff=<signed>` (Tables 4.2 and
 4.5), the clause prints that same text, so Table 4.8 never re-rounds the long CSV's four-decimal effect_size (which
 double-rounded +0.03147 -> 0.0315 -> +0.032 against the shell's +0.031); otherwise fmt.fmt_signed(effect_size, 3)."""
    m = DIFF_RE.search(row.get(SHELL_KEY) or "")
    if m:
        return m.group(1)
    return fmt.fmt_signed(float(row["effect_size"]), 3)


def _cell(cells: dict, claim: dict) -> dict:
    key = (claim["row_label"], claim["col_label"])
    if key not in cells:
        raise RuntimeError(f"hypothesis rule {claim['id']}: no long row for table {claim['table']} cell {key}")
    return cells[key]


def _num(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


P_FLOOR_TEXT = "p<1e-06"  # the long CSVs store p at six decimals, so a stored 0.000000 means p < 5e-07


def fmt_p_long(value: str) -> str:
    """A p-value read from a long CSV: one that parses to 0.0 is below the six-decimal floor and renders `p<1e-06`
 (never `p=0.00`); any other value renders through fmt.fmt_p (pre-registered specification)."""
    p = float(value)
    if p == 0.0:
        return P_FLOOR_TEXT
    return fmt.fmt_p(p)


SENSITIVITY = "sensitivity"
SENSITIVITY_PREFIX = "sensitivity (not in the verdict): "


def _present(s: str | None) -> bool:
    return s not in (None, "", "n/a")


def _n_text(n_items: str) -> str:
    return n_items.replace(";", "/")  # a two-sample row stores `n_a;n_b`


def _sensitivity(claim: dict, cells: dict) -> str:
    """One report-only clause: the cell's value (with its CI when the value does not already carry one), p_raw, n_items
 and n_seeds as stored in the long CSV; `n/a (row absent)` when the namespace lacks the row. Never a status."""
    label = claim["label"]
    row = cells.get((claim["row_label"], claim["col_label"]))
    if row is None:
        return f"{label}: n/a (row absent)"
    if claim.get("metric") and row["metric"] != claim["metric"]:
        raise RuntimeError(f"hypothesis rule {claim['id']}: long row metric {row['metric']!r} is not {claim['metric']!r}")
    value = row["value"]
    if _present(row["ci_lo"]) and _present(row["ci_hi"]) and "[" not in value:
        value += f" [{row['ci_lo']}, {row['ci_hi']}]"
    paren = []
    if _present(row["p_raw"]):
        paren.append(fmt_p_long(row["p_raw"]))
    if _present(row["n_items"]):
        paren.append(f"n={_n_text(row['n_items'])}")
    if _present(row["n_seeds"]):
        paren.append(f"seeds={row['n_seeds']}")
    return f"{label}: {value}" + (f" ({'; '.join(paren)})" if paren else "")


def _evaluate(claim: dict, cells: dict, alpha: float, other_cells: dict | None = None) -> tuple[str, str]:
    """(status, clause) with status in held | not held | not testable."""
    kind = claim["kind"]
    cid = claim["id"] + (f" ({claim['basis']})" if claim.get("basis") else "")
    row = _cell(cells, claim)
    if kind == "holm_test":
        p_holm, effect = _num(row["p_holm"]), _num(row["effect_size"])
        if p_holm is None or effect is None:
            kind_txt = row["effect_size_kind"] or "n/a"
            return "not testable", f"{cid}: not testable ({kind_txt}; p n/a)"
        sign_ok = effect < 0 if claim["direction"] == "negative" else effect > 0
        held = sign_ok and p_holm < alpha
        kind_txt = row["effect_size_kind"]
        n_txt = f"; n={_n_text(row['n_items'])}" if _present(row.get("n_items")) else ""
        return ("held" if held else "not held"), f"{cid}: {'held' if held else 'not held'} ({fmt_p_long(row['p_holm'])} Holm; {kind_txt} {effect_text(row)}{n_txt})"
    if kind == "ci_excludes_zero":
        lo = _num(row["ci_lo"])
        if lo is None:
            return "not testable", f"{cid}: not testable (no CI)"
        held = lo > 0.0
        return ("held" if held else "not held"), f"{cid}: {'held' if held else 'not held'} (rate {row['value']} percent, 95% CI [{row['ci_lo']}, {row['ci_hi']}])"
    if kind == "compare_rows":
        other = cells.get((claim["other_row_label"], claim["col_label"]))
        if other is None:
            raise RuntimeError(f"hypothesis rule {cid}: no long row for {claim['other_row_label']}")
        a, b = _num(row["value"]), _num(other["value"])
        if a is None or b is None:
            return "not testable", f"{cid}: not testable"
        held = a > b if claim["direction"] == "greater" else a < b
        return ("held" if held else "not held"), f"{cid}: {'held' if held else 'not held'} ({row['value']} at {claim['row_label']} vs {other['value']} at {claim['other_row_label']}; descriptive)"
    raise ValueError(f"unknown claim kind {kind!r}")


def _suggest(statuses: list[str]) -> str:
    held = sum(s == "held" for s in statuses)
    if held == len(statuses):
        return VERDICTS[0]
    if held > 0:
        return VERDICTS[1]
    return VERDICTS[2]


def _rq4(root: Path, out_dir: Path, claim: dict) -> tuple[str, str, bool]:
    path = out_dir / T47_FILE
    if not path.is_file():
        return f"{T47_FILE} absent: make verify-pack has not written it", claim["pending_text"], False
    _, rows = fmt.read_table(path)
    parts = [f"{r['Artifact']}: {r['Result']}" for r in rows]
    results = {r["Result"] for r in rows}
    verdict = "pass per artifact" if results == {"pass"} else "pass / fail per artifact: " + "; ".join(parts)
    return "; ".join(parts), verdict, True


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    rules = _load_yaml(root / "configs" / "hypothesis-rules.yaml")
    verdicts = _load_yaml(root / "configs" / "verdicts.yaml")["verdicts"]
    real_ns = out_root(root).resolve() == root.resolve()  # analyst verdicts are real-data verdicts
    alpha = float(rules["alpha"])
    shell = fmt.load_shells(root / "contracts" / "table-shells.json")[TABLE]
    shell_rows, long_rows = [], []
    hashes: set[str] = set()
    data_versions: set[str] = set()
    cache: dict[str, tuple[dict, dict]] = {}
    for hyp in rules["hypotheses"]:
        hid, row_label = hyp["id"], hyp["row"]
        if row_label not in shell["rows"]:
            raise RuntimeError(f"hypothesis-rules.yaml row {row_label!r} is not a Table 4.8 shell row")
        clauses, statuses, sources_txt, sens = [], [], [], []
        row_hashes: set[str] = set()
        rq4_verdict, rq4_pending = None, False
        for claim in hyp["claims"]:
            if claim["kind"] == "table_4_7":
                outcome, rq4_verdict, present = _rq4(root, out_dir, claim)
                rq4_pending = not present
                clauses.append(outcome)
                sources_txt.append(T47_FILE if present else f"{T47_FILE} (absent)")
                continue
            tid = claim["table"]
            if tid not in cache:
                cache[tid] = _long(root, out_dir, tid)
            header, cells = cache[tid]
            if claim["kind"] == SENSITIVITY:  # report-only: no status, so never in _suggest or a Holm family
                sens.append(_sensitivity(claim, cells))
                row_hashes.update(h for h in header["config_hash"].split(";") if h)
                data_versions.add(header["data_version"])
                sources_txt.append(f"sensitivity: table {tid} ({claim['row_label']} / {claim['col_label']})")
                continue
            status, clause = _evaluate(claim, cells, alpha)
            statuses.append(status)
            clauses.append(clause)
            row_hashes.update(h for h in header["config_hash"].split(";") if h)
            data_versions.add(header["data_version"])
            sources_txt.append(f"table {tid} ({claim['row_label']} / {claim['col_label']})")
        clauses.extend(SENSITIVITY_PREFIX + c for c in sens)  # after every verdict clause
        outcome = "; ".join(clauses)
        confirmed = (verdicts.get(hid) or {}).get("confirmed")
        file_confirmed = confirmed
        if not real_ns:
            confirmed = None
        if hid == "RQ4":
            suggested = rq4_verdict
        else:
            suggested = _suggest(statuses)
            if file_confirmed is not None and file_confirmed not in VERDICTS:
                raise ValueError(f"configs/verdicts.yaml {hid}: confirmed must be one of {VERDICTS}, got {file_confirmed!r}")
        if confirmed is not None:
            verdict_cell = str(confirmed)
            verdict_note = f"analyst-confirmed (configs/verdicts.yaml); rule suggested: {suggested}"
        elif hid == "RQ4" and rq4_pending:
            verdict_cell = rq4_verdict
            verdict_note = "pending until make verify-pack writes table-4.7"
        else:  # namespace builds land here for every row: same text as the golden fixture table
            verdict_cell = f"suggested: {suggested}"
            verdict_note = "rule-suggested (configs/hypothesis-rules.yaml); analyst confirms in configs/verdicts.yaml"
        hashes.update(row_hashes)
        shell_rows.append({"hypothesis": row_label, "outcome_text": outcome, "verdict": verdict_cell})
        row_ch = ";".join(sorted(row_hashes)) if row_hashes else None
        held = sum(s == "held" for s in statuses)
        long_rows.append(fmt.long_row(TABLE, row_label, "Outcome", "outcome_clauses", outcome, n_items=sum(c["kind"] != SENSITIVITY for c in hyp["claims"]),
                                      test=f"{held} of {len(statuses)} sub-claims held; sources: " + "; ".join(sources_txt),
                                      family=f"Hypothesis items {hyp.get('items', hyp.get('_items', []))}", config_hash=row_ch or "PLACEHOLDER",
                                      data_version=";".join(sorted(data_versions)) if data_versions else "PLACEHOLDER",
                                      run_ids="via the long CSVs named in test", source_shards_sha256="via the long CSVs named in test"))
        long_rows.append(fmt.long_row(TABLE, row_label, "Verdict", "verdict", verdict_cell, test=verdict_note,
                                      family=f"Hypothesis items {hyp.get('items', hyp.get('_items', []))}", config_hash=row_ch or "PLACEHOLDER",
                                      data_version=";".join(sorted(data_versions)) if data_versions else "PLACEHOLDER",
                                      run_ids="via the long CSVs named in test", source_shards_sha256="via the long CSVs named in test"))
    all_ch = ";".join(sorted(hashes))
    all_dv = ";".join(sorted(data_versions))
    for lr in long_rows:  # the RQ4 row (no long-CSV source yet) inherits the provenance of the tables the other rows read
        if lr["config_hash"] == "PLACEHOLDER":
            lr["config_hash"], lr["data_version"] = all_ch, all_dv
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=all_ch, data_version=all_dv, out_dir=out_dir))
