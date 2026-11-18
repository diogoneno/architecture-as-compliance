"""Shell-bound CSV emitter and the frozen formatting rules (/ mechanics).

emit writes reports/<file> (header = shell LABELS, rows in shell order, cells
pre-formatted, a leading `# config_hash=... data_version=...` comment line, optional extra
`# key: value` comment lines) and reports/long/<stem>.long.csv (21 provenance columns, rows
sorted by (row_label in shell order, col_label, metric)). It refuses any long row with an
empty config_hash unless its value is a NOT_RUN cut marker, refuses a table with zero long
rows or a shell row that no long row supports, and asserts that every shell cell is
derivable from exactly one long row.

Formatting rules (ARCHITECTURE section 11; pre-registered specification
1 d.p. in percent; CI as `0.612 [0.581, 0.641]`; p-values 3 significant figures (`p=1.00`, `p=0.0412`,
`p=5.65e-05`), a p at a known floor with the floor stated (`p = 0.0625 (floor at n = 5)`); `p=0.0123 (Holm);
d=0.800`; literal `n/a`; `not run (cut: <step> on <date>)` read from configs/cuts.yaml; never `____`, NaN or an
empty cell. Rounding is decimal half-up on the shortest repr (23.45 -> 23.5), not binary (: the p-value
rule equals format(p, '#.3g') except at an exact decimal tie).

Interim cells (finding 2): `pending (Phase N)` (PENDING) marks a cell a later
phase fills. emit refuses it unless the caller passes allow_pending=True (CLI: `aac reports build
--table <id> --allow-pending`, only for generators whose build takes allow_pending); the header then
carries `pending=<phase, spaces as hyphens>`, and a pending long row may have an empty config_hash
(like a NOT_RUN cell). The value never travels further: `aac pack build` refuses any copied CSV with a
pending cell and assert_no_pending is the check the Phase 7 fill generator must call."""
from __future__ import annotations

import csv
import os
import json
import math
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import yaml

from aac.run import paths as aac_paths
from aac.run.paths import root

# : a real long row's source_shards_sha256 lists every decision-log shard hash of its run union (Table 4.4's
# victim-A pooled rows read the 220,738-record RD run: 224,574 characters), beyond csv's 131,072-character default
# field limit; every reader of a written report (read_table, the contract tests, the pack) imports this module.
CSV_FIELD_LIMIT = 2**31 - 1
csv.field_size_limit(CSV_FIELD_LIMIT)

LONG_COLUMNS = (
    "table", "row_label", "col_label", "metric", "value", "ci_lo", "ci_hi", "n_items", "n_seeds", "seeds",
    "p_raw", "p_holm", "effect_size", "effect_size_kind", "test", "family", "config_hash", "data_version",
    "model_version", "run_ids", "source_shards_sha256",
)
NOT_RUN_PREFIX = "not run (cut: "
PENDING_PREFIX = "pending ("
NA = "n/a"
FORBIDDEN = ("", "NaN", "nan")


class MissingConfigHash(Exception):
    pass


class ShellMismatch(Exception):
    pass


class ShellContractError(Exception):
    pass


class PendingNotAllowed(Exception):
    """emit met a `pending (...)` cell without allow_pending=True."""


class PendingCellFound(Exception):
    """assert_no_pending met a `pending (...)` cell: (file, row, column) name it."""

    def __init__(self, file, row: str, column: str) -> None:
        self.file, self.row, self.column = Path(file), str(row), str(column)
        super().__init__(f"pending cell in {self.file} ({self.row}, {self.column})")


def PENDING(phase: str) -> str:
    """`pending (<phase>)`: the explicit interim value of a cell a later phase fills, e.g. PENDING("Phase 3").
 An empty phase raises ValueError (an interim value always names who fills it)."""
    phase = str(phase or "").strip()
    if not phase:
        raise ValueError("PENDING needs the phase that fills the cell, e.g. PENDING('Phase 3')")
    return f"{PENDING_PREFIX}{phase})"


def is_pending(cell) -> bool:
    return str(cell).startswith(PENDING_PREFIX)


def _pending_header_value(cell) -> str:
    """`pending ` -> `Phase-3` (spaces as hyphens so header_fields still splits on spaces)."""
    inner = str(cell)[len(PENDING_PREFIX):]
    inner = inner[:-1] if inner.endswith(")") else inner
    return "-".join(inner.split())


def assert_no_pending(paths) -> None:
    """Raise PendingCellFound(file, row, column) on the first `pending (...)` cell of the given written
 shell or long CSVs (row = the long row_label, else the first column's value); None when there is none.
 `aac pack build` calls it on every reports CSV it copies; the Phase 7 fill generator must call it."""
    for p in paths:
        _, rows = read_table(Path(p))
        for row in rows:
            for col, cell in row.items():
                if cell is not None and is_pending(cell):
                    label = row.get("row_label") or next(iter(row.values()), "")
                    raise PendingCellFound(p, label, col)
    return None


# --------------------------------------------------------------------------- cuts.yaml

def cuts_path() -> Path:
    return root() / "configs" / "cuts.yaml"


def load_cuts(path: Path | None = None) -> dict:
    with open(path or cuts_path(), encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "cuts" not in data or not data["cuts"].get("steps"):
        raise ValueError("configs/cuts.yaml: expected a top-level `cuts:` mapping with `steps`")
    return data["cuts"]


def cut_step(step_id: str, path: Path | None = None) -> dict:
    """The cuts.yaml step with this id; KeyError when no such step exists (never a guessed cut)."""
    for step in load_cuts(path)["steps"]:
        if step["id"] == step_id:
            return step
    raise KeyError(f"configs/cuts.yaml has no step {step_id!r}")


def NOT_RUN(step_id: str, date: str | None = None) -> str:
    """`not run (cut: <step> on <date>)` for a cuts.yaml step. The date is the step's `fired_on`
 when the analyst has fired it, else its trigger date; an explicit `date` overrides both."""
    step = cut_step(step_id)
    when = date or step.get("fired_on") or step["trigger_date"]
    text = str(load_cuts().get("cell_text") or "not run (cut: {step} on {date})")
    return text.format(step=step_id, date=str(when))


def _row_matches(spec_rows, ident: str) -> bool:
    if isinstance(spec_rows, str):
        return False  # prose description of long-CSV rows (e.g. "long-CSV rows with subset rd"): not a shell row
    for r in spec_rows:
        key = " | ".join(r) if isinstance(r, list) else str(r)
        if key == ident:
            return True
    return False


def active_cut_for(table: str, row_ident: str, col_key: str) -> str | None:
    """The id of the FIRED cuts.yaml step that blanks (table, row, column), else None. A step
 that has not fired covers nothing: while no trigger has fired no `not run` value may appear."""
    for step in load_cuts()["steps"]:
        if not step.get("active") or not step.get("fired_on"):
            continue
        for blank in step.get("blanks") or []:
            if str(blank.get("table")) == str(table) and col_key in (blank.get("columns") or []) \
                    and _row_matches(blank.get("rows"), row_ident):
                return step["id"]
    return None


# --------------------------------------------------------------------------- shells

def shells_path() -> Path:
    return root() / "contracts" / "table-shells.json"


def validate_shells(tables: dict) -> dict:
    """every table has >= 1 column and >= 1 row label; no duplicate column
 keys or labels within a table; row identities are unique and match label_columns.
 Strings are compared exactly (UTF-8, no normalization)."""
    for tid, t in tables.items():
        cols = t.get("columns") or []
        rows = t.get("rows") or []
        if not cols:
            raise ShellContractError(f"table {tid}: zero columns")
        if not rows:
            raise ShellContractError(f"table {tid}: zero row labels")
        keys = [c["key"] for c in cols]
        labels = [c["label"] for c in cols]
        if len(set(keys)) != len(keys):
            raise ShellContractError(f"table {tid}: duplicate column keys {sorted(k for k in keys if keys.count(k) > 1)}")
        if len(set(labels)) != len(labels):
            raise ShellContractError(f"table {tid}: duplicate column labels {sorted(l for l in labels if labels.count(l) > 1)}")
        if any(not k or not l for k, l in zip(keys, labels)):
            raise ShellContractError(f"table {tid}: empty key or label")
        n_label = int(t.get("label_columns", 1))
        if n_label < 1 or n_label >= len(cols):
            raise ShellContractError(f"table {tid}: label_columns {n_label} out of range")
        idents = [row_identity(r, n_label) for r in rows]
        if len(set(idents)) != len(idents):
            raise ShellContractError(f"table {tid}: duplicate row labels")
    return tables


def row_identity(row, n_label: int) -> tuple:
    if n_label == 1:
        if not isinstance(row, str) or not row:
            raise ShellContractError(f"row label must be a non-empty string, got {row!r}")
        return (row,)
    if not isinstance(row, list) or len(row) != n_label or any(not isinstance(x, str) or not x for x in row):
        raise ShellContractError(f"row identity must be a list of {n_label} non-empty strings, got {row!r}")
    return tuple(row)


def load_shells(path: Path | None = None) -> dict:
    with open(path or shells_path(), encoding="utf-8") as fh:
        data = json.load(fh)
    return validate_shells(data["tables"])


def ident_of(shell: dict, row) -> str:
    return " | ".join(row_identity(row, int(shell.get("label_columns", 1))))


# --------------------------------------------------------------------------- formatting

def _finite(x, name: str) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: not a number: {x!r}") from exc
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name}: non-finite value {x!r} must never reach a report")
    return v


def _round(x: float, nd: int, signed: bool = False) -> str:
    """Decimal half-up rounding of the shortest repr; a value that rounds to zero is rendered without a sign, signed
 or not (`-0.0` and `+0.0` both print `0.0`; value-audit observation 1: Table 4.1 printed `+0.000`
 for -0.00004)."""
    q = Decimal(1).scaleb(-nd)
    d = Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)
    if d == 0:
        d = abs(d)
    s = f"{d:f}"
    if signed and d != 0 and not s.startswith("-"):
        s = "+" + s
    return s


def fmt_f1(x: float) -> str:
    return _round(_finite(x, "fmt_f1"), 3)


def fmt_metric(x: float) -> str:
    """F1 / AUROC / ECE: 3 d.p."""
    return _round(_finite(x, "fmt_metric"), 3)


def fmt_rate(x: float) -> str:
    """Rates in percent, 1 d.p. (x is a fraction)."""
    return _round(100.0 * _finite(x, "fmt_rate"), 1)


def fmt_pp(x: float) -> str:
    """Signed change in percentage points."""
    return _round(100.0 * _finite(x, "fmt_pp"), 1, signed=True)


def fmt_ci(x: float, lo: float, hi: float) -> str:
    return f"{_round(_finite(x, 'fmt_ci'), 3)} [{_round(_finite(lo, 'fmt_ci'), 3)}, {_round(_finite(hi, 'fmt_ci'), 3)}]"


def fmt_rate_ci(x: float, lo: float, hi: float) -> str:
    """A rate in percent with its CI: `33.3 [10.0, 60.0]`."""
    return f"{fmt_rate(x)} [{fmt_rate(lo)}, {fmt_rate(hi)}]"


def _sig3(v: float) -> str:
    """3 significant figures, decimal half-up on the shortest repr; scientific notation below 1e-4 with a two-digit
 exponent, as format(v, '#.3g') writes it (`5.65e-05`); 0 is `0.00`."""
    d = Decimal(repr(float(v)))
    if d == 0:
        return "0.00"
    adj = d.adjusted()
    q = d.quantize(Decimal(1).scaleb(adj - 2), rounding=ROUND_HALF_UP)
    if q.adjusted() != adj:  # the rounding carried into a new leading digit (0.0009996 -> 0.00100)
        adj = q.adjusted()
        q = d.quantize(Decimal(1).scaleb(adj - 2), rounding=ROUND_HALF_UP)
    if adj < -4:
        return f"{q.scaleb(-adj).quantize(Decimal('0.01')):f}e{adj:+03d}"
    return f"{q:f}"


def _p_value(p, name: str) -> float:
    v = _finite(p, name)
    if v < 0.0 or v > 1.0:
        raise ValueError(f"{name}: p-value {v} outside [0, 1]")
    return v


def fmt_p(p: float) -> str:
    """pre-registered specification`)."""
    return f"p={_sig3(_p_value(p, 'fmt_p'))}"


def fmt_p_floor(p: float, n: int) -> str:
    """pre-registered specification
 stated, `p = 0.0625 (floor at n = 5)`; any other p is fmt_p(p)."""
    v = _p_value(p, "fmt_p_floor")
    if abs(v - 2.0 / 2 ** int(n)) <= 1e-12:
        return f"p = {_sig3(v)} (floor at n = {int(n)})"
    return fmt_p(v)


def fmt_effect(kind: str, size: float) -> str:
    """pre-registered specification`)."""
    return f"{kind}={_round(_finite(size, 'fmt_effect'), 3)}"


def fmt_p_effect(p, effect, kind: str = "d", *, degenerate: bool = False, correction: str = "Holm") -> str:
    """`p=0.0123 (Holm); d=0.800`; a degenerate paired test (zero variance) is `n/a (zero variance)`."""
    if degenerate:
        return "n/a (zero variance)"
    return f"{fmt_p(p)} ({correction}); {fmt_effect(kind, effect)}"


def fmt_signed(x: float, nd: int = 3) -> str:
    return _round(_finite(x, "fmt_signed"), nd, signed=True)


def fmt_count(x) -> str:
    v = _finite(x, "fmt_count")
    return str(int(v)) if float(v).is_integer() else _round(v, 1)


# --------------------------------------------------------------------------- emit

def _cell_text(cell, where: str) -> str:
    if isinstance(cell, float) and (math.isnan(cell) or math.isinf(cell)):
        raise ValueError(f"{where}: non-finite cell value {cell!r}")
    s = str(cell)
    if s in FORBIDDEN or "____" in s:
        raise ValueError(f"{where}: forbidden cell value {s!r}")
    return s


def _model_versions_by_config() -> dict[str, set[str]]:
    """config_hash -> model_version(s) of the sealed runs listed in runs/INDEX.json (run-record.json)."""
    runs_dir = aac_paths.runs_dir()
    index = runs_dir / "INDEX.json"
    out: dict[str, set[str]] = {}
    if not index.is_file():
        return out
    for r in json.loads(index.read_text(encoding="utf-8")).get("runs", []):
        rr = runs_dir / r["run_id"] / "run-record.json"
        if rr.is_file():
            out.setdefault(r["config_hash"], set()).add(json.loads(rr.read_text(encoding="utf-8"))["model_version"])
    return out


def with_model_version(table_id: str, long_rows: list[dict]) -> list[dict]:
    """every long row carries the model_version(s) of the sealed runs its config_hash names
 (';'-joined, sorted, like config_hash); a cut cell with no run carries `n/a`. A config_hash that no
 indexed run carries raises MissingConfigHash (never a guessed identity). Rows that already carry a
 model_version are kept as given. Returns new dicts; the caller's rows are not mutated."""
    mapping: dict[str, set[str]] | None = None
    out = []
    for lr in long_rows:
        lr = dict(lr)
        if str(lr.get("model_version") or "").strip():
            out.append(lr)
            continue
        chs = [c for c in str(lr.get("config_hash", "")).split(";") if c.strip()]
        if not chs:
            lr["model_version"] = NA
        else:
            mapping = _model_versions_by_config() if mapping is None else mapping
            mvs: set[str] = set()
            for c in chs:
                if c not in mapping:
                    raise MissingConfigHash(f"{table_id} cell ({lr.get('row_label')}, {lr.get('col_label')}): config_hash "
                                            f"{c[:12]} is no indexed sealed run's, so its model_version is unknown")
                mvs |= mapping[c]
            lr["model_version"] = ";".join(sorted(mvs))
        out.append(lr)
    return out


def _check_long(table_id: str, long_rows: list[dict]) -> None:
    for r in long_rows:
        extra = set(r) - set(LONG_COLUMNS)
        missing = set(LONG_COLUMNS) - set(r)
        if extra or missing:
            raise ValueError(f"long row for {table_id}: extra {sorted(extra)}, missing {sorted(missing)}")
        if "|" in str(r["row_label"]).replace(" | ", ""):
            raise ValueError(f"{table_id}: row_label may not contain '|' except as the ' | ' identity separator")
        value = _cell_text(r["value"], f"{table_id} long row {r['row_label']}/{r['col_label']} value")
        no_run = value.startswith(NOT_RUN_PREFIX) or value.startswith(PENDING_PREFIX)
        if not str(r["config_hash"]).strip() and not no_run:
            raise MissingConfigHash(f"{table_id} cell ({r['row_label']}, {r['col_label']}) has no config_hash")
        for col, cell in r.items():
            if col == "config_hash" and no_run:
                continue  # the legitimate empties: a cut cell or a pending cell has no run
            _cell_text(cell, f"{table_id} long row {r['row_label']}/{r['col_label']} column {col}")


def emit(table_id: str, shell_rows: list[dict], long_rows: list[dict], *, config_hash: str, data_version: str,
         out_dir: Path, shells: dict | None = None, header_extra: dict | None = None,
         comment_lines: list[str] | None = None, allow_pending: bool = False) -> tuple[Path, Path]:
    """Write the shell and its long sidecar. `header_extra` adds ` key=value` fields to the first
 comment line; `comment_lines` adds further `# ...` lines (e.g. `# host_change: ...`). Long rows
 whose row_label is not a shell row (supplementary provenance, e.g. Table D.2's knowledge-store
 row) are written after the shell rows. A `pending (...)` cell raises PendingNotAllowed unless
 allow_pending=True, which adds ` pending=<phase>` (from the first pending cell) to the header."""
    shells = shells or load_shells()
    if table_id not in shells:
        raise KeyError(f"table {table_id!r} is not in contracts/table-shells.json")
    shell = shells[table_id]
    keys = [c["key"] for c in shell["columns"]]
    labels = [c["label"] for c in shell["columns"]]
    n_label = int(shell.get("label_columns", 1))
    row_labels = [row_identity(r, n_label) for r in shell["rows"]]
    if not long_rows:
        raise ValueError(f"{table_id}: zero long rows; an empty shell is never written")
    if not shell_rows:
        raise ValueError(f"{table_id}: zero shell rows; an empty shell is never written")
    got = [tuple(r[k] for k in keys[:n_label]) for r in shell_rows]
    if got != row_labels:
        raise ShellMismatch(f"{table_id}: row labels {got} != shell {row_labels}")
    # interim cells are checked before anything is written 
    first_pending, where = None, ""
    for r in shell_rows:
        for k, lab in zip(keys, labels):
            if k in r and is_pending(r[k]) and first_pending is None:
                first_pending, where = r[k], f"row {' | '.join(str(r[x]) for x in keys[:n_label])!r} column {lab!r} (key {k!r})"
    for lr in long_rows:
        for col, cell in lr.items():
            if is_pending(cell) and first_pending is None:
                first_pending = cell
                where = f"long row {lr.get('row_label')!r} column {lr.get('col_label')!r} ({col})"
    if first_pending is not None and not allow_pending:
        raise PendingNotAllowed(f"{table_id} {where}: {first_pending!r} is an interim value; only "
                                f"`aac reports build --table {table_id} --allow-pending` may write it")
    for r in shell_rows:
        if list(r.keys()) != keys:
            raise ShellMismatch(f"{table_id}: columns {list(r.keys())} != shell keys {keys}")
        for k, v in r.items():
            _cell_text(v, f"{table_id} row {r[keys[0]]!r} column {k!r}")
    long_rows = with_model_version(table_id, long_rows)
    _check_long(table_id, long_rows)
    if not str(config_hash).strip() or not str(data_version).strip():
        raise MissingConfigHash(f"{table_id}: header config_hash / data_version must not be empty")
    # every non-label shell cell must be derivable from exactly one long row
    seen: dict[tuple[str, str], int] = {}
    for lr in long_rows:
        if lr["table"] != table_id:
            raise ValueError(f"long row table {lr['table']!r} != {table_id!r}")
        key = (lr["row_label"], lr["col_label"])
        seen[key] = seen.get(key, 0) + 1
    for r in shell_rows:
        ident = " | ".join(r[k] for k in keys[:n_label])
        for lab in labels[n_label:]:
            n = seen.get((ident, lab), 0)
            if n != 1:
                raise ValueError(f"{table_id}: cell ({ident!r}, {lab!r}) derives from {n} long rows, expected 1")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "long").mkdir(exist_ok=True)
    header = f"# config_hash={config_hash} data_version={data_version}"
    for k, v in sorted((header_extra or {}).items()):
        header += f" {k}={v}"
    if first_pending is not None:
        header += f" pending={_pending_header_value(first_pending)}"
    header += "\n"
    extra_lines = "".join(f"# {ln}\n" for ln in (comment_lines or []))
    shell_path = out_dir / shell["file"]
    
    shell_tmp = shell_path.with_name(shell_path.name + ".tmp")
    with open(shell_tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        fh.write(extra_lines)
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(labels)
        for r in shell_rows:
            w.writerow([str(r[k]) for k in keys])
    long_path = out_dir / "long" / (Path(shell["file"]).stem + ".long.csv")
    idents = [" | ".join(t) for t in row_labels]

    def order(r: dict) -> tuple:
        pos = idents.index(r["row_label"]) if r["row_label"] in idents else len(idents)
        return (pos, r["row_label"] if pos == len(idents) else "", r["col_label"], r["metric"])

    long_tmp = long_path.with_name(long_path.name + ".tmp")
    with open(long_tmp, "w", encoding="utf-8", newline="") as fh:
        fh.write(header)
        fh.write(extra_lines)
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(LONG_COLUMNS)
        for lr in sorted(long_rows, key=order):
            w.writerow([str(lr[c]) for c in LONG_COLUMNS])
    os.replace(long_tmp, long_path)
    os.replace(shell_tmp, shell_path)
    return shell_path, long_path


def long_row(table: str, row_label: str, col_label: str, metric: str, value, *, config_hash: str, data_version: str,
             run_ids: str, source_shards_sha256: str, ci_lo="n/a", ci_hi="n/a", n_items="n/a", n_seeds="n/a",
             seeds="n/a", p_raw="n/a", p_holm="n/a", effect_size="n/a", effect_size_kind="n/a", test="n/a",
             family="n/a", model_version="") -> dict:
    """One long-sidecar row. model_version is normally left empty and filled by emit from the
 sealed runs the row's config_hash names ."""
    return {
        "table": table, "row_label": row_label, "col_label": col_label, "metric": metric, "value": value,
        "ci_lo": ci_lo, "ci_hi": ci_hi, "n_items": n_items, "n_seeds": n_seeds, "seeds": seeds, "p_raw": p_raw,
        "p_holm": p_holm, "effect_size": effect_size, "effect_size_kind": effect_size_kind, "test": test,
        "family": family, "config_hash": config_hash, "data_version": data_version, "model_version": model_version,
        "run_ids": run_ids, "source_shards_sha256": source_shards_sha256,
    }


def read_table(path: Path) -> tuple[list[str], list[dict]]:
    """Read a written shell or long CSV: (comment lines, rows as dicts keyed by header)."""
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.read().splitlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    body = [ln for ln in lines if not ln.startswith("#")]
    return comments, list(csv.DictReader(body))


def header_fields(comment: str) -> dict[str, str]:
    """Parse `# config_hash=... data_version=... key=value` into a dict."""
    out = {}
    for tok in comment.lstrip("#").split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out
