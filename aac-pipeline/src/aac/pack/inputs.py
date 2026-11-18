"""Shared readers and renderers for the pack generators (ARCHITECTURE §12).

Every generator reads only reports/ (shell CSVs and their long sidecars), sealed run records
listed in runs/INDEX.json, committed configs and the fixture/raw hash manifests. Nothing here
re-runs inference or recomputes a metric: a value shown in a pack file is either copied from a
shell cell (which fmt.emit derived from exactly one long row) or read from a long row. Templates
are versioned inputs under configs/pack/ rendered with Jinja2 StrictUndefined, so a template
asking for a value the generator did not supply fails instead of printing an empty string."""
from __future__ import annotations

import csv
import io
import json

csv.field_size_limit(2**31 - 1)  # : real long rows exceed the 131,072-character default (aac.reports.fmt)
from dataclasses import dataclass, field
from pathlib import Path

from aac.run.hashing import sha256_file
from aac.run.index import load_index
from aac.run.seal import verify_sealed
from aac.run.paths import reports_dir, runs_dir

TEMPLATE_DIR = ("configs", "pack")


class PackInputError(RuntimeError):
    """A pack generator could not find the input it is allowed to read (never a silent blank)."""


@dataclass
class PackContext:
    """What every generator receives from aac.pack.manifest.build_pack."""
    root: Path
    pack: Path
    runs: list[dict]
    config_hashes: list[str]
    data_versions: list[str]
    counts: object = None
    written: list[dict] = field(default_factory=list)
    # run_id -> SEALED digest that build_pack recomputed (check_input_run) and found equal to runs/INDEX.json in this
    # process : a generator that needs the same proof reads it here instead of hashing the run again
    # (the 182 real runs hash in about 66 s); empty when the caller verified nothing, and then generators recompute.
    verified_seals: dict[str, str] = field(default_factory=dict)

    def record(self, rel: str, generated_by: str, inputs: str) -> None:
        self.written.append({"file": rel, "generated_by": generated_by, "inputs": inputs})


# -- reports ----------------------------------------------------------------------------------------

def _split_header(text: str) -> tuple[dict, str]:
    first, _, rest = text.partition("\n")
    if not first.startswith("# "):
        raise PackInputError("report CSV lacks the `# config_hash=... data_version=...` first line")
    meta = {}
    for tok in first[2:].split():
        k, _, v = tok.partition("=")
        meta[k] = v
    # further comment lines (Table D.2 host_change lines) are part of the header block
    body_lines = [ln for ln in rest.split("\n") if not ln.startswith("#")]
    return meta, "\n".join(body_lines)


def read_shell(path: Path) -> tuple[dict, list[str], list[list[str]]]:
    """(header meta, column labels, rows) of a reports/table-*.csv shell."""
    path = Path(path)
    if not path.is_file():
        raise PackInputError(f"missing report {path.name}; run `make reports` first")
    meta, body = _split_header(path.read_text(encoding="utf-8"))
    rows = list(csv.reader(io.StringIO(body)))
    rows = [r for r in rows if r]
    return meta, rows[0], rows[1:]


def read_long(path: Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise PackInputError(f"missing long CSV {path.name}; run `make reports` first")
    _, body = _split_header(path.read_text(encoding="utf-8"))
    return list(csv.DictReader(io.StringIO(body)))


def long_cell(rows: list[dict], row_label: str, col_label: str, metric: str | None = None) -> dict:
    hits = [r for r in rows if r["row_label"] == row_label and r["col_label"] == col_label
            and (metric is None or r["metric"] == metric)]
    if len(hits) != 1:
        raise PackInputError(f"expected one long row for ({row_label!r}, {col_label!r}, {metric!r}), found {len(hits)}")
    return hits[0]


def shell_value(header: list[str], rows: list[list[str]], row_label: str, col_label: str, label_columns: int = 1) -> str:
    col = header.index(col_label)
    hits = [r for r in rows if " | ".join(r[:label_columns]) == row_label]
    if len(hits) != 1:
        raise PackInputError(f"expected one shell row {row_label!r}, found {len(hits)}")
    return hits[0][col]


def as_float(value: str) -> float | None:
    """A long-row value as a number, or None for n/a / not run / composite text."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# -- markdown ---------------------------------------------------------------------------------------

def md_escape(cell: str) -> str:
    return str(cell).replace("|", "\\|").replace("\n", " ")


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(md_escape(h) for h in header) + " |",
           "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(md_escape(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def shell_md(root: Path, file: str) -> tuple[dict, str]:
    meta, header, rows = read_shell(reports_dir(root) / file)
    return meta, md_table(header, rows)


# -- sealed runs ------------------------------------------------------------------------------------

def index_entries(root: Path) -> list[dict]:
    return sorted(load_index(runs_dir(root) / "INDEX.json")["runs"], key=lambda r: r["run_id"])


def sealed_run(root: Path, *, regime: str, name: str, seed: int) -> tuple[dict, Path, dict]:
    """(INDEX entry, run dir, config.resolved.json) of one sealed run; the seal is verified."""
    hits = [e for e in index_entries(root) if e["regime"] == regime and e.get("name") == name and int(e["seed"]) == seed]
    if len(hits) != 1:
        raise PackInputError(f"runs/INDEX.json lists {len(hits)} sealed {regime}/{name} seed {seed} runs (need exactly 1)")
    entry = hits[0]
    run_dir = runs_dir(root) / entry["run_id"]
    if verify_sealed(run_dir) != entry["sealed_sha256"]:
        raise PackInputError(f"{entry['run_id']}: SEALED disagrees with runs/INDEX.json")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    return entry, run_dir, resolved


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_sha(root: Path, rel: str) -> str:
    return sha256_file(Path(root) / rel)


# -- rendering --------------------------------------------------------------------------------------

def template_path(root: Path, name: str) -> Path:
    p = Path(root).joinpath(*TEMPLATE_DIR, name)
    if not p.is_file():
        raise PackInputError(f"missing pack template {p.relative_to(root)}")
    return p


def render(root: Path, name: str, *, html: bool = False, **ctx) -> str:
    import jinja2

    env = jinja2.Environment(undefined=jinja2.StrictUndefined, keep_trailing_newline=True,
                             autoescape=html, trim_blocks=True, lstrip_blocks=True)
    tpl = env.from_string(template_path(root, name).read_text(encoding="utf-8"))
    return tpl.render(**ctx)


def write_text(path: Path, text: str) -> Path:
    """UTF-8, LF only, exactly one trailing newline (byte-stable across hosts)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = text.replace("\r\n", "\n").rstrip("\n") + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


def require_headings(text: str, headings: list[str], file: str) -> None:
    """Every contracts/pack-files.yaml required heading appears as a Markdown heading line."""
    lines = {ln.lstrip("#").strip() for ln in text.splitlines() if ln.startswith("#")}
    missing = [h for h in headings if not any(line == h or line.startswith(h + " ") or line.endswith(" " + h) for line in lines)]
    if missing:
        raise PackInputError(f"{file}: required headings missing {missing}")
