"""Article 15 attack-success-rate report: evidence-pack/asr-report.md (Table D.3 row 6; brief §1
item 6).

Reads the Table 4.4 shell and long CSV only (ASR, BODEGA score or its `not computed on stubs`
note, median queries per success and deflection per attack x victim, pooled and per subset), the
recipe pins and the `not_assessed` attack-type list from configs/attacks/bodega-recipes.yaml. No
adversarial text: the attack records carry claim hashes, and this report carries only rates.

the template is chosen by the namespace kind (configs/pack/datasheet-profiles.yaml, keyed by the
data manifest the sealed runs name). `fixture` renders asr-report.template.md exactly as before (stub goldens
unchanged); `real` renders asr-report.v2.template.md, which adds, per Table 4.4 shell row and scope (pooled, then each
subset in long-CSV order, never merged), the success rates and the denominators (ledger rows, attacked n, strict
successes, ASR, ASR excluding budget-exhausted items, budget-exhausted n and share, skipped and timed-out counts),
every value the shell or long-row string verbatim; the shell's analyst note lines byte for byte; the threat model and
the Art. 15(4) no-feedback-loop statement from configs/pack/art15-scope.yaml; and the not-assessed table: the
bodega-recipes.yaml entries (that file is not edited), a stale entry overridden with the run that partly assessed it
(the run name must be indexed and the long rows must exist), then the Art. 15(5) additions. Every entry needs a source."""
from __future__ import annotations

from pathlib import Path

import yaml

from aac.pack.datasheet import load_profile
from aac.pack.inputs import (PackContext, PackInputError, index_entries, md_table, read_json, read_long, read_shell,
                             render, require_headings, shell_md, write_text)
from aac.run.hashing import sha256_file
from aac.run.paths import reports_dir, runs_dir

GENERATED_BY = "aac.pack.asr_report"
FILE = "asr-report.md"
REQUIRED = ["Attack battery", "Success rates", "Deflection"]
RECIPES = "configs/attacks/bodega-recipes.yaml"
METRIC_COLS = ("Attack-success rate (percent)", "BODEGA score", "Queries per success (median)", "Deflection rate (percent)")
TEMPLATES = {"fixture": "asr-report.template.md", "real": "asr-report.v2.template.md"}
REAL_HEADINGS = REQUIRED + ["Threat model", "Denominators", "Not assessed", "No feedback loop"]
SCOPE = "configs/pack/art15-scope.yaml"
SHELL, LONG = "table-4.4-r3.csv", "table-4.4-r3.long.csv"
POOLED, POOLED_SHELL = "pooled", "pooled (shell)"
MISSING = "n/a (no long row)"
# Long-CSV column label (pooled form) of each metric the real report copies; a subset row is `<base> (<subset>)`.
LONG_BASE = {
    "ledger_rows_n": "Ledger rows (planned items with a ledger row)",
    "attacked_n": "Attacked items (ASR denominator)",
    "strict_success_n": "Strict successes",
    "asr_pct": "Attack-success rate (percent)",
    "asr_excl_budget_pct": ("Attack-success rate, alternative reading: strict successes / (attacked - budget-exhausted) "
                            "(percent; not the shell value)"),
    "budget_exhausted_n": "Budget-exhausted items (own category; in the ASR denominator as non-successes)",
    "budget_exhausted_share_pct": "Budget-exhausted share (percent)",
    "skipped_n": "Skipped items (clean decision already wrong, 0 queries; outside the ASR denominator)",
    "timed_out_n": "Timed-out items (wall-clock deadline, not finished; outside the ASR denominator, R47)",
    "bodega_score_mean": "BODEGA score",
    "queries_per_success_median": "Queries per success (median)",
    "deflection_rate_pct": "Deflection rate (percent)",
}
DENOM_METRICS = ("ledger_rows_n", "attacked_n", "strict_success_n", "asr_pct", "asr_excl_budget_pct",
                 "budget_exhausted_n", "budget_exhausted_share_pct", "skipped_n", "timed_out_n")
DENOM_HEADER = ["Attack", "Victim", "Scope", "Ledger rows", "Attacked n", "Strict successes", "ASR (percent)",
                "ASR excl. budget-exhausted", "Budget-exhausted n", "Budget-exhausted share", "Skipped", "Timed out"]
SUCCESS_METRICS = ("asr_pct", "bodega_score_mean", "queries_per_success_median", "deflection_rate_pct")
SUCCESS_HEADER = ["Attack", "Victim", "Scope", "ASR (percent)", "BODEGA score", "Median queries per success",
                  "Deflection (percent)"]


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    # the Table 4.4 shell and long CSV first: a missing report is a PackInputError naming the file
    read_shell(reports_dir(root) / SHELL)
    read_long(reports_dir(root) / "long" / LONG)
    with open(root / RECIPES, encoding="utf-8") as fh:
        recipes = yaml.safe_load(fh)
    not_assessed = recipes.get("not_assessed") or []
    if not not_assessed:
        raise PackInputError(f"{RECIPES} has no not_assessed list; the ASR report must state what the battery is silent on")
    meta44, t44 = shell_md(root, "table-4.4-r3.csv")
    long44 = read_long(reports_dir(root) / "long" / "table-4.4-r3.long.csv")
    subset_rows, deflect_rows = [], []
    for r in long44:
        col = r["col_label"]
        if col in METRIC_COLS:
            continue  # pooled cell: shown in the shell table above
        base, _, subset = col.rpartition(" (")
        if base not in METRIC_COLS or not subset.endswith(")"):
            continue
        subset = subset[:-1]
        attack, victim = [s.strip() for s in r["row_label"].split(" | ")]
        value = r["value"]
        if base == "Deflection rate (percent)":
            if victim == "Full pipeline":
                deflect_rows.append([attack, subset, value, r["test"]])
            continue
        subset_rows.append([attack, victim, subset, base, value])
    subset_rows.sort()
    deflect_rows.sort()
    battery = [[name, spec["bodega_name"], spec["constructor"]] for name, spec in recipes["recipes"].items()]
    beam = recipes["beamattack"]
    battery.append(["BeamAttack", beam["entry_point"], f"{beam['chosen']}; commit {beam['commit'][:8]}"])
    kind = _kind(root)
    extra, inputs = {}, f"reports/table-4.4-r3.csv; reports/long/table-4.4-r3.long.csv; {RECIPES}"
    if kind == "real":
        extra = _real_fields(root, not_assessed, long44)
        inputs += f"; {SCOPE}"
    text = render(root, TEMPLATES[kind], library=recipes["library"], library_version=recipes["library_version"],
                  bodega_commit=recipes["bodega_commit"], query_budget=recipes["evaluation"]["query_budget"],
                  battery=md_table(["Attack", "Implementation", "Constructor / setting"], battery),
                  t44=t44, table_config_hash=meta44["config_hash"],
                  subset_table=md_table(["Attack", "Victim", "Subset", "Metric", "Value"], subset_rows),
                  deflect_table=md_table(["Attack", "Subset", "Deflection rate (percent)", "Basis"], deflect_rows),
                  not_assessed=md_table(["Attack type", "Why not assessed", "Source"],
                                        [[n["attack_type"], n["why"], n["source"]] for n in not_assessed]), **extra)
    require_headings(text, REAL_HEADINGS if kind == "real" else REQUIRED, FILE)
    write_text(pack / FILE, text)
    ctx.record(FILE, GENERATED_BY, inputs)
    return [FILE]


def _kind(root: Path) -> str:
    """Namespace kind from the data manifest the sealed runs of the namespace name (config.resolved.json only; no
 shard read). Every indexed run is read, not only R3: a scratch or test namespace may hold no R3 run, and the
 datasheet generator already requires one manifest per namespace."""
    manifests = {read_json(runs_dir(root) / e["run_id"] / "config.resolved.json")["data"]["manifest"]
                 for e in index_entries(root)}
    if len(manifests) != 1:
        raise PackInputError(f"ASR report: sealed runs name {len(manifests)} data manifests: {sorted(manifests)}")
    kind = load_profile(root, next(iter(manifests)))["kind"]
    if kind not in TEMPLATES:
        raise PackInputError(f"ASR report: no template for namespace kind {kind!r}")
    return kind


def _scope_of(row: dict) -> str | None:
    """POOLED, the subset label, or None when the row is not one of the LONG_BASE cells."""
    base = LONG_BASE.get(row["metric"])
    if base is None:
        return None
    col = row["col_label"]
    if col == base:
        return POOLED
    if col.startswith(base + " (") and col.endswith(")"):
        return col[len(base) + 2:-1]
    return None


def _cells(long_rows: list[dict], label: str) -> tuple[list[str], dict]:
    """(scopes: pooled first, then subsets in first-appearance order of the long CSV; {(metric, scope): value})."""
    scopes, values = [POOLED], {}
    for r in long_rows:
        if r["row_label"] != label:
            continue
        scope = _scope_of(r)
        if scope is None:
            continue
        if (r["metric"], scope) in values:
            raise PackInputError(f"{LONG}: two rows for ({label!r}, {r['metric']}, {scope!r})")
        values[(r["metric"], scope)] = r["value"]
        if scope not in scopes:
            scopes.append(scope)
    return scopes, values


def denominator_rows(shell_rows: list[list[str]], long_rows: list[dict]) -> list[list[str]]:
    """Per shell row (shell order) and scope: the denominator metrics, each the long-row string verbatim or MISSING."""
    out = []
    for r in shell_rows:
        label = f"{r[0]} | {r[1]}"
        scopes, values = _cells(long_rows, label)
        if ("attacked_n", POOLED) not in values:
            raise PackInputError(f"{LONG}: no pooled attacked_n row for {label!r} (every ASR needs its denominator)")
        for scope in scopes:
            out.append([r[0], r[1], scope] + [values.get((m, scope), MISSING) for m in DENOM_METRICS])
    return out


def success_rows(shell_rows: list[list[str]], long_rows: list[dict]) -> list[list[str]]:
    """Per shell row: the pooled shell cells verbatim, then one row per subset from the long CSV (side by side)."""
    out = []
    for r in shell_rows:
        scopes, values = _cells(long_rows, f"{r[0]} | {r[1]}")
        out.append([r[0], r[1], POOLED_SHELL] + list(r[2:6]))
        for scope in scopes[1:]:
            out.append([r[0], r[1], scope] + [values.get((m, scope), MISSING) for m in SUCCESS_METRICS])
    return out


def _need(entry: dict, keys: tuple[str, ...], where: str) -> None:
    missing = [k for k in keys if not entry.get(k)]
    if missing:
        raise PackInputError(f"{where}: entry {entry.get('attack_type')!r} has no {', '.join(missing)} "
                             "(every not-assessed entry needs a why and a source)")


def applied_overrides(scope: dict, run_names: set[str]) -> list[dict]:
    """The overrides whose named run is a sealed run of this namespace (runs/INDEX.json). : an override
 whose run the namespace does not hold (a thin scratch selection without the R3 run) is not applied, so its entry
 degrades to the file's `not assessed` row; a namespace never claims an assessment it does not contain."""
    return [o for o in scope.get("overrides") or [] if o.get("run_name") in run_names]


def not_assessed_rows(entries: list[dict], scope: dict, *, run_names: set[str], long_rows: list[dict]) -> list[list[str]]:
    """bodega-recipes entries in file order (overrides applied by attack_type), then the additions in file order.
 An override applies only when its run is in `run_names` (applied_overrides); otherwise the entry stays
 `not assessed`. Every override must still match an entry by attack_type."""
    overrides = {}
    known = {e.get("attack_type") for e in entries}
    for o in scope.get("overrides") or []:
        _need(o, ("attack_type", "status", "why", "source", "run_name", "long_row_prefix"), f"{SCOPE} overrides")
        if o["attack_type"] not in known:
            raise PackInputError(f"{SCOPE}: overrides match no not_assessed entry: {[o['attack_type']]}")
    for o in applied_overrides(scope, run_names):
        if not any(r["row_label"].startswith(o["long_row_prefix"]) for r in long_rows):
            raise PackInputError(f"{SCOPE}: override {o['attack_type']!r} rests on long rows {o['long_row_prefix']!r}, "
                                 f"which {LONG} does not carry")
        overrides[o["attack_type"]] = o
    out, seen = [], set()
    for e in entries:
        _need(e, ("attack_type", "why", "source"), RECIPES)
        o = overrides.pop(e["attack_type"], None)
        out.append([e["attack_type"], o["status"], o["why"], o["source"]] if o
                   else [e["attack_type"], "not assessed", e["why"], e["source"]])
        seen.add(e["attack_type"].lower())
    if overrides:
        raise PackInputError(f"{SCOPE}: overrides match no not_assessed entry: {sorted(overrides)}")
    for a in scope.get("not_assessed_add") or []:
        _need(a, ("attack_type", "why", "source"), f"{SCOPE} not_assessed_add")
        if a["attack_type"].lower() in seen:
            raise PackInputError(f"{SCOPE}: {a['attack_type']!r} is already listed")
        out.append([a["attack_type"], "not assessed", a["why"], a["source"]])
        seen.add(a["attack_type"].lower())
    return out


def _real_fields(root: Path, entries: list[dict], long44: list[dict]) -> dict:
    """Template fields of the real profile; every number is a shell or long-row string. Nothing is computed."""
    rep = reports_dir(root)
    meta, _, shell_rows = read_shell(rep / SHELL)
    notes = [ln[len("# note: "):] for ln in (rep / SHELL).read_text(encoding="utf-8").splitlines()
             if ln.startswith("# note: ")]
    if not notes:
        raise PackInputError(f"reports/{SHELL}: the analyst note lines are missing from the header")
    with open(Path(root) / SCOPE, encoding="utf-8") as fh:
        scope = yaml.safe_load(fh)
    for c in scope["threat_model"]:
        if not c.get("clause") or not c.get("source"):
            raise PackInputError(f"{SCOPE}: threat-model clause without a source: {c}")
    nfl = scope["no_feedback_loop"]
    if not nfl.get("text") or not nfl.get("source"):
        raise PackInputError(f"{SCOPE}: the no-feedback-loop statement needs a text and a source")
    run_names = {e.get("name") for e in index_entries(root)}
    rows = not_assessed_rows(entries, scope, run_names=run_names, long_rows=long44)
    evidence = []
    for o in applied_overrides(scope, run_names):
        for r in long44:
            if r["row_label"].startswith(o["long_row_prefix"]):
                evidence.append([r["row_label"], r["col_label"], r["value"], r["p_raw"], r["n_items"], r["run_ids"]])
    return {
        "success_table": md_table(SUCCESS_HEADER, success_rows(shell_rows, long44)),
        "denominator_table": md_table(DENOM_HEADER, denominator_rows(shell_rows, long44)),
        "notes": notes,
        "threat_model": [f"{c['clause']} (source: {c['source']})" for c in scope["threat_model"]],
        "no_feedback_loop": nfl["text"], "no_feedback_loop_source": nfl["source"],
        "not_assessed_table": md_table(["Attack type", "Status", "Why", "Source"], rows),
        "override_evidence": md_table(["Long-CSV row", "Metric", "Value", "p (raw)", "n", "Runs"], evidence),
        "shell_sha256": sha256_file(rep / SHELL), "long_sha256": sha256_file(rep / "long" / LONG),
        "data_version": meta["data_version"], "config_hashes": meta["config_hash"].split(";"),
    }
