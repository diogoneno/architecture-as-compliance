"""Appendix subgroup deferral table (methodological note`).

Source: the sealed r1-chrono-full seed-0 run (the same run and loader as Table 4.6), its chronological test
records, deferral = the logged C5 decision at the 80 percent coverage threshold (abstention_flag; Table 4.6
checks that its own 80 percent rule reproduces these flags), reason = the logged abstention_reason.

Claim metadata (claim_types, location_ISO_code) is read from the raw AVeriTeC files after verify_entries (the
directory rule of aac.data.manifest.verify_entries, restated here) has re-hashed them against the run's data
manifest. Rows:
 - Claim type: one row per claim_types value of the shell (multi-type claims count once per type); a value
 outside the shell rows stops the build (fixed row identities)
 - Location: Rank 1 to Rank 10 = the ten most frequent location codes of the test set, by count then code (the
 code in the Code column); Other = every remaining code; Unknown = no code recorded
Per row: n, deferral rate with a percentile item bootstrap CI (1,000 resamples, seed key
boot.subgroup-deferral.<Field>.<Group>.deferral_rate), and the reason shares of the row's deferrals.

A namespace whose run reads the fixture corpus (the stub smoke chain) gets no table: the generator returns 
(registry.REAL_ONLY), so the stub goldens gain no file."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from aac.data.averitec import parse_claim_id, read_raw
from aac.reports import fmt
from aac.reports.table_4_6 import load_test_records, select_run
from aac.run.hashing import derive_seed, sha256_file
from aac.run.paths import reports_dir, runs_dir
from aac.run.seal import verify_sealed
from aac.stats.bootstrap import ci_index

TABLE = "subgroup_deferral"
TOP_LOCATIONS = 10
REASONS = (("reason_low_conf", "low_confidence"), ("reason_no_evidence", "no_evidence"), ("reason_conflict", "conflicting_evidence"))
NO_DEFERRALS = "n/a (no deferrals)"
OTHER_CODE = "n/a (other codes)"
UNKNOWN_CODE = "n/a (none recorded)"


class RawDrift(RuntimeError):
    """A raw file the table reads is not covered by the data manifest or differs from its listed sha256."""


def verify_entries(root: Path, manifest: str, directory: str) -> int:
    """Re-hash every file the data manifest lists under `directory` (project-relative, under data/raw/) before the
 table reads it; at least one entry must be listed and every listed file present on this host must match.
 Returns the number of files re-hashed. This is the directory rule of aac.data.manifest.verify_entries; that
 module is not imported because its CLI gate imports aac.components, which the reports layer may not reach
 (import-linter contract; tests/unit/test_table_subgroup.py checks both functions agree on the real files)."""
    root = Path(root)
    raw_prefix = "data/raw/"
    d = Path(directory).as_posix().rstrip("/")
    if not d.startswith(raw_prefix):
        raise RawDrift(f"{directory}: the subgroup table reads raw AVeriTeC files under {raw_prefix} only")
    rel_dir = "raw/" + d[len(raw_prefix):]
    listed = {}
    for n, line in enumerate((root / manifest).read_text(encoding="ascii").splitlines(), 1):
        if not line.strip():
            continue
        digest, sep, rel = line.partition("  ")
        if not sep or len(digest) != 64:
            raise RawDrift(f"{manifest}:{n}: not a sha256sum line")
        listed[rel] = digest
    under = sorted(r for r in listed if r.startswith(rel_dir + "/"))
    if not under:
        raise RawDrift(f"{directory}: no entry under {rel_dir}/ in {manifest}")
    count = 0
    for rel in under:
        path = root / raw_prefix / rel[len("raw/"):]
        if path.is_file():
            got = sha256_file(path)
            if got != listed[rel]:
                raise RawDrift(f"{rel}: on disk {got[:12]} != listed {listed[rel][:12]} in {manifest}")
            count += 1
    if not count:
        raise RawDrift(f"{directory}: no listed file is present on this host")
    return count


def _blank(v) -> bool:
    return v is None or (isinstance(v, str) and not v.strip())


def location_ranks(codes: list) -> tuple[list[str], set[str]]:
    """(the ten most frequent codes, by count descending then code; the remaining codes). Blank codes are unknown."""
    counts = Counter(c for c in codes if not _blank(c))
    ranked = [c for c, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
    return ranked[:TOP_LOCATIONS], set(ranked[TOP_LOCATIONS:])


def group_members(meta: list[dict], claim_type_rows: list[str]) -> list[tuple[str, str, str, np.ndarray]]:
    """[(field, group, code, member mask)] in shell order. meta[i] = {"claim_types": [...], "location": code|None}."""
    n = len(meta)
    types = [set(m["claim_types"] or []) for m in meta]
    unknown_types = sorted(set().union(*types) - set(claim_type_rows)) if types else []
    if unknown_types:
        raise fmt.ShellMismatch(f"{TABLE}: claim types {unknown_types} are not shell rows {claim_type_rows}")
    out = []
    for ct in claim_type_rows:
        out.append(("Claim type", ct, fmt.NA, np.asarray([ct in t for t in types], dtype=bool)))
    locs = [m["location"] for m in meta]
    top, rest = location_ranks(locs)
    for k in range(TOP_LOCATIONS):
        code = top[k] if k < len(top) else None
        mask = np.asarray([code is not None and loc == code for loc in locs], dtype=bool)
        out.append(("Location", f"Rank {k + 1}", code if code is not None else "n/a (fewer than ten codes)", mask))
    out.append(("Location", "Other", OTHER_CODE, np.asarray([not _blank(l) and l in rest for l in locs], dtype=bool)))
    out.append(("Location", "Unknown", UNKNOWN_CODE, np.asarray([_blank(l) for l in locs], dtype=bool)))
    assert all(m.shape == (n,) for *_, m in out)
    return out


def rows_for(meta: list[dict], deferred: np.ndarray, reasons: list[str | None], claim_type_rows: list[str], *,
             n_resamples: int, level: float) -> list[dict]:
    """One dict per shell row: field, group, code, n, rate, ci, deferrals, reason counts, other codes."""
    deferred = np.asarray(deferred, dtype=bool)
    out = []
    _, rest = location_ranks([m["location"] for m in meta])
    for field, group, code, mask in group_members(meta, claim_type_rows):
        n = int(mask.sum())
        d = deferred[mask].astype(float)
        if n:
            rate = float(d.mean())
            lo, hi = ci_index(lambda idx, d=d: float(d[idx].mean()), n, n_resamples=n_resamples,
                              seed=derive_seed(f"boot.subgroup-deferral.{field}.{group}.deferral_rate"), level=level)
        else:
            rate, lo, hi = None, None, None
        rs = [r for r, m, x in zip(reasons, mask, deferred) if m and x]
        out.append({"field": field, "group": group, "code": code, "n": n, "rate": rate, "ci": (lo, hi), "n_deferred": len(rs),
                    "reasons": {key: rs.count(name) for key, name in REASONS},
                    "other_codes": sorted(rest) if group == "Other" else []})
    return out


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    entry = select_run(root)
    run_dir = runs_dir(root) / entry["run_id"]
    sealed = verify_sealed(run_dir)
    if sealed != entry["sealed_sha256"]:
        raise RuntimeError(f"{entry['run_id']}: SEALED {sealed[:12]} disagrees with runs/INDEX.json")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    data = resolved["data"]
    if data.get("dataset") != "averitec":
        return []  # fixture corpus (stub namespace): the table is real-only (registry.REAL_ONLY)
    verify_entries(root, data["manifest"], data["claims"])
    raw = read_raw(root / data["claims"])
    recs, shard_hashes = load_test_records(run_dir)
    meta = []
    for r in recs:
        split, idx = parse_claim_id(r["claim_id"])
        claim = raw[split][idx]
        meta.append({"claim_types": list(claim.get("claim_types") or []), "location": claim.get("location_ISO_code")})
    deferred = np.asarray([bool(r["abstention_flag"]) for r in recs], dtype=bool)
    reasons = [r["abstention_reason"] for r in recs]
    shell = fmt.load_shells()[TABLE]
    claim_type_rows = [g for f, g in shell["rows"] if f == "Claim type"]
    n_boot = int(resolved["stats"]["bootstrap"])
    level = float(resolved["stats"]["level"])
    rows = rows_for(meta, deferred, reasons, claim_type_rows, n_resamples=n_boot, level=level)
    ch, dv = entry["config_hash"], resolved["data_version"]
    common = dict(config_hash=ch, data_version=dv, run_ids=entry["run_id"], source_shards_sha256=shard_hashes, n_seeds=1, seeds="0")
    labels = {c["key"]: c["label"] for c in shell["columns"]}
    shell_rows, long_rows = [], []
    for row in rows:
        ident = f"{row['field']} | {row['group']}"
        L = lambda key, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, ident, labels[key], metric, value, **{**common, **kw}))
        if row["rate"] is None:
            rate_cell = "n/a (no items)"
            L("deferral_rate", "deferral_rate_pct", rate_cell, n_items=0)
        else:
            rate_cell = fmt.fmt_rate_ci(row["rate"], *row["ci"])
            L("deferral_rate", "deferral_rate_pct", fmt.fmt_rate(row["rate"]), ci_lo=fmt.fmt_rate(row["ci"][0]), ci_hi=fmt.fmt_rate(row["ci"][1]),
              n_items=row["n"], test=(f"share of the row's test items deferred by C5 at the 80 percent threshold (abstention_flag); "
                                     f"percentile item bootstrap, {n_boot} resamples, seed key boot.subgroup-deferral.{row['field']}.{row['group']}.deferral_rate"))
        code_test = ("location_ISO_code of this rank (count then code)" if row["group"].startswith("Rank")
                     else ("codes: " + ", ".join(row["other_codes"]) if row["group"] == "Other" else "n/a"))
        L("code", "location_code", row["code"], test=code_test)
        L("n_items", "n_items", str(row["n"]), n_items=row["n"],
          test="test items of the row (a claim with several claim types counts once per type)" if row["field"] == "Claim type" else "test items of the row")
        cells = {"field": row["field"], "group": row["group"], "code": row["code"], "n_items": str(row["n"]), "deferral_rate": rate_cell}
        for key, name in REASONS:
            if row["n_deferred"] == 0:
                cell = NO_DEFERRALS
                L(key, f"{key}_pct", cell, n_items=0)
            else:
                cell = fmt.fmt_rate(row["reasons"][key] / row["n_deferred"])
                L(key, f"{key}_pct", cell, n_items=row["n_deferred"], test=f"logged abstention_reason {name} among the row's {row['n_deferred']} deferrals")
            cells[key] = cell
        shell_rows.append(cells)
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=ch, data_version=dv, out_dir=out_dir))
