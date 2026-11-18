"""Table D.2 (run record): sealed runs/<run_id>/run-record.json grouped into the shell rows (Baselines, R1, R2, R3, R4,
Evidence pack, Total) through the versioned regime -> row map configs/reports/d2-attribution.yaml (analyst
item O5a). Every indexed run belongs to exactly one row: a regime absent from the map, or listed twice, raises
UnattributedRegime. Configurations, Seeds, GPU-hours and Configuration hash cover the row's whole group (its own regime
plus the auxiliary regimes attributed to it, e.g. C1 / C2S training into R1); Items (train / test) are read from the
row's own regime only (BASE: test = run-record items.items). GPU-hours = sum of CUDA stage seconds / 3600. The Total
row reads `x of the 50 to 75 planned` where x is the sum over ALL indexed run records (computed, never typed); the
shell rows plus the long-only `Knowledge-store preparation (shared)` row (text-contradictions item 7) add up to it.
Each run has one long row `Run record (<run_id>)` with its host ROLE (GPU host / CPU host, from the env.json torch
build), release git_sha, determinism level, depends_on, lock sha256 prefix, torch build and driver (GPU-host runs:
runs/_probes/gpu-host.json), never a host name. Header comment lines `# host_change: <host> <date> <command>` come from
configs/host-changes.yaml (the only sanctioned host mention)."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR
from aac.reports import fmt, sources
from aac.run.paths import probes_dir, reports_dir, runs_dir
from aac.run.seal import verify_sealed

TABLE = "D.2"
GROUPS = (("Baselines (4.2)", "BASE"), ("R1 time-split", "R1"), ("R2 cross-dataset", "R2"), ("R3 adversarial", "R3"), ("R4 ablations", "R4"))
PACK_ROW = "Evidence pack (4.7)"
TOTAL_ROW = "Total"
KS_ROW = "Knowledge-store preparation (shared)"
KS_EMPTY = "n/a (no knowledge-store preparation run in this namespace)"
ATTACK_KEYS = ("attack_subset", "fc_positive", "rd_positive")
PLANNED = "of the 50 to 75 planned"
COLS = {"configurations": "Configurations run", "seeds": "Seeds", "items_train_test": "Items (train / test)",
        "gpu_hours": "GPU-hours", "config_hash": "Configuration hash"}
ATTRIBUTION = ("configs", "reports", "d2-attribution.yaml")
GPU_PROBE = "gpu_probe.json"
GPU_HOURS_DEF = ("sum of CUDA stage seconds / 3600 from run-record.json (stages with device == cuda); model load, "
                 "checkpoint hashing and sealing are not counted")


class UnattributedRegime(LookupError):
    """A regime of an indexed run has no single Table D.2 row in configs/reports/d2-attribution.yaml."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """safe_load that refuses a duplicate mapping key (plain safe_load keeps the last one silently)."""


def _construct_unique(loader: yaml.SafeLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    seen: set = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in seen:
            raise UnattributedRegime(f"configs/reports/d2-attribution.yaml: regime {key!r} is listed more than once; "
                                     f"a run belongs to exactly one Table D.2 row")
        seen.add(key)
    return yaml.SafeLoader.construct_mapping(loader, node, deep=deep)


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique)


def load_attribution(root: Path) -> dict[str, str]:
    """regime -> Table D.2 row label (a shell regime row or KS_ROW), from configs/reports/d2-attribution.yaml."""
    path = Path(root).joinpath(*ATTRIBUTION)
    with open(path, encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_UniqueKeyLoader) or {}
    rows = data.get("rows")
    if not isinstance(rows, dict) or not rows:
        raise ValueError(f"{path}: expected a non-empty top-level `rows:` mapping regime -> row label")
    allowed = {label for label, _ in GROUPS} | {KS_ROW}
    for regime, label in rows.items():
        if label not in allowed:
            raise ValueError(f"{path}: regime {regime!r} maps to {label!r}, which is not a Table D.2 regime row or {KS_ROW!r}")
    for label, own in GROUPS:
        if own in rows and rows[own] != label:
            raise ValueError(f"{path}: {own} must map to its own row {label!r} (the row's Items cell reads {own} alone)")
    return {str(k): str(v) for k, v in rows.items()}


def attribute(regimes, attribution: dict[str, str]) -> dict[str, str]:
    """regime -> row for every regime given; a regime absent from the map raises UnattributedRegime naming it."""
    missing = sorted({r for r in regimes if r not in attribution})
    if missing:
        raise UnattributedRegime(f"Table D.2: regime(s) {', '.join(missing)} of indexed runs have no row in "
                                 f"configs/reports/d2-attribution.yaml; add each to exactly one row")
    return {r: attribution[r] for r in regimes}


def host_changes(root: Path) -> list[str]:
    with open(root / "configs" / "host-changes.yaml", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    lines = []
    for c in data.get("changes") or []:
        lines.append(f"host_change: {c['host']} {c['date']} {c['command']}")
    return lines


def load_run_meta(root: Path, entry: dict) -> sources.Run:
    """sources.load_run without reading the decision-log records (Table D.2 needs none of them): the seal is verified
 against runs/INDEX.json and the shard hashes come from the sidecars in shard order, exactly as load_run does."""
    run_dir = runs_dir(root) / entry["run_id"]
    sealed = verify_sealed(run_dir)
    if sealed != entry["sealed_sha256"]:
        raise RuntimeError(f"{entry['run_id']}: SEALED {sealed[:12]} disagrees with runs/INDEX.json")
    resolved = json.loads((run_dir / "config.resolved.json").read_text(encoding="utf-8"))
    sidecars = [p.with_name(p.name + ".sha256").read_text(encoding="ascii").split()[0] for p in list_shards(run_dir / SHARD_DIR)]
    return sources.Run(entry=entry, run_dir=run_dir, resolved=resolved, records=(), shard_hashes=";".join(sidecars))


def _prov(runs: list) -> dict:
    """sources.provenance; a run set without any decision-log shard (external, training, evidence runs) names that."""
    p = sources.provenance(runs)
    if not p["source_shards_sha256"].strip(";"):
        p["source_shards_sha256"] = "n/a (no decision-log shards)"
    return p


def gpu_hours(record: dict) -> float:
    return sum(float(s["seconds"]) for s in record["stages"] if s.get("device") == "cuda") / 3600.0


def _items(record: dict) -> tuple[str, str]:
    items = record["items"]
    train = str(items["train"]) if "train" in items else fmt.NA
    if record["regime"] == "R3":
        n = sum(int(items[k]) for k in ATTACK_KEYS if k in items)
        return train, f"{n} attack candidates ({', '.join(f'{k} {items[k]}' for k in ATTACK_KEYS if k in items)})"
    if record["regime"] == "BASE" and "items" in items:
        return train, str(items["items"])  # external baseline runs record their item count as items.items
    n = sum(int(s["items"]) for s in record["stages"] if str(s["name"]).startswith("infer_"))
    return train, str(n)


def _distinct(values: list[str]) -> str:
    uniq = sorted(set(values), key=values.index)
    return uniq[0] if len(uniq) == 1 else "; ".join(uniq)


def host_role(torch_version: str) -> str:
    """GPU host / CPU host from the env.json torch build suffix (`+cu126` / `+cpu`); never a host name."""
    build = str(torch_version).partition("+")[2]
    if build.startswith("cu"):
        return "GPU host"
    if build == "cpu":
        return "CPU host"
    raise ValueError(f"torch build {torch_version!r} has no +cuXXX / +cpu suffix; host role unknown")


def _gpu_driver(root: Path) -> str:
    path = probes_dir(root) / GPU_PROBE
    if not path.is_file():
        return f"n/a (no runs/_probes/{GPU_PROBE})"
    return str(json.loads(path.read_text(encoding="utf-8")).get("driver") or f"n/a (runs/_probes/{GPU_PROBE} has no driver)")


def run_record_text(record: dict, env: dict, driver: str) -> str:
    role = host_role(env["torch"])
    deps = ", ".join(record.get("depends_on") or []) or "none"
    return (f"host role {role}; release {str(record['git_sha'])[:12]}; determinism {record['determinism_level']}; "
            f"depends_on {deps}; lock {str(env['lock_sha256'])[:12]}; torch {env['torch']}; "
            f"driver {driver if role == 'GPU host' else fmt.NA}")


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    entries = sources.index_runs(root)
    if not entries:
        raise sources.MissingRun("Table D.2: runs/INDEX.json lists no sealed run")
    row_of = attribute(sorted({e["regime"] for e in entries}), load_attribution(root))
    runs = [load_run_meta(root, e) for e in entries]
    records = {r.run_id: json.loads((r.run_dir / "run-record.json").read_text(encoding="utf-8")) for r in runs}
    envs = {r.run_id: json.loads((r.run_dir / "env.json").read_text(encoding="utf-8")) for r in runs}
    driver = _gpu_driver(root)
    hours_of = {rid: gpu_hours(rec) for rid, rec in records.items()}
    shell_rows, long_rows = [], []
    all_prov = _prov(runs)
    n_runs = len(runs)
    all_seeds = ", ".join(map(str, sorted({r.seed for r in runs})))
    pack_common = dict(n_items=n_runs, n_seeds=len({r.seed for r in runs}), seeds=all_seeds, **all_prov)
    row_hours: dict[str, float] = {}

    def regime_rows(row_label: str, group: list, own: str | None) -> None:
        """Per-regime GPU-hours (6 decimals) and, for auxiliary regimes, their item counts."""
        for regime in sorted({r.entry["regime"] for r in group}):
            sub = [r for r in group if r.entry["regime"] == regime]
            recs = [records[r.run_id] for r in sub]
            prov = _prov(sub)
            seeds = sorted({int(rec["seed"]) for rec in recs})
            kw = dict(n_items=len(sub), n_seeds=len(seeds), seeds=", ".join(map(str, seeds)), **prov)
            long_rows.append(fmt.long_row(TABLE, row_label, f"{COLS['gpu_hours']} ({regime})", "gpu_hours",
                                          fmt._round(sum(hours_of[r.run_id] for r in sub), 6),
                                          test=f"{GPU_HOURS_DEF}; {len(sub)} {regime} run record(s)", **kw))
            if regime == own:
                continue
            for key in sorted({k for rec in recs for k in rec["items"]}):
                vals = [str(rec["items"][key]) for rec in recs if key in rec["items"]]
                long_rows.append(fmt.long_row(TABLE, row_label, f"{COLS['items_train_test']} ({key}, {regime})",
                                              f"items.{key}", _distinct(vals),
                                              test=f"run-record.json items.{key} of the {regime} runs attributed to this row "
                                                   f"(configs/reports/d2-attribution.yaml)", **kw))

    def run_rows(row_label: str, group: list) -> None:
        for r in group:
            rec = records[r.run_id]
            long_rows.append(fmt.long_row(TABLE, row_label, f"Run record ({r.run_id})", "run_record",
                                          run_record_text(rec, envs[r.run_id], driver), n_items=1, n_seeds=1, seeds=str(r.seed),
                                          test=("host role from the env.json torch build (+cuXXX GPU host, +cpu CPU host); release = "
                                                "run-record git_sha (12 hex); lock = env.json lock_sha256 (12 hex); driver from "
                                                f"runs/_probes/{GPU_PROBE} for GPU-host runs"),
                                          **_prov([r])))

    for row_label, own in GROUPS:
        group = [r for r in runs if row_of[r.entry["regime"]] == row_label]
        own_runs = [r for r in group if r.entry["regime"] == own]
        if not own_runs:
            raise sources.MissingRun(f"Table D.2 row {row_label!r}: no sealed {own} run in runs/INDEX.json")
        recs = [records[r.run_id] for r in group]
        own_recs = [records[r.run_id] for r in own_runs]
        configs = sorted({c for rec in recs for c in rec["configurations"]})
        cfg_cell = "; ".join(configs)
        if own == "R3":
            attacks = sorted({a for r in group for a in (r.resolved.get("attack") or {}).get("attacks", [])})
            if attacks:
                cfg_cell += "; attacks: " + ", ".join(attacks)
        seeds = sorted({int(rec["seed"]) for rec in recs})
        trains, tests = zip(*(_items(rec) for rec in own_recs))
        items_cell = f"{_distinct(list(trains))} / {_distinct(list(tests))}"
        hours = sum(hours_of[r.run_id] for r in group)
        row_hours[row_label] = hours
        hashes = sorted({rec["config_hash"] for rec in recs})
        hash_cell = "; ".join(h[:12] for h in hashes)
        prov = _prov(group)
        common = dict(n_items=len(group), n_seeds=len(seeds), seeds=", ".join(map(str, seeds)), **prov)
        L = lambda col, metric, value, **kw: long_rows.append(fmt.long_row(TABLE, row_label, col, metric, value, **{**common, **kw}))
        shell_rows.append({"regime": row_label, "configurations": cfg_cell, "seeds": ", ".join(map(str, seeds)),
                           "items_train_test": items_cell, "gpu_hours": fmt._round(hours, 1), "config_hash": hash_cell})
        aux = sorted({r.entry["regime"] for r in group} - {own})
        L(COLS["configurations"], "configurations", cfg_cell, test=f"{len(group)} sealed run(s): " + "; ".join(r.run_id for r in group))
        L(COLS["seeds"], "seeds", ", ".join(map(str, seeds)))
        if own == "R3":
            items_test = "attack candidates = " + " + ".join(ATTACK_KEYS)
        elif own == "BASE":
            items_test = "train from run-record.json items.train; test = items.items (external runs) or the items of the infer_* stages"
        else:
            items_test = "train from run-record.json items.train (split manifest); test = items of the infer_* stages"
        L(COLS["items_train_test"], "n_train / n_test", items_cell,
          test=f"{items_test}; {own} runs only" + (f" (attributed {', '.join(aux)} runs: long rows by regime)" if aux else ""),
          **{**dict(n_items=len(own_runs)), **_prov(own_runs)})
        for key in sorted({k for rec in own_recs for k in rec["items"]}):
            vals = [str(rec["items"][key]) for rec in own_recs if key in rec["items"]]
            L(f"{COLS['items_train_test']} ({key})", f"items.{key}", _distinct(vals), n_items=len(own_runs), **_prov(own_runs))
        L(COLS["gpu_hours"], "gpu_hours", fmt._round(hours, 1),
          test=f"{GPU_HOURS_DEF}; over the row's group ({', '.join(sorted({r.entry['regime'] for r in group}))}); "
               f"configs/reports/d2-attribution.yaml")
        L(f"{COLS['gpu_hours']} (exact)", "gpu_hours_exact", fmt._round(hours, 6), test=f"{GPU_HOURS_DEF}; 6 decimals")
        L(COLS["config_hash"], "config_hash_short", hash_cell, test="first 12 hex of each distinct config_hash; full hashes in this row's config_hash column")
        regime_rows(row_label, group, own)
        run_rows(row_label, group)

    shell_rows.append({"regime": PACK_ROW, "configurations": "all records", "seeds": fmt.NA, "items_train_test": fmt.NA,
                       "gpu_hours": "under 1", "config_hash": f"all {len(all_prov['config_hash'].split(';'))} hashes (long CSV)"})
    for key, col in COLS.items():
        value = {"configurations": "all records", "seeds": fmt.NA, "items_train_test": fmt.NA, "gpu_hours": "under 1",
                 "config_hash": f"all {len(all_prov['config_hash'].split(';'))} hashes (long CSV)"}[key]
        long_rows.append(fmt.long_row(TABLE, PACK_ROW, col, key, value, test=f"evidence pack over all {n_runs} sealed runs (CPU regeneration)", **pack_common))

    # shared knowledge-store preparation (long CSV only; text-contradictions item 7 asks the analyst to add the shell row)
    ks_group = [r for r in runs if row_of[r.entry["regime"]] == KS_ROW]
    ks_hours = sum(hours_of[r.run_id] for r in ks_group)
    if not ks_group:
        for key, col in COLS.items():
            long_rows.append(fmt.long_row(TABLE, KS_ROW, col, key, KS_EMPTY,
                                          test="no indexed run's regime maps to this row in configs/reports/d2-attribution.yaml",
                                          **pack_common))
    else:
        ks_recs = [records[r.run_id] for r in ks_group]
        ks_seeds = sorted({int(rec["seed"]) for rec in ks_recs})
        ks_prov = _prov(ks_group)
        ks_common = dict(n_items=len(ks_group), n_seeds=len(ks_seeds), seeds=", ".join(map(str, ks_seeds)), **ks_prov)
        ks_regimes = ", ".join(sorted({r.entry["regime"] for r in ks_group}))
        ks_hashes = sorted({rec["config_hash"] for rec in ks_recs})
        values = {"configurations": "; ".join(sorted({c for rec in ks_recs for c in rec["configurations"]})),
                  "seeds": ", ".join(map(str, ks_seeds)),
                  "items_train_test": "n/a (shared preparation; item counts per regime in the rows below)",
                  "gpu_hours": fmt._round(ks_hours, 1),
                  "config_hash": "; ".join(h[:12] for h in ks_hashes)}
        tests = {"configurations": f"{len(ks_group)} sealed run(s) ({ks_regimes}): " + "; ".join(r.run_id for r in ks_group),
                 "gpu_hours": f"{GPU_HOURS_DEF}; over the {ks_regimes} run records (BM25 shortlists on CPU, BGE-M3 scoring and "
                              f"re-ranking on GPU); the draft shell lacks this row (text-contradictions item 7)"}
        for key, col in COLS.items():
            long_rows.append(fmt.long_row(TABLE, KS_ROW, col, key, values[key], test=tests.get(key, "n/a"), **ks_common))
        long_rows.append(fmt.long_row(TABLE, KS_ROW, f"{COLS['gpu_hours']} (exact)", "gpu_hours_exact", fmt._round(ks_hours, 6),
                                      test=f"{GPU_HOURS_DEF}; 6 decimals", **ks_common))
        regime_rows(KS_ROW, ks_group, None)
        run_rows(KS_ROW, ks_group)

    total_hours = sum(hours_of[r.run_id] for r in runs)  # over ALL indexed run records, never the sum of printed cells
    if abs(total_hours - (sum(row_hours.values()) + ks_hours)) > 1e-9:
        raise RuntimeError(f"Table D.2: rows ({sum(row_hours.values()) + ks_hours!r}) do not add up to the total over all "
                           f"run records ({total_hours!r}); a run is attributed twice or not at all")
    total_cell = f"{fmt._round(total_hours, 1)} {PLANNED}"
    n_hashes = len(all_prov["config_hash"].split(";"))
    shell_rows.append({"regime": TOTAL_ROW, "configurations": f"{n_runs} sealed runs", "seeds": all_seeds,
                       "items_train_test": fmt.NA, "gpu_hours": total_cell, "config_hash": f"{n_hashes} distinct"})
    for key, col in COLS.items():
        value = {"configurations": f"{n_runs} sealed runs", "seeds": all_seeds, "items_train_test": fmt.NA,
                 "gpu_hours": total_cell, "config_hash": f"{n_hashes} distinct"}[key]
        long_rows.append(fmt.long_row(TABLE, TOTAL_ROW, col, key, value,
                                      test="sum over all indexed run records (GPU-hours) / distinct over all sealed runs", **pack_common))
    long_rows.append(fmt.long_row(TABLE, TOTAL_ROW, f"{COLS['gpu_hours']} (sum over all run records)", "gpu_hours_total",
                                  fmt._round(total_hours, 6),
                                  test=f"{GPU_HOURS_DEF}; over all {n_runs} indexed run records; equals the five shell regime rows "
                                       f"plus the {KS_ROW} long row", **pack_common))
    return list(fmt.emit(TABLE, shell_rows, long_rows, config_hash=all_prov["config_hash"], data_version=all_prov["data_version"],
                         out_dir=out_dir, comment_lines=host_changes(root)))
