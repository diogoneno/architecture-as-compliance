"""Run identity from (config, seed): resolved config, config_hash, data_version, run_id,
model_version. Shared by the CLI and aac.run.matrix.

Config kinds (aac.run.config.SCHEMAS): a regime run (`aac-config/1`, model_version over the
component configs), a C1 training run (`aac-train/1`, model_version over the train block) and a benchmark
run (`aac-bench/1`, model_version over the bench block, whose c1.trained_by is resolved into
c1.checkpoint exactly like components.c1.trained_by) and an external baseline run (`aac-external/1`, model_version over the external block, whose checkpoint_sha256 and bodega_commit are resolved here) and a shortlist
run (`aac-shortlist/1`, model_version over the shortlist block) and an evidence run (`aac-evidence/1`,
model_version over the evidence block, whose shortlist_from / pool_from name is resolved into a RunRef by
resolve_run_ref) and a C2 scorer training run (`aac-train-scorer/1`, model_version over the scorer block,
whose evidence_from name is resolved into scorer.evidence) and a BODEGA scoring run (`aac-bodega/1`, model_version over the bodega block, whose `sources` R3 run names are resolved into source_runs by
resolve_bodega_sources). A regime run whose components.c2 is the real C2 gets
c2.evidence (resolve_real_c2) and c2.scorer (resolve_real_c2_scorer: the C2S run scorer_trained_by at scorer_seed,
null = the run seed) filled before config_hash. `_MV_BLOCK` maps each config class to its block; a regime run (RunConfig) uses the per-component
hashes instead."""
from __future__ import annotations

from pathlib import Path

from aac.run import hashing
from aac.run.clock import git_sha
from aac.run.config import (MDEBERTA_KIND, REAL_C2_KIND, BenchConfig, BodegaScoreConfig, EvidenceConfig, ExternalConfig,
                             RunConfig, ScorerTrainConfig, ShortlistConfig, TrainConfig, load_any, resolve)

CHECKPOINT_FILE = "checkpoint.safetensors"
# config class -> the resolved block whose hash is model_version's single part (RunConfig: per-component hashes)
_MV_BLOCK = {TrainConfig: "train", BenchConfig: "bench", ExternalConfig: "external", ShortlistConfig: "shortlist",
             EvidenceConfig: "evidence", ScorerTrainConfig: "scorer", BodegaScoreConfig: "bodega"}


class MissingCheckpointRun(LookupError):
    """components.c1.trained_by (or components.c2.scorer_trained_by) names no sealed C1 (C2S) run at this seed in
 runs_dir/INDEX.json."""


# regime of a weights-file run -> the config field that names it (error messages only)
_TRAINED_BY_FIELD = {"C1": "components.c1.trained_by", "C2S": "components.c2.scorer_trained_by"}


def lock_path(root: Path) -> Path:
    return Path(root) / "uv.lock"


def resolve_checkpoint(root: Path, trained_by: str, seed: int, regime: str = "C1") -> dict:
    """{run_id, sha256} of the sealed training run `trained_by` at `seed` (regime `regime`, C1 or C2S; name ==
 trained_by).

 The checkpoint is re-hashed here and must equal the sha256 its sealed model.json records
 (ChecksumMismatch otherwise), so a replaced checkpoint file can never enter a run identity silently."""
    import json

    from aac.run.checkpoints import ChecksumMismatch
    from aac.run.index import load_index
    from aac.run.paths import runs_dir

    runs = runs_dir(root)
    hits = sorted(r["run_id"] for r in load_index(runs / "INDEX.json")["runs"]
                  if r.get("regime") == regime and r.get("name") == trained_by and int(r.get("seed", -1)) == int(seed))
    if len(hits) != 1:
        what = "no" if not hits else f"{len(hits)} ({', '.join(hits)})"
        field = _TRAINED_BY_FIELD.get(regime, f"{regime} trained_by")
        raise MissingCheckpointRun(f"{field} {trained_by!r}: {what} sealed {regime} run at seed {seed} in "
                                   f"{runs / 'INDEX.json'}; train it first (aac train --seed {seed})")
    rid = hits[0]
    run_dir = runs / rid
    if not (run_dir / "SEALED").is_file() or not (run_dir / CHECKPOINT_FILE).is_file():
        raise MissingCheckpointRun(f"{rid} (trained_by {trained_by!r}, seed {seed}) is indexed but has no SEALED "
                                   f"{CHECKPOINT_FILE} under {runs}")
    sha = hashing.sha256_file(run_dir / CHECKPOINT_FILE)
    recorded = json.loads((run_dir / "model.json").read_text(encoding="utf-8"))["checkpoint_sha256"]
    if sha != recorded:
        raise ChecksumMismatch(f"{run_dir / CHECKPOINT_FILE}: sha256 {sha} != model.json {recorded}")
    return {"run_id": rid, "sha256": sha}


class MissingRun(LookupError):
    """A `*_from` run name resolves to no single sealed run of that regime at that seed in runs_dir/INDEX.json."""


def resolve_run_ref(root: Path, regime: str, name: str, seed: int = 0) -> dict:
    """{run_id, sha256} of the one indexed run of `regime` named `name` at `seed`; sha256 is its SEALED digest,
 recomputed here by aac.run.seal.verify_sealed (UnsealedRun when a file changed after sealing)."""
    from aac.run.index import load_index
    from aac.run.paths import runs_dir
    from aac.run.seal import verify_sealed

    runs = runs_dir(root)
    hits = sorted(r["run_id"] for r in load_index(runs / "INDEX.json")["runs"]
                  if r.get("regime") == regime and r.get("name") == name and int(r.get("seed", -1)) == int(seed))
    if len(hits) != 1:
        what = "no" if not hits else f"{len(hits)} ({', '.join(hits)})"
        raise MissingRun(f"{regime} run {name!r}: {what} indexed run at seed {seed} in {runs / 'INDEX.json'}")
    rid = hits[0]
    if not (runs / rid / "SEALED").is_file():
        raise MissingRun(f"{rid} ({regime} {name!r}, seed {seed}) is indexed but has no SEALED run directory under {runs}")
    return {"run_id": rid, "sha256": verify_sealed(runs / rid)}


# evidence.mode -> (name field, ref field, regime of the referenced run)
_EV_REFS = {"per_claim": ("shortlist_from", "shortlist", "SL"), "pooled_rd": ("pool_from", "pool", "EV")}


def resolve_evidence(root: Path, ev: dict) -> None:
    """Fill evidence.shortlist (per_claim, from shortlist_from) or evidence.pool (pooled_rd, from pool_from) in place,
 BEFORE config_hash (the external-run pattern). A ref typed in the YAML is refused: leave it unset."""
    name_key, ref_key, regime = _EV_REFS[ev["mode"]]
    if ev.get(ref_key) is not None:
        raise ValueError(f"evidence.{ref_key} is resolved from evidence.{name_key}; leave it unset")
    ev[ref_key] = resolve_run_ref(root, regime, ev[name_key], 0)


def resolve_real_c2(root: Path, c2: dict) -> None:
    """Fill components.c2.evidence from evidence_from (the EV run at seed 0; retrieval is seed-independent) in place,
 BEFORE config_hash. Refused: a typed evidence ref; dense or reranker blocks, or k / k_dense , that differ from
 the EV run's config.resolved.json (the C2 recompute path must embed and re-rank exactly as the EV run did); a variant the EV
 run did not compute; an EV run whose mode does not fit the provider. Provider `pooled` reads the RD evidence
 cache (aac.data.evidence_providers.PooledRdStore), so its evidence_from must name a pooled_rd EV run (e.g.
 `ev-rd-pooled`); the pooled bundle is checked through that run's own pool ref. Provider `averitec_store` reads a
 per_claim EV run (integration item 1 of wave 3). The scorer ref: resolve_real_c2_scorer."""
    import json

    from aac.run.paths import runs_dir

    if c2.get("evidence") is not None:
        raise ValueError("components.c2.evidence is resolved from components.c2.evidence_from; leave it unset")
    if not c2.get("evidence_from"):
        return
    ref = resolve_run_ref(root, "EV", c2["evidence_from"], 0)
    ev = json.loads((runs_dir(root) / ref["run_id"] / "config.resolved.json").read_text(encoding="utf-8"))["evidence"]
    for block in ("dense", "reranker"):
        if c2[block] != ev[block]:
            diff = sorted(k for k in set(c2[block]) | set(ev[block]) if c2[block].get(k) != ev[block].get(k))
            raise ValueError(f"components.c2.{block} differs from EV run {ref['run_id']} in {diff}; they must be equal")
    
    # the EV run did not see, e.g. an adversarial text) ranks with c2's values. They must be equal or the provider path,
    # not the perturbation, changes n_evidence / top_rerank_score.
    for key in ("k", "k_dense"):
        if int(c2[key]) != int(ev[key]):
            raise ValueError(f"components.c2.{key} {c2[key]} differs from EV run {ref['run_id']} ({ev[key]}); "
                             "they must be equal")
    if c2["variant"] not in ev["variants"]:
        raise ValueError(f"components.c2.variant {c2['variant']!r} is not a list of EV run {ref['run_id']} {ev['variants']}")
    want_mode = {"pooled": "pooled_rd", "averitec_store": "per_claim"}.get(c2["provider"])
    if want_mode is not None and ev["mode"] != want_mode:
        raise ValueError(f"components.c2.provider {c2['provider']} reads a {want_mode} EV run; {ref['run_id']} is mode "
                         f"{ev['mode']}")
    c2["evidence"] = ref


def resolve_real_c2_scorer(root: Path, c2: dict, seed: int) -> None:
    """Fill components.c2.scorer in place, BEFORE config_hash: the C2S run `scorer_trained_by` at `scorer_seed` (null =
 the run seed), its checkpoint re-hashed against its sealed model.json (resolve_checkpoint). A typed ref is refused."""
    if c2.get("scorer") is not None:
        raise ValueError("components.c2.scorer is resolved from components.c2.scorer_trained_by; leave it unset")
    at = int(seed) if c2.get("scorer_seed") is None else int(c2["scorer_seed"])
    c2["scorer"] = resolve_checkpoint(root, c2["scorer_trained_by"], at, regime="C2S")


def resolve_scorer_evidence(root: Path, sc: dict) -> None:
    """Fill scorer.evidence (the EV run `evidence_from` at seed 0; retrieval is seed-independent) in place, BEFORE
 config_hash. A ref typed in the YAML is refused: leave it unset."""
    if sc.get("evidence") is not None:
        raise ValueError("scorer.evidence is resolved from scorer.evidence_from; leave it unset")
    sc["evidence"] = resolve_run_ref(root, "EV", sc["evidence_from"], 0)


def resolve_bodega_sources(root: Path, resolved: dict, seed: int) -> None:
    """Fill source_runs [{name, run_id, sealed_sha256}] of a BODEGA scoring run in place, BEFORE config_hash : each `sources` name is the one indexed R3 run of that name at `seed`, its SEALED digest recomputed here
 (resolve_run_ref; UnsealedRun when a file changed after sealing). A re-sealed source therefore changes config_hash
 and the run id. A typed source_runs is refused (the config model refuses it too)."""
    if resolved.get("source_runs") is not None:
        raise ValueError("source_runs is resolved from sources; leave it unset")
    refs = []
    for name in resolved["sources"]:
        ref = resolve_run_ref(root, "R3", name, seed)
        refs.append({"name": name, "run_id": ref["run_id"], "sealed_sha256": ref["sha256"]})
    resolved["source_runs"] = refs


class UnknownCheckpoint(LookupError):
    """external.checkpoint_path has no recorded sha256 in the raw or extra manifest."""


def recorded_sha256(root: Path, rel: str) -> str:
    """The recorded sha256 of data/raw/<rel>: data/raw/MANIFEST.expected.sha256 or configs/data/expected-extra.sha256
 (aac.data.verify.load_expected; a path in both must agree). Never a measured value."""
    from aac.data.verify import load_expected  # noqa: PLC0415 (aac.data sits above aac.run)

    exp = load_expected(root)
    if rel not in exp:
        raise UnknownCheckpoint(f"external.checkpoint_path {rel!r} has no recorded sha256 in "
                                "data/raw/MANIFEST.expected.sha256 or configs/data/expected-extra.sha256")
    return exp[rel]


def gitlink_commit(root: Path, rel: str = "third_party/BODEGA") -> str:
    """The submodule commit recorded at HEAD for `rel` (`git ls-tree HEAD <rel>`), not the checkout's state."""
    import re
    import subprocess

    out = subprocess.run(["git", "ls-tree", "HEAD", rel], cwd=str(root), capture_output=True, text=True,
                         check=True).stdout.split()
    if len(out) < 3 or out[1] != "commit" or not re.fullmatch(r"[0-9a-f]{40}", out[2]):
        raise LookupError(f"git ls-tree HEAD {rel}: no gitlink commit ({out!r})")
    return out[2]


def resolve_external(root: Path, ext: dict) -> None:
    """Fill external.checkpoint_sha256 (recorded manifests) and, for BODEGA tools, external.bodega_commit (gitlink
 at HEAD) in place, BEFORE config_hash. A value typed in the YAML is refused: leave both unset."""
    for key in ("checkpoint_sha256", "bodega_commit"):
        if ext.get(key) is not None:
            raise ValueError(f"external.{key} is resolved from the recorded manifests / git; leave it unset")
    ext["checkpoint_sha256"] = recorded_sha256(root, ext["checkpoint_path"])
    if str(ext["tool"]).startswith("bodega_"):
        ext["bodega_commit"] = gitlink_commit(root)


def identity(root: Path, config: Path, seed: int) -> dict:
    cfg = load_any(config)
    sha = git_sha(cwd=root)
    lock = lock_path(root)
    lock_sha = hashing.sha256_file(lock)
    resolved = resolve(cfg, git_sha=sha, lock_sha256=lock_sha, seed=seed, lock_path=lock)
    c1, where = None, None
    if isinstance(cfg, ExternalConfig):
        resolve_external(root, resolved["external"])
    elif isinstance(cfg, EvidenceConfig):
        resolve_evidence(root, resolved["evidence"])
    elif isinstance(cfg, ScorerTrainConfig):
        resolve_scorer_evidence(root, resolved["scorer"])
    elif isinstance(cfg, BodegaScoreConfig):
        resolve_bodega_sources(root, resolved, seed)
    elif isinstance(cfg, BenchConfig):
        c1, where = resolved["bench"]["c1"], "bench.c1"
    elif isinstance(cfg, RunConfig) and resolved["components"]["c1"]["kind"] == MDEBERTA_KIND:
        c1, where = resolved["components"]["c1"], "components.c1"
    if isinstance(cfg, RunConfig) and resolved["components"]["c2"]["kind"] == REAL_C2_KIND:
        resolve_real_c2(root, resolved["components"]["c2"])
        resolve_real_c2_scorer(root, resolved["components"]["c2"], seed)
    if c1 is not None:
        
        ref = resolve_checkpoint(root, c1["trained_by"], seed)
        if c1.get("checkpoint") not in (None, ref):
            raise ValueError(f"{where}.checkpoint {c1['checkpoint']} differs from the sealed run {ref}; leave it unset")
        c1["checkpoint"] = ref
    if resolved["data"]["dataset"] != "fixture":
        
        # a run can never carry a data_version that does not cover its inputs.
        from aac.data.manifest import data_paths, verify_entries  # noqa: PLC0415 (aac.data sits above aac.run)

        verify_entries(root, resolved["data"]["manifest"], data_paths(resolved["data"]))
    dv = hashing.data_version(root / resolved["data"]["manifest"])
    ch = hashing.config_hash(resolved, git_sha=sha, lock_sha256=lock_sha, data_version=dv)
    resolved["data_version"] = dv
    rid = hashing.run_id(resolved["regime"], ch, seed)
    block = _MV_BLOCK.get(type(cfg))
    if block is not None:
        parts = {block: hashing.sha256_bytes(hashing.canonical(resolved[block]))[:16]}
    else:
        parts = {k: hashing.sha256_bytes(hashing.canonical(v))[:16] for k, v in resolved["components"].items()}
    mv = hashing.model_version({**parts, "git": sha[:7]})
    return {"cfg": cfg, "resolved": resolved, "git_sha": sha, "lock_sha256": lock_sha, "data_version": dv,
            "config_hash": ch, "run_id": rid, "model_version": mv}
