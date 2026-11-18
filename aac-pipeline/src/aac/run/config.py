"""Experiment config: YAML on disk -> pydantic v2 models (extra="forbid", no defaults
for required fields) -> resolved dict (the thing that is hashed).

A regime file may list `extends: [relative paths]`; the files are deep-merged in
order, the regime file last. `?` placeholders fail validation (wrong type)."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Annotated, Literal, Union

import yaml
from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator

from aac.run.env import env_block

_FORBID = ConfigDict(extra="forbid", populate_by_name=True)


class StubC1Cfg(BaseModel):
    """The Phase 1 stub encoder (any kind other than mdeberta-v3-base)."""

    model_config = _FORBID
    enabled: bool
    kind: str
    num_labels: int
    concentration: float


class CheckpointRef(BaseModel):
    """The sealed C1 training run whose checkpoint a run loads; filled by aac.run.identity, never by hand."""

    model_config = _FORBID
    run_id: str = Field(pattern=r"^C1-[0-9a-f]{10}-s\d+$")
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class MdebertaC1Cfg(BaseModel):
    """The real C1 : mDeBERTa-v3-base fine-tuned by the C1 run named `trained_by` at the same seed."""

    model_config = _FORBID
    enabled: bool
    kind: Literal["mdeberta-v3-base"]
    num_labels: int
    hf_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(gt=0)
    trained_by: str
    checkpoint: CheckpointRef | None = None


MDEBERTA_KIND = "mdeberta-v3-base"


def _c1_kind(v) -> str:
    kind = v.get("kind") if isinstance(v, dict) else getattr(v, "kind", None)
    return "mdeberta" if kind == MDEBERTA_KIND else "stub"


C1Cfg = Annotated[Union[Annotated[StubC1Cfg, Tag("stub")], Annotated[MdebertaC1Cfg, Tag("mdeberta")]],
                  Discriminator(_c1_kind)]


class C2Cfg(BaseModel):
    model_config = _FORBID
    enabled: bool
    kind: str
    provider: str
    k_retrieve: int
    relevance_floor: float


RUN_ID_PATTERN = r"^[A-Z0-9]+-[0-9a-f]{10}-s\d+$"


class RunRef(BaseModel):
    """A sealed run another run reads : its run id and its SEALED digest (aac.run.seal.verify_sealed).
 Filled by aac.run.identity from a `*_from` run name, never typed in the YAML, so both enter config_hash."""

    model_config = _FORBID
    run_id: str = Field(pattern=RUN_ID_PATTERN)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DenseCfg(BaseModel):
    """BGE-M3 dense stage : own CLS pooling of AutoModel, L2-normalized (aac.components.c2_retrieval).
 `precision: bfloat16` means bf16 autocast on CUDA only; weights stay float32 and CPU computes in float32."""

    model_config = _FORBID
    hf_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    pooling: Literal["cls"]
    normalize: Literal[True]
    max_seq_length: int = Field(gt=0)
    precision: Literal["bfloat16", "float32"]
    batch_size: int = Field(gt=0)


class RerankCfg(BaseModel):
    """bge-reranker-v2-m3 cross-encoder on (claim, passage) pairs, sigmoid scores in [0, 1] ."""

    model_config = _FORBID
    hf_id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    activation: Literal["sigmoid"]


REAL_C2_KIND = "bge-m3-rerank-entail"


class RealC2Cfg(BaseModel):
    """The real C2 (config; Plans 03-06/03-07 implement the scorer and inference): frozen evidence from the
 EV run `evidence_from` (resolved by aac.run.identity into `evidence`, a RunRef; its dense and reranker blocks must
 equal this block's), list `variant`, an entailment scorer trained by `scorer_trained_by` at `scorer_seed`
 (null = the run seed; aac.run.identity fills `scorer` with the C2S run id and its checkpoint sha256) and the
 relevance floor rule."""

    model_config = _FORBID
    enabled: bool
    kind: Literal["bge-m3-rerank-entail"]
    provider: Literal["averitec_store", "supplied", "pooled", "goldqa"]   # goldqa: addition C oracle 
    evidence_from: str | None
    evidence: RunRef | None = None
    variant: Literal["primary", "sensitivity-b"]
    dense: DenseCfg
    reranker: RerankCfg
    k_dense: int = Field(gt=0)
    k: int = Field(gt=0)
    scorer_trained_by: str
    scorer_seed: int | None
    scorer: RunRef | None = None
    floor_rule: Literal["p10-top1-rerank-val_cal"]

    @model_validator(mode="after")
    def _fields(self) -> "RealC2Cfg":
        if self.k > self.k_dense:
            raise ValueError(f"components.c2.k ({self.k}) must not exceed k_dense ({self.k_dense})")
        if self.provider in ("averitec_store", "pooled") and not self.evidence_from:
            raise ValueError(f"components.c2.provider {self.provider} reads an EV run: set evidence_from")
        return self


def _c2_kind(v) -> str:
    kind = v.get("kind") if isinstance(v, dict) else getattr(v, "kind", None)
    return "real" if kind == REAL_C2_KIND else "stub"


C2Choice = Annotated[Union[Annotated[C2Cfg, Tag("stub")], Annotated[RealC2Cfg, Tag("real")]], Discriminator(_c2_kind)]


class C3Cfg(BaseModel):
    model_config = _FORBID
    gate: str
    fit_on: str
    features: list[str]
    weights: list[float]
    bias: float


class C4Cfg(BaseModel):
    model_config = _FORBID
    enabled: bool
    methods: list[Literal["temperature", "isotonic"]]
    select_by: str
    bins: int
    fit_on: str
    temperature: float


class C5Cfg(BaseModel):
    model_config = _FORBID
    enabled: bool
    coverage_target: float
    report_coverages: list[float]
    fit_on: str
    conflict_floor: float



# configs validate into the old models and dump the same bytes) ------------------------------------------------

LOGISTIC_GATE = "logistic-fitted"
FITTED_C4_KIND = "fitted"
FITTED_C5_KIND = "fitted-gate"
REAL_C3_FEATURES = ("encoder_probs", "support_agg", "refute_agg", "top_rerank_score", "n_evidence",
                    "evidence_coverage")


class LogisticC3Cfg(BaseModel):
    """The fitted C3 gate (aac.components.c3_fusion.LogisticGate): StandardScaler then LogisticRegression on
 the design rows C3 collects in the logged val_fuse pass. `fitted_from` names a sealed R1 full run (same seed) whose
 fusion.json, calibration.json, gate.json and evidence-floor.json this run loads instead of fitting (R2 full)."""

    model_config = _FORBID
    gate: Literal["logistic-fitted"]
    fit_on: Literal["val_fuse"]
    features: list[str]
    transform: Literal["logratio-encoder-v1"]
    standardise: Literal[True]
    C: float = Field(gt=0)
    solver: Literal["lbfgs"]
    max_iter: int = Field(gt=0)
    k: int = Field(gt=0)
    fitted_from: str | None = None

    @field_validator("features")
    @classmethod
    def _features(cls, v: list[str]) -> list[str]:
        if tuple(v) != REAL_C3_FEATURES:
            raise ValueError(f"components.c3.features must be {list(REAL_C3_FEATURES)}")
        return v


class FittedC4Cfg(BaseModel):
    """The fitted C4 (aac.components.c4_calibration.FittedCalibration): own bounded-NLL temperature and
 isotonic, chosen by out-of-fold ECE over 2 index-parity folds of val_cal."""

    model_config = _FORBID
    enabled: bool
    kind: Literal["fitted"]
    methods: list[Literal["temperature", "isotonic"]] = Field(min_length=1)
    select_by: Literal["oof_ece"]
    bins: int = Field(gt=0)
    fit_on: Literal["val_cal"]
    temperature_bounds: list[float] = Field(min_length=2, max_length=2)
    xatol: float = Field(gt=0)
    folds: Literal[2]

    @model_validator(mode="after")
    def _bounds(self) -> "FittedC4Cfg":
        lo, hi = self.temperature_bounds
        if not 0.0 < lo < hi:
            raise ValueError("components.c4.temperature_bounds must satisfy 0 < lo < hi")
        return self


class GateC5Cfg(BaseModel):
    """The fitted C5 (aac.components.c5_abstention.FittedGate): threshold pair on val_cal with the
 raw-confidence tie-break; `pause: true` defers every item (Art. 14(4)(e) stop control)."""

    model_config = _FORBID
    enabled: bool
    kind: Literal["fitted-gate"]
    coverage_target: float = Field(ge=0.0, le=1.0)
    report_coverages: list[float] = Field(min_length=1)
    fit_on: Literal["val_cal"]
    conflict_floor: float
    tie_break: Literal["raw-confidence"]
    pause: bool


def _field_tag(key: str, value: str, tag: str):
    def pick(v) -> str:
        got = v.get(key) if isinstance(v, dict) else getattr(v, key, None)
        return tag if got == value else "stub"
    return pick


C3Choice = Annotated[Union[Annotated[C3Cfg, Tag("stub")], Annotated[LogisticC3Cfg, Tag("logistic")]],
                     Discriminator(_field_tag("gate", LOGISTIC_GATE, "logistic"))]
C4Choice = Annotated[Union[Annotated[C4Cfg, Tag("stub")], Annotated[FittedC4Cfg, Tag("fitted")]],
                     Discriminator(_field_tag("kind", FITTED_C4_KIND, "fitted"))]
C5Choice = Annotated[Union[Annotated[C5Cfg, Tag("stub")], Annotated[GateC5Cfg, Tag("fitted")]],
                     Discriminator(_field_tag("kind", FITTED_C5_KIND, "fitted"))]


class Components(BaseModel):
    model_config = _FORBID
    c1: C1Cfg
    c2: C2Choice   # kind bge-m3-rerank-entail -> RealC2Cfg; any other kind -> the stub C2Cfg (same dump as before)
    c3: C3Choice   # gate logistic-fitted -> LogisticC3Cfg; any other gate -> the stub C3Cfg (same dump as before)
    c4: C4Choice   # kind fitted -> FittedC4Cfg; no kind -> the stub C4Cfg (same dump as before)
    c5: C5Choice   # kind fitted-gate -> GateC5Cfg; no kind -> the stub C5Cfg (same dump as before)


class DataCfg(BaseModel):
    """`dataset` is hashed; the path-valued keys are excluded from config_hash and
 covered by data_version (the hash manifest named by `manifest`)."""

    model_config = _FORBID
    dataset: Literal["fixture", "averitec", "incrediblae"]   # incrediblae: external BODEGA baselines 
    manifest: str
    split_manifest: str
    claims: str
    labels: str
    # Optional since : a training config or a real content-only run has no evidence file.
    # Every stub config sets it, so stub dumps (hence config_hash and the goldens) do not change.
    evidence: str | None = None
    # Evaluation targets of the run (BUILD-BRIEF section 2): `averitec_test` = the test partition of
    # `split_manifest` (new configs use it); `averitec_chrono` = the legacy label of the same partition,
    
    # Empty = the regime's default.
    targets: list[Literal["averitec_test", "averitec_chrono", "fc", "rd"]] = Field(default_factory=list)
    fc_items: str | None = None       # path (excluded from config_hash; content is in data_version)
    rd_items: str | None = None
    attack_subset: str | None = None  # R3: ids drawn from the chronological test split


class WilcoxonFloorCfg(BaseModel):
    model_config = _FORBID
    n: int
    p: float
    rule: str


class StatsCfg(BaseModel):
    """The frozen statistics block of configs/stats.yaml : nothing here is chosen in code."""

    model_config = _FORBID
    bootstrap: int
    level: float
    unit: Literal["item"]
    test_family: str
    primary: list[str]
    secondary: list[str]
    descriptive: list[str]
    wilcoxon_floor: WilcoxonFloorCfg
    correction: Literal["holm"]
    families: dict[str, list[str]]


class EnvCfg(BaseModel):
    model_config = _FORBID
    lock_sha256: str
    python: str
    torch: str
    transformers: str
    numpy: str


class CodeCfg(BaseModel):
    model_config = _FORBID
    git_sha: str


ATTACKS = ("bert_attack", "textfooler", "deepwordbug", "pwws", "beamattack")
SUBSETS = ("averitec_chrono", "fc", "rd")


class AttackVictimsCfg(BaseModel):
    """: victim A (content_only) on all three subsets; victim B (full, un-gated C1->C4) on the AVeriTeC subset."""

    model_config = _FORBID
    content_only: list[Literal["averitec_chrono", "fc", "rd"]]
    full: list[Literal["averitec_chrono", "fc", "rd"]]


class AttackCfg(BaseModel):
    model_config = _FORBID
    attacks: list[Literal["bert_attack", "textfooler", "deepwordbug", "pwws", "beamattack"]]
    victims: AttackVictimsCfg
    budget: int            # attack-phase queries per item, counted by aac.attacks.budget.QueryBudget 
    mode: Literal["targeted"]
    scope: Literal["claim"]
    adversary: Literal["stub"]
    victim_source: str     # name of the sealed R1 run (same seed) whose fitted C4/C5 the victim uses


OPENATTACK_ADVERSARY = "openattack"


class OpenAttackCfg(BaseModel):
    """R3 against the real adversary : the OpenAttack runner (env-attack) attacks the victim server
 (aac.serve.victim_server) over the AF_UNIX socket . A separate model, selected by `adversary: openattack`,
 so a stub config validates into AttackCfg and keeps its bytes and config_hash. regimes/r3.py still refuses it:
 only the server runs real attacks."""

    model_config = _FORBID
    attacks: list[Literal["bert_attack", "textfooler", "deepwordbug", "pwws", "beamattack"]] = Field(min_length=1)
    victims: AttackVictimsCfg
    budget: int = Field(gt=0)   # attack-phase queries per item: server QueryBudget + AttackEval(invoke_limit) 
    mode: Literal["targeted"]
    scope: Literal["claim"]
    adversary: Literal["openattack"]
    victim_source: str          # name of the sealed R1 full run (same seed) whose fitted C3/C4/C5 the victim loads
    item_limit: Annotated[int, Field(gt=0)] | None          # required key; null = every positive item of the subset
    runner_device: Literal["cpu", "cuda"]                   # device of the OpenAttack runner (its MLM / encoders)
    session_timeout_s: Annotated[int, Field(gt=0)] | None   # required key; null = no session timeout

    @field_validator("victims")
    @classmethod
    def _victim_b_averitec_only(cls, v: AttackVictimsCfg) -> AttackVictimsCfg:
        """ (pre-registered specification
 only components.c2.provider, so a victim-B FC or RD cell would fault the server on its first query. A
 validator changes no dump, so no sealed config_hash moves."""
        if list(v.full) not in ([], ["averitec_chrono"]):
            raise ValueError(f"attack.victims.full must be [] or [averitec_chrono] (victim B is AVeriTeC-only), got {list(v.full)}")
        return v


GATED_MASKED_VIEW = "gated-masked"


class OpenAttackGatedCfg(OpenAttackCfg):
    """Addition A (gap review item 3, methodological note): the gate-aware attack. Every OpenAttackCfg
 field plus `victim_view: gated-masked`: the victim server answers victim B's clean and attack queries through the
 GATED decision function with the pre-registered specification
 deferral reads as still Refuted and the search continues; success needs an auto-decided wrong final and no post-hoc
 query runs. A separate model chosen by the presence of the `victim_view` key, so every OpenAttackCfg block (the
 pilot, the sealed R3 runs) validates into OpenAttackCfg and keeps its bytes and config_hash (operational constraint).
 A score-visible view is not built (analyst default NO; it would need a deviation row)."""

    victim_view: Literal["gated-masked"]

    @field_validator("victims")
    @classmethod
    def _victim_b_only(cls, v: AttackVictimsCfg) -> AttackVictimsCfg:
        """The gated view exists for victim B on the AVeriTeC subset only (victim A has no gate)."""
        if list(v.content_only) != [] or list(v.full) != ["averitec_chrono"]:
            raise ValueError("attack.victim_view gated-masked needs victims {content_only: [], full: [averitec_chrono]}, "
                             f"got content_only={list(v.content_only)} full={list(v.full)}")
        return v


def _attack_kind(v) -> str:
    """stub / openattack / openattack-gated: the gated variant only when the block carries a `victim_view` key."""
    if isinstance(v, dict):
        adversary, has_view = v.get("adversary"), "victim_view" in v
    else:
        adversary, has_view = getattr(v, "adversary", None), getattr(v, "victim_view", None) is not None
    if adversary != OPENATTACK_ADVERSARY:
        return "stub"
    return "openattack-gated" if has_view else "openattack"


AttackChoice = Annotated[Union[Annotated[AttackCfg, Tag("stub")], Annotated[OpenAttackCfg, Tag("openattack")],
                               Annotated[OpenAttackGatedCfg, Tag("openattack-gated")]],
                         Discriminator(_attack_kind)]


class ReplayCfg(BaseModel):
    """replay rule: victim B's final adversarial examples re-run through an ablated config."""

    model_config = _FORBID
    source: str            # name of the sealed R3 run
    seed: int


STORE_REPLAY_KIND = "private-store"


class StoreReplayCfg(BaseModel):
    """The real R4 replay (pre-registered specification
 in `sources` (at `seed`), read from the private store of the host that made them (aac.attacks.private_store), each
 claim_hash asserted against the store and the R3 run's attack-adversarial.jsonl, re-run through this run's system.
 A separate model chosen by `kind: private-store`, so a stub `replay: {source, seed}` block validates into ReplayCfg
 and keeps its bytes and config_hash (operational constraint). `victim` selects the source ledger rows (B = full,
 A = content_only for the transfer and gate-isolation replays); the subset is the AVeriTeC attack subset only."""

    model_config = _FORBID
    kind: Literal["private-store"]
    sources: list[str] = Field(min_length=1)   # names of sealed R3 runs, resolved by name in runs/INDEX.json
    victim: Literal["full", "content_only"]
    subset: Literal["averitec_chrono"]
    seed: int

    @field_validator("sources")
    @classmethod
    def _unique(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("replay.sources names a run twice")
        return v


ReplayChoice = Annotated[Union[Annotated[ReplayCfg, Tag("stub")], Annotated[StoreReplayCfg, Tag("store")]],
                         Discriminator(_field_tag("kind", STORE_REPLAY_KIND, "store"))]


class RunConfig(BaseModel):
    model_config = _FORBID
    schema_: str = Field(alias="schema")
    name: str
    regime: str
    system: Literal[
        "content_only", "full", "ablation-no-retrieval", "ablation-no-calibration",
        "ablation-no-abstention", "ablation-all-removed",
    ]
    seed: int
    batch_size: int
    data: DataCfg
    components: Components
    stats: StatsCfg
    env: EnvCfg
    code: CodeCfg
    attack: AttackChoice | None = None   # adversary openattack -> OpenAttackCfg (+ victim_view -> OpenAttackGatedCfg); else AttackCfg
    replay: ReplayChoice | None = None   # kind private-store -> StoreReplayCfg; else the stub ReplayCfg


HEX40 = r"^[0-9a-f]{40}$"


class TrainModelCfg(BaseModel):
    model_config = _FORBID
    key: str
    hf_id: str
    revision: str = Field(pattern=HEX40)
    num_labels: int
    dtype: Literal["float32"]


class OptimizerCfg(BaseModel):
    model_config = _FORBID
    name: Literal["adamw"]
    lr: float
    weight_decay: float
    betas: list[float] = Field(min_length=2, max_length=2)
    eps: float


class ScheduleCfg(BaseModel):
    model_config = _FORBID
    name: Literal["linear"]
    warmup_fraction: float = Field(ge=0.0, le=1.0)


class PrecisionCfg(BaseModel):
    model_config = _FORBID
    weights: Literal["float32"]
    autocast_cuda: Literal["bfloat16"]
    autocast_cpu: Literal["none"]


class LoaderCfg(BaseModel):
    model_config = _FORBID
    shuffle: bool
    num_workers: int = Field(ge=0)
    seed_key_template: str


class TrainCfg(BaseModel):
    """The C1 recipe .
 Every field is required: a recipe value is never chosen in code."""

    model_config = _FORBID
    model: TrainModelCfg
    max_length: int = Field(gt=0)
    labels: list[str]
    optimizer: OptimizerCfg
    schedule: ScheduleCfg
    grad_clip_norm: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_steps: Annotated[int, Field(gt=0)] | None  # required key; null = run the fixed epochs
    class_weights: Literal["none"]
    precision: PrecisionCfg
    early_stopping: Literal[False]
    checkpoint: Literal["last"]
    train_partition: Literal["train"]
    loader: LoaderCfg
    determinism: Literal["strict", "warn"]

    @field_validator("labels")
    @classmethod
    def _labels_are_the_label_space(cls, v: list[str]) -> list[str]:
        from aac.data.fixtures import LABELS  # noqa: PLC0415 (function-level: aac.run sits below aac.data)

        if tuple(v) != LABELS:
            raise ValueError(f"train.labels must equal aac.data.fixtures.LABELS {list(LABELS)} (index 1 = Refuted)")
        return v


class TrainConfig(BaseModel):
    """A C1 training run (schema `aac-train/1`): seals `C1-<hash10>-s<seed>` holding checkpoint.safetensors."""

    model_config = _FORBID
    schema_: Literal["aac-train/1"] = Field(alias="schema")
    name: str
    regime: Literal["C1"]
    seed: int
    data: DataCfg
    train: TrainCfg
    env: EnvCfg
    code: CodeCfg


SCORER_LABELS = ("SUPPORT", "REFUTE", "NEUTRAL")   # C2 entailment scorer head order ; index 1 = REFUTE


class PairsCfg(BaseModel):
    """Weak-labeled (passage, claim) pairs of the C2 scorer (methodological note
 train-partition claim (Boolean: question + answer + boolean_explanation; other types: the answer text), labeled by
 the claim label through label_map, Unanswerable answers `unanswerable`. Retrieved pairs: the first retrieved_top_k
 re-ranked passages of the claim's primary EV list; a passage whose SL row has url_eq_answer 1 takes the claim label
 through label_map, every other passage NEUTRAL. Claims labeled in exclude_claim_labels give no pair at all."""

    model_config = _FORBID
    gold: bool
    retrieved_top_k: int = Field(ge=0)
    label_map: dict[str, Literal["SUPPORT", "REFUTE", "NEUTRAL"]]
    exclude_claim_labels: list[str]
    unanswerable: Literal["NEUTRAL"]
    gold_text: Literal["answer; boolean: question+answer+explanation"]
    retrieved_label: Literal["url-match-claim-label-else-neutral"]
    order: Literal["passage-first"]
    truncation: Literal["only_first"]

    @model_validator(mode="after")
    def _claim_labels(self) -> "PairsCfg":
        from aac.data.fixtures import LABELS  # noqa: PLC0415 (function-level: aac.run sits below aac.data)

        mapped, excluded = set(self.label_map), set(self.exclude_claim_labels)
        if mapped & excluded:
            raise ValueError(f"scorer.pairs: {sorted(mapped & excluded)} both mapped and excluded")
        if mapped | excluded != set(LABELS) or len(self.exclude_claim_labels) != len(excluded):
            raise ValueError(f"scorer.pairs: label_map keys plus exclude_claim_labels must be the claim labels {list(LABELS)} "
                             "exactly once each")
        if not self.gold and self.retrieved_top_k == 0:
            raise ValueError("scorer.pairs: gold false and retrieved_top_k 0 give no pair")
        return self


class ScorerTrainCfg(BaseModel):
    """The C2 scorer recipe : the C1 recipe fields (TrainCfg) with the scorer labels, the EV run
 `evidence_from` (resolved by aac.run.identity into `evidence`, a RunRef, never typed) and the pair rules."""

    model_config = _FORBID
    model: TrainModelCfg
    max_length: int = Field(gt=0)
    labels: list[str]
    optimizer: OptimizerCfg
    schedule: ScheduleCfg
    grad_clip_norm: float = Field(gt=0.0)
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    max_steps: Annotated[int, Field(gt=0)] | None  # required key; null = run the fixed epochs
    class_weights: Literal["none"]
    precision: PrecisionCfg
    early_stopping: Literal[False]
    checkpoint: Literal["last"]
    train_partition: Literal["train"]
    loader: LoaderCfg
    determinism: Literal["strict", "warn"]
    evidence_from: str
    evidence: RunRef | None = None
    pairs: PairsCfg

    @model_validator(mode="after")
    def _head(self) -> "ScorerTrainCfg":
        if tuple(self.labels) != SCORER_LABELS:
            raise ValueError(f"scorer.labels must equal {list(SCORER_LABELS)}")
        if self.model.num_labels != len(SCORER_LABELS):
            raise ValueError(f"scorer.model.num_labels must be {len(SCORER_LABELS)}, got {self.model.num_labels}")
        return self


class ScorerTrainConfig(BaseModel):
    """A C2 scorer training run (schema `aac-train-scorer/1`): seals `C2S-<hash10>-s<seed>` holding
 checkpoint.safetensors, model.json, train-log.jsonl and pairs.json (text-free)."""

    model_config = _FORBID
    schema_: Literal["aac-train-scorer/1"] = Field(alias="schema")
    name: str
    regime: Literal["C2S"]
    seed: int
    data: DataCfg
    scorer: ScorerTrainCfg
    env: EnvCfg
    code: CodeCfg


class BenchInputCfg(BaseModel):
    """The benchmark input : the first `n_passages` distinct url2text sentences of the stores of `zip`,
 read in claim order through aac.data.kstore.iter_docs (streamed, never extracted). `zip` is relative to
 data/raw (`zip_base: data_raw`, the real store) or to the project root (`zip_base: project`, the smoke's
 synthetic store); either way a relative path, so no host path enters config_hash."""

    model_config = _FORBID
    kind: Literal["store_sentences"]
    zip: str
    zip_base: Literal["data_raw", "project"]
    n_passages: int = Field(gt=0)
    max_stores: int = Field(gt=0)


class BenchModelCfg(BaseModel):
    model_config = _FORBID
    key: str
    hf_id: str
    revision: str = Field(pattern=HEX40)
    loader: Literal["sentence_transformer", "cross_encoder"]
    max_seq_length: int = Field(gt=0)   


class BenchLatencyCfg(BaseModel):
    model_config = _FORBID
    queries: int = Field(gt=0)
    stages: list[Literal["c1", "bge-m3", "reranker"]] = Field(min_length=1)
    reranker_pairs_per_query: int = Field(gt=0)


class BenchC1Cfg(BaseModel):
    """The sealed C1 training run whose checkpoint the c1 latency stage loads (same seed); `checkpoint` is
 filled by aac.run.identity from `trained_by`, never by hand, so its sha256 enters config_hash."""

    model_config = _FORBID
    trained_by: str
    checkpoint: CheckpointRef | None = None


class BenchProjectionCfg(BaseModel):
    model_config = _FORBID
    n_claims: int = Field(gt=0)
    shortlist_sizes: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)


class BenchCfg(BaseModel):
    model_config = _FORBID
    input: BenchInputCfg
    models: list[BenchModelCfg] = Field(min_length=1)
    precisions: list[Literal["float32", "bfloat16"]] = Field(min_length=1)
    batch_sizes: list[Annotated[int, Field(gt=0)]] = Field(min_length=1)
    latency: BenchLatencyCfg
    c1: BenchC1Cfg
    projection: BenchProjectionCfg

    @field_validator("models")
    @classmethod
    def _one_model_per_loader(cls, v: list[BenchModelCfg]) -> list[BenchModelCfg]:
        keys = [m.key for m in v]
        if len(keys) != len(set(keys)):
            raise ValueError(f"bench.models keys must be unique, got {keys}")
        for loader in ("sentence_transformer", "cross_encoder"):
            if sum(1 for m in v if m.loader == loader) > 1:
                raise ValueError(f"bench.models: at most one {loader} model (the latency stages name one each)")
        return v


class BenchConfig(BaseModel):
    """A benchmark run (schema `aac-bench/1`): seals `BENCH-<hash10>-s<seed>` holding bench.json
 (record_count 0). The input text itself is written only to <out_root>/data/derived/private/."""

    model_config = _FORBID
    schema_: Literal["aac-bench/1"] = Field(alias="schema")
    name: str
    regime: Literal["BENCH"]
    seed: int
    data: DataCfg
    bench: BenchCfg
    env: EnvCfg
    code: CodeCfg


class ExternalCfg(BaseModel):
    """A third-party baseline reproduced unchanged (Table 4.1 rows 1-5).

 `checkpoint_path` is relative to data/raw. `checkpoint_sha256` and `bodega_commit` are filled by
 aac.run.identity (from data/raw/MANIFEST.expected.sha256 or configs/data/expected-extra.sha256, and from the
 BODEGA gitlink at HEAD), never typed in the YAML, so both enter config_hash. There is deliberately no
 query-budget field: the published BODEGA protocol is unbudgeted (pre-registered specification
 addition of R3, not of the reproduction)."""

    model_config = _FORBID
    tool: Literal["bodega_test_victims", "bodega_attack", "averitec_veracity"]
    task: Literal["FC", "RD"] | None
    attack: Literal["BERTattack"] | None
    victim: Literal["BERT"] | None
    targeted: bool
    checkpoint_path: str
    threads: int = Field(gt=0)
    base_model: str                          # HF id of the architecture the checkpoint is loaded into
    base_revision: str = Field(pattern=HEX40)
    checkpoint_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    bodega_commit: Annotated[str, Field(pattern=HEX40)] | None = None
    # [start, stop): 0-based data rows of attack.tsv in file order (the file has no header line; BODEGA parses every
    # line's first field as the label, victims/transformer.py:44-46). Set: the private staging copy of attack.tsv holds
    # only those rows, byte for byte (aac.baselines.bodega.restrict_rows), so attack.py stays unchanged. bodega_attack
    # only; null = every row. Part of config_hash (: the untargeted RD run is chunked
    # because a single attack.py process grows its memory with the items processed).
    row_range: list[Annotated[int, Field(ge=0)]] | None = None

    @model_validator(mode="after")
    def _tool_fields(self) -> "ExternalCfg":
        bodega = self.tool.startswith("bodega_")
        if bodega and (self.task is None or self.victim is None):
            raise ValueError(f"external.tool {self.tool} needs task and victim")
        if self.tool == "bodega_attack" and self.attack is None:
            raise ValueError("external.tool bodega_attack needs attack")
        if self.tool != "bodega_attack" and (self.attack is not None or self.targeted):
            raise ValueError(f"external.tool {self.tool} takes no attack and targeted false")
        if not bodega and (self.task is not None or self.victim is not None):
            raise ValueError(f"external.tool {self.tool} takes no task or victim")
        if Path(self.checkpoint_path).is_absolute() or ".." in Path(self.checkpoint_path).parts:
            raise ValueError("external.checkpoint_path must be relative to data/raw")
        if self.row_range is not None:
            if self.tool != "bodega_attack":
                raise ValueError(f"external.row_range applies to bodega_attack only, not {self.tool}")
            if len(self.row_range) != 2 or not self.row_range[0] < self.row_range[1]:
                raise ValueError(f"external.row_range must be [start, stop) with start < stop, got {self.row_range}")
        return self


class ExternalConfig(BaseModel):
    """An external baseline run (schema `aac-external/1`): seals `BASE-<hash10>-s<seed>` holding result.json and
 items.jsonl, record_count 0 (methodological note)."""

    model_config = _FORBID
    schema_: Literal["aac-external/1"] = Field(alias="schema")
    name: str
    regime: Literal["BASE"]
    seed: int
    data: DataCfg
    external: ExternalCfg
    stats: StatsCfg
    env: EnvCfg
    code: CodeCfg


class StoreSelectCfg(BaseModel):
    """Which stores a shortlist run reads: `limit` = the first N stores in (split rank, idx) order (null = all);
 `include` = claim ids added to that selection (the pilot adds the largest store). Either one set makes the
 run a subset run (`limit: true` in shortlist.json)."""

    model_config = _FORBID
    limit: Annotated[int, Field(gt=0)] | None
    include: list[str]


class ShortlistCfg(BaseModel):
    """The S1 BM25 shortlist . `store_zips` are relative to data/raw (`zip_base: data_raw`,
 the real stores) or to the project root (`zip_base: project`, the tiny world), so no host path enters
 config_hash. `domains` is a project-relative host list (pre-registered specification
 field here: it changes speed, not results, so it must not move config_hash."""

    model_config = _FORBID
    store_zips: list[str] = Field(min_length=1)
    zip_base: Literal["data_raw", "project"]
    domains: str
    top_n: int = Field(gt=0)
    reserve: int = Field(ge=0)
    variant_exclude: list[Literal["url_eq_answer", "url_date_after_claim"]]
    tokenizer: Literal["bm25s-pystemmer-english-v1"]
    dedup: Literal["first-occurrence"]
    stores: StoreSelectCfg

    @field_validator("store_zips", "variant_exclude")
    @classmethod
    def _unique_relative(cls, v: list[str]) -> list[str]:
        if len(v) != len(set(v)):
            raise ValueError(f"duplicate entries: {v}")
        for p in v:
            if Path(p).is_absolute() or ".." in Path(p).parts:
                raise ValueError(f"{p!r} must be a relative path without '..'")
        return v


class ShortlistConfig(BaseModel):
    """A shortlist run (schema `aac-shortlist/1`): seals `SL-<hash10>-s<seed>` holding shortlist.jsonl.gz and
 shortlist.json (record_count 0). Sentence text goes only to <out_root>/data/derived/private/shortlist/."""

    model_config = _FORBID
    schema_: Literal["aac-shortlist/1"] = Field(alias="schema")
    name: str
    regime: Literal["SL"]
    seed: int
    data: DataCfg
    shortlist: ShortlistCfg
    env: EnvCfg
    code: CodeCfg


VARIANTS = ("primary", "sensitivity-b")   # EV list name -> SL list: primary -> "primary", sensitivity-b -> "variant"
PRE_CUTOFF_PARTITIONS = ("train", "val_fuse", "val_cal")


class PooledBundleCfg(BaseModel):
    """The pooled RD evidence corpus (pre-registered specification
 passages of the claims in `partitions` (chronological pre-cutoff partitions only; test is not allowed),
 deduplicated by sentence sha256, with their fp16 vectors."""

    model_config = _FORBID
    enabled: bool
    partitions: list[Literal["train", "val_fuse", "val_cal"]]
    dedup: Literal["sentence-sha256"]

    @model_validator(mode="after")
    def _partitions(self) -> "PooledBundleCfg":
        if len(self.partitions) != len(set(self.partitions)):
            raise ValueError(f"pooled.partitions has duplicates: {self.partitions}")
        if self.enabled and not self.partitions:
            raise ValueError("pooled.enabled needs at least one partition")
        return self


class EvidenceCfg(BaseModel):
    """The S2 evidence stage . `per_claim`: dense top k_dense and re-ranked top k over the
 first n_passages rows of each configured list of the SL run `shortlist_from`. `pooled_rd` (validated here,
 implemented by ): BM25 top bm25_top_n over the pooled bundle of the EV run `pool_from`.
 `shortlist` / `pool` are filled by aac.run.identity (RunRef), never typed."""

    model_config = _FORBID
    mode: Literal["per_claim", "pooled_rd"]
    shortlist_from: str | None = None
    shortlist: RunRef | None = None
    n_passages: Annotated[int, Field(gt=0)] | None = None
    variants: list[Literal["primary", "sensitivity-b"]] = Field(min_length=1)
    pool_from: str | None = None
    pool: RunRef | None = None
    bm25_top_n: Annotated[int, Field(gt=0)] | None = None
    dense: DenseCfg
    reranker: RerankCfg
    k_dense: int = Field(gt=0)
    k: int = Field(gt=0)
    pooled: PooledBundleCfg
    stores: StoreSelectCfg
    determinism: Literal["strict", "warn"]

    @model_validator(mode="after")
    def _mode_fields(self) -> "EvidenceCfg":
        if len(self.variants) != len(set(self.variants)):
            raise ValueError(f"evidence.variants has duplicates: {self.variants}")
        if self.k > self.k_dense:
            raise ValueError(f"evidence.k ({self.k}) must not exceed evidence.k_dense ({self.k_dense})")
        if self.mode == "per_claim":
            if not self.shortlist_from or self.n_passages is None:
                raise ValueError("evidence.mode per_claim needs shortlist_from and n_passages")
            mixed = [f for f in ("pool_from", "pool", "bm25_top_n") if getattr(self, f) is not None]
            if mixed:
                raise ValueError(f"evidence.mode per_claim takes no pooled_rd fields: {mixed}")
            if self.k_dense > self.n_passages:
                raise ValueError(f"evidence.k_dense ({self.k_dense}) must not exceed n_passages ({self.n_passages})")
            if self.pooled.enabled and "primary" not in self.variants:
                raise ValueError("evidence.pooled.enabled needs the primary list in evidence.variants")
        else:
            if not self.pool_from or self.bm25_top_n is None:
                raise ValueError("evidence.mode pooled_rd needs pool_from and bm25_top_n")
            mixed = [f for f in ("shortlist_from", "shortlist", "n_passages") if getattr(self, f) is not None]
            if self.pooled.enabled:
                mixed.append("pooled.enabled")
            if mixed:
                raise ValueError(f"evidence.mode pooled_rd takes no per_claim fields: {mixed}")
        return self


class EvidenceConfig(BaseModel):
    """An evidence run (schema `aac-evidence/1`): seals `EV-<hash10>-s<seed>` holding evidence.jsonl.gz and
 evidence.json (record_count 0, text-free). Passage vectors and pooled texts go only to
 <out_root>/data/derived/private/evidence/<run_id>/. Retrieval is seed-independent: one run serves every seed."""

    model_config = _FORBID
    schema_: Literal["aac-evidence/1"] = Field(alias="schema")
    name: str
    regime: Literal["EV"]
    seed: int
    data: DataCfg
    evidence: EvidenceCfg
    env: EnvCfg
    code: CodeCfg


# (4; methodological note): the BODEGA score as a sealed post-attack CPU step.
BODEGA_COMMIT = "2dda5a6e6d8a291327df79a44767efea8cab556f"         # third_party/BODEGA gitlink (metrics/BODEGAScore.py)
BLEURT_REVISION = "e22b9eb071dcb939da9fff08b20a7e746c72a830"       # lucadiliello/BLEURT-20 (HF), BODEGA's semantic scorer
BLEURT_PYTORCH_COMMIT = "279ca1bb4106bde5a89f0f82723197e23d8446cb"  # lucadiliello/bleurt-pytorch (env-attack)
LAMBO_COMMIT = "c699c6d07fc4b75d2baf758181c4e796220df238"          # LAMBO segmenter (RD sentence alignment)
# subset -> BODEGA task id: BODEGAScore.segment_sentences knows only PR2 / FC / RD / HN / C19 (any other id returns )
BODEGA_SHIM = {"averitec_chrono": "PR2", "fc": "FC", "rd": "RD"}


class BodegaCfg(BaseModel):
    """How BODEGA's own BODEGAScore is run: the pinned code and models (hashed into config_hash), BLEURT as
 the semantic scorer (attack.py:151), sentence alignment for RD only (PR2 and claim-only FC are one segment), and
 the CPU thread count (fixed, so two runs on one host give the same floats)."""

    model_config = _FORBID
    commit: Literal["2dda5a6e6d8a291327df79a44767efea8cab556f"]
    semantic_scorer: Literal["BLEURT"]
    bleurt_revision: Literal["e22b9eb071dcb939da9fff08b20a7e746c72a830"]
    bleurt_pytorch_commit: Literal["279ca1bb4106bde5a89f0f82723197e23d8446cb"]
    lambo_commit: Literal["c699c6d07fc4b75d2baf758181c4e796220df238"]
    align_rd: Literal[True]
    device: Literal["cpu"]
    threads: int = Field(gt=0)


class BodegaScoreConfig(BaseModel):
    """A BODEGA scoring run (schema `aac-bodega/1`, regime BS): seals `BS-<hash10>-s<seed>` holding
 bodega-scores.jsonl (one text-free row per attacked ledger row of the source R3 runs) and aggregate.json,
 record_count 0. `sources` names sealed R3 runs; aac.run.identity resolves each at the run seed into
 `source_runs` [{name, run_id, sealed_sha256}] BEFORE config_hash (leave it unset), so a re-sealed source moves the
 run id. A new schema, so no existing config dump changes (the identity guard stays green)."""

    model_config = _FORBID
    schema_: Literal["aac-bodega/1"] = Field(alias="schema")
    name: str
    regime: Literal["BS"]
    seed: int
    data: DataCfg
    sources: list[str] = Field(min_length=1)
    source_runs: list[dict] | None = None
    shim: dict[str, str]
    bodega: BodegaCfg
    stats: StatsCfg
    env: EnvCfg
    code: CodeCfg

    @field_validator("sources")
    @classmethod
    def _unique_sources(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError("sources names a run twice")
        return v

    @field_validator("shim")
    @classmethod
    def _shim_is_the_task_map(cls, v: dict[str, str]) -> dict[str, str]:
        if v != BODEGA_SHIM:
            raise ValueError(f"shim must be exactly {BODEGA_SHIM} (04-RESEARCH Pattern 4), got {v}")
        return v

    @field_validator("source_runs")
    @classmethod
    def _source_runs_resolved(cls, v):
        if v is not None:
            raise ValueError("source_runs is resolved from sources by aac.run.identity; leave it unset")
        return v


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, _Replace):
            out[k] = copy.deepcopy(dict(v))
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


class _Replace(dict):
    """A mapping tagged `!replace` in YAML: it REPLACES the inherited mapping instead of being deep-merged
 into it (used by real configs whose c1 block must not inherit base.yaml's stub fields)."""


class _Loader(yaml.SafeLoader):
    pass


def _construct_replace(loader: yaml.SafeLoader, node: yaml.Node) -> _Replace:
    if not isinstance(node, yaml.MappingNode):
        raise yaml.constructor.ConstructorError(None, None, "!replace applies to a mapping", node.start_mark)
    return _Replace(loader.construct_mapping(node, deep=True))


_Loader.add_constructor("!replace", _construct_replace)


def _plain(obj):
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_plain(v) for v in obj]
    return obj


def load_raw(path: Path) -> dict:
    path = Path(path)
    with open(path, encoding="utf-8") as fh:
        data = yaml.load(fh, Loader=_Loader) or {}  # noqa: S506 (SafeLoader subclass + one mapping tag)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    merged: dict = {}
    for rel in data.pop("extends", []) or []:
        merged = _deep_merge(merged, load_raw(path.parent / rel))
    return _plain(_deep_merge(merged, data))


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(load_raw(path))


# `schema` key -> config model. Later run kinds (e.g. external baselines) register here.
SCHEMAS: dict[str, type[BaseModel]] = {"aac-config/1": RunConfig, "aac-train/1": TrainConfig, "aac-bench/1": BenchConfig,
                                        "aac-external/1": ExternalConfig, "aac-shortlist/1": ShortlistConfig,
                                        "aac-evidence/1": EvidenceConfig, "aac-train-scorer/1": ScorerTrainConfig,
                                        "aac-bodega/1": BodegaScoreConfig}


def load_any(path: Path) -> BaseModel:
    """Validate a config file with the model its `schema` key names (SCHEMAS)."""
    raw = load_raw(path)
    key = raw.get("schema")
    if key not in SCHEMAS:
        raise ValueError(f"{path}: unknown config schema {key!r} (have {sorted(SCHEMAS)})")
    return SCHEMAS[key].model_validate(raw)


def resolve(cfg: BaseModel, *, git_sha: str, lock_sha256: str, seed: int | None = None,
            lock_path: Path | None = None) -> dict:
    """Resolved dict: AUTO env/code fields filled; `seed` overridden by the CLI value."""
    resolved = cfg.model_dump(by_alias=True)
    if seed is not None:
        resolved["seed"] = int(seed)
    env = env_block(lock_path) if lock_path is not None else {}
    env["lock_sha256"] = lock_sha256
    for key in ("python", "torch", "transformers", "numpy"):
        if resolved["env"].get(key, "AUTO") == "AUTO":
            resolved["env"][key] = env.get(key, "absent")
    resolved["env"]["lock_sha256"] = lock_sha256
    if resolved["code"].get("git_sha", "AUTO") == "AUTO":
        resolved["code"]["git_sha"] = git_sha
    return resolved
