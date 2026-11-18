"""Article 14 analyst-interface mock-up: evidence-pack/analyst-ui-mockup.html (Table D.3 row 4;
draft-v0.5.md:279, brief §1 item 5, ARCHITECTURE §8 and §12).

The one deferred case shown is a versioned input, configs/analyst/demo-case.json, written by rule
(`aac pack demo-case`) from the sealed R1 chronological full-pipeline seed-0 run: among its test
records with abstention_flag true, the one with the lowest calibrated confidence max(p, 1 - p),
tie-break claim_id. It carries the fixture claim text (the only claim text allowed into the pack),
the scored passages as hashes with scores, the calibrated probability, the deferral reason and the
record_id. A real namespace (datasheet profile kind `real`) uses
configs/analyst/demo-case.real.json, chosen by the same rule: claim_text null and claim_text_shown
false, claim_hash kept, evidence as ids with scores; the page shows CLAIM_PLACEHOLDER in the claim
slot and the evidence ids with their scores; an RD-derived source run is refused. The page is static: no clock, no
network request, no external resource. Its three
buttons compose the append-only analyst-events payload {record_id, analyst_action,
analyst_action_timestamp} in a visible text box (the timestamp is filled by the Phase 5 CLI; the
sealed record's two analyst slots stay null), and a pause control shows the threshold = infinity
setting. The build refuses a demo case that no longer matches the sealed run, and a page carrying
an `@handle`.

: the template is chosen by namespace kind (TEMPLATES). The fixture page keeps the stub template byte for
byte; the real page uses configs/pack/analyst-ui.v2.template.html: evidence ids with their re-rank scores and a note
that passage text stays in the knowledge-store archives, the sealed threshold pair at the coverage target, the reason
definition, and a pause control that shows the sealed gate.json `pause` flag and `pause_rule` (its button composes the
pause payload {c5_pause: true, ...}). The case is re-derived by rule at every build; a gate.json that disagrees with
the case (threshold, coverage target, reason) is refused.

analyst item O1: configs/pack/analyst-mock.yaml (closed keys SWITCH_KEYS) holds `show_claim_text`
(default false: placeholder). With true, the page shows the claim text only when `approved_by`, `approval_quote` and
`attribution` are filled, the source is AVeriTeC and not RD-derived, and the text, loaded at render time from the
hash-pinned raw files, hashes to the record's claim_hash; it is never written to the demo case or a log. Any other
combination raises PackInputError before the page is written."""
from __future__ import annotations

import json
import re
from pathlib import Path

from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.pack.inputs import PackContext, PackInputError, read_json, render, sealed_run, write_text
from aac.run.hashing import claim_hash, dumps_sorted

GENERATED_BY = "aac.pack.analyst_mock"
FILE = "analyst-ui-mockup.html"
DEMO_CASE = "configs/analyst/demo-case.json"  # the fixture (stub) case, kept byte for byte
DEMO_CASES = {"fixture": DEMO_CASE, "real": "configs/analyst/demo-case.real.json"}
# : the page template by namespace kind (the datasheet's pattern); the fixture page stays byte-identical.
TEMPLATES = {"fixture": "analyst-ui.template.html", "real": "analyst-ui.v2.template.html"}
SWITCH = "configs/pack/analyst-mock.yaml"
SWITCH_KEYS = frozenset({"show_claim_text", "approved_by", "approval_quote", "attribution"})
GATE_FILE = "gate.json"
GATE_KEYS = ("threshold_pair", "pause", "pause_rule", "reason_definitions", "coverage_target", "threshold")
PAUSE_EFFECT = "every item is routed to the analyst queue"
PAUSE_ARTICLE = "14(4)(e)"
# : no passage text or passage hash exists on cpu-host; ids and re-rank scores only.
EVIDENCE_NOTE = ("Passage text is held in the AVeriTeC knowledge-store archives, not in this pack: each retrieved "
                 "passage is shown by its evidence id (claim id, document index d, sentence index s) and its "
                 "re-rank score.")
# (default "no"): a real case never carries claim text; the page shows
# claim_hash and this placeholder. Plain text only (the page refuses http:// and https://).
CLAIM_PLACEHOLDER = ("Claim text withheld: the source corpus (AVeriTeC) is licensed CC BY-NC 4.0, so the pack carries "
                     "no claim text; the claim is identified by its claim_hash below.")
RULE = ("R1 chronological full pipeline, seed 0, test split: the deferred record (abstention_flag true) with the "
        "lowest calibrated confidence max(p, 1 - p), tie-break claim_id")
HANDLE_RE = re.compile(r"@\w+")
ACTIONS = ("confirm", "override", "return")


def _compact(obj: dict) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(", ", ": "))


def _confidence(p: float) -> float:
    return max(p, 1.0 - p)


def _records(run_dir: Path) -> list[dict]:
    return sorted((r for s in list_shards(run_dir / SHARD_DIR) for r in read_shard(s)), key=lambda r: r["seq"])


def namespace_kind(root: Path, resolved: dict) -> str:
    """`fixture` or `real`: the datasheet profile kind of the source run's data manifest
 (configs/pack/datasheet-profiles.yaml; the datasheet generator enforces one manifest per namespace)."""
    from aac.pack.datasheet import load_profile

    kind = load_profile(root, resolved.get("data", {}).get("manifest"))["kind"]
    if kind not in DEMO_CASES:
        raise PackInputError(f"unknown datasheet profile kind {kind!r} (known: {sorted(DEMO_CASES)})")
    return kind


def _source_run(root: Path) -> tuple[dict, Path, dict, str]:
    entry, run_dir, resolved = sealed_run(root, regime="R1", name="r1-chrono-full", seed=0)
    kind = namespace_kind(root, resolved)
    data = resolved.get("data", {})
    if kind == "fixture" and data.get("dataset") != "fixture":
        raise PackInputError(f"{entry['run_id']}: dataset {data.get('dataset')!r} is not the fixture but its manifest "
                             "has the fixture profile")
    if kind == "real" and data.get("rd_items"):  
        raise PackInputError(f"{entry['run_id']}: data block names rd_items; an RD-derived source run is refused")
    return entry, run_dir, resolved, kind


def _pick(entry: dict, run_dir: Path) -> dict:
    deferred = [r for r in _records(run_dir) if r["split"] == "test" and r["abstention_flag"]]
    if not deferred:
        raise PackInputError(f"{entry['run_id']}: no deferred test record; the mock-up needs one deferred case")
    return min(deferred, key=lambda r: (_confidence(r["calibrated_probability"]), r["claim_id"]))


def _real_case(entry: dict, resolved: dict, rec: dict) -> dict:
    """A real case by the same rule: claim_hash kept, claim text withheld (null, claim_text_shown false), evidence as
 ids with scores (no passage text or passage hash exists on cpu-host)."""
    if len(rec["evidence_ids"]) != len(rec["evidence_scores"]):
        raise PackInputError(f"{rec['record_id']}: evidence_ids and evidence_scores differ in length")
    return {
        "selection_rule": RULE, "source_run_id": entry["run_id"], "source_config_hash": entry["config_hash"],
        "data_version": resolved["data_version"], "record_id": rec["record_id"], "claim_id": rec["claim_id"],
        "claim_hash": rec["claim_hash"], "claim_text": None, "claim_text_shown": False,
        "evidence": [{"evidence_id": eid, "score": score} for eid, score in zip(rec["evidence_ids"], rec["evidence_scores"])],
        "evidence_coverage": rec["evidence_coverage"], "calibrated_probability": rec["calibrated_probability"],
        "calibration_method": rec["calibration_method"], "calibrated_confidence": round(_confidence(rec["calibrated_probability"]), 6),
        "abstention_reason": rec["abstention_reason"], "threshold": rec["threshold"], "coverage_target": rec["coverage_target"],
        "decision": rec["decision"], "analyst_queue_id": rec["analyst_queue_id"],
    }


def _demo(root: Path) -> tuple[dict, str, str, dict, Path, dict]:
    """(case, versioned-input path, kind, INDEX entry, run dir, resolved config), one seal check."""
    root = Path(root)
    entry, run_dir, resolved, kind = _source_run(root)
    rec = _pick(entry, run_dir)
    case = _real_case(entry, resolved, rec) if kind == "real" else _fixture_case(root, entry, resolved, rec)
    return case, DEMO_CASES[kind], kind, entry, run_dir, resolved


def demo_case(root: Path) -> tuple[dict, str, str]:
    """(case, versioned-input path, kind): the demo case by rule from the sealed run, one seal check."""
    case, rel, kind, *_ = _demo(root)
    return case, rel, kind


def _gate(entry: dict, run_dir: Path, case: dict) -> dict:
    """The sealed gate.json of the source run, checked against the case it explains ."""
    path = Path(run_dir) / GATE_FILE
    if not path.is_file():
        raise PackInputError(f"{entry['run_id']}: no sealed {GATE_FILE}; the pause control and threshold pair read it")
    gate = read_json(path)
    missing = [k for k in GATE_KEYS if k not in gate]
    if missing:
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: keys missing {missing}")
    if not isinstance(gate["pause"], bool):
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: pause is not true or false")
    if not isinstance(gate["pause_rule"], str) or not gate["pause_rule"].strip():
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: pause_rule is empty")
    pair = gate["threshold_pair"]
    if not (isinstance(pair, list) and len(pair) == 2 and all(isinstance(x, (int, float)) for x in pair)):
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: threshold_pair is not a pair of numbers")
    if float(gate["coverage_target"]) != float(case["coverage_target"]):
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: coverage_target differs from the case's")
    if not gate["pause"] and case["threshold"] != pair[0]:
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: the case threshold differs from threshold_pair[0]")
    if case["abstention_reason"] not in gate["reason_definitions"]:
        raise PackInputError(f"{entry['run_id']}/{GATE_FILE}: no definition for reason {case['abstention_reason']!r}")
    return gate


def build_demo_case(root: Path) -> dict:
    """The demo case by rule from the sealed run (used by `aac pack demo-case`)."""
    return demo_case(root)[0]


def _fixture_case(root: Path, entry: dict, resolved: dict, rec: dict) -> dict:
    claims = {c["claim_id"]: c for c in json.loads((root / resolved["data"]["claims"]).read_text(encoding="utf-8"))}
    text = claims[rec["claim_id"]]["text"]
    if claim_hash(text) != rec["claim_hash"]:
        raise PackInputError(f"{rec['claim_id']}: fixture text does not hash to the record's claim_hash")
    if HANDLE_RE.search(text):
        raise PackInputError(f"{rec['claim_id']}: claim text carries a user handle; choose by rule only among handle-free texts")
    passages = {p["id"]: p for p in json.loads((root / resolved["data"]["evidence"]).read_text(encoding="utf-8")).get(rec["claim_id"], [])}
    evidence = []
    for eid, score in zip(rec["evidence_ids"], rec["evidence_scores"]):
        if eid not in passages:
            raise PackInputError(f"{eid}: not in {resolved['data']['evidence']}")
        evidence.append({"evidence_id": eid, "passage_sha256": passages[eid]["text_hash"], "score": score})
    return {
        "selection_rule": RULE, "source_run_id": entry["run_id"], "source_config_hash": entry["config_hash"],
        "data_version": resolved["data_version"], "record_id": rec["record_id"], "claim_id": rec["claim_id"],
        "claim_hash": rec["claim_hash"], "claim_text": text, "evidence": evidence,
        "evidence_coverage": rec["evidence_coverage"], "calibrated_probability": rec["calibrated_probability"],
        "calibration_method": rec["calibration_method"], "calibrated_confidence": round(_confidence(rec["calibrated_probability"]), 6),
        "abstention_reason": rec["abstention_reason"], "threshold": rec["threshold"], "coverage_target": rec["coverage_target"],
        "decision": rec["decision"], "analyst_queue_id": rec["analyst_queue_id"],
    }


def write_demo_case(root: Path) -> Path:
    case, rel, _ = demo_case(root)
    target = Path(root) / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(dumps_sorted(case))
    return target


def load_switch(root: Path) -> dict:
    """the analyst's claim-text switch (closed vocabulary; item O1)."""
    import yaml

    path = Path(root) / SWITCH
    if not path.is_file():
        raise PackInputError(f"missing versioned input {SWITCH} (analyst item O1; default show_claim_text: false)")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise PackInputError(f"{SWITCH}: not a mapping")
    unknown = sorted(set(data) - SWITCH_KEYS)
    missing = sorted(SWITCH_KEYS - set(data))
    if unknown or missing:
        raise PackInputError(f"{SWITCH}: unknown key(s) {unknown}, missing key(s) {missing} (closed set {sorted(SWITCH_KEYS)})")
    if not isinstance(data["show_claim_text"], bool):
        raise PackInputError(f"{SWITCH}: show_claim_text must be true or false")
    return data


def _filled(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _load_claim_text(root: Path, claim_id: str) -> str:
    """The raw AVeriTeC claim text, read at render time from the hash-pinned raw files; returned, never logged."""
    from aac.data.averitec import RAW_REL, check_raw_hashes, load_claims
    from aac.data.verify import load_expected

    raw = Path(root) / RAW_REL
    try:
        check_raw_hashes(raw, load_expected(root))
        claims = load_claims(raw)
    except Exception as exc:  # IngestError, VerifyError, OSError: the text cannot be proven, so it is not shown
        raise PackInputError(f"raw AVeriTeC files under {RAW_REL} cannot be read or fail their pinned hashes "
                             f"({type(exc).__name__})") from None
    if claim_id not in claims:
        raise PackInputError(f"{claim_id}: not in the raw AVeriTeC files")
    return claims[claim_id]["text"]


def _claim_slot(root: Path, case: dict, resolved: dict) -> tuple[str, str | None, bool]:
    """(claim slot text, attribution or None, shown): the placeholder unless the analyst's guarded switch is on."""
    switch = load_switch(root)
    if not switch["show_claim_text"]:
        return CLAIM_PLACEHOLDER, None, False
    for key in ("approved_by", "approval_quote", "attribution"):
        if not _filled(switch[key]):
            raise PackInputError(f"{SWITCH}: show_claim_text is true but {key} is empty; the analyst's recorded "
                                 "approval (approved_by, approval_quote) and the attribution are required")
    data = resolved.get("data", {})
    if data.get("rd_items"):
        raise PackInputError(f"{SWITCH}: the source run names rd_items; RD text is never shown")
    if data.get("dataset") != "averitec" or not str(case["claim_id"]).startswith("averitec/"):
        raise PackInputError(f"{SWITCH}: only AVeriTeC claim text may be shown (dataset {data.get('dataset')!r})")
    text = _load_claim_text(root, case["claim_id"])
    if claim_hash(text) != case["claim_hash"]:
        raise PackInputError(f"{case['claim_id']}: the raw text does not hash to the record's claim_hash; not shown")
    return text, switch["attribution"].strip(), True


def _check_page(html: str) -> None:
    hits = HANDLE_RE.findall(html)
    if hits:
        raise PackInputError(f"{FILE}: user-identifier pattern {sorted(set(hits))[:3]} in the rendered page")
    for banned in ("<script src=", "http://", "https://", "Date("):
        if banned in html:
            raise PackInputError(f"{FILE}: forbidden token {banned!r} (static page: no external resource, no clock)")


def _render_real(root: Path, case: dict, gate: dict, entry: dict, payloads: dict, claim_slot: str,
                 attribution: str | None) -> str:
    pause = _compact({"c5_pause": True, "effect": PAUSE_EFFECT, "rule": gate["pause_rule"], "article": PAUSE_ARTICLE})
    pair = gate["threshold_pair"]
    return render(root, TEMPLATES["real"], html=True, case=case, claim_slot=claim_slot, attribution=attribution,
                  rule=RULE, source_run_id=entry["run_id"], payloads=payloads, pause=pause,
                  pause_flag="true" if gate["pause"] else "false", pause_rule=gate["pause_rule"],
                  probability=f"{case['calibrated_probability']:.6f}", confidence=f"{case['calibrated_confidence']:.6f}",
                  threshold_pair=f"({float(pair[0]):.6f}, {float(pair[1]):.6f})",
                  coverage=f"{round(float(case['coverage_target']) * 100)}",
                  reason_definition=gate["reason_definitions"][case["abstention_reason"]],
                  evidence_coverage="yes" if case["evidence_coverage"] else "no",
                  evidence=[(e["evidence_id"], f"{e['score']:.6f}") for e in case["evidence"]], evidence_note=EVIDENCE_NOTE)


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    expected, rel, kind, entry, run_dir, resolved = _demo(root)
    path = root / rel
    if not path.is_file():
        raise PackInputError(f"missing versioned input {rel}; write it by rule with `aac pack demo-case`")
    case = read_json(path)
    if case != expected:
        diff = sorted(k for k in set(case) | set(expected) if case.get(k) != expected.get(k))
        raise PackInputError(f"{rel} no longer matches the rule applied to the sealed run (fields {diff}); "
                             "regenerate it with `aac pack demo-case` and commit it")
    real = kind == "real"
    if real and (case.get("claim_text") is not None or case.get("claim_text_shown") is not False):
        raise PackInputError(f"{rel}: claim_text_shown is false but claim_text is set; the pack never carries it")
    payloads = {a: _compact({"record_id": case["record_id"], "analyst_action": a, "analyst_action_timestamp": None})
                for a in ACTIONS}
    if real:
        gate = _gate(entry, run_dir, case)
        claim_slot, attribution, shown = _claim_slot(root, case, resolved)
        html = _render_real(root, case, gate, entry, payloads, claim_slot, attribution)
        inputs = (f"{rel} (versioned input, chosen by rule); runs/{entry['run_id']}/{GATE_FILE}; {SWITCH}"
                  + (" (claim text shown: analyst approval recorded; raw AVeriTeC text hash-checked)" if shown else "")
                  + f"; configs/pack/{TEMPLATES['real']}")
    else:
        pause = _compact({"c5_threshold": "infinity", "effect": PAUSE_EFFECT, "article": PAUSE_ARTICLE})
        evidence = [(e["evidence_id"], e["passage_sha256"], f"{e['score']:.3f}") for e in case["evidence"]]
        html = render(root, TEMPLATES["fixture"], html=True, case=case, payloads=payloads, pause=pause,
                      probability=f"{case['calibrated_probability']:.3f}", confidence=f"{case['calibrated_confidence']:.3f}",
                      threshold="none" if case["threshold"] is None else f"{case['threshold']:.3f}",
                      coverage=f"{round(float(case['coverage_target']) * 100)}",
                      evidence_coverage="yes" if case["evidence_coverage"] else "no",
                      evidence=evidence)
        inputs = f"{rel} (versioned input, chosen by rule); configs/pack/{TEMPLATES['fixture']}"
    _check_page(html)
    write_text(pack / FILE, html)  
    ctx.record(FILE, GENERATED_BY, inputs)
    return [FILE]
