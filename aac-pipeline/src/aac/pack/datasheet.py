"""Article 10 datasheets: evidence-pack/datasheet-averitec.md and datasheet-incrediblae.md, plus
evidence-pack/data_version.txt (Table D.3 row 1; draft-v0.5.md:256, brief §1 item 3).

Gebru et al. (2021) sections and audience-structured sections after Pushkarna, Zaldivar and
Kjartansson (2022) are headings of a versioned template under configs/pack/; the known-gaps narrative
under configs/datasheet/ is included verbatim (a versioned input, never edited into the output). Which
template, narrative and statistics files a render uses is chosen by the data manifest through
configs/pack/datasheet-profiles.yaml:

 fixture tests/fixtures/corpus/MANIFEST.sha256 -> datasheet.template.md + known-gaps.v1.md (the Phase 1
 skeleton datasheets, byte for byte)
 real data/derived/MANIFEST.sha256 -> datasheet.v2.template.md + known-gaps.v2.md : the
 Gebru et al. sections plus `Knowledge store` and `Bias and gap audit` Every statistics file a profile names must be listed in that manifest (the datasheet reads only hashed
statistics); a missing key fails the render naming the file and the key. Counts are template fields, never
typed into a template.

`data_version` is never derived from the rendered datasheets: it is aac.run.hashing.data_version over the
data manifest. Two entry points share one render function:

 render_pre_run(root, pack, manifest) no sealed run needed: data_version straight from the manifest. This
 is the Art. 10 artifact rendered before C1 is fine-tuned ;
 the committed data_version.txt opens the C1 training gate.
 build(root, pack, ctx) `aac pack build`: data_version shared by every sealed run and every
 record (sealed_data_version refuses DataVersionMismatch otherwise),
 then the same render, so the post-run pack re-renders byte-identically.

Each file is written through a temp file and os.replace; data_version.txt is written last, so an
interrupted render leaves the training gate closed. Two concurrent renders into one pack are unsupported."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from aac.logging.validate import list_shards
from aac.logging.writer import SHARD_DIR, read_shard
from aac.pack.inputs import PackContext, PackInputError, read_json, render, require_headings
from aac.run.hashing import data_version, sha256_file
from aac.run.paths import runs_dir

GENERATED_BY = "aac.pack.datasheet"
REQUIRED = ["Motivation", "Composition", "Collection process", "Preprocessing", "Uses", "Distribution",
            "Maintenance", "Known gaps"]
KNOWN_GAPS = "configs/datasheet/known-gaps.v1.md"  # the fixture profile's narrative (also read by the model card)
EXPECTED_RAW = "data/raw/MANIFEST.expected.sha256"
PROFILES = "configs/pack/datasheet-profiles.yaml"
DOCS = ("datasheet-averitec.md", "datasheet-incrediblae.md")
DATA_VERSION_FILE = "data_version.txt"


class DataVersionMismatch(RuntimeError):
    """Sealed runs disagree on the data manifest or on the data_version they recorded."""


def sealed_data_version(root: Path, runs: list[dict]) -> tuple[str, str]:
    """(manifest path, data_version) shared by every sealed run and every record in it."""
    root = Path(root)
    manifests = {}
    for e in runs:
        resolved = read_json(runs_dir(root) / e["run_id"] / "config.resolved.json")
        manifests.setdefault(resolved["data"]["manifest"], []).append(e["run_id"])
    if len(manifests) != 1:
        raise DataVersionMismatch(f"sealed runs name {len(manifests)} data manifests: {sorted(manifests)}")
    (manifest, _), = manifests.items()
    dv = data_version(root / manifest)
    for e in runs:
        run_dir = runs_dir(root) / e["run_id"]
        resolved = read_json(run_dir / "config.resolved.json")
        if resolved["data_version"] != dv:
            raise DataVersionMismatch(f"{e['run_id']}: config.resolved.json data_version {resolved['data_version'][:12]} != {dv[:12]} of {manifest}")
        for shard in list_shards(run_dir / SHARD_DIR):
            for rec in read_shard(shard):
                if rec["data_version"] != dv:
                    raise DataVersionMismatch(f"{e['run_id']} {shard.name} seq {rec['seq']}: record data_version "
                                              f"{rec['data_version'][:12]} != {dv[:12]} of {manifest}")
    return manifest, dv


def _manifest_files(root: Path, manifest: str) -> list[tuple[str, str]]:
    out = []
    for line in (Path(root) / manifest).read_text(encoding="ascii").splitlines():
        if line.strip():
            digest, _, rel = line.partition("  ")
            out.append((rel, digest))
    return sorted(out)


def _expected_raw(root: Path, prefix: str) -> list[tuple[str, str]]:
    out = []
    for line in (Path(root) / EXPECTED_RAW).read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            digest, _, rel = line.partition("  ")
            if rel.startswith(prefix):
                out.append((rel, digest))
    return sorted(out)


def _pct(x: float) -> str:
    from aac.reports.fmt import fmt_rate

    return fmt_rate(x)


def _items_stats(path: Path) -> dict:
    """Composition of an InCrediblAE-shaped item file (counts only; no text leaves this function)."""
    items = json.loads(path.read_text(encoding="utf-8"))
    rows = list(items.values()) if isinstance(items, dict) else list(items)
    texts = [r["text"] for r in rows]
    n = len(rows)
    pos = sum(1 for r in rows if int(r["label"]) == 1)
    return {"n_items": n, "n_label_1": pos, "label_1_share": _pct(pos / n) if n else "n/a",
            "n_with_supplied_evidence": sum(1 for r in rows if r.get("evidence")),
            "n_duplicate_texts": n - len(set(texts)), "n_non_ascii_texts": sum(1 for t in texts if not t.isascii())}


def _licences(root: Path) -> dict:
    with open(Path(root) / "configs" / "licences.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["data"]


def _known_gaps(root: Path, rel: str = KNOWN_GAPS) -> str:
    p = Path(root) / rel
    if not p.is_file():
        raise PackInputError(f"missing versioned input {rel}")
    return p.read_text(encoding="utf-8").rstrip("\n")


# -- profiles ---------------------------------------------------------------------------------------

def load_profile(root: Path, manifest: str) -> dict:
    p = Path(root) / PROFILES
    if not p.is_file():
        raise PackInputError(f"missing versioned input {PROFILES}")
    profiles = yaml.safe_load(p.read_text(encoding="utf-8"))["profiles"]
    if manifest not in profiles:
        raise PackInputError(f"no datasheet profile for manifest {manifest} in {PROFILES} (known: {sorted(profiles)})")
    return profiles[manifest]


def _stats(root: Path, manifest: str, profile: dict, listed: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """({key: parsed stats file}, {key: path shown as source}); every file must be listed in the manifest."""
    base = (Path(root) / manifest).parent
    shown_base = manifest.rsplit("/", 1)[0]
    out, shown = {}, {}
    for key, rel in profile["stats"].items():
        if rel not in listed:
            raise PackInputError(f"{manifest} does not list {rel}; the datasheet reads only hashed statistics")
        p = base / rel
        if not p.is_file():
            raise PackInputError(f"{shown_base}/{rel} is listed in {manifest} but missing")
        if sha256_file(p) != listed[rel]:  
            raise PackInputError(f"{shown_base}/{rel}: sha256 differs from the hash listed in {manifest}; "
                                 "run `make data-manifest` before rendering the datasheet")
        out[key] = p
        shown[key] = f"{shown_base}/{rel}"
    return out, shown


class _Stats:
    """Checked access to one statistics file: a missing key raises PackInputError naming file and key."""

    def __init__(self, data: dict, shown: str) -> None:
        self.data, self.shown = data, shown

    def __call__(self, dotted: str):
        cur = self.data
        for part in dotted.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise PackInputError(f"{self.shown}: required key {dotted!r} missing")
            cur = cur[part]
        return cur


def _n(x) -> str:
    """An integer count with thousands separators (a non-integer is refused, never rounded)."""
    if isinstance(x, bool) or not isinstance(x, int):
        raise PackInputError(f"expected an integer count, got {x!r}")
    return f"{x:,}"


def _share(num: int, den: int) -> str:
    """num/den in percent, 2 d.p.; `n/a` for an empty denominator."""
    return f"{100.0 * num / den:.2f}" if den else "n/a"


def _frac_pct(s: str) -> str:
    """A `.6f` share string of ks-stats.json shown in percent, 2 d.p."""
    return f"{100.0 * float(s):.2f}"


def _no_at(s: str) -> str:
    """ is literally a test failing on `@\\w+`: a contact address is written with [at]."""
    return s.replace("@", " [at] ")




def _fixture_docs(root: Path, manifest: str, profile: dict, listed: dict[str, str]) -> dict[str, dict]:
    files, shown = _stats(root, manifest, profile, listed)
    av = _Stats(read_json(files["averitec"]), shown["averitec"])
    lic = _licences(root)
    labels = sorted(av("label_counts"))
    av_rows = [[lab, str(av("label_counts")[lab]), _pct(av("label_shares")[lab])] for lab in labels]
    docs = {
        "datasheet-averitec.md": dict(
            dataset_key="averitec", dataset_title="AVeriTeC (fixture stand-in in the Phase 1 skeleton)",
            licence=lic["averitec"]["licence"], licence_source=lic["averitec"]["source"],
            expected_raw=_expected_raw(root, "averitec/"),
            composition={"Claims": av("n_claims"), "Dated claims": av("n_dated"), "Undated claims (excluded from both R1 splits)": av("n_undated"),
                         "Earliest claim date": av("date_min"), "Latest claim date": av("date_max"),
                         "Duplicate texts (claims minus distinct texts)": av("n_duplicate_texts"),
                         "Texts with non-ASCII characters": av("n_non_ascii_texts")},
            label_rows=av_rows, stats_source=shown["averitec"]),
    }
    inc_comp, inc_sources = {}, []
    for key, skey in (("FC", "fc_items"), ("RD", "rd_items")):
        s = _items_stats(files[skey])
        inc_sources.append(shown[skey])
        inc_comp[f"{key} items"] = s["n_items"]
        inc_comp[f"{key} label 1 (non-credible) items"] = s["n_label_1"]
        inc_comp[f"{key} label 1 share (percent)"] = s["label_1_share"]
        inc_comp[f"{key} items with supplied evidence"] = s["n_with_supplied_evidence"]
        inc_comp[f"{key} duplicate texts"] = s["n_duplicate_texts"]
        inc_comp[f"{key} texts with non-ASCII characters"] = s["n_non_ascii_texts"]
    inc = lic["incrediblae"]
    docs["datasheet-incrediblae.md"] = dict(
        dataset_key="incrediblae", dataset_title="InCrediblAE FC and RD (fixture stand-ins in the Phase 1 skeleton)",
        licence=(f"organisers' packaging terms {inc['packaging_terms']}; FC from {inc['FC']['source_corpus']} ({inc['FC']['licence']}); "
                 f"RD from {inc['RD']['source_corpus']} ({inc['RD']['licence']})"),
        licence_source=f"{inc['source']}; FC: {inc['FC']['source']}; RD: {inc['RD']['source']}",
        expected_raw=_expected_raw(root, "incrediblae/"), composition=inc_comp, label_rows=[],
        stats_source="; ".join(inc_sources))
    common = {"manifest_files": sorted(listed.items()), "expected_raw_file": EXPECTED_RAW}
    return {name: {**common, **fields} for name, fields in docs.items()}


# -- the real profile -----------------------------------------------------------------

def _file_rows(root: Path, listed: dict[str, str]) -> list[list[str]]:
    expected = {f"raw/{rel}": d for rel, d in _expected_raw(root, "")}
    rows = []
    for rel, digest in sorted(listed.items()):
        if rel.startswith("raw/"):
            kind = ("raw release file, equal to the published hash list" if expected.get(rel) == digest
                    else "raw release file, measured only (no published hash)" if rel not in expected
                    else "raw release file, DIFFERS from the published hash list")
        elif rel.startswith("configs/"):
            kind = "committed rule file"
        elif rel.startswith("private/"):
            kind = "private pseudonymized lane (hash only; never distributed)"
        else:
            kind = "derived table or statistics (text-free)"
        rows.append([rel, kind, digest])
    if any("DIFFERS" in r[1] for r in rows):
        bad = [r[0] for r in rows if "DIFFERS" in r[1]]
        raise PackInputError(f"raw files listed with a hash that differs from {EXPECTED_RAW}: {bad}")
    return rows


def _averitec_fields(st: dict[str, _Stats]) -> dict:
    av, dup, ks, tok, ch = st["averitec"], st["duplicates"], st["ks"], st["tokens"], st["chrono"]
    by = av("n_claims_by_split")
    n_train, n_dev = by["train"], by["dev"]
    
    if ks("provenance.limit") is not None or ks("provenance.stores") != av("n_claims"):
        raise PackInputError(f"{ks.shown}: provenance limit={ks('provenance.limit')} stores={ks('provenance.stores')}; "
                             f"the datasheet needs the full audit of all {av('n_claims')} labeled claims")
    und = av("n_undated_by_split")
    q, a = av("questions_per_claim"), av("answers_per_question")
    composition = [
        ["Claims (train / dev)", f"{_n(av('n_claims'))} ({_n(n_train)} / {_n(n_dev)})"],
        ["Dated claims", _n(av("n_dated"))],
        ["Undated claims, train / dev (excluded from both R1 splits, listed in the split manifests)",
         f"{_n(av('n_undated'))} ({_n(und['train'])} / {_n(und['dev'])})"],
        ["Earliest claim date", av("date_min")],
        ["Latest claim date", av("date_max")],
        ["Empty claim strings", f"{_n(len(av('empty_text_ids')))} ({', '.join(f'`{i}`' for i in av('empty_text_ids'))})"],
        ["Whitespace-only non-empty claim strings", _n(av("n_blank_nonempty_texts"))],
        ["Claim texts with non-ASCII characters", _n(av("n_non_ascii_texts"))],
        ["Questions per claim (min / median / mean / max)", f"{q['min']} / {q['median']} / {q['mean']} / {q['max']}"],
        ["Answers per question (min / median / mean / max)", f"{a['min']} / {a['median']} / {a['mean']} / {a['max']}"],
        [f"mDeBERTa-v3-base tokens per claim, special tokens included ({tok('percentile_rule')} p50 / p95 / p99 / max)",
         f"{tok('p50')} / {tok('p95')} / {tok('p99')} / {tok('max')}"],
        [f"Claims longer than max_length {tok('max_length')} tokens", _n(tok("n_over_max_length"))],
    ]
    lc, ls, lbs = av("label_counts"), av("label_shares"), av("label_counts_by_split")
    label_rows = []
    for lab in sorted(lc):
        tr, dv = lbs["train"].get(lab, 0), lbs["dev"].get(lab, 0)
        label_rows.append([lab, _n(lc[lab]), _pct(ls[lab]), f"{_n(tr)} ({_share(tr, n_train)})", f"{_n(dv)} ({_share(dv, n_dev)})"])
    decades = [[d, _n(c)] for d, c in sorted(av("claim_date_decade_histogram").items())]
    groups = dup("groups")
    conflicting = [g for g in groups if g["conflicting"]]
    if len(groups) != av("n_duplicate_groups") or len(conflicting) != av("n_conflicting_groups"):
        raise PackInputError(f"{dup.shown} ({len(groups)} groups, {len(conflicting)} conflicting) disagrees with "
                             f"{av.shown} ({av('n_duplicate_groups')}, {av('n_conflicting_groups')})")
    conflict_rows = [[g["group_id"][:12], ", ".join(f"`{m}`" for m in g["members"]), ", ".join(g["labels"]),
                      f"`{g['representative']}`", "yes" if g["spans_train_dev"] else "no"] for g in conflicting]
    duplicates = [
        ["Duplicate groups (byte-equal claim strings)", _n(av("n_duplicate_groups"))],
        ["Members removed (one representative kept per group)", _n(av("n_duplicate_members_removed"))],
        ["Groups whose members carry different labels", _n(av("n_conflicting_groups"))],
        ["Groups spanning train and dev", _n(av("n_groups_spanning_train_dev"))],
        ["Groups whose members carry different claim dates", _n(av("n_groups_with_differing_dates"))],
    ]
    counts = ch("counts")
    test_ids = ch("test")
    population = [*ch("train"), *ch("val_fuse"), *ch("val_cal"), *test_ids]
    dev_pop = sum(1 for c in population if c.startswith("averitec/dev/"))
    dev_test = sum(1 for c in test_ids if c.startswith("averitec/dev/"))
    split_rows = [
        ["Split manifest", f"`{ch('name')}`"],
        ["Cutoff date (claims dated on it are in train/val)", ch("cutoff_date")],
        ["First test date", ch("first_test_date")],
        ["Dated population after exclusions", _n(counts["population_dated"])],
        ["Train / val_fuse / val_cal / test", f"{_n(counts['train'])} / {_n(counts['val_fuse'])} / {_n(counts['val_cal'])} / {_n(counts['test'])}"],
        ["Excluded: undated / removed duplicates / empty claim string", f"{_n(counts['excluded_undated'])} / {_n(counts['excluded_duplicates'])} / {_n(counts['excluded_empty_text'])}"],
        ["Dev claims in the population / of which in the test partition", f"{_n(dev_pop)} / {_n(dev_test)}"],
    ]
    if any(v is not True for v in ch("assertions").values()):
        raise PackInputError(f"{ch.shown}: an assertion is not true")
    tot = ks("totals")
    ks_totals = []
    for label, key in (("Stores", "n_stores"), ("Empty stores", "n_stores_empty"), ("Documents", "n_docs"),
                       ("Sentences", "n_sentences"), ("Unique sentences, summed per store", "n_unique_sentences_summed_over_stores"),
                       ("Bytes uncompressed", "bytes_uncompressed"), ("Unparsable lines", "n_lines_unparsable")):
        ks_totals.append([label, _n(tot["train"][key]), _n(tot["dev"][key]), _n(tot["all"][key])])
    n_docs = tot["all"]["n_docs"]
    researched = tot["all"]["n_docs_by_type"]
    type_rows = [[f"`{t}`", "researched", _n(c), _share(c, n_docs)] for t, c in sorted(researched.items(), key=lambda kv: (-kv[1], kv[0])) if t != "other_type"]
    type_rows += [[f"`{t}`", "outside the researched six", _n(c), _share(c, n_docs)]
                  for t, c in sorted(ks("other_type_values").items(), key=lambda kv: (-kv[1], kv[0]))]
    flt = ks("filter")
    filt_rows = [
        ["Hosts on the frozen list", _n(flt["n_hosts_listed"])],
        ["Documents dropped, train / dev / all", f"{_n(flt['n_docs_filtered_by_split']['train'])} / {_n(flt['n_docs_filtered_by_split']['dev'])} / {_n(flt['n_docs_filtered'])}"],
        ["Share of all documents dropped (percent)", _frac_pct(flt["share_docs_filtered"])],
        ["Sentences dropped", _n(flt["n_sentences_filtered"])],
        ["Share of all sentences dropped (percent)", _frac_pct(flt["share_sentences_filtered"])],
        ["Stores losing at least one document", _n(flt["n_stores_with_filtered_docs"])],
    ]
    per_host = sorted(flt["per_host"].items(), key=lambda kv: (-kv[1], kv[0]))
    host_rows = [[f"`{h}`", _n(c), _share(c, flt["n_docs_filtered"])] for h, c in per_host if c > 0]
    never = [h for h, c in per_host if c == 0]
    lk = ks("leakage")
    leak_rows = [
        ["Documents on a fact-checker host (equals the filter count by construction)", _n(lk["n_docs_factchecker_host"]), _frac_pct(lk["share_factchecker_host"])],
        ["Documents whose URL equals an answer `source_url`, exact", _n(lk["n_docs_url_eq_answer_exact"]), _frac_pct(lk["share_url_eq_answer_exact"])],
        ["Documents whose URL equals an answer `source_url`, normalized", _n(lk["n_docs_url_eq_answer_norm"]), _frac_pct(lk["share_url_eq_answer_norm"])],
        ["`gold` documents whose URL equals an answer `source_url` (share of `gold` documents)", _n(lk["n_docs_url_eq_answer_norm_gold"]), _frac_pct(lk["share_gold_docs_url_eq_answer_norm"])],
        ["Stores with at least one normalized answer-URL match", _n(lk["n_stores_with_url_eq_answer_norm"]), _share(lk["n_stores_with_url_eq_answer_norm"], tot["all"]["n_stores"])],
    ]
    ud = ks("url_date")
    if ud["after_claim_is_lower_bound"] is not True:
        raise PackInputError(f"{ks.shown}: url_date.after_claim_is_lower_bound must be true")
    yr = ud["year_range"]
    url_rows = [
        [f"Documents with a URL-path date (years {yr[0]} to {yr[1]})", _n(ud["n_docs_url_date_parsed"]), _frac_pct(ud["coverage"])],
        ["Of these, dated strictly after the claim date (share of parsed)", _n(ud["n_docs_url_date_after_claim"]), _frac_pct(ud["share_of_parsed_after_claim"])],
        ["The same documents as a share of all documents", _n(ud["n_docs_url_date_after_claim"]), _frac_pct(ud["share_of_docs_after_claim"])],
        ["Documents in the stores of undated claims (not testable)", _n(ud["n_docs_in_undated_stores"]), _share(ud["n_docs_in_undated_stores"], n_docs)],
    ]
    nulls = av("null_or_empty_counts")
    null_rows = []
    for field in ("location_ISO_code", "reporting_source", "speaker", "original_claim_url"):
        tr, dv = nulls["train"][field], nulls["dev"][field]
        null_rows.append([f"`{field}`", f"{_n(tr)} ({_share(tr, n_train)})", f"{_n(dv)} ({_share(dv, n_dev)})",
                          f"{_n(tr + dv)} ({_share(tr + dv, n_train + n_dev)})"])
    n_claims = av("n_claims")
    loc_rows = [[f"`{r['value']}`", _n(r["count"]), _share(r["count"], n_claims)] for r in av("top_location_ISO_code")]
    src_rows = [[f"`{r['value']}`", _n(r["count"]), _share(r["count"], n_claims)] for r in av("top_reporting_source")]
    return dict(composition=composition, label_rows=label_rows, decade_rows=decades, duplicate_rows=duplicates,
                conflict_rows=conflict_rows, split_rows=split_rows, ks_totals=ks_totals, type_rows=type_rows,
                filter_rule=flt["rule"], filter_rows=filt_rows, host_rows=host_rows, hosts_never=never,
                leak_rows=leak_rows, url_rows=url_rows, null_rows=null_rows, location_rows=loc_rows,
                source_rows=src_rows, gold_docs=_n(researched["gold"]))


def _incrediblae_fields(st: dict[str, _Stats]) -> dict:
    inc = st["incrediblae"]
    rows = []
    for key in ("FC", "RD"):
        rows += [
            [key, "Items (= attack-file rows)", _n(inc(f"{key}.n_items"))],
            [key, "Label 1 (non-credible) items (percent)", f"{_n(inc(f'{key}.n_label_1'))} ({inc(f'{key}.label_1_share')})"],
            [key, "Distinct organiser ids", _n(inc(f"{key}.n_unique_organiser_ids"))],
            [key, "Items with supplied evidence", _n(inc(f"{key}.n_with_supplied_evidence"))],
            [key, "Empty texts", _n(inc(f"{key}.n_empty_text"))],
            [key, "Duplicate texts", _n(inc(f"{key}.n_duplicate_texts"))],
            [key, "Texts with non-ASCII characters", _n(inc(f"{key}.n_non_ascii_texts"))],
        ]
    rows += [
        ["FC", "Empty evidence fields", _n(inc("FC.n_empty_evidence"))],
        ["FC", "Evidence texts with non-ASCII characters", _n(inc("FC.n_non_ascii_evidence"))],
        ["RD", "Texts with user handles (before pseudonymization)", _n(inc("RD.n_texts_with_handles"))],
        ["RD", "Texts with URLs (before pseudonymization)", _n(inc("RD.n_texts_with_urls"))],
        ["RD", "Texts with bare 15-20 digit runs", _n(inc("RD.n_texts_with_digit_runs"))],
        ["RD", "Text length in characters (median / nearest-rank p95 / max)",
         f"{inc('RD.text_length_chars.median')} / {_n(inc('RD.text_length_chars.p95_nearest_rank'))} / {_n(inc('RD.text_length_chars.max'))}"],
    ]
    bases = [[k, inc(f"{k}.text_basis")] for k in ("FC", "RD")]
    sources = sorted(inc("source_sha256").items())
    return dict(composition_rows=rows, text_bases=bases, source_rows=[[r, d] for r, d in sources])


def _real_docs(root: Path, manifest: str, profile: dict, listed: dict[str, str]) -> dict[str, dict]:
    files, shown = _stats(root, manifest, profile, listed)
    st = {k: _Stats(read_json(p), shown[k]) for k, p in files.items()}
    lic = _licences(root)
    common = {"file_rows": _file_rows(root, listed), "expected_raw_file": EXPECTED_RAW,
              "n_files": len(listed)}
    inc = lic["incrediblae"]
    return {
        "datasheet-averitec.md": dict(
            common, dataset_key="averitec", dataset_title="AVeriTeC (train and dev, labeled claims)",
            licence=lic["averitec"]["licence"], licence_source=_no_at(lic["averitec"]["source"]),
            stats_sources=[shown[k] for k in ("averitec", "duplicates", "ks", "tokens", "chrono")],
            **_averitec_fields(st)),
        "datasheet-incrediblae.md": dict(
            common, dataset_key="incrediblae", dataset_title="InCrediblAE FC and RD (CLEF 2024 CheckThat! Task 6 attack files)",
            licence=(f"organisers' packaging terms {inc['packaging_terms']}; FC from {inc['FC']['source_corpus']} ({inc['FC']['licence']}); "
                     f"RD from {inc['RD']['source_corpus']} ({inc['RD']['licence']})"),
            licence_source=_no_at(f"{inc['source']}; FC: {inc['FC']['source']}; RD: {inc['RD']['source']}"),
            stats_sources=[shown["incrediblae"]], **_incrediblae_fields(st)),
    }


# -- shared render ---------------------------------------------------------------------------------

def _write_atomic(path: Path, text: str) -> Path:
    """UTF-8, LF only, exactly one trailing newline (inputs.write_text's bytes), via temp file + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = text.replace("\r\n", "\n").rstrip("\n") + "\n"
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def render_datasheets(root: Path, pack: Path, manifest: str, dv: str, ctx: PackContext | None = None) -> list[str]:
    """Render both datasheets for `manifest` under its profile, then data_version.txt (last).
 Both texts are rendered and heading-checked in memory before anything is written."""
    root, pack = Path(root), Path(pack)
    if not (root / manifest).is_file():
        raise PackInputError(f"missing data manifest {manifest}")
    profile = load_profile(root, manifest)
    listed = dict(_manifest_files(root, manifest))
    builders = {"fixture": _fixture_docs, "real": _real_docs}
    if profile["kind"] not in builders:
        raise PackInputError(f"{PROFILES}: unknown profile kind {profile['kind']!r} for {manifest}")
    docs = builders[profile["kind"]](root, manifest, profile, listed)
    gaps = _known_gaps(root, profile["known_gaps"])
    common = {"data_version": dv, "manifest": manifest, "known_gaps": gaps, "known_gaps_file": profile["known_gaps"]}
    texts = {}
    for name in DOCS:
        text = render(root, profile["template"], **common, **docs[name])
        require_headings(text, profile["required"], name)
        texts[name] = text
    written = []
    for name in DOCS:
        _write_atomic(pack / name, texts[name])
        if ctx is not None:
            sources = docs[name].get("stats_source") or "; ".join(docs[name]["stats_sources"])
            ctx.record(name, GENERATED_BY, f"{sources}; {EXPECTED_RAW}; {profile['known_gaps']} (versioned input)")
        written.append(name)
    _write_atomic(pack / DATA_VERSION_FILE, dv)
    if ctx is not None:
        ctx.record(DATA_VERSION_FILE, GENERATED_BY, f"aac.run.hashing.data_version({manifest}) shared by every sealed run")
    written.append(DATA_VERSION_FILE)
    return written


def render_pre_run(root: Path, pack: Path, manifest: str) -> list[str]:
    """The Art. 10 datasheets before any training: no sealed run, data_version straight from the manifest."""
    root = Path(root)
    if not (root / manifest).is_file():
        raise PackInputError(f"missing data manifest {manifest}")
    return render_datasheets(root, pack, manifest, data_version(root / manifest))


def build(root: Path, pack: Path, ctx: PackContext) -> list[str]:
    root, pack = Path(root), Path(pack)
    manifest, dv = sealed_data_version(root, ctx.runs)
    return render_datasheets(root, pack, manifest, dv, ctx)


def main(argv: list[str] | None = None) -> int:
    from aac.run.paths import pack_dir, root as project_root

    ap = argparse.ArgumentParser(prog="python -m aac.pack.datasheet", description="Render the Article 10 datasheets before any run.")
    ap.add_argument("command", choices=("render",))
    ap.add_argument("--manifest", required=True, help="data manifest relative to the root, e.g. data/derived/MANIFEST.sha256")
    ap.add_argument("--root", default=None, help="project root (default: the aac-pipeline checkout)")
    ap.add_argument("--pack", default=None, help="target directory (default: pack_dir(), the selected namespace's evidence-pack/)")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else project_root()
    pack = Path(args.pack) if args.pack else pack_dir(root)
    try:
        written = render_pre_run(root, pack, args.manifest)
    except PackInputError as exc:
        print(f"datasheet: {exc}", file=sys.stderr)
        return 1
    dv = (pack / DATA_VERSION_FILE).read_text(encoding="ascii").strip()
    print(f"datasheet: wrote {', '.join(written)} data_version={dv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
