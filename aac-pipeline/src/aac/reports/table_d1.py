"""Table D.1 (stack and licences): rows exactly as the shell; `choice` from configs/models.yaml
and configs/attacks/bodega-recipes.yaml , `licence` from the hand-verified configs/licences.yaml
(InCrediblAE packaging `unverified`), `version` parsed from uv.lock, attacks/attack-cpu.lock.txt
(`commit-pinned` for the two VCS requirements per attacks/README.md lock_mode), the model revisions,
the tracer run's env.json (python, torch build) and the raw-data manifest names. No version string
is typed in this module: every value is read from a file."""
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import yaml

from aac.reports import fmt, sources
from aac.run.paths import reports_dir

TABLE = "D.1"
MAIN_PACKAGES = ("torch", "transformers", "sentence-transformers", "scikit-learn", "netcal", "huggingface-hub",
                 "statsmodels", "scipy", "jsonschema", "bm25s", "matplotlib")
ATTACK_PACKAGES = ("transformers", "openattack")
VCS_PACKAGES = ("bleurt-pytorch", "lambo")
LOCK_MODE_RE = re.compile(r"^lock_mode:\s*(\S+)", re.M)


def _norm(name: str) -> str:
    return name.lower().replace("_", "-")


def main_lock_versions(root: Path, torch_build: str) -> dict[str, str]:
    """Package -> version from uv.lock (tomllib). torch has one entry per index; the one whose version
 equals the run's recorded torch build is reported."""
    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    by_name: dict[str, list[str]] = {}
    for pkg in lock["package"]:
        by_name.setdefault(_norm(pkg["name"]), []).append(str(pkg["version"]))
    out = {}
    for name in MAIN_PACKAGES:
        versions = sorted(set(by_name.get(name, [])))
        if not versions:
            raise RuntimeError(f"uv.lock has no package {name!r}")
        if name == "torch":
            match = [v for v in versions if v == torch_build]
            if not match:
                raise RuntimeError(f"uv.lock torch versions {versions} do not include the run's build {torch_build!r}")
            out[name] = match[0]
        elif len(versions) == 1:
            out[name] = versions[0]
        else:
            raise RuntimeError(f"uv.lock lists {name} at several versions {versions}")
    return out


def attack_lock(root: Path) -> tuple[dict[str, str], dict[str, str], str]:
    """(pinned versions, VCS commit pins, python version) from attacks/attack-cpu.lock.txt."""
    text = (root / "attacks" / "attack-cpu.lock.txt").read_text(encoding="utf-8")
    pins, vcs = {}, {}
    for line in text.splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s\\]+)", line)
        if m:
            pins[_norm(m.group(1))] = m.group(2)
        m = re.match(r"^([A-Za-z0-9_.-]+) @ git\+\S+@([0-9a-f]{7,40})", line)
        if m:
            vcs[_norm(m.group(1))] = m.group(2)
    py = re.search(r"--python-version\s+(\d+\.\d+)", text)
    if not py:
        raise RuntimeError("attack-cpu.lock.txt header does not record --python-version")
    for name in ATTACK_PACKAGES:
        if name not in pins:
            raise RuntimeError(f"attack-cpu.lock.txt has no pin for {name}")
    for name in VCS_PACKAGES:
        if name not in vcs:
            raise RuntimeError(f"attack-cpu.lock.txt has no commit pin for {name}")
    return pins, vcs, py.group(1)


def lock_mode(root: Path) -> str:
    m = LOCK_MODE_RE.search((root / "attacks" / "README.md").read_text(encoding="utf-8"))
    if not m:
        raise RuntimeError("attacks/README.md does not state lock_mode")
    return m.group(1)


def models(root: Path) -> dict[str, dict]:
    with open(root / "configs" / "models.yaml", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    out = {m["key"]: m for m in data["models"]}
    for m in data.get("not_cached_yet") or []:
        out[m["key"]] = m
    return out


def licences(root: Path) -> dict:
    with open(root / "configs" / "licences.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def recipes(root: Path) -> dict:
    with open(root / "configs" / "attacks" / "bodega-recipes.yaml", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def manifest_hashes(root: Path) -> list[tuple[str, str]]:
    out = []
    for line in (root / "data" / "raw" / "MANIFEST.expected.sha256").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        digest, _, name = line.partition("  ")
        out.append((name.strip(), digest.strip()))
    return sorted(out)


def _short(rev: str) -> str:
    return rev[:8]


def _m(ms: dict, key: str) -> str:
    return ms[key]["hf_id"]


def _lic(lics: dict, group: str, key: str) -> str:
    return str(lics[group][key]["licence"])


def rows(root: Path) -> tuple[list[dict], dict]:
    root = Path(root)
    state, entries = sources.require_runs(root, regime="R1", name="r1-chrono-full", table=TABLE, row_ident="Runtime", col_key="version")
    if state == "cut":
        raise sources.MissingRun("Table D.1 needs the tracer run; it is never blanked by a cut")
    seed0 = [e for e in entries if int(e["seed"]) == 0]
    if not seed0:
        raise sources.MissingRun("Table D.1 needs the seed-0 r1-chrono-full run for env.json")
    run = sources.load_run(root, seed0[0])
    env = json.loads((run.run_dir / "env.json").read_text(encoding="utf-8"))
    main = main_lock_versions(root, env["torch"])
    pins, vcs, py_attack = attack_lock(root)
    mode = lock_mode(root)
    ms, lics, rc = models(root), licences(root), recipes(root)
    recipe_names = ", ".join(rc["recipes"].keys())
    torch_build = env["torch"]
    cuda = ("n/a (CPU torch build; no CUDA runtime recorded yet)" if torch_build.endswith("+cpu")
            else f"CUDA per torch build {torch_build.split('+', 1)[-1]} (nvidia-smi on the GPU host records the driver)")
    inc = lics["data"]["incrediblae"]
    hashes = manifest_hashes(root)
    data_cell = "; ".join(f"{name} {digest[:12]}" for name, digest in hashes)
    out = [
        {"layer": "Content encoder",
         "choice": f"mDeBERTa-v3-base ({_m(ms, 'mdeberta-v3-base')}); {ms['mdeberta-v3-base']['loader']}",
         "licence": _lic(lics, "models", "mdeberta-v3-base"),
         "version": f"rev {ms['mdeberta-v3-base']['revision']}; transformers {main['transformers']} (uv.lock)"},
        {"layer": "Retrieval and re-ranking",
         "choice": f"BM25 shortlist (bm25s) then BGE-M3 dense scoring ({_m(ms, 'bge-m3')}) then cross-encoder {_m(ms, 'bge-reranker-v2-m3')}",
         "licence": f"{_lic(lics, 'code', 'bm25s')} (bm25s); {_lic(lics, 'models', 'bge-m3')} (BGE-M3); {_lic(lics, 'models', 'bge-reranker-v2-m3')} (reranker)",
         "version": f"bm25s {main['bm25s']}; bge-m3 rev {_short(ms['bge-m3']['revision'])}; reranker rev {_short(ms['bge-reranker-v2-m3']['revision'])}; sentence-transformers {main['sentence-transformers']} (uv.lock)"},
        {"layer": "Claim-to-evidence scorer",
         "choice": f"Entailment head fine-tuned from mDeBERTa-v3-base ({_m(ms, 'mdeberta-v3-base')}) on AVeriTeC Q&A evidence",
         "licence": _lic(lics, "models", "mdeberta-v3-base"),
         "version": f"as encoder: rev {ms['mdeberta-v3-base']['revision']}"},
        {"layer": "Fusion",
         "choice": "Logistic gating layer, scikit-learn LogisticRegression (fitted on the validation split)",
         "licence": _lic(lics, "code", "scikit-learn"),
         "version": f"scikit-learn {main['scikit-learn']} (uv.lock)"},
        {"layer": "Calibration and ECE",
         "choice": "Temperature scaling (re-implemented); isotonic regression (scikit-learn); netcal ECE (15 bins) and reliability diagram",
         "licence": f"{_lic(lics, 'code', 'scikit-learn')} (scikit-learn); {_lic(lics, 'code', 'netcal')} (netcal)",
         "version": f"scikit-learn {main['scikit-learn']}; netcal {main['netcal']} (uv.lock)"},
        {"layer": "Attack battery",
         "choice": f"BODEGA harness ({recipe_names} via {rc['library']}); BeamAttack ({rc['beamattack']['entry_point']}, {rc['beamattack']['chosen'].split(' (')[0]})",
         "licence": f"{_lic(lics, 'code', 'BODEGA')} (BODEGA, README statement; LAMBO {_lic(lics, 'code', 'LAMBO')}); {_lic(lics, 'code', 'OpenAttack')} (OpenAttack); {_lic(lics, 'code', 'BeamAttack')} (BeamAttack); {_lic(lics, 'code', 'bleurt-pytorch')} (bleurt-pytorch)",
         "version": f"BODEGA {_short(rc['bodega_commit'])}; OpenAttack {pins['openattack']}; BeamAttack {_short(rc['beamattack']['commit'])}; transformers {pins['transformers']} in the attack environment; "
                    f"bleurt-pytorch and lambo commit-pinned ({_short(vcs['bleurt-pytorch'])}, {_short(vcs['lambo'])}); attack lock mode {mode}"},
        {"layer": "Model card and datasheet",
         "choice": "huggingface_hub ModelCard from an own template; datasheet script (own code, Gebru et al. structure)",
         "licence": f"{_lic(lics, 'code', 'huggingface_hub')} (huggingface_hub); own code",
         "version": f"huggingface_hub {main['huggingface-hub']} (uv.lock)"},
        {"layer": "Runtime",
         "choice": f"Python {env['python']} (pipeline), {py_attack} (attacks); PyTorch {torch_build}; {cuda}; transformers {main['transformers']} ({pins['transformers']} in attacks); sentence-transformers {main['sentence-transformers']}",
         "licence": f"PSF (Python); {_lic(lics, 'code', 'torch')} (torch); {_lic(lics, 'code', 'transformers')} (transformers); {_lic(lics, 'code', 'sentence-transformers')} (sentence-transformers)",
         "version": f"python {env['python']}; torch {torch_build}; transformers {env['transformers']}; numpy {env['numpy']} (tracer run env.json, lock sha256 {env['lock_sha256'][:12]}); scipy {main['scipy']}; statsmodels {main['statsmodels']}; jsonschema {main['jsonschema']}; matplotlib {main['matplotlib']} (uv.lock)"},
        {"layer": "Data",
         "choice": f"AVeriTeC train and dev with the knowledge store; InCrediblAE FC (source {inc['FC']['source_corpus']}) and RD (source {inc['RD']['source_corpus']}); organisers' packaging terms {inc['packaging_terms']}",
         "licence": f"{lics['data']['averitec']['licence']} (AVeriTeC); {inc['FC']['licence']} (FC source); {inc['RD']['licence']} (RD source); InCrediblAE packaging {inc['packaging_terms']}",
         "version": f"data/raw/MANIFEST.expected.sha256 ({len(hashes)} files): {data_cell}"},
    ]
    return out, {"run": run, "main": main, "pins": pins, "vcs": vcs}


def build(root: Path, out_dir: Path | None = None) -> list[Path]:
    root = Path(root)
    out_dir = Path(out_dir) if out_dir else reports_dir(root)
    shell = fmt.load_shells(root / "contracts" / "table-shells.json")[TABLE]
    table_rows, ctx = rows(root)
    if [r["layer"] for r in table_rows] != list(shell["rows"]):
        raise fmt.ShellMismatch(f"Table D.1 rows {[r['layer'] for r in table_rows]} != shell {shell['rows']}")
    run = ctx["run"]
    prov = sources.provenance([run])
    long_rows = []
    srcs = {"choice": "configs/models.yaml; configs/attacks/bodega-recipes.yaml", "licence": "configs/licences.yaml (hand-verified)",
            "version": "uv.lock; attacks/attack-cpu.lock.txt; configs/models.yaml; runs/<tracer>/env.json; data/raw/MANIFEST.expected.sha256"}
    for r in table_rows:
        for key, label in (("choice", "Choice"), ("licence", "Licence"), ("version", "Version recorded at build")):
            long_rows.append(fmt.long_row(TABLE, r["layer"], label, key, r[key], test=srcs[key], **prov))
    return list(fmt.emit(TABLE, table_rows, long_rows, config_hash=prov["config_hash"], data_version=prov["data_version"], out_dir=out_dir))
