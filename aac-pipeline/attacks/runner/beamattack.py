"""BeamAttack as an OpenAttack attacker .

BeamAttack (LucK1Y/BeamAttack, branch public_branch, commit ba974db9451bea005d52a065d6d44e19f0bcea59, MIT) ships a
search function, not an OpenAttack attacker. BeamAttackAttacker wraps `attack_text_bfs` in an OpenAttack
ClassificationAttacker, so the runner runs it through the same AttackEval(invoke_limit=budget) path as the four
BODEGA recipes: AacVictim counts every text, the server QueryBudget charges it, OpenAttack's invoke limit ends the
item, and the three counters must agree per item . No fourth counter exists.

Parameters (transcribed in configs/attacks/beamattack.yaml; tests/unit/test_attack_recipes.py compares both by AST):
imp="bert" (leave-one-out logit importance; no LIME explainer is ever built, so LIME_QUERIES_PER_ITEM is 0),
early_stop=1 (strict success needs one flip), k=3 and width=10 (repo defaults), add_semantic_pruning=False (no BLEURT
in the search loop), and the repo defaults for positional, temperature, remove_words and keep_original.

The masked LM is FacebookAI/roberta-large at the pinned revision. load_masked_lm checks the sha256 of the weight
file against configs/models.yaml before any load, loads safetensors only, offline.

Imports: the standard library, PyYAML, torch, transformers, OpenAttack and beam_attack only (never `aac`).
"""
import hashlib
import os
import sys
from pathlib import Path

import torch
import yaml
from OpenAttack.attackers.classification import ClassificationAttacker
from OpenAttack.tags import TAG_English, Tag

AAC_ROOT = Path(__file__).resolve().parents[2]
BEAMATTACK_DIR = AAC_ROOT / "third_party" / "BeamAttack"
MODELS_YAML = AAC_ROOT / "configs" / "models.yaml"
if str(BEAMATTACK_DIR) not in sys.path:  # scripts/attack_env.sh puts only third_party/BODEGA on PYTHONPATH
    sys.path.append(str(BEAMATTACK_DIR))
from beam_attack.attack import attack_text_bfs  # noqa: E402 (vendored submodule at the pinned commit)

BEAMATTACK_PARAMS = {
    "imp": "bert",
    "k": 3,
    "width": 10,
    "early_stop": 1,
    "add_semantic_pruning": False,
    "positional": True,
    "temperature": 1,
    "remove_words": False,
    "keep_original": False,
}
LIME_QUERIES_PER_ITEM = 0
MASKED_LM_KEY = "roberta-large"
MASKED_LM_HF_ID = "FacebookAI/roberta-large"
MASKED_LM_REVISION = "722cf37b1afa9454edce342e7895e588b6ff1d59"
MASKED_LM_WEIGHT_FILE = "model.safetensors"


class WeightHashMismatch(RuntimeError):
    """The masked-LM weight file in the cache is not the file whose sha256 configs/models.yaml records."""


def _sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _recorded_entry(models_yaml):
    data = yaml.safe_load(Path(models_yaml).read_text(encoding="utf-8"))
    for m in data.get("models") or []:
        if m.get("key") == MASKED_LM_KEY:
            return m
    raise WeightHashMismatch(f"{MASKED_LM_KEY} is not a cached model in {models_yaml}")


def _hf_home():
    return Path(os.environ.get("HF_HOME") or (AAC_ROOT / ".hf"))


def weight_path(hf_home=None):
    """The pinned snapshot's weight file (a symlink into blobs/ in the HF cache layout)."""
    hub = Path(hf_home) if hf_home else _hf_home()
    return (hub / "hub" / ("models--" + MASKED_LM_HF_ID.replace("/", "--")) / "snapshots" / MASKED_LM_REVISION
            / MASKED_LM_WEIGHT_FILE)


def check_weights(models_yaml=MODELS_YAML, hf_home=None):
    """Return the weight path after its sha256 equals the recorded value; raise WeightHashMismatch otherwise."""
    entry = _recorded_entry(models_yaml)
    if entry.get("hf_id") != MASKED_LM_HF_ID or entry.get("revision") != MASKED_LM_REVISION:
        raise WeightHashMismatch(f"{models_yaml}: {MASKED_LM_KEY} id or revision differs from the adapter pin")
    if entry.get("weight_file") != MASKED_LM_WEIGHT_FILE:
        raise WeightHashMismatch(f"{models_yaml}: {MASKED_LM_KEY} weight_file is not {MASKED_LM_WEIGHT_FILE}")
    expected = str(entry.get("weight_sha256") or "")
    path = weight_path(hf_home)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: download the pinned revision first")
    actual = _sha256_file(path)
    if len(expected) != 64 or actual != expected:
        raise WeightHashMismatch(f"{MASKED_LM_WEIGHT_FILE}: sha256 {actual} != recorded {expected or '(none)'}")
    return path


def load_masked_lm(device, models_yaml=MODELS_YAML):
    """roberta-large (MLM head) and its tokenizer at the pinned revision, offline, after the sha256 check."""
    check_weights(models_yaml)  # raises before transformers sees any file
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MASKED_LM_HF_ID, revision=MASKED_LM_REVISION, local_files_only=True)
    model = AutoModelForMaskedLM.from_pretrained(MASKED_LM_HF_ID, revision=MASKED_LM_REVISION, local_files_only=True,
                                                 use_safetensors=True, torch_dtype=torch.float32)
    device = device if isinstance(device, torch.device) else torch.device(device)
    model.to(device)
    model.eval()
    return model, tokenizer


class BeamAttackAttacker(ClassificationAttacker):
    """attack_text_bfs with BEAMATTACK_PARAMS; returns the first successful text or None.

 ClassificationAttacker.__call__ asks the victim for the origin label first and re-checks the returned text with
 get_pred, both through the counted victim. attack_text_bfs reads probabilities with victim.get_prob only.
 """

    TAGS = {TAG_English, Tag("get_pred", "victim"), Tag("get_prob", "victim")}

    def __init__(self, device, masked_lm=None):
        if BEAMATTACK_PARAMS["imp"] != "bert" or LIME_QUERIES_PER_ITEM != 0:
            raise ValueError("BeamAttack is pinned to imp='bert' with 0 LIME queries; the LIME path is refused")
        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.model, self.tokenizer = masked_lm if masked_lm is not None else load_masked_lm(self.device)

    def attack(self, victim, sentence, goal):
        outputs, _, _ = attack_text_bfs(sentence, None, victim, self.device, self.model, self.tokenizer,
                                        verbose=False, **BEAMATTACK_PARAMS)
        return outputs[0][0] if outputs else None
