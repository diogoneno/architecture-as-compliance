"""The four BODEGA recipes, built exactly as BODEGA builds them .

Source: third_party/BODEGA/runs/attack.py at commit 2dda5a6e6d8a291327df79a44767efea8cab556f, lines 118 (filter
words), 122 (PWWS), 127-128 (TextFooler), 130 (DeepWordBug), 140 (BERTattack); transcribed in
configs/attacks/bodega-recipes.yaml. tests/unit/test_attack_recipes.py compares the constructor calls below with
both files by AST, so an edit here that changes a class or a keyword argument fails that test.

The calls are plain Python calls. Nothing here turns a configuration string into code.
BeamAttack is built here too, as attacks/runner/beamattack.py BeamAttackAttacker (an OpenAttack
ClassificationAttacker, so the same AttackEval(invoke_limit=budget) path and counters apply); it is not a BODEGA
recipe, so it is in NAMES but not in LEDGER_TO_BODEGA. Its parameters are pinned in beamattack.py and
configs/attacks/beamattack.yaml. The module is imported only when BeamAttack is built (roberta-large load).
"""
import OpenAttack

UNK_TEXT = "\U0001F525"   # BODEGA victims/unk_fix_wrapper.py line 3: UNK_TEXT = '🔥'
SEPARATOR_CHAR = "~"      # BODEGA utils/data_mappings.py line 1

# ledger name -> BODEGA attack name (runs/attack.py argv[3])
LEDGER_TO_BODEGA = {
    "pwws": "PWWS",
    "textfooler": "TextFooler",
    "deepwordbug": "DeepWordBug",
    "bert_attack": "BERTattack",
}
NAMES = tuple(LEDGER_TO_BODEGA) + ("beamattack",)


def filter_words():
    """OpenAttack's default English filter words plus BODEGA's pair separator (attack.py line 118)."""
    return OpenAttack.attack_assist.filter_words.get_default_filter_words('english') + [SEPARATOR_CHAR]


def build(name, device):
    """Return the OpenAttack attacker for a ledger name; `device` is a torch.device (BODEGA's attacker_device)."""
    words = filter_words()
    if name == "pwws":
        return OpenAttack.attackers.PWWSAttacker(token_unk=UNK_TEXT, lang='english', filter_words=words)
    if name == "textfooler":
        return OpenAttack.attackers.TextFoolerAttacker(token_unk=UNK_TEXT, lang='english', filter_words=words)
    if name == "deepwordbug":
        return OpenAttack.attackers.DeepWordBugAttacker(token_unk=UNK_TEXT)
    if name == "bert_attack":
        return OpenAttack.attackers.BERTAttacker(filter_words=words, use_bpe=False, device=device)
    if name == "beamattack":
        from attacks.runner.beamattack import BeamAttackAttacker
        return BeamAttackAttacker(device=device)
    raise ValueError(f"no recipe for attack {name!r} (known: {', '.join(NAMES)})")
