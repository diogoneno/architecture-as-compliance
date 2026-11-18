# BODEGA Adversarial Attack Suite

Adversarial attack execution and evaluation harness for testing fact-checking robustness against textual evasion attacks on the BODEGA benchmark.

---

## Supported Attacks

The following adversarial evasion algorithms are integrated from OpenAttack and BeamAttack:
- **BERT-Attack:** Contextual masked language model perturbation.
- **TextFooler:** Word-level semantic substitution via counter-fitted embedding similarity.
- **PWWS:** Probability Weighted Word Saliency replacement.
- **DeepWordBug:** Character-level perturbations (swap, substitution, deletion, insertion).
- **BeamAttack:** Beam-search guided adversarial generation using RoBERTa-large.

---

## Running Attacks

Attacks run through the environment wrapper:

```bash
make -f mk/attack.mk attack-fc-clean
```
