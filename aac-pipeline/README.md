# Architecture-as-Compliance (AAC) Fact-Checking Pipeline

Source code, configurations, and benchmark datasets for the MSc Cyber Security dissertation:
*Architecture-as-Compliance: Mitigating Adversarial Vulnerability and Temporal Drift in Fact-Checking Systems Under the EU AI Act* by Diogo Pereira.

The pipeline implements an automated misinformation detection system designed to evaluate robustness against temporal drift and adversarial evasion attacks, featuring selective classification and confidence calibration to support human oversight.

---

## Architecture Overview

The system consists of five modular stages:
1. **C1 (Claim Encoder):** Fine-tuned transformer (`mDeBERTa-v3-base`) for stance and claim encoding.
2. **C2 (Evidence Retrieval & Scoring):** Dense retrieval (`BGE-M3`) and cross-encoder relevance reranking over evidence sources.
3. **C3 (Multi-Evidence Fusion):** Cross-attention network combining claim representations and top retrieved evidence snippets for 4-way classification (*Supported*, *Refuted*, *Not Enough Info*, *Conflicting*).
4. **C4 (Probability Calibration):** Temperature scaling and isotonic regression to calibrate model confidence scores.
5. **C5 (Selective Classification Gate):** Decision rule determining whether to output an automated verdict or route the claim to a human analyst based on confidence and risk thresholds.

---

## Directory Structure

```
aac-pipeline/
├── Makefile # Build automation and report generation
├── configs/ # Model, training, attack, and report configurations
├── data/
│ ├── raw/ # Benchmark datasets (AVeriTeC & BODEGA)
│ │ ├── averitec/ # AVeriTeC claims and knowledge store
│ │ └── incrediblae/ # BODEGA benchmark and pre-trained weights
│ └── derived/ # Chronological and random split partitions
├── reports/ # Generated figures (4.1, 4.2) and tables (4.1–4.8)
├── attacks/ # Adversarial evasion battery (BERT-Attack, TextFooler, PWWS, BeamAttack)
├── scripts/ # Environment wrapper scripts
├── src/aac/ # Core pipeline Python package
└── pyproject.toml # Package dependencies and installation specification
```

---

## Setup & Environment

The pipeline runs on Python 3.11 with virtual environment `.venv`:

```bash
# Activate virtual environment
source .venv/bin/activate

# Verify CLI
aac --help
```

To install dependencies in a fresh Python 3.11 environment:
```bash
pip install -e .
```

---

## Benchmark Datasets

Raw benchmark files are stored in `data/raw/`:
- **AVeriTeC:** Real-world fact-checking benchmark with multi-source evidence crawls.
- **BODEGA Benchmark:** Adversarial robustness benchmark for misinformation and rumor detection.

To verify dataset integrity:
```bash
cd data/raw && sha256sum -c MANIFEST.expected.sha256
```

---

## Reproducing Results

All dissertation tables and figures can be generated via `make`:

```bash
# Individual tables (4.1 to 4.8)
make report TABLE=4.1 # Baseline fact-checking comparison
make report TABLE=4.2 # Temporal drift evaluation
make report TABLE=4.3 # Rumor detection evaluation
make report TABLE=4.4 # Adversarial evasion resistance
make report TABLE=4.5 # Architectural ablation experiments
make report TABLE=4.6 # Selective prediction & risk-coverage metrics

# Generate all report tables and figures
make reports
```
