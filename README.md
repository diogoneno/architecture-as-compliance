# Architecture-as-Compliance

**A Robust and Governance-Aware Pipeline for Misinformation Detection on Social Media under the EU AI Act**

- **Author:** Diogo Pereira (`diogoneno@gmail.com`)
- **Supervisor:** Dr. Anupam Mazumdar
- **Degree:** MSc Cyber Security
- **Institution:** University of Essex Online
- **Submission:** November 2026

---

## Overview

This repository contains the full source code, benchmark datasets, evaluation harness, and submission documents for the MSc Cyber Security dissertation *Architecture-as-Compliance*.

The project addresses the fragility of automated fact-checking systems when subjected to temporal drift and adversarial evasion attacks, while structuring compliance with high-risk obligations under the EU Artificial Intelligence Act (Regulation (EU) 2024/1689, Articles 10–15) and independent auditability under the Digital Services Act (Regulation (EU) 2022/2065, Article 37).

---

## Five-Component Architecture

The pipeline pairs technical failure mitigations with regulatory obligations:

1. **C1 (Claim & Stance Encoder):** Fine-tuned transformer (`mDeBERTa-v3-base`) for claim encoding (*Art. 10 Data Governance & Art. 15 Accuracy*).
2. **C2 (Dense Retrieval & Evidence Scorer):** Multi-stage retrieval with `BGE-M3` and cross-encoder relevance re-ranking (*Art. 13 Transparency*).
3. **C3 (Multi-Evidence Fusion):** Cross-attention gating network combining claim representations and evidence passages into a 4-way verdict (*Art. 12 Cryptographic Decision Logging*).
4. **C4 (Post-Hoc Probability Calibration):** Temperature scaling and isotonic regression to calibrate confidence scores (*Art. 15 Accuracy & Error Metrics*).
5. **C5 (Selective Classification Gate):** Risk-coverage decision rule routing uncertain predictions to human analysts (*Art. 14 Human Oversight*).

---

## Repository Structure

- **[`docs/dissertation/`](docs/dissertation/):** Submission documents, including PDF, DOCX, ODT, defense presentation (`.pptx`), calculations workbook (`.xlsx`), and SHA-256 checksum manifest.
- **[`aac-pipeline/`](aac-pipeline/):** Complete standalone fact-checking pipeline, CLI, and evaluation battery.
  - `configs/`: Model, training, regime, and attack configurations.
  - `data/`: Raw and derived benchmark data (AVeriTeC & InCrediblAE/BODEGA).
  - `src/aac/`: Core implementation package.
  - `attacks/`: Adversarial evasion battery (BERT-Attack, TextFooler, PWWS, BeamAttack).
  - `reports/`: Generated dissertation tables (4.1 to 4.8) and figures (4.1, 4.2).

---

## Quick Start

### 1. Environment Setup

```bash
cd aac-pipeline
source .venv/bin/activate
aac --help
```

Or install in a fresh Python 3.11 environment:
```bash
cd aac-pipeline
pip install -e .
```

### 2. Verify Checksums

```bash
cd docs/dissertation
sha256sum -c AAC-Dissertation-DiogoPereira.sha256
```

### 3. Reproduce Experimental Tables

```bash
cd aac-pipeline
make report TABLE=4.1  # Baseline reproductions
make report TABLE=4.2  # Temporal drift (R1)
make report TABLE=4.3  # Domain transfer (R2)
make report TABLE=4.4  # Adversarial evasion (R3)
make report TABLE=4.5  # Architectural ablations (R4)
make report TABLE=4.6  # Selective prediction & risk curve
make reports           # All tables and figures
```

---

## License

This project is released for academic evaluation and research reproducibility. Datasets remain under their respective source licenses (AVeriTeC and CheckThat! Lab / InCrediblAE).
