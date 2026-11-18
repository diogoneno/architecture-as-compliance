# MSc Cyber Security Dissertation Submission

## Architecture-as-Compliance: A Robust and Governance-Aware Pipeline for Misinformation Detection on Social Media under the EU AI Act

- **Candidate:** Diogo Pereira (`diogoneno@gmail.com`)
- **Supervisor:** Dr. Anupam Mazumdar
- **Institution:** University of Essex Online
- **Program:** MSc Cyber Security
- **Module:** CYSPROJ (Cyber) — Dissertation
- **Submission Date:** November 2026

---

### Overview

This package contains the submission documents, presentation slides, and supporting calculations workbook for the MSc Cyber Security dissertation *Architecture-as-Compliance* (CYSPROJ).

The dissertation evaluates automated misinformation detection under the regulatory requirements of the EU AI Act, testing robustness against temporal drift and adversarial evasion attacks while implementing confidence calibration and selective human routing.

---

### Dissertation Documents

| File | Format | Description |
| :--- | :---: | :--- |
| **[`AAC-Dissertation-DiogoPereira.pdf`](./AAC-Dissertation-DiogoPereira.pdf)** | PDF | Primary submission manuscript with all embedded figures, tables, and references. |
| **[`AAC-Dissertation-DiogoPereira.docx`](./AAC-Dissertation-DiogoPereira.docx)** | Word (.docx) | Complete Microsoft Word document including all tables and figures. |
| **[`AAC-Dissertation-DiogoPereira.odt`](./AAC-Dissertation-DiogoPereira.odt)** | ODT (.odt) | OpenDocument Text format. |
| **[`Architecture-as-Compliance-Presentation.pptx`](./Architecture-as-Compliance-Presentation.pptx)** | PPTX (.pptx) | Dissertation defense presentation slides (14 slides). |
| **[`MSc-Dissertation-Calculations-and-Tables.xlsx`](./MSc-Dissertation-Calculations-and-Tables.xlsx)** | Excel (.xlsx) | Calculations workbook with experimental tables, confidence intervals, and hypothesis tests. |
| **[`AAC-Dissertation-DiogoPereira.sha256`](./AAC-Dissertation-DiogoPereira.sha256)** | Checksums | Cryptographic SHA-256 digests for all submission files above. |

---

### Checksum Verification

To verify the integrity of all delivery files:

```bash
sha256sum -c AAC-Dissertation-DiogoPereira.sha256
```

All 5 files will report `OK`.

---

### Pipeline Source Code

The complete standalone source code, configuration files, and benchmark datasets are located in [`aac-pipeline/`](../../aac-pipeline/).
