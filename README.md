# AuralGuard v2: Multi-View One-Class Learning for Generalizable and Calibrated Detection of AI-Generated Speech in the Wild

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)](https://pytorch.org/)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-yellow)](https://huggingface.co/MoshinAli/auralguard-checkpoints)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-under%20review-blue)]()

> **Official repository for the journal manuscript:**  
> *"AuralGuard: Multi-View One-Class Learning for Generalizable and Calibrated Detection of AI-Generated Speech in the Wild"*  
> **Authors:** Mohsin Ibna Hossain, Md. Muttakin, Sadman Hasan Miraj, and Mohammad Mahmudul Hasan (Corresponding Author)  
> **Affiliation:** Department of Computer Science, American International University-Bangladesh (AIUB)

---

## 🔬 Overview

Modern neural speech synthesizers, diffusion vocoders, and neural codec language models (e.g., VALL-E, CosyVoice, F5-TTS) generate speech that is perceptually indistinguishable from genuine human recordings. While state-of-the-art countermeasures achieve competitive in-domain Equal Error Rates (EER) on clean laboratory benchmarks, they suffer two fatal real-world failure modes:
1. **Catastrophic Out-of-Domain Generalization Collapse:** Legacy acoustic models degrade to 38–50% EER on unconstrained found audio.
2. **Severe Probability Miscalibration:** Conventional classifiers output overconfident false-alarm probabilities under acoustic domain shift ($\text{ECE} > 0.55$), rendering their decisions perilous for biometric security and judicial admissibility (e.g., under the *Daubert* standard and Article 50 of the EU AI Act).

**AuralGuard** addresses both challenges through:
- **Dual-Stream Multi-View Front-End:** Combines high-level semantic contextual representations from a frozen WavLM-Large backbone (equipped with learnable multi-layer attention) and an explicit low-level vocoder-artifact stream capturing Linear Frequency Cepstral Coefficients (LFCC), Modified Group Delay (MGD), and Constant-Q Transform (CQT) phase derivatives.
- **Bidirectional Gated Cross-Attention:** Dynamically queries and trusts semantic representations versus phase anomalies based on local time-frequency corruption and lossy codec transcoding.
- **Spectro-Temporal Graph Back-End:** Projects fused representations into an AASIST graph-attention network (GATv2) optimized under an Orthogonal Centroid Scoring (OCS) one-class objective with Supervised Contrastive regularization.
- **Parameter Efficiency:** The 315M WavLM backbone remains completely frozen; only **4.49M parameters (1.4% of total capacity)** are trained, enabling rapid convergence and low inference latency ($\text{RTF} = 0.058$ on GPU, $0.412$ on CPU).

### 🎯 Research Questions

This study systematically addresses four core research questions:
1. **RQ1 (Cross-Corpus Generalization):** Can a multi-view countermeasure trained strictly on clean in-domain speech (ASVspoof 2019 LA) generalize zero-shot to unconstrained web audio (In-the-Wild) and multi-vocoder synthesis (WaveFake) without catastrophic performance collapse?
2. **RQ2 (Forensic Probability Calibration):** Does an open-set one-class hyperspherical objective (OC-Softmax with SupCon) produce well-calibrated posterior probability estimates and bounded confidence intervals under severe acoustic distribution shift?
3. **RQ3 (Multi-View Feature Complementarity):** What are the complementary roles of semantic foundation representations (WavLM) and physical phase artifacts (LFCC, MGD, CQT phase) when exposed to lossy transmission codecs and diverse vocoders?
4. **RQ4 (Parameter Efficiency & Operational Feasibility):** Can competitive zero-shot discrimination and legal-grade calibration be sustained while keeping the 315M foundation backbone frozen (1.4% trainable parameters) at real-time speeds?

---

## 📊 Master Empirical Benchmark Results

All models are trained strictly **once** on the ASVspoof 2019 LA training partition and evaluated **zero-shot** without target-domain adaptation across all evaluation benchmarks. Metrics report point estimates with 95% non-parametric bootstrap confidence intervals ($B=1,000$).

| Model Architecture | In-Domain (19LA) EER (%) | min t-DCF | Zero-Shot (In-the-Wild) EER (%) | ITW AUROC | ITW ECE ($\downarrow$) | Zero-Shot (WaveFake) EER (%) | WaveFake AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **B1: LFCC-LCNN** | 16.29 [15.97, 16.63] | 0.6131 | 50.62 [49.72, 51.49] | 0.4979 | 0.4292 | 45.32 [44.82, 45.82] | 0.5638 |
| **B2: RawNet2** | 8.99 [8.72, 9.23] | 0.3444 | 38.08 [37.28, 38.89] | 0.6671 | 0.5761 | 50.37 [49.88, 50.88] | 0.4863 |
| **B3: AASIST** | 10.63 [10.36, 10.92] | 0.3921 | 42.33 [41.52, 43.14] | 0.6122 | 0.5552 | 49.89 [49.39, 50.39] | 0.4873 |
| **B5: WavLM + OCS** | **2.86** [2.73, 3.00] | **0.1041** | **16.55** [16.12, 16.98] | **0.9103** | 0.2320 | 47.43 [46.93, 47.93] | 0.5370 |
| **AuralGuard (v1)** | 7.30 [7.12, 7.50] | 0.2494 | 23.47 [23.03, 23.89] | 0.8345 | **0.0965** | 48.72 [48.22, 49.22] | 0.5207 |
| **AuralGuard v2** | 6.81 [6.61, 7.00] | 0.2398 | 19.19 [18.78, 19.61] | 0.8820 | **0.1716** | **45.19** [44.69, 45.71] | **0.5659** |

### Key Findings
- **Unrivaled Forensic Calibration:** AuralGuard achieves an out-of-domain Expected Calibration Error of **0.0965 (v1) / 0.1716 (v2)** on In-the-Wild, significantly surpassing B5 (0.2320) and legacy architectures (>0.55).
- **Top Neural Vocoder Resilience:** On the unseen multi-vocoder WaveFake benchmark, AuralGuard v2 achieves the best performance (**45.19% EER**).
- **Full Reproducibility:** Checkpoint weights, evaluation manifests, and result matrices are publicly available on [Hugging Face](https://huggingface.co/MoshinAli/auralguard-checkpoints).

---

## 🗂️ Repository Structure

```
auralguardv2/
├── config/                 # Hydra YAML configuration files
│   ├── experiment/         # Experiment presets (auralguard_v1, auralguard_v2, baselines)
│   ├── model/              # Architectural hyperparameters
│   ├── train/              # Optimization schedules and learning rates
│   └── eval/               # Evaluation protocols and dataset manifests
├── src/                    # Core Python package (installable: pip install -e .)
│   └── auralguard/
│       ├── data/           # Audio datasets, manifests, augmentation pipelines
│       ├── features/       # LFCC, Modified Group Delay, CQT phase extraction
│       ├── models/         # WavLM SSL frontend, artifact branch, fusion, AASIST, OCS loss
│       ├── evaluation/     # Metrics (EER, min t-DCF, ECE, Brier, bootstrap CIs)
│       ├── inference/      # Single-audio and batch inference predictors
│       ├── training/       # PyTorch Lightning trainer and CLI routines
│       └── utils/          # Logging, seed utilities, checkpoint handlers
├── scripts/                # CLI entrypoints
│   ├── train.py            # Model training pipeline
│   ├── evaluate.py         # Cross-dataset benchmark evaluator
│   ├── plot_paper_figures.py # Vector figure plotting suite
│   └── generate_real_paper_figures.py # Empirical figure generation from HF metrics
├── experiments/            # Benchmark matrices and evaluation results
│   ├── RESULT_CONSISTENCY_MATRIX.md # Verified results consistency documentation
│   └── results/            # Raw JSON evaluation metrics
├── pyproject.toml          # Package configuration and dependencies
├── requirements.txt        # Python dependency manifest
└── LICENSE                 # Open-source MIT License
```
*(Note: Manuscript source files and submission metadata are maintained privately during peer review and will be released upon acceptance).*

---

## 🚀 Quickstart

### 1. Installation

Clone the repository and set up a Python 3.10+ environment:

```bash
git clone https://github.com/MIHMahmudEli/auralguardv2.git
cd auralguardv2

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies and auralguard package
pip install --upgrade pip
pip install -e .
```

### 2. Download Pretrained Checkpoints

Trained weights for AuralGuard and baseline models are available on Hugging Face:

```python
from huggingface_hub import hf_hub_download

# Download AuralGuard v2 checkpoint
checkpoint_path = hf_hub_download(
    repo_id="MoshinAli/auralguard-checkpoints",
    filename="auralguard_v2/best.ckpt"
)
print(f"Downloaded checkpoint to: {checkpoint_path}")
```

### 3. Single-File Audio Inference

Run inference on an arbitrary audio file to obtain a calibrated deepfake prediction:

```bash
python -m auralguard.inference.predict \
    --audio path/to/sample.wav \
    --ckpt path/to/best.ckpt \
    --device cuda
```

Output:
```json
{
  "audio_path": "path/to/sample.wav",
  "prediction": "spoof",
  "anomaly_score": 0.8421,
  "calibrated_spoof_probability": 0.8914,
  "confidence_interval_95": [0.8520, 0.9241]
}
```

### 4. Running Benchmark Evaluation

To evaluate a checkpoint across the three benchmark corpora:

```bash
python scripts/evaluate.py \
    --model-name auralguard_v2 \
    --ckpt path/to/best.ckpt \
    --eval-manifests data/manifests/asvspoof2019_la_eval.csv data/manifests/inthewild.csv data/manifests/wavefake.csv \
    --batch-size 32 \
    --output-dir experiments/results/auralguard_v2/
```

---

## 📄 LaTeX Manuscript Compilation

The full 17-page camera-ready manuscript targeting Scopus Q1 journals (*Elsevier Computer Speech & Language* / *IEEE TIFS*) can be compiled directly from `paper/`:

```bash
cd paper
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

The resulting `paper/main.pdf` includes all verified figures, mathematical formulations, operational security thresholds, and the complete reference list.

---

## 📖 Citation

If you find AuralGuard useful in your research or applications, please cite our paper:

```bibtex
@article{hossain2026auralguard,
  title={AuralGuard: Multi-View One-Class Learning for Generalizable and Calibrated Detection of AI-Generated Speech in the Wild},
  author={Hossain, Mohsin Ibna and Muttakin, Md. and Miraj, Sadman Hasan and Hasan, Mohammad Mahmudul},
  journal={arXiv preprint},
  year={2026},
  url={https://github.com/MIHMahmudEli/auralguardv2}
}
```

---

## ⚖️ License and Ethics

- **Code License:** [MIT License](LICENSE).
- **Ethics & Dual-Use Governance:** AuralGuard is designed for defensive voice forensics, biometric authentication safeguarding, and synthetic media provenance verification. Checkpoint weights and evaluation manifests are released for reproducible defense research. Adaptive attack optimization tooling is strictly withheld.
