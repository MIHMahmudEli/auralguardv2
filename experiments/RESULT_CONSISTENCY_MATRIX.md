# AuralGuard — Result Consistency Matrix

This document tracks all empirical experimental metrics across the project. Every numerical claim in the manuscript (`main.tex`, `sections/*.tex`, figures, and tables) must trace directly and consistently to this matrix.

---

## 1. Master Metric Matrix

| Model / Architecture | Dataset / Split | Role | N Samples (Spoof / Total) | EER (%) | 95% Confidence Interval | AUROC | min t-DCF | F1 Score | Balanced Accuracy | ECE | Source File | Source Path | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **B1 (LFCC-LCNN)** | ASVspoof 2019 LA eval | Held-out In-Domain Test | 63,882 / 71,237 | **16.29%** | [15.97%, 16.63%] | 0.9104 | 0.6131 | 0.9022 | 0.8372 | 0.2651 | `results.json` | `results/b1_lcnn/eval_results/` | Verified Empirical |
| **B1 (LFCC-LCNN)** | In-the-Wild | Out-of-Domain Zero-Shot | 11,816 / 31,779 | **50.62%** | [50.00%, 51.19%] | 0.4979 | 0.9997 | 0.4205 | 0.4939 | 0.4292 | `results.json` | `results/b1_lcnn/eval_results/` | Verified Empirical |
| **B1 (LFCC-LCNN)** | WaveFake | Out-of-Domain Zero-Shot | 117,985 / 134,097 | **45.32%** | [44.86%, 45.76%] | 0.5638 | 1.0000 | 0.6542 | 0.5468 | 0.1965 | `results.json` | `results/b1_lcnn/eval_results/` | Verified Empirical |
| **B2 (RawNet2)** | ASVspoof 2019 LA eval | Held-out In-Domain Test | 63,882 / 71,237 | **8.99%** | [8.72%, 9.23%] | 0.9655 | 0.3444 | 0.9478 | 0.9101 | 0.1181 | `results.json` | `results/b2_rawnet2/eval_results/` | Verified Empirical |
| **B2 (RawNet2)** | In-the-Wild | Out-of-Domain Zero-Shot | 11,816 / 31,779 | **38.08%** | [37.57%, 38.59%] | 0.6671 | 0.9341 | 0.5474 | 0.6192 | 0.5761 | `results.json` | `results/b2_rawnet2/eval_results/` | Verified Empirical |
| **B2 (RawNet2)** | WaveFake | Out-of-Domain Zero-Shot | 117,985 / 134,097 | **50.37%** | [49.89%, 50.80%] | 0.4863 | 1.0000 | 0.6070 | 0.4963 | 0.1996 | `results.json` | `results/b2_rawnet2/eval_results/` | Verified Empirical |
| **B3 (AASIST)** | ASVspoof 2019 LA eval | Held-out In-Domain Test | 63,882 / 71,237 | **10.63%** | [10.36%, 10.92%] | 0.9574 | 0.3921 | 0.9378 | 0.8937 | 0.0545 | `results.json` | `results/b3_aasist/eval_results/` | Verified Empirical |
| **B3 (AASIST)** | In-the-Wild | Out-of-Domain Zero-Shot | 11,816 / 31,779 | **42.33%** | [41.80%, 42.85%] | 0.6122 | 0.9733 | 0.5032 | 0.5766 | 0.5552 | `results.json` | `results/b3_aasist/eval_results/` | Verified Empirical |
| **B3 (AASIST)** | WaveFake | Out-of-Domain Zero-Shot | 117,985 / 134,097 | **49.89%** | [49.37%, 50.34%] | 0.4873 | 1.0000 | 0.6116 | 0.5012 | 0.2472 | `results.json` | `results/b3_aasist/eval_results/` | Verified Empirical |
| **B5 (WavLM+OCS)** | ASVspoof 2019 LA eval | Held-out In-Domain Test | 63,882 / 71,237 | **2.86%** | [2.73%, 3.00%] | 0.9845 | 0.1041 | 0.9839 | 0.9714 | 0.4487 | `results.json` | `results/b5_wavlm_ocs/eval_results/` | Verified Empirical |
| **B5 (WavLM+OCS)** | In-the-Wild | Out-of-Domain Zero-Shot | 11,816 / 31,779 | **16.55%** | [16.12%, 16.98%] | 0.9103 | 0.4396 | 0.7895 | 0.8345 | 0.2320 | `results.json` | `results/b5_wavlm_ocs/eval_results/` | Verified Empirical |
| **B5 (WavLM+OCS)** | WaveFake | Out-of-Domain Zero-Shot | 117,985 / 134,097 | **47.43%** | [46.91%, 47.89%] | 0.5370 | 1.0000 | 0.6347 | 0.5257 | 0.4531 | `results.json` | `results/b5_wavlm_ocs/eval_results/` | Verified Empirical |
| **AuralGuard (v1)** | ASVspoof 2019 LA eval | Held-out In-Domain Test | 63,882 / 71,237 | **7.30%** | [7.12%, 7.50%] | 0.9425 | 0.2494 | 0.9579 | 0.9269 | 0.4524 | `results.json` | `results/auralguard/eval_results/` | Verified Empirical |
| **AuralGuard (v1)** | In-the-Wild | Out-of-Domain Zero-Shot | 11,816 / 31,779 | **23.47%** | [23.03%, 23.89%] | 0.8345 | 0.7268 | 0.7081 | 0.7654 | **0.0965** | `results.json` | `results/auralguard/eval_results/` | Verified Empirical |
| **AuralGuard (v1)** | WaveFake | Out-of-Domain Zero-Shot | 117,985 / 134,097 | **48.72%** | [48.25%, 49.22%] | 0.5207 | 1.0000 | 0.6227 | 0.5128 | 0.5048 | `results.json` | `results/auralguard/eval_results/` | Verified Empirical |
| **AuralGuard v2** | ASVspoof 2019 LA eval | Held-out In-Domain Test | 63,882 / 71,237 | **6.81%** | [6.61%, 7.00%] | 0.9498 | 0.2398 | 0.9608 | 0.9318 | 0.4746 | `results.json` | `results/auralguard_v2/eval_results/` | Verified Empirical |
| **AuralGuard v2** | In-the-Wild | Out-of-Domain Zero-Shot | 11,816 / 31,779 | **19.19%** | [18.78%, 19.61%] | 0.8820 | 0.5295 | 0.7580 | 0.8081 | 0.1716 | `results.json` | `results/auralguard_v2/eval_results/` | Verified Empirical |
| **AuralGuard v2** | WaveFake | Out-of-Domain Zero-Shot | 117,985 / 134,097 | **45.19%** | [44.69%, 45.71%] | 0.5659 | 0.9999 | 0.6553 | 0.5481 | 0.5001 | `results.json` | `results/auralguard_v2/eval_results/` | Verified Empirical |

---

## 2. Checkpoint Provenance Audit

| Checkpoint Path | Architecture | Epoch | Dev EER | File Size | Authoritative Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `checkpoints/b1_lcnn/best.ckpt` | LFCC-LCNN | 6 | 0.0086 (0.86%) | ~21.8 MB | Hugging Face `MoshinAli/auralguard-checkpoints` |
| `checkpoints/b2_rawnet2/best.ckpt` | RawNet2 | 25 | 0.0169 (1.69%) | ~54.2 MB | Hugging Face `MoshinAli/auralguard-checkpoints` |
| `checkpoints/b3_aasist/best.ckpt` | AASIST | 35 | 0.0474 (4.74%) | ~3.6 MB | Hugging Face `MoshinAli/auralguard-checkpoints` |
| `checkpoints/b5_wavlm_ocs/best.ckpt` | WavLM-Large + AASIST + OCSoftmax | 11 | 0.0020 (0.20%) | ~1.26 GB | Hugging Face `MoshinAli/auralguard-checkpoints` |
| `checkpoints/auralguard/best.ckpt` | AuralGuard v1 | 16 | 0.0016 (0.16%) | ~1.28 GB | Hugging Face `MoshinAli/auralguard-checkpoints` |
| `checkpoints/auralguard_v2/best.ckpt` | AuralGuard v2 | 5 | 0.0052 (0.52%) | ~1.28 GB | Hugging Face `MoshinAli/auralguard-checkpoints` |

---

## 3. Dataset Verification Matrix

| Dataset | Split | Total Utterances | Bona Fide Utterances | Spoofed Utterances | Audio Format | Sampling Rate | Description / Generators |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **ASVspoof 2019 LA** | Train | 25,380 | 2,580 | 22,800 | FLAC | 16 kHz | 6 TTS/VC algorithms (A01-A06) |
| **ASVspoof 2019 LA** | Development | 24,844 | 2,548 | 22,296 | FLAC | 16 kHz | Same 6 algorithms as train (A01-A06) |
| **ASVspoof 2019 LA** | Evaluation | 71,237 | 7,355 | 63,882 | FLAC | 16 kHz | 13 unseen algorithms (A07-A19) |
| **In-the-Wild** | Evaluation | 31,779 | 19,963 | 11,816 | WAV | 16 kHz | Real-world deepfakes of public figures |
| **WaveFake** | Evaluation | 134,097 | 16,112 | 117,985 | WAV | 16 kHz | MelGAN, Parallel WaveGAN, HiFi-GAN, WaveGlow |
| **ASVspoof 2021 LA** | Evaluation | 181,566 | 18,452 | 163,114 | FLAC | 16 kHz | Transmission channels (VoIP, PSTN) |
| **ASVspoof 2021 DF** | Evaluation | 611,829 | 22,617 | 589,212 | FLAC | 16 kHz | Lossy audio codecs |
| **MLAAD v5** | Evaluation | 328,000 | 164,100 | 163,900 | WAV | 16 kHz | 38 languages, modern neural TTS engines |
