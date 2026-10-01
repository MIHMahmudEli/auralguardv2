# Cover Letter for Manuscript Submission

**Date:** October 1, 2026  
**Target Journal:** *Computer Speech & Language* (Elsevier) / *IEEE Transactions on Information Forensics and Security* (IEEE)  
**Manuscript Title:** *AuralGuard: Multi-View One-Class Learning for Generalizable and Calibrated Detection of AI-Generated Speech in the Wild*  

**To:** The Editor-in-Chief  

Dear Editor,

We are pleased to submit our original research manuscript titled **"AuralGuard: Multi-View One-Class Learning for Generalizable and Calibrated Detection of AI-Generated Speech in the Wild"** for consideration for publication as a regular research article.

### Context and Motivation
Recent advances in generative speech synthesis (diffusion vocoders, flow-matching, and neural codec language models such as VALL-E and CosyVoice) have made AI-generated voice cloning virtually indistinguishable from bona fide human speech. While state-of-the-art countermeasures achieve competitive Equal Error Rates (EER) on constrained in-domain benchmarks, they suffer two severe operational failure modes when deployed in real-world environments:
1. **Catastrophic Out-of-Domain Collapse:** Traditional acoustic models (e.g., LFCC-LCNN, RawNet2) degrade to near-chance discrimination (38–50% EER) on unconstrained audio found "in the wild".
2. **Severe Probability Miscalibration:** Existing deep learning classifiers output overconfident false-alarm probabilities under acoustic domain shift (Expected Calibration Error exceeding 0.55), rendering their predictions legally inadmissible and dangerous in biometric authentication and judicial proceedings (e.g., under the *Daubert* standard and Article 50 of the EU AI Act).

### Key Contributions
In this manuscript, we present **AuralGuard**, an open-set multi-view framework designed specifically for generalizable and calibrated audio deepfake detection:
- **Multi-View Complementary Architecture:** We combine a frozen 315M self-supervised foundation-model front-end (WavLM-Large with learnable multi-layer attention) with an explicit 1.12M physical artifact stream capturing Linear Frequency Cepstral Coefficients (LFCC), Modified Group Delay (MGD), and Constant-Q Transform (CQT) phase derivatives.
- **Bidirectional Gated Cross-Attention Fusion:** A dynamic gating mechanism selectively queries and trusts semantic representations versus phase anomalies based on local time-frequency corruption and lossy codec transcoding.
- **One-Class Metric Back-End:** By projecting fused representations into a spectro-temporal graph attention network (AASIST back-end) optimized under an Orthogonal Centroid Scoring (OCS) objective with Supervised Contrastive regularization, AuralGuard maps genuine speech to a compact hyperspherical centroid while leaving the open set of synthetic speech unbounded.
- **Empirical Rigor and Verified Benchmark Validation:** Under a strict train-once (ASVspoof 2019 LA) evaluate-everywhere protocol, AuralGuard v2 achieves **19.19% EER** on the unconstrained *In-the-Wild* benchmark while maintaining an outstanding Expected Calibration Error of **0.0965 (v1) / 0.1716 (v2)**—vastly superior to single-view foundation baselines (ECE = 0.2320) and legacy acoustic architectures (ECE > 0.55). On the multi-vocoder *WaveFake* benchmark, AuralGuard v2 attains the top performance (**45.19% EER**) across all evaluated models.
- **Reproducibility:** All checkpoint weights, evaluation manifests, and training codes are publicly accessible at Hugging Face (`MoshinAli/auralguard-checkpoints`) and GitHub (`https://github.com/MIHMahmudEli/auralguardv2`).

### Authorship and Declarations
- **Authors:**
  1. Mohsin Ibna Hossain (First Author, ORCID: 0009-0003-9123-9040, `23-50194-1@student.aiub.edu`)
  2. Md. Muttakin (`23-50987-1@student.aiub.edu`)
  3. Sadman Hasan Miraj (`23-50380-1@student.aiub.edu`)
  4. Mohammad Mahmudul Hasan (Corresponding Author, Assistant Professor, `m.hasan@aiub.edu`)
- **Affiliation:** Department of Computer Science, American International University-Bangladesh (AIUB), Dhaka 1229, Bangladesh.
- **Originality:** This manuscript represents original work that has neither been published nor is under concurrent consideration for publication elsewhere.
- **Conflicts of Interest:** The authors declare no competing financial or non-financial interests.

Thank you very much for your time and consideration of our work.

Sincerely yours,

**Mohammad Mahmudul Hasan**  
Corresponding Author  
Assistant Professor, Department of Computer Science  
American International University-Bangladesh (AIUB)  
Dhaka 1229, Bangladesh  
Email: `m.hasan@aiub.edu`
