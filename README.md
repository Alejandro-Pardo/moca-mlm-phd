# Modular Language and Domain Adaptation with DoRA

This repository explores whether language adaptation and domain-specific knowledge can be learned as independent parameter updates using **DoRA (Weight-Decomposed Low-Rank Adaptation)** and subsequently merged via parameter arithmetic.

The target task is causal language modeling on held-out **Spanish legal text (OPUS DGT)** evaluated by next-token perplexity (PPL) and paired block-level negative log-likelihood (NLL).

---

## 📁 Repository Structure

```text
├── report.pdf / report/        # Main project report (based on the Qwen model)
├── qwen-adaptation.ipynb       # Experiment notebook: Qwen/Qwen3-0.6B-Base (Report Basis)
├── lfm-adaptation.ipynb        # Comparative notebook: LiquidAI/LFM2.5-350M-Base
└── README.md                   # Project overview and documentation

```

### File Descriptions

* **Project Report**: The core written report containing in-depth theoretical analysis, training dynamics, paired statistical tests, and conclusions. **Note:** The empirical results and analysis presented in the main report are based on the **Qwen (`Qwen/Qwen3-0.6B-Base`)** notebook.
* **`qwen-adaptation.ipynb` (Qwen3-0.6B-Base)**: The primary experiment notebook. Demonstrates DoRA training, zero-shot task arithmetic, supervised refinement, and bootstrap statistical validation on Qwen3-0.6B.
* **`lfm-adaptation.ipynb` (LFM2.5-350M-Base)**: A parallel comparative experiment running the identical pipeline on Liquid AI's LFM2.5 architecture to assess cross-architecture transfer and tokenization effects.

---

## 🔬 Experimental Setup

Both notebooks follow a strictly controlled data budget and training protocol:

| Factor / Split | Dataset Source | Language(s) | Budget (Words) |
| --- | --- | --- | --- |
| **Language Factor** | OPUS Books (`en-es`) | Spanish (`es`) | 1,000,000 train / 100,000 val |
| **Legal Domain Factor** | OPUS DGT (Multilingual) | 9 non-Spanish EU languages (`bg, fi, ga, hr, mt, nl, sh, sk, sv`) | 1,000,000 train / 100,000 val (balanced) |
| **Target Domain** | OPUS DGT (`es-ga`) | Spanish (`es`) | 1,000,000 train / 100,000 val / 100,000 test |

### Methodology

1. **Independent DoRA Training**:
* Rank $r = 8$, Alpha $\alpha = 16$, applied to all linear layers (`all-linear`).
* Tokenized texts are packed into 512-token blocks and equalized across factors to ensure identical model-token exposure and optimizer step counts.


2. **Zero-Shot Task Arithmetic**:
* Adapters are merged into base backbones in full precision (`float32`), and parameter deltas are added linearly:

$$W_{\text{composed}} = W_{\text{Spanish}} + W_{\text{legal}} - W_{\text{base}}$$




3. **Supervised Refinement**:
* A secondary DoRA adapter is trained on target Spanish DGT initialized from the frozen composed model and compared against direct end-to-end target adaptation.



---

## 📊 Summary of Results

### 1. Primary Results (Qwen3-0.6B-Base — Report Basis)

*Evaluation on 326 held-out Spanish DGT test blocks (512 tokens each):*

| Model Variant | Mean NLL | Perplexity (PPL) | 95% Bootstrap CI | Reduction vs. Base |
| --- | --- | --- | --- | --- |
| **Direct Spanish-Legal DoRA** | 2.0983 | **8.15** | [7.94 – 8.36] | 36.1% |
| **Composition + Supervised DoRA** | 2.1096 | **8.24** | [8.03 – 8.45] | 35.4% |
| **Base Model (Unadapted)** | 2.5462 | **12.76** | [12.46 – 13.05] | — |
| **Legal DoRA (Non-Spanish)** | 2.5572 | **12.90** | [12.61 – 13.18] | -1.1% |
| **Zero-Shot Composition** | 2.7047 | **14.95** | [14.58 – 15.31] | -17.2% |
| **Spanish DoRA (Books)** | 2.8371 | **17.07** | [16.60 – 17.53] | -33.8% |

---

### 2. Comparative Results (LFM2.5-350M-Base)

*Evaluation on 306 held-out Spanish DGT test blocks (512 tokens each):*

| Model Variant | Mean NLL | Perplexity (PPL) | 95% Bootstrap CI | Reduction vs. Base |
| --- | --- | --- | --- | --- |
| **Composition + Supervised DoRA** | 2.0797 | **8.00** | [7.83 – 8.17] | 91.8% |
| **Direct Spanish-Legal DoRA** | 2.0906 | **8.09** | [7.92 – 8.26] | 91.7% |
| **Legal DoRA (Non-Spanish)** | 2.9005 | **18.18** | [17.84 – 18.52] | 81.3% |
| **Zero-Shot Composition** | 3.1359 | **23.01** | [22.59 – 23.42] | 76.3% |
| **Spanish DoRA (Books)** | 3.1751 | **23.93** | [23.43 – 24.42] | 75.4% |
| **Base Model (Unadapted)** | 4.5770 | **97.22** | [94.58 – 99.82] | — |

---

## 🛠️ Requirements & Setup

To reproduce the experiments in Kaggle:

```bash
pip install -q \
    "transformers==5.15.0" \
    "peft==0.20.0" \
    "datasets==5.0.1" \
    "accelerate>=1.8.0,<2" \
    safetensors sentencepiece matplotlib pandas numpy

```

### Hardware Requirements

* **GPU**: 2x NVIDIA Tesla T4 with CUDA enabled.
* **Internet**: Enabled (for downloading OPUS datasets and model weights from Hugging Face).
