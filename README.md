# AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives

<p align="center">
    <img src="https://github.com/LLM-VLM-GSL/AHA/blob/main/assets/logo.png" width="200">
</p>

[![Project Page](https://img.shields.io/badge/Project-Page-Green)](https://github.com/LLM-VLM-GSL/AHA)
[![Arxiv](https://img.shields.io/badge/Arxiv-2512.24052-red)](https://arxiv.org/abs/2512.24052)
[![Model](https://img.shields.io/badge/HuggingFace-Model-blue)](https://huggingface.co/ASU-GSL/Qwen-Audio-AHA)
[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-orange)](https://huggingface.co/datasets/ASU-GSL/AHA)

## 🚀 News
* **[2024.12]** We release **Qwen-Audio-AHA**, the dataset, and the inference code!
* **[2024.12]** Our paper **"AHA: Aligning Large Audio-Language Models for Reasoning Hallucinations via Counterfactual Hard Negatives"** is available on [arXiv](https://arxiv.org/abs/2512.24052).

---

## 💡 Highlights
**AHA (Audio Hallucination Alignment)** is a unified framework designed to mitigate hallucinations in Large Audio-Language Models (LALMs).

* **Fine-grained Taxonomy:** We identify four core hallucination types: Event Omission, False Event Identity, Temporal Relation Error, and Quantitative Temporal Error.
* **Counterfactual Hard Negative Mining:** A novel pipeline to construct preference pairs that force models to ground responses in actual acoustic evidence rather than linguistic priors.
* **AHA-Eval:** A rigorous diagnostic benchmark for fine-grained temporal and causal reasoning in audio.
* **Superior Performance:** Significant improvements over base models (e.g., +13.7% on AHA-Eval) and competitive results on public benchmarks like MMAU and MMAR.

---

## 🛠 Installation

1. **Clone the repository and navigate to the AHA folder**
```bash
git clone [https://github.com/LLM-VLM-GSL/AHA.git](https://github.com/LLM-VLM-GSL/AHA.git)
cd AHA
```
2. **Install Packages**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

