# HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse

[![Project Status](https://img.shields.io/badge/Status-Active-success)](https://github.com/Sai-Kartheek-Reddy/HateMirage/)
[![Conference](https://img.shields.io/badge/LREC-2026-informational)](https://github.com/Sai-Kartheek-Reddy/HateMirage/)
[![License: Research Only](https://img.shields.io/badge/License-Research%20Only-red)](https://github.com/Sai-Kartheek-Reddy/HateMirage/)

**Official repository for the paper: "HateMirage: An Explainable Multi-Dimensional Dataset for Decoding Faux Hate and Subtle Online Abuse" (LREC 2026).**

---

## 📖 Overview

**HateMirage** is a novel benchmark dataset designed to advance research into **Faux Hate** a sophisticated form of online abuse where hostile narratives are constructed or amplified through misinformation rather than explicit toxicity.

Unlike traditional hate speech datasets that focus on overt slurs, HateMirage captures implicit, context-dependent harm. It bridges the gap between **misinformation** and **hate speech** by providing structured, multi-dimensional explanations for *why* a comment is harmful.

This repository contains the code, prompt templates, and evaluation scripts used to generate and benchmark the dataset.

### Key Features
* **Scale:** Contains **4,530** user comments sourced from YouTube discussions surrounding widely debunked claims.
* **Explainability:** Beyond binary classification, every instance is annotated with three reasoning dimensions: **Target**, **Intent**, and **Implication**.
* **Methodology:** Constructed using a **Retrieval-Augmented Generation (RAG)** pipeline with GPT-4, rigorous fact-checking (AltNews, FactChecker), and a **Human-in-the-loop** validation process for quality assurance.
* **Benchmarks:** Includes baselines for state-of-the-art SLMs and LLMs (Llama-3, Phi-3, Mistral, Qwen) in both Zero-shot and RAG settings.

---

## 🏗️ Dataset Structure

HateMirage moves beyond token-level highlighting. It decomposes "Faux Hate" into a structured reasoning framework:

| Dimension | Definition | Research Utility |
| :--- | :--- | :--- |
| **Target** | The specific entity, community, or individual being attacked. | Entity recognition, victim identification. |
| **Intent** | The underlying motivation (e.g., to delegitimize, polarize, or vilify). | Psychological analysis, intent detection. |
| **Implication** | The potential downstream societal or emotional consequence. | Causal reasoning, harm assessment. |

### Example
> **Comment:** *"The virus came from [Country] labs; they engineered it to destroy us."*

* **Target:** [Country]
* **Intent:** To accuse [Country] of intentionally creating a biological weapon to harm others.
* **Implication:** Could incite geopolitical hostility and xenophobia against citizens of [Country].

---

## 📂 Repository Organization

```text
HateMirage/
├── Annotation Guidelines/
│   ├── Hate Mirage Data Annotation Guidelines (Rating).pdf
│   └── HateMirage Data Annotation Guidelines.pdf
├── code/
│   ├── Code-README.md
│   ├── zero_shot.py          # Script for Zero-Shot explanation generation, including evaluation Metrics (ROUGE-L, SBERT Similarity)
│   └── rag.py                # Script for RAG-based explanation generation, including evaluation Metrics (ROUGE-L, SBERT Similarity)
│
├── source_docs/            # Reference materials for RAG (fact-checked claims)
│
├── requirements.txt
│
└── README.md
```

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Required libraries (see `requirements.txt`)
* Access to relevant LLM APIs (if reproducing generation steps)

### Benchmarking
This repository supports reproducing the paper's experiments. We evaluate models in two settings:

* **Zero-Shot:** The model generates explanations based solely on the comment.
* **RAG-Based:** The model is provided with retrieved fact-checking context to ground its reasoning.

---

## 🔐 Data Access & Ethics

### Access Policy
Due to the sensitive nature of the content (hate speech and misinformation) and to prevent misuse, the **HateMirage** dataset is not publicly downloadable. It is available exclusively for **academic and research purposes**.

To request access, please fill out the data usage agreement form below. Requests are manually reviewed to ensure compliance with ethical guidelines.

👉 **[Request Access to HateMirage](https://forms.gle/WhQ6U9yU8tZG8RrZ9)**

### Ethical Considerations
* **Privacy:** All Personally Identifiable Information (PII), including usernames, profile links, and timestamps, has been removed to protect individuals' privacy.  
* **Responsible Use:** The dataset is intended solely for research on detecting and understanding online abuse. It **must not** be used to train generative models for creating harmful content or for commercial surveillance purposes.  
* **Content Warning:** This dataset contains examples of hate speech, offensive language, and misleading claims. Reader discretion is strongly advised.  

---

## 📜 Citation

If you find the HateMirage dataset helpful for your research, please consider citing our paper:

```bibtex
soon
```
