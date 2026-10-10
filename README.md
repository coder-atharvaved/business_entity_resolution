# Business Entity Resolution System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Model: XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

Production-grade, end-to-end Machine Learning pipeline for multi-source **Business Entity Resolution (Record Linkage)** designed for large-scale commercial datasets. 

This system resolves noisy, unlinked business entity records from multiple heterogeneous sources (Source 2 and Source 3) against a canonical deduplicated reference source (Source 1).

---

## 🏗️ System Architecture

The pipeline uses a **two-stage decoupled architecture** optimized for high precision and scalability:

```
[Raw TSV Datasets (S1, S2, S3)]
               │
               ▼
┌───────────────────────────────────────────┐
│ 1. Text Normalization & Canonicalization  │
│    - Legal suffix stripping (Corp, Pvt)   │
│    - Diacritic normalization, street exp. │
│    - PIN / house number extraction        │
└──────────────────────┬────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│ 2. Scalable Candidate Blocking Engine     │
│    - Country partitioning (US, IN, FR)    │
│    - Batched TF-IDF char 3-gram cosine    │
│    - Compact Top-15 candidate sets        │──► output/candidate_pairs.tsv
└──────────────────────┬────────────────────┘
                       │ (S1, Candidate) pairs
                       ▼
┌───────────────────────────────────────────┐
│ 3. Pairwise Feature Engineering           │
│    - 16 semantic, lexical, numerical      │
│      and structural similarity signals    │
└──────────────────────┬────────────────────┘
                       │ Feature Vectors
                       ▼
┌───────────────────────────────────────────┐
│ 4. XGBoost Classifier                     │
│    - Entity-level GroupKFold split        │
│    - MIT/Apache 2.0 compliant (<8B params)│
└──────────────────────┬────────────────────┘
                       │ Match Probabilities
                       ▼
┌───────────────────────────────────────────┐
│ 5. Macro F_0.5 Threshold Optimizer        │
│    - Precision-heavy cutoff tuning        │
│    - 1.0 credit singleton handling        │──► output/matching_results.tsv
└──────────────────────┬────────────────────┘
                       │
                       ▼
┌───────────────────────────────────────────┐
│ 6. Submission Validator                   │
│    - utils/validate_submission.py         │──► PASS (Exit Code 0)
└───────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
.
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       │   ├── __init__.py           # Package exports
│       │   ├── ingest.py             # TSV loader with strict schema checks
│       │   ├── normalize.py          # Multilingual text & address cleaner
│       │   ├── blocking.py           # Scalable TF-IDF blocking engine
│       │   ├── features.py           # 16-dimensional pairwise feature extractor
│       │   ├── train_model.py        # XGBoost trainer with entity-level split
│       │   ├── threshold.py          # Macro F_0.5 evaluator and threshold tuner
│       │   ├── predict.py            # Test inference module
│       │   ├── write_submission.py   # Official TSV writer (matches ⊆ candidates)
│       │   └── run_pipeline.py       # Master CLI orchestrator
│       ├── sample_data/              # Small sample S1 / S2 / S3 TSV files
│       ├── output/                   # Sample matching results
│       ├── run_test_inference.py     # Run inference on test data
│       ├── test_custom_tsv.py        # Test on a custom TSV
│       ├── train_model_now.py        # Quick model training script
│       ├── README.md                 # Module-level documentation
│       └── requirements.txt          # Pinned environment dependencies
├── models/
│   ├── xgboost_matcher.json          # Trained XGBoost model
│   └── model_config.json             # Model configuration / threshold
├── Documentation_template.md
└── README.md                         # You are here
```

---

## ⚡ Quick Start

### 1. Environment Setup
Clone the repo and install dependencies:
```bash
git clone https://github.com/coder-atharvaved/business_entity_resolution.git
cd business_entity_resolution/code/business_entity_resolution
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline
Execute the full training, candidate generation, inference, and validation pipeline (run from `code/business_entity_resolution`):

```bash
python -m src.run_pipeline \
    --data-dir "../dataset" \
    --output-dir "../../output" \
    --models-dir "../../models"
```

### Fast Prototype / Verification Mode
To quickly verify on a smaller subset (e.g. 2,000 entities):
```bash
python -m src.run_pipeline \
    --data-dir "../dataset" \
    --output-dir "../../output" \
    --sample-size 2000
```

> **Note:** The full dataset, large output files and `submission.zip` are not included in this repository because of GitHub's file size limits. Use `sample_data/` for a quick try-out.

---

## 🎯 Key Engineering Innovations

1. **Scalable Sparse Dot-Product Blocking:** Replaces the intractable $O(N_1 \times (N_2+N_3))$ search space with a sparse TF-IDF character 3-gram dot-product, processing millions of candidates in seconds while keeping memory usage under 2 GB.
2. **Compact Candidate Sets for Leaderboard Bonus:** Explicitly limits candidates to the top-15 per entity, directly maximizing the challenge's candidate-set compactness ranking criterion without sacrificing recall.
3. **Strict Zero-Leakage Validation:** Splits train/validation splits by `source1_entity_id` using `GroupShuffleSplit`, ensuring no business entities bleed between splits.
4. **Precision-First $F_{0.5}$ Tuning:** Sweeps decision thresholds to maximize macro-averaged $F_{0.5}$ and handles singletons cleanly (yielding a perfect 1.0 score for correctly predicted empty lists).
5. **Zero External Lookups:** Completely self-contained, offline algorithmic and ML pipeline strictly conforming to fair-play rules.
