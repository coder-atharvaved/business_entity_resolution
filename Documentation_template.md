# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Team EntitySync  
**Team Members:** Atharva & Team  
**Submission Date:** September 2026  

---

## 1. Executive Summary
We designed and implemented a production-grade, two-stage **Decoupled Business Entity Resolution System** that maps heterogeneous, noisy business identity records from Source 2 and Source 3 to a deduplicated reference source (Source 1). By combining partition-based character 3-gram TF-IDF sparse matrix blocking with a 16-dimensional pairwise XGBoost gradient boosted decision tree classifier, our solution achieves high recall while constraining candidate set sizes to top-15 per entity. The decision threshold is directly calibrated to optimize the macro-averaged $F_{0.5}$ metric, maximizing precision and earning full credit on singleton records.

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis and domain inspection, the following critical patterns and noise challenges were observed:
- **Lexical and Suffix Variations:** Names contain diverse corporate abbreviations (`Corp`, `Corporation`, `Pvt Ltd`, `Private Limited`, `LLC`, `SA`, `SAS`) and conjunction variations (`&` vs. `and`).
- **Address Noise:** Missing components (e.g., absent PIN/ZIP codes or states), format transpositions, landmark-based descriptions (`Near SBI ATM`), and standard street abbreviations (`Rd`, `St`, `Ave`, `Blvd`, `Bldg`).
- **Open-Set Country Distribution:** While training data covers the `US` and `India`, test evaluation introduces unseen countries (such as `France`). The pipeline avoids country-specific hardcoding, treating country as an open string partition.
- **Extreme Class Imbalance:** Comparing all entities would require $O(N_1 \times (N_2 + N_3)) \approx 10^{11}$ comparisons. High-quality blocking is imperative to reduce computational complexity while maintaining high recall ceiling.
- **Precision-Weighted Evaluation:** The $F_{0.5}$ metric places 2x higher importance on precision than recall, meaning false merges (wrong linkages) are penalized severely compared to missed links.

### 2.2 Solution Strategy
We adopt a **Two-Stage Decoupled Hybrid Pipeline**:
1. **Stage 1 (Scalable Blocking & Candidate Generation):** Text is normalized and partitioned by country. A sublinear character 3-gram TF-IDF sparse dot-product engine generates a compact pool of top-15 candidates per Source 1 entity, directly satisfying the challenge update ranking bonus for small candidate sets.
2. **Stage 2 (Pairwise Machine Learning Classifier):** For each generated candidate pair, a 16-dimensional feature vector captures token overlap, character n-gram Jaccard similarity, SequenceMatcher ratio, numerical/PIN alignment, and rank metadata. An XGBoost classifier scores candidate pairs.
3. **Stage 3 (Precision Optimization & Singleton Handling):** Optimal decision thresholds are swept to maximize macro $F_{0.5}$. Entities with all candidate scores below threshold are cleanly emitted as singletons (empty match lists), capturing maximum score credit.

**Approach Type:** Multi-Stage Blocking + Feature Engineering + XGBoost Pairwise Classifier  
**Core Innovation:** Partition-aware character 3-gram sparse matrix candidate generation coupled with entity-level cross-validation and $F_{0.5}$ threshold calibration that guarantees matches $\subseteq$ candidates and strict format compliance.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  - **Country Partitioning:** Coarse partition by `country` (open-set string equality).
  - **Sublinear Char 3-gram TF-IDF Dot Product:** Dense-free sparse matrix multiplication on combined normalized business names and addresses.
  - **Compact Top-K Filtering:** Top-15 candidates per Source 1 entity with minimum similarity cutoff.
- **Candidate pairs generated:** Average of ~12–15 candidate pairs per Source 1 entity.
- **How true matches were not lost:** Character n-grams (`char_wb`, range 3-3) are highly resilient to misspellings, transliterations, word-order swaps, and dropped suffixes. Sublinear term frequency prevents common words from drowning distinctive business name tokens.

---

## 4. Matching Model

**Features used:**
- **Name features:** Token Jaccard similarity, character 3-gram Jaccard similarity, `difflib` sequence matcher ratio, length differential ratio, exact equality flag, substring containment flag, and first-token match flag.
- **Address features:** Address token Jaccard similarity, character 3-gram Jaccard similarity, address sequence ratio, length differential.
- **Numerical & Anchor features:** Extracted numbers/PIN/house numbers Jaccard similarity and shared numbers count.
- **Structural metadata:** Candidate ranking within blocking pool, total candidates count, source origin flag (`S2-` vs `S3-`).

**Model type:** XGBoost Classifier (`objective='binary:logistic'`, `max_depth=5`, `learning_rate=0.08`, `n_estimators=250`, `scale_pos_weight` tuned for class imbalance).  
**Threshold selection method:** Empirical threshold sweep maximizing macro $F_{0.5}$ on an entity-level validation set (`GroupShuffleSplit` by `source1_entity_id` to strictly prevent data leakage).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro):** **0.9621 (96.21%)** on held-out validation set of 2,400 Source 1 entities (36,000 candidate pairs) using Entity-Level GroupShuffleSplit.
- **Optimal Decision Threshold:** **0.824** (calibrated specifically for precision-heavy evaluation to prevent false merges).
- **Key Feature Importances:**
  1. `addr_tok_jaccard` (43.8%) — Address token overlap is the single strongest discriminator.
  2. `addr_ng_jaccard` (27.7%) — Character 3-gram address similarity captures format variations and abbreviations.
  3. `candidate_rank` (6.4%) — Structural rank in the blocking candidate pool.
  4. `name_seq_ratio` (4.2%) — Levenshtein letter-level similarity for business names.
  5. `num_jaccard` (4.2%) — Invariant house/suite number and PIN code overlap.
- **Common false positives (wrong merges):** Franchises or branch businesses sharing identical brand names in adjacent commercial zones or similar streets. Mitigated by enforcing strict address token and numerical overlap requirements.
- **Common false negatives (missed matches):** Entities where both name and address were drastically abbreviated or completely transformed via non-standard phonetic transliteration.

---

## 6. Conclusion
The developed system demonstrates that a disciplined, two-stage entity resolution architecture is capable of scaling to millions of commercial records while upholding high precision. By combining sparse matrix candidate generation, entity-level validation to prevent leakage, and $F_{0.5}$ threshold calibration, the pipeline produces compact, highly accurate submissions that strictly comply with all challenge rules and fair-play constraints.

---

## Appendix

### A. Code Artefacts
All runnable code is located in `code/business_entity_resolution/`:
- `src/ingest.py`: Multi-source TSV loader and schema validator.
- `src/normalize.py`: Multilingual text normalizer and numerical extractor.
- `src/blocking.py`: Scalable country-partitioned TF-IDF candidate generator.
- `src/features.py`: 16-dimensional feature extractor.
- `src/train_model.py`: Zero-leakage XGBoost trainer with GroupShuffleSplit.
- `src/threshold.py`: Official macro $F_{0.5}$ evaluation and threshold optimizer.
- `src/predict.py`: Inference engine generating final match predictions.
- `src/write_submission.py`: Official TSV serializer ensuring matches $\subseteq$ candidates.
- `src/run_pipeline.py`: Master CLI orchestrator executing the full workflow end-to-end.

To reproduce results:
```bash
python -m src.run_pipeline --data-dir "dataset" --output-dir "output"
```
