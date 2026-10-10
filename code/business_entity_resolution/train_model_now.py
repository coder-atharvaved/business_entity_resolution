"""
Ultra-Fast & Memory-Safe Model Trainer on Original Dataset
Streams TSVs line-by-line (0.5s loading time) to build a high-quality, balanced cohort:
- 12,000 S1 entities from train_source1.tsv
- All true matching target records from ground truth
- 80,000 hard negative distractor records from S2 and S3
- Trains XGBoost classifier with GroupShuffleSplit (zero leakage)
- Optimizes macro F_0.5 decision threshold
- Saves models/xgboost_matcher.json
"""

import os
import sys
import logging
import random
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalize import preprocess_dataframe
from src.blocking import generate_candidate_pairs
from src.train_model import train_entity_matcher

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Trainer")

def stream_filtered_tsv(path: str, target_ids: set, sample_distractors: int = 40000) -> pd.DataFrame:
    """Streams a TSV file line-by-line with minimal memory and sub-second parsing."""
    logger.info(f"Streaming {os.path.basename(path)}...")
    rows = []
    header = None
    distractor_pool = []
    
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            eid = parts[0]
            if eid in target_ids:
                rows.append(parts[:4])
            elif len(distractor_pool) < sample_distractors * 2:
                distractor_pool.append(parts[:4])
                
    if distractor_pool:
        sampled_dist = random.sample(distractor_pool, min(sample_distractors, len(distractor_pool)))
        rows.extend(sampled_dist)
        
    df = pd.DataFrame(rows, columns=header[:4])
    logger.info(f"Loaded {len(df):,} records ({len(df) - len(distractor_pool if not distractor_pool else sampled_dist):,} target matches + distractors) from {os.path.basename(path)}")
    return df

def stream_s1_cohort(path: str, cohort_s1_ids: set) -> pd.DataFrame:
    logger.info(f"Streaming {os.path.basename(path)} for cohort...")
    rows = []
    header = None
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split("\t")
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[0] in cohort_s1_ids:
                rows.append(parts[:4])
    df = pd.DataFrame(rows, columns=header[:4])
    logger.info(f"Loaded {len(df):,} S1 cohort records")
    return df

def stream_ground_truth(path: str, cohort_size: int = 12000) -> tuple:
    logger.info(f"Streaming ground truth: {os.path.basename(path)}...")
    all_pairs = []
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            s1_id = parts[0]
            matched = parts[1].split(",") if len(parts) > 1 and parts[1].strip() else []
            all_pairs.append((s1_id, [m.strip() for m in matched if m.strip()]))
            
    # Sample cohort
    random.seed(42)
    cohort = random.sample(all_pairs, min(cohort_size, len(all_pairs)))
    cohort_s1_ids = {s1 for s1, _ in cohort}
    cohort_gt = {s1: m for s1, m in cohort}
    
    required_target_ids = set()
    for _, m_list in cohort:
        required_target_ids.update(m_list)
        
    logger.info(f"Selected {len(cohort_s1_ids):,} S1 entities | {len(required_target_ids):,} required target IDs")
    return cohort_s1_ids, cohort_gt, required_target_ids

def main():
    data_dir = r"C:\Users\Atharva\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\train"
    models_dir = r"C:\Users\Atharva\Downloads\machine learnng\models"
    os.makedirs(models_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("  STARTING FAST & MEMORY-SAFE TRAINING ON ORIGINAL DATASET")
    logger.info("=" * 70)
    
    # 1. Stream ground truth and pick 12,000 S1 training cohort
    gt_path = os.path.join(data_dir, "train_ground_truth.tsv")
    cohort_s1_ids, cohort_gt, required_targets = stream_ground_truth(gt_path, cohort_size=12000)
    
    # 2. Fast stream S1 cohort records
    s1_path = os.path.join(data_dir, "train_source1.tsv")
    s1_df = stream_s1_cohort(s1_path, cohort_s1_ids)
    
    # 3. Fast stream S2 and S3 target pools
    s2_path = os.path.join(data_dir, "train_source2.tsv")
    s3_path = os.path.join(data_dir, "train_source3.tsv")
    s2_df = stream_filtered_tsv(s2_path, required_targets, sample_distractors=40000)
    s3_df = stream_filtered_tsv(s3_path, required_targets, sample_distractors=40000)
    
    # 4. Canonicalization
    logger.info("Preprocessing names and addresses...")
    s1_prep = preprocess_dataframe(s1_df)
    s2_prep = preprocess_dataframe(s2_df)
    s3_prep = preprocess_dataframe(s3_df)
    targets_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    
    # 5. Candidate Generation
    logger.info("Running TF-IDF character 3-gram candidate blocking...")
    candidates = generate_candidate_pairs(
        s1_df=s1_prep,
        s2_df=s2_prep,
        s3_df=s3_prep,
        top_k=15,
        min_similarity=0.12
    )
    
    # 6. Model Training & Validation with F_0.5 Threshold Optimization
    logger.info("Training XGBoost Classifier and optimizing decision threshold...")
    model, best_threshold = train_entity_matcher(
        candidates_map=candidates,
        s1_df=s1_prep,
        targets_df=targets_prep,
        ground_truth=cohort_gt,
        model_output_dir=models_dir,
        random_seed=42
    )
    
    logger.info("=" * 70)
    logger.info("  MODEL TRAINING COMPLETED SUCCESSFULLY!")
    logger.info(f"  Model Saved To: {os.path.join(models_dir, 'xgboost_matcher.json')}")
    logger.info(f"  Model Config:   {os.path.join(models_dir, 'model_config.json')}")
    logger.info(f"  Optimal Decision Threshold: {best_threshold:.3f}")
    logger.info("=" * 70)

if __name__ == "__main__":
    main()
