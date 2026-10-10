"""
End-to-End Orchestrator Pipeline
================================
Runs the entire business entity resolution pipeline from raw TSVs to final scored submission.
Can be executed end-to-end via CLI.
"""

import os
import sys
import argparse
import logging
import subprocess
import pandas as pd

from .ingest import load_dataset_split
from .normalize import preprocess_dataframe
from .blocking import generate_candidate_pairs
from .train_model import train_entity_matcher
from .predict import load_trained_matcher, predict_matches
from .write_submission import generate_submission_files
from .threshold import compute_macro_f05

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("Pipeline")

def run_end_to_end_pipeline(
    data_dir: str,
    output_dir: str = "output",
    models_dir: str = "models",
    sample_size: int = None,
    top_k: int = 15,
    run_validation: bool = True
):
    """
    Executes the complete Machine Learning lifecycle:
    Ingestion -> Preprocessing -> Blocking -> Training -> Inference -> Formatting -> Validation
    """
    logger.info("=" * 70)
    logger.info("  STARTING BUSINESS ENTITY RESOLUTION PIPELINE")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # STAGE 1: INGESTION & TRAINING PREPARATION
    # -------------------------------------------------------------------------
    logger.info(">>> STAGE 1: Loading & Preprocessing Training Data...")
    s1_train, s2_train, s3_train, gt_train = load_dataset_split(data_dir, split="train")
    
    if sample_size and sample_size < len(s1_train):
        logger.info(f"Subsampling {sample_size:,} S1 entities for rapid development...")
        s1_train = s1_train.iloc[:sample_size].copy()
        valid_s1_set = set(s1_train["entity_id"])
        gt_train = {k: v for k, v in gt_train.items() if k in valid_s1_set}
        
    logger.info("Canonicalizing names and addresses...")
    s1_train_prep = preprocess_dataframe(s1_train)
    s2_train_prep = preprocess_dataframe(s2_train)
    s3_train_prep = preprocess_dataframe(s3_train)
    targets_train = pd.concat([s2_train_prep, s3_train_prep], ignore_index=True)
    
    # -------------------------------------------------------------------------
    # STAGE 2: BLOCKING / CANDIDATE GENERATION (TRAIN)
    # -------------------------------------------------------------------------
    logger.info(">>> STAGE 2: Running Blocking on Training Split...")
    train_candidates = generate_candidate_pairs(
        s1_train_prep, s2_train_prep, s3_train_prep, top_k=top_k
    )
    
    # Audit train recall ceiling
    total_true_links = 0
    captured_true_links = 0
    for s1_id, true_targets in gt_train.items():
        cand_set = set(train_candidates.get(s1_id, []))
        for tgt in true_targets:
            total_true_links += 1
            if tgt in cand_set:
                captured_true_links += 1
                
    recall_ceiling = captured_true_links / max(total_true_links, 1)
    logger.info(
        f"Blocking Recall Ceiling: {recall_ceiling:.2%} "
        f"({captured_true_links:,}/{total_true_links:,} true matches preserved in candidate pool)"
    )
    
    # -------------------------------------------------------------------------
    # STAGE 3: MODEL TRAINING & THRESHOLD OPTIMIZATION
    # -------------------------------------------------------------------------
    logger.info(">>> STAGE 3: Training Pairwise XGBoost Classifier & Sweeping Threshold...")
    model, optimal_threshold = train_entity_matcher(
        candidates_map=train_candidates,
        s1_df=s1_train_prep,
        targets_df=targets_train,
        ground_truth=gt_train,
        model_output_dir=models_dir
    )
    
    # -------------------------------------------------------------------------
    # STAGE 4: INGESTION & INFERENCE ON TEST SET
    # -------------------------------------------------------------------------
    logger.info(">>> STAGE 4: Loading & Preprocessing Test Set...")
    s1_test, s2_test, s3_test, _ = load_dataset_split(data_dir, split="test")
    
    s1_test_prep = preprocess_dataframe(s1_test)
    s2_test_prep = preprocess_dataframe(s2_test)
    s3_test_prep = preprocess_dataframe(s3_test)
    targets_test = pd.concat([s2_test_prep, s3_test_prep], ignore_index=True)
    
    logger.info(">>> STAGE 5: Generating Test Candidate Pairs...")
    test_candidates = generate_candidate_pairs(
        s1_test_prep, s2_test_prep, s3_test_prep, top_k=top_k
    )
    
    logger.info(">>> STAGE 6: Predicting Test Matches...")
    test_predictions = predict_matches(
        model=model,
        threshold=optimal_threshold,
        candidates_map=test_candidates,
        s1_df=s1_test_prep,
        targets_df=targets_test
    )
    
    # -------------------------------------------------------------------------
    # STAGE 5: SERIALIZATION OF SUBMISSION FILES
    # -------------------------------------------------------------------------
    logger.info(">>> STAGE 7: Writing Submission Files...")
    s1_test_ordered_ids = s1_test["entity_id"].tolist()
    match_file, cand_file = generate_submission_files(
        candidates_map=test_candidates,
        predictions_map=test_predictions,
        s1_ordered_ids=s1_test_ordered_ids,
        output_dir=output_dir
    )
    
    logger.info("=" * 70)
    logger.info("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    logger.info(f"  Matching Results : {os.path.abspath(match_file)}")
    logger.info(f"  Candidate Pairs  : {os.path.abspath(cand_file)}")
    logger.info("=" * 70)
    
    # -------------------------------------------------------------------------
    # STAGE 6: AUTOMATED VALIDATION
    # -------------------------------------------------------------------------
    if run_validation:
        validator_script = os.path.join(data_dir, "..", "utils", "validate_submission.py")
        if not os.path.exists(validator_script):
            validator_script = os.path.join("utils", "validate_submission.py")
            
        test_dir = os.path.join(data_dir, "test") if os.path.isdir(os.path.join(data_dir, "test")) else data_dir
        if os.path.exists(validator_script):
            logger.info(">>> Running Official Submission Validator...")
            cmd = [
                sys.executable,
                validator_script,
                "--matching", match_file,
                "--candidate", cand_file,
                "--test-dir", test_dir
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            print(res.stdout)
            if res.stderr:
                print(res.stderr, file=sys.stderr)
            if res.returncode == 0:
                logger.info("✅ Official Validator returned PASS (Exit Code 0)!")
            else:
                logger.warning(f"Validator returned Exit Code {res.returncode}")

def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution End-to-End Pipeline")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to dataset root folder")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for output TSVs")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory for saving models")
    parser.add_argument("--sample-size", type=int, default=None, help="Sample size for training data (optional)")
    parser.add_argument("--top-k", type=int, default=15, help="Max candidates per S1 entity")
    parser.add_argument("--skip-validation", action="store_true", help="Skip running validator script")
    
    args = parser.parse_args()
    
    run_end_to_end_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        models_dir=args.models_dir,
        sample_size=args.sample_size,
        top_k=args.top_k,
        run_validation=not args.skip_validation
    )

if __name__ == "__main__":
    main()
