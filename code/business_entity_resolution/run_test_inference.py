"""
Production Inverted-Index Streaming Test Inference Engine
Scalable Search Engine Architecture (sub-second inverted indexing + XGBoost scoring):
- Inverted index on business name tokens (with frequency capping for stop words)
- Fast ranking & candidate selection (Top-10 per S1 entity for compactness bonus)
- Feature extraction & XGBoost prediction (Optimal Threshold = 0.824)
- Direct streaming to output/candidate_pairs.tsv and output/matching_results.tsv
- Automatic execution of utils/validate_submission.py
"""

import os
import sys
import json
import logging
import subprocess
from collections import defaultdict
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.normalize import normalize_name, normalize_address
from src.features import compute_pair_features, FEATURE_COLUMNS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("TestInference")

# Generic multilingual stop words that provide low discriminatory value across languages
STOPWORDS = {
    "and", "the", "of", "in", "at", "for", "to", "on", "by", "with", "a", "an",
    "de", "la", "le", "et", "des", "du", "en", "les", "pour", "un", "une", "par",
    "pvt", "ltd", "inc", "corp", "llc", "co", "sa", "sas", "sarl"
}

def get_tokens(text: str) -> list[str]:
    """Extracts distinctive tokens (length >= 2, non-stopwords)."""
    return [t for t in text.split() if len(t) >= 2 and t not in STOPWORDS]

def stream_source_file_by_country(path: str, country: str) -> tuple[list[str], list[str], list[str]]:
    """Streams a source file and extracts (entity_ids, norm_names, norm_addrs) for a specific country."""
    logger.info(f"Loading {os.path.basename(path)} for country [{country}]...")
    ids = []
    norm_names = []
    norm_addrs = []
    
    with open(path, "r", encoding="utf-8") as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 4 and parts[3].strip() == country:
                eid = parts[0].strip()
                n = normalize_name(parts[1])
                a = normalize_address(parts[2])
                ids.append(eid)
                norm_names.append(n)
                norm_addrs.append(a)
                
    logger.info(f"Loaded {len(ids):,} records for [{country}] from {os.path.basename(path)}")
    return ids, norm_names, norm_addrs

def run_country_inference(
    country: str,
    test_dir: str,
    model: XGBClassifier,
    threshold: float,
    top_k: int = 10,
    batch_size: int = 10000
) -> dict[str, tuple[list[str], list[str]]]:
    """
    Runs inverted-index blocking and XGBoost scoring for a single country.
    Returns: {s1_id: (candidate_list, matched_list)}
    """
    logger.info("=" * 70)
    logger.info(f"  PROCESSING TEST INFERENCE FOR COUNTRY: [{country}]")
    logger.info("=" * 70)
    
    # 1. Load S1 records for this country
    s1_path = os.path.join(test_dir, "test_source1.tsv")
    s1_ids, s1_names, s1_addrs = stream_source_file_by_country(s1_path, country)
    if not s1_ids:
        logger.warning(f"No S1 records found for country [{country}]")
        return {}
        
    # 2. Load Target records (S2 + S3)
    s2_path = os.path.join(test_dir, "test_source2.tsv")
    s3_path = os.path.join(test_dir, "test_source3.tsv")
    s2_ids, s2_names, s2_addrs = stream_source_file_by_country(s2_path, country)
    s3_ids, s3_names, s3_addrs = stream_source_file_by_country(s3_path, country)
    
    target_ids = s2_ids + s3_ids
    target_names = s2_names + s3_names
    target_addrs = s2_addrs + s3_addrs
    
    logger.info(f"Building token inverted index over {len(target_ids):,} target entities...")
    target_dict = {}
    inverted_index = defaultdict(list)
    
    for tid, n, a in zip(target_ids, target_names, target_addrs):
        target_dict[tid] = {"norm_name": n, "norm_addr": a, "entity_id": tid}
        for tok in get_tokens(n):
            # Cap posting list to avoid massive over-matching on generic words
            if len(inverted_index[tok]) < 800:
                inverted_index[tok].append(tid)
                
    logger.info(f"Inverted index built: {len(inverted_index):,} distinct vocabulary tokens")
    
    results: dict[str, tuple[list[str], list[str]]] = {}
    n_s1 = len(s1_ids)
    logger.info(f"Scoring {n_s1:,} S1 entities in streaming batches of {batch_size:,}...")
    
    for start_i in range(0, n_s1, batch_size):
        end_i = min(start_i + batch_size, n_s1)
        sub_s1_ids = s1_ids[start_i:end_i]
        sub_s1_names = s1_names[start_i:end_i]
        sub_s1_addrs = s1_addrs[start_i:end_i]
        
        pair_keys = []
        feature_rows = []
        cands_per_s1 = {}
        
        for curr_s1_id, n1, a1 in zip(sub_s1_ids, sub_s1_names, sub_s1_addrs):
            s1_row_data = {"norm_name": n1, "norm_addr": a1, "entity_id": curr_s1_id}
            s1_toks = get_tokens(n1)
            
            # Retrieve candidate targets from inverted index
            candidate_counts = defaultdict(int)
            for tok in s1_toks:
                for tid in inverted_index.get(tok, []):
                    candidate_counts[tid] += 1
                    
            if not candidate_counts:
                cands_per_s1[curr_s1_id] = []
                continue
                
            # Quick sort candidates by token count overlap
            sorted_candidates = sorted(candidate_counts.keys(), key=lambda x: -candidate_counts[x])[:top_k]
            cands_per_s1[curr_s1_id] = sorted_candidates
            total_c = len(sorted_candidates)
            
            for rank_idx, tgt_id in enumerate(sorted_candidates, start=1):
                tgt_data = target_dict.get(tgt_id, {})
                feats = compute_pair_features(s1_row_data, tgt_data, rank_idx, total_c)
                feature_rows.append(feats)
                pair_keys.append((curr_s1_id, tgt_id))
                
        # Run XGBoost inference on chunk
        matched_per_s1 = {eid: [] for eid in sub_s1_ids}
        if feature_rows:
            X_df = pd.DataFrame(feature_rows, columns=FEATURE_COLUMNS)
            probs = model.predict_proba(X_df)[:, 1]
            for (s1_id, tgt_id), prob in zip(pair_keys, probs):
                if prob >= threshold:
                    matched_per_s1[s1_id].append(tgt_id)
                    
        for eid in sub_s1_ids:
            results[eid] = (cands_per_s1.get(eid, []), matched_per_s1.get(eid, []))
            
        pct = (end_i / n_s1) * 100
        logger.info(f"[{country}] Progress: {end_i:,}/{n_s1:,} ({pct:.1f}%) entities scored")
        
    return results

def main():
    test_dir = r"C:\Users\Atharva\Downloads\6ab10eb3b23ba_student_resource\student_resource\dataset\test"
    models_dir = r"C:\Users\Atharva\Downloads\machine learnng\models"
    output_dir = r"C:\Users\Atharva\Downloads\machine learnng\output"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Trained Model & Threshold
    logger.info("Loading trained XGBoost model and configuration...")
    model_path = os.path.join(models_dir, "xgboost_matcher.json")
    config_path = os.path.join(models_dir, "model_config.json")
    
    model = XGBClassifier()
    model.load_model(model_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = float(config.get("best_threshold", 0.824))
    logger.info(f"Model loaded. Decision Threshold: {threshold:.3f}")
    
    # 2. Get master ordered list of test S1 IDs
    s1_test_path = os.path.join(test_dir, "test_source1.tsv")
    logger.info("Reading test_source1.tsv master ID list...")
    ordered_s1_ids = []
    with open(s1_test_path, "r", encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if parts and parts[0]:
                ordered_s1_ids.append(parts[0].strip())
    logger.info(f"Total Test Entities to generate: {len(ordered_s1_ids):,}")
    
    # 3. Run Inference per Country (France, US, India)
    all_results = {}
    for country in ["France", "US", "India"]:
        c_res = run_country_inference(
            country=country,
            test_dir=test_dir,
            model=model,
            threshold=threshold,
            top_k=10,
            batch_size=10000
        )
        all_results.update(c_res)
        
    # 4. Stream write output/candidate_pairs.tsv and output/matching_results.tsv
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    logger.info("Writing official submission TSV files...")
    
    with open(candidate_file, "w", encoding="utf-8", newline="\n") as fc, \
         open(matching_file, "w", encoding="utf-8", newline="\n") as fm:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        
        for s1_id in ordered_s1_ids:
            cands, matches = all_results.get(s1_id, ([], []))
            cands_set = set(cands)
            clean_matches = [m for m in matches if m in cands_set]
            
            fc.write(f"{s1_id}\t{','.join(cands)}\n")
            fm.write(f"{s1_id}\t{','.join(clean_matches)}\n")
            
    logger.info(f"Successfully generated:")
    logger.info(f"  1. {matching_file}")
    logger.info(f"  2. {candidate_file}")
    
    # 5. Run Official Submission Validator
    validator_path = r"C:\Users\Atharva\Downloads\6ab10eb3b23ba_student_resource\student_resource\utils\validate_submission.py"
    if os.path.exists(validator_path):
        logger.info("Running official submission validator...")
        cmd = [
            sys.executable,
            validator_path,
            "--matching", matching_file,
            "--candidate", candidate_file,
            "--test-dir", test_dir
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        if res.returncode == 0:
            logger.info("🏆 SUBMISSION VALIDATION PASSED (EXIT CODE 0)! 🏆")
        else:
            logger.warning(f"Validator reported issues (Exit code {res.returncode})")

if __name__ == "__main__":
    main()
