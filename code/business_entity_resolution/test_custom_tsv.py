"""
Custom TSV Test Runner
Allows testing ANY custom source TSV files through the trained XGBoost Entity Resolution system.
Prints matched entities and confidence scores directly to console and saves to TSV.

Usage:
  python test_custom_tsv.py --s1 "path/to/my_s1.tsv" --s2 "path/to/my_s2.tsv" --s3 "path/to/my_s3.tsv" --output "output/my_results.tsv"
"""

import os
import sys
import json
import argparse
import pandas as pd
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ingest import load_source_tsv
from src.normalize import preprocess_dataframe
from src.blocking import generate_candidate_pairs
from src.features import build_feature_matrix

def test_custom_files(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    output_path: str = "output/custom_matching_results.tsv",
    models_dir: str = "models",
    top_k: int = 10
):
    print("=" * 70)
    print("  RUNNING CUSTOM TSV ENTITY RESOLUTION TEST")
    print("=" * 70)
    
    # 1. Load Trained Model
    model_path = os.path.join(models_dir, "xgboost_matcher.json")
    config_path = os.path.join(models_dir, "model_config.json")
    if not os.path.exists(model_path) or not os.path.exists(config_path):
        raise FileNotFoundError(f"Model not found in {models_dir}. Please train model first.")
        
    model = XGBClassifier()
    model.load_model(model_path)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = float(config.get("best_threshold", 0.824))
    print(f"[*] Loaded XGBoost Model (Decision Threshold: {threshold:.3f})")
    
    # 2. Ingest Custom Files
    print(f"[*] Loading S1: {s1_path}")
    s1_df = load_source_tsv(s1_path)
    print(f"[*] Loading S2: {s2_path}")
    s2_df = load_source_tsv(s2_path)
    print(f"[*] Loading S3: {s3_path}")
    s3_df = load_source_tsv(s3_path)
    
    # 3. Preprocess
    print("[*] Normalizing names and addresses...")
    s1_prep = preprocess_dataframe(s1_df)
    s2_prep = preprocess_dataframe(s2_df)
    s3_prep = preprocess_dataframe(s3_df)
    targets_prep = pd.concat([s2_prep, s3_prep], ignore_index=True)
    
    # 4. Generate Candidates
    print(f"[*] Generating candidates (top_k={top_k})...")
    candidates_map = generate_candidate_pairs(s1_prep, s2_prep, s3_prep, top_k=top_k)
    
    # 5. Extract Features & Predict
    print("[*] Scoring candidate pairs with XGBoost...")
    X_df, pair_ids = build_feature_matrix(candidates_map, s1_prep, targets_prep)
    
    predictions = {s1_id: [] for s1_id in s1_df["entity_id"]}
    match_details = []
    
    if len(pair_ids) > 0:
        probs = model.predict_proba(X_df)[:, 1]
        for (s1_id, tgt_id), prob in zip(pair_ids, probs):
            if prob >= threshold:
                predictions[s1_id].append(tgt_id)
                match_details.append((s1_id, tgt_id, float(prob)))
                
    # 6. Display Results
    print("\n" + "=" * 70)
    print("  DETAILED MATCHING RESULTS")
    print("=" * 70)
    s1_lookup = s1_df.set_index("entity_id").to_dict("index")
    tgt_lookup = targets_prep.set_index("entity_id").to_dict("index")
    
    for s1_id in s1_df["entity_id"]:
        s1_info = s1_lookup.get(s1_id, {})
        matched_ids = predictions.get(s1_id, [])
        print(f"\n[Source 1 Entity] ID: {s1_id}")
        print(f"  Name   : {s1_info.get('business_name')}")
        print(f"  Address: {s1_info.get('business_address')} ({s1_info.get('country')})")
        
        if matched_ids:
            print(f"  --> MATCHED ENTITIES ({len(matched_ids)}):")
            for tid in matched_ids:
                t_info = tgt_lookup.get(tid, {})
                print(f"      [{tid}] {t_info.get('business_name')} | {t_info.get('business_address')}")
        else:
            print("  --> SINGLETON (No true match found in S2/S3)")
            
    # 7. Write to TSV
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_df["entity_id"]:
            f.write(f"{s1_id}\t{','.join(predictions.get(s1_id, []))}\n")
            
    print("\n" + "=" * 70)
    print(f"[*] Results saved to: {output_path}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Test Custom TSV Files")
    parser.add_argument("--s1", type=str, required=True, help="Path to Source 1 TSV")
    parser.add_argument("--s2", type=str, required=True, help="Path to Source 2 TSV")
    parser.add_argument("--s3", type=str, required=True, help="Path to Source 3 TSV")
    parser.add_argument("--output", type=str, default="output/custom_matching_results.tsv", help="Output path")
    parser.add_argument("--models-dir", type=str, default="models", help="Path to models folder")
    parser.add_argument("--top-k", type=int, default=10, help="Top candidates per entity")
    
    args = parser.parse_args()
    test_custom_files(args.s1, args.s2, args.s3, args.output, args.models_dir, args.top_k)

if __name__ == "__main__":
    main()
